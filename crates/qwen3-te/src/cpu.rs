//! CPU kernels of the reference graph. Every output element is computed by one thread in a fixed
//! order, so results do not depend on the thread count (tests check this).

use crate::weights::Mat;
use kernels_sys::QType;
use oxn::linear::WeightRows;
use std::ops::Range;

/// Raw pointer that may cross a scoped-thread boundary; users write disjoint elements only.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// Run `f` on `threads` contiguous pieces of `0..n` (fewer when `n` is small).
pub fn par_ranges(n: usize, threads: usize, f: impl Fn(Range<usize>) + Sync) {
    let t = threads.max(1).min(n.max(1));
    if t <= 1 {
        f(0..n);
        return;
    }
    let per = n.div_ceil(t);
    std::thread::scope(|s| {
        for i in 0..t {
            let r = (i * per).min(n)..((i + 1) * per).min(n);
            if r.is_empty() {
                continue;
            }
            let f = &f;
            s.spawn(move || f(r));
        }
    });
}

/// Fixed-order dot product: 8 lane partial sums, then the lanes left to right, then the tail.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// `y[t][..] = x[t][..] W^T` for `n` tokens, f32 arithmetic on dequantized weight rows.
/// `y` is `n x o`, `x` is `n x k`.
pub fn gemm_f32(m: &Mat, x: &[f32], n: usize, y: &mut [f32], threads: usize) -> Result<(), String> {
    let (o, k) = m.shape();
    if x.len() != n * k || y.len() != n * o {
        return Err(format!("gemm: x {} y {} for n {n} o {o} k {k}", x.len(), y.len()));
    }
    const RB: usize = 4;
    let yp = SendPtr(y.as_mut_ptr());
    let err = std::sync::Mutex::new(None::<String>);
    par_ranges(o.div_ceil(RB), threads, |blocks| {
        let yp = yp;
        let mut w = vec![0f32; RB * k];
        for b in blocks {
            let r0 = b * RB;
            let nr = RB.min(o - r0);
            for r in 0..nr {
                if let Err(e) = m.read_row(r0 + r, &mut w[r * k..(r + 1) * k]) {
                    *err.lock().unwrap() = Some(e);
                    return;
                }
            }
            for t in 0..n {
                let xr = &x[t * k..(t + 1) * k];
                for r in 0..nr {
                    let v = dot(xr, &w[r * k..(r + 1) * k]);
                    // SAFETY: column r0 + r of row t is written by this block only.
                    unsafe { *yp.0.add(t * o + r0 + r) = v };
                }
            }
        }
    });
    match err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// ggml `quantize_row_q8_0` (Arm NEON flavour): `d = amax / 127` stored as fp16, codes
/// `round_ties_even(x / d)` computed with the f32 `1/d` (not the rounded fp16 d).
pub fn quantize_q8_0(x: &[f32], d: &mut [f32], q: &mut [i8]) {
    let nb = x.len() / 32;
    debug_assert!(x.len() % 32 == 0 && d.len() == nb && q.len() == x.len());
    for b in 0..nb {
        let xb = &x[b * 32..(b + 1) * 32];
        let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
        d[b] = oxn::tiles::half_to_f32(oxn::tiles::f32_to_half(dd));
        for i in 0..32 {
            q[b * 32 + i] = (xb[i] * id).round_ties_even() as i8;
        }
    }
}

/// Emulation of sd.cpp's CPU arithmetic for a Q8_0 weight: the activation row is quantized to Q8_0
/// and each 32-block contributes `isum * (d_w * d_x)` (ggml `vec_dot_q8_0_q8_0`). Summation order
/// differs from ggml's SIMD order only by f32 rounding. Errors for non-Q8_0 weights.
pub fn gemm_q8(m: &Mat, x: &[f32], n: usize, y: &mut [f32], threads: usize) -> Result<(), String> {
    let (o, k) = m.shape();
    let wq = m.q8_0().ok_or("gemm_q8: weight is not Q8_0")?;
    if x.len() != n * k || y.len() != n * o || k % 32 != 0 {
        return Err(format!("gemm_q8: x {} y {} for n {n} o {o} k {k}", x.len(), y.len()));
    }
    let nb = k / 32;
    let mut xd = vec![0f32; n * nb];
    let mut xq = vec![0i8; n * k];
    {
        let (dp, qp) = (SendPtr(xd.as_mut_ptr()), xq.as_mut_ptr() as usize);
        par_ranges(n, threads, |ts| {
            let (dp, qp) = (dp, qp);
            for t in ts {
                // SAFETY: row t of each buffer belongs to this range only.
                let d = unsafe { std::slice::from_raw_parts_mut(dp.0.add(t * nb), nb) };
                let q = unsafe { std::slice::from_raw_parts_mut((qp as *mut i8).add(t * k), k) };
                quantize_q8_0(&x[t * k..(t + 1) * k], d, q);
            }
        });
    }
    let rb = QType::Q8_0.row_bytes(k);
    let yp = SendPtr(y.as_mut_ptr());
    par_ranges(o, threads, |rows| {
        let yp = yp;
        let mut wd = vec![0f32; nb];
        let mut wqv = vec![0i8; k];
        for r in rows {
            let row = &wq[r * rb..(r + 1) * rb];
            for b in 0..nb {
                let blk = &row[b * 34..(b + 1) * 34];
                wd[b] = oxn::tiles::half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..32 {
                    wqv[b * 32 + i] = blk[2 + i] as i8;
                }
            }
            for t in 0..n {
                let (xdq, xqq) = (&xd[t * nb..(t + 1) * nb], &xq[t * k..(t + 1) * k]);
                let mut acc = 0f32;
                for b in 0..nb {
                    let mut isum = 0i32;
                    let (a, c) = (&wqv[b * 32..(b + 1) * 32], &xqq[b * 32..(b + 1) * 32]);
                    for i in 0..32 {
                        isum += a[i] as i32 * c[i] as i32;
                    }
                    acc += isum as f32 * (wd[b] * xdq[b]);
                }
                // SAFETY: column r of row t is written by this range only.
                unsafe { *yp.0.add(t * o + r) = acc };
            }
        }
    });
    Ok(())
}

/// ggml `rms_norm` (sum of squares in f64, scale in f32) followed by the weight multiply, in place
/// on every `w.len()`-wide row of `x`.
pub fn rms_norm_rows(x: &mut [f32], w: &[f32], eps: f32) {
    let d = w.len();
    for row in x.chunks_exact_mut(d) {
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let mean = (ss / d as f64) as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        for (v, &g) in row.iter_mut().zip(w) {
            *v = (*v * scale) * g;
        }
    }
}

/// NeoX RoPE tables for positions `0..n`: `cos/sin[p][i]`, `i < head_dim/2`, with ggml's
/// recurrence `theta_{i+1} = theta_i * base^(-2/head_dim)` in f32.
pub fn rope_tables(n: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let theta_scale = base.powf(-2.0 / head_dim as f32);
    let (mut c, mut s) = (vec![0f32; n * half], vec![0f32; n * half]);
    for p in 0..n {
        let mut theta = p as f32;
        for i in 0..half {
            c[p * half + i] = theta.cos();
            s[p * half + i] = theta.sin();
            theta *= theta_scale;
        }
    }
    (c, s)
}

/// Rotate every head of a `[n][heads * head_dim]` buffer in place (NeoX pairs `(i, i + half)`).
pub fn rope_neox(x: &mut [f32], n: usize, heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for t in 0..n {
        for h in 0..heads {
            let v = &mut x[(t * heads + h) * head_dim..(t * heads + h + 1) * head_dim];
            for i in 0..half {
                let (a, b) = (v[i], v[i + half]);
                let (c, s) = (cos[t * half + i], sin[t * half + i]);
                v[i] = a * c - b * s;
                v[i + half] = a * s + b * c;
            }
        }
    }
}

/// Causal attention with masked (pad) keys, grouped KV heads. `q` is `[n][heads*hd]`, `k`/`v`
/// `[n][kv_heads*hd]`, output `[n][heads*hd]`. Softmax as ggml `soft_max_ext`: scale, max, exp in
/// f32, sum in f64. A query with no visible key gets zeros (cannot happen for klein: token 0 is real).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[bool],
    n: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    out: &mut [f32],
    threads: usize,
) {
    let group = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let (qd, kd) = (heads * hd, kv_heads * hd);
    let op = SendPtr(out.as_mut_ptr());
    par_ranges(heads * n, threads, |items| {
        let op = op;
        let mut p = vec![0f32; n];
        let mut acc = vec![0f32; hd];
        for it in items {
            let (h, i) = (it / n, it % n);
            let kh = h / group;
            let qi = &q[i * qd + h * hd..i * qd + (h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                if !mask[j] {
                    continue;
                }
                let s = dot(qi, &k[j * kd + kh * hd..j * kd + (kh + 1) * hd]) * scale;
                p[j] = s;
                mx = mx.max(s);
            }
            acc.iter_mut().for_each(|a| *a = 0.0);
            if mx.is_finite() {
                let mut sum = 0f64;
                for j in 0..=i {
                    if mask[j] {
                        let e = (p[j] - mx).exp();
                        p[j] = e;
                        sum += e as f64;
                    }
                }
                let inv = (1.0 / sum) as f32;
                for j in 0..=i {
                    if mask[j] {
                        let w = p[j] * inv;
                        let vj = &v[j * kd + kh * hd..j * kd + (kh + 1) * hd];
                        for d in 0..hd {
                            acc[d] += w * vj[d];
                        }
                    }
                }
            }
            // SAFETY: (query i, head h) is written by this item only.
            unsafe { std::ptr::copy_nonoverlapping(acc.as_ptr(), op.0.add(i * qd + h * hd), hd) };
        }
    });
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
