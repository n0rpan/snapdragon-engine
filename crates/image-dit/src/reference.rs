//! Straightforward f32 implementation of one single-stream block (the math in `types.rs`).
//!
//! This is the yardstick for every error number, not the fast path. Per token, linear1 output is
//! computed and immediately turned into q/k/v rows and the gated MLP row, so the full
//! `[n_tokens][lin1_out]` matrix is never materialised. Parallelism is over contiguous token
//! ranges with `std::thread::scope`; results do not depend on the thread count.

use crate::arith::{round_activation, Arith, WType};
use crate::types::{BlockInput, BlockOutput, BlockWeightsF32, Dims};

/// Output of the linear1 stage for a token range: (q, k, v, m, (x_mod, q_norm, k_norm)) rows, each
/// concatenated in token order. The last three (LayerNorm+modulate output before rounding; q and
/// k after QK norm, before RoPE) are empty unless taps are on.
type Lin1Part = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, (Vec<f32>, Vec<f32>, Vec<f32>));

/// Per-stage intermediates of one block (W5-G taps), `[n_tokens][width]` each.
#[derive(Default, Debug, Clone)]
pub struct StageTaps {
    /// LayerNorm + modulate output (f32, before `arith` rounds it for linear1; sd.cpp's `dump.s0.xmod`).
    pub x_mod: Vec<f32>,
    /// q and k after QK RMS norm, before RoPE (sd.cpp's `query_norm` / `key_norm` outputs);
    /// v straight from linear1. `[n][hidden]`.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Run one block. `threads` is clamped to at least 1; results are bitwise identical for any count.
pub fn single_block_ref(w: &BlockWeightsF32, input: &BlockInput, threads: usize) -> BlockOutput {
    single_block_ref_arith(w, input, threads, Arith::F32)
}

/// [`single_block_ref`] with the activation arithmetic chosen by `arith` (linear1 and linear2 are
/// Q8_0 in klein GGUFs, so [`Arith::SdParity`] rounds both inputs to Q8_0 blocks).
pub fn single_block_ref_arith(w: &BlockWeightsF32, input: &BlockInput, threads: usize, arith: Arith) -> BlockOutput {
    block_impl(w, input, threads, arith, false).0
}

/// [`single_block_ref_arith`] that also returns the per-stage taps.
pub fn single_block_ref_taps(
    w: &BlockWeightsF32,
    input: &BlockInput,
    threads: usize,
    arith: Arith,
) -> (BlockOutput, StageTaps) {
    block_impl(w, input, threads, arith, true)
}

fn block_impl(
    w: &BlockWeightsF32,
    input: &BlockInput,
    threads: usize,
    arith: Arith,
    want_taps: bool,
) -> (BlockOutput, StageTaps) {
    let d = w.dims;
    d.check();
    let (h, n) = (d.hidden, input.n_tokens);
    assert_eq!(w.w1.len(), d.lin1_out() * h, "w1 length");
    assert_eq!(w.w2.len(), h * d.lin2_in(), "w2 length");
    assert_eq!(w.q_scale.len(), d.head_dim, "q_scale length");
    assert_eq!(w.k_scale.len(), d.head_dim, "k_scale length");
    assert_eq!(input.x.len(), n * h, "x length");
    assert_eq!(input.pe.len(), n * (d.head_dim / 2) * 4, "pe length");
    assert!(
        input.shift.len() == h && input.scale.len() == h && input.gate.len() == h,
        "modulation lengths"
    );

    let ranges = split_ranges(n, threads.max(1));

    // Stage 1: LayerNorm + modulate, linear1, QK norm, RoPE, SwiGLU, per token range.
    let parts: Vec<Lin1Part> = std::thread::scope(|sc| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(s, e)| sc.spawn(move || lin1_stage(w, input, s, e, arith, want_taps)))
            .collect();
        handles
            .into_iter()
            .map(|hd| hd.join().expect("lin1 worker panicked"))
            .collect()
    });
    let mut q: Vec<f32> = Vec::with_capacity(n * h);
    let mut k: Vec<f32> = Vec::with_capacity(n * h);
    let mut v: Vec<f32> = Vec::with_capacity(n * h);
    let mut m: Vec<f32> = Vec::with_capacity(n * d.mlp);
    let (mut x_mod, mut q_norm, mut k_norm) = (Vec::new(), Vec::new(), Vec::new());
    for (pq, pk, pv, pm, (px, pqn, pkn)) in parts {
        q.extend(pq);
        k.extend(pk);
        v.extend(pv);
        m.extend(pm);
        x_mod.extend(px);
        q_norm.extend(pqn);
        k_norm.extend(pkn);
    }

    // Stage 2: attention, split over query tokens (q/k/v shared read-only).
    let (qr, kr, vr) = (&q[..], &k[..], &v[..]);
    let attn_parts: Vec<Vec<f32>> = std::thread::scope(|sc| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(s, e)| sc.spawn(move || attn_stage(d, qr, kr, vr, n, s, e)))
            .collect();
        handles
            .into_iter()
            .map(|hd| hd.join().expect("attention worker panicked"))
            .collect()
    });
    let mut attn: Vec<f32> = Vec::with_capacity(n * h);
    for p in attn_parts {
        attn.extend(p);
    }

    // Stage 3: linear2 over concat(attn, m), gate and residual, per token range.
    let (ar, mr) = (&attn[..], &m[..]);
    let lin2_parts: Vec<(Vec<f32>, Vec<f32>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = ranges
            .iter()
            .map(|&(s, e)| sc.spawn(move || lin2_stage(w, input, ar, mr, s, e, arith)))
            .collect();
        handles
            .into_iter()
            .map(|hd| hd.join().expect("linear2 worker panicked"))
            .collect()
    });
    let mut z: Vec<f32> = Vec::with_capacity(n * h);
    let mut out: Vec<f32> = Vec::with_capacity(n * h);
    for (pz, po) in lin2_parts {
        z.extend(pz);
        out.extend(po);
    }

    let taps = if want_taps { StageTaps { x_mod, q: q_norm, k: k_norm, v } } else { StageTaps::default() };
    (BlockOutput { out, attn, mlp: m, z }, taps)
}

/// Contiguous `[start, end)` token ranges covering `0..n`, at most `parts` of them.
fn split_ranges(n: usize, parts: usize) -> Vec<(usize, usize)> {
    let chunk = n.div_ceil(parts.max(1)).max(1);
    (0..n).step_by(chunk).map(|s| (s, (s + chunk).min(n))).collect()
}

fn lin1_stage(w: &BlockWeightsF32, input: &BlockInput, s: usize, e: usize, arith: Arith, want_taps: bool) -> Lin1Part {
    let d = w.dims;
    let (h, hd, mlp) = (d.hidden, d.head_dim, d.mlp);
    let half = hd / 2;
    let lin1 = d.lin1_out();
    let rows = e - s;
    let mut q = Vec::with_capacity(rows * h);
    let mut k = Vec::with_capacity(rows * h);
    let mut v = Vec::with_capacity(rows * h);
    let mut m = Vec::with_capacity(rows * mlp);
    let cap = if want_taps { rows * h } else { 0 };
    let (mut xs, mut qn, mut kn) = (Vec::with_capacity(cap), Vec::with_capacity(cap), Vec::with_capacity(cap));
    // Tokens in groups of TOKEN_GROUP so each weight row is reused from cache; every value is the
    // same `dot` call as per-token processing, so results are bitwise unchanged.
    let mut xg = vec![0f32; TOKEN_GROUP * h];
    let mut yg = vec![0f32; TOKEN_GROUP * lin1];
    let mut g0 = s;
    while g0 < e {
        let g1 = (g0 + TOKEN_GROUP).min(e);
        for t in g0..g1 {
            let xmod = &mut xg[(t - g0) * h..(t - g0 + 1) * h];
            layer_norm_modulate(&input.x[t * h..(t + 1) * h], &input.shift, &input.scale, xmod);
            if want_taps {
                xs.extend_from_slice(xmod); // before rounding, as sd.cpp's dump.s0.xmod
            }
            round_activation(arith, WType::Q8_0, xmod);
        }
        for r in 0..lin1 {
            let wr = &w.w1[r * h..(r + 1) * h];
            for t in g0..g1 {
                yg[(t - g0) * lin1 + r] = dot(wr, &xg[(t - g0) * h..(t - g0 + 1) * h]);
            }
        }
        for t in g0..g1 {
            let y = &yg[(t - g0) * lin1..(t - g0 + 1) * lin1];
            let pe_t = &input.pe[t * half * 4..(t + 1) * half * 4];
            let mut qrow = y[..h].to_vec();
            let mut krow = y[h..2 * h].to_vec();
            for hh in 0..d.heads {
                let (a, b) = (hh * hd, (hh + 1) * hd);
                rms_norm_scale(&mut qrow[a..b], &w.q_scale);
                rms_norm_scale(&mut krow[a..b], &w.k_scale);
            }
            if want_taps {
                qn.extend_from_slice(&qrow);
                kn.extend_from_slice(&krow);
            }
            for hh in 0..d.heads {
                let (a, b) = (hh * hd, (hh + 1) * hd);
                rope(&mut qrow[a..b], pe_t);
                rope(&mut krow[a..b], pe_t);
            }
            q.extend_from_slice(&qrow);
            k.extend_from_slice(&krow);
            v.extend_from_slice(&y[2 * h..3 * h]);
            for j in 0..mlp {
                m.push(silu(y[3 * h + j]) * y[3 * h + mlp + j]);
            }
        }
        g0 = g1;
    }
    (q, k, v, m, (xs, qn, kn))
}

/// Tokens per weight-row pass in the reference linears (cache reuse only; no effect on values).
const TOKEN_GROUP: usize = 8;

/// Attention output rows `[s, e)` (query tokens); keys and values run over all `n` tokens.
fn attn_stage(d: Dims, q: &[f32], k: &[f32], v: &[f32], n: usize, s: usize, e: usize) -> Vec<f32> {
    let (h, hd) = (d.hidden, d.head_dim);
    let sqrt_hd = (hd as f32).sqrt();
    let mut out = vec![0f32; (e - s) * h];
    let mut logits = vec![0f64; n];
    let mut acc = vec![0f64; hd];
    for tq in s..e {
        for hh in 0..d.heads {
            let base = hh * hd;
            let qv = &q[tq * h + base..tq * h + base + hd];
            let mut mx = f64::NEG_INFINITY;
            for tk in 0..n {
                let kv = &k[tk * h + base..tk * h + base + hd];
                let l = (dot(qv, kv) / sqrt_hd) as f64;
                logits[tk] = l;
                mx = mx.max(l);
            }
            let mut sum = 0f64;
            for l in logits.iter_mut() {
                *l = (*l - mx).exp();
                sum += *l;
            }
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            for tk in 0..n {
                let p = logits[tk] / sum;
                let vv = &v[tk * h + base..tk * h + base + hd];
                for dd in 0..hd {
                    acc[dd] += p * vv[dd] as f64;
                }
            }
            let orow = &mut out[(tq - s) * h + base..(tq - s) * h + base + hd];
            for dd in 0..hd {
                orow[dd] = acc[dd] as f32;
            }
        }
    }
    out
}

/// The reference attention of all `n` query tokens (F-IMG-ATTN tests compare the fast path's float attention to it).
#[cfg(test)]
pub(crate) fn attn_stage_for_tests(d: Dims, q: &[f32], k: &[f32], v: &[f32], n: usize) -> Vec<f32> {
    attn_stage(d, q, k, v, n, 0, n)
}

/// Returns (z, out) rows `[s, e)`.
fn lin2_stage(
    w: &BlockWeightsF32,
    input: &BlockInput,
    attn: &[f32],
    m: &[f32],
    s: usize,
    e: usize,
    arith: Arith,
) -> (Vec<f32>, Vec<f32>) {
    let d = w.dims;
    let (h, mlp) = (d.hidden, d.mlp);
    let kin = d.lin2_in();
    let mut cat = vec![0f32; TOKEN_GROUP * kin];
    let mut z = vec![0f32; (e - s) * h];
    let mut out = vec![0f32; (e - s) * h];
    let mut g0 = s;
    while g0 < e {
        let g1 = (g0 + TOKEN_GROUP).min(e);
        for t in g0..g1 {
            let c = &mut cat[(t - g0) * kin..(t - g0 + 1) * kin];
            c[..h].copy_from_slice(&attn[t * h..(t + 1) * h]);
            c[h..].copy_from_slice(&m[t * mlp..(t + 1) * mlp]);
            round_activation(arith, WType::Q8_0, c);
        }
        for r in 0..h {
            let wr = &w.w2[r * kin..(r + 1) * kin];
            for t in g0..g1 {
                let zr = dot(wr, &cat[(t - g0) * kin..(t - g0 + 1) * kin]);
                z[(t - s) * h + r] = zr;
                out[(t - s) * h + r] = input.x[t * h + r] + input.gate[r] * zr;
            }
        }
        g0 = g1;
    }
    (z, out)
}

/// `y = (LN(x) + LN(x) * scale) + shift` with LN over the slice, no affine, eps 1e-6.
/// Mean and variance are computed in f64.
pub(crate) fn layer_norm_modulate(x: &[f32], shift: &[f32], scale: &[f32], out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let dv = v as f64 - mean;
            dv * dv
        })
        .sum::<f64>()
        / n;
    let denom = (var + 1e-6).sqrt();
    for (((o, &xv), &sc), &sh) in out.iter_mut().zip(x).zip(scale).zip(shift) {
        let y = ((xv as f64 - mean) / denom) as f32;
        *o = (y + y * sc) + sh;
    }
}

/// In place: `x / sqrt(mean(x^2) + 1e-6)` over the slice (mean in f64), then times `scale[d]`.
pub(crate) fn rms_norm_scale(x: &mut [f32], scale: &[f32]) {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let denom = (ms + 1e-6).sqrt();
    for (v, &sc) in x.iter_mut().zip(scale) {
        *v = ((*v as f64 / denom) as f32) * sc;
    }
}

/// Interleaved RoPE in place on one head. `pe` holds `[cos, -sin, sin, cos]` per pair
/// (`pe.len() == x.len() / 2 * 4`).
pub(crate) fn rope(x: &mut [f32], pe: &[f32]) {
    for (pair, p) in x.chunks_exact_mut(2).zip(pe.chunks_exact(4)) {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = p[0] * a + p[1] * b;
        pair[1] = p[2] * a + p[3] * b;
    }
}

/// `a / (1 + exp(-a))`.
pub(crate) fn silu(a: f32) -> f32 {
    a / (1.0 + (-a).exp())
}

/// Dot product: 8 independent f32 lanes over the element index, lanes summed in f64 at the end.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let main = n - n % 8;
    let mut acc = [0.0f32; 8];
    for (ca, cb) in a[..main].chunks_exact(8).zip(b[..main].chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += ca[l] * cb[l];
        }
    }
    for (l, (x, y)) in a[main..].iter().zip(&b[main..]).enumerate() {
        acc[l] += x * y;
    }
    acc.iter().map(|&v| v as f64).sum::<f64>() as f32
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
