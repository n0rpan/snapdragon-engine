//! Faster CPU attention for the encoder (TE-CPU-674), BITWISE equal to [`crate::cpu::attention`].
//!
//! The reference walks one (head, query) at a time and reloads every key/value row for each of the
//! `heads / kv_heads` query heads that share it. Here one work item is (kv head, query): each K and
//! V row is loaded once and used by the whole GQA group (4 heads for Qwen3-4B), and the q.k dots of
//! the group run interleaved, so the 8-lane accumulators of all heads stay in registers.
//!
//! Every output element keeps the reference's exact arithmetic: the same 8-lane partial sums in the
//! same order, the same lane tree and tail, the same f32 max / f64 sum softmax, and the same
//! key-ordered `acc[d] += w * v[d]` (Rust never contracts a multiply-add, so no FMA appears in either).
//! The result is therefore independent of the group size and thread count, and tests compare bits.

use crate::cpu::{dot, par_ranges, SendPtr};

/// [`dot`] of `G` query rows against one key row: per query exactly `dot(q[g], k)`.
#[inline(always)]
fn dots<const G: usize>(q: [&[f32]; G], k: &[f32]) -> [f32; G] {
    let len = k.len();
    let full = len / 8 * 8;
    let mut acc = [[0f32; 8]; G];
    let mut c = 0;
    while c < full {
        let kc = &k[c..c + 8];
        for g in 0..G {
            let qc = &q[g][c..c + 8];
            for i in 0..8 {
                acc[g][i] += qc[i] * kc[i];
            }
        }
        c += 8;
    }
    let mut out = [0f32; G];
    for g in 0..G {
        let a = &acc[g];
        let mut s = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
        for i in full..len {
            s += q[g][i] * k[i];
        }
        out[g] = s;
    }
    out
}

/// Same contract and result bits as [`crate::cpu::attention`].
#[allow(clippy::too_many_arguments)]
pub fn attention_gqa(
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
    match group {
        1 => run::<1>(q, k, v, mask, n, heads, kv_heads, hd, out, threads),
        2 => run::<2>(q, k, v, mask, n, heads, kv_heads, hd, out, threads),
        4 => run::<4>(q, k, v, mask, n, heads, kv_heads, hd, out, threads),
        8 => run::<8>(q, k, v, mask, n, heads, kv_heads, hd, out, threads),
        _ => crate::cpu::attention(q, k, v, mask, n, heads, kv_heads, hd, out, threads),
    }
}

#[allow(clippy::too_many_arguments)]
fn run<const G: usize>(
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
    debug_assert_eq!(heads, kv_heads * G);
    let scale = 1.0 / (hd as f32).sqrt();
    let (qd, kd) = (heads * hd, kv_heads * hd);
    assert!(q.len() >= n * qd && k.len() >= n * kd && v.len() >= n * kd && out.len() >= n * qd && mask.len() >= n);
    // the visible keys (mask) in ascending order, shared by every query: query i uses the prefix <= i
    let keys: Vec<usize> = (0..n).filter(|&j| mask[j]).collect();
    let op = SendPtr(out.as_mut_ptr());
    // items are kv-head major like the reference's head-major order; long (late) queries are spread
    // over the threads the same way
    par_ranges(kv_heads * n, threads, |items| {
        let op = op;
        let mut p = vec![0f32; G * n];
        let mut acc = vec![0f32; G * hd];
        for it in items {
            let (kh, i) = (it / n, it % n);
            let h0 = kh * G;
            let qi: [&[f32]; G] = std::array::from_fn(|g| &q[i * qd + (h0 + g) * hd..i * qd + (h0 + g + 1) * hd]);
            let vis = &keys[..keys.partition_point(|&j| j <= i)];
            let mut mx = [f32::NEG_INFINITY; G];
            for &j in vis {
                let s = dots::<G>(qi, &k[j * kd + kh * hd..j * kd + (kh + 1) * hd]);
                for g in 0..G {
                    let sg = s[g] * scale;
                    p[g * n + j] = sg;
                    mx[g] = mx[g].max(sg);
                }
            }
            acc.iter_mut().for_each(|a| *a = 0.0);
            let mut inv = [0f32; G];
            let mut live = [false; G];
            for g in 0..G {
                if mx[g].is_finite() {
                    live[g] = true;
                    let mut sum = 0f64;
                    for &j in vis {
                        let e = (p[g * n + j] - mx[g]).exp();
                        p[g * n + j] = e;
                        sum += e as f64;
                    }
                    inv[g] = (1.0 / sum) as f32;
                }
            }
            for &j in vis {
                let vj = &v[j * kd + kh * hd..j * kd + (kh + 1) * hd];
                for g in 0..G {
                    if !live[g] {
                        continue;
                    }
                    let w = p[g * n + j] * inv[g];
                    let a = &mut acc[g * hd..(g + 1) * hd];
                    for d in 0..hd {
                        a[d] += w * vj[d];
                    }
                }
            }
            for g in 0..G {
                // SAFETY: (query i, head h0 + g) is written by this item only.
                unsafe { std::ptr::copy_nonoverlapping(acc[g * hd..].as_ptr(), op.0.add(i * qd + (h0 + g) * hd), hd) };
            }
        }
    });
}
