//! O-70 model-free tests of the fused single-pass decode / verify attention
//! (`model::engine::fattn`) at production shapes (head_dim 256, 2 KV heads x 8 query heads,
//! 1 and 2 verify lanes), against the old tiled path (a replica of the engine's f16 steps) and an
//! exact f64 attention over the same f16 inputs. Thread counts 1 / 4 / 8.
//!
//! Exactness: neither kernel can agree with f64 to 1e-6 on random data (both use f16 q / K / V
//! values, f16 accumulators and f16 weights, as ggml's FA-off path does), so random data checks a
//! bound and "no worse than the old path"; the exact-count fixtures (uniform attention or +-128
//! logit spikes over one-hot V rows, where every product and sum is exact in f16 / f32) check the
//! fused kernel to 1e-6 against f64, which catches any masking, chunk-boundary or merge slip.
//!
//! `cargo test --release -p model --test fattn -- --nocapture --test-threads 1`

mod common;

use common::*;
use std::time::Duration;

const CTX: [usize; 8] = [1, 31, 32, 33, 2048, 6144, 16384, 32000];
const THREADS: [usize; 3] = [1, 4, 8];

fn n_ctx_for(pos0: usize, nl: usize) -> usize {
    (pos0 + nl).div_ceil(256).max(1) * 256
}

/// Runs `f` with a fresh pool of n threads on its own scoped thread (the pool pins its caller).
fn with_pool<R: Send>(n: usize, f: impl FnOnce(&sched::Pool) -> R + Send) -> R {
    std::thread::scope(|s| {
        s.spawn(move || {
            let pool = sched::Pool::new(&cpus(n), Duration::from_micros(100)).unwrap();
            f(&pool)
        })
        .join()
        .unwrap()
    })
}

fn fused_out(c: &Case, nth: usize, ch: usize) -> Vec<f32> {
    with_pool(nth, |pool| {
        let fr = FusedRun::new(c, nth);
        let fa = fr.fa(c, ch);
        fr.run(pool, &fa);
        fr.out.into_inner()
    })
}

fn old_out(c: &Case, nth: usize) -> Vec<f32> {
    with_pool(nth, |pool| {
        let or = OldRun::new(c, nth);
        or.run(pool, c);
        or.out.into_inner()
    })
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Random q / K / V at every listed context, 1 and 2 lanes: the fused output is finite, the same
/// bits at 1 / 4 / 8 threads, within 3e-3 (relative L2 per head; 1.2e-2 at the peakiest qs 16) of exact f64 attention, and no
/// worse than the old tiled path's own error (+ slack for f16 noise).
#[test]
fn fused_random_matches_f64_and_old_path() {
    for &qs in &[1.0f32, 4.0, 16.0] {
        for &n in &CTX {
            for nl in [1usize, 2] {
                let pos0 = n - 1;
                let c = Case::new(n_ctx_for(pos0, nl), nl, pos0, Kind::Random, qs, 0x5eed ^ (n as u64) ^ ((nl as u64) << 40), &[]);
                let r = ref_f64(&c);
                let old = old_out(&c, 8);
                let f1 = fused_out(&c, 1, 256);
                assert!(f1.iter().all(|v| v.is_finite()), "n {n} nl {nl}: non-finite output");
                for &nth in &THREADS[1..] {
                    assert!(same_bits(&fused_out(&c, nth, 256), &f1), "n {n} nl {nl} qs {qs}: {nth} threads differ from 1");
                }
                let (ef, af) = rel_err(&f1, &r);
                let (eo, ao) = rel_err(&old, &r);
                let (efo, _) = rel_err(&f1, &old.iter().map(|&x| x as f64).collect::<Vec<_>>());
                println!("qs {qs:>4} n {n:>5} lanes {nl}: rel L2 vs f64 fused {ef:.2e} (abs {af:.1e}) old {eo:.2e} (abs {ao:.1e}); fused vs old {efo:.2e}");
                // peakier softmax (qs 16: logits ~ +-5) -> f16 score / weight rounding dominates both kernels
                let tol = 3e-3 * (qs as f64 / 4.0).max(1.0);
                assert!(ef <= tol, "n {n} nl {nl} qs {qs}: fused error {ef:.3e} vs f64 (bound {tol:.1e})");
                assert!(ef <= eo * 1.25 + 3e-4, "n {n} nl {nl} qs {qs}: fused error {ef:.3e} worse than old {eo:.3e}");
            }
        }
    }
}

/// Uniform attention over one-hot V rows: every output is count / n exactly; fused agrees with
/// f64 to 1e-6 relative at every context (plus the 128 / 256 / 512 chunk boundaries), 1 and 2
/// lanes, chunk sizes 128 / 256 / 512 and 1 / 4 / 8 threads. A masked position read, a dropped
/// or doubled chunk, or a wrong lane mask moves some count by 1.
#[test]
fn fused_onehot_exact_counts() {
    let mut ctx: Vec<usize> = CTX.to_vec();
    ctx.extend([127, 128, 129, 255, 256, 257, 511, 512, 513]);
    for &n in &ctx {
        for nl in [1usize, 2] {
            let pos0 = n - 1;
            let c = Case::new(n_ctx_for(pos0, nl), nl, pos0, Kind::OneHot, 0.0, 7, &[]);
            let r = ref_f64(&c);
            for ch in [128usize, 256, 512] {
                for &nth in &THREADS {
                    let f = fused_out(&c, nth, ch);
                    for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                        assert!((x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12, "n {n} nl {nl} ch {ch} nth {nth}: out[{i}] = {x} want {y}");
                    }
                }
            }
        }
    }
}

/// Large logits (+-128 after scaling: exp overflows / underflows without max subtraction) and
/// masking: +128 spikes at chunk / 32-block boundaries and at the last cache row, a spike inside a
/// chunk of -128 logits, and a spike at the second lane's own position (lane 0 must not see it).
/// Exact counts again: 1e-6 against f64; old path within its f16 noise.
#[test]
fn fused_spikes_masking_and_large_logits() {
    for &n in &[1usize, 2, 33, 300, 600, 2048, 6144, 16384] {
        for nl in [1usize, 2] {
            let pos0 = n - 1;
            let mut sp = vec![0usize, 31, 32, 127, 128, 255, 383, 512];
            if pos0 >= 1 {
                sp.push(pos0 - 1);
            }
            sp.push(pos0);
            sp.retain(|&p| p <= pos0);
            let c = Case::new(n_ctx_for(pos0, nl), nl, pos0, Kind::Spikes, 0.0, 11, &sp);
            let r = ref_f64(&c);
            let old = old_out(&c, 4);
            let (eo, _) = rel_err(&old, &r);
            assert!(eo < 5e-3, "n {n} nl {nl}: old path error {eo:.3e} on spikes");
            for &nth in &THREADS {
                let f = fused_out(&c, nth, 256);
                for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                    assert!((x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12, "n {n} nl {nl} nth {nth}: out[{i}] = {x} want {y}");
                }
            }
        }
    }
}

/// Batch invariance: lane 0 of a 2-lane run equals a 1-lane run at the same position bit for
/// bit, and lane 1 equals a 1-lane run at pos0 + 1 (the engine's MTP verify relies on this kind of
/// equality for its rollback / replay tests), including when pos0 + 1 opens a new 32-block or chunk.
#[test]
fn fused_lane_results_are_batch_invariant() {
    for &n in &[31usize, 32, 255, 256, 2047, 6144] {
        let pos0 = n - 1;
        let n_ctx = n_ctx_for(pos0, 2) + 256;
        let two = Case::new(n_ctx, 2, pos0, Kind::Random, 4.0, 99, &[]);
        let f2 = fused_out(&two, 4, 256);
        // the same data seen as two single-lane runs: cache rows filled from the local copies
        for ln in 0..2 {
            let mut one = Case::new(n_ctx, 2, pos0, Kind::Random, 4.0, 99, &[]);
            let p = pos0 + ln;
            for h in 0..NKV {
                for i in 0..ln {
                    let row = two.k_local[i][h * HD..(h + 1) * HD].to_vec();
                    one.k[(h * n_ctx + pos0 + i) * HD..(h * n_ctx + pos0 + i + 1) * HD].copy_from_slice(&row);
                }
            }
            one.k_local = vec![two.k_local[ln].clone()];
            one.nl = 1;
            one.pos0 = p;
            // q / gate columns of lane ln -> lane 0
            for h in 0..NKV {
                for g in 0..GROUP {
                    let (src, dst) = (h * two.cpk + ln * GROUP + g, h * one.cpk + g);
                    let q = two.q16[src * HD..(src + 1) * HD].to_vec();
                    one.q16[dst * HD..(dst + 1) * HD].copy_from_slice(&q);
                    let gt = two.gate[src * HD..(src + 1) * HD].to_vec();
                    one.gate[dst * HD..(dst + 1) * HD].copy_from_slice(&gt);
                }
            }
            let f1 = fused_out(&one, 4, 256);
            let lane = &f2[ln * NHEAD * HD..(ln + 1) * NHEAD * HD];
            assert!(same_bits(lane, &f1), "n {n}: lane {ln} of the 2-lane run differs from a 1-lane run");
        }
    }
}
