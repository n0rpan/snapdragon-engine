//! v0.1.3 APPEND-PREFILL micro-benchmark (model-free; needs a QUIET machine: hold the build
//! locks, ~3-6 min): the prefill attention core of ONE full-attention layer for an append of A
//! tokens at depth D, old tiled kernel (replica of `pf_attention_tiled`'s f16 unit at the default
//! mask: scores over all rows, FA-off softmax rows, V^T . p without position blocking) vs fa2
//! (`pfattn::Tile::run`, lane A) vs fa2x (`pfattn::Tile::run_x`, this lane), on a `sched::Pool`
//! with dynamically scheduled units as the engine runs them.
//!
//! Shapes: head_dim 256, 2 KV heads x 8 query heads, units of 8 tokens (64 columns), K rows
//! [kvh][n_ctx][256] and V^T rows with the engine's stride (n_ctx + 32), f16. K / V^T of the
//! layer (~290 MB at 136K) do not fit any cache, as in the engine.
//! Work: per (D, A) a sample of `OXE_PFBENCH_UNITS` units (default 96, spread evenly over the
//! append's units and both KV heads) is timed; us per append token = sample wall x (all units /
//! sampled) / A for ONE layer (x 10 for Ornith's attention layers).
//!
//! Run (Snapdragon, quiet):
//! `cargo test --release -p model --test pfattn_kbench -- --ignored --nocapture --test-threads 1`
//! Optional: OXE_PFBENCH_THREADS (default 12), OXE_PFBENCH_CPUS (cpu list), OXE_PFBENCH_DEPTH
//! (default "32768,65536,131072"), OXE_PFBENCH_APPEND (default "512,2048,8192"),
//! OXE_PFBENCH_X ("fold:ch" list, default "8:128,16:128,8:256"), OXE_PFBENCH_REPS (default 3),
//! OXE_PFBENCH_UNITS, OXE_PFBENCH_OLD=0 (skip the tiled replica: it needs ~54 MB scratch per thread).
//!
//! Kill (APPEND-PREFILL): at D 64K, A 2048, fa2x (fold 8, ch 128) < 1.5x the old tiled kernel ->
//! stop the lane; fa2x < 1.2x fa2 -> fa2x is not worth a second numerics class over fa2.

mod common;

use common::*;
use kernels_sys::ffi;
use model::engine::padded_n_kv;
use model::engine::pfattn::{Scratch, Tile, XParams};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

const TT: usize = 8;
const NCOL: usize = TT * GROUP;
/// the engine's PF_ATTN_PAD column offset (default mask has it)
const ATT_PAD: usize = 208;

fn env_list(k: &str, d: &str) -> Vec<usize> {
    std::env::var(k)
        .unwrap_or_else(|_| d.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kern {
    Old,
    Fa2,
    Fa2x(XParams),
}

impl Kern {
    fn name(self) -> String {
        match self {
            Kern::Old => "tiled".into(),
            Kern::Fa2 => "fa2".into(),
            Kern::Fa2x(x) => format!("fa2x f{} c{}", x.fold, x.ch),
        }
    }
}

/// Per-thread scratch of all kernels.
struct Thr {
    fa: Scratch,
    akq: Vec<f32>,
    ap16: Vec<u16>,
    wp: Vec<f32>,
    dp: Vec<f32>,
    aot: Vec<f32>,
}

struct Layer {
    n_ctx: usize,
    v_stride: usize,
    k: Vec<u16>,
    vt: Vec<u16>,
    q16: Vec<u16>,
}

impl Layer {
    fn new(n_ctx: usize) -> Layer {
        let v_stride = n_ctx + VPAD;
        let mut r = Lcg(0xa99e_d0);
        let blk = f16s(&(0..(1 << 16) + 97).map(|_| r.f()).collect::<Vec<_>>());
        let fill = |n: usize, salt: usize| -> Vec<u16> {
            let mut v = vec![0u16; n];
            let mut o = salt % 97;
            for c in v.chunks_mut(1 << 16) {
                c.copy_from_slice(&blk[o..o + c.len()]);
                o = (o + 31) % 97;
            }
            v
        };
        let q16 = f16s(&(0..NCOL * HD).map(|_| r.f() * 4.0).collect::<Vec<_>>());
        Layer {
            n_ctx,
            v_stride,
            k: fill(NKV * n_ctx * HD, 1),
            vt: fill(NKV * HD * v_stride, 2),
            q16,
        }
    }

    fn tile(&self, kvh: usize, pos0: usize) -> Tile {
        Tile {
            hd: HD,
            ncol: NCOL,
            group: GROUP,
            pos0,
            scale: 1.0 / (HD as f32).sqrt(),
            k: self.k[kvh * self.n_ctx * HD..].as_ptr(),
            vt: self.vt[kvh * HD * self.v_stride..].as_ptr(),
            v_stride: self.v_stride,
            n_ctx: self.n_ctx,
            q16: self.q16.as_ptr(),
        }
    }

    /// One unit of the old tiled path into thr.aot ([d][col]).
    unsafe fn old_unit(&self, t: &Tile, thr: &mut Thr) {
        // SAFETY: the caller's contract (tile within the layer, scratch sized for n_ctx).
        unsafe {
            let rows = t.last() + 1;
            let n32 = rows.div_ceil(32) * 32;
            let cs = self.n_ctx + ATT_PAD;
            let nkv = padded_n_kv(t.last(), self.n_ctx);
            let nf = [nkv as i32; NCOL];
            ffi::oxe_dot_f16_tile(
                HD as i32,
                t.k,
                HD as i64,
                rows as i32,
                t.q16,
                HD as i64,
                NCOL as i32,
                thr.akq.as_mut_ptr(),
                1,
                cs as i64,
                std::ptr::null(),
            );
            for col in 0..NCOL {
                let pos = t.pos0 + col / GROUP;
                ffi::oxe_attn_softmax_row_f16(
                    thr.akq.as_ptr().add(col * cs),
                    pos as i32,
                    nkv as i32,
                    t.scale,
                    thr.wp.as_mut_ptr(),
                    thr.dp.as_mut_ptr(),
                    thr.ap16.as_mut_ptr().add(col * cs),
                    n32 as i32,
                );
            }
            ffi::oxe_dot_f16_tile(
                n32 as i32,
                t.vt,
                t.v_stride as i64,
                HD as i32,
                thr.ap16.as_ptr(),
                cs as i64,
                NCOL as i32,
                thr.aot.as_mut_ptr(),
                NCOL as i64,
                1,
                nf.as_ptr(),
            );
        }
    }

    unsafe fn run(&self, k: Kern, t: &Tile, thr: &mut Thr) {
        // SAFETY: the caller's contract.
        unsafe {
            match k {
                Kern::Old => self.old_unit(t, thr),
                Kern::Fa2 => t.run(&mut thr.fa),
                Kern::Fa2x(x) => t.run_x(&mut thr.fa, x),
            }
        }
    }

    /// Output column-major [col][hd] of the unit (spot check).
    fn out(&self, k: Kern, thr: &Thr) -> Vec<f32> {
        match k {
            Kern::Old => (0..NCOL * HD)
                .map(|i| thr.aot[(i % HD) * NCOL + i / HD])
                .collect(),
            _ => (0..NCOL).flat_map(|c| thr.fa.out(c).to_vec()).collect(),
        }
    }
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        e += (x as f64 - y as f64).powi(2);
        n += (y as f64).powi(2);
    }
    (e / n.max(1e-300)).sqrt()
}

#[test]
#[ignore]
fn kbench_append_prefill_attention() {
    let nth = env_list("OXE_PFBENCH_THREADS", "12")[0].max(1);
    let depths = env_list("OXE_PFBENCH_DEPTH", "32768,65536,131072");
    let appends = env_list("OXE_PFBENCH_APPEND", "512,2048,8192");
    let reps = env_list("OXE_PFBENCH_REPS", "3")[0].max(1);
    let sample = env_list("OXE_PFBENCH_UNITS", "96")[0].max(1);
    let old_on = std::env::var("OXE_PFBENCH_OLD")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    let xs: Vec<XParams> = std::env::var("OXE_PFBENCH_X")
        .unwrap_or_else(|_| "8:128,16:128,8:256".into())
        .split(',')
        .filter_map(|s| {
            let (f, c) = s.trim().split_once(':')?;
            XParams::new(f.parse().ok()?, c.parse().ok()?)
        })
        .collect();
    let mut kerns = Vec::new();
    if old_on {
        kerns.push(Kern::Old);
    }
    kerns.push(Kern::Fa2);
    kerns.extend(xs.iter().map(|&x| Kern::Fa2x(x)));
    let n_ctx =
        (depths.iter().max().unwrap() + appends.iter().max().unwrap()).div_ceil(256) * 256 + 256;
    let cpus = std::env::var("OXE_PFBENCH_CPUS")
        .ok()
        .map(|s| sched::parse_cpu_list(&s).expect("OXE_PFBENCH_CPUS"))
        .unwrap_or_else(|| cpus(nth));
    let cpus = cpus[..nth.min(cpus.len())].to_vec();
    assert_eq!(
        cpus.len(),
        nth,
        "OXE_PFBENCH_CPUS has fewer cpus than threads"
    );
    println!("APPEND-PREFILL kbench: head_dim {HD}, {NKV} KV heads x {GROUP}, units of {TT} tokens, n_ctx {n_ctx}, threads {nth} on {cpus:?}, reps {reps}, sampled units {sample}");
    let layer = Layer::new(n_ctx);
    let thrs: Vec<Shared<Thr>> = (0..nth)
        .map(|_| {
            let big = if old_on { NCOL * (n_ctx + ATT_PAD) } else { 0 };
            Shared::new(vec![Thr {
                fa: Scratch::new(NCOL, HD),
                akq: vec![0.0; big],
                ap16: vec![0; big],
                wp: vec![0.0; if old_on { n_ctx } else { 0 }],
                dp: vec![0.0; if old_on { n_ctx } else { 0 }],
                aot: vec![0.0; HD * NCOL],
            }])
        })
        .collect();
    let mut table: Vec<(usize, usize, Kern, f64)> = Vec::new();
    std::thread::scope(|s| {
        s.spawn(|| {
            let pool = sched::Pool::new(&cpus, Duration::from_micros(200)).unwrap();
            for &d in &depths {
                for &a in &appends {
                    let total = NKV * a.div_ceil(TT);
                    let ns = sample.min(total);
                    let units: Vec<(usize, usize)> = (0..ns).map(|j| j * total / ns).map(|u| (u % NKV, d + (u / NKV) * TT)).collect();
                    let ctr = AtomicUsize::new(0);
                    let run = |k: Kern| {
                        ctr.store(0, Relaxed);
                        pool.run(|ctx| loop {
                            let i = ctr.fetch_add(1, Relaxed);
                            if i >= units.len() {
                                break;
                            }
                            let (kvh, pos0) = units[i];
                            // SAFETY: slot tid is this thread's; the layer holds every row read.
                            unsafe { layer.run(k, &layer.tile(kvh, pos0), &mut *thrs[ctx.tid].ptr()) };
                        });
                    };
                    // spot check on one unit (thread 0's scratch, outside the pool)
                    let spot: Vec<(Kern, Vec<f32>)> = kerns
                        .iter()
                        .map(|&k| {
                            // SAFETY: the pool is idle; slot 0 used by this thread only.
                            let thr = unsafe { &mut *thrs[0].ptr() };
                            unsafe { layer.run(k, &layer.tile(0, units[ns / 2].1), thr) };
                            (k, layer.out(k, thr))
                        })
                        .collect();
                    let base = &spot.iter().find(|(k, _)| *k == Kern::Fa2).unwrap().1;
                    let mut times = vec![Vec::new(); kerns.len()];
                    for k in &kerns {
                        run(*k); // warm: page faults, scratch
                    }
                    for r in 0..reps {
                        for j in 0..kerns.len() {
                            // rotate the order so drift hits every kernel
                            let ki = (j + r) % kerns.len();
                            let t0 = Instant::now();
                            run(kerns[ki]);
                            times[ki].push(t0.elapsed().as_secs_f64());
                        }
                    }
                    for (ki, k) in kerns.iter().enumerate() {
                        let w = median(&mut times[ki]);
                        let us_tok = w * 1e6 * (total as f64 / ns as f64) / a as f64;
                        // multiply-adds of the sampled units (scores + V^T . p over each column's rows)
                        let fma: f64 = units.iter().map(|&(_, p0)| (0..TT).map(|t| (p0 + t + 1) as f64).sum::<f64>() * GROUP as f64 * HD as f64 * 2.0).sum();
                        let dsp = rel(&spot[ki].1, base);
                        println!("D {d:>6} A {a:>5} {:<16} {us_tok:>10.2} us/append-token/layer  {:>7.1} GFMA/s  (vs fa2 rel {dsp:.1e})", k.name(), fma / w / 1e9);
                        table.push((d, a, *k, us_tok));
                    }
                }
            }
        })
        .join()
        .unwrap()
    });
    println!("speedups (us per append token, one layer; x10 = Ornith's attention per token):");
    let get = |d: usize, a: usize, k: Kern| {
        table
            .iter()
            .find(|r| r.0 == d && r.1 == a && r.2 == k)
            .map(|r| r.3)
    };
    for &d in &depths {
        for &a in &appends {
            for &x in &xs {
                let fx = get(d, a, Kern::Fa2x(x)).unwrap();
                let vs_old = get(d, a, Kern::Old)
                    .map(|o| format!("{:.2}x", o / fx))
                    .unwrap_or("-".into());
                let vs_fa2 = get(d, a, Kern::Fa2).unwrap() / fx;
                println!(
                    "D {d:>6} A {a:>5} fa2x f{} c{}: vs tiled {vs_old}, vs fa2 {vs_fa2:.2}x",
                    x.fold, x.ch
                );
            }
        }
    }
    let dx = XParams::default();
    if let (Some(o), Some(f), Some(x)) = (
        get(65536, 2048, Kern::Old),
        get(65536, 2048, Kern::Fa2),
        get(65536, 2048, Kern::Fa2x(dx)),
    ) {
        let (so, sf) = (o / x, f / x);
        println!(
            "KILL CHECK D 64K A 2048 fa2x(f{} c{}): vs tiled {so:.2}x -> {}; vs fa2 {sf:.2}x -> {}",
            dx.fold,
            dx.ch,
            if so < 1.5 {
                "KILL (< 1.5x): stop the lane"
            } else {
                "pass (>= 1.5x)"
            },
            if sf < 1.2 {
                "fa2x not worth a numerics class over fa2 (< 1.2x)"
            } else {
                "pass (>= 1.2x)"
            }
        );
    } else {
        println!("KILL CHECK needs D 65536, A 2048, the tiled replica and fa2x {dx:?} in the run");
    }
}
