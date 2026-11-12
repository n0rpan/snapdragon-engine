//! O-70 micro-benchmark (model-free, needs a QUIET machine: hold all build locks, <= 5 min):
//! one decode / verify attention phase of the full-attention layers at production shapes,
//! old tiled path (replica of the engine's f16 `attn_core_tiled` steps, 3 barriers) vs the fused
//! single-pass kernel (`model::engine::fattn`, 1 barrier), on a `sched::Pool` as the engine runs
//! them (one pool job per layer phase; both include one dispatch + join).
//!
//! Shapes: head_dim 256, 2 KV heads x 8 query heads, n_ctx 32768 strides (V^T rows + 32 pad),
//! context 6144 / 8192 / 16384 / 32000, 1 and 2 lanes (plain decode / MTP depth-1 verify).
//! `OXE_KBENCH_LAYERS` (default 10) distinct layers are cycled, so K / V come from DRAM as in the
//! engine (10 x 21 MB at 8K), not from L2.
//!
//! Run:
//! `OXE_KBENCH_THREADS=8 cargo test --release -p model --test fattn_kbench -- --ignored --nocapture --test-threads 1`
//! Optional: OXE_KBENCH_REPS (default 30), OXE_KBENCH_THREADS (default "8"; comma list),
//! OXE_KBENCH_CPUS (cpu list for the pool, e.g. "4-11"; default: the top cores),
//! OXE_KBENCH_CH (fused chunk sizes, default "256"; e.g. "128,256,512"), OXE_KBENCH_CTX (comma list),
//! OXE_KBENCH_LAYERS.
//!
//! Kill (KV-LEVERS-PRICING R1): fused attention-time cut < 20% for 2 lanes at 8K (printed).

mod common;

use common::*;
use std::time::{Duration, Instant};

fn env_list(k: &str, d: &str) -> Vec<usize> {
    std::env::var(k).unwrap_or_else(|_| d.to_string()).split(',').filter_map(|s| s.trim().parse().ok()).collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn kbench_fused_verify_attention() {
    const N_CTX: usize = 32768;
    let reps = env_list("OXE_KBENCH_REPS", "30")[0].max(3);
    let threads = env_list("OXE_KBENCH_THREADS", "8");
    let chs = env_list("OXE_KBENCH_CH", "256");
    let ctxs = env_list("OXE_KBENCH_CTX", "6144,8192,16384,32000");
    let nlayer = env_list("OXE_KBENCH_LAYERS", "10")[0].max(1);
    let cpu_list = std::env::var("OXE_KBENCH_CPUS").ok().map(|s| sched::parse_cpu_list(&s).expect("OXE_KBENCH_CPUS"));
    println!("O-70 kbench: head_dim {HD}, {NKV} KV heads x {GROUP}, n_ctx {N_CTX}, {nlayer} layers cycled, reps {reps}, threads {threads:?}, chunks {chs:?}");
    println!("{:>6} {:>5} {:>3} {:>5} {:>10} {:>10} {:>9} {:>9} {:>7} {:>8}", "ctx", "lanes", "thr", "ch", "old_us", "fused_us", "old_ns/p", "fus_ns/p", "cut%", "fus_GB/s");
    let mut verdict = Vec::new();
    for &ctx in &ctxs {
        let pos0 = ctx - 1;
        let mut layers: Vec<Case> = (0..nlayer).map(|l| Case::new(N_CTX, 2, pos0, Kind::Random, 4.0, 0xbe7c4 + l as u64, &[])).collect();
        for nl in [1usize, 2] {
            for c in layers.iter_mut() {
                c.nl = nl;
            }
            for &nth in &threads {
                let cpus = cpu_list.clone().map(|v| v[..nth.min(v.len())].to_vec()).unwrap_or_else(|| cpus(nth));
                assert_eq!(cpus.len(), nth, "OXE_KBENCH_CPUS has fewer cpus than threads");
                let rows: Vec<(usize, f64, f64)> = std::thread::scope(|s| {
                    s.spawn(|| {
                        let pool = sched::Pool::new(&cpus, Duration::from_micros(200)).unwrap();
                        let or = OldRun::new(&layers[0], nth);
                        let mut out = Vec::new();
                        for &ch in &chs {
                            let fr = FusedRun::new(&layers[0], nth);
                            let fas: Vec<_> = layers.iter().map(|c| fr.fa(c, ch)).collect();
                            // warm: page faults, scratch growth
                            for (c, fa) in layers.iter().zip(&fas) {
                                or.run(&pool, c);
                                fr.run(&pool, fa);
                            }
                            let (mut to, mut tf) = (Vec::new(), Vec::new());
                            for r in 0..reps {
                                // alternate the order (ABBA) so drift hits both
                                for k in 0..2 {
                                    let old_first = (r + k) % 2 == 0;
                                    let t0 = Instant::now();
                                    for (c, fa) in layers.iter().zip(&fas) {
                                        if old_first {
                                            or.run(&pool, c);
                                        } else {
                                            fr.run(&pool, fa);
                                        }
                                    }
                                    let dt = t0.elapsed().as_secs_f64() / nlayer as f64;
                                    if old_first {
                                        to.push(dt);
                                    } else {
                                        tf.push(dt);
                                    }
                                }
                            }
                            out.push((ch, median(&mut to), median(&mut tf)));
                        }
                        out
                    })
                    .join()
                    .unwrap()
                });
                for (ch, to, tf) in rows {
                    let cut = 100.0 * (1.0 - tf / to);
                    let bytes = (ctx * NKV * HD * 2 * 2) as f64; // K + V^T of one layer
                    println!(
                        "{ctx:>6} {nl:>5} {nth:>3} {ch:>5} {:>10.1} {:>10.1} {:>9.3} {:>9.3} {cut:>7.1} {:>8.1}",
                        to * 1e6,
                        tf * 1e6,
                        to * 1e9 / ctx as f64,
                        tf * 1e9 / ctx as f64,
                        bytes / tf / 1e9
                    );
                    if nl == 2 && (6144..=8192).contains(&ctx) {
                        verdict.push(format!("ctx {ctx} lanes 2 threads {nth} ch {ch}: cut {cut:.1}% -> {}", if cut < 20.0 { "KILL (< 20%)" } else { "pass (>= 20%)" }));
                    }
                }
            }
        }
    }
    println!("per-layer times; x10 attention layers = one verify step's attention. Kill rule (R1): < 20% cut at 8K, 2 lanes.");
    for v in verdict {
        println!("{v}");
    }
}
