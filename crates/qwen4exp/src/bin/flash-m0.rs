//! flash-m0: the Flash M0 timing prototype.
//!
//!   flash-m0 sidecar [--model-dir D] [--out F] [--verify N]      build the R4 expert sidecar (heavy I/O)
//!   flash-m0 verify-sidecar [--model-dir D] [--out F] [--n N]     re-read N records, compare with the shards
//!   flash-m0 run [options]                                        timing run (loads ~4.5 GB dense + cache)
//!
//! Every command that reads model bytes must run under the inference lease, e.g.
//!   flock -n -E 75 "$LOCAL_AI_LEASE" ./target/release/flash-m0 run ...

use qwen4exp::cache::Cache;
use qwen4exp::dense::{Dense, DenseMode};
use qwen4exp::exec::{Cfg, Engine, IoMode, Weights, KINDS, KIND_NAMES};
use qwen4exp::io::{self, IoConfig, IoQueue};
use qwen4exp::model::{Shards, N_LAYER, N_VOCAB};
use qwen4exp::ngram::NgramTable;
use qwen4exp::sidecar::{self, Layout};
use qwen4exp::trace::Trace;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME"))
}
fn die(m: &str) -> ! {
    eprintln!("flash-m0: {m}");
    std::process::exit(2)
}

struct Args(HashMap<String, String>);
impl Args {
    fn parse(v: &[String]) -> Args {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].strip_prefix("--").unwrap_or_else(|| die(&format!("unexpected {}", v[i])));
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                m.insert(k.to_string(), v[i + 1].clone());
                i += 2;
            } else {
                m.insert(k.to_string(), "1".into());
                i += 1;
            }
        }
        Args(m)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn n<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        match self.0.get(k) {
            Some(v) => v.parse().unwrap_or_else(|_| die(&format!("--{k} {v}: bad value"))),
            None => d,
        }
    }
    fn p(&self, k: &str, d: PathBuf) -> PathBuf {
        self.0.get(k).map(PathBuf::from).unwrap_or(d)
    }
}

fn default_model_dir() -> PathBuf {
    home().join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS")
}
fn default_sidecar() -> PathBuf {
    home().join(".cache/local-ai/snapdragon-engine/flash/experts-r4.bin")
}
fn default_trace() -> PathBuf {
    // A recorded expert-id trace (CSV). Override with OXF_EXPERT_TRACE or --trace.
    std::env::var_os("OXF_EXPERT_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".cache/local-ai/snapdragon-engine/flash/flash-fixed353-expert-ids.csv"))
}

fn status_kib(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok()
}
fn cgroup_file(name: &str) -> Option<String> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cg.lines().find_map(|l| l.strip_prefix("0::"))?;
    let p = format!("/sys/fs/cgroup{}/{name}", path.trim());
    std::fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).map(|s| s.as_str()).unwrap_or("help");
    let a = Args::parse(&argv[2.min(argv.len())..]);
    match cmd {
        "sidecar" => cmd_sidecar(&a),
        "verify-sidecar" => cmd_verify(&a),
        "run" => cmd_run(&a),
        "info" => cmd_info(&a),
        _ => {
            eprintln!("{}", include_str!("flash-m0.rs").lines().take(8).map(|l| l.trim_start_matches("//!")).collect::<Vec<_>>().join("\n"));
            std::process::exit(2)
        }
    }
}

fn open(a: &Args) -> (Shards, Layout) {
    let s = Shards::open(&a.p("model-dir", default_model_dir())).unwrap_or_else(|e| die(&e));
    let lay = Layout::new(&s, true);
    (s, lay)
}

/// Header-only: layout, per-token bytes, quotas (no weights read).
fn cmd_info(a: &Args) {
    let (s, lay) = open(a);
    let trace = Trace::load(&a.p("trace", default_trace())).unwrap_or_else(|e| die(&e));
    let budget = (a.n("cache-gib", 16.0f64) * (1u64 << 30) as f64) as usize;
    let wq = prefill_weights(&trace, &lay);
    let q = Cache::quotas(&lay, budget, &wq, 64);
    let attn: Vec<usize> = (0..N_LAYER).filter(|&l| s.is_attn_layer(l)).collect();
    let (ho, hv) = s.ple_heads();
    println!(
        "{}",
        json!({"shards": s.paths, "tensors": s.tensors.len(), "attn_layers": attn, "sidecar_total_bytes": lay.total,
               "slots": lay.geom.iter().map(|g| g.slot).collect::<Vec<_>>(), "quota_prefill_16gib": q,
               "quota_bytes": (0..N_LAYER).map(|l| q[l] * lay.geom[l].slot).sum::<usize>(),
               "prefill_distinct": trace.prefill.iter().map(|p| p.len()).collect::<Vec<_>>(),
               "decode_tokens": trace.decode.len(), "ple_head_offsets": ho, "ple_head_vocab": hv,
               "ngram_tensor": s.req("per_layer_token_embd.weight").off})
    );
}

fn cmd_sidecar(a: &Args) {
    let (s, lay) = open(a);
    let out = a.p("out", default_sidecar());
    eprintln!("sidecar: {} bytes -> {}", lay.total, out.display());
    let r = sidecar::build(&s, &lay, &out).unwrap_or_else(|e| die(&e));
    let n = a.n("verify", 64usize);
    let v = sidecar::verify(&s, &lay, &out, n).unwrap_or_else(|e| die(&e));
    println!("{}", json!({"sidecar": out, "build": r, "verified_records": v, "commit": qwen4exp::ENGINE_COMMIT}));
}

fn cmd_verify(a: &Args) {
    let (s, lay) = open(a);
    let out = a.p("out", default_sidecar());
    sidecar::check(&s, &lay, &out).unwrap_or_else(|e| die(&e));
    let v = sidecar::verify(&s, &lay, &out, a.n("n", 256usize)).unwrap_or_else(|e| die(&e));
    println!("{}", json!({"sidecar": out, "verified_records": v}));
}

/// Prefill-derived quota weights: distinct experts the prompt used per layer x slot bytes. The
/// trace's last layer only ran the final prompt position (output cropping), so a layer with
/// fewer than 2 x top-k distinct experts gets the mean weight of the others.
fn prefill_weights(trace: &Trace, lay: &Layout) -> Vec<f64> {
    let n: Vec<f64> = trace.prefill.iter().map(|p| p.len() as f64).collect();
    let ok: Vec<f64> = n.iter().cloned().filter(|&x| x >= 20.0).collect();
    let mean = ok.iter().sum::<f64>() / ok.len().max(1) as f64;
    (0..N_LAYER).map(|l| (if n[l] >= 20.0 { n[l] } else { mean }) * lay.geom[l].slot as f64).collect()
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn cmd_run(a: &Args) {
    let t_start = Instant::now();
    let cpus = sched::parse_cpu_list(&a.s("cpu-list", "0,1,2,4,5,6,8,9")).unwrap_or_else(|e| die(&e));
    let io_cpu: Option<usize> = match a.s("io-cpu", "3").as_str() {
        "none" => None,
        c => Some(c.parse().unwrap_or_else(|_| die("--io-cpu"))),
    };
    let io_mode = match a.s("io", "lookahead").as_str() {
        "none" => IoMode::None,
        "demand" => IoMode::Demand,
        "lookahead" => IoMode::Lookahead,
        x => die(&format!("--io {x}: none|demand|lookahead")),
    };
    let dense_mode = match a.s("dense", "vecdot").as_str() {
        "vecdot" => DenseMode::VecDot,
        "repack" => DenseMode::Repack,
        x => die(&format!("--dense {x}: vecdot|repack")),
    };
    let cache_gib: f64 = a.n("cache-gib", 16.0);
    let quota_mode = a.s("quota", "prefill");
    let tokens_max: usize = a.n("tokens", 353);
    let warmup: usize = a.n("warmup", 8);
    let ctx: usize = a.n("ctx", 2048);
    let ngram_on = a.s("ngram", "on") == "on";
    let ngram_mib: usize = a.n("ngram-cache-mib", 64);
    let qd: u32 = a.n("qd", 32);
    let pf_qd: u32 = a.n("prefetch-qd", 16);
    let recall: f64 = a.n("recall", 0.7);
    let depth: usize = a.n("depth", 1);
    let false_pos = a.s("false-pos", "on") == "on";
    let max_gib: f64 = a.n("max-gib", 26.0);

    let (s, lay) = open(a);
    let sc = a.p("sidecar", default_sidecar());
    sidecar::check(&s, &lay, &sc).unwrap_or_else(|e| die(&e));
    let trace = Trace::load(&a.p("trace", default_trace())).unwrap_or_else(|e| die(&e));
    let tokens = tokens_max.min(trace.decode.len());
    if tokens <= warmup {
        die("--tokens must exceed --warmup");
    }
    let pos0: usize = a.n("pos0", 156); // the fixed screen's prompt length (trace prefill steps)

    // memory admission (rough): dense ~4.6 GB + cache + n-gram + KV/state
    let budget = (cache_gib * (1u64 << 30) as f64) as usize;
    let est = 4.6e9 + budget as f64 + (ngram_mib << 20) as f64 + 0.4e9;
    if est / (1u64 << 30) as f64 > max_gib {
        die(&format!("estimated footprint {:.1} GiB > --max-gib {max_gib}", est / (1u64 << 30) as f64));
    }

    // quotas
    let weights: Vec<f64> = match quota_mode.as_str() {
        "equal" => (0..N_LAYER).map(|l| lay.geom[l].slot as f64).collect(), // equal slot counts
        "bytes" => vec![1.0; N_LAYER],
        "prefill" => prefill_weights(&trace, &lay),
        x => die(&format!("--quota {x}: prefill|equal|bytes")),
    };
    let quota = Cache::quotas(&lay, budget, &weights, a.n("quota-floor", 64usize));

    eprintln!("flash-m0: loading dense weights ({:?})", dense_mode);
    let dense = Dense::load(&s, dense_mode).unwrap_or_else(|e| die(&e));
    let w = Weights::new(&dense, &s);
    eprintln!(
        "flash-m0: dense arena {:.2} GB in {:.1} s; per-token bytes: dense {:.3} GB, router {:.3} GB, experts {:.3} GB",
        dense.arena_bytes as f64 / 1e9,
        dense.load_seconds,
        w.bytes_dense as f64 / 1e9,
        w.bytes_router as f64 / 1e9,
        w.bytes_experts as f64 / 1e9
    );
    let sc_fd = qwen4exp::uring::open_direct(&sc).unwrap_or_else(|e| die(&format!("{}: {e}", sc.display())));
    let t_cache = Instant::now();
    let cache = Cache::new(lay.clone(), &quota, sc_fd, io_mode == IoMode::None, true).unwrap_or_else(|e| die(&e.to_string()));
    let cache_alloc_s = t_cache.elapsed().as_secs_f64();
    let ioq = Arc::new(IoQueue::new());
    let io_thread = io::spawn(ioq.clone(), IoConfig { qd, prefetch_qd: pf_qd, cpu: io_cpu });

    let ngram = if ngram_on {
        let t = s.req("per_layer_token_embd.weight");
        let fd = qwen4exp::uring::open_direct(&s.paths[t.shard]).unwrap_or_else(|e| die(&e.to_string()));
        let (ho, hv) = s.ple_heads();
        Some(NgramTable::new(fd, t.off, ho, hv, ngram_mib << 20).unwrap_or_else(|e| die(&e.to_string())))
    } else {
        None
    };

    // warm the cache with the prefill routes (what a real prompt would leave behind)
    let t_warm = Instant::now();
    let mut stamp = 1u64 << 40;
    let done = |q: &IoQueue| q.counters.bytes.iter().map(|b| b.load(std::sync::atomic::Ordering::Acquire)).sum::<u64>();
    let drain = |q: &IoQueue| {
        while done(q) < cache.peek_stats().read_bytes {
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    for l in 0..N_LAYER {
        // in batches of half the layer's slots: a PENDING slot cannot be evicted
        for chunk in trace.prefill[l].chunks((quota[l] / 2).max(1)) {
            for &e in chunk {
                stamp += 1;
                // SAFETY: control thread (no job running).
                unsafe { cache.lookup(&ioq, l, e as usize, stamp, qwen4exp::io::ReqClass::Demand) };
            }
            drain(&ioq);
        }
    }
    let warm_issued = cache.peek_stats().read_bytes;
    drain(&ioq);
    let warm_s = t_warm.elapsed().as_secs_f64();
    // SAFETY: control thread.
    let warm_stats = unsafe { cache.take_stats() };
    eprintln!("flash-m0: cache warmed from the prefill routes: {:.2} GB in {:.1} s", warm_issued as f64 / 1e9, warm_s);

    let pool = sched::Pool::new(&cpus, Duration::from_millis(50)).unwrap_or_else(|e| die(&e.to_string()));
    let nt = pool.nthreads();
    let cfg = Cfg { io: io_mode, recall, depth, false_pos, ctx, pos0, ngram: ngram_on, seed: a.n("seed", 12345u64) };
    let eng = Engine::new(&w, &cache, &ioq, ngram, &trace, cfg, nt);

    // synthetic token stream (the fixed-screen token ids are not in the trace)
    let mut toks: Vec<u32> = Vec::new();
    let mut rs = 0x2545F4914F6CDD1Du64;
    let mut per_tok = Vec::new();
    let mut bd_sum = [[0f64; 4]; KINDS];
    let mut work_sum = [0f64; KINDS];
    let mut mgmt_ms = 0.0;
    let mut ng_before = (0u64, 0u64, 0u64);
    let b0 = pool.barrier_episodes();
    let mut barriers_meas = 0u64;
    let mut io_meas_start = 0u64;
    let snap = |q: &IoQueue| -> [[u64; 3]; 3] {
        let c = &q.counters;
        let l = |x: &std::sync::atomic::AtomicU64| x.load(std::sync::atomic::Ordering::Acquire);
        [[l(&c.completed[0]), l(&c.completed[1]), l(&c.completed[2])], [l(&c.bytes[0]), l(&c.bytes[1]), l(&c.bytes[2])], [l(&c.lat_ns[0]), l(&c.lat_ns[1]), l(&c.lat_ns[2])]]
    };
    let mut io_snap = snap(&ioq);
    for t in 0..tokens {
        rs ^= rs << 13;
        rs ^= rs >> 7;
        rs ^= rs << 17;
        toks.push((rs % N_VOCAB as u64) as u32);
        if t == warmup {
            // SAFETY: control thread.
            unsafe { cache.take_stats() };
            if let Some(n) = eng.ngram_stats() {
                ng_before = n;
            }
            barriers_meas = pool.barrier_episodes();
            io_meas_start = done(&ioq);
            io_snap = snap(&ioq);
        }
        let ts0 = Instant::now();
        // SAFETY: control thread between jobs.
        unsafe { *eng.tok.get() = t };
        eng.begin_token(&toks);
        pool.run(|c| eng.token_job(c));
        let wall = ts0.elapsed().as_nanos() as u64;
        let (bd, _n) = eng.breakdown(wall);
        let mw = eng.mean_work();
        let (_arg, mg) = eng.token_result();

        if t >= warmup {
            per_tok.push(wall as f64 / 1e6);
            for k in 0..KINDS {
                for j in 0..4 {
                    bd_sum[k][j] += bd[k][j];
                }
                work_sum[k] += mw[k];
            }
            mgmt_ms += mg as f64 / 1e6;
        }
    }
    let nmeas = per_tok.len() as f64;
    // SAFETY: control thread.
    let st = unsafe { cache.take_stats() };
    let barriers = pool.barrier_episodes() - barriers_meas;
    let io_meas_bytes = done(&ioq) - io_meas_start;
    ioq.stop();
    io_thread.join().unwrap();
    let _ = b0;

    let mean = per_tok.iter().sum::<f64>() / nmeas;
    let mut v = per_tok.clone();
    let median = pct(&mut v, 0.5);
    let p90 = pct(&mut v, 0.9);
    let dense_kinds = [0usize, 1, 2, 3, 4, 5, 8];
    let sum_k = |ks: &[usize], j: usize| ks.iter().map(|&k| bd_sum[k][j]).sum::<f64>() / nmeas;
    let all: Vec<usize> = (0..KINDS).collect();
    let breakdown = json!({
        "dense_compute_ms": sum_k(&dense_kinds, 0),
        "experts_compute_ms": sum_k(&[6, 7], 0),
        "ssd_wait_ms": sum_k(&all, 1),
        "ngram_wait_ms": sum_k(&all, 2),
        "sync_ms": sum_k(&all, 3),
        "outside_job_ms": mean - (0..4).map(|j| sum_k(&all, j)).sum::<f64>(),
        "cache_mgmt_thread0_ms": mgmt_ms / nmeas,
        "note": "each phase's wall (thread 0 barrier exits) = slowest thread's compute + its SSD/N-gram wait + sync (rest); outside_job = embedding row, N-gram issue, pool wake/join",
    });
    let phases: Vec<_> = (0..KINDS)
        .map(|k| {
            json!({"kind": KIND_NAMES[k], "compute_ms": bd_sum[k][0]/nmeas, "ssd_ms": bd_sum[k][1]/nmeas,
                   "ngram_ms": bd_sum[k][2]/nmeas, "sync_ms": bd_sum[k][3]/nmeas, "mean_thread_work_ms": work_sum[k]/nmeas})
        })
        .collect();
    let dl = st.demand_lookups.max(1) as f64;
    let c = &ioq.counters;
    let ld = |x: &std::sync::atomic::AtomicU64| x.load(std::sync::atomic::Ordering::Relaxed);
    let io_json: Vec<_> = ["demand", "ngram", "prefetch"]
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let cnt = ld(&c.completed[i]) - io_snap[0][i];
            json!({"class": n, "decode_completed": cnt, "decode_bytes": ld(&c.bytes[i]) - io_snap[1][i],
                   "decode_mean_latency_us": (ld(&c.lat_ns[i]) - io_snap[2][i]) as f64 / cnt.max(1) as f64 / 1e3,
                   "max_latency_us_incl_warm": ld(&c.lat_max_ns[i]) as f64 / 1e3})
        })
        .collect();
    let meas_s = per_tok.iter().sum::<f64>() / 1e3;
    let ng_json = eng.ngram_stats().map(|ng| {
        let (l, h, r) = (ng.0 - ng_before.0, ng.1 - ng_before.1, ng.2 - ng_before.2);
        json!({"page_lookups": l, "page_hits": h, "page_reads": r, "hit_rate": h as f64 / l.max(1) as f64, "cache_mib": ngram_mib})
    });
    let base_ms = 1000.0 / 4.057;
    let tok_s = 1000.0 / median;
    let verdict = if median <= 90.0 || base_ms / median >= 1.5 {
        "GO"
    } else if median > 140.0 {
        "STOP"
    } else {
        "GREY (90-140 ms: try the alternative layout once, then decide)"
    };
    let rep = json!({
        "kind": "flash-m0 timing (approximate math, replayed routing)",
        "engine_commit": qwen4exp::ENGINE_COMMIT,
        "kernel_cflags": qwen4exp::KERNEL_CFLAGS,
        "config": {"cpus": cpus, "io_cpu": io_cpu, "io": format!("{io_mode:?}"), "recall": recall, "depth": depth,
                   "false_pos": false_pos, "dense": format!("{dense_mode:?}"), "cache_gib": cache_gib, "quota": quota_mode,
                   "quota_slots": quota, "qd": qd, "prefetch_qd": pf_qd, "ngram": ngram_on, "ctx": ctx, "pos0": pos0,
                   "tokens": tokens, "warmup": warmup, "sidecar": sc},
        "ms_per_token": {"median": median, "mean": mean, "p90": p90, "min": v[0], "max": v[v.len()-1]},
        "tok_s": tok_s,
        "breakdown_ms": breakdown,
        "phases": phases,
        "barriers_per_token": barriers as f64 / nmeas,
        "bytes_per_token": {"dense": w.bytes_dense, "router_f32": w.bytes_router, "experts_touched": w.bytes_experts},
        "cache": {"demand_lookups": st.demand_lookups, "hit_rate_resident": st.demand_hits_ready as f64 / dl,
                  "hit_rate_incl_inflight": (st.demand_hits_ready + st.demand_hits_pending) as f64 / dl,
                  "demand_misses_per_token": st.demand_misses as f64 / nmeas,
                  "prefetch_issued_per_token": st.prefetch_issued as f64 / nmeas,
                  "prefetch_used_per_token": st.demand_hits_prefetched as f64 / nmeas,
                  "read_mib_per_token": st.read_bytes as f64 / nmeas / (1u64<<20) as f64,
                  "ssd_gb_s_during_decode": io_meas_bytes as f64 / 1e9 / meas_s,
                  "arena_bytes": cache.arena_bytes(), "alloc_prefault_s": cache_alloc_s,
                  "warm": {"bytes": warm_issued, "seconds": warm_s, "lookups": warm_stats.demand_lookups}},
        "io": io_json,
        "io_max_inflight": ld(&c.max_inflight),
        "ngram": ng_json,
        "memory": {"vm_hwm_kib": status_kib("VmHWM:"), "vm_rss_kib": status_kib("VmRSS:"),
                   "cgroup_memory_peak": cgroup_file("memory.peak"), "cgroup_memory_max": cgroup_file("memory.max"),
                   "cgroup_swap_max": cgroup_file("memory.swap.max")},
        "dense": {"arena_bytes": dense.arena_bytes, "load_s": dense.load_seconds},
        "gate": {"baseline_tok_s": 4.057, "baseline_ms": base_ms, "speedup": base_ms / median,
                 "go": "<= 90 ms/token or >= 1.5x baseline", "stop": "> 140 ms/token", "verdict": verdict},
        "per_token_ms": per_tok,
        "wall_s": t_start.elapsed().as_secs_f64(),
    });
    let out = serde_json::to_string_pretty(&rep).unwrap();
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, &out).unwrap_or_else(|e| die(&format!("{p}: {e}")));
    }
    println!("{out}");
    let bdv = &rep["breakdown_ms"];
    eprintln!(
        "flash-m0: {:.1} ms/token median ({:.2} tok/s, {:.2}x of 4.057), dense {:.1} experts {:.1} ssd {:.1} ngram {:.2} sync {:.1} other {:.1} ms; hit {:.1}%, {:.1} MiB/token read, {:.0} barriers/token, VmHWM {:.2} GiB -> {}",
        median,
        tok_s,
        base_ms / median,
        bdv["dense_compute_ms"].as_f64().unwrap(),
        bdv["experts_compute_ms"].as_f64().unwrap(),
        bdv["ssd_wait_ms"].as_f64().unwrap(),
        bdv["ngram_wait_ms"].as_f64().unwrap(),
        bdv["sync_ms"].as_f64().unwrap(),
        bdv["outside_job_ms"].as_f64().unwrap(),
        100.0 * st.demand_hits_ready as f64 / dl,
        st.read_bytes as f64 / nmeas / (1u64 << 20) as f64,
        barriers as f64 / nmeas,
        status_kib("VmHWM:").unwrap_or(0) as f64 / (1u64 << 20) as f64,
        verdict
    );
}
