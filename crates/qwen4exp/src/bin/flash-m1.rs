//! flash-m1: Flash M1 exactness driver.
//!
//!   flash-m1 ref --oracle DIR [options]    greedy fixed353 through the graph port (production ggml),
//!                                           experts from shard mappings; compares the tokens with
//!                                           production's (DIR/tokens.bin from the oracle dump run)
//!
//!   flash-m1 stream --oracle DIR [options] prefill through the graph (mapped experts), then decode with
//!                                           the streaming expert op (M0 cache + IO thread + R4 sidecar);
//!                                           the first --check-steps decode steps also run a mapped-expert
//!                                           context on a copy of the state and compare every
//!                                           ffn_moe_{up,gate,swiglu,down,out}-L, l_last-L and the logits
//!                                           bitwise; then the rest of fixed353 streams only
//!     stream options: --sidecar F, --cache-gib 16, --qd 32, --io-cpu N, --check-steps 3,
//!                     --check-dir D (default results/m1/check), --prefill-state F (save the prompt's
//!                     state after the prefill, or load it if F exists: skips the mapped prefill)
//!
//! Options: --model-dir D, --threads 4 (decode, production), --threads-batch 6 (prefill, production),
//! --n-ctx 16384, --max-new 400, --dump-dir D --dump-names a,b* --dump-ubatches 1,2,3 (production
//! numbering: u0 = production's warm-up, u1 = the prompt, u2.. = decode steps), --out F.json.
//!
//! Loads model bytes: run under the inference lease, inside the 28G/no-swap scope.

use qwen4exp::dense::{Dense, DenseMode};
use qwen4exp::graph::{argmax, Dump, GraphCtx, GraphModel};
use qwen4exp::model::{Shards, N_VOCAB};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const EOG: [i32; 2] = [248046, 248044];

fn die(m: &str) -> ! {
    eprintln!("flash-m1: {m}");
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
}

/// Generated token i matches production: equal to the oracle's, or the end-of-generation token
/// right after the oracle's last generated token (the oracle records only tokens fed back).
fn want_ok(want: &[i32], i: usize, next: i32) -> bool {
    match want.get(i) {
        Some(&w) => w == next,
        None => i == want.len() && EOG.contains(&next),
    }
}

/// The oracle dump's token records: one per production ubatch (u0 warm-up, u1 prompt, u2.. decode).
pub fn read_tokens(p: &Path) -> Vec<Vec<i32>> {
    let b = std::fs::read(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        assert_eq!(&b[i..i + 4], b"FGD1");
        let mut n = 1i64;
        for k in 0..4 {
            n *= i64::from_le_bytes(b[i + 8 + 8 * k..i + 16 + 8 * k].try_into().unwrap());
        }
        let n = n as usize;
        let v = (0..n).map(|j| i32::from_le_bytes(b[i + 40 + 4 * j..i + 44 + 4 * j].try_into().unwrap())).collect();
        out.push(v);
        i += 40 + 4 * n;
    }
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let a = Args::parse(&argv[1.min(argv.len())..]);
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let model_dir = PathBuf::from(a.s("model-dir", &home.join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS").to_string_lossy()));
    match cmd.as_str() {
        "ref" => cmd_ref(&a, &model_dir),
        "stream" => cmd_stream(&a, &model_dir),
        _ => die("usage: flash-m1 ref|stream --oracle DIR [...]"),
    }
}

fn cmd_ref(a: &Args, model_dir: &Path) {
    let oracle = PathBuf::from(a.s("oracle", ""));
    let recs = read_tokens(&oracle.join("tokens.bin"));
    if recs.len() < 3 || recs[1].len() < 2 {
        die("oracle tokens.bin: expected warm-up, prompt, decode records");
    }
    let prompt = recs[1].clone();
    let want: Vec<i32> = recs[2..].iter().map(|r| r[0]).collect();
    let threads: usize = a.n("threads", 4);
    let threads_batch: usize = a.n("threads-batch", 6);
    let n_ctx: u32 = a.n("n-ctx", 16384);
    let max_new: usize = a.n("max-new", 400);
    let dump_dir = a.s("dump-dir", "");
    let dump_names: Vec<String> = a.s("dump-names", "").split(',').filter(|s| !s.is_empty()).map(String::from).collect();
    let dump_ub: Vec<usize> = a.s("dump-ubatches", "").split(',').filter_map(|s| s.parse().ok()).collect();

    let t0 = Instant::now();
    let s = Shards::open(model_dir).unwrap_or_else(|e| die(&e));
    let d = Dense::load(&s, DenseMode::VecDot).unwrap_or_else(|e| die(&e));
    let load_s = t0.elapsed().as_secs_f64();
    let gm = GraphModel::new(&s, &d, true).unwrap_or_else(|e| die(&e));
    let mut g = GraphCtx::new(&gm, n_ctx);
    eprintln!("flash-m1: dense {:.2} GB in {load_s:.1} s, state {:.0} MiB", d.arena_bytes as f64 / 1e9, g.state_bytes() as f64 / 1048576.0);

    let mut logits = vec![0f32; N_VOCAB];
    let dump_for = |ub: usize| -> Option<Dump> {
        if dump_dir.is_empty() || !dump_ub.contains(&ub) {
            return None;
        }
        let dir = format!("{dump_dir}/u{ub}");
        std::fs::create_dir_all(&dir).ok();
        Some(Dump { dir, names: dump_names.clone() })
    };
    let tp = Instant::now();
    g.decode(&prompt, threads_batch, &mut logits, dump_for(1).as_ref()).unwrap_or_else(|e| die(&e));
    let prefill_s = tp.elapsed().as_secs_f64();
    let mut got: Vec<i32> = vec![argmax(&logits) as i32];
    eprintln!("flash-m1: prefill {} tokens {prefill_s:.1} s, first token {} (want {})", prompt.len(), got[0], want[0]);
    let td = Instant::now();
    let mut first_mismatch: Option<usize> = if got[0] != want[0] { Some(0) } else { None };
    while got.len() < max_new && !EOG.contains(got.last().unwrap()) {
        let ub = got.len() + 1; // production ubatch index of this decode step
        let tok = *got.last().unwrap();
        g.decode(&[tok], threads, &mut logits, dump_for(ub).as_ref()).unwrap_or_else(|e| die(&e));
        let next = argmax(&logits) as i32;
        let i = got.len();
        got.push(next);
        if first_mismatch.is_none() && !want_ok(&want, i, next) {
            first_mismatch = Some(i);
            eprintln!("flash-m1: MISMATCH at generated token {i}: got {next}, want {:?}", want.get(i));
            if !a.0.contains_key("keep-going") {
                break;
            }
        }
        if i % 50 == 0 {
            eprintln!("flash-m1: {i} tokens, {:.1} ms/token", td.elapsed().as_secs_f64() * 1e3 / i as f64);
        }
    }
    let decode_s = td.elapsed().as_secs_f64();
    // production's answer = the generated tokens up to (not including) the end-of-generation token
    let ans: Vec<i32> = got.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    let want_ans: Vec<i32> = want.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    let identical = ans == want_ans && first_mismatch.is_none();
    let out = json!({
        "kind": "flash-m1-ref",
        "engine_commit": qwen4exp::ENGINE_COMMIT,
        "ggml_prod_cpu_sha256": env!("OXF_GGML_PROD_CPU_SHA"),
        "prompt_tokens": prompt.len(),
        "generated": ans.len(),
        "want_generated": want_ans.len(),
        "first_mismatch": first_mismatch,
        "tokens_identical": identical,
        "threads": threads, "threads_batch": threads_batch, "n_ctx": n_ctx,
        "load_s": load_s, "prefill_s": prefill_s, "decode_s": decode_s,
        "ms_per_token": decode_s * 1e3 / (got.len().max(2) - 1) as f64,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, serde_json::to_string_pretty(&out).unwrap()).ok();
    }
    std::process::exit(if identical { 0 } else { 1 });
}

fn status_kib(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok()
}
fn cgroup_file(name: &str) -> Option<String> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cg.lines().find_map(|l| l.strip_prefix("0::"))?;
    std::fs::read_to_string(format!("/sys/fs/cgroup{}/{name}", path.trim())).ok().map(|s| s.trim().to_string())
}

/// Compare the dump files of `a` that also exist in `b` (the oracle dumps a subset of names).
fn compare_common(a: &str, b: &str) -> (usize, usize, Vec<String>) {
    let mut names: Vec<String> = std::fs::read_dir(a)
        .map(|r| r.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    names.retain(|n| Path::new(b).join(n).exists());
    names.sort_by_key(|n| {
        let stem = n.trim_end_matches(".bin").to_string();
        (stem.rsplit('-').next().and_then(|x| x.parse::<i64>().ok()).unwrap_or(99), stem)
    });
    let mut same = 0;
    let mut diff = Vec::new();
    for n in &names {
        if std::fs::read(Path::new(a).join(n)).ok() == std::fs::read(Path::new(b).join(n)).ok() {
            same += 1;
        } else {
            diff.push(n.clone());
        }
    }
    (same, names.len(), diff)
}

fn cmd_stream(a: &Args, model_dir: &Path) {
    use qwen4exp::stream::{Stream, StreamConfig};
    let oracle = PathBuf::from(a.s("oracle", ""));
    let recs = read_tokens(&oracle.join("tokens.bin"));
    if recs.len() < 3 || recs[1].len() < 2 {
        die("oracle tokens.bin: expected warm-up, prompt, decode records");
    }
    let prompt = recs[1].clone();
    let want: Vec<i32> = recs[2..].iter().map(|r| r[0]).collect();
    let threads: usize = a.n("threads", 4);
    let threads_batch: usize = a.n("threads-batch", 6);
    let n_ctx: u32 = a.n("n-ctx", 16384);
    let max_new: usize = a.n("max-new", 400);
    let check_steps: usize = a.n("check-steps", 3);
    let check_dir = a.s("check-dir", "results/m1/check");
    let cache_gib: f64 = a.n("cache-gib", 16.0);
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let sidecar = PathBuf::from(a.s("sidecar", &home.join(".cache/local-ai/snapdragon-engine/flash/experts-r4.bin").to_string_lossy()));
    let io_cpu: Option<usize> = a.0.get("io-cpu").and_then(|v| v.parse().ok());

    let t0 = Instant::now();
    let s = Shards::open(model_dir).unwrap_or_else(|e| die(&e));
    let d = Dense::load(&s, DenseMode::VecDot).unwrap_or_else(|e| die(&e));
    let load_s = t0.elapsed().as_secs_f64();
    let gm = GraphModel::new(&s, &d, true).unwrap_or_else(|e| die(&e));
    let mut gref = GraphCtx::new(&gm, n_ctx);
    let mut logits = vec![0f32; N_VOCAB];
    let tp = Instant::now();
    // --prefill-state F: reuse the prompt's state (saved by an earlier run with the same prompt,
    // threads-batch and build) instead of the 3-4 min mapped-expert prefill
    let pstate = a.0.get("prefill-state").map(|p| PathBuf::from(format!("{p}.ctx{n_ctx}.tb{threads_batch}")));
    let mut prefill_from_file = false;
    if let Some(p) = pstate.as_ref().filter(|p| p.exists()) {
        let (toks, ex) = gref.load_file(p).unwrap_or_else(|e| die(&e));
        if toks != prompt || ex.len() != N_VOCAB {
            die("prefill state file is for another prompt");
        }
        logits.copy_from_slice(&ex);
        prefill_from_file = true;
    } else {
        gref.decode(&prompt, threads_batch, &mut logits, None).unwrap_or_else(|e| die(&e));
        if let Some(p) = &pstate {
            gref.save_file(p, &logits).unwrap_or_else(|e| die(&e));
        }
    }
    let prefill_s = tp.elapsed().as_secs_f64();
    let mut got: Vec<i32> = vec![argmax(&logits) as i32];
    eprintln!("flash-m1 stream: prefill (from file: {prefill_from_file}) {} tokens {prefill_s:.1} s, first token {} (want {})", prompt.len(), got[0], want[0]);

    let ts = Instant::now();
    let cfg = StreamConfig { cache_bytes: (cache_gib * (1u64 << 30) as f64) as usize, qd: a.n("qd", 32u32), io_cpu, prefault: true };
    let st = Box::new(Stream::new(&s, &sidecar, &cfg).unwrap_or_else(|e| die(&e)));
    let stream_setup_s = ts.elapsed().as_secs_f64();
    eprintln!("flash-m1 stream: cache {:.2} GiB ready in {stream_setup_s:.1} s", st.cache.arena_bytes() as f64 / (1u64 << 30) as f64);
    let mut gs = GraphCtx::new(&gm, n_ctx);
    gs.copy_state_from(&gref).unwrap_or_else(|e| die(&e));
    // SAFETY: `st` outlives every decode of `gs` (dropped after it below).
    unsafe { gs.set_expert_fn(Some(Stream::callback()), &*st as *const Stream as *mut std::ffi::c_void) };

    let mut first_mismatch: Option<usize> = if got[0] != want[0] { Some(0) } else { None };
    let names: Vec<String> = ["ffn_moe_up-*", "ffn_moe_gate-*", "ffn_moe_swiglu-*", "ffn_moe_down-*", "ffn_moe_out-*", "l_last-*"]
        .iter()
        .map(|x| x.to_string())
        .collect();
    let mut checks = Vec::new();
    let mut check_ok = true;
    let mut lref = vec![0f32; N_VOCAB];
    let mut step_ms: Vec<f64> = Vec::new();
    // no check steps: free the reference context (state + its prefill compute buffer) right away
    let mut gref = if check_steps > 0 { Some(gref) } else { None };
    let mut snapshot_test: Option<serde_json::Value> = None;
    while got.len() < max_new && !EOG.contains(got.last().unwrap()) {
        let i = got.len();
        let tok = *got.last().unwrap();
        if i <= check_steps {
            let (dr, ds) = (format!("{check_dir}/ref/s{i}"), format!("{check_dir}/stream/s{i}"));
            for dd in [&dr, &ds] {
                std::fs::remove_dir_all(dd).ok();
                std::fs::create_dir_all(dd).ok();
            }
            let g = gref.as_mut().unwrap();
            g.decode(&[tok], threads, &mut lref, Some(&Dump { dir: dr.clone(), names: names.clone() })).unwrap_or_else(|e| die(&e));
            gs.decode(&[tok], threads, &mut logits, Some(&Dump { dir: ds.clone(), names: names.clone() })).unwrap_or_else(|e| die(&e));
            let logits_same = lref.iter().zip(&logits).all(|(x, y)| x.to_bits() == y.to_bits());
            // tensors: the streaming context's dumps against the PRODUCTION oracle's dumps of the same
            // ubatch (production numbering: decode step i = ubatch i+1), every name both have. The
            // mapped reference context's own dumps are also written (dr) but are not trusted: in a
            // two-context process some of its expert-node dumps come out stale (see docs/lab-notes/M1-FLASH.md).
            let od = oracle.join(format!("u{}", i + 1));
            let (same, total, diff) = compare_common(&ds, &od.to_string_lossy());
            let ok = logits_same && same == total;
            check_ok &= ok;
            eprintln!("flash-m1 stream: check step {i}: logits identical to the mapped-expert context {logits_same}; {same}/{total} tensors identical to the production oracle u{}{}",
                i + 1, if diff.is_empty() { String::new() } else { format!(", first differing {}", diff[0]) });
            checks.push(json!({"step": i, "oracle_ubatch": i + 1, "oracle_tensors": total, "identical": same, "logits_identical": logits_same,
                               "differing": diff.iter().take(10).collect::<Vec<_>>()}));
            if i == check_steps {
                gref = None; // free the reference context
            }
        } else {
            if i == check_steps + 1 && !a.0.contains_key("no-snapshot-test") {
                // snapshot / restore: save, decode two tokens, restore, decode the first again
                let (p, snap) = gs.save_recurrent();
                gs.decode(&[tok], threads, &mut lref, None).unwrap_or_else(|e| die(&e));
                let first = lref.clone();
                let t2 = argmax(&lref) as i32;
                gs.decode(&[t2], threads, &mut lref, None).unwrap_or_else(|e| die(&e));
                gs.restore_recurrent(p, &snap).unwrap_or_else(|e| die(&e));
                gs.decode(&[tok], threads, &mut logits, None).unwrap_or_else(|e| die(&e));
                let same = first.iter().zip(&logits).all(|(x, y)| x.to_bits() == y.to_bits());
                check_ok &= same;
                eprintln!("flash-m1 stream: snapshot/restore at position {p}: logits after restore identical {same}");
                snapshot_test = Some(json!({"position": p, "bytes": snap.len(), "logits_identical": same}));
                let next = argmax(&logits) as i32;
                got.push(next);
                if first_mismatch.is_none() && !want_ok(&want, i, next) {
                    first_mismatch = Some(i);
                    break;
                }
                continue;
            }
            let t = Instant::now();
            gs.decode(&[tok], threads, &mut logits, None).unwrap_or_else(|e| die(&e));
            step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let next = argmax(&logits) as i32;
        got.push(next);
        if first_mismatch.is_none() && !want_ok(&want, i, next) {
            first_mismatch = Some(i);
            eprintln!("flash-m1 stream: MISMATCH at generated token {i}: got {next}, want {:?}", want.get(i));
            if !a.0.contains_key("keep-going") {
                break;
            }
        }
        if i % 50 == 0 && !step_ms.is_empty() {
            eprintln!("flash-m1 stream: {i} tokens, {:.1} ms/token (streaming steps)", step_ms.iter().sum::<f64>() / step_ms.len() as f64);
        }
    }
    drop(gref);
    let ans: Vec<i32> = got.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    let want_ans: Vec<i32> = want.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    let identical = ans == want_ans && first_mismatch.is_none();
    let n = step_ms.len().max(1) as f64;
    let mut sorted = step_ms.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = if sorted.is_empty() { 0.0 } else { sorted[sorted.len() / 2] };
    let cs = st.stats();
    let calls = st.counters.calls.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64;
    let toks_streamed = calls / 48.0;
    let io = &st.io.counters;
    let ld = |x: &std::sync::atomic::AtomicU64| x.load(std::sync::atomic::Ordering::Relaxed);
    let out = json!({
        "kind": "flash-m1-stream",
        "engine_commit": qwen4exp::ENGINE_COMMIT,
        "ggml_prod_cpu_sha256": env!("OXF_GGML_PROD_CPU_SHA"),
        "prompt_tokens": prompt.len(),
        "generated": ans.len(),
        "want_generated": want_ans.len(),
        "first_mismatch": first_mismatch,
        "tokens_identical": identical,
        "check_steps": checks,
        "check_all_identical": check_ok,
        "snapshot_test": snapshot_test,
        "threads": threads, "threads_batch": threads_batch, "n_ctx": n_ctx,
        "cache_gib": cache_gib, "cache_arena_bytes": st.cache.arena_bytes(), "qd": cfg.qd, "io_cpu": io_cpu,
        "load_s": load_s, "prefill_s": prefill_s, "prefill_from_file": prefill_from_file, "stream_setup_s": stream_setup_s,
        "stream_steps": step_ms.len(),
        "ms_per_token": {"mean": step_ms.iter().sum::<f64>() / n, "median": median,
                         "p90": sorted.get(sorted.len() * 9 / 10).copied(), "max": sorted.last().copied()},
        "expert_wait_ms_per_token_thread0": ld(&st.counters.wait0_ns) as f64 / 1e6 / toks_streamed,
        "expert_wait_ms_per_token_all_threads": ld(&st.counters.wait_ns) as f64 / 1e6 / toks_streamed,
        "fetch_ms_per_token": ld(&st.counters.fetch_ns) as f64 / 1e6 / toks_streamed,
        "cache": {"demand_lookups": cs.demand_lookups, "hits": cs.demand_hits_ready + cs.demand_hits_pending,
                  "misses": cs.demand_misses, "hit_rate": (cs.demand_hits_ready + cs.demand_hits_pending) as f64 / cs.demand_lookups.max(1) as f64,
                  "read_mib_per_token": cs.read_bytes as f64 / 1048576.0 / toks_streamed, "evictions": cs.evictions},
        "io": {"completed": ld(&io.completed[0]), "mean_latency_us": ld(&io.lat_ns[0]) as f64 / 1e3 / ld(&io.completed[0]).max(1) as f64,
               "max_latency_us": ld(&io.lat_max_ns[0]) as f64 / 1e3, "max_inflight": ld(&io.max_inflight)},
        "memory": {"vm_hwm_kib": status_kib("VmHWM:"), "cgroup_memory_peak": cgroup_file("memory.peak"), "cgroup_memory_max": cgroup_file("memory.max")},
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, serde_json::to_string_pretty(&out).unwrap()).ok();
    }
    drop(gs);
    drop(st);
    std::process::exit(if identical && check_ok { 0 } else { 1 });
}
