#![recursion_limit = "256"]
//! flash-m3: Flash fast decode path driver (fixed353), with the M1 exact path selectable.
//!
//!   flash-m3 run --oracle DIR [options]
//!
//! Paths:
//!   --exact          M1: production-identical graph (4 decode / 6 prefill threads, mapped-expert prefill,
//!                    4-node streaming expert op, production flash attention). Tokens = production.
//!   (default) fast   --threads 8, fixed-chunk attention (256-cell blocks merged in order: thread-count and
//!                    batch independent), shared expert ahead of the routed experts, resident-first
//!                    expert units, streamed batched prefill (the prompt's experts go through the cache:
//!                    no shard page faults, and the cache is warm when decode starts).
//! Fast-path switches (each 0/1): --fixed-attn, --shexp-first, --resident-first, --stream-prefill,
//! --fuse N (S-c fused decode phases, bitmask: 1 hc-mix, 2 router, 4 shared expert + combine, 8 hc-combine,
//!   16 GDN pre/post, 32 attention pre/post; 63 = all; bitwise),
//! --trace-work (FD-1: forces --executor 1; per node kind work vs graph-barrier wait per thread, fused-barrier spin,
//!   fused / fallback counts, per-read IO stamps; JSON "trace"; ~1-3% overhead, not a speed number),
//! --executor (S-b: cached decode graphs run by ggml's own op functions under the engine's barrier schedule; same bits;
//!   `executor` in the JSON: graphs scheduled, nodes, barriers),
//! --rs-view (decode reads the recurrent states in place instead of a 1-thread 3 MB get_rows copy per GDN layer),
//! --graph-reuse (one-token decodes reuse the graph + allocation while the shape holds; bitwise the same),
//! --batch-gemm (streamed prefill: i8mm gemm4 over 4-token groups on the R4 layers; bitwise = gemv).
//!
//!   flash-m3 export --text F --n-tokens N [--prefix 64] --out F.lae --route-trace F.csv (learned-lookahead data: the
//!     first `prefix` tokens prefilled, the rest teacher-forced one by one; per step i the input embedding of token i
//!     and the final residual h_nextn after it; record = i32 step, f32 embd[2560], f32 h[10240]; header "LAE1", u32 dims)
//!   flash-m3 kld --ref A.flg --test B.flg [--windows a:b,...] [--out F]   (no model: per-step KL, windows)
//! Prefill compute switches (G4, default ON on the fast path; each takes 0/1 and `--name 0` / `false` / `off` turns it
//! off): --attn-tile (register-tile attention for prefill ubatches), --dense-gemm (multi-column dense matmul for
//! ubatches of >= 256 tokens), --expert-tile (IQ2_S gate/up in 8-token tiles, streamed prefill). All three are bitwise
//! equal to the switched-off path (G4: prefill 30.28 -> 45.32 tok/s, byte-identical state).
//! Numerics: --attn precise (fixed-chunk attention with f32 Q, f64 KQ and accumulators: a higher-precision
//! reference for the G-num question), --kld-out F.csv (per-step KLD of --kld-ref), --profile N --profile-from S.
//! Cache options (run and prefill): --cache-policy lfu|clock (default lfu since ticket 035), --quota-weights
//! builtin|equal|FILE (slot counts per layer ~ the 48 weights; default builtin = FLASH_QUOTA_WEIGHTS), --route-trace F.csv.
//! Options: --threads N, --threads-batch N, --ubatch 4096 (prefill ubatch, production's), --cache-gib 20,
//! --qd 32, --io-cpu N, --n-ctx 4096, --max-new 400, --teacher-force (feed the oracle's tokens instead of
//! the argmax: logits stay aligned with a reference run), --save-logits F (every step's logits, step 0 =
//! after the prompt), --kld-ref F (KL(ref || this) per step against a --save-logits file of the same
//! steps), --answer F (decoded answer text, for tools/flash_answer_check.py), --prefill-state F,
//! --profile N (first N decode steps node by node, per-kind ms/token in the JSON), --out F.json.
//!
//!   flash-m3 prefill --text F --n-tokens N [--mapped] [--profile] [--save-state F] [--save-logits F] [--reps 2]
//!
//! prefill (WP-S1): the first N tokens of F (model tokenizer, no specials) prefilled `--reps` times in one
//! process (rep 1 with a cold expert cache, later reps warm, each on a cleared context), streamed through the
//! cache (default) or with mapped experts (`--mapped`, the G-exact reference). Fast-path options as `run`
//! (--threads 8, --ubatch 4096, --fixed-attn ...). Per rep: seconds, tok/s, expert read GiB, IO wait; last-
//! token logits of every rep must be bitwise equal (determinism); `--save-state`/`--save-logits` write rep 1's
//! state / logits for a cross-run `cmp` (streamed vs mapped). `--profile`: rep 1 node by node (per-kind ms).
//!
//! Loads model bytes: run under the inference lease, inside the 28G/no-swap scope
//! (tools/run_flash_m3.sh does both).

use qwen4exp::cli::{cgroup_file, die, kld, read_tokens, status_kib, want_ok, Args, EOG};
use qwen4exp::dense::{Dense, DenseMode};
use qwen4exp::graph::{argmax, FgOptions, GraphCtx, GraphModel};
use qwen4exp::model::{Shards, N_VOCAB};

use qwen4exp::cache::Policy;
use qwen4exp::stream::{Stream, StreamConfig};
use serde_json::json;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn main() {
    sched::cli::exit_zero_on_epipe(); // F-CI9: `| head` ends with status 0, not a core dump
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let a = Args::parse(&argv[1.min(argv.len())..]);
    match cmd.as_str() {
        "run" => run(&a),
        "prefill" => prefill(&a),
        "kld" => kld_files(&a),
        "la-check" => {
            // model-free: flash-m3 la-check --lookahead F.lat --vectors F.lav (tools/la_dense.py reference vectors)
            let mut la = qwen4exp::lookahead::Lookahead::load(Path::new(&a.s("lookahead", "")), 8).unwrap_or_else(|e| die(&e));
            let v = std::fs::read(a.s("vectors", "")).unwrap_or_else(|e| die(&format!("--vectors: {e}")));
            let (n, bad) = qwen4exp::lookahead::check_vectors(&mut la, &v).unwrap_or_else(|e| die(&e));
            println!("{}", json!({"kind": "flash-m3-la-check", "vectors": n, "mismatches": bad, "pass": n > 0 && bad == 0}));
            if n == 0 || bad != 0 {
                std::process::exit(1);
            }
        }
        "export" => export(&a),
        "bench-dense" => {
            // model-free: flash-m3 bench-dense [--copies 4] [--reps 5] [--threads 1,4,8]
            let copies: usize = a.n("copies", 4);
            let reps: usize = a.n("reps", 5);
            for t in a.s("threads", "1,4,8").split(',') {
                qwen4exp::graph::bench_dense(t.parse().unwrap_or_else(|_| die("--threads a,b,c")), copies, reps);
            }
        }
        "selftest" => {
            let mut ok = true;
            for (kv, r) in [(256, 1), (512, 1), (512, 4), (1024, 3), (2304, 2), (1024, 20), (2304, 17), (512, 64), (32768, 1), (32768, 2)] {
                ok &= qwen4exp::graph::selftest_attn(kv, r);
            }
            println!("selftest attention: {}", if ok { "PASS" } else { "FAIL" });
            let mut ok2 = true;
            for nth in [1, 3, 8, 11] {
                ok2 &= qwen4exp::graph::selftest_exec(nth);
            }
            println!("selftest executor: {}", if ok2 { "PASS" } else { "FAIL" });
            ok &= ok2;
            let mut ok3 = true;
            for nth in [1, 3, 8, 11] {
                ok3 &= qwen4exp::graph::selftest_dense(nth);
            }
            println!("selftest dense gemm: {}", if ok3 { "PASS" } else { "FAIL" });
            ok &= ok3;
            // PREFILL FP-1: register-tile attention part op == the graph port's tiled path, bitwise, whole output
            let mut ok4 = true;
            for (kv, rows, pat) in [(256, 16, 0), (512, 24, 1), (1024, 37, 2), (2304, 64, 0), (2304, 40, 1), (768, 96, 2)] {
                for nth in [1, 3, 8, 11] {
                    ok4 &= qwen4exp::graph::selftest_attn_tile(kv, rows, nth, pat, 1).0;
                }
            }
            println!("selftest attention tiles: {}", if ok4 { "PASS" } else { "FAIL" });
            ok &= ok4;
            let ok5 = qwen4exp::graph::selftest_abort();
            println!("selftest abort: {}", if ok5 { "PASS" } else { "FAIL" });
            ok &= ok5;
            // S-c fused phases: reported, not yet part of the gate (not wired into the graph)
            let (mut h, mut f, mut pc) = (true, true, true);
            for nth in [1, 3, 8] {
                let (a, b, d) = qwen4exp::graph::selftest_fuse(nth);
                h &= a;
                f &= b;
                pc &= d;
            }
            let pf = |x: bool| if x { "PASS" } else { "FAIL" };
            let mut gd = true;
            for nth in [1, 3, 8] {
                gd &= qwen4exp::graph::selftest_fuse_gdn(nth);
            }
            let mut at = true;
            for nth in [1, 3, 8] {
                at &= qwen4exp::graph::selftest_fuse_attn(nth);
            }
            println!("selftest fused hc-mix: {}  fused router/shexp: {}  fused hc-combine: {}  fused gdn pre/post: {}  fused attn pre/post: {}",
                     pf(h), pf(f), pf(pc), pf(gd), pf(at));
            // chunked-prefill split evidence: printed, not part of the exit code (matmul lines are width-dependent by design)
            let sp = qwen4exp::graph::selftest_split(8);
            println!("selftest chat-split: {}", if sp { "expected components bitwise (see split lines)" } else { "EXPECTED COMPONENT DIFFERS (see split lines)" });
            std::process::exit(if ok { 0 } else { 1 })
        }
        // model-free timing of the multi-column dense matmul vs ggml_mul_mat on the Flash shapes (DEV windows only)
        "bench-dense-gemm" => {
            let nth: usize = a.n("threads", 11);
            let t: usize = a.n("n-tokens", 2048);
            let reps: usize = a.n("reps", 3);
            // (name, ggml type, K, R): Q6_K = 14, Q8_0 = 8
            let shapes = [("attn_qkv", 14, 2560, 10240), ("attn_gate z", 14, 2560, 6144), ("ssm_out", 14, 6144, 2560), ("attn_q", 14, 2560, 12288),
                ("shexp_up", 14, 2560, 640), ("hc_down", 8, 10240, 320), ("hc_up", 8, 320, 10240), ("shexp_down", 8, 640, 2560)];
            let mut rows = Vec::new();
            for (name, ty, k, r) in shapes {
                let (ms_ggml, ms_pf, same) = qwen4exp::graph::bench_dense_gemm(ty, k, r, t, nth, reps);
                let gmac = (k * r * t) as f64 / 1e9;
                println!("{name:12} K {k:5} R {r:5} T {t}: ggml {ms_ggml:8.1} ms ({:6.1} GMAC/s)  multi-column {ms_pf:8.1} ms ({:6.1} GMAC/s)  x{:.2}  bitwise {}",
                    gmac / (ms_ggml / 1e3), gmac / (ms_pf / 1e3), ms_ggml / ms_pf, same);
                rows.push(serde_json::json!({"name": name, "type": ty, "k": k, "r": r, "t": t, "ms_ggml": ms_ggml, "ms_multi_column": ms_pf, "bitwise": same}));
            }
            println!("{}", serde_json::json!({"kind": "flash-m3-bench-dense-gemm", "threads": nth, "rows": rows}));
        }
        // model-free single-thread timing of the expert kernels on one synthetic expert: today's path (gemm4 over 4-token
        // groups / vec_dot per row) vs the 8-token tiles, with the bitwise comparison (DEV windows only)
        "bench-expert" => bench_expert(&a),
        // model-free timing of the fixed-chunk attention part op, graph port's tiled path vs the register tiles
        // (the last `rows` positions of an `n-kv` context; DEV windows only)
        "bench-attn" => {
            let nth: usize = a.n("threads", 11);
            let reps: usize = a.n("reps", 2);
            let mut rows = Vec::new();
            for (kv, r, pat) in [(2048usize, 512usize, 0usize), (2048, 512, 2), (5632, 256, 1), (5632, 256, 0)] {
                let (same, o) = qwen4exp::graph::selftest_attn_tile4(kv, r, nth, pat, reps);
                let (off, on) = (o[0], o[1]);
                // pairs = (query, visible cell) for 24 heads; 512 FMA per pair (score + P.V)
                println!("n_kv {kv} rows {r} pattern {pat}: graph tiled {off:.1} ms  register tiles {on:.1} ms  x{:.2}; part + merge {:.1} ms  fused node {:.1} ms  bitwise (all) {same}", off / on, o[2], o[3]);
                rows.push(serde_json::json!({"n_kv": kv, "rows": r, "pattern": pat, "ms_off": off, "ms_on": on, "ms_part_merge": o[2], "ms_fused": o[3], "bitwise": same}));
            }
            println!("{}", serde_json::json!({"kind": "flash-m3-bench-attn", "threads": nth, "rows": rows}));
        }
        _ => die("usage: flash-m3 run --oracle DIR [--exact] [...] (see the source header)"),
    }
}

/// The expert stream with the cache options: --cache-policy clock|lfu, --quota-weights FILE (48 numbers, '#'
/// comments; tools/flash-quota-weights.txt), --route-trace F (written at the end by `finish_stream`).
fn make_stream(a: &Args, s: &Shards, sidecar: &Path, cfg: &StreamConfig) -> Box<Stream> {
    // default (ticket 035): the built-in per-layer weights; "equal" = the old equal slot counts; else a file
    let qw = a.s("quota-weights", "builtin");
    let w: Option<Vec<f64>> = if qw == "builtin" {
        Some(qwen4exp::stream::FLASH_QUOTA_WEIGHTS.to_vec())
    } else if qw == "equal" {
        None
    } else {
        Some(&qw).map(|p| {
        let t = std::fs::read_to_string(p).unwrap_or_else(|e| die(&format!("--quota-weights {p}: {e}")));
        t.lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .flat_map(|l| l.split_whitespace())
            .map(|x| x.parse().unwrap_or_else(|_| die(&format!("--quota-weights: bad value {x}"))))
            .collect()
    })
    };
    let mut st = Box::new(Stream::new_with(s, sidecar, cfg, w.as_deref()).unwrap_or_else(|e| die(&e)));
    st.cache.policy = match a.s("cache-policy", "lfu").as_str() {
        "clock" => Policy::Clock,
        "lfu" => Policy::Lfu,
        x => die(&format!("--cache-policy {x}: clock|lfu")),
    };
    if a.has("route-trace") {
        st.enable_trace();
    }
    // QC-A (FLASH-LOOKAHEAD-DESIGN): N wasted Prefetch reads per decode token; off by default
    st.enable_stage_junk(a.n("stage-junk", 0usize)).unwrap_or_else(|e| die(&e));
    // expert lookahead (FLASH-LOOKAHEAD-DESIGN section 6): speculate layer l+1 at layer l's FETCH; off by default
    if let Some(p) = a.0.get("lookahead") {
        st.enable_lookahead(Path::new(p), a.n("la-max", 2usize), a.n("la-pending", 2usize)).unwrap_or_else(|e| die(&e));
    }
    st
}

fn finish_stream(a: &Args, st: &mut Stream) {
    if let Some(p) = a.0.get("route-trace") {
        match st.write_trace(Path::new(p)) {
            Ok(n) => eprintln!("flash-m3: route trace {p}: {n} (layer, token) records"),
            Err(e) => eprintln!("flash-m3: route trace {p}: {e}"),
        }
    }
}

/// Mean and median of steps `a..b` (clamped); null when the range is empty.
fn split_mean(ms: &[f64], a: usize, b: usize) -> serde_json::Value {
    let v = &ms[a.min(ms.len())..b.min(ms.len())];
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    json!({"steps": v.len(), "mean": v.iter().sum::<f64>() / v.len() as f64, "median": pct(v, 0.5)})
}

/// Lookahead receipt: the table, the per-layer cap, the scorer's counters and the cache's speculation counters (whole
/// run; speculation only happens in one-token decodes). Null when off.
fn lookahead_json(a: &Args, st: &Stream) -> serde_json::Value {
    let Some(la) = st.lookahead_stats() else { return serde_json::Value::Null };
    let cs = st.stats();
    json!({"table": a.0.get("lookahead"), "la_max": a.n("la-max", 2usize), "la_pending": a.n("la-pending", 2usize),
           "layers_scored": la.scored, "above_threshold": la.above_thr, "issued": la.issued, "no_slot": la.no_slot,
           "issued_per_scored_layer": if la.scored > 0 { la.issued as f64 / la.scored as f64 } else { 0.0 },
           "issued_per_layer": la.issued_per_layer.to_vec(),
           "skipped_pressure_layers": la.skipped_pressure_layers, "skipped_pressure_reads": la.skipped_pressure_reads,
           "max_in_flight": la.max_in_flight,
           "score_ms_total": la.score_ns as f64 / 1e6,
           "spec_issued": cs.spec_issued, "spec_hit_ready": cs.spec_hit_ready, "spec_hit_pending": cs.spec_hit_pending,
           "spec_evicted_unused": cs.spec_evicted_unused})
}

fn cache_opts_json(a: &Args, st: &Stream) -> serde_json::Value {
    json!({"policy": format!("{:?}", st.cache.policy), "quota_weights": a.s("quota-weights", "builtin"),
           "quota_min": st.cache.quota.iter().min(), "quota_max": st.cache.quota.iter().max(), "route_trace": a.0.get("route-trace"),
           "stage_junk": a.n("stage-junk", 0usize), "junk_issued_skipped": st.junk_counts(),
           "lookahead": lookahead_json(a, st)})
}

/// fixed-chunk attention mode: 1 = online f32 update (default; the bits of tickets 002-010), 3 = `--attn twopass`
/// (two-pass softmax + NEON P.V; faster, different summation order), 2 = `--attn precise` (f64 reference),
/// 0 = `--fixed-attn 0` (production flash attention). Prefill ubatches of >= 16 rows use the row/head-tiled
/// variant of the same mode (bitwise equal per query).
fn attn_mode(a: &Args) -> i32 {
    match a.s("attn", "").as_str() {
        "precise" => 2,
        "twopass" => 3,
        "online" | "" => a.flag("fixed-attn", true) as i32,
        x => die(&format!("--attn {x}: online|twopass|precise")),
    }
}

struct LogitsWriter(std::io::BufWriter<std::fs::File>);
impl LogitsWriter {
    fn new(p: &Path) -> LogitsWriter {
        let mut f = std::io::BufWriter::new(std::fs::File::create(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
        f.write_all(b"FLG1").unwrap();
        f.write_all(&(N_VOCAB as u64).to_le_bytes()).unwrap();
        LogitsWriter(f)
    }
    fn push(&mut self, l: &[f32]) {
        // SAFETY: plain f32 slice as bytes.
        let b = unsafe { std::slice::from_raw_parts(l.as_ptr() as *const u8, l.len() * 4) };
        self.0.write_all(b).unwrap();
    }
}

struct LogitsReader(std::io::BufReader<std::fs::File>, Vec<u8>);
impl LogitsReader {
    fn open(p: &Path) -> LogitsReader {
        let mut f = std::io::BufReader::new(std::fs::File::open(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))));
        let mut h = [0u8; 12];
        f.read_exact(&mut h).unwrap_or_else(|e| die(&e.to_string()));
        if &h[..4] != b"FLG1" || u64::from_le_bytes(h[4..12].try_into().unwrap()) != N_VOCAB as u64 {
            die("--kld-ref: not an FLG1 logits file of this vocabulary");
        }
        LogitsReader(f, vec![0u8; N_VOCAB * 4])
    }
    fn next(&mut self) -> Option<Vec<f32>> {
        self.0.read_exact(&mut self.1).ok()?;
        Some(self.1.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

fn pct(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(s[((s.len() - 1) as f64 * p).round() as usize])
}

fn run(a: &Args) {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let model_dir = PathBuf::from(a.s("model-dir", &home.join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS").to_string_lossy()));
    // `--text F [--text-prefix 64]` (with --teacher-force): the prompt is the first `text-prefix` tokens of a text file
    // and the decode follows the text, as `export` does (lookahead QC on code text); otherwise the oracle dump.
    let (prompt, want): (Vec<i32>, Vec<i32>) = if let Some(tp) = a.0.get("text") {
        if !a.has("teacher-force") {
            die("--text needs --teacher-force (the decode follows the text)");
        }
        let text = std::fs::read_to_string(tp).unwrap_or_else(|e| die(&format!("--text: {e}")));
        let sh = Shards::open(&model_dir).unwrap_or_else(|e| die(&e));
        let tk = tokenizer::Tokenizer::from_gguf(&sh.paths[0]).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
        let all: Vec<i32> = tk.encode(&text, false, false).into_iter().map(|t| t as i32).collect();
        let pre: usize = a.n("text-prefix", 64);
        let n_new: usize = a.n("max-new", 400);
        if pre < 2 || all.len() < pre + n_new + 1 {
            die(&format!("--text: {} tokens, need --text-prefix {pre} + --max-new {n_new} + 1", all.len()));
        }
        (all[..pre].to_vec(), all[pre..pre + n_new + 1].to_vec())
    } else {
        let oracle = PathBuf::from(a.s("oracle", ""));
        let recs = read_tokens(&oracle.join("tokens.bin"));
        if recs.len() < 3 || recs[1].len() < 2 {
            die("oracle tokens.bin: expected warm-up, prompt, decode records");
        }
        (recs[1].clone(), recs[2..].iter().map(|r| r[0]).collect())
    };

    let exact = a.has("exact");
    let threads: usize = a.n("threads", if exact { 4 } else { 8 });
    let threads_batch: usize = a.n("threads-batch", if exact { 6 } else { threads });
    let ubatch: usize = a.n("ubatch", 4096);
    let n_ctx: u32 = a.n("n-ctx", 4096);
    let max_new: usize = a.n("max-new", 400);
    let cache_gib: f64 = a.n("cache-gib", 20.0);
    let teacher = a.has("teacher-force");
    let opts = if exact {
        FgOptions::default()
    } else {
        FgOptions {
            fixed_attn: attn_mode(a),
            shexp_first: a.flag("shexp-first", true) as i32,
            batch_min_tokens: if a.flag("stream-prefill", true) { 17 } else { 0 },
            profile: 0,
            rs_view: a.flag("rs-view", true) as i32,
            // FD-1 --trace-work needs the executor as its harness (bitwise and speed-neutral, ticket 017)
            executor: (a.flag("executor", false) || a.has("trace-work")) as i32,
            fuse: a.n("fuse", 0i32),
        }
    };
    let resident_first = !exact && a.flag("resident-first", true);
    let stream_prefill = opts.batch_min_tokens > 0;
    let sidecar = PathBuf::from(a.s("sidecar", &qwen4exp::sidecar::default_path().to_string_lossy()));
    let io_cpu: Option<usize> = a.0.get("io-cpu").and_then(|v| v.parse().ok());

    let t0 = Instant::now();
    let s = Shards::open(&model_dir).unwrap_or_else(|e| die(&e));
    let d = Dense::load(&s, DenseMode::VecDot).unwrap_or_else(|e| die(&e));
    let load_s = t0.elapsed().as_secs_f64();
    // the shard mappings of the experts are only needed for a mapped-expert prefill
    let gm = GraphModel::new(&s, &d, !stream_prefill).unwrap_or_else(|e| die(&e));
    let cfg = StreamConfig { cache_bytes: (cache_gib * (1u64 << 30) as f64) as usize, qd: a.n("qd", 32u32), io_cpu, prefault: true };
    let ts = Instant::now();
    let mut st = make_stream(a, &s, &sidecar, &cfg);
    st.resident_first = resident_first;
    st.batch_gemm = a.flag("batch-gemm", true);
    st.batch_tile = a.flag("expert-tile", !exact); // PREFILL: IQ2_S gate/up in 8-token tiles (bitwise = gemm4); default on (G4)
    // W4-I policy (G4): order on; read-ahead "auto" = 64 experts per layer on passes that start warm, 0 on cold ones;
    // `--pf-readahead N` forces N on every pass (0 = off); `--pf-ra-warm X` = the warm threshold (resident share)
    st.pf_order = a.flag("pf-order", !exact); // the exact path keeps the prefill switches off unless given (as W4-C)
    st.pf_order_last = a.flag("pf-order-last", false); // F123 (server --flash-pf-order 2)
    // G4' (W5-C): read-ahead default OFF (auto bought nothing beyond the order); `auto` stays selectable
    match a.s("pf-readahead", "0").as_str() {
        "auto" => {
            st.pf_readahead = 64;
            st.pf_readahead_auto = true;
        }
        v => {
            st.pf_readahead = v.parse().unwrap_or_else(|_| die("--pf-readahead: auto or a number"));
            st.pf_readahead_auto = false;
        }
    }
    st.pf_ra_warm = a.n("pf-ra-warm", 0.5f64);
    // FP-8: one pass per prompt: `--pf-max-pass N|auto` merges full `--ubatch` chunks into passes of up to N tokens
    // (auto = 8192), limited by MemAvailable; 0 (default) = plain chunks of `--ubatch`
    let pf_max_pass: usize = match a.s("pf-max-pass", "0").as_str() {
        "auto" => 8192,
        v => v.parse().unwrap_or_else(|_| die("--pf-max-pass: auto or a number")),
    };
    // W4-I per-phase IO core: the IO thread(s) move to these CPUs for the batched prefill (comma list), back to
    // `--io-cpu` afterwards (decode)
    let io_cpu_prefill: Vec<usize> = a.0.get("io-cpu-prefill").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let mut io_moved = 0usize; // IO threads re-pinned by the last --io-cpu-prefill switch (0 = none / not found)
    let stream_setup_s = ts.elapsed().as_secs_f64();
    // PREFILL: multi-column dense matmul for ubatches of >= 256 tokens (bitwise = ggml's mul_mat); default on (G4)
    let dense_gemm = a.flag("dense-gemm", !exact);
    qwen4exp::graph::set_dense_gemm(dense_gemm);
    // PREFILL FP-1: register-tile attention for prefill ubatches (bitwise): 0 off, 1 tiles (default on the fast path, G4),
    // 2 = 1 + the fused part+merge node (no partial tensor; W4-I, default only after QC-FP8 / G4')
    let attn_tile: i32 = match a.s("attn-tile", if exact { "0" } else { "1" }).as_str() {
        "true" | "on" => 1, // the boolean spellings
        "false" | "off" => 0,
        v => v.parse().ok().filter(|m| (0..=2).contains(m)).unwrap_or_else(|| die("--attn-tile: 0, 1 or 2")),
    };
    qwen4exp::graph::set_attn_tile_mode(attn_tile);
    qwen4exp::graph::attn_tile_units(true);
    qwen4exp::graph::dense_gemm_stats(true);
    let mut g = GraphCtx::new(&gm, n_ctx);
    g.set_options(&opts);
    g.set_graph_reuse(a.flag("graph-reuse", true));
    let ud = &*st as *const Stream as *mut std::ffi::c_void;
    eprintln!(
        "flash-m3: {} path, threads {threads}/{threads_batch}, {opts:?}, resident_first {resident_first}, cache {:.2} GiB ({stream_setup_s:.1} s), dense {:.2} GB ({load_s:.1} s)",
        if exact { "EXACT (M1)" } else { "fast" },
        st.cache.arena_bytes() as f64 / (1u64 << 30) as f64,
        d.arena_bytes as f64 / 1e9
    );

    let mut logits = vec![0f32; N_VOCAB];
    let mut save = a.0.get("save-logits").map(|p| LogitsWriter::new(Path::new(p)));
    let mut kref = a.0.get("kld-ref").map(|p| LogitsReader::open(Path::new(p)));
    let mut klds: Vec<f64> = Vec::new();
    let mut top1s: Vec<bool> = Vec::new();
    let mut top1_agree = 0usize;
    let mut on_logits = |l: &[f32], save: &mut Option<LogitsWriter>, kref: &mut Option<LogitsReader>| {
        if let Some(w) = save.as_mut() {
            w.push(l);
        }
        if let Some(r) = kref.as_mut() {
            if let Some(rl) = r.next() {
                let (k, same) = kld(&rl, l);
                klds.push(k);
                top1s.push(same);
                top1_agree += same as usize;
            }
        }
    };

    // ---- prefill ----
    let tp = Instant::now();
    let pstate = a.0.get("prefill-state").map(|p| PathBuf::from(format!("{p}.ctx{n_ctx}.tb{threads_batch}.{}", if exact { "exact" } else { "fast" })));
    let mut prefill_from_file = false;
    let stats_before_prefill = st.stats();
    if let Some(p) = pstate.as_ref().filter(|p| p.exists()) {
        let (toks, ex) = g.load_file(p).unwrap_or_else(|e| die(&e));
        if toks != prompt || ex.len() != N_VOCAB {
            die("prefill state file is for another prompt");
        }
        logits.copy_from_slice(&ex);
        prefill_from_file = true;
    } else {
        // SAFETY: `st` outlives every decode of `g`.
        unsafe {
            if stream_prefill {
                g.set_expert_fn(Some(Stream::callback()), ud);
            } else {
                g.set_expert_fn(None, std::ptr::null_mut());
            }
        }
        if !io_cpu_prefill.is_empty() {
            io_moved = st.set_io_cpus(&io_cpu_prefill);
        }
        let plan = pf_plan_for(prompt.len(), ubatch, pf_max_pass, attn_tile == 2);
        let mut at = 0;
        for n in plan {
            let ch = &prompt[at..at + n];
            at += n;
            g.decode(ch, threads_batch, &mut logits, None).unwrap_or_else(|e| die(&e));
            if let Some(e) = st.take_error() {
                die(&format!("prefill: {e}"));
            }
        }
        if let Some(p) = &pstate {
            g.save_file(p, &logits).unwrap_or_else(|e| die(&e));
        }
    }
    // decode: the IO thread(s) back on `--io-cpu`
    if !io_cpu_prefill.is_empty() {
        if let Some(c) = io_cpu {
            st.set_io_cpus(&[c]);
        }
    }
    let prefill_s = tp.elapsed().as_secs_f64();
    let prefill_cache = st.stats();
    on_logits(&logits, &mut save, &mut kref);
    let mut got: Vec<i32> = vec![argmax(&logits) as i32];
    eprintln!(
        "flash-m3: prefill {} tokens {prefill_s:.1} s (from file {prefill_from_file}, streamed {stream_prefill}), first token {} (want {})",
        prompt.len(),
        got[0],
        want[0]
    );

    // ---- decode ----
    // SAFETY: as above.
    unsafe { g.set_expert_fn(Some(Stream::callback()), ud) };
    // --profile N: the first N decode steps run node by node with per-kind timing (then normal speed;
    // those steps are excluded from ms_per_token)
    // --profile-from S: profile decode steps S..S+N-1 instead (warm cache: the first steps after the prefill
    // carry SSD reads that node-by-node execution cannot hide, ticket 002)
    let trace_work = a.has("trace-work") && !exact;
    if trace_work {
        g.set_trace(true);
        st.io.set_trace(true);
    }
    let w_all0 = st.counters.wait_ns.load(Relaxed);
    let profile_steps: usize = a.n("profile", 0);
    let profile_from: usize = a.n("profile-from", 1).max(1);
    let profile_last = profile_from + profile_steps - 1;
    let mut profile_rows: Vec<(String, u64, f64)> = Vec::new();
    let c0 = st.stats();
    let (w0, f0, calls0) = (st.counters.wait0_ns.load(Relaxed), st.counters.fetch_ns.load(Relaxed), st.counters.calls.load(Relaxed));
    let io0 = (st.io.counters.completed[0].load(Relaxed), st.io.counters.lat_ns[0].load(Relaxed));
    let mut first_mismatch: Option<usize> = if got[0] != want[0] { Some(0) } else { None };
    g.timing(true);
    let mut stage_ms = [0f64; 4];
    let mut step_ms: Vec<f64> = Vec::new();
    loop {
        let i = got.len();
        let input = if teacher {
            match want.get(i - 1) {
                Some(&t) => t,
                None => break,
            }
        } else {
            *got.last().unwrap()
        };
        if i >= max_new || EOG.contains(&input) {
            break;
        }
        let profiling = profile_steps > 0 && i >= profile_from && i <= profile_last;
        if profiling && i == profile_from {
            g.set_options(&FgOptions { profile: 1, ..opts });
            g.profile_reset();
        }
        let t = Instant::now();
        g.decode(&[input], threads, &mut logits, None).unwrap_or_else(|e| die(&e));
        if let Some(e) = st.take_error() {
            die(&format!("decode step {i}: {e}"));
        }
        if profiling {
            if i == profile_last {
                profile_rows = g.profile();
                g.set_options(&opts);
            }
        } else {
            step_ms.push(t.elapsed().as_secs_f64() * 1e3);
            let (st4, _) = g.timing(true);
            for k in 0..4 {
                stage_ms[k] += st4[k];
            }
        }
        if profiling {
            g.timing(true);
        }
        on_logits(&logits, &mut save, &mut kref);
        let next = argmax(&logits) as i32;
        got.push(next);
        if first_mismatch.is_none() && !want_ok(&want, i, next) {
            first_mismatch = Some(i);
            eprintln!("flash-m3: first token difference at generated token {i}: got {next}, want {:?}", want.get(i));
            // a teacher-forced run keeps going (its logits file must cover every step)
            if exact && !teacher && !a.has("keep-going") {
                break;
            }
        }
        if i % 50 == 0 {
            eprintln!("flash-m3: {i} tokens, {:.1} ms/token", step_ms.iter().sum::<f64>() / step_ms.len() as f64);
        }
        if !teacher && EOG.contains(&next) {
            break;
        }
    }
    drop(save);
    let ans: Vec<i32> = got.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    let want_ans: Vec<i32> = want.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    if let Some(p) = a.0.get("answer") {
        let tk = tokenizer::Tokenizer::from_gguf(&s.paths[0]).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
        let toks: Vec<u32> = ans.iter().map(|&t| t as u32).collect();
        std::fs::write(p, tk.decode_bytes(&toks, false)).unwrap_or_else(|e| die(&e.to_string()));
    }
    let n = step_ms.len().max(1) as f64;
    let toks = step_ms.len().max(1) as f64;
    let cs = st.stats();
    let dl = cs.demand_lookups - c0.demand_lookups;
    let dh = (cs.demand_hits_ready + cs.demand_hits_pending) - (c0.demand_hits_ready + c0.demand_hits_pending);
    let io_n = st.io.counters.completed[0].load(Relaxed) - io0.0;
    let io_lat = st.io.counters.lat_ns[0].load(Relaxed) - io0.1;
    let out = json!({
        "kind": "flash-m3-run",
        "path": if exact { "exact" } else { "fast" },
        "engine_commit": qwen4exp::ENGINE_COMMIT,
        "ggml_prod_cpu_sha256": env!("OXF_GGML_PROD_CPU_SHA"),
        "expert_tile": st.batch_tile,
        "prefill_io": {"pf_order": st.pf_order, "pf_order_last": st.pf_order_last, "pf_readahead": st.pf_readahead, "pf_readahead_auto": st.pf_readahead_auto, "pf_ra_warm": st.pf_ra_warm,
                       "readahead_issued": st.counters.readahead.load(Relaxed), "warm_passes": st.counters.ra_warm_passes.load(Relaxed),
                       "cold_passes": st.counters.ra_cold_passes.load(Relaxed), "pf_max_pass": pf_max_pass, "io_cpu_prefill": io_cpu_prefill, "io_threads_moved": io_moved},
        "attn_tile": {"on": attn_tile > 0, "mode": attn_tile, "units": qwen4exp::graph::attn_tile_units(false)},
        "dense_gemm": {"on": dense_gemm, "nodes_multi_column": qwen4exp::graph::dense_gemm_stats(false)[0], "nodes_ggml_fallback": qwen4exp::graph::dense_gemm_stats(false)[1]},
        "options": {"fixed_attn": opts.fixed_attn, "shexp_first": opts.shexp_first, "batch_min_tokens": opts.batch_min_tokens, "rs_view": opts.rs_view, "executor": opts.executor, "fuse": opts.fuse,
                    "resident_first": resident_first, "stream_prefill": stream_prefill, "batch_gemm": st.batch_gemm, "teacher_force": teacher,
                    "graph_reuse": a.flag("graph-reuse", true)},
        "cache_options": cache_opts_json(a, &st),
        "threads": threads, "threads_batch": threads_batch, "ubatch": ubatch, "n_ctx": n_ctx,
        "cache_gib": cache_gib, "cache_arena_bytes": st.cache.arena_bytes(), "qd": cfg.qd, "io_cpu": io_cpu,
        "gomp_cpu_affinity": std::env::var("GOMP_CPU_AFFINITY").ok(),
        "prompt_tokens": prompt.len(),
        "generated": ans.len(), "want_generated": want_ans.len(),
        "first_mismatch": first_mismatch,
        "tokens_identical": ans == want_ans && first_mismatch.is_none(),
        "load_s": load_s, "stream_setup_s": stream_setup_s,
        "prefill_s": prefill_s, "prefill_from_file": prefill_from_file,
        "prefill_cache": {"lookups": prefill_cache.demand_lookups - stats_before_prefill.demand_lookups,
                          "read_gib": (prefill_cache.read_bytes - stats_before_prefill.read_bytes) as f64 / (1u64 << 30) as f64},
        "decode_steps": step_ms.len(),
        "ms_per_token": {"mean": step_ms.iter().sum::<f64>() / n, "median": pct(&step_ms, 0.5), "p90": pct(&step_ms, 0.9), "max": pct(&step_ms, 1.0)},
        // lookahead QC: the first 64 decode steps (cache still warming) and the rest, separately
        "ms_per_token_split64": {"first": split_mean(&step_ms, 0, 64), "rest": split_mean(&step_ms, 64, usize::MAX)},
        "executor": {"graphs_scheduled": g.exec_stats()[0], "nodes": g.exec_stats()[1], "barriers": g.exec_stats()[2]},
        "trace": if trace_work { trace_json(&g.trace_text(), &st.io.take_stamps(), (st.counters.wait_ns.load(Relaxed) - w_all0) as f64 / 1e6 / threads as f64, step_ms.len()) } else { serde_json::Value::Null },
        "decode_stage_ms_per_token": {"graph_build": stage_ms[0] / n, "gallocr": stage_ms[1] / n, "inputs": stage_ms[2] / n, "compute": stage_ms[3] / n},
        "expert_wait_ms_per_token_thread0": (st.counters.wait0_ns.load(Relaxed) - w0) as f64 / 1e6 / toks,
        "fetch_ms_per_token": (st.counters.fetch_ns.load(Relaxed) - f0) as f64 / 1e6 / toks,
        "expert_calls_per_token": (st.counters.calls.load(Relaxed) - calls0) as f64 / toks,
        "decode_cache": {"lookups": dl, "hit_rate": dh as f64 / dl.max(1) as f64,
                         "read_mib_per_token": (cs.read_bytes - c0.read_bytes) as f64 / 1048576.0 / toks},
        "decode_io": {"reads": io_n, "mean_latency_us": io_lat as f64 / 1e3 / io_n.max(1) as f64,
                      // F-IO: the IO configuration in effect (OXF_IO), whole-run worker parks / parts / promotions, and the
                      // PENDING-slot spin summed over ALL compute threads (whole run; thread 0's share is above)
                      "config": st.io_desc.clone(), "parks_total": st.io.counters.parks.load(Relaxed),
                      "parts_total": st.io.counters.parts.load(Relaxed), "promoted_total": st.io.counters.promoted.load(Relaxed),
                      "expert_wait_ms_all_threads_total": st.counters.wait_ns.load(Relaxed) as f64 / 1e6},
        "kld": if klds.is_empty() { serde_json::Value::Null } else { json!({
            "steps": klds.len(), "mean": klds.iter().sum::<f64>() / klds.len() as f64,
            "median": pct(&klds, 0.5), "p99": pct(&klds, 0.99), "max": pct(&klds, 1.0),
            "top1_agree": top1_agree as f64 / klds.len() as f64 }) },
        "profile_from": profile_from,
        "profile_ms_per_token": if profile_rows.is_empty() { serde_json::Value::Null } else {
            json!(profile_rows.iter().map(|(k, c, ms)| json!({"kind": k, "calls_per_token": *c as f64 / profile_steps as f64,
                                                               "ms_per_token": ms / profile_steps as f64})).collect::<Vec<_>>()) },
        "memory": {"vm_hwm_kib": status_kib("VmHWM:"), "cgroup_memory_peak": cgroup_file("memory.peak"), "cgroup_memory_max": cgroup_file("memory.max"),
                   "anon_huge_kib": qwen4exp::cli::anon_huge_kib(), "dense_arena_bytes": d.arena_bytes, "cache_arena_bytes": st.cache.arena_bytes()},
    });
    let txt = serde_json::to_string_pretty(&out).unwrap();
    println!("{txt}");
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, &txt).ok();
    }
    if let Some(p) = a.0.get("kld-out") {
        // step 0 = the logits after the prompt; step i = after decoding generated token i
        let csv: String = std::iter::once("step,kld,top1_same\n".to_string())
            .chain(klds.iter().zip(&top1s).enumerate().map(|(i, (k, t))| format!("{i},{k:e},{}\n", *t as u8)))
            .collect();
        std::fs::write(p, csv).unwrap_or_else(|e| die(&format!("--kld-out {p}: {e}")));
    }
    drop(g);
    finish_stream(a, &mut st);
    drop(st);
}

/// `flash-m3 kld --ref A.flg --test B.flg [--windows 0:1,1:100,100:354] [--out F.json]` (no model): KL(ref || test)
/// per step of two --save-logits files, summarised over windows [a, b) of steps, plus top-1 agreement.
fn kld_files(a: &Args) {
    let mut ra = LogitsReader::open(Path::new(&a.s("ref", "")));
    let mut rb = LogitsReader::open(Path::new(&a.s("test", "")));
    let mut ks: Vec<(f64, bool)> = Vec::new();
    while let (Some(x), Some(y)) = (ra.next(), rb.next()) {
        ks.push(kld(&x, &y));
    }
    let n = ks.len();
    let wins: Vec<(usize, usize)> = a
        .s("windows", &format!("0:1,1:100,100:{n},0:{n}"))
        .split(',')
        .map(|w| {
            let (x, y) = w.split_once(':').unwrap_or_else(|| die("--windows a:b,..."));
            (x.parse().unwrap_or_else(|_| die("--windows")), y.parse::<usize>().unwrap_or_else(|_| die("--windows")).min(n))
        })
        .collect();
    let rows: Vec<serde_json::Value> = wins
        .iter()
        .filter(|(x, y)| x < y)
        .map(|&(x, y)| {
            let v: Vec<f64> = ks[x..y].iter().map(|k| k.0).collect();
            json!({"from": x, "to": y, "steps": y - x, "mean": v.iter().sum::<f64>() / v.len() as f64,
                   "median": pct(&v, 0.5), "p90": pct(&v, 0.9), "max": pct(&v, 1.0),
                   "top1_flips": ks[x..y].iter().filter(|k| !k.1).count()})
        })
        .collect();
    let out = json!({"kind": "flash-m3-kld", "ref": a.s("ref", ""), "test": a.s("test", ""), "steps": n, "windows": rows,
                     "per_step": ks.iter().map(|k| k.0).collect::<Vec<f64>>()});
    let txt = serde_json::to_string_pretty(&out).unwrap();
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, &txt).ok();
    }
    println!("{}", serde_json::to_string_pretty(&out["windows"]).unwrap());
}

fn prefill(a: &Args) {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let model_dir = PathBuf::from(a.s("model-dir", &home.join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS").to_string_lossy()));
    let n_tok: usize = a.n("n-tokens", 0);
    let text = std::fs::read_to_string(a.s("text", "")).unwrap_or_else(|e| die(&format!("--text: {e}")));
    let mapped = a.has("mapped");
    let threads: usize = a.n("threads", 8);
    let ubatch: usize = a.n("ubatch", 4096);
    let reps: usize = a.n("reps", 2).max(1);
    let n_ctx: u32 = a.n("n-ctx", (((n_tok + 64) as u32).div_ceil(256) * 256).max(4096));
    let cache_gib: f64 = a.n("cache-gib", 20.0);
    let opts = FgOptions {
        fixed_attn: attn_mode(a),
        shexp_first: a.flag("shexp-first", true) as i32,
        batch_min_tokens: if mapped { 0 } else { 17 },
        profile: 0,
        rs_view: a.flag("rs-view", true) as i32,
        executor: a.flag("executor", false) as i32,
        fuse: a.n("fuse", 0i32),
    };
    let sidecar = PathBuf::from(a.s("sidecar", &qwen4exp::sidecar::default_path().to_string_lossy()));
    let io_cpu: Option<usize> = a.0.get("io-cpu").and_then(|v| v.parse().ok());

    let t0 = Instant::now();
    let s = Shards::open(&model_dir).unwrap_or_else(|e| die(&e));
    let tk = tokenizer::Tokenizer::from_gguf(&s.paths[0]).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
    let all: Vec<i32> = tk.encode(&text, false, false).into_iter().map(|t| t as i32).collect();
    if n_tok == 0 || all.len() < n_tok {
        die(&format!("--n-tokens {n_tok}: the text has {} tokens", all.len()));
    }
    let prompt = &all[..n_tok];
    let d = Dense::load(&s, DenseMode::VecDot).unwrap_or_else(|e| die(&e));
    let gm = GraphModel::new(&s, &d, mapped).unwrap_or_else(|e| die(&e));
    let load_s = t0.elapsed().as_secs_f64();
    let cfg = StreamConfig { cache_bytes: (cache_gib * (1u64 << 30) as f64) as usize, qd: a.n("qd", 32u32), io_cpu, prefault: true };
    let mut st = make_stream(a, &s, &sidecar, &cfg);
    st.resident_first = a.flag("resident-first", true);
    st.batch_gemm = a.flag("batch-gemm", true);
    st.batch_tile = a.flag("expert-tile", true); // PREFILL: IQ2_S gate/up in 8-token tiles (bitwise = gemm4); default on (G4)
    // W4-I policy (G4): order on; read-ahead "auto" = 64 experts per layer on passes that start warm, 0 on cold ones;
    // `--pf-readahead N` forces N on every pass (0 = off); `--pf-ra-warm X` = the warm threshold (resident share)
    st.pf_order = a.flag("pf-order", true);
    st.pf_order_last = a.flag("pf-order-last", false); // F123 (server --flash-pf-order 2)
    // G4' (W5-C): read-ahead default OFF; `auto` stays selectable
    match a.s("pf-readahead", "0").as_str() {
        "auto" => {
            st.pf_readahead = 64;
            st.pf_readahead_auto = true;
        }
        v => {
            st.pf_readahead = v.parse().unwrap_or_else(|_| die("--pf-readahead: auto or a number"));
            st.pf_readahead_auto = false;
        }
    }
    st.pf_ra_warm = a.n("pf-ra-warm", 0.5f64);
    // FP-8: one pass per prompt: `--pf-max-pass N|auto` merges full `--ubatch` chunks into passes of up to N tokens
    // (auto = 8192), limited by MemAvailable; 0 (default) = plain chunks of `--ubatch`
    let pf_max_pass: usize = match a.s("pf-max-pass", "0").as_str() {
        "auto" => 8192,
        v => v.parse().unwrap_or_else(|_| die("--pf-max-pass: auto or a number")),
    };
    // W4-I per-phase IO core: the IO thread(s) move to these CPUs for the batched prefill (comma list), back to
    // `--io-cpu` afterwards (decode)
    let io_cpu_prefill: Vec<usize> = a.0.get("io-cpu-prefill").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let mut io_moved = 0usize; // IO threads re-pinned by the last --io-cpu-prefill switch (0 = none / not found)
    // PREFILL: multi-column dense matmul for ubatches of >= 256 tokens (bitwise = ggml's mul_mat); default on (G4)
    let dense_gemm = a.flag("dense-gemm", true);
    qwen4exp::graph::set_dense_gemm(dense_gemm);
    // PREFILL FP-1: register-tile attention for prefill ubatches (bitwise): 0 off, 1 tiles (default on, G4),
    // 2 = 1 + the fused part+merge node (no partial tensor; W4-I, default only after QC-FP8 / G4')
    let attn_tile: i32 = match a.s("attn-tile", "1").as_str() {
        "true" | "on" => 1, // the boolean spellings
        "false" | "off" => 0,
        v => v.parse().ok().filter(|m| (0..=2).contains(m)).unwrap_or_else(|| die("--attn-tile: 0, 1 or 2")),
    };
    qwen4exp::graph::set_attn_tile_mode(attn_tile);
    qwen4exp::graph::attn_tile_units(true);
    qwen4exp::graph::dense_gemm_stats(true);
    let mut g = GraphCtx::new(&gm, n_ctx);
    let ud = &*st as *const Stream as *mut std::ffi::c_void;
    // SAFETY: `st` outlives every decode of `g`.
    unsafe {
        if mapped {
            g.set_expert_fn(None, std::ptr::null_mut());
        } else {
            g.set_expert_fn(Some(Stream::callback()), ud);
        }
    }
    eprintln!("flash-m3 prefill: {n_tok} tokens, {} experts, threads {threads}, ubatch {ubatch}, n_ctx {n_ctx}, {opts:?}",
              if mapped { "mapped" } else { "streamed" });
    let mut logits = vec![0f32; N_VOCAB];
    let mut first: Option<Vec<f32>> = None;
    let mut rows = Vec::new();
    let mut profile_rows: Vec<(String, u64, f64)> = Vec::new();
    let mut deterministic = true;
    let mut plan_used: Vec<usize> = Vec::new();
    for rep in 0..reps {
        g.clear();
        let prof = rep == 0 && a.has("profile");
        g.set_options(&FgOptions { profile: prof as i32, ..opts });
        if prof {
            g.profile_reset();
        }
        let c0 = st.stats();
        let (w0, wall0) = (st.counters.wait0_ns.load(Relaxed), st.counters.wait_ns.load(Relaxed));
        let tp = Instant::now();
        let mut chunk_s = Vec::new();
        if !io_cpu_prefill.is_empty() {
            io_moved = st.set_io_cpus(&io_cpu_prefill);
        }
        let plan = pf_plan_for(prompt.len(), ubatch, pf_max_pass, attn_tile == 2);
        plan_used = plan.clone();
        qwen4exp::graph::pf_phase_ms(true);
        let stage0: Vec<u64> = st.counters.batch_stage_ns.iter().map(|c| c.load(Relaxed)).collect();
        let mut at = 0;
        for n in plan {
            let ch = &prompt[at..at + n];
            at += n;
            let tc = Instant::now();
            g.decode(ch, threads, &mut logits, None).unwrap_or_else(|e| die(&e));
            if let Some(e) = st.take_error() {
                die(&format!("prefill: {e}"));
            }
            chunk_s.push(tc.elapsed().as_secs_f64());
        }
        let secs = tp.elapsed().as_secs_f64();
        let c1 = st.stats();
        // FP-9: where the pass went, from clocks inside the ops (thread 0 wall; no node-by-node mode needed)
        let (pms, pcalls) = qwen4exp::graph::pf_phase_ms(false);
        let stage: Vec<f64> = st.counters.batch_stage_ns.iter().zip(&stage0).map(|(c, s0)| (c.load(Relaxed) - s0) as f64 / 1e6).collect();
        let expert_ms: f64 = stage.iter().sum();
        let phase = json!({"attention_tiles_ms": pms[0], "dense_pack_ms": pms[1], "dense_mm_ms": pms[2],
            "expert_plan_quant_ms": stage[0], "expert_gate_up_ms": stage[1], "expert_swiglu_ms": stage[2], "expert_down_ms": stage[3],
            "rest_ms": secs * 1e3 - pms[0] - pms[1] - pms[2] - expert_ms, "calls": {"attention": pcalls[0], "dense_pack": pcalls[1], "dense_mm": pcalls[2]},
            "note": "thread-0 wall inside the ops; attention only when --attn-tile is on, dense only for multi-column nodes; expert stages include their SSD waits"});
        if prof {
            profile_rows = g.profile();
        }
        match &first {
            None => {
                if let Some(p) = a.0.get("save-state") {
                    g.save_file(Path::new(p), &logits).unwrap_or_else(|e| die(&e));
                }
                if let Some(p) = a.0.get("save-logits") {
                    let mut w = LogitsWriter::new(Path::new(p));
                    w.push(&logits);
                }
                first = Some(logits.clone());
            }
            Some(f) => {
                // SAFETY: f32 slices compared as bytes.
                let same = unsafe {
                    std::slice::from_raw_parts(f.as_ptr() as *const u8, f.len() * 4) == std::slice::from_raw_parts(logits.as_ptr() as *const u8, logits.len() * 4)
                };
                deterministic &= same;
            }
        }
        let row = json!({
            "rep": rep + 1, "profiled": prof, "seconds": secs, "tok_per_s": n_tok as f64 / secs, "ubatch_seconds": chunk_s,
            "phase_ms": phase, "reads_per_token_mib": (c1.read_bytes - c0.read_bytes) as f64 / (1u64 << 20) as f64 / n_tok as f64,
            "expert_lookups": c1.demand_lookups - c0.demand_lookups,
            "expert_hits": (c1.demand_hits_ready + c1.demand_hits_pending) - (c0.demand_hits_ready + c0.demand_hits_pending),
            "read_gib": (c1.read_bytes - c0.read_bytes) as f64 / (1u64 << 30) as f64,
            "read_gb_per_s": (c1.read_bytes - c0.read_bytes) as f64 / 1e9 / secs,
            "expert_wait_s_thread0": (st.counters.wait0_ns.load(Relaxed) - w0) as f64 / 1e9,
            "expert_wait_s_all_threads": (st.counters.wait_ns.load(Relaxed) - wall0) as f64 / 1e9,
        });
        eprintln!("flash-m3 prefill: {row}");
        rows.push(row);
    }
    let out = json!({
        "kind": "flash-m3-prefill", "engine_commit": qwen4exp::ENGINE_COMMIT, "ggml_prod_cpu_sha256": env!("OXF_GGML_PROD_CPU_SHA"),
        "experts": if mapped { "mapped" } else { "streamed" },
        "expert_tile": st.batch_tile,
        "prefill_io": {"pf_order": st.pf_order, "pf_order_last": st.pf_order_last, "pf_readahead": st.pf_readahead, "pf_readahead_auto": st.pf_readahead_auto, "pf_ra_warm": st.pf_ra_warm,
                       "readahead_issued": st.counters.readahead.load(Relaxed), "warm_passes": st.counters.ra_warm_passes.load(Relaxed),
                       "cold_passes": st.counters.ra_cold_passes.load(Relaxed), "pf_max_pass": pf_max_pass, "io_cpu_prefill": io_cpu_prefill, "io_threads_moved": io_moved},
        "attn_tile": {"on": attn_tile > 0, "mode": attn_tile, "units": qwen4exp::graph::attn_tile_units(false)},
        "dense_gemm": {"on": dense_gemm, "nodes_multi_column": qwen4exp::graph::dense_gemm_stats(false)[0], "nodes_ggml_fallback": qwen4exp::graph::dense_gemm_stats(false)[1]},
        "options": {"fixed_attn": opts.fixed_attn, "shexp_first": opts.shexp_first, "batch_min_tokens": opts.batch_min_tokens, "resident_first": st.resident_first, "batch_gemm": st.batch_gemm},
        "cache_options": cache_opts_json(a, &st),
        "ubatch_plan": plan_used,
        "n_tokens": n_tok, "text_tokens": all.len(), "threads": threads, "ubatch": ubatch, "n_ctx": n_ctx,
        "cache_gib": cache_gib, "qd": cfg.qd, "io_cpu": io_cpu, "gomp_cpu_affinity": std::env::var("GOMP_CPU_AFFINITY").ok(),
        "load_s": load_s, "reps": rows, "deterministic": deterministic,
        "argmax": argmax(first.as_ref().unwrap()),
        "profile_ms": if profile_rows.is_empty() { serde_json::Value::Null } else {
            json!(profile_rows.iter().map(|(k, c, ms)| json!({"kind": k, "calls": c, "ms": ms})).collect::<Vec<_>>()) },
        "memory": {"vm_hwm_kib": status_kib("VmHWM:"), "cgroup_memory_peak": cgroup_file("memory.peak"), "cgroup_memory_max": cgroup_file("memory.max")},
    });
    let txt = serde_json::to_string_pretty(&out).unwrap();
    println!("{txt}");
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, &txt).ok();
    }
    drop(g);
    finish_stream(a, &mut st);
    drop(st);
    if !deterministic {
        std::process::exit(1);
    }
}

/// `flash-m3 bench-expert [--pairs N] [--reps R]`: one expert's gate (640 x 2560, IQ2_S / IQ3_XXS R4) and down
/// (2560 x 640, IQ4_NL) against N tokens, single thread, random weights and activations.
fn bench_expert(a: &Args) {
    use qwen4exp::ffi::{self, W};
    use std::ffi::c_void;
    let np: usize = a.n("pairs", 40);
    let reps: usize = a.n("reps", 5);
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut rows_out = Vec::new();
    for w in [W::Iq2S, W::Iq3Xxs, W::Iq4Nl] {
        let down = w == W::Iq4Nl;
        let (k, rows) = if down { (640usize, 2560usize) } else { (2560usize, 640usize) };
        let rb = w.row_bytes(k);
        let mut m: Vec<u8> = (0..rows * rb).map(|_| rnd() as u8).collect();
        let bb = w.block().1;
        for b in 0..rows * rb / bb {
            let h: u16 = 0x2000 + (b as u16 % 512);
            m[b * bb..b * bb + 2].copy_from_slice(&h.to_le_bytes());
        }
        let at = if down { kernels_sys::ActType::Q8_0 } else { kernels_sys::ActType::Q8K };
        let ab = at.bytes(k);
        let mut acts = vec![0u8; np * ab];
        for t in 0..np {
            let f: Vec<f32> = (0..k).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
            kernels_sys::quantize_act(at, &f, &mut acts[t * ab..(t + 1) * ab]);
        }
        if let Some(ty) = w.r4() {
            assert!(unsafe { ffi::ggml_cpu_expert_r4_slice_to_r4(ty, m.as_mut_ptr(), k as i64, rows as i64) });
        }
        let (mut old, mut new) = (vec![0f32; np * rows], vec![0f32; np * rows]);
        let (mut t_old, mut t_new) = (f64::MAX, f64::MAX);
        let ng = np.div_ceil(8);
        let (mut xc, mut xd) = (vec![0i8; ng * k * 8], vec![0f32; ng * (k / 32) * 8]);
        const TR: usize = 32;
        let mut scratch = vec![0u32; unsafe { ffi::pf_expert_iq2s_scratch(2560, TR as i64).max(ffi::pf_expert_iq3_scratch(2560, TR as i64)).max(ffi::pf_expert_iq4nl_scratch(640, TR as i64)) }.div_ceil(4)];
        for _ in 0..reps {
            // today's path, in the stream op's loop order (chunk-major)
            let t0 = Instant::now();
            unsafe {
                if let Some(ty) = w.r4() {
                    for r0 in (0..rows).step_by(16) {
                        for g in (0..np).step_by(4) {
                            if g + 4 <= np {
                                let s4: [*mut f32; 4] = std::array::from_fn(|t| old.as_mut_ptr().add((g + t) * rows + r0));
                                let y4: [*const u8; 4] = std::array::from_fn(|t| acts.as_ptr().add((g + t) * ab));
                                ffi::ggml_cpu_expert_r4_gemm4(ty, k as i64, s4.as_ptr(), m.as_ptr().add(r0 * rb), y4.as_ptr(), 16);
                            } else {
                                for t in g..np {
                                    ffi::ggml_cpu_expert_r4_gemv(ty, k as i64, old.as_mut_ptr().add(t * rows + r0), m.as_ptr().add(r0 * rb), acts.as_ptr().add(t * ab), 16);
                                }
                            }
                        }
                    }
                } else {
                    for r0 in (0..rows).step_by(32) {
                        for t in 0..np {
                            ffi::gemv_vecdot(w, m.as_ptr(), k, r0, r0 + 32, acts.as_ptr().add(t * ab), old.as_mut_ptr().add(t * rows + r0));
                        }
                    }
                }
            }
            t_old = t_old.min(t0.elapsed().as_secs_f64());
            // 8-token tiles (pack included)
            let t0 = Instant::now();
            unsafe {
                for g in 0..ng {
                    let n = (np - g * 8).min(8);
                    let vy: [*const u8; 8] = std::array::from_fn(|t| if t < n { acts.as_ptr().add((g * 8 + t) * ab) } else { std::ptr::null() });
                    if down {
                        ffi::pf_expert_pack_x8_q8_0(k as i64, n as i32, vy.as_ptr(), xc.as_mut_ptr().add(g * k * 8), xd.as_mut_ptr().add(g * (k / 32) * 8));
                    } else {
                        ffi::pf_expert_pack_x8(k as i64, n as i32, vy.as_ptr(), xc.as_mut_ptr().add(g * k * 8), xd.as_mut_ptr().add(g * (k / 256) * 8));
                    }
                }
                let f = match w {
                    W::Iq2S => ffi::pf_expert_iq2s_rows,
                    W::Iq3Xxs => ffi::pf_expert_iq3_rows,
                    _ => ffi::pf_expert_iq4nl_rows,
                };
                let mut s: Vec<*mut f32> = vec![std::ptr::null_mut(); ng * 8];
                for r0 in (0..rows).step_by(TR) {
                    for t in 0..np {
                        s[t] = new.as_mut_ptr().add(t * rows + r0);
                    }
                    f(k as i64, m.as_ptr().add(r0 * rb), TR as i64, xc.as_ptr(), xd.as_ptr(), ng as i64, s.as_ptr(), scratch.as_mut_ptr() as *mut c_void);
                }
            }
            t_new = t_new.min(t0.elapsed().as_secs_f64());
        }
        let same = old.iter().zip(&new).all(|(a, b)| a.to_bits() == b.to_bits());
        let gmac = (k * rows * np) as f64 / 1e9;
        println!("{w:?} K {k} rows {rows} pairs {np}: today {:.2} ms ({:.1} GMAC/s)  tiles {:.2} ms ({:.1} GMAC/s)  x{:.2}  bitwise {same}",
            t_old * 1e3, gmac / t_old, t_new * 1e3, gmac / t_new, t_old / t_new);
        rows_out.push(json!({"type": format!("{w:?}"), "k": k, "rows": rows, "pairs": np, "ms_today": t_old * 1e3, "ms_tiles": t_new * 1e3, "bitwise": same}));
    }
    println!("{}", json!({"kind": "flash-m3-bench-expert", "threads": 1, "rows": rows_out}));
}

fn read_fgd(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
    if b.len() < 40 || &b[..4] != b"FGD1" || i32::from_le_bytes(b[4..8].try_into().unwrap()) != 0 {
        die(&format!("{}: not an F32 FGD1 dump", p.display()));
    }
    b[40..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Learned-lookahead training data (PROGRESS-M3 "Slotstream comparison"): teacher-forced decode over a text, dumping
/// per step the input embedding and the final residual; the route trace (--route-trace) gives each step's experts.
fn export(a: &Args) {
    use qwen4exp::graph::Dump;
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let model_dir = PathBuf::from(a.s("model-dir", &home.join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS").to_string_lossy()));
    let n_tok: usize = a.n("n-tokens", 0);
    let prefix: usize = a.n("prefix", 64);
    let text = std::fs::read_to_string(a.s("text", "")).unwrap_or_else(|e| die(&format!("--text: {e}")));
    let threads: usize = a.n("threads", 8);
    let cache_gib: f64 = a.n("cache-gib", 21.0);
    let n_ctx: u32 = a.n("n-ctx", (((n_tok + 64) as u32).div_ceil(256) * 256).max(4096));
    let s = Shards::open(&model_dir).unwrap_or_else(|e| die(&e));
    let tk = tokenizer::Tokenizer::from_gguf(&s.paths[0]).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
    let all: Vec<i32> = tk.encode(&text, false, false).into_iter().map(|t| t as i32).collect();
    if n_tok <= prefix + 1 || all.len() < n_tok {
        die(&format!("--n-tokens {n_tok} (> --prefix {prefix}): the text has {} tokens", all.len()));
    }
    let d = Dense::load(&s, DenseMode::VecDot).unwrap_or_else(|e| die(&e));
    let gm = GraphModel::new(&s, &d, false).unwrap_or_else(|e| die(&e));
    let sidecar = PathBuf::from(a.s("sidecar", &qwen4exp::sidecar::default_path().to_string_lossy()));
    let cfg = StreamConfig { cache_bytes: (cache_gib * (1u64 << 30) as f64) as usize, qd: 32, io_cpu: a.0.get("io-cpu").and_then(|v| v.parse().ok()), prefault: true };
    let mut st = make_stream(a, &s, &sidecar, &cfg);
    if !a.has("route-trace") {
        die("export needs --route-trace F.csv (the routes are the labels)");
    }
    let opts = FgOptions { fixed_attn: attn_mode(a), shexp_first: 1, batch_min_tokens: 17, profile: 0, rs_view: 1, executor: 0, fuse: 0 };
    let mut g = GraphCtx::new(&gm, n_ctx);
    g.set_options(&opts);
    let ud = &*st as *const Stream as *mut std::ffi::c_void;
    // SAFETY: `st` outlives every decode of `g`.
    unsafe { g.set_expert_fn(Some(Stream::callback()), ud) };
    let mut logits = vec![0f32; N_VOCAB];
    let t0 = Instant::now();
    g.decode(&all[..prefix], threads, &mut logits, None).unwrap_or_else(|e| die(&e));
    if let Some(e) = st.take_error() {
        die(&format!("prefill: {e}"));
    }
    let outp = a.s("out", "");
    let dir = format!("{outp}.dumpdir");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| die(&e.to_string()));
    let dump = Dump { dir: dir.clone(), names: vec!["h_nextn".into(), "model.input_embed".into()] };
    let mut w = std::io::BufWriter::new(std::fs::File::create(&outp).unwrap_or_else(|e| die(&format!("--out {outp}: {e}"))));
    w.write_all(b"LAE1").unwrap();
    for v in [2560u32, 10240u32] {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    for i in prefix..n_tok {
        g.decode(&all[i..i + 1], threads, &mut logits, Some(&dump)).unwrap_or_else(|e| die(&e));
        if let Some(e) = st.take_error() {
            die(&format!("step {i}: {e}"));
        }
        let h = read_fgd(&Path::new(&dir).join("h_nextn.bin"));
        let e = read_fgd(&Path::new(&dir).join("model.input_embed.bin"));
        if h.len() != 10240 || e.len() != 2560 {
            die(&format!("step {i}: dump sizes h {} embd {}", h.len(), e.len()));
        }
        w.write_all(&(i as i32).to_le_bytes()).unwrap();
        for x in e.iter().chain(h.iter()) {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
        if (i - prefix) % 500 == 0 {
            eprintln!("flash-m3 export: step {i}/{n_tok}, {:.0} s", t0.elapsed().as_secs_f64());
        }
    }
    drop(w);
    std::fs::remove_dir_all(&dir).ok();
    drop(g);
    finish_stream(a, &mut st);
    println!("{}", json!({"kind": "flash-m3-export", "out": outp, "steps": n_tok - prefix, "prefix": prefix, "text_tokens": all.len(),
                         "seconds": t0.elapsed().as_secs_f64(), "route_trace": a.0.get("route-trace")}));
}

/// FD-1: the executor trace table + IO stamps -> JSON (per token: work / graph-barrier wait per kind, sorted by work+wait;
/// fused-barrier spin; fused / fallback counts; read queue wait and device latency; expert-unit spin per thread).
fn trace_json(text: &str, stamps: &[qwen4exp::io::ReadStamp], unit_wait_ms_per_thread: f64, tokens: usize) -> serde_json::Value {
    let mut kinds = Vec::new();
    let mut tot = serde_json::Map::new();
    let mut fused = Vec::new();
    let mut ntok = tokens.max(1) as f64;
    for l in text.lines() {
        let f: Vec<&str> = l.split('\t').collect();
        if l.starts_with("#fused") && f.len() >= 4 {
            fused.push(json!({"phase": f[1], "built": f[2].parse::<u64>().unwrap_or(0), "fallback": f[3].parse::<u64>().unwrap_or(0)}));
        } else if let Some(k) = l.strip_prefix('#') {
            if f.len() >= 2 {
                let v: f64 = f[1].parse().unwrap_or(0.0);
                if k.starts_with("tokens") {
                    ntok = v.max(1.0);
                }
                tot.insert(f[0].trim_start_matches('#').to_string(), json!(v));
            }
        } else if f.len() >= 4 {
            let c: f64 = f[1].parse().unwrap_or(0.0);
            let w: f64 = f[2].parse().unwrap_or(0.0);
            let b: f64 = f[3].parse().unwrap_or(0.0);
            kinds.push((f[0].to_string(), c, w, b));
        }
    }
    kinds.sort_by(|a, b| (b.2 + b.3).partial_cmp(&(a.2 + a.3)).unwrap());
    let (sw, sb): (f64, f64) = kinds.iter().fold((0.0, 0.0), |(x, y), k| (x + k.2, y + k.3));
    let mut q: Vec<f64> = stamps.iter().map(|s| (s.submit_ns.saturating_sub(s.push_ns)) as f64 / 1e6).collect();
    let mut d: Vec<f64> = stamps.iter().map(|s| (s.done_ns.saturating_sub(s.submit_ns)) as f64 / 1e6).collect();
    q.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |v: &[f64], p: f64| if v.is_empty() { 0.0 } else { v[((v.len() - 1) as f64 * p) as usize] };
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    json!({
        "tokens": ntok,
        "work_ms_per_token": sw / ntok, "graph_barrier_wait_ms_per_token": sb / ntok,
        "region_ms_per_token": tot.get("region_ms").and_then(|v| v.as_f64()).unwrap_or(0.0) / ntok,
        "fused_barrier_spin_ms_per_token_per_thread": tot.get("fuse_spin_ms_all_threads").and_then(|v| v.as_f64()).unwrap_or(0.0) / ntok,
        "expert_unit_spin_ms_per_token_per_thread": unit_wait_ms_per_thread / ntok,
        "fused_counts_last_graph": fused,
        "kinds": kinds.iter().take(60).map(|k| json!({"kind": k.0, "calls_per_token": k.1 / ntok, "work_ms": k.2 / ntok, "wait_ms": k.3 / ntok})).collect::<Vec<_>>(),
        "io": {"reads": stamps.len(), "reads_per_token": stamps.len() as f64 / ntok,
               "queue_wait_ms": {"mean": mean(&q), "p90": pct(&q, 0.9), "max": pct(&q, 1.0)},
               "device_ms": {"mean": mean(&d), "p50": pct(&d, 0.5), "p90": pct(&d, 0.9), "max": pct(&d, 1.0)}},
    })
}

/// FP-8: the prefill ubatches of a prompt of `n` tokens. `max_pass` 0 = plain chunks of `ubatch`; otherwise full
/// chunks are merged into passes of up to `max_pass` tokens, as far as MemAvailable (minus a 1.5 GiB margin) allows by
/// the planner's estimate (the attention partial tensor counts unless `--attn-tile 2` fuses it away). See qwen4exp::pf_plan.
fn pf_plan_for(n: usize, ubatch: usize, max_pass: usize, fused_attn: bool) -> Vec<usize> {
    use qwen4exp::pf_plan;
    if max_pass == 0 {
        return pf_plan::plan_ubatches(n, ubatch, 0);
    }
    let budget = pf_plan::mem_available_bytes().map_or(0, |b| b.saturating_sub(3 << 29));
    let admitted = pf_plan::max_pass_tokens(ubatch, max_pass, 0, budget, fused_attn);
    let plan = pf_plan::plan_ubatches(n, ubatch, admitted);
    eprintln!("flash-m3: prefill plan {plan:?} (ubatch {ubatch}, max pass {max_pass}, admitted {admitted} by {:.1} GiB available)", budget as f64 / (1u64 << 30) as f64);
    plan
}
