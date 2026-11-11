//! flash-t2 (WP-T2): MTP speculative decoding in the Flash engine.
//!
//!   flash-t2 selftest                       no model: mm_cols vs one-column mul_mat, GDN snapshot commit (bitwise)
//!   flash-t2 vocab --roots A,B --out F      no model weights (tokenizer from the GGUF header): top-N token ids
//!       [--exts rs,py,ts,md,...] [--top 32768] [--max-files 20000] [--holdout 5]
//!       --holdout K: a file is held out iff fnv1a(path) % K == 0; ids come from the rest (train) only,
//!       and the JSON reports coverage_train / coverage_holdout (0 = no split, old behaviour)
//!   flash-t2 vocab --roots A,B --coverage-of F  score an existing id list F on all walked files; writes nothing
//!   flash-t2 selfcheck --oracle DIR         MODEL: the bug detector. Teacher-forced on the oracle tokens:
//!       (1) plain fast decode, one hash per logit row; (2) the same positions again through verify / commit
//!       in widths 1..=--max-width with partial commits (rollbacks); every verify row must be BITWISE the plain
//!       row; (3) a `decode_h` prefill must give the plain prefill's last-row logits bitwise.
//!   flash-t2 run --oracle DIR               MODEL: free-running greedy with MTP (or --plain: one-token decodes).
//!       --depth N (max drafts, default 2; 0 = verify of one row), --floor P (0.3), --vocab F (restricted draft
//!       ids from `flash-t2 vocab`), --head F.gguf, --head-willneed, --max-new 400, --out F.json.
//! Common model options as flash-m3 run's fast path: --threads 8 --threads-batch N --ubatch 4096 --n-ctx 4096
//! --cache-gib 18 --qd 32 --io-cpu N --attn online|twopass --executor 0|1 --sidecar F --model-dir D
//! --fuse N (S-c phases as flash-m3; the GDN phases are not used in verify graphs) --verify-cache 0|1 (keep the
//! verify graph per width; default 0) --cache-policy lfu|clock --quota-weights builtin|equal (defaults as flash-m3).
//! MODEL commands load model bytes: Codex runs them (inference lease, 28G scope), never a Claude agent.

use qwen4exp::cli::{cgroup_file, die, read_tokens, status_kib, Args, EOG};
use qwen4exp::cache::Policy;
use qwen4exp::dense::{Dense, DenseMode};
use qwen4exp::graph::{argmax, FgOptions, GraphCtx, GraphModel};
use qwen4exp::model::{Shards, N_VOCAB};
use qwen4exp::mtp::{MtpConfig, MtpDriver};
use qwen4exp::mtp_engine::MtpEngine;
use qwen4exp::mtp_graph::H_DIM;
use qwen4exp::mtp_head::HeadFile;
use qwen4exp::stream::{Stream, StreamConfig};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let a = Args::parse(&argv[1.min(argv.len())..]);
    // an unknown or mistyped flag is an error, not a silent default (TEST-ECONOMY R1b); --check-args: parse only
    const KNOWN: [&str; 39] = ["always", "answer", "attn", "cache-gib", "cache-policy", "check-args", "check-prefill-h", "coverage-of", "depth", "draft-cost",
        "executor", "exts", "floor", "fuse", "head", "head-willneed", "holdout", "io-cpu", "max-file-bytes", "max-files", "max-new", "max-width",
        "model-dir", "n-ctx", "oracle", "out", "plain", "qd", "quota-weights", "roots", "row-cost", "sidecar", "steps", "threads",
        "threads-batch", "top", "ubatch", "verify-cache", "vocab"];
    if let Some(k) = a.0.keys().find(|k| !KNOWN.contains(&k.as_str())) {
        die(&format!("flash-t2: unknown option --{k}"));
    }
    if a.has("check-args") {
        if !["selftest", "vocab", "selfcheck", "run"].contains(&cmd.as_str()) {
            die("flash-t2: unknown command");
        }
        println!("flash-t2 {cmd}: arguments OK");
        return;
    }
    match cmd.as_str() {
        "selftest" => {
            let mut ok = true;
            for nth in [1, 3, 8] {
                ok &= qwen4exp::mtp_graph::selftest_mtp(nth);
            }
            println!("selftest mtp (mm_cols, GDN commit): {}", if ok { "PASS" } else { "FAIL" });
            std::process::exit(if ok { 0 } else { 1 })
        }
        "vocab" => vocab(&a),
        "selfcheck" => selfcheck(&a),
        "run" => run(&a),
        _ => die("usage: flash-t2 selftest | vocab | selfcheck | run (see the source header)"),
    }
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn model_dir(a: &Args) -> PathBuf {
    PathBuf::from(a.s("model-dir", &home().join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS").to_string_lossy()))
}

fn vocab(a: &Args) {
    use qwen4exp::mtp_vocab as v;
    let dir = model_dir(a);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap_or_else(|e| die(&format!("{}: {e}", dir.display()))).filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false)).collect();
    paths.sort();
    let first = paths.first().unwrap_or_else(|| die("no .gguf in the model dir"));
    let tk = tokenizer::Tokenizer::from_gguf(first).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
    let roots: Vec<PathBuf> = a.s("roots", "").split(',').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
    if roots.is_empty() {
        die("--roots A,B");
    }
    let exts_s = a.s("exts", "rs,py,ts,tsx,js,jsx,go,c,h,cpp,hpp,java,kt,swift,rb,sh,md,txt,json,toml,yaml,yml,html,css,sql");
    let exts: Vec<&str> = exts_s.split(',').collect();
    let files = v::walk(&roots, &exts, a.n("max-files", 20000usize));
    let max_bytes: u64 = a.n("max-file-bytes", 1u64 << 20);
    if let Some(cf) = a.0.get("coverage-of") {
        // score an existing id list on every walked file; writes nothing
        let ids = v::read_ids(Path::new(cf)).unwrap_or_else(|e| die(&e));
        let (counts, total, used) = v::count_files(&tk, &files, N_VOCAB, max_bytes);
        println!("{}", json!({"kind": "flash-t2-vocab-coverage", "ids": ids.len(), "files": used, "tokens": total,
                              "coverage": v::coverage(&counts, &ids), "ids_file": cf}));
        return;
    }
    // held-out split: a file is held out iff fnv1a(path) % K == 0 (K = 0: no split); ids come from train only
    let holdout_k: usize = a.n("holdout", 5usize);
    let (train_files, hold_files) = v::split_holdout(&files, holdout_k);
    let (counts, total, used) = v::count_files(&tk, &train_files, N_VOCAB, max_bytes);
    let (hcounts, htotal, hused) = v::count_files(&tk, &hold_files, N_VOCAB, max_bytes);
    let top: usize = a.n("top", 32768);
    let mut always: Vec<i32> = EOG.to_vec();
    always.extend(a.s("always", "").split(',').filter_map(|s| s.parse::<i32>().ok()));
    let ids = v::top_ids(&counts, top, &always);
    let out = a.s("out", "mtp-vocab.bin");
    v::write_ids(Path::new(&out), &ids).unwrap_or_else(|e| die(&e));
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    println!("{}", json!({"kind": "flash-t2-vocab", "files": used, "tokens": total, "distinct_ids": distinct, "top": ids.len(),
                          "coverage": v::coverage(&counts, &ids), "out": out,
                          "holdout_k": holdout_k, "train_files": used, "train_tokens": total,
                          "holdout_files": hused, "holdout_tokens": htotal,
                          "coverage_train": v::coverage(&counts, &ids), "coverage_holdout": v::coverage(&hcounts, &ids),
                          "holdout_distinct_ids_missing": v::distinct_missing(&hcounts, &ids)}));
}

struct Loaded {
    s: Shards,
    _d: Dense,
    gm: GraphModel,
    st: Box<Stream>,
    opts: FgOptions,
    threads: usize,
    threads_batch: usize,
    ubatch: usize,
    n_ctx: u32,
    cache_gib: f64,
    _head: Option<HeadFile>,
}

/// The fast path of flash-m3 run (fixed-chunk attention, streamed prefill, shared expert first, rs-view).
fn load(a: &Args, want_head: bool) -> Loaded {
    let threads: usize = a.n("threads", 8);
    let threads_batch: usize = a.n("threads-batch", threads);
    let attn = match a.s("attn", "online").as_str() {
        "online" => 1,
        "twopass" => 3,
        x => die(&format!("--attn {x}: online|twopass")),
    };
    let opts = FgOptions { fixed_attn: attn, shexp_first: 1, batch_min_tokens: 17, profile: 0, rs_view: 1, executor: a.flag("executor", false) as i32, fuse: a.n("fuse", 0i32) };
    qwen4exp::mtp_graph::set_verify_cache(a.flag("verify-cache", false));
    let sidecar = PathBuf::from(a.s("sidecar", &home().join(".cache/local-ai/snapdragon-engine/flash/experts-r4.bin").to_string_lossy()));
    let s = Shards::open(&model_dir(a)).unwrap_or_else(|e| die(&e));
    let d = Dense::load(&s, DenseMode::VecDot).unwrap_or_else(|e| die(&e));
    let mut gm = GraphModel::new(&s, &d, false).unwrap_or_else(|e| die(&e));
    let head = if want_head {
        let hp = a.0.get("head").map(PathBuf::from).unwrap_or_else(qwen4exp::mtp_head::default_path);
        let h = HeadFile::open(&hp).unwrap_or_else(|e| die(&e));
        h.attach(&mut gm).unwrap_or_else(|e| die(&e));
        h.advise_experts(a.has("head-willneed"));
        if let Some(vp) = a.0.get("vocab") {
            let ids = qwen4exp::mtp_vocab::read_ids(Path::new(vp)).unwrap_or_else(|e| die(&e));
            gm.mtp_set_vocab(&ids).unwrap_or_else(|e| die(&e));
            eprintln!("flash-t2: draft vocabulary {} ids", ids.len());
        }
        Some(h)
    } else {
        None
    };
    let cache_gib: f64 = a.n("cache-gib", 18.0);
    let cfg = StreamConfig { cache_bytes: (cache_gib * (1u64 << 30) as f64) as usize, qd: a.n("qd", 32u32), io_cpu: a.0.get("io-cpu").and_then(|v| v.parse().ok()), prefault: true };
    // cache defaults as flash-m3 run on the integrated tree (ticket 035): LFU + the built-in per-layer quotas
    let qw = a.s("quota-weights", "builtin");
    let w: Option<&[f64]> = match qw.as_str() {
        "builtin" => Some(&qwen4exp::stream::FLASH_QUOTA_WEIGHTS),
        "equal" => None,
        x => die(&format!("--quota-weights {x}: builtin|equal")),
    };
    let mut st = Box::new(Stream::new_with(&s, &sidecar, &cfg, w).unwrap_or_else(|e| die(&e)));
    st.cache.policy = match a.s("cache-policy", "lfu").as_str() {
        "clock" => Policy::Clock,
        "lfu" => Policy::Lfu,
        x => die(&format!("--cache-policy {x}: clock|lfu")),
    };
    st.resident_first = true;
    st.batch_gemm = true;
    Loaded { s, _d: d, gm, st, opts, threads, threads_batch, ubatch: a.n("ubatch", 4096), n_ctx: a.n("n-ctx", 4096), cache_gib, _head: head }
}

fn oracle(a: &Args) -> (Vec<i32>, Vec<i32>) {
    let recs = read_tokens(&PathBuf::from(a.s("oracle", "")).join("tokens.bin"));
    if recs.len() < 3 || recs[1].len() < 2 {
        die("oracle tokens.bin: expected warm-up, prompt, decode records");
    }
    (recs[1].clone(), recs[2..].iter().map(|r| r[0]).collect())
}

/// FNV-1a over the row's bytes: equal hashes <=> bitwise equal rows (for this check's purposes).
fn row_hash(l: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for x in l {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn new_ctx(l: &Loaded) -> GraphCtx {
    let mut g = GraphCtx::new(&l.gm, l.n_ctx);
    g.set_options(&l.opts);
    let ud = &*l.st as *const Stream as *mut std::ffi::c_void;
    // SAFETY: the stream outlives the context (both live in `Loaded`'s scope; the context is dropped first).
    unsafe { g.set_expert_fn(Some(Stream::callback()), ud) };
    g
}

fn selfcheck(a: &Args) {
    let l = load(a, false);
    let (prompt, want) = oracle(a);
    let steps: usize = a.n("steps", want.len()).min(want.len());
    let max_w: usize = a.n("max-width", 4).clamp(1, 16);
    let mut g = new_ctx(&l);
    let mut logits = vec![0f32; N_VOCAB];
    // (1) plain: prefill + teacher-forced one-token decodes; ref[i] = hash of the row after position P + i - 1
    for ch in prompt.chunks(l.ubatch) {
        g.decode(ch, l.threads_batch, &mut logits, None).unwrap_or_else(|e| die(&e));
    }
    let prefill_hash = row_hash(&logits);
    let (ptoks, pstate) = g.save_compact();
    let mut refh: Vec<u64> = Vec::with_capacity(steps);
    let t0 = Instant::now();
    for &t in &want[..steps] {
        g.decode(&[t], l.threads, &mut logits, None).unwrap_or_else(|e| die(&e));
        refh.push(row_hash(&logits));
    }
    let plain_ms = t0.elapsed().as_secs_f64() * 1e3 / steps.max(1) as f64;
    // (2) the same positions through verify / commit
    g.load_compact(&ptoks, &pstate).unwrap_or_else(|e| die(&e));
    let mut la = vec![0f32; max_w * N_VOCAB];
    let mut ha = vec![0f32; max_w * H_DIM];
    let (mut i, mut cyc, mut rows, mut bad, mut rollbacks) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut first_bad: Option<(usize, usize, usize)> = None; // (step, width, row)
    let mut by_width = vec![(0usize, 0usize); max_w + 1]; // (rows, mismatches)
    let t1 = Instant::now();
    while i < steps {
        let w = (1 + cyc % max_w).min(steps - i);
        g.verify(&want[i..i + w], l.threads, &mut la, &mut ha).unwrap_or_else(|e| die(&e));
        for r in 0..w {
            rows += 1;
            by_width[w].0 += 1;
            if row_hash(&la[r * N_VOCAB..(r + 1) * N_VOCAB]) != refh[i + r] {
                bad += 1;
                by_width[w].1 += 1;
                first_bad.get_or_insert((i + r, w, r));
            }
        }
        // partial commits exercise the rollback: keep 1..=w rows in a fixed pattern
        let m = 1 + (cyc * 7 + 3) % w;
        rollbacks += (m < w) as usize;
        g.commit(m).unwrap_or_else(|e| die(&e));
        i += m;
        cyc += 1;
    }
    let verify_ms = t1.elapsed().as_secs_f64() * 1e3 / cyc.max(1) as f64;
    let stats = g.mtp_stats();
    // (3) decode_h prefill == plain prefill (last row)
    let mut ph_same = None;
    if a.flag("check-prefill-h", true) {
        g.clear();
        let mut h = vec![0f32; prompt.len().min(l.ubatch) * H_DIM];
        for ch in prompt.chunks(l.ubatch) {
            g.decode_h(ch, l.threads_batch, &mut logits, &mut h).unwrap_or_else(|e| die(&e));
        }
        ph_same = Some(row_hash(&logits) == prefill_hash);
    }
    let pass = bad == 0 && stats[2] == 0.0 && ph_same != Some(false);
    let out = json!({
        "kind": "flash-t2-selfcheck", "engine_commit": qwen4exp::ENGINE_COMMIT, "pass": pass,
        "steps": steps, "max_width": max_w, "verify_cycles": cyc, "verify_rows": rows, "rollbacks": rollbacks,
        "rows_not_bitwise": bad, "first_mismatch_step_width_row": first_bad.map(|(s, w, r)| vec![s, w, r]),
        "by_width": by_width.iter().enumerate().skip(1).map(|(w, x)| json!({"width": w, "rows": x.0, "mismatches": x.1})).collect::<Vec<_>>(),
        "mm_fallbacks": stats[2], "prefill_h_last_row_bitwise": ph_same,
        "plain_ms_per_token": plain_ms, "verify_ms_per_cycle_mixed_widths": verify_ms,
        "options": format!("{:?}", l.opts), "threads": l.threads, "cache_gib": l.cache_gib,
        "memory": {"vm_hwm_kib": status_kib("VmHWM:"), "cgroup_memory_peak": cgroup_file("memory.peak")},
    });
    let txt = serde_json::to_string_pretty(&out).unwrap();
    println!("{txt}");
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, &txt).ok();
    }
    drop(g);
    std::process::exit(if pass { 0 } else { 1 })
}

fn run(a: &Args) {
    let plain = a.has("plain");
    let l = load(a, !plain);
    let (prompt, want) = oracle(a);
    let max_new: usize = a.n("max-new", 400);
    let depth: usize = a.n("depth", 2);
    let mut g = new_ctx(&l);
    if !plain {
        g.mtp_init().unwrap_or_else(|e| die(&e));
    }
    let mut logits = vec![0f32; N_VOCAB];
    let mut eng = MtpEngine::new(&mut g, l.n_ctx as usize, l.threads, !plain);
    let tp = Instant::now();
    eng.prefill(&prompt, l.ubatch, l.threads_batch, &mut logits).unwrap_or_else(|e| die(&e));
    let prefill_s = tp.elapsed().as_secs_f64();
    let mut got: Vec<i32> = vec![argmax(&logits) as i32];
    let mut cfg = MtpConfig { max_depth: depth, floor: a.n("floor", 0.3f32), ..MtpConfig::default() };
    cfg.verify_cost = (0..=depth).map(|i| 1.0 + a.n("row-cost", 0.25f32) * i as f32).collect();
    cfg.draft_cost = a.n("draft-cost", 0.06f32);
    let mut drv = MtpDriver::new(cfg);
    eng.g.timing(true);
    let c0 = l.st.stats();
    let w0 = l.st.counters.wait0_ns.load(Relaxed);
    let t0 = Instant::now();
    let mut cycle_ms: Vec<f64> = Vec::new();
    while got.len() < max_new && !EOG.contains(got.last().unwrap()) {
        let t = *got.last().unwrap();
        let tc = Instant::now();
        if plain {
            eng.g.decode(&[t], l.threads, &mut logits, None).unwrap_or_else(|e| die(&e));
            got.push(argmax(&logits) as i32);
        } else {
            let new = drv.cycle(&mut eng, t).unwrap_or_else(|e| die(&e));
            for x in new {
                got.push(x);
                if EOG.contains(&x) {
                    break; // tokens drafted past the end of generation are dropped
                }
            }
        }
        cycle_ms.push(tc.elapsed().as_secs_f64() * 1e3);
    }
    let decode_s = t0.elapsed().as_secs_f64();
    let n_new = (got.len() - 1).max(1) as f64;
    let ans: Vec<i32> = got.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    let want_ans: Vec<i32> = want.iter().copied().take_while(|t| !EOG.contains(t)).collect();
    if let Some(p) = a.0.get("answer") {
        let tk = tokenizer::Tokenizer::from_gguf(&l.s.paths[0]).unwrap_or_else(|e| die(&format!("tokenizer: {e}")));
        let toks: Vec<u32> = ans.iter().map(|&t| t as u32).collect();
        std::fs::write(p, tk.decode_bytes(&toks, false)).unwrap_or_else(|e| die(&e.to_string()));
    }
    let cs = l.st.stats();
    let st = &drv.stats;
    let tm = eng.times.clone();
    let ms = eng.g.mtp_stats();
    let (gt, gcalls) = eng.g.timing(false);
    let out = json!({
        "kind": "flash-t2-run", "mode": if plain { "plain" } else { "mtp" }, "engine_commit": qwen4exp::ENGINE_COMMIT,
        "depth": depth, "floor": drv.cfg.floor, "vocab": a.0.get("vocab"),
        "threads": l.threads, "cache_gib": l.cache_gib, "options": format!("{:?}", l.opts),
        "prompt_tokens": prompt.len(), "prefill_s": prefill_s,
        "generated": ans.len(), "tokens": got, "matches_oracle_tokens": ans == want_ans,
        "ms_per_token": decode_s * 1e3 / n_new, "decode_s": decode_s, "cycles": cycle_ms.len(),
        "ms_per_cycle_mean": cycle_ms.iter().sum::<f64>() / cycle_ms.len().max(1) as f64,
        "mtp": {"cycles": st.cycles, "drafted": st.drafted, "accepted": st.accepted, "committed": st.committed,
                "tokens_per_verify": st.tokens_per_verify(), "pos_proposed": st.pos_proposed, "pos_accepted": st.pos_accepted,
                "floor_stops": st.floor_stops, "mm_fallbacks": ms[2], "head_passes": ms[3]},
        "ms_per_token_parts": {"verify": tm.verify_s * 1e3 / n_new, "commit": tm.commit_s * 1e3 / n_new, "draft": tm.draft_s * 1e3 / n_new,
                               "catch_up": tm.catch_up_s * 1e3 / n_new, "draft_ms_per_step": tm.draft_s * 1e3 / tm.draft_steps.max(1) as f64},
        "verify_cache": a.flag("verify-cache", false), "cache_policy": format!("{:?}", l.st.cache.policy), "quota_weights": a.s("quota-weights", "builtin"),
        "trunk_graph_ms_per_token": {"build": gt[0] / n_new, "alloc": gt[1] / n_new, "inputs": gt[2] / n_new, "compute": gt[3] / n_new, "calls": gcalls},
        "expert_wait_ms_per_token_thread0": (l.st.counters.wait0_ns.load(Relaxed) - w0) as f64 / 1e6 / n_new,
        "read_mib_per_token": (cs.read_bytes - c0.read_bytes) as f64 / 1048576.0 / n_new,
        "memory": {"vm_hwm_kib": status_kib("VmHWM:"), "cgroup_memory_peak": cgroup_file("memory.peak")},
    });
    let txt = serde_json::to_string_pretty(&out).unwrap();
    println!("{txt}");
    if let Some(p) = a.0.get("out") {
        std::fs::write(p, &txt).ok();
    }
    drop(eng);
    drop(g);
}
