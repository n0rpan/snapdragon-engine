//! `oxe fused-ref --candidate prefill-fa2|prefill-fa2x|fused-decode` (bench only, `--features
//! f32-reference`): the O-117 quality runner, with the declared scope of
//! [`model::fused_ref::SCOPE`]. See `FUSED-REF.md` for the recipe, costs and gate. Never a server
//! setting.
use super::*;
use model::fused_ref::{self, Candidate, Control, Gate, Report, Verdict};

/// Synthetic mixed continuation (chat markup, tool-call JSON, code, logs, prose); no real data.
const FIXTURE: &str = include_str!("../fixtures/fused-ref-continuation.txt");

/// First `need` tokens of a text file, tokenising only as much text as needed (as `kv-kld`).
fn text_tokens(tk: &tokenizer::Tokenizer, path: &str, need: usize) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    let mut bytes = (need * 8).max(4096).min(text.len());
    loop {
        while bytes < text.len() && !text.is_char_boundary(bytes) {
            bytes += 1;
        }
        let t = tk.encode(&text[..bytes], false, false);
        if t.len() >= need || bytes >= text.len() {
            return t;
        }
        bytes = (bytes * 2).min(text.len());
    }
}

/// The validated fused-ref options (dies on a bad value): shared by the run and `--check-args`.
struct Opts {
    cand: Candidate,
    control: Control,
    gate: Gate,
    p: usize,
    n: usize,
    n_samples: usize,
    ubatch: usize,
}

fn opts(a: &Args) -> Opts {
    let cand = a.get("candidate").and_then(Candidate::parse).unwrap_or_else(|| {
        die("--candidate prefill-fa2 | prefill-fa2x | fused-decode required (O-117: the candidate is explicit; prefill candidates run genuinely batched prefill)")
    });
    let p = a.get("prefix-len").map(|_| a.num("prefix-len", 0)).unwrap_or_else(|| die("--prefix-len P required (32768, 65536, ..)"));
    let n = a.num("n", 1000);
    let n_samples = a.num("ref-samples", 200);
    let ubatch = a.num("ubatch", model::engine::UBATCH);
    if p < 64 || n == 0 || n_samples == 0 || ubatch == 0 {
        die("fused-ref needs --prefix-len >= 64, --n >= 1, --ref-samples >= 1, --ubatch >= 1");
    }
    let control = match a.get("control") {
        Some(c) => Control::parse(c).unwrap_or_else(|| die(&format!("--control {c}: decode | ubatch:N"))),
        None if cand.is_prefill() => Control::Decode,
        None => Control::Ubatch(model::engine::UBATCH),
    };
    if cand == Candidate::FusedDecode && control == Control::Decode {
        die("fused-decode: the incumbent is the tiled decode; the control must be batched (--control ubatch:N)");
    }
    let num = |k: &str, d: f64| {
        let v = a.get(k).map(|v| v.parse::<f64>().unwrap_or_else(|_| die(&format!("--{k}: not a number")))).unwrap_or(d);
        if !v.is_finite() || v < 0.0 {
            die(&format!("--{k}: a finite number >= 0"));
        }
        v
    };
    let def = Gate::default();
    let gate = Gate {
        kld_factor: num("kld-factor", def.kld_factor),
        top1_slack: num("top1-slack-pp", def.top1_slack * 100.0) / 100.0,
        ref_ratio_max: num("ref-ratio-max", def.ref_ratio_max),
        jump_threshold: num("jump-threshold", def.jump_threshold),
        ..def
    };
    Opts { cand, control, gate, p, n, n_samples, ubatch }
}

/// `oxe fused-ref ... --check-args`: the resolved options (no tokenizer, no model), then the
/// generic unknown-option check (exit 0, or 1 on any bad value / unknown option).
pub(super) fn check_args(a: &Args) {
    let o = opts(a);
    for k in ["model", "state-file", "out"] {
        if a.get(k).is_none() {
            die(&format!("--{k} required"));
        }
    }
    if a.get("prefix-ids-file").is_some() == a.get("text-file").is_some() {
        die("exactly one of --prefix-ids-file F or --text-file F");
    }
    println!(
        "{}",
        json!({"fused_ref_resolved": {"candidate": o.cand.name(), "control": o.control.name(), "prefix_len": o.p, "n": o.n, "ref_samples": o.n_samples,
            "ubatch": o.ubatch, "kld_factor": o.gate.kld_factor, "top1_slack": o.gate.top1_slack, "ref_ratio_max": o.gate.ref_ratio_max,
            "jump_threshold": o.gate.jump_threshold, "min_positions": o.gate.min_positions, "min_ref_samples": o.gate.min_ref_samples}})
    );
    a.check_known();
}

pub(super) fn run(a: &Args) {
    let Opts { cand, control, gate, p, n, n_samples, ubatch } = opts(a);
    let tk = tokenizer::Tokenizer::from_gguf(a.model()).unwrap_or_else(|e| die(&format!("tokenizer: {e:?}")));
    // prefix: ids file (the same ids as a prefill-bench --prefix-ids-file state) or a text file
    let src = match (a.get("prefix-ids-file"), a.get("text-file")) {
        (Some(f), None) => read_ids_file(f),
        (None, Some(f)) => text_tokens(&tk, f, p),
        _ => die("exactly one of --prefix-ids-file F or --text-file F"),
    };
    if src.len() < p {
        die(&format!("prefix source has {} tokens, --prefix-len needs {p}", src.len()));
    }
    let prefix = &src[..p];
    let (cont_src, cont_name) = match a.get("continuation-file") {
        Some(f) => (std::fs::read_to_string(f).unwrap_or_else(|e| die(&format!("{f}: {e}"))), f.to_string()),
        None => (FIXTURE.to_string(), "builtin:fused-ref-continuation.txt".to_string()),
    };
    let cont_all = tk.encode(&cont_src, false, true);
    if cont_all.len() < n {
        die(&format!("continuation has {} tokens, --n needs {n}", cont_all.len()));
    }
    let cont = &cont_all[..n];
    let n_ctx = a.num("ctx", (p + n + 16).next_multiple_of(1024));
    if p + n >= n_ctx {
        die("--ctx must exceed prefix + n");
    }
    let samples = fused_ref::sample_positions(n, n_samples);
    let scope_label = format!(
        "O-117 R2 clauses (a)-(c): candidate {} vs tiled incumbent, prefix {p} (one saved state), {n} teacher-forced positions all compared, control {} (measured, same state and positions), f32-attention reference at {} sampled positions (NOT a full-model f32 reference); clause (d) tool-JSON runs separately (closed-loop runner)",
        cand.name(),
        control.name(),
        samples.len()
    );
    let cont_hash = format!("{:016x}", model::weights::hash_bytes(cont_src.as_bytes()));
    let arms = if cand.is_prefill() {
        json!(["incumbent: tiled batched prefill (ubatch)", "candidate: batched prefill with the candidate kernel (ubatch)", "reference: batched prefill, f32-attention reference for every row", "control: tiled decode or tiled batched prefill at another ubatch"])
    } else {
        json!(["control: tiled batched prefill (ubatch)", "incumbent: tiled decode", "candidate: fused decode", "reference: forked decode steps of the incumbent"])
    };
    let plan = json!({"candidate": cand.name(), "prefix_tokens": p, "positions": n, "ref_samples": samples.len(), "n_ctx": n_ctx,
        "ubatch": ubatch, "control": control.name(), "prefix_ids": sha_ids(prefix), "continuation": cont_name, "continuation_text_hash": cont_hash,
        "continuation_tokens_available": cont_all.len(), "continuation_ids": sha_ids(cont), "state_file": a.get("state-file"),
        "engines": if cand.is_prefill() { 1 } else { 2 }, "arms": arms, "scope": fused_ref::SCOPE, "scope_label": scope_label});
    if a.flag("dry-run") {
        // tokenizer only: no weights, no engine
        println!("{}", serde_json::to_string_pretty(&plan).unwrap());
        return;
    }
    let state_file = a.get("state-file").unwrap_or_else(|| die("--state-file F required (built once, reused)"));
    let out = a.get("out").unwrap_or_else(|| die("--out NEW_FILE required"));
    if Path::new(out).exists() {
        die("--out exists; pick a new file");
    }
    let s = setup(a, n_ctx);
    if prefix.iter().chain(cont).any(|&t| t as usize >= s.loaded.plan.desc.n_vocab) {
        die("token id outside vocabulary");
    }
    let cfg = EngineConfig { n_ctx, logits: true, trace: false, profile: false };
    let mk = || Engine::with_options(&s.loaded.plan, &s.loaded.ws, cfg.clone(), EngineOptions { verify_lanes: 1, mtp: None, kv_type: Some(model::engine::KvType::F16) }).unwrap_or_else(|e| die(&e));
    let mut ea = mk();
    // the saved state is built by the tiled incumbent (the same kind as prefill-bench
    // --prefix-len/--state-file: one state serves both tools)
    ea.set_pf_attn(model::engine::PfAttn::Tiled);
    let header = state_header(&ea, "prefill-bench-prefix", prefix);
    let t0 = Instant::now();
    let built = if state_file_load(state_file, &ea, &header, p).is_none() {
        for c in prefix.chunks(STATE_CHUNK) {
            ea.prefill(&s.pool, c, model::engine::UBATCH).unwrap();
            eprintln!("fused-ref: prefix {}/{p} after {:.1} s", ea.pos(), t0.elapsed().as_secs_f64());
        }
        state_file_save(state_file, &ea, &header, 0, &[]);
        true
    } else {
        false
    };
    let state_s = t0.elapsed().as_secs_f64();
    let mut load = |e: &Engine| {
        if state_file_load(state_file, e, &header, p).is_none() || e.pos() != p {
            die("state file could not be read back (the same saved state before every arm)");
        }
    };
    let t1 = Instant::now();
    let mut progress = |m: &str| eprintln!("fused-ref {}: {m} ({:.1} s)", cand.name(), t1.elapsed().as_secs_f64());
    let (rep, fused_chunk) = if cand.is_prefill() {
        let r = fused_ref::run_prefill(&mut ea, &s.pool, &mut load, cand, cont, ubatch, control, &samples, &mut progress);
        (r, None)
    } else {
        let mut eb = mk();
        if let Some(ch) = a.get("fused-chunk") {
            eb.set_attn_fused_chunk(ch.parse().unwrap_or_else(|_| die("--fused-chunk")));
            ea.set_attn_fused_chunk(eb.attn_fused_chunk());
        }
        let r = fused_ref::run_fused_decode(&mut ea, &mut eb, &s.pool, &mut load, cont, control, &samples, &mut progress);
        (r, Some(eb.attn_fused_chunk()))
    };
    let run_s = t1.elapsed().as_secs_f64();
    let v = rep.verdict(&gate);
    if !v.finite_ok {
        eprintln!("fused-ref: FAIL: non-finite values ({} compared rows, {} reference rows): see \"finite\" in the receipt", v.nonfinite_rows, v.nonfinite_refs);
    }
    let res = receipt(&rep, &v, &gate, &scope_label);
    let mut res = res;
    let o = res.as_object_mut().unwrap();
    o.insert("plan".into(), plan);
    o.insert("model".into(), json!(a.model().display().to_string()));
    o.insert("model_sha256".into(), json!(s.loaded.ws.info.model_sha256));
    o.insert("threads".into(), json!(s.pool.nthreads()));
    o.insert("fused_chunk".into(), json!(fused_chunk));
    o.insert("effective".into(), json!(model::engine::effective_switches()));
    o.insert("engine_commit".into(), json!(option_env!("OXE_GIT_COMMIT").unwrap_or("unknown")));
    o.insert("state".into(), json!({"built": built, "seconds": state_s, "built_with": "tiled batched prefill, ubatch 256, chunks of 4096"}));
    o.insert("run_seconds".into(), json!(run_s));
    let text = serde_json::to_string_pretty(&res).unwrap();
    std::fs::write(out, &text).unwrap_or_else(|e| die(&format!("{out}: {e}")));
    let mut short = res.clone();
    for k in ["per_position", "refs", "plan"] {
        short.as_object_mut().unwrap().remove(k);
    }
    println!("{}", serde_json::to_string_pretty(&short).unwrap());
    if !v.passed {
        std::process::exit(1);
    }
}

/// The O-117 receipt: PASS / FAIL per clause, row counts, scope label, the evidence of every arm,
/// per-position rows and the reference rows.
fn receipt(r: &Report, v: &Verdict, g: &Gate, scope_label: &str) -> serde_json::Value {
    fn pf(s: &'static str) -> &'static str {
        if s.is_empty() {
            "n/a"
        } else {
            s
        }
    }
    let verdict = |b: bool| if b { "PASS" } else { "FAIL" };
    let st = |s: &fused_ref::Stats| json!({"rows": s.rows, "mean": s.mean, "p50": s.p50, "p90": s.p90, "p99": s.p99, "max": s.max, "top1_agreement": s.top1});
    let arms: Vec<_> = r.arms.iter().map(|x| json!({"role": x.role, "path": x.path, "decode_steps": x.decode_steps, "fused_decode": x.fused, "pf_kernel": pf(x.pf_kernel),
        "batched_prefill": {"passes": x.pf.passes, "rows": x.pf.rows, "max_rows_per_pass": x.pf.max_rows,
            "attention_units": {"tiled": x.pf.units_tiled, "per_head": x.pf.units_per_head, "fa2": x.pf.units_fa2, "fa2x": x.pf.units_fa2x, "f32_reference": x.pf.units_ref}}})).collect();
    let per_position: Vec<_> = r.rows.iter().zip(&r.control_rows).map(|(c, k)| json!([c.position, c.kld, c.top1, c.inc_top1, k.kld, k.top1, c.finite && k.finite])).collect();
    let refs: Vec<_> = r.refs.iter().map(|x| json!({"index": x.index, "position": x.position, "sampled": x.sampled, "disagreement": x.disagreement,
        "ref_top1": x.ref_top1, "ref_margin": x.ref_margin, "ref_absmax": x.ref_absmax, "tiled_top1": x.inc_top1, "cand_top1": x.cand_top1,
        "tiled_rms": x.inc_rms, "cand_rms": x.cand_rms, "tiled_kld_vs_ref": x.inc_kld_vs_ref, "cand_kld_vs_ref": x.cand_kld_vs_ref,
        "fork_replay_exact": x.fork_replay_exact, "finite": x.all_finite(), "tiled_attn_err2_ref2": [x.inc_attn.0, x.inc_attn.1], "cand_attn_err2_ref2": [x.cand_attn.0, x.cand_attn.1]})).collect();
    json!({
        "kind": "o117-quality/v1", "scope": fused_ref::SCOPE, "scope_label": scope_label, "candidate": r.candidate.name(),
        "passed": v.passed, "verdict": verdict(v.passed),
        "clauses": {
            "a": {"verdict": verdict(v.pass_a), "rule": "mean KLD(cand || tiled) <= factor x mean KLD(control || tiled), measured on the same state and positions",
                  "kld_direction": "KL(P_cand || P_tiled)", "cand_mean": v.cand.mean, "control_mean": v.control.mean, "factor": g.kld_factor, "max_allowed": v.kld_max},
            "b": {"verdict": verdict(v.pass_b), "rule": "top-1 agreement(cand, tiled) >= control agreement - slack",
                  "cand_top1": v.cand.top1, "control_top1": v.control.top1, "slack_pp": g.top1_slack * 100.0, "min_allowed": v.top1_min},
            "c": {"verdict": verdict(v.pass_c), "rule": "f32-attention reference: median and p90 logit RMS error <= ratio x tiled; jumps <= tiled + max(jump_tol_min, ceil(jump_tol_frac x samples)); attention-output rel-L2 <= ratio x tiled",
                  "samples": v.ref_samples, "ratio": g.ref_ratio_max, "jump_threshold": g.jump_threshold, "jump_allowance": v.jump_allowance,
                  "jump_allowance_rule": format!("max({}, ceil({} x {}))", g.jump_tol_min, g.jump_tol_frac, v.ref_samples),
                  "logit_rms": {"tiled": {"p50": v.inc_rms_p50, "p90": v.inc_rms_p90, "mean": v.inc_rms_mean, "jumps": v.inc_jumps},
                                "cand": {"p50": v.cand_rms_p50, "p90": v.cand_rms_p90, "mean": v.cand_rms_mean, "jumps": v.cand_jumps}},
                  "attention_rel_l2": {"tiled": v.inc_attn_rel_l2, "cand": v.cand_attn_rel_l2},
                  "p50": verdict(v.pass_c_p50), "p90": verdict(v.pass_c_p90), "jumps": verdict(v.pass_c_jumps), "attention": verdict(v.pass_c_attn)},
            "d": {"verdict": "NOT RUN HERE", "rule": "tool-JSON 100% on one long-session closed-loop replay at >= 48K (closed-loop runner)"}},
        "rows": {"compared": v.cand.rows, "control": v.control.rows, "same_positions": v.same_positions, "ref_samples": v.ref_samples,
                 "ref_forced_forks": v.ref_forced, "min_positions": g.min_positions, "min_ref_samples": g.min_ref_samples},
        "scope_met": v.scope_met, "fork_replay_exact": v.fork_replay_exact,
        "finite": {"verdict": verdict(v.finite_ok), "nonfinite_rows": v.nonfinite_rows, "nonfinite_ref_rows": v.nonfinite_refs,
                   "rule": "every compared logit row (candidate, incumbent, control) and every reference row / metric finite; any NaN or inf fails"},
        "path": {"ok": v.path_ok, "problems": v.path_problems, "ubatch": r.ubatch, "arms": arms},
        "candidate_kld": st(&v.cand),
        "control": {"kind": r.control.name(), "kld_direction": "KL(P_control || P_tiled)", "kld": st(&v.control)},
        "per_position": {"columns": ["position", "kld_cand_to_tiled", "cand_top1", "tiled_top1", "kld_control_to_tiled", "control_top1", "finite"], "rows": per_position},
        "refs": refs})
}
