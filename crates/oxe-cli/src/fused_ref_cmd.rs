//! `oxe fused-ref` (bench only, `--features f32-reference`): long-context numeric reference for
//! the O-70 fused attention, with the declared scope of [`model::fused_ref::SCOPE`]. See
//! `FUSED-REF.md` for the recipe, costs and gate. Never a server setting.
use super::*;
use model::fused_ref::{self, Gate};

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

pub(super) fn run(a: &Args) {
    let p = a.get("prefix-len").map(|_| a.num("prefix-len", 0)).unwrap_or_else(|| die("--prefix-len P required (32768, 65536, ..)"));
    let n = a.num("n", 1000);
    let n_samples = a.num("ref-samples", 200);
    if p < 64 || n == 0 || n_samples == 0 {
        die("fused-ref needs --prefix-len >= 64, --n >= 1, --ref-samples >= 1");
    }
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
    let mut gate = Gate {
        noise_floor_kld: a.get("noise-floor-kld").map(|v| v.parse().unwrap_or_else(|_| die("--noise-floor-kld"))).unwrap_or(f64::NAN),
        ref_ratio_max: a.get("ref-ratio-max").map(|v| v.parse().unwrap_or_else(|_| die("--ref-ratio-max"))).unwrap_or(1.1),
        jump_threshold: a.get("jump-threshold").map(|v| v.parse().unwrap_or_else(|_| die("--jump-threshold"))).unwrap_or(1e-3),
        margin: a.get("margin").map(|v| v.parse().unwrap_or_else(|_| die("--margin"))).unwrap_or(5.0),
        ..Gate::default()
    };
    let cont_hash = format!("{:016x}", model::weights::hash_bytes(cont_src.as_bytes()));
    let plan = json!({"prefix_tokens": p, "positions": n, "ref_samples": samples.len(), "n_ctx": n_ctx,
        "prefix_ids": sha_ids(prefix), "continuation": cont_name, "continuation_text_hash": cont_hash,
        "continuation_tokens_available": cont_all.len(), "continuation_ids": sha_ids(cont), "state_file": a.get("state-file"),
        "engines": 2, "scope": fused_ref::SCOPE});
    if a.flag("dry-run") {
        // tokenizer only: no weights, no engine
        println!("{}", serde_json::to_string_pretty(&plan).unwrap());
        return;
    }
    let control_on = a.flag("control");
    let control_w = a.num("control-ubatch", 4);
    let control_factor: f64 = a.get("control-factor").map(|v| v.parse().unwrap_or_else(|_| die("--control-factor"))).unwrap_or(1.0);
    if control_on && (control_w == 0 || control_w > n) {
        die("--control-ubatch must be in 1..=n");
    }
    if !gate.noise_floor_kld.is_finite() && !control_on {
        die("--control (measures the floor on this state) or --noise-floor-kld X (provided, not measured) required");
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
    let (mut ea, mut eb) = (mk(), mk());
    if let Some(ch) = a.get("fused-chunk") {
        eb.set_attn_fused_chunk(ch.parse().unwrap_or_else(|_| die("--fused-chunk")));
        ea.set_attn_fused_chunk(eb.attn_fused_chunk());
    }
    // the same kind as prefill-bench --prefix-len/--state-file: one state serves both tools
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
    if state_file_load(state_file, &eb, &header, p).is_none() {
        die("state file could not be read back into the second engine");
    }
    if ea.pos() != p || eb.pos() != p {
        die("engines are not at the prefix end");
    }
    // --control: the incumbent's own noise on the SAME state (decode vs batched prefill, both
    // tiled), then both engines are reloaded from the state file for the fused comparison
    let mut control_json = serde_json::Value::Null;
    let mut floor_source = "provided (not measured)".to_string();
    if control_on {
        fused_ref::configure(&mut ea, &mut eb);
        eb.set_attn_fused(false);
        let tc = Instant::now();
        let cv = fused_ref::control(&ea, &eb, &s.pool, cont, control_w, &mut |i| {
            if (i + 1) % 200 < control_w {
                eprintln!("fused-ref control: {}/{n} positions, {:.1} s", i + 1, tc.elapsed().as_secs_f64());
            }
        });
        let (cn, mean, p50, p90, p99, max, top1) = fused_ref::control_summary(&cv);
        eprintln!("fused-ref control: {cn} chunk-end positions, kld mean {mean:.3e} p99 {p99:.3e}, top1 {top1:.4}");
        if !gate.noise_floor_kld.is_finite() {
            gate.noise_floor_kld = control_factor * mean;
            floor_source = format!("measured: {control_factor} x control mean on this state");
        }
        control_json = json!({"kind": "tiled decode (1 token per step) vs tiled batched prefill (ubatch = chunk) from the same saved state, teacher-forced; chunk-end positions",
            "ubatch": control_w, "positions": cn, "kld_direction": "KL(P_decode || P_prefill_batch)",
            "kld": {"mean": mean, "p50": p50, "p90": p90, "p99": p99, "max": max}, "top1_agreement": top1,
            "factor": control_factor, "seconds": tc.elapsed().as_secs_f64(),
            "rows": cv.iter().map(|x| json!([x.position, x.kld, x.tiled_top1, x.fused_top1])).collect::<Vec<_>>()});
        for e in [&ea, &eb] {
            if state_file_load(state_file, e, &header, p).is_none() {
                die("state file could not be re-read after the control");
            }
        }
    }
    fused_ref::configure(&mut ea, &mut eb);
    let t1 = Instant::now();
    let rep = fused_ref::run(&mut ea, &mut eb, &s.pool, cont, &samples, &mut |i, r| {
        if (i + 1) % 50 == 0 || i + 1 == n {
            let v = r.verdict(&gate);
            eprintln!("fused-ref: {}/{n} positions, {} forks, kld mean {:.3e}, top1 {:.4}, {:.1} s", i + 1, r.refs.len(), v.kld_mean, v.top1_agreement, t1.elapsed().as_secs_f64());
        }
    });
    let run_s = t1.elapsed().as_secs_f64();
    let v = rep.verdict(&gate);
    let positions: Vec<_> = rep.positions.iter().map(|x| json!([x.position, x.kld, x.tiled_top1, x.fused_top1])).collect();
    let refs: Vec<_> = rep.refs.iter().map(|r| json!({"index": r.index, "position": r.position, "sampled": r.sampled, "disagreement": r.disagreement,
        "ref_top1": r.ref_top1, "ref_margin": r.ref_margin, "ref_absmax": r.ref_absmax, "tiled_top1": r.tiled_top1, "fused_top1_shared_state": r.fused_top1,
        "fused_top1_continuous": r.fused_cont_top1, "tiled_rms": r.tiled_rms, "fused_rms": r.fused_rms,
        "tiled_kld_vs_ref": r.tiled_kld, "fused_kld_vs_ref": r.fused_kld, "fork_replay_exact": r.fork_replay_exact,
        "tiled_attn_err2_ref2": [r.tiled_attn.0, r.tiled_attn.1], "fused_attn_err2_ref2": [r.fused_attn.0, r.fused_attn.1]})).collect();
    let res = json!({
        "kind": "fused-ref/v1", "scope": fused_ref::SCOPE, "passed": v.passed,
        "noise_floor_source": floor_source, "control": control_json,
        "gate": {"a_mean_kld_tiled_to_fused_max": gate.noise_floor_kld, "a_top1_min": gate.min_top1,
                 "wrong_top1_margin": gate.margin, "wrong_top1_max": 0, "b_fused_le_ratio_x_tiled": gate.ref_ratio_max,
                 "b_applies_to": "median and p90 of the per-position logit RMS error, attention-level rel-L2",
                 "b_jump_threshold": gate.jump_threshold, "b_jump_tolerance": v.jump_tolerance,
                 "min_positions": gate.min_positions, "min_ref_samples": gate.min_ref_samples},
        "a_candidate_vs_incumbent": {"kld_direction": "KL(P_tiled || P_fused) per teacher-forced position", "positions": v.positions,
            "kld": {"mean": v.kld_mean, "p50": v.kld_p50, "p90": v.kld_p90, "p99": v.kld_p99, "max": v.kld_max},
            "top1_agreement": v.top1_agreement, "pass_kld": v.pass_kld, "pass_top1": v.pass_top1},
        "b_f32_attention_reference": {"label": "f32-attention reference with production weights", "samples": v.ref_samples,
            "forced_disagreement_forks": v.ref_forced, "error_metric": "RMS logit difference vs the reference over the full vocabulary, same stored state",
            "note": "the per-position logit error is heavy-tailed (~10% of positions jump to 1e-2..1e-1 for any ~1e-4 attention change, likely MoE routing flips): gate on median / p90 / jump count, the mean is informational",
            "logit_rms": {"tiled": {"p50": v.tiled_rms_p50, "p90": v.tiled_rms_p90, "mean": v.tiled_rms_mean, "jumps": v.tiled_jumps},
                          "fused": {"p50": v.fused_rms_p50, "p90": v.fused_rms_p90, "mean": v.fused_rms_mean, "jumps": v.fused_jumps}},
            "attention_rel_l2": {"tiled": v.tiled_attn_rel_l2, "fused": v.fused_attn_rel_l2,
                                 "metric": "sqrt(sum |gated output - f32-attention reference|^2 / sum |reference|^2), same q and stored KV, all layers/heads of the sampled forks"},
            "pass_ref_p50": v.pass_ref_p50, "pass_ref_p90": v.pass_ref_p90, "pass_ref_jumps": v.pass_ref_jumps, "pass_ref_attn": v.pass_ref_attn,
            "tiled_kld_vs_ref_mean": v.tiled_kld_ref_mean, "fused_kld_vs_ref_mean": v.fused_kld_ref_mean,
            "fused_le_tiled_share": v.fused_le_tiled_share, "pass_ref": v.pass_ref},
        "wrong_top1_high_margin": {"fused": v.fused_wrong_top1_high_margin, "tiled_incumbent": v.tiled_wrong_top1_high_margin, "pass": v.pass_wrong_top1},
        "fork_replay_exact": v.fork_replay_exact, "scope_met": v.scope_met,
        "plan": plan, "model": a.model().display().to_string(), "model_sha256": s.loaded.ws.info.model_sha256,
        "threads": s.pool.nthreads(), "fused_chunk": eb.attn_fused_chunk(), "effective": model::engine::effective_switches(),
        "engine_commit": option_env!("OXE_GIT_COMMIT").unwrap_or("unknown"),
        "state": {"built": built, "seconds": state_s}, "run_seconds": run_s,
        "per_position": {"columns": ["position", "kld_tiled_to_fused", "tiled_top1", "fused_top1"], "rows": positions},
        "forks": refs});
    let text = serde_json::to_string_pretty(&res).unwrap();
    std::fs::write(out, &text).unwrap_or_else(|e| die(&format!("{out}: {e}")));
    let mut short = res.clone();
    for k in ["per_position", "forks", "plan"] {
        short.as_object_mut().unwrap().remove(k);
    }
    println!("{}", serde_json::to_string_pretty(&short).unwrap());
    if !v.passed {
        std::process::exit(1);
    }
}
