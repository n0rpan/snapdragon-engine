//! O-117 quality runner for the PREFILL candidates (model-free, mini model from testkit; the
//! module name carries `fused_ref` so the f32-reference CI step runs it too): the candidate rows
//! really come from batched prefill with the candidate kernel (engine counters), every position is
//! compared for the candidate and the control (row counts), the KLD direction is KL(cand || tiled)
//! (recomputed independently), the floors are the measured control's, the jump allowance is
//! max(2, ceil(3% x samples)), a degraded candidate fails, and a scalar-step run is never a pass.
//!
//! Deep contexts are synthetic states ([`synth_state`], 9,000 positions: every prefill unit of
//! the continuation is beyond `PFA_MIN_KV`, so fa2 / fa2x run for all of them).

use super::{long_engine, long_parts, rec_bytes, synth_state};
use crate::engine::{head_logits, Engine, KvType, PfAttn, PfCounts};
use crate::fused_ref::{compare_row, kld, ref_attn_f64, run_prefill, sample_positions, ArmRun, Candidate, Control, Gate, RefRow, Report, Row};
use crate::tests::{bits_eq, print_verdict};

const N_CTX: usize = 16_384;
const N_POS: usize = 9_000;

fn toks(n: usize, a: u32, b: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * a + b) % 64).collect()
}

/// One [`run_prefill`] on the mini model from the synthetic 9,000-position state.
fn o117(tag: &str, cand: Candidate, n: usize, ub: usize, control: Control, n_samples: usize, degrade: f32) -> Report {
    let (dir, plan, ws, _, pool) = long_parts(tag, 2, false);
    let mut e = long_engine(&plan, &ws, N_CTX, KvType::F16, 0, None);
    let rec = rec_bytes(&e, &pool);
    let blob = synth_state(&e, N_POS, &rec);
    e.set_pf_degrade(degrade);
    let cont = toks(n, 29, 5);
    let samples = sample_positions(n, n_samples);
    let mut load = |x: &Engine| {
        x.reset();
        x.read_state(&mut &blob[..], N_POS).unwrap();
    };
    let r = run_prefill(&mut e, &pool, &mut load, cand, &cont, ub, control, &samples, &mut |m| eprintln!("o117 {tag}: {m}"));
    drop(e);
    std::fs::remove_dir_all(&dir).ok();
    r
}

fn arm<'r>(r: &'r Report, role: &str) -> &'r ArmRun {
    r.arms.iter().find(|a| a.role == role).unwrap_or_else(|| panic!("no {role} arm"))
}

/// The full O-117 shape for prefill-fa2: 1,000 teacher-forced positions through batched prefill
/// (ubatch 256), 1,000 compared candidate rows and 1,000 control rows (tiled decode) at the same
/// positions, 200 reference samples; the counters show that the candidate's rows all came from
/// batched passes of the fa2 kernel, the incumbent's from the tiled kernel, the reference pass ran
/// the f32-attention reference for every (row, KV head) and the control decoded every position.
#[test]
fn fused_ref_o117_prefill_fa2_full_shape_and_batched_evidence() {
    let r = o117("o117fa2", Candidate::PrefillFa2, 1000, 256, Control::Decode, 200, 0.0);
    let v = r.verdict(&Gate::default());
    print_verdict("mini prefill-fa2 (1000 rows, control decode)", &v);
    assert_eq!((r.rows.len(), r.control_rows.len(), r.refs.len()), (1000, 1000, 200));
    assert_eq!((r.rows[0].position, r.rows[999].position), (N_POS, N_POS + 999));
    assert!(v.same_positions && v.scope_met && v.path_ok, "{v:?}");
    assert_eq!((v.cand.rows, v.control.rows, v.ref_samples), (1000, 1000, 200));
    let (inc, cand, refa, ctl) = (arm(&r, "incumbent"), arm(&r, "candidate"), arm(&r, "reference"), arm(&r, "control"));
    // 1000 rows from 9000 (not 256-aligned: one ubatch per pass, 256 + 256 + 256 + 232, unless
    // OXE_PF_GROUP_UNALIGNED groups them)
    let batched = |c: &PfCounts| c.rows == 1000 && c.passes >= 1 && c.passes <= 4 && c.max_rows >= 256;
    assert!(batched(&cand.pf) && cand.decode_steps == 0, "{cand:?}");
    assert!(cand.pf.units_fa2 > 0 && cand.pf.units_fa2x == 0 && cand.pf.units_tiled == 0 && cand.pf.units_ref == 0, "every candidate unit is fa2: {cand:?}");
    assert_eq!(cand.pf_kernel, "fa2");
    assert!(batched(&inc.pf) && inc.pf.units_tiled > 0 && inc.pf.units_fa2 + inc.pf.units_fa2x + inc.pf.units_ref == 0, "{inc:?}");
    assert!(batched(&refa.pf) && refa.pf.units_ref > 0 && refa.pf.units_ref % 1000 == 0, "{refa:?}");
    assert_eq!(refa.pf.units_tiled + refa.pf.units_fa2 + refa.pf.units_fa2x + refa.pf.units_per_head, 0);
    assert!(ctl.decode_steps == 1000 && ctl.pf == PfCounts::default(), "{ctl:?}");
    // the candidate's own attention check and the tiled one saw every sampled row
    for f in &r.refs {
        assert!(f.inc_attn.1 > 0.0 && f.cand_attn.1 > 0.0, "position {} not checked", f.position);
        assert!((f.inc_attn.0 / f.inc_attn.1).sqrt() < 5e-2 && (f.cand_attn.0 / f.cand_attn.1).sqrt() < 5e-2, "position {}", f.position);
        assert!(f.inc_rms.is_finite() && f.cand_rms.is_finite() && f.ref_absmax > 0.0);
    }
    // fa2 is its own numerics class: somewhere it differs from tiled
    assert!(r.refs.iter().any(|f| f.cand_rms.to_bits() != f.inc_rms.to_bits()), "fa2 bit-identical to tiled: the candidate did not run");
    // the dynamic floors are the measured control's
    assert_eq!(v.kld_max.to_bits(), v.control.mean.to_bits());
    assert_eq!(v.top1_min.to_bits(), (v.control.top1 - 0.005).to_bits());
    assert!(v.control.mean.is_finite() && v.cand.mean.is_finite());
}

/// prefill-fa2x with a batched control (ubatch 16): the runner's rows are recomputed
/// independently (fresh engines, `prefill_ext`, the production head): candidate rows are
/// KL(P_cand || P_tiled) and control rows KL(P_control || P_tiled), bit for bit; the head the runner
/// applies to every row is the batched prefill's own head (last row bit-identical); the counters
/// name fa2x.
#[test]
fn fused_ref_o117_prefill_fa2x_rows_recompute_kld_direction_and_head() {
    let (n, ub, cu) = (64usize, 32usize, 16usize);
    let r = o117("o117fa2x", Candidate::PrefillFa2x, n, ub, Control::Ubatch(cu), 16, 0.0);
    assert_eq!((r.rows.len(), r.control_rows.len(), r.refs.len()), (n, n, 16));
    let cand = arm(&r, "candidate");
    assert!(cand.pf.rows == n as u64 && cand.pf.max_rows == ub as u64 && cand.pf.units_fa2x > 0 && cand.pf.units_fa2 == 0, "{cand:?}");
    assert_eq!(cand.pf_kernel, "fa2x");
    let ctl = arm(&r, "control");
    assert!(ctl.pf.rows == n as u64 && ctl.pf.max_rows == cu as u64 && ctl.pf.units_fa2 + ctl.pf.units_fa2x == 0, "{ctl:?}");
    let v = r.verdict(&Gate { min_positions: n, min_ref_samples: 16, ..Gate::default() });
    print_verdict("mini prefill-fa2x (64 rows, control ubatch 16)", &v);
    assert!(v.path_ok && v.scope_met, "{v:?}");

    // independent recomputation
    let (dir, plan, ws, _, pool) = long_parts("o117fa2xre", 2, false);
    let probe = long_engine(&plan, &ws, N_CTX, KvType::F16, 0, None);
    let rec = rec_bytes(&probe, &pool);
    let blob = synth_state(&probe, N_POS, &rec);
    drop(probe);
    let cont = toks(n, 29, 5);
    let pass = |mode: PfAttn, u: usize| {
        let mut e = long_engine(&plan, &ws, N_CTX, KvType::F16, 0, None);
        e.set_pf_attn(mode);
        e.read_state(&mut &blob[..], N_POS).unwrap();
        let mut h = Vec::new();
        e.prefill_ext(&pool, &cont, u, Some(&mut h)).unwrap();
        let ne = plan.desc.n_embd;
        let mut last = vec![0f32; plan.desc.n_vocab];
        head_logits(&plan, &ws, &pool, &h[(n - 1) * ne..], &mut last);
        assert!(bits_eq(&last, e.logits()), "{mode:?}: head_logits is not the batched prefill's head");
        h
    };
    let (ht, hc, hk) = (pass(PfAttn::Tiled, ub), pass(PfAttn::Fa2x, ub), pass(PfAttn::Tiled, cu));
    let (ne, nv) = (plan.desc.n_embd, plan.desc.n_vocab);
    let logits = |h: &[f32], i: usize| {
        let mut o = vec![0f32; nv];
        head_logits(&plan, &ws, &pool, &h[i * ne..(i + 1) * ne], &mut o);
        o
    };
    let mut asym = false;
    for i in 0..n {
        let (lt, lc, lk) = (logits(&ht, i), logits(&hc, i), logits(&hk, i));
        assert_eq!(r.rows[i].kld.to_bits(), kld(&lc, &lt).to_bits(), "row {i}: candidate row is not KL(P_cand || P_tiled)");
        assert_eq!(r.control_rows[i].kld.to_bits(), kld(&lk, &lt).to_bits(), "row {i}: control row is not KL(P_control || P_tiled)");
        assert_eq!((r.rows[i].top1, r.rows[i].inc_top1, r.control_rows[i].top1), (crate::fused_ref::argmax(&lc), crate::fused_ref::argmax(&lt), crate::fused_ref::argmax(&lk)));
        asym |= kld(&lc, &lt) > 0.0 && kld(&lc, &lt).to_bits() != kld(&lt, &lc).to_bits();
    }
    eprintln!("o117 fa2x recompute: rows bit-identical; directions distinguishable at some row: {asym}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A deliberately degraded candidate (fa2 unit outputs scaled by 1 + 0.1, far above the kernels'
/// own attention error on this peaked synthetic context; the tiled incumbent, reference and control
/// are untouched) fails clause (c) with the rows and the path in scope.
#[test]
fn fused_ref_o117_degraded_prefill_candidate_fails() {
    let r = o117("o117bad", Candidate::PrefillFa2, 300, 256, Control::Decode, 200, 0.1);
    let v = r.verdict(&Gate { min_positions: 300, ..Gate::default() });
    print_verdict("mini prefill-fa2 degraded 0.1", &v);
    assert!(v.scope_met && v.path_ok, "{v:?}");
    assert!(!v.pass_c_attn && !v.pass_c && !v.passed, "{v:?}");
    assert!(v.cand_attn_rel_l2 > 3.0 * v.inc_attn_rel_l2, "{v:?}");
}

/// A run whose candidate passes are single tokens (ubatch 1: the scalar-step shape) is never a
/// pass (the path evidence fails, whatever the numbers; see also the synthetic case in
/// `fused_ref_o117_gate_arithmetic`, where every clause passes).
#[test]
fn fused_ref_o117_scalar_step_is_never_a_pass() {
    let r = o117("o117scalar", Candidate::PrefillFa2, 40, 1, Control::Ubatch(8), 10, 0.0);
    let lenient = Gate { kld_factor: 1e30, top1_slack: 1.0, ref_ratio_max: 1e30, jump_tol_min: 1000, min_positions: 1, min_ref_samples: 1, ..Gate::default() };
    let v = r.verdict(&lenient);
    print_verdict("mini prefill-fa2 ubatch 1", &v);
    assert!(v.scope_met && v.pass_b && v.pass_c, "{v:?}");
    assert!(!v.path_ok && !v.passed, "{v:?}");
    assert!(v.path_problems.iter().any(|p| p.contains("scalar-step")), "{:?}", v.path_problems);
    assert_eq!(arm(&r, "candidate").pf.max_rows, 1);
}

/// The prefill f32-attention reference reads the f16 cache in place and is bit for bit
/// [`ref_attn_f64`] over the widened rows ([`Engine::kv_head_f32`]).
#[test]
fn fused_ref_o117_pf_ref_head_is_ref_attn_f64() {
    let (dir, plan, ws, _, pool) = long_parts("o117refhead", 2, false);
    let e = long_engine(&plan, &ws, N_CTX, KvType::F16, 0, None);
    let rec = rec_bytes(&e, &pool);
    e.read_state(&mut &synth_state(&e, N_POS, &rec)[..], N_POS).unwrap();
    let hd = plan.desc.head_dim;
    let scale = 1.0 / (hd as f32).sqrt();
    let q: Vec<f32> = (0..hd).map(|j| ((j * 37 % 23) as f32 - 11.0) * 0.004).collect();
    for (kvh, npos) in [(0usize, N_POS), (plan.desc.n_head_kv - 1, 4_321)] {
        let (k, v) = e.kv_head_f32(0, kvh, npos);
        let (mut a, mut b) = (vec![0f32; hd], vec![0f32; hd]);
        ref_attn_f64(&q, &k, &v, npos, hd, scale, &mut a);
        let mut s = vec![0f64; npos];
        e.pf_ref_head(0, kvh, &q, npos, &mut s, &mut b);
        assert!(bits_eq(&a, &b), "kvh {kvh} npos {npos}: in-place f16 reference differs from ref_attn_f64");
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn row(i: usize, kld: f64, agree: bool) -> Row {
    Row { index: i, position: 100 + i, kld, top1: if agree { 1 } else { 2 }, inc_top1: 1, finite: true }
}

fn refrow(i: usize, inc: f64, cand: f64) -> RefRow {
    RefRow {
        index: i,
        position: 100 + i,
        sampled: true,
        disagreement: false,
        ref_top1: 1,
        ref_margin: 1.0,
        ref_absmax: 1.0,
        inc_top1: 1,
        cand_top1: 1,
        inc_rms: inc,
        cand_rms: cand,
        inc_kld_vs_ref: 0.0,
        cand_kld_vs_ref: 0.0,
        fork_replay_exact: true,
        inc_attn: (1e-6, 1.0),
        cand_attn: (1e-6, 1.0),
        finite: true,
    }
}

fn synthetic(n: usize, cand_kld: f64, ctl_kld: f64, cand_disagree: usize, ctl_disagree: usize, refs: Vec<RefRow>) -> Report {
    let ok = PfCounts { passes: 4, rows: n as u64, max_rows: 256, units_tiled: 8, ..Default::default() };
    let fa2 = PfCounts { units_tiled: 0, units_fa2: 8, ..ok };
    let rf = PfCounts { units_tiled: 0, units_ref: 8, ..ok };
    let arm = |role, pf| ArmRun { role, path: String::new(), decode_steps: 0, fused: false, pf, pf_kernel: "tiled" };
    Report {
        candidate: Candidate::PrefillFa2,
        control: Control::Decode,
        ubatch: 256,
        rows: (0..n).map(|i| row(i, cand_kld, i >= cand_disagree)).collect(),
        control_rows: (0..n).map(|i| row(i, ctl_kld, i >= ctl_disagree)).collect(),
        refs,
        arms: vec![arm("incumbent", ok), arm("candidate", fa2), arm("reference", rf), ArmRun { role: "control", decode_steps: n as u64, ..arm("control", PfCounts::default()) }],
    }
}

/// Gate arithmetic (no engine): the jump allowance, the dynamic floors at their boundaries, the
/// row-count and same-position scope, the path evidence.
#[test]
fn fused_ref_o117_gate_arithmetic() {
    let g = Gate::default();
    // max(2, ceil(3% x samples)); float rounding never adds one (0.03 x 100 = 3.0000000000000004)
    for (s, want) in [(0usize, 2usize), (10, 2), (66, 2), (67, 3), (100, 3), (200, 6), (201, 7), (1000, 30)] {
        assert_eq!(g.jump_allowance(s), want, "{s} samples");
    }
    let refs = |cj: usize| (0..200).map(|i| refrow(i, if i < 3 { 2e-3 } else { 1e-4 }, if i < cj { 2e-3 } else { 1e-4 })).collect::<Vec<_>>();
    // all clauses at their limits: KLD equal to the control's, top-1 exactly 0.5 pp below, jumps
    // tiled 3 + allowance 6
    let r = synthetic(1000, 0.02, 0.02, 25, 20, refs(9));
    let v = r.verdict(&g);
    assert!(v.path_ok && v.scope_met && v.pass_a && v.pass_b && v.pass_c && v.passed, "{v:?}");
    assert_eq!((v.inc_jumps, v.cand_jumps, v.jump_allowance), (3, 9, 6));
    assert!((v.top1_min - 0.975).abs() < 1e-12 && (v.cand.top1 - 0.975).abs() < 1e-12);
    // one step beyond each limit fails that clause only
    let v = synthetic(1000, 0.0201, 0.02, 25, 20, refs(9)).verdict(&g);
    assert!(!v.pass_a && v.pass_b && v.pass_c && !v.passed);
    let v = synthetic(1000, 0.02, 0.02, 26, 20, refs(9)).verdict(&g);
    assert!(v.pass_a && !v.pass_b && v.pass_c && !v.passed);
    let v = synthetic(1000, 0.02, 0.02, 25, 20, refs(10)).verdict(&g);
    assert!(v.pass_a && v.pass_b && !v.pass_c_jumps && !v.passed);
    // a better control floor is a stricter gate (dynamic, not absolute)
    let v = synthetic(1000, 0.02, 0.01, 25, 0, refs(9)).verdict(&g);
    assert!(!v.pass_a && !v.pass_b);
    // 999 rows, or control rows at other positions, do not meet the scope
    assert!(!synthetic(999, 0.0, 0.0, 0, 0, refs(0)).verdict(&g).scope_met);
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs(0));
    r.control_rows[7].position += 1;
    let v = r.verdict(&g);
    assert!(!v.same_positions && !v.scope_met && !v.passed);
    // fewer than 200 sampled reference positions do not meet the scope
    assert!(!synthetic(1000, 0.0, 0.0, 0, 0, refs(0)[..199].to_vec()).verdict(&g).scope_met);
    // every clause passes, but the candidate's passes were single tokens: never a pass
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs(0));
    r.arms[1].pf.max_rows = 1;
    let v = r.verdict(&g);
    assert!(v.pass_a && v.pass_b && v.pass_c && v.scope_met && !v.path_ok && !v.passed, "{v:?}");
    // path evidence: candidate rows not all batched, the wrong kernel, a tiled-looking incumbent
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs(0));
    r.arms[1].pf.rows = 999;
    assert!(!r.verdict(&g).path_ok);
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs(0));
    r.candidate = Candidate::PrefillFa2x;
    assert!(!r.verdict(&g).path_ok, "fa2 units do not evidence an fa2x candidate");
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs(0));
    r.arms[0].pf.units_fa2 = 1;
    assert!(!r.verdict(&g).path_ok);
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs(0));
    r.arms[3].decode_steps = 250;
    assert!(!r.verdict(&g).path_ok, "a control over 250 of 1000 positions");
    // KL direction helper: KL(p || q) = sum p (log p - log q)
    let (p, q) = ([0.0f32, 1.0, 2.0], [2.0f32, 0.0, 0.0]);
    let sm = |v: &[f32]| {
        let z: f64 = v.iter().map(|&x| (x as f64).exp()).sum();
        v.iter().map(|&x| (x as f64).exp() / z).collect::<Vec<f64>>()
    };
    let (ps, qs) = (sm(&p), sm(&q));
    let want: f64 = ps.iter().zip(&qs).map(|(a, b)| a * (a.ln() - b.ln())).sum();
    assert!((kld(&p, &q) - want).abs() < 1e-12 && (kld(&p, &q) - kld(&q, &p)).abs() > 1e-3);
}

/// Fail closed on non-finite values: one NaN or inf logit in ONE unsampled candidate or control
/// row, or one non-finite sampled reference row / metric, fails a report that otherwise passes
/// every clause; `kld` never turns a NaN into 0 and `compare_row` marks the row whatever argmax
/// picks.
#[test]
fn fused_ref_o117_non_finite_values_fail_closed() {
    let fin = [0.0f32, 1.0, 2.0];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let x = [0.0f32, bad, 2.0];
        for r in [compare_row(0, 0, &x, &fin), compare_row(0, 0, &fin, &x)] {
            assert!(!r.finite && r.kld.is_nan(), "{bad}: {r:?}");
        }
        assert!(!kld(&x, &fin).is_finite() || bad == f32::NEG_INFINITY, "{bad}");
    }
    assert!(kld(&[0.0, f32::NAN], &[0.0, 1.0]).is_nan(), "a NaN KL sum must stay NaN (f64::max would make it 0)");
    assert!(compare_row(0, 0, &fin, &fin).finite);
    let refs = || (0..200).map(|i| refrow(i, 1e-4, 1e-4)).collect::<Vec<_>>();
    let g = Gate::default();
    assert!(synthetic(1000, 0.0, 0.0, 0, 0, refs()).verdict(&g).passed, "baseline must pass");
    // row 501 is not a sampled position (samples are the even refs 0..200 here): one NaN logit
    for control in [false, true] {
        let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs());
        let rows = if control { &mut r.control_rows } else { &mut r.rows };
        rows[501] = compare_row(501, rows[501].position, &[0.0, f32::NAN, 1.0], &[0.0, 1.0, 1.0]);
        let v = r.verdict(&g);
        assert!(v.nonfinite_rows == 1 && !v.finite_ok && !v.passed, "control {control}: {v:?}");
    }
    // a row marked finite but with an inf KLD
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs());
    r.rows[3].kld = f64::INFINITY;
    assert!(!r.verdict(&g).passed);
    // one non-finite sampled reference row, and one non-finite metric in an otherwise finite row
    let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs());
    r.refs[17].finite = false;
    let v = r.verdict(&g);
    assert!(v.nonfinite_refs == 1 && !v.passed, "{v:?}");
    for k in 0..3 {
        let mut r = synthetic(1000, 0.0, 0.0, 0, 0, refs());
        match k {
            0 => r.refs[40].cand_rms = f64::NAN,
            1 => r.refs[40].inc_attn.0 = f64::INFINITY,
            _ => r.refs[40].cand_attn.1 = f64::NAN,
        }
        let v = r.verdict(&g);
        assert!(v.nonfinite_refs == 1 && !v.passed, "metric {k}: {v:?}");
    }
}

