//! O-117 quality runner (bench / tests only: compiled with `cfg(test)` or the bench-only
//! `f32-reference` feature, never in a production build; `oxe fused-ref --candidate ..`).
//!
//! Declared scope (receipts carry [`SCOPE`] verbatim): NOT a full-model f32 reference (a 35B model
//! in f32 at 32K-64K is ~140 GB). On ONE saved prefix state, teacher-forced over a fixed
//! continuation of N tokens, every position compared (row counts reported):
//!
//! * candidate [`Candidate::PrefillFa2`] / [`Candidate::PrefillFa2x`]: the continuation goes through
//!   GENUINELY BATCHED prefill (`Engine::prefill_ext`, the production ubatch / grouping, every row's
//!   output-normed hidden state, logits of every row through the production head
//!   [`head_logits`]) once with the tiled incumbent and once with the candidate kernel. The
//!   engine's prefill counters ([`PfCounts`]) prove which kernel ran how many units in how many
//!   batched passes; a run whose candidate rows did not all come from batched passes, or whose
//!   candidate kernel ran no unit, is never a pass ([`Verdict::path_ok`]).
//! * candidate [`Candidate::FusedDecode`]: the O-70 fused decode, engines A (tiled) and B (fused)
//!   stepping the same tokens ([`run_decode`]).
//!
//! (a) KLD per position, direction KL(P_cand || P_tiled); (b) top-1 agreement. The floor is
//! MEASURED on the same state and the same positions: a control arm with a legitimate production
//! variation of the incumbent ([`Control`]: tiled decode vs tiled batched, or another ubatch),
//! KL(P_control || P_tiled), the same direction. Gate (O-117): mean KLD(cand) <= factor x mean
//! KLD(control) (factor 1.0), top-1(cand) >= top-1(control) - 0.5 pp.
//!
//! (c) f32-attention reference with production weights at >= 200 sampled positions: for prefill a
//! third batched pass whose attention core is the f32 reference (f64 accumulation over the same
//! roped f32 query and the stored f16 K / V, `Engine::set_pf_ref`) for every row, and in the
//! incumbent's and the candidate's own passes the gated attention outputs of the sampled rows are
//! compared with the reference on THEIR OWN inputs (`Engine::set_pf_check`); for fused decode the
//! existing forked steps (engine A's state forked, one step three ways, the fork checked to replay
//! bit for bit). Per sampled position the logit RMS error vs the reference: candidate median and
//! p90 <= 1.1 x tiled; jumps (> 1e-3) <= tiled jumps + max(2, ceil(3% x sampled positions)); the
//! attention-output rel-L2 <= 1.1 x tiled.
//!
//! (d) (tool-JSON 100% at >= 48K) is run separately by the closed-loop runner; not here.
use crate::engine::{head_logits, AttnRef, Engine, PfAttn, PfCounts};
use sched::Team;

/// Scope statement for receipts: what this reference is and is not.
pub const SCOPE: &str = "O-117 quality runner: teacher-forced from one saved prefix state, every \
position compared (candidate vs tiled incumbent, and a measured control of the incumbent on the \
same positions); f32-attention reference with production weights (f64-accumulated attention over \
the same q and stored f16 KV; weights, matmuls and recurrent layers are the production CPU path); \
NOT a full-model f32 reference; clause (d) tool-JSON is run separately";

/// f32-attention reference of one query head: f32 inputs (the stored f16 K / V widened, the
/// roped f32 query), dot products and softmax accumulated in f64, output rounded to f32.
pub fn ref_attn_f64(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, scale: f32, out: &mut [f32]) {
    let mut s = vec![0f64; n];
    let mut m = f64::NEG_INFINITY;
    for p in 0..n {
        let kr = &k[p * hd..(p + 1) * hd];
        let mut a = 0f64;
        for j in 0..hd {
            a += q[j] as f64 * kr[j] as f64;
        }
        s[p] = a * scale as f64;
        m = m.max(s[p]);
    }
    let mut sum = 0f64;
    for x in s.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    let mut acc = vec![0f64; hd];
    for p in 0..n {
        let w = s[p];
        for (a, &x) in acc.iter_mut().zip(&v[p * hd..(p + 1) * hd]) {
            *a += w * x as f64;
        }
    }
    for j in 0..hd {
        out[j] = (acc[j] / sum) as f32;
    }
}

/// Which arm a forked one-step run uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    Tiled,
    Fused,
    Ref,
}

fn set_arm(e: &mut Engine, arm: Arm) {
    // the reference hook takes precedence over the fused path; keep fused off while it runs so
    // the barrier count is the tiled one the hook mirrors
    e.set_attn_ref(if arm == Arm::Ref { Some(ref_attn_f64 as AttnRef) } else { None });
    e.set_attn_fused(arm == Arm::Fused);
}

pub fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b
}

/// (top-1, top-1 minus top-2 logit).
pub fn top1_margin(v: &[f32]) -> (usize, f32) {
    let t = argmax(v);
    let second = v.iter().enumerate().filter(|&(i, _)| i != t).fold(f32::NEG_INFINITY, |m, (_, &x)| m.max(x));
    (t, v[t] - second)
}

/// KL(p_r || q_o) over the full vocabulary, in f64.
pub fn kld(r: &[f32], o: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lr, lo) = (lse(r), lse(o));
    let s = r
        .iter()
        .zip(o)
        .map(|(&a, &b)| {
            let lp = a as f64 - lr;
            (lp.exp()) * (lp - (b as f64 - lo))
        })
        .sum::<f64>();
    // (f64::max would turn a NaN sum into 0: a non-finite row must never look perfect)
    if s.is_finite() {
        s.max(0.0)
    } else {
        f64::NAN
    }
}

/// Root-mean-square logit difference (the per-position logit error of (b)).
pub fn rms(a: &[f32], b: &[f32]) -> f64 {
    (a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64).sqrt()
}

/// Every value finite (no NaN / inf).
pub fn all_finite(v: &[f32]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// One compared position: `arm` (candidate or control logits) vs `inc` (tiled incumbent).
/// KL(P_arm || P_inc); a row with any non-finite logit is marked (`finite: false`, kld NaN) and
/// fails the verdict, whatever argmax would pick.
pub fn compare_row(index: usize, position: usize, arm: &[f32], inc: &[f32]) -> Row {
    let finite = all_finite(arm) && all_finite(inc);
    Row { index, position, kld: if finite { kld(arm, inc) } else { f64::NAN }, top1: argmax(arm), inc_top1: argmax(inc), finite }
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// `count` sample indices in [0, n), evenly spaced and deterministic (first at 0, last at n-1).
pub fn sample_positions(n: usize, count: usize) -> Vec<usize> {
    if n == 0 || count == 0 {
        return Vec::new();
    }
    let c = count.min(n);
    if c == 1 {
        return vec![0];
    }
    let mut v: Vec<usize> = (0..c).map(|i| i * (n - 1) / (c - 1)).collect();
    v.dedup();
    v
}


fn quant(s: &[f64], f: f64) -> f64 {
    if s.is_empty() {
        return f64::NAN;
    }
    s[((s.len() as f64 - 1.0) * f).round() as usize]
}

/// The candidate under test (`--candidate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Candidate {
    /// batched prefill, fa2 units beyond `PFA_MIN_KV` (`OXE_PF_ATTN=fa2`)
    PrefillFa2,
    /// batched prefill, fa2x units beyond `PFA_MIN_KV` (`OXE_PF_ATTN=fa2x`)
    PrefillFa2x,
    /// O-70 fused decode attention (`OXE_ATTN=fused`, cutoffs 0)
    FusedDecode,
}

impl Candidate {
    pub fn parse(s: &str) -> Option<Candidate> {
        match s.trim() {
            "prefill-fa2" => Some(Candidate::PrefillFa2),
            "prefill-fa2x" => Some(Candidate::PrefillFa2x),
            "fused-decode" => Some(Candidate::FusedDecode),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Candidate::PrefillFa2 => "prefill-fa2",
            Candidate::PrefillFa2x => "prefill-fa2x",
            Candidate::FusedDecode => "fused-decode",
        }
    }

    pub fn is_prefill(self) -> bool {
        self != Candidate::FusedDecode
    }

    fn pf_attn(self) -> PfAttn {
        match self {
            Candidate::PrefillFa2 => PfAttn::Fa2,
            Candidate::PrefillFa2x => PfAttn::Fa2x,
            Candidate::FusedDecode => PfAttn::Tiled,
        }
    }
}

/// The control arm: the tiled incumbent with a legitimate production variation, from the same
/// state over the same positions (`--control`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// tiled decode, one token per step (production generation); for prefill candidates
    Decode,
    /// tiled batched prefill in ubatches of this size (production prompt processing at another
    /// ubatch; for fused decode: the batched prefill vs the decode incumbent)
    Ubatch(usize),
}

impl Control {
    /// `decode` or `ubatch:N` (N >= 1).
    pub fn parse(s: &str) -> Option<Control> {
        match s.trim() {
            "decode" => Some(Control::Decode),
            x => x.strip_prefix("ubatch:").and_then(|n| n.parse().ok()).filter(|&n| n >= 1).map(Control::Ubatch),
        }
    }

    pub fn name(self) -> String {
        match self {
            Control::Decode => "decode".into(),
            Control::Ubatch(u) => format!("ubatch:{u}"),
        }
    }
}

/// One compared position: candidate (or control) vs the tiled incumbent.
#[derive(Clone, Debug)]
pub struct Row {
    /// index into the continuation
    pub index: usize,
    /// absolute position of the token (prefix length + index)
    pub position: usize,
    /// KL(P_cand || P_tiled) for candidate rows, KL(P_control || P_tiled) for control rows
    pub kld: f64,
    /// top-1 of the candidate (control) arm
    pub top1: usize,
    /// top-1 of the tiled incumbent
    pub inc_top1: usize,
    /// every logit of both rows finite (a non-finite row fails the verdict)
    pub finite: bool,
}

/// One f32-attention reference position of (c).
#[derive(Clone, Debug)]
pub struct RefRow {
    pub index: usize,
    pub position: usize,
    /// the even sample (gated), or (decode) only a forced fork at a top-1 disagreement
    pub sampled: bool,
    pub disagreement: bool,
    pub ref_top1: usize,
    pub ref_margin: f32,
    /// max |logit| of the reference (scale of the errors)
    pub ref_absmax: f32,
    pub inc_top1: usize,
    pub cand_top1: usize,
    /// RMS logit difference vs the reference over the vocabulary
    pub inc_rms: f64,
    pub cand_rms: f64,
    /// KL(P_arm || P_ref) (informational)
    pub inc_kld_vs_ref: f64,
    pub cand_kld_vs_ref: f64,
    /// decode forks: the replayed tiled step equals the un-forked one bit for bit (prefill: true)
    pub fork_replay_exact: bool,
    /// attention level, summed over layers and heads: (squared error of the arm's gated
    /// attention output vs the f32-attention reference on the same inputs, squared reference)
    pub inc_attn: (f64, f64),
    pub cand_attn: (f64, f64),
    /// the reference, incumbent and candidate logit rows were all finite
    pub finite: bool,
}

/// What one arm actually ran (the evidence behind [`Verdict::path_ok`]).
#[derive(Clone, Debug, Default)]
pub struct ArmRun {
    /// incumbent | candidate | reference | control
    pub role: &'static str,
    pub path: String,
    /// one-token decode steps of the continuation (forks excluded)
    pub decode_steps: u64,
    /// fused decode attention on
    pub fused: bool,
    /// batched-prefill counters of the arm's pass
    pub pf: PfCounts,
    /// the engine's resolved prefill kernel beyond the cutoff (tiled | fa2 | fa2x)
    pub pf_kernel: &'static str,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub candidate: Candidate,
    pub control: Control,
    /// candidate / incumbent ubatch (prefill candidates)
    pub ubatch: usize,
    pub rows: Vec<Row>,
    pub control_rows: Vec<Row>,
    pub refs: Vec<RefRow>,
    pub arms: Vec<ArmRun>,
}

/// O-117 thresholds (the receipt states them).
#[derive(Clone, Debug)]
pub struct Gate {
    /// (a) mean KLD(cand || tiled) <= kld_factor x mean KLD(control || tiled)
    pub kld_factor: f64,
    /// (b) top-1(cand, tiled) >= top-1(control, tiled) - top1_slack
    pub top1_slack: f64,
    /// (c) candidate <= ratio x tiled: median and p90 of the logit error, attention rel-L2
    pub ref_ratio_max: f64,
    /// (c) a sampled position "jumps" when its logit RMS error exceeds this
    pub jump_threshold: f64,
    /// (c) candidate jumps <= tiled jumps + max(jump_tol_min, ceil(jump_tol_frac x samples))
    pub jump_tol_min: usize,
    pub jump_tol_frac: f64,
    /// compared rows (candidate and control, the same positions)
    pub min_positions: usize,
    pub min_ref_samples: usize,
}

impl Default for Gate {
    fn default() -> Gate {
        Gate { kld_factor: 1.0, top1_slack: 0.005, ref_ratio_max: 1.1, jump_threshold: 1e-3, jump_tol_min: 2, jump_tol_frac: 0.03, min_positions: 1000, min_ref_samples: 200 }
    }
}

impl Gate {
    /// (c) jump allowance over `samples` sampled positions: max(jump_tol_min, ceil(jump_tol_frac x
    /// samples)) (O-117 "tiled + max(2, 3%)"; 200 samples -> 6, 1000 -> 30).
    pub fn jump_allowance(&self, samples: usize) -> usize {
        self.jump_tol_min.max((self.jump_tol_frac * samples as f64 - 1e-9).ceil() as usize)
    }
}

/// KLD distribution and top-1 agreement of a set of rows.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub rows: usize,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub top1: f64,
}

pub fn stats(rows: &[Row]) -> Stats {
    let mut k: Vec<f64> = rows.iter().map(|p| p.kld).collect();
    k.sort_by(|a, b| a.total_cmp(b));
    let n = k.len();
    if n == 0 {
        return Stats { rows: 0, mean: f64::NAN, p50: f64::NAN, p90: f64::NAN, p99: f64::NAN, max: f64::NAN, top1: f64::NAN };
    }
    let agree = rows.iter().filter(|p| p.top1 == p.inc_top1).count();
    Stats { rows: n, mean: k.iter().sum::<f64>() / n as f64, p50: quant(&k, 0.5), p90: quant(&k, 0.9), p99: quant(&k, 0.99), max: k[n - 1], top1: agree as f64 / n as f64 }
}

/// Summary numbers and the per-clause verdict.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub candidate: Candidate,
    pub cand: Stats,
    pub control: Stats,
    /// candidate and control rows cover the same positions
    pub same_positions: bool,
    /// (a) limit: kld_factor x control mean
    pub kld_max: f64,
    /// (b) limit: control top-1 - slack
    pub top1_min: f64,
    pub pass_a: bool,
    pub pass_b: bool,
    pub ref_samples: usize,
    /// decode: extra forks at top-1 disagreements (not gated)
    pub ref_forced: usize,
    pub inc_rms_p50: f64,
    pub inc_rms_p90: f64,
    pub cand_rms_p50: f64,
    pub cand_rms_p90: f64,
    pub inc_rms_mean: f64,
    pub cand_rms_mean: f64,
    pub inc_jumps: usize,
    pub cand_jumps: usize,
    pub jump_allowance: usize,
    /// sqrt(sum err^2 / sum ref^2) of the gated attention outputs over the sampled positions
    pub inc_attn_rel_l2: f64,
    pub cand_attn_rel_l2: f64,
    pub pass_c_p50: bool,
    pub pass_c_p90: bool,
    pub pass_c_jumps: bool,
    pub pass_c_attn: bool,
    pub pass_c: bool,
    /// the arms ran the paths they claim (batched candidate prefill with the candidate kernel,
    /// batched tiled incumbent, reference units, control path); why not, when not
    pub path_ok: bool,
    pub path_problems: Vec<String>,
    pub fork_replay_exact: bool,
    /// candidate + control rows with a non-finite logit or KLD (any one fails the verdict)
    pub nonfinite_rows: usize,
    /// reference rows (sampled or forced) with a non-finite logit row or metric
    pub nonfinite_refs: usize,
    pub finite_ok: bool,
    pub scope_met: bool,
    pub passed: bool,
}

impl RefRow {
    /// The row and every metric the verdict reads from it are finite.
    pub fn all_finite(&self) -> bool {
        self.finite
            && [self.inc_rms, self.cand_rms, self.inc_attn.0, self.inc_attn.1, self.cand_attn.0, self.cand_attn.1].iter().all(|x| x.is_finite())
            && self.ref_margin.is_finite()
            && self.ref_absmax.is_finite()
    }
}

/// Checks that the arms of `r` ran what they claim; returns the problems (empty = ok).
pub fn path_problems(r: &Report) -> Vec<String> {
    let n = r.rows.len() as u64;
    let mut bad = Vec::new();
    let arm = |role: &str| r.arms.iter().find(|a| a.role == role);
    let mut need = |ok: bool, what: String| {
        if !ok {
            bad.push(what);
        }
    };
    let (Some(inc), Some(cand)) = (arm("incumbent"), arm("candidate")) else {
        return vec!["incumbent or candidate arm missing".into()];
    };
    if r.candidate.is_prefill() {
        let (want, other) = if r.candidate == Candidate::PrefillFa2 { (cand.pf.units_fa2, cand.pf.units_fa2x) } else { (cand.pf.units_fa2x, cand.pf.units_fa2) };
        need(cand.decode_steps == 0 && cand.pf.rows == n && n > 0, format!("candidate rows not all from batched prefill ({} of {n} rows batched, {} decode steps)", cand.pf.rows, cand.decode_steps));
        need(cand.pf.max_rows > 1, format!("candidate passes are not batched (largest pass {} rows): a scalar-step run is never a pass", cand.pf.max_rows));
        need(want > 0 && other == 0 && cand.pf.units_ref == 0, format!("candidate kernel {} did not run (units fa2 {} fa2x {} ref {})", r.candidate.name(), cand.pf.units_fa2, cand.pf.units_fa2x, cand.pf.units_ref));
        need(inc.decode_steps == 0 && inc.pf.rows == n, format!("incumbent rows not all from batched prefill ({} of {n})", inc.pf.rows));
        need(inc.pf.units_fa2 + inc.pf.units_fa2x + inc.pf.units_ref == 0 && inc.pf.units_tiled + inc.pf.units_per_head > 0, format!("incumbent is not the tiled prefill ({:?})", inc.pf));
        if !r.refs.is_empty() {
            match arm("reference") {
                Some(a) => need(a.pf.rows == n && a.pf.units_ref > 0 && a.pf.units_fa2 + a.pf.units_fa2x + a.pf.units_tiled + a.pf.units_per_head == 0, format!("reference pass did not run the f32-attention reference everywhere ({:?})", a.pf)),
                None => need(false, "reference arm missing".into()),
            }
        }
    } else {
        need(cand.fused && cand.decode_steps == n && n > 0, format!("candidate is not the fused decode over every position (fused {}, {} steps of {n})", cand.fused, cand.decode_steps));
        need(!inc.fused && inc.decode_steps == n, format!("incumbent is not the tiled decode over every position ({} steps)", inc.decode_steps));
    }
    match (arm("control"), r.control) {
        (Some(c), Control::Decode) => need(c.decode_steps == n && !c.fused, format!("control decode ran {} of {n} steps", c.decode_steps)),
        (Some(c), Control::Ubatch(_)) => need(c.pf.rows == n && c.pf.units_fa2 + c.pf.units_fa2x + c.pf.units_ref == 0, format!("control is not a tiled batched prefill of every row ({:?})", c.pf)),
        (None, _) => need(false, "control arm missing".into()),
    }
    if r.candidate == Candidate::FusedDecode && r.control == Control::Decode {
        bad.push("fused-decode needs a batched control (the decode incumbent compared with itself is no floor)".into());
    }
    bad
}

impl Report {
    pub fn verdict(&self, g: &Gate) -> Verdict {
        let (cand, control) = (stats(&self.rows), stats(&self.control_rows));
        let same_positions = self.rows.len() == self.control_rows.len() && self.rows.iter().zip(&self.control_rows).all(|(a, b)| a.position == b.position);
        let kld_max = g.kld_factor * control.mean;
        let top1_min = control.top1 - g.top1_slack;
        // (a) an absolute floor so two bit-exact arms (both 0) compare equal
        let pass_a = cand.rows > 0 && cand.mean <= kld_max + 1e-15;
        let pass_b = cand.rows > 0 && cand.top1 >= top1_min - 1e-12;
        let sampled: Vec<&RefRow> = self.refs.iter().filter(|r| r.sampled).collect();
        let m = sampled.len().max(1) as f64;
        let sorted = |f: &dyn Fn(&RefRow) -> f64| {
            let mut v: Vec<f64> = sampled.iter().map(|r| f(r)).collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v
        };
        let (is, cs) = (sorted(&|r| r.inc_rms), sorted(&|r| r.cand_rms));
        let (i50, i90, c50, c90) = (quant(&is, 0.5), quant(&is, 0.9), quant(&cs, 0.5), quant(&cs, 0.9));
        let inc_jumps = is.iter().filter(|&&x| x > g.jump_threshold).count();
        let cand_jumps = cs.iter().filter(|&&x| x > g.jump_threshold).count();
        let jump_allowance = g.jump_allowance(sampled.len());
        let rel = |f: &dyn Fn(&RefRow) -> (f64, f64)| {
            let (e, r) = sampled.iter().fold((0.0, 0.0), |a, x| (a.0 + f(x).0, a.1 + f(x).1));
            if r > 0.0 {
                (e / r).sqrt()
            } else {
                f64::NAN
            }
        };
        let (ia, ca) = (rel(&|r| r.inc_attn), rel(&|r| r.cand_attn));
        let within = |c: f64, t: f64| c <= g.ref_ratio_max * t + 1e-12;
        let pass_c_p50 = !sampled.is_empty() && within(c50, i50);
        let pass_c_p90 = !sampled.is_empty() && within(c90, i90);
        let pass_c_jumps = !sampled.is_empty() && cand_jumps <= inc_jumps + jump_allowance;
        let pass_c_attn = ia.is_finite() && ca.is_finite() && within(ca, ia);
        let pass_c = pass_c_p50 && pass_c_p90 && pass_c_jumps && pass_c_attn;
        let path_problems = path_problems(self);
        let path_ok = path_problems.is_empty();
        let fork_replay_exact = self.refs.iter().all(|r| r.fork_replay_exact);
        let scope_met = cand.rows >= g.min_positions && same_positions && sampled.len() >= g.min_ref_samples;
        let nonfinite_rows = self.rows.iter().chain(&self.control_rows).filter(|r| !r.finite || !r.kld.is_finite()).count();
        let nonfinite_refs = self.refs.iter().filter(|r| !r.all_finite()).count();
        let finite_ok = nonfinite_rows == 0 && nonfinite_refs == 0;
        Verdict {
            candidate: self.candidate,
            same_positions,
            kld_max,
            top1_min,
            pass_a,
            pass_b,
            ref_samples: sampled.len(),
            ref_forced: self.refs.len() - sampled.len(),
            inc_rms_p50: i50,
            inc_rms_p90: i90,
            cand_rms_p50: c50,
            cand_rms_p90: c90,
            inc_rms_mean: is.iter().sum::<f64>() / m,
            cand_rms_mean: cs.iter().sum::<f64>() / m,
            inc_jumps,
            cand_jumps,
            jump_allowance,
            inc_attn_rel_l2: ia,
            cand_attn_rel_l2: ca,
            pass_c_p50,
            pass_c_p90,
            pass_c_jumps,
            pass_c_attn,
            pass_c,
            path_ok,
            path_problems,
            fork_replay_exact,
            nonfinite_rows,
            nonfinite_refs,
            finite_ok,
            scope_met,
            passed: scope_met && path_ok && fork_replay_exact && finite_ok && pass_a && pass_b && pass_c,
            cand,
            control,
        }
    }
}

// ------------------------------------------------------------------ prefill candidates

/// One batched teacher-forced pass of `cont` from the loaded state: every row's output-normed
/// hidden state, the attention check of `check` positions, what ran.
#[allow(clippy::too_many_arguments)]
fn batched_arm<'a, P: Team + ?Sized>(e: &mut Engine<'a>, pool: &P, load: &mut dyn FnMut(&Engine<'a>), base: usize, cont: &[u32], ubatch: usize, mode: PfAttn, pf_ref: bool, check: Option<Vec<usize>>, role: &'static str) -> (Vec<f32>, Vec<(usize, f64, f64)>, ArmRun) {
    load(e);
    assert_eq!(e.pos(), base, "load() must restore the saved prefix state");
    e.set_pf_attn(mode);
    e.set_pf_ref(pf_ref);
    e.set_pf_check(check);
    e.reset_pf_counts();
    let mut h = Vec::new();
    e.prefill_ext(pool, cont, ubatch, Some(&mut h));
    let chk = e.take_pf_check();
    let kernel = if pf_ref { "f32-attention reference" } else { e.pf_attn_kernel() };
    let run = ArmRun { role, path: format!("batched prefill, ubatch {ubatch}, attention {kernel}"), decode_steps: 0, fused: false, pf: e.pf_counts(), pf_kernel: e.pf_attn_kernel() };
    e.set_pf_ref(false);
    e.set_pf_check(None);
    (h, chk, run)
}

fn check_of(chk: &[(usize, f64, f64)], pos: usize) -> (f64, f64) {
    chk.iter().find(|c| c.0 == pos).map_or((0.0, 0.0), |c| (c.1, c.2))
}

/// O-117 for a prefill candidate on engine `e` (f16 KV). `load(e)` restores the saved prefix
/// state (the same bytes before every arm); `cont` is the teacher-forced continuation, compared at
/// every position; `samples` are indices into `cont` for the f32-attention reference (an empty
/// set skips the reference pass: (c) then fails). Arms: tiled batched (incumbent), candidate
/// batched, reference batched, control; the logits of every row through [`head_logits`].
#[allow(clippy::too_many_arguments)]
pub fn run_prefill<'a, P: Team + ?Sized>(e: &mut Engine<'a>, pool: &P, load: &mut dyn FnMut(&Engine<'a>), cand: Candidate, cont: &[u32], ubatch: usize, control: Control, samples: &[usize], progress: &mut dyn FnMut(&str)) -> Report {
    assert!(cand.is_prefill(), "run_prefill: a prefill candidate");
    assert!(ubatch >= 1 && !cont.is_empty());
    let (plan, ws) = (e.plan, e.weight_store());
    let (de, nv) = (plan.desc.n_embd, plan.desc.n_vocab);
    e.set_attn_fused(false);
    load(e);
    let base = e.pos();
    let check: Vec<usize> = samples.iter().map(|&i| base + i).collect();
    let (h_inc, chk_inc, a_inc) = batched_arm(e, pool, &mut *load, base, cont, ubatch, PfAttn::Tiled, false, Some(check.clone()), "incumbent");
    progress("incumbent (tiled batched) pass done");
    let (h_cand, chk_cand, a_cand) = batched_arm(e, pool, &mut *load, base, cont, ubatch, cand.pf_attn(), false, Some(check.clone()), "candidate");
    progress("candidate batched pass done");
    let mut arms = vec![a_inc, a_cand];
    let h_ref = if samples.is_empty() {
        Vec::new()
    } else {
        let (h, _, a) = batched_arm(e, pool, &mut *load, base, cont, ubatch, PfAttn::Tiled, true, None, "reference");
        arms.push(a);
        progress("reference (f32-attention) batched pass done");
        h
    };
    let h_ctl = match control {
        Control::Ubatch(u) => {
            let (h, _, a) = batched_arm(e, pool, &mut *load, base, cont, u, PfAttn::Tiled, false, None, "control");
            arms.push(a);
            h
        }
        Control::Decode => {
            load(e);
            assert_eq!(e.pos(), base);
            Vec::new()
        }
    };
    let (mut li, mut lc, mut lr) = (vec![0f32; nv], vec![0f32; nv], vec![0f32; nv]);
    let mut lk = vec![0f32; nv];
    let (mut rows, mut control_rows, mut refs) = (Vec::new(), Vec::new(), Vec::new());
    for (i, &t) in cont.iter().enumerate() {
        let row = i * de..(i + 1) * de;
        head_logits(plan, ws, pool, &h_inc[row.clone()], &mut li);
        head_logits(plan, ws, pool, &h_cand[row.clone()], &mut lc);
        let lctl: &[f32] = match control {
            Control::Decode => {
                e.step(pool, t);
                e.logits()
            }
            Control::Ubatch(_) => {
                head_logits(plan, ws, pool, &h_ctl[row.clone()], &mut lk);
                &lk
            }
        };
        let (ti, tc) = (argmax(&li), argmax(&lc));
        let position = base + i;
        control_rows.push(compare_row(i, position, lctl, &li));
        rows.push(compare_row(i, position, &lc, &li));
        if samples.binary_search(&i).is_ok() {
            head_logits(plan, ws, pool, &h_ref[row.clone()], &mut lr);
            let (rt, rm) = top1_margin(&lr);
            refs.push(RefRow {
                index: i,
                position,
                sampled: true,
                disagreement: ti != tc,
                ref_top1: rt,
                ref_margin: rm,
                ref_absmax: lr.iter().fold(0f32, |m, x| m.max(x.abs())),
                inc_top1: ti,
                cand_top1: tc,
                inc_rms: rms(&li, &lr),
                cand_rms: rms(&lc, &lr),
                inc_kld_vs_ref: kld(&li, &lr),
                cand_kld_vs_ref: kld(&lc, &lr),
                fork_replay_exact: true,
                inc_attn: check_of(&chk_inc, position),
                cand_attn: check_of(&chk_cand, position),
                finite: all_finite(&lr) && all_finite(&li) && all_finite(&lc),
            });
        }
        if (i + 1) % 100 == 0 || i + 1 == cont.len() {
            progress(&format!("compared {}/{} rows", i + 1, cont.len()));
        }
    }
    if control == Control::Decode {
        arms.push(ArmRun { role: "control", path: "tiled decode, one token per step".into(), decode_steps: cont.len() as u64, fused: false, pf: PfCounts::default(), pf_kernel: "n/a" });
    }
    Report { candidate: cand, control, ubatch, rows, control_rows, refs, arms }
}

// ------------------------------------------------------------------ fused decode candidate

/// Prepare engine A (incumbent) and B (candidate) for [`run_decode`]: tiled attention at every
/// n_kv (cutoff 0, so the reference hook is reachable at any depth), legacy off, fused on B only.
pub fn configure(a: &mut Engine, b: &mut Engine) {
    fn base(e: &mut Engine) {
        e.set_attn_legacy(false);
        e.set_attn_tiled_min_kv(0);
        e.set_attn_ref(None);
    }
    base(a);
    base(b);
    a.set_attn_fused(false);
    b.set_attn_fused(true);
    // v0.1.3 (lane A): fused has its own n_kv cutoffs; the reference harness measures the fused
    // kernel itself at every position, as before the cutoffs (0 = always)
    b.set_attn_fused_min_kv(0, 0);
}

/// Decode core: A and B hold the same prefix state, configured by [`configure`]. Steps `cont[i]`
/// on both; candidate rows KL(P_fused || P_tiled); forks A at `samples` (indices into `cont`) and
/// at every top-1 disagreement: one step three ways (f32-attention reference, fused, tiled) over
/// A's stored state, the tiled one checked to replay bit for bit. `on_pos(i, tiled, fused)` sees
/// both continuous logit rows of every position.
pub fn run_decode<P: Team + ?Sized>(a: &mut Engine, b: &mut Engine, pool: &P, cont: &[u32], samples: &[usize], on_pos: &mut dyn FnMut(usize, &[f32], &[f32]), progress: &mut dyn FnMut(usize, &[Row])) -> (Vec<Row>, Vec<RefRow>) {
    assert_eq!(a.pos(), b.pos(), "A and B must start from the same state");
    assert!(!a.attn_fused() && b.attn_fused(), "configure() first");
    let base = a.pos();
    let (mut rows, mut refs) = (Vec::new(), Vec::new());
    let mut snap = None;
    for (i, &t) in cont.iter().enumerate() {
        b.step(pool, t);
        let lb = b.logits().to_vec();
        let s = match snap.take() {
            Some(old) => a.save_recurrent_into(old),
            None => a.save_recurrent(),
        };
        a.step(pool, t);
        let la = a.logits().to_vec();
        on_pos(i, &la, &lb);
        let (ta, tb) = (argmax(&la), argmax(&lb));
        rows.push(compare_row(i, base + i, &lb, &la));
        let sampled = samples.binary_search(&i).is_ok();
        if sampled || ta != tb {
            let mut out = [Vec::new(), Vec::new()];
            let mut fused_attn = (0.0, 0.0);
            for (k, arm) in [Arm::Ref, Arm::Fused].into_iter().enumerate() {
                a.restore_recurrent(&s);
                set_arm(a, arm);
                a.set_attn_check(arm == Arm::Fused);
                a.step(pool, t);
                out[k] = a.logits().to_vec();
                if arm == Arm::Fused {
                    fused_attn = a.take_attn_check().unwrap();
                }
            }
            a.restore_recurrent(&s);
            set_arm(a, Arm::Tiled);
            a.set_attn_check(true);
            a.step(pool, t);
            let tiled_attn = a.take_attn_check().unwrap();
            a.set_attn_check(false);
            let exact = bits_eq(a.logits(), &la);
            let [lr, lf] = out;
            let (rt, rm) = top1_margin(&lr);
            refs.push(RefRow {
                index: i,
                position: base + i,
                sampled,
                disagreement: ta != tb,
                ref_top1: rt,
                ref_margin: rm,
                ref_absmax: lr.iter().fold(0f32, |m, x| m.max(x.abs())),
                inc_top1: ta,
                cand_top1: argmax(&lf),
                inc_rms: rms(&la, &lr),
                cand_rms: rms(&lf, &lr),
                inc_kld_vs_ref: kld(&la, &lr),
                cand_kld_vs_ref: kld(&lf, &lr),
                fork_replay_exact: exact,
                inc_attn: tiled_attn,
                cand_attn: fused_attn,
                finite: all_finite(&lr) && all_finite(&lf) && all_finite(&la),
            });
        }
        snap = Some(s);
        progress(i, &rows);
    }
    (rows, refs)
}

/// O-117 for the fused decode: the control is the tiled BATCHED prefill of the same tokens
/// (`Control::Ubatch`), compared with the tiled decode incumbent at every position
/// (KL(P_batched || P_decode)); then [`run_decode`]. `load(e)` restores the saved prefix state.
#[allow(clippy::too_many_arguments)]
pub fn run_fused_decode<'a, P: Team + ?Sized>(a: &mut Engine<'a>, b: &mut Engine<'a>, pool: &P, load: &mut dyn FnMut(&Engine<'a>), cont: &[u32], control: Control, samples: &[usize], progress: &mut dyn FnMut(&str)) -> Report {
    let Control::Ubatch(cu) = control else { panic!("fused-decode needs a batched control (--control ubatch:N)") };
    let (plan, ws) = (a.plan, a.weight_store());
    let (de, nv) = (plan.desc.n_embd, plan.desc.n_vocab);
    a.set_attn_fused(false);
    load(a);
    let base = a.pos();
    let (h_ctl, _, a_ctl) = batched_arm(a, pool, &mut *load, base, cont, cu, PfAttn::Tiled, false, None, "control");
    progress("control (tiled batched) pass done");
    load(a);
    load(b);
    configure(a, b);
    let mut lk = vec![0f32; nv];
    let mut control_rows = Vec::new();
    let n = cont.len();
    let (rows, refs) = run_decode(
        a,
        b,
        pool,
        cont,
        samples,
        &mut |i, la, _| {
            head_logits(plan, ws, pool, &h_ctl[i * de..(i + 1) * de], &mut lk);
            control_rows.push(compare_row(i, base + i, &lk, la));
        },
        &mut |i, _| {
            if (i + 1) % 100 == 0 || i + 1 == n {
                progress(&format!("decoded {}/{n} positions", i + 1));
            }
        },
    );
    let forks = refs.len() as u64;
    let arms = vec![
        ArmRun { role: "incumbent", path: "tiled decode, one token per step".into(), decode_steps: n as u64, fused: a.attn_fused(), pf: PfCounts::default(), pf_kernel: "n/a" },
        ArmRun { role: "candidate", path: "fused decode (O-70), one token per step, cutoffs 0".into(), decode_steps: n as u64, fused: b.attn_fused(), pf: PfCounts::default(), pf_kernel: "n/a" },
        ArmRun { role: "reference", path: format!("forked decode steps of A, f32-attention reference ({forks} forks)"), decode_steps: 0, fused: false, pf: PfCounts::default(), pf_kernel: "n/a" },
        a_ctl,
    ];
    Report { candidate: Candidate::FusedDecode, control, ubatch: cu, rows, control_rows, refs, arms }
}
