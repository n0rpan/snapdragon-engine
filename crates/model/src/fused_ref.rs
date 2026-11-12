//! Long-context numeric reference for the O-70 fused attention (bench / tests only: compiled with
//! `cfg(test)` or the bench-only `f32-reference` feature, never in a production build).
//!
//! Declared scope (the receipt carries it verbatim, [`SCOPE`]): this is NOT a full-model f32
//! reference. A full f32 evaluation of a 35B model at 32K-64K context is infeasible on this
//! machine (about 140 GB of f32 weights). What it measures instead, on the SAME saved prefix state:
//!
//! (a) candidate vs incumbent, teacher-forced: engine A (tiled attention, the incumbent) and
//!     engine B (fused attention, the candidate) step the same fixed continuation from the same
//!     state; per position KL(P_tiled || P_fused) and top-1 agreement.
//! (b) f32-attention reference with production weights: at sampled positions (and at every
//!     position where A and B disagree on top-1), engine A's state is forked (recurrent snapshot +
//!     KV truncation) and the one step is re-run three ways over the SAME stored f16 KV: f32
//!     attention math (f64 accumulation, [`ref_attn_f64`]), fused, tiled. Weights, matmuls,
//!     recurrent layers and the stored KV are production; only the attention core differs. The
//!     per-position logit error of fused and tiled against that reference is compared by its
//!     median, p90 and count of jumps (> 1e-3), and the gated attention outputs of both arms are
//!     compared with the reference on the same inputs (rel-L2), see [`Gate::ref_ratio_max`].
//!
//! The fork is checked: after the three forked steps, the tiled step is replayed and must equal
//! the un-forked step bit for bit (`fork_replay_exact`).
use crate::engine::{AttnRef, Engine};
use sched::Team;

/// Scope statement for receipts: what this reference is and is not.
pub const SCOPE: &str = "f32-attention reference with production weights (fused vs tiled \
attention over the same stored f16 KV and the same saved prefix state; weights, matmuls and \
recurrent layers are the production CPU path); NOT a full-model f32 reference";

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
    r.iter()
        .zip(o)
        .map(|(&a, &b)| {
            let lp = a as f64 - lr;
            (lp.exp()) * (lp - (b as f64 - lo))
        })
        .sum::<f64>()
        .max(0.0)
}

/// Root-mean-square logit difference (the per-position logit error of (b)).
pub fn rms(a: &[f32], b: &[f32]) -> f64 {
    (a.iter().zip(b).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len().max(1) as f64).sqrt()
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

/// One teacher-forced position of (a).
#[derive(Clone, Debug)]
pub struct Pos {
    /// absolute position of the stepped token (prefix length + index)
    pub position: usize,
    /// KL(P_tiled || P_fused), continuous engines
    pub kld: f64,
    pub tiled_top1: usize,
    pub fused_top1: usize,
}

/// One forked reference position of (b), all three arms over engine A's stored state.
#[derive(Clone, Debug)]
pub struct RefPos {
    pub index: usize,
    pub position: usize,
    /// why it was forked: the even sample, a top-1 disagreement of the continuous engines, or both
    pub sampled: bool,
    pub disagreement: bool,
    pub ref_top1: usize,
    pub ref_margin: f32,
    /// max |logit| of the reference (scale of the errors)
    pub ref_absmax: f32,
    pub tiled_top1: usize,
    pub fused_top1: usize,
    /// top-1 of the continuous fused engine B at this position
    pub fused_cont_top1: usize,
    pub tiled_rms: f64,
    pub fused_rms: f64,
    pub tiled_kld: f64,
    pub fused_kld: f64,
    pub fork_replay_exact: bool,
    /// attention level, summed over layers and heads of the forked step: (squared error of the
    /// arm's gated attention output vs the f32-attention reference on the same inputs, squared
    /// reference)
    pub tiled_attn: (f64, f64),
    pub fused_attn: (f64, f64),
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub positions: Vec<Pos>,
    pub refs: Vec<RefPos>,
}

/// Gate thresholds (the receipt states them).
#[derive(Clone, Debug)]
pub struct Gate {
    /// incumbent noise floor for the mean KL(P_tiled || P_fused) (Ornith 0.027, Qwen 0.01)
    pub noise_floor_kld: f64,
    pub min_top1: f64,
    /// a wrong top-1 counts when the f32-attention reference's top-1 margin exceeds this
    pub margin: f32,
    /// (b): fused <= ratio x tiled for the median and the p90 of the per-position logit error
    /// and for the attention-level rel-L2 (the logit error is heavy-tailed: ~10% of positions
    /// jump to 1e-2..1e-1 for ANY ~1e-4 attention change, likely MoE routing flips; a mean over
    /// a few samples measures where the jumps landed, not the kernel)
    pub ref_ratio_max: f64,
    /// (b): a position "jumps" when its logit RMS error exceeds this
    pub jump_threshold: f64,
    /// (b): fused jumps <= tiled jumps + max(jump_tol_min, jump_tol_frac x samples)
    pub jump_tol_min: usize,
    pub jump_tol_frac: f64,
    pub min_positions: usize,
    pub min_ref_samples: usize,
}

impl Default for Gate {
    fn default() -> Gate {
        Gate { noise_floor_kld: 0.027, min_top1: 0.995, margin: 5.0, ref_ratio_max: 1.1, jump_threshold: 1e-3, jump_tol_min: 2, jump_tol_frac: 0.03, min_positions: 1000, min_ref_samples: 200 }
    }
}

/// Summary numbers and the verdict.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub positions: usize,
    pub kld_mean: f64,
    pub kld_p50: f64,
    pub kld_p90: f64,
    pub kld_p99: f64,
    pub kld_max: f64,
    pub top1_agreement: f64,
    pub ref_samples: usize,
    pub ref_forced: usize,
    pub tiled_rms_mean: f64,
    pub tiled_rms_p50: f64,
    pub tiled_rms_p90: f64,
    pub fused_rms_p50: f64,
    pub fused_rms_p90: f64,
    pub tiled_jumps: usize,
    pub fused_jumps: usize,
    pub jump_tolerance: usize,
    /// sqrt(sum err^2 / sum ref^2) of the gated attention outputs over all sampled forks
    pub tiled_attn_rel_l2: f64,
    pub fused_attn_rel_l2: f64,
    pub pass_ref_p50: bool,
    pub pass_ref_p90: bool,
    pub pass_ref_jumps: bool,
    pub pass_ref_attn: bool,
    pub fused_rms_mean: f64,
    pub tiled_kld_ref_mean: f64,
    pub fused_kld_ref_mean: f64,
    /// share of sampled positions where fused is at least as close to the reference as tiled
    pub fused_le_tiled_share: f64,
    /// continuous fused top-1 != reference top-1 at margin > gate.margin (forked positions)
    pub fused_wrong_top1_high_margin: usize,
    /// the same count for the incumbent (tiled), for comparison
    pub tiled_wrong_top1_high_margin: usize,
    pub fork_replay_exact: bool,
    pub scope_met: bool,
    pub pass_kld: bool,
    pub pass_top1: bool,
    pub pass_wrong_top1: bool,
    pub pass_ref: bool,
    pub passed: bool,
}

fn quant(s: &[f64], f: f64) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    s[((s.len() as f64 - 1.0) * f).round() as usize]
}

impl Report {
    pub fn verdict(&self, g: &Gate) -> Verdict {
        let mut k: Vec<f64> = self.positions.iter().map(|p| p.kld).collect();
        k.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = self.positions.len();
        let kld_mean = if n > 0 { k.iter().sum::<f64>() / n as f64 } else { 0.0 };
        let agree = self.positions.iter().filter(|p| p.tiled_top1 == p.fused_top1).count();
        let top1_agreement = if n > 0 { agree as f64 / n as f64 } else { 0.0 };
        let sampled: Vec<&RefPos> = self.refs.iter().filter(|r| r.sampled).collect();
        let m = sampled.len().max(1) as f64;
        let mean = |f: &dyn Fn(&RefPos) -> f64| sampled.iter().map(|r| f(r)).sum::<f64>() / m;
        let tiled_rms_mean = mean(&|r| r.tiled_rms);
        let fused_rms_mean = mean(&|r| r.fused_rms);
        let le = sampled.iter().filter(|r| r.fused_rms <= r.tiled_rms).count();
        let fused_wrong = self.refs.iter().filter(|r| r.fused_cont_top1 != r.ref_top1 && r.ref_margin > g.margin).count();
        let tiled_wrong = self.refs.iter().filter(|r| r.tiled_top1 != r.ref_top1 && r.ref_margin > g.margin).count();
        let fork_ok = self.refs.iter().all(|r| r.fork_replay_exact);
        let scope_met = n >= g.min_positions && sampled.len() >= g.min_ref_samples;
        let pass_kld = n > 0 && kld_mean <= g.noise_floor_kld;
        let pass_top1 = n > 0 && top1_agreement >= g.min_top1;
        let pass_wrong_top1 = fused_wrong == 0;
        let sorted = |f: &dyn Fn(&RefPos) -> f64| {
            let mut v: Vec<f64> = sampled.iter().map(|r| f(r)).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };
        let (ts, fs) = (sorted(&|r| r.tiled_rms), sorted(&|r| r.fused_rms));
        let (t50, t90, f50, f90) = (quant(&ts, 0.5), quant(&ts, 0.9), quant(&fs, 0.5), quant(&fs, 0.9));
        let tiled_jumps = ts.iter().filter(|&&x| x > g.jump_threshold).count();
        let fused_jumps = fs.iter().filter(|&&x| x > g.jump_threshold).count();
        let jump_tolerance = g.jump_tol_min.max((g.jump_tol_frac * sampled.len() as f64).ceil() as usize);
        let rel = |f: &dyn Fn(&RefPos) -> (f64, f64)| {
            let (e, r) = sampled.iter().fold((0.0, 0.0), |a, x| (a.0 + f(x).0, a.1 + f(x).1));
            if r > 0.0 { (e / r).sqrt() } else { f64::NAN }
        };
        let (ta, fa) = (rel(&|r| r.tiled_attn), rel(&|r| r.fused_attn));
        // an absolute floor so two bit-exact arms (both 0) compare equal
        let within = |f: f64, t: f64| f <= g.ref_ratio_max * t + 1e-12;
        let pass_ref_p50 = !sampled.is_empty() && within(f50, t50);
        let pass_ref_p90 = !sampled.is_empty() && within(f90, t90);
        let pass_ref_jumps = !sampled.is_empty() && fused_jumps <= tiled_jumps + jump_tolerance;
        let pass_ref_attn = ta.is_finite() && fa.is_finite() && within(fa, ta);
        let pass_ref = pass_ref_p50 && pass_ref_p90 && pass_ref_jumps && pass_ref_attn;
        Verdict {
            positions: n,
            kld_mean,
            kld_p50: quant(&k, 0.5),
            kld_p90: quant(&k, 0.9),
            kld_p99: quant(&k, 0.99),
            kld_max: k.last().copied().unwrap_or(0.0),
            top1_agreement,
            ref_samples: sampled.len(),
            ref_forced: self.refs.len() - sampled.len(),
            tiled_rms_mean,
            tiled_rms_p50: t50,
            tiled_rms_p90: t90,
            fused_rms_p50: f50,
            fused_rms_p90: f90,
            tiled_jumps,
            fused_jumps,
            jump_tolerance,
            tiled_attn_rel_l2: ta,
            fused_attn_rel_l2: fa,
            pass_ref_p50,
            pass_ref_p90,
            pass_ref_jumps,
            pass_ref_attn,
            fused_rms_mean,
            tiled_kld_ref_mean: mean(&|r| r.tiled_kld),
            fused_kld_ref_mean: mean(&|r| r.fused_kld),
            fused_le_tiled_share: le as f64 / m,
            fused_wrong_top1_high_margin: fused_wrong,
            tiled_wrong_top1_high_margin: tiled_wrong,
            fork_replay_exact: fork_ok,
            scope_met,
            pass_kld,
            pass_top1,
            pass_wrong_top1,
            pass_ref,
            passed: scope_met && fork_ok && pass_kld && pass_top1 && pass_wrong_top1 && pass_ref,
        }
    }
}

/// Prepare engine A (incumbent) and B (candidate) for [`run`]: tiled attention at every n_kv
/// (cutoff 0, so the reference hook is reachable at any depth), legacy off, fused on B only.
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
}

/// Run (a) and (b). A and B must hold the same prefix state (position `a.pos() == b.pos()`),
/// be configured by [`configure`], and have room for `cont.len()` more positions. Steps
/// `cont[i]` for every i; forks A at `samples` (indices into `cont`) and at every top-1
/// disagreement. `progress(i)` is called after each position.
pub fn run<P: Team + ?Sized>(a: &mut Engine, b: &mut Engine, pool: &P, cont: &[u32], samples: &[usize], progress: &mut dyn FnMut(usize, &Report)) -> Report {
    assert_eq!(a.pos(), b.pos(), "A and B must start from the same state");
    assert!(!a.attn_fused() && b.attn_fused(), "configure() first");
    let base = a.pos();
    let mut rep = Report::default();
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
        let (ta, tb) = (argmax(&la), argmax(&lb));
        rep.positions.push(Pos { position: base + i, kld: kld(&la, &lb), tiled_top1: ta, fused_top1: tb });
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
            rep.refs.push(RefPos {
                index: i,
                position: base + i,
                sampled,
                disagreement: ta != tb,
                ref_top1: rt,
                ref_margin: rm,
                ref_absmax: lr.iter().fold(0f32, |m, x| m.max(x.abs())),
                tiled_top1: ta,
                fused_top1: argmax(&lf),
                fused_cont_top1: tb,
                tiled_rms: rms(&la, &lr),
                fused_rms: rms(&lf, &lr),
                tiled_kld: kld(&lr, &la),
                fused_kld: kld(&lr, &lf),
                fork_replay_exact: exact,
                tiled_attn,
                fused_attn,
            });
        }
        snap = Some(s);
        progress(i, &rep);
    }
    rep
}

/// `--control`: the incumbent's own numerical noise on the same saved state. Engine `a` steps
/// every token through the decode path (production generation); engine `c` takes the same
/// tokens through the batched prefill path in chunks of `w` (production prompt processing, one
/// ubatch per chunk). Both are tiled and teacher-forced from the same state; the chunk-end
/// positions are compared: KL(P_decode || P_prefill-batch) and top-1 (`fused_top1` holds the
/// batched arm's top-1 here). Both engines end at `base + w * (cont.len() / w)`.
pub fn control<P: Team + ?Sized>(a: &Engine, c: &Engine, pool: &P, cont: &[u32], w: usize, progress: &mut dyn FnMut(usize)) -> Vec<Pos> {
    assert!(w >= 1);
    assert_eq!(a.pos(), c.pos(), "A and C must start from the same state");
    assert!(!a.attn_fused() && !c.attn_fused(), "control compares the tiled incumbent with itself");
    let base = a.pos();
    let mut out = Vec::new();
    for (k, chunk) in cont.chunks_exact(w).enumerate() {
        c.prefill(pool, chunk, w);
        let lc = c.logits().to_vec();
        for &t in chunk {
            a.step(pool, t);
        }
        let la = a.logits();
        let i = k * w + w - 1;
        out.push(Pos { position: base + i, kld: kld(la, &lc), tiled_top1: argmax(la), fused_top1: argmax(&lc) });
        progress(i);
    }
    out
}

/// Summary of a control run: (positions, mean, p50, p90, p99, max KLD, top-1 agreement).
pub fn control_summary(v: &[Pos]) -> (usize, f64, f64, f64, f64, f64, f64) {
    let mut k: Vec<f64> = v.iter().map(|p| p.kld).collect();
    k.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = k.len();
    let mean = if n > 0 { k.iter().sum::<f64>() / n as f64 } else { f64::NAN };
    let agree = if n > 0 { v.iter().filter(|p| p.tiled_top1 == p.fused_top1).count() as f64 / n as f64 } else { f64::NAN };
    (n, mean, quant(&k, 0.5), quant(&k, 0.9), quant(&k, 0.99), k.last().copied().unwrap_or(f64::NAN), agree)
}
