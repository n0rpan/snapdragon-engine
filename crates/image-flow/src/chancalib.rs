//! Static channel calibration for coarse int8 scale chunks (F94 / F-IMG-256, v0.1.3 lane H).
//!
//! [`crate::actstats`] showed WHY chunk 256 misses the precision gate: one max-based scale per token
//! per 256 channels is set by a few large channels, so the other 255 lose resolution (scale
//! resolution, not summation: every chunk's integer sum is exact). This module turns that analysis
//! into a calibration TOOL: it accumulates per-channel magnitudes of a block's linear-input over
//! calibration rows (several prompts / steps / blocks), proposes a static fix per operand, checks the
//! proposal on HELD-OUT rows, and writes a calibration record the fast path can later apply at pack
//! time. Two fixes, both static (no run-time search):
//!
//! - `sorted`: a column permutation by typical magnitude, so each chunk holds channels of similar
//!   size. Exact for the weights apart from their own per-chunk re-quantisation, which the fast path
//!   already does at chunk != 32; zero run-time cost (the operand writer stores channel `perm[i]`
//!   at position `i`).
//! - `equalized(alpha)`: SmoothQuant-style per-channel division `x_k / s_k` with
//!   `s_k = rms_k^alpha` (normalised to geometric mean 1), the weights' column `k` multiplied by
//!   `s_k`. The activation side is measured here; the WEIGHT side error (re-quantising scaled columns)
//!   needs the real weights and is the hardware step.
//!
//! Selection rule (frozen here so the hardware step only measures): per operand, the error target
//! is the natural arrangement at chunk 32 (the setting that passes today's block gate) on the same
//! held-out rows. Prefer `sorted` if its held-out balanced error is within `slack` of the target;
//! otherwise the smallest-alpha `equalized` that is; otherwise the best of all, marked
//! `meets_target: false` (then chunk 256 stays off for that operand and the record says so).
//!
//! Error measure = the `balanced` relative error of [`crate::actstats`] (each channel divided by its
//! magnitude, so small channels count as much as large ones: the better proxy for `W x`).

use serde_json::{json, Value};

/// Schema of the calibration record.
pub const SCHEMA: &str = "oxi-chan-calib/1";
/// Alphas tried for `equalized`, smallest (closest to no scaling) first.
pub const ALPHAS: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// Per-channel statistics of one operand over calibration rows.
#[derive(Clone, Debug)]
pub struct ChanAccum {
    pub h: usize,
    pub rows: u64,
    sumsq: Vec<f64>,
    absmax: Vec<f32>,
}

impl ChanAccum {
    pub fn new(h: usize) -> ChanAccum {
        ChanAccum { h, rows: 0, sumsq: vec![0.0; h], absmax: vec![0.0; h] }
    }

    /// Add `n` rows of width `h`.
    pub fn add(&mut self, x: &[f32], n: usize) -> Result<(), String> {
        if x.len() != n * self.h {
            return Err(format!("chan-calib: {} values for {n} rows of {}", x.len(), self.h));
        }
        for row in x.chunks_exact(self.h) {
            for k in 0..self.h {
                let v = row[k];
                if !v.is_finite() {
                    return Err("chan-calib: non-finite calibration value".into());
                }
                self.sumsq[k] += (v as f64) * (v as f64);
                self.absmax[k] = self.absmax[k].max(v.abs());
            }
        }
        self.rows += n as u64;
        Ok(())
    }

    /// Root mean square per channel (0 for an unseen or all-zero channel).
    pub fn rms(&self) -> Vec<f32> {
        self.sumsq.iter().map(|s| (s / self.rows.max(1) as f64).sqrt() as f32).collect()
    }

    pub fn absmax(&self) -> &[f32] {
        &self.absmax
    }
}

/// Magnitude-sorted permutation (ascending, ties by channel index: deterministic).
pub fn sorted_perm(mag: &[f32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..mag.len() as u32).collect();
    order.sort_by(|&a, &b| mag[a as usize].partial_cmp(&mag[b as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    order
}

/// Equalisation divisors `s_k = mag_k^alpha`, normalised to geometric mean 1 over the non-zero
/// channels; zero channels get 1.
pub fn eq_scale(mag: &[f32], alpha: f32) -> Vec<f32> {
    let logs: Vec<f64> = mag.iter().filter(|m| **m > 0.0).map(|m| (*m as f64).ln() * alpha as f64).collect();
    let mean = if logs.is_empty() { 0.0 } else { logs.iter().sum::<f64>() / logs.len() as f64 };
    mag.iter().map(|&m| if m > 0.0 { ((m as f64).ln() * alpha as f64 - mean).exp() as f32 } else { 1.0 }).collect()
}

/// Balanced (and plain) relative int8 error of `n` rows with one max-based scale per token per
/// `chunk` positions, channels taken in `order` and divided by `div` before quantisation (and
/// multiplied back after). `bal` is the per-channel magnitude of the balanced measure.
#[allow(clippy::too_many_arguments)]
pub fn quant_error_with(x: &[f32], n: usize, h: usize, order: &[u32], div: &[f32], bal: &[f32], chunk: usize) -> (f64, f64) {
    assert!(chunk > 0 && h % chunk == 0 && order.len() == h && div.len() == h && bal.len() == h && x.len() == n * h);
    let (mut e_plain, mut s_plain, mut e_bal, mut s_bal) = (0f64, 0f64, 0f64, 0f64);
    let mut buf = vec![0f32; chunk];
    for t in 0..n {
        let row = &x[t * h..(t + 1) * h];
        for c in 0..h / chunk {
            let idx = &order[c * chunk..(c + 1) * chunk];
            let mut amax = 0f32;
            for (b, &k) in buf.iter_mut().zip(idx) {
                *b = row[k as usize] / div[k as usize];
                amax = amax.max(b.abs());
            }
            let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for (b, &k) in buf.iter().zip(idx) {
                let k = k as usize;
                let q = (b * inv).round().clamp(-127.0, 127.0) * s * div[k];
                let d = (q - row[k]) as f64;
                e_plain += d * d;
                s_plain += (row[k] as f64).powi(2);
                if bal[k] > 0.0 {
                    let m = bal[k] as f64;
                    e_bal += (d / m).powi(2);
                    s_bal += (row[k] as f64 / m).powi(2);
                }
            }
        }
    }
    ((e_plain / s_plain.max(f64::MIN_POSITIVE)).sqrt(), (e_bal / s_bal.max(f64::MIN_POSITIVE)).sqrt())
}

/// The proposal for one operand.
#[derive(Clone, Debug)]
pub struct OperandPlan {
    pub name: String,
    pub h: usize,
    pub chunk: usize,
    /// "natural" | "sorted" | "equalized"
    pub arrange: &'static str,
    pub alpha: Option<f32>,
    pub meets_target: bool,
    pub perm: Option<Vec<u32>>,
    pub scale: Option<Vec<f32>>,
    /// held-out balanced errors: target (natural at chunk 32), natural, sorted, equalized per alpha
    pub errors: Value,
    pub calib_rows: u64,
    pub holdout_rows: usize,
}

impl OperandPlan {
    pub fn json(&self) -> Value {
        json!({
            "name": self.name, "h": self.h, "chunk": self.chunk, "arrange": self.arrange, "alpha": self.alpha,
            "meets_target": self.meets_target, "calib_rows": self.calib_rows, "holdout_rows": self.holdout_rows,
            "errors_balanced_holdout": self.errors, "perm": self.perm, "scale": self.scale,
        })
    }
}

/// Plan one operand: magnitudes from `calib`, decision on the held-out rows `hold` (`n_hold` rows).
pub fn plan_operand(name: &str, calib: &ChanAccum, hold: &[f32], n_hold: usize, chunk: usize, slack: f64) -> Result<OperandPlan, String> {
    let h = calib.h;
    if chunk == 0 || chunk % 32 != 0 || h % chunk != 0 || h % 32 != 0 {
        return Err(format!("chan-calib: {name}: chunk {chunk} must be a multiple of 32 dividing {h}"));
    }
    if n_hold == 0 || hold.len() != n_hold * h || calib.rows == 0 {
        return Err(format!("chan-calib: {name}: needs calibration rows and held-out rows"));
    }
    let mag = calib.rms();
    // the balanced measure uses the held-out rows' own magnitudes (independent of the plan)
    let mut hacc = ChanAccum::new(h);
    hacc.add(hold, n_hold)?;
    let bal = hacc.rms();
    let natural: Vec<u32> = (0..h as u32).collect();
    let ones = vec![1f32; h];
    let target = quant_error_with(hold, n_hold, h, &natural, &ones, &bal, 32).1;
    let nat = quant_error_with(hold, n_hold, h, &natural, &ones, &bal, chunk).1;
    let perm = sorted_perm(&mag);
    let sorted = quant_error_with(hold, n_hold, h, &perm, &ones, &bal, chunk).1;
    let mut eqs = Vec::new();
    for a in ALPHAS {
        let s = eq_scale(&mag, a);
        let e = quant_error_with(hold, n_hold, h, &natural, &s, &bal, chunk).1;
        eqs.push((a, s, e));
    }
    let limit = target * (1.0 + slack);
    let errors = json!({
        "target_natural_chunk32": target, "natural": nat, "sorted": sorted,
        "equalized": eqs.iter().map(|(a, _, e)| json!({"alpha": a, "error": e})).collect::<Vec<_>>(),
        "limit": limit,
    });
    let base = OperandPlan {
        name: name.to_string(), h, chunk, arrange: "natural", alpha: None, meets_target: nat <= limit, perm: None, scale: None,
        errors, calib_rows: calib.rows, holdout_rows: n_hold,
    };
    if nat <= limit {
        return Ok(base);
    }
    if sorted <= limit {
        return Ok(OperandPlan { arrange: "sorted", meets_target: true, perm: Some(perm), ..base });
    }
    if let Some((a, s, _)) = eqs.iter().find(|(_, _, e)| *e <= limit) {
        return Ok(OperandPlan { arrange: "equalized", alpha: Some(*a), meets_target: true, scale: Some(s.clone()), ..base });
    }
    // nothing meets the target: keep the best one, flagged (chunk stays off for this operand)
    let (best_a, best_s, best_e) = eqs.iter().min_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal)).expect("alphas");
    if sorted <= *best_e && sorted < nat {
        Ok(OperandPlan { arrange: "sorted", perm: Some(perm), ..base })
    } else if *best_e < nat {
        Ok(OperandPlan { arrange: "equalized", alpha: Some(*best_a), scale: Some(best_s.clone()), ..base })
    } else {
        Ok(base)
    }
}

/// The calibration record (`--out`): every operand's plan plus provenance (no paths, no prompts).
pub fn record(chunk: usize, slack: f64, plans: &[OperandPlan], sources: &[String]) -> Value {
    let met = plans.iter().filter(|p| p.meets_target).count();
    json!({
        "schema": SCHEMA,
        "chunk": chunk,
        "slack": slack,
        "selection_rule": "target = natural arrangement at chunk 32 on held-out rows; prefer sorted, then smallest-alpha equalized, within (1+slack)*target",
        "weight_side": "not measured (equalized changes the weights' per-chunk quantisation): hardware step",
        "operands_meeting_target": met,
        "operands": plans.iter().map(|p| p.json()).collect::<Vec<_>>(),
        "sources_sha256": sources,
    })
}

/// Read back a permutation / scale from a record operand (validated: a real permutation of `0..h`,
/// finite positive scales), for the pack-time application and for tests.
pub fn operand_transform(op: &Value) -> Result<(usize, Option<Vec<u32>>, Option<Vec<f32>>), String> {
    let h = op["h"].as_u64().ok_or("operand without h")? as usize;
    let perm = match op["perm"].as_array() {
        None => None,
        Some(a) => {
            let p: Vec<u32> = a.iter().map(|v| v.as_u64().map(|x| x as u32).ok_or("perm entry")).collect::<Result<_, _>>()?;
            let mut seen = vec![false; h];
            if p.len() != h || p.iter().any(|&i| (i as usize) >= h || std::mem::replace(&mut seen[i as usize], true)) {
                return Err("perm is not a permutation of 0..h".into());
            }
            Some(p)
        }
    };
    let scale = match op["scale"].as_array() {
        None => None,
        Some(a) => {
            let s: Vec<f32> = a.iter().map(|v| v.as_f64().map(|x| x as f32).ok_or("scale entry")).collect::<Result<_, _>>()?;
            if s.len() != h || s.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                return Err("scale must be h finite positive values".into());
            }
            Some(s)
        }
    };
    Ok((h, perm, scale))
}

#[cfg(test)]
#[path = "chancalib_tests.rs"]
mod tests;
