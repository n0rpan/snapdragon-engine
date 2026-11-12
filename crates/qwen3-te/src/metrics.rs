//! Hidden-state agreement between two `[n][segments * width]` context tensors (native vs reference).
//!
//! Reported per output layer (segment) and per row class (real prompt tokens, pad tokens, all):
//! cosine of the flattened values, relative RMSE `||a - b|| / ||b||` (b = reference), max |a - b|,
//! and the worst per-row cosine. The gate in TEXT-ENCODER-NPU-DESIGN.md reads `real` and `all`.

use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, Default)]
pub struct Agree {
    pub cos: f64,
    pub rel_rmse: f64,
    pub max_abs: f64,
    pub worst_row_cos: f64,
    pub rows: usize,
}

impl Agree {
    pub fn json(&self) -> Value {
        json!({"cos": self.cos, "rel_rmse": self.rel_rmse, "max_abs": self.max_abs,
               "worst_row_cos": self.worst_row_cos, "rows": self.rows})
    }
}

/// Agreement over `rows` of segment `seg` (`None` = all segments).
pub fn agree(a: &[f32], b: &[f32], width: usize, nseg: usize, seg: Option<usize>, rows: &[usize]) -> Agree {
    let ctx = width * nseg;
    let (c0, c1) = match seg {
        Some(s) => (s * width, (s + 1) * width),
        None => (0, ctx),
    };
    let (mut ab, mut aa, mut bb, mut dd, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut worst = 1f64;
    for &r in rows {
        let (mut rab, mut raa, mut rbb) = (0f64, 0f64, 0f64);
        for c in c0..c1 {
            let (x, y) = (a[r * ctx + c] as f64, b[r * ctx + c] as f64);
            rab += x * y;
            raa += x * x;
            rbb += y * y;
            dd += (x - y) * (x - y);
            mx = mx.max((x - y).abs());
        }
        ab += rab;
        aa += raa;
        bb += rbb;
        let rc = if raa > 0.0 && rbb > 0.0 { rab / (raa.sqrt() * rbb.sqrt()) } else if raa == rbb { 1.0 } else { 0.0 };
        worst = worst.min(rc);
    }
    let cos = if aa > 0.0 && bb > 0.0 { ab / (aa.sqrt() * bb.sqrt()) } else if aa == bb { 1.0 } else { 0.0 };
    let rel = if bb > 0.0 { (dd / bb).sqrt() } else if dd == 0.0 { 0.0 } else { f64::INFINITY };
    Agree { cos, rel_rmse: rel, max_abs: mx, worst_row_cos: worst, rows: rows.len() }
}

/// Full report: `{"layers": [..], "segments": [{"layer", "real", "pad", "all"}], "real", "all"}`.
pub fn report(a: &[f32], b: &[f32], n: usize, n_real: usize, width: usize, layers: &[usize]) -> Result<Value, String> {
    let nseg = layers.len();
    if a.len() != n * width * nseg || b.len() != a.len() {
        return Err(format!("compare: {} vs {} values for n {n} x {} x {width}", a.len(), b.len(), nseg));
    }
    let real: Vec<usize> = (0..n_real.min(n)).collect();
    let pad: Vec<usize> = (n_real.min(n)..n).collect();
    let all: Vec<usize> = (0..n).collect();
    let segs: Vec<Value> = (0..nseg)
        .map(|s| {
            json!({"layer": layers[s],
                   "real": agree(a, b, width, nseg, Some(s), &real).json(),
                   "pad": agree(a, b, width, nseg, Some(s), &pad).json(),
                   "all": agree(a, b, width, nseg, Some(s), &all).json()})
        })
        .collect();
    Ok(json!({
        "n": n, "n_real": n_real, "layers": layers, "segments": segs,
        "real": agree(a, b, width, nseg, None, &real).json(),
        "all": agree(a, b, width, nseg, None, &all).json(),
    }))
}

/// The hidden-state gate of the design doc on a [`report`]: every segment's `real` and `all` rows
/// need `cos >= min_cos` and `rel_rmse <= max_rel`.
pub fn gate(rep: &Value, min_cos: f64, max_rel: f64) -> (bool, Vec<String>) {
    let mut why = Vec::new();
    for s in rep["segments"].as_array().into_iter().flatten() {
        for class in ["real", "all"] {
            let (c, r) = (s[class]["cos"].as_f64().unwrap_or(0.0), s[class]["rel_rmse"].as_f64().unwrap_or(f64::INFINITY));
            if s[class]["rows"].as_u64().unwrap_or(0) == 0 {
                continue;
            }
            if !(c >= min_cos) || !(r <= max_rel) {
                why.push(format!("layer {} {class}: cos {c:.6} rel_rmse {r:.4e}", s["layer"]));
            }
        }
    }
    (why.is_empty(), why)
}

/// O-46 form of the gate: the candidate (`cand`, e.g. NPU vs the f32 reference) must be at least as
/// close to the exact reference as the installed path (`inst`, sd.cpp's context vs the same reference),
/// per segment and row class: `rel_rmse(cand) <= slack * rel_rmse(inst)` and `cos(cand) >= cos(inst)`
/// up to the same slack on `1 - cos`.
pub fn gate_vs_installed(cand: &Value, inst: &Value, slack: f64) -> (bool, Vec<String>) {
    let mut why = Vec::new();
    let (cs, is) = (cand["segments"].as_array(), inst["segments"].as_array());
    let (Some(cs), Some(is)) = (cs, is) else {
        return (false, vec!["missing segments".into()]);
    };
    if cs.len() != is.len() {
        return (false, vec!["segment count differs".into()]);
    }
    for (c, i) in cs.iter().zip(is) {
        for class in ["real", "all"] {
            let f = |v: &Value, k: &str| v[class][k].as_f64().unwrap_or(f64::NAN);
            let (cr, ir) = (f(c, "rel_rmse"), f(i, "rel_rmse"));
            let (cc, ic) = (f(c, "cos"), f(i, "cos"));
            if !(cr <= slack * ir) || !(1.0 - cc <= slack * (1.0 - ic)) {
                why.push(format!(
                    "layer {} {class}: candidate rel {cr:.4e} cos {cc:.6} vs installed rel {ir:.4e} cos {ic:.6}",
                    c["layer"]
                ));
            }
        }
    }
    (why.is_empty(), why)
}
