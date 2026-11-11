//! W5-G drift localisation: component probes of the Rust reference against sd.cpp dumps, in f32
//! and in sd.cpp arithmetic parity ([`Arith::SdParity`]).
//!
//! Every probe feeds one component sd.cpp's OWN dumped input and compares with sd.cpp's dumped
//! output, so the numbers are local (no error carried in from earlier blocks). Scopes:
//! - `component` (one block, offline lane): time/modulation vectors vs `dump.single.mod.*`
//!   (+ `dump.flux.vec`), token embedders vs `dump.flux.{img,txt}_emb`, and single block 0 on
//!   `dump.single.0.in` vs `dump.single.1.in`, with the per-stage taps `dump.s0.*` when present.
//! - `blocks` (whole model, one block at a time; Codex run): additionally every double block on
//!   `dump.double.{i-1}.*_out` (block 0 on the embeddings), every single block on
//!   `dump.single.{i}.in`, and the final layer on `dump.final.in` vs `dump.flux.out`.
//!   Tags beyond the 0002 probe patch come from `0003-w5g-stage-taps.patch` (sd.cpp side).
//!
//! Block errors are reported on the block's UPDATE (`out - in`), the quantity that carries the
//! block's own arithmetic; `out` itself is dominated by the residual. Double blocks use the Rust
//! modulations of the same arithmetic (sd.cpp does not dump them); single blocks use sd.cpp's
//! dumped single-stream modulation.

use crate::full_ref::{RefModel, TimeVectors};
use crate::handoff::{read_tag_call, Handoff};
use image_dit::arith::Arith;
use image_dit::types::{rel_rmse, BlockInput};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Component,
    Blocks,
}

impl Scope {
    pub fn parse(s: &str) -> Result<Scope, String> {
        match s {
            "component" => Ok(Scope::Component),
            "blocks" => Ok(Scope::Blocks),
            other => Err(format!("--scope must be component or blocks, got '{other}'")),
        }
    }
}

/// One dumped tensor of sd.cpp's step `call` as f32, if present.
fn dump(dir: &Path, tag: &str, call: usize) -> Result<Option<Vec<f32>>, String> {
    Ok(read_tag_call(dir, tag, call)?.map(|t| t.data))
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// `{rel_rmse, delta_rel_rmse}` of a block output against sd.cpp's, both relative to the input.
fn block_err(got: &[f32], want: &[f32], input: &[f32]) -> Value {
    json!({
        "delta_rel_rmse": rel_rmse(&sub(got, input), &sub(want, input)),
        "out_rel_rmse": rel_rmse(got, want),
        "update_over_input_rms": (sumsq(&sub(want, input)) / sumsq(input).max(1e-30)).sqrt(),
    })
}

fn sumsq(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64 * x as f64).sum()
}

/// Options of the `blocks` scope.
#[derive(Clone, Copy, Debug, Default)]
pub struct BlocksOpts {
    /// Self-noise control: also run every block on its input with each value moved by up to this
    /// many ulps (deterministic pseudo-random sign/size) and report how far the block's own update
    /// moves (`self_noise`). 0 = off. A block whose sd.cpp error is of the order of its self-noise
    /// is at the arithmetic's noise floor; an error far above it points to a real op difference.
    pub perturb_ulp: u32,
    /// Stop after this many single blocks (None = all).
    pub max_single: Option<usize>,
}

/// Moves each value by up to `ulp` ulps (xorshift, fixed seed).
pub fn perturb(x: &[f32], ulp: u32) -> Vec<f32> {
    let mut st = 0x9e37_79b9_7f4a_7c15u64;
    x.iter()
        .map(|&v| {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            let k = (st % (2 * ulp as u64 + 1)) as i64 - ulp as i64;
            if v == 0.0 || !v.is_finite() {
                v
            } else {
                f32::from_bits((v.to_bits() as i64 + k) as u32)
            }
        })
        .collect()
}

/// Runs the probes for each arithmetic in `ariths` on step `step` of the hand-over in `dir`.
pub fn run(model: &mut RefModel, h: &Handoff, dir: &Path, step: usize, scope: Scope, ariths: &[Arith]) -> Result<Value, String> {
    run_with(model, h, dir, step, scope, ariths, BlocksOpts::default())
}

/// [`run`] with the `blocks` scope options.
pub fn run_with(
    model: &mut RefModel,
    h: &Handoff,
    dir: &Path,
    step: usize,
    scope: Scope,
    ariths: &[Arith],
    bo: BlocksOpts,
) -> Result<Value, String> {
    let d = model.dims();
    let hd = d.block.hidden;
    let (n_txt, n_img) = (h.n_txt, h.n_img);
    let n = n_txt + n_img;
    let sigma = h.sigma_pair(step).0;
    let mut notes: Vec<String> = Vec::new();

    // sd.cpp side, loaded once
    let sd_shift = dump(dir, "dump.single.mod.shift", step)?;
    let sd_scale = dump(dir, "dump.single.mod.scale", step)?;
    let sd_gate = dump(dir, "dump.single.mod.gate", step)?;
    let sd_vec = dump(dir, "dump.flux.vec", step)?;
    let sd_img_emb = dump(dir, "dump.flux.img_emb", step)?;
    let sd_txt_emb = dump(dir, "dump.flux.txt_emb", step)?;
    let s0_in = h.single_in_at(0, step).ok_or("dump.single.0.in missing for this step")?.to_vec();
    let s1_in = h.single_in_at(1, step).ok_or("dump.single.1.in missing for this step")?.to_vec();
    let stage_tags = ["xmod", "q", "k", "v", "attn", "mlp", "z"];
    let mut sd_stage: Vec<(&str, Option<Vec<f32>>)> = Vec::new();
    for t in stage_tags {
        sd_stage.push((t, dump(dir, &format!("dump.s0.{t}"), step)?));
    }
    if sd_stage.iter().all(|(_, v)| v.is_none()) {
        notes.push("no dump.s0.* stage taps (sd.cpp built without 0003-w5g-stage-taps.patch): stage rows compare f32 vs parity only".into());
    }

    let mut per_arith: Vec<Value> = Vec::new();
    // stage vectors of the first arithmetic, to measure what the dtype choice alone moves
    let mut first_stages: Option<Vec<(&str, Vec<f32>)>> = None;
    for &arith in ariths {
        model.set_arith(arith);
        let t0 = Instant::now();
        let mut r = json!({"arith": arith.name()});

        // A. time embedding and modulations
        let tv = model.time_vectors(sigma);
        let mut a = json!({});
        if let (Some(s), Some(c), Some(g)) = (&sd_shift, &sd_scale, &sd_gate) {
            a["single_shift"] = json!(rel_rmse(&tv.mod_single[..hd], s));
            a["single_scale"] = json!(rel_rmse(&tv.mod_single[hd..2 * hd], c));
            a["single_gate"] = json!(rel_rmse(&tv.mod_single[2 * hd..], g));
        }
        if let Some(v) = &sd_vec {
            a["vec"] = json!(rel_rmse(&tv.vec, v));
        }
        r["time_vectors_vs_sdcpp"] = a;

        // B. token embedders
        if let (Some(ie), Some(te)) = (&sd_img_emb, &sd_txt_emb) {
            let (ri, rt) = model.embed_tokens(h.img_at(step), n_img, &h.txt, n_txt);
            r["embed_vs_sdcpp"] = json!({"img_in": rel_rmse(&ri, ie), "txt_in": rel_rmse(&rt, te)});
        }

        // C. single block 0 on sd.cpp's input and modulation, per stage
        let (shift, scale, gate) = match (&sd_shift, &sd_scale, &sd_gate) {
            (Some(s), Some(c), Some(g)) => (s.clone(), c.clone(), g.clone()),
            _ => {
                notes.push("single block 0 uses the Rust modulation (dump.single.mod.* absent)".into());
                (tv.mod_single[..hd].to_vec(), tv.mod_single[hd..2 * hd].to_vec(), tv.mod_single[2 * hd..].to_vec())
            }
        };
        let inp = BlockInput { n_tokens: n, x: s0_in.clone(), pe: h.pe.clone(), shift, scale, gate };
        let tb = Instant::now();
        let (o, st) = model.single_block(0, &inp)?;
        let block_s = tb.elapsed().as_secs_f64();
        let ours: Vec<(&str, Vec<f32>)> = vec![
            ("xmod", st.x_mod),
            ("q", st.q),
            ("k", st.k),
            ("v", st.v),
            ("attn", o.attn),
            ("mlp", o.mlp),
            ("z", o.z),
            ("delta", sub(&o.out, &s0_in)),
        ];
        let mut stages = json!({});
        for (name, v) in &ours {
            let mut row = json!({});
            if let Some((_, Some(want))) = sd_stage.iter().find(|(t, _)| t == name) {
                row["vs_sdcpp"] = json!(rel_rmse(v, want));
            }
            if *name == "delta" {
                row["vs_sdcpp"] = json!(rel_rmse(v, &sub(&s1_in, &s0_in)));
            }
            if let Some(first) = &first_stages {
                if let Some((_, f)) = first.iter().find(|(t, _)| t == name) {
                    row[format!("vs_{}", ariths[0].name())] = json!(rel_rmse(v, f));
                }
            }
            stages[*name] = row;
        }
        r["single0"] = json!({
            "block": block_err(&o.out, &s1_in, &s0_in),
            "stages": stages,
            "seconds": block_s,
        });
        if first_stages.is_none() {
            first_stages = Some(ours);
        }

        // D. every block, teacher-forced on sd.cpp's own inputs (Codex scope)
        if scope == Scope::Blocks {
            r["blocks"] = blocks_probe(model, h, dir, step, &tv, bo, &mut notes)?;
        }
        r["seconds"] = json!(t0.elapsed().as_secs_f64());
        per_arith.push(r);
    }
    model.set_arith(Arith::F32);
    notes.sort();
    notes.dedup();
    Ok(json!({
        "probe": "w5g-drift",
        "step": step,
        "sigma": sigma,
        "scope": format!("{scope:?}").to_lowercase(),
        "n_txt": n_txt,
        "n_img": n_img,
        "metric": "relative RMSE; block rows compare the block update (out - in) and the full output",
        "results": per_arith,
        "notes": notes,
    }))
}

/// Every double and single block and the final layer on sd.cpp's own inputs, one at a time.
fn blocks_probe(
    model: &RefModel,
    h: &Handoff,
    dir: &Path,
    step: usize,
    tv: &TimeVectors,
    bo: BlocksOpts,
    notes: &mut Vec<String>,
) -> Result<Value, String> {
    let d = model.dims();
    let hd = d.block.hidden;
    let (n_txt, n) = (h.n_txt, h.n_txt + h.n_img);
    let mut rows: Vec<Value> = Vec::new();

    // double blocks
    let mut prev: Option<(Vec<f32>, Vec<f32>)> = match (dump(dir, "dump.flux.img_emb", step)?, dump(dir, "dump.flux.txt_emb", step)?) {
        (Some(i), Some(t)) => Some((i, t)),
        _ => None,
    };
    for i in 0..d.n_double {
        let want = match (dump(dir, &format!("dump.double.{i}.img_out"), step)?, dump(dir, &format!("dump.double.{i}.txt_out"), step)?) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        };
        match (prev.take(), &want) {
            (Some((img, txt)), Some((wi, wt))) => {
                let t = Instant::now();
                let (gi, gt) = model.double_block_at(i, img.clone(), txt.clone(), tv, &h.pe)?;
                let mut row = json!({
                    "block": format!("double.{i}"),
                    "img": block_err(&gi, wi, &img),
                    "txt": block_err(&gt, wt, &txt),
                });
                if bo.perturb_ulp > 0 {
                    let (pimg, ptxt) = (perturb(&img, bo.perturb_ulp), perturb(&txt, bo.perturb_ulp));
                    let (qi, qt) = model.double_block_at(i, pimg.clone(), ptxt.clone(), tv, &h.pe)?;
                    row["self_noise"] = json!({
                        "img_delta_rel_rmse": rel_rmse(&sub(&qi, &pimg), &sub(&gi, &img)),
                        "txt_delta_rel_rmse": rel_rmse(&sub(&qt, &ptxt), &sub(&gt, &txt)),
                    });
                }
                row["seconds"] = json!(t.elapsed().as_secs_f64());
                rows.push(row);
            }
            _ => notes.push(format!("double.{i}: sd.cpp input or output dump missing, skipped")),
        }
        prev = want;
    }

    // single blocks
    let modv = |tag: &str| dump(dir, tag, step);
    let (shift, scale, gate) = match (modv("dump.single.mod.shift")?, modv("dump.single.mod.scale")?, modv("dump.single.mod.gate")?) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => (tv.mod_single[..hd].to_vec(), tv.mod_single[hd..2 * hd].to_vec(), tv.mod_single[2 * hd..].to_vec()),
    };
    let final_in = dump(dir, "dump.final.in", step)?;
    let n_single = bo.max_single.map_or(d.n_single, |m| m.min(d.n_single));
    for i in 0..n_single {
        let x = dump(dir, &format!("dump.single.{i}.in"), step)?;
        let next = if i + 1 < d.n_single { dump(dir, &format!("dump.single.{}.in", i + 1), step)? } else { None };
        let Some(x) = x else {
            notes.push(format!("single.{i}: dump.single.{i}.in missing, skipped"));
            continue;
        };
        let inp = BlockInput { n_tokens: n, x, pe: h.pe.clone(), shift: shift.clone(), scale: scale.clone(), gate: gate.clone() };
        let t = Instant::now();
        let row = match (&next, &final_in) {
            (Some(want), _) => {
                let (o, _) = model.single_block(i, &inp)?;
                let mut r = json!({"block": format!("single.{i}"), "all": block_err(&o.out, want, &inp.x)});
                if bo.perturb_ulp > 0 {
                    let px = perturb(&inp.x, bo.perturb_ulp);
                    let pinp = BlockInput {
                        n_tokens: inp.n_tokens,
                        x: px.clone(),
                        pe: inp.pe.clone(),
                        shift: inp.shift.clone(),
                        scale: inp.scale.clone(),
                        gate: inp.gate.clone(),
                    };
                    let (po, _) = model.single_block(i, &pinp)?;
                    r["self_noise"] = json!({"delta_rel_rmse": rel_rmse(&sub(&po.out, &px), &sub(&o.out, &inp.x))});
                }
                r
            }
            (None, Some(fin)) if i + 1 == d.n_single => {
                // last block: sd.cpp dumps only the image tokens entering the final layer
                let (o, _) = model.single_block(i, &inp)?;
                let img_part = &o.out[n_txt * hd..];
                json!({"block": format!("single.{i}"), "img_only": block_err(img_part, fin, &inp.x[n_txt * hd..])})
            }
            _ => {
                notes.push(format!("single.{i}: no sd.cpp output dump, skipped"));
                continue;
            }
        };
        let mut row = row;
        row["seconds"] = json!(t.elapsed().as_secs_f64());
        rows.push(row);
    }

    // final layer
    if let (Some(fin), Some(out)) = (&final_in, h.out_at(step)) {
        let y = model.final_layer(fin, tv);
        rows.push(json!({"block": "final", "out_rel_rmse": rel_rmse(&y, out)}));
    } else {
        notes.push("final: dump.final.in or dump.flux.out missing, skipped".into());
    }
    Ok(json!(rows))
}
