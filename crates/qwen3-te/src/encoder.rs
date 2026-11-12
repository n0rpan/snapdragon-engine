//! The klein text-encoder graph: Qwen3 blocks up to the deepest requested layer, CPU attention and
//! norms, linears through a [`Linears`] backend (CPU f32 reference, CPU Q8 emulation of sd.cpp, or
//! oxn NPU in `npu.rs`).

use crate::config::KleinSpec;
use crate::cpu::{self, attention, rms_norm_rows, rope_neox, rope_tables, silu};
use crate::prompt::Encoded;
use crate::weights::{Kind, TeWeights};
use oxn::linear::WeightRows;
use std::time::Instant;

/// Linear-projection backend. `project` computes, for every kind in `kinds`, `outs[i] = x W_kind^T`
/// (`n x o_kind`, row-major) from one shared `n x k` input; callers group the projections that
/// share an input (q/k/v, gate/up) so an accelerator writes the activation once.
pub trait Linears: Sync {
    fn name(&self) -> String;
    fn project(&self, layer: usize, kinds: &[Kind], x: &[f32], n: usize, outs: &mut [Vec<f32>]) -> Result<(), String>;
    /// Backend counters for the JSON summary.
    fn summary(&self) -> serde_json::Value {
        serde_json::Value::Null
    }
}

/// CPU arithmetic of a linear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuMode {
    /// f32 dot products on dequantized weights (the reference)
    F32,
    /// Q8_0 x Q8_0 integer blocks, as sd.cpp's CPU backend computes them (identity bug detector)
    Q8,
}

pub struct CpuLinears<'w, 'a> {
    pub w: &'w TeWeights<'a>,
    pub threads: usize,
    pub mode: CpuMode,
}

impl Linears for CpuLinears<'_, '_> {
    fn name(&self) -> String {
        format!("cpu-{}", if self.mode == CpuMode::F32 { "f32" } else { "q8" })
    }
    fn project(&self, layer: usize, kinds: &[Kind], x: &[f32], n: usize, outs: &mut [Vec<f32>]) -> Result<(), String> {
        for (i, &kd) in kinds.iter().enumerate() {
            let m = self.w.layers[layer].mat(kd);
            outs[i].resize(n * m.o(), 0.0);
            match self.mode {
                CpuMode::F32 => cpu::gemm_f32(m, x, n, &mut outs[i], self.threads)?,
                CpuMode::Q8 => cpu::gemm_q8(m, x, n, &mut outs[i], self.threads)?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub embed_s: f64,
    pub norm_rope_s: f64,
    pub linear_s: f64,
    pub attn_s: f64,
    pub total_s: f64,
    pub layers: usize,
}

impl Timings {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "embed_s": self.embed_s, "norm_rope_s": self.norm_rope_s, "linear_s": self.linear_s,
            "attn_s": self.attn_s, "total_s": self.total_s, "layers": self.layers,
        })
    }
}

pub struct Encoder<'w, 'a> {
    pub w: &'w TeWeights<'a>,
    pub spec: KleinSpec,
    pub threads: usize,
}

impl<'w, 'a> Encoder<'w, 'a> {
    pub fn new(w: &'w TeWeights<'a>, spec: KleinSpec, threads: usize) -> Result<Self, String> {
        spec.check(&w.cfg)?;
        if w.layers.len() < spec.layers_needed() {
            return Err(format!("encoder: {} blocks loaded, the spec needs {}", w.layers.len(), spec.layers_needed()));
        }
        Ok(Encoder { w, spec, threads: threads.max(1) })
    }

    /// `[n][out_layers.len() * hidden]` (layer-major within a token row), `n = e.tokens.len()`.
    pub fn encode(&self, lin: &dyn Linears, e: &Encoded) -> Result<(Vec<f32>, Timings), String> {
        let c = &self.w.cfg;
        let (n, hdim, hd) = (e.tokens.len(), c.hidden, c.head_dim);
        if n == 0 || e.mask.len() != n || !e.mask[0] {
            return Err("encoder: empty input or first token masked".into());
        }
        let t_all = Instant::now();
        let mut tm = Timings { layers: self.spec.layers_needed(), ..Default::default() };
        let t0 = Instant::now();
        let mut x = vec![0f32; n * hdim];
        for (t, &id) in e.tokens.iter().enumerate() {
            if id as usize >= c.vocab {
                return Err(format!("encoder: token id {id} outside the vocabulary ({})", c.vocab));
            }
            self.w.embd.read_row(id as usize, &mut x[t * hdim..(t + 1) * hdim])?;
        }
        tm.embed_s = t0.elapsed().as_secs_f64();
        let (cos, sin) = rope_tables(n, hd, c.rope_theta);
        let nout = self.spec.out_layers.len();
        let ctx = nout * hdim;
        let mut out = vec![0f32; n * ctx];
        let mut h = vec![0f32; n * hdim];
        let mut qkv = vec![Vec::new(), Vec::new(), Vec::new()];
        let mut att = vec![0f32; n * c.q_dim()];
        let mut one = vec![Vec::new()];
        let mut gu = vec![Vec::new(), Vec::new()];
        for l in 0..self.spec.layers_needed() {
            let lw = &self.w.layers[l];
            // attention
            let t = Instant::now();
            h.copy_from_slice(&x);
            rms_norm_rows(&mut h, &lw.attn_norm, c.eps);
            tm.norm_rope_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            lin.project(l, &[Kind::Q, Kind::K, Kind::V], &h, n, &mut qkv)?;
            tm.linear_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            {
                let (q, rest) = qkv.split_at_mut(1);
                let (q, k) = (&mut q[0], &mut rest[0]);
                rms_norm_rows(q, &lw.q_norm, c.eps);
                rms_norm_rows(k, &lw.k_norm, c.eps);
                rope_neox(q, n, c.n_heads, hd, &cos, &sin);
                rope_neox(k, n, c.n_kv_heads, hd, &cos, &sin);
            }
            tm.norm_rope_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            attention(&qkv[0], &qkv[1], &qkv[2], &e.mask, n, c.n_heads, c.n_kv_heads, hd, &mut att, self.threads);
            tm.attn_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            lin.project(l, &[Kind::O], &att, n, &mut one)?;
            tm.linear_s += t.elapsed().as_secs_f64();
            x.iter_mut().zip(&one[0]).for_each(|(a, b)| *a += b);
            // feed-forward
            let t = Instant::now();
            h.copy_from_slice(&x);
            rms_norm_rows(&mut h, &lw.ffn_norm, c.eps);
            tm.norm_rope_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            lin.project(l, &[Kind::Gate, Kind::Up], &h, n, &mut gu)?;
            tm.linear_s += t.elapsed().as_secs_f64();
            {
                let (g, u) = gu.split_at_mut(1);
                g[0].iter_mut().zip(&u[0]).for_each(|(a, b)| *a = silu(*a) * b);
            }
            let t = Instant::now();
            lin.project(l, &[Kind::Down], &gu[0], n, &mut one)?;
            tm.linear_s += t.elapsed().as_secs_f64();
            x.iter_mut().zip(&one[0]).for_each(|(a, b)| *a += b);
            if let Some(slot) = self.spec.out_layers.iter().position(|&o| o == l + 1) {
                for t in 0..n {
                    out[t * ctx + slot * hdim..t * ctx + (slot + 1) * hdim].copy_from_slice(&x[t * hdim..(t + 1) * hdim]);
                }
            }
            if x.iter().any(|v| !v.is_finite()) {
                return Err(format!("encoder: non-finite residual after block {l}"));
            }
        }
        tm.total_s = t_all.elapsed().as_secs_f64();
        Ok((out, tm))
    }
}
