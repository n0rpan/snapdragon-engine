//! Hybrid DiT forward: every block linear (the 80 Q8_0 tensors of klein) runs as a device publish,
//! everything else runs on the CPU in f32.
//!
//! The CPU formulas are ported from `image_flow::full_ref` (the whole-model f32 reference) and
//! `image_dit::reference` (single block). Source lines are cited next to each port. Only the
//! linears differ from the reference: the device computes them from the same dequantised Q8_0
//! weights, so `HybridModel` on a mock device should match `image_flow::RefModel` to f32 rounding.
//!
//! Two schedules, same math:
//! - `Merged`: one publish per dependency stage (4 per double block, 2 per single block).
//!   Independent streams share a publish. 60 publishes per step for klein.
//! - `StreamSplit`: one publish per stream and stage (8 per double block, 2 per single). CPU work
//!   of one stream runs while the other stream's publish is in flight. 80 publishes per step.

use std::ops::Range;

use image_dit::fast_model::WeightSource;
use image_dit::model::{DitBackend, ModelInput};

use crate::backend::{Handle, LinearCall, LinearDevice, RunnerStats, WindowedRunner};
use crate::plan::Window;

/// `image_flow::full_ref::TIME_DIM`, `TIME_MAX_PERIOD`, `TIME_FACTOR`, `EPS`.
pub const TIME_DIM: usize = 256;
pub const TIME_MAX_PERIOD: f32 = 10000.0;
pub const TIME_FACTOR: f32 = 1000.0;
const EPS: f32 = 1e-6;

/// Model shapes (same meaning as `image_flow::ModelDims`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HybridDims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp: usize,
    pub in_ch: usize,
    pub ctx_dim: usize,
    pub out_ch: usize,
    pub n_double: usize,
    pub n_single: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Merged,
    StreamSplit,
}

struct Embed {
    img_in: Vec<f32>,
    txt_in: Vec<f32>,
    time_in0: Vec<f32>,
    time_in2: Vec<f32>,
    mod_img: Vec<f32>,
    mod_txt: Vec<f32>,
    mod_single: Vec<f32>,
    final_adaln: Vec<f32>,
    final_linear: Vec<f32>,
}

struct Norms {
    q: Vec<f32>,
    k: Vec<f32>,
}

struct DoubleNorms {
    img: Norms,
    txt: Norms,
}

/// Names of one stream's device linears in one double block.
struct StreamNames {
    qkv: String,
    proj: String,
    mlp0: String,
    mlp2: String,
}

fn stream_names(i: usize, s: &str) -> StreamNames {
    StreamNames {
        qkv: format!("double_blocks.{i}.{s}_attn.qkv.weight"),
        proj: format!("double_blocks.{i}.{s}_attn.proj.weight"),
        mlp0: format!("double_blocks.{i}.{s}_mlp.0.weight"),
        mlp2: format!("double_blocks.{i}.{s}_mlp.2.weight"),
    }
}

/// The whole DiT with block linears on a device. Name: `hybrid-npu-linear`.
pub struct HybridModel<'a, D: LinearDevice> {
    dims: HybridDims,
    embed: Embed,
    double: Vec<DoubleNorms>,
    single: Vec<Norms>,
    runner: WindowedRunner<'a, D>,
    schedule: Schedule,
    trace: Option<Vec<String>>,
}

impl<'a, D: LinearDevice> HybridModel<'a, D> {
    /// `plan` must cover the block linears of `src` (see `plan::block_linear_order` or
    /// `plan::klein_linear_order`). Errors on shape mismatches.
    pub fn new(
        src: &'a dyn WeightSource,
        dims: HybridDims,
        plan: Vec<Window>,
        dev: D,
        schedule: Schedule,
    ) -> Result<Self, String> {
        let h = dims.hidden;
        if dims.heads * dims.head_dim != h || dims.head_dim % 2 != 0 {
            return Err("bad attention shape".into());
        }
        let embed = Embed {
            img_in: mat(src, "img_in.weight", h, dims.in_ch)?,
            txt_in: mat(src, "txt_in.weight", h, dims.ctx_dim)?,
            time_in0: mat(src, "time_in.in_layer.weight", h, TIME_DIM)?,
            time_in2: mat(src, "time_in.out_layer.weight", h, h)?,
            mod_img: mat(src, "double_stream_modulation_img.lin.weight", 6 * h, h)?,
            mod_txt: mat(src, "double_stream_modulation_txt.lin.weight", 6 * h, h)?,
            mod_single: mat(src, "single_stream_modulation.lin.weight", 3 * h, h)?,
            final_adaln: mat(src, "final_layer.adaLN_modulation.1.weight", 2 * h, h)?,
            final_linear: mat(src, "final_layer.linear.weight", dims.out_ch, h)?,
        };
        let mut double = Vec::with_capacity(dims.n_double);
        for i in 0..dims.n_double {
            let norms = |s: &str| -> Result<Norms, String> {
                Ok(Norms {
                    q: vecn(src, &format!("double_blocks.{i}.{s}_attn.norm.query_norm.scale"), dims.head_dim)?,
                    k: vecn(src, &format!("double_blocks.{i}.{s}_attn.norm.key_norm.scale"), dims.head_dim)?,
                })
            };
            double.push(DoubleNorms { img: norms("img")?, txt: norms("txt")? });
        }
        let mut single = Vec::with_capacity(dims.n_single);
        for i in 0..dims.n_single {
            single.push(Norms {
                q: vecn(src, &format!("single_blocks.{i}.norm.query_norm.scale"), dims.head_dim)?,
                k: vecn(src, &format!("single_blocks.{i}.norm.key_norm.scale"), dims.head_dim)?,
            });
        }
        let runner = WindowedRunner::new(plan, src, dev)?;
        Ok(HybridModel { dims, embed, double, single, runner, schedule, trace: None })
    }

    pub fn stats(&self) -> RunnerStats {
        self.runner.stats()
    }

    pub fn device(&self) -> &D {
        self.runner.device()
    }

    /// Starts recording submit/wait/cpu markers (for the overlap checks).
    pub fn enable_trace(&mut self) {
        self.trace = Some(Vec::new());
    }

    pub fn take_trace(&mut self) -> Vec<String> {
        self.trace.as_mut().map(std::mem::take).unwrap_or_default()
    }

    fn mark(&mut self, s: String) {
        if let Some(t) = self.trace.as_mut() {
            t.push(s);
        }
    }

    fn sub(&mut self, name: &str, calls: &[LinearCall]) -> Result<Handle, String> {
        self.mark(format!("submit:{name}"));
        self.runner.submit(calls)
    }

    fn wai(&mut self, name: &str, h: Handle) -> Result<Vec<Vec<f32>>, String> {
        self.mark(format!("wait:{name}"));
        self.runner.wait(h)
    }

    fn cpu(&mut self, name: &str) {
        self.mark(format!("cpu:{name}"));
    }

    /// One denoising forward. Device errors are returned, not panicked.
    pub fn try_forward(&mut self, input: &ModelInput) -> Result<Vec<f32>, String> {
        let d = self.dims;
        let h = d.hidden;
        let half = d.head_dim / 2;
        let (n_img, n_txt) = (input.n_img, input.n_txt);
        let n = n_txt + n_img;
        if input.img.len() != n_img * d.in_ch || input.txt.len() != n_txt * d.ctx_dim || input.pe.len() != n * half * 4 {
            return Err("input shapes do not match the model".into());
        }
        let split = self.schedule == Schedule::StreamSplit;

        // embedders and shared modulations (CPU)
        let mut img = linear_rows(&self.embed.img_in, d.in_ch, h, &input.img, n_img);
        let mut txt = linear_rows(&self.embed.txt_in, d.ctx_dim, h, &input.txt, n_txt);
        let temb = timestep_embedding(input.timestep, TIME_DIM, TIME_MAX_PERIOD, TIME_FACTOR);
        let mut t1 = linear_rows(&self.embed.time_in0, TIME_DIM, h, &temb, 1);
        for v in t1.iter_mut() {
            *v = silu(*v);
        }
        let vec = linear_rows(&self.embed.time_in2, h, h, &t1, 1);
        let svec: Vec<f32> = vec.iter().map(|&v| silu(v)).collect();
        let mi = mods6(&linear_rows(&self.embed.mod_img, h, 6 * h, &svec, 1), h);
        let mt = mods6(&linear_rows(&self.embed.mod_txt, h, 6 * h, &svec, 1), h);
        let ms = mods3(&linear_rows(&self.embed.mod_single, h, 3 * h, &svec, 1), h);
        let mfin = linear_rows(&self.embed.final_adaln, h, 2 * h, &svec, 1);
        self.cpu("embed");

        for i in 0..d.n_double {
            let (ni, nt) = if split {
                self.double_split(i, img, txt, &mi, &mt, &input.pe, n_txt, n_img)?
            } else {
                self.double_merged(i, img, txt, &mi, &mt, &input.pe, n_txt, n_img)?
            };
            img = ni;
            txt = nt;
        }

        let mut x = [txt, img].concat();
        for i in 0..d.n_single {
            x = self.single(i, x, n, &ms, &input.pe)?;
        }

        // final layer on image tokens (CPU), `img_tok` = x[n_txt*h..]
        let y = ln_rows(&x[n_txt * h..], n_img, h, &mfin[..h], &mfin[h..]);
        Ok(linear_rows(&self.embed.final_linear, h, d.out_ch, &y, n_img))
    }

    /// Double block, merged schedule: publishes A {qkv}, B {proj}, C {mlp0}, D {mlp2}, each with
    /// both streams.
    #[allow(clippy::too_many_arguments)]
    fn double_merged(
        &mut self,
        i: usize,
        img: Vec<f32>,
        txt: Vec<f32>,
        mi: &Mods6,
        mt: &Mods6,
        pe: &[f32],
        n_txt: usize,
        n_img: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let d = self.dims;
        let (h, hd, m) = (d.hidden, d.head_dim, d.mlp);
        let half = hd / 2;
        let n = n_txt + n_img;
        let (nt, ni) = (stream_names(i, "txt"), stream_names(i, "img"));
        let mut txt = txt;
        let mut img = img;

        let xt = ln_rows(&txt, n_txt, h, &mt.shift1, &mt.scale1);
        let xi = ln_rows(&img, n_img, h, &mi.shift1, &mi.scale1);
        let ha = self.sub("A", &[call(&nt.qkv, n_txt, xt), call(&ni.qkv, n_img, xi)])?;
        let ya = self.wai("A", ha)?;
        let dn = &self.double[i];
        let (qt, kt, vt) = post_qkv(&d, &ya[0], n_txt, &dn.txt.q, &dn.txt.k, &pe[..n_txt * half * 4]);
        let (qi, ki, vi) = post_qkv(&d, &ya[1], n_img, &dn.img.q, &dn.img.k, &pe[n_txt * half * 4..]);
        self.cpu("A.post");
        let q = [qt, qi].concat();
        let k = [kt, ki].concat();
        let v = [vt, vi].concat();
        let attn = attention_f32_rows(&d, &q, &k, &v, n, 0..n);
        self.cpu("attention");
        let (attn_t, attn_i) = attn.split_at(n_txt * h);

        let hb = self.sub("B", &[call(&nt.proj, n_txt, attn_t.to_vec()), call(&ni.proj, n_img, attn_i.to_vec())])?;
        let pb = self.wai("B", hb)?;
        residual(&mut txt, &mt.gate1, &pb[0], h);
        residual(&mut img, &mi.gate1, &pb[1], h);
        let xt = ln_rows(&txt, n_txt, h, &mt.shift2, &mt.scale2);
        let xi = ln_rows(&img, n_img, h, &mi.shift2, &mi.scale2);
        self.cpu("B.post");

        let hc = self.sub("C", &[call(&nt.mlp0, n_txt, xt), call(&ni.mlp0, n_img, xi)])?;
        let mc = self.wai("C", hc)?;
        let at = swiglu(&mc[0], n_txt, m);
        let ai = swiglu(&mc[1], n_img, m);
        self.cpu("C.post");

        let hd_ = self.sub("D", &[call(&nt.mlp2, n_txt, at), call(&ni.mlp2, n_img, ai)])?;
        let md = self.wai("D", hd_)?;
        residual(&mut txt, &mt.gate2, &md[0], h);
        residual(&mut img, &mi.gate2, &md[1], h);
        self.cpu("D.post");
        Ok((img, txt))
    }

    /// Double block, stream-split schedule. Publishes per stream; CPU work of one stream runs while
    /// the other stream's publish is in flight. Same math as `double_merged` (rows are independent).
    #[allow(clippy::too_many_arguments)]
    fn double_split(
        &mut self,
        i: usize,
        img: Vec<f32>,
        txt: Vec<f32>,
        mi: &Mods6,
        mt: &Mods6,
        pe: &[f32],
        n_txt: usize,
        n_img: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let d = self.dims;
        let (h, hd, m) = (d.hidden, d.head_dim, d.mlp);
        let half = hd / 2;
        let n = n_txt + n_img;
        let (nt, ni) = (stream_names(i, "txt"), stream_names(i, "img"));
        let mut txt = txt;
        let mut img = img;

        let xt = ln_rows(&txt, n_txt, h, &mt.shift1, &mt.scale1);
        let xi = ln_rows(&img, n_img, h, &mi.shift1, &mi.scale1);
        let ha_t = self.sub("A.txt", &[call(&nt.qkv, n_txt, xt)])?;
        let ha_i = self.sub("A.img", &[call(&ni.qkv, n_img, xi)])?;
        let ya_t = self.wai("A.txt", ha_t)?;
        let dn = &self.double[i];
        let (qt, kt, vt) = post_qkv(&d, &ya_t[0], n_txt, &dn.txt.q, &dn.txt.k, &pe[..n_txt * half * 4]);
        self.cpu("A.txt.post"); // overlaps the img qkv publish
        let ya_i = self.wai("A.img", ha_i)?;
        let dn = &self.double[i];
        let (qi, ki, vi) = post_qkv(&d, &ya_i[0], n_img, &dn.img.q, &dn.img.k, &pe[n_txt * half * 4..]);
        self.cpu("A.img.post");

        let q = [qt, qi].concat();
        let k = [kt, ki].concat();
        let v = [vt, vi].concat();
        let attn_t = attention_f32_rows(&d, &q, &k, &v, n, 0..n_txt);
        let hb_t = self.sub("B.txt", &[call(&nt.proj, n_txt, attn_t)])?;
        let attn_i = attention_f32_rows(&d, &q, &k, &v, n, n_txt..n);
        self.cpu("attention.img"); // overlaps the txt proj publish

        let pt = self.wai("B.txt", hb_t)?;
        residual(&mut txt, &mt.gate1, &pt[0], h);
        let xt2 = ln_rows(&txt, n_txt, h, &mt.shift2, &mt.scale2);
        let hc_t = self.sub("C.txt", &[call(&nt.mlp0, n_txt, xt2)])?;
        let hb_i = self.sub("B.img", &[call(&ni.proj, n_img, attn_i)])?;
        let pi = self.wai("B.img", hb_i)?;
        residual(&mut img, &mi.gate1, &pi[0], h);
        let xi2 = ln_rows(&img, n_img, h, &mi.shift2, &mi.scale2);
        let hc_i = self.sub("C.img", &[call(&ni.mlp0, n_img, xi2)])?;

        let mct = self.wai("C.txt", hc_t)?;
        let at = swiglu(&mct[0], n_txt, m);
        self.cpu("swiglu.txt"); // overlaps the img mlp0 publish
        let hd_t = self.sub("D.txt", &[call(&nt.mlp2, n_txt, at)])?;

        let mci = self.wai("C.img", hc_i)?;
        let ai = swiglu(&mci[0], n_img, m);
        self.cpu("swiglu.img"); // overlaps the txt mlp2 publish
        let hd_i = self.sub("D.img", &[call(&ni.mlp2, n_img, ai)])?;

        let mdt = self.wai("D.txt", hd_t)?;
        residual(&mut txt, &mt.gate2, &mdt[0], h);
        self.cpu("residual.txt"); // overlaps the img mlp2 publish
        let mdi = self.wai("D.img", hd_i)?;
        residual(&mut img, &mi.gate2, &mdi[0], h);
        Ok((img, txt))
    }

    /// Single block: publish {linear1} (all tokens), CPU attention and gate, publish {linear2}.
    fn single(&mut self, i: usize, x: Vec<f32>, n: usize, ms: &Mods3, pe: &[f32]) -> Result<Vec<f32>, String> {
        let d = self.dims;
        let (h, hd, m) = (d.hidden, d.head_dim, d.mlp);
        let half = hd / 2;
        let lin1 = format!("single_blocks.{i}.linear1.weight");
        let lin2 = format!("single_blocks.{i}.linear2.weight");
        let xm = ln_rows(&x, n, h, &ms.shift, &ms.scale);
        let h1 = self.sub("S1", &[call(&lin1, n, xm)])?;
        let y = self.wai("S1", h1)?.remove(0);
        let row = 3 * h + 2 * m;
        let nrm = &self.single[i];
        let mut q = vec![0f32; n * h];
        let mut k = vec![0f32; n * h];
        let mut v = vec![0f32; n * h];
        let mut mm = vec![0f32; n * m];
        for t in 0..n {
            let yr = &y[t * row..(t + 1) * row];
            let pe_t = &pe[t * half * 4..(t + 1) * half * 4];
            let qr = &mut q[t * h..(t + 1) * h];
            qr.copy_from_slice(&yr[..h]);
            let kr = &mut k[t * h..(t + 1) * h];
            kr.copy_from_slice(&yr[h..2 * h]);
            for hh in 0..d.heads {
                let (a, b) = (hh * hd, (hh + 1) * hd);
                rms_norm_scale(&mut qr[a..b], &nrm.q);
                rope(&mut qr[a..b], pe_t);
                rms_norm_scale(&mut kr[a..b], &nrm.k);
                rope(&mut kr[a..b], pe_t);
            }
            v[t * h..(t + 1) * h].copy_from_slice(&yr[2 * h..3 * h]);
            for j in 0..m {
                mm[t * m + j] = silu(yr[3 * h + j]) * yr[3 * h + m + j];
            }
        }
        self.cpu("S.post");
        let attn = attention_f64_rows(&d, &q, &k, &v, n, 0..n);
        let kin = h + m;
        let mut cat = vec![0f32; n * kin];
        for t in 0..n {
            cat[t * kin..t * kin + h].copy_from_slice(&attn[t * h..(t + 1) * h]);
            cat[t * kin + h..(t + 1) * kin].copy_from_slice(&mm[t * m..(t + 1) * m]);
        }
        self.cpu("S.attention");
        let h2 = self.sub("S2", &[call(&lin2, n, cat)])?;
        let z = self.wai("S2", h2)?.remove(0);
        let mut out = x;
        residual(&mut out, &ms.gate, &z, h);
        self.cpu("S.post2");
        Ok(out)
    }
}

impl<D: LinearDevice> DitBackend for HybridModel<'_, D> {
    fn name(&self) -> &'static str {
        "hybrid-npu-linear"
    }

    /// Panics on device errors; use `try_forward` to get them.
    fn forward(&mut self, input: &ModelInput) -> Vec<f32> {
        self.try_forward(input).expect("hybrid forward")
    }
}

fn call(name: &str, n_tokens: usize, input: Vec<f32>) -> LinearCall {
    LinearCall { tensor: name.to_string(), n_tokens, input }
}

// ---------------------------------------------------------------------------------------------
// loading (shape-checked, like full_ref::get_f32)
// ---------------------------------------------------------------------------------------------

fn mat(src: &dyn WeightSource, name: &str, rows: usize, k: usize) -> Result<Vec<f32>, String> {
    let (v, r, kk) = src.f32(name)?;
    if r != rows || kk != k || v.len() != rows * k {
        return Err(format!("{name}: shape [{kk}, {r}], want [{k}, {rows}]"));
    }
    Ok(v)
}

fn vecn(src: &dyn WeightSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let (v, r, k) = src.f32(name)?;
    if r != 1 || k != len || v.len() != len {
        return Err(format!("{name}: shape ({r}, {k}), want {len}"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// CPU math, ported from image_flow::full_ref (cited per function) and image_dit::reference
// ---------------------------------------------------------------------------------------------

struct Mods6 {
    shift1: Vec<f32>,
    scale1: Vec<f32>,
    gate1: Vec<f32>,
    shift2: Vec<f32>,
    scale2: Vec<f32>,
    gate2: Vec<f32>,
}

/// `full_ref::Mods6::from_vec` (chunk order of sd.cpp `Modulation::forward`).
fn mods6(v: &[f32], h: usize) -> Mods6 {
    let c = |k: usize| v[k * h..(k + 1) * h].to_vec();
    Mods6 { shift1: c(0), scale1: c(1), gate1: c(2), shift2: c(3), scale2: c(4), gate2: c(5) }
}

struct Mods3 {
    shift: Vec<f32>,
    scale: Vec<f32>,
    gate: Vec<f32>,
}

/// `full_ref::Mods3::from_vec`.
fn mods3(v: &[f32], h: usize) -> Mods3 {
    let c = |k: usize| v[k * h..(k + 1) * h].to_vec();
    Mods3 { shift: c(0), scale: c(1), gate: c(2) }
}

/// `full_ref::linear_rows`: `y[t][o] = sum_i w[o][i] x[t][i]`, `w` is `[out][in]`.
fn linear_rows(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32], rows: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out_dim];
    for t in 0..rows {
        let xr = &x[t * in_dim..(t + 1) * in_dim];
        for r in 0..out_dim {
            y[t * out_dim + r] = dot8(&w[r * in_dim..(r + 1) * in_dim], xr);
        }
    }
    y
}

/// `full_ref::dot`: 8 independent lanes.
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let main = n - n % 8;
    let mut acc = [0f32; 8];
    for (ca, cb) in a[..main].chunks_exact(8).zip(b[..main].chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += ca[l] * cb[l];
        }
    }
    for (l, (x, y)) in a[main..].iter().zip(&b[main..]).enumerate() {
        acc[l] += x * y;
    }
    (acc[0] + acc[1] + acc[2] + acc[3]) + (acc[4] + acc[5] + acc[6] + acc[7])
}

/// LayerNorm + modulate for `rows` tokens of width `width`.
fn ln_rows(x: &[f32], rows: usize, width: usize, shift: &[f32], scale: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; rows * width];
    for t in 0..rows {
        ln_modulate(&x[t * width..(t + 1) * width], shift, scale, &mut out[t * width..(t + 1) * width]);
    }
    out
}

/// `full_ref::ln_modulate`: `(y + y*scale) + shift` after LayerNorm without affine; mean and
/// variance in f64. (`reference::layer_norm_modulate` is the same formula.)
fn ln_modulate(x: &[f32], shift: &[f32], scale: &[f32], out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let denom = (var + EPS as f64).sqrt();
    for (((o, &xv), &sc), &sh) in out.iter_mut().zip(x).zip(scale).zip(shift) {
        let y = ((xv as f64 - mean) / denom) as f32;
        *o = (y + y * sc) + sh;
    }
}

/// `full_ref::rms_norm_scale`: `x / sqrt(mean(x^2) + eps) * scale`, over one head.
fn rms_norm_scale(x: &mut [f32], scale: &[f32]) {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let denom = (ms + EPS as f64).sqrt();
    for (v, &sc) in x.iter_mut().zip(scale) {
        *v = ((*v as f64 / denom) as f32) * sc;
    }
}

/// `full_ref::rope`: interleaved RoPE, `pe` = `[cos, -sin, sin, cos]` per pair.
fn rope(x: &mut [f32], pe: &[f32]) {
    for (pair, p) in x.chunks_exact_mut(2).zip(pe.chunks_exact(4)) {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = p[0] * a + p[1] * b;
        pair[1] = p[2] * a + p[3] * b;
    }
}

/// `full_ref::silu`.
fn silu(a: f32) -> f32 {
    a / (1.0 + (-a).exp())
}

/// `full_ref::timestep_embedding` (`ggml_ext_timestep_embedding`).
fn timestep_embedding(t: f32, dim: usize, max_period: f32, time_factor: f32) -> Vec<f32> {
    let tt = t * time_factor;
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for j in 0..half {
        let freq = (-(max_period.ln()) * j as f32 / half as f32).exp();
        let arg = tt * freq;
        out[j] = arg.cos();
        out[j + half] = arg.sin();
    }
    out
}

/// `full_ref::stream_qkv` after the qkv linear: split rows into q, k, v; per head QK RMS norm and
/// RoPE. `y` is `[n][3h]`.
fn post_qkv(d: &HybridDims, y: &[f32], n: usize, qs: &[f32], ks: &[f32], pe: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (h, hd) = (d.hidden, d.head_dim);
    let half = hd / 2;
    let mut q = vec![0f32; n * h];
    let mut k = vec![0f32; n * h];
    let mut v = vec![0f32; n * h];
    for t in 0..n {
        let yr = &y[t * 3 * h..(t + 1) * 3 * h];
        q[t * h..(t + 1) * h].copy_from_slice(&yr[..h]);
        k[t * h..(t + 1) * h].copy_from_slice(&yr[h..2 * h]);
        v[t * h..(t + 1) * h].copy_from_slice(&yr[2 * h..]);
        let pe_t = &pe[t * half * 4..(t + 1) * half * 4];
        for hh in 0..d.heads {
            let r = t * h + hh * hd..t * h + (hh + 1) * hd;
            rms_norm_scale(&mut q[r.clone()], qs);
            rope(&mut q[r.clone()], pe_t);
            rms_norm_scale(&mut k[r.clone()], ks);
            rope(&mut k[r], pe_t);
        }
    }
    (q, k, v)
}

/// `x += gate * p` per token row (`full_ref::stream_post`, `reference::lin2_stage`).
fn residual(x: &mut [f32], gate: &[f32], p: &[f32], h: usize) {
    for (t, row) in x.chunks_exact_mut(h).enumerate() {
        for r in 0..h {
            row[r] += gate[r] * p[t * h + r];
        }
    }
}

/// `ggml_swiglu`: `silu(first half) * second half` of each `[2*mlp]` row.
fn swiglu(m0: &[f32], n: usize, mlp: usize) -> Vec<f32> {
    let mut out = vec![0f32; n * mlp];
    for t in 0..n {
        let row = &m0[t * 2 * mlp..(t + 1) * 2 * mlp];
        for j in 0..mlp {
            out[t * mlp + j] = silu(row[j]) * row[mlp + j];
        }
    }
    out
}

/// Joint attention in f32 for query rows `rows` over all `n` tokens (`full_ref::attention`).
fn attention_f32_rows(d: &HybridDims, q: &[f32], k: &[f32], v: &[f32], n: usize, rows: Range<usize>) -> Vec<f32> {
    let (h, hd) = (d.hidden, d.head_dim);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; rows.len() * h];
    let mut logits = vec![0f32; n];
    for (ri, t) in rows.enumerate() {
        for hh in 0..d.heads {
            let base = hh * hd;
            let qv = &q[t * h + base..t * h + base + hd];
            let mut mx = f32::NEG_INFINITY;
            for tk in 0..n {
                let l = dot8(qv, &k[tk * h + base..tk * h + base + hd]) * scale;
                logits[tk] = l;
                mx = mx.max(l);
            }
            let mut sum = 0f32;
            for l in logits.iter_mut() {
                *l = (*l - mx).exp();
                sum += *l;
            }
            let inv = 1.0 / sum;
            let o = &mut out[ri * h + base..ri * h + base + hd];
            for tk in 0..n {
                let p = logits[tk] * inv;
                let vv = &v[tk * h + base..tk * h + base + hd];
                for dd in 0..hd {
                    o[dd] += p * vv[dd];
                }
            }
        }
    }
    out
}

/// Single-block attention with f64 softmax and accumulation (`reference::attn_stage`).
fn attention_f64_rows(d: &HybridDims, q: &[f32], k: &[f32], v: &[f32], n: usize, rows: Range<usize>) -> Vec<f32> {
    let (h, hd) = (d.hidden, d.head_dim);
    let sqrt_hd = (hd as f32).sqrt();
    let mut out = vec![0f32; rows.len() * h];
    let mut logits = vec![0f64; n];
    let mut acc = vec![0f64; hd];
    for (ri, tq) in rows.enumerate() {
        for hh in 0..d.heads {
            let base = hh * hd;
            let qv = &q[tq * h + base..tq * h + base + hd];
            let mut mx = f64::NEG_INFINITY;
            for tk in 0..n {
                let l = (dot8(qv, &k[tk * h + base..tk * h + base + hd]) / sqrt_hd) as f64;
                logits[tk] = l;
                mx = mx.max(l);
            }
            let mut sum = 0f64;
            for l in logits.iter_mut() {
                *l = (*l - mx).exp();
                sum += *l;
            }
            acc.iter_mut().for_each(|a| *a = 0.0);
            for tk in 0..n {
                let p = logits[tk] / sum;
                let vv = &v[tk * h + base..tk * h + base + hd];
                for dd in 0..hd {
                    acc[dd] += p * vv[dd] as f64;
                }
            }
            let orow = &mut out[ri * h + base..ri * h + base + hd];
            for dd in 0..hd {
                orow[dd] = acc[dd] as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MockDevice;
    use crate::plan::{assign_slots, block_linear_order, plan_windows, steady_reloads_per_step, stream_split_publishes, ActReserve, PlanConfig};
    use image_dit::fast_model::MemSource;
    use image_dit::model::DitBackend;
    use image_dit::types::rel_rmse;
    use image_dit::weights::quant_q8_0;
    use image_flow::RefModel;

    struct Rng(u64);

    impl Rng {
        /// Uniform in [-1, 1).
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
        }
        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale).collect()
        }
    }

    /// The brief's tiny klein: hidden 64, heads 2, head_dim 32, mlp 192, in 16, ctx 48, out 16,
    /// 1 double + 2 single blocks.
    fn tiny_dims() -> HybridDims {
        HybridDims { hidden: 64, heads: 2, head_dim: 32, mlp: 192, in_ch: 16, ctx_dim: 48, out_ch: 16, n_double: 1, n_single: 2 }
    }

    fn put_f(src: &mut MemSource, name: &str, v: Vec<f32>, rows: usize, k: usize) {
        src.f.insert(name.to_string(), (v, rows, k));
    }

    fn put_q(src: &mut MemSource, name: &str, r: &mut Rng, rows: usize, k: usize) {
        let v = r.vec(rows * k, 1.0 / (k as f32).sqrt());
        src.q8.insert(name.to_string(), (quant_q8_0(&v, rows, k), rows, k));
    }

    fn put_qg(src: &mut MemSource, name: &str, r: &mut Rng, rows: usize, k: usize, g: f32) {
        let v: Vec<f32> = r.vec(rows * k, 1.0 / (k as f32).sqrt()).iter().map(|x| x * g).collect();
        src.q8.insert(name.to_string(), (quant_q8_0(&v, rows, k), rows, k));
    }

    fn put_norm(src: &mut MemSource, name: &str, r: &mut Rng, len: usize) {
        let v: Vec<f32> = (0..len).map(|_| 1.0 + 0.3 * r.next()).collect();
        put_f(src, name, v, 1, len);
    }

    /// The tiny model of `image_flow::full_ref_tests::tiny_source` (same recipe and gain 8).
    fn tiny_source(c: HybridDims, seed: u64) -> MemSource {
        let g = 8.0f32;
        let mut r = Rng(seed.wrapping_mul(2654435761).wrapping_add(12345));
        let mut src = MemSource::default();
        let (h, m, hd) = (c.hidden, c.mlp, c.head_dim);
        let fi = |k: usize| 1.0 / (k as f32).sqrt();
        let v = r.vec(h * c.in_ch, fi(c.in_ch));
        put_f(&mut src, "img_in.weight", v, h, c.in_ch);
        let v = r.vec(h * c.ctx_dim, fi(c.ctx_dim));
        put_f(&mut src, "txt_in.weight", v, h, c.ctx_dim);
        let v = r.vec(h * TIME_DIM, fi(TIME_DIM));
        put_f(&mut src, "time_in.in_layer.weight", v, h, TIME_DIM);
        let v = r.vec(h * h, fi(h));
        put_f(&mut src, "time_in.out_layer.weight", v, h, h);
        for name in ["double_stream_modulation_img.lin.weight", "double_stream_modulation_txt.lin.weight"] {
            let v: Vec<f32> = r.vec(6 * h * h, fi(h)).iter().map(|x| x * g).collect();
            put_f(&mut src, name, v, 6 * h, h);
        }
        let v: Vec<f32> = r.vec(3 * h * h, fi(h)).iter().map(|x| x * g).collect();
        put_f(&mut src, "single_stream_modulation.lin.weight", v, 3 * h, h);
        let v: Vec<f32> = r.vec(2 * h * h, fi(h)).iter().map(|x| x * g).collect();
        put_f(&mut src, "final_layer.adaLN_modulation.1.weight", v, 2 * h, h);
        let v = r.vec(c.out_ch * h, fi(h));
        put_f(&mut src, "final_layer.linear.weight", v, c.out_ch, h);
        for i in 0..c.n_double {
            for s in ["img", "txt"] {
                put_q(&mut src, &format!("double_blocks.{i}.{s}_attn.qkv.weight"), &mut r, 3 * h, h);
                put_qg(&mut src, &format!("double_blocks.{i}.{s}_attn.proj.weight"), &mut r, h, h, g);
                put_q(&mut src, &format!("double_blocks.{i}.{s}_mlp.0.weight"), &mut r, 2 * m, h);
                put_qg(&mut src, &format!("double_blocks.{i}.{s}_mlp.2.weight"), &mut r, h, m, g);
                put_norm(&mut src, &format!("double_blocks.{i}.{s}_attn.norm.query_norm.scale"), &mut r, hd);
                put_norm(&mut src, &format!("double_blocks.{i}.{s}_attn.norm.key_norm.scale"), &mut r, hd);
            }
        }
        for i in 0..c.n_single {
            put_q(&mut src, &format!("single_blocks.{i}.linear1.weight"), &mut r, 3 * h + 2 * m, h);
            put_qg(&mut src, &format!("single_blocks.{i}.linear2.weight"), &mut r, h, h + m, g);
            put_norm(&mut src, &format!("single_blocks.{i}.norm.query_norm.scale"), &mut r, hd);
            put_norm(&mut src, &format!("single_blocks.{i}.norm.key_norm.scale"), &mut r, hd);
        }
        src
    }

    fn random_pe(r: &mut Rng, n: usize, half: usize) -> Vec<f32> {
        let mut pe = Vec::with_capacity(n * half * 4);
        for _ in 0..n * half {
            let a = 3.0 * r.next();
            let (s, c) = a.sin_cos();
            pe.extend_from_slice(&[c, -s, s, c]);
        }
        pe
    }

    fn input(c: HybridDims, n_txt: usize, n_img: usize, seed: u64, timestep: f32) -> ModelInput {
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let n = n_txt + n_img;
        ModelInput {
            n_img,
            n_txt,
            img: r.vec(n_img * c.in_ch, 1.0),
            txt: r.vec(n_txt * c.ctx_dim, 1.0),
            timestep,
            pe: random_pe(&mut r, n, c.head_dim / 2),
        }
    }

    const N_TXT: usize = 4;
    const N_IMG: usize = 8;

    fn plan_with_budget(c: HybridDims, budget: u64) -> Vec<Window> {
        let linears = block_linear_order(c.n_double, c.n_single, c.hidden, c.mlp);
        let cfg = PlanConfig {
            budget_bytes: budget,
            overhead_bytes: 0,
            n_tokens: N_TXT + N_IMG,
            act_reserve: ActReserve::PeakPerLinear,
            ..PlanConfig::klein_default(N_TXT + N_IMG)
        };
        plan_windows(&linears, &cfg).unwrap()
    }

    /// Multi-window budget for the tiny model (80 KB: several windows, every window fits).
    const SMALL: u64 = 80_000;
    const BIG: u64 = 1 << 30;

    fn check_close(got: &[f32], want: &[f32], what: &str) {
        let e = rel_rmse(got, want);
        assert!(e < 1e-5, "{what}: relative RMSE {e:e}");
    }

    fn ref_forward(src: &MemSource, inp: &ModelInput) -> Vec<f32> {
        let mut r = RefModel::from_source(src, 1).unwrap();
        r.forward(inp)
    }

    #[test]
    fn merged_matches_reference_single_window() {
        let c = tiny_dims();
        let src = tiny_source(c, 1);
        let plan = plan_with_budget(c, BIG);
        assert_eq!(plan.len(), 1);
        let mut m = HybridModel::new(&src, c, plan, MockDevice::new(1, BIG), Schedule::Merged).unwrap();
        let inp = input(c, N_TXT, N_IMG, 7, 0.8);
        let got = m.forward(&inp);
        check_close(&got, &ref_forward(&src, &inp), "merged vs ref");
        assert_eq!(m.name(), "hybrid-npu-linear");
    }

    #[test]
    fn stream_split_matches_reference_and_publish_count() {
        let c = tiny_dims();
        let src = tiny_source(c, 2);
        let plan = plan_with_budget(c, BIG);
        let expect_pubs = 8 * c.n_double + 2 * c.n_single;
        assert_eq!(stream_split_publishes(&plan), expect_pubs);
        let mut m = HybridModel::new(&src, c, plan, MockDevice::new(1, BIG), Schedule::StreamSplit).unwrap();
        let inp = input(c, N_TXT, N_IMG, 9, 0.6);
        let got = m.forward(&inp);
        check_close(&got, &ref_forward(&src, &inp), "stream-split vs ref");
        assert_eq!(m.stats().publishes, expect_pubs);
    }

    #[test]
    fn merged_multi_window_matches_reference_with_one_slot_per_window() {
        let c = tiny_dims();
        let src = tiny_source(c, 3);
        let plan = plan_with_budget(c, SMALL);
        let w = plan.len();
        assert!(w >= 3, "the small budget should give several windows, got {w}");
        let mut m = HybridModel::new(&src, c, plan.clone(), MockDevice::new(w, SMALL), Schedule::Merged).unwrap();
        let inp = input(c, N_TXT, N_IMG, 11, 0.4);
        let got = m.forward(&inp);
        check_close(&got, &ref_forward(&src, &inp), "multi-window vs ref");
        assert_eq!(m.stats().loads, w, "every window loaded once");
        assert_eq!(m.stats().reloads, 0);
        assert_eq!(m.stats().publishes, plan.iter().map(|x| x.stage_publishes).sum::<usize>());
    }

    #[test]
    fn four_steps_with_enough_slots_load_once_and_match_reference_each_step() {
        let c = tiny_dims();
        let src = tiny_source(c, 4);
        let plan = plan_with_budget(c, SMALL);
        let w = plan.len();
        let per_step: usize = plan.iter().map(|x| x.stage_publishes).sum();
        let mut m = HybridModel::new(&src, c, plan, MockDevice::new(w, SMALL), Schedule::Merged).unwrap();
        let mut loads_after = Vec::new();
        for (step, ts) in [0.9f32, 0.7, 0.5, 0.3].iter().enumerate() {
            let inp = input(c, N_TXT, N_IMG, 20 + step as u64, *ts);
            let got = m.forward(&inp);
            check_close(&got, &ref_forward(&src, &inp), &format!("step {step}"));
            loads_after.push(m.stats().loads);
            assert_eq!(m.stats().publishes, per_step * (step + 1), "publishes per step equal the plan");
        }
        assert_eq!(loads_after, vec![w, w, w, w], "no loads after the first step");
        assert_eq!(m.stats().reloads, 0);
        assert_eq!(m.device().total_loads(), w);
    }

    #[test]
    fn one_slot_reloads_every_window_every_step_as_planned() {
        let c = tiny_dims();
        let src = tiny_source(c, 5);
        let plan = plan_with_budget(c, SMALL);
        let w = plan.len();
        let predicted = steady_reloads_per_step(w, 1);
        assert_eq!(predicted, w);
        let mut m = HybridModel::new(&src, c, plan, MockDevice::new(1, SMALL), Schedule::Merged).unwrap();
        for step in 0..4 {
            let inp = input(c, N_TXT, N_IMG, 30 + step as u64, 0.5);
            check_close(&m.forward(&inp), &ref_forward(&src, &inp), &format!("1 slot step {step}"));
        }
        assert_eq!(m.stats().loads, 4 * w);
        assert_eq!(m.stats().reloads, 3 * predicted);
        for &c_ in &m.device().max_charged {
            assert!(c_ <= SMALL, "slot budget exceeded");
        }
    }

    #[test]
    fn two_slots_reload_count_matches_prediction() {
        let c = tiny_dims();
        let src = tiny_source(c, 6);
        let plan = plan_with_budget(c, SMALL);
        let w = plan.len();
        let predicted = steady_reloads_per_step(w, 2);
        let slots = assign_slots(w, 2);
        assert_eq!(slots.iter().filter(|&&s| s == 0).count(), 1, "one pinned window");
        let mut m = HybridModel::new(&src, c, plan, MockDevice::new(2, SMALL), Schedule::Merged).unwrap();
        for step in 0..4 {
            let inp = input(c, N_TXT, N_IMG, 40 + step as u64, 0.5);
            check_close(&m.forward(&inp), &ref_forward(&src, &inp), &format!("2 slots step {step}"));
        }
        assert_eq!(m.stats().reloads, 3 * predicted);
        assert!(m.device().max_charged.iter().all(|&x| x <= SMALL));
    }

    #[test]
    fn trace_shows_overlap_only_in_stream_split() {
        let c = tiny_dims();
        let src = tiny_source(c, 8);
        let plan = plan_with_budget(c, BIG);
        let inp = input(c, N_TXT, N_IMG, 50, 0.5);

        let between = |t: &[String], a: &str, b: &str| -> Vec<String> {
            let s = t.iter().position(|x| x == a).unwrap_or_else(|| panic!("no {a}"));
            let e = t.iter().skip(s).position(|x| x == b).map(|k| s + k).unwrap_or_else(|| panic!("no {b}"));
            t[s + 1..e].iter().filter(|x| x.starts_with("cpu:")).cloned().collect()
        };

        let mut sp = HybridModel::new(&src, c, plan.clone(), MockDevice::new(1, BIG), Schedule::StreamSplit).unwrap();
        sp.enable_trace();
        sp.forward(&inp);
        let t = sp.take_trace();
        assert_eq!(between(&t, "submit:A.img", "wait:A.img"), vec!["cpu:A.txt.post".to_string()]);
        assert_eq!(between(&t, "submit:B.txt", "wait:B.txt"), vec!["cpu:attention.img".to_string()]);

        let mut mg = HybridModel::new(&src, c, plan, MockDevice::new(1, BIG), Schedule::Merged).unwrap();
        mg.enable_trace();
        mg.forward(&inp);
        let t = mg.take_trace();
        // No CPU work sits between a merged submit and its wait: the graph has no independent work.
        assert!(between(&t, "submit:A", "wait:A").is_empty());
        assert!(between(&t, "submit:B", "wait:B").is_empty());
        assert!(between(&t, "submit:C", "wait:C").is_empty());
        assert!(between(&t, "submit:D", "wait:D").is_empty());
    }
}
