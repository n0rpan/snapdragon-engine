//! Whole FLUX.2 klein DiT in plain f32: the reference for the model, as `image_dit::reference` is
//! for one block.
//!
//! Mirrors sd.cpp `Flux::forward_orig` (`model/diffusion/flux.hpp`, ~1080-1260) for klein: no
//! guidance embed, no vector_in, no biases, shared modulation, SiLU-gated MLP (`use_mlp_silu_act`).
//! Every linear is row-major `[out][in]` (GGUF dims `[in, out]`). Tensors are read through
//! [`WeightSource`]: `open` keeps the GGUF mapped and dequantises one block only while it runs;
//! `from_source` decodes everything up front (tests, small models).

use gguf::Gguf;
use image_dit::arith::{round_activation, Arith, WType};
use image_dit::fast_model::{GgufSource, WeightSource};
use image_dit::model::{DitBackend, ModelInput};
use image_dit::reference::{single_block_ref_arith, single_block_ref_taps, StageTaps};
use image_dit::types::{BlockInput, BlockOutput, BlockWeightsF32, Dims};
use image_dit::weights::dequant_q8_0;
use std::path::Path;

/// Timestep embedding width (sd.cpp `time_in` input).
pub const TIME_DIM: usize = 256;
/// `ggml_ext_timestep_embedding(t, 256, 10000, 1000)` in `forward_orig`.
pub const TIME_MAX_PERIOD: f32 = 10000.0;
pub const TIME_FACTOR: f32 = 1000.0;
const EPS: f32 = 1e-6;

/// Model-level shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelDims {
    /// Block shapes (hidden, heads, head_dim, mlp). klein 4B: 3072, 24, 128, 9216.
    pub block: Dims,
    /// Latent channels entering `img_in` (klein: 128).
    pub in_ch: usize,
    /// Text context width entering `txt_in` (klein: 7680).
    pub ctx_dim: usize,
    /// Output channels of `final_layer.linear` (klein: 128).
    pub out_ch: usize,
    pub n_double: usize,
    pub n_single: usize,
}

/// One stream's (image or text) double-block weights, row-major `[out][in]`.
pub struct StreamF32 {
    /// `[3h][h]`, rows q | k | v.
    pub qkv: Vec<f32>,
    /// `[h][h]`
    pub proj: Vec<f32>,
    /// `[2m][h]`, first half of the rows is the SiLU gate.
    pub mlp0: Vec<f32>,
    /// `[h][m]`
    pub mlp2: Vec<f32>,
    pub q_scale: Vec<f32>,
    pub k_scale: Vec<f32>,
}

pub struct DoubleBlockF32 {
    pub img: StreamF32,
    pub txt: StreamF32,
}

/// Embedders and shared modulations.
pub struct EmbedF32 {
    /// `[h][in_ch]`
    pub img_in: Vec<f32>,
    /// `[h][ctx]`
    pub txt_in: Vec<f32>,
    /// `[h][256]` (`time_in.in_layer`)
    pub time_in0: Vec<f32>,
    /// `[h][h]` (`time_in.out_layer`)
    pub time_in2: Vec<f32>,
    /// `[6h][h]` (`double_stream_modulation_img.lin`)
    pub mod_img: Vec<f32>,
    /// `[6h][h]` (`double_stream_modulation_txt.lin`)
    pub mod_txt: Vec<f32>,
    /// `[3h][h]` (`single_stream_modulation.lin`)
    pub mod_single: Vec<f32>,
    /// `[2h][h]` (`final_layer.adaLN_modulation.1`)
    pub final_adaln: Vec<f32>,
    /// `[out][h]` (`final_layer.linear`)
    pub final_linear: Vec<f32>,
    /// Storage type of each matrix above (decides the activation rounding of [`Arith::SdParity`]).
    pub types: EmbedTypes,
}

/// File storage types of the embedder / modulation / head matrices.
#[derive(Clone, Copy, Debug)]
pub struct EmbedTypes {
    pub img_in: WType,
    pub txt_in: WType,
    pub time_in0: WType,
    pub time_in2: WType,
    pub mod_img: WType,
    pub mod_txt: WType,
    pub mod_single: WType,
    pub final_adaln: WType,
    pub final_linear: WType,
}

/// Records the token matrices entering single blocks 0 and 1, `[n_txt+n_img][h]` each, when set,
/// and (W5-G, [`Taps::all`]) the per-stage intermediates of the first forward.
#[derive(Default)]
pub struct Taps {
    pub single0_in: Option<Vec<f32>>,
    pub single1_in: Option<Vec<f32>>,
    /// `img_in(img)` `[n_img][h]`, `txt_in(txt)` `[n_txt][h]`.
    pub img_emb: Option<Vec<f32>>,
    pub txt_emb: Option<Vec<f32>>,
    /// `time_in(sinusoid(t))` `[h]` (before the SiLU the modulations apply).
    pub vec: Option<Vec<f32>>,
    /// Single-stream modulation `[shift | scale | gate]`, `[3h]`.
    pub mod_single: Option<Vec<f32>>,
    /// Output of every double block: (img `[n_img][h]`, txt `[n_txt][h]`).
    pub double_out: Option<Vec<(Vec<f32>, Vec<f32>)>>,
    /// Single block 0: outputs with attention / MLP / linear2 intermediates, plus x_mod and q/k/v.
    pub single0: Option<(BlockOutput, StageTaps)>,
    /// Image tokens entering the final layer `[n_img][h]`.
    pub final_in: Option<Vec<f32>>,
}

impl Taps {
    /// Both single-block input taps enabled.
    pub fn recording() -> Taps {
        Taps { single0_in: Some(Vec::new()), single1_in: Some(Vec::new()), ..Taps::default() }
    }

    /// Every tap enabled (about 220 MB at 512 px with five double blocks).
    pub fn all() -> Taps {
        Taps {
            single0_in: Some(Vec::new()),
            single1_in: Some(Vec::new()),
            img_emb: Some(Vec::new()),
            txt_emb: Some(Vec::new()),
            vec: Some(Vec::new()),
            mod_single: Some(Vec::new()),
            double_out: Some(Vec::new()),
            single0: Some(Default::default()),
            final_in: Some(Vec::new()),
        }
    }
}

/// Embedding-path vectors of one timestep (no token work).
pub struct TimeVectors {
    /// `time_in(sinusoid(t))`, `[h]`.
    pub vec: Vec<f32>,
    /// Single-stream modulation `[shift | scale | gate]`, `[3h]`.
    pub mod_single: Vec<f32>,
    /// Double-stream modulations, `[6h]` each.
    pub mod_img: Vec<f32>,
    pub mod_txt: Vec<f32>,
    /// Final-layer `[shift | scale]`, `[2h]`.
    pub final_mod: Vec<f32>,
}

enum Blocks {
    /// Blocks are dequantised from the mapped GGUF while they run.
    File(Gguf),
    /// Blocks decoded up front.
    Mem { double: Vec<DoubleBlockF32>, single: Vec<BlockWeightsF32> },
}

struct TxtCache {
    txt: Vec<f32>,
    emb: Vec<f32>,
}

/// The whole klein DiT in f32. Name: `ref-f32`.
pub struct RefModel {
    dims: ModelDims,
    embed: EmbedF32,
    blocks: Blocks,
    threads: usize,
    txt_cache: Option<TxtCache>,
    arith: Arith,
    perturb_ulp: u32,
}

impl RefModel {
    /// Map a klein GGUF. Only shapes are checked up front; block weights are dequantised per block.
    pub fn open(gguf_path: &Path, threads: usize) -> Result<RefModel, String> {
        let g = Gguf::open(gguf_path).map_err(|e| e.0)?;
        let dims = model_dims(&GgufSource(&g))?;
        let embed = decode_embed(&GgufSource(&g), &dims)?;
        for i in 0..dims.n_double {
            check_double(&GgufSource(&g), i, &dims)?;
        }
        for i in 0..dims.n_single {
            check_single(&GgufSource(&g), i, &dims)?;
        }
        Ok(RefModel { dims, embed, blocks: Blocks::File(g), threads: threads.max(1), txt_cache: None, arith: Arith::F32, perturb_ulp: 0 })
    }

    /// Decode every tensor from `src` (tests and small models).
    pub fn from_source(src: &dyn WeightSource, threads: usize) -> Result<RefModel, String> {
        let dims = model_dims(src)?;
        let embed = decode_embed(src, &dims)?;
        let mut double = Vec::with_capacity(dims.n_double);
        for i in 0..dims.n_double {
            double.push(decode_double(src, i, &dims)?);
        }
        let mut single = Vec::with_capacity(dims.n_single);
        for i in 0..dims.n_single {
            single.push(decode_single(src, i, &dims)?);
        }
        Ok(RefModel {
            dims,
            embed,
            blocks: Blocks::Mem { double, single },
            threads: threads.max(1),
            txt_cache: None,
            arith: Arith::F32,
            perturb_ulp: 0,
        })
    }

    pub fn dims(&self) -> ModelDims {
        self.dims
    }

    /// Activation arithmetic (default [`Arith::F32`], the original reference).
    pub fn set_arith(&mut self, arith: Arith) {
        if arith != self.arith {
            self.txt_cache = None;
        }
        self.arith = arith;
    }

    pub fn arith(&self) -> Arith {
        self.arith
    }

    /// Self-noise control (W5-G): move every value of the img and txt token embeddings by up to
    /// `ulp` ulps (fixed pseudo-random pattern) before the double blocks. 0 = off (default).
    pub fn set_perturb_ulp(&mut self, ulp: u32) {
        self.perturb_ulp = ulp;
    }

    pub fn embed_types(&self) -> EmbedTypes {
        self.embed.types
    }

    /// Time embedding and every modulation vector for timestep `t` (component probe: no tokens).
    pub fn time_vectors(&self, t: f32) -> TimeVectors {
        let (h, th, a, ty) = (self.dims.block.hidden, self.threads, self.arith, &self.embed.types);
        let temb = timestep_embedding(t, TIME_DIM, TIME_MAX_PERIOD, TIME_FACTOR);
        let mut t1 = linear_a(&self.embed.time_in0, TIME_DIM, h, &temb, 1, th, a, ty.time_in0);
        for v in t1.iter_mut() {
            *v = silu(*v);
        }
        let vec = linear_a(&self.embed.time_in2, h, h, &t1, 1, th, a, ty.time_in2);
        let svec: Vec<f32> = vec.iter().map(|&v| silu(v)).collect();
        TimeVectors {
            mod_img: linear_a(&self.embed.mod_img, h, 6 * h, &svec, 1, th, a, ty.mod_img),
            mod_txt: linear_a(&self.embed.mod_txt, h, 6 * h, &svec, 1, th, a, ty.mod_txt),
            mod_single: linear_a(&self.embed.mod_single, h, 3 * h, &svec, 1, th, a, ty.mod_single),
            final_mod: linear_a(&self.embed.final_adaln, h, 2 * h, &svec, 1, th, a, ty.final_adaln),
            vec,
        }
    }

    /// Single block `i` alone on a given input (component probe; e.g. sd.cpp's dumped block input).
    pub fn single_block(&self, i: usize, input: &BlockInput) -> Result<(BlockOutput, StageTaps), String> {
        let (th, a) = (self.threads, self.arith);
        with_single(self, i, |w| single_block_ref_taps(w, input, th, a))
    }

    /// Double block `i` alone on given `(img, txt)` token matrices with the modulations of `tv`
    /// (component probe). `pe` covers `[txt | img]` tokens. Returns `(img, txt)`.
    pub fn double_block_at(
        &self,
        i: usize,
        img: Vec<f32>,
        txt: Vec<f32>,
        tv: &TimeVectors,
        pe: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let h = self.dims.block.hidden;
        let (mi, mt) = (Mods6::from_vec(&tv.mod_img, h), Mods6::from_vec(&tv.mod_txt, h));
        let (b, th, a) = (self.dims.block, self.threads, self.arith);
        with_double(self, i, |blk| double_block_a(b, blk, img, txt, &mi, &mt, pe, th, a))
    }

    /// Final layer on the image tokens `[n_img][h]` with the final modulation of `tv`.
    pub fn final_layer(&self, img_tok: &[f32], tv: &TimeVectors) -> Vec<f32> {
        let (h, th, a) = (self.dims.block.hidden, self.threads, self.arith);
        let n_img = img_tok.len() / h;
        let (fshift, fscale) = (&tv.final_mod[..h], &tv.final_mod[h..]);
        let y = par_rows(n_img, h, th, |s, e, out| {
            for t in s..e {
                let o = (t - s) * h;
                ln_modulate(&img_tok[t * h..(t + 1) * h], fshift, fscale, &mut out[o..o + h]);
            }
        });
        linear_a(&self.embed.final_linear, h, self.dims.out_ch, &y, n_img, th, a, self.embed.types.final_linear)
    }

    /// `img_in(img)` and `txt_in(txt)` (component probe).
    pub fn embed_tokens(&self, img: &[f32], n_img: usize, txt: &[f32], n_txt: usize) -> (Vec<f32>, Vec<f32>) {
        let (d, th, a, ty) = (self.dims, self.threads, self.arith, &self.embed.types);
        let h = d.block.hidden;
        (
            linear_a(&self.embed.img_in, d.in_ch, h, img, n_img, th, a, ty.img_in),
            linear_a(&self.embed.txt_in, d.ctx_dim, h, txt, n_txt, th, a, ty.txt_in),
        )
    }

    /// Same as [`DitBackend::forward`], also recording the requested taps.
    pub fn forward_with_taps(&mut self, input: &ModelInput, taps: &mut Taps) -> Vec<f32> {
        self.ensure_txt(input);
        let emb = &self.txt_cache.as_ref().expect("txt cache set above").emb;
        self.run(input, emb, taps).expect("weights were validated at construction")
    }

    /// `txt_in(txt)` does not depend on the step: recompute only when the context changes.
    fn ensure_txt(&mut self, input: &ModelInput) {
        if let Some(c) = &self.txt_cache {
            if c.txt == input.txt {
                return;
            }
        }
        let h = self.dims.block.hidden;
        let ty = self.embed.types.txt_in;
        let emb = linear_a(&self.embed.txt_in, self.dims.ctx_dim, h, &input.txt, input.n_txt, self.threads, self.arith, ty);
        self.txt_cache = Some(TxtCache { txt: input.txt.clone(), emb });
    }

    fn run(&self, input: &ModelInput, txt_emb: &[f32], taps: &mut Taps) -> Result<Vec<f32>, String> {
        let d = self.dims;
        let b = d.block;
        let (h, hd) = (b.hidden, b.head_dim);
        let half = hd / 2;
        let (n_img, n_txt) = (input.n_img, input.n_txt);
        let n = n_txt + n_img;
        let th = self.threads;
        assert_eq!(input.img.len(), n_img * d.in_ch, "img shape");
        assert_eq!(input.txt.len(), n_txt * d.ctx_dim, "txt shape");
        assert_eq!(input.pe.len(), n * half * 4, "pe shape");
        assert_eq!(txt_emb.len(), n_txt * h, "txt embedding shape");

        let a = self.arith;

        // 1. embedders
        let mut img = linear_a(&self.embed.img_in, d.in_ch, h, &input.img, n_img, th, a, self.embed.types.img_in);
        let mut txt = txt_emb.to_vec();
        if self.perturb_ulp > 0 {
            img = crate::drift::perturb(&img, self.perturb_ulp);
            txt = crate::drift::perturb(&txt, self.perturb_ulp);
        }
        tap(&mut taps.img_emb, &img);
        tap(&mut taps.txt_emb, &txt);

        // 2. time embedding: vec = time_in(sinusoid(t)); 3. shared modulations from silu(vec)
        let tv = self.time_vectors(input.timestep);
        tap(&mut taps.vec, &tv.vec);
        tap(&mut taps.mod_single, &tv.mod_single);
        let mi = Mods6::from_vec(&tv.mod_img, h);
        let mt = Mods6::from_vec(&tv.mod_txt, h);
        let ms = Mods3::from_vec(&tv.mod_single, h);

        // 4. double blocks (joint attention, text tokens first)
        for i in 0..d.n_double {
            let (ni, nt) = with_double(self, i, |blk| {
                double_block_a(b, blk, std::mem::take(&mut img), std::mem::take(&mut txt), &mi, &mt, &input.pe, th, a)
            })?;
            img = ni;
            txt = nt;
            if let Some(v) = taps.double_out.as_mut() {
                v.push((img.clone(), txt.clone()));
            }
        }

        // 5. single blocks on the concatenation (text first)
        let mut x = [txt, img].concat();
        for i in 0..d.n_single {
            if i == 0 {
                if let Some(t) = taps.single0_in.as_mut() {
                    *t = x.clone();
                }
            }
            if i == 1 {
                if let Some(t) = taps.single1_in.as_mut() {
                    *t = x.clone();
                }
            }
            let inp = BlockInput {
                n_tokens: n,
                x: std::mem::take(&mut x),
                pe: input.pe.clone(),
                shift: ms.shift.clone(),
                scale: ms.scale.clone(),
                gate: ms.gate.clone(),
            };
            if i == 0 && taps.single0.is_some() {
                let (o, st) = with_single(self, i, |w| single_block_ref_taps(w, &inp, th, a))?;
                x = o.out.clone();
                taps.single0 = Some((o, st));
            } else {
                x = with_single(self, i, |w| single_block_ref_arith(w, &inp, th, a).out)?;
            }
        }

        // 6. final layer on the image tokens
        let img_tok = &x[n_txt * h..];
        tap(&mut taps.final_in, img_tok);
        Ok(self.final_layer(img_tok, &tv))
    }
}

/// Fills an enabled tap.
fn tap(slot: &mut Option<Vec<f32>>, v: &[f32]) {
    if let Some(t) = slot.as_mut() {
        *t = v.to_vec();
    }
}

impl DitBackend for RefModel {
    fn name(&self) -> &'static str {
        "ref-f32"
    }

    fn forward(&mut self, input: &ModelInput) -> Vec<f32> {
        self.forward_with_taps(input, &mut Taps::default())
    }
}

// ---------------------------------------------------------------------------------------------
// block access (lazy for File, owned for Mem)
// ---------------------------------------------------------------------------------------------

fn with_double<R>(m: &RefModel, i: usize, f: impl FnOnce(&DoubleBlockF32) -> R) -> Result<R, String> {
    match &m.blocks {
        Blocks::Mem { double, .. } => Ok(f(&double[i])),
        Blocks::File(g) => {
            let blk = decode_double(&GgufSource(g), i, &m.dims)?;
            Ok(f(&blk))
        }
    }
}

fn with_single<R>(m: &RefModel, i: usize, f: impl FnOnce(&BlockWeightsF32) -> R) -> Result<R, String> {
    match &m.blocks {
        Blocks::Mem { single, .. } => Ok(f(&single[i])),
        Blocks::File(g) => {
            let blk = decode_single(&GgufSource(g), i, &m.dims)?;
            Ok(f(&blk))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------------

/// Shapes from the tensor table. Block dims come from single block 0 when present (klein), else
/// from double block 0, so models with only one kind of block still load.
fn model_dims(src: &dyn WeightSource) -> Result<ModelDims, String> {
    let n_double = count_blocks(src, "double_blocks", "img_attn.qkv.weight");
    let n_single = count_blocks(src, "single_blocks", "linear1.weight");
    if n_double == 0 && n_single == 0 {
        return Err("no double or single blocks found".into());
    }
    let (_, img_rows, in_ch) = src.f32("img_in.weight")?;
    let hidden = img_rows;
    let (_, _, ctx_dim) = src.f32("txt_in.weight")?;
    let (_, out_ch, fin_k) = src.f32("final_layer.linear.weight")?;
    if fin_k != hidden {
        return Err(format!("final_layer.linear input {fin_k} != hidden {hidden}"));
    }
    let (head_dim, mlp) = if n_single > 0 {
        let (_, _, hd) = src.f32("single_blocks.0.norm.query_norm.scale")?;
        let (_, rows1, k1) = src.q8("single_blocks.0.linear1.weight")?;
        let (_, rows2, k2) = src.q8("single_blocks.0.linear2.weight")?;
        if k1 != hidden || rows2 != hidden || k2 <= hidden {
            return Err(format!("single block shapes: linear1 in {k1}, linear2 out {rows2}, in {k2}"));
        }
        let mlp = k2 - hidden;
        if rows1 != 3 * hidden + 2 * mlp {
            return Err(format!("linear1 rows {rows1} != 3*hidden + 2*mlp"));
        }
        (hd, mlp)
    } else {
        let (_, _, hd) = src.f32("double_blocks.0.img_attn.norm.query_norm.scale")?;
        let (_, rows, k) = src.q8("double_blocks.0.img_mlp.2.weight")?;
        if rows != hidden {
            return Err(format!("img_mlp.2 rows {rows} != hidden {hidden}"));
        }
        (hd, k)
    };
    if head_dim == 0 || hidden % head_dim != 0 || head_dim % 2 != 0 {
        return Err(format!("bad attention shape: hidden {hidden}, head_dim {head_dim}"));
    }
    let block = Dims { hidden, heads: hidden / head_dim, head_dim, mlp };
    block.check();
    Ok(ModelDims { block, in_ch, ctx_dim, out_ch, n_double, n_single })
}

fn count_blocks(src: &dyn WeightSource, kind: &str, probe: &str) -> usize {
    (0usize..).take_while(|i| src.has(&format!("{kind}.{i}.{probe}"))).count()
}

fn decode_embed(src: &dyn WeightSource, d: &ModelDims) -> Result<EmbedF32, String> {
    let h = d.block.hidden;
    let wt = |name: &str| WType::from_name(src.dtype(name).as_deref());
    Ok(EmbedF32 {
        types: EmbedTypes {
            img_in: wt("img_in.weight"),
            txt_in: wt("txt_in.weight"),
            time_in0: wt("time_in.in_layer.weight"),
            time_in2: wt("time_in.out_layer.weight"),
            mod_img: wt("double_stream_modulation_img.lin.weight"),
            mod_txt: wt("double_stream_modulation_txt.lin.weight"),
            mod_single: wt("single_stream_modulation.lin.weight"),
            final_adaln: wt("final_layer.adaLN_modulation.1.weight"),
            final_linear: wt("final_layer.linear.weight"),
        },
        img_in: get_f32(src, "img_in.weight", h, d.in_ch)?,
        txt_in: get_f32(src, "txt_in.weight", h, d.ctx_dim)?,
        time_in0: get_f32(src, "time_in.in_layer.weight", h, TIME_DIM)?,
        time_in2: get_f32(src, "time_in.out_layer.weight", h, h)?,
        mod_img: get_f32(src, "double_stream_modulation_img.lin.weight", 6 * h, h)?,
        mod_txt: get_f32(src, "double_stream_modulation_txt.lin.weight", 6 * h, h)?,
        mod_single: get_f32(src, "single_stream_modulation.lin.weight", 3 * h, h)?,
        final_adaln: get_f32(src, "final_layer.adaLN_modulation.1.weight", 2 * h, h)?,
        final_linear: get_f32(src, "final_layer.linear.weight", d.out_ch, h)?,
    })
}

/// Shape-checks a double block's tensors without decoding them.
fn check_double(src: &dyn WeightSource, i: usize, d: &ModelDims) -> Result<(), String> {
    for s in ["img", "txt"] {
        let names = double_names(i, s);
        let (h, hd, m) = (d.block.hidden, d.block.head_dim, d.block.mlp);
        check_q8(src, &names.qkv, 3 * h, h)?;
        check_q8(src, &names.proj, h, h)?;
        check_q8(src, &names.mlp0, 2 * m, h)?;
        check_q8(src, &names.mlp2, h, m)?;
        check_f32_len(src, &names.q_scale, hd)?;
        check_f32_len(src, &names.k_scale, hd)?;
    }
    Ok(())
}

fn check_single(src: &dyn WeightSource, i: usize, d: &ModelDims) -> Result<(), String> {
    let (h, hd, m) = (d.block.hidden, d.block.head_dim, d.block.mlp);
    check_q8(src, &format!("single_blocks.{i}.linear1.weight"), 3 * h + 2 * m, h)?;
    check_q8(src, &format!("single_blocks.{i}.linear2.weight"), h, h + m)?;
    check_f32_len(src, &format!("single_blocks.{i}.norm.query_norm.scale"), hd)?;
    check_f32_len(src, &format!("single_blocks.{i}.norm.key_norm.scale"), hd)
}

struct StreamNames {
    qkv: String,
    proj: String,
    mlp0: String,
    mlp2: String,
    q_scale: String,
    k_scale: String,
}

fn double_names(i: usize, s: &str) -> StreamNames {
    StreamNames {
        qkv: format!("double_blocks.{i}.{s}_attn.qkv.weight"),
        proj: format!("double_blocks.{i}.{s}_attn.proj.weight"),
        mlp0: format!("double_blocks.{i}.{s}_mlp.0.weight"),
        mlp2: format!("double_blocks.{i}.{s}_mlp.2.weight"),
        q_scale: format!("double_blocks.{i}.{s}_attn.norm.query_norm.scale"),
        k_scale: format!("double_blocks.{i}.{s}_attn.norm.key_norm.scale"),
    }
}

fn decode_double(src: &dyn WeightSource, i: usize, d: &ModelDims) -> Result<DoubleBlockF32, String> {
    let (h, hd, m) = (d.block.hidden, d.block.head_dim, d.block.mlp);
    let stream = |s: &str| -> Result<StreamF32, String> {
        let n = double_names(i, s);
        Ok(StreamF32 {
            qkv: get_q8_f32(src, &n.qkv, 3 * h, h)?,
            proj: get_q8_f32(src, &n.proj, h, h)?,
            mlp0: get_q8_f32(src, &n.mlp0, 2 * m, h)?,
            mlp2: get_q8_f32(src, &n.mlp2, h, m)?,
            q_scale: get_f32(src, &n.q_scale, 1, hd)?,
            k_scale: get_f32(src, &n.k_scale, 1, hd)?,
        })
    };
    Ok(DoubleBlockF32 { img: stream("img")?, txt: stream("txt")? })
}

fn decode_single(src: &dyn WeightSource, i: usize, d: &ModelDims) -> Result<BlockWeightsF32, String> {
    let (h, hd, m) = (d.block.hidden, d.block.head_dim, d.block.mlp);
    Ok(BlockWeightsF32 {
        dims: d.block,
        w1: get_q8_f32(src, &format!("single_blocks.{i}.linear1.weight"), 3 * h + 2 * m, h)?,
        w2: get_q8_f32(src, &format!("single_blocks.{i}.linear2.weight"), h, h + m)?,
        q_scale: get_f32(src, &format!("single_blocks.{i}.norm.query_norm.scale"), 1, hd)?,
        k_scale: get_f32(src, &format!("single_blocks.{i}.norm.key_norm.scale"), 1, hd)?,
    })
}

fn check_q8(src: &dyn WeightSource, name: &str, rows: usize, k: usize) -> Result<(), String> {
    let (bytes, r, kk) = src.q8(name)?;
    if r != rows || kk != k {
        return Err(format!("{name}: shape [{kk}, {r}], want [{k}, {rows}]"));
    }
    if bytes.len() != rows * (k / 32) * 34 {
        return Err(format!("{name}: {} bytes, want {}", bytes.len(), rows * (k / 32) * 34));
    }
    Ok(())
}

fn check_f32_len(src: &dyn WeightSource, name: &str, len: usize) -> Result<(), String> {
    let (v, rows, k) = src.f32(name)?;
    if rows != 1 || k != len || v.len() != len {
        return Err(format!("{name}: shape ({rows}, {k}), want {len}"));
    }
    Ok(())
}

/// A float matrix `[rows][k]` (GGUF `[k, rows]`), shape-checked.
fn get_f32(src: &dyn WeightSource, name: &str, rows: usize, k: usize) -> Result<Vec<f32>, String> {
    let (v, r, kk) = src.f32(name)?;
    if r != rows || kk != k || v.len() != rows * k {
        return Err(format!("{name}: shape [{kk}, {r}], want [{k}, {rows}]"));
    }
    Ok(v)
}

/// A Q8_0 matrix `[rows][k]`, dequantised.
fn get_q8_f32(src: &dyn WeightSource, name: &str, rows: usize, k: usize) -> Result<Vec<f32>, String> {
    check_q8(src, name, rows, k)?;
    let (bytes, _, _) = src.q8(name)?;
    Ok(dequant_q8_0(bytes, rows, k))
}

// ---------------------------------------------------------------------------------------------
// math
// ---------------------------------------------------------------------------------------------

/// Modulation vectors of one double block stream pair: `[shift1, scale1, gate1, shift2, scale2, gate2]`.
pub(crate) struct Mods6 {
    pub shift1: Vec<f32>,
    pub scale1: Vec<f32>,
    pub gate1: Vec<f32>,
    pub shift2: Vec<f32>,
    pub scale2: Vec<f32>,
    pub gate2: Vec<f32>,
}

impl Mods6 {
    /// Chunk order of sd.cpp `Modulation::forward` (`ModulationOut` offsets 0 and 3).
    pub(crate) fn from_vec(v: &[f32], h: usize) -> Mods6 {
        assert_eq!(v.len(), 6 * h);
        let c = |k: usize| v[k * h..(k + 1) * h].to_vec();
        Mods6 { shift1: c(0), scale1: c(1), gate1: c(2), shift2: c(3), scale2: c(4), gate2: c(5) }
    }
}

/// Single block modulation `[shift, scale, gate]`.
pub(crate) struct Mods3 {
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
    pub gate: Vec<f32>,
}

impl Mods3 {
    pub(crate) fn from_vec(v: &[f32], h: usize) -> Mods3 {
        assert_eq!(v.len(), 3 * h);
        let c = |k: usize| v[k * h..(k + 1) * h].to_vec();
        Mods3 { shift: c(0), scale: c(1), gate: c(2) }
    }
}

/// One double block: both streams, joint attention over `[txt | img]` tokens.
#[allow(dead_code)]
pub(crate) fn double_block(
    d: Dims,
    b: &DoubleBlockF32,
    img: Vec<f32>,
    txt: Vec<f32>,
    mi: &Mods6,
    mt: &Mods6,
    pe: &[f32],
    threads: usize,
) -> (Vec<f32>, Vec<f32>) {
    double_block_a(d, b, img, txt, mi, mt, pe, threads, Arith::F32)
}

/// [`double_block`] with activation arithmetic `a` (every double-block matrix is Q8_0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn double_block_a(
    d: Dims,
    b: &DoubleBlockF32,
    img: Vec<f32>,
    txt: Vec<f32>,
    mi: &Mods6,
    mt: &Mods6,
    pe: &[f32],
    threads: usize,
    a: Arith,
) -> (Vec<f32>, Vec<f32>) {
    let h = d.hidden;
    let half = d.head_dim / 2;
    let n_txt = txt.len() / h;
    let n_img = img.len() / h;
    let n = n_txt + n_img;
    assert_eq!(pe.len(), n * half * 4, "pe shape");

    let (qt, kt, vt) = stream_qkv(d, &b.txt, &txt, mt, &pe[..n_txt * half * 4], threads, a);
    let (qi, ki, vi) = stream_qkv(d, &b.img, &img, mi, &pe[n_txt * half * 4..], threads, a);
    let q = [qt, qi].concat();
    let k = [kt, ki].concat();
    let v = [vt, vi].concat();
    let attn = attention(d, &q, &k, &v, n, threads);
    let (attn_t, attn_i) = attn.split_at(n_txt * h);

    let txt = stream_post(d, &b.txt, txt, attn_t, mt, threads, a);
    let img = stream_post(d, &b.img, img, attn_i, mi, threads, a);
    (img, txt)
}

/// LayerNorm + modulate, then the stream's qkv, QK RMS norm and RoPE. Returns q, k, v `[n][h]`.
fn stream_qkv(
    d: Dims,
    s: &StreamF32,
    x: &[f32],
    m: &Mods6,
    pe: &[f32],
    threads: usize,
    a: Arith,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (h, hd) = (d.hidden, d.head_dim);
    let half = hd / 2;
    let n = x.len() / h;
    let xm = par_rows(n, h, threads, |a, b, out| {
        for t in a..b {
            let o = (t - a) * h;
            ln_modulate(&x[t * h..(t + 1) * h], &m.shift1, &m.scale1, &mut out[o..o + h]);
        }
    });
    let y = linear_a(&s.qkv, h, 3 * h, &xm, n, threads, a, WType::Q8_0);
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
            rms_norm_scale(&mut q[r.clone()], &s.q_scale);
            rope(&mut q[r.clone()], pe_t);
            rms_norm_scale(&mut k[r.clone()], &s.k_scale);
            rope(&mut k[r], pe_t);
        }
    }
    (q, k, v)
}

/// Attention output projection with gate, then LayerNorm+modulate and the SiLU-gated MLP with gate.
fn stream_post(d: Dims, s: &StreamF32, mut x: Vec<f32>, attn: &[f32], m: &Mods6, threads: usize, a: Arith) -> Vec<f32> {
    let (h, mlp) = (d.hidden, d.mlp);
    let n = x.len() / h;
    let p = linear_a(&s.proj, h, h, attn, n, threads, a, WType::Q8_0);
    for t in 0..n {
        for r in 0..h {
            x[t * h + r] += m.gate1[r] * p[t * h + r];
        }
    }
    let xm = par_rows(n, h, threads, |a, b, out| {
        for t in a..b {
            let o = (t - a) * h;
            ln_modulate(&x[t * h..(t + 1) * h], &m.shift2, &m.scale2, &mut out[o..o + h]);
        }
    });
    let m0 = linear_a(&s.mlp0, h, 2 * mlp, &xm, n, threads, a, WType::Q8_0);
    // ggml_swiglu: silu(first half) * second half
    let act = par_rows(n, mlp, threads, |a, b, out| {
        for t in a..b {
            let row = &m0[t * 2 * mlp..(t + 1) * 2 * mlp];
            let o = (t - a) * mlp;
            for j in 0..mlp {
                out[o + j] = silu(row[j]) * row[mlp + j];
            }
        }
    });
    let m2 = linear_a(&s.mlp2, mlp, h, &act, n, threads, a, WType::Q8_0);
    for t in 0..n {
        for r in 0..h {
            x[t * h + r] += m.gate2[r] * m2[t * h + r];
        }
    }
    x
}

/// Joint attention over `n` tokens, `softmax(q k^T / sqrt(head_dim)) v` per head. Rows split over threads.
fn attention(d: Dims, q: &[f32], k: &[f32], v: &[f32], n: usize, threads: usize) -> Vec<f32> {
    let (h, hd) = (d.hidden, d.head_dim);
    let scale = 1.0 / (hd as f32).sqrt();
    par_rows(n, h, threads, |s, e, out| {
        let mut logits = vec![0f32; n];
        for t in s..e {
            for hh in 0..d.heads {
                let base = hh * hd;
                let qv = &q[t * h + base..t * h + base + hd];
                let mut mx = f32::NEG_INFINITY;
                for tk in 0..n {
                    let l = dot(qv, &k[tk * h + base..tk * h + base + hd]) * scale;
                    logits[tk] = l;
                    mx = mx.max(l);
                }
                let mut sum = 0f32;
                for l in logits.iter_mut() {
                    *l = (*l - mx).exp();
                    sum += *l;
                }
                let inv = 1.0 / sum;
                let o = &mut out[(t - s) * h + base..(t - s) * h + base + hd];
                o.iter_mut().for_each(|x| *x = 0.0);
                for tk in 0..n {
                    let p = logits[tk] * inv;
                    let vv = &v[tk * h + base..tk * h + base + hd];
                    for dd in 0..hd {
                        o[dd] += p * vv[dd];
                    }
                }
            }
        }
    })
}

/// `y[t][o] = sum_i w[o][i] x[t][i]` for `rows` tokens; `w` is `[out][in]`.
pub(crate) fn linear_rows(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32], rows: usize, threads: usize) -> Vec<f32> {
    assert_eq!(w.len(), out_dim * in_dim, "linear weight shape");
    assert_eq!(x.len(), rows * in_dim, "linear input shape");
    // Tokens in groups of TOKEN_GROUP so a weight row is reused from cache; every (token, row)
    // value is the same `dot` call as before, so the result is bitwise unchanged.
    par_rows(rows, out_dim, threads, |s, e, out| {
        let mut g0 = s;
        while g0 < e {
            let g1 = (g0 + TOKEN_GROUP).min(e);
            for r in 0..out_dim {
                let wr = &w[r * in_dim..(r + 1) * in_dim];
                for t in g0..g1 {
                    out[(t - s) * out_dim + r] = dot(wr, &x[t * in_dim..(t + 1) * in_dim]);
                }
            }
            g0 = g1;
        }
    })
}

/// Tokens per weight-row pass in the reference linears (cache reuse only; no effect on values).
pub(crate) const TOKEN_GROUP: usize = 8;

/// [`linear_rows`] after rounding each input row as sd.cpp would for a `wt` weight under `a`
/// (no copy when nothing is rounded).
#[allow(clippy::too_many_arguments)]
pub(crate) fn linear_a(
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
    x: &[f32],
    rows: usize,
    threads: usize,
    a: Arith,
    wt: WType,
) -> Vec<f32> {
    if a == Arith::F32 || wt == WType::F32 {
        return linear_rows(w, in_dim, out_dim, x, rows, threads);
    }
    let mut xr = x.to_vec();
    for row in xr.chunks_exact_mut(in_dim) {
        round_activation(a, wt, row);
    }
    linear_rows(w, in_dim, out_dim, &xr, rows, threads)
}

/// Splits `n` rows into contiguous ranges, one per thread; `f(start, end, out)` fills `out`
/// (`(end-start) * width` values). Each row is computed independently, so the result does not
/// depend on the thread count.
pub(crate) fn par_rows<F>(n: usize, width: usize, threads: usize, f: F) -> Vec<f32>
where
    F: Fn(usize, usize, &mut [f32]) + Sync,
{
    let mut out = vec![0f32; n * width];
    if n == 0 {
        return out;
    }
    let t = threads.max(1).min(n);
    if t == 1 {
        f(0, n, &mut out);
        return out;
    }
    let per = n.div_ceil(t);
    let fref = &f;
    std::thread::scope(|sc| {
        for (i, chunk) in out.chunks_mut(per * width).enumerate() {
            let s = i * per;
            let e = (s + per).min(n);
            sc.spawn(move || fref(s, e, chunk));
        }
    });
    out
}

/// `(x + x*scale) + shift` after LayerNorm without affine (eps 1e-6). Mean and variance in f64.
pub(crate) fn ln_modulate(x: &[f32], shift: &[f32], scale: &[f32], out: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let denom = (var + EPS as f64).sqrt();
    for (((o, &xv), &sc), &sh) in out.iter_mut().zip(x).zip(scale).zip(shift) {
        let y = ((xv as f64 - mean) / denom) as f32;
        *o = (y + y * sc) + sh;
    }
}

/// `x / sqrt(mean(x^2) + eps) * scale[d]`, in place over one head.
pub(crate) fn rms_norm_scale(x: &mut [f32], scale: &[f32]) {
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let denom = (ms + EPS as f64).sqrt();
    for (v, &sc) in x.iter_mut().zip(scale) {
        *v = ((*v as f64 / denom) as f32) * sc;
    }
}

/// Interleaved RoPE on one head; `pe` holds `[cos, -sin, sin, cos]` per pair.
pub(crate) fn rope(x: &mut [f32], pe: &[f32]) {
    for (pair, p) in x.chunks_exact_mut(2).zip(pe.chunks_exact(4)) {
        let (a, b) = (pair[0], pair[1]);
        pair[0] = p[0] * a + p[1] * b;
        pair[1] = p[2] * a + p[3] * b;
    }
}

#[inline]
pub(crate) fn silu(a: f32) -> f32 {
    a / (1.0 + (-a).exp())
}

/// Dot product with 8 independent lanes.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
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

/// `ggml_ext_timestep_embedding`: `t * time_factor` (f32), then `[cos(arg)..., sin(arg)...]` with
/// `freq_j = exp(-ln(max_period) * j / half)`. An odd `dim` gets a trailing zero.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32, time_factor: f32) -> Vec<f32> {
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

#[cfg(test)]
#[path = "full_ref_tests.rs"]
mod tests;
