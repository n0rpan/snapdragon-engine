//! Weight views. Linears stay in their file format (borrowed from the GGUF mmap) and are read one
//! row at a time; nothing makes a full f32 copy of the model. Norm vectors are small and kept in f32.

use crate::config::Qwen3Config;
use gguf::{GgmlType, Gguf, TensorInfo};
use kernels_sys::QType;
use oxn::linear::WeightRows;
use std::borrow::Cow;

/// One `o x k` row-major matrix (row = output channel), in its storage format.
pub enum Mat<'a> {
    F32 { o: usize, k: usize, data: Cow<'a, [f32]> },
    F16 { o: usize, k: usize, bytes: &'a [u8] },
    Quant { ty: QType, o: usize, k: usize, bytes: &'a [u8] },
}

impl<'a> Mat<'a> {
    pub fn o(&self) -> usize {
        self.shape().0
    }
    pub fn k(&self) -> usize {
        self.shape().1
    }
    /// The raw GGUF Q8_0 bytes, when this matrix is Q8_0 (the oxn W8 format takes them as they are).
    pub fn q8_0(&self) -> Option<&[u8]> {
        match self {
            Mat::Quant { ty: QType::Q8_0, bytes, .. } => Some(bytes),
            _ => None,
        }
    }

    pub fn from_tensor(t: &TensorInfo, bytes: &'a [u8]) -> Result<Mat<'a>, String> {
        if t.shape.len() != 2 {
            return Err(format!("{}: {}-D, want a 2-D matrix", t.name, t.shape.len()));
        }
        let (k, o) = (t.shape[0] as usize, t.shape[1] as usize);
        let want = |bpe_num: usize, bpe_den: usize| o * k * bpe_num / bpe_den;
        let m = match t.ty {
            GgmlType::F32 => {
                if bytes.len() != want(4, 1) || bytes.as_ptr() as usize % 4 != 0 {
                    return Err(format!("{}: F32 size/alignment", t.name));
                }
                // SAFETY: length and alignment checked; f32 has no invalid bit patterns.
                let s = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, o * k) };
                Mat::F32 { o, k, data: Cow::Borrowed(s) }
            }
            GgmlType::F16 => {
                if bytes.len() != want(2, 1) {
                    return Err(format!("{}: F16 size", t.name));
                }
                Mat::F16 { o, k, bytes }
            }
            ty => {
                let qt = match ty {
                    GgmlType::Q8_0 => QType::Q8_0,
                    GgmlType::Q4_K => QType::Q4K,
                    GgmlType::Q5_K => QType::Q5K,
                    GgmlType::Q6_K => QType::Q6K,
                    GgmlType::Q3_K => QType::Q3K,
                    other => return Err(format!("{}: unsupported type {}", t.name, other.name())),
                };
                if k % qt.qk() != 0 || bytes.len() != o * qt.row_bytes(k) {
                    return Err(format!("{}: quantized size mismatch", t.name));
                }
                Mat::Quant { ty: qt, o, k, bytes }
            }
        };
        Ok(m)
    }
}

impl WeightRows for Mat<'_> {
    fn shape(&self) -> (usize, usize) {
        match self {
            Mat::F32 { o, k, .. } | Mat::F16 { o, k, .. } | Mat::Quant { o, k, .. } => (*o, *k),
        }
    }
    fn read_row(&self, row: usize, dst: &mut [f32]) -> Result<(), String> {
        let (o, k) = self.shape();
        if row >= o || dst.len() != k {
            return Err(format!("row {row} of {o}, dst {} for k {k}", dst.len()));
        }
        match self {
            Mat::F32 { data, .. } => dst.copy_from_slice(&data[row * k..(row + 1) * k]),
            Mat::F16 { bytes, .. } => {
                for (i, d) in dst.iter_mut().enumerate() {
                    let p = 2 * (row * k + i);
                    *d = oxn::tiles::half_to_f32(u16::from_le_bytes([bytes[p], bytes[p + 1]]));
                }
            }
            Mat::Quant { ty, bytes, .. } => {
                let rb = ty.row_bytes(k);
                kernels_sys::dequantize_row(*ty, &bytes[row * rb..(row + 1) * rb], dst);
            }
        }
        Ok(())
    }
}

/// The seven linears of a block, in the order the forward uses them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
}

impl Kind {
    pub const ALL: [Kind; 7] = [Kind::Q, Kind::K, Kind::V, Kind::O, Kind::Gate, Kind::Up, Kind::Down];
    pub fn gguf(self) -> &'static str {
        match self {
            Kind::Q => "attn_q",
            Kind::K => "attn_k",
            Kind::V => "attn_v",
            Kind::O => "attn_output",
            Kind::Gate => "ffn_gate",
            Kind::Up => "ffn_up",
            Kind::Down => "ffn_down",
        }
    }
}

pub struct Layer<'a> {
    pub attn_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// indexed by `Kind as usize`
    pub mats: Vec<Mat<'a>>,
}

impl<'a> Layer<'a> {
    pub fn mat(&self, k: Kind) -> &Mat<'a> {
        &self.mats[k as usize]
    }
}

pub struct TeWeights<'a> {
    pub cfg: Qwen3Config,
    /// `vocab x hidden`
    pub embd: Mat<'a>,
    /// only the blocks the spec needs (27 of 36 for klein)
    pub layers: Vec<Layer<'a>>,
}

impl<'a> TeWeights<'a> {
    /// Borrow the first `n_layers` blocks of a qwen3 GGUF. Shapes are checked against the config.
    pub fn from_gguf(g: &'a Gguf, n_layers: usize) -> Result<TeWeights<'a>, String> {
        let cfg = Qwen3Config::from_header(&g.header)?;
        if n_layers > cfg.n_layers {
            return Err(format!("text encoder: {n_layers} layers requested, the model has {}", cfg.n_layers));
        }
        let tensor = |name: &str| g.header.tensor(name).ok_or_else(|| format!("text encoder: missing {name}"));
        let vec = |name: &str, len: usize| -> Result<Vec<f32>, String> {
            let t = tensor(name)?;
            let m = Mat::from_tensor(&TensorInfo { shape: vec![t.shape[0], 1], ..t.clone() }, g.tensor_data(t))?;
            if m.k() != len || t.shape.len() != 1 {
                return Err(format!("text encoder: {name} has shape {:?}, want [{len}]", t.shape));
            }
            let mut v = vec![0f32; len];
            m.read_row(0, &mut v)?;
            Ok(v)
        };
        let mat = |name: &str| -> Result<Mat<'a>, String> {
            let t = g.header.tensor(name).ok_or_else(|| format!("text encoder: missing {name}"))?;
            Mat::from_tensor(t, g.tensor_data(t))
        };
        let embd = mat("token_embd.weight")?;
        if embd.shape() != (cfg.vocab, cfg.hidden) {
            return Err(format!("token_embd: {:?}, want ({}, {})", embd.shape(), cfg.vocab, cfg.hidden));
        }
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("blk.{l}.");
            let mut mats = Vec::with_capacity(7);
            for kd in Kind::ALL {
                let m = mat(&format!("{p}{}.weight", kd.gguf()))?;
                let want = expected_shape(&cfg, kd);
                if m.shape() != want {
                    return Err(format!("{p}{}: {:?}, want {want:?}", kd.gguf(), m.shape()));
                }
                mats.push(m);
            }
            layers.push(Layer {
                attn_norm: vec(&format!("{p}attn_norm.weight"), cfg.hidden)?,
                q_norm: vec(&format!("{p}attn_q_norm.weight"), cfg.head_dim)?,
                k_norm: vec(&format!("{p}attn_k_norm.weight"), cfg.head_dim)?,
                ffn_norm: vec(&format!("{p}ffn_norm.weight"), cfg.hidden)?,
                mats,
            });
        }
        Ok(TeWeights { cfg, embd, layers })
    }

    /// Bytes of the placed linears if they were fp16 (the NPU fp16 residency).
    pub fn linear_fp16_bytes(&self) -> u64 {
        self.layers.iter().flat_map(|l| l.mats.iter()).map(|m| (m.o() * m.k() * 2) as u64).sum()
    }
}

/// `(o, k)` of each linear.
pub fn expected_shape(c: &Qwen3Config, kd: Kind) -> (usize, usize) {
    match kd {
        Kind::Q => (c.q_dim(), c.hidden),
        Kind::K | Kind::V => (c.kv_dim(), c.hidden),
        Kind::O => (c.hidden, c.q_dim()),
        Kind::Gate | Kind::Up => (c.ff, c.hidden),
        Kind::Down => (c.hidden, c.ff),
    }
}
