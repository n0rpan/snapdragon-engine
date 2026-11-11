//! GGUF tensor mapping for FLUX.2 klein single-stream blocks: dims from tensor shapes, Q8_0
//! access, dequantisation to f32, and the matching quantiser used by tests and synthetic runs.
//!
//! The klein GGUF has no metadata. GGUF dims are `[in, out]` and data is row-major by output row:
//! `linear1` is `[hidden, 3*hidden + 2*mlp]`, `linear2` is `[hidden + mlp, hidden]`.

use crate::types::{BlockWeightsF32, Dims};
use gguf::{Gguf, Header, TensorInfo};
use int8_gemm::f16_to_f32;

/// Block shapes from the first single block's tensor shapes.
pub fn dims_from_header(h: &Header) -> Result<Dims, String> {
    let l1 = req(h, "single_blocks.0.linear1.weight")?;
    let l2 = req(h, "single_blocks.0.linear2.weight")?;
    let qn = req(h, "single_blocks.0.norm.query_norm.scale")?;
    if l1.shape.len() != 2 || l2.shape.len() != 2 {
        return Err(format!(
            "linear weights must be 2-D, got {:?} and {:?}",
            l1.shape, l2.shape
        ));
    }
    let hidden = l1.shape[0] as usize;
    let lin1_out = l1.shape[1] as usize;
    let lin2_in = l2.shape[0] as usize;
    let lin2_out = l2.shape[1] as usize;
    let head_dim = qn.elements() as usize;
    if hidden == 0 || head_dim == 0 || head_dim % 2 != 0 || hidden % head_dim != 0 {
        return Err(format!(
            "bad attention shape: hidden {hidden}, head_dim {head_dim} (need head_dim even and dividing hidden)"
        ));
    }
    if lin2_in < hidden {
        return Err(format!("linear2 input {lin2_in} is smaller than hidden {hidden}"));
    }
    let mlp = lin2_in - hidden;
    if lin1_out != 3 * hidden + 2 * mlp {
        return Err(format!(
            "linear1 output {lin1_out} != 3*hidden + 2*mlp = 3*{hidden} + 2*{mlp}"
        ));
    }
    if lin2_out != hidden {
        return Err(format!("linear2 output {lin2_out} != hidden {hidden}"));
    }
    Ok(Dims { hidden, heads: hidden / head_dim, head_dim, mlp })
}

/// Number of single blocks: consecutive indices from 0 that have a `linear1.weight` tensor.
pub fn count_single_blocks(h: &Header) -> usize {
    (0usize..)
        .take_while(|i| h.tensor(&format!("single_blocks.{i}.linear1.weight")).is_some())
        .count()
}

/// One single block's quantised linears (borrowed from the mmap) and f32 norm scales.
pub struct SingleBlockRaw<'a> {
    pub dims: Dims,
    pub linear1_q8: &'a [u8],
    pub linear2_q8: &'a [u8],
    /// (absolute file offset of the tensor data, byte length).
    pub linear1_range: (u64, u64),
    pub linear2_range: (u64, u64),
    pub q_scale: Vec<f32>,
    pub k_scale: Vec<f32>,
}

/// Map single block `idx` of an opened klein GGUF.
pub fn single_block_raw<'a>(g: &'a Gguf, idx: usize) -> Result<SingleBlockRaw<'a>, String> {
    let h: &'a Header = &g.header;
    let dims = dims_from_header(h)?;
    let t1 = req(h, &format!("single_blocks.{idx}.linear1.weight"))?;
    let t2 = req(h, &format!("single_blocks.{idx}.linear2.weight"))?;
    check_ty(t1, "Q8_0")?;
    check_ty(t2, "Q8_0")?;
    let want1 = vec![dims.hidden as u64, dims.lin1_out() as u64];
    let want2 = vec![dims.lin2_in() as u64, dims.hidden as u64];
    if t1.shape != want1 {
        return Err(format!("{}: shape {:?}, want {:?}", t1.name, t1.shape, want1));
    }
    if t2.shape != want2 {
        return Err(format!("{}: shape {:?}, want {:?}", t2.name, t2.shape, want2));
    }
    let qn = req(h, &format!("single_blocks.{idx}.norm.query_norm.scale"))?;
    let kn = req(h, &format!("single_blocks.{idx}.norm.key_norm.scale"))?;
    let q_scale = read_f32_vec(g, qn, dims.head_dim)?;
    let k_scale = read_f32_vec(g, kn, dims.head_dim)?;
    // The mmap base plus data_offset plus the tensor offset is the absolute file offset.
    let abs = |t: &TensorInfo| (h.data_offset + t.offset, t.bytes());
    Ok(SingleBlockRaw {
        dims,
        linear1_q8: g.tensor_data(t1),
        linear2_q8: g.tensor_data(t2),
        linear1_range: abs(t1),
        linear2_range: abs(t2),
        q_scale,
        k_scale,
    })
}

/// Dequantise a whole block to the f32 layout `BlockWeightsF32` expects.
pub fn block_weights_f32(raw: &SingleBlockRaw) -> BlockWeightsF32 {
    let d = raw.dims;
    BlockWeightsF32 {
        dims: d,
        w1: dequant_q8_0(raw.linear1_q8, d.lin1_out(), d.hidden),
        w2: dequant_q8_0(raw.linear2_q8, d.hidden, d.lin2_in()),
        q_scale: raw.q_scale.clone(),
        k_scale: raw.k_scale.clone(),
    }
}

/// Q8_0 dequantisation. Block = 2-byte little-endian half `d`, then 32 int8 codes (34 bytes);
/// a row of `k` values is `k/32` blocks; value = `d * code`.
pub fn dequant_q8_0(src: &[u8], rows: usize, k: usize) -> Vec<f32> {
    assert!(k % 32 == 0, "Q8_0 row length {k} is not a multiple of 32");
    let blocks = rows * (k / 32);
    assert_eq!(src.len(), blocks * 34, "Q8_0 byte length for {rows}x{k}");
    let mut out = Vec::with_capacity(rows * k);
    for blk in src.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        out.extend(blk[2..].iter().map(|&c| d * (c as i8) as f32));
    }
    out
}

/// Q8_0 quantisation (inverse of [`dequant_q8_0`]). Per 32-block: `d = amax/127` stored as a
/// half; codes are `round(x / d_decoded)` clamped to [-127, 127]. An all-zero block gets `d = 0`
/// and zero codes.
pub fn quant_q8_0(rows_f32: &[f32], rows: usize, k: usize) -> Vec<u8> {
    assert!(k % 32 == 0, "Q8_0 row length {k} is not a multiple of 32");
    assert_eq!(rows_f32.len(), rows * k, "quant_q8_0 input length");
    let mut out = Vec::with_capacity(rows * (k / 32) * 34);
    for blk in rows_f32.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d_half = f32_to_f16(amax / 127.0);
        let d = f16_to_f32(d_half);
        out.extend_from_slice(&d_half.to_le_bytes());
        for &v in blk {
            let c = if d == 0.0 { 0i8 } else { (v / d).round().clamp(-127.0, 127.0) as i8 };
            out.push(c as u8);
        }
    }
    out
}

/// f32 to IEEE half, round to nearest even. Overflow gives infinity, tiny values give subnormals
/// or zero, NaN stays NaN.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x007f_ffff;
    if exp == 0xff {
        return if man == 0 { sign | 0x7c00 } else { sign | 0x7e00 };
    }
    if exp == 0 {
        // f32 zero or subnormal: far below the smallest half subnormal (2^-24).
        return sign;
    }
    // Half exponent field before the bias is applied: (exp - 127) + 15.
    let e = exp - 112;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e >= 1 {
        let mut half_man = man >> 13;
        let rem = man & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (half_man & 1) == 1) {
            half_man += 1;
        }
        // A mantissa carry bumps the exponent; exponent 31 with zero mantissa is infinity.
        return sign | ((((e as u32) << 10) + half_man) as u16);
    }
    // Subnormal target: value = (man | 2^23) * 2^(exp-150) = half_mantissa * 2^-24.
    let m24 = man | 0x0080_0000;
    let shift = (126 - exp) as u32;
    if shift > 24 {
        return sign;
    }
    let mut q = m24 >> shift;
    let rem = m24 & ((1u32 << shift) - 1);
    let half = 1u32 << (shift - 1);
    if rem > half || (rem == half && (q & 1) == 1) {
        q += 1;
    }
    sign | (q as u16)
}

fn req<'a>(h: &'a Header, name: &str) -> Result<&'a TensorInfo, String> {
    h.tensor(name).ok_or_else(|| format!("missing tensor {name}"))
}

fn check_ty(t: &TensorInfo, want: &str) -> Result<(), String> {
    if t.ty.name() != want {
        return Err(format!("{}: type {}, want {want}", t.name, t.ty.name()));
    }
    Ok(())
}

/// Read a small norm tensor as f32; F32, F16 and BF16 are accepted.
fn read_f32_vec(g: &Gguf, t: &TensorInfo, len: usize) -> Result<Vec<f32>, String> {
    let bytes = g.tensor_data(t);
    let out: Vec<f32> = match t.ty.name() {
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "F16" => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        // BF16 is the upper 16 bits of the f32 pattern.
        "BF16" => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        other => return Err(format!("{}: unsupported norm type {other}", t.name)),
    };
    if out.len() != len {
        return Err(format!("{}: {} elements, want {len}", t.name, out.len()));
    }
    Ok(out)
}
