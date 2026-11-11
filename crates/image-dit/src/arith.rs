//! Activation arithmetic of the reference paths (W5-G arithmetic-parity control).
//!
//! The f32 reference multiplies dequantised weights with f32 activations. sd.cpp's CPU graph does
//! not: ggml converts a matmul's f32 activations to the weight type's `vec_dot_type` first
//! (`ggml-cpu.c` type traits): Q8_0 weights take Q8_0 activations (32 values per block, one f16
//! scale, `ggml-cpu/arch/arm/quants.c` `quantize_row_q8_0`; the repacked 4x4/4x8 paths use the
//! same per-block maths), BF16 weights take BF16 activations (round to nearest even), F32 weights
//! take F32. [`Arith::SdParity`] rounds the reference's activations the same way, so the remaining
//! difference to sd.cpp is summation order plus whatever is a real port difference.
//!
//! Rounding an activation to `q * f16(d)` and taking an f32 dot with the dequantised weight gives
//! sd.cpp's `sum_b (d_w * d_x) * sum_i q_w q_x` up to the order of the float additions.
//! Not emulated (none of these is used by klein with `--flash-attn off`): F16 K/V of flash
//! attention, sage attention, KleidiAI kernels.

/// Activation arithmetic of a reference forward.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Arith {
    /// Plain f32 activations (the original reference; bitwise unchanged).
    #[default]
    F32,
    /// sd.cpp CPU parity: Q8_0 activations into Q8_0 weights, BF16 activations into BF16 weights.
    SdParity,
}

impl Arith {
    pub fn parse(s: &str) -> Result<Arith, String> {
        match s {
            "f32" => Ok(Arith::F32),
            "sdcpp" | "sd-parity" | "parity" => Ok(Arith::SdParity),
            other => Err(format!("--arith must be f32 or sdcpp, got '{other}'")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Arith::F32 => "f32",
            Arith::SdParity => "sdcpp",
        }
    }
}

/// Storage type of a weight matrix, which decides how sd.cpp rounds its activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WType {
    Q8_0,
    Bf16,
    F16,
    F32,
}

impl WType {
    /// From a GGUF type name; unknown or absent types count as F32 (no rounding).
    pub fn from_name(name: Option<&str>) -> WType {
        match name {
            Some("Q8_0") => WType::Q8_0,
            Some("BF16") => WType::Bf16,
            Some("F16") => WType::F16,
            _ => WType::F32,
        }
    }
}

/// Rounds one activation row in place the way sd.cpp would before a matmul with a `wt` weight.
/// No-op for [`Arith::F32`] and for F32 weights. Q8_0 needs `x.len() % 32 == 0`.
pub fn round_activation(arith: Arith, wt: WType, x: &mut [f32]) {
    if arith == Arith::F32 {
        return;
    }
    match wt {
        WType::Q8_0 => round_q8_0(x),
        WType::Bf16 => round_bf16(x),
        WType::F16 => x.iter_mut().for_each(|v| *v = f16_bits_to_f32(f32_to_f16_bits(*v))),
        WType::F32 => {}
    }
}

/// `quantize_row_q8_0` (ARM NEON) then dequantise: per 32 values, `d = amax / 127` (f32),
/// `q = round_half_even(x * (1/d))`, stored scale `f16(d)`; each value becomes `q * f16(d)`.
pub fn round_q8_0(x: &mut [f32]) {
    assert!(x.len() % 32 == 0, "Q8_0 rounding needs whole 32-value blocks, got {}", x.len());
    for blk in x.chunks_exact_mut(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let dh = f16_bits_to_f32(f32_to_f16_bits(d));
        for v in blk.iter_mut() {
            let q = (*v * id).round_ties_even();
            *v = q * dh;
        }
    }
}

/// f32 -> bf16 -> f32 with ggml's rounding (`ggml_compute_fp32_to_bf16`: nearest even, NaN quiet).
pub fn round_bf16(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = f32::from_bits((f32_to_bf16_bits(*v) as u32) << 16);
    }
}

pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 64) as u16;
    }
    ((u + (0x7fff + ((u >> 16) & 1))) >> 16) as u16
}

/// IEEE f32 -> f16 bits, round to nearest even (what `vcvt_f16_f32` / F16C do).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = b & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        return sign | 0x7c00 | if a > 0x7f80_0000 { 0x200 } else { 0 };
    }
    if a >= 0x477f_f000 {
        return sign | 0x7c00; // >= 65520 rounds to infinity
    }
    if a < 0x3880_0000 {
        // below 2^-14: f16 subnormal (or zero), unit 2^-24; the scaling is exact
        let q = (f32::from_bits(a) * 16_777_216.0).round_ties_even() as u16;
        return sign | q;
    }
    let e = ((a >> 23) as i32 - 127 + 15) as u32;
    let m = a & 0x7f_ffff;
    let mut h = (e << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// IEEE f16 bits -> f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    int8_gemm::f16_to_f32(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_and_rounding() {
        for v in [0.0f32, 1.0, -2.5, 65504.0, 6.103_515_6e-5, 5.960_464_5e-8, 0.333_333_34] {
            let r = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!((r - v).abs() <= v.abs() * 1e-3 + 6e-8, "{v} -> {r}");
        }
        // exact f16 values are fixed points (normal and subnormal)
        for h in [0x3c00u16, 0x0001, 0x03ff, 0x0400, 0x7bff, 0xbc00] {
            assert_eq!(f32_to_f16_bits(f16_bits_to_f32(h)), h);
        }
        // ties go to even: 1 + 2^-11 lies halfway between 1 and 1 + 2^-10
        assert_eq!(f32_to_f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        // 1 + 2^-8 is halfway between bf16 1.0 and 1 + 2^-7: even -> 1.0
        assert_eq!(f32_to_bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        let mut x = [3.141_592_7f32, -0.1];
        round_bf16(&mut x);
        assert_eq!(x, [3.140625, -0.10009765625]);
    }

    #[test]
    fn q8_0_matches_ggml_block_maths() {
        // amax 127 -> d = 1 exactly, so values round to integers (half to even)
        let mut x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0 + 0.5).collect();
        x[31] = 127.0;
        let mut r = x.clone();
        round_q8_0(&mut r);
        for (i, (&a, &b)) in x.iter().zip(&r).enumerate() {
            if i == 31 {
                assert_eq!(b, 127.0);
            } else {
                assert_eq!(b, a.round_ties_even(), "index {i}");
            }
        }
        // zero block stays zero; the error of a general block is at most d/2 per value
        let mut z = vec![0f32; 32];
        round_q8_0(&mut z);
        assert!(z.iter().all(|&v| v == 0.0));
        let mut s: Vec<f32> = (0..64).map(|i| ((i * 37 % 64) as f32 - 31.7) * 0.013).collect();
        let orig = s.clone();
        round_q8_0(&mut s);
        for (blk_o, blk_r) in orig.chunks(32).zip(s.chunks(32)) {
            let amax = blk_o.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            for (a, b) in blk_o.iter().zip(blk_r) {
                // |x - q d| <= d/2 from the rounding, plus |q| |f16(d) - d| <= 127 d 2^-11 from the f16 scale
                assert!((a - b).abs() <= 0.5 * d + 127.0 * d / 2048.0 + 1e-7, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn f32_mode_is_a_no_op() {
        let mut x: Vec<f32> = (0..64).map(|i| (i as f32).sin() * 3.3).collect();
        let orig = x.clone();
        round_activation(Arith::F32, WType::Q8_0, &mut x);
        round_activation(Arith::F32, WType::Bf16, &mut x);
        round_activation(Arith::SdParity, WType::F32, &mut x);
        assert_eq!(x, orig);
    }
}
