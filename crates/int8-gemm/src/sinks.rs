//! Sink kernels: what happens to a GEMM's f32 output while it is still in cache (quantise to
//! int8 codes, SiLU gate, RMS-norm scale + RoPE). Each NEON kernel (csrc/i8g_sinks.c) has a scalar
//! definition here and is tested BITWISE against it.

/// The C kernels (csrc/i8g_sinks.c). Lengths are checked by the safe wrappers below.
pub mod raw {
    extern "C" {
        /// Quantise `n` floats (n % 8 == 0, n >= 8) as one chunk; codes of x[8s..8s+8) go to
        /// `dst + s * stride` (bytes). Returns the chunk scale.
        pub fn i8g_quant_chunk(x: *const f32, n: usize, dst: *mut i8, stride: isize) -> f32;
        /// `out[j] = silu(row[j]) * row[u + j]` for j in 0..u (u % 8 == 0). No aliasing.
        pub fn i8g_silu_gate(row: *const f32, u: usize, out: *mut f32);
        /// RMS-scale and RoPE for hd values (hd % 8 == 0); `pe` has 2*hd floats.
        pub fn i8g_rope_scale(
            row: *const f32,
            hd: usize,
            r: f32,
            scale: *const f32,
            pe: *const f32,
            vals: *mut f32,
        );
    }
}

/// Scalar reference of [`quant_chunk`]: one scale for all of `x`, codes at `dst[s * stride + i]`.
pub fn quant_chunk_ref(x: &[f32], dst: &mut [i8], stride: usize) -> f32 {
    let mut amax = 0.0f32;
    for &a in x {
        amax = amax.max(a.abs()); // f32::max ignores a NaN operand
    }
    let (scale, inv) = if amax > 0.0 && amax.is_finite() { (amax / 127.0, 127.0 / amax) } else { (0.0, 0.0) };
    for (s, piece) in x.chunks_exact(8).enumerate() {
        for i in 0..8 {
            dst[s * stride + i] = (piece[i] * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    scale
}

/// Scalar reference of the crate's fast exp (polynomial on the reduced argument, 2^n from bits).
pub fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_2e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 1.666_666_5e-1)
        * r
        + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

fn silu(a: f32) -> f32 {
    a / (1.0 + exp_fast(-a))
}

/// Scalar reference of [`silu_gate`]: `out[j] = silu(row[j]) * row[u + j]`, u = out.len().
pub fn silu_gate_ref(row: &[f32], out: &mut [f32]) {
    let u = out.len();
    for j in 0..u {
        out[j] = silu(row[j]) * row[u + j];
    }
}

/// Scalar reference of [`rope_scale`], written pair by pair as in the kernel contract.
pub fn rope_scale_ref(row: &[f32], r: f32, scale: &[f32], pe: &[f32], vals: &mut [f32]) {
    let hd = row.len();
    for i in 0..hd / 2 {
        let a = (row[2 * i] * r) * scale[2 * i];
        let b = (row[2 * i + 1] * r) * scale[2 * i + 1];
        vals[2 * i] = pe[4 * i] * a + pe[4 * i + 1] * b;
        vals[2 * i + 1] = pe[4 * i + 2] * a + pe[4 * i + 3] * b;
    }
}

/// Quantise `x` (one chunk, one scale) to int8 codes; codes of `x[8s..8s+8)` go to
/// `dst[s * stride ..]`. Returns the scale. Requires `x.len() % 8 == 0`, `x.len() >= 8`,
/// `stride >= 8`, `dst.len() >= (x.len()/8 - 1) * stride + 8`.
pub fn quant_chunk(x: &[f32], dst: &mut [i8], stride: usize) -> f32 {
    let n = x.len();
    assert!(n >= 8 && n % 8 == 0, "quant_chunk: n must be a multiple of 8 and >= 8");
    assert!(stride >= 8, "quant_chunk: stride must be >= 8");
    assert!(dst.len() >= (n / 8 - 1) * stride + 8, "quant_chunk: dst too short");
    // SAFETY: lengths checked above; x and dst are valid for the reads/writes the kernel performs.
    unsafe { raw::i8g_quant_chunk(x.as_ptr(), n, dst.as_mut_ptr(), stride as isize) }
}

/// `out[j] = silu(row[j]) * row[out.len() + j]`. Requires `out.len() % 8 == 0` and
/// `row.len() == 2 * out.len()`.
pub fn silu_gate(row: &[f32], out: &mut [f32]) {
    let u = out.len();
    assert!(u % 8 == 0, "silu_gate: u must be a multiple of 8");
    assert_eq!(row.len(), 2 * u, "silu_gate: row must hold 2*u floats");
    // SAFETY: lengths checked; `row` and `out` cannot alias (shared vs exclusive borrow).
    unsafe { raw::i8g_silu_gate(row.as_ptr(), u, out.as_mut_ptr()) }
}

/// RMS-scale + RoPE of `row` (hd = row.len(), hd % 8 == 0). `scale` has hd floats, `pe` has 2*hd.
pub fn rope_scale(row: &[f32], r: f32, scale: &[f32], pe: &[f32], vals: &mut [f32]) {
    let hd = row.len();
    assert!(hd % 8 == 0, "rope_scale: hd must be a multiple of 8");
    assert_eq!(scale.len(), hd, "rope_scale: scale length");
    assert_eq!(pe.len(), 2 * hd, "rope_scale: pe length");
    assert_eq!(vals.len(), hd, "rope_scale: vals length");
    // SAFETY: lengths checked above.
    unsafe { raw::i8g_rope_scale(row.as_ptr(), hd, r, scale.as_ptr(), pe.as_ptr(), vals.as_mut_ptr()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// xorshift64 (no external crates).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    const LENS: [usize; 6] = [8, 16, 64, 128, 256, 3072];
    const MAGS: [f32; 5] = [1e-6, 1e-3, 1.0, 1e3, 1e6];
    const SPECIALS: [f32; 16] = [
        0.0,
        -0.0,
        100.0,
        -100.0,
        87.0,
        -87.0,
        88.0,
        -88.0,
        1e-30,
        -1e-30,
        1e-40, // denormal
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        3.0e38,
    ];

    /// Bitwise comparison of every lane. One deliberate exception: when BOTH sides are NaN the sign
    /// and payload are not compared. IEEE 754 leaves the choice of NaN operand unspecified when two
    /// NaNs of opposite sign meet in one multiply/add, and LLVM may commute the scalar operands
    /// differently from the NEON code (observed: +NaN vs -NaN in rope_scale with NaN inputs of both
    /// signs). A NaN on one side only is still an error, and all finite/inf lanes are exact.
    fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{}: length", what);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            if g.is_nan() && w.is_nan() {
                continue;
            }
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{}: index {}: got {:?} ({:#010x}), want {:?} ({:#010x})",
                what,
                i,
                g,
                g.to_bits(),
                w,
                w.to_bits()
            );
        }
    }

    /// Guard byte pattern for the gaps of a strided destination.
    fn guard(p: usize) -> i8 {
        ((p * 37) % 251) as u8 as i8
    }

    fn quant_case(what: &str, x: &[f32], stride: usize) {
        let nchunks = x.len() / 8;
        let used = (nchunks - 1) * stride + 8;
        let total = used + 16; // trailing guard, catches overruns
        let init: Vec<i8> = (0..total).map(guard).collect();
        let mut want = init.clone();
        let mut got = init.clone();
        let want_scale = quant_chunk_ref(x, &mut want[..used], stride);
        let got_scale = quant_chunk(x, &mut got[..used], stride);
        assert_eq!(got_scale.to_bits(), want_scale.to_bits(), "{}: scale got {} want {}", what, got_scale, want_scale);
        assert_eq!(got, want, "{}: codes (stride {})", what, stride);
        for p in 0..total {
            let is_code = p < used && p / stride < nchunks && p % stride < 8;
            if !is_code {
                assert_eq!(got[p], init[p], "{}: guard byte {} modified (stride {})", what, p, stride);
            }
        }
    }

    #[test]
    fn quant_chunk_matches_scalar_random() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for &n in &LENS {
            for &m in &MAGS {
                for &stride in &[8usize, 64] {
                    for _ in 0..4 {
                        let x: Vec<f32> = (0..n).map(|_| rng.unit() * m).collect();
                        quant_case(&format!("random n={} m={}", n, m), &x, stride);
                    }
                }
            }
        }
    }

    #[test]
    fn quant_chunk_special_values() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for &n in &LENS {
            for &stride in &[8usize, 64] {
                // all zero (scale 0), negative zero mixed in
                quant_case("all +0", &vec![0.0f32; n], stride);
                let mut z = vec![0.0f32; n];
                for (i, v) in z.iter_mut().enumerate() {
                    if i % 3 == 0 {
                        *v = -0.0;
                    }
                }
                quant_case("+0/-0 mixed", &z, stride);
                // all NaN (scale 0, codes 0)
                quant_case("all NaN", &vec![f32::NAN; n], stride);
                quant_case("all -NaN", &vec![-f32::NAN; n], stride);
                // all inf / -inf (amax inf -> scale 0)
                quant_case("all inf", &vec![f32::INFINITY; n], stride);
                // NaN lanes mixed with finite values (NaN ignored by amax, code 0)
                let mut nan_mix: Vec<f32> = (0..n).map(|_| rng.unit() * 5.0).collect();
                for i in (0..n).step_by(5) {
                    nan_mix[i] = if i % 2 == 0 { f32::NAN } else { -f32::NAN };
                }
                quant_case("NaN lanes", &nan_mix, stride);
                // inf lanes mixed with finite values (scale 0, every code 0 or NaN->0)
                let mut inf_mix: Vec<f32> = (0..n).map(|_| rng.unit() * 5.0).collect();
                for i in (0..n).step_by(7) {
                    inf_mix[i] = if i % 2 == 0 { f32::INFINITY } else { f32::NEG_INFINITY };
                }
                quant_case("inf lanes", &inf_mix, stride);
                // huge and denormal magnitudes
                let huge: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 3.0e38 } else { -2.5e38 }).collect();
                quant_case("huge", &huge, stride);
                let tiny: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1e-40 } else { -3e-41 }).collect();
                quant_case("denormal", &tiny, stride);
                // a mix of specials and random values
                let mixed: Vec<f32> = (0..n)
                    .map(|_| {
                        if rng.below(4) == 0 {
                            SPECIALS[rng.below(SPECIALS.len())]
                        } else {
                            rng.unit() * 1000.0
                        }
                    })
                    .collect();
                quant_case("specials mixed", &mixed, stride);
                // exact .5 ties: amax = 127 gives inv = 1, so x = k + 0.5 is an exact tie
                let ties: Vec<f32> = (0..n)
                    .map(|i| match i % 8 {
                        0 => 127.0,
                        1 => 0.5,
                        2 => 1.5,
                        3 => -2.5,
                        4 => 3.5,
                        5 => -0.5,
                        6 => 126.5,
                        _ => -126.5,
                    })
                    .collect();
                quant_case("ties", &ties, stride);
            }
        }
    }

    #[test]
    fn quant_chunk_tie_rounding_is_half_away_from_zero() {
        let x = [127.0f32, 0.5, 1.5, -2.5, 3.5, -0.5, 126.5, -126.5];
        let mut dst = [0i8; 8];
        let scale = quant_chunk(&x, &mut dst, 8);
        assert_eq!(scale.to_bits(), 1.0f32.to_bits());
        assert_eq!(dst, [127, 1, 2, -3, 4, -1, 127, -127]);
    }

    #[test]
    #[should_panic(expected = "stride")]
    fn quant_chunk_rejects_small_stride() {
        let x = [1.0f32; 16];
        let mut dst = [0i8; 32];
        quant_chunk(&x, &mut dst, 4);
    }

    fn silu_case(what: &str, row: &[f32]) {
        let u = row.len() / 2;
        let mut want = vec![0.0f32; u];
        let mut got = vec![0.0f32; u];
        silu_gate_ref(row, &mut want);
        silu_gate(row, &mut got);
        assert_bits(what, &got, &want);
    }

    #[test]
    fn silu_gate_matches_scalar() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        for &u in &LENS {
            for &m in &MAGS {
                for _ in 0..4 {
                    let row: Vec<f32> = (0..2 * u).map(|_| rng.unit() * m).collect();
                    silu_case(&format!("silu random u={} m={}", u, m), &row);
                }
            }
            // every special value in both halves, plus random values with specials mixed in
            let row: Vec<f32> = (0..2 * u).map(|i| SPECIALS[i % SPECIALS.len()]).collect();
            silu_case("silu specials (rotating)", &row);
            let row: Vec<f32> = (0..2 * u).map(|i| SPECIALS[(i / 2) % SPECIALS.len()]).collect();
            silu_case("silu specials (gate side)", &row);
            for _ in 0..8 {
                let row: Vec<f32> = (0..2 * u)
                    .map(|_| {
                        if rng.below(3) == 0 {
                            SPECIALS[rng.below(SPECIALS.len())]
                        } else {
                            rng.unit() * 120.0
                        }
                    })
                    .collect();
                silu_case("silu specials mixed", &row);
            }
            // large |a| around the exp clamp, both signs
            let row: Vec<f32> = (0..2 * u)
                .map(|i| [100.0, -100.0, 87.0, -87.0, 88.0, -88.0, 86.9, -86.9][i % 8] * (1.0 + (i % 3) as f32))
                .collect();
            silu_case("silu clamp region", &row);
        }
    }

    fn rope_case(what: &str, row: &[f32], r: f32, scale: &[f32], pe: &[f32]) {
        let mut want = vec![0.0f32; row.len()];
        let mut got = vec![0.0f32; row.len()];
        rope_scale_ref(row, r, scale, pe, &mut want);
        rope_scale(row, r, scale, pe, &mut got);
        assert_bits(what, &got, &want);
    }

    #[test]
    fn rope_scale_matches_scalar() {
        let mut rng = Rng(0x5151_7A7A_9393_1010);
        let rs = [1.0f32, -0.0, 0.0, 3.7e-3, 1e4, -2.5, f32::NAN, f32::INFINITY];
        for &hd in &LENS {
            for &m in &MAGS {
                for &r in &rs {
                    let row: Vec<f32> = (0..hd).map(|_| rng.unit() * m).collect();
                    let scale: Vec<f32> = (0..hd).map(|_| rng.unit().abs() * 2.0).collect();
                    let pe: Vec<f32> = (0..2 * hd).map(|_| rng.unit()).collect();
                    rope_case(&format!("rope hd={} m={} r={}", hd, m, r), &row, r, &scale, &pe);
                }
            }
            // specials in row, scale and pe
            let mut row: Vec<f32> = (0..hd).map(|_| rng.unit() * 10.0).collect();
            let mut scale: Vec<f32> = (0..hd).map(|_| rng.unit()).collect();
            let mut pe: Vec<f32> = (0..2 * hd).map(|_| rng.unit()).collect();
            for i in 0..hd {
                if i % 5 == 0 {
                    row[i] = SPECIALS[i % SPECIALS.len()];
                }
                if i % 7 == 0 {
                    scale[i] = SPECIALS[(i + 3) % SPECIALS.len()];
                }
            }
            for i in 0..2 * hd {
                if i % 9 == 0 {
                    pe[i] = SPECIALS[(i + 5) % SPECIALS.len()];
                }
            }
            rope_case("rope specials", &row, 0.125, &scale, &pe);
            rope_case("rope specials r=NaN", &row, f32::NAN, &scale, &pe);
            // exact ties and zeros
            let zero = vec![0.0f32; hd];
            let ones = vec![1.0f32; hd];
            let pe1 = vec![1.0f32; 2 * hd];
            rope_case("rope zeros", &zero, 2.0, &ones, &pe1);
            rope_case("rope -0", &vec![-0.0f32; hd], -1.0, &ones, &pe1);
        }
    }

    #[test]
    fn exp_fast_is_sane() {
        // Spot checks of the scalar reference itself (the kernels are tested against it bitwise).
        assert_eq!(exp_fast(0.0).to_bits(), 1.0f32.to_bits());
        assert!((exp_fast(1.0) - std::f32::consts::E).abs() < 1e-5);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-5);
    }
}
