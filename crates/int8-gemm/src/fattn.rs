//! Float flash-attention kernels (F-IMG-ATTN): q.k^T, softmax of a key chunk and p.v in f32, or with
//! f16 operands and f32 accumulation (FMLAL). Layouts are documented in csrc/i8g_fattn.c; the
//! driver lives in `image-dit` (`attn_float`). f16 values are passed as their raw `u16` bits.

/// The C kernels (csrc/i8g_fattn.c). See the C file for the exact layouts.
pub mod raw {
    extern "C" {
        pub fn i8g_fattn_qk_f32(qt: *const f32, kt: *const f32, hd: usize, steps: usize, s: *mut f32, ss: usize);
        pub fn i8g_fattn_qk_f16(qt: *const u16, kt: *const u16, hd: usize, steps: usize, s: *mut f32, ss: usize);
        pub fn i8g_fattn_softmax_f32(s: *const f32, kc: usize, nvalid: usize, m: f32, pt: *mut f32, qi: usize) -> f32;
        pub fn i8g_fattn_softmax_f16(s: *const f32, kc: usize, nvalid: usize, m: f32, pt: *mut u16, qi: usize) -> f32;
        pub fn i8g_fattn_pv_f32(pt: *const f32, vt: *const f32, kc: usize, hd: usize, vstride: usize, o: *mut f32);
        pub fn i8g_fattn_pv_f16(pt: *const u16, vt: *const u16, kc: usize, hd: usize, vstride: usize, o: *mut f32);
        pub fn i8g_fattn_to_f16(x: *const f32, n: usize, dst: *mut u16, stride: usize);
        pub fn i8g_fattn_from_f16(src: *const u16, n: usize, x: *mut f32);
    }
}

/// f32 -> f16 bits, round to nearest even (overflow -> inf), scalar definition of the NEON `vcvt`.
pub fn f16_bits_ref(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // subnormal f16 (or zero): value = man_full * 2^(exp-150); f16 subnormal unit 2^-24
        if e < -10 {
            return sign;
        }
        let full = man | 0x80_0000;
        let shift = (14 - e) as u32; // 24-bit mantissa -> 10-bit subnormal field
        let q = full >> shift;
        let rem = full & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let q = if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q };
        return sign | q as u16;
    }
    let q = man >> 13;
    let rem = man & 0x1fff;
    let mut h = ((e as u32) << 10) | q;
    if rem > 0x1000 || (rem == 0x1000 && q & 1 == 1) {
        h += 1; // may carry into the exponent (and to inf): correct
    }
    sign | h as u16
}

/// f16 bits -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        // zero or subnormal: man * 2^-24 (exact in f32)
        let mag = man as f32 * (1.0 / 16_777_216.0);
        return if sign != 0 { -mag } else { mag };
    }
    let bits = if exp == 0x1f { sign | 0x7f80_0000 | (man << 13) } else { sign | ((exp + 127 - 15) << 23) | (man << 13) };
    f32::from_bits(bits)
}

/// [`f16_bits_ref`] after the saturation of `to_f16`: finite values beyond the f16 range become
/// +-65504; NaN and +-inf are kept (F-IMG-ATTN-b).
pub fn f16_bits_sat_ref(x: f32) -> u16 {
    if x.is_finite() {
        f16_bits_ref(x.clamp(-65504.0, 65504.0))
    } else {
        f16_bits_ref(x)
    }
}

/// `dst[i * stride] = f16(x[i])` (NEON conversion), SATURATING: finite values beyond the f16 range
/// become +-65504 instead of +-inf; NaN and +-inf pass through (see csrc/i8g_fattn.c).
pub fn to_f16(x: &[f32], dst: &mut [u16], stride: usize) {
    assert!(stride >= 1);
    if x.is_empty() {
        return;
    }
    assert!(dst.len() > (x.len() - 1) * stride);
    // SAFETY: bounds checked above.
    unsafe { raw::i8g_fattn_to_f16(x.as_ptr(), x.len(), dst.as_mut_ptr(), stride) }
}

#[cfg(test)]
#[path = "fattn_sat_tests.rs"]
mod sat_tests;

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    #[test]
    fn f16_conversion_matches_neon() {
        let mut rng = Rng(0xf16);
        let mut xs: Vec<f32> = vec![0.0, -0.0, 1.0, 65504.0, 65520.0, 1e9, -1e9, 6.0e-8, 2.9e-8, 3.0e-8, 5.96e-8, 6.1e-5, 1.0 / 3.0];
        for _ in 0..20000 {
            let e = (rng.next() % 40) as i32 - 28;
            xs.push(rng.unit() * 2f32.powi(e));
        }
        // ties at the f16 grid
        for k in 0..64u32 {
            xs.push(f32::from_bits(0x3f80_0000 | (k << 13) | 0x1000));
        }
        let mut got = vec![0u16; xs.len()];
        to_f16(&xs, &mut got, 1);
        for (i, &x) in xs.iter().enumerate() {
            assert_eq!(got[i], f16_bits_sat_ref(x), "x {x:e}");
            if got[i] & 0x7c00 != 0x7c00 {
                let back = f16_to_f32(got[i]);
                let mut one = [0f32];
                // SAFETY: one element each.
                unsafe { raw::i8g_fattn_from_f16(&got[i], 1, one.as_mut_ptr()) };
                assert_eq!(back.to_bits(), one[0].to_bits(), "back {x:e}");
            }
        }
        // strided
        let mut s = vec![0u16; 8 * 5];
        to_f16(&xs[..5], &mut s, 8);
        for i in 0..5 {
            assert_eq!(s[i * 8], f16_bits_sat_ref(xs[i]));
        }
    }

    /// q.k, softmax and p.v kernels against plain f64 sums on the same (rounded) operands.
    #[test]
    fn kernels_match_f64() {
        let (hd, kc, steps) = (32usize, 24usize, 3usize);
        let mut rng = Rng(77);
        let q: Vec<f32> = (0..8 * hd).map(|_| rng.unit()).collect(); // [qi][d]
        let k: Vec<f32> = (0..kc * hd).map(|_| rng.unit()).collect(); // [key][d]
        let v: Vec<f32> = (0..kc * hd).map(|_| rng.unit() * 3.0).collect(); // [key][d]
        for half in [false, true] {
            let rnd = |x: f32| if half { f16_to_f32(f16_bits_ref(x)) } else { x };
            // layouts
            let mut qt = vec![0f32; hd * 8];
            let mut kt = vec![0f32; steps * hd * 8];
            let mut vt = vec![0f32; (hd / 8) * kc * 8];
            for qi in 0..8 {
                for d in 0..hd {
                    qt[d * 8 + qi] = q[qi * hd + d];
                }
            }
            for key in 0..kc {
                for d in 0..hd {
                    kt[(key / 8) * hd * 8 + d * 8 + key % 8] = k[key * hd + d];
                    vt[(d / 8) * kc * 8 + key * 8 + d % 8] = v[key * hd + d];
                }
            }
            let ss = kc + 8;
            let mut s = vec![0f32; 8 * ss];
            let mut o = vec![0.5f32; 8 * hd]; // accumulates onto existing values
            let mut sums = [0f32; 8];
            let nvalid = kc - 3;
            let m = 0.7f32;
            if half {
                let h = |x: &[f32]| x.iter().map(|&a| f16_bits_ref(a)).collect::<Vec<u16>>();
                let (qh, kh, vh) = (h(&qt), h(&kt), h(&vt));
                let mut pt = vec![0u16; kc * 8];
                unsafe {
                    raw::i8g_fattn_qk_f16(qh.as_ptr(), kh.as_ptr(), hd, steps, s.as_mut_ptr(), ss);
                    for qi in 0..8 {
                        sums[qi] = raw::i8g_fattn_softmax_f16(s.as_ptr().add(qi * ss), kc, nvalid, m, pt.as_mut_ptr(), qi);
                    }
                    let mut pf = vec![0f32; kc * 8];
                    raw::i8g_fattn_from_f16(pt.as_ptr(), kc * 8, pf.as_mut_ptr());
                    check(&q, &k, &v, hd, kc, nvalid, m, &s, ss, &pf, &sums, &o.clone(), &mut o, &rnd, half, |o| {
                        raw::i8g_fattn_pv_f16(pt.as_ptr(), vh.as_ptr(), kc, hd, kc * 8, o.as_mut_ptr())
                    });
                }
            } else {
                let mut pt = vec![0f32; kc * 8];
                unsafe {
                    raw::i8g_fattn_qk_f32(qt.as_ptr(), kt.as_ptr(), hd, steps, s.as_mut_ptr(), ss);
                    for qi in 0..8 {
                        sums[qi] = raw::i8g_fattn_softmax_f32(s.as_ptr().add(qi * ss), kc, nvalid, m, pt.as_mut_ptr(), qi);
                    }
                    let pf = pt.clone();
                    check(&q, &k, &v, hd, kc, nvalid, m, &s, ss, &pf, &sums, &o.clone(), &mut o, &rnd, half, |o| {
                        raw::i8g_fattn_pv_f32(pt.as_ptr(), vt.as_ptr(), kc, hd, kc * 8, o.as_mut_ptr())
                    });
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        hd: usize,
        kc: usize,
        nvalid: usize,
        m: f32,
        s: &[f32],
        ss: usize,
        pf: &[f32],
        sums: &[f32; 8],
        o0: &[f32],
        o: &mut Vec<f32>,
        rnd: &dyn Fn(f32) -> f32,
        half: bool,
        pv: impl FnOnce(&mut Vec<f32>),
    ) {
        pv(o);
        for qi in 0..8 {
            let mut sum_ref = 0f64;
            for key in 0..kc {
                let want: f64 = (0..hd).map(|d| rnd(q[qi * hd + d]) as f64 * rnd(k[key * hd + d]) as f64).sum();
                let got = s[qi * ss + key] as f64;
                assert!((got - want).abs() < 1e-5 * (1.0 + want.abs()), "qk qi {qi} key {key}: {got} vs {want}");
                let p_want = if key < nvalid { (s[qi * ss + key] as f64 - m as f64).exp() } else { 0.0 };
                let p_got = pf[key * 8 + qi] as f64;
                // f16 storage: half-ulp relative 2^-11 (plus exp's ~1e-7)
                let tol = if half { 6e-4 } else { 1e-6 };
                assert!((p_got - p_want).abs() <= tol * p_want.abs() + 1e-30, "p qi {qi} key {key}: {p_got} vs {p_want}");
                sum_ref += p_got;
            }
            assert!((sums[qi] as f64 - sum_ref).abs() < 1e-5 * sum_ref, "sum qi {qi}");
            for d in 0..hd {
                let want = o0[qi * hd + d] as f64 + (0..kc).map(|key| pf[key * 8 + qi] as f64 * rnd(v[key * hd + d]) as f64).sum::<f64>();
                let got = o[qi * hd + d] as f64;
                assert!((got - want).abs() < 1e-5 * (1.0 + want.abs()), "pv qi {qi} d {d}: {got} vs {want}");
            }
        }
    }
}
