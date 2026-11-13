//! v0.1.3 lane H review (F15 / W4-6): model-free edge cases of the int8 tile kernel, its safe
//! wrappers and the softmax kernels (exp range, max subtraction, masked lanes, NaN propagation).
//! Complements `tests.rs` (bitwise contract, accuracy) and `fattn.rs` (f64 comparison).
use super::*;

/// n = 0: the empty buffers must still give aligned (empty) views. Before the fix the dangling
/// pointer of an empty `AlignedBytes` was 1-aligned, so `scales()` built a misaligned `&[f32]`
/// (UB; a precondition abort in debug builds).
#[test]
fn empty_packed_has_aligned_empty_views() {
    let p = Packed::zeroed(0, 64, 32);
    assert_eq!(p.groups(), 0);
    assert!(p.data().is_empty());
    let s = p.scales();
    assert!(s.is_empty());
    assert_eq!(s.as_ptr() as usize % 64, 0);
    let w = Packed::zeroed(8, 64, 32);
    let mut y: Vec<f32> = Vec::new();
    gemm(&p, &w, 0..0, 0..1, &mut y, 8, Order::MulAdd, false);
    gemm_ref(&p, &w, 0..0, 0..1, &mut y, 8, Order::MulAdd, false);
}

// ---- the safe wrappers refuse what would make the raw tile kernel read or write out of bounds ----

#[test]
#[should_panic(expected = "y too small")]
fn gemm_rejects_short_output() {
    let x = Packed::zeroed(16, 64, 64);
    let w = Packed::zeroed(16, 64, 64);
    // 16 tokens x 16 rows at stride 16 need 15 * 16 + 16 = 256 floats
    let mut y = vec![0.0f32; 255];
    gemm(&x, &w, 0..2, 0..2, &mut y, 16, Order::MulAdd, false);
}

#[test]
#[should_panic(expected = "ystride >= nr")]
fn gemm_rejects_overlapping_rows() {
    let x = Packed::zeroed(8, 64, 64);
    let w = Packed::zeroed(16, 64, 64);
    let mut y = vec![0.0f32; 1024];
    gemm(&x, &w, 0..1, 0..2, &mut y, 8, Order::MulAdd, false);
}

#[test]
#[should_panic(expected = "x.groups()")]
fn gemm_rejects_group_range_past_operand() {
    let x = Packed::zeroed(8, 64, 64);
    let w = Packed::zeroed(8, 64, 64);
    let mut y = vec![0.0f32; 1024];
    gemm(&x, &w, 0..1, 0..2, &mut y, 16, Order::MulAdd, false);
}

#[test]
#[should_panic(expected = "k mismatch")]
fn gemm_rejects_k_mismatch() {
    let x = Packed::zeroed(8, 64, 64);
    let w = Packed::zeroed(8, 128, 64);
    let mut y = vec![0.0f32; 64];
    gemm(&x, &w, 0..1, 0..1, &mut y, 8, Order::MulAdd, false);
}

#[test]
#[should_panic(expected = "single-chunk")]
fn gemm_rows1_rejects_multi_chunk_operands() {
    let x = Packed::zeroed(8, 64, 32);
    let w = Packed::zeroed(8, 64, 32);
    let mut y = vec![0.0f32; 64];
    gemm_rows1(&x, &w, 0, 0..1, &mut y, 8);
}

/// The largest chunk `Packed::zeroed` accepts (131072) with worst-case codes: +-127 on the
/// activation side (the crate's quantiser never emits -128) and -128 on the weight side (a foreign
/// Q8_0 file may). |sum| = 127 * 128 * 131072 = 2_130_706_432 < 2^31 - 1, so the int32 SMMLA sums
/// are exact; both values are exactly representable in f32. (Both operands at -128 would be
/// 2^31 and overflow: the documented bound assumes one side comes from the quantiser.)
#[test]
fn max_chunk_sums_stay_exact() {
    const K: usize = 131072;
    let mut x = Packed::zeroed(8, K, K);
    let mut w = Packed::zeroed(8, K, K);
    let plus = vec![127i8; K];
    let minus = vec![-127i8; K];
    let wmin = vec![-128i8; K];
    for v in 0..8 {
        let xc: &[i8] = if v % 2 == 0 { &plus[..] } else { &minus[..] };
        let wc: &[i8] = if v < 4 { &wmin[..] } else { &plus[..] };
        x.set_codes(v, 0, xc, &[1.0]);
        w.set_codes(v, 0, wc, &[1.0]);
    }
    for order in [Order::MulAdd, Order::Fma] {
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        gemm(&x, &w, 0..1, 0..1, &mut a, 8, order, false);
        gemm_ref(&x, &w, 0..1, 0..1, &mut b, 8, order, false);
        for t in 0..8 {
            for r in 0..8 {
                let want = (if t % 2 == 0 { 127i64 } else { -127 }) * (if r < 4 { -128 } else { 127 }) * K as i64;
                assert!(want.abs() < i32::MAX as i64);
                assert_eq!(a[t * 8 + r], want as f32, "{order:?} t{t} r{r}");
                assert_eq!(b[t * 8 + r].to_bits(), a[t * 8 + r].to_bits(), "{order:?} t{t} r{r}");
            }
        }
    }
}

// ---- i8g_softmax_pack / i8g_row_max (int8 flash attention) ----

/// Arguments at and beyond the -87 clamp, the maximum itself and -inf: every exp is finite, in
/// [0, 1] and within 1e-4 of exp(max(x, -87)); exp(0) is exactly 1, so the row maximum becomes
/// code 255 with scale exactly 1/255. Only the probed query's bytes are written.
#[test]
fn softmax_pack_exp_range_and_codes() {
    let s0 = [0.0f32, -1.0, -10.0, -50.0, -86.99, -87.0, -88.0, f32::NEG_INFINITY];
    let mut s = s0.to_vec();
    let sv = [1.0f32; 8];
    let qi = 5usize;
    let mut codes = vec![0xAAu8; 64];
    let (mut sum, mut scale) = (0.0f32, 0.0f32);
    unsafe { raw::i8g_softmax_pack(s.as_mut_ptr(), 8, 0.0, sv.as_ptr(), codes.as_mut_ptr(), qi, &mut sum, &mut scale) };
    assert_eq!(s[0].to_bits(), 1.0f32.to_bits(), "exp(0) must be exactly 1");
    assert_eq!(scale.to_bits(), (1.0f32 / 255.0).to_bits());
    let mut want_sum = 0.0f64;
    for (j, &e) in s.iter().enumerate() {
        let want = (s0[j].max(-87.0) as f64).exp();
        want_sum += want;
        assert!(e.is_finite() && (0.0..=1.0).contains(&e), "j {j}: {e:e}");
        assert!(((e as f64) - want).abs() <= 1e-4 * want, "j {j}: {e:e} vs {want:e}");
    }
    assert!(((sum as f64 - want_sum) / want_sum).abs() < 1e-4, "sum {sum} vs {want_sum}");
    // 255 * e^-1 = 93.81 -> 94; 255 * e^-10 = 0.012 -> 0
    assert_eq!(&codes[qi * 8..qi * 8 + 8], &[255u8, 94, 0, 0, 0, 0, 0, 0]);
    for (i, &b) in codes.iter().enumerate() {
        if i / 8 != qi {
            assert_eq!(b, 0xAA, "byte {i} of another query written");
        }
    }
}

/// A NaN score is not hidden: it reaches the row sum (so the row's normaliser is NaN) and the row
/// writes zero codes with scale 0 instead of garbage codes. `i8g_row_max` propagates NaN too.
#[test]
fn softmax_pack_nan_score_surfaces_in_sum() {
    let mut s = vec![0.0f32, -0.5, f32::NAN, -2.0, -3.0, -4.0, -5.0, -6.0];
    let cmax = unsafe { raw::i8g_row_max(s.as_ptr(), 8) };
    assert!(cmax.is_nan());
    let sv = [0.5f32; 8];
    let mut codes = vec![0xAAu8; 64];
    let (mut sum, mut scale) = (0.0f32, 1.0f32);
    unsafe { raw::i8g_softmax_pack(s.as_mut_ptr(), 8, 0.0, sv.as_ptr(), codes.as_mut_ptr(), 0, &mut sum, &mut scale) };
    assert!(sum.is_nan(), "sum {sum}");
    assert_eq!(scale, 0.0);
    assert!(codes[..8].iter().all(|&b| b == 0));
    assert!(codes[8..].iter().all(|&b| b == 0xAA));
}

/// A key chunk that is all padding (-inf) under a finite running maximum: the clamp gives about
/// 1.6e-38 per key, so the sum and the P.V weight (code * scale) are negligible and finite even
/// though the maximum pv is subnormal (its reciprocal overflows; the codes saturate).
#[test]
fn softmax_pack_all_masked_chunk_is_negligible() {
    let mut s = vec![f32::NEG_INFINITY; 16];
    let sv = [0.05f32; 16];
    let mut codes = vec![0u8; 128];
    let (mut sum, mut scale) = (0.0f32, 0.0f32);
    unsafe { raw::i8g_softmax_pack(s.as_mut_ptr(), 16, 3.0, sv.as_ptr(), codes.as_mut_ptr(), 1, &mut sum, &mut scale) };
    assert!(sum.is_finite() && sum >= 0.0 && sum < 1e-36, "sum {sum:e}");
    assert!(scale.is_finite() && scale >= 0.0 && scale < 1e-38, "scale {scale:e}");
    for j in 0..16 {
        let c = codes[(j / 8) * 64 + 8 + j % 8] as f32;
        assert!(c * scale < 1e-37, "j {j}: code {c} scale {scale:e}");
    }
}

// ---- i8g_fattn_softmax_f32 / _f16 (float flash attention) ----

/// Lanes at or past `nvalid` are exactly +0 whatever the score buffer holds there (NaN, +-inf),
/// and they add nothing to the sum; `nvalid = 0` gives an exact zero sum; a NaN in a valid lane
/// reaches the sum.
#[test]
fn fattn_softmax_masked_lanes_and_nan() {
    use crate::fattn::raw as fr;
    let (kc, nvalid) = (16usize, 13usize);
    let mut s: Vec<f32> = (0..kc).map(|j| -0.25 * j as f32).collect();
    s[13] = f32::NAN;
    s[14] = f32::INFINITY;
    s[15] = f32::NEG_INFINITY;
    let m = 0.0f32;
    let want: f64 = (0..nvalid).map(|j| (s[j] as f64 - m as f64).exp()).sum();
    for qi in [0usize, 7] {
        let mut pt = vec![f32::NAN; kc * 8];
        let sum = unsafe { fr::i8g_fattn_softmax_f32(s.as_ptr(), kc, nvalid, m, pt.as_mut_ptr(), qi) };
        assert!(((sum as f64 - want) / want).abs() < 1e-5, "f32 sum {sum} vs {want}");
        for j in 0..kc {
            let p = pt[j * 8 + qi];
            if j < nvalid {
                let w = (s[j] as f64 - m as f64).exp();
                assert!((p as f64 - w).abs() <= 1e-6 * w, "qi {qi} j {j}: {p} vs {w}");
            } else {
                assert_eq!(p.to_bits(), 0, "qi {qi} masked j {j}: {p}");
            }
        }
        let mut ph = vec![0x7e00u16; kc * 8];
        let sum_h = unsafe { fr::i8g_fattn_softmax_f16(s.as_ptr(), kc, nvalid, m, ph.as_mut_ptr(), qi) };
        assert!(sum_h.is_finite() && ((sum_h as f64 - want) / want).abs() < 2e-3, "f16 sum {sum_h} vs {want}");
        for j in nvalid..kc {
            assert_eq!(ph[j * 8 + qi], 0, "qi {qi} masked f16 j {j}");
        }
    }
    let mut pt = vec![1.0f32; kc * 8];
    let sum = unsafe { fr::i8g_fattn_softmax_f32(s.as_ptr(), kc, 0, m, pt.as_mut_ptr(), 3) };
    assert_eq!(sum.to_bits(), 0);
    for j in 0..kc {
        assert_eq!(pt[j * 8 + 3].to_bits(), 0);
    }
    let mut t = s.clone();
    t[4] = f32::NAN;
    let sum = unsafe { fr::i8g_fattn_softmax_f32(t.as_ptr(), kc, nvalid, m, pt.as_mut_ptr(), 3) };
    assert!(sum.is_nan(), "sum {sum}");
}
