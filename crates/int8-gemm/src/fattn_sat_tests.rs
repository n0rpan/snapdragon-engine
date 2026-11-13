//! F-IMG-ATTN-b: the f16 operand store saturates finite overflow and keeps non-finite values.

use super::*;

#[test]
fn to_f16_saturates_finite_overflow() {
    // vector body (4 at a time) and the scalar tail (n % 4 != 0) both covered
    let xs = [65504.0f32, 65519.0, 65520.0, 1e9, -65520.0, -1e30, f32::MAX, -f32::MAX, 70000.0, -70000.0, 1.5];
    let mut got = vec![0u16; xs.len()];
    to_f16(&xs, &mut got, 1);
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(got[i], f16_bits_sat_ref(x), "x {x:e}");
        assert_ne!(got[i] & 0x7c00, 0x7c00, "finite x {x:e} must stay finite");
    }
    assert_eq!(got[2], 0x7bff); // 65520 would round to inf unsaturated
    assert_eq!(got[4], 0xfbff);
}

#[test]
fn to_f16_keeps_nan_and_inf() {
    let xs = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0, f32::NAN, f32::INFINITY, -f32::NAN];
    let mut got = vec![0u16; xs.len()];
    to_f16(&xs, &mut got, 1);
    assert_eq!(got[0], 0x7c00);
    assert_eq!(got[1], 0xfc00);
    assert_eq!(got[5], 0x7c00);
    for i in [2, 4, 6] {
        assert_eq!(got[i] & 0x7c00, 0x7c00, "NaN exponent at {i}");
        assert_ne!(got[i] & 0x3ff, 0, "NaN mantissa at {i}");
    }
    assert_eq!(got[3], 0x3c00);
}

#[test]
fn to_f16_in_range_is_unchanged() {
    // every value inside the range converts exactly as the plain round-to-nearest-even reference
    let mut x = 1.0e-8f32;
    let mut xs = Vec::new();
    while x < 65504.0 {
        xs.push(x);
        xs.push(-x);
        x *= 1.37;
    }
    let mut got = vec![0u16; xs.len()];
    to_f16(&xs, &mut got, 1);
    for (i, &x) in xs.iter().enumerate() {
        assert_eq!(got[i], f16_bits_ref(x), "x {x:e}");
    }
}
