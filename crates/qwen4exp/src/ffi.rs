//! FFI over the vendored Flash kernels (`vendor/ggml-flash`, BigMoeOnEdge llama.cpp fork
//! `claude/flash-r4`): production NEON `vec_dot` for every quant type Flash uses, and the R4
//! interleaved IQ2_S / IQ3_XXS expert layout with its gemv. Activation quantisation and the f32
//! ops come from `kernels-sys` (same functions llama.cpp uses on ARM).

use std::os::raw::c_int;

pub const GGML_TYPE_IQ3_XXS: c_int = 18;
pub const GGML_TYPE_IQ2_S: c_int = 22;

pub type VecDot = unsafe extern "C" fn(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);

use std::ffi::c_void;

extern "C" {
    pub fn oxf_vec_dot_q6_K_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
    pub fn oxf_vec_dot_q8_0_q8_0(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
    pub fn oxf_vec_dot_iq2_s_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
    pub fn oxf_vec_dot_iq3_xxs_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
    pub fn oxf_vec_dot_iq4_xs_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
    pub fn oxf_vec_dot_iq4_nl_q8_0(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
    pub fn oxf_vec_dot_q2_0_q8_0(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);

    pub fn ggml_cpu_expert_r4_slice_to_r4(ty: c_int, data: *mut u8, ne00: i64, ne01: i64) -> bool;
    pub fn ggml_cpu_expert_r4_gemv(ty: c_int, ne00: i64, s: *mut f32, vx: *const u8, vy: *const u8, nrows: i64);
    /// 4 rows x 4 tokens per i8mm tile set; s[t][r] for rows [0, nrows) of vx; bitwise equal to the gemv per (row, token).
    pub fn ggml_cpu_expert_r4_gemm4(ty: c_int, ne00: i64, s: *const *mut f32, vx: *const u8, vy: *const *const u8, nrows: i64);
    pub fn ggml_cpu_expert_r4_has_gemm() -> bool;

    // PREFILL (csrc/pf_expert.c): 8-token tiles for the IQ2_S R4 gate/up rows; bitwise equal to the gemv per (row, token)
    /// Pack `n <= 8` q8_K rows into one X group: `xc` = ne00 * 8 bytes, `xd` = ne00 / 256 * 8 floats.
    pub fn pf_expert_pack_x8(ne00: i64, n: c_int, vy: *const *const u8, xc: *mut i8, xd: *mut f32);
    pub fn pf_expert_iq2s_scratch(ne00: i64, nrows: i64) -> usize;
    /// Rows [0, nrows) of the R4 matrix `vx` against `ngroups` consecutive X groups: `s[8g + t][r]` (NULL = skip).
    pub fn pf_expert_iq2s_rows(ne00: i64, vx: *const u8, nrows: i64, xc: *const i8, xd: *const f32, ngroups: i64, s: *const *mut f32, scratch: *mut c_void);
    /// Down rows (plain IQ4_NL matrix, Q8_0 activations): pack `n <= 8` q8_0 rows of `len` values (`xc` = len * 8 bytes,
    /// `xd` = len / 32 * 8 floats), then rows [0, nrows) against the groups; bitwise equal to vec_dot per (row, token).
    pub fn pf_expert_pack_x8_q8_0(len: i64, n: c_int, vy: *const *const u8, xc: *mut i8, xd: *mut f32);
    pub fn pf_expert_iq4nl_scratch(ne00: i64, nrows: i64) -> usize;
    pub fn pf_expert_iq4nl_rows(ne00: i64, vx: *const u8, nrows: i64, xc: *const i8, xd: *const f32, ngroups: i64, s: *const *mut f32, scratch: *mut c_void);
    pub fn pf_expert_iq3_scratch(ne00: i64, nrows: i64) -> usize;
    /// As `pf_expert_iq2s_rows` for an IQ3_XXS R4 matrix.
    pub fn pf_expert_iq3_rows(ne00: i64, vx: *const u8, nrows: i64, xc: *const i8, xd: *const f32, ngroups: i64, s: *const *mut f32, scratch: *mut c_void);
}

/// Weight quant types present in the Flash GGUF (ggml type ids).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W {
    F32,
    Bf16,
    Q8_0,
    Q6K,
    Iq2S,
    Iq3Xxs,
    Iq4Xs,
    Iq4Nl,
    Q2_0,
}

impl W {
    pub fn from_ggml(t: u32) -> Option<W> {
        Some(match t {
            0 => W::F32,
            30 => W::Bf16,
            8 => W::Q8_0,
            14 => W::Q6K,
            22 => W::Iq2S,
            18 => W::Iq3Xxs,
            23 => W::Iq4Xs,
            20 => W::Iq4Nl,
            42 => W::Q2_0,
            _ => return None,
        })
    }
    pub fn ggml_type(self) -> i32 {
        match self {
            W::F32 => 0,
            W::Bf16 => 30,
            W::Q8_0 => 8,
            W::Q6K => 14,
            W::Iq2S => 22,
            W::Iq3Xxs => 18,
            W::Iq4Xs => 23,
            W::Iq4Nl => 20,
            W::Q2_0 => 42,
        }
    }
    /// (block elements, block bytes)
    pub fn block(self) -> (usize, usize) {
        match self {
            W::F32 => (1, 4),
            W::Bf16 => (1, 2),
            W::Q8_0 => (32, 34),
            W::Q6K => (256, 210),
            W::Iq2S => (256, 82),
            W::Iq3Xxs => (256, 98),
            W::Iq4Xs => (256, 136),
            W::Iq4Nl => (32, 18),
            W::Q2_0 => (64, 18),
        }
    }
    pub fn row_bytes(self, k: usize) -> usize {
        let (qk, bb) = self.block();
        assert!(k % qk == 0, "{self:?}: k={k} not a multiple of {qk}");
        k / qk * bb
    }
    /// Activation format of the vec_dot (None = f32 / bf16 handled in Rust).
    pub fn act(self) -> Option<kernels_sys::ActType> {
        use kernels_sys::ActType::*;
        match self {
            W::Q6K | W::Iq2S | W::Iq3Xxs | W::Iq4Xs => Some(Q8K),
            W::Q8_0 | W::Iq4Nl | W::Q2_0 => Some(Q8_0),
            W::F32 | W::Bf16 => None,
        }
    }
    pub fn vec_dot(self) -> Option<VecDot> {
        Some(match self {
            W::Q6K => oxf_vec_dot_q6_K_q8_K,
            W::Q8_0 => oxf_vec_dot_q8_0_q8_0,
            W::Iq2S => oxf_vec_dot_iq2_s_q8_K,
            W::Iq3Xxs => oxf_vec_dot_iq3_xxs_q8_K,
            W::Iq4Xs => oxf_vec_dot_iq4_xs_q8_K,
            W::Iq4Nl => oxf_vec_dot_iq4_nl_q8_0,
            W::Q2_0 => oxf_vec_dot_q2_0_q8_0,
            W::F32 | W::Bf16 => return None,
        })
    }
    /// The R4 layout exists for this type (gate/up expert slices only).
    pub fn r4(self) -> Option<c_int> {
        match self {
            W::Iq2S => Some(GGML_TYPE_IQ2_S),
            W::Iq3Xxs => Some(GGML_TYPE_IQ3_XXS),
            _ => None,
        }
    }
}

/// Rows `[r0, r1)` of a row-major quantised matrix (`k` weights per row) times one activation
/// vector `act` (already quantised to `w.act()`), written to `out[r - r0]`.
///
/// # Safety
/// `base` must point at the matrix's row 0 and the rows must exist; `act` must hold
/// `w.act().bytes(k)` bytes; `out` must hold `r1 - r0` floats.
#[inline]
pub unsafe fn gemv_vecdot(w: W, base: *const u8, k: usize, r0: usize, r1: usize, act: *const u8, out: *mut f32) {
    let rb = w.row_bytes(k);
    let f = w.vec_dot().expect("quantised type");
    for r in r0..r1 {
        f(k as c_int, out.add(r - r0), 0, base.add(r * rb), 0, act, 0, 1);
    }
}

/// R4 gemv over rows `[r0, r1)` (both multiples of 4) of an R4-encoded slice.
///
/// # Safety
/// As [`gemv_vecdot`]; the slice must have been transformed by `slice_to_r4`.
#[inline]
pub unsafe fn gemv_r4(ty: c_int, w: W, base: *const u8, k: usize, r0: usize, r1: usize, act: *const u8, out: *mut f32) {
    debug_assert!(r0 % 4 == 0 && r1 % 4 == 0);
    let rb = w.row_bytes(k);
    ggml_cpu_expert_r4_gemv(ty, k as i64, out, base.add(r0 * rb), act, (r1 - r0) as i64);
}

/// f32 rows `[r0, r1)` of a row-major f32 matrix times x.
///
/// # Safety
/// `base` must hold the rows.
#[inline]
pub unsafe fn gemv_f32(base: *const f32, k: usize, r0: usize, r1: usize, x: &[f32], out: &mut [f32]) {
    for r in r0..r1 {
        out[r - r0] = kernels_sys::ffi::oxe_vec_dot_f32(k as c_int, base.add(r * k), x.as_ptr());
    }
}

/// bf16 rows `[r0, r1)` times x (bf16 = upper half of an f32).
///
/// # Safety
/// `base` must hold the rows.
#[inline]
pub unsafe fn gemv_bf16(base: *const u16, k: usize, r0: usize, r1: usize, x: &[f32], out: &mut [f32]) {
    for r in r0..r1 {
        let row = std::slice::from_raw_parts(base.add(r * k), k);
        let mut acc = [0f32; 8];
        for (c, xs) in row.chunks_exact(8).zip(x.chunks_exact(8)) {
            for j in 0..8 {
                acc[j] += f32::from_bits((c[j] as u32) << 16) * xs[j];
            }
        }
        out[r - r0] = acc.iter().sum();
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Dequantise IQ4_NL blocks (18 B: fp16 d, 16 nibble bytes; low nibbles = weights 0-15,
/// high nibbles = 16-31, as ggml's dequantize_row_iq4_nl).
pub fn dequant_iq4_nl(src: &[u8], out: &mut [f32]) {
    for (b, o) in src.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        for j in 0..16 {
            o[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xf) as usize] as f32;
            o[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f16_roundtrip_basics() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!((f16_to_f32(0x0001) - 5.960464e-8).abs() < 1e-12);
    }
    /// R4 gemv through the crate's own FFI is bitwise equal to the NEON vec_dot on the file layout.
    #[test]
    fn r4_gemv_bitwise_vs_vec_dot() {
        for (w, d_off) in [(W::Iq2S, 0usize), (W::Iq3Xxs, 0usize)] {
            let (k, rows) = (2560usize, 64usize);
            let rb = w.row_bytes(k);
            let mut x = 0x1234_5678_9abc_def1u64;
            let mut m: Vec<u8> = (0..rows * rb)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    x as u8
                })
                .collect();
            // sane fp16 scale at the start of every block (d is the first field)
            let bb = w.block().1;
            for b in 0..rows * rb / bb {
                let h: u16 = 0x2000 + (b as u16 % 512);
                m[b * bb + d_off..b * bb + d_off + 2].copy_from_slice(&h.to_le_bytes());
            }
            let act_f: Vec<f32> = (0..k).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let mut act = vec![0u8; kernels_sys::ActType::Q8K.bytes(k)];
            kernels_sys::quantize_act(kernels_sys::ActType::Q8K, &act_f, &mut act);
            let mut want = vec![0f32; rows];
            let mut got = vec![0f32; rows];
            unsafe {
                gemv_vecdot(w, m.as_ptr(), k, 0, rows, act.as_ptr(), want.as_mut_ptr());
                let t = w.r4().unwrap();
                assert!(ggml_cpu_expert_r4_slice_to_r4(t, m.as_mut_ptr(), k as i64, rows as i64));
                gemv_r4(t, w, m.as_ptr(), k, 0, rows, act.as_ptr(), got.as_mut_ptr());
            }
            assert!(want.iter().all(|v| v.is_finite()));
            assert_eq!(want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{w:?}");
        }
    }

    /// R4 gemm4 (4 tokens per call, i8mm) is bitwise equal to the NEON vec_dot per (row, token).
    #[test]
    fn r4_gemm4_bitwise_vs_vec_dot() {
        assert!(unsafe { ggml_cpu_expert_r4_has_gemm() }, "build without i8mm gemm");
        for w in [W::Iq2S, W::Iq3Xxs] {
            let (k, rows) = (2560usize, 64usize);
            let rb = w.row_bytes(k);
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            let mut rnd = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let mut m: Vec<u8> = (0..rows * rb).map(|_| rnd() as u8).collect();
            let bb = w.block().1;
            for b in 0..rows * rb / bb {
                let h: u16 = 0x2000 + (b as u16 % 512);
                m[b * bb..b * bb + 2].copy_from_slice(&h.to_le_bytes());
            }
            let ab = kernels_sys::ActType::Q8K.bytes(k);
            let mut acts = vec![0u8; 4 * ab];
            for t in 0..4 {
                let f: Vec<f32> = (0..k).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
                kernels_sys::quantize_act(kernels_sys::ActType::Q8K, &f, &mut acts[t * ab..(t + 1) * ab]);
            }
            let mut want = vec![0f32; 4 * rows];
            for t in 0..4 {
                unsafe { gemv_vecdot(w, m.as_ptr(), k, 0, rows, acts[t * ab..].as_ptr(), want[t * rows..].as_mut_ptr()) };
            }
            let ty = w.r4().unwrap();
            assert!(unsafe { ggml_cpu_expert_r4_slice_to_r4(ty, m.as_mut_ptr(), k as i64, rows as i64) });
            let mut got = vec![0f32; 4 * rows];
            // two calls of 16 + 48 rows at the row offsets the stream op uses
            for (r0, r1) in [(0usize, 16usize), (16, 64)] {
                let s: [*mut f32; 4] = std::array::from_fn(|t| unsafe { got.as_mut_ptr().add(t * rows + r0) });
                let vy: [*const u8; 4] = std::array::from_fn(|t| acts[t * ab..].as_ptr());
                unsafe { ggml_cpu_expert_r4_gemm4(ty, k as i64, s.as_ptr(), m.as_ptr().add(r0 * rb), vy.as_ptr(), (r1 - r0) as i64) };
            }
            assert!(want.iter().all(|v| v.is_finite() && *v != 0.0));
            assert_eq!(want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{w:?}");
        }
    }

    /// PREFILL: the 8-token tile paths (IQ2_S, IQ3_XXS) are bitwise equal to the NEON vec_dot per (row, token), incl. partial groups.
    #[test]
    fn pf_expert_rows_bitwise() {
        for w in [W::Iq2S, W::Iq3Xxs] {
            pf_rows_case(w);
        }
    }

    fn pf_rows_case(w: W) {
        let (k, rows, nt) = (2560usize, 64usize, 19usize);
        let rb = w.row_bytes(k);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut m: Vec<u8> = (0..rows * rb).map(|_| rnd() as u8).collect();
        let bb = w.block().1;
        for b in 0..rows * rb / bb {
            let h: u16 = 0x2000 + (b as u16 % 512);
            m[b * bb..b * bb + 2].copy_from_slice(&h.to_le_bytes());
        }
        let ab = kernels_sys::ActType::Q8K.bytes(k);
        let mut acts = vec![0u8; nt * ab];
        for t in 0..nt {
            let mag = if t == 3 { 0.0 } else if t % 5 == 1 { 250.0 } else { 1.0 };
            let f: Vec<f32> = (0..k).map(|_| (((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * mag).collect();
            kernels_sys::quantize_act(kernels_sys::ActType::Q8K, &f, &mut acts[t * ab..(t + 1) * ab]);
        }
        let mut want = vec![0f32; nt * rows];
        for t in 0..nt {
            unsafe { gemv_vecdot(w, m.as_ptr(), k, 0, rows, acts[t * ab..].as_ptr(), want[t * rows..].as_mut_ptr()) };
        }
        assert!(unsafe { ggml_cpu_expert_r4_slice_to_r4(w.r4().unwrap(), m.as_mut_ptr(), k as i64, rows as i64) });
        let ng = nt.div_ceil(8);
        let mut xc = vec![0i8; ng * k * 8];
        let mut xd = vec![0f32; ng * (k / 256) * 8];
        for g in 0..ng {
            let n = (nt - g * 8).min(8);
            let vy: [*const u8; 8] = std::array::from_fn(|t| if t < n { acts[(g * 8 + t) * ab..].as_ptr() } else { std::ptr::null() });
            unsafe { pf_expert_pack_x8(k as i64, n as c_int, vy.as_ptr(), xc[g * k * 8..].as_mut_ptr(), xd[g * (k / 256) * 8..].as_mut_ptr()) };
        }
        let mut got = vec![0f32; nt * rows];
        // two calls of 32 rows at the row offsets the stream op uses
        for r0 in [0usize, 32] {
            let n = 32usize;
            let iq2 = w == W::Iq2S;
            let mut scratch = vec![0u32; unsafe { if iq2 { pf_expert_iq2s_scratch(k as i64, n as i64) } else { pf_expert_iq3_scratch(k as i64, n as i64) } }.div_ceil(4)];
            let s: Vec<*mut f32> = (0..ng * 8).map(|t| if t < nt { unsafe { got.as_mut_ptr().add(t * rows + r0) } } else { std::ptr::null_mut() }).collect();
            let f = if iq2 { pf_expert_iq2s_rows } else { pf_expert_iq3_rows };
            unsafe { f(k as i64, m.as_ptr().add(r0 * rb), n as i64, xc.as_ptr(), xd.as_ptr(), ng as i64, s.as_ptr(), scratch.as_mut_ptr() as *mut c_void) };
        }
        assert!(want.iter().enumerate().all(|(i, v)| v.is_finite() && (*v != 0.0 || i / rows == 3)));
        assert_eq!(want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{w:?}");
    }

    /// PREFILL: the 8-token IQ4_NL down tile is bitwise equal to the NEON vec_dot per (row, token).
    #[test]
    fn pf_expert_down_bitwise() {
        let w = W::Iq4Nl;
        let (k, rows, nt) = (640usize, 96usize, 21usize);
        let rb = w.row_bytes(k);
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut m: Vec<u8> = (0..rows * rb).map(|_| rnd() as u8).collect();
        let bb = w.block().1;
        for b in 0..rows * rb / bb {
            let h: u16 = 0x2000 + (b as u16 % 512);
            m[b * bb..b * bb + 2].copy_from_slice(&h.to_le_bytes());
        }
        let ab = kernels_sys::ActType::Q8_0.bytes(k);
        let mut acts = vec![0u8; nt * ab];
        for t in 0..nt {
            let mag = if t == 3 { 0.0 } else if t % 5 == 1 { 250.0 } else { 1.0 };
            let f: Vec<f32> = (0..k).map(|_| (((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * mag).collect();
            kernels_sys::quantize_act(kernels_sys::ActType::Q8_0, &f, &mut acts[t * ab..(t + 1) * ab]);
        }
        let mut want = vec![0f32; nt * rows];
        for t in 0..nt {
            unsafe { gemv_vecdot(w, m.as_ptr(), k, 0, rows, acts[t * ab..].as_ptr(), want[t * rows..].as_mut_ptr()) };
        }
        let ng = nt.div_ceil(8);
        let mut xc = vec![0i8; ng * k * 8];
        let mut xd = vec![0f32; ng * (k / 32) * 8];
        for g in 0..ng {
            let n = (nt - g * 8).min(8);
            let vy: [*const u8; 8] = std::array::from_fn(|t| if t < n { acts[(g * 8 + t) * ab..].as_ptr() } else { std::ptr::null() });
            unsafe { pf_expert_pack_x8_q8_0(k as i64, n as c_int, vy.as_ptr(), xc[g * k * 8..].as_mut_ptr(), xd[g * (k / 32) * 8..].as_mut_ptr()) };
        }
        let mut got = vec![0f32; nt * rows];
        for r0 in [0usize, 32, 64] {
            let n = 32usize;
            let mut scratch = vec![0u32; unsafe { pf_expert_iq4nl_scratch(k as i64, n as i64) }.div_ceil(4)];
            let s: Vec<*mut f32> = (0..ng * 8).map(|t| if t < nt { unsafe { got.as_mut_ptr().add(t * rows + r0) } } else { std::ptr::null_mut() }).collect();
            unsafe { pf_expert_iq4nl_rows(k as i64, m.as_ptr().add(r0 * rb), n as i64, xc.as_ptr(), xd.as_ptr(), ng as i64, s.as_ptr(), scratch.as_mut_ptr() as *mut c_void) };
        }
        assert!(want.iter().enumerate().all(|(i, v)| v.is_finite() && (*v != 0.0 || i / rows == 3)));
        assert_eq!(want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), got.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn row_bytes_match_header_audit() {
        // bytes_per_expert from flash-post-memlat-header-audit.json, 640 rows each
        assert_eq!(W::Iq2S.row_bytes(2560) * 640, 524800);
        assert_eq!(W::Iq3Xxs.row_bytes(2560) * 640, 627200);
        assert_eq!(W::Iq4Xs.row_bytes(2560) * 640, 870400);
        assert_eq!(W::Iq4Nl.row_bytes(640) * 2560, 921600);
        assert_eq!(W::Q2_0.row_bytes(640) * 2560, 460800);
    }
}
