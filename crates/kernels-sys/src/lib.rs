//! Thin FFI over the vendored ggml CPU kernels and the ported f32 ops
//! (see `vendor/ggml-subset/PROVENANCE.md` and `csrc/oxe_ops.cpp`).
//!
//! - Interleaved GEMVs: Q3_K / Q4_K / Q5_K / Q6_K (8 rows x 8-byte interleave, q8_K activations)
//!   and Q8_0 (4 rows x 8-byte interleave, q8_0 activations), plus their load-time repack.
//! - Activation quantisation (q8_K, q8_0) and embedding-row dequantisation.
//! - The ported decode ops (RMSNorm, softmax, delta-net step, conv, RoPE, attention, ...),
//!   which call ggml's own `vec.h`/`vec.cpp` helpers for bit-exactness with llama.cpp.

use std::os::raw::c_int;

pub const QK_K: usize = 256;
pub const QK8_0: usize = 32;

/// Build flags the C/C++ subset was compiled with (recorded in reports).
pub const KERNEL_CFLAGS: &str = env!("OXE_KERNEL_CFLAGS");

/// `block_q8_K` from ggml-common.h.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlockQ8K {
    pub d: f32,
    pub qs: [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}

impl Default for BlockQ8K {
    fn default() -> Self {
        BlockQ8K { d: 0.0, qs: [0; QK_K], bsums: [0; QK_K / 16] }
    }
}

pub const Q8K_BLOCK_BYTES: usize = 292;
pub const Q8_0_BLOCK_BYTES: usize = 34;

pub mod ffi {
    use std::os::raw::c_int;
    extern "C" {
        pub fn oxe_native_expert_tile(out: *mut u16, out_halfs: usize,
            src: *const u8, bytes: usize, kind: u32, k: u32, rows: u32,
            experts: u32, expert: u32, row0: u32, k0: u32) -> i32;

        pub fn oxe_sizeof_block_q8_K() -> usize;
        pub fn oxe_sizeof_block_q3_K() -> usize;
        pub fn oxe_sizeof_block_q4_K() -> usize;
        pub fn oxe_sizeof_block_q5_K() -> usize;
        pub fn oxe_sizeof_block_q6_K() -> usize;
        pub fn oxe_sizeof_block_q8_0() -> usize;
        pub fn oxe_sizeof_block_q3_Kx8() -> usize;
        pub fn oxe_sizeof_block_q4_Kx8() -> usize;
        pub fn oxe_sizeof_block_q5_Kx8() -> usize;
        pub fn oxe_sizeof_block_q6_Kx8() -> usize;
        pub fn oxe_sizeof_block_q8_0x4() -> usize;

        pub fn oxe_repack_q3_K_8x8(src: *const u8, dst: *mut u8, nrows: i64, k: i64) -> c_int;
        pub fn oxe_repack_q4_K_8x8(src: *const u8, dst: *mut u8, nrows: i64, k: i64) -> c_int;
        pub fn oxe_repack_q5_K_8x8(src: *const u8, dst: *mut u8, nrows: i64, k: i64) -> c_int;
        pub fn oxe_repack_q6_K_8x8(src: *const u8, dst: *mut u8, nrows: i64, k: i64) -> c_int;
        pub fn oxe_repack_q8_0_4x8(src: *const u8, dst: *mut u8, nrows: i64, k: i64) -> c_int;

        pub fn oxe_quantize_row_q8_K(x: *const f32, y: *mut u8, k: i64);
        pub fn quantize_row_q8_0(x: *const f32, y: *mut u8, k: i64);
        pub fn oxe_f32_to_f16(x: *const f32, y: *mut u16, n: i64);

        pub fn dequantize_row_q8_0(x: *const u8, y: *mut f32, k: i64);
        pub fn dequantize_row_q3_K(x: *const u8, y: *mut f32, k: i64);
        pub fn dequantize_row_q4_K(x: *const u8, y: *mut f32, k: i64);
        pub fn dequantize_row_q5_K(x: *const u8, y: *mut f32, k: i64);
        pub fn dequantize_row_q6_K(x: *const u8, y: *mut f32, k: i64);

        pub fn ggml_gemv_q3_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q4_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q5_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q6_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q8_0_4x8_q8_0(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q3_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q4_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q5_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q6_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemv_q8_0_4x8_q8_0_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);

        pub fn ggml_vec_dot_q3_K_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
        pub fn ggml_vec_dot_q4_K_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
        pub fn ggml_vec_dot_q5_K_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
        pub fn ggml_vec_dot_q6_K_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);
        pub fn ggml_vec_dot_q8_0_q8_0_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, bx: usize, vy: *const u8, by: usize, nrc: c_int);

        // M3.2 batched prefill: 4-row activation quantisers, interleaved GEMMs, MUL_MAT_ID tiles
        pub fn ggml_quantize_mat_q8_K_4x8_generic(x: *const f32, vy: *mut u8, k: i64);
        pub fn ggml_quantize_mat_q8_0_4x8(x: *const f32, vy: *mut u8, k: i64);
        pub fn ggml_gemm_q3_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q4_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q5_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q6_K_8x8_q8_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q8_0_4x8_q8_0(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q4_K_8x8_q8_K_gemv_order(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q5_K_8x8_q8_K_gemv_order(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q3_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q4_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q5_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q6_K_8x8_q8_K_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn ggml_gemm_q8_0_4x8_q8_0_generic(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn oxe_interleave_q8_K_rows_x4(rows: *const *const u8, out: *mut u8, n_per_row: i64);
        pub fn oxe_interleave_q8_0_rows_x4(rows: *const *const u8, out: *mut u8, n_per_row: i64);
        pub fn oxe_sizeof_block_q8_Kx4() -> usize;
        pub fn oxe_pf_gemm_q4_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn oxe_pf_gemm_q6_K(n: c_int, s: *mut f32, bs: usize, vx: *const u8, vy: *const u8, nr: c_int, nc: c_int);
        pub fn oxe_pf_tiles_available() -> c_int;
        pub fn oxe_pf_xblk_bytes() -> usize;
        pub fn oxe_pf_pack_q8K_pairs(rows: *const *const u8, nr: c_int, n: c_int, out: *mut u8);
        pub fn oxe_pf_mmid_tile(ty: c_int, n: c_int, orow: *const *mut f32, vx: *const u8, xq: *const u8, nr: c_int, nc: c_int) -> c_int;
        // verify-lane GEMVs (csrc/oxe_vl.inc)
        pub fn oxe_vl_gemv(ty: c_int, n: c_int, s: *const *mut f32, vx: *const u8, y: *const *const u8, nl: c_int, nc: c_int) -> c_int;

        // csrc/oxe_ops.cpp
        pub fn oxe_rms_norm_f32(x: *const f32, y: *mut f32, n: i64, eps: f32);
        pub fn oxe_rms_norm_mul_f32(x: *const f32, w: *const f32, y: *mut f32, n: i64, eps: f32);
        pub fn oxe_gdn_l2_norm_f32(x: *mut f32, n: i64, eps: f32);
        pub fn oxe_vec_scale_f32(n: c_int, y: *mut f32, v: f32);
        pub fn oxe_vec_dot_f32(n: c_int, x: *const f32, y: *const f32) -> f32;
        pub fn oxe_vec_dot_f16(n: c_int, x: *const u16, y: *const u16) -> f32;
        pub fn oxe_fp32_to_fp16_row(x: *const f32, y: *mut u16, n: i64);
        pub fn oxe_silu_f32(n: c_int, y: *mut f32, x: *const f32);
        pub fn oxe_swiglu_f32(n: c_int, y: *mut f32, gate: *const f32, up: *const f32);
        pub fn oxe_sigmoid_f32(n: i64, y: *mut f32, x: *const f32);
        pub fn oxe_softplus_f32(n: i64, y: *mut f32, x: *const f32);
        pub fn oxe_expf(x: f32) -> f32;
        pub fn oxe_vec_sum_f32(n: c_int, x: *const f32) -> f32;
        pub fn oxe_soft_max_f32(n: c_int, dp: *mut f32, sp: *const f32, scale: f32, mask: *const f32, wp: *mut f32);
        pub fn oxe_argsort_top_k_f32(x: *const f32, n: c_int, k: c_int, out: *mut i32, scratch: *mut i32);
        pub fn oxe_ssm_conv_step_f32(state: *mut f32, x_new: *const f32, c: *const f32, nr: c_int, nc: c_int, out: *mut f32);
        pub fn oxe_gated_delta_net_step_f32(
            s: *mut f32,
            q: *const f32,
            k: *const f32,
            v: *const f32,
            g: f32,
            beta: f32,
            s_v: i64,
            delta: *mut f32,
            attn: *mut f32,
        );
        /// The same step, passes fused and 4 state rows interleaved; bitwise the same (S_v % 16 == 0).
        pub fn oxe_gated_delta_net_step_f32_x4(
            s: *mut f32,
            q: *const f32,
            k: *const f32,
            v: *const f32,
            g: f32,
            beta: f32,
            s_v: i64,
            delta: *mut f32,
            attn: *mut f32,
        );
        /// OD-6b: the step reading the old state from `s_in` and writing the new one to `s` (bitwise the
        /// in-place step on a copy of `s_in`; `s_in == s` allowed).
        pub fn oxe_gated_delta_net_step_f32_from(
            s: *mut f32,
            s_in: *const f32,
            q: *const f32,
            k: *const f32,
            v: *const f32,
            g: f32,
            beta: f32,
            s_v: i64,
            delta: *mut f32,
            attn: *mut f32,
        );
        /// The fused 4-row step, out of place as [`oxe_gated_delta_net_step_f32_from`].
        pub fn oxe_gated_delta_net_step_f32_x4_from(
            s: *mut f32,
            s_in: *const f32,
            q: *const f32,
            k: *const f32,
            v: *const f32,
            g: f32,
            beta: f32,
            s_v: i64,
            delta: *mut f32,
            attn: *mut f32,
        );
        pub fn oxe_mrope_cache_init(
            p: i64,
            sections: *const i32,
            is_imrope: c_int,
            n_dims: c_int,
            ne0: i64,
            freq_base: f32,
            freq_scale: f32,
            ext_factor: f32,
            attn_factor: f32,
            beta_fast: f32,
            beta_slow: f32,
            n_ctx_orig: c_int,
            truncate: c_int,
            cache: *mut f32,
        );
        pub fn oxe_rope_neox_inplace(x: *mut f32, n_dims: c_int, cache: *const f32);
        pub fn oxe_attn_head_f16(
            q: *const f32,
            k: *const u16,
            k_stride: i64,
            k_last: *const u16,
            vt: *const u16,
            v_stride: i64,
            n_kv: c_int,
            pos_last: c_int,
            head_dim: c_int,
            scale: f32,
            q16: *mut u16,
            kq: *mut f32,
            mask: *mut f32,
            wp: *mut f32,
            p16: *mut u16,
            out: *mut f32,
        );
        /// Block of `ggml_vec_dot_f16` results, bit-identical per element (docs/lab-notes/M3-ATTN.md):
        /// out[r*ox + c*oy] = dot(n, x + r*xs, y + c*ys); optional per-column `n_full` tail rule.
        pub fn oxe_dot_f16_tile(
            n: c_int,
            x: *const u16,
            xs: i64,
            nx: c_int,
            y: *const u16,
            ys: i64,
            ny: c_int,
            out: *mut f32,
            ox: i64,
            oy: i64,
            n_full: *const c_int,
        );
        /// One FA-off softmax row (causal mask at `pos`, n_kv padded) evaluated on round_up(pos+1, 4)
        /// entries; f16 probabilities to p16[0, n_out), +0 beyond pos.
        /// `oxe_dot_f16_tile` blocked by position (`pb` positions of all columns per pass);
        /// `st` = nx * ny * 64 bytes of 16-byte aligned scratch. Bitwise the same result.
        pub fn oxe_dot_f16_tile_pb(
            n: c_int,
            x: *const u16,
            xs: i64,
            nx: c_int,
            y: *const u16,
            ys: i64,
            ny: c_int,
            out: *mut f32,
            ox: i64,
            oy: i64,
            n_full: *const c_int,
            pb: c_int,
            st: *mut u8,
        );
        pub fn oxe_attn_softmax_row_f16(kq: *const f32, pos: c_int, n_kv: c_int, scale: f32, wp: *mut f32, dp: *mut f32, p16: *mut u16, n_out: c_int);
        /// O-70 fused attention, one (KV head, chunk) item: chunk-local softmax of the unscaled scores
        /// `s` ([col][ss], overwritten; column c valid on [0, nvalid[c])), f16 weights exp(x - m_c)
        /// into `p16` (ncol * len), partial out[c*hd + d] = V^T[d][0..len) . p (f16 accumulation),
        /// ml[c] = m_c (-inf if empty), ml[ncol + c] = sum of the f16 weights. len % 32 == 0; the dot
        /// equals the one of length `len_full` (tail rule; V^T readable and finite up to it).
        pub fn oxe_attn_fchunk_f16(
            s: *mut f32,
            ss: i64,
            ncol: c_int,
            nvalid: *const c_int,
            len: c_int,
            len_full: c_int,
            scale: f32,
            vt: *const u16,
            v_stride: i64,
            hd: c_int,
            p16: *mut u16,
            out: *mut f32,
            ml: *mut f32,
        );
        /// O-70: merge one column's chunk partials (part + ch*ps, m/l at ch*mls) for dims [d0, d1):
        /// out[d] = (sum_ch exp(m_ch - M) part[ch][d]) / (sum_ch exp(m_ch - M) l_ch) * gate[d], in
        /// chunk order (independent of the dim range and of trailing empty chunks).
        /// w: round_up(nch, 4) floats of scratch.
        pub fn oxe_attn_fmerge_f32(
            nch: c_int,
            part: *const f32,
            ps: i64,
            m: *const f32,
            l: *const f32,
            mls: i64,
            d0: c_int,
            d1: c_int,
            gate: *const f32,
            out: *mut f32,
            w: *mut f32,
        );
        /// O-13: Q8_0 rows x Q8_0 columns, each element bitwise `ggml_vec_dot_q8_0_q8_0_generic`:
        /// out[r*ox + c*oy] = dot(n, x + r*xs, y + c*ys) (strides in bytes).
        pub fn oxe_dot_q8_tile(n: c_int, x: *const u8, xs: i64, nx: c_int, y: *const u8, ys: i64, ny: c_int, out: *mut f32, ox: i64, oy: i64);
        /// O-13: out[c*hd + d] = fma chain over p in [p0, p1) of (q*d)(V[p])[d] * f32(prob[c*ps + p]);
        /// V rows `vs` bytes apart; pscr = (p1-p0) * round_up(ncol, 4) floats of scratch.
        pub fn oxe_pv_q8(hd: c_int, v: *const u8, vs: i64, p0: c_int, p1: c_int, prob: *const u16, ps: i64, ncol: c_int, out: *mut f32, pscr: *mut f32);
        /// OD-4d: `oxe_pv_q8` bit for bit, faster (8 dims x 8 columns in registers, weights d * prob
        /// formed once per block). scr = 2 * (p1-p0) * round_up(ncol, 4) floats.
        pub fn oxe_pv_q8_fast(hd: c_int, v: *const u8, vs: i64, p0: c_int, p1: c_int, prob: *const u16, ps: i64, ncol: c_int, out: *mut f32, scr: *mut f32);
        /// OD-4d: scratch BYTES of `oxe_pv_q8_int` for np positions and ncol columns.
        pub fn oxe_pv_q8_int_scratch(hd: c_int, np: c_int, ncol: c_int) -> i64;
        /// OD-4d, LOSSY (KLD gated): integer V . p. The weights d * prob are quantised per chunk and
        /// (block, column) to `bits` = 16 or 8 unsigned bits and summed against the int8 quants in
        /// int32 (usmmla). Needs prob >= 0, d >= 0, p1 - p0 <= 256; scr 4-aligned.
        pub fn oxe_pv_q8_int(hd: c_int, v: *const u8, vs: i64, p0: c_int, p1: c_int, prob: *const u16, ps: i64, ncol: c_int, out: *mut f32, scr: *mut core::ffi::c_void, bits: c_int);
    }
    #[allow(dead_code)]
    fn _unused(_: c_int) {}
}

/// Activation format a weight type's GEMV consumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActType {
    Q8K,
    Q8_0,
}

impl ActType {
    pub const fn bytes(self, k: usize) -> usize {
        match self {
            ActType::Q8K => k / QK_K * Q8K_BLOCK_BYTES,
            ActType::Q8_0 => k / QK8_0 * Q8_0_BLOCK_BYTES,
        }
    }
}

/// Quantise `x` into activation blocks of type `at` (`out` must hold `at.bytes(x.len())`).
pub fn quantize_act(at: ActType, x: &[f32], out: &mut [u8]) {
    let k = x.len();
    assert!(out.len() >= at.bytes(k));
    // SAFETY: sizes checked; the C code writes exactly at.bytes(k) bytes.
    unsafe {
        match at {
            ActType::Q8K => {
                assert!(k % QK_K == 0);
                ffi::oxe_quantize_row_q8_K(x.as_ptr(), out.as_mut_ptr(), k as i64)
            }
            ActType::Q8_0 => {
                assert!(k % QK8_0 == 0);
                ffi::quantize_row_q8_0(x.as_ptr(), out.as_mut_ptr(), k as i64)
            }
        }
    }
}

/// The quantised weight types the engine's kernel set handles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QType {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

impl QType {
    pub const ALL: [QType; 5] = [QType::Q3K, QType::Q4K, QType::Q5K, QType::Q6K, QType::Q8_0];
    /// Values per block.
    pub const fn qk(self) -> usize {
        match self {
            QType::Q8_0 => QK8_0,
            _ => QK_K,
        }
    }
    /// Bytes of one block of one row (row-major ggml layout).
    pub const fn block_bytes(self) -> usize {
        match self {
            QType::Q3K => 110,
            QType::Q4K => 144,
            QType::Q5K => 176,
            QType::Q6K => 210,
            QType::Q8_0 => 34,
        }
    }
    /// Rows interleaved per repacked block (8 for the K-quants, 4 for q8_0_4x8).
    pub const fn nr(self) -> usize {
        match self {
            QType::Q8_0 => 4,
            _ => 8,
        }
    }
    /// Bytes of one interleaved block (`block_<t>x8` / `block_q8_0x4`); all size-neutral.
    pub const fn block_x_bytes(self) -> usize {
        self.nr() * self.block_bytes()
    }
    pub const fn act(self) -> ActType {
        match self {
            QType::Q8_0 => ActType::Q8_0,
            _ => ActType::Q8K,
        }
    }
    pub const fn name(self) -> &'static str {
        match self {
            QType::Q3K => "Q3_K",
            QType::Q4K => "Q4_K",
            QType::Q5K => "Q5_K",
            QType::Q6K => "Q6_K",
            QType::Q8_0 => "Q8_0",
        }
    }
    /// Bytes of one row of `k` columns.
    pub const fn row_bytes(self, k: usize) -> usize {
        k / self.qk() * self.block_bytes()
    }
    /// Byte offset of interleaved row group containing `row` (a multiple of nr()).
    pub const fn group_offset(self, row: usize, k: usize) -> usize {
        row / self.nr() * (k / self.qk()) * self.block_x_bytes()
    }
}

/// Repack `nrows` row-major rows of length `k` into the interleaved layout the GEMV kernels
/// expect (exactly ggml's `repack_<t>_to_<t>_<n>_bl` loop).
pub fn repack(ty: QType, src: &[u8], dst: &mut [u8], nrows: usize, k: usize) {
    let bytes = nrows * ty.row_bytes(k);
    assert!(nrows % ty.nr() == 0 && k % ty.qk() == 0, "repack needs nrows % {} == 0 and k % {} == 0", ty.nr(), ty.qk());
    assert!(src.len() >= bytes && dst.len() >= bytes);
    // SAFETY: both buffers hold at least nrows*k worth of blocks (checked above).
    let rc = unsafe {
        let (s, d, n, kk) = (src.as_ptr(), dst.as_mut_ptr(), nrows as i64, k as i64);
        match ty {
            QType::Q3K => ffi::oxe_repack_q3_K_8x8(s, d, n, kk),
            QType::Q4K => ffi::oxe_repack_q4_K_8x8(s, d, n, kk),
            QType::Q5K => ffi::oxe_repack_q5_K_8x8(s, d, n, kk),
            QType::Q6K => ffi::oxe_repack_q6_K_8x8(s, d, n, kk),
            QType::Q8_0 => ffi::oxe_repack_q8_0_4x8(s, d, n, kk),
        }
    };
    assert_eq!(rc, 0);
}

/// Back-compat name used by the M0 code and tests.
pub fn repack_8x8(ty: QType, src: &[u8], dst: &mut [u8], nrows: usize, k: usize) {
    repack(ty, src, dst, nrows, k)
}

/// Quantise `x` (length multiple of 256) into q8_K blocks.
pub fn quantize_q8k(x: &[f32], y: &mut [BlockQ8K]) {
    assert!(x.len() % QK_K == 0 && y.len() >= x.len() / QK_K);
    // SAFETY: y has x.len()/256 blocks; the C code writes exactly that many.
    unsafe { ffi::oxe_quantize_row_q8_K(x.as_ptr(), y.as_mut_ptr() as *mut u8, x.len() as i64) }
}

/// Raw GEMV: `out[0..nc) = W . a` where `w` points at the first interleaved block of the
/// row range and `a` holds the activation blocks for `k` (`ty.act()` format).
///
/// # Safety
/// `w` must point at `nc/nr * k/qk` readable interleaved blocks, `a` at `ty.act().bytes(k)`
/// readable bytes and `out` at `nc` writable floats; `nc % nr == 0`, `k % qk == 0`.
#[inline]
pub unsafe fn gemv_raw(ty: QType, k: usize, out: *mut f32, w: *const u8, a: *const u8, nc: usize) {
    debug_assert!(nc % ty.nr() == 0 && k % ty.qk() == 0);
    let (k, nc) = (k as c_int, nc as c_int);
    match ty {
        QType::Q3K => ffi::ggml_gemv_q3_K_8x8_q8_K(k, out, 0, w, a, 1, nc),
        QType::Q4K => ffi::ggml_gemv_q4_K_8x8_q8_K(k, out, 0, w, a, 1, nc),
        QType::Q5K => ffi::ggml_gemv_q5_K_8x8_q8_K(k, out, 0, w, a, 1, nc),
        QType::Q6K => ffi::ggml_gemv_q6_K_8x8_q8_K(k, out, 0, w, a, 1, nc),
        QType::Q8_0 => ffi::ggml_gemv_q8_0_4x8_q8_0(k, out, 0, w, a, 1, nc),
    }
}

/// Quantise 4 contiguous rows of `k` floats (`x[r*k..]`) into one 4-row interleaved activation
/// group (`ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>`: q8_K 4x8 generic, q8_0 4x8 NEON), which
/// occupies exactly `4 * at.bytes(k)` bytes.
///
/// # Safety
/// `x` must hold `4*k` floats and `out` `4 * at.bytes(k)` writable bytes.
#[inline]
pub unsafe fn quantize_mat4_raw(at: ActType, x: *const f32, out: *mut u8, k: usize) {
    match at {
        ActType::Q8K => ffi::ggml_quantize_mat_q8_K_4x8_generic(x, out, k as i64),
        ActType::Q8_0 => ffi::ggml_quantize_mat_q8_0_4x8(x, out, k as i64),
    }
}

/// One activation row (`from_float` of the weight type's vec_dot type).
///
/// # Safety
/// `x` must hold `k` floats and `out` `at.bytes(k)` writable bytes.
#[inline]
pub unsafe fn quantize_row_raw(at: ActType, x: *const f32, out: *mut u8, k: usize) {
    match at {
        ActType::Q8K => ffi::oxe_quantize_row_q8_K(x, out, k as i64),
        ActType::Q8_0 => ffi::quantize_row_q8_0(x, out, k as i64),
    }
}

/// Interleave 4 activation rows (each `at.bytes(k)`) into one 4-row tile
/// (`ggml_repack_interleave_q8_rows_x4`, the MUL_MAT_ID gemm path).
///
/// # Safety
/// Each row pointer must hold `at.bytes(k)` bytes; `out` `4 * at.bytes(k)` writable bytes.
#[inline]
pub unsafe fn interleave_rows4_raw(at: ActType, rows: &[*const u8; 4], out: *mut u8, k: usize) {
    match at {
        ActType::Q8K => ffi::oxe_interleave_q8_K_rows_x4(rows.as_ptr(), out, k as i64),
        ActType::Q8_0 => ffi::oxe_interleave_q8_0_rows_x4(rows.as_ptr(), out, k as i64),
    }
}

/// Interleaved GEMM of repack's MUL_MAT (`gemm<BLOC, INTER, NB_COLS, PARAM>`):
/// `s[r*bs + c] = W[c] . a[r]` for r < nr (multiple of 4), c < nc (multiple of the row group).
///
/// # Safety
/// As [`gemv_raw`], with `a` holding `nr/4` 4-row interleaved groups and `s` `nr` rows of stride `bs`.
#[inline]
pub unsafe fn gemm_raw(ty: QType, k: usize, s: *mut f32, bs: usize, w: *const u8, a: *const u8, nr: usize, nc: usize) {
    debug_assert!(nr % 4 == 0 && nc % ty.nr() == 0 && k % ty.qk() == 0);
    let (k, nr, nc) = (k as c_int, nr as c_int, nc as c_int);
    match ty {
        QType::Q3K => ffi::ggml_gemm_q3_K_8x8_q8_K(k, s, bs, w, a, nr, nc),
        QType::Q4K => ffi::ggml_gemm_q4_K_8x8_q8_K(k, s, bs, w, a, nr, nc),
        QType::Q5K => ffi::ggml_gemm_q5_K_8x8_q8_K(k, s, bs, w, a, nr, nc),
        QType::Q6K => ffi::ggml_gemm_q6_K_8x8_q8_K(k, s, bs, w, a, nr, nc),
        QType::Q8_0 => ffi::ggml_gemm_q8_0_4x8_q8_0(k, s, bs, w, a, nr, nc),
    }
}

/// Whether [`gemm_tile_raw`] has a multi-token tile kernel for `ty` (moe-prefill; otherwise it
/// is the reference [`gemm_raw`]).
pub fn gemm_tile_has(ty: QType) -> bool {
    // SAFETY: no arguments.
    matches!(ty, QType::Q4K | QType::Q6K) && unsafe { ffi::oxe_pf_tiles_available() } != 0
}

/// [`gemm_raw`] through the multi-token tile kernels (csrc/oxe_pf_tiles.inc): the weight rows
/// are decoded once and multiplied with all tokens in 8-token x 4-row SMMLA tiles. Bitwise
/// equal to [`gemm_raw`] (unit test `pf_tile_dense_bitwise`); types without a tile kernel use
/// the reference.
///
/// # Safety
/// As [`gemm_raw`].
#[inline]
pub unsafe fn gemm_tile_raw(ty: QType, k: usize, s: *mut f32, bs: usize, w: *const u8, a: *const u8, nr: usize, nc: usize) {
    debug_assert!(nr % 4 == 0 && nc % ty.nr() == 0 && k % ty.qk() == 0);
    match ty {
        QType::Q4K => ffi::oxe_pf_gemm_q4_K(k as c_int, s, bs, w, a, nr as c_int, nc as c_int),
        QType::Q6K => ffi::oxe_pf_gemm_q6_K(k as c_int, s, bs, w, a, nr as c_int, nc as c_int),
        _ => gemm_raw(ty, k, s, bs, w, a, nr, nc),
    }
}

/// Bytes of one packed pair block of the routed-expert tiles (`oxe_pf_xblk`: two from_float
/// q8_K rows, one 256-block).
pub const PF_XBLK_BYTES: usize = 688;

/// Whether [`mmid_tile_raw`] has a multi-token tile kernel for the MUL_MAT_ID of `ty`.
pub fn mmid_tile_has(ty: QType) -> bool {
    // SAFETY: no arguments.
    matches!(ty, QType::Q3K | QType::Q4K | QType::Q5K | QType::Q6K) && unsafe { ffi::oxe_pf_tiles_available() } != 0
}

/// Pack `rows.len()` from_float q8_K activation rows of `k` values into pairs for
/// [`mmid_tile_raw`]: `out` must hold `rows.len().div_ceil(2) * k / 256 * PF_XBLK_BYTES` bytes.
///
/// # Safety
/// Every row pointer must hold `ActType::Q8K.bytes(k)` readable bytes; `out` as stated, 4-aligned.
#[inline]
pub unsafe fn mmid_tile_pack_raw(rows: &[*const u8], k: usize, out: *mut u8) {
    ffi::oxe_pf_pack_q8K_pairs(rows.as_ptr(), rows.len() as c_int, k as c_int, out)
}

/// MUL_MAT_ID of one expert through the multi-token tiles: `orow[t][c] = W[c] . row t` for the
/// packed rows (csrc/oxe_pf_tiles.inc). Bitwise equal to [`gemm_mmid_raw`] and to the GEMV per
/// row (unit test `pf_tile_mmid_bitwise`). Returns false (nothing written) without a kernel.
///
/// # Safety
/// `w` as [`gemv_raw`] for `nc` rows; `xq` from [`mmid_tile_pack_raw`] with `orow.len()` rows;
/// each `orow[t]` must hold `nc` writable floats; `nc % 8 == 0`.
#[inline]
pub unsafe fn mmid_tile_raw(ty: QType, k: usize, orow: &[*mut f32], w: *const u8, xq: *const u8, nc: usize) -> bool {
    debug_assert!(nc % 8 == 0 && k % QK_K == 0);
    let t = match ty {
        QType::Q3K => 3,
        QType::Q4K => 4,
        QType::Q5K => 5,
        QType::Q6K => 6,
        _ => return false,
    };
    ffi::oxe_pf_mmid_tile(t, k as c_int, orow.as_ptr(), w, xq, orow.len() as c_int, nc as c_int) != 0
}

/// The MUL_MAT_ID gemm (`gemm_mmid`): the MUL_MAT gemm, except Q4_K/Q5_K 8x8 which use the
/// `_gemv_order` kernels (i8mm integer work, the gemv's float accumulation order).
///
/// # Safety
/// As [`gemm_raw`].
#[inline]
pub unsafe fn gemm_mmid_raw(ty: QType, k: usize, s: *mut f32, bs: usize, w: *const u8, a: *const u8, nr: usize, nc: usize) {
    debug_assert!(nr % 4 == 0 && nc % ty.nr() == 0 && k % ty.qk() == 0);
    match ty {
        QType::Q4K => ffi::ggml_gemm_q4_K_8x8_q8_K_gemv_order(k as c_int, s, bs, w, a, nr as c_int, nc as c_int),
        QType::Q5K => ffi::ggml_gemm_q5_K_8x8_q8_K_gemv_order(k as c_int, s, bs, w, a, nr as c_int, nc as c_int),
        _ => gemm_raw(ty, k, s, bs, w, a, nr, nc),
    }
}

/// The scalar `_generic` GEMM (reference for tests).
///
/// # Safety
/// As [`gemm_raw`].
pub unsafe fn gemm_generic_raw(ty: QType, k: usize, s: *mut f32, bs: usize, w: *const u8, a: *const u8, nr: usize, nc: usize) {
    let (k, nr, nc) = (k as c_int, nr as c_int, nc as c_int);
    match ty {
        QType::Q3K => ffi::ggml_gemm_q3_K_8x8_q8_K_generic(k, s, bs, w, a, nr, nc),
        QType::Q4K => ffi::ggml_gemm_q4_K_8x8_q8_K_generic(k, s, bs, w, a, nr, nc),
        QType::Q5K => ffi::ggml_gemm_q5_K_8x8_q8_K_generic(k, s, bs, w, a, nr, nc),
        QType::Q6K => ffi::ggml_gemm_q6_K_8x8_q8_K_generic(k, s, bs, w, a, nr, nc),
        QType::Q8_0 => ffi::ggml_gemm_q8_0_4x8_q8_0_generic(k, s, bs, w, a, nr, nc),
    }
}

/// Most lanes one [`gemv_lanes_raw`] call takes.
pub const VL_MAX_LANES: usize = 4;

/// Verify-lane GEMV (csrc/oxe_vl.inc, OD-6b): `outs[l][0..nc) = W . acts[l]` for 2..=4 lanes over the
/// same weight rows. The rows are read and unpacked once and dotted with the lanes in pairs (SMMLA).
/// Bitwise equal to [`gemv_raw`] called once per lane (unit test `vl_gemv_bitwise`). One lane is
/// accepted too (paired with itself; faster than [`gemv_raw`] only for Q6_K, about 1.15x on one
/// thread with hot weights). Returns false, with nothing written, when there is no lane kernel for
/// `ty` in this build or the lane count is outside 1..=4; the caller then runs [`gemv_raw`] per lane.
///
/// # Safety
/// As [`gemv_raw`] for every lane: `w` at `nc` rows of `k` weights, every `acts[l]` at
/// `ty.act().bytes(k)` readable bytes, every `outs[l]` at `nc` writable floats (lanes must not
/// overlap); `outs.len() == acts.len()`.
#[inline]
pub unsafe fn gemv_lanes_raw(ty: QType, k: usize, outs: &[*mut f32], w: *const u8, acts: &[*const u8], nc: usize) -> bool {
    debug_assert!(outs.len() == acts.len() && nc % ty.nr() == 0 && k % ty.qk() == 0);
    let nl = outs.len();
    if !(1..=VL_MAX_LANES).contains(&nl) || nc == 0 {
        return false;
    }
    let t = match ty {
        QType::Q3K => 3,
        QType::Q4K => 4,
        QType::Q5K => 5,
        QType::Q6K => 6,
        QType::Q8_0 => 80,
    };
    ffi::oxe_vl_gemv(t, k as c_int, outs.as_ptr(), w, acts.as_ptr(), nl as c_int, nc as c_int) != 0
}

/// Whether [`gemv_lanes_raw`] has a kernel for `ty` in this build.
pub fn gemv_lanes_has(ty: QType) -> bool {
    let k = ty.qk() * 8;
    let w = vec![0u8; ty.block_x_bytes() * 8];
    let a = vec![0u8; ty.act().bytes(k)];
    let mut o = [[0f32; 8]; 2];
    let (o0, o1) = o.split_at_mut(1);
    // SAFETY: one zeroed row group of k weights, two zeroed activations, two 8-float outputs (>= nr).
    unsafe { gemv_lanes_raw(ty, k, &[o0[0].as_mut_ptr(), o1[0].as_mut_ptr()], w.as_ptr(), &[a.as_ptr(), a.as_ptr()], ty.nr()) }
}

/// Safe wrapper of [`gemv_raw`] over a repacked weight slice and activation bytes.
pub fn gemv(ty: QType, k: usize, out: &mut [f32], w: &[u8], a: &[u8]) {
    let nc = out.len();
    assert!(nc % ty.nr() == 0 && k % ty.qk() == 0);
    assert!(w.len() >= nc * ty.row_bytes(k) && a.len() >= ty.act().bytes(k));
    // SAFETY: sizes checked above.
    unsafe { gemv_raw(ty, k, out.as_mut_ptr(), w.as_ptr(), a.as_ptr(), nc) }
}

/// The scalar `_generic` GEMV on the same interleaved layout (reference for tests).
pub fn gemv_generic(ty: QType, k: usize, out: &mut [f32], w: &[u8], a: &[u8]) {
    let nc = out.len();
    assert!(nc % ty.nr() == 0 && k % ty.qk() == 0);
    assert!(w.len() >= nc * ty.row_bytes(k) && a.len() >= ty.act().bytes(k));
    let (kk, ncc) = (k as c_int, nc as c_int);
    let (o, wp, ap) = (out.as_mut_ptr(), w.as_ptr(), a.as_ptr());
    // SAFETY: sizes checked above.
    unsafe {
        match ty {
            QType::Q3K => ffi::ggml_gemv_q3_K_8x8_q8_K_generic(kk, o, 0, wp, ap, 1, ncc),
            QType::Q4K => ffi::ggml_gemv_q4_K_8x8_q8_K_generic(kk, o, 0, wp, ap, 1, ncc),
            QType::Q5K => ffi::ggml_gemv_q5_K_8x8_q8_K_generic(kk, o, 0, wp, ap, 1, ncc),
            QType::Q6K => ffi::ggml_gemv_q6_K_8x8_q8_K_generic(kk, o, 0, wp, ap, 1, ncc),
            QType::Q8_0 => ffi::ggml_gemv_q8_0_4x8_q8_0_generic(kk, o, 0, wp, ap, 1, ncc),
        }
    }
}

/// Row-major reference: dot of one un-repacked weight row with an activation row.
pub fn vec_dot_generic(ty: QType, k: usize, row: &[u8], a: &[u8]) -> f32 {
    assert!(k % ty.qk() == 0 && row.len() >= ty.row_bytes(k) && a.len() >= ty.act().bytes(k));
    let mut s = 0f32;
    let kk = k as c_int;
    let (r, ap) = (row.as_ptr(), a.as_ptr());
    // SAFETY: sizes checked above; nrc = 1 writes one float.
    unsafe {
        match ty {
            QType::Q3K => ffi::ggml_vec_dot_q3_K_q8_K_generic(kk, &mut s, 0, r, 0, ap, 0, 1),
            QType::Q4K => ffi::ggml_vec_dot_q4_K_q8_K_generic(kk, &mut s, 0, r, 0, ap, 0, 1),
            QType::Q5K => ffi::ggml_vec_dot_q5_K_q8_K_generic(kk, &mut s, 0, r, 0, ap, 0, 1),
            QType::Q6K => ffi::ggml_vec_dot_q6_K_q8_K_generic(kk, &mut s, 0, r, 0, ap, 0, 1),
            QType::Q8_0 => ffi::ggml_vec_dot_q8_0_q8_0_generic(kk, &mut s, 0, r, 0, ap, 0, 1),
        }
    }
    s
}

/// Dequantise one row (`ggml_get_rows` -> `to_float`) of a supported embedding type.
pub fn dequantize_row(ty: QType, src: &[u8], out: &mut [f32]) {
    let k = out.len();
    assert!(k % ty.qk() == 0 && src.len() >= ty.row_bytes(k));
    // SAFETY: sizes checked.
    unsafe {
        let (s, o, kk) = (src.as_ptr(), out.as_mut_ptr(), k as i64);
        match ty {
            QType::Q8_0 => ffi::dequantize_row_q8_0(s, o, kk),
            QType::Q4K => ffi::dequantize_row_q4_K(s, o, kk),
            QType::Q5K => ffi::dequantize_row_q5_K(s, o, kk),
            QType::Q6K => ffi::dequantize_row_q6_K(s, o, kk),
            QType::Q3K => ffi::dequantize_row_q3_K(s, o, kk),
        }
    }
}

/// f32 -> f16 bits.
pub fn f32_to_f16(x: &[f32], y: &mut [u16]) {
    assert!(y.len() >= x.len());
    // SAFETY: y has at least x.len() elements.
    unsafe { ffi::oxe_f32_to_f16(x.as_ptr(), y.as_mut_ptr(), x.len() as i64) }
}

/// Safe wrappers of the ported ops (see csrc/oxe_ops.cpp for the upstream function each mirrors).
pub mod ops {
    use super::ffi;

    pub fn rms_norm(x: &[f32], y: &mut [f32], eps: f32) {
        assert!(y.len() >= x.len());
        unsafe { ffi::oxe_rms_norm_f32(x.as_ptr(), y.as_mut_ptr(), x.len() as i64, eps) }
    }
    /// y = rms_norm(x) * w (llama.cpp's fused RMS_NORM + MUL).
    pub fn rms_norm_mul(x: &[f32], w: &[f32], y: &mut [f32], eps: f32) {
        assert!(y.len() >= x.len() && w.len() >= x.len());
        unsafe { ffi::oxe_rms_norm_mul_f32(x.as_ptr(), w.as_ptr(), y.as_mut_ptr(), x.len() as i64, eps) }
    }
    pub fn gdn_l2_norm(x: &mut [f32], eps: f32) {
        unsafe { ffi::oxe_gdn_l2_norm_f32(x.as_mut_ptr(), x.len() as i64, eps) }
    }
    pub fn scale(y: &mut [f32], v: f32) {
        unsafe { ffi::oxe_vec_scale_f32(y.len() as i32, y.as_mut_ptr(), v) }
    }
    pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
        assert!(y.len() >= x.len());
        unsafe { ffi::oxe_vec_dot_f32(x.len() as i32, x.as_ptr(), y.as_ptr()) }
    }
    pub fn fp32_to_fp16(x: &[f32], y: &mut [u16]) {
        assert!(y.len() >= x.len());
        unsafe { ffi::oxe_fp32_to_fp16_row(x.as_ptr(), y.as_mut_ptr(), x.len() as i64) }
    }
    pub fn silu(x: &[f32], y: &mut [f32]) {
        assert!(y.len() >= x.len());
        unsafe { ffi::oxe_silu_f32(x.len() as i32, y.as_mut_ptr(), x.as_ptr()) }
    }
    pub fn swiglu(gate: &[f32], up: &[f32], y: &mut [f32]) {
        assert!(y.len() >= gate.len() && up.len() >= gate.len());
        unsafe { ffi::oxe_swiglu_f32(gate.len() as i32, y.as_mut_ptr(), gate.as_ptr(), up.as_ptr()) }
    }
    pub fn sigmoid(x: &[f32], y: &mut [f32]) {
        assert!(y.len() >= x.len());
        unsafe { ffi::oxe_sigmoid_f32(x.len() as i64, y.as_mut_ptr(), x.as_ptr()) }
    }
    pub fn sigmoid1(x: f32) -> f32 {
        let mut y = 0f32;
        unsafe { ffi::oxe_sigmoid_f32(1, &mut y, &x) };
        y
    }
    pub fn softplus(x: &[f32], y: &mut [f32]) {
        assert!(y.len() >= x.len());
        unsafe { ffi::oxe_softplus_f32(x.len() as i64, y.as_mut_ptr(), x.as_ptr()) }
    }
    pub fn sum(x: &[f32]) -> f32 {
        unsafe { ffi::oxe_vec_sum_f32(x.len() as i32, x.as_ptr()) }
    }
    /// ggml_soft_max (scale 1, no mask) of one row; `wp` is scratch of x.len() floats.
    pub fn soft_max(x: &[f32], y: &mut [f32], wp: &mut [f32]) {
        let n = x.len();
        assert!(y.len() >= n && wp.len() >= n);
        unsafe { ffi::oxe_soft_max_f32(n as i32, y.as_mut_ptr(), x.as_ptr(), 1.0, std::ptr::null(), wp.as_mut_ptr()) }
    }
    /// ggml_argsort_top_k (descending, std::sort tie order); scratch >= x.len().
    pub fn argsort_top_k(x: &[f32], k: usize, out: &mut [i32], scratch: &mut [i32]) {
        assert!(out.len() >= k && scratch.len() >= x.len() && k < x.len());
        unsafe { ffi::oxe_argsort_top_k_f32(x.as_ptr(), x.len() as i32, k as i32, out.as_mut_ptr(), scratch.as_mut_ptr()) }
    }
}

#[cfg(test)]
mod tests;

/// Scalar correctness oracle: native packed expert -> one temporary HMX tile.
/// No allocation or resident format change. Not a performance implementation.
pub fn native_expert_tile(ty: QType, src: &[u8], k: usize, rows: usize, experts: usize,
        expert: usize, row0: usize, k0: usize, out: &mut [u16;1024]) -> Result<(), &'static str> {
    let kind=match ty {QType::Q3K=>3,QType::Q4K=>4,_=>return Err("native expert type unsupported")};
    let cv=|n|u32::try_from(n).map_err(|_|"native expert geometry exceeds protocol");
    let (k,rows,experts,expert,row0,k0)=(cv(k)?,cv(rows)?,cv(experts)?,cv(expert)?,cv(row0)?,cv(k0)?);
    // SAFETY: source/output slices are disjoint and live; C validates all geometry,
    // source bounds and fixed tile capacity before indexing either slice.
    let rc=unsafe{ffi::oxe_native_expert_tile(out.as_mut_ptr(),1024,src.as_ptr(),src.len(),kind,k,rows,experts,expert,row0,k0)};
    match rc {0=>Ok(()),-2=>Err("non-finite native expert tile"),_=>Err("native expert tile geometry/bounds")}
}

#[cfg(test)]
mod native_expert_tests {
    use super::*;
    fn source(ty:QType,k:usize,rows:usize,experts:usize,seed:u32)->Vec<u8>{
        let block=ty.block_bytes();let mut data=vec![0u8;ty.row_bytes(k)*rows*experts];let mut rng=seed;
        for b in data.chunks_exact_mut(block){
            for v in b.iter_mut(){rng=rng.wrapping_mul(1664525).wrapping_add(1013904223);*v=(rng>>24)as u8;}
            let put=|b:&mut[u8],at:usize,h:u16|b[at..at+2].copy_from_slice(&h.to_le_bytes());
            // finite scales with varied mantissas/exponents, including subnormals
            let h=if rng&7==0{(rng as u16&0x03ff)|1}else{0x1000|(rng as u16&0x0fff)};
            if ty==QType::Q3K{put(b,108,h);}else{put(b,0,h);put(b,2,0x0c00|(rng as u16&0x03ff));}
        }
        data
    }
    #[test]
    fn native_tiles_equal_original_row_dequantization_for_all_elements(){
        for ty in [QType::Q3K,QType::Q4K]{for (k,rows,experts,seed) in [(256,32,1,1),(512,64,3,29),(1024,96,2,149)]{
            let raw=source(ty,k,rows,experts,seed);let mut packed=vec![0u8;raw.len()];
            repack(ty,&raw,&mut packed,rows*experts,k);
            let mut reference=vec![0f32;k*rows*experts];
            for (r,row) in reference.chunks_exact_mut(k).enumerate(){let bytes=ty.row_bytes(k);dequantize_row(ty,&raw[r*bytes..(r+1)*bytes],row);}
            let mut expected=vec![0u16;reference.len()];f32_to_f16(&reference,&mut expected);
            for e in 0..experts{for row0 in (0..rows).step_by(32){for k0 in (0..k).step_by(32){
                let mut tile=[0u16;1024];native_expert_tile(ty,&packed,k,rows,experts,e,row0,k0,&mut tile).unwrap();
                for n in 0..32{for c in 0..32{
                    let at=(c/2)*64+2*n+c%2;
                    assert_eq!(tile[at],expected[(e*rows+row0+n)*k+k0+c],"{ty:?} e{e} row{} k{}",row0+n,k0+c);
                }}
            }}}
        }}
    }
    #[test]
    fn malformed_geometry_refused_without_output_write(){
        let src=vec![0u8;3520];
        for (ty,k,rows,experts,e,r,c,len) in [
            (QType::Q5K,256,32,1,0,0,0,3520),(QType::Q3K,0,32,1,0,0,0,3520),
            (QType::Q3K,256,31,1,0,0,0,3520),(QType::Q3K,256,32,0,0,0,0,3520),
            (QType::Q3K,256,32,1,1,0,0,3520),(QType::Q3K,256,32,1,0,1,0,3520),
            (QType::Q3K,256,32,1,0,0,1,3520),(QType::Q3K,256,32,1,0,32,0,3520),
            (QType::Q3K,256,32,1,0,0,256,3520),(QType::Q3K,256,32,1,0,0,0,3519),
            (QType::Q3K,usize::MAX,32,1,0,0,0,3520),
        ]{
            let mut out=[0x2a2a;1024];assert!(native_expert_tile(ty,&src[..len],k,rows,experts,e,r,c,&mut out).is_err());assert_eq!(out,[0x2a2a;1024]);
        }
    }
}
