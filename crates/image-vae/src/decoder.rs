//! FLUX.2 VAE decoder (port of the decode path of sd.cpp `AutoEncoderKLModel`).
//!
//! # Layouts
//!
//! * Input latent `z`: f32, row-major `[channels][h][w]` (index `(c*h + y)*w + x`),
//!   `channels = 4 * z_channels` (128 for FLUX.2, `z_channels` = 32). The latent must
//!   already be de-normalised: sd.cpp does `z * std[c] + mean[c]` with fixed tables
//!   (`auto_encoder_kl.hpp`, `get_latents_mean_std`, called from
//!   `AutoEncoderKLModel::diffusion_to_vae_latents`) before calling decode. The
//!   `bn.running_mean/var` tensors of the weight file are NOT used by sd.cpp.
//! * Output: f32, row-major `[3][H][W]`, `H = h * 2 * 2^(levels-1)` (16*h for
//!   FLUX.2 with 4 levels). Raw decoder output, not clamped; sd.cpp maps it to
//!   pixels with `(x + 1) * 0.5` clamped to [0, 1] (see `crate::to_rgb`).
//!
//! # Forward pass (sd.cpp order)
//!
//! 1. un-patchify `[C*4][h][w]` -> `[C][2h][2w]` (see [`unpatchify`]).
//! 2. `post_quant_conv` (1x1, C -> z_channels).
//! 3. `decoder.conv_in` (3x3, z_channels -> mid channels).
//! 4. `mid.block_1`, `mid.attn_1` (single-head self-attention over h*w positions,
//!    residual), `mid.block_2`.
//! 5. up levels in the order sd.cpp runs them (highest index first). Each level has
//!    `num_res_blocks + 1` ResnetBlocks; every level except the last (index 0) ends
//!    with nearest x2 upsampling followed by a 3x3 conv.
//! 6. `norm_out` (GroupNorm), SiLU, `conv_out` (3x3 -> 3).
//!
//! Channel counts, number of levels and blocks per level are read from the tensor
//! names/shapes, not hard-coded.
//!
//! All GroupNorms use `eps = 1e-6` (sd.cpp `GroupNorm32`, not the 1e-5 default of
//! `GroupNorm`). Attention scale is `1/sqrt(C)` (one head, d_head = C).
//! Convolutions use zero padding `k/2` (1 for 3x3, 0 for 1x1) and always have a bias.

use std::collections::HashMap;
use std::time::Instant;

use crate::safetensors::SafeTensors;

mod nhwc;

/// Un-patchify factor (sd.cpp `int64_t p = 2`).
pub const PATCH: usize = 2;
/// GroupNorm group count used by sd.cpp (`GroupNorm32`).
pub const DEFAULT_GROUPS: usize = 32;
const GN_EPS: f64 = 1e-6;
/// Query rows per block in attention.
const ATTN_QB: usize = 32;

/// Tensor map used to build a decoder: name -> (shape, values).
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Channel-major feature map `[c][h][w]`.
#[derive(Clone)]
struct Feat {
    c: usize,
    h: usize,
    w: usize,
    d: Vec<f32>,
}

impl Feat {
    fn zeros(c: usize, h: usize, w: usize) -> Feat {
        Feat {
            c,
            h,
            w,
            d: vec![0.0; c * h * w],
        }
    }
}

struct Conv {
    /// `[cout][cin][k][k]` row-major.
    w: Vec<f32>,
    b: Vec<f32>,
    cout: usize,
    cin: usize,
    k: usize,
    /// GEMM-ordered weights `[cout block][tap][cin][GC]`, built on first use (see `Conv::forward`).
    wp: std::sync::OnceLock<Vec<f32>>,
    /// int8 weights for the SMMLA path, rows = output channels over `[tap][cin]`, built on first use.
    wq: std::sync::OnceLock<int8_gemm::Packed>,
    /// Folded weights of "nearest x2 upsample, then this 3x3 conv" (channel-last path, see `nhwc.rs`), f32 GEMM order.
    wup: std::sync::OnceLock<Vec<f32>>,
    /// The same folded weights as int8 rows, one matrix per output phase.
    wq_up: std::sync::OnceLock<Vec<int8_gemm::Packed>>,
    /// Winograd F(2x2, 3x3) transformed weights (channel-last path, see `nhwc.rs`).
    wwin: std::sync::OnceLock<Vec<f32>>,
}

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct ResBlock {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Attn {
    norm: Norm,
    q: Conv,
    k: Conv,
    v: Conv,
    proj: Conv,
}

struct UpLevel {
    blocks: Vec<ResBlock>,
    upsample: Option<Conv>,
}

/// The decoder. `decode` is pure: the same input always yields the same output,
/// and the output does not depend on `threads`.
pub struct VaeDecoder {
    groups: usize,
    post_quant: Conv,
    conv_in: Conv,
    mid1: ResBlock,
    mid_attn: Attn,
    mid2: ResBlock,
    /// Processing order: highest level index first.
    up: Vec<UpLevel>,
    norm_out: Norm,
    conv_out: Conv,
}

/// Un-patchify (sd.cpp `AutoEncoderKLModel::decode`, flux2 branch).
///
/// Derivation from the ggml sequence (ggml `ne` order is `[w, h, C*4, N]`, `w`
/// fastest). Let the input element be `z[c4][y][x]`, `c4 = 4*cc + q`, `q` the
/// index in the 2x2 group.
///
/// 1. `reshape(z, [w*h, 4, C, N])`: memory unchanged, so `q = c4 % 4`, `cc = c4 / 4`.
/// 2. `permute(1,0,2,3)` + cont: `[4, w*h, C, N]`, element `(q, i)` with `i = x + w*y`.
/// 3. `reshape(z, [2, 2, w, h*C*N])`: `a + 2*b = q`, so `a = q % 2`, `b = q / 2`.
/// 4. `permute(0,2,1,3)` (ggml axes 0,2,1,3) + cont: `[2, w, 2, h*C*N]`, element
///    `(a, x, b, yy)` with `yy = y + h*cc`.
/// 5. `reshape(z, [2w, 2h, C, N])`: element `(X, Y, cc)` with `X = a + 2x`,
///    `Y = b + 2y`.
///
/// Result: `out[cc][2y + b][2x + a] = in[4*cc + 2*b + a][y][x]` (row offset `b`,
/// column offset `a`). This equals the torch/diffusers reference
/// `x.reshape(C,2,2,h,w).permute(0,3,1,4,2).reshape(C,2h,2w)`. The unit test in
/// `tests/decoder_reference.rs` checks the formula by literally emulating the ggml
/// reshape/permute sequence on an index-valued tensor.
///
/// `z` is `[c*4][h][w]`; the result is `[c][2h][2w]`.
pub fn unpatchify(z: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let p = PATCH;
    assert_eq!(z.len(), c * p * p * h * w, "unpatchify: input length");
    let (oh, ow) = (h * p, w * p);
    let mut out = vec![0f32; c * oh * ow];
    for cc in 0..c {
        for b in 0..p {
            for a in 0..p {
                let in_c = cc * p * p + b * p + a;
                for y in 0..h {
                    let src = &z[(in_c * h + y) * w..][..w];
                    let dst_row = (cc * oh + p * y + b) * ow;
                    for x in 0..w {
                        out[dst_row + p * x + a] = src[x];
                    }
                }
            }
        }
    }
    out
}

/// Runs `f(chunk_index, chunk)` over consecutive chunks of `data`, distributing
/// chunks round-robin over `threads` scoped threads. Each chunk is written by exactly
/// one call, so results do not depend on the thread count.
fn for_each_chunk_mut<T: Send>(
    data: &mut [T],
    chunk: usize,
    threads: usize,
    f: &(dyn Fn(usize, &mut [T]) + Sync),
) {
    let n_chunks = data.len() / chunk.max(1);
    let t = threads.max(1).min(n_chunks.max(1));
    if t <= 1 {
        for (i, c) in data.chunks_mut(chunk).enumerate() {
            f(i, c);
        }
        return;
    }
    let mut groups: Vec<Vec<(usize, &mut [T])>> = (0..t).map(|_| Vec::new()).collect();
    for (i, c) in data.chunks_mut(chunk).enumerate() {
        groups[i % t].push((i, c));
    }
    std::thread::scope(|s| {
        for g in groups {
            s.spawn(move || {
                for (i, c) in g {
                    f(i, c);
                }
            });
        }
    });
}

/// Output channels computed together per task (one input-row load feeds all of them).
const CO_BLOCK: usize = 4;
/// Output columns held in registers per tile (4 NEON vectors per output channel).
const TILE: usize = 16;

/// Padded-input view used by the blocked convolution.
struct ConvJob<'a> {
    w: &'a [f32],
    b: &'a [f32],
    cin: usize,
    k: usize,
    padded: &'a [f32],
    plane_p: usize,
    pw: usize,
}

impl ConvJob<'_> {
    /// `NB` output channels x `TILE` columns starting at `(y, x0)`. The accumulation order
    /// per output element is bias, then (ci, ky, kx), the same as the reference loop.
    #[inline(always)]
    fn tile<const NB: usize>(&self, co0: usize, y: usize, x0: usize) -> [[f32; TILE]; NB] {
        let (cin, k, pw) = (self.cin, self.k, self.pw);
        let mut acc = [[0f32; TILE]; NB];
        for (j, a) in acc.iter_mut().enumerate() {
            *a = [self.b[co0 + j]; TILE];
        }
        for ci in 0..cin {
            let pin = &self.padded[ci * self.plane_p..];
            for ky in 0..k {
                let row = &pin[(y + ky) * pw..];
                for kx in 0..k {
                    let s: &[f32; TILE] = row[x0 + kx..x0 + kx + TILE].try_into().unwrap();
                    for (j, a) in acc.iter_mut().enumerate() {
                        let wv = self.w[((co0 + j) * cin + ci) * k * k + ky * k + kx];
                        for t in 0..TILE {
                            a[t] += wv * s[t];
                        }
                    }
                }
            }
        }
        acc
    }

    /// One output value, same accumulation order as [`ConvJob::tile`].
    fn scalar(&self, co: usize, y: usize, x: usize) -> f32 {
        let (cin, k, pw) = (self.cin, self.k, self.pw);
        let mut acc = self.b[co];
        for ci in 0..cin {
            for ky in 0..k {
                for kx in 0..k {
                    let wv = self.w[((co * cin + ci) * k + ky) * k + kx];
                    acc += wv * self.padded[ci * self.plane_p + (y + ky) * pw + x + kx];
                }
            }
        }
        acc
    }
}

fn store_tile<const NB: usize>(planes: &mut [&mut [f32]], w: usize, y: usize, x0: usize, acc: &[[f32; TILE]; NB]) {
    for (j, a) in acc.iter().enumerate() {
        planes[j][y * w + x0..y * w + x0 + TILE].copy_from_slice(a);
    }
}

/// Fills `nb` consecutive output planes (channels `co0..co0+nb`) of one block.
fn conv_block(job: &ConvJob, co0: usize, planes: &mut [&mut [f32]], h: usize, w: usize) {
    let nb = planes.len();
    for y in 0..h {
        let mut x0 = 0;
        while x0 + TILE <= w {
            match nb {
                4 => store_tile(planes, w, y, x0, &job.tile::<4>(co0, y, x0)),
                3 => store_tile(planes, w, y, x0, &job.tile::<3>(co0, y, x0)),
                2 => store_tile(planes, w, y, x0, &job.tile::<2>(co0, y, x0)),
                _ => store_tile(planes, w, y, x0, &job.tile::<1>(co0, y, x0)),
            }
            x0 += TILE;
        }
        for x in x0..w {
            for (j, p) in planes.iter_mut().enumerate() {
                p[y * w + x] = job.scalar(co0 + j, y, x);
            }
        }
    }
}

/// Process-wide switch for the lossy int8 convolution path (default off = f32).
static INT8_CONV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the int8 SMMLA convolution path on or off for the heavy 3x3 convolutions (process-wide).
/// Off (the default) is the f32 decoder. On is lossy; see `Conv::forward_int8` and VAE-SPEED.md.
pub fn set_int8_conv(on: bool) {
    INT8_CONV.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Process-wide switch for the Winograd form of the plain 3x3 f32 convolutions (channel-last path only).
static WINOGRAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the Winograd F(2x2, 3x3) form of the f32 3x3 convolutions on or off (process-wide).
/// It is f32 arithmetic with 16 instead of 36 products per 2x2 outputs; the result differs from
/// the direct convolution by rounding only (see VAE-SPEED.md for the measured distance).
/// Ignored while the int8 path is on.
pub fn set_winograd(on: bool) {
    WINOGRAD.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the Winograd form is on.
pub fn winograd() -> bool {
    WINOGRAD.load(std::sync::atomic::Ordering::Relaxed)
}

/// Select the convolution arithmetic of the decoder by name (process-wide):
/// `f32` = direct f32 (the default), `winograd` = f32 with the Winograd form of the plain 3x3
/// convolutions (rounding-level difference), `int8` = the lossy int8 path.
pub fn set_conv_mode(name: &str) -> Result<(), String> {
    let (int8, wino) = match name {
        "f32" => (false, false),
        "winograd" => (false, true),
        "int8" => (true, false),
        other => return Err(format!("--vae-conv must be f32, winograd or int8, got '{other}'")),
    };
    set_int8_conv(int8);
    set_winograd(wino);
    Ok(())
}

/// Whether the int8 convolution path is on.
pub fn int8_conv() -> bool {
    INT8_CONV.load(std::sync::atomic::Ordering::Relaxed)
}

/// Channels per activation/weight scale of the int8 convolution.
const I8_CHUNK: usize = 64;
/// Output pixels per batch of the int8 convolution (4 tile groups).
const I8_BATCH: usize = 32;
/// Weight row groups (of 8 output channels) per cache block of the int8 convolution.
const I8_ROW_GROUPS: usize = 8;

/// Output channels per GEMM tile (4 NEON vectors).
const GC: usize = 16;
/// Output pixels per GEMM tile.
const GP: usize = 4;

/// One GEMM tile of the convolution: `GP` consecutive output pixels x `GC` output channels,
/// accumulated over (tap, input channel) with the 16 accumulators in registers.
/// `wp`: `[taps][cin][GC]` weights of one output-channel block; `xin`: zero-padded NHWC input
/// with row length `pw` pixels; `bias`: `GC` values. Writes `GP * GC` floats to `out`.
///
/// # Safety
/// `xin` must hold rows `y .. y + k` and columns `x0 .. x0 + GP + k - 1` (in pixels of `cin` floats);
/// `wp` must hold `k * k * cin * GC` floats; `out` must hold `GP * GC` floats.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_tile(wp: *const f32, xin: *const f32, cin: usize, k: usize, pw: usize, y: usize, x0: usize, bias: *const f32, out: *mut f32) {
    use core::arch::aarch64::*;
    let (b0, b1, b2, b3) = (vld1q_f32(bias), vld1q_f32(bias.add(4)), vld1q_f32(bias.add(8)), vld1q_f32(bias.add(12)));
    let (mut a00, mut a01, mut a02, mut a03) = (b0, b1, b2, b3);
    let (mut a10, mut a11, mut a12, mut a13) = (b0, b1, b2, b3);
    let (mut a20, mut a21, mut a22, mut a23) = (b0, b1, b2, b3);
    let (mut a30, mut a31, mut a32, mut a33) = (b0, b1, b2, b3);
    let mut w = wp;
    for ky in 0..k {
        for kx in 0..k {
            let p0 = xin.add(((y + ky) * pw + x0 + kx) * cin);
            let (p1, p2, p3) = (p0.add(cin), p0.add(2 * cin), p0.add(3 * cin));
            for ci in 0..cin {
                let (w0, w1, w2, w3) = (vld1q_f32(w), vld1q_f32(w.add(4)), vld1q_f32(w.add(8)), vld1q_f32(w.add(12)));
                w = w.add(GC);
                let s0 = vld1q_dup_f32(p0.add(ci));
                let s1 = vld1q_dup_f32(p1.add(ci));
                let s2 = vld1q_dup_f32(p2.add(ci));
                let s3 = vld1q_dup_f32(p3.add(ci));
                a00 = vfmaq_f32(a00, w0, s0);
                a01 = vfmaq_f32(a01, w1, s0);
                a02 = vfmaq_f32(a02, w2, s0);
                a03 = vfmaq_f32(a03, w3, s0);
                a10 = vfmaq_f32(a10, w0, s1);
                a11 = vfmaq_f32(a11, w1, s1);
                a12 = vfmaq_f32(a12, w2, s1);
                a13 = vfmaq_f32(a13, w3, s1);
                a20 = vfmaq_f32(a20, w0, s2);
                a21 = vfmaq_f32(a21, w1, s2);
                a22 = vfmaq_f32(a22, w2, s2);
                a23 = vfmaq_f32(a23, w3, s2);
                a30 = vfmaq_f32(a30, w0, s3);
                a31 = vfmaq_f32(a31, w1, s3);
                a32 = vfmaq_f32(a32, w2, s3);
                a33 = vfmaq_f32(a33, w3, s3);
            }
        }
    }
    for (i, v) in [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33].into_iter().enumerate() {
        vst1q_f32(out.add(i * 4), v);
    }
}

/// Portable form of [`conv_tile`] (same order of operations up to FMA contraction).
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_tile(wp: *const f32, xin: *const f32, cin: usize, k: usize, pw: usize, y: usize, x0: usize, bias: *const f32, out: *mut f32) {
    let mut acc = [0f32; GP * GC];
    for p in 0..GP {
        for j in 0..GC {
            acc[p * GC + j] = *bias.add(j);
        }
    }
    let mut w = wp;
    for ky in 0..k {
        for kx in 0..k {
            for ci in 0..cin {
                for p in 0..GP {
                    let s = *xin.add(((y + ky) * pw + x0 + kx + p) * cin + ci);
                    for j in 0..GC {
                        acc[p * GC + j] = (*w.add(j)).mul_add(s, acc[p * GC + j]);
                    }
                }
                w = w.add(GC);
            }
        }
    }
    std::ptr::copy_nonoverlapping(acc.as_ptr(), out, GP * GC);
}

impl Conv {
    fn new(w: Vec<f32>, b: Vec<f32>, cout: usize, cin: usize, k: usize) -> Conv {
        Conv { w, b, cout, cin, k, wp: std::sync::OnceLock::new(), wq: std::sync::OnceLock::new(), wup: std::sync::OnceLock::new(), wq_up: std::sync::OnceLock::new(), wwin: std::sync::OnceLock::new() }
    }

    /// Weights in GEMM order `[cout block][tap][cin][GC]` (missing output channels of the last block are 0).
    fn packed(&self) -> &[f32] {
        self.wp.get_or_init(|| {
            let (cin, cout, taps) = (self.cin, self.cout, self.k * self.k);
            let nb = cout.div_ceil(GC);
            let mut p = vec![0f32; nb * taps * cin * GC];
            for co in 0..cout {
                let (cb, j) = (co / GC, co % GC);
                for ci in 0..cin {
                    for tap in 0..taps {
                        p[((cb * taps + tap) * cin + ci) * GC + j] = self.w[(co * cin + ci) * taps + tap];
                    }
                }
            }
            p
        })
    }

    /// Same-size convolution with zero padding `k/2` and bias, as a blocked GEMM.
    ///
    /// The input is copied once into a zero-padded channel-last layout, so that for one tap the
    /// `cin` values of a pixel are contiguous; a tile computes `GP` pixels x `GC` output channels
    /// with all 16 accumulators in registers (16 FMLA per 8 loads). Each thread works on one
    /// output-channel block (its weights, `taps * cin * GC` floats, stay in its cache) and a band
    /// of rows. Per output value the terms are added in (tap, input channel) order with fused
    /// multiply-adds: the result differs from the reference loop only by float rounding.
    fn forward(&self, x: &Feat, threads: usize) -> Feat {
        if int8_conv() && self.int8_ok(x) {
            return self.forward_int8(x, threads);
        }
        self.forward_f32(x, threads)
    }

    /// The f32 GEMM path (see [`Conv::forward`]).
    fn forward_f32(&self, x: &Feat, threads: usize) -> Feat {
        assert_eq!(x.c, self.cin, "conv input channels");
        let (cin, cout, k) = (self.cin, self.cout, self.k);
        let pad = k / 2;
        let (h, w) = (x.h, x.w);
        let w4 = w.div_ceil(GP) * GP;
        let (ph, pw) = (h + 2 * pad, w4 + 2 * pad);
        let hw = h * w;
        // 1. zero-padded channel-last copy, tiled so that few input planes are live at a time
        let mut xin = vec![0f32; ph * pw * cin];
        let xd = &x.d;
        for_each_chunk_mut(&mut xin, pw * cin, threads, &|yy: usize, row: &mut [f32]| {
            if yy < pad || yy >= h + pad {
                return;
            }
            let y = yy - pad;
            for x0 in (0..w).step_by(16) {
                let x1 = (x0 + 16).min(w);
                for c0 in (0..cin).step_by(4) {
                    let c1 = (c0 + 4).min(cin);
                    for ci in c0..c1 {
                        let src = &xd[ci * hw + y * w..][..w];
                        for xx in x0..x1 {
                            row[(xx + pad) * cin + ci] = src[xx];
                        }
                    }
                }
            }
        });
        // 2. GEMM tiles into a block-major buffer [cout block][h][w4][GC]
        let wp = self.packed();
        let nb = cout.div_ceil(GC);
        let taps = k * k;
        let mut bias = vec![0f32; nb * GC];
        bias[..cout].copy_from_slice(&self.b);
        let band = if h % 4 == 0 { h / 4 } else { 1 };
        let bands = h / band;
        let mut blk = vec![0f32; nb * h * w4 * GC];
        let xin_ref = &xin;
        let bias_ref = &bias;
        for_each_chunk_mut(&mut blk, band * w4 * GC, threads, &|i: usize, out: &mut [f32]| {
            let (cb, y0) = (i / bands, (i % bands) * band);
            let wb = &wp[cb * taps * cin * GC..(cb + 1) * taps * cin * GC];
            for dy in 0..band {
                for x0 in (0..w4).step_by(GP) {
                    // SAFETY: xin has rows y0+dy .. +k and columns x0 .. x0+GP+k-1 (pw = w4 + 2*pad);
                    // wb holds taps*cin*GC floats; out holds band*w4*GC floats.
                    unsafe {
                        conv_tile(wb.as_ptr(), xin_ref.as_ptr(), cin, k, pw, y0 + dy, x0, bias_ref.as_ptr().add(cb * GC), out.as_mut_ptr().add((dy * w4 + x0) * GC));
                    }
                }
            }
        });
        // 3. back to channel-major planes
        let mut out = vec![0f32; cout * hw];
        let blk_ref = &blk;
        for_each_chunk_mut(&mut out, hw, threads, &|co: usize, plane: &mut [f32]| {
            let (cb, j) = (co / GC, co % GC);
            let src = &blk_ref[cb * h * w4 * GC..(cb + 1) * h * w4 * GC];
            for y in 0..h {
                for xx in 0..w {
                    plane[y * w + xx] = src[(y * w4 + xx) * GC + j];
                }
            }
        });
        Feat { c: cout, h, w, d: out }
    }

    /// Shapes the int8 path takes: the heavy 3x3 convolutions (the 1x1 and small ones stay f32).
    fn int8_ok(&self, x: &Feat) -> bool {
        self.k == 3 && self.cin >= I8_CHUNK && self.cin % I8_CHUNK == 0 && self.cout % 8 == 0 && x.w % 8 == 0
    }

    fn packed_i8(&self) -> &int8_gemm::Packed {
        self.wq.get_or_init(|| {
            let (cin, cout, taps) = (self.cin, self.cout, self.k * self.k);
            let kk = taps * cin;
            let mut rows = vec![0f32; cout * kk];
            for co in 0..cout {
                for ci in 0..cin {
                    for tap in 0..taps {
                        rows[co * kk + tap * cin + ci] = self.w[(co * cin + ci) * taps + tap];
                    }
                }
            }
            int8_gemm::Packed::from_f32(&rows, cout, kk, I8_CHUNK)
        })
    }

    /// LOSSY int8 path: the convolution as an int8 x int8 GEMM on the SMMLA tiles of `int8-gemm`
    /// (four times the MAC rate of the f32 tile). Activations: int8 with one scale per pixel per
    /// `I8_CHUNK` channels; weights: int8 with one scale per output channel per `I8_CHUNK` inputs
    /// of one tap. A patch of an output pixel is the concatenation of 9 input pixels, so its
    /// chunk scales are simply those pixels' scales: no per-patch quantisation is needed.
    /// Per conv the error is of the order of 1% of the output RMS; it must be judged on the final
    /// image against the f32 decode (see VAE-SPEED.md).
    fn forward_int8(&self, x: &Feat, threads: usize) -> Feat {
        use int8_gemm::{gemm, Order, Packed, G};
        assert_eq!(x.c, self.cin, "conv input channels");
        let (cin, cout, k) = (self.cin, self.cout, self.k);
        let pad = k / 2;
        let (h, w) = (x.h, x.w);
        let (ph, pw) = (h + 2 * pad, w + 2 * pad);
        let hw = h * w;
        let nch = cin / I8_CHUNK;
        let kk = k * k * cin;
        // 1. channel-last copy of the input, then int8 codes and scales per (pixel, chunk); padding stays 0
        let mut xin = vec![0f32; ph * pw * cin];
        let xd = &x.d;
        for_each_chunk_mut(&mut xin, pw * cin, threads, &|yy: usize, row: &mut [f32]| {
            if yy < pad || yy >= h + pad {
                return;
            }
            let y = yy - pad;
            for x0 in (0..w).step_by(16) {
                let x1 = (x0 + 16).min(w);
                for ci in 0..cin {
                    let src = &xd[ci * hw + y * w..][..w];
                    for xx in x0..x1 {
                        row[(xx + pad) * cin + ci] = src[xx];
                    }
                }
            }
        });
        let mut qs = vec![0f32; ph * pw * nch];
        let xin_ref = &xin;
        for_each_chunk_mut(&mut qs, pw * nch, threads, &|yy: usize, srow: &mut [f32]| {
            let row = &xin_ref[yy * pw * cin..(yy + 1) * pw * cin];
            for (c, s) in srow.iter_mut().enumerate() {
                let mut amax = 0f32;
                for &v in &row[c * I8_CHUNK..(c + 1) * I8_CHUNK] {
                    amax = amax.max(v.abs());
                }
                *s = if amax > 0.0 && amax.is_finite() { amax / 127.0 } else { 0.0 };
            }
        });
        let mut q = vec![0i8; ph * pw * cin];
        let qs_ref = &qs;
        for_each_chunk_mut(&mut q, pw * cin, threads, &|yy: usize, qrow: &mut [i8]| {
            let row = &xin_ref[yy * pw * cin..(yy + 1) * pw * cin];
            let srow = &qs_ref[yy * pw * nch..(yy + 1) * pw * nch];
            for (c, (qc, xc)) in qrow.chunks_exact_mut(I8_CHUNK).zip(row.chunks_exact(I8_CHUNK)).enumerate() {
                let s = srow[c];
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                for (o, &v) in qc.iter_mut().zip(xc) {
                    *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        });
        drop(xin);
        // 2. per output row: batches of 32 pixels (4 tile groups) against blocks of 64 output channels
        let wq = self.packed_i8();
        let row_groups = cout / G;
        let mut outn = vec![0f32; hw * cout];
        let (q_ref, bias) = (&q, &self.b);
        for_each_chunk_mut(&mut outn, w * cout, threads, &|y: usize, orow: &mut [f32]| {
            let mut ybuf = vec![0f32; I8_BATCH * cout];
            for x0 in (0..w).step_by(I8_BATCH) {
                let nb = I8_BATCH.min(w - x0);
                let mut xp = Packed::zeroed(nb, kk, I8_CHUNK);
                for j in 0..nb {
                    for ky in 0..k {
                        for kx in 0..k {
                            let px = (y + ky) * pw + x0 + j + kx;
                            xp.set_codes(j, (ky * k + kx) * cin, &q_ref[px * cin..(px + 1) * cin], &qs_ref[px * nch..(px + 1) * nch]);
                        }
                    }
                }
                let mut rb = 0;
                while rb < row_groups {
                    let re = (rb + I8_ROW_GROUPS).min(row_groups);
                    gemm(&xp, wq, 0..nb / G, rb..re, &mut ybuf[rb * G..], cout, Order::MulAdd, false);
                    rb = re;
                }
                for t in 0..nb {
                    let dst = &mut orow[(x0 + t) * cout..(x0 + t + 1) * cout];
                    let src = &ybuf[t * cout..(t + 1) * cout];
                    for co in 0..cout {
                        dst[co] = src[co] + bias[co];
                    }
                }
            }
        });
        // 3. back to channel-major planes
        let mut out = vec![0f32; cout * hw];
        let outn_ref = &outn;
        for_each_chunk_mut(&mut out, hw, threads, &|co: usize, plane: &mut [f32]| {
            for (i, o) in plane.iter_mut().enumerate() {
                *o = outn_ref[i * cout + co];
            }
        });
        Feat { c: cout, h, w, d: out }
    }

    /// The earlier channel-major blocked kernel (4 output channels x 16 columns), bitwise equal to
    /// the reference loop; kept for A/B timing and as a second implementation in tests.
    #[allow(dead_code)]
    fn forward_blocked(&self, x: &Feat, threads: usize) -> Feat {
        assert_eq!(x.c, self.cin, "conv input channels");
        let (cin, cout, k) = (self.cin, self.cout, self.k);
        let pad = k / 2;
        let (h, w) = (x.h, x.w);
        let (ph, pw) = (h + 2 * pad, w + 2 * pad);
        let plane_p = ph * pw;
        // Zero-padded copy of the input so the inner loops need no bounds logic.
        let mut padded = vec![0f32; cin * plane_p];
        for ci in 0..cin {
            for y in 0..h {
                let src = &x.d[(ci * h + y) * w..][..w];
                let off = ci * plane_p + (y + pad) * pw + pad;
                padded[off..off + w].copy_from_slice(src);
            }
        }
        let hw = h * w;
        let mut out = vec![0f32; cout * hw];
        let job = ConvJob {
            w: &self.w,
            b: &self.b,
            cin,
            k,
            padded: &padded,
            plane_p,
            pw,
        };
        for_each_chunk_mut(&mut out, CO_BLOCK * hw, threads, &|bi: usize, blk: &mut [f32]| {
            let mut planes: Vec<&mut [f32]> = blk.chunks_mut(hw).collect();
            conv_block(&job, bi * CO_BLOCK, &mut planes, h, w);
        });
        Feat {
            c: cout,
            h,
            w,
            d: out,
        }
    }

    /// Straightforward per-channel loop kept as the reference for the blocked kernel.
    #[cfg(test)]
    fn forward_reference(&self, x: &Feat, threads: usize) -> Feat {
        assert_eq!(x.c, self.cin, "conv input channels");
        let (cin, cout, k) = (self.cin, self.cout, self.k);
        let pad = k / 2;
        let (h, w) = (x.h, x.w);
        let (ph, pw) = (h + 2 * pad, w + 2 * pad);
        let plane_p = ph * pw;
        let mut padded = vec![0f32; cin * plane_p];
        for ci in 0..cin {
            for y in 0..h {
                let src = &x.d[(ci * h + y) * w..][..w];
                let off = ci * plane_p + (y + pad) * pw + pad;
                padded[off..off + w].copy_from_slice(src);
            }
        }
        let hw = h * w;
        let mut out = vec![0f32; cout * hw];
        let weights = &self.w;
        let bias = &self.b;
        let padded = &padded;
        for_each_chunk_mut(&mut out, hw, threads, &|co: usize, plane: &mut [f32]| {
            plane.fill(bias[co]);
            for ci in 0..cin {
                let pin = &padded[ci * plane_p..(ci + 1) * plane_p];
                for ky in 0..k {
                    for kx in 0..k {
                        let wv = weights[((co * cin + ci) * k + ky) * k + kx];
                        for y in 0..h {
                            let src = &pin[(y + ky) * pw + kx..][..w];
                            let dst = &mut plane[y * w..][..w];
                            for (d, s) in dst.iter_mut().zip(src) {
                                *d += wv * *s;
                            }
                        }
                    }
                }
            }
        });
        Feat {
            c: cout,
            h,
            w,
            d: out,
        }
    }
}

/// GroupNorm with `groups` groups, f64 statistics, eps 1e-6, per-channel affine, optionally
/// followed by SiLU in the same pass. Parallel over groups; one task writes one group's
/// channels, so the result does not depend on the thread count.
fn group_norm_act(x: &Feat, norm: &Norm, groups: usize, silu: bool, threads: usize) -> Feat {
    let c = x.c;
    assert!(groups > 0 && c % groups == 0, "group_norm: {c} channels, {groups} groups");
    let cg = c / groups;
    let l = x.h * x.w;
    let mut out = Feat::zeros(c, x.h, x.w);
    for_each_chunk_mut(&mut out.d, cg * l, threads, &|g: usize, dst: &mut [f32]| {
        let src_all = &x.d[g * cg * l..(g + 1) * cg * l];
        let n = (cg * l) as f64;
        // statistics in f64, four lanes so the loops vectorise
        let mean = sum4(src_all, 0.0) / n;
        let var = sumsq4(src_all, mean) / n;
        let inv = 1.0 / (var + GN_EPS).sqrt();
        let mean32 = mean as f32;
        for cl in 0..cg {
            let ch = g * cg + cl;
            let a = (inv * norm.w[ch] as f64) as f32;
            let bc = norm.b[ch];
            let src = &src_all[cl * l..(cl + 1) * l];
            let d = &mut dst[cl * l..(cl + 1) * l];
            if silu {
                for (o, &v) in d.iter_mut().zip(src) {
                    let y = (v - mean32) * a + bc;
                    *o = y / (1.0 + exp_fast(-y));
                }
            } else {
                for (o, &v) in d.iter_mut().zip(src) {
                    *o = (v - mean32) * a + bc;
                }
            }
        }
    });
    out
}

/// `sum(v - shift)` in f64 with four independent lanes.
fn sum4(v: &[f32], shift: f64) -> f64 {
    let mut acc = [0f64; 4];
    let chunks = v.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..4 {
            acc[i] += c[i] as f64 - shift;
        }
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3];
    for &x in rem {
        s += x as f64 - shift;
    }
    s
}

/// `sum((v - shift)^2)` in f64 with four independent lanes.
fn sumsq4(v: &[f32], shift: f64) -> f64 {
    let mut acc = [0f64; 4];
    let chunks = v.chunks_exact(4);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..4 {
            let d = c[i] as f64 - shift;
            acc[i] += d * d;
        }
    }
    let mut s = acc[0] + acc[1] + acc[2] + acc[3];
    for &x in rem {
        let d = x as f64 - shift;
        s += d * d;
    }
    s
}

/// exp(x) for x clamped to [-87, 87], relative error about 1e-7 (Cephes expf polynomial), branch-free.
#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_2e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1) * r + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

fn add_inplace(dst: &mut Feat, src: &Feat) {
    assert_eq!(dst.d.len(), src.d.len());
    for (d, s) in dst.d.iter_mut().zip(&src.d) {
        *d += *s;
    }
}

/// Nearest-neighbour x2 upsampling.
fn upsample_nearest2(x: &Feat) -> Feat {
    let (h, w) = (x.h, x.w);
    let (oh, ow) = (2 * h, 2 * w);
    let mut out = Feat::zeros(x.c, oh, ow);
    for c in 0..x.c {
        for oy in 0..oh {
            let src = &x.d[(c * h + oy / 2) * w..][..w];
            let dst = &mut out.d[(c * oh + oy) * ow..][..ow];
            for (ox, o) in dst.iter_mut().enumerate() {
                *o = src[ox / 2];
            }
        }
    }
    out
}

impl ResBlock {
    fn forward(&self, x: &Feat, groups: usize, threads: usize) -> Feat {
        let h = group_norm_act(x, &self.norm1, groups, true, threads);
        let h = self.conv1.forward(&h, threads);
        let h2 = group_norm_act(&h, &self.norm2, groups, true, threads);
        let mut h = self.conv2.forward(&h2, threads);
        let shortcut_owned;
        let sc: &Feat = match &self.shortcut {
            Some(conv) => {
                shortcut_owned = conv.forward(x, threads);
                &shortcut_owned
            }
            None => x,
        };
        add_inplace(&mut h, sc);
        h
    }
}

impl Attn {
    fn forward(&self, x: &Feat, groups: usize, threads: usize) -> Feat {
        let (c, h, w) = (x.c, x.h, x.w);
        let l = h * w;
        let xn = group_norm_act(x, &self.norm, groups, false, threads);
        // Projections are [C][L]; attention works on position-major [L][C].
        let qt = transpose(&self.q.forward(&xn, threads).d, c, l);
        let kt = transpose(&self.k.forward(&xn, threads).d, c, l);
        let vt = transpose(&self.v.forward(&xn, threads).d, c, l);
        let o = attention(&qt, &kt, &vt, l, c, threads);
        let ot = Feat {
            c,
            h,
            w,
            d: transpose(&o, l, c),
        };
        let mut out = self.proj.forward(&ot, threads);
        add_inplace(&mut out, x);
        out
    }
}

/// Transposes a row-major `rows x cols` matrix into `cols x rows`, in 16x16 tiles.
///
/// A plain double loop writes `dst[c * rows + r]` for consecutive `c`: a write stride of
/// `rows * 4` bytes, which is 16 KB / 64 KB at the production mid-block sizes (4096 / 16384
/// positions), i.e. every write of a row lands in the same L1 set. Tiling bounds the live
/// lines to 16 per side. Element values are copied, so the result is identical.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const T: usize = 16;
    let mut dst = vec![0f32; rows * cols];
    for r0 in (0..rows).step_by(T) {
        let r1 = (r0 + T).min(rows);
        for c0 in (0..cols).step_by(T) {
            let c1 = (c0 + T).min(cols);
            for r in r0..r1 {
                for c in c0..c1 {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
    dst
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = acc.iter().sum::<f32>();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// Extra floats per score row. The score block is `ATTN_QB` rows of `l` floats read column-wise
/// (all rows for one key). With `l` = 4096 or 16384 (512 px / 1024 px) the row stride would be
/// 16 KB / 64 KB, so the 32 rows of a block would share one L1 set (6 ways); the pad makes
/// consecutive rows land 64 bytes further on. Addresses only: results are bit-identical.
const ATTN_ROW_PAD: usize = 16;

/// Single-head softmax attention. `q`, `k`, `v` are `[l][c]`; scale `1/sqrt(c)`.
/// Query rows are processed in blocks of `ATTN_QB`; within a block each key/value
/// row is reused for all queries of the block. Scores for one block are `ATTN_QB x l`
/// floats (row stride `l + ATTN_ROW_PAD`), so the full `l x l` matrix is never materialised.
fn attention(q: &[f32], k: &[f32], v: &[f32], l: usize, c: usize, threads: usize) -> Vec<f32> {
    attention_with_stride(q, k, v, l, c, threads, l + ATTN_ROW_PAD)
}

/// [`attention`] with an explicit score row stride `ls >= l` (tests compare strides bitwise).
fn attention_with_stride(q: &[f32], k: &[f32], v: &[f32], l: usize, c: usize, threads: usize, ls: usize) -> Vec<f32> {
    assert!(ls >= l);
    let scale = 1.0 / (c as f32).sqrt();
    let mut out = vec![0f32; l * c];
    for_each_chunk_mut(&mut out, ATTN_QB * c, threads, &|bi: usize, oblk: &mut [f32]| {
        let nb = oblk.len() / c;
        let i0 = bi * ATTN_QB;
        let mut s = vec![0f32; nb * ls];
        for j in 0..l {
            let kj = &k[j * c..][..c];
            for r in 0..nb {
                let qi = &q[(i0 + r) * c..][..c];
                s[r * ls + j] = dot(qi, kj) * scale;
            }
        }
        for r in 0..nb {
            let row = &mut s[r * ls..][..l];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for x in row.iter_mut() {
                *x = (*x - m).exp();
                sum += *x;
            }
            let inv = 1.0 / sum;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
        for j in 0..l {
            let vj = &v[j * c..][..c];
            for r in 0..nb {
                let p = s[r * ls + j];
                let od = &mut oblk[r * c..][..c];
                for (o, vv) in od.iter_mut().zip(vj) {
                    *o += p * *vv;
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod aliasing_tests {
    use super::*;

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn transpose_tiles_match_plain() {
        for (rows, cols) in [(1usize, 1usize), (5, 37), (16, 16), (33, 70), (64, 17)] {
            let src = fill(rows * cols, 3);
            let t = transpose(&src, rows, cols);
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(t[c * rows + r].to_bits(), src[r * cols + c].to_bits());
                }
            }
        }
    }

    #[test]
    fn padded_score_rows_are_bit_identical() {
        let (l, c) = (80usize, 24usize);
        let (q, k, v) = (fill(l * c, 1), fill(l * c, 2), fill(l * c, 5));
        let a = attention_with_stride(&q, &k, &v, l, c, 3, l);
        let b = attention(&q, &k, &v, l, c, 3);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    /// Production-shape fixture: the mid-block attention of a 512 px decode (4096 positions x 512
    /// channels) and, with OXI_VAE_FIXTURE_1024=1, of a 1024 px decode (16384 positions).
    /// Checks bit-identity of the padded stride against the aliased one and prints both times.
    /// Heavy: run explicitly in a DEV window,
    /// `cargo test --release -p image-vae -- --ignored --nocapture prod_shape`.
    #[test]
    #[ignore]
    fn prod_shape_attention_512_and_1024() {
        let threads = 12;
        let mut sizes = vec![4096usize];
        if std::env::var("OXI_VAE_FIXTURE_1024").is_ok() {
            sizes.push(16384);
        }
        for l in sizes {
            let c = 512;
            let (q, k, v) = (fill(l * c, 11), fill(l * c, 12), fill(l * c, 13));
            let t0 = std::time::Instant::now();
            let a = attention_with_stride(&q, &k, &v, l, c, threads, l);
            let t_plain = t0.elapsed().as_secs_f64();
            let t0 = std::time::Instant::now();
            let b = attention(&q, &k, &v, l, c, threads);
            let t_pad = t0.elapsed().as_secs_f64();
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
            let t0 = std::time::Instant::now();
            let tr = transpose(&a, l, c);
            let t_tr = t0.elapsed().as_secs_f64();
            assert_eq!(tr.len(), l * c);
            println!("vae attention fixture l={l} c={c} threads={threads}: stride l {t_plain:.3} s, stride l+{ATTN_ROW_PAD} {t_pad:.3} s, transpose {t_tr:.4} s");
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

fn take(t: &mut Tensors, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    t.remove(name)
        .ok_or_else(|| format!("vae: missing tensor {name}"))
}

/// Loads `base.weight` / `base.bias` as a `k x k` conv. A 2-D weight `[cout, cin]` is
/// accepted for `k == 1` (linear layout, identical memory).
fn load_conv(t: &mut Tensors, base: &str, k: usize) -> Result<Conv, String> {
    let (shape, w) = take(t, &format!("{base}.weight"))?;
    let (cout, cin) = match shape.as_slice() {
        &[co, ci, kh, kw] if kh == k && kw == k => (co, ci),
        &[co, ci] if k == 1 => (co, ci),
        _ => return Err(format!("vae: {base}.weight shape {shape:?}, expected [co, ci, {k}, {k}]")),
    };
    let (bshape, b) = take(t, &format!("{base}.bias"))?;
    if bshape != [cout] {
        return Err(format!("vae: {base}.bias shape {bshape:?}, expected [{cout}]"));
    }
    Ok(Conv::new(w, b, cout, cin, k))
}

fn load_norm(t: &mut Tensors, base: &str) -> Result<Norm, String> {
    let (ws, w) = take(t, &format!("{base}.weight"))?;
    let (bs, b) = take(t, &format!("{base}.bias"))?;
    if ws.len() != 1 || bs != ws {
        return Err(format!("vae: {base} norm shapes {ws:?} / {bs:?}"));
    }
    Ok(Norm { w, b })
}

fn load_resblock(t: &mut Tensors, base: &str) -> Result<ResBlock, String> {
    let norm1 = load_norm(t, &format!("{base}.norm1"))?;
    let conv1 = load_conv(t, &format!("{base}.conv1"), 3)?;
    let norm2 = load_norm(t, &format!("{base}.norm2"))?;
    let conv2 = load_conv(t, &format!("{base}.conv2"), 3)?;
    let shortcut = if t.contains_key(&format!("{base}.nin_shortcut.weight")) {
        Some(load_conv(t, &format!("{base}.nin_shortcut"), 1)?)
    } else {
        None
    };
    let cin = conv1.cin;
    let cout = conv2.cout;
    if norm1.w.len() != cin || conv1.cout != conv2.cin || norm2.w.len() != conv1.cout || cout != conv1.cout {
        return Err(format!("vae: {base}: inconsistent channel counts"));
    }
    match &shortcut {
        Some(s) if s.cin == cin && s.cout == cout => {}
        Some(_) => return Err(format!("vae: {base}: nin_shortcut channel mismatch")),
        None if cin == cout => {}
        None => return Err(format!("vae: {base}: channels change ({cin} -> {cout}) without nin_shortcut")),
    }
    Ok(ResBlock {
        norm1,
        conv1,
        norm2,
        conv2,
        shortcut,
    })
}

fn load_attn(t: &mut Tensors, base: &str) -> Result<Attn, String> {
    let norm = load_norm(t, &format!("{base}.norm"))?;
    let q = load_conv(t, &format!("{base}.q"), 1)?;
    let k = load_conv(t, &format!("{base}.k"), 1)?;
    let v = load_conv(t, &format!("{base}.v"), 1)?;
    let proj = load_conv(t, &format!("{base}.proj_out"), 1)?;
    let c = norm.w.len();
    for conv in [&q, &k, &v, &proj] {
        if conv.cin != c || conv.cout != c {
            return Err(format!("vae: {base}: attention projections must be {c}x{c}"));
        }
    }
    Ok(Attn {
        norm,
        q,
        k,
        v,
        proj,
    })
}

impl VaeDecoder {
    /// Builds the decoder from `decoder.*` tensors of a safetensors file.
    /// `groups` is the GroupNorm group count (sd.cpp uses 32).
    pub fn from_safetensors(st: &SafeTensors, groups: usize) -> Result<VaeDecoder, String> {
        let mut t: Tensors = HashMap::new();
        for name in st.names() {
            if name.starts_with("decoder.") {
                t.insert(name.to_string(), st.tensor_f32(name)?);
            }
        }
        VaeDecoder::from_tensors(t, groups)
    }

    /// Builds the decoder from an in-memory tensor map (names as in the safetensors
    /// file, e.g. `decoder.conv_in.weight`). Consumes the entries it uses.
    pub fn from_tensors(mut t: Tensors, groups: usize) -> Result<VaeDecoder, String> {
        if groups == 0 {
            return Err("vae: groups must be > 0".into());
        }
        // Discover the up-level layout from names.
        let mut n_levels = 0usize;
        while t.contains_key(&format!("decoder.up.{n_levels}.block.0.conv1.weight")) {
            n_levels += 1;
        }
        if n_levels == 0 {
            return Err("vae: no decoder.up.* levels found".into());
        }

        let post_quant = load_conv(&mut t, "decoder.post_quant_conv", 1)?;
        let conv_in = load_conv(&mut t, "decoder.conv_in", 3)?;
        let mid1 = load_resblock(&mut t, "decoder.mid.block_1")?;
        let mid_attn = load_attn(&mut t, "decoder.mid.attn_1")?;
        let mid2 = load_resblock(&mut t, "decoder.mid.block_2")?;

        let mut up = Vec::with_capacity(n_levels);
        for i in (0..n_levels).rev() {
            let mut blocks = Vec::new();
            while t.contains_key(&format!("decoder.up.{i}.block.{}.conv1.weight", blocks.len())) {
                let j = blocks.len();
                blocks.push(load_resblock(&mut t, &format!("decoder.up.{i}.block.{j}"))?);
            }
            let upsample = if i != 0 {
                Some(load_conv(&mut t, &format!("decoder.up.{i}.upsample.conv"), 3)?)
            } else {
                None
            };
            up.push(UpLevel { blocks, upsample });
        }
        let norm_out = load_norm(&mut t, "decoder.norm_out")?;
        let conv_out = load_conv(&mut t, "decoder.conv_out", 3)?;

        // Channel chain checks.
        let z = post_quant.cin;
        if post_quant.cout != z || conv_in.cin != z {
            return Err("vae: post_quant_conv / conv_in channel mismatch".into());
        }
        let mut ch = conv_in.cout;
        if mid1.conv1.cin != ch || mid2.conv1.cin != ch || mid_attn.norm.w.len() != ch {
            return Err("vae: mid block channel mismatch".into());
        }
        for level in &up {
            for b in &level.blocks {
                if b.conv1.cin != ch {
                    return Err(format!("vae: up block expects {} input channels, have {ch}", b.conv1.cin));
                }
                ch = b.conv2.cout;
            }
        }
        if norm_out.w.len() != ch || conv_out.cin != ch {
            return Err("vae: norm_out / conv_out channel mismatch".into());
        }
        let mut norm_channels = vec![norm_out.w.len(), mid_attn.norm.w.len()];
        for r in [&mid1, &mid2].into_iter().chain(up.iter().flat_map(|l| l.blocks.iter())) {
            norm_channels.push(r.norm1.w.len());
            norm_channels.push(r.norm2.w.len());
        }
        if let Some(n) = norm_channels.iter().find(|&&n| n % groups != 0) {
            return Err(format!("vae: {n} channels not divisible by {groups} groups"));
        }

        Ok(VaeDecoder {
            groups,
            post_quant,
            conv_in,
            mid1,
            mid_attn,
            mid2,
            up,
            norm_out,
            conv_out,
        })
    }

    /// Latent channels before un-patchify (`z_channels`, 32 for FLUX.2).
    pub fn z_channels(&self) -> usize {
        self.post_quant.cin
    }

    /// Latent channels expected by [`VaeDecoder::decode`] (`4 * z_channels`, 128 for FLUX.2).
    pub fn latent_channels(&self) -> usize {
        PATCH * PATCH * self.z_channels()
    }

    /// Number of output channels (3).
    pub fn out_channels(&self) -> usize {
        self.conv_out.cout
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Output `(H, W)` for a latent of `h x w` (16*h, 16*w for FLUX.2).
    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        let f = PATCH << (self.up.len() - 1);
        (h * f, w * f)
    }

    /// Decodes a latent `[channels][h][w]` (see module docs) into `[3][H][W]` f32.
    pub fn decode(&self, z: &[f32], channels: usize, h: usize, w: usize, threads: usize) -> Result<Vec<f32>, String> {
        self.decode_timed(z, channels, h, w, threads).map(|(out, _)| out)
    }

    /// Same as [`VaeDecoder::decode`], also returning wall-clock time per stage.
    /// Runs the channel-last path (`decoder/nhwc.rs`).
    pub fn decode_timed(&self, z: &[f32], channels: usize, h: usize, w: usize, threads: usize) -> Result<(Vec<f32>, DecodeTimes), String> {
        self.check_latent(z, channels, h, w)?;
        Ok(self.decode_nhwc(z, channels, h, w, threads))
    }

    fn check_latent(&self, z: &[f32], channels: usize, h: usize, w: usize) -> Result<(), String> {
        if channels != self.latent_channels() {
            return Err(format!("vae decode: {channels} latent channels, decoder expects {}", self.latent_channels()));
        }
        if h == 0 || w == 0 {
            return Err("vae decode: empty latent".into());
        }
        if z.len() != channels * h * w {
            return Err(format!("vae decode: latent has {} values, expected {}", z.len(), channels * h * w));
        }
        Ok(())
    }

    /// The channel-major decode of commit 8531151 (GEMM-tile convolutions with a layout copy
    /// around each, separate upsampling), kept as the A/B reference of the channel-last path.
    pub fn decode_timed_channel_major(
        &self,
        z: &[f32],
        channels: usize,
        h: usize,
        w: usize,
        threads: usize,
    ) -> Result<(Vec<f32>, DecodeTimes), String> {
        if channels != self.latent_channels() {
            return Err(format!(
                "vae decode: {channels} latent channels, decoder expects {}",
                self.latent_channels()
            ));
        }
        if h == 0 || w == 0 {
            return Err("vae decode: empty latent".into());
        }
        if z.len() != channels * h * w {
            return Err(format!(
                "vae decode: latent has {} values, expected {}",
                z.len(),
                channels * h * w
            ));
        }
        let mut tm = DecodeTimes::default();
        let t_all = Instant::now();
        let g = self.groups;
        let base_c = channels / (PATCH * PATCH);
        let (mut hh, mut ww) = (PATCH * h, PATCH * w);
        let t0 = Instant::now();
        let x = Feat {
            c: base_c,
            h: hh,
            w: ww,
            d: unpatchify(z, base_c, h, w),
        };
        let x = self.post_quant.forward(&x, threads);
        tm.push("unpatchify+post_quant", t0);
        let t0 = Instant::now();
        let mut x = self.conv_in.forward(&x, threads);
        tm.push("conv_in", t0);
        let t0 = Instant::now();
        x = self.mid1.forward(&x, g, threads);
        tm.push("mid_resnet_1", t0);
        let t0 = Instant::now();
        x = self.mid_attn.forward(&x, g, threads);
        tm.push("mid_attn", t0);
        let t0 = Instant::now();
        x = self.mid2.forward(&x, g, threads);
        tm.push("mid_resnet_2", t0);
        for (li, level) in self.up.iter().enumerate() {
            let t0 = Instant::now();
            for b in &level.blocks {
                x = b.forward(&x, g, threads);
            }
            tm.push(&format!("up_level{li}_resnets ({}x{})", x.h, x.w), t0);
            if let Some(up) = &level.upsample {
                let t0 = Instant::now();
                x = upsample_nearest2(&x);
                hh *= 2;
                ww *= 2;
                x = up.forward(&x, threads);
                tm.push(&format!("up_level{li}_upsample_conv"), t0);
            }
        }
        debug_assert_eq!((x.h, x.w), (hh, ww));
        let t0 = Instant::now();
        let x = group_norm_act(&x, &self.norm_out, g, true, threads);
        let x = self.conv_out.forward(&x, threads);
        tm.push("norm_out+conv_out", t0);
        tm.total = t_all.elapsed().as_secs_f64();
        Ok((x.d, tm))
    }
}

/// Wall-clock seconds per decode stage, in execution order.
#[derive(Debug, Clone, Default)]
pub struct DecodeTimes {
    pub stages: Vec<(String, f64)>,
    pub total: f64,
    /// Seconds per kind of work over the whole decode (channel-last path only; empty otherwise).
    pub kinds: Vec<(String, f64)>,
}

impl DecodeTimes {
    fn push(&mut self, name: &str, since: Instant) {
        self.stages.push((name.to_string(), since.elapsed().as_secs_f64()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpatchify_formula_small() {
        // c=1 output channel, latent [4][2][3] with value = index.
        let (c, h, w) = (1usize, 2usize, 3usize);
        let z: Vec<f32> = (0..c * 4 * h * w).map(|i| i as f32).collect();
        let out = unpatchify(&z, c, h, w);
        let (oh, ow) = (2 * h, 2 * w);
        for y in 0..h {
            for x in 0..w {
                for b in 0..2 {
                    for a in 0..2 {
                        let in_c = 2 * b + a;
                        let expect = z[(in_c * h + y) * w + x];
                        let got = out[(2 * y + b) * ow + 2 * x + a];
                        assert_eq!(got, expect, "y={y} x={x} b={b} a={a}");
                    }
                }
            }
        }
        assert_eq!(out.len(), c * oh * ow);
    }

    #[test]
    fn thread_chunks_cover_all() {
        for threads in [1usize, 2, 3, 8] {
            let mut v = vec![0u32; 10 * 4];
            for_each_chunk_mut(&mut v, 4, threads, &|i: usize, c: &mut [u32]| {
                for x in c.iter_mut() {
                    *x += i as u32 + 1;
                }
            });
            for (i, c) in v.chunks(4).enumerate() {
                assert!(c.iter().all(|&x| x == i as u32 + 1), "threads={threads}");
            }
        }
    }

    fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn blocked_conv_matches_reference() {
        // (cin, cout, k, h, w): odd sizes, widths below and above the 16-column tile,
        // cout not a multiple of CO_BLOCK, 1x1 and 3x3.
        let shapes = [
            (5usize, 7usize, 3usize, 9usize, 13usize),
            (8, 4, 1, 6, 16),
            (16, 16, 3, 5, 33),
            (3, 3, 3, 1, 1),
            (4, 9, 3, 4, 32),
            (12, 12, 1, 7, 5),
            (6, 10, 3, 2, 17),
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (cin, cout, k, h, w) in shapes {
            let conv = Conv {
                w: rand_vec(cout * cin * k * k, &mut seed).into_iter().map(|v| v * 0.3).collect(),
                b: rand_vec(cout, &mut seed),
                cout,
                cin,
                k,
                wp: std::sync::OnceLock::new(),
                wq: std::sync::OnceLock::new(),
                wup: std::sync::OnceLock::new(),
                wq_up: std::sync::OnceLock::new(),
                wwin: std::sync::OnceLock::new(),
            };
            let x = Feat {
                c: cin,
                h,
                w,
                d: rand_vec(cin * h * w, &mut seed),
            };
            let reference = conv.forward_reference(&x, 1);
            let blocked = conv.forward_blocked(&x, 3);
            assert!(blocked.d.iter().zip(&reference.d).all(|(a, b)| a.to_bits() == b.to_bits()), "blocked kernel is not bitwise equal to the reference");
            let fast = conv.forward(&x, 1);
            assert_eq!(fast.d.len(), reference.d.len());
            let num: f64 = fast.d.iter().zip(&reference.d).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = reference.d.iter().map(|b| (*b as f64).powi(2)).sum();
            let rel = (num / den).sqrt();
            assert!(rel < 1e-6, "cin={cin} cout={cout} k={k} {h}x{w}: rel RMSE {rel:e}");
            for threads in [2usize, 5, 12] {
                let many = conv.forward(&x, threads);
                assert!(
                    many.d.iter().zip(&fast.d).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "threads={threads} differs from 1 thread"
                );
            }
        }
    }

    #[test]
    fn int8_conv_is_close_to_f32_and_thread_invariant() {
        let mut seed = 0x51ed_270b_9f3c_1a77u64;
        for (cin, cout, h, w) in [(64usize, 16usize, 5usize, 8usize), (128, 64, 4, 40), (64, 72, 3, 16)] {
            let conv = Conv::new(rand_vec(cout * cin * 9, &mut seed).into_iter().map(|v| v * 0.05).collect(), rand_vec(cout, &mut seed), cout, cin, 3);
            let x = Feat { c: cin, h, w, d: rand_vec(cin * h * w, &mut seed) };
            assert!(conv.int8_ok(&x));
            let f = conv.forward_f32(&x, 2);
            let a = conv.forward_int8(&x, 1);
            let b = conv.forward_int8(&x, 5);
            assert!(a.d.iter().zip(&b.d).all(|(p, q)| p.to_bits() == q.to_bits()), "int8 conv depends on the thread count");
            let num: f64 = a.d.iter().zip(&f.d).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
            let den: f64 = f.d.iter().map(|q| (*q as f64).powi(2)).sum();
            let rel = (num / den).sqrt();
            assert!(rel < 0.03, "cin={cin} cout={cout} {h}x{w}: int8 vs f32 rel RMSE {rel:e}");
            assert!(rel > 1e-6, "int8 path suspiciously exact: is it running?");
        }
        // shapes the int8 path must refuse
        let small = Conv::new(vec![0.0; 8 * 32 * 9], vec![0.0; 8], 8, 32, 3);
        assert!(!small.int8_ok(&Feat::zeros(32, 4, 8)));
        let odd_w = Conv::new(vec![0.0; 8 * 64 * 9], vec![0.0; 8], 8, 64, 3);
        assert!(!odd_w.int8_ok(&Feat::zeros(64, 4, 12)));
    }

    fn rel_rmse(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let num: f64 = a.iter().zip(b).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = b.iter().map(|q| (*q as f64).powi(2)).sum();
        (num / den).sqrt()
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
    }

    /// Channel-last f32 conv: bitwise equal to the channel-major GEMM path (same order of fused
    /// multiply-adds), also with the fused residual add, for any thread count.
    #[test]
    fn nhwc_conv_is_bitwise_the_channel_major_gemm() {
        use nhwc::testing as t;
        let shapes = [
            (5usize, 7usize, 3usize, 9usize, 13usize),
            (8, 4, 1, 6, 16),
            (16, 16, 3, 5, 33),
            (3, 3, 3, 1, 1),
            (4, 9, 3, 4, 32),
            (12, 12, 1, 7, 5),
            (6, 10, 3, 2, 17),
            (32, 48, 3, 9, 8),
            (64, 35, 3, 3, 3),
            (20, 16, 3, 6, 2),
        ];
        let mut seed = 0x1357_9bdf_2468_ace0u64;
        for (cin, cout, k, h, w) in shapes {
            let conv = Conv::new(rand_vec(cout * cin * k * k, &mut seed).into_iter().map(|v| v * 0.3).collect(), rand_vec(cout, &mut seed), cout, cin, k);
            let x = Feat { c: cin, h, w, d: rand_vec(cin * h * w, &mut seed) };
            let res = Feat { c: cout, h, w, d: rand_vec(cout * h * w, &mut seed) };
            let reference = conv.forward_reference(&x, 1);
            let old = conv.forward_f32(&x, 2);
            let xn = t::to_nhwc(&x);
            let new = t::to_planes(&t::conv(&conv, &xn, false, None, 1));
            assert!(rel_rmse(&new.d, &reference.d) < 1e-6, "cin={cin} cout={cout} k={k} {h}x{w}");
            assert!(same_bits(&new.d, &old.d), "cin={cin} cout={cout} k={k} {h}x{w}: not bitwise the channel-major GEMM");
            let mut old_res = old.clone();
            add_inplace(&mut old_res, &res);
            let rn = t::to_nhwc(&res);
            for threads in [1usize, 3, 12] {
                let many = t::to_planes(&t::conv(&conv, &xn, false, None, threads));
                assert!(same_bits(&many.d, &new.d), "threads={threads}");
                let with_res = t::to_planes(&t::conv(&conv, &xn, false, Some(&rn), threads));
                assert!(same_bits(&with_res.d, &old_res.d), "residual, threads={threads}");
            }
        }
    }

    /// Fused nearest-upsample + 3x3 conv against "upsample, then the reference conv".
    #[test]
    fn nhwc_upsample_conv_matches_upsample_then_conv() {
        use nhwc::testing as t;
        let mut seed = 0x0f1e_2d3c_4b5a_6978u64;
        for (cin, cout, h, w) in [(5usize, 7usize, 3usize, 5usize), (16, 16, 1, 1), (8, 20, 4, 9), (32, 32, 6, 8), (12, 3, 2, 1), (4, 4, 1, 6)] {
            let conv = Conv::new(rand_vec(cout * cin * 9, &mut seed).into_iter().map(|v| v * 0.3).collect(), rand_vec(cout, &mut seed), cout, cin, 3);
            let x = Feat { c: cin, h, w, d: rand_vec(cin * h * w, &mut seed) };
            let want = conv.forward_reference(&upsample_nearest2(&x), 1);
            let xn = t::to_nhwc(&x);
            let got1 = t::conv(&conv, &xn, true, None, 1);
            assert_eq!((got1.c, got1.h, got1.w), (cout, 2 * h, 2 * w));
            let got = t::to_planes(&got1);
            let e = rel_rmse(&got.d, &want.d);
            assert!(e < 2e-6, "cin={cin} cout={cout} {h}x{w}: rel RMSE {e:e}");
            for threads in [2usize, 7] {
                let many = t::to_planes(&t::conv(&conv, &xn, true, None, threads));
                assert!(same_bits(&many.d, &got.d), "threads={threads}");
            }
        }
    }

    /// Channel-last int8 conv: bitwise the channel-major int8 path (same codes, scales and
    /// chunk order); the fused upsample form stays close to f32.
    #[test]
    fn nhwc_int8_conv_matches_channel_major_int8() {
        use nhwc::testing as t;
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let mut seed = 0x2468_ace0_1357_9bdfu64;
        for (cin, cout, h, w) in [(64usize, 16usize, 5usize, 8usize), (128, 64, 4, 40), (64, 72, 3, 16), (192, 8, 1, 8)] {
            let conv = Conv::new(rand_vec(cout * cin * 9, &mut seed).into_iter().map(|v| v * 0.05).collect(), rand_vec(cout, &mut seed), cout, cin, 3);
            let x = Feat { c: cin, h, w, d: rand_vec(cin * h * w, &mut seed) };
            let res = Feat { c: cout, h, w, d: rand_vec(cout * h * w, &mut seed) };
            let xn = t::to_nhwc(&x);
            assert!(t::int8_ok(&conv, w));
            let old = conv.forward_int8(&x, 2);
            let new = t::to_planes(&t::conv_int8(&conv, &xn, false, None, 1));
            assert!(same_bits(&new.d, &old.d), "cin={cin} cout={cout} {h}x{w}: not bitwise the channel-major int8 path");
            let mut old_res = old.clone();
            add_inplace(&mut old_res, &res);
            let with_res = t::to_planes(&t::conv_int8(&conv, &xn, false, Some(&t::to_nhwc(&res)), 5));
            assert!(same_bits(&with_res.d, &old_res.d), "residual");
            // fused upsample
            let want = conv.forward_reference(&upsample_nearest2(&x), 1);
            let up1 = t::to_planes(&t::conv_int8(&conv, &xn, true, None, 1));
            let up5 = t::to_planes(&t::conv_int8(&conv, &xn, true, None, 5));
            assert!(same_bits(&up1.d, &up5.d), "int8 upsample conv depends on the thread count");
            let e = rel_rmse(&up1.d, &want.d);
            assert!(e < 0.03 && e > 1e-6, "cin={cin} cout={cout} {h}x{w}: int8 upsample conv rel RMSE {e:e}");
        }
        assert!(!t::int8_ok(&Conv::new(vec![0.0; 8 * 32 * 9], vec![0.0; 8], 8, 32, 3), 8));
        assert!(!t::int8_ok(&Conv::new(vec![0.0; 8 * 64 * 9], vec![0.0; 8], 8, 64, 3), 12));
    }

    /// Channel-last GroupNorm (+ SiLU) against the channel-major one (statistics are summed in a
    /// different order, so close, not bitwise); in place == out of place; thread invariant.
    #[test]
    fn nhwc_group_norm_matches_channel_major() {
        use nhwc::testing as t;
        let mut seed = 0x7777_1234_abcd_0001u64;
        for (c, groups, h, w) in [(32usize, 32usize, 5usize, 7usize), (64, 32, 8, 8), (24, 4, 3, 1), (6, 3, 2, 5), (128, 32, 4, 4)] {
            let norm = Norm { w: rand_vec(c, &mut seed), b: rand_vec(c, &mut seed) };
            let x = Feat { c, h, w, d: rand_vec(c * h * w, &mut seed).into_iter().map(|v| v * 3.0 + 0.7).collect() };
            let xn = t::to_nhwc(&x);
            for silu in [false, true] {
                let want = group_norm_act(&x, &norm, groups, silu, 2);
                let a = t::norm_silu(&xn, &norm, groups, silu, false, 1);
                let b = t::norm_silu(&xn, &norm, groups, silu, true, 5);
                assert!(same_bits(&a.d[..c * h * w], &b.d[..c * h * w]), "in place / threads");
                let e = rel_rmse(&t::to_planes(&a).d, &want.d);
                assert!(e < 1e-6, "c={c} groups={groups} silu={silu}: rel RMSE {e:e}");
            }
        }
    }

    /// Winograd F(2x2, 3x3) against the reference loop: rounding-level distance, thread invariant,
    /// residual add included.
    #[test]
    fn nhwc_winograd_conv_matches_reference() {
        use nhwc::testing as t;
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let mut seed = 0x0bad_cafe_1234_5678u64;
        for (cin, cout, h, w) in [(32usize, 32usize, 2usize, 2usize), (32, 8, 4, 6), (64, 20, 6, 2), (36, 48, 10, 14), (128, 16, 2, 300), (32, 4, 34, 2), (256, 204, 4, 6), (224, 256, 2, 2), (256, 256, 18, 10)] {
            // the last three have transformed weights above 3 MB: the shared super-band mode
            let conv = Conv::new(rand_vec(cout * cin * 9, &mut seed).into_iter().map(|v| v * 0.1).collect(), rand_vec(cout, &mut seed), cout, cin, 3);
            let x = Feat { c: cin, h, w, d: rand_vec(cin * h * w, &mut seed) };
            let res = Feat { c: cout, h, w, d: rand_vec(cout * h * w, &mut seed) };
            let want = conv.forward_reference(&x, 1);
            let xn = t::to_nhwc(&x);
            assert!(t::wino_ok(&conv, h, w));
            let a = t::to_planes(&t::conv_winograd(&conv, &xn, None, 1));
            let e = rel_rmse(&a.d, &want.d);
            assert!(e < 3e-6, "cin={cin} cout={cout} {h}x{w}: rel RMSE {e:e}");
            let b = t::to_planes(&t::conv_winograd(&conv, &xn, None, 7));
            assert!(same_bits(&a.d, &b.d), "winograd conv depends on the thread count");
            let mut want_res = a.clone();
            add_inplace(&mut want_res, &res);
            let c = t::to_planes(&t::conv_winograd(&conv, &xn, Some(&t::to_nhwc(&res)), 3));
            assert!(same_bits(&c.d, &want_res.d), "residual");
        }
        // shapes it must refuse: odd size, few channels
        assert!(!t::wino_ok(&Conv::new(vec![0.0; 32 * 32 * 9], vec![0.0; 32], 32, 32, 3), 3, 4));
        assert!(!t::wino_ok(&Conv::new(vec![0.0; 3 * 32 * 9], vec![0.0; 3], 3, 32, 3), 4, 4));
        assert!(!t::wino_ok(&Conv::new(vec![0.0; 32 * 32], vec![0.0; 32], 32, 32, 1), 4, 4));
    }

    /// GEMM attention against the dot-product attention of the channel-major path.
    #[test]
    fn nhwc_attention_matches_channel_major() {
        let mut seed = 0x4242_4242_1111_0001u64;
        for (l, c) in [(6usize, 8usize), (64, 32), (37, 20), (150, 16), (256, 64)] {
            let q = rand_vec(l * c, &mut seed);
            let k = rand_vec(l * c, &mut seed);
            let v = rand_vec(l * c, &mut seed);
            let want = attention(&q, &k, &v, l, c, 2);
            let a = nhwc::testing::attention(&q, &k, &v, l, c, 1);
            let b = nhwc::testing::attention(&q, &k, &v, l, c, 5);
            assert!(same_bits(&a, &b), "l={l} c={c}: attention depends on the thread count");
            let e = rel_rmse(&a, &want);
            assert!(e < 2e-6, "l={l} c={c}: rel RMSE {e:e}");
        }
    }

    /// Kernel A/B in one process (interleaved, so machine load hits all alike). Run under
    /// `tools/lane_quiet.sh cargo test --release -p image-vae --lib -- --ignored --nocapture bench_nhwc_ab`.
    #[test]
    #[ignore]
    fn bench_nhwc_ab() {
        use nhwc::testing as t;
        let mut seed = 0x1234_5678_9abc_def1u64;
        for (cin, cout, h, w) in [(512usize, 512usize, 128usize, 128usize), (256, 256, 256, 256), (128, 128, 512, 512)] {
            let conv = Conv::new(rand_vec(cout * cin * 9, &mut seed).into_iter().map(|v| v * 0.05).collect(), rand_vec(cout, &mut seed), cout, cin, 3);
            let x = Feat { c: cin, h, w, d: rand_vec(cin * h * w, &mut seed) };
            let xn = t::to_nhwc(&x);
            let gmac = (cout * cin * 9 * h * w) as f64 / 1e9;
            for _ in 0..3 {
                let t0 = Instant::now();
                let a = conv.forward_f32(&x, 12);
                let t_old = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let b = t::conv(&conv, &xn, false, None, 12);
                let t_new = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let a8 = conv.forward_int8(&x, 12);
                let t_old8 = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let b8 = t::conv_int8(&conv, &xn, false, None, 12);
                let t_new8 = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let bw = t::conv_winograd(&conv, &xn, None, 12);
                let t_w = t0.elapsed().as_secs_f64();
                println!("    winograd f32: {t_w:.4} s ({:.0} GMAC/s equivalent), rel RMSE vs direct {:.1e}", gmac / t_w, rel_rmse(&bw.d[..b.d.len().min(bw.d.len()).min(cout * h * w)], &b.d[..cout * h * w]));
                let bitwise = same_bits(&t::to_planes(&b).d, &a.d);
                let bitwise8 = same_bits(&t::to_planes(&b8).d, &a8.d);
                println!(
                    "conv3x3 {cin}->{cout} {h}x{w} ({gmac:.1} GMAC): f32 channel-major {t_old:.4} s ({:.0} GMAC/s), channel-last {t_new:.4} s ({:.0} GMAC/s, bitwise {bitwise}); int8 channel-major {t_old8:.4} s ({:.0}), channel-last incl. quantise {t_new8:.4} s ({:.0}, bitwise {bitwise8})",
                    gmac / t_old,
                    gmac / t_new,
                    gmac / t_old8,
                    gmac / t_new8
                );
            }
        }
        // upsample + conv: separate (channel-major) against fused (channel-last)
        for (c, h, w) in [(512usize, 128usize, 128usize), (256, 256, 256)] {
            let conv = Conv::new(rand_vec(c * c * 9, &mut seed).into_iter().map(|v| v * 0.05).collect(), rand_vec(c, &mut seed), c, c, 3);
            let x = Feat { c, h, w, d: rand_vec(c * h * w, &mut seed) };
            let xn = t::to_nhwc(&x);
            for _ in 0..2 {
                let t0 = Instant::now();
                let a = conv.forward_f32(&upsample_nearest2(&x), 12);
                let t_old = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let b = t::conv(&conv, &xn, true, None, 12);
                let t_new = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let b8 = t::conv_int8(&conv, &xn, true, None, 12);
                let t_new8 = t0.elapsed().as_secs_f64();
                println!(
                    "upsample+conv3x3 {c}->{c} from {h}x{w}: separate f32 {t_old:.4} s, fused f32 {t_new:.4} s (rel RMSE {:.1e}), fused int8 {t_new8:.4} s (rel RMSE {:.1e})",
                    rel_rmse(&t::to_planes(&b).d, &a.d),
                    rel_rmse(&t::to_planes(&b8).d, &a.d)
                );
            }
        }
    }

    /// A/B of the blocked kernel against the reference loop, interleaved in one process so
    /// machine load affects both equally. Run: `cargo test --release -p image-vae --lib -- --ignored --nocapture bench_conv_ab`.
    #[test]
    #[ignore]
    fn bench_conv_ab() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for (cin, cout, k, h, w) in [(512usize, 512usize, 3usize, 128usize, 128usize), (256, 256, 3, 256, 256), (128, 128, 3, 512, 512)] {
            let conv = Conv {
                w: rand_vec(cout * cin * k * k, &mut seed).into_iter().map(|v| v * 0.05).collect(),
                b: rand_vec(cout, &mut seed),
                cout,
                cin,
                k,
                wp: std::sync::OnceLock::new(),
                wq: std::sync::OnceLock::new(),
                wup: std::sync::OnceLock::new(),
                wq_up: std::sync::OnceLock::new(),
                wwin: std::sync::OnceLock::new(),
            };
            let x = Feat {
                c: cin,
                h,
                w,
                d: rand_vec(cin * h * w, &mut seed),
            };
            for _ in 0..2 {
                let t0 = Instant::now();
                let a = conv.forward_reference(&x, 12);
                let t_ref = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let b = conv.forward_blocked(&x, 12);
                let t_blk = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let c = conv.forward(&x, 12);
                let t_gemm = t0.elapsed().as_secs_f64();
                let t0 = Instant::now();
                let d8 = conv.forward_int8(&x, 12);
                let t_i8 = t0.elapsed().as_secs_f64();
                let n8: f64 = d8.d.iter().zip(&a.d).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
                let d8den: f64 = a.d.iter().map(|q| (*q as f64).powi(2)).sum();
                println!("    int8: {t_i8:.4} s = {:.0} GMAC/s (rel RMSE {:.1e})", (cout * cin * k * k * h * w) as f64 / 1e9 / t_i8, (n8 / d8den).sqrt());
                let same = a.d.iter().zip(&b.d).all(|(p, q)| p.to_bits() == q.to_bits());
                let num: f64 = c.d.iter().zip(&a.d).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
                let den: f64 = a.d.iter().map(|q| (*q as f64).powi(2)).sum();
                let gmac = (cout * cin * k * k * h * w) as f64 / 1e9;
                println!(
                    "conv3x3 {cin}->{cout} {h}x{w} ({gmac:.1} GMAC): reference {t_ref:.4} s, blocked {t_blk:.4} s (bitwise {same}), gemm {t_gemm:.4} s = {:.0} GMAC/s (rel RMSE {:.1e})",
                    gmac / t_gemm,
                    (num / den).sqrt()
                );
            }
        }
    }

    #[test]
    fn dot_matches_naive() {
        let a: Vec<f32> = (0..19).map(|i| (i as f32) * 0.37 - 2.0).collect();
        let b: Vec<f32> = (0..19).map(|i| 1.0 - (i as f32) * 0.11).collect();
        let naive: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!((dot(&a, &b) as f64 - naive).abs() < 1e-4);
    }
}
