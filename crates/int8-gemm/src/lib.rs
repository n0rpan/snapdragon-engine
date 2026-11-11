//! Shared int8 SMMLA GEMM (image engine and LLM engines).
//!
//! `Y[t][r] = sum_k X[t][k] * W[r][k]` with both operands held as int8 codes plus float scales,
//! computed with SMMLA (32 int8 MACs per instruction) in 8x8 tiles, output f32.
//!
//! # Number format: chunk scales
//!
//! A vector (one activation token, or one weight row) of length `k` is cut into chunks of `chunk`
//! consecutive elements (`chunk` is a multiple of 8 and divides `k`). Every (vector, chunk) has
//! one f32 scale: `value[i] = scale[i / chunk] * code[i]`. One parameter covers the whole range:
//!
//! | `chunk`      | meaning                                                                    |
//! |--------------|----------------------------------------------------------------------------|
//! | 32           | exactly ggml Q8_0 (weights: codes and `d` verbatim; activations: q8_0)     |
//! | 64 .. 1024   | image-engine fast path: one float fix-up per `chunk` MACs instead of per 32 |
//! | `k`          | pure per-row weight scale x per-token activation scale (one fix-up per dot) |
//! | 256 + `sub`  | K-quant shape: f32 scale per 256 super-block, integer scale per 16 (or 32)  |
//!
//! Both operands of one GEMM must use the same `k` and `chunk`.
//!
//! # Float contract (fixed accumulation order)
//!
//! The integer part of every chunk is exact (int32, no overflow for `chunk <= 131072`), so tile
//! shape, SIMD width and thread count cannot change it. The only floating-point freedom is how
//! chunks are combined, and that is pinned. For every output element independently:
//!
//! ```text
//! acc = +0.0                      (or the previous y when accumulate = true)
//! for c in 0..nchunks (ascending):
//!     s   = sx[t][c] * sw[r][c]                 one f32 multiply, round-to-nearest-even
//!     f   = f32(isum_c[t][r])                   int32 -> f32, round-to-nearest-even
//!     acc = acc + f * s                         Order::MulAdd   (two roundings)
//!     acc = fma(f, s, acc)                      Order::Fma      (one rounding)
//! y[t][r] = acc
//! ```
//!
//! [`gemm_ref`] is the scalar definition of this contract; the SMMLA path is tested bitwise
//! against it. A consumer that needs bit-identity with one particular ggml kernel must check that
//! kernel's order against this one: ggml's NEON `vec_dot_q8_0_q8_0` converts per-lane partial
//! sums to float (4 lanes x 2 accumulators) and therefore is NOT this order; it needs its own
//! kernel behind the same packed format (the integer part stays shared and exact).
//!
//! # Packed layout (same for activations and weights)
//!
//! Vectors are grouped by 8 (the last group is zero-padded). Per group, per k-step (8 consecutive
//! k): 64 bytes, vector-major (bytes `8*j .. 8*j+8` belong to vector `j` of the group). Scales:
//! per group, per chunk, 8 f32 (one per vector of the group). See [`Packed`].
//!
//! # Threading and cache blocking
//!
//! The crate is single-threaded by design: [`gemm`] computes a (token-group range) x (row-group
//! range) block. The caller's static plan gives each thread its own row-group range and chooses
//! block sizes (a row block of `rows * k` bytes should sit in the core's L2 share; one token
//! group of `8 * k` bytes is the L1 operand). [`gemm_blocked`] is the usual driver: for a row
//! block, walk all tokens in small batches and hand each finished f32 batch to a sink (the fused
//! post-op: activation, norm, re-quantisation), so the f32 output never leaves the cache.

use std::ops::Range;

pub mod raw {
    //! The C micro-kernels (csrc/i8g_kernels.c). All pointers are to packed data as described in
    //! the crate docs; `steps` counts k-steps of 8.
    extern "C" {
        pub fn i8g_dot8x8(x: *const i8, w: *const i8, steps: usize, out: *mut i32);
        pub fn i8g_udot8x8(x: *const u8, w: *const i8, steps: usize, out: *mut i32);
        pub fn i8g_tile8x8(
            x: *const i8,
            w: *const i8,
            steps_per_chunk: usize,
            nchunks: usize,
            sx: *const f32,
            sw: *const f32,
            y: *mut f32,
            ystride: isize,
            accumulate: i32,
            fused: i32,
        );
        pub fn i8g_utile8x8(
            x: *const u8,
            w: *const i8,
            steps_per_chunk: usize,
            nchunks: usize,
            sx: *const f32,
            sw: *const f32,
            y: *mut f32,
            ystride: isize,
            accumulate: i32,
        );
        /// Max of `n` floats (`n` a multiple of 4, at least 4).
        pub fn i8g_row_max(s: *const f32, n: usize) -> f32;
        /// One query row of one key chunk of the int8 flash attention: exp(s - m), sum, times the
        /// value scales `sv`, u8 codes at `pcodes[(j / 8) * 64 + qi * 8 + j % 8]` with one scale.
        /// `kc` is a multiple of 8; `s` is overwritten. See csrc/i8g_kernels.c for the exact contract.
        pub fn i8g_softmax_pack(s: *mut f32, kc: usize, m: f32, sv: *const f32, pcodes: *mut u8, qi: usize, sum: *mut f32, scale: *mut f32);
        /// Roofline probe: 4x4 SDOT tile (`steps` of 16 k; 64 bytes per step per operand); 16 dots out.
        pub fn i8g_probe_sdot4x4(x: *const i8, w: *const i8, steps: usize, out: *mut i32);
        /// One token group x `ngroups` consecutive row groups, one chunk (see csrc): bitwise `i8g_tile8x8` per group.
        pub fn i8g_rows1(x: *const i8, w: *const i8, steps: usize, ngroups: usize, sx: *const f32, sw: *const f32, y: *mut f32, ystride: isize);
        pub fn i8g_kernel_abi() -> i32;
    }
}

/// Vectors per tile side.
pub const G: usize = 8;
/// k consumed per SMMLA step.
pub const KS: usize = 8;

/// How chunk contributions are added (see the crate-level float contract).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    MulAdd,
    Fma,
}

/// 64-byte-aligned byte buffer.
pub struct AlignedBytes {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: plain owned memory.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}

impl AlignedBytes {
    pub fn zeroed(len: usize) -> AlignedBytes {
        if len == 0 {
            return AlignedBytes { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(), len: 0 };
        }
        let layout = std::alloc::Layout::from_size_align(len, 64).expect("layout");
        // SAFETY: len > 0.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        AlignedBytes { ptr, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr/len describe our allocation (or a dangling pointer with len 0).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, unique borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated with this layout in `zeroed`.
            unsafe { std::alloc::dealloc(self.ptr, std::alloc::Layout::from_size_align_unchecked(self.len, 64)) }
        }
    }
}

/// Integer sub-block scales of a two-level (K-quant shaped) weight matrix:
/// `value = sup_scale[chunk] * sub_scale[sub-block] * code`.
pub struct SubScales {
    /// Elements per integer scale (16 for Q6_K/Q4_K/Q5_K/Q3_K, must be a multiple of 8 and divide `chunk`).
    pub sub: usize,
    /// `[group][sub-block][8 vectors]`, i16 so that both the int8 scales of Q6_K/Q3_K and the
    /// 6-bit unsigned scales of Q4_K/Q5_K fit.
    pub scales: Vec<i16>,
}

/// `n` int8 vectors of length `k` with chunk scales, in the tile layout of the micro-kernels.
pub struct Packed {
    n: usize,
    k: usize,
    chunk: usize,
    data: AlignedBytes,
    /// f32 scales, `[group][chunk][8]`, kept as raw aligned bytes so that `quantize_unsync` can write through `&self`.
    scales: AlignedBytes,
    /// Present only for two-level weights.
    pub sub: Option<SubScales>,
}

impl Packed {
    /// All-zero codes and scales. `k % 8 == 0`, `chunk % 8 == 0`, `k % chunk == 0`.
    pub fn zeroed(n: usize, k: usize, chunk: usize) -> Packed {
        assert!(k > 0 && k % KS == 0, "k must be a positive multiple of 8");
        assert!(chunk > 0 && chunk % KS == 0 && k % chunk == 0, "chunk must be a multiple of 8 that divides k");
        assert!(chunk <= 131072, "int32 chunk sums need chunk <= 131072");
        let groups = n.div_ceil(G);
        Packed { n, k, chunk, data: AlignedBytes::zeroed(groups * k * G), scales: AlignedBytes::zeroed(groups * (k / chunk) * G * 4), sub: None }
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn chunk(&self) -> usize {
        self.chunk
    }
    pub fn groups(&self) -> usize {
        self.n.div_ceil(G)
    }
    pub fn nchunks(&self) -> usize {
        self.k / self.chunk
    }
    pub fn steps(&self) -> usize {
        self.k / KS
    }
    pub fn data(&self) -> &[u8] {
        self.data.as_slice()
    }
    pub fn scales(&self) -> &[f32] {
        // SAFETY: 64-byte aligned, len is a multiple of 4, any bit pattern is a valid f32.
        unsafe { std::slice::from_raw_parts(self.scales.as_ptr() as *const f32, self.scales.len() / 4) }
    }
    fn scales_mut(&mut self) -> &mut [f32] {
        // SAFETY: as above, unique borrow.
        unsafe { std::slice::from_raw_parts_mut(self.scales.as_mut_ptr() as *mut f32, self.scales.len() / 4) }
    }
    /// Bytes of one group's codes.
    pub fn group_bytes(&self) -> usize {
        self.k * G
    }
    #[inline]
    fn code_index(&self, v: usize, i: usize) -> usize {
        ((v / G) * self.steps() + i / KS) * 64 + (v % G) * KS + i % KS
    }
    #[inline]
    fn scale_index(&self, v: usize, c: usize) -> usize {
        ((v / G) * self.nchunks() + c) * G + v % G
    }
    pub fn code(&self, v: usize, i: usize) -> i8 {
        self.data.as_slice()[self.code_index(v, i)] as i8
    }
    pub fn scale(&self, v: usize, c: usize) -> f32 {
        self.scales()[self.scale_index(v, c)]
    }

    /// Store ready-made codes for elements `[k0, k0 + codes.len())` of vector `v`, with one scale
    /// per chunk. `k0` and `codes.len()` are multiples of `chunk`.
    pub fn set_codes(&mut self, v: usize, k0: usize, codes: &[i8], scales: &[f32]) {
        assert!(v < self.n && k0 % self.chunk == 0 && codes.len() % self.chunk == 0 && k0 + codes.len() <= self.k);
        assert_eq!(scales.len(), codes.len() / self.chunk);
        let c0 = k0 / self.chunk;
        for (c, &s) in scales.iter().enumerate() {
            let si = self.scale_index(v, c0 + c);
            self.scales_mut()[si] = s;
        }
        let base = (v / G) * self.steps() * 64 + (v % G) * KS;
        let d = self.data.as_mut_slice();
        for (s, piece) in codes.chunks_exact(KS).enumerate() {
            let o = base + (k0 / KS + s) * 64;
            for i in 0..KS {
                d[o + i] = piece[i] as u8;
            }
        }
    }

    /// Quantise `x` into elements `[k0, k0 + x.len())` of vector `v`: per chunk
    /// `scale = max|x| / 127`, `code = round_half_away(x / scale)` clamped to +-127
    /// (an all-zero or non-finite-max chunk gets scale 0 and zero codes). `k0` and `x.len()` are
    /// multiples of `chunk`.
    pub fn quantize(&mut self, v: usize, k0: usize, x: &[f32]) {
        // SAFETY: unique borrow, so no concurrent access at all.
        unsafe { self.quantize_unsync(v, k0, x) }
    }

    /// [`Packed::quantize`] through a shared reference, for static plans in which several threads
    /// fill disjoint parts of one activation matrix.
    ///
    /// # Safety
    /// No other thread may read or write the same (vector, chunk) cells while this runs, and no
    /// `&[u8]`/`&[f32]` borrow of `data()`/`scales()` may be alive across the call.
    pub unsafe fn quantize_unsync(&self, v: usize, k0: usize, x: &[f32]) {
        assert!(v < self.n && k0 % self.chunk == 0 && x.len() % self.chunk == 0 && k0 + x.len() <= self.k);
        let chunk = self.chunk;
        let c0 = k0 / chunk;
        let base = (v / G) * self.steps() * 64 + (v % G) * KS;
        let data = self.data.ptr;
        let scales = self.scales.ptr as *mut f32;
        for (c, xc) in x.chunks_exact(chunk).enumerate() {
            let s0 = (k0 + c * chunk) / KS;
            // NEON kernel, bitwise equal to the scalar rule in the doc comment (`sinks::quant_chunk_ref`):
            // the 8 codes of k-step s go to this vector's 8 bytes of that step (64 bytes per step).
            let scale = sinks::raw::i8g_quant_chunk(xc.as_ptr(), chunk, data.add(base + s0 * 64) as *mut i8, 64);
            *scales.add(self.scale_index(v, c0 + c)) = scale;
        }
    }

    /// Decode vector `v` back to f32 (`scale * code`; two-level weights include the sub scale).
    pub fn dequantize(&self, v: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.k];
        for (i, o) in out.iter_mut().enumerate() {
            let mut val = self.scale(v, i / self.chunk) * self.code(v, i) as f32;
            if let Some(sub) = &self.sub {
                val *= sub.scales[((v / G) * (self.k / sub.sub) + i / sub.sub) * G + v % G] as f32;
            }
            *o = val;
        }
        out
    }

    /// Pack f32 rows (`rows[r*k .. (r+1)*k]`), quantising each row per chunk.
    pub fn from_f32(rows: &[f32], n: usize, k: usize, chunk: usize) -> Packed {
        assert_eq!(rows.len(), n * k);
        let mut p = Packed::zeroed(n, k, chunk);
        for v in 0..n {
            p.quantize(v, 0, &rows[v * k..(v + 1) * k]);
        }
        p
    }

    /// Pack a ggml Q8_0 matrix (`n_src` rows of `k/32` blocks of `{f16 d; i8 qs[32]}`).
    /// Packed row `i` is source row `perm[i]` (`perm = None`: identity); `perm` may select a subset.
    /// `chunk == 32` keeps codes and scales verbatim (exact Q8_0). A larger chunk (multiple of 32)
    /// re-quantises each chunk of the decoded row to one scale: lossy, error <= scale/2 per weight.
    pub fn from_q8_0(src: &[u8], n_src: usize, k: usize, chunk: usize, perm: Option<&[u32]>) -> Packed {
        const BLOCK: usize = 34;
        assert!(k % 32 == 0 && chunk % 32 == 0);
        assert_eq!(src.len(), n_src * (k / 32) * BLOCK, "Q8_0 byte size");
        let n = perm.map_or(n_src, |p| p.len());
        let mut p = Packed::zeroed(n, k, chunk);
        let mut codes = vec![0i8; k];
        let mut d = vec![0.0f32; k / 32];
        let mut vals = vec![0.0f32; k];
        for v in 0..n {
            let r = perm.map_or(v, |pm| pm[v] as usize);
            assert!(r < n_src);
            let row = &src[r * (k / 32) * BLOCK..(r + 1) * (k / 32) * BLOCK];
            for (b, blk) in row.chunks_exact(BLOCK).enumerate() {
                d[b] = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for i in 0..32 {
                    codes[b * 32 + i] = blk[2 + i] as i8;
                }
            }
            if chunk == 32 {
                p.set_codes(v, 0, &codes, &d);
            } else {
                for i in 0..k {
                    vals[i] = d[i / 32] * codes[i] as f32;
                }
                p.quantize(v, 0, &vals);
            }
        }
        p
    }
}

/// IEEE half -> f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: normalise
            let shift = man.leading_zeros() - 21; // man has 10 bits; bring the top set bit to bit 10
            let man = (man << shift) & 0x3ff;
            let exp = 1 - shift as i32 + 127 - 15;
            sign | ((exp as u32) << 23) | (man << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

fn check_pair(x: &Packed, w: &Packed, tg: &Range<usize>, rg: &Range<usize>, ylen: usize, ystride: usize) {
    assert_eq!(x.k, w.k, "k mismatch");
    assert_eq!(x.chunk, w.chunk, "chunk mismatch");
    assert!(x.sub.is_none(), "activations cannot have sub scales");
    assert!(tg.end <= x.groups() && rg.end <= w.groups());
    let (nt, nr) = (tg.len() * G, rg.len() * G);
    assert!(ystride >= nr);
    if nt > 0 && nr > 0 {
        assert!(ylen >= (nt - 1) * ystride + nr, "y too small");
    }
}

/// `y[(t - 8*tg.start) * ystride + (r - 8*rg.start)] = X[t] . W[r]` for the tokens of the token
/// groups `tg` and the rows of the row groups `rg` (padded vectors included: they give 0).
/// Loop order: token group outer (its `8*k` bytes are the L1 operand), row groups inner.
/// Works for single-level and two-level (`w.sub`) weights; both follow the crate float contract.
pub fn gemm(x: &Packed, w: &Packed, tg: Range<usize>, rg: Range<usize>, y: &mut [f32], ystride: usize, order: Order, accumulate: bool) {
    check_pair(x, w, &tg, &rg, y.len(), ystride);
    if let Some(sub) = &w.sub {
        return gemm_two_level(x, w, sub, tg, rg, y, ystride, order, accumulate);
    }
    let gb = x.group_bytes();
    let (spc, nch) = (x.chunk / KS, x.nchunks());
    let xd = x.data.as_ptr() as *const i8;
    let wd = w.data.as_ptr() as *const i8;
    for (ti, t) in tg.clone().enumerate() {
        for (ri, r) in rg.clone().enumerate() {
            // SAFETY: group indices and the y extent were checked in check_pair; the kernel reads
            // exactly one group of each operand (gb bytes) and nch*8 scales, and writes an 8x8 tile.
            unsafe {
                raw::i8g_tile8x8(
                    xd.add(t * gb),
                    wd.add(r * gb),
                    spc,
                    nch,
                    x.scales().as_ptr().add(t * nch * G),
                    w.scales().as_ptr().add(r * nch * G),
                    y.as_mut_ptr().add(ti * G * ystride + ri * G),
                    ystride as isize,
                    accumulate as i32,
                    (order == Order::Fma) as i32,
                );
            }
        }
    }
}

/// [`gemm`] (`Order::MulAdd`, no accumulate) for ONE token group `tg` and operands with a single
/// chunk (`chunk == k`, no sub scales): the short-vector case (attention q.k^T, k = head_dim), where
/// the per-tile set-up of the general path is a large part of the cost. Bitwise equal to `gemm`.
pub fn gemm_rows1(x: &Packed, w: &Packed, tg: usize, rg: Range<usize>, y: &mut [f32], ystride: usize) {
    check_pair(x, w, &(tg..tg + 1), &rg, y.len(), ystride);
    assert!(x.nchunks() == 1 && w.sub.is_none(), "gemm_rows1 needs single-chunk operands");
    if rg.is_empty() {
        return;
    }
    let gb = x.group_bytes();
    // SAFETY: group indices and the y extent were checked in check_pair; the kernel reads one group
    // of x, rg.len() groups of w, 8 scales per group, and writes 8 rows of rg.len()*8 floats.
    unsafe {
        raw::i8g_rows1(
            (x.data.as_ptr() as *const i8).add(tg * gb),
            (w.data.as_ptr() as *const i8).add(rg.start * gb),
            x.steps(),
            rg.len(),
            x.scales().as_ptr().add(tg * G),
            w.scales().as_ptr().add(rg.start * G),
            y.as_mut_ptr(),
            ystride as isize,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_two_level(x: &Packed, w: &Packed, sub: &SubScales, tg: Range<usize>, rg: Range<usize>, y: &mut [f32], ystride: usize, order: Order, accumulate: bool) {
    assert!(sub.sub % KS == 0 && x.chunk % sub.sub == 0);
    let nsub_k = x.k / sub.sub;
    assert_eq!(sub.scales.len(), w.groups() * nsub_k * G);
    let gb = x.group_bytes();
    let nch = x.nchunks();
    let sub_per_chunk = x.chunk / sub.sub;
    let sub_steps = sub.sub / KS;
    let xd = x.data.as_ptr() as *const i8;
    let wd = w.data.as_ptr() as *const i8;
    let mut part = [0i32; 64];
    for (ti, t) in tg.clone().enumerate() {
        for (ri, r) in rg.clone().enumerate() {
            let mut acc = [0.0f32; 64];
            if accumulate {
                for a in 0..G {
                    for b in 0..G {
                        acc[a * G + b] = y[(ti * G + a) * ystride + ri * G + b];
                    }
                }
            }
            for c in 0..nch {
                let mut isum = [0i32; 64];
                for j in 0..sub_per_chunk {
                    let sb = c * sub_per_chunk + j;
                    let off = sb * sub_steps * 64;
                    // SAFETY: off + sub_steps*64 <= gb for both operands; `part` holds 64 i32.
                    unsafe { raw::i8g_dot8x8(xd.add(t * gb + off), wd.add(r * gb + off), sub_steps, part.as_mut_ptr()) };
                    let ss = &sub.scales[(r * nsub_k + sb) * G..(r * nsub_k + sb) * G + G];
                    for a in 0..G {
                        for b in 0..G {
                            isum[a * G + b] += part[a * G + b] * ss[b] as i32;
                        }
                    }
                }
                let sx = &x.scales()[(t * nch + c) * G..(t * nch + c) * G + G];
                let sw = &w.scales()[(r * nch + c) * G..(r * nch + c) * G + G];
                for a in 0..G {
                    for b in 0..G {
                        let s = sx[a] * sw[b];
                        let f = isum[a * G + b] as f32;
                        acc[a * G + b] = match order {
                            Order::MulAdd => acc[a * G + b] + f * s,
                            Order::Fma => f.mul_add(s, acc[a * G + b]),
                        };
                    }
                }
            }
            for a in 0..G {
                for b in 0..G {
                    y[(ti * G + a) * ystride + ri * G + b] = acc[a * G + b];
                }
            }
        }
    }
}

/// Scalar definition of the float contract (and the fallback for machines without i8mm).
/// Same arguments and results as [`gemm`], bit for bit.
pub fn gemm_ref(x: &Packed, w: &Packed, tg: Range<usize>, rg: Range<usize>, y: &mut [f32], ystride: usize, order: Order, accumulate: bool) {
    check_pair(x, w, &tg, &rg, y.len(), ystride);
    let nch = x.nchunks();
    let chunk = x.chunk;
    for t in tg.start * G..tg.end * G {
        for r in rg.start * G..rg.end * G {
            let yi = (t - tg.start * G) * ystride + (r - rg.start * G);
            let mut acc = if accumulate { y[yi] } else { 0.0f32 };
            for c in 0..nch {
                let mut isum = 0i32;
                match &w.sub {
                    None => {
                        for i in c * chunk..(c + 1) * chunk {
                            isum += padded_code(x, t, i) as i32 * padded_code(w, r, i) as i32;
                        }
                    }
                    Some(sub) => {
                        let nsub_k = x.k / sub.sub;
                        for sb in c * (chunk / sub.sub)..(c + 1) * (chunk / sub.sub) {
                            let mut part = 0i32;
                            for i in sb * sub.sub..(sb + 1) * sub.sub {
                                part += padded_code(x, t, i) as i32 * padded_code(w, r, i) as i32;
                            }
                            isum += part * sub.scales[((r / G) * nsub_k + sb) * G + r % G] as i32;
                        }
                    }
                }
                let s = x.scales()[((t / G) * nch + c) * G + t % G] * w.scales()[((r / G) * nch + c) * G + r % G];
                let f = isum as f32;
                acc = match order {
                    Order::MulAdd => acc + f * s,
                    Order::Fma => f.mul_add(s, acc),
                };
            }
            y[yi] = acc;
        }
    }
}

#[inline]
fn padded_code(p: &Packed, v: usize, i: usize) -> i8 {
    p.data.as_slice()[((v / G) * p.steps() + i / KS) * 64 + (v % G) * KS + i % KS] as i8
}

/// Cache-blocked driver for one row block: walks the token groups `tg` in batches of
/// `batch_groups`, computes the f32 block `[batch tokens][rows of rg]` (row-major, stride
/// `rg.len()*8`) into `scratch`, and calls `sink(first_token, n_tokens_in_batch, block)`.
/// `first_token` and the token count include padding tokens of the last group.
pub fn gemm_blocked<S: FnMut(usize, usize, &[f32])>(
    x: &Packed,
    w: &Packed,
    tg: Range<usize>,
    rg: Range<usize>,
    batch_groups: usize,
    order: Order,
    scratch: &mut Vec<f32>,
    mut sink: S,
) {
    assert!(batch_groups > 0);
    let nr = rg.len() * G;
    scratch.resize(batch_groups * G * nr, 0.0);
    let mut t = tg.start;
    while t < tg.end {
        let e = (t + batch_groups).min(tg.end);
        let block = &mut scratch[..(e - t) * G * nr];
        gemm(x, w, t..e, rg.clone(), block, nr, order, false);
        sink(t * G, (e - t) * G, block);
        t = e;
    }
}

pub mod sinks;
pub mod fattn;

#[cfg(test)]
mod tests;
