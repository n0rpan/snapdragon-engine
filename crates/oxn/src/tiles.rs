//! Host-side tile packing for the oxn fp16 32x32 accelerator.
//!
//! Computes Y[T,O] = X[T,K] * W[O,K]^T on 32x32 fp16 tiles. Layouts match
//! npu/oxn/common/oxn_tiles.h and oxn_w8.h bit for bit:
//!
//!   activation (m, k): (m/2)*64 + 2*k + (m&1)      (local m, k < 32)
//!   weight     (k, n): (k/2)*64 + 2*n + (k&1)
//!   output     (m, n): (m/2)*64 + 2*n + (m&1)
//!
//! K is split into chunks of `kc` columns; each chunk produces a partial output
//! and the host sums them in `read_out_row`.
//!
//! Self-contained: no other crate module, no external crates, no C. NEON (fcvtn /
//! fcvtl through `core::arch::asm!`, because the half-precision NEON intrinsics are
//! not stable in rustc 1.99) on aarch64, scalar code elsewhere.

pub const TILE: usize = 32;
pub const TILE_ELMS: usize = 1024;
pub const TILE_BYTES: usize = 2048;
pub const W8_TILE_BYTES: usize = 1152;
const W8_SCALE_OFF: usize = 1024;

/// Element index inside a tile of activation (m, k), local coordinates < 32.
pub const fn act_elem_idx(m: usize, k: usize) -> usize {
    (m >> 1) * 64 + 2 * k + (m & 1)
}

/// Element index inside a tile of weight (k, n), local coordinates < 32.
pub const fn wt_elem_idx(k: usize, n: usize) -> usize {
    (k >> 1) * 64 + 2 * n + (k & 1)
}

/// Element index inside a tile of output (m, n), local coordinates < 32.
pub const fn out_elem_idx(m: usize, n: usize) -> usize {
    (m >> 1) * 64 + 2 * n + (m & 1)
}

// ---------------------------------------------------------------------------
// Scalar half conversions (IEEE binary16, round-to-nearest-even)
// ---------------------------------------------------------------------------

/// f32 -> binary16, round to nearest even, overflow to inf, subnormals handled.
/// NaN keeps its sign and the top payload bits and is quieted (same as fcvtn).
pub fn f32_to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        if mant != 0 {
            return sign | 0x7c00 | 0x200 | ((mant >> 13) as u16 & 0x1ff);
        }
        return sign | 0x7c00;
    }
    // half exponent field before rounding: exp - 127 + 15
    let e = exp - 112;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal half: value = m24 * 2^(e-25) in units of 2^-24 -> shift = 14 - e
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut hm = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (hm & 1) == 1) {
            hm += 1;
        }
        return sign | hm as u16;
    }
    let hm = mant >> 13;
    let rem = mant & 0x1fff;
    let mut h = ((e as u32) << 10) | hm;
    if rem > 0x1000 || (rem == 0x1000 && (hm & 1) == 1) {
        h += 1; // carry may propagate into the exponent, up to inf
    }
    sign | h as u16
}

/// binary16 -> f32 (exact). Signalling NaN inputs are quieted, as fcvtl does.
pub fn half_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16;
    let e = (h >> 10) & 0x1f;
    let m = h & 0x3ff;
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            // normalise the subnormal
            let mut m = m;
            let mut e2 = 113u32;
            while m & 0x400 == 0 {
                m <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 31 {
        if m == 0 {
            sign | 0x7f80_0000
        } else {
            sign | 0x7f80_0000 | ((m | 0x200) << 13)
        }
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Vector half conversions
// ---------------------------------------------------------------------------

/// Vector f32 -> half for the first `n` elements (n = len rounded down to 8). Returns n.
#[cfg(target_arch = "aarch64")]
fn f32s_to_halfs_vec(src: &[f32], dst: &mut [u16]) -> usize {
    use core::arch::aarch64::{uint16x8_t, vld1q_f32, vst1q_u16};
    let n = src.len() & !7;
    let mut i = 0;
    while i < n {
        // SAFETY: i + 8 <= n <= src.len() == dst.len(); NEON is baseline on aarch64.
        unsafe {
            let a = vld1q_f32(src.as_ptr().add(i));
            let b = vld1q_f32(src.as_ptr().add(i + 4));
            let h: uint16x8_t;
            core::arch::asm!(
                "fcvtn {0:v}.4h, {1:v}.4s",
                "fcvtn2 {0:v}.8h, {2:v}.4s",
                out(vreg) h,
                in(vreg) a,
                in(vreg) b,
                options(pure, nomem, nostack),
            );
            vst1q_u16(dst.as_mut_ptr().add(i), h);
        }
        i += 8;
    }
    n
}

#[cfg(not(target_arch = "aarch64"))]
fn f32s_to_halfs_vec(_src: &[f32], _dst: &mut [u16]) -> usize {
    0
}

/// Vector half -> f32 for the first `n` elements (n = len rounded down to 8). Returns n.
#[cfg(target_arch = "aarch64")]
fn halfs_to_f32s_vec(src: &[u16], dst: &mut [f32]) -> usize {
    use core::arch::aarch64::{float32x4_t, vld1q_u16, vst1q_f32};
    let n = src.len() & !7;
    let mut i = 0;
    while i < n {
        // SAFETY: i + 8 <= n <= src.len() == dst.len(); NEON is baseline on aarch64.
        unsafe {
            let v = vld1q_u16(src.as_ptr().add(i));
            let lo: float32x4_t;
            let hi: float32x4_t;
            core::arch::asm!(
                "fcvtl {0:v}.4s, {2:v}.4h",
                "fcvtl2 {1:v}.4s, {2:v}.8h",
                out(vreg) lo,
                out(vreg) hi,
                in(vreg) v,
                options(pure, nomem, nostack),
            );
            vst1q_f32(dst.as_mut_ptr().add(i), lo);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), hi);
        }
        i += 8;
    }
    n
}

#[cfg(not(target_arch = "aarch64"))]
fn halfs_to_f32s_vec(_src: &[u16], _dst: &mut [f32]) -> usize {
    0
}

/// dst[i] = f32_to_half(src[i]); bit-identical to the scalar function, NaN included.
pub fn f32s_to_halfs(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len(), "f32s_to_halfs: length mismatch");
    let done = f32s_to_halfs_vec(src, dst);
    for (d, &s) in dst[done..].iter_mut().zip(&src[done..]) {
        *d = f32_to_half(s);
    }
}

/// dst[i] = half_to_f32(src[i]).
pub fn halfs_to_f32s(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "halfs_to_f32s: length mismatch");
    let done = halfs_to_f32s_vec(src, dst);
    for (d, &h) in dst[done..].iter_mut().zip(&src[done..]) {
        *d = half_to_f32(h);
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActLayout {
    pub t_pad: usize,
    pub k: usize,
    pub kc: usize,
}

impl ActLayout {
    pub fn new(t: usize, k: usize, kc: usize) -> ActLayout {
        assert!(k % TILE == 0, "ActLayout: k={k} not a multiple of 32");
        assert!(kc > 0 && kc % TILE == 0, "ActLayout: kc={kc} not a positive multiple of 32");
        assert!(k % kc == 0, "ActLayout: kc={kc} does not divide k={k}");
        ActLayout { t_pad: t.div_ceil(TILE) * TILE, k, kc }
    }

    pub fn nchunks(&self) -> usize {
        self.k / self.kc
    }

    pub fn bytes(&self) -> usize {
        self.t_pad * self.k * 2
    }

    pub fn chunk_bytes(&self) -> usize {
        self.t_pad * self.kc * 2
    }
}

const ROW_PIECE: usize = 256;

/// Write columns k0 .. k0+row.len() of token m into the X tile buffer.
///
/// `k0` and `row.len()` must be multiples of 32. Only the elements of the written
/// columns are touched, so a zeroed buffer keeps padding rows (and unwritten columns) zero.
///
/// # Safety
/// `dst` must point to `lay.bytes()` writable bytes, 2-byte aligned. Concurrent calls
/// for different `m` are fine: they write disjoint u16 elements.
pub unsafe fn write_act_row(dst: *mut u16, lay: &ActLayout, m: usize, k0: usize, row: &[f32]) {
    assert!(m < lay.t_pad, "write_act_row: row {m} >= t_pad {}", lay.t_pad);
    assert!(k0 % TILE == 0 && row.len() % TILE == 0, "write_act_row: unaligned k0/len");
    assert!(k0 + row.len() <= lay.k, "write_act_row: columns exceed k");
    let kct = lay.kc / TILE;
    let rtiles = lay.t_pad / TILE;
    let rt = m / TILE;
    let row_off = act_elem_idx(m % TILE, 0);
    let mut buf = [0u16; ROW_PIECE];
    let mut col = 0;
    while col < row.len() {
        let n = core::cmp::min(ROW_PIECE, row.len() - col);
        f32s_to_halfs(&row[col..col + n], &mut buf[..n]);
        let mut t = 0;
        while t < n {
            let kt = (k0 + col + t) / TILE;
            let c = kt / kct;
            let ktl = kt % kct;
            let tile = c * rtiles * kct + rt * kct + ktl;
            let base = tile * TILE_ELMS + row_off;
            for j in 0..TILE {
                // SAFETY: caller guarantees dst covers lay.bytes(); index < lay.bytes()/2.
                unsafe { dst.add(base + 2 * j).write(buf[t + j]) };
            }
            t += TILE;
        }
        col += n;
    }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutLayout {
    pub t_pad: usize,
    pub o: usize,
    pub nchunks: usize,
    pub ycol: bool,
}

impl OutLayout {
    pub fn bytes(&self) -> usize {
        self.nchunks * self.t_pad * self.o * 2
    }

    pub fn chunk_bytes(&self) -> usize {
        self.t_pad * self.o * 2
    }
}

/// out[i] = sum over chunks c (ascending) of half_to_f32(Y_c(m, n0 + i)).
///
/// `n0` and `out.len()` are arbitrary. The first chunk assigns, later chunks add.
///
/// # Safety
/// `src` must point to `lay.bytes()` readable bytes, 2-byte aligned.
pub unsafe fn read_out_row(src: *const u16, lay: &OutLayout, m: usize, n0: usize, out: &mut [f32]) {
    assert!(m < lay.t_pad, "read_out_row: row {m} >= t_pad {}", lay.t_pad);
    assert!(n0 + out.len() <= lay.o, "read_out_row: columns exceed o");
    assert!(lay.nchunks >= 1, "read_out_row: nchunks must be >= 1");
    let len = out.len();
    if len == 0 {
        return;
    }
    let ot_n = lay.o / TILE;
    let rtiles = lay.t_pad / TILE;
    let rt = m / TILE;
    let row_off = out_elem_idx(m % TILE, 0);
    let chunk_elems = rtiles * ot_n * TILE_ELMS;
    let mut h = [0u16; TILE];
    let mut f = [0f32; TILE];
    for c in 0..lay.nchunks {
        let mut i = 0;
        while i < len {
            let n = n0 + i;
            let ot = n / TILE;
            let nl = n % TILE;
            let cnt = core::cmp::min(TILE - nl, len - i);
            let tile = if lay.ycol { ot * rtiles + rt } else { rt * ot_n + ot };
            let base = c * chunk_elems + tile * TILE_ELMS + row_off + 2 * nl;
            for (j, hj) in h[..cnt].iter_mut().enumerate() {
                // SAFETY: caller guarantees src covers lay.bytes(); index < lay.bytes()/2.
                *hj = unsafe { src.add(base + 2 * j).read() };
            }
            halfs_to_f32s(&h[..cnt], &mut f[..cnt]);
            if c == 0 {
                out[i..i + cnt].copy_from_slice(&f[..cnt]);
            } else {
                for (o, &v) in out[i..i + cnt].iter_mut().zip(&f[..cnt]) {
                    *o += v;
                }
            }
            i += cnt;
        }
    }
}

// ---------------------------------------------------------------------------
// Whole-tile output reader and activation writer
// ---------------------------------------------------------------------------

fn check_read_tile(lay: &OutLayout, rt: usize, n0: usize, len: usize, out_len: usize) {
    assert!(lay.nchunks >= 1, "read_out_tile_rows: nchunks must be >= 1");
    assert!(rt < lay.t_pad / TILE, "read_out_tile_rows: row tile {rt} out of range");
    assert!(n0 + len <= lay.o, "read_out_tile_rows: columns exceed o");
    assert_eq!(out_len, TILE * len, "read_out_tile_rows: out length");
}

fn check_write_tile(lay: &ActLayout, rt: usize, k0: usize, len: usize, rows_len: usize) {
    assert!(rt < lay.t_pad / TILE, "write_act_tile_rows: row tile {rt} out of range");
    assert!(k0 % TILE == 0 && len % TILE == 0, "write_act_tile_rows: unaligned k0/len");
    assert!(k0 + len <= lay.k, "write_act_tile_rows: columns exceed k");
    assert_eq!(rows_len, TILE * len, "write_act_tile_rows: rows length");
}

/// Rows of one row tile: out[r * len + i] = sum over chunks of half_to_f32(Y_c(32*rt + r, n0 + i)) for r in 0..32, i in 0..len.
///
/// `out.len() == 32 * len`. `n0` and `len` are arbitrary. The per-element summation order is the same as
/// `read_out_row`'s (chunk 0 assigns, later chunks add in ascending order), so the results are bit-identical
/// to 32 calls of `read_out_row`.
///
/// # Safety
/// `src` must point to `lay.bytes()` readable bytes, 2-byte aligned. `rt < lay.t_pad / 32`, `n0 + len <= lay.o`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn read_out_tile_rows(src: *const u16, lay: &OutLayout, rt: usize, n0: usize, len: usize, out: &mut [f32]) {
    check_read_tile(lay, rt, n0, len, out.len());
    if len == 0 {
        return;
    }
    let ot_n = lay.o / TILE;
    let rtiles = lay.t_pad / TILE;
    let chunk_elems = rtiles * ot_n * TILE_ELMS;
    let mut even = [0f32; TILE];
    let mut odd = [0f32; TILE];
    for c in 0..lay.nchunks {
        for ot in n0 / TILE..=(n0 + len - 1) / TILE {
            let tile = if lay.ycol { ot * rtiles + rt } else { rt * ot_n + ot };
            // SAFETY: tile < rtiles * ot_n and c < nchunks, so the tile lies inside lay.bytes().
            let tp = unsafe { src.add(c * chunk_elems + tile * TILE_ELMS) };
            let col0 = ot * TILE;
            let j0 = n0.max(col0) - col0;
            let j1 = (n0 + len).min(col0 + TILE) - col0;
            let cnt = j1 - j0;
            let d0 = col0 + j0 - n0;
            for p in 0..TILE / 2 {
                // SAFETY: p < 16, so line p (64 halfs) lies inside the tile.
                unsafe { line_to_f32(tp.add(p * 64), even.as_mut_ptr(), odd.as_mut_ptr()) };
                let re = 2 * p;
                assign_or_add(&mut out[re * len + d0..re * len + d0 + cnt], &even[j0..j1], c == 0);
                assign_or_add(&mut out[(re + 1) * len + d0..(re + 1) * len + d0 + cnt], &odd[j0..j1], c == 0);
            }
        }
    }
}

/// Portable version: one `read_out_row` per token row.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn read_out_tile_rows(src: *const u16, lay: &OutLayout, rt: usize, n0: usize, len: usize, out: &mut [f32]) {
    // SAFETY: the portable implementation has the same contract.
    unsafe { read_out_tile_rows_portable(src, lay, rt, n0, len, out) };
}

// Also compile the portable branch on ARM in tests, so both implementations are checked
// against the same row-wise oracle without needing a second machine.
#[cfg(any(not(target_arch = "aarch64"), test))]
unsafe fn read_out_tile_rows_portable(src: *const u16, lay: &OutLayout, rt: usize, n0: usize, len: usize, out: &mut [f32]) {
    check_read_tile(lay, rt, n0, len, out.len());
    for r in 0..TILE {
        // SAFETY: forwarded from the caller's contract (row 32*rt + r < t_pad).
        unsafe { read_out_row(src, lay, TILE * rt + r, n0, &mut out[r * len..(r + 1) * len]) };
    }
}

/// Write columns k0 .. k0+len of the 32 token rows of row tile `rt` at once: `rows[r * len + i]` is the value for
/// token 32*rt + r, column k0 + i. `k0` and `len` are multiples of 32, `rows.len() == 32 * len`. Tokens beyond the
/// real token count must be passed as zeros by the caller. Produces exactly the bytes that 32 calls of
/// `write_act_row` would produce.
///
/// # Safety
/// As `write_act_row`: `dst` must point to `lay.bytes()` writable bytes, 2-byte aligned. Concurrent calls for
/// different `rt` are fine: they write disjoint u16 elements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn write_act_tile_rows(dst: *mut u16, lay: &ActLayout, rt: usize, k0: usize, len: usize, rows: &[f32]) {
    check_write_tile(lay, rt, k0, len, rows.len());
    let kct = lay.kc / TILE;
    let rtiles = lay.t_pad / TILE;
    for kt in k0 / TILE..(k0 + len) / TILE {
        let c = kt / kct;
        let ktl = kt % kct;
        let tile = c * rtiles * kct + rt * kct + ktl;
        // SAFETY: tile < lay.bytes() / 2048 by the layout arithmetic of write_act_row.
        let tp = unsafe { dst.add(tile * TILE_ELMS) };
        let col = kt * TILE - k0;
        for p in 0..TILE / 2 {
            let even = &rows[(2 * p) * len + col..][..TILE];
            let odd = &rows[(2 * p + 1) * len + col..][..TILE];
            // SAFETY: p < 16, so line p (64 halfs) lies inside the tile.
            unsafe { f32s_to_line(even.as_ptr(), odd.as_ptr(), tp.add(p * 64)) };
        }
    }
}

/// Portable version: one `write_act_row` per token row.
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn write_act_tile_rows(dst: *mut u16, lay: &ActLayout, rt: usize, k0: usize, len: usize, rows: &[f32]) {
    check_write_tile(lay, rt, k0, len, rows.len());
    for r in 0..TILE {
        // SAFETY: forwarded from the caller's contract (token 32*rt + r < t_pad).
        unsafe { write_act_row(dst, lay, TILE * rt + r, k0, &rows[r * len..(r + 1) * len]) };
    }
}

/// `dst[i] = src[i]` when `assign` is true (first chunk), else `dst[i] += src[i]` (later chunks, same operand order as `read_out_row`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn assign_or_add(dst: &mut [f32], src: &[f32], assign: bool) {
    if assign {
        dst.copy_from_slice(src);
    } else {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    }
}

/// Eight binary16 lanes -> eight f32 at `dst` (fcvtl + fcvtl2, exact, as `halfs_to_f32s`).
///
/// # Safety
/// `dst` must be valid for 8 f32 writes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn half8_to_f32x8(v: core::arch::aarch64::uint16x8_t, dst: *mut f32) {
    use core::arch::aarch64::{float32x4_t, vst1q_f32};
    let lo: float32x4_t;
    let hi: float32x4_t;
    // SAFETY: pure register-to-register conversion; NEON is baseline on aarch64.
    unsafe {
        core::arch::asm!(
            "fcvtl {0:v}.4s, {2:v}.4h",
            "fcvtl2 {1:v}.4s, {2:v}.8h",
            out(vreg) lo,
            out(vreg) hi,
            in(vreg) v,
            options(pure, nomem, nostack),
        );
        vst1q_f32(dst, lo);
        vst1q_f32(dst.add(4), hi);
    }
}

/// Eight f32 at `src` -> eight binary16 lanes (fcvtn + fcvtn2, as `f32s_to_halfs`).
///
/// # Safety
/// `src` must be valid for 8 f32 reads.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn f32x8_to_half8(src: *const f32) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::{float32x4_t, uint16x8_t, vld1q_f32};
    // SAFETY: caller guarantees 8 readable f32; NEON is baseline on aarch64.
    unsafe {
        let a: float32x4_t = vld1q_f32(src);
        let b: float32x4_t = vld1q_f32(src.add(4));
        let h: uint16x8_t;
        core::arch::asm!(
            "fcvtn {0:v}.4h, {1:v}.4s",
            "fcvtn2 {0:v}.8h, {2:v}.4s",
            out(vreg) h,
            in(vreg) a,
            in(vreg) b,
            options(pure, nomem, nostack),
        );
        h
    }
}

/// One row-pair line (64 halfs, element (m, n) at 2*n + (m & 1)): the even positions go to `even[0..32]`
/// (row 2p), the odd positions to `odd[0..32]` (row 2p+1).
///
/// # Safety
/// `line` must be valid for 64 u16 reads; `even` and `odd` for 32 f32 writes each.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn line_to_f32(line: *const u16, even: *mut f32, odd: *mut f32) {
    use core::arch::aarch64::{vld1q_u16, vuzp1q_u16, vuzp2q_u16};
    // SAFETY: 8 vectors of 8 halfs cover the 64-half line; the outputs have 32 slots each.
    unsafe {
        for g in 0..4 {
            let a = vld1q_u16(line.add(16 * g));
            let b = vld1q_u16(line.add(16 * g + 8));
            half8_to_f32x8(vuzp1q_u16(a, b), even.add(8 * g));
            half8_to_f32x8(vuzp2q_u16(a, b), odd.add(8 * g));
        }
    }
}

/// Inverse of `line_to_f32`: 32 f32 of row 2p (`even`) and of row 2p+1 (`odd`) -> one 64-half line.
///
/// # Safety
/// `even` and `odd` must be valid for 32 f32 reads each; `line` for 64 u16 writes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn f32s_to_line(even: *const f32, odd: *const f32, line: *mut u16) {
    use core::arch::aarch64::{vst1q_u16, vzip1q_u16, vzip2q_u16};
    // SAFETY: 4 groups of 8 columns; each group writes 16 halfs of the line.
    unsafe {
        for g in 0..4 {
            let e = f32x8_to_half8(even.add(8 * g));
            let o = f32x8_to_half8(odd.add(8 * g));
            vst1q_u16(line.add(16 * g), vzip1q_u16(e, o));
            vst1q_u16(line.add(16 * g + 8), vzip2q_u16(e, o));
        }
    }
}

// ---------------------------------------------------------------------------
// W8 weights
// ---------------------------------------------------------------------------

/// Bytes of a W8 buffer for an O x K weight (independent of kc).
pub fn w8_bytes(o: usize, k: usize) -> usize {
    assert!(o % TILE == 0 && k % TILE == 0, "w8_bytes: o={o}, k={k} must be multiples of 32");
    (o / TILE) * (k / TILE) * W8_TILE_BYTES
}

/// Bytes of one W8 strip: the kc/32 records of one output tile (32 rows) and one K chunk of kc columns.
/// In the chunked W8 layout the strip of chunk `c` and output tile `ot` starts at byte
/// `(c * (o/32) + ot) * w8_strip_bytes(kc)` of the whole linear.
pub fn w8_strip_bytes(kc: usize) -> usize {
    (kc / TILE) * W8_TILE_BYTES
}

/// Pack one W8 strip: K chunk `chunk` (columns chunk*kc .. (chunk+1)*kc) of output tile `ot` (rows
/// 32*ot .. 32*ot+31) of the Q8_0 source into `dst`, which is exactly `w8_strip_bytes(kc)` bytes long.
/// Record `ktl` of the strip occupies `dst[ktl * 1152 ..]`. Packed row n takes source row perm[n]
/// (identity if None). Every byte of `dst` is written (padding zeroed).
#[allow(clippy::too_many_arguments)]
pub fn pack_w8_strip(dst: &mut [u8], q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot: usize) {
    assert!(o % TILE == 0, "pack_w8: o={o} not a multiple of 32");
    assert!(k % TILE == 0, "pack_w8: k={k} not a multiple of 32");
    assert!(kc > 0 && kc % TILE == 0 && k % kc == 0, "pack_w8: bad kc={kc} for k={k}");
    assert_eq!(q8.len(), o * (k / TILE) * 34, "pack_w8: q8 length");
    if let Some(p) = perm {
        assert_eq!(p.len(), o, "pack_w8: perm length");
    }
    assert!(chunk < k / kc, "pack_w8: chunk {chunk} out of range");
    assert!(ot < o / TILE, "pack_w8: output tile {ot} out of range");
    assert_eq!(dst.len(), w8_strip_bytes(kc), "pack_w8: strip length");

    let nb = k / TILE;
    let kct = kc / TILE;
    for ktl in 0..kct {
        let kt = chunk * kct + ktl;
        let rec = &mut dst[ktl * W8_TILE_BYTES..(ktl + 1) * W8_TILE_BYTES];
        let (codes, scales) = rec.split_at_mut(W8_SCALE_OFF);
        let codes: &mut [u8; W8_SCALE_OFF] = codes.try_into().expect("record codes");
        scales[2 * TILE..].fill(0);
        for nl in 0..TILE {
            let n = ot * TILE + nl;
            let s = match perm {
                Some(p) => p[n] as usize,
                None => n,
            };
            // one read of the 34-byte Q8_0 block: scale, then 32 codes scattered with a fixed stride
            let blk: &[u8; 34] = (&q8[(s * nb + kt) * 34..(s * nb + kt + 1) * 34]).try_into().expect("Q8_0 block");
            scales[2 * nl..2 * nl + 2].copy_from_slice(&blk[0..2]);
            let qs: &[u8; TILE] = (&blk[2..]).try_into().expect("Q8_0 codes");
            // element (kl, nl) is at (kl / 2) * 64 + 2 * nl + kl % 2, see `wt_elem_idx`
            for j in 0..TILE / 2 {
                codes[j * 64 + 2 * nl] = qs[2 * j];
                codes[j * 64 + 2 * nl + 1] = qs[2 * j + 1];
            }
        }
    }
}

/// Pack GGUF Q8_0 rows into chunked W8 records. Packed row n takes source row perm[n]
/// (identity if None). Every byte of `dst` is written (padding zeroed).
pub fn pack_w8_from_q8_0(dst: &mut [u8], q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>) {
    assert!(o % TILE == 0, "pack_w8: o={o} not a multiple of 32");
    assert!(k % TILE == 0, "pack_w8: k={k} not a multiple of 32");
    assert!(kc > 0 && kc % TILE == 0 && k % kc == 0, "pack_w8: bad kc={kc} for k={k}");
    assert_eq!(q8.len(), o * (k / TILE) * 34, "pack_w8: q8 length");
    if let Some(p) = perm {
        assert_eq!(p.len(), o, "pack_w8: perm length");
    }
    assert_eq!(dst.len(), w8_bytes(o, k), "pack_w8: dst length");

    let sb = w8_strip_bytes(kc);
    let ntile_o = o / TILE;
    let nchunks = k / kc;
    for c in 0..nchunks {
        for ot in 0..ntile_o {
            let off = (c * ntile_o + ot) * sb;
            pack_w8_strip(&mut dst[off..off + sb], q8, o, k, kc, perm, c, ot);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tiles_ref.rs"]
mod tiles_ref;

#[cfg(test)]
mod tests {
    use super::tiles_ref::*;
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Rng {
            assert!(seed != 0);
            Rng(seed)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_u32(&mut self) -> u32 {
            (self.next_u64() >> 32) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
        fn byte(&mut self) -> u8 {
            self.next_u64() as u8
        }
    }

    fn random_perm(rng: &mut Rng, o: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..o as u32).collect();
        for i in (1..o).rev() {
            let j = rng.below(i + 1);
            p.swap(i, j);
        }
        p
    }

    #[test]
    fn half_all_patterns_roundtrip() {
        for h in 0..=65535u16 {
            let f = half_to_f32(h);
            let is_nan = h & 0x7fff > 0x7c00;
            if is_nan {
                assert!(f.is_nan(), "h={h:04x} should be NaN");
                // quiet bit is set by the conversion (as fcvtl/fcvtn do)
                assert_eq!(f32_to_half(f), h | 0x200, "NaN h={h:04x}");
            } else {
                assert_eq!(f32_to_half(f), h, "round trip h={h:04x}");
                // independent check of half_to_f32 from the IEEE definition
                let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
                let e = ((h >> 10) & 31) as i32;
                let m = (h & 0x3ff) as f64;
                let exp = if e == 0 {
                    m * 2f64.powi(-24)
                } else if e == 31 {
                    f64::INFINITY
                } else {
                    (1024.0 + m) * 2f64.powi(e - 25)
                };
                assert_eq!(f as f64, sign * exp, "value of h={h:04x}");
            }
        }
    }

    #[test]
    fn half_halfway_round_to_even() {
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-11)), 0x3c00); // tie, even mantissa stays
        assert_eq!(f32_to_half(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02); // tie, rounds to even
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3c01); // above tie
        assert_eq!(f32_to_half(2f32.powi(-25)), 0x0000); // subnormal tie -> 0
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 0x0002); // subnormal tie -> 2
        assert_eq!(f32_to_half(-(2f32.powi(-25))), 0x8000);
        assert_eq!(f32_to_half(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_half(2f32.powi(-14)), 0x0400);
        assert_eq!(f32_to_half(65504.0), 0x7bff);
        assert_eq!(f32_to_half(65519.0), 0x7bff);
        assert_eq!(f32_to_half(65520.0), 0x7c00); // tie between 65504 (odd) and 65536 -> inf
        assert_eq!(f32_to_half(f32::MAX), 0x7c00);
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_half(-0.0), 0x8000);
    }

    #[test]
    fn vector_conversions_match_scalar() {
        let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
        let n = 1_000_003;
        let src: Vec<f32> = (0..n)
            .map(|i| match i % 3 {
                0 => rng.range(-70000.0, 70000.0),
                1 => rng.range(-6.2e-5, 6.2e-5),
                _ => f32::from_bits(rng.next_u32()),
            })
            .collect();
        let mut dst = vec![0u16; n];
        f32s_to_halfs(&src, &mut dst);
        for (i, &x) in src.iter().enumerate() {
            assert_eq!(dst[i], f32_to_half(x), "x bits {:08x}", x.to_bits());
        }
        for len in 0..40 {
            let mut d = vec![0u16; len];
            f32s_to_halfs(&src[..len], &mut d);
            for i in 0..len {
                assert_eq!(d[i], f32_to_half(src[i]), "tail len {len} idx {i}");
            }
        }
    }

    #[test]
    fn halfs_to_f32_matches_scalar_all_patterns() {
        let hs: Vec<u16> = (0..=65535u16).collect();
        let mut fs = vec![0f32; hs.len()];
        halfs_to_f32s(&hs, &mut fs);
        for h in 0..=65535u16 {
            assert_eq!(fs[h as usize].to_bits(), half_to_f32(h).to_bits(), "h={h:04x}");
        }
        for len in 0..40 {
            let mut f = vec![0f32; len];
            halfs_to_f32s(&hs[..len], &mut f);
            for i in 0..len {
                assert_eq!(f[i].to_bits(), half_to_f32(hs[i]).to_bits());
            }
        }
        // and back: f32s_to_halfs(halfs_to_f32s(h)) == h, NaN quieted
        let mut back = vec![0u16; hs.len()];
        f32s_to_halfs(&fs, &mut back);
        for h in 0..=65535u16 {
            let want = if h & 0x7fff > 0x7c00 { h | 0x200 } else { h };
            assert_eq!(back[h as usize], want, "h={h:04x}");
        }
    }

    #[test]
    fn act_row_plus_split_matches_ref() {
        check_write_act(5, 64, 64, 1);
        check_write_act(33, 128, 32, 2);
        check_write_act(64, 3072, 1536, 3);
        check_write_act(70, 12288, 3072, 4);
    }

    fn check_write_act(t: usize, k: usize, kc: usize, seed: u64) {
        let lay = ActLayout::new(t, k, kc);
        assert_eq!(lay.nchunks(), k / kc);
        assert_eq!(lay.bytes(), lay.t_pad * k * 2);
        assert_eq!(lay.chunk_bytes(), lay.t_pad * kc * 2);
        let n16 = lay.bytes() / 2;
        let mut rng = Rng::new(seed);
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for _ in 0..t {
            let mut r = Vec::with_capacity(k);
            for _ in 0..k {
                r.push(rng.range(-60000.0, 60000.0));
            }
            rows.push(r);
        }

        // one call per row
        let mut buf = vec![0u16; n16];
        let mut refb = vec![0u16; n16];
        for m in 0..t {
            unsafe { write_act_row(buf.as_mut_ptr(), &lay, m, 0, &rows[m]) };
            ref_write_act(&mut refb, lay.t_pad, k, kc, m, 0, &rows[m]);
        }
        assert_eq!(buf, refb, "t={t} k={k} kc={kc} single call");

        // two calls split at a multiple of 32 (second has k0 > 0)
        let s = ((k / 2) / 32).max(1) * 32;
        let mut buf2 = vec![0u16; n16];
        for m in 0..t {
            unsafe {
                write_act_row(buf2.as_mut_ptr(), &lay, m, 0, &rows[m][..s]);
                write_act_row(buf2.as_mut_ptr(), &lay, m, s, &rows[m][s..]);
            }
        }
        assert_eq!(buf2, refb, "t={t} k={k} kc={kc} split");

        // padding rows stay zero
        for m in t..lay.t_pad {
            for kk in 0..k {
                assert_eq!(buf[ref_act_index(lay.t_pad, k, kc, m, kk)], 0, "pad row {m}");
            }
        }
    }

    #[test]
    fn out_row_matches_ref() {
        for &(t_pad, o, nchunks) in &[(32usize, 64usize, 1usize), (64, 96, 3), (96, 1056, 2)] {
            for &ycol in &[false, true] {
                check_read_out(t_pad, o, nchunks, ycol, 10 + o as u64);
            }
        }
    }

    fn check_read_out(t_pad: usize, o: usize, nchunks: usize, ycol: bool, seed: u64) {
        let lay = OutLayout { t_pad, o, nchunks, ycol };
        assert_eq!(lay.bytes(), nchunks * t_pad * o * 2);
        assert_eq!(lay.chunk_bytes(), t_pad * o * 2);
        let mut rng = Rng::new(seed);
        let buf: Vec<u16> = (0..lay.bytes() / 2).map(|_| f32_to_half(rng.range(-50.0, 50.0))).collect();
        let combos = [(0usize, o), (8, 24), (40, 200), (o - 8, 8)];
        for m in 0..t_pad {
            for &(n0, len) in &combos {
                if n0 + len > o {
                    continue;
                }
                let mut out = vec![0f32; len];
                unsafe { read_out_row(buf.as_ptr(), &lay, m, n0, &mut out) };
                let mut r = vec![0f32; len];
                ref_read_out(&buf, t_pad, o, nchunks, ycol, m, n0, &mut r);
                assert_eq!(out, r, "t_pad={t_pad} o={o} nc={nchunks} ycol={ycol} m={m} n0={n0} len={len}");
            }
        }
    }

    /// Random half: infinities, NaNs (incl. signalling-style patterns), subnormals, bounded finite values, raw bits.
    fn rand_half(rng: &mut Rng) -> u16 {
        match rng.below(8) {
            0 => [0x7c00u16, 0xfc00, 0x7e00, 0xfe01, 0x7d00][rng.below(5)],
            1 => (rng.next_u64() as u16) & 0x83ff,
            2 | 3 => f32_to_half(rng.range(-50.0, 50.0)),
            _ => rng.next_u64() as u16,
        }
    }

    #[test]
    fn out_tile_rows_match_row_reads() {
        for &(t_pad, o, nchunks) in &[(32usize, 64usize, 1usize), (64, 96, 3), (96, 1056, 2), (1536, 256, 2)] {
            for &ycol in &[false, true] {
                let lay = OutLayout { t_pad, o, nchunks, ycol };
                let mut rng = Rng::new(0x51 + o as u64 + 7 * nchunks as u64 + ycol as u64);
                let buf: Vec<u16> = (0..lay.bytes() / 2).map(|_| rand_half(&mut rng)).collect();
                for rt in 0..t_pad / TILE {
                    for &(n0, len) in &[(0, o), (8, 24), (40, 200), (o - 8, 8), (32, 32), (31, 34)] {
                        if n0 + len > o {
                            continue;
                        }
                        let mut got = vec![0f32; TILE * len];
                        unsafe { read_out_tile_rows(buf.as_ptr(), &lay, rt, n0, len, &mut got) };
                        let mut want = vec![0f32; TILE * len];
                        for r in 0..TILE {
                            unsafe { read_out_row(buf.as_ptr(), &lay, TILE * rt + r, n0, &mut want[r * len..(r + 1) * len]) };
                        }
                        let g: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
                        let w: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
                        assert_eq!(g, w, "t_pad={t_pad} o={o} nc={nchunks} ycol={ycol} rt={rt} n0={n0} len={len}");
                        let mut portable = vec![0f32; TILE * len];
                        unsafe { read_out_tile_rows_portable(buf.as_ptr(), &lay, rt, n0, len, &mut portable) };
                        assert_eq!(portable.iter().map(|x|x.to_bits()).collect::<Vec<_>>(), w,
                            "portable t_pad={t_pad} o={o} nc={nchunks} ycol={ycol} rt={rt} n0={n0} len={len}");
                    }
                }
            }
        }
    }

    /// Token rows of row tile `rt`, columns k0 .. k0+len, zero for tokens >= t.
    fn tile_rows(vals: &[f32], t: usize, k: usize, rt: usize, k0: usize, len: usize) -> Vec<f32> {
        let mut rows = vec![0f32; TILE * len];
        for r in 0..TILE {
            let m = TILE * rt + r;
            if m < t {
                rows[r * len..(r + 1) * len].copy_from_slice(&vals[m * k + k0..m * k + k0 + len]);
            }
        }
        rows
    }

    #[test]
    fn act_tile_rows_match_row_writes() {
        for &(t, k, kc) in &[(64usize, 128usize, 32usize), (70, 3072, 1536), (96, 12288, 3072)] {
            let lay = ActLayout::new(t, k, kc);
            let mut rng = Rng::new(0x77 + k as u64);
            let vals: Vec<f32> = (0..t * k)
                .map(|i| if i % 5 == 0 { f32::from_bits(rng.next_u32()) } else { rng.range(-60000.0, 60000.0) })
                .collect();
            let n16 = lay.bytes() / 2;
            let mut want = vec![0u16; n16];
            for m in 0..t {
                unsafe { write_act_row(want.as_mut_ptr(), &lay, m, 0, &vals[m * k..(m + 1) * k]) };
            }
            for split in [false, true] {
                let mut got = vec![0u16; n16];
                for rt in 0..lay.t_pad / TILE {
                    if split {
                        let a = tile_rows(&vals, t, k, rt, 0, 64);
                        let b = tile_rows(&vals, t, k, rt, 64, k - 64);
                        unsafe {
                            write_act_tile_rows(got.as_mut_ptr(), &lay, rt, 0, 64, &a);
                            write_act_tile_rows(got.as_mut_ptr(), &lay, rt, 64, k - 64, &b);
                        }
                    } else {
                        let a = tile_rows(&vals, t, k, rt, 0, k);
                        unsafe { write_act_tile_rows(got.as_mut_ptr(), &lay, rt, 0, k, &a) };
                    }
                }
                assert_eq!(got, want, "t={t} k={k} kc={kc} split={split}");
            }
        }
    }

    #[test]
    #[ignore]
    fn bench_read_out() {
        use std::time::Instant;
        let (t_pad, o) = (1536usize, 27648usize);
        let lay = OutLayout { t_pad, o, nchunks: 1, ycol: true };
        let mut rng = Rng::new(0xbe7c_0001);
        let buf: Vec<u16> = (0..lay.bytes() / 2).map(|_| rng.next_u64() as u16).collect();
        let elems = (t_pad * o) as f64;
        let mut out = vec![0f32; TILE * 128];
        let mut acc = 0u32;
        for rep in 0..3 {
            let t0 = Instant::now();
            for m in 0..t_pad {
                for n0 in (0..o).step_by(128) {
                    let len = (o - n0).min(128);
                    unsafe { read_out_row(buf.as_ptr(), &lay, m, n0, &mut out[..len]) };
                    acc ^= out[0].to_bits();
                }
            }
            let ns_a = t0.elapsed().as_nanos() as f64 / elems;

            let t1 = Instant::now();
            for rt in 0..t_pad / TILE {
                for n0 in (0..o).step_by(128) {
                    let len = (o - n0).min(128);
                    unsafe { read_out_tile_rows(buf.as_ptr(), &lay, rt, n0, len, &mut out[..TILE * len]) };
                    acc ^= out[0].to_bits();
                }
            }
            let ns_b = t1.elapsed().as_nanos() as f64 / elems;
            println!("rep {rep}: read_out_row {ns_a:.3} ns/elem, read_out_tile_rows {ns_b:.3} ns/elem");
        }
        std::hint::black_box(acc);
    }

    #[test]
    fn pack_w8_matches_ref_and_source() {
        for &(o, k, kc) in &[(32usize, 32usize, 32usize), (64, 128, 64), (96, 3072, 1536)] {
            for &use_perm in &[false, true] {
                check_pack(o, k, kc, use_perm, 100 + o as u64 + use_perm as u64);
            }
        }
    }

    fn check_pack(o: usize, k: usize, kc: usize, use_perm: bool, seed: u64) {
        let mut rng = Rng::new(seed);
        let nb = k / 32;
        let mut q8 = Vec::with_capacity(o * nb * 34);
        for _ in 0..o * nb {
            let d = f32_to_half(rng.range(-0.5, 0.5));
            q8.extend_from_slice(&d.to_le_bytes());
            for _ in 0..32 {
                q8.push(rng.byte());
            }
        }
        let perm = if use_perm { Some(random_perm(&mut rng, o)) } else { None };
        let perm_ref = perm.as_deref();

        let mut dst = vec![0xA5u8; w8_bytes(o, k)]; // dirty: every byte must be overwritten
        pack_w8_from_q8_0(&mut dst, &q8, o, k, kc, perm_ref);
        let r = ref_pack_w8(o, k, kc, &q8, perm_ref);
        assert_eq!(dst, r, "o={o} k={k} kc={kc} perm={use_perm}");

        // decode the packed buffer and compare against the Q8_0 source directly
        for n in 0..o {
            let s = perm_ref.map_or(n, |p| p[n] as usize);
            for kk in 0..k {
                let blk = (s * nb + kk / 32) * 34;
                let d = u16::from_le_bytes([q8[blk], q8[blk + 1]]);
                let want = half_to_f32(d) * (q8[blk + 2 + kk % 32] as i8) as f32;
                assert_eq!(ref_w8_get(&dst, o, k, kc, n, kk), want, "n={n} k={kk}");
            }
        }
    }

    /// Random Q8_0 source of an o x k linear. No byte is 0xEE, so a 0xEE sentinel survives only
    /// where the packer writes nothing.
    fn random_q8(rng: &mut Rng, o: usize, k: usize) -> Vec<u8> {
        let nb = k / 32;
        let mut q8 = Vec::with_capacity(o * nb * 34);
        for _ in 0..o * nb {
            let d = loop {
                let b = f32_to_half(rng.range(-0.5, 0.5)).to_le_bytes();
                if b[0] != 0xEE && b[1] != 0xEE {
                    break b;
                }
            };
            q8.extend_from_slice(&d);
            for _ in 0..32 {
                let c = rng.byte();
                q8.push(if c == 0xEE { 0x11 } else { c });
            }
        }
        q8
    }

    #[test]
    fn pack_w8_strip_concatenation_matches_ref() {
        for &(o, k, kc) in &[(64usize, 128usize, 64usize), (96, 3072, 1536)] {
            for &use_perm in &[false, true] {
                let mut rng = Rng::new(500 + o as u64 + use_perm as u64);
                let q8 = random_q8(&mut rng, o, k);
                let perm = if use_perm { Some(random_perm(&mut rng, o)) } else { None };
                let perm_ref = perm.as_deref();
                let sb = w8_strip_bytes(kc);
                let ntile_o = o / TILE;
                let mut dst = vec![0xA5u8; w8_bytes(o, k)];
                for c in 0..k / kc {
                    for ot in 0..ntile_o {
                        let off = (c * ntile_o + ot) * sb;
                        pack_w8_strip(&mut dst[off..off + sb], &q8, o, k, kc, perm_ref, c, ot);
                    }
                }
                let r = ref_pack_w8(o, k, kc, &q8, perm_ref);
                assert_eq!(dst, r, "strips o={o} k={k} kc={kc} perm={use_perm}");
            }
        }
    }

    #[test]
    fn pack_w8_strip_writes_every_byte() {
        for &(o, k, kc) in &[(64usize, 128usize, 64usize), (96, 3072, 1536)] {
            let mut rng = Rng::new(900 + o as u64);
            let q8 = random_q8(&mut rng, o, k);
            let perm = random_perm(&mut rng, o);
            for chunk in 0..k / kc {
                for ot in 0..o / TILE {
                    let mut strip = vec![0xEEu8; w8_strip_bytes(kc)];
                    pack_w8_strip(&mut strip, &q8, o, k, kc, Some(&perm[..]), chunk, ot);
                    assert!(!strip.contains(&0xEE), "o={o} k={k} kc={kc} chunk={chunk} ot={ot}: unwritten byte");
                    for rec in strip.chunks(W8_TILE_BYTES) {
                        assert!(rec[W8_SCALE_OFF + 2 * TILE..].iter().all(|&b| b == 0), "padding not zero");
                    }
                }
            }
        }
    }

    #[test]
    fn element_indices_match_c_header() {
        let mut seen_a = [false; 1024];
        let mut seen_w = [false; 1024];
        let mut seen_o = [false; 1024];
        for m in 0..32usize {
            for k in 0..32usize {
                let a = act_elem_idx(m, k);
                assert_eq!(a, (m / 2) * 64 + 2 * k + (m % 2));
                let w = wt_elem_idx(m, k); // weight (k=m, n=k)
                assert_eq!(w, (m / 2) * 64 + 2 * k + (m % 2));
                let o = out_elem_idx(m, k);
                assert_eq!(o, (m / 2) * 64 + 2 * k + (m % 2));
                seen_a[a] = true;
                seen_w[w] = true;
                seen_o[o] = true;
            }
        }
        assert!(seen_a.iter().all(|&b| b) && seen_w.iter().all(|&b| b) && seen_o.iter().all(|&b| b));
        assert_eq!(act_elem_idx(0, 0), 0);
        assert_eq!(act_elem_idx(1, 0), 1);
        assert_eq!(act_elem_idx(0, 1), 2);
        assert_eq!(act_elem_idx(2, 0), 64);
        assert_eq!(act_elem_idx(31, 31), 1023);
        assert_eq!(wt_elem_idx(31, 31), 1023);
        assert_eq!(out_elem_idx(31, 31), 1023);
    }

    #[test]
    #[should_panic]
    fn act_layout_rejects_kc_not_dividing_k() {
        let _ = ActLayout::new(10, 96, 64);
    }
}
