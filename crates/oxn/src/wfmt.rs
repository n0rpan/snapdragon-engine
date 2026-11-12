//! Weight formats of the accelerator linears.
//!
//! `W8`: int8 codes and one half scale per 32 codes in 1152-byte records (`GF_W8`, `tiles::pack_w8_strip`).
//! `Fp16`: each weight as one binary16 element of a 2048-byte tile (no `GF_W8`), the value
//! `f32_to_half(half_to_f32(d) * q)`. Both hold the same Q8_0 values; fp16 rounds `scale * q` to half.
//!
//! A strip is the kc/32 tiles (or records) of one output tile (32 rows) and one K chunk of kc columns.
//! A blob is a run of strips of consecutive output tiles, strip i holding output tile ot0 + i.

use crate::tiles::{self, TILE, TILE_BYTES};
use crate::GF_W8;

/// Bytes of one W8 record (1024 codes, then 32 half scales).
const W8_RECORD_BYTES: usize = 1152;
const W8_SCALE_OFF: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WFmt {
    #[default]
    W8,
    Fp16,
}

impl WFmt {
    /// Bytes of one strip: one output tile (32 rows) x `kc` columns.
    pub fn strip_bytes(self, kc: usize) -> usize {
        match self {
            WFmt::W8 => tiles::w8_strip_bytes(kc),
            WFmt::Fp16 => (kc / TILE).checked_mul(TILE_BYTES).expect("wfmt: fp16 strip size overflow"),
        }
    }

    /// The GEMM flags that select this format for the weight operand.
    pub fn gemm_flag(self) -> u32 {
        match self {
            WFmt::W8 => GF_W8,
            WFmt::Fp16 => 0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            WFmt::W8 => "w8",
            WFmt::Fp16 => "fp16",
        }
    }
}

fn check_pack(q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot: usize) {
    assert!(o % TILE == 0, "wfmt: o={o} not a multiple of 32");
    assert!(k % TILE == 0, "wfmt: k={k} not a multiple of 32");
    assert!(kc > 0 && kc % TILE == 0 && k % kc == 0, "wfmt: bad kc={kc} for k={k}");
    let q8_bytes = o.checked_mul(k / TILE).and_then(|v| v.checked_mul(34)).expect("wfmt: q8 shape overflow");
    assert_eq!(q8.len(), q8_bytes, "wfmt: q8 length");
    if let Some(p) = perm {
        assert_eq!(p.len(), o, "wfmt: perm length");
    }
    assert!(chunk < k / kc, "wfmt: chunk {chunk} out of range");
    assert!(ot < o / TILE, "wfmt: output tile {ot} out of range");
}

/// Pack one fp16 strip: K chunk `chunk` of output tile `ot` of the Q8_0 source into `dst`, exactly
/// `WFmt::Fp16.strip_bytes(kc)` bytes. Tile `ktl` of the strip is `dst[ktl * 2048 ..]`, element (k%32, n%32)
/// at u16 index `tiles::wt_elem_idx(k % 32, n % 32)`, little-endian. Packed row n takes source row perm[n]
/// (identity if None). Every byte of `dst` is written.
#[allow(clippy::too_many_arguments)]
pub fn pack_fp16_strip(dst: &mut [u8], q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot: usize) {
    check_pack(q8, o, k, kc, perm, chunk, ot);
    assert_eq!(dst.len(), WFmt::Fp16.strip_bytes(kc), "pack_fp16_strip: strip length");
    let nb = k / TILE;
    let kct = kc / TILE;
    for nl in 0..TILE {
        let n = ot * TILE + nl;
        let s = match perm {
            Some(p) => p[n] as usize,
            None => n,
        };
        assert!(s < o, "wfmt: permutation row out of range");
        for ktl in 0..kct {
            let kt = chunk * kct + ktl;
            // one 34-byte Q8_0 block: half scale, then 32 int8 codes (code kl of the block is blk[2 + kl])
            let blk: &[u8; 34] = q8[(s * nb + kt) * 34..(s * nb + kt + 1) * 34].try_into().expect("Q8_0 block");
            let d = tiles::half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let mut prod = [0f32; TILE];
            for (p, &c) in prod.iter_mut().zip(&blk[2..]) {
                *p = d * (c as i8) as f32;
            }
            let mut h = [0u16; TILE];
            tiles::f32s_to_halfs(&prod, &mut h);
            // Element (k, n) with k = 2j + b is at u16 index j * 64 + 2 * nl + b (`wt_elem_idx`), so the
            // column pair (h[2j], h[2j + 1]) is one little-endian u32 at u16 index j * 64 + 2 * nl.
            // The checked strip size above makes the multiplication non-overflowing. Keep a
            // checked tile slice as a second bound before any raw stores.
            let tile_base = ktl * TILE_BYTES;
            let tile = &mut dst[tile_base..tile_base + TILE_BYTES];
            let base = tile.as_mut_ptr();
            for j in 0..TILE / 2 {
                let pair = (h[2 * j] as u32) | ((h[2 * j + 1] as u32) << 16);
                let off = (j * 64 + 2 * nl) * 2;
                // SAFETY: nl < 32 and j < 16 imply off + 4 <= 2048, the checked tile's length.
                // The store is unaligned-safe and writes u32 little-endian.
                unsafe { base.add(off).cast::<u32>().write_unaligned(pair.to_le()) };
            }
        }
    }
}

/// The previous element-wise fp16 strip packer, kept as the bit-exact reference for the tests.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn pack_fp16_strip_ref(dst: &mut [u8], q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot: usize) {
    check_pack(q8, o, k, kc, perm, chunk, ot);
    assert_eq!(dst.len(), WFmt::Fp16.strip_bytes(kc), "pack_fp16_strip: strip length");
    let nb = k / TILE;
    let kct = kc / TILE;
    for ktl in 0..kct {
        let kt = chunk * kct + ktl;
        let tile = &mut dst[ktl * TILE_BYTES..(ktl + 1) * TILE_BYTES];
        for nl in 0..TILE {
            let n = ot * TILE + nl;
            let s = match perm {
                Some(p) => p[n] as usize,
                None => n,
            };
            // one 34-byte Q8_0 block: half scale, then 32 int8 codes (code kl of the block is blk[2 + kl])
            let blk = &q8[(s * nb + kt) * 34..(s * nb + kt + 1) * 34];
            let d = tiles::half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for kl in 0..TILE {
                let q = blk[2 + kl] as i8;
                let v = tiles::f32_to_half(d * q as f32);
                let idx = tiles::wt_elem_idx(kl, nl);
                tile[2 * idx..2 * idx + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
}

/// Pack one strip in format `fmt` (see `tiles::pack_w8_strip` and [`pack_fp16_strip`]).
#[allow(clippy::too_many_arguments)]
pub fn pack_strip(fmt: WFmt, dst: &mut [u8], q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot: usize) {
    match fmt {
        WFmt::W8 => tiles::pack_w8_strip(dst, q8, o, k, kc, perm, chunk, ot),
        WFmt::Fp16 => pack_fp16_strip(dst, q8, o, k, kc, perm, chunk, ot),
    }
}

/// The kc weights of packed output row `n` (`32 * ot0 <= n`) for one K chunk, as f32, into `out` (len kc).
/// `strips` points at the strips of output tiles `ot0..` of that chunk (strip i = output tile ot0 + i, each
/// `fmt.strip_bytes(kc)` long). W8: `half_to_f32(scale) * q`; Fp16: `half_to_f32(value)`.
///
/// # Safety
/// `strips` must be valid for `((n / 32 - ot0) + 1) * fmt.strip_bytes(kc)` bytes.
pub unsafe fn weight_row(fmt: WFmt, strips: *const u8, kc: usize, ot0: usize, n: usize, out: &mut [f32]) {
    assert!(kc > 0 && kc % TILE == 0, "weight_row: bad kc {kc}");
    assert_eq!(out.len(), kc, "weight_row: out length");
    assert!(n / TILE >= ot0, "weight_row: row {n} before output tile {ot0}");
    let sb = fmt.strip_bytes(kc);
    // SAFETY: the caller's contract covers this strip (strip index n/32 - ot0).
    let strip = unsafe { std::slice::from_raw_parts(strips.add((n / TILE - ot0) * sb), sb) };
    let nl = n % TILE;
    for (kl, o) in out.iter_mut().enumerate() {
        let (kt, kli) = (kl / TILE, kl % TILE);
        *o = match fmt {
            WFmt::W8 => {
                let rec = &strip[kt * W8_RECORD_BYTES..(kt + 1) * W8_RECORD_BYTES];
                let s = 2 * nl;
                let scale = tiles::half_to_f32(u16::from_le_bytes([rec[W8_SCALE_OFF + s], rec[W8_SCALE_OFF + s + 1]]));
                let q = rec[tiles::wt_elem_idx(kli, nl)] as i8;
                scale * q as f32
            }
            WFmt::Fp16 => {
                let tile = &strip[kt * TILE_BYTES..(kt + 1) * TILE_BYTES];
                let i = 2 * tiles::wt_elem_idx(kli, nl);
                tiles::half_to_f32(u16::from_le_bytes([tile[i], tile[i + 1]]))
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emu::Emu;
    use crate::{Backend, GemmArgs, GF_YCOL};

    #[test]
    #[should_panic(expected = "fp16 strip size overflow")]
    fn fp16_strip_size_rejects_overflow() {
        WFmt::Fp16.strip_bytes((usize::MAX / TILE_BYTES + 1) * TILE);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "q8 shape overflow")]
    fn fp16_pack_rejects_wrapped_shape_before_raw_stores() {
        let k = TILE * ((1usize << 53) + 1);
        let q8 = vec![0; 34816];
        let mut dst = vec![0; TILE_BYTES];
        pack_fp16_strip(&mut dst, &q8, 1024, k, k, None, 0, 0);
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        /// uniform in [-1, 1)
        fn sym(&mut self) -> f32 {
            (self.next() >> 40) as f32 / 16_777_216.0 * 2.0 - 1.0
        }
    }

    /// Random Q8_0 bytes of an o x k linear: a small half scale and 32 random codes per block.
    fn random_q8(r: &mut Rng, o: usize, k: usize) -> Vec<u8> {
        let mut q8 = Vec::with_capacity(o * (k / 32) * 34);
        for _ in 0..o * (k / 32) {
            let d = tiles::f32_to_half(((r.below(2000) as f32) - 1000.0) * 1e-4);
            q8.extend_from_slice(&d.to_le_bytes());
            for _ in 0..32 {
                q8.push(r.next() as u8);
            }
        }
        q8
    }

    fn random_perm(r: &mut Rng, o: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..o as u32).collect();
        for i in (1..o).rev() {
            p.swap(i, r.below(i + 1));
        }
        p
    }

    /// The weight (packed row n, reduction index kk) of the packed linear, written from the formulas:
    /// block scale d and code q of source row perm[n], block kk / 32, code kk % 32.
    fn ref_weight(q8: &[u8], k: usize, perm: Option<&[u32]>, fmt: WFmt, n: usize, kk: usize) -> f32 {
        let s = perm.map_or(n, |p| p[n] as usize);
        let blk = (s * (k / 32) + kk / 32) * 34;
        let d = tiles::half_to_f32(u16::from_le_bytes([q8[blk], q8[blk + 1]]));
        let q = q8[blk + 2 + kk % 32] as i8;
        match fmt {
            WFmt::W8 => d * q as f32,
            WFmt::Fp16 => tiles::half_to_f32(tiles::f32_to_half(d * q as f32)),
        }
    }

    /// Reference bytes of one fp16 strip, element by element.
    fn ref_fp16_strip(q8: &[u8], k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot: usize) -> Vec<u8> {
        let mut buf = vec![0u8; (kc / 32) * 2048];
        for ktl in 0..kc / 32 {
            for nl in 0..32 {
                for kl in 0..32 {
                    let kk = chunk * kc + ktl * 32 + kl;
                    let n = ot * 32 + nl;
                    let v = ref_weight(q8, k, perm, WFmt::Fp16, n, kk);
                    let idx = ktl * 1024 + (kl / 2) * 64 + 2 * nl + kl % 2;
                    let h = tiles::f32_to_half(v);
                    buf[2 * idx..2 * idx + 2].copy_from_slice(&h.to_le_bytes());
                }
            }
        }
        buf
    }

    #[test]
    fn fp16_strips_match_reference() {
        let mut r = Rng(0x1f2e_3d4c_5b6a_7988);
        for &(o, k, kc) in &[(64usize, 128usize, 64usize), (96, 256, 256)] {
            let q8 = random_q8(&mut r, o, k);
            let perm = random_perm(&mut r, o);
            for p in [None, Some(&perm[..])] {
                for chunk in 0..k / kc {
                    for ot in 0..o / 32 {
                        let mut got = vec![0xabu8; WFmt::Fp16.strip_bytes(kc)];
                        pack_fp16_strip(&mut got, &q8, o, k, kc, p, chunk, ot);
                        let want = ref_fp16_strip(&q8, k, kc, p, chunk, ot);
                        assert!(got == want, "o {o} k {k} kc {kc} perm {} chunk {chunk} ot {ot}", p.is_some());
                    }
                }
            }
        }
    }

    /// Q8_0 bytes whose scales hit the special half patterns (zero, subnormal, max finite, inf, NaN, smallest
    /// normal), random bit patterns, and small normals so that products underflow; codes include -128, -1, 0, 127.
    fn random_q8_special(r: &mut Rng, o: usize, k: usize) -> Vec<u8> {
        const SPECIAL: [u16; 16] = [
            0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0200, 0x0400, 0x0800, 0x7bff, 0xfbff, 0x3c00, 0xbc00, 0x7c00,
            0xfc00, 0x7c01, 0xfe55,
        ];
        let mut q8 = Vec::with_capacity(o * (k / 32) * 34);
        for _ in 0..o * (k / 32) {
            let d: u16 = match r.below(4) {
                0 => SPECIAL[r.below(SPECIAL.len())],
                1 => r.next() as u16,
                _ => {
                    let mag = 10f32.powi(-(r.below(12) as i32));
                    tiles::f32_to_half(r.sym() * mag)
                }
            };
            q8.extend_from_slice(&d.to_le_bytes());
            for _ in 0..32 {
                let c = match r.below(8) {
                    0 => -128i8 as u8,
                    1 => 127,
                    2 => 0,
                    3 => 0xff,
                    _ => r.next() as u8,
                };
                q8.push(c);
            }
        }
        q8
    }

    #[test]
    fn fp16_strips_match_ref_bitwise() {
        let mut r = Rng(0x7f4a_7c15_9e37_79b9);
        for &(o, k, kc) in &[(64usize, 128usize, 64usize), (96, 256, 256), (96, 3072, 1536)] {
            for _ in 0..2 {
                let q8 = random_q8_special(&mut r, o, k);
                let perm = random_perm(&mut r, o);
                for p in [None, Some(&perm[..])] {
                    for chunk in 0..k / kc {
                        for ot in 0..o / 32 {
                            let sb = WFmt::Fp16.strip_bytes(kc);
                            let mut got = vec![0xabu8; sb];
                            pack_fp16_strip(&mut got, &q8, o, k, kc, p, chunk, ot);
                            let mut want = vec![0x5au8; sb];
                            pack_fp16_strip_ref(&mut want, &q8, o, k, kc, p, chunk, ot);
                            assert!(got == want, "o {o} k {k} kc {kc} perm {} chunk {chunk} ot {ot}", p.is_some());
                        }
                    }
                }
            }
        }
    }

    /// Micro-benchmark: `cargo test --release -- --ignored --nocapture bench_pack_fp16`. Packs every strip of a
    /// 3072 x 3072 linear (one K chunk, 96 output tiles) single-threaded: old and new fp16 packer, and
    /// `tiles::pack_w8_strip` for comparison. MB = 10^6 bytes; fp16 rate on output bytes, w8 rate on o*k source bytes.
    #[test]
    #[ignore]
    fn bench_pack_fp16() {
        use std::time::Instant;
        let (o, k, kc) = (3072usize, 3072usize, 3072usize);
        let nt = o / TILE;
        let mut r = Rng(0x0123_4567_89ab_cdef);
        let q8 = random_q8(&mut r, o, k);
        let sb = WFmt::Fp16.strip_bytes(kc);
        let wb = tiles::w8_strip_bytes(kc);
        let mut old = vec![0u8; nt * sb];
        let mut new = vec![0u8; nt * sb];
        let mut w8 = vec![0u8; nt * wb];
        let mb_out = (o * k * 2) as f64 / 1e6;
        let mb_src = (o * k) as f64 / 1e6;
        let (mut sum_old, mut sum_new, mut sum_w8) = (0f64, 0f64, 0f64);
        let reps = 3;
        for rep in 0..reps {
            let t = Instant::now();
            for ot in 0..nt {
                pack_fp16_strip_ref(&mut old[ot * sb..(ot + 1) * sb], std::hint::black_box(&q8), o, k, kc, None, 0, ot);
            }
            let t_old = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for ot in 0..nt {
                pack_fp16_strip(&mut new[ot * sb..(ot + 1) * sb], std::hint::black_box(&q8), o, k, kc, None, 0, ot);
            }
            let t_new = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for ot in 0..nt {
                tiles::pack_w8_strip(&mut w8[ot * wb..(ot + 1) * wb], std::hint::black_box(&q8), o, k, kc, None, 0, ot);
            }
            let t_w8 = t.elapsed().as_secs_f64();
            assert!(old == new, "bench: new fp16 output differs from the reference");
            let (rate_old, rate_new, rate_w8) = (mb_out / t_old, mb_out / t_new, mb_src / t_w8);
            println!("rep {rep}: fp16 old {rate_old:.0} MB/s, fp16 new {rate_new:.0} MB/s, w8 {rate_w8:.0} MB/s");
            sum_old += rate_old;
            sum_new += rate_new;
            sum_w8 += rate_w8;
        }
        let n = reps as f64;
        println!(
            "mean over {reps} reps: fp16 old {:.0} MB/s, fp16 new {:.0} MB/s ({:.2}x), w8 {:.0} MB/s",
            sum_old / n,
            sum_new / n,
            sum_new / sum_old,
            sum_w8 / n
        );
    }

    #[test]
    fn weight_row_matches_reference_both_formats() {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let (o, k, kc) = (96usize, 256usize, 64usize);
        let q8 = random_q8(&mut r, o, k);
        let perm = random_perm(&mut r, o);
        for fmt in [WFmt::W8, WFmt::Fp16] {
            let sb = fmt.strip_bytes(kc);
            // output tiles 1 and 2 (ot0 = 1, nt = 2) of every chunk, packed as a blob per chunk
            let (ot0, nt) = (1usize, 2usize);
            let blobs: Vec<Vec<u8>> = (0..k / kc)
                .map(|c| {
                    let mut b = vec![0u8; nt * sb];
                    for i in 0..nt {
                        pack_strip(fmt, &mut b[i * sb..(i + 1) * sb], &q8, o, k, kc, Some(&perm), c, ot0 + i);
                    }
                    b
                })
                .collect();
            let mut out = vec![0f32; kc];
            for _ in 0..200 {
                let n = (ot0 + r.below(nt)) * 32 + r.below(32);
                for (c, blob) in blobs.iter().enumerate() {
                    unsafe { weight_row(fmt, blob.as_ptr(), kc, ot0, n, &mut out) };
                    for (kl, &w) in out.iter().enumerate() {
                        let want = ref_weight(&q8, k, Some(&perm), fmt, n, c * kc + kl);
                        assert_eq!(w.to_bits(), want.to_bits(), "{fmt:?} row {n} chunk {c} kl {kl}");
                    }
                }
            }
        }
    }

    fn round_up(n: usize) -> usize {
        n.div_ceil(4096) * 4096
    }

    /// One GEMM per K chunk through the emulator: X (activation tiles), the weight blobs of `fmt`, Y column-major.
    fn emu_gemm(fmt: WFmt, q8: &[u8], x_rows: &[Vec<f32>], t: usize, k: usize, o: usize, kc: usize) -> Vec<f32> {
        let mut e = Emu::new(8 << 20, 1);
        let lay = tiles::ActLayout::new(t, k, kc);
        let xb = e.alloc(0, round_up(lay.bytes())).expect("x buffer");
        for (m, row) in x_rows.iter().enumerate() {
            unsafe { tiles::write_act_row(xb.ptr as *mut u16, &lay, m, 0, row) };
        }
        let nchunks = k / kc;
        let ol = tiles::OutLayout { t_pad: lay.t_pad, o, nchunks, ycol: true };
        let yb = e.alloc(0, round_up(ol.bytes())).expect("y buffer");
        let sb = fmt.strip_bytes(kc);
        let wbytes = nchunks * (o / 32) * sb;
        let wb = e.alloc(0, round_up(wbytes)).expect("w buffer");
        for c in 0..nchunks {
            let base = c * (o / 32) * sb;
            let blob = unsafe { std::slice::from_raw_parts_mut(wb.ptr.add(base), (o / 32) * sb) };
            for ot in 0..o / 32 {
                pack_strip(fmt, &mut blob[ot * sb..(ot + 1) * sb], q8, o, k, kc, None, c, ot);
            }
            let a = GemmArgs {
                x_slot: xb.slot,
                x_off: (c * lay.chunk_bytes()) as u32,
                w_slot: wb.slot,
                w_off: base as u32,
                y_slot: yb.slot,
                y_off: (c * ol.chunk_bytes()) as u32,
                t: lay.t_pad as u32,
                k: kc as u32,
                o: o as u32,
                flags: fmt.gemm_flag() | GF_YCOL,
                ..Default::default()
            };
            e.gemm(0, &a).expect("gemm");
        }
        let mut y = vec![0f32; t * o];
        for m in 0..t {
            unsafe { tiles::read_out_row(yb.ptr as *const u16, &ol, m, 0, &mut y[m * o..(m + 1) * o]) };
        }
        y
    }

    #[test]
    fn fp16_strips_match_w8_in_the_emulator() {
        let (t, k, o, kc) = (64usize, 128usize, 64usize, 64usize);
        let mut r = Rng(0x2545_f491_4f6c_dd1d);
        let q8 = random_q8(&mut r, o, k);
        let x_rows: Vec<Vec<f32>> = (0..t).map(|_| (0..k).map(|_| r.sym()).collect()).collect();
        let y_w8 = emu_gemm(WFmt::W8, &q8, &x_rows, t, k, o, kc);
        let y_fp = emu_gemm(WFmt::Fp16, &q8, &x_rows, t, k, o, kc);
        let (mut se, mut sr) = (0f64, 0f64);
        for (a, b) in y_fp.iter().zip(&y_w8) {
            se += ((a - b) as f64).powi(2);
            sr += (*b as f64).powi(2);
        }
        let rel = (se / sr).sqrt();
        assert!(sr > 0.0, "W8 result is zero");
        assert!(rel < 1e-3, "fp16 vs W8 rel RMSE {rel}");
    }
}
