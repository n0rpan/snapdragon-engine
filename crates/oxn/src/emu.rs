//! Pure-Rust CPU emulation of the oxn GEMM contract on heap memory. For tests without hardware.
//!
//! Validation mirrors `oxn_gemm_check` (npu/oxn/common/oxn_check.h); the arithmetic is an f32
//! accumulation in ascending k, rounded to IEEE half on store. Sessions are independent slot tables
//! (as the device's FastRPC sessions): a slot of one session cannot be used in another.

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::time::Instant;

use crate::{Backend, Buf, GemmArgs, GemmStats, GF_NO_HMX, GF_NO_WFETCH, GF_NO_YWRITE, GF_W8, GF_YCOL};

const ALIGN: usize = 4096;
/// slots per session, as the device (OXN_MAX_SLOTS)
const MAX_SLOTS: usize = 32;
const MAX_BYTES: usize = 128 << 20;
const TILE_ELMS: usize = 1024;
const W8_TILE_BYTES: usize = 1152;
const W8_SCALE_OFF: usize = 1024;
const KNOWN: u32 = GF_YCOL | GF_NO_WFETCH | GF_NO_YWRITE | GF_NO_HMX | GF_W8;
const PROBES: u32 = GF_NO_WFETCH | GF_NO_YWRITE | GF_NO_HMX;

struct Slot {
    ptr: *mut u8,
    len: usize,
}

/// CPU emulation backend. Owns its slots; they are freed on drop.
pub struct Emu {
    vtcm: u32,
    /// slot table per session
    sessions: Vec<Vec<Slot>>,
}

// The raw pointers are exclusively owned by this Emu (allocated in `alloc`, freed in `drop`).
unsafe impl Send for Emu {}

impl Emu {
    /// `n_sessions` >= 1 independent sessions, each with VTCM `vtcm_bytes` (the same size) and its own slots.
    pub fn new(vtcm_bytes: u32, n_sessions: usize) -> Emu {
        assert!(n_sessions >= 1, "Emu needs at least one session");
        Emu { vtcm: vtcm_bytes, sessions: (0..n_sessions).map(|_| Vec::new()).collect() }
    }
}

impl Drop for Emu {
    fn drop(&mut self) {
        for slots in &self.sessions {
            for s in slots {
                unsafe { dealloc(s.ptr, Layout::from_size_align(s.len, ALIGN).unwrap()) };
            }
        }
    }
}

/// IEEE binary16 bits -> f32 (scalar, exact).
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            sign
        } else {
            // subnormal: value = m * 2^-24; normalise so bit 10 is the implicit one
            let mut mm = m;
            let mut ex: u32 = 113;
            while mm & 0x400 == 0 {
                mm <<= 1;
                ex -= 1;
            }
            sign | (ex << 23) | ((mm & 0x3ff) << 13)
        }
    } else if e == 31 {
        sign | 0x7f80_0000 | (m << 13)
    } else {
        sign | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE binary16 bits, round to nearest even. NaN maps to a quiet NaN.
fn f32_to_half(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return if man != 0 { sign | 0x7e00 } else { sign | 0x7c00 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // half subnormal or zero: result = round(m * 2^(exp-126)) with m the 24-bit significand
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut hm = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (hm & 1) == 1) {
            hm += 1;
        }
        // hm == 1024 after rounding is the smallest normal, which the bit pattern already encodes
        return sign | hm as u16;
    }
    let mut hm = man >> 13;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (hm & 1) == 1) {
        hm += 1;
    }
    // a carry out of the mantissa bumps the exponent (and to inf at the top), which the sum encodes
    sign | (((e as u32) << 10) + hm) as u16
}

/// Element index within a 1024-element tile: activation (m,k), weight (k,n) and output (m,n) use the same rule
/// with the first index in pairs: (i/2)*64 + 2*j + (i&1), where i is the row-like and j the column-like index.
#[inline]
fn tile_idx(i: usize, j: usize) -> usize {
    let i = i % 32;
    let j = j % 32;
    (i / 2) * 64 + 2 * j + (i & 1)
}

#[inline]
fn u16_at(b: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([b[pos], b[pos + 1]])
}

impl Backend for Emu {
    fn name(&self) -> &'static str {
        "emu"
    }

    fn sessions(&self) -> usize {
        self.sessions.len()
    }

    fn vtcm_bytes(&self) -> u32 {
        self.vtcm
    }

    fn alloc(&mut self, session: usize, bytes: usize) -> Result<Buf, String> {
        if session >= self.sessions.len() {
            return Err(format!("alloc: session {session} out of range (have {})", self.sessions.len()));
        }
        if bytes == 0 || bytes % ALIGN != 0 || bytes > MAX_BYTES {
            return Err(format!("alloc: {bytes} bytes must be a non-zero multiple of 4096 and <= 128 MiB"));
        }
        let tab = &mut self.sessions[session];
        if tab.len() >= MAX_SLOTS {
            return Err(format!("alloc: session {session}: all {MAX_SLOTS} slots in use"));
        }
        let layout = Layout::from_size_align(bytes, ALIGN).map_err(|e| e.to_string())?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(format!("alloc: out of memory for {bytes} bytes"));
        }
        let slot = tab.len() as u32;
        tab.push(Slot { ptr, len: bytes });
        Ok(Buf { session: session as u32, slot, ptr, len: bytes })
    }

    fn gemm(&mut self, session: usize, a: &GemmArgs) -> Result<GemmStats, String> {
        let t0 = Instant::now();
        if session >= self.sessions.len() {
            return Err(format!("GEMM_SESSION: session {session} out of range (have {})", self.sessions.len()));
        }
        let tab = &self.sessions[session];
        let (t, k, o) = (a.t as usize, a.k as usize, a.o as usize);
        for (name, d) in [("T", a.t), ("K", a.k), ("O", a.o)] {
            if d == 0 || d % 32 != 0 || d > 65536 {
                return Err(format!("GEMM_DIMS: {name}={d} must be a non-zero multiple of 32 and <= 65536"));
            }
        }
        if a.flags & !KNOWN != 0 {
            return Err(format!("GEMM_FLAGS: unknown flag bits 0x{:x}", a.flags & !KNOWN));
        }
        if a.flags & PROBES != 0 {
            return Err("probe flags not emulated".to_string());
        }
        if a.r > 255 || a.c > 255 {
            return Err(format!("GEMM_FLAGS: R/C override above 255 (r {}, c {})", a.r, a.c));
        }
        let w8 = a.flags & GF_W8 != 0;
        let nkt = (k / 32) as u64;
        let wlen: u64 = if w8 { (o / 32) as u64 * nkt * W8_TILE_BYTES as u64 } else { 2 * o as u64 * k as u64 };
        let len = [2 * t as u64 * k as u64, wlen, 2 * t as u64 * o as u64];
        let slot = [a.x_slot, a.w_slot, a.y_slot];
        let off = [a.x_off as u64, a.w_off as u64, a.y_off as u64];
        let names = ["X", "W", "Y"];
        for i in 0..3 {
            let s = slot[i] as usize;
            if s >= tab.len() {
                return Err(format!("GEMM_SLOT_{}: slot {} not mapped in session {session}", names[i], slot[i]));
            }
        }
        for i in 0..3 {
            let cap = tab[slot[i] as usize].len as u64;
            if off[i] + len[i] > cap {
                return Err(format!("GEMM_BOUNDS_{}: {} + {} bytes past slot of {} bytes", names[i], off[i], len[i], cap));
            }
        }
        for i in 0..3 {
            if off[i] % 128 != 0 {
                return Err(format!("GEMM_ALIGN: {} offset {} not 128-byte aligned", names[i], off[i]));
            }
        }
        for i in 0..2 {
            if slot[i] == slot[2] && off[i] < off[2] + len[2] && off[2] < off[i] + len[i] {
                return Err(format!("GEMM_OVERLAP: Y overlaps {}", names[i]));
            }
        }
        let min_vtcm = 4096u64 + 2048 * nkt + 2 * 2048 * nkt + 2 * 2048;
        if (self.vtcm as u64) < min_vtcm {
            return Err(format!("GEMM_VTCM_CAP: VTCM {} below the minimum {} bytes", self.vtcm, min_vtcm));
        }

        // decode X [T][K] and W [O][K] to f32 (tile layouts)
        let mut x = vec![0f32; t * k];
        {
            let xb = unsafe { std::slice::from_raw_parts(tab[a.x_slot as usize].ptr.add(off[0] as usize), len[0] as usize) };
            for m in 0..t {
                for kk in 0..k {
                    let tile = (m / 32) * (k / 32) + kk / 32;
                    let pos = (tile * TILE_ELMS + tile_idx(m, kk)) * 2;
                    x[m * k + kk] = half_to_f32(u16_at(xb, pos));
                }
            }
        }
        let mut w = vec![0f32; o * k];
        {
            let wb = unsafe { std::slice::from_raw_parts(tab[a.w_slot as usize].ptr.add(off[1] as usize), len[1] as usize) };
            for n in 0..o {
                for kk in 0..k {
                    let tile = (n / 32) * (k / 32) + kk / 32;
                    let idx = tile_idx(kk, n);
                    w[n * k + kk] = if w8 {
                        let rec = tile * W8_TILE_BYTES;
                        let q = wb[rec + idx] as i8;
                        let s = half_to_f32(u16_at(wb, rec + W8_SCALE_OFF + 2 * (n % 32)));
                        s * q as f32
                    } else {
                        half_to_f32(u16_at(wb, (tile * TILE_ELMS + idx) * 2))
                    };
                }
            }
        }

        // accumulate and round to half, logical [T][O]
        let mut y = vec![0u16; t * o];
        for m in 0..t {
            let xr = &x[m * k..(m + 1) * k];
            for n in 0..o {
                let wr = &w[n * k..(n + 1) * k];
                let mut acc = 0f32;
                for kk in 0..k {
                    acc += xr[kk] * wr[kk];
                }
                y[m * o + n] = f32_to_half(acc);
            }
        }

        // store into the Y tile layout
        let ycol = a.flags & GF_YCOL != 0;
        let (nrt, not) = (t / 32, o / 32);
        let yb = tab[a.y_slot as usize].ptr;
        for m in 0..t {
            for n in 0..o {
                let tile = if ycol { (n / 32) * nrt + m / 32 } else { (m / 32) * not + n / 32 };
                let pos = off[2] as usize + (tile * TILE_ELMS + tile_idx(m, n)) * 2;
                let bytes = y[m * o + n].to_le_bytes();
                unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), yb.add(pos), 2) };
            }
        }

        Ok(GemmStats { wall_s: t0.elapsed().as_secs_f64(), ..Default::default() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u32) -> u32 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *s = x;
        x
    }

    /// uniform in [-scale, scale)
    fn rand_f(s: &mut u32, scale: f32) -> f32 {
        let u = (rng(s) >> 8) as f32 * (1.0 / 16_777_216.0);
        (2.0 * u - 1.0) * scale
    }

    fn round_up(n: usize) -> usize {
        n.div_ceil(ALIGN) * ALIGN
    }

    fn write_u16(buf: &Buf, byte_off: usize, vals: &[u16]) {
        for (i, v) in vals.iter().enumerate() {
            let b = v.to_le_bytes();
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), buf.ptr.add(byte_off + 2 * i), 2) };
        }
    }

    fn read_u16(buf: &Buf, byte_off: usize, count: usize) -> Vec<u16> {
        let s = unsafe { std::slice::from_raw_parts(buf.ptr.add(byte_off), 2 * count) };
        (0..count).map(|i| u16_at(s, 2 * i)).collect()
    }

    #[test]
    fn half_round_trip_all_patterns() {
        for h in 0..=u16::MAX {
            let f = half_to_f32(h);
            let is_nan = (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0;
            if is_nan {
                assert!(f.is_nan(), "pattern 0x{h:04x} should decode to NaN");
                assert_eq!(f32_to_half(f) & 0x7c00, 0x7c00);
                assert_ne!(f32_to_half(f) & 0x3ff, 0);
            } else {
                assert_eq!(f32_to_half(f), h, "pattern 0x{h:04x} ({f}) did not round-trip");
            }
        }
    }

    #[test]
    fn half_ties_round_to_even() {
        // midpoints between adjacent positive finite halves; the result must be the one with an even bit pattern
        for h in 0u16..0x7bff {
            let mid = (half_to_f32(h) + half_to_f32(h + 1)) * 0.5;
            let even = if h & 1 == 0 { h } else { h + 1 };
            assert_eq!(f32_to_half(mid), even, "midpoint above 0x{h:04x}");
            assert_eq!(f32_to_half(-mid), even | 0x8000, "negative midpoint above 0x{h:04x}");
        }
        assert_eq!(f32_to_half(65519.0), 0x7bff);
        assert_eq!(f32_to_half(65520.0), 0x7c00, "65520 is the tie between max half and inf: even is inf");
        assert_eq!(f32_to_half(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_half(1e-9), 0);
    }

    /// Logical X [T][K] -> activation tiles [rt][kt]; element (m,k) uses the row-paired index.
    fn pack_x(logical: &[u16], t: usize, k: usize) -> Vec<u16> {
        let nkt = k / 32;
        let mut out = vec![0u16; (t / 32) * nkt * TILE_ELMS];
        for m in 0..t {
            for kk in 0..k {
                out[((m / 32) * nkt + kk / 32) * TILE_ELMS + tile_idx(m, kk)] = logical[m * k + kk];
            }
        }
        out
    }

    /// Logical W [O][K] -> weight tiles [ot][kt]; element (k,n) uses the k-paired index.
    fn pack_w(logical: &[u16], o: usize, k: usize) -> Vec<u16> {
        let nkt = k / 32;
        let mut out = vec![0u16; (o / 32) * nkt * TILE_ELMS];
        for n in 0..o {
            for kk in 0..k {
                out[((n / 32) * nkt + kk / 32) * TILE_ELMS + tile_idx(kk, n)] = logical[n * k + kk];
            }
        }
        out
    }

    struct Case {
        t: usize,
        k: usize,
        o: usize,
        flags: u32,
        x: Vec<u16>,
        w_fp16: Vec<u16>,
        w_deq: Vec<f64>,
        w8: Vec<u8>,
    }

    /// Random X [T][K] and W [O][K]: fp16 weights, or W8 records (random int8 and random half scale per (n, k-block)).
    fn make_case(t: usize, k: usize, o: usize, w8: bool, seed: u32) -> Case {
        let mut s = seed;
        let xf: Vec<f32> = (0..t * k).map(|_| rand_f(&mut s, 1.0)).collect();
        let x: Vec<u16> = xf.iter().map(|&v| f32_to_half(v)).collect();
        let nkt = k / 32;
        if !w8 {
            let wf: Vec<f32> = (0..o * k).map(|_| rand_f(&mut s, 1.0 / (k as f32).sqrt())).collect();
            let wh: Vec<u16> = wf.iter().map(|&v| f32_to_half(v)).collect();
            let w_deq = wh.iter().map(|&h| half_to_f32(h) as f64).collect();
            let w_fp16 = pack_w(&wh, o, k);
            return Case { t, k, o, flags: 0, x: pack_x(&x, t, k), w_fp16, w_deq, w8: Vec::new() };
        }
        let mut rec = vec![0u8; (o / 32) * nkt * W8_TILE_BYTES];
        let mut w_deq = vec![0f64; o * k];
        for n in 0..o {
            for kb in 0..nkt {
                let scale_bits = f32_to_half(0.01 + 0.04 * (rng(&mut s) >> 8) as f32 / 16_777_216.0);
                let tile = (n / 32) * nkt + kb;
                let base = tile * W8_TILE_BYTES;
                let j = n % 32;
                rec[base + W8_SCALE_OFF + 2 * j..base + W8_SCALE_OFF + 2 * j + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for kk in 0..32 {
                    let q = (rng(&mut s) % 255) as i32 - 127;
                    rec[base + tile_idx(kk, j)] = q as i8 as u8;
                    w_deq[n * k + kb * 32 + kk] = half_to_f32(scale_bits) as f64 * q as f64;
                }
            }
        }
        Case { t, k, o, flags: GF_W8, x: pack_x(&x, t, k), w_fp16: Vec::new(), w_deq, w8: rec }
    }

    /// Runs the GEMM and returns the relative RMSE against the f64 reference of the same inputs.
    fn run_case(c: &Case, ycol: bool) -> f64 {
        let mut e = Emu::new(8 << 20, 1);
        let (t, k, o) = (c.t, c.k, c.o);
        let xb = e.alloc(0, round_up(2 * t * k)).unwrap();
        let wlen = if c.flags & GF_W8 != 0 { c.w8.len() } else { 2 * o * k };
        let wb = e.alloc(0, round_up(wlen)).unwrap();
        let yb = e.alloc(0, round_up(2 * t * o)).unwrap();
        write_u16(&xb, 0, &c.x);
        if c.flags & GF_W8 != 0 {
            unsafe { std::ptr::copy_nonoverlapping(c.w8.as_ptr(), wb.ptr, c.w8.len()) };
        } else {
            write_u16(&wb, 0, &c.w_fp16);
        }
        let flags = c.flags | if ycol { GF_YCOL } else { 0 };
        let a = GemmArgs {
            x_slot: xb.slot,
            w_slot: wb.slot,
            y_slot: yb.slot,
            t: t as u32,
            k: k as u32,
            o: o as u32,
            flags,
            ..Default::default()
        };
        e.gemm(0, &a).unwrap();
        let yt = read_u16(&yb, 0, (t / 32) * (o / 32) * TILE_ELMS);
        let nrt = t / 32;
        let not = o / 32;
        let (mut se, mut sr) = (0f64, 0f64);
        for m in 0..t {
            for n in 0..o {
                let tile = if ycol { (n / 32) * nrt + m / 32 } else { (m / 32) * not + n / 32 };
                let got = half_to_f32(yt[tile * TILE_ELMS + tile_idx(m, n)]) as f64;
                let reference: f64 = (0..k)
                    .map(|kk| {
                        let xv = half_to_f32(read_x(&c.x, t, k, m, kk)) as f64;
                        xv * c.w_deq[n * k + kk]
                    })
                    .sum();
                se += (got - reference).powi(2);
                sr += reference * reference;
            }
        }
        (se / sr).sqrt()
    }

    fn read_x(xt: &[u16], _t: usize, k: usize, m: usize, kk: usize) -> u16 {
        let tile = (m / 32) * (k / 32) + kk / 32;
        xt[tile * TILE_ELMS + tile_idx(m, kk)]
    }

    #[test]
    fn gemm_fp16_matches_reference() {
        let c = make_case(64, 96, 160, false, 0x1234_5678);
        let rows = run_case(&c, false);
        assert!(rows < 1e-3, "row-major rel RMSE {rows}");
        let cols = run_case(&c, true);
        assert!(cols < 1e-3, "column-major rel RMSE {cols}");
    }

    #[test]
    fn gemm_w8_matches_dequantised_reference() {
        let c = make_case(64, 96, 160, true, 0x9abc_def1);
        let rel = run_case(&c, false);
        assert!(rel < 1e-3, "W8 rel RMSE {rel}");
        let rel_col = run_case(&c, true);
        assert!(rel_col < 1e-3, "W8 column-major rel RMSE {rel_col}");
    }

    #[test]
    fn gemm_rejections() {
        let mut e = Emu::new(8 << 20, 1);
        let big = e.alloc(0, 40960).unwrap();   // X at 0 (12288 bytes), Y in the same slot
        let w = e.alloc(0, 32768).unwrap();     // W [160][96] fp16 = 30720 bytes
        let y = e.alloc(0, 20480).unwrap();
        let base = GemmArgs {
            x_slot: big.slot,
            w_slot: w.slot,
            y_slot: y.slot,
            t: 64,
            k: 96,
            o: 160,
            ..Default::default()
        };
        let expect = |a: GemmArgs, needle: &str, e: &mut Emu| {
            let err = e.gemm(0, &a).expect_err("expected rejection");
            assert!(err.contains(needle), "wanted '{needle}' in '{err}'");
        };
        // bad dims
        expect(GemmArgs { t: 0, ..base }, "GEMM_DIMS", &mut e);
        expect(GemmArgs { t: 33, ..base }, "GEMM_DIMS", &mut e);
        expect(GemmArgs { k: 100, ..base }, "GEMM_DIMS", &mut e);
        // unmapped slot
        expect(GemmArgs { x_slot: 9, ..base }, "not mapped", &mut e);
        // out of bounds: X offset pushes past its 40960-byte slot
        expect(GemmArgs { x_off: 40960 - 12288 + 128, ..base }, "GEMM_BOUNDS_X", &mut e);
        // Y in the same slot as X, overlapping [0, 12288)
        let same = GemmArgs { y_slot: big.slot, y_off: 8192, ..base };
        expect(same, "GEMM_OVERLAP", &mut e);
        // unknown flag bit
        expect(GemmArgs { flags: 0x20, ..base }, "unknown flag", &mut e);
        // probe flags
        expect(GemmArgs { flags: GF_NO_HMX, ..base }, "probe flags not emulated", &mut e);
        // alignment
        expect(GemmArgs { x_off: 64, ..base }, "GEMM_ALIGN", &mut e);
        // R/C override above 255
        expect(GemmArgs { r: 256, ..base }, "GEMM_FLAGS", &mut e);
        // VTCM below the R=C=1 minimum
        let mut small = Emu::new(4096, 1);
        let sx = small.alloc(0, 12288).unwrap();
        let sw = small.alloc(0, 30720_usize.div_ceil(4096) * 4096).unwrap();
        let sy = small.alloc(0, 20480).unwrap();
        let a = GemmArgs { x_slot: sx.slot, w_slot: sw.slot, y_slot: sy.slot, t: 64, k: 96, o: 160, ..Default::default() };
        expect(a, "GEMM_VTCM_CAP", &mut small);
        // a valid request on the same buffers still runs
        assert!(e.gemm(0, &GemmArgs { ..base }).is_ok());
    }

    #[test]
    fn two_sessions_are_independent() {
        let mut e = Emu::new(8 << 20, 2);
        assert_eq!(e.sessions(), 2);
        // same shape on both sessions, different data: slot 0 = X, 1 = W, 2 = Y in each session
        let (t, k, o) = (64usize, 96usize, 160usize);
        let mut runs = Vec::new();
        for (sess, seed) in [(0usize, 0x1111_2222u32), (1, 0x3333_4444u32)] {
            let c = make_case(t, k, o, false, seed);
            let xb = e.alloc(sess, round_up(2 * t * k)).unwrap();
            let wb = e.alloc(sess, round_up(2 * o * k)).unwrap();
            let yb = e.alloc(sess, round_up(2 * t * o)).unwrap();
            assert_eq!((xb.session as usize, wb.session as usize, yb.session as usize), (sess, sess, sess));
            assert_eq!((xb.slot, wb.slot, yb.slot), (0, 1, 2), "slots count from 0 in each session");
            write_u16(&xb, 0, &c.x);
            write_u16(&wb, 0, &c.w_fp16);
            let a = GemmArgs {
                x_slot: xb.slot,
                w_slot: wb.slot,
                y_slot: yb.slot,
                t: t as u32,
                k: k as u32,
                o: o as u32,
                ..Default::default()
            };
            e.gemm(sess, &a).unwrap();
            runs.push((c, read_u16(&yb, 0, (t / 32) * (o / 32) * TILE_ELMS)));
        }
        // each session's result matches its own reference (run_case checks the same path)
        for (c, yt) in &runs {
            let not = o / 32;
            let mut se = 0f64;
            let mut sr = 0f64;
            for m in 0..t {
                for n in 0..o {
                    let tile = (m / 32) * not + n / 32;
                    let got = half_to_f32(yt[tile * TILE_ELMS + tile_idx(m, n)]) as f64;
                    let reference: f64 = (0..k)
                        .map(|kk| half_to_f32(read_x(&c.x, t, k, m, kk)) as f64 * c.w_deq[n * k + kk])
                        .sum();
                    se += (got - reference).powi(2);
                    sr += reference * reference;
                }
            }
            assert!((se / sr).sqrt() < 1e-3, "session result off the reference");
        }
        // the two sessions hold different data, so their results differ
        assert_ne!(runs[0].1, runs[1].1);
        // a slot that exists only in session 0 is not a slot of session 1
        let extra = e.alloc(0, 4096).unwrap();
        let only0 = GemmArgs { x_slot: extra.slot, w_slot: 1, y_slot: 2, t: 32, k: 32, o: 32, ..Default::default() };
        let err = e.gemm(1, &only0).expect_err("session 0's slot 3 used in session 1");
        assert!(err.contains("not mapped in session 1"), "{err}");
        // and the session index itself is checked
        assert!(e.alloc(2, 4096).is_err());
        assert!(e.gemm(2, &only0).is_err());
    }
}
