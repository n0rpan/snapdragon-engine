//! Float attention for the fast path (F-IMG-ATTN): the linears stay int8, only q.k^T and
//! softmax.v run in f32 ([`AttnPrec::F32`]) or with f16 operands and f32 accumulation
//! ([`AttnPrec::F16`], NEON FMLAL). [`AttnPrec::Int8`] (the default) is the original int8 flash
//! attention and is not routed through this module at all.
//!
//! W5-G measured the int8 attention as the fast block's whole excess over sd.cpp arithmetic at
//! chunk 32 (attn 1.25% vs MLP 0.025% per block); sd.cpp runs attention in f32
//! (`--flash-attn off`), so [`AttnPrec::F32`] is its arithmetic class.
//!
//! Per head the store holds q, k (final values: RMS-normed, scaled, RoPE'd; q also times
//! 1/sqrt(head_dim)) and v in the layouts of `int8_gemm::fattn` (csrc/i8g_fattn.c). The driver is
//! flash-style: per query group of 8 and key chunk, scores -> running max -> exp (relative error
//! about 1e-7) -> p.v into f32 accumulators; the row sum is kept in f64.

use crate::fast::exp_fast;
use int8_gemm::fattn::raw;
use int8_gemm::{AlignedBytes, G};

/// Precision of the fast path's attention (the linears are int8 in every mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AttnPrec {
    /// int8 q/k/v, u8 probabilities (the original fast path).
    #[default]
    Int8,
    /// f16 q/k/v and probabilities, products and sums in f32.
    F16,
    /// f32 throughout.
    F32,
}

impl AttnPrec {
    pub fn parse(s: &str) -> Result<AttnPrec, String> {
        match s {
            "int8" => Ok(AttnPrec::Int8),
            "f16" => Ok(AttnPrec::F16),
            "f32" => Ok(AttnPrec::F32),
            _ => Err(format!("--attn-prec must be int8, f16 or f32 (got {s})")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            AttnPrec::Int8 => "int8",
            AttnPrec::F16 => "f16",
            AttnPrec::F32 => "f32",
        }
    }

    pub fn is_float(self) -> bool {
        self != AttnPrec::Int8
    }
}

struct FloatHead {
    /// `[query group][d][8]`
    qt: AlignedBytes,
    /// `[key step][d][8]`
    kt: AlignedBytes,
    /// `[dim tile][key (n_pad)][8]`
    vt: AlignedBytes,
}

/// q, k, v of all heads for `n` tokens in f32 or f16 (padding tokens are zero).
pub struct FloatStore {
    half: bool,
    n: usize,
    n_pad: usize,
    hd: usize,
    heads: Vec<FloatHead>,
}

/// Per-thread scratch of [`FloatStore::attend`].
pub struct FloatScratch {
    kc_max: usize,
    s: Vec<f32>,
    pt32: Vec<f32>,
    pt16: Vec<u16>,
    o: Vec<f32>,
    row: Vec<f32>,
}

impl FloatStore {
    /// `prec` must be F16 or F32.
    pub fn new(heads: usize, hd: usize, n: usize, prec: AttnPrec) -> FloatStore {
        assert!(prec.is_float(), "FloatStore is for f16/f32 attention");
        assert!(hd % G == 0 && hd > 0 && n > 0);
        let half = prec == AttnPrec::F16;
        let n_pad = n.div_ceil(G) * G;
        let bytes = n_pad * hd * if half { 2 } else { 4 };
        let heads = (0..heads).map(|_| FloatHead { qt: AlignedBytes::zeroed(bytes), kt: AlignedBytes::zeroed(bytes), vt: AlignedBytes::zeroed(bytes) }).collect();
        FloatStore { half, n, n_pad, hd, heads }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn heads(&self) -> usize {
        self.heads.len()
    }

    pub fn scratch(&self, key_chunk: usize) -> FloatScratch {
        assert!(key_chunk % G == 0 && key_chunk > 0);
        let kc_max = key_chunk.min(self.n_pad);
        FloatScratch {
            kc_max,
            s: vec![0.0; G * kc_max],
            pt32: if self.half { Vec::new() } else { vec![0.0; kc_max * G] },
            pt16: if self.half { vec![0; kc_max * G] } else { Vec::new() },
            o: vec![0.0; G * self.hd],
            row: vec![0.0; self.hd],
        }
    }

    /// `[group/step][d][8]` element (t, d) of `buf`, stride 8 over d.
    ///
    /// # Safety
    /// Element (t, 0..hd) of this buffer must be written by one thread only, with no concurrent reader.
    unsafe fn put_tr(&self, buf: &AlignedBytes, t: usize, vals: &[f32]) {
        let hd = self.hd;
        assert!(t < self.n && vals.len() >= hd);
        let off = ((t / G) * hd * G) + t % G;
        if self.half {
            raw::i8g_fattn_to_f16(vals.as_ptr(), hd, (buf.as_ptr() as *mut u16).add(off), G);
        } else {
            let dst = (buf.as_ptr() as *mut f32).add(off);
            for d in 0..hd {
                *dst.add(d * G) = vals[d];
            }
        }
    }

    /// Final q of (head, token): RMS-normed, scaled, RoPE'd and times 1/sqrt(head_dim).
    ///
    /// # Safety
    /// (head, q, token) must be written by one thread only, with no concurrent reader.
    pub unsafe fn put_q(&self, head: usize, t: usize, vals: &[f32]) {
        self.put_tr(&self.heads[head].qt, t, vals)
    }

    /// Final k of (head, token).
    ///
    /// # Safety
    /// As [`FloatStore::put_q`].
    pub unsafe fn put_k(&self, head: usize, t: usize, vals: &[f32]) {
        self.put_tr(&self.heads[head].kt, t, vals)
    }

    /// v of (head, token), straight from linear1.
    ///
    /// # Safety
    /// As [`FloatStore::put_q`].
    pub unsafe fn put_v(&self, head: usize, t: usize, row: &[f32]) {
        let (hd, n_pad) = (self.hd, self.n_pad);
        assert!(t < self.n && row.len() >= hd);
        let vt = &self.heads[head].vt;
        for dt in 0..hd / G {
            let off = (dt * n_pad + t) * G;
            if self.half {
                raw::i8g_fattn_to_f16(row.as_ptr().add(dt * G), G, (vt.as_ptr() as *mut u16).add(off), 1);
            } else {
                std::ptr::copy_nonoverlapping(row.as_ptr().add(dt * G), (vt.as_ptr() as *mut f32).add(off), G);
            }
        }
    }

    /// Attention of query group `g` (queries `8g..8g+8`) of `head` over all keys. Writes the
    /// normalised output of each REAL query `t` to `out(t, row)` (`row` has head_dim floats).
    /// The producing stage must have finished (barrier) before any call.
    pub fn attend(&self, head: usize, g: usize, key_chunk: usize, sc: &mut FloatScratch, mut out: impl FnMut(usize, &[f32])) {
        let (n, n_pad, hd) = (self.n, self.n_pad, self.hd);
        assert!(g * G < n_pad);
        let kc_max = sc.kc_max;
        assert!(key_chunk.min(n_pad) == kc_max, "scratch made for another key chunk");
        let hs = &self.heads[head];
        let mut m = [f32::NEG_INFINITY; G];
        let mut l = [0.0f64; G];
        sc.o.iter_mut().for_each(|v| *v = 0.0);
        let mut c0 = 0;
        while c0 < n_pad {
            let kc = key_chunk.min(n_pad - c0);
            // c0 < n_pad and n_pad - n < 8 with c0 a multiple of 8: at least one real key here.
            let nvalid = (n - c0).min(kc);
            // SAFETY: qt/kt hold n_pad*hd elements; group g and key steps c0/8.. (c0+kc)/8 are in
            // range; s holds 8 rows of kc_max >= kc floats.
            unsafe {
                if self.half {
                    raw::i8g_fattn_qk_f16((hs.qt.as_ptr() as *const u16).add(g * hd * G), (hs.kt.as_ptr() as *const u16).add(c0 * hd), hd, kc / G, sc.s.as_mut_ptr(), kc_max);
                } else {
                    raw::i8g_fattn_qk_f32((hs.qt.as_ptr() as *const f32).add(g * hd * G), (hs.kt.as_ptr() as *const f32).add(c0 * hd), hd, kc / G, sc.s.as_mut_ptr(), kc_max);
                }
            }
            for qi in 0..G {
                let row = &sc.s[qi * kc_max..qi * kc_max + kc];
                let cmax = row[..nvalid].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                if cmax > m[qi] {
                    let f = if m[qi] == f32::NEG_INFINITY { 0.0 } else { exp_fast(m[qi] - cmax) };
                    for v in &mut sc.o[qi * hd..(qi + 1) * hd] {
                        *v *= f;
                    }
                    l[qi] *= f as f64;
                    m[qi] = cmax;
                }
                // SAFETY: row holds kc floats (kc % 8 == 0); pt holds kc_max x 8 elements.
                let sum = unsafe {
                    if self.half {
                        raw::i8g_fattn_softmax_f16(row.as_ptr(), kc, nvalid, m[qi], sc.pt16.as_mut_ptr(), qi)
                    } else {
                        raw::i8g_fattn_softmax_f32(row.as_ptr(), kc, nvalid, m[qi], sc.pt32.as_mut_ptr(), qi)
                    }
                };
                l[qi] += sum as f64;
            }
            // SAFETY: vt holds hd/8 tiles of n_pad x 8; keys c0..c0+kc are in range; o is 8 x hd.
            unsafe {
                if self.half {
                    raw::i8g_fattn_pv_f16(sc.pt16.as_ptr(), (hs.vt.as_ptr() as *const u16).add(c0 * G), kc, hd, n_pad * G, sc.o.as_mut_ptr());
                } else {
                    raw::i8g_fattn_pv_f32(sc.pt32.as_ptr(), (hs.vt.as_ptr() as *const f32).add(c0 * G), kc, hd, n_pad * G, sc.o.as_mut_ptr());
                }
            }
            c0 += kc;
        }
        for qi in 0..G {
            let t = g * G + qi;
            if t >= n {
                break;
            }
            let inv = (1.0 / l[qi]) as f32;
            for d in 0..hd {
                sc.row[d] = sc.o[qi * hd + d] * inv;
            }
            out(t, &sc.row);
        }
    }
}

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

    fn rel_rmse(a: &[f32], b: &[f32]) -> f64 {
        let (mut e, mut r) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            e += (*x as f64 - *y as f64).powi(2);
            r += (*y as f64).powi(2);
        }
        (e / r.max(1e-300)).sqrt()
    }

    /// The reference block's attention (`reference::attn_stage`) against the float store on the
    /// same q/k/v. f32: float noise only (different summation order, exp ~1e-7); f16: operand
    /// rounding (2^-11 relative) and nothing structural.
    #[test]
    fn float_attention_matches_reference_attention() {
        let (heads, hd) = (3usize, 32usize);
        let h = heads * hd;
        for &(n, kc, logit_scale) in &[(37usize, 16usize, 1.0f32), (64, 64, 3.0), (29, 256, 6.0), (8, 8, 1.0)] {
            let mut rng = Rng(11 + n as u64);
            let q: Vec<f32> = (0..n * h).map(|_| rng.unit() * logit_scale).collect();
            let k: Vec<f32> = (0..n * h).map(|_| rng.unit() * logit_scale).collect();
            let v: Vec<f32> = (0..n * h).map(|_| rng.unit() * 4.0).collect();
            let dims = crate::types::Dims { hidden: h, heads, head_dim: hd, mlp: 4 * h };
            let want = crate::reference::attn_stage_for_tests(dims, &q, &k, &v, n);
            let inv_sqrt = 1.0 / (hd as f32).sqrt();
            for prec in [AttnPrec::F32, AttnPrec::F16] {
                let st = FloatStore::new(heads, hd, n, prec);
                let mut qs = vec![0f32; hd];
                for t in 0..n {
                    for hh in 0..heads {
                        let r = t * h + hh * hd..t * h + (hh + 1) * hd;
                        for (d, &x) in q[r.clone()].iter().enumerate() {
                            qs[d] = x * inv_sqrt;
                        }
                        unsafe {
                            st.put_q(hh, t, &qs);
                            st.put_k(hh, t, &k[r.clone()]);
                            st.put_v(hh, t, &v[r]);
                        }
                    }
                }
                let mut got = vec![f32::NAN; n * h];
                let mut sc = st.scratch(kc);
                for hh in 0..heads {
                    for g in 0..n.div_ceil(G) {
                        st.attend(hh, g, kc, &mut sc, |t, row| got[t * h + hh * hd..t * h + (hh + 1) * hd].copy_from_slice(row));
                    }
                }
                let e = rel_rmse(&got, &want);
                let lim = if prec == AttnPrec::F32 { 2e-6 } else { 2e-3 };
                eprintln!("attn_float n {n} kc {kc} logit_scale {logit_scale} {}: rel rmse vs reference {e:.3e}", prec.name());
                assert!(e < lim, "n {n} kc {kc} scale {logit_scale} {prec:?}: rel rmse {e:e} >= {lim:e}");
                // repeatable, and independent of the key chunk up to float noise
                let mut again = vec![0f32; n * h];
                let mut sc2 = st.scratch(8);
                for hh in 0..heads {
                    for g in 0..n.div_ceil(G) {
                        st.attend(hh, g, 8, &mut sc2, |t, row| again[t * h + hh * hd..t * h + (hh + 1) * hd].copy_from_slice(row));
                    }
                }
                assert!(rel_rmse(&again, &got) < if prec == AttnPrec::F32 { 2e-6 } else { 2e-3 });
            }
        }
    }

    #[test]
    fn parse_names() {
        for p in [AttnPrec::Int8, AttnPrec::F16, AttnPrec::F32] {
            assert_eq!(AttnPrec::parse(p.name()).unwrap(), p);
        }
        assert!(AttnPrec::parse("bf16").is_err());
        assert_eq!(AttnPrec::default(), AttnPrec::Int8);
    }
}
