//! O-70: fused single-pass decode / verify attention (flash-decoding style) over the f16 KV
//! cache, for the full-attention layers. Opt-in (`OXE_ATTN=fused` or
//! [`super::Engine::set_attn_fused`]); the tiled path stays the default.
//!
//! The tiled path ([`super::Engine`]'s `attn_core_tiled`) runs three steps with a barrier after
//! each: [1] all scores K^T q into an f32 matrix (positions split over the threads), [2] softmax
//! rows -> f16 probabilities (rows split), [3] V^T p (head dims split). The score matrix is written
//! by one set of threads and read by another (~3.8 KB per position per verify step).
//!
//! Here the work items are (KV head, position chunk) pairs with fixed chunk boundaries
//! (`ch` positions, a multiple of 32). One item computes the scores of its chunk for every query
//! column of the KV head (all query heads of the group and all lanes: each K row is read once),
//! a chunk-local softmax (max m, f16 weights exp(x - m), their sum l) and the chunk's V^T . p
//! (each V^T row segment read once for all columns), all while the chunk's scores and weights sit
//! in L1, and writes a small partial ([col][head_dim] f32 + m, l). After ONE barrier, every
//! output element is the rescaled sum of its chunk partials in chunk order. Results are
//! therefore independent of the thread count (bitwise), but not bitwise equal to the ggml FA-off
//! order: see `oxe_attn_fchunk_f16` in `oxe_ops.cpp` for the exact numerics.
//!
//! The same code runs inside the engine and in the model-free tests / micro-benchmark
//! (`tests/fattn.rs`, `tests/fattn_kbench.rs`).

use kernels_sys::ffi;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::MAX_LANES;

/// Default positions per work item (a multiple of 32): 128 KiB of K and V^T per KV head and the
/// scores / weights of 16 columns (16 KiB + 8 KiB) in L1.
pub const FUSED_CH_DEFAULT: usize = 256;
/// Smallest / largest accepted chunk (the partial buffers are sized for the smallest).
pub const FUSED_CH_MIN: usize = 128;
pub const FUSED_CH_MAX: usize = 4096;

/// A chunk size from a request: a multiple of 32 in [FUSED_CH_MIN, FUSED_CH_MAX].
pub fn chunk_from(req: usize) -> usize {
    (req.clamp(FUSED_CH_MIN, FUSED_CH_MAX) / 32) * 32
}

/// `OXE_ATTN_FUSED_CH` (default [`FUSED_CH_DEFAULT`]).
pub fn chunk_from_env() -> usize {
    chunk_from(std::env::var("OXE_ATTN_FUSED_CH").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(FUSED_CH_DEFAULT))
}

/// Partial-output floats for `n_ctx` positions (a multiple of 256) and up to `lanes` lanes.
pub fn part_len(n_ctx: usize, nkv: usize, group: usize, hd: usize, lanes: usize) -> usize {
    nkv * n_ctx.div_ceil(FUSED_CH_MIN) * lanes * group * hd
}

/// (m, l) floats for the same.
pub fn ml_len(n_ctx: usize, nkv: usize, group: usize, lanes: usize) -> usize {
    nkv * n_ctx.div_ceil(FUSED_CH_MIN) * 2 * lanes * group
}

/// Per-thread scratch (grown on first use).
#[derive(Default)]
pub struct FaScratch {
    s: Vec<f32>,
    p: Vec<u16>,
    nv: Vec<i32>,
    w: Vec<f32>,
}

/// One attention layer's fused attention for a run of `nl` lanes at positions pos0..pos0+nl
/// (lane i attends to positions <= pos0 + i). Raw pointers: the engine's buffers or a test's.
#[derive(Clone, Copy)]
pub struct Fa {
    pub hd: usize,
    pub nkv: usize,
    /// query heads per KV head
    pub group: usize,
    pub nl: usize,
    pub pos0: usize,
    /// positions per work item ([`chunk_from`])
    pub ch: usize,
    pub scale: f32,
    /// K row (kv head h, position p) at `k + (h * n_ctx + p) * hd`; rows >= pos0 are NOT read
    pub k: *const u16,
    pub n_ctx: usize,
    /// lane i's own K rows (position pos0 + i): kv head h at `k_local[i] + h * hd`
    pub k_local: [*const u16; MAX_LANES],
    /// V^T row (kv head h, dim d) at `vt + (h * hd + d) * v_stride`; read on [0, round_up(last+1, 32)),
    /// and must be finite up to n_ctx (the tail rule of the last chunk may read up to its end)
    pub vt: *const u16,
    pub v_stride: usize,
    /// query columns (f16) and sigmoid gates (f32), `hd` each, column (kvh, lane, g) at
    /// `kvh * cpk + lane * group + g`
    pub q16: *const u16,
    pub gate: *const f32,
    pub cpk: usize,
    /// partials: [`part_len`] / [`ml_len`] floats (shared by the threads)
    pub part: *mut f32,
    pub ml: *mut f32,
    /// output (lane ln, query head h, dim d) at `out + ln * out_ls + h * hd + d` (gated)
    pub out: *mut f32,
    pub out_ls: usize,
}
// SAFETY: plain pointers into buffers whose ownership rules the callers uphold (see `chunks`).
unsafe impl Send for Fa {}
unsafe impl Sync for Fa {}

impl Fa {
    #[inline]
    pub fn last(&self) -> usize {
        self.pos0 + self.nl - 1
    }

    #[inline]
    pub fn ncol(&self) -> usize {
        self.nl * self.group
    }

    /// Chunks per KV head of this run.
    #[inline]
    pub fn nch(&self) -> usize {
        ((self.last() + 1).div_ceil(32) * 32).div_ceil(self.ch)
    }

    /// Step 1 (before the barrier): this thread's (KV head, chunk) items.
    ///
    /// # Safety
    /// All pointers valid for the shapes above; K rows < pos0, the local rows, V^T rows up to
    /// the chunk end (min(chunk start + ch, n_ctx)), q16 and gates complete and not written during the call; the partial
    /// slots of this thread's items are written by no one else; `ch % 32 == 0`, nl <= MAX_LANES.
    pub unsafe fn chunks(&self, tid: usize, nth: usize, scr: &mut FaScratch) {
        self.chunks_stats(tid, nth, scr, None)
    }

    /// [`Self::chunks`] that also counts `OXE_ATTN_STATS` (O70-KEEP): per column, the 32-position
    /// blocks of the chunk entirely inside the column's valid prefix (total, all f16 weights exactly
    /// zero, all below 2^-10). The weights seen here are the chunk-local exp(x - m_chunk), the
    /// values a fused zero-block skip would test; since m_chunk <= the row max, a block that is zero
    /// here is zero in the normalised row too (the count is exact for "zero", and "< 2^-10" is
    /// relative to the chunk max). Totals equal the tiled path's (floor((pos + 1) / 32) per column).
    ///
    /// # Safety
    /// As [`Self::chunks`].
    pub unsafe fn chunks_stats(&self, tid: usize, nth: usize, scr: &mut FaScratch, stats: Option<&[AtomicU64; 3]>) {
        let (hd, group, nl, pos0) = (self.hd, self.group, self.nl, self.pos0);
        let ncol = self.ncol();
        let last = self.last();
        let n32 = (last + 1).div_ceil(32) * 32;
        let nch = self.nch();
        let items = self.nkv * nch;
        let ss = self.ch;
        if scr.s.len() < ncol * ss {
            scr.s.resize(ncol * ss, 0.0);
        }
        if scr.p.len() < ncol * ss {
            scr.p.resize(ncol * ss, 0);
        }
        if scr.nv.len() < ncol {
            scr.nv.resize(ncol, 0);
        }
        let s = scr.s.as_mut_ptr();
        for it in items * tid / nth..items * (tid + 1) / nth {
            let (kvh, ch) = (it / nch, it % nch);
            let s0 = ch * self.ch;
            let e = (s0 + self.ch).min(n32);
            let len = e - s0;
            let qc = self.q16.add(kvh * self.cpk * hd);
            // cache rows (< pos0): every lane attends to them
            let ce = e.min(pos0);
            if s0 < ce {
                ffi::oxe_dot_f16_tile(
                    hd as i32,
                    self.k.add((kvh * self.n_ctx + s0) * hd),
                    hd as i64,
                    (ce - s0) as i32,
                    qc,
                    hd as i64,
                    ncol as i32,
                    s,
                    1,
                    ss as i64,
                    std::ptr::null(),
                );
            }
            // this run's rows: row pos0+i is seen by lanes i..nl (their columns are contiguous)
            for p in s0.max(pos0)..e.min(last + 1) {
                let i = p - pos0;
                ffi::oxe_dot_f16_tile(
                    hd as i32,
                    self.k_local[i].add(kvh * hd),
                    hd as i64,
                    1,
                    qc.add(i * group * hd),
                    hd as i64,
                    ((nl - i) * group) as i32,
                    s.add(i * group * ss + (p - s0)),
                    1,
                    ss as i64,
                    std::ptr::null(),
                );
            }
            for (c, v) in scr.nv.iter_mut().enumerate().take(ncol) {
                *v = (pos0 + c / group + 1).saturating_sub(s0).min(len) as i32;
            }
            let idx = kvh * nch + ch;
            ffi::oxe_attn_fchunk_f16(
                s,
                ss as i64,
                ncol as i32,
                scr.nv.as_ptr(),
                len as i32,
                ((s0 + self.ch).min(self.n_ctx) - s0) as i32,
                self.scale,
                self.vt.add(kvh * hd * self.v_stride + s0),
                self.v_stride as i64,
                hd as i32,
                scr.p.as_mut_ptr(),
                self.part.add(idx * ncol * hd),
                self.ml.add(idx * 2 * ncol),
            );
            if let Some(st) = stats {
                // weights of column c at p[c * len ..], valid prefix nv[c] (a multiple of 32 except
                // in the column's last chunk; only whole blocks are counted)
                let (mut full, mut z, mut t) = (0u64, 0u64, 0u64);
                for c in 0..ncol {
                    let nv = scr.nv[c].max(0) as usize;
                    let w = &scr.p[c * len..c * len + nv];
                    for b in w.chunks_exact(32) {
                        let m = b.iter().map(|&x| x & 0x7fff).max().unwrap_or(0);
                        full += 1;
                        z += (m == 0) as u64;
                        t += (m < 0x1400) as u64; // < 2^-10
                    }
                }
                st[0].fetch_add(full, Relaxed);
                st[1].fetch_add(z, Relaxed);
                st[2].fetch_add(t, Relaxed);
            }
        }
    }

    /// Step 2 (after the barrier): outputs of this thread's (KV head, dim) rows, every column.
    ///
    /// # Safety
    /// As [`Self::chunks`]; every item's partial complete; the output elements of this thread's
    /// rows are written by no one else.
    pub unsafe fn merge(&self, tid: usize, nth: usize, scr: &mut FaScratch) {
        let (hd, group) = (self.hd, self.group);
        let ncol = self.ncol();
        let nch = self.nch();
        if scr.w.len() < nch.div_ceil(4) * 4 {
            scr.w.resize(nch.div_ceil(4) * 4, 0.0);
        }
        let rows = self.nkv * hd;
        let my = rows * tid / nth..rows * (tid + 1) / nth;
        for kvh in 0..self.nkv {
            let a = my.start.max(kvh * hd).min((kvh + 1) * hd) - kvh * hd;
            let b = my.end.max(kvh * hd).min((kvh + 1) * hd) - kvh * hd;
            if a >= b {
                continue;
            }
            let base = kvh * nch;
            for col in 0..ncol {
                let (ln, g) = (col / group, col % group);
                let h = kvh * group + g;
                let ml = self.ml.add(base * 2 * ncol + col);
                ffi::oxe_attn_fmerge_f32(
                    nch as i32,
                    self.part.add(base * ncol * hd + col * hd),
                    (ncol * hd) as i64,
                    ml,
                    ml.add(ncol),
                    (2 * ncol) as i64,
                    a as i32,
                    b as i32,
                    self.gate.add((kvh * self.cpk + col) * hd),
                    self.out.add(ln * self.out_ls + h * hd),
                    scr.w.as_mut_ptr(),
                );
            }
        }
    }
}
