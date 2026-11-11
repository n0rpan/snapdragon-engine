//! The whole klein DiT on the int8 SMMLA path (M1): 5 double-stream + 20 single-stream blocks,
//! embedders/modulation/final layer in f32 (they are a rounding error of the cost). Implements
//! [`DitBackend`]. The heavy stages (linears, attention) are work-shared ([`Share`]): fixed work
//! items, claimed by whichever thread is free, so no thread waits for a slower one; results do
//! not depend on who runs what.
//!
//! Built from the same stage functions for both block kinds:
//! - `norm_mod_quant`: LayerNorm + modulation -> packed int8 (scale per token per chunk);
//! - `run_units`: a linear whose weight rows were re-ordered at load into units that are complete
//!   for their consumer (one head of q / k / v, or a block of SiLU-gate rows with their partners),
//!   each unit's f32 tile consumed in cache by a sink;
//! - `attention`: int8 flash attention over an `AttnStore` (steps: query groups of a head);
//! - `run_plain`: a linear with rows in file order (steps: row block x token batch), sink adds the
//!   gated result to the residual stream in place.
//!
//! Double block: per stream qkv (units) -> joint attention -> proj (plain) -> mlp.0 (units,
//! SiLU gate) -> mlp.2 (plain). Single block: linear1 (units: q, k, v and MLP) -> attention ->
//! linear2 (plain). The single-block code in `fast.rs` is the frozen M0 prototype (ticket 015) and
//! is deliberately not touched; this module duplicates its kernels' call patterns.

use crate::attn_float::{AttnPrec, FloatStore};
use crate::fast::{exp_fast, FastConfig};
use crate::model::{DitBackend, ModelInput};
use int8_gemm::{gemm, gemm_rows1, raw, sinks, AlignedBytes, Order, Packed, G};
use sched::Pool;
use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

/// Stage clock of the block stages (nanoseconds, summed over a forward). `*_busy` is the sum of
/// the threads' own work times and `*_busy_max` the sum over runs of the slowest thread's time:
/// `busy_max * threads / busy` is the load imbalance of the stage (1.0 = perfect).
#[derive(Default)]
pub struct StageClock {
    quant: AtomicU64,
    alloc: AtomicU64,
    units: AtomicU64,
    units_busy: AtomicU64,
    units_busy_max: AtomicU64,
    plain: AtomicU64,
    plain_busy: AtomicU64,
    plain_busy_max: AtomicU64,
}

fn clk_add(a: &AtomicU64, t0: Instant) {
    a.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
}

fn clk_take(a: &AtomicU64) -> f64 {
    a.swap(0, Relaxed) as f64 * 1e-9
}

// ------------------------------------------------------------------------------------------
// weights
// ------------------------------------------------------------------------------------------

/// Where the model's tensors come from (a GGUF mapping, or memory in tests).
pub trait WeightSource {
    /// A Q8_0 matrix: (bytes, rows = out features, k = in features).
    fn q8(&self, name: &str) -> Result<(&[u8], usize, usize), String>;
    /// A float tensor (F32, F16 or BF16 in the file) as f32: (values, rows, k); 1-D tensors have rows = 1.
    fn f32(&self, name: &str) -> Result<(Vec<f32>, usize, usize), String>;
    fn has(&self, name: &str) -> bool;
    /// Storage type name in the file ("Q8_0", "BF16", "F32", ...), when known (the sd.cpp
    /// arithmetic-parity reference rounds a matmul's activations by it).
    fn dtype(&self, _name: &str) -> Option<String> {
        None
    }
}

/// GGUF-backed source.
pub struct GgufSource<'a>(pub &'a gguf::Gguf);

impl WeightSource for GgufSource<'_> {
    fn q8(&self, name: &str) -> Result<(&[u8], usize, usize), String> {
        let t = self.0.header.tensor(name).ok_or_else(|| format!("{name}: missing"))?;
        if t.ty.name() != "Q8_0" || t.shape.len() != 2 {
            return Err(format!("{name}: expected a 2-D Q8_0 tensor, found {} {:?}", t.ty.name(), t.shape));
        }
        Ok((self.0.tensor_data(t), t.shape[1] as usize, t.shape[0] as usize))
    }
    fn f32(&self, name: &str) -> Result<(Vec<f32>, usize, usize), String> {
        let t = self.0.header.tensor(name).ok_or_else(|| format!("{name}: missing"))?;
        let b = self.0.tensor_data(t);
        let v: Vec<f32> = match t.ty.name() {
            "F32" => b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => b.chunks_exact(2).map(|c| int8_gemm::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            other => return Err(format!("{name}: unsupported float type {other}")),
        };
        let (k, rows) = match t.shape.len() {
            1 => (t.shape[0] as usize, 1),
            2 => (t.shape[0] as usize, t.shape[1] as usize),
            _ => return Err(format!("{name}: unexpected shape {:?}", t.shape)),
        };
        Ok((v, rows, k))
    }
    fn has(&self, name: &str) -> bool {
        self.0.header.tensor(name).is_some()
    }
    fn dtype(&self, name: &str) -> Option<String> {
        self.0.header.tensor(name).map(|t| t.ty.name().to_string())
    }
}

/// In-memory source (tests, synthetic models).
#[derive(Default)]
pub struct MemSource {
    pub q8: HashMap<String, (Vec<u8>, usize, usize)>,
    pub f: HashMap<String, (Vec<f32>, usize, usize)>,
}

impl WeightSource for MemSource {
    fn q8(&self, name: &str) -> Result<(&[u8], usize, usize), String> {
        self.q8.get(name).map(|(b, r, k)| (&b[..], *r, *k)).ok_or_else(|| format!("{name}: missing"))
    }
    fn f32(&self, name: &str) -> Result<(Vec<f32>, usize, usize), String> {
        self.f.get(name).cloned().ok_or_else(|| format!("{name}: missing"))
    }
    fn has(&self, name: &str) -> bool {
        self.q8.contains_key(name) || self.f.contains_key(name)
    }
}

// ------------------------------------------------------------------------------------------
// plumbing shared by the stages
// ------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
// SAFETY: only used for writes to disjoint regions: every work step is handed out once (see each use).
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

fn even_split(n: usize, parts: usize) -> Vec<Range<usize>> {
    (0..parts).map(|p| n * p / parts..n * (p + 1) / parts).collect()
}

/// Work-shared two-level loop for one parallel stage: `n_outer` items (a weight block), each with
/// `n_inner` steps (token batches). A thread claims whole items while there are any (so a weight
/// block normally stays in one core's L2), then helps to finish the items of other threads step
/// by step. So no thread waits at the barrier for more than about one step of another thread:
/// the split does not depend on a cost model, and a core that is slower for a while (another
/// process, frequency) does not hold the stage up. Which thread runs a step never changes a result.
struct Share {
    next_outer: AtomicUsize,
    inner: Vec<sched::Padded<AtomicUsize>>,
    n_inner: usize,
}

impl Share {
    fn new(n_outer: usize, n_inner: usize) -> Share {
        Share { next_outer: AtomicUsize::new(0), inner: (0..n_outer).map(|_| sched::Padded(AtomicUsize::new(0))).collect(), n_inner }
    }
    #[inline]
    fn drain<F: FnMut(usize, usize)>(&self, o: usize, body: &mut F) {
        // the cheap load keeps finished items from being written to by every helper
        while self.inner[o].0.load(Relaxed) < self.n_inner {
            let i = self.inner[o].0.fetch_add(1, Relaxed);
            if i >= self.n_inner {
                break;
            }
            body(o, i);
        }
    }
    /// Call from every thread of the stage: takes unclaimed items, then helps with the rest.
    fn work<F: FnMut(usize, usize)>(&self, mut body: F) {
        let n = self.inner.len();
        loop {
            let o = self.next_outer.fetch_add(1, Relaxed);
            if o >= n {
                break;
            }
            self.drain(o, &mut body);
        }
        for o in 0..n {
            self.drain(o, &mut body);
        }
    }
}

/// Cache-blocking knobs of the linear stages. None of them changes a result (the GEMM's float
/// order is fixed per output element; the knobs only choose which tiles are computed when).
/// Measured 2026-11-10 at klein shapes, variants interleaved [M]: the defaults are the best of batch 2/4/8 and
/// plain rows 64/128; merging units into 384-row passes and software prefetch of the next batch were both slower.
#[derive(Clone, Copy, Debug)]
pub struct LinTune {
    /// Token groups (of 8) per batch of a unit linear (qkv, mlp.0, linear1).
    pub unit_batch: usize,
    /// Token groups per batch of a plain linear (proj, mlp.2, linear2).
    pub plain_batch: usize,
    /// Rows per weight block of a plain linear.
    pub plain_rows: usize,
}

impl LinTune {
    fn from_cfg(cfg: &FastConfig) -> LinTune {
        LinTune { unit_batch: cfg.batch_groups, plain_batch: cfg.batch_groups, plain_rows: cfg.lin2_row_block }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitKind {
    Q(usize),
    K(usize),
    V(usize),
    Mlp(usize),
}

struct Unit {
    kind: UnitKind,
    rg: Range<usize>,
}

/// A linear with unit-ordered rows.
struct UnitLinear {
    w: Packed,
    units: Vec<Unit>,
    /// outputs per MLP unit (0 when the linear has no MLP part)
    mlp_unit: usize,
}

/// Row layout of a unit linear in the SOURCE matrix.
struct Layout {
    /// first row of q (k follows at +hidden, v at +2*hidden), if the linear has q/k/v
    qkv: Option<usize>,
    /// (first gate row, first partner row, mlp width), if the linear has a gated MLP part
    mlp: Option<(usize, usize, usize)>,
}

#[allow(clippy::too_many_arguments)]
fn build_unit_linear(src: &[u8], rows: usize, k: usize, chunk_in: usize, chunk_out: usize, heads: usize, hd: usize, layout: &Layout) -> UnitLinear {
    let hidden = heads * hd;
    let mut perm: Vec<u32> = Vec::new();
    let mut units = Vec::new();
    let mut push = |kind: UnitKind, rows: &mut dyn Iterator<Item = usize>, perm: &mut Vec<u32>| {
        let g0 = perm.len() / G;
        perm.extend(rows.map(|r| r as u32));
        assert!(perm.len() % G == 0, "unit rows must be a multiple of 8");
        units.push(Unit { kind, rg: g0..perm.len() / G });
    };
    if let Some(q0) = layout.qkv {
        for head in 0..heads {
            push(UnitKind::Q(head), &mut (q0 + head * hd..q0 + (head + 1) * hd), &mut perm);
            push(UnitKind::K(head), &mut (q0 + hidden + head * hd..q0 + hidden + (head + 1) * hd), &mut perm);
            push(UnitKind::V(head), &mut (q0 + 2 * hidden + head * hd..q0 + 2 * hidden + (head + 1) * hd), &mut perm);
        }
    }
    let mut mlp_unit = 0;
    if let Some((gate0, lin0, mlp)) = layout.mlp {
        mlp_unit = chunk_out;
        while mlp_unit < hd && mlp % (mlp_unit * 2) == 0 {
            mlp_unit *= 2;
        }
        assert!(mlp % mlp_unit == 0 && mlp_unit % G == 0);
        for u in 0..mlp / mlp_unit {
            let gate = gate0 + u * mlp_unit..gate0 + (u + 1) * mlp_unit;
            let lin = lin0 + u * mlp_unit..lin0 + (u + 1) * mlp_unit;
            push(UnitKind::Mlp(u), &mut gate.chain(lin), &mut perm);
        }
    }
    assert_eq!(perm.len(), rows, "layout must cover every row once");
    let w = Packed::from_q8_0(src, rows, k, chunk_in, Some(&perm));
    UnitLinear { w, units, mlp_unit }
}

/// Run a unit linear over the `n` tokens of `x`. `sink(unit kind, token, row, scratch)` is called
/// exactly once per (unit, real token), on whichever thread runs that step; `row` is that token's f32 output
/// for the unit's rows; `scratch` has `scratch_len` elements.
fn run_units<F>(pool: &Pool, clk: &StageClock, tune: &LinTune, x: &Packed, ul: &UnitLinear, n: usize, scratch_len: usize, sink: F)
where
    F: Fn(UnitKind, usize, &[f32], &mut [f32]) + Sync,
{
    let groups = n.div_ceil(G);
    let bg = tune.unit_batch.max(1);
    let t_run = Instant::now();
    let busy_max = AtomicU64::new(0);
    let share = Share::new(ul.units.len(), groups.div_ceil(bg));
    pool.run(|_ctx| {
        let t_me = Instant::now();
        let mut block_buf: Vec<f32> = Vec::new();
        let mut vals = vec![0.0f32; scratch_len];
        share.work(|u, b| {
            let unit = &ul.units[u];
            let nr = unit.rg.len() * G;
            let (g0, g1) = (b * bg, ((b + 1) * bg).min(groups));
            let ntok = (g1 - g0) * G;
            if block_buf.len() < ntok * nr {
                block_buf.resize(ntok * nr, 0.0);
            }
            let block = &mut block_buf[..ntok * nr];
            gemm(x, &ul.w, g0..g1, unit.rg.clone(), block, nr, Order::MulAdd, false);
            for ti in 0..ntok {
                let t = g0 * G + ti;
                if t >= n {
                    break;
                }
                sink(unit.kind, t, &block[ti * nr..(ti + 1) * nr], &mut vals);
            }
        });
        let d = t_me.elapsed().as_nanos() as u64;
        clk.units_busy.fetch_add(d, Relaxed);
        busy_max.fetch_max(d, Relaxed);
    });
    clk.units_busy_max.fetch_add(busy_max.load(Relaxed), Relaxed);
    clk_add(&clk.units, t_run);
}

/// A linear with rows in file order.
struct PlainLinear {
    w: Packed,
}

fn build_plain_linear(src: &[u8], rows: usize, k: usize, chunk: usize) -> PlainLinear {
    assert!(rows % G == 0);
    PlainLinear { w: Packed::from_q8_0(src, rows, k, chunk, None) }
}

/// Run a plain linear. `sink(token, first column, z)` gets the f32 outputs `z` of columns
/// `first column .. + z.len()` for one real token; every (row block, token) is delivered exactly once.
fn run_plain<F>(pool: &Pool, clk: &StageClock, tune: &LinTune, x: &Packed, pl: &PlainLinear, n: usize, sink: F)
where
    F: Fn(usize, usize, &[f32]) + Sync,
{
    let groups = n.div_ceil(G);
    let rb = (tune.plain_rows / G).max(1);
    let bg = tune.plain_batch.max(1);
    let row_groups = pl.w.groups();
    let t_run = Instant::now();
    let busy_max = AtomicU64::new(0);
    let share = Share::new(row_groups.div_ceil(rb), groups.div_ceil(bg));
    pool.run(|_ctx| {
        let t_me = Instant::now();
        let mut block_buf: Vec<f32> = Vec::new();
        share.work(|o, b| {
            let (r0, r1) = (o * rb, ((o + 1) * rb).min(row_groups));
            let nr = (r1 - r0) * G;
            let (g0, g1) = (b * bg, ((b + 1) * bg).min(groups));
            let ntok = (g1 - g0) * G;
            if block_buf.len() < ntok * nr {
                block_buf.resize(ntok * nr, 0.0);
            }
            let block = &mut block_buf[..ntok * nr];
            gemm(x, &pl.w, g0..g1, r0..r1, block, nr, Order::MulAdd, false);
            for ti in 0..ntok {
                let t = g0 * G + ti;
                if t >= n {
                    break;
                }
                sink(t, r0 * G, &block[ti * nr..(ti + 1) * nr]);
            }
        });
        let d = t_me.elapsed().as_nanos() as u64;
        clk.plain_busy.fetch_add(d, Relaxed);
        busy_max.fetch_max(d, Relaxed);
    });
    clk.plain_busy_max.fetch_add(busy_max.load(Relaxed), Relaxed);
    clk_add(&clk.plain, t_run);
}

/// LayerNorm (no affine, eps 1e-6) + modulation of the `n` rows of `x` (`[n][h]`), quantised into `dst`.
fn norm_mod_quant(pool: &Pool, clk: &StageClock, x: &[f32], n: usize, h: usize, shift: &[f32], scale: &[f32], dst: &Packed) {
    let t_run = Instant::now();
    let plan = even_split(n.div_ceil(G), pool.nthreads());
    pool.run(|ctx| {
        let mut buf = vec![0.0f32; h];
        for g in plan[ctx.tid].clone() {
            for t in g * G..((g + 1) * G).min(n) {
                let row = &x[t * h..(t + 1) * h];
                layer_norm_modulate(row, shift, scale, &mut buf);
                // SAFETY: token t belongs to exactly one thread.
                unsafe { dst.quantize_unsync(t, 0, &buf) };
            }
        }
    });
    clk_add(&clk.quant, t_run);
}

fn layer_norm_modulate(x: &[f32], shift: &[f32], scale: &[f32], out: &mut [f32]) {
    let h = x.len();
    let (mut s, mut ss) = (0.0f64, 0.0f64);
    for &v in x {
        s += v as f64;
        ss += v as f64 * v as f64;
    }
    let mean = s / h as f64;
    let var = (ss / h as f64 - mean * mean).max(0.0);
    let (mean, inv) = (mean as f32, (1.0 / (var + 1e-6).sqrt()) as f32);
    for i in 0..h {
        let y = (x[i] - mean) * inv;
        out[i] = (y + y * scale[i]) + shift[i];
    }
}

/// Quantise rows `src[(off + t) * width ..][..width]`, t in 0..n, into columns `k0..k0+width` of `dst`.
fn quant_rows(pool: &Pool, clk: &StageClock, src: &[f32], off: usize, width: usize, n: usize, dst: &Packed, k0: usize) {
    let t_run = Instant::now();
    let plan = even_split(n.div_ceil(G), pool.nthreads());
    pool.run(|ctx| {
        for g in plan[ctx.tid].clone() {
            for t in g * G..((g + 1) * G).min(n) {
                // SAFETY: token t belongs to exactly one thread; no one else touches these columns now.
                unsafe { dst.quantize_unsync(t, k0, &src[(off + t) * width..(off + t + 1) * width]) };
            }
        }
    });
    clk_add(&clk.quant, t_run);
}

// ------------------------------------------------------------------------------------------
// attention
// ------------------------------------------------------------------------------------------

struct HeadStore {
    q: Packed,
    k: Packed,
    /// v transposed: `[dim group][key step][64]`
    vt: AlignedBytes,
    /// value scale per key
    sv: AlignedBytes,
}

/// q, k, v of all heads for `n` tokens: int8, or (F-IMG-ATTN) f16/f32 in `float` with `heads` empty.
struct AttnStore {
    heads: Vec<HeadStore>,
    float: Option<FloatStore>,
    n: usize,
    n_pad: usize,
    hd: usize,
    nsteps: usize,
}

impl AttnStore {
    fn new(heads: usize, hd: usize, n: usize, prec: AttnPrec) -> AttnStore {
        let n_pad = n.div_ceil(G) * G;
        let nsteps = n_pad / G;
        let float = if prec.is_float() { Some(FloatStore::new(heads, hd, n, prec)) } else { None };
        let heads = (0..if float.is_some() { 0 } else { heads })
            .map(|_| HeadStore { q: Packed::zeroed(n, hd, hd), k: Packed::zeroed(n, hd, hd), vt: AlignedBytes::zeroed((hd / G) * nsteps * 64), sv: AlignedBytes::zeroed(n_pad * 4) })
            .collect();
        AttnStore { heads, float, n, n_pad, hd, nsteps }
    }

    /// RMS norm, scale, RoPE, then int8 for one (head, token) of q or k. For q the 1/sqrt(d) of
    /// the logits is folded in.
    ///
    /// # Safety
    /// (head, q|k, token) must be written by one thread only, with no concurrent reader.
    unsafe fn put_qk(&self, head: usize, t: usize, is_q: bool, row: &[f32], scale: &[f32], pe_t: &[f32], vals: &mut [f32]) {
        let hd = self.hd;
        let mut ms = 0.0f32;
        for &v in row {
            ms += v * v;
        }
        let mut r = 1.0 / (ms / hd as f32 + 1e-6).sqrt();
        if is_q {
            r /= (hd as f32).sqrt();
        }
        // per pair: a = row*r*scale, b likewise; vals = [[p0, p1], [p2, p3]] . [a, b] (NEON, bitwise the scalar rule)
        sinks::rope_scale(row, r, scale, pe_t, &mut vals[..hd]);
        if let Some(fs) = &self.float {
            if is_q {
                fs.put_q(head, t, &vals[..hd]);
            } else {
                fs.put_k(head, t, &vals[..hd]);
            }
            return;
        }
        let dst = if is_q { &self.heads[head].q } else { &self.heads[head].k };
        dst.quantize_unsync(t, 0, &vals[..hd]);
    }

    /// int8 for one (head, token) of v, stored transposed.
    ///
    /// # Safety
    /// (head, token) must be written by one thread only, with no concurrent reader.
    unsafe fn put_v(&self, head: usize, t: usize, row: &[f32]) {
        if let Some(fs) = &self.float {
            fs.put_v(head, t, row);
            return;
        }
        let hs = &self.heads[head];
        let vt = hs.vt.as_ptr() as *mut u8;
        let mut codes = [0i8; 512];
        if self.hd <= codes.len() {
            // scale = max|row| / 127, code = round_half_away(row / scale) clamped to +-127 (NEON, same rule as below)
            *(hs.sv.as_ptr() as *mut f32).add(t) = sinks::quant_chunk(&row[..self.hd], &mut codes[..self.hd], 8);
            for dg in 0..self.hd / G {
                let dst = vt.add((dg * self.nsteps + t / G) * 64 + t % G);
                for i in 0..G {
                    *dst.add(i * G) = codes[dg * G + i] as u8;
                }
            }
            return;
        }
        let mut amax = 0.0f32;
        for &v in row {
            amax = amax.max(v.abs());
        }
        let (scale, inv) = if amax > 0.0 && amax.is_finite() { (amax / 127.0, 127.0 / amax) } else { (0.0, 0.0) };
        *(hs.sv.as_ptr() as *mut f32).add(t) = scale;
        for d in 0..self.hd {
            let code = (row[d] * inv).round().clamp(-127.0, 127.0) as i8;
            *vt.add(((d / G) * self.nsteps + t / G) * 64 + (d % G) * G + t % G) = code as u8;
        }
    }
}

/// int8 flash attention; writes `out[t * h + head * hd + d]` for all tokens (`h = heads * hd`).
fn attention(pool: &Pool, store: &AttnStore, key_chunk: usize, out: &mut [f32]) {
    if let Some(fs) = &store.float {
        return attention_float(pool, fs, store.hd, key_chunk, out);
    }
    let (n, n_pad, hd, nsteps) = (store.n, store.n_pad, store.hd, store.nsteps);
    let h = store.heads.len() * hd;
    assert_eq!(out.len(), n * h);
    let groups = n_pad / G;
    let out_ptr = SyncPtr(out.as_mut_ptr());
    let kc_max = key_chunk.min(n_pad);
    // a step = QB query groups of one head (work-shared like the linears: see `Share`)
    const QB: usize = 4;
    let share = Share::new(store.heads.len(), groups.div_ceil(QB));
    pool.run(|_ctx| {
        let mut scores = vec![0.0f32; G * kc_max];
        let mut pcodes = vec![0u8; (kc_max / G) * 64];
        let mut o = vec![0.0f32; G * hd];
        let ones = [1.0f32; G];
        share.work(|head, qb| {
            let hs = &store.heads[head];
            // SAFETY: the producing stage has finished (barrier); sv is read-only from here.
            let sv = unsafe { std::slice::from_raw_parts(hs.sv.as_ptr() as *const f32, n_pad) };
            for g in qb * QB..((qb + 1) * QB).min(groups) {
                let mut m = [f32::NEG_INFINITY; G];
                let mut l = [0.0f64; G];
                o.iter_mut().for_each(|v| *v = 0.0);
                let mut c0 = 0;
                while c0 < n_pad {
                    let kc = key_chunk.min(n_pad - c0);
                    gemm_rows1(&hs.q, &hs.k, g, c0 / G..(c0 + kc) / G, &mut scores[..(G - 1) * kc_max + kc], kc_max);
                    let mut sp = [0.0f32; G];
                    for qi in 0..G {
                        let s = &mut scores[qi * kc_max..qi * kc_max + kc];
                        if c0 + kc > n {
                            for key in n.saturating_sub(c0)..kc {
                                s[key] = f32::NEG_INFINITY; // padding keys
                            }
                        }
                        // SAFETY: s holds kc floats, kc is a multiple of 8.
                        let cmax = unsafe { raw::i8g_row_max(s.as_ptr(), kc) };
                        if cmax > m[qi] {
                            let f = if m[qi] == f32::NEG_INFINITY { 0.0 } else { exp_fast(m[qi] - cmax) };
                            for v in &mut o[qi * hd..(qi + 1) * hd] {
                                *v *= f;
                            }
                            l[qi] *= f as f64;
                            m[qi] = cmax;
                        }
                        let (mut sum, mut scale) = (0.0f32, 0.0f32);
                        // SAFETY: s and sv[c0..] hold kc floats; pcodes holds kc/8 steps of 64 bytes; qi < 8.
                        unsafe { raw::i8g_softmax_pack(s.as_mut_ptr(), kc, m[qi], sv.as_ptr().add(c0), pcodes.as_mut_ptr(), qi, &mut sum, &mut scale) };
                        l[qi] += sum as f64;
                        sp[qi] = scale;
                    }
                    for dg in 0..hd / G {
                        // SAFETY: pcodes holds kc/8 steps; vt holds (hd/8) x nsteps steps and
                        // c0/8 + kc/8 <= nsteps; o is 8 x hd with stride hd.
                        unsafe {
                            raw::i8g_utile8x8(pcodes.as_ptr(), (hs.vt.as_ptr() as *const i8).add((dg * nsteps + c0 / G) * 64), kc / G, 1, sp.as_ptr(), ones.as_ptr(), o.as_mut_ptr().add(dg * G), hd as isize, 1);
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
                    // SAFETY: (head, query group) is delivered exactly once: these columns of row t are ours.
                    let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.get().add(t * h + head * hd), hd) };
                    for d in 0..hd {
                        dst[d] = o[qi * hd + d] * inv;
                    }
                }
            }
        });
    });
}

/// f16/f32 flash attention (F-IMG-ATTN), work-shared like [`attention`]: a step is QB query groups of one head.
fn attention_float(pool: &Pool, fs: &FloatStore, hd: usize, key_chunk: usize, out: &mut [f32]) {
    let n = fs.n();
    let h = fs.heads() * hd;
    assert_eq!(out.len(), n * h);
    let groups = n.div_ceil(G);
    let out_ptr = SyncPtr(out.as_mut_ptr());
    const QB: usize = 4;
    let share = Share::new(fs.heads(), groups.div_ceil(QB));
    pool.run(|_ctx| {
        let mut sc = fs.scratch(key_chunk);
        share.work(|head, qb| {
            for g in qb * QB..((qb + 1) * QB).min(groups) {
                fs.attend(head, g, key_chunk, &mut sc, |t, row| {
                    // SAFETY: (head, query group) is delivered exactly once: these columns of row t are ours.
                    let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr.get().add(t * h + head * hd), hd) };
                    dst.copy_from_slice(row);
                });
            }
        });
    });
}

// ------------------------------------------------------------------------------------------
// small f32 pieces
// ------------------------------------------------------------------------------------------

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let main = n - n % 8;
    let mut acc = [0.0f32; 8];
    for (ca, cb) in a[..main].chunks_exact(8).zip(b[..main].chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += ca[l] * cb[l];
        }
    }
    let mut tail = 0.0f32;
    for i in main..n {
        tail += a[i] * b[i];
    }
    acc.iter().sum::<f32>() + tail
}

/// Float matrix `[rows][k]`.
struct Mat {
    w: Vec<f32>,
    rows: usize,
    k: usize,
}

impl Mat {
    fn load(src: &dyn WeightSource, name: &str) -> Result<Mat, String> {
        let (w, rows, k) = src.f32(name)?;
        Ok(Mat { w, rows, k })
    }
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.k);
        (0..self.rows).map(|r| dot(&self.w[r * self.k..(r + 1) * self.k], x)).collect()
    }
    /// `[n][k] -> [n][rows]`, parallel over tokens.
    fn apply_tokens(&self, pool: &Pool, x: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(x.len(), n * self.k);
        let mut out = vec![0.0f32; n * self.rows];
        let ptr = SyncPtr(out.as_mut_ptr());
        let plan = even_split(n, pool.nthreads());
        pool.run(|ctx| {
            for t in plan[ctx.tid].clone() {
                let xt = &x[t * self.k..(t + 1) * self.k];
                // SAFETY: row t of out is written by one thread.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(t * self.rows), self.rows) };
                for r in 0..self.rows {
                    dst[r] = dot(&self.w[r * self.k..(r + 1) * self.k], xt);
                }
            }
        });
        out
    }
}

/// ggml_timestep_embedding of `t * time_factor`: `[cos(t f_j) for j < half] ++ [sin(t f_j)]`, `f_j = exp(-ln(max_period) j / half)`.
pub fn timestep_embedding(t: f32, dim: usize, max_period: f32, time_factor: f32) -> Vec<f32> {
    let half = dim / 2;
    let ts = t * time_factor;
    let mut e = vec![0.0f32; dim];
    for j in 0..half {
        let freq = (-(max_period.ln()) * j as f32 / half as f32).exp();
        let arg = ts * freq;
        e[j] = arg.cos();
        e[j + half] = arg.sin();
    }
    e
}

fn silu_exact(a: f32) -> f32 {
    a / (1.0 + (-a).exp())
}

// ------------------------------------------------------------------------------------------
// the model
// ------------------------------------------------------------------------------------------

struct Stream {
    qkv: UnitLinear,
    proj: PlainLinear,
    mlp0: UnitLinear,
    mlp2: PlainLinear,
    q_scale: Vec<f32>,
    k_scale: Vec<f32>,
}

struct Double {
    img: Stream,
    txt: Stream,
}

struct Single {
    lin1: UnitLinear,
    lin2: PlainLinear,
    q_scale: Vec<f32>,
    k_scale: Vec<f32>,
}

/// Wall time of the last forward by category, seconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelTimes {
    pub embed: f64,
    pub double_blocks: f64,
    pub single_blocks: f64,
    pub final_layer: f64,
    pub attention: f64,
    pub total: f64,
    /// LayerNorm/modulation/re-quantisation stages
    pub quant: f64,
    /// buffer allocation and zeroing
    pub alloc: f64,
    /// unit linears (qkv, mlp.0, linear1) including their sinks: wall, thread-busy sum, slowest-thread sum
    pub units: f64,
    pub units_busy: f64,
    pub units_busy_max: f64,
    /// plain linears (proj, mlp.2, linear2) including their sinks
    pub plain: f64,
    pub plain_busy: f64,
    pub plain_busy_max: f64,
}

/// The whole DiT, int8 fast path.
pub struct FastModel<'p> {
    pool: &'p Pool,
    cfg: FastConfig,
    hidden: usize,
    heads: usize,
    hd: usize,
    mlp: usize,
    img_in: Mat,
    txt_in: Mat,
    time_in: Mat,
    time_out: Mat,
    mod_img: Mat,
    mod_txt: Mat,
    mod_single: Mat,
    final_ada: Mat,
    final_lin: Mat,
    doubles: Vec<Double>,
    singles: Vec<Single>,
    txt_cache: Option<(Vec<f32>, Vec<f32>)>,
    clk: StageClock,
    /// Cache-blocking knobs of the linears (defaults from `FastConfig`; results do not depend on them).
    pub tune: LinTune,
    pub last_times: ModelTimes,
}

fn count_blocks(src: &dyn WeightSource, prefix: &str, suffix: &str) -> usize {
    let mut n = 0;
    while src.has(&format!("{prefix}.{n}.{suffix}")) {
        n += 1;
    }
    n
}

impl<'p> FastModel<'p> {
    /// Map a klein GGUF and pack every block (int8 codes + chunk scales stay resident).
    pub fn open(gguf_path: &Path, pool: &'p Pool, cfg: FastConfig) -> Result<FastModel<'p>, String> {
        let g = gguf::Gguf::open(gguf_path).map_err(|e| e.0)?;
        FastModel::from_source(&GgufSource(&g), pool, cfg)
    }

    pub fn from_source(src: &dyn WeightSource, pool: &'p Pool, cfg: FastConfig) -> Result<FastModel<'p>, String> {
        // dims from the first block that exists (klein always has both kinds; tests may have one)
        let (hidden, hd, mlp) = if src.has("single_blocks.0.linear1.weight") {
            let (_, _, hd) = src.f32("single_blocks.0.norm.query_norm.scale")?;
            let (_, rows1, hidden) = src.q8("single_blocks.0.linear1.weight")?;
            let (_, _, k2) = src.q8("single_blocks.0.linear2.weight")?;
            if k2 <= hidden || rows1 != 3 * hidden + 2 * (k2 - hidden) {
                return Err(format!("unexpected single block shapes: hidden {hidden}, linear1 rows {rows1}, linear2 in {k2}"));
            }
            (hidden, hd, k2 - hidden)
        } else {
            let (_, _, hd) = src.f32("double_blocks.0.img_attn.norm.query_norm.scale")?;
            let (_, _, hidden) = src.q8("double_blocks.0.img_attn.qkv.weight")?;
            let (_, _, mlp) = src.q8("double_blocks.0.img_mlp.2.weight")?;
            (hidden, hd, mlp)
        };
        if hd == 0 || hidden % hd != 0 {
            return Err(format!("hidden {hidden} is not a multiple of head_dim {hd}"));
        }
        let heads = hidden / hd;
        if hd % 8 != 0 || hidden % cfg.chunk1 != 0 || hidden % cfg.chunk2 != 0 || mlp % cfg.chunk2 != 0 || cfg.chunk1 % 32 != 0 || cfg.chunk2 % 32 != 0 {
            return Err(format!("chunk sizes {}/{} do not fit hidden {hidden}, mlp {mlp}, head_dim {hd}", cfg.chunk1, cfg.chunk2));
        }
        let scale = |name: String| -> Result<Vec<f32>, String> {
            let (v, _, k) = src.f32(&name)?;
            if k != hd {
                return Err(format!("{name}: length {k}, expected {hd}"));
            }
            Ok(v)
        };
        let unit = |name: String, layout: Layout| -> Result<UnitLinear, String> {
            let (b, rows, k) = src.q8(&name)?;
            if k != hidden {
                return Err(format!("{name}: in features {k}, expected {hidden}"));
            }
            Ok(build_unit_linear(b, rows, k, cfg.chunk1, cfg.chunk2, heads, hd, &layout))
        };
        let plain = |name: String, want_rows: usize, want_k: usize| -> Result<PlainLinear, String> {
            let (b, rows, k) = src.q8(&name)?;
            if rows != want_rows || k != want_k {
                return Err(format!("{name}: shape [{rows}][{k}], expected [{want_rows}][{want_k}]"));
            }
            Ok(build_plain_linear(b, rows, k, cfg.chunk2))
        };
        let n_double = count_blocks(src, "double_blocks", "img_attn.qkv.weight");
        let n_single = count_blocks(src, "single_blocks", "linear1.weight");
        let mut doubles = Vec::with_capacity(n_double);
        for i in 0..n_double {
            let stream = |s: &str| -> Result<Stream, String> {
                let p = format!("double_blocks.{i}.{s}");
                Ok(Stream {
                    qkv: unit(format!("{p}_attn.qkv.weight"), Layout { qkv: Some(0), mlp: None })?,
                    proj: plain(format!("{p}_attn.proj.weight"), hidden, hidden)?,
                    mlp0: unit(format!("{p}_mlp.0.weight"), Layout { qkv: None, mlp: Some((0, mlp, mlp)) })?,
                    mlp2: plain(format!("{p}_mlp.2.weight"), hidden, mlp)?,
                    q_scale: scale(format!("{p}_attn.norm.query_norm.scale"))?,
                    k_scale: scale(format!("{p}_attn.norm.key_norm.scale"))?,
                })
            };
            doubles.push(Double { img: stream("img")?, txt: stream("txt")? });
        }
        let mut singles = Vec::with_capacity(n_single);
        for i in 0..n_single {
            let p = format!("single_blocks.{i}");
            singles.push(Single {
                lin1: unit(format!("{p}.linear1.weight"), Layout { qkv: Some(0), mlp: Some((3 * hidden, 3 * hidden + mlp, mlp)) })?,
                lin2: plain(format!("{p}.linear2.weight"), hidden, hidden + mlp)?,
                q_scale: scale(format!("{p}.norm.query_norm.scale"))?,
                k_scale: scale(format!("{p}.norm.key_norm.scale"))?,
            });
        }
        let m = FastModel {
            pool,
            cfg,
            hidden,
            heads,
            hd,
            mlp,
            img_in: Mat::load(src, "img_in.weight")?,
            txt_in: Mat::load(src, "txt_in.weight")?,
            time_in: Mat::load(src, "time_in.in_layer.weight")?,
            time_out: Mat::load(src, "time_in.out_layer.weight")?,
            mod_img: Mat::load(src, "double_stream_modulation_img.lin.weight")?,
            mod_txt: Mat::load(src, "double_stream_modulation_txt.lin.weight")?,
            mod_single: Mat::load(src, "single_stream_modulation.lin.weight")?,
            final_ada: Mat::load(src, "final_layer.adaLN_modulation.1.weight")?,
            final_lin: Mat::load(src, "final_layer.linear.weight")?,
            doubles,
            singles,
            txt_cache: None,
            clk: StageClock::default(),
            tune: LinTune::from_cfg(&cfg),
            last_times: ModelTimes::default(),
        };
        let h = hidden;
        for (name, mat, rows, k) in [
            ("img_in", &m.img_in, h, m.img_in.k),
            ("txt_in", &m.txt_in, h, m.txt_in.k),
            ("time_in.in_layer", &m.time_in, h, 256),
            ("time_in.out_layer", &m.time_out, h, h),
            ("double_stream_modulation_img", &m.mod_img, 6 * h, h),
            ("double_stream_modulation_txt", &m.mod_txt, 6 * h, h),
            ("single_stream_modulation", &m.mod_single, 3 * h, h),
            ("final_layer.adaLN_modulation.1", &m.final_ada, 2 * h, h),
            ("final_layer.linear", &m.final_lin, m.final_lin.rows, h),
        ] {
            if mat.rows != rows || mat.k != k {
                return Err(format!("{name}: shape [{}][{}], expected [{rows}][{k}]", mat.rows, mat.k));
            }
        }
        Ok(m)
    }

    pub fn n_blocks(&self) -> (usize, usize) {
        (self.doubles.len(), self.singles.len())
    }

    /// One stream's qkv into the joint attention store at token offset `off`.
    #[allow(clippy::too_many_arguments)]
    fn stream_qkv(&self, st: &Stream, x: &[f32], n_s: usize, off: usize, shift: &[f32], scale: &[f32], pe: &[f32], store: &AttnStore) {
        let (h, hd) = (self.hidden, self.hd);
        let x1 = Packed::zeroed(n_s, h, self.cfg.chunk1);
        norm_mod_quant(self.pool, &self.clk, x, n_s, h, shift, scale, &x1);
        let pe_w = (hd / 2) * 4;
        run_units(self.pool, &self.clk, &self.tune, &x1, &st.qkv, n_s, hd, |kind, t, row, vals| {
            let tt = off + t;
            // SAFETY: every (unit, token) is delivered exactly once (`Share`), and units write disjoint cells.
            unsafe {
                match kind {
                    UnitKind::Q(head) => store.put_qk(head, tt, true, row, &st.q_scale, &pe[tt * pe_w..(tt + 1) * pe_w], vals),
                    UnitKind::K(head) => store.put_qk(head, tt, false, row, &st.k_scale, &pe[tt * pe_w..(tt + 1) * pe_w], vals),
                    UnitKind::V(head) => store.put_v(head, tt, row),
                    UnitKind::Mlp(_) => unreachable!("qkv linear has no MLP units"),
                }
            }
        });
    }

    /// `x += gate * W x_q` for a plain linear, in place on the residual stream `x` (`[n][h]`).
    fn plain_residual(&self, pl: &PlainLinear, xq: &Packed, n_s: usize, gate: &[f32], x: &mut [f32]) {
        let h = self.hidden;
        let ptr = SyncPtr(x.as_mut_ptr());
        run_plain(self.pool, &self.clk, &self.tune, xq, pl, n_s, |t, c0, z| {
            // SAFETY: (row block, token) is delivered exactly once, so these columns of row t are ours; row t is real.
            let dst = unsafe { std::slice::from_raw_parts_mut(ptr.get().add(t * h + c0), z.len()) };
            for j in 0..z.len() {
                dst[j] += gate[c0 + j] * z[j];
            }
        });
    }

    /// Second half of a double-block stream: proj of the attention rows, then the gated MLP.
    #[allow(clippy::too_many_arguments)]
    fn stream_post(&self, st: &Stream, x: &mut [f32], n_s: usize, off: usize, attn: &[f32], m: &[f32]) {
        let (h, mlp) = (self.hidden, self.mlp);
        let (gate1, shift2, scale2, gate2) = (&m[2 * h..3 * h], &m[3 * h..4 * h], &m[4 * h..5 * h], &m[5 * h..6 * h]);
        let xa = Packed::zeroed(n_s, h, self.cfg.chunk2);
        quant_rows(self.pool, &self.clk, attn, off, h, n_s, &xa, 0);
        self.plain_residual(&st.proj, &xa, n_s, gate1, x);
        let xm = Packed::zeroed(n_s, h, self.cfg.chunk1);
        norm_mod_quant(self.pool, &self.clk, x, n_s, h, shift2, scale2, &xm);
        let xh = Packed::zeroed(n_s, mlp, self.cfg.chunk2);
        let u_len = st.mlp0.mlp_unit;
        run_units(self.pool, &self.clk, &self.tune, &xm, &st.mlp0, n_s, u_len, |kind, t, row, vals| {
            if let UnitKind::Mlp(u) = kind {
                sinks::silu_gate(&row[..2 * u_len], &mut vals[..u_len]);
                // SAFETY: (MLP unit u, token t) is delivered exactly once; whole chunks of columns u*U.. of token t.
                unsafe { xh.quantize_unsync(t, u * u_len, &vals[..u_len]) };
            }
        });
        self.plain_residual(&st.mlp2, &xh, n_s, gate2, x);
    }

    fn single_block(&self, sb: &Single, x: &mut [f32], n: usize, pe: &[f32], m: &[f32], attn_secs: &mut f64) {
        let (h, hd, mlp) = (self.hidden, self.hd, self.mlp);
        let (shift, scale, gate) = (&m[..h], &m[h..2 * h], &m[2 * h..3 * h]);
        let t_al = Instant::now();
        let x1 = Packed::zeroed(n, h, self.cfg.chunk1);
        let store = AttnStore::new(self.heads, hd, n, self.cfg.attn_prec);
        let x2 = Packed::zeroed(n, h + mlp, self.cfg.chunk2);
        let mut attn = vec![0.0f32; n * h];
        clk_add(&self.clk.alloc, t_al);
        norm_mod_quant(self.pool, &self.clk, x, n, h, shift, scale, &x1);
        let u_len = sb.lin1.mlp_unit;
        let pe_w = (hd / 2) * 4;
        run_units(self.pool, &self.clk, &self.tune, &x1, &sb.lin1, n, hd.max(u_len), |kind, t, row, vals| {
            // SAFETY: every (unit, token) is delivered exactly once (`Share`); MLP unit u
            // covers whole chunks of columns h + u*U.. of token t.
            unsafe {
                match kind {
                    UnitKind::Q(head) => store.put_qk(head, t, true, row, &sb.q_scale, &pe[t * pe_w..(t + 1) * pe_w], vals),
                    UnitKind::K(head) => store.put_qk(head, t, false, row, &sb.k_scale, &pe[t * pe_w..(t + 1) * pe_w], vals),
                    UnitKind::V(head) => store.put_v(head, t, row),
                    UnitKind::Mlp(u) => {
                        sinks::silu_gate(&row[..2 * u_len], &mut vals[..u_len]);
                        x2.quantize_unsync(t, h + u * u_len, &vals[..u_len]);
                    }
                }
            }
        });
        let t0 = Instant::now();
        attention(self.pool, &store, self.cfg.key_chunk, &mut attn);
        *attn_secs += t0.elapsed().as_secs_f64();
        let t_al = Instant::now();
        drop(store);
        drop(x1);
        clk_add(&self.clk.alloc, t_al);
        quant_rows(self.pool, &self.clk, &attn, 0, h, n, &x2, 0);
        self.plain_residual(&sb.lin2, &x2, n, gate, x);
    }
}

impl DitBackend for FastModel<'_> {
    fn name(&self) -> &'static str {
        "cpu-int8-smmla"
    }

    fn forward(&mut self, input: &ModelInput) -> Vec<f32> {
        let (h, hd) = (self.hidden, self.hd);
        let (n_img, n_txt) = (input.n_img, input.n_txt);
        let n = n_img + n_txt;
        assert_eq!(input.img.len(), n_img * self.img_in.k, "img shape");
        assert_eq!(input.txt.len(), n_txt * self.txt_in.k, "txt shape");
        assert_eq!(input.pe.len(), n * (hd / 2) * 4, "pe shape");
        let mut times = ModelTimes::default();
        let t_all = Instant::now();

        // embedders and modulation (f32)
        let t0 = Instant::now();
        let mut img = self.img_in.apply_tokens(self.pool, &input.img, n_img);
        let cached = matches!(&self.txt_cache, Some((key, _)) if key == &input.txt);
        if !cached {
            let v = self.txt_in.apply_tokens(self.pool, &input.txt, n_txt);
            self.txt_cache = Some((input.txt.clone(), v));
        }
        let mut txt = self.txt_cache.as_ref().expect("set above").1.clone();
        let emb = timestep_embedding(input.timestep, 256, 10000.0, 1000.0);
        let mut vec = self.time_in.apply(&emb);
        vec.iter_mut().for_each(|v| *v = silu_exact(*v));
        let vec = self.time_out.apply(&vec);
        let sv: Vec<f32> = vec.iter().map(|&v| silu_exact(v)).collect();
        let m_img = self.mod_img.apply(&sv);
        let m_txt = self.mod_txt.apply(&sv);
        let m_single = self.mod_single.apply(&sv);
        times.embed = t0.elapsed().as_secs_f64();

        // double blocks
        let t0 = Instant::now();
        for db in &self.doubles {
            let store = AttnStore::new(self.heads, hd, n, self.cfg.attn_prec);
            self.stream_qkv(&db.txt, &txt, n_txt, 0, &m_txt[..h], &m_txt[h..2 * h], &input.pe, &store);
            self.stream_qkv(&db.img, &img, n_img, n_txt, &m_img[..h], &m_img[h..2 * h], &input.pe, &store);
            let mut attn = vec![0.0f32; n * h];
            let ta = Instant::now();
            attention(self.pool, &store, self.cfg.key_chunk, &mut attn);
            times.attention += ta.elapsed().as_secs_f64();
            drop(store);
            self.stream_post(&db.img, &mut img, n_img, n_txt, &attn, &m_img);
            self.stream_post(&db.txt, &mut txt, n_txt, 0, &attn, &m_txt);
        }
        times.double_blocks = t0.elapsed().as_secs_f64();

        // single blocks on concat(txt, img)
        let t0 = Instant::now();
        let mut x = txt;
        x.extend_from_slice(&img);
        drop(img);
        for sb in &self.singles {
            self.single_block(sb, &mut x, n, &input.pe, &m_single, &mut times.attention);
        }
        times.single_blocks = t0.elapsed().as_secs_f64();

        // final layer on the image tokens (f32)
        let t0 = Instant::now();
        let ada = self.final_ada.apply(&sv);
        let (shift, scale) = (&ada[..h], &ada[h..2 * h]);
        let mut xm = vec![0.0f32; n_img * h];
        for t in 0..n_img {
            layer_norm_modulate(&x[(n_txt + t) * h..(n_txt + t + 1) * h], shift, scale, &mut xm[t * h..(t + 1) * h]);
        }
        let out = self.final_lin.apply_tokens(self.pool, &xm, n_img);
        times.final_layer = t0.elapsed().as_secs_f64();
        times.total = t_all.elapsed().as_secs_f64();
        let c = &self.clk;
        times.quant = clk_take(&c.quant);
        times.alloc = clk_take(&c.alloc);
        times.units = clk_take(&c.units);
        times.units_busy = clk_take(&c.units_busy);
        times.units_busy_max = clk_take(&c.units_busy_max);
        times.plain = clk_take(&c.plain);
        times.plain_busy = clk_take(&c.plain_busy);
        times.plain_busy_max = clk_take(&c.plain_busy_max);
        self.last_times = times;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_layout() {
        let e = timestep_embedding(0.5, 8, 10000.0, 1000.0);
        // half = 4, f_j = 10000^(-j/4)
        for j in 0..4 {
            let arg = 500.0f64 * (10000.0f64).powf(-(j as f64) / 4.0);
            assert!((e[j] as f64 - arg.cos()).abs() < 2e-3, "cos {j}");
            assert!((e[j + 4] as f64 - arg.sin()).abs() < 2e-3, "sin {j}");
        }
    }

    #[test]
    fn splits_cover_everything() {
        for (n, p) in [(24usize, 12usize), (5, 12), (384, 10), (0, 3)] {
            let s = even_split(n, p);
            assert_eq!(s.len(), p);
            assert_eq!(s.iter().map(|r| r.len()).sum::<usize>(), n);
            for w in s.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
        }
    }
}
