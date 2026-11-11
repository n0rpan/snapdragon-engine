//! int8 SMMLA path of one single-stream block on a static per-thread plan.
//!
//! Five parallel stages, each one `Pool::run` (one barrier each), no dynamic scheduling:
//!
//! 1. `norm_quant`: per token LayerNorm + modulation, quantised straight into the packed int8
//!    operand of linear1 (scale per token per `chunk1`).
//! 2. `linear1`: the weight rows are re-ordered at pack time into UNITS that are complete for
//!    their consumer: one head of q, of k, of v (`head_dim` rows), or one MLP unit (`U` gate rows
//!    followed by their `U` partner rows). A thread owns a fixed list of units; per unit it walks
//!    all tokens in small batches and the batch's f32 block is consumed while still in cache:
//!    q/k -> RMS norm, scale, RoPE, int8 per (token, head); v -> int8 per (token, head) stored
//!    transposed for P.V; MLP -> SiLU gate, int8 into linear2's operand. The 510 MB f32 output of
//!    linear1 (at 4608 tokens) never exists.
//! 3. `attention`: a thread owns whole heads. Flash-style: per 8 queries, per key chunk:
//!    q.k^T by SMMLA -> exp with a running max -> probabilities (times the value scale) as u8 per
//!    (query, key chunk) -> P.V by USMMLA into f32 accumulators.
//! 4. `requant`: attention output -> int8 into linear2's operand.
//! 5. `linear2`: a thread owns a fixed range of output rows; the sink adds the gated result to x.
//!
//! Numbers: weights are the file's Q8_0 codes (verbatim at chunk 32, re-quantised per chunk
//! otherwise); every float fix-up follows `int8_gemm`'s fixed order, so the output does not depend
//! on the thread count.

use crate::attn_float::{AttnPrec, FloatStore};
use crate::backend::{SingleBlockBackend, StageTimes};
use crate::types::{BlockInput, BlockOutput, Dims};
use int8_gemm::{gemm, gemm_blocked, raw, AlignedBytes, Order, Packed, G};
use sched::Pool;
use std::ops::Range;
use std::time::Instant;

/// Tunables of the fast path. All sizes in elements.
#[derive(Clone, Copy, Debug)]
pub struct FastConfig {
    /// Scale chunk of linear1's operands (multiple of 32, divides `hidden`).
    pub chunk1: usize,
    /// Scale chunk of linear2's operands (multiple of 32, divides `hidden` and `mlp`).
    pub chunk2: usize,
    /// Token groups (of 8) per sink batch in the linears.
    pub batch_groups: usize,
    /// Rows per cache block of linear2 (multiple of 8).
    pub lin2_row_block: usize,
    /// Keys per attention chunk (multiple of 8).
    pub key_chunk: usize,
    /// Attention arithmetic (F-IMG-ATTN). `Int8` (default) is the original path, bitwise unchanged;
    /// `F16`/`F32` keep the int8 linears and run q.k^T and softmax.v in float (`attn_float`).
    pub attn_prec: AttnPrec,
}

impl Default for FastConfig {
    fn default() -> Self {
        FastConfig { chunk1: 256, chunk2: 256, batch_groups: 4, lin2_row_block: 64, key_chunk: 256, attn_prec: AttnPrec::Int8 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitKind {
    Q(usize),
    K(usize),
    V(usize),
    /// MLP unit index (outputs `u*U .. (u+1)*U`)
    Mlp(usize),
}

#[derive(Clone, Debug)]
struct Unit {
    kind: UnitKind,
    /// row groups of the packed linear1 matrix
    rg: Range<usize>,
}

/// Per-head attention operands.
struct HeadStore {
    q: Packed,
    k: Packed,
    /// v transposed: `[dim group][key step][64]` (8 dims x 8 keys per step), codes i8
    vt: AlignedBytes,
    /// value scale per key (f32)
    sv: AlignedBytes,
}

#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
// SAFETY: used only for writes to disjoint regions under the static plan (see each use).
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    /// By-value accessor: closures must capture the whole wrapper, not its raw-pointer field.
    #[inline(always)]
    fn get(self) -> *mut T {
        self.0
    }
}

/// Contiguous split of `costs` into `parts` ranges of nearly equal total cost.
fn partition(costs: &[usize], parts: usize) -> Vec<Range<usize>> {
    let total: usize = costs.iter().sum();
    let mut out = Vec::with_capacity(parts);
    let (mut start, mut acc) = (0usize, 0usize);
    for p in 0..parts {
        let target = total * (p + 1) / parts;
        let mut end = start;
        while end < costs.len() && (acc + costs[end] <= target || (end == start && p + 1 == parts)) {
            acc += costs[end];
            end += 1;
        }
        if p + 1 == parts {
            end = costs.len();
        }
        out.push(start..end);
        start = end;
    }
    out
}

/// exp(x) for x clamped to [-87, 87]; relative error about 1e-7 (Cephes expf polynomial).
/// Branch-free so the loops that call it vectorise.
#[inline(always)]
pub fn exp_fast(x: f32) -> f32 {
    let x = x.clamp(-87.0, 87.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_2e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 1.666_666_5e-1) * r + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

#[inline(always)]
fn silu(a: f32) -> f32 {
    a / (1.0 + exp_fast(-a))
}

/// A prepared block: packed weights and the static plan for `pool`'s thread count.
pub struct FastBlock<'p> {
    dims: Dims,
    cfg: FastConfig,
    pool: &'p Pool,
    w1: Packed,
    w2: Packed,
    q_scale: Vec<f32>,
    k_scale: Vec<f32>,
    /// MLP outputs per unit
    mlp_unit: usize,
    units: Vec<Unit>,
    /// per thread: range into `units`
    unit_plan: Vec<Range<usize>>,
    /// per thread: heads
    head_plan: Vec<Range<usize>>,
    /// per thread: row groups of linear2
    lin2_plan: Vec<Range<usize>>,
}

impl<'p> FastBlock<'p> {
    /// `w1_q8`: Q8_0 bytes of linear1 (`lin1_out` rows of `hidden`), `w2_q8`: linear2 (`hidden` rows of `lin2_in`).
    pub fn new(dims: Dims, w1_q8: &[u8], w2_q8: &[u8], q_scale: &[f32], k_scale: &[f32], cfg: FastConfig, pool: &'p Pool) -> FastBlock<'p> {
        dims.check();
        let (h, hd, mlp) = (dims.hidden, dims.head_dim, dims.mlp);
        assert!(hd % 8 == 0 && h % 32 == 0 && mlp % 32 == 0);
        assert!(cfg.chunk1 % 32 == 0 && h % cfg.chunk1 == 0, "chunk1 must be a multiple of 32 dividing hidden");
        assert!(cfg.chunk2 % 32 == 0 && h % cfg.chunk2 == 0 && mlp % cfg.chunk2 == 0, "chunk2 must be a multiple of 32 dividing hidden and mlp");
        assert!(cfg.key_chunk % 8 == 0 && cfg.key_chunk > 0 && cfg.lin2_row_block % 8 == 0 && cfg.lin2_row_block > 0 && cfg.batch_groups > 0);
        assert_eq!(q_scale.len(), hd);
        assert_eq!(k_scale.len(), hd);
        // MLP unit: a multiple of chunk2 (so a unit can be quantised on its own), about head_dim or more.
        let mut mlp_unit = cfg.chunk2;
        while mlp_unit < hd && mlp % (mlp_unit * 2) == 0 {
            mlp_unit *= 2;
        }
        assert!(mlp % mlp_unit == 0 && mlp_unit % 8 == 0);

        // Row order of the packed linear1 and its units.
        let mut perm: Vec<u32> = Vec::with_capacity(dims.lin1_out());
        let mut units = Vec::new();
        let mut push = |kind: UnitKind, rows: &mut dyn Iterator<Item = usize>, perm: &mut Vec<u32>| {
            let g0 = perm.len() / G;
            perm.extend(rows.map(|r| r as u32));
            assert!(perm.len() % G == 0);
            units.push(Unit { kind, rg: g0..perm.len() / G });
        };
        for head in 0..dims.heads {
            push(UnitKind::Q(head), &mut (head * hd..(head + 1) * hd), &mut perm);
            push(UnitKind::K(head), &mut (h + head * hd..h + (head + 1) * hd), &mut perm);
            push(UnitKind::V(head), &mut (2 * h + head * hd..2 * h + (head + 1) * hd), &mut perm);
        }
        for u in 0..mlp / mlp_unit {
            let gate = 3 * h + u * mlp_unit..3 * h + (u + 1) * mlp_unit;
            let lin = 3 * h + mlp + u * mlp_unit..3 * h + mlp + (u + 1) * mlp_unit;
            push(UnitKind::Mlp(u), &mut gate.chain(lin), &mut perm);
        }
        assert_eq!(perm.len(), dims.lin1_out());
        let w1 = Packed::from_q8_0(w1_q8, dims.lin1_out(), h, cfg.chunk1, Some(&perm));
        let w2 = Packed::from_q8_0(w2_q8, h, dims.lin2_in(), cfg.chunk2, None);

        let nt = pool.nthreads();
        let unit_costs: Vec<usize> = units.iter().map(|u| u.rg.len()).collect();
        let unit_plan = partition(&unit_costs, nt);
        let head_plan = partition(&vec![1; dims.heads], nt);
        let lin2_plan = partition(&vec![1; h / G], nt);
        FastBlock { dims, cfg, pool, w1, w2, q_scale: q_scale.to_vec(), k_scale: k_scale.to_vec(), mlp_unit, units, unit_plan, head_plan, lin2_plan }
    }

    pub fn config(&self) -> FastConfig {
        self.cfg
    }

    /// Bytes of packed weights (codes + scales).
    pub fn weight_bytes(&self) -> usize {
        self.w1.data().len() + self.w1.scales().len() * 4 + self.w2.data().len() + self.w2.scales().len() * 4
    }
}

impl SingleBlockBackend for FastBlock<'_> {
    fn name(&self) -> &'static str {
        "cpu-int8-smmla"
    }

    fn forward(&mut self, input: &BlockInput, want_intermediates: bool) -> (BlockOutput, StageTimes) {
        let dims = self.dims;
        let (h, hd, mlp, heads) = (dims.hidden, dims.head_dim, dims.mlp, dims.heads);
        let n = input.n_tokens;
        assert!(n > 0);
        assert_eq!(input.x.len(), n * h);
        assert_eq!(input.pe.len(), n * (hd / 2) * 4);
        assert!(input.shift.len() == h && input.scale.len() == h && input.gate.len() == h);
        let cfg = self.cfg;
        let nt = self.pool.nthreads();
        let groups = n.div_ceil(G);
        let n_pad = groups * G;
        let tok_plan = partition(&vec![1; groups], nt);
        let mut times = StageTimes::default();
        let t_all = Instant::now();

        // ---- stage 1: LayerNorm + modulation -> int8 ----
        let x1 = Packed::zeroed(n, h, cfg.chunk1);
        let t0 = Instant::now();
        self.pool.run(|ctx| {
            let mut buf = vec![0.0f32; h];
            for g in tok_plan[ctx.tid].clone() {
                for t in g * G..((g + 1) * G).min(n) {
                    let x = &input.x[t * h..(t + 1) * h];
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
                        buf[i] = (y + y * input.scale[i]) + input.shift[i];
                    }
                    // SAFETY: token t belongs to exactly one thread (tok_plan is a partition).
                    unsafe { x1.quantize_unsync(t, 0, &buf) };
                }
            }
        });
        times.norm_quant = t0.elapsed().as_secs_f64();

        // ---- stage 2: linear1 with fused sinks ----
        let nsteps_keys = n_pad / G;
        let float = cfg.attn_prec.is_float();
        let fstore = if float { Some(FloatStore::new(heads, hd, n, cfg.attn_prec)) } else { None };
        let store: Vec<HeadStore> = (0..if float { 0 } else { heads })
            .map(|_| HeadStore {
                q: Packed::zeroed(n, hd, hd),
                k: Packed::zeroed(n, hd, hd),
                vt: AlignedBytes::zeroed((hd / G) * nsteps_keys * 64),
                sv: AlignedBytes::zeroed(n_pad * 4),
            })
            .collect();
        let x2 = Packed::zeroed(n, dims.lin2_in(), cfg.chunk2);
        let mut mlp_out = if want_intermediates { vec![0.0f32; n * mlp] } else { Vec::new() };
        let mlp_ptr = SyncPtr(mlp_out.as_mut_ptr());
        let inv_sqrt_hd = 1.0 / (hd as f32).sqrt();
        let mlp_unit = self.mlp_unit;
        let (w1, units, unit_plan, q_scale, k_scale) = (&self.w1, &self.units, &self.unit_plan, &self.q_scale, &self.k_scale);
        let fstore_ref = fstore.as_ref();
        let t0 = Instant::now();
        self.pool.run(|ctx| {
            let mut scratch = Vec::new();
            let mut vals = vec![0.0f32; hd.max(mlp_unit)];
            for unit in &units[unit_plan[ctx.tid].clone()] {
                let nr = unit.rg.len() * G;
                gemm_blocked(&x1, w1, 0..groups, unit.rg.clone(), cfg.batch_groups, Order::MulAdd, &mut scratch, |t0, ntok, block| {
                    for ti in 0..ntok {
                        let t = t0 + ti;
                        if t >= n {
                            break;
                        }
                        let row = &block[ti * nr..(ti + 1) * nr];
                        match unit.kind {
                            UnitKind::Q(head) | UnitKind::K(head) => {
                                let is_q = matches!(unit.kind, UnitKind::Q(_));
                                let sc = if is_q { q_scale } else { k_scale };
                                let mut ms = 0.0f32;
                                for &v in row {
                                    ms += v * v;
                                }
                                let mut r = 1.0 / (ms / hd as f32 + 1e-6).sqrt();
                                if is_q {
                                    r *= inv_sqrt_hd; // the 1/sqrt(d) of the logits lives in q
                                }
                                let pe = &input.pe[t * (hd / 2) * 4..(t + 1) * (hd / 2) * 4];
                                for i in 0..hd / 2 {
                                    let a = row[2 * i] * r * sc[2 * i];
                                    let b = row[2 * i + 1] * r * sc[2 * i + 1];
                                    let p = &pe[i * 4..i * 4 + 4];
                                    vals[2 * i] = p[0] * a + p[1] * b;
                                    vals[2 * i + 1] = p[2] * a + p[3] * b;
                                }
                                // SAFETY: (head, q|k) is one unit, owned by this thread; token t is
                                // visited once per unit.
                                if let Some(fs) = fstore_ref {
                                    unsafe {
                                        if is_q {
                                            fs.put_q(head, t, &vals[..hd])
                                        } else {
                                            fs.put_k(head, t, &vals[..hd])
                                        }
                                    };
                                    continue;
                                }
                                let dst = if is_q { &store[head].q } else { &store[head].k };
                                unsafe { dst.quantize_unsync(t, 0, &vals[..hd]) };
                            }
                            UnitKind::V(head) => {
                                if let Some(fs) = fstore_ref {
                                    // SAFETY: unit (head, v) is owned by this thread; token t once.
                                    unsafe { fs.put_v(head, t, row) };
                                    continue;
                                }
                                let mut amax = 0.0f32;
                                for &v in row {
                                    amax = amax.max(v.abs());
                                }
                                let (scale, inv) = if amax > 0.0 && amax.is_finite() { (amax / 127.0, 127.0 / amax) } else { (0.0, 0.0) };
                                let hs = &store[head];
                                // SAFETY: unit (head, v) is owned by this thread; writes go to key
                                // column t of this head only.
                                unsafe {
                                    *(hs.sv.as_ptr() as *mut f32).add(t) = scale;
                                    let vt = hs.vt.as_ptr() as *mut u8;
                                    for d in 0..hd {
                                        let code = (row[d] * inv).round().clamp(-127.0, 127.0) as i8;
                                        *vt.add(((d / G) * nsteps_keys + t / G) * 64 + (d % G) * G + t % G) = code as u8;
                                    }
                                }
                            }
                            UnitKind::Mlp(u) => {
                                for j in 0..mlp_unit {
                                    vals[j] = silu(row[j]) * row[mlp_unit + j];
                                }
                                // SAFETY: MLP unit u is owned by this thread; it covers columns
                                // h + u*U .. h + (u+1)*U of token t (whole chunks of chunk2).
                                unsafe {
                                    x2.quantize_unsync(t, h + u * mlp_unit, &vals[..mlp_unit]);
                                    if want_intermediates {
                                        std::ptr::copy_nonoverlapping(vals.as_ptr(), mlp_ptr.get().add(t * mlp + u * mlp_unit), mlp_unit);
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });
        times.linear1 = t0.elapsed().as_secs_f64();

        // ---- stage 3: attention ----
        let mut attn = vec![0.0f32; n * h];
        let attn_ptr = SyncPtr(attn.as_mut_ptr());
        let head_plan = &self.head_plan;
        let kc_max = cfg.key_chunk.min(n_pad);
        let t0 = Instant::now();
        if let Some(fs) = fstore_ref {
            self.pool.run(|ctx| {
                let mut sc = fs.scratch(cfg.key_chunk);
                for head in head_plan[ctx.tid].clone() {
                    for g in 0..groups {
                        fs.attend(head, g, cfg.key_chunk, &mut sc, |t, row| {
                            // SAFETY: columns head*hd..(head+1)*hd of row t are written by this thread only.
                            let dst = unsafe { std::slice::from_raw_parts_mut(attn_ptr.get().add(t * h + head * hd), hd) };
                            dst.copy_from_slice(row);
                        });
                    }
                }
            });
        } else {
        self.pool.run(|ctx| {
            let mut scores = vec![0.0f32; G * kc_max];
            let mut pcodes = vec![0u8; (kc_max / G) * 64];
            let mut o = vec![0.0f32; G * hd];
                let ones = [1.0f32; G];
            for head in head_plan[ctx.tid].clone() {
                let hs = &store[head];
                // SAFETY: stage 2 has finished (barrier); from here sv is read-only.
                let sv = unsafe { std::slice::from_raw_parts(hs.sv.as_ptr() as *const f32, n_pad) };
                for g in 0..groups {
                    let mut m = [f32::NEG_INFINITY; G];
                    let mut l = [0.0f64; G];
                    o.iter_mut().for_each(|v| *v = 0.0);
                    let mut c0 = 0;
                    while c0 < n_pad {
                        let kc = cfg.key_chunk.min(n_pad - c0);
                        gemm(&hs.q, &hs.k, g..g + 1, c0 / G..(c0 + kc) / G, &mut scores[..(G - 1) * kc_max + kc], kc_max, Order::MulAdd, false);
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
                            // SAFETY: pcodes holds kc/8 steps; vt holds (hd/8) x nsteps_keys steps and
                            // c0/8 + kc/8 <= nsteps_keys; o is 8 x hd with stride hd.
                            unsafe {
                                raw::i8g_utile8x8(
                                    pcodes.as_ptr(),
                                    (hs.vt.as_ptr() as *const i8).add((dg * nsteps_keys + c0 / G) * 64),
                                    kc / G,
                                    1,
                                    sp.as_ptr(),
                                    ones.as_ptr(),
                                    o.as_mut_ptr().add(dg * G),
                                    hd as isize,
                                    1,
                                );
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
                        // SAFETY: columns head*hd..(head+1)*hd of row t are written by this thread only.
                        let dst = unsafe { std::slice::from_raw_parts_mut(attn_ptr.get().add(t * h + head * hd), hd) };
                        for d in 0..hd {
                            dst[d] = o[qi * hd + d] * inv;
                        }
                    }
                }
            }
        });
        }
        times.attention = t0.elapsed().as_secs_f64();
        drop(fstore);

        // ---- stage 4: attention output -> int8 operand of linear2 ----
        let t0 = Instant::now();
        let attn_ref = &attn;
        self.pool.run(|ctx| {
            for g in tok_plan[ctx.tid].clone() {
                for t in g * G..((g + 1) * G).min(n) {
                    // SAFETY: token t belongs to one thread; columns 0..h are not touched by anyone else now.
                    unsafe { x2.quantize_unsync(t, 0, &attn_ref[t * h..(t + 1) * h]) };
                }
            }
        });
        times.requant = t0.elapsed().as_secs_f64();

        // ---- stage 5: linear2 + gate + residual ----
        let mut out = vec![0.0f32; n * h];
        let mut z = if want_intermediates { vec![0.0f32; n * h] } else { Vec::new() };
        let (out_ptr, z_ptr) = (SyncPtr(out.as_mut_ptr()), SyncPtr(z.as_mut_ptr()));
        let (w2, lin2_plan) = (&self.w2, &self.lin2_plan);
        let rb = cfg.lin2_row_block / G;
        let t0 = Instant::now();
        self.pool.run(|ctx| {
            let mut scratch = Vec::new();
            let mine = lin2_plan[ctx.tid].clone();
            let mut r0 = mine.start;
            while r0 < mine.end {
                let r1 = (r0 + rb).min(mine.end);
                let nr = (r1 - r0) * G;
                gemm_blocked(&x2, w2, 0..groups, r0..r1, cfg.batch_groups, Order::MulAdd, &mut scratch, |t0, ntok, block| {
                    for ti in 0..ntok {
                        let t = t0 + ti;
                        if t >= n {
                            break;
                        }
                        let zr = &block[ti * nr..(ti + 1) * nr];
                        let c0 = r0 * G;
                        // SAFETY: columns c0..c0+nr belong to this thread's row range.
                        unsafe {
                            let dst = std::slice::from_raw_parts_mut(out_ptr.get().add(t * h + c0), nr);
                            for j in 0..nr {
                                dst[j] = input.x[t * h + c0 + j] + input.gate[c0 + j] * zr[j];
                            }
                            if want_intermediates {
                                std::ptr::copy_nonoverlapping(zr.as_ptr(), z_ptr.get().add(t * h + c0), nr);
                            }
                        }
                    }
                });
                r0 = r1;
            }
        });
        times.linear2 = t0.elapsed().as_secs_f64();
        times.total = t_all.elapsed().as_secs_f64();

        let attn = if want_intermediates { attn } else { Vec::new() };
        (BlockOutput { out, attn, mlp: mlp_out, z }, times)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_fast_accuracy() {
        let mut x = -87.0f32;
        while x <= 87.0 {
            let (a, b) = (exp_fast(x) as f64, (x as f64).exp());
            assert!(((a - b) / b).abs() < 1e-6, "x {x}: {a} vs {b}");
            x += 0.37;
        }
        assert!(exp_fast(f32::NEG_INFINITY) < 2e-38);
        assert_eq!(exp_fast(0.0), 1.0);
    }

    #[test]
    fn partition_is_contiguous_and_balanced() {
        let costs = vec![16usize; 72].into_iter().chain(vec![32usize; 72]).collect::<Vec<_>>();
        for parts in [1, 3, 10, 12] {
            let p = partition(&costs, parts);
            assert_eq!(p.len(), parts);
            assert_eq!(p[0].start, 0);
            assert_eq!(p[parts - 1].end, costs.len());
            let total: usize = costs.iter().sum();
            for w in p.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
            for r in &p {
                let c: usize = costs[r.clone()].iter().sum();
                assert!(c * parts <= total + 32 * parts && c * parts + 32 * parts >= total, "parts {parts}: {c}");
            }
        }
        // more parts than items: every item exactly once
        let p = partition(&[1, 1], 5);
        assert_eq!(p.iter().map(|r| r.len()).sum::<usize>(), 2);
    }
}
