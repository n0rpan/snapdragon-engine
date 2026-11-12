//! O-70 shared fixtures for the fused-attention tests and micro-benchmark (model-free).
//!
//! Production shapes of the full-attention layers (Ornith / Qwen3.6 35B-A3B model desc: 16 query
//! heads, 2 KV heads, head_dim 256, f16 K rows and f16 V^T rows with the engine's strides).
//! `old_tiled` is a step-for-step replica of the engine's f16 `attn_core_tiled` steps [1]-[3]
//! (same FFI calls, same splits, same barriers); `fused` drives `model::engine::fattn::Fa`, the code
//! the engine runs; `ref_f64` is the exact attention over the same f16 inputs.
#![allow(dead_code)]

use kernels_sys::ffi;
use model::engine::fattn::{Fa, FaScratch};
use model::engine::{padded_n_kv, MAX_LANES};
use std::cell::UnsafeCell;

pub const HD: usize = 256;
pub const NKV: usize = 2;
pub const GROUP: usize = 8;
pub const NHEAD: usize = NKV * GROUP;
/// engine default OXE_KV_VPAD
pub const VPAD: usize = 32;
/// f16 NaN: K rows nobody may read (positions > last, and this run's cache rows)
pub const F16_NAN: u16 = 0x7e00;

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// uniform in [-1, 1]
    pub fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

pub fn f16s(v: &[f32]) -> Vec<u16> {
    let mut o = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::oxe_fp32_to_fp16_row(v.as_ptr(), o.as_mut_ptr(), v.len() as i64) };
    o
}

pub fn h2f(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                s * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// One layer's inputs for a run of `nl` lanes at positions pos0..pos0+nl, laid out as the engine
/// keeps them.
pub struct Case {
    pub n_ctx: usize,
    pub nl: usize,
    pub pos0: usize,
    /// query columns per KV head in q16 / gate (the engine's `lanes * group`; > nl * group)
    pub cpk: usize,
    pub v_stride: usize,
    /// [kvh][n_ctx][hd]
    pub k: Vec<u16>,
    /// per lane: [kvh][hd]
    pub k_local: Vec<Vec<u16>>,
    /// [kvh * hd + d][v_stride]
    pub vt: Vec<u16>,
    /// [col][hd], col = kvh * cpk + lane * group + g
    pub q16: Vec<u16>,
    pub gate: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    /// random q (x qs), K, V in [-1, 1], random gates
    Random,
    /// q = 0 (uniform attention), V^T one-hot (V[p][d] = 1 iff d == p % hd), gates 1:
    /// every output is an exact count / n, so a masking or chunk-boundary slip is visible at 1e-6
    OneHot,
    /// all query columns of a KV head = u (entries +-1), K rows 0 except spikes: +8u at the
    /// positions in `spikes` (+8u at the second lane's own position too, invisible to lane 0),
    /// -8u over the whole chunk [256, 512); V^T one-hot, gates 1. Logits up to +-128.
    Spikes,
}

impl Case {
    pub fn new(n_ctx: usize, nl: usize, pos0: usize, kind: Kind, qs: f32, seed: u64, spikes: &[usize]) -> Case {
        assert!(nl >= 1 && nl <= MAX_LANES && pos0 + nl <= n_ctx);
        let last = pos0 + nl - 1;
        let cpk = (nl + 1) * GROUP;
        let v_stride = n_ctx + VPAD;
        let mut r = Lcg(seed);
        let ncols = NKV * cpk;
        let mut u = vec![0f32; NKV * HD];
        for x in u.iter_mut() {
            *x = if r.next() & 1 == 0 { 1.0 } else { -1.0 };
        }
        // K rows (f32 first, so the local copies are the same bits)
        let mut kf = vec![0f32; NKV * n_ctx * HD];
        let mut vf = vec![0f32; NKV * HD * v_stride];
        let mut qf = vec![0f32; ncols * HD];
        let mut gate = vec![1f32; ncols * HD];
        match kind {
            Kind::Random => {
                for x in kf.iter_mut() {
                    *x = r.f();
                }
                for x in vf.iter_mut() {
                    *x = r.f();
                }
                for x in qf.iter_mut() {
                    *x = r.f() * qs;
                }
                for x in gate.iter_mut() {
                    *x = 0.5 + 0.5 * r.f();
                }
            }
            Kind::OneHot | Kind::Spikes => {
                for h in 0..NKV {
                    for d in 0..HD {
                        for p in 0..n_ctx {
                            // beyond last: finite garbage (weights there are +0)
                            vf[(h * HD + d) * v_stride + p] = if p > last { 1000.0 * r.f() } else { (p % HD == d) as u8 as f32 };
                        }
                    }
                }
                if kind == Kind::Spikes {
                    for h in 0..NKV {
                        for c in 0..cpk {
                            qf[(h * cpk + c) * HD..(h * cpk + c + 1) * HD].copy_from_slice(&u[h * HD..(h + 1) * HD]);
                        }
                        let mut put = |p: usize, s: f32| {
                            if p <= last {
                                for j in 0..HD {
                                    kf[(h * n_ctx + p) * HD + j] = s * u[h * HD + j];
                                }
                            }
                        };
                        for p in 256..512 {
                            put(p, -8.0);
                        }
                        for &p in spikes {
                            put(p, 8.0);
                        }
                        if nl > 1 {
                            put(pos0 + 1, 8.0);
                        }
                    }
                }
            }
        }
        let k16 = f16s(&kf);
        let mut k = k16.clone();
        let mut k_local = vec![vec![0u16; NKV * HD]; nl];
        for h in 0..NKV {
            for p in 0..n_ctx {
                let row = &mut k[(h * n_ctx + p) * HD..(h * n_ctx + p + 1) * HD];
                if p >= pos0 && p <= last {
                    k_local[p - pos0][h * HD..(h + 1) * HD].copy_from_slice(row);
                }
                if p >= pos0 {
                    // this run's cache rows and everything beyond: never read
                    row.fill(F16_NAN);
                }
            }
        }
        Case { n_ctx, nl, pos0, cpk, v_stride, k, k_local, vt: f16s(&vf), q16: f16s(&qf), gate }
    }

    pub fn last(&self) -> usize {
        self.pos0 + self.nl - 1
    }

    fn k_row(&self, h: usize, p: usize) -> &[u16] {
        if p >= self.pos0 {
            &self.k_local[p - self.pos0][h * HD..(h + 1) * HD]
        } else {
            &self.k[(h * self.n_ctx + p) * HD..(h * self.n_ctx + p + 1) * HD]
        }
    }

    /// Output length: [lane][head][dim].
    pub fn out_len(&self) -> usize {
        self.nl * NHEAD * HD
    }
}

/// Exact attention (f64) over the f16 inputs: [lane][head][dim], gated.
pub fn ref_f64(c: &Case) -> Vec<f64> {
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0f64; c.out_len()];
    let vt: Vec<f64> = c.vt.iter().map(|&h| h2f(h)).collect();
    for kvh in 0..NKV {
        // K rows in f64, shared by the group's heads and the lanes
        let n = c.last() + 1;
        let mut kf = vec![0f64; n * HD];
        for p in 0..n {
            for (j, &h) in c.k_row(kvh, p).iter().enumerate() {
                kf[p * HD + j] = h2f(h);
            }
        }
        for ln in 0..c.nl {
            let np = c.pos0 + ln + 1;
            for g in 0..GROUP {
                let col = kvh * c.cpk + ln * GROUP + g;
                let q: Vec<f64> = c.q16[col * HD..(col + 1) * HD].iter().map(|&h| h2f(h)).collect();
                let mut s = vec![0f64; np];
                let mut m = f64::NEG_INFINITY;
                for p in 0..np {
                    let mut a = 0f64;
                    for j in 0..HD {
                        a += q[j] * kf[p * HD + j];
                    }
                    s[p] = a * scale;
                    m = m.max(s[p]);
                }
                let mut sum = 0f64;
                for x in s.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                let h = kvh * GROUP + g;
                let o = &mut out[(ln * NHEAD + h) * HD..(ln * NHEAD + h + 1) * HD];
                for (d, od) in o.iter_mut().enumerate() {
                    let row = &vt[(kvh * HD + d) * c.v_stride..];
                    let mut a = 0f64;
                    for p in 0..np {
                        a += s[p] * row[p];
                    }
                    *od = a / sum * c.gate[col * HD + d] as f64;
                }
            }
        }
    }
    out
}

/// CPUs for a pool of n threads: the top of the allowed range (as the model unit tests do).
pub fn cpus(n: usize) -> Vec<usize> {
    let avail = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    (0..n).map(|i| avail - 1 - i % avail).collect()
}

/// Shared mutable buffer for a pool job (each thread writes disjoint parts).
pub struct Shared<T>(pub UnsafeCell<Vec<T>>);
// SAFETY: the jobs below write disjoint ranges between barriers.
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    pub fn new(v: Vec<T>) -> Self {
        Shared(UnsafeCell::new(v))
    }
    pub fn ptr(&self) -> *mut T {
        // SAFETY: only the pointer is taken.
        unsafe { (*self.0.get()).as_mut_ptr() }
    }
    pub fn into_inner(self) -> Vec<T> {
        self.0.into_inner()
    }
}

/// Fused-attention buffers and the `Fa` describing a case.
pub struct FusedRun {
    pub part: Shared<f32>,
    pub ml: Shared<f32>,
    pub out: Shared<f32>,
    pub scr: Vec<Shared<FaScratch>>,
}

impl FusedRun {
    pub fn new(c: &Case, nth: usize) -> FusedRun {
        let lanes = c.cpk / GROUP;
        FusedRun {
            part: Shared::new(vec![0f32; model::engine::fattn::part_len(c.n_ctx, NKV, GROUP, HD, lanes)]),
            ml: Shared::new(vec![0f32; model::engine::fattn::ml_len(c.n_ctx, NKV, GROUP, lanes)]),
            out: Shared::new(vec![f32::NAN; c.out_len()]),
            scr: (0..nth).map(|_| Shared::new(vec![FaScratch::default()])).collect(),
        }
    }

    pub fn fa(&self, c: &Case, ch: usize) -> Fa {
        let mut k_local = [std::ptr::null::<u16>(); MAX_LANES];
        for (i, kl) in c.k_local.iter().enumerate() {
            k_local[i] = kl.as_ptr();
        }
        Fa {
            hd: HD,
            nkv: NKV,
            group: GROUP,
            nl: c.nl,
            pos0: c.pos0,
            ch: model::engine::fattn::chunk_from(ch),
            scale: 1.0 / (HD as f32).sqrt(),
            k: c.k.as_ptr(),
            n_ctx: c.n_ctx,
            k_local,
            vt: c.vt.as_ptr(),
            v_stride: c.v_stride,
            q16: c.q16.as_ptr(),
            gate: c.gate.as_ptr(),
            cpk: c.cpk,
            part: self.part.ptr(),
            ml: self.ml.ptr(),
            out: self.out.ptr(),
            out_ls: NHEAD * HD,
        }
    }

    /// One fused attention on the pool: items, barrier, merge (the engine's phase).
    pub fn run(&self, pool: &sched::Pool, fa: &Fa) {
        pool.run(|ctx| {
            let scr = self.scr[ctx.tid].ptr();
            // SAFETY: Fa's contract (see fattn); scratch slot tid is this thread's.
            unsafe { fa.chunks(ctx.tid, ctx.nthreads, &mut *scr) };
            ctx.barrier();
            // SAFETY: as above.
            unsafe { fa.merge(ctx.tid, ctx.nthreads, &mut *scr) };
        });
    }
}

/// Buffers of the old tiled path (`akq` / `ap16` at column stride acs, per-thread scratch).
pub struct OldRun {
    pub acs: usize,
    pub akq: Shared<f32>,
    pub ap16: Shared<u16>,
    pub out: Shared<f32>,
    /// per thread: wp, dp (n_ctx each), tmp (32 * ncol)
    pub scr: Vec<Shared<f32>>,
}

impl OldRun {
    pub fn new(c: &Case, nth: usize) -> OldRun {
        let acs = c.n_ctx;
        OldRun {
            acs,
            akq: Shared::new(vec![0f32; NKV * c.cpk * acs]),
            ap16: Shared::new(vec![0u16; NKV * c.cpk * acs]),
            out: Shared::new(vec![f32::NAN; c.out_len()]),
            scr: (0..nth).map(|_| Shared::new(vec![0f32; 2 * c.n_ctx + 32 * c.cpk])).collect(),
        }
    }

    /// Replica of the engine's f16 `attn_core_tiled` steps [1] scores, barrier, [2] softmax rows,
    /// barrier, [3] V^T p + gate (the phase's own barrier follows in the engine).
    pub fn run(&self, pool: &sched::Pool, c: &Case) {
        let (hd, group, nkv, nl, pos0, n_ctx, acs) = (HD, GROUP, NKV, c.nl, c.pos0, c.n_ctx, self.acs);
        let scale = 1.0 / (hd as f32).sqrt();
        let last = c.last();
        let (akq, ap16, out) = (self.akq.ptr() as usize, self.ap16.ptr() as usize, self.out.ptr() as usize);
        pool.run(|ctx| {
            let (tid, nth) = (ctx.tid, ctx.nthreads);
            let (akq, ap16, out) = (akq as *mut f32, ap16 as *mut u16, out as *mut f32);
            let scr = self.scr[tid].ptr();
            // SAFETY (whole job): the engine's ownership rules (disjoint rows / columns per thread
            // between barriers); all buffers sized for the case.
            unsafe {
                // [1]
                let rows = last + 1;
                let my = rows * nkv * tid / nth..rows * nkv * (tid + 1) / nth;
                const RB: usize = 64;
                for kvh in 0..nkv {
                    let a = my.start.max(kvh * rows).min((kvh + 1) * rows) - kvh * rows;
                    let b = my.end.max(kvh * rows).min((kvh + 1) * rows) - kvh * rows;
                    let c0 = kvh * c.cpk;
                    let kbase = kvh * n_ctx * hd;
                    let mut p = a;
                    while p < b.min(pos0) {
                        let pe = (p + RB).min(b.min(pos0));
                        ffi::oxe_dot_f16_tile(hd as i32, c.k.as_ptr().add(kbase + p * hd), hd as i64, (pe - p) as i32, c.q16.as_ptr().add(c0 * hd), hd as i64, (nl * group) as i32, akq.add(c0 * acs + p), 1, acs as i64, std::ptr::null());
                        p = pe;
                    }
                    for p in a.max(pos0)..b {
                        let i = p - pos0;
                        ffi::oxe_dot_f16_tile(hd as i32, c.k_local[i].as_ptr().add(kvh * hd), hd as i64, 1, c.q16.as_ptr().add((c0 + i * group) * hd), hd as i64, ((nl - i) * group) as i32, akq.add((c0 + i * group) * acs + p), 1, acs as i64, std::ptr::null());
                    }
                }
                ctx.barrier();
                // [2]
                let n32 = (last + 1).div_ceil(32) * 32;
                let nrow = nl * NHEAD;
                for r in nrow * tid / nth..nrow * (tid + 1) / nth {
                    let (ln, h) = (r / NHEAD, r % NHEAD);
                    let pos = pos0 + ln;
                    let col = (h / group) * c.cpk + ln * group + h % group;
                    ffi::oxe_attn_softmax_row_f16(akq.add(col * acs), pos as i32, padded_n_kv(pos, n_ctx) as i32, scale, scr, scr.add(n_ctx), ap16.add(col * acs), n32 as i32);
                }
                ctx.barrier();
                // [3]
                let mut nf = [0i32; MAX_LANES * 16];
                for (cc, v) in nf.iter_mut().enumerate().take(nl * group) {
                    *v = padded_n_kv(pos0 + cc / group, n_ctx) as i32;
                }
                let my = nkv * hd * tid / nth..nkv * hd * (tid + 1) / nth;
                let ncol = nl * group;
                const DB: usize = 32;
                let tmp = scr.add(2 * n_ctx);
                for kvh in 0..nkv {
                    let a = my.start.max(kvh * hd).min((kvh + 1) * hd) - kvh * hd;
                    let b = my.end.max(kvh * hd).min((kvh + 1) * hd) - kvh * hd;
                    let c0 = kvh * c.cpk;
                    let mut r = a;
                    while r < b {
                        let re = (r + DB).min(b);
                        ffi::oxe_dot_f16_tile(n32 as i32, c.vt.as_ptr().add((kvh * hd + r) * c.v_stride), c.v_stride as i64, (re - r) as i32, ap16.add(c0 * acs), acs as i64, ncol as i32, tmp, ncol as i64, 1, nf.as_ptr());
                        for col in 0..ncol {
                            let (ln, g) = (col / group, col % group);
                            let h = kvh * group + g;
                            for dd in r..re {
                                *out.add((ln * NHEAD + h) * hd + dd) = *tmp.add((dd - r) * ncol + col) * c.gate[(c0 + col) * hd + dd];
                            }
                        }
                        r = re;
                    }
                }
            }
        });
    }
}

/// Max over (lane, head) of ||x - ref||_2 / ||ref||_2, and the max abs error.
pub fn rel_err(x: &[f32], r: &[f64]) -> (f64, f64) {
    let mut worst = 0f64;
    let mut amax = 0f64;
    for (xs, rs) in x.chunks(HD).zip(r.chunks(HD)) {
        let (mut e2, mut r2) = (0f64, 0f64);
        for (&a, &b) in xs.iter().zip(rs) {
            let e = a as f64 - b;
            e2 += e * e;
            r2 += b * b;
            amax = amax.max(e.abs());
        }
        worst = worst.max((e2 / r2.max(1e-300)).sqrt());
    }
    (worst, amax)
}
