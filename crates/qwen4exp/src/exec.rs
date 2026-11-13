//! The M0 token executor: static per-thread plan over the real Flash weights and kernels,
//! dummy activations, replayed routing, real expert streaming through the cache + io_uring.
//!
//! Per layer (10 barriers; 11 on the PLE layer):
//!   P1 hc_attn: norm(X 4x2560) + q8_0; hc_attn_down rows (320); hc_attn_inject rows (4)  | B
//!   P2 hc_attn_up rows (10240)                                                            | B
//!   P3 mixer projections: GDN qkv/gate/alpha/beta rows, or attention q/k/v + indexer rows | B
//!   P4 mixer core: GDN conv + delta rule per value head, or attention per query head      | B
//!   P5 mixer out rows (ssm_out / attn_output, 2560), X update, conv state / KV append     | B
//!   P6 hc_ffn_down, P7 hc_ffn_up                                                          | B B
//!  (PLE, layer 1: N-gram rows (pages read at token start) -> ple_value / ple_key rows     | B)
//!   P8 router rows (F32 512) + shared expert gate/up rows                                 | B
//!   P9 thread 0 looks up the replayed top-10 in the cache (demand reads) and issues the
//!      lookahead prefetch for L+1..L+d; every thread: its rows of the 10 experts' gate/up,
//!      resident experts first, misses as they land (SSD wait)                             | B
//!   P10 expert down rows (2560) + shared down, weighted sum, X update                     | B
//! Head: output_hc down | up | final norm + Q6_K output GEMV with running argmax (join).
//!
//! Faithful: every weight byte a decode token reads, by the production kernels (NEON vec_dot,
//! R4 gemv for IQ2_S/IQ3_XXS experts), the work partition and barrier structure, the expert
//! byte stream from SSD (sidecar records, O_DIRECT, io_uring) and the N-gram page reads.
//! Approximated: activations (dummy, bounded by norms), hyper-connection mixing math, the PLE
//! combination, the indexer/compression of the attention layers (all keys used: ctx < 2048),
//! RoPE, routing (replayed from the P2 trace instead of computed).

use crate::cache::{Cache, Entry};
use crate::dense::{Dense, Mat};
use crate::ffi::W;
use crate::io::{IoQueue, ReqClass};
use crate::model::*;
use crate::ngram::NgramTable;
use crate::trace::Trace;
use kernels_sys::{ffi as kf, quantize_act, ActType};
use sched::Ctx;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering::*};
use std::time::Instant;

const EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Hc = 0,
    MixProj,
    MixCore,
    MixOut,
    Ple,
    Router,
    ExpGateUp,
    ExpDown,
    Head,
}
pub const KINDS: usize = 9;
pub const KIND_NAMES: [&str; KINDS] = ["hc", "mix_proj", "mix_core", "mix_out", "ple", "router", "exp_gate_up", "exp_down", "head"];

#[derive(Clone, Copy)]
pub struct Hc {
    norm: *const f32,
    down: Mat,
    up: Mat,
    inject: Option<Mat>,
}

#[derive(Clone, Copy)]
pub struct LayerW {
    pub attn: bool,
    hc: [Hc; 2],
    // gdn
    qkv: Option<Mat>,
    zgate: Option<Mat>,
    alpha: Option<Mat>,
    beta: Option<Mat>,
    conv: *const f32,
    ssm_a: *const f32,
    dt_bias: *const f32,
    ssm_norm: *const f32,
    ssm_out: Option<Mat>,
    // attention
    q: Option<Mat>,
    k: Option<Mat>,
    v: Option<Mat>,
    q_norm: *const f32,
    k_norm: *const f32,
    idx_q: Option<Mat>,
    idx_k: Option<Mat>,
    out: Option<Mat>,
    // moe
    router: Mat,
    sh_inp: *const f32,
    sh_gate: Mat,
    sh_up: Mat,
    sh_down: Mat,
    // ple
    ple_key: Option<Mat>,
    ple_value: Option<Mat>,
    pub exp: ExpertGeom,
}
unsafe impl Send for LayerW {}
unsafe impl Sync for LayerW {}

pub struct Weights {
    pub layers: Vec<LayerW>,
    out_hc: Hc,
    output: Mat,
    token_embd: Mat,
    /// unique weight bytes one decode token reads, by group
    pub bytes_dense: usize,
    pub bytes_router: usize,
    pub bytes_experts: usize,
}

fn null() -> *const f32 {
    std::ptr::null()
}

impl Weights {
    pub fn new(d: &Dense, s: &Shards) -> Weights {
        let hc = |p: &str| Hc {
            norm: d.f32(&format!("{p}_norm.weight")),
            down: d.get(&format!("{p}_down.weight")),
            up: d.get(&format!("{p}_up.weight")),
            inject: d.opt(&format!("{p}_inject.weight")),
        };
        let mut layers = Vec::new();
        let (mut bd, mut br, mut be) = (0usize, 0usize, 0usize);
        for l in 0..N_LAYER {
            let n = |x: &str| format!("blk.{l}.{x}");
            let o = |x: &str| d.opt(&n(x));
            let f = |x: &str| d.opt(&n(x)).map(|m| m.p as *const f32).unwrap_or(null());
            let lw = LayerW {
                attn: s.is_attn_layer(l),
                hc: [hc(&n("hc_attn")), hc(&n("hc_ffn"))],
                qkv: o("attn_qkv.weight"),
                zgate: o("attn_gate.weight"),
                alpha: o("ssm_alpha.weight"),
                beta: o("ssm_beta.weight"),
                conv: f("ssm_conv1d.weight"),
                ssm_a: f("ssm_a"),
                dt_bias: f("ssm_dt.bias"),
                ssm_norm: f("ssm_norm.weight"),
                ssm_out: o("ssm_out.weight"),
                q: o("attn_q.weight"),
                k: o("attn_k.weight"),
                v: o("attn_v.weight"),
                q_norm: f("attn_q_norm.weight"),
                k_norm: f("attn_k_norm.weight"),
                idx_q: o("indexer.q_proj.weight"),
                idx_k: o("indexer.k_proj.weight"),
                out: o("attn_output.weight"),
                router: d.get(&n("ffn_gate_inp.weight")),
                sh_inp: d.f32(&n("ffn_gate_inp_shexp.weight")),
                sh_gate: d.get(&n("ffn_gate_shexp.weight")),
                sh_up: d.get(&n("ffn_up_shexp.weight")),
                sh_down: d.get(&n("ffn_down_shexp.weight")),
                ple_key: o("ple_key.weight"),
                ple_value: o("ple_value.weight"),
                exp: expert_geom(s, l),
            };
            for m in [Some(lw.hc[0].down), Some(lw.hc[0].up), lw.hc[0].inject, Some(lw.hc[1].down), Some(lw.hc[1].up), lw.hc[1].inject]
                .into_iter()
                .chain([lw.qkv, lw.zgate, lw.alpha, lw.beta, lw.ssm_out, lw.q, lw.k, lw.v, lw.idx_q, lw.idx_k, lw.out])
                .chain([Some(lw.sh_gate), Some(lw.sh_up), Some(lw.sh_down), lw.ple_key, lw.ple_value])
                .flatten()
            {
                bd += m.bytes();
            }
            br += lw.router.bytes();
            be += TOP_K * lw.exp.record;
            assert_eq!(lw.attn, lw.q.is_some() && lw.qkv.is_none(), "layer {l}: mixer tensors");
            layers.push(lw);
        }
        let out_hc = hc("output_hc");
        let output = d.get("output.weight");
        bd += out_hc.down.bytes() + out_hc.up.bytes() + output.bytes();
        Weights { layers, out_hc, output, token_embd: d.get("token_embd.weight"), bytes_dense: bd, bytes_router: br, bytes_experts: be }
    }
}

/// Shared f32 buffer written in disjoint ranges between barriers.
/// Shared f32 scratch written by the team on disjoint ranges between barriers.
///
/// S7 (REVIEW-v0.1.1): slices are built from a raw pointer + length taken once at construction; no thread ever
/// forms a `&mut Vec` (the old `UnsafeCell<Vec>` handed every thread a `&mut` to the same Vec header). Writers that
/// split work by rows take only their range with [`SBuf::range`].
struct SBuf {
    ptr: *mut f32,
    len: usize,
}
unsafe impl Sync for SBuf {}
unsafe impl Send for SBuf {}
impl SBuf {
    fn new(n: usize) -> SBuf {
        let b: Box<[f32]> = vec![0.0; n].into_boxed_slice();
        SBuf { len: b.len(), ptr: Box::into_raw(b) as *mut f32 }
    }
    /// The whole buffer, mutable. Callers write only the elements their phase owns (disjoint between threads).
    #[allow(clippy::mut_from_ref)]
    unsafe fn m(&self) -> &mut [f32] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }
    /// Elements `r0..r1` only (bounds checked: cold relative to the GEMV it feeds).
    #[allow(clippy::mut_from_ref)]
    unsafe fn range(&self, r0: usize, r1: usize) -> &mut [f32] {
        assert!(r0 <= r1 && r1 <= self.len, "SBuf range {r0}..{r1} of {}", self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(r0), r1 - r0)
    }
    unsafe fn r(&self) -> &[f32] {
        std::slice::from_raw_parts(self.ptr, self.len)
    }
}
impl Drop for SBuf {
    fn drop(&mut self) {
        // SAFETY: ptr/len come from Box::into_raw of a boxed slice of exactly this length, dropped once.
        unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr, self.len))) }
    }
}

/// Per-thread scratch (own 128-byte lines: aliasing sweep, no false sharing between threads' headers).
#[repr(align(128))]
struct Ts {
    a: Vec<u8>,
    a2: Vec<u8>,
    xn: Vec<f32>,
    tmp: Vec<f32>,
    tmp2: Vec<f32>,
    delta: Vec<f32>,
    /// per phase of the current token: (kind, work ns excl. waits, ssd wait ns, ngram wait ns)
    prof: Vec<(u8, u64, u64, u64)>,
    /// thread 0 only: barrier exit timestamps (ns since token start)
    walls: Vec<u64>,
    mgmt_ns: u64,
    best: (f32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoMode {
    /// misses keep the slot's old bytes: compute-only timing (all experts "resident")
    None,
    Demand,
    Lookahead,
}

pub struct Cfg {
    pub io: IoMode,
    pub recall: f64,
    pub depth: usize,
    pub false_pos: bool,
    pub ctx: usize,
    pub pos0: usize,
    pub ngram: bool,
    pub seed: u64,
}

pub struct Engine<'a> {
    pub w: &'a Weights,
    pub cache: &'a Cache,
    pub io: &'a IoQueue,
    ngram: Option<UnsafeCell<NgramTable>>,
    pub trace: &'a Trace,
    pub cfg: Cfg,
    nt: usize,
    x: SBuf,       // 4 x 2560 hyper-connection streams
    hcw: SBuf,     // 4 inject weights
    low: SBuf,     // 320
    hup: SBuf,     // 10240
    xin: SBuf,     // 2560 (block input, recomputed redundantly)
    qkv: SBuf,     // 10240 (GDN) / 12288+512+512 (attn: q|k|v)
    z: SBuf,       // 6144
    ab: SBuf,      // 96
    idx: SBuf,     // 512 + 128
    mix: SBuf,     // 6144
    blk: SBuf,     // 2560
    ple: SBuf,     // 2560 + 10240
    logits: SBuf,  // 512
    sh: SBuf,      // 640 gate | 640 up
    eh: SBuf,      // 10 x 640 SwiGLU outputs
    moe: SBuf,     // 2560
    gdn_state: Vec<SBuf>,
    conv_state: Vec<SBuf>,
    kv_k: Vec<SBuf>,
    kv_v: Vec<SBuf>,
    kv_i: Vec<SBuf>,
    work: UnsafeCell<[Entry; TOP_K]>,
    work_epoch: AtomicU64,
    ts: Vec<UnsafeCell<Ts>>,
    // control-thread state
    pub tok: UnsafeCell<usize>,
    rng: UnsafeCell<u64>,
}
unsafe impl Sync for Engine<'_> {}

fn part(n: usize, nt: usize, t: usize, align: usize) -> (usize, usize) {
    assert!(n % align == 0, "{n} rows not a multiple of {align}");
    let b = n / align;
    (b * t / nt * align, b * (t + 1) / nt * align)
}

#[inline]
fn now_ns(t0: Instant) -> u64 {
    t0.elapsed().as_nanos() as u64
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

impl<'a> Engine<'a> {
    pub fn new(w: &'a Weights, cache: &'a Cache, io: &'a IoQueue, ngram: Option<NgramTable>, trace: &'a Trace, cfg: Cfg, nt: usize) -> Engine<'a> {
        let n_attn = w.layers.iter().filter(|l| l.attn).count();
        let n_gdn = N_LAYER - n_attn;
        let ctx = cfg.ctx;
        let seed = cfg.seed;
        let ts = (0..nt)
            .map(|_| {
                UnsafeCell::new(Ts {
                    a: vec![0; 16384],
                    a2: vec![0; TOP_K * 20 * 34 + 1024],
                    xn: vec![0.0; N_HC * N_EMBD],
                    tmp: vec![0.0; 16384],
                    tmp2: vec![0.0; 16384],
                    delta: vec![0.0; 1024],
                    prof: Vec::with_capacity(1024),
                    walls: Vec::with_capacity(1024),
                    mgmt_ns: 0,
                    best: (f32::MIN, 0),
                })
            })
            .collect();
        let e = Engine {
            w,
            cache,
            io,
            ngram: ngram.map(UnsafeCell::new),
            trace,
            cfg,
            nt,
            x: SBuf::new(N_HC * N_EMBD),
            hcw: SBuf::new(N_HC),
            low: SBuf::new(HC_RANK),
            hup: SBuf::new(N_HC * N_EMBD),
            xin: SBuf::new(N_EMBD),
            qkv: SBuf::new(N_HEAD * HEAD_D * 2 + 2 * N_HEAD_KV * HEAD_D),
            z: SBuf::new(SSM_INNER),
            ab: SBuf::new(2 * SSM_V_HEADS),
            idx: SBuf::new(IDX_HEADS * IDX_D + IDX_D),
            mix: SBuf::new(N_HEAD * HEAD_D),
            blk: SBuf::new(N_EMBD),
            ple: SBuf::new(N_EMBD + SSM_QKV),
            logits: SBuf::new(N_EXPERT),
            sh: SBuf::new(2 * FF_EXP),
            eh: SBuf::new(TOP_K * FF_EXP),
            moe: SBuf::new(N_EMBD),
            gdn_state: (0..n_gdn).map(|_| SBuf::new(SSM_V_HEADS * SSM_D * SSM_D)).collect(),
            conv_state: (0..n_gdn).map(|_| SBuf::new((SSM_CONV - 1) * SSM_QKV)).collect(),
            kv_k: (0..n_attn).map(|_| SBuf::new(ctx * N_HEAD_KV * HEAD_D)).collect(),
            kv_v: (0..n_attn).map(|_| SBuf::new(ctx * N_HEAD_KV * HEAD_D)).collect(),
            kv_i: (0..n_attn).map(|_| SBuf::new(ctx * IDX_D)).collect(),
            work: UnsafeCell::new([Entry { ptr: std::ptr::null(), state: std::ptr::null() }; TOP_K]),
            work_epoch: AtomicU64::new(0),
            ts,
            tok: UnsafeCell::new(0),
            rng: UnsafeCell::new(seed | 1),
        };
        // bounded, non-zero initial state (dummy)
        unsafe {
            for (i, v) in e.x.m().iter_mut().enumerate() {
                *v = ((i * 7919 % 1000) as f32 - 500.0) * 1e-3;
            }
            for s in &e.kv_k {
                s.m().iter_mut().enumerate().for_each(|(i, v)| *v = ((i % 97) as f32 - 48.0) * 0.01);
            }
            for s in &e.kv_v {
                s.m().iter_mut().enumerate().for_each(|(i, v)| *v = ((i % 89) as f32 - 44.0) * 0.01);
            }
        }
        e
    }

    /// (page lookups, page hits, page reads) of the N-gram page cache so far.
    pub fn ngram_stats(&self) -> Option<(u64, u64, u64)> {
        // SAFETY: control thread between jobs.
        self.ngram.as_ref().map(|n| unsafe {
            let n = &*n.get();
            (n.lookups, n.hits, n.reads)
        })
    }

    pub fn nthreads(&self) -> usize {
        self.nt
    }

    /// Token start (control thread, outside the job): embedding row into X, N-gram reads.
    pub fn begin_token(&self, toks: &[u32]) {
        let t = *toks.last().unwrap() as usize % N_VOCAB;
        let m = self.w.token_embd;
        let rb = m.w.row_bytes(N_EMBD);
        let mut row = vec![0f32; N_EMBD];
        // SAFETY: row t of the resident token_embd (file layout).
        unsafe { kf::dequantize_row_q6_K(m.p.add(t * rb), row.as_mut_ptr(), N_EMBD as i64) };
        let x = unsafe { self.x.m() };
        for s in 0..N_HC {
            for i in 0..N_EMBD {
                x[s * N_EMBD + i] = 0.5 * x[s * N_EMBD + i] + row[i];
            }
        }
        if let Some(ng) = &self.ngram {
            // SAFETY: control thread between jobs; workers only read it inside the job.
            unsafe { (*ng.get()).issue(self.io, toks) };
        }
    }

    /// Run one decode token on the team (`ctx` = this thread's handle).
    pub fn token_job(&self, c: &mut Ctx) {
        let tid = c.tid;
        // SAFETY: slot tid is owned by this thread.
        let ts = unsafe { &mut *self.ts[tid].get() };
        ts.prof.clear();
        ts.walls.clear();
        ts.mgmt_ns = 0;
        ts.best = (f32::MIN, 0);
        let t0 = Instant::now();
        let tok = unsafe { *self.tok.get() };
        let pos = (self.cfg.pos0 + tok).min(self.cfg.ctx - 1);
        let mut gi = 0usize; // gdn layer index
        let mut ai = 0usize; // attn layer index
        for l in 0..N_LAYER {
            let lw = &self.w.layers[l];
            self.hc_phases(c, ts, t0, &lw.hc[0]);
            // P3
            let s = Instant::now();
            unsafe { self.mixer_proj(c, ts, lw) };
            self.end_phase(c, ts, t0, Kind::MixProj, s, 0, 0);
            // P4
            let s = Instant::now();
            unsafe {
                if lw.attn {
                    self.attn_core(c, ts, lw, ai, pos)
                } else {
                    self.gdn_core(c, ts, lw, gi)
                }
            };
            self.end_phase(c, ts, t0, Kind::MixCore, s, 0, 0);
            // P5
            let s = Instant::now();
            unsafe { self.mixer_out(c, ts, lw, gi, ai, pos) };
            self.end_phase(c, ts, t0, Kind::MixOut, s, 0, 0);
            if lw.attn {
                ai += 1;
            } else {
                gi += 1;
            }
            self.hc_phases(c, ts, t0, &lw.hc[1]);
            if l == PLE_LAYER {
                let s = Instant::now();
                let ng = unsafe { self.ple_phase(c, ts, lw) };
                self.end_phase(c, ts, t0, Kind::Ple, s, 0, ng);
            }
            // P8
            let s = Instant::now();
            unsafe { self.router_phase(c, ts, lw) };
            self.end_phase(c, ts, t0, Kind::Router, s, 0, 0);
            // P9
            let s = Instant::now();
            let ssd = unsafe { self.experts_gate_up(c, ts, l, tok) };
            self.end_phase(c, ts, t0, Kind::ExpGateUp, s, ssd, 0);
            // P10
            let s = Instant::now();
            unsafe { self.experts_down(c, ts, lw) };
            self.end_phase(c, ts, t0, Kind::ExpDown, s, 0, 0);
        }
        unsafe { self.head(c, ts, t0) };
    }

    fn end_phase(&self, c: &mut Ctx, ts: &mut Ts, t0: Instant, k: Kind, s: Instant, ssd: u64, ng: u64) {
        let w = s.elapsed().as_nanos() as u64;
        ts.prof.push((k as u8, w.saturating_sub(ssd + ng), ssd, ng));
        c.barrier();
        if c.tid == 0 {
            ts.walls.push(now_ns(t0));
        }
    }

    fn hc_phases(&self, c: &mut Ctx, ts: &mut Ts, t0: Instant, hc: &Hc) {
        let (tid, nt) = (c.tid, c.nthreads);
        // P1: norm + q8_0 of all 4 streams, down rows, inject rows
        let s = Instant::now();
        unsafe {
            let x = self.x.r();
            kf::oxe_rms_norm_mul_f32(x.as_ptr(), hc.norm, ts.xn.as_mut_ptr(), (N_HC * N_EMBD) as i64, EPS);
            quantize_act(ActType::Q8_0, &ts.xn, &mut ts.a);
            let (r0, r1) = part(HC_RANK, nt, tid, 8);
            hc.down.gemv(r0, r1, &ts.a, &ts.xn, self.low.range(r0, r1));
            if let Some(inj) = hc.inject {
                if tid < N_HC.min(nt) {
                    let (i0, i1) = if nt >= N_HC { (tid, tid + 1) } else { part(N_HC, nt, tid, 1) };
                    inj.gemv(i0, i1, &ts.a, &ts.xn, self.hcw.range(i0, i1));
                }
            }
        }
        self.end_phase(c, ts, t0, Kind::Hc, s, 0, 0);
        // P2: up rows
        let s = Instant::now();
        unsafe {
            let low = self.low.r();
            // bounded dummy: normalise the low-rank vector before quantising
            kf::oxe_rms_norm_f32(low.as_ptr(), ts.tmp.as_mut_ptr(), HC_RANK as i64, EPS);
            quantize_act(ActType::Q8_0, &ts.tmp[..HC_RANK], &mut ts.a);
            let (r0, r1) = part(N_HC * N_EMBD, nt, tid, 8);
            hc.up.gemv(r0, r1, &ts.a, &ts.tmp[..HC_RANK], self.hup.range(r0, r1));
        }
        self.end_phase(c, ts, t0, Kind::Hc, s, 0, 0);
    }

    /// Block input (redundant on every thread): weighted stream sum + low-rank term, normalised.
    unsafe fn block_input(&self, ts: &mut Ts) {
        let x = self.x.r();
        let h = self.hup.r();
        let w = self.hcw.r();
        let xin = &mut ts.xn[..N_EMBD];
        for i in 0..N_EMBD {
            let mut v = 0.0;
            for s in 0..N_HC {
                v += x[s * N_EMBD + i] * (0.25 + 0.01 * w[s].tanh()) + 0.01 * h[s * N_EMBD + i];
            }
            xin[i] = v;
        }
        kf::oxe_rms_norm_f32(xin.as_ptr(), ts.tmp2.as_mut_ptr(), N_EMBD as i64, EPS);
        xin.copy_from_slice(&ts.tmp2[..N_EMBD]);
    }

    unsafe fn mixer_proj(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW) {
        let (tid, nt) = (c.tid, c.nthreads);
        self.block_input(ts);
        quantize_act(ActType::Q8K, &ts.xn[..N_EMBD], &mut ts.a);
        let xin = &ts.xn[..N_EMBD];
        let q = self.qkv.m();
        if lw.attn {
            let (mq, mk, mv) = (lw.q.unwrap(), lw.k.unwrap(), lw.v.unwrap());
            let (r0, r1) = part(mq.n, nt, tid, 8);
            mq.gemv(r0, r1, &ts.a, xin, &mut q[r0..r1]);
            let o = mq.n;
            let (r0, r1) = part(mk.n, nt, tid, 8);
            mk.gemv(r0, r1, &ts.a, xin, &mut q[o + r0..o + r1]);
            let o = o + mk.n;
            mv.gemv(r0, r1, &ts.a, xin, &mut q[o + r0..o + r1]);
            let (iq, ik) = (lw.idx_q.unwrap(), lw.idx_k.unwrap());
            let idx = self.idx.m();
            let (r0, r1) = part(iq.n, nt, tid, 8);
            iq.gemv(r0, r1, &ts.a, xin, &mut idx[r0..r1]);
            let (r0, r1) = part(ik.n, nt, tid, 8);
            ik.gemv(r0, r1, &ts.a, xin, &mut idx[iq.n + r0..iq.n + r1]);
        } else {
            let (mq, mz, ma, mb) = (lw.qkv.unwrap(), lw.zgate.unwrap(), lw.alpha.unwrap(), lw.beta.unwrap());
            let (r0, r1) = part(mq.n, nt, tid, 8);
            mq.gemv(r0, r1, &ts.a, xin, &mut q[r0..r1]);
            let (r0, r1) = part(mz.n, nt, tid, 8);
            mz.gemv(r0, r1, &ts.a, xin, self.z.range(r0, r1));
            let ab = self.ab.m();
            let (r0, r1) = part(ma.n, nt, tid, 1);
            ma.gemv(r0, r1, &ts.a, xin, &mut ab[r0..r1]);
            mb.gemv(r0, r1, &ts.a, xin, &mut ab[SSM_V_HEADS + r0..SSM_V_HEADS + r1]);
        }
    }

    unsafe fn gdn_core(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW, gi: usize) {
        let (tid, nt) = (c.tid, c.nthreads);
        let qkv = self.qkv.r();
        let cs = self.conv_state[gi].r();
        let st = self.gdn_state[gi].m();
        let z = self.z.r();
        let ab = self.ab.r();
        let mix = self.mix.m();
        let (h0, h1) = part(SSM_V_HEADS, nt, tid, 1);
        let conv = |ch0: usize, out: &mut [f32]| {
            for (j, o) in out.iter_mut().enumerate() {
                let ch = ch0 + j;
                let w = std::slice::from_raw_parts(lw.conv.add(ch * SSM_CONV), SSM_CONV);
                let mut v = qkv[ch] * w[SSM_CONV - 1];
                for t in 0..SSM_CONV - 1 {
                    v += cs[t * SSM_QKV + ch] * w[t];
                }
                *o = v / (1.0 + (-v).exp()); // SiLU
            }
        };
        let mut qb = [0f32; SSM_D];
        let mut kb = [0f32; SSM_D];
        let mut vb = [0f32; SSM_D];
        let mut last_kh = usize::MAX;
        for h in h0..h1 {
            let kh = h * SSM_K_HEADS / SSM_V_HEADS;
            if kh != last_kh {
                conv(kh * SSM_D, &mut qb);
                conv(SSM_K_HEADS * SSM_D + kh * SSM_D, &mut kb);
                kf::oxe_gdn_l2_norm_f32(qb.as_mut_ptr(), SSM_D as i64, EPS);
                kf::oxe_gdn_l2_norm_f32(kb.as_mut_ptr(), SSM_D as i64, EPS);
                last_kh = kh;
            }
            conv(2 * SSM_K_HEADS * SSM_D + h * SSM_D, &mut vb);
            let sp = (ab[h] + *lw.dt_bias.add(h)).max(-20.0).min(20.0);
            let softplus = (1.0 + sp.exp()).ln();
            let g = -(*lw.ssm_a.add(h)).abs().min(10.0) * softplus; // log-decay (the op applies expf)
            let beta = 1.0 / (1.0 + (-ab[SSM_V_HEADS + h]).exp());
            let out = &mut ts.tmp[..SSM_D];
            kf::oxe_gated_delta_net_step_f32(
                st.as_mut_ptr().add(h * SSM_D * SSM_D),
                qb.as_ptr(),
                kb.as_ptr(),
                vb.as_ptr(),
                g,
                beta,
                SSM_D as i64,
                ts.delta.as_mut_ptr(),
                out.as_mut_ptr(),
            );
            // gated RMSNorm
            kf::oxe_rms_norm_mul_f32(out.as_ptr(), lw.ssm_norm, ts.tmp2.as_mut_ptr(), SSM_D as i64, EPS);
            for i in 0..SSM_D {
                let zz = z[h * SSM_D + i];
                mix[h * SSM_D + i] = ts.tmp2[i] * zz / (1.0 + (-zz).exp());
            }
        }
    }

    unsafe fn attn_core(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW, ai: usize, pos: usize) {
        let (tid, nt) = (c.tid, c.nthreads);
        let q = self.qkv.r();
        let kc = self.kv_k[ai].r();
        let vc = self.kv_v[ai].r();
        let mix = self.mix.m();
        let n_kv = pos + 1;
        let knew = &q[N_HEAD * 2 * HEAD_D..N_HEAD * 2 * HEAD_D + N_HEAD_KV * HEAD_D];
        let vnew = &q[N_HEAD * 2 * HEAD_D + N_HEAD_KV * HEAD_D..];
        let (h0, h1) = part(N_HEAD, nt, tid, 1);
        let scale = 1.0 / (HEAD_D as f32).sqrt();
        for h in h0..h1 {
            let kvh = h * N_HEAD_KV / N_HEAD;
            let qh = &q[h * 2 * HEAD_D..h * 2 * HEAD_D + HEAD_D];
            let gh = &q[h * 2 * HEAD_D + HEAD_D..(h + 1) * 2 * HEAD_D];
            let mut qn = [0f32; HEAD_D];
            kf::oxe_rms_norm_mul_f32(qh.as_ptr(), lw.q_norm, qn.as_mut_ptr(), HEAD_D as i64, EPS);
            let sc = &mut ts.tmp[..n_kv];
            for p in 0..n_kv {
                let kp = if p == pos { &knew[kvh * HEAD_D..(kvh + 1) * HEAD_D] } else { &kc[(p * N_HEAD_KV + kvh) * HEAD_D..(p * N_HEAD_KV + kvh + 1) * HEAD_D] };
                sc[p] = kf::oxe_vec_dot_f32(HEAD_D as i32, qn.as_ptr(), kp.as_ptr()) * scale;
            }
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            for v in sc.iter_mut() {
                *v = (*v - m).exp();
                sum += *v;
            }
            let o = &mut ts.tmp2[..HEAD_D];
            o.fill(0.0);
            for p in 0..n_kv {
                let vp = if p == pos { &vnew[kvh * HEAD_D..(kvh + 1) * HEAD_D] } else { &vc[(p * N_HEAD_KV + kvh) * HEAD_D..(p * N_HEAD_KV + kvh + 1) * HEAD_D] };
                let wgt = sc[p] / sum;
                for i in 0..HEAD_D {
                    o[i] += wgt * vp[i];
                }
            }
            for i in 0..HEAD_D {
                mix[h * HEAD_D + i] = o[i] / (1.0 + (-gh[i]).exp());
            }
        }
        // lightning indexer scores (all keys are kept below top_k = 2048; computed for the cost)
        if tid < IDX_HEADS {
            let idx = self.idx.r();
            let ic = self.kv_i[ai].r();
            let qh = &idx[tid * IDX_D..(tid + 1) * IDX_D];
            let mut acc = 0f32;
            for p in 0..pos {
                acc += kf::oxe_vec_dot_f32(IDX_D as i32, qh.as_ptr(), ic[p * IDX_D..].as_ptr()).max(0.0);
            }
            ts.tmp[0] = acc;
        }
    }

    unsafe fn mixer_out(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW, gi: usize, ai: usize, pos: usize) {
        let (tid, nt) = (c.tid, c.nthreads);
        let (m, k) = if lw.attn { (lw.out.unwrap(), N_HEAD * HEAD_D) } else { (lw.ssm_out.unwrap(), SSM_INNER) };
        assert_eq!(m.k, k);
        let mix = &self.mix.r()[..k];
        quantize_act(ActType::Q8K, mix, &mut ts.a);
        let (r0, r1) = part(N_EMBD, nt, tid, 8);
        let blk = self.blk.m();
        m.gemv(r0, r1, &ts.a, mix, &mut blk[r0..r1]);
        let x = self.x.m();
        let w = self.hcw.r();
        for s in 0..N_HC {
            let ws = 0.1 * w[s].tanh();
            for i in r0..r1 {
                x[s * N_EMBD + i] = 0.9 * x[s * N_EMBD + i] + ws * blk[i] + 0.01 * blk[i];
            }
        }
        if lw.attn {
            // append the new K/V (and indexer key) at pos
            let q = self.qkv.r();
            let kn = &q[N_HEAD * 2 * HEAD_D..];
            if tid == 0 {
                let kc = self.kv_k[ai].m();
                let vc = self.kv_v[ai].m();
                for h in 0..N_HEAD_KV {
                    let dst = &mut kc[(pos * N_HEAD_KV + h) * HEAD_D..(pos * N_HEAD_KV + h + 1) * HEAD_D];
                    kf::oxe_rms_norm_mul_f32(kn[h * HEAD_D..].as_ptr(), lw.k_norm, dst.as_mut_ptr(), HEAD_D as i64, EPS);
                    vc[(pos * N_HEAD_KV + h) * HEAD_D..(pos * N_HEAD_KV + h + 1) * HEAD_D]
                        .copy_from_slice(&kn[N_HEAD_KV * HEAD_D + h * HEAD_D..N_HEAD_KV * HEAD_D + (h + 1) * HEAD_D]);
                }
                let ic = self.kv_i[ai].m();
                let idx = self.idx.r();
                ic[pos * IDX_D..(pos + 1) * IDX_D].copy_from_slice(&idx[IDX_HEADS * IDX_D..IDX_HEADS * IDX_D + IDX_D]);
            }
        } else {
            // shift the conv state over this thread's channels
            let cs = self.conv_state[gi].m();
            let q = self.qkv.r();
            let (c0, c1) = part(SSM_QKV, nt, tid, 1);
            for t in 0..SSM_CONV - 2 {
                for ch in c0..c1 {
                    cs[t * SSM_QKV + ch] = cs[(t + 1) * SSM_QKV + ch];
                }
            }
            for ch in c0..c1 {
                cs[(SSM_CONV - 2) * SSM_QKV + ch] = q[ch];
            }
        }
    }

    /// PLE at layer 1: wait for the token's N-gram pages, dequantise 16 rows, ple_value/ple_key rows.
    unsafe fn ple_phase(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW) -> u64 {
        let (tid, nt) = (c.tid, c.nthreads);
        let e = &mut ts.tmp2[..N_EMBD];
        let waited = match &self.ngram {
            Some(ng) => (*ng.get()).gather(e),
            None => {
                e.iter_mut().enumerate().for_each(|(i, v)| *v = ((i % 13) as f32 - 6.0) * 0.1);
                0
            }
        };
        kf::oxe_rms_norm_f32(e.as_ptr(), ts.xn.as_mut_ptr(), N_EMBD as i64, EPS);
        quantize_act(ActType::Q8_0, &ts.xn[..N_EMBD], &mut ts.a);
        let (pv, pk) = (lw.ple_value.unwrap(), lw.ple_key.unwrap());
        let ple = self.ple.m();
        let (r0, r1) = part(pv.n, nt, tid, 8);
        pv.gemv(r0, r1, &ts.a, &ts.xn[..N_EMBD], &mut ple[r0..r1]);
        let (k0, k1) = part(pk.n, nt, tid, 8);
        pk.gemv(k0, k1, &ts.a, &ts.xn[..N_EMBD], &mut ple[N_EMBD + k0..N_EMBD + k1]);
        let x = self.x.m();
        for s in 0..N_HC {
            for i in r0..r1 {
                x[s * N_EMBD + i] += 0.01 * ple[i];
            }
        }
        waited
    }

    unsafe fn router_phase(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW) {
        let (tid, nt) = (c.tid, c.nthreads);
        self.block_input(ts);
        quantize_act(ActType::Q8K, &ts.xn[..N_EMBD], &mut ts.a);
        let xin = &ts.xn[..N_EMBD];
        let (r0, r1) = part(N_EXPERT, nt, tid, 1);
        lw.router.gemv(r0, r1, &ts.a, xin, self.logits.range(r0, r1));
        let sh = self.sh.m();
        let (r0, r1) = part(FF_EXP, nt, tid, 8);
        lw.sh_gate.gemv(r0, r1, &ts.a, xin, &mut sh[r0..r1]);
        lw.sh_up.gemv(r0, r1, &ts.a, xin, &mut sh[FF_EXP + r0..FF_EXP + r1]);
        for r in r0..r1 {
            let g = sh[r];
            sh[r] = g / (1.0 + (-g).exp()) * sh[FF_EXP + r];
        }
        // the shared-expert gate scalar, redundantly (2560 MACs)
        if tid == 0 {
            let s = kf::oxe_vec_dot_f32(N_EMBD as i32, lw.sh_inp, xin.as_ptr());
            self.xin.m()[0] = 1.0 / (1.0 + (-s).exp());
        }
    }

    /// Thread 0: cache lookups for layer l (demand) and the lookahead prefetch.
    unsafe fn control_lookups(&self, l: usize, tok: usize) {
        let stamp = tok as u64 + 1;
        let truth = &self.trace.decode[tok];
        let work = &mut *self.work.get();
        let class = ReqClass::Demand;
        for j in 0..TOP_K {
            let (e, _) = self.cache.lookup(self.io, l, truth[l][j] as usize, stamp, class);
            work[j] = e;
        }
        if self.cfg.io == IoMode::Lookahead {
            let rng = &mut *self.rng.get();
            for d in 1..=self.cfg.depth {
                let l2 = l + d;
                if l2 >= N_LAYER {
                    break;
                }
                let mut n_ok = 0;
                for j in 0..TOP_K {
                    if (xorshift(rng) % 1_000_000) as f64 / 1e6 < self.cfg.recall {
                        self.cache.lookup(self.io, l2, truth[l2][j] as usize, stamp, ReqClass::Prefetch);
                        n_ok += 1;
                    }
                }
                if self.cfg.false_pos {
                    for _ in n_ok..TOP_K {
                        let e = (xorshift(rng) % N_EXPERT as u64) as usize;
                        self.cache.lookup(self.io, l2, e, stamp, ReqClass::Prefetch);
                    }
                }
            }
        }
    }

    unsafe fn experts_gate_up(&self, c: &Ctx, ts: &mut Ts, l: usize, tok: usize) -> u64 {
        let (tid, nt) = (c.tid, c.nthreads);
        let lw = &self.w.layers[l];
        let epoch = (tok * N_LAYER + l + 1) as u64;
        if tid == 0 {
            let s = Instant::now();
            self.control_lookups(l, tok);
            ts.mgmt_ns += s.elapsed().as_nanos() as u64;
            self.work_epoch.store(epoch, Release);
        } else {
            while self.work_epoch.load(Acquire) < epoch {
                std::hint::spin_loop();
            }
        }
        let work = &*self.work.get();
        // activation: the router-phase input (recomputed redundantly, bounded)
        self.block_input(ts);
        quantize_act(ActType::Q8K, &ts.xn[..N_EMBD], &mut ts.a);
        let g = lw.exp;
        let (r0, r1) = part(FF_EXP, nt, tid, 4);
        let eh = self.eh.m();
        let mut done = [false; TOP_K];
        let mut left = TOP_K;
        let mut waited = 0u64;
        let mut wait_start: Option<Instant> = None;
        while left > 0 {
            let mut progressed = false;
            for j in 0..TOP_K {
                if done[j] || !work[j].ready() {
                    continue;
                }
                if let Some(ws) = wait_start.take() {
                    waited += ws.elapsed().as_nanos() as u64;
                }
                let p = work[j].ptr;
                let (gt, ut) = (&mut ts.tmp[..r1 - r0], &mut ts.tmp2[..r1 - r0]);
                match g.gate_w.r4() {
                    Some(t) => {
                        crate::ffi::gemv_r4(t, g.gate_w, p, N_EMBD, r0, r1, ts.a.as_ptr(), gt.as_mut_ptr());
                        crate::ffi::gemv_r4(t, g.gate_w, p.add(g.gate_bytes), N_EMBD, r0, r1, ts.a.as_ptr(), ut.as_mut_ptr());
                    }
                    None => {
                        crate::ffi::gemv_vecdot(g.gate_w, p, N_EMBD, r0, r1, ts.a.as_ptr(), gt.as_mut_ptr());
                        crate::ffi::gemv_vecdot(g.gate_w, p.add(g.gate_bytes), N_EMBD, r0, r1, ts.a.as_ptr(), ut.as_mut_ptr());
                    }
                }
                for r in 0..r1 - r0 {
                    let gg = gt[r];
                    eh[j * FF_EXP + r0 + r] = gg / (1.0 + (-gg).exp()) * ut[r];
                }
                done[j] = true;
                left -= 1;
                progressed = true;
            }
            if !progressed && wait_start.is_none() {
                wait_start = Some(Instant::now());
            }
            if !progressed {
                std::hint::spin_loop();
            }
        }
        if let Some(ws) = wait_start {
            waited += ws.elapsed().as_nanos() as u64;
        }
        waited
    }

    unsafe fn experts_down(&self, c: &Ctx, ts: &mut Ts, lw: &LayerW) {
        let (tid, nt) = (c.tid, c.nthreads);
        let work = &*self.work.get();
        let g = lw.exp;
        let eh = self.eh.r();
        let sh = self.sh.r();
        let qb = ActType::Q8_0.bytes(FF_EXP);
        for j in 0..TOP_K {
            // bounded dummy: normalise before quantising
            kf::oxe_rms_norm_f32(eh[j * FF_EXP..].as_ptr(), ts.tmp.as_mut_ptr(), FF_EXP as i64, EPS);
            quantize_act(ActType::Q8_0, &ts.tmp[..FF_EXP], &mut ts.a2[j * qb..(j + 1) * qb]);
        }
        kf::oxe_rms_norm_f32(sh.as_ptr(), ts.tmp.as_mut_ptr(), FF_EXP as i64, EPS);
        quantize_act(ActType::Q8_0, &ts.tmp[..FF_EXP], &mut ts.a);
        let (r0, r1) = part(N_EMBD, nt, tid, 8);
        let moe = self.moe.range(r0, r1);
        let t = &mut ts.tmp[..r1 - r0];
        lw.sh_down.gemv(r0, r1, &ts.a, &[], t);
        let sg = self.xin.r()[0];
        for r in 0..r1 - r0 {
            moe[r] = sg * t[r];
        }
        let logits = self.logits.r();
        for j in 0..TOP_K {
            let p = work[j].ptr.add(2 * g.gate_bytes);
            crate::ffi::gemv_vecdot(g.down_w, p, FF_EXP, r0, r1, ts.a2[j * qb..].as_ptr(), t.as_mut_ptr());
            let wj = 0.1 + 0.001 * logits[j].tanh();
            for r in 0..r1 - r0 {
                moe[r] += wj * t[r];
            }
        }
        let x = self.x.m();
        for s in 0..N_HC {
            for (r, i) in (r0..r1).enumerate() {
                x[s * N_EMBD + i] = 0.9 * x[s * N_EMBD + i] + 0.05 * moe[r];
            }
        }
    }

    unsafe fn head(&self, c: &mut Ctx, ts: &mut Ts, t0: Instant) {
        let (tid, nt) = (c.tid, c.nthreads);
        let hc = &self.w.out_hc;
        self.hc_phases(c, ts, t0, hc);
        let s = Instant::now();
        self.block_input(ts);
        quantize_act(ActType::Q8K, &ts.xn[..N_EMBD], &mut ts.a);
        let m = self.w.output;
        let (r0, r1) = part(m.n, nt, tid, 8);
        let mut best = (f32::MIN, 0u32);
        let chunk = 2048;
        let mut r = r0;
        while r < r1 {
            let e = (r + chunk).min(r1);
            m.gemv(r, e, &ts.a, &ts.xn[..N_EMBD], &mut ts.tmp[..e - r]);
            for (i, &v) in ts.tmp[..e - r].iter().enumerate() {
                if v > best.0 {
                    best = (v, (r + i) as u32);
                }
            }
            r = e;
        }
        ts.best = best;
        ts.prof.push((Kind::Head as u8, s.elapsed().as_nanos() as u64, 0, 0));
        // the pool's join barrier ends the phase; thread 0 records its wall after run()
    }

    /// After a token: the argmax over threads, thread 0's mgmt time.
    pub fn token_result(&self) -> (u32, u64) {
        let mut best = (f32::MIN, 0);
        for t in &self.ts {
            // SAFETY: the job has joined.
            let b = unsafe { (*t.get()).best };
            if b.0 > best.0 {
                best = b;
            }
        }
        (best.1, unsafe { (*self.ts[0].get()).mgmt_ns })
    }

    /// Per-token breakdown (ms), attributing each phase's wall time (thread 0's barrier exits)
    /// to: compute of the slowest thread, its SSD / N-gram wait, and sync (the rest).
    /// Returns per kind [compute, ssd, ngram, sync] and the number of phases.
    pub fn breakdown(&self, wall_ns: u64) -> ([[f64; 4]; KINDS], usize) {
        let mut out = [[0f64; 4]; KINDS];
        // SAFETY: the job has joined.
        let tss: Vec<&Ts> = self.ts.iter().map(|t| unsafe { &*t.get() }).collect();
        let n = tss[0].prof.len();
        let mut walls = tss[0].walls.clone();
        walls.push(wall_ns); // the head phase ends at the join
        assert_eq!(walls.len(), n, "phase count mismatch");
        let mut prev = 0u64;
        for i in 0..n {
            let wall = walls[i].saturating_sub(prev);
            prev = walls[i];
            let k = tss[0].prof[i].0 as usize;
            // slowest thread by work + waits
            let (mut crit, mut ssd, mut ng) = (0u64, 0u64, 0u64);
            for t in &tss {
                let (_, w, s, g) = t.prof[i];
                if w + s + g > crit {
                    crit = w + s + g;
                    ssd = s;
                    ng = g;
                }
            }
            let crit = crit.min(wall);
            let ssd = ssd.min(crit);
            let ng = ng.min(crit - ssd);
            out[k][0] += (crit - ssd - ng) as f64 / 1e6;
            out[k][1] += ssd as f64 / 1e6;
            out[k][2] += ng as f64 / 1e6;
            out[k][3] += (wall - crit) as f64 / 1e6;
        }
        (out, n)
    }

    /// Mean per-thread work (excl. waits) per kind, ms: the balance view.
    pub fn mean_work(&self) -> [f64; KINDS] {
        let mut out = [0f64; KINDS];
        for t in &self.ts {
            // SAFETY: the job has joined.
            for &(k, w, _, _) in unsafe { &(*t.get()).prof } {
                out[k as usize] += w as f64 / 1e6 / self.nt as f64;
            }
        }
        out
    }
}

#[allow(dead_code)]
fn _w(_: W) {}

#[cfg(test)]
mod tests {
    use super::part;
    #[test]
    fn partitions_cover_rows() {
        for &(n, a) in &[(320usize, 8usize), (10240, 8), (640, 4), (2560, 8), (248320, 8), (48, 1), (512, 1)] {
            for nt in [1, 2, 4, 6, 8, 12] {
                let mut next = 0;
                for t in 0..nt {
                    let (r0, r1) = part(n, nt, t, a);
                    assert_eq!(r0, next);
                    assert!(r0 % a == 0 && r1 % a == 0);
                    next = r1;
                }
                assert_eq!(next, n);
            }
        }
    }
}
