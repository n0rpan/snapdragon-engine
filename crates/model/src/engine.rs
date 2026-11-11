//! The qwen35moe decode engine (M1/M2): the static plan executed with real math.
//!
//! Every op mirrors llama.cpp's CPU graph for one token (see `csrc/oxe_ops.cpp` for the
//! ported f32 ops; the quantised GEMVs are ggml's own kernels on ggml's repacked layouts):
//!
//! layer il (delta-net):  RecMix  -B-  RecOut  -B-  MoE
//! layer il (attention):  AttnQkv -B-  AttnCore -B- AttnOut -B- MoE
//! MoE:                   MoeRouter -B- MoeGateUp -B- MoeDown -B-
//! after the last layer:  Head (pool join)
//!
//! M3.1 (MTP) adds **lanes**: one run may carry up to `MAX_LANES` consecutive tokens
//! (positions pos0, pos0+1, ...). Every lane goes through exactly the per-token arithmetic
//! of a one-token step (same kernels, same inputs, same summation order); lanes only share
//! the weight reads (GEMVs are interleaved over lanes per row chunk, and the MoE runs over
//! the union of the lanes' experts). So `verify(&[t0, t1, t2])` gives bit-identical logits
//! to three `step` calls; tests check it. Recurrent state lives in `n_slots` pre-allocated
//! slots: lane 0 updates the current slot in place, lane i writes slot cur+i, and `commit(m)`
//! makes the slot of lane m-1 current (an index change, no copy).
//!
//! Ownership invariants (all `unsafe` slice accesses rely on these):
//! - `ThreadState` (t, lane) is touched only by worker t (the main thread reads it only between runs).
//! - Shared buffers are written by a thread only in the rows/heads/channels/units its `Work`
//!   slice owns, and read by others only after the barrier that ends that phase.
//! - Recurrent state (conv rows, S of value heads) and KV rows are owned the same way; the K
//!   row of a position of a KV head is written by exactly one designated thread.
//! - Every thread computes the cheap redundant prologues itself (norms, quantisation, routing),
//!   all with identical inputs and code, so they agree bit for bit.

use crate::desc::{LayerKind, ModelDesc};
use crate::plan::{LayerTensors, Mixer, Phase, PhaseKind, Plan, Tid, Work};
use crate::weights::{Mapping, WeightRef, WeightStore};
use kernels_sys::{ffi, gemv_raw, ops, quantize_act, ActType, QType, QK_K};
use sched::{Ctx, Team};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::Mutex;

mod kv8;
mod prefill;
pub use kv8::{KvType, PvMode};
use kv8::Kv8Scratch;
pub use prefill::{group_by_size, pf_group, pf_group_for, pf_group_unaligned, set_pf_group_unaligned, pf_group_thresholds, pf_tiles, set_pf_group_thresholds, PF_GROUP_T1_DEFAULT, PF_TILES_DEFAULT, prefill_rows_hist, reset_prefill_rows_hist, set_pf_group, set_pf_tiles, PfProfile, PF_ATTN_PAD, PF_ATTN_PB, PF_ATTN_TT8, PF_CONV_FUSE, PF_DYN_SPLIT, PF_F32_CU8, PF_GDN_X4, PF_MMID_GEMV1, PF_MMID_TILE, PF_PHASES, PF_Q6K_TILE, PF_SUBS, PF_TILE_DENSE, PF_WAIT_DIET, UBATCH};

/// Default decode cutoff for the tiled attention (padded n_kv); below it the per-head path
/// (fewer barriers) is faster at short context. Codex: Qwen3.6 1K decode -12 % tiled.
pub const ATTN_TILED_MIN_KV: usize = 4096;

/// Most tokens one run can carry (verify width; MTP catch-up + draft lane).
pub const MAX_LANES: usize = 8;

// ------------------------------------------------------------------ timing

#[inline(always)]
pub fn ticks() -> u64 {
    let v: u64;
    // SAFETY: reading the virtual counter is allowed at EL0 on Linux/aarch64.
    unsafe { std::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) };
    v
}

pub fn tick_hz() -> u64 {
    let v: u64;
    // SAFETY: as above.
    unsafe { std::arch::asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack)) };
    v
}

/// Per-thread accumulators (ticks), summed over profiled tokens (runs: a verify run of n lanes
/// counts once). `phase_wait` includes the barriers inside a phase (attention's inner barriers);
/// `phase_bytes` are the bytes this thread asked for (weight rows once per run whatever the lane
/// count, KV positions x row bytes for its share of the heads, delta-net state per lane).
#[derive(Clone, Default, Debug)]
pub struct Prof {
    pub phase_work: [u64; 9],
    pub phase_wait: [u64; 9],
    pub tokens: u64,
    pub phase_bytes: [u64; 9],
    /// work ticks of sub-steps, see [`PROF_SUB_NAMES`]
    pub sub_work: [u64; 2],
}

/// OD-1: sub-steps inside a phase kind (work ticks; they sum to that phase's work).
pub const PROF_SUB_NAMES: [&str; 2] = ["deltanet_mix.gemv (qkv, z, beta, alpha rows)", "deltanet_mix.recur (conv, l2, gates, delta rule, norm)"];

/// What a decode phase kind is made of (for the OD-1 table).
pub fn phase_class(k: PhaseKind) -> &'static str {
    match k {
        PhaseKind::RecMix => "dense GEMV + delta-net",
        PhaseKind::RecOut | PhaseKind::AttnQkv | PhaseKind::AttnOut => "dense GEMV",
        PhaseKind::AttnCore => "attention (KV)",
        PhaseKind::MoeRouter => "router + shared gate/up",
        PhaseKind::MoeGateUp => "expert gate/up",
        PhaseKind::MoeDown => "expert down + shared down",
        PhaseKind::Head => "output head",
    }
}

// OD-1 per-thread profile counters of the phase in flight (only touched with `profile` on)
thread_local! {
    static PROF_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static PROF_IWAIT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static PROF_SUB0: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[inline]
fn prof_add(c: &'static std::thread::LocalKey<std::cell::Cell<u64>>, v: u64) {
    c.with(|x| x.set(x.get() + v));
}

#[inline]
fn prof_take(c: &'static std::thread::LocalKey<std::cell::Cell<u64>>) -> u64 {
    c.with(|x| x.replace(0))
}

// ------------------------------------------------------------------ buffers

struct Raw<T> {
    ptr: *mut T,
    len: usize,
}
// SAFETY: see the module-level ownership invariants.
unsafe impl<T> Send for Raw<T> {}
unsafe impl<T> Sync for Raw<T> {}
impl<T> Raw<T> {
    fn new(a: &Mapping, off_bytes: usize, len: usize) -> Raw<T> {
        assert!(off_bytes + len * std::mem::size_of::<T>() <= a.len());
        Raw { ptr: (a.base() as usize + off_bytes) as *mut T, len }
    }
    /// SAFETY: caller upholds the ownership invariants for `r`.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn at(&self, r: Range<usize>) -> &mut [T] {
        debug_assert!(r.end <= self.len && r.start <= r.end, "{r:?} of {}", self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(r.start), r.end - r.start)
    }
    #[inline]
    fn ptr_at(&self, i: usize) -> *mut T {
        debug_assert!(i <= self.len);
        // SAFETY: within the buffer.
        unsafe { self.ptr.add(i) }
    }
}

/// A shared buffer with one 64-byte aligned copy per lane.
struct LBuf<T> {
    raw: Raw<T>,
    stride: usize,
}
impl<T> LBuf<T> {
    /// SAFETY: as [`Raw::at`].
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn at(&self, ln: usize, r: Range<usize>) -> &mut [T] {
        debug_assert!(r.end <= self.stride);
        self.raw.at(ln * self.stride + r.start..ln * self.stride + r.end)
    }
    #[inline]
    fn ptr_at(&self, ln: usize, i: usize) -> *mut T {
        self.raw.ptr_at(ln * self.stride + i)
    }
}

/// Bump allocator over an arena (64-byte aligned pieces).
struct Carve {
    off: usize,
}
impl Carve {
    fn take<T>(&mut self, n: usize) -> (usize, usize) {
        let o = self.off.div_ceil(64) * 64;
        self.off = o + n * std::mem::size_of::<T>();
        (o, n)
    }
    /// `lanes` copies of `n` elements, each 64-byte aligned: (offset, total len, stride).
    fn lanes<T>(&mut self, n: usize, lanes: usize) -> (usize, usize, usize) {
        let per = (64 / std::mem::size_of::<T>()).max(1);
        let stride = n.div_ceil(per) * per;
        let (o, len) = self.take::<T>(stride * lanes);
        (o, len, stride)
    }
}

/// Activation scratch for both formats, quantised on demand from the current f32 vector.
struct Acts {
    q8k: Vec<u8>,
    q80: Vec<u8>,
    ok: [bool; 2],
}
impl Acts {
    fn new(n: usize) -> Acts {
        Acts { q8k: vec![0; ActType::Q8K.bytes(n)], q80: vec![0; ActType::Q8_0.bytes(n)], ok: [false; 2] }
    }
    fn invalidate(&mut self) {
        self.ok = [false; 2];
    }
    /// Activation for weight type `q` from `x` (quantised once per invalidation).
    fn get(&mut self, q: QType, x: &[f32]) -> *const u8 {
        let at = q.act();
        let i = (at == ActType::Q8_0) as usize;
        let buf = if i == 0 { &mut self.q8k } else { &mut self.q80 };
        if !self.ok[i] {
            quantize_act(at, x, &mut buf[..at.bytes(x.len())]);
            self.ok[i] = true;
        }
        buf.as_ptr()
    }
}

struct ThreadState {
    xin: Vec<f32>,
    xn: Vec<f32>,
    acts: Acts,
    /// recurrent: conv input / output channels (global channel index; own channels only)
    qkv: Vec<f32>,
    conv: Vec<f32>,
    z: Vec<f32>,
    zs: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    gate: Vec<f32>,
    beta_s: Vec<f32>,
    delta: Vec<f32>,
    gdn_out: Vec<f32>,
    tmpn: Vec<f32>,
    /// attention: q + gate rows (global index, own heads), roped q, K for the current position
    qg: Vec<f32>,
    q: Vec<f32>,
    kn: Vec<f32>,
    k16: Vec<u16>,
    q16: Vec<u16>,
    kq: Vec<f32>,
    mask: Vec<f32>,
    wp: Vec<f32>,
    p16: Vec<u16>,
    aout: Vec<f32>,
    gsig: Vec<f32>,
    /// int8 KV (O-13): one quantised row, the dequantisation scratch, output accumulators
    /// (empty in f16 mode)
    kq8: Vec<u8>,
    kv8s: Kv8Scratch,
    kv8acc: Vec<f32>,
    /// generic GEMV output scratch
    tmp: Vec<f32>,
    g: Vec<f32>,
    u: Vec<f32>,
    /// MoE
    probs: Vec<f32>,
    pwp: Vec<f32>,
    ids: Vec<i32>,
    iscr: Vec<i32>,
    w: Vec<f32>,
    sh_gate: f32,
    dtmp: Vec<f32>,
    /// MTP entry: [enorm(embd); hnorm(h)] (empty without MTP)
    cat: Vec<f32>,
    /// multi-lane MoE: union of the lanes' experts, and of their (expert, block) units
    uni_e: Vec<u32>,
    uni_u: Vec<(u32, u32)>,
    prof: Prof,
}

/// Debug capture of named intermediates (llama.cpp tensor names), enabled per engine.
#[derive(Default)]
pub struct Trace {
    bufs: Mutex<HashMap<String, Vec<f32>>>,
}
impl Trace {
    fn put(&self, name: &str, il: Option<usize>, total: usize, off: usize, data: &[f32]) {
        let key = match il {
            Some(l) => format!("{name}-{l}"),
            None => name.to_string(),
        };
        let mut m = self.bufs.lock().unwrap();
        let v = m.entry(key).or_insert_with(|| vec![f32::NAN; total]);
        v[off..off + data.len()].copy_from_slice(data);
    }
    pub fn take(&self) -> HashMap<String, Vec<f32>> {
        std::mem::take(&mut *self.bufs.lock().unwrap())
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// KV-cache capacity in positions (rounded up to a multiple of 256).
    pub n_ctx: usize,
    /// Keep the full logits of every step (KLD / debugging).
    pub logits: bool,
    pub trace: bool,
    pub profile: bool,
}

/// Optional engine features beyond one-token decode (M3.1).
#[derive(Clone, Copy, Default)]
pub struct EngineOptions<'a> {
    /// Tokens per `verify` run (and recurrent-state slots); 0 or 1 = plain decode only.
    pub verify_lanes: usize,
    /// The MTP head (adds one lane for the MTP catch-up + draft run).
    pub mtp: Option<&'a crate::mtp::MtpWeights>,
    /// KV cache element type; None: `OXE_KV_TYPE` (default f16).
    pub kv_type: Option<KvType>,
}

/// MTP head resources inside the engine.
struct MtpRt<'a> {
    w: &'a crate::mtp::MtpWeights,
    phases: Vec<Phase>,
    entry_rows: Vec<Range<usize>>,
}

/// What one run computes.
struct Run<'r> {
    phases: &'r [Phase],
    toks: &'r [u32],
    pos0: usize,
    /// lanes whose output-head logits/argmax are computed
    head_mask: u32,
    head_norm: Tid,
    mtp: bool,
    /// MTP KV seeding only (entry + K/V of the MTP layer; no attention, MoE or head)
    seed: bool,
}

/// Column padding of the decode attention scratch with OXE_DEC_ATTN_PAD=1 (elements; 13 cache
/// lines of f32), so the group's score / probability columns no longer alias one L1 set.
pub const DEC_ATT_PAD: usize = 208;

/// Test reference of one query head's attention: q (f32, roped), K / V of positions [0, n) as
/// f32 [position][dim], scale; writes the head_dim outputs (before the sigmoid gate).
#[cfg(test)]
pub(crate) type AttnRef = fn(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, scale: f32, out: &mut [f32]);

pub struct Engine<'a> {
    pub plan: &'a Plan,
    ws: &'a WeightStore,
    pub cfg: EngineConfig,
    pub n_ctx: usize,
    _state: Mapping,
    x2: LBuf<f32>,
    x1: LBuf<f32>,
    o: LBuf<f32>,
    kraw: LBuf<f32>,
    router: LBuf<f32>,
    hsh: LBuf<f32>,
    hsh_q: LBuf<u8>,
    h: LBuf<f32>,
    hq: LBuf<u8>,
    logits: LBuf<f32>,
    argmax: LBuf<(f32, u32)>,
    rope: LBuf<f32>,
    /// normalised hidden state after the last layer (llama.cpp `h_nextn`), per lane
    hout: LBuf<f32>,
    /// MTP input hidden state per lane
    hin: LBuf<f32>,
    conv_state: Raw<f32>,
    ssm_state: Raw<f32>,
    kcache: Raw<u16>,
    vcache: Raw<u16>,
    /// V^T cache: elements between consecutive rows (n_ctx + `OXE_KV_VPAD`) and per layer
    v_stride: usize,
    v_len: usize,
    /// KV element type (O-13). In int8 mode K is `kcache8` ([layer][kv head][n_ctx] Q8_0 rows,
    /// `k8_len` bytes per layer) instead of `kcache`, and V is `vcache8` (per position, same
    /// order, `v8_len` bytes per layer) instead of the V^T `vcache`; the replaced buffer is empty.
    kv: KvType,
    kcache8: Raw<u8>,
    vcache8: Raw<u8>,
    k8_len: usize,
    v8_len: usize,
    /// attention layers with KV (the MTP layer included)
    n_att: usize,
    /// int8 V: raw V rows per lane (f32), quantised after the AttnQkv barrier by the designated
    /// writer of the KV head (whole Q8_0 blocks; the V row split of the plan is by 8 rows)
    vraw: LBuf<f32>,
    /// int8 reads through the fused kernels (`oxe_dot_q8_tile` / `oxe_pv_q8`; `OXE_KV8_FUSED=0`:
    /// the f16-scratch first version)
    kv8_fused: bool,
    /// kernel of the fused int8 V . p (`OXE_KV8_PV`; `Fast` and `Ref` are bit-identical)
    kv8_pv: kv8::PvMode,
    /// fused int8 K: Q8_0 query columns (as `aq16`, row_bytes(head_dim) bytes each)
    aq8: Raw<u8>,
    /// fused int8 V: per (KV head, 256-position chunk) partial outputs [col][dim], reduced in
    /// chunk order after a barrier (thread-count invariant)
    vpart: Raw<f32>,
    /// test hook: int8 decode attention through the straightforward f32 reference
    #[cfg(test)]
    attn_ref: Option<AttnRef>,
    /// Tiled attention (M3-ATTN.md), columns in [kv head][lane][query head in group] order:
    /// f16 q (col * head_dim), sigmoid gate (col * head_dim), scores and f16 probabilities
    /// (col * n_ctx).
    aq16: Raw<u16>,
    agate: Raw<f32>,
    akq: Raw<f32>,
    ap16: Raw<u16>,
    /// column stride (elements) of akq / ap16 in the tiled decode attention: n_ctx, or
    /// n_ctx + DEC_ATT_PAD with OXE_DEC_ATTN_PAD=1 (with a power-of-two n_ctx the group's columns
    /// otherwise share one L1 set; the prefill twin is OXE_PF_TILES bit 4 on moe-prefill).
    /// Only addresses change: the results are bit-identical either way.
    acs: usize,
    /// OXE_ATTN_STATS=1: count 32-position probability blocks of the tiled decode attention
    /// (total, all exactly zero, all below 2^-10): skippable V^T . p work at long context
    att_stats: Option<[AtomicU64; 3]>,
    /// false: the per-head M1 attention path (`OXE_ATTN=legacy`; always used when tracing)
    attn_tiled: bool,
    /// OD-2: decode / verify delta-net recurrence through the fused 4-row step
    /// (`oxe_gated_delta_net_step_f32_x4`, prefill bit 64); bitwise equal. Default on; `OXE_DEC_FUSED_DN=0` = off.
    dec_fused_dn: bool,
    /// OD-6b: verify runs of 2+ lanes use the lane GEMVs (`kernels_sys::gemv_lanes_raw`: weight rows
    /// read and unpacked once for 2-4 lanes); bitwise equal to the GEMV per lane. Bits: [`VL_MV`]
    /// dense / router / shared / expert rows, [`VL_HEAD`] the output head, [`VL_Q6K_ONE`] one-lane
    /// Q6_K GEMVs (plain steps too), [`VL_DN_FROM`] delta-net step without the state copy. `OXE_VL=<mask>`, default 0.
    dec_vl: u32,
    /// decode / verify runs use the tiled attention only when the run's padded n_kv is at least
    /// this (`OXE_ATTN_TILED_MIN_KV`, default 4096); prefill always uses it. Both are exact.
    attn_min_kv: usize,
    /// [thread][lane]
    threads: Vec<Box<UnsafeCell<Vec<ThreadState>>>>,
    rec_index: Vec<usize>,
    att_index: Vec<usize>,
    pos: AtomicUsize,
    /// KV high-water mark: no position >= hw has been written since the last reset
    hw: AtomicUsize,
    pub trace: Option<Trace>,
    lanes: usize,
    n_slots: usize,
    cur_slot: AtomicUsize,
    /// last verify run: (pos0, slot of lane 0, lanes, lanes committed so far or 0)
    pend: [AtomicUsize; 4],
    mtp: Option<MtpRt<'a>>,
    conv_len_all: usize,
    ssm_len_all: usize,
    /// batched-prefill scratch, allocated on the first [`Engine::prefill`]
    /// prefill scratch: slot 0 for the decode team's thread count, slot 1 for any other team (the
    /// larger prefill team), so small turns on the decode team do not rebuild the big scratch
    pf: [Mutex<Option<Box<prefill::Pf>>>; 2],
}

// SAFETY: see the module-level ownership invariants; ThreadState cells are per-thread.
unsafe impl Sync for Engine<'_> {}

fn index_of(d: &ModelDesc, kind: LayerKind) -> Vec<usize> {
    let mut c = 0;
    d.layers
        .iter()
        .map(|k| {
            if *k == kind {
                c += 1;
                c - 1
            } else {
                usize::MAX
            }
        })
        .collect()
}

/// Recurrent state + position at one point of the sequence (see [`Engine::save_recurrent`]).
#[derive(Clone)]
pub struct RecurrentSnapshot {
    pub pos: usize,
    conv: Vec<f32>,
    ssm: Vec<f32>,
}

impl RecurrentSnapshot {
    pub fn bytes(&self) -> usize {
        4 * (self.conv.len() + self.ssm.len())
    }
}

/// Sequence state in llama.cpp's `llama_state_seq_get_data` element order, positions [0, n_pos):
/// per attention layer K as [pos][n_embd_k_gqa] and V^T as [n_embd_v_gqa][pos] (f16 bits);
/// per delta-net layer the conv state [channel][d_conv-1] and S [head][S_v*S_v] (f32).
pub struct StateDump {
    pub n_pos: usize,
    pub k: Vec<Vec<u16>>,
    pub v: Vec<Vec<u16>>,
    pub conv: Vec<Vec<f32>>,
    pub ssm: Vec<Vec<f32>>,
}

/// llama_kv_cache::get_n_kv: min(size, max(256, PAD(used_max_p1, 256))).
/// `OXE_KV_VPAD`: extra elements between V^T cache rows (default `KV_VPAD_DEFAULT`); read when an
/// engine is created. [`set_kv_vpad`] overrides the environment (tests, CLI).
/// Default V^T row pad: one cache line (32 f16), so the rows of a tile and the 512 rows a token's
/// V write touches fall in different L1 sets at a power-of-two n_ctx. `OXE_KV_VPAD=0` = stride n_ctx.
pub const KV_VPAD_DEFAULT: usize = 32;
static KV_VPAD: AtomicUsize = AtomicUsize::new(usize::MAX);
/// Decode attention scratch off the power-of-two n_ctx stride: ON by default since MoE candidate 2
/// (batch 1 G1: 26.38 -> 25.43 ms/token at 8K, token-identical); `OXE_DEC_ATTN_PAD=0` turns it off.
pub fn dec_attn_pad_default() -> bool {
    std::env::var("OXE_DEC_ATTN_PAD").map(|v| v.trim() != "0").unwrap_or(true)
}
/// Fused 4-row delta-net step in decode / verify: ON by default since MoE candidate 2 (bitwise
/// equal; with the pad 25.07 ms/token at 8K); `OXE_DEC_FUSED_DN=0` turns it off.
pub fn dec_fused_dn_default() -> bool {
    std::env::var("OXE_DEC_FUSED_DN").map(|v| v.trim() != "0").unwrap_or(true)
}
/// One line with the switches in effect for engines built now, and every OXE_* variable of the
/// environment (W4-5: goes into every result row and the server's start-up log).
pub fn effective_switches() -> String {
    let (t1, t2) = pf_group_thresholds();
    let group = match pf_group() {
        0 => format!("auto(t1={t1},t2={})", if t2 == 0 { "off".to_string() } else { t2.to_string() }),
        g => g.to_string(),
    };
    let mut env: Vec<String> = std::env::vars().filter(|(k, _)| k.starts_with("OXE_") && k != "OXE_MODEL_SHA256").map(|(k, v)| format!("{k}={v}")).collect();
    env.sort();
    format!(
        "pf_tiles={} pf_group={group} pf_group_unaligned={} dec_attn_pad={} dec_fused_dn={} dec_vl={} kv_vpad={} env=[{}]",
        pf_tiles(),
        pf_group_unaligned() as u8,
        dec_attn_pad_default() as u8,
        dec_fused_dn_default() as u8,
        std::env::var("OXE_VL").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0),
        kv_vpad(),
        env.join(" ")
    )
}
pub fn set_kv_vpad(n: usize) {
    KV_VPAD.store(n.min(4096), Relaxed);
}
pub fn kv_vpad() -> usize {
    let v = KV_VPAD.load(Relaxed);
    if v != usize::MAX {
        return v;
    }
    let n = std::env::var("OXE_KV_VPAD").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(KV_VPAD_DEFAULT).min(4096);
    KV_VPAD.store(n, Relaxed);
    n
}

pub fn padded_n_kv(pos: usize, n_ctx: usize) -> usize {
    n_ctx.min(256usize.max((pos + 1).div_ceil(256) * 256))
}

/// `OXE_VL` bit: lane GEMVs for the dense, router, shared-expert and routed-expert rows of a verify run.
pub const VL_MV: u32 = 1;
/// `OXE_VL` bit: lane GEMVs for the output head of a verify run.
pub const VL_HEAD: u32 = 2;
/// `OXE_VL` bit: every ONE-lane Q6_K GEMV (plain steps too: the output head and the shared experts
/// of Ornith / Qwen) through the lane kernel with one lane. Bitwise equal; the Q6_K lane kernel
/// folds the -32 offset into the codes and is faster than the reference GEMV even for one lane.
pub const VL_Q6K_ONE: u32 = 4;
/// `OXE_VL` bit: a verify lane's delta-net step reads the previous lane's state slot directly
/// (`oxe_gated_delta_net_step_f32*_from`) instead of copying the state into its slot first. Bitwise equal.
pub const VL_DN_FROM: u32 = 8;

/// Rows per lane-interleaved GEMV chunk: about 24 KiB of weights (stays in L1 across lanes).
#[inline]
fn lane_chunk(w: &WeightRef) -> usize {
    let rb = (w.bytes / w.n_expert / w.rows.max(1)).max(1);
    ((24 * 1024 / rb) / 8 * 8).max(8)
}

impl<'a> Engine<'a> {
    pub fn new(plan: &'a Plan, ws: &'a WeightStore, cfg: EngineConfig) -> Result<Engine<'a>, String> {
        Self::with_options(plan, ws, cfg, EngineOptions::default())
    }

    pub fn with_options(plan: &'a Plan, ws: &'a WeightStore, cfg: EngineConfig, opt: EngineOptions<'a>) -> Result<Engine<'a>, String> {
        let d = &plan.desc;
        let n = plan.nthreads;
        let n_ctx = cfg.n_ctx.max(1).div_ceil(256) * 256;
        let hd = d.head_dim;
        let att_width = d.n_head * hd;
        let hv = d.ssm_head_v_dim();
        let k_used = d.n_expert_used;
        let n_slots = opt.verify_lanes.max(1);
        // MTP engines get every lane for catch-up runs (prompt seeding); verify uses n_slots
        let lanes = if opt.mtp.is_some() { MAX_LANES.max(n_slots + 1) } else { n_slots };
        if n > 255 || d.ssm_conv_kernel > 16 {
            return Err("thread count or conv kernel beyond the engine limits".into());
        }
        if lanes > MAX_LANES {
            return Err(format!("{lanes} lanes > MAX_LANES {MAX_LANES}"));
        }
        let mut downs: Vec<(Tid, Tid)> = plan.layers.iter().map(|l| (l.down_exps, l.down_shexp)).collect();
        if let Some(m) = opt.mtp {
            downs.push((m.layer.down_exps, m.layer.down_shexp));
        }
        let get = |id: Tid| -> &WeightRef {
            if id >= crate::mtp::MTP_TID_BASE {
                return opt.mtp.and_then(|m| m.get(id)).expect("MTP tensor not loaded");
            }
            ws.refs[id].as_ref().expect("tensor not loaded")
        };
        let hq_blk = downs.iter().map(|&(a, _)| get(a).qtype().act().bytes(QK_K)).max().unwrap();
        let hsh_blk = downs.iter().map(|&(_, b)| get(b).qtype().act().bytes(QK_K)).max().unwrap();
        let n_rec = d.n_recurrent();
        let n_att = d.n_attention() + opt.mtp.is_some() as usize;
        let conv_len = (d.ssm_conv_kernel - 1) * d.ssm_conv_channels();
        let ssm_len = d.ssm_n_vheads * hv * hv;
        let kv = match opt.kv_type {
            Some(t) => t,
            None => KvType::from_env()?,
        };
        if kv.any8() && hd % kv8::QK != 0 {
            return Err(format!("int8 KV needs head_dim % 32 == 0 (head_dim {hd})"));
        }
        let kv8_len = d.n_head_kv * n_ctx * kv8::row_bytes(hd);
        let kv_len = if kv.k8() { 0 } else { d.n_head_kv * n_ctx * hd };
        // V^T row stride (elements). With a power-of-two n_ctx the rows of a tile (4 V^T rows read
        // together, n_ctx * 2 bytes apart) share one L1 set; `OXE_KV_VPAD=<elements>` (e.g. 32 =
        // one cache line) moves each row off its neighbours. Layout only: no result and no state
        // file changes (state files hold the first n positions of every row, not the stride).
        let v_stride = n_ctx + kv_vpad();
        let v_len = if kv.v8() { 0 } else { d.n_head_kv * hd * v_stride };

        let mut c = Carve { off: 0 };
        let x2 = c.lanes::<f32>(d.n_embd, lanes);
        let x1 = c.lanes::<f32>(d.n_embd, lanes);
        let o = c.lanes::<f32>(d.ssm_d_inner.max(att_width), lanes);
        let kraw = c.lanes::<f32>(d.n_head_kv * hd, lanes);
        let router = c.lanes::<f32>(d.n_expert, lanes);
        let hsh = c.lanes::<f32>(d.ff_shexp, lanes);
        let hsh_q = c.lanes::<u8>(d.ff_shexp / QK_K * hsh_blk, lanes);
        let h = c.lanes::<f32>(k_used * d.ff_exp, lanes);
        let hq = c.lanes::<u8>(k_used * d.ff_exp / QK_K * hq_blk, lanes);
        let logits = c.lanes::<f32>(if cfg.logits { d.n_vocab } else { 0 }, lanes);
        let argmax = c.lanes::<(f32, u32)>(n * 16, lanes);
        let rope = c.lanes::<f32>(hd, lanes);
        let hout = c.lanes::<f32>(d.n_embd, lanes);
        let hin = c.lanes::<f32>(d.n_embd, lanes);
        let conv_state = c.take::<f32>(n_slots * n_rec * conv_len);
        let ssm_state = c.take::<f32>(n_slots * n_rec * ssm_len);
        let kcache = c.take::<u16>(n_att * kv_len);
        let vcache = c.take::<u16>(n_att * v_len);
        let aq16 = c.take::<u16>(lanes * att_width);
        let agate = c.take::<f32>(lanes * att_width);
        let acs = if dec_attn_pad_default() { n_ctx + DEC_ATT_PAD } else { n_ctx };
        let akq = c.take::<f32>(lanes * d.n_head * acs);
        let ap16 = c.take::<u16>(lanes * d.n_head * acs);
        // int8 KV (after every f16-mode buffer, so the f16 arena layout is unchanged)
        let (k8_len, v8_len) = (if kv.k8() { kv8_len } else { 0 }, if kv.v8() { kv8_len } else { 0 });
        let kcache8 = c.take::<u8>(n_att * k8_len);
        let vcache8 = c.take::<u8>(n_att * v8_len);
        let vraw = c.lanes::<f32>(if kv.v8() { d.n_head_kv * hd } else { 0 }, lanes);
        let group = d.n_head / d.n_head_kv;
        let aq8 = c.take::<u8>(if kv.k8() { lanes * d.n_head * kv8::row_bytes(hd) } else { 0 });
        let vpart = c.take::<f32>(if kv.v8() { d.n_head_kv * (n_ctx / kv8::PV_CHUNK) * lanes * group * hd } else { 0 });
        let state = Mapping::anon(c.off, true).map_err(|e| format!("state arena: {e}"))?;
        let f = |p: (usize, usize)| Raw::<f32>::new(&state, p.0, p.1);
        let lf = |p: (usize, usize, usize)| LBuf { raw: Raw::<f32>::new(&state, p.0, p.1), stride: p.2 };
        let lu = |p: (usize, usize, usize)| LBuf { raw: Raw::<u8>::new(&state, p.0, p.1), stride: p.2 };
        let mtp_width = if opt.mtp.is_some() { 2 * d.n_embd } else { 0 };
        let wide = d.n_embd.max(d.ssm_d_inner).max(att_width).max(d.ff_exp).max(d.ff_shexp).max(mtp_width);
        let mut att_index = index_of(d, LayerKind::Attention);
        let mtp = opt.mtp.map(|w| {
            att_index.push(n_att - 1);
            let (phases, entry_rows) = crate::mtp::phases(d, n, w.draft.as_ref().map(|dh| dh.w.rows));
            MtpRt { w, phases, entry_rows }
        });
        let new_ts = || ThreadState {
            xin: vec![0.0; d.n_embd],
            xn: vec![0.0; d.n_embd],
            acts: Acts::new(wide),
            qkv: vec![0.0; d.ssm_conv_channels()],
            conv: vec![0.0; d.ssm_conv_channels()],
            z: vec![0.0; d.ssm_d_inner],
            zs: vec![0.0; hv],
            alpha: vec![0.0; d.ssm_n_vheads.div_ceil(8) * 8],
            beta: vec![0.0; d.ssm_n_vheads.div_ceil(8) * 8],
            gate: vec![0.0; d.ssm_n_vheads],
            beta_s: vec![0.0; d.ssm_n_vheads],
            delta: vec![0.0; hv],
            gdn_out: vec![0.0; hv],
            tmpn: vec![0.0; hv.max(hd)],
            qg: vec![0.0; 2 * att_width],
            q: vec![0.0; att_width],
            kn: vec![0.0; hd],
            k16: vec![0; d.n_head_kv * hd],
            q16: vec![0; hd],
            kq: vec![0.0; n_ctx],
            mask: vec![0.0; n_ctx],
            wp: vec![0.0; n_ctx],
            p16: vec![0; n_ctx],
            aout: vec![0.0; hd],
            gsig: vec![0.0; hd],
            kq8: vec![0; if kv.any8() { d.n_head_kv * kv8::row_bytes(hd) } else { 0 }],
            kv8s: if kv.any8() { Kv8Scratch::new(hd, lanes * (d.n_head / d.n_head_kv)) } else { Kv8Scratch::empty() },
            kv8acc: vec![0.0; if kv.v8() { kv8::QK * lanes * (d.n_head / d.n_head_kv) } else { 0 }],
            tmp: vec![0.0; d.n_embd.max(plan.head_chunk).max(2 * att_width).max(d.ssm_conv_channels())],
            g: vec![0.0; QK_K.max(d.ff_shexp)],
            u: vec![0.0; QK_K.max(d.ff_shexp)],
            probs: vec![0.0; d.n_expert],
            pwp: vec![0.0; d.n_expert],
            ids: vec![0; k_used],
            iscr: vec![0; d.n_expert],
            w: vec![0.0; k_used],
            sh_gate: 0.0,
            dtmp: vec![0.0; (k_used + 1) * d.n_embd],
            cat: vec![0.0; mtp_width],
            uni_e: Vec::with_capacity(MAX_LANES * k_used),
            uni_u: Vec::with_capacity(MAX_LANES * k_used * (d.ff_exp / QK_K)),
            prof: Prof::default(),
        };
        let e = Engine {
            plan,
            ws,
            n_ctx,
            v_stride,
            v_len,
            kv,
            kcache8: Raw::new(&state, kcache8.0, kcache8.1),
            vcache8: Raw::new(&state, vcache8.0, vcache8.1),
            k8_len,
            v8_len,
            n_att,
            vraw: lf(vraw),
            kv8_fused: kv8::fused_from_env(),
            kv8_pv: kv8::pv_mode_from_env(),
            aq8: Raw::new(&state, aq8.0, aq8.1),
            vpart: f(vpart),
            #[cfg(test)]
            attn_ref: None,
            x2: lf(x2),
            x1: lf(x1),
            o: lf(o),
            kraw: lf(kraw),
            router: lf(router),
            hsh: lf(hsh),
            hsh_q: lu(hsh_q),
            h: lf(h),
            hq: lu(hq),
            logits: lf(logits),
            argmax: LBuf { raw: Raw::new(&state, argmax.0, argmax.1), stride: argmax.2 },
            rope: lf(rope),
            hout: lf(hout),
            hin: lf(hin),
            conv_state: f(conv_state),
            ssm_state: f(ssm_state),
            kcache: Raw::new(&state, kcache.0, kcache.1),
            vcache: Raw::new(&state, vcache.0, vcache.1),
            aq16: Raw::new(&state, aq16.0, aq16.1),
            agate: Raw::new(&state, agate.0, agate.1),
            akq: Raw::new(&state, akq.0, akq.1),
            ap16: Raw::new(&state, ap16.0, ap16.1),
            acs,
            att_stats: std::env::var_os("OXE_ATTN_STATS").map(|_| [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]),
            attn_tiled: std::env::var("OXE_ATTN").map(|v| v != "legacy").unwrap_or(true),
            dec_fused_dn: dec_fused_dn_default(),
            dec_vl: std::env::var("OXE_VL").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            attn_min_kv: std::env::var("OXE_ATTN_TILED_MIN_KV").ok().and_then(|v| v.parse().ok()).unwrap_or(ATTN_TILED_MIN_KV),
            _state: state,
            threads: (0..n).map(|_| Box::new(UnsafeCell::new((0..lanes).map(|_| new_ts()).collect()))).collect(),
            rec_index: index_of(d, LayerKind::Recurrent),
            att_index,
            pos: AtomicUsize::new(0),
            hw: AtomicUsize::new(0),
            trace: if cfg.trace { Some(Trace::default()) } else { None },
            cfg,
            lanes,
            n_slots,
            cur_slot: AtomicUsize::new(0),
            pend: Default::default(),
            mtp,
            conv_len_all: n_rec * conv_len,
            ssm_len_all: n_rec * ssm_len,
            pf: [Mutex::new(None), Mutex::new(None)],
        };
        Ok(e)
    }

    /// Use the per-head M1 attention path instead of the tiled one (A/B and tests; the results
    /// are bit-identical). Also selected by `OXE_ATTN=legacy` at construction.
    pub fn set_attn_legacy(&mut self, legacy: bool) {
        self.attn_tiled = !legacy;
    }

    /// OD-2: decode / verify delta-net step through the fused 4-row kernel (bit-identical).
    /// Default on at construction (`OXE_DEC_FUSED_DN=0` = off).
    pub fn set_dec_fused_dn(&mut self, on: bool) {
        self.dec_fused_dn = on;
    }

    pub fn dec_fused_dn(&self) -> bool {
        self.dec_fused_dn
    }

    /// OD-6b: lane GEMVs for verify runs of 2+ lanes ([`VL_MV`] | [`VL_HEAD`]) and for one-lane Q6_K
    /// GEMVs ([`VL_Q6K_ONE`]); bit-identical.
    /// Also selected by `OXE_VL=<mask>` at construction (default 0).
    pub fn set_verify_lane_kernels(&mut self, mask: u32) {
        self.dec_vl = mask;
    }

    pub fn verify_lane_kernels(&self) -> u32 {
        self.dec_vl
    }

    /// Decode cutoff for the tiled attention (padded n_kv; 0 = always tiled).
    pub fn set_attn_tiled_min_kv(&mut self, n_kv: usize) {
        self.attn_min_kv = n_kv;
    }

    pub fn attn_tiled_min_kv(&self) -> usize {
        self.attn_min_kv
    }

    /// Whether decode / prefill use the tiled attention.
    /// OXE_ATTN_STATS counters: (32-position probability blocks, all exactly zero, all < 2^-10).
    pub fn attn_stats(&self) -> Option<(u64, u64, u64)> {
        self.att_stats.as_ref().map(|s| (s[0].load(Relaxed), s[1].load(Relaxed), s[2].load(Relaxed)))
    }
    /// Column stride of the decode attention scratch (n_ctx, or padded with OXE_DEC_ATTN_PAD=1).
    pub fn attn_col_stride(&self) -> usize {
        self.acs
    }
    pub fn attn_tiled(&self) -> bool {
        self.kv.any8() || (self.attn_tiled && self.trace.is_none())
    }

    /// KV cache element type (O-13).
    pub fn kv_type(&self) -> KvType {
        self.kv
    }

    /// Whether a decode / verify run whose last position is `last` takes the tiled attention:
    /// always with an int8 KV cache (the per-head C path reads f16 only; the cutoff is 0 then).
    #[inline]
    fn tiled_at(&self, last: usize) -> bool {
        self.kv.any8() || (self.attn_tiled && self.trace.is_none() && padded_n_kv(last, self.n_ctx) >= self.attn_min_kv)
    }

    /// Barriers per one-token step at the current position: the plan's, plus two inside each
    /// attention phase when the tiled path runs (n_kv >= the cutoff).
    pub fn barriers_per_token(&self) -> usize {
        let n_att = self.plan.desc.n_attention();
        let tiled = self.tiled_at(self.pos());
        // fused int8 V: one more barrier before the chunk-ordered reduction
        let per = 2 + (self.kv.v8() && self.kv8_fused) as usize;
        self.plan.barriers_per_token() + if tiled { per * n_att } else { 0 }
    }

    /// Test hook: run the int8 decode attention through `f` (a straightforward f32 reference
    /// over the stored, dequantised K / V values) instead of the tiled kernels.
    #[cfg(test)]
    pub(crate) fn set_attn_ref(&mut self, f: Option<AttnRef>) {
        self.attn_ref = f;
    }

    /// The stored K and V bytes of attention layer `ai` (MTP layer = the last) at positions
    /// < pos, in [kv head][position] order: Q8_0 rows in int8 mode, f16 K rows / V^T rows (each
    /// row's first pos elements) in f16 mode. For tests and debugging.
    pub fn kv_layer_bytes(&self, ai: usize) -> (Vec<u8>, Vec<u8>) {
        let d = &self.plan.desc;
        let (n, hd, rb) = (self.pos(), d.head_dim, kv8::row_bytes(d.head_dim));
        let mut k = Vec::new();
        let mut v = Vec::new();
        // SAFETY: between runs by the owner thread.
        unsafe {
            for h in 0..d.n_head_kv {
                if self.kv.k8() {
                    let at = ai * self.k8_len + h * self.n_ctx * rb;
                    k.extend_from_slice(self.kcache8.at(at..at + n * rb));
                } else {
                    let at = ai * d.n_head_kv * self.n_ctx * hd + h * self.n_ctx * hd;
                    k.extend(self.kcache.at(at..at + n * hd).iter().flat_map(|x| x.to_le_bytes()));
                }
                if self.kv.v8() {
                    let at = ai * self.v8_len + h * self.n_ctx * rb;
                    v.extend_from_slice(self.vcache8.at(at..at + n * rb));
                } else {
                    for j in 0..hd {
                        let at = self.vrow(ai, h * hd + j);
                        v.extend(self.vcache.at(at..at + n).iter().flat_map(|x| x.to_le_bytes()));
                    }
                }
            }
        }
        (k, v)
    }

    /// Stored K and V of positions [0, n) of one KV head as f32 ([position][dim] each): the
    /// dequantised Q8_0 rows (q * d) in int8 mode, the f16 values otherwise.
    pub fn kv_head_f32(&self, ai: usize, kvh: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
        let d = &self.plan.desc;
        let (hd, rb) = (d.head_dim, kv8::row_bytes(d.head_dim));
        let mut k = vec![0f32; n * hd];
        let mut v = vec![0f32; n * hd];
        // SAFETY: rows < n are complete (between runs, or after the barrier that ends their writes).
        unsafe {
            for p in 0..n {
                if self.kv.k8() {
                    let at = ai * self.k8_len + (kvh * self.n_ctx + p) * rb;
                    kv8::deq_f32(self.kcache8.at(at..at + rb), hd, &mut k[p * hd..(p + 1) * hd]);
                } else {
                    let at = ai * d.n_head_kv * self.n_ctx * hd + (kvh * self.n_ctx + p) * hd;
                    for (o, &x) in k[p * hd..(p + 1) * hd].iter_mut().zip(self.kcache.at(at..at + hd).iter()) {
                        *o = kv8::h2f(x);
                    }
                }
                if self.kv.v8() {
                    let at = ai * self.v8_len + (kvh * self.n_ctx + p) * rb;
                    kv8::deq_f32(self.vcache8.at(at..at + rb), hd, &mut v[p * hd..(p + 1) * hd]);
                } else {
                    for j in 0..hd {
                        v[p * hd + j] = kv8::h2f(*self.vcache.ptr_at(self.vrow(ai, kvh * hd + j) + p));
                    }
                }
            }
        }
        (k, v)
    }

    /// Byte offset of the Q8_0 K row (attention layer ai, KV head kvh, position p).
    #[inline]
    fn k8row(&self, ai: usize, kvh: usize, p: usize) -> usize {
        ai * self.k8_len + (kvh * self.n_ctx + p) * kv8::row_bytes(self.plan.desc.head_dim)
    }

    /// Byte offset of the Q8_0 V row (attention layer ai, KV head kvh, position p).
    #[inline]
    fn v8row(&self, ai: usize, kvh: usize, p: usize) -> usize {
        ai * self.v8_len + (kvh * self.n_ctx + p) * kv8::row_bytes(self.plan.desc.head_dim)
    }

    /// int8 V: quantise lane ln's raw V row of KV head kvh into the cache at `pos` (designated
    /// writer only; vraw complete after the AttnQkv barrier).
    fn store_v8(&self, ln: usize, ai: usize, kvh: usize, pos: usize) {
        let hd = self.plan.desc.head_dim;
        let rb = kv8::row_bytes(hd);
        let at = self.v8row(ai, kvh, pos);
        // SAFETY: vraw rows complete and read-only after the barrier; the designated writer of
        // (kvh, pos) is the only writer of that cache row and nobody reads it before the next barrier.
        unsafe { kv8::quant_row(self.vraw.at(ln, kvh * hd..(kvh + 1) * hd), self.vcache8.at(at..at + rb)) };
    }

    /// int8 K: quantise `kn` (normalised, roped) into slot `kvh` of `ts.kq8` (the bytes the
    /// fused scores of this run's own row use), store it at the row `at` when `store`, and leave
    /// in `k16[kvh]` the f16 values the first-version reads see.
    fn k8_from(&self, ts: &mut ThreadState, kvh: usize, store: Option<usize>) {
        let hd = self.plan.desc.head_dim;
        let rb = kv8::row_bytes(hd);
        let q = &mut ts.kq8[kvh * rb..(kvh + 1) * rb];
        kv8::quant_row(&ts.kn, q);
        if let Some(at) = store {
            // SAFETY: designated writer of this row; no reader before the next barrier.
            unsafe { self.kcache8.at(at..at + rb).copy_from_slice(q) };
        }
        kv8::deq_f32(q, hd, &mut ts.kn);
        ops::fp32_to_fp16(&ts.kn, &mut ts.k16[kvh * hd..(kvh + 1) * hd]);
    }

    /// Fused int8 kernels on / off (`OXE_KV8_FUSED`, read at construction): A/B and tests.
    pub fn set_kv8_fused(&mut self, on: bool) {
        self.kv8_fused = on;
    }

    /// Kernel of the fused int8 V . p (`OXE_KV8_PV`, read at construction): A/B and tests.
    pub fn set_kv8_pv(&mut self, mode: kv8::PvMode) {
        self.kv8_pv = mode;
    }

    /// Current position (number of tokens already in the state).
    pub fn pos(&self) -> usize {
        self.pos.load(Relaxed)
    }

    /// Tokens one `verify` run may carry.
    pub fn verify_lanes(&self) -> usize {
        self.n_slots
    }

    /// Lanes one `mtp_run` may carry (0 without an MTP head).
    pub fn mtp_lanes(&self) -> usize {
        if self.mtp.is_some() {
            self.lanes
        } else {
            0
        }
    }

    #[inline]
    fn w(&self, id: Tid) -> &WeightRef {
        if id >= crate::mtp::MTP_TID_BASE {
            return self.mtp.as_ref().and_then(|m| m.w.get(id)).expect("MTP tensor not loaded");
        }
        self.ws.refs[id].as_ref().expect("tensor not loaded")
    }

    #[inline]
    fn layer(&self, l: usize) -> &LayerTensors {
        if l < self.plan.layers.len() {
            &self.plan.layers[l]
        } else {
            &self.mtp.as_ref().expect("MTP layer without an MTP head").w.layer
        }
    }

    /// Forget everything (recurrent state, KV cache, position). Only V^T rows up to the
    /// high-water mark are zeroed (as a fresh cache, finite); K rows at or past the position
    /// are never weighted (the same invariant `restore_recurrent` relies on), so they are
    /// left as they are. A cold request no longer writes (and makes resident) the whole arena.
    pub fn reset(&self) {
        let hw = self.hw.swap(0, Relaxed).min(self.n_ctx);
        // SAFETY: called between runs by the owner thread.
        unsafe {
            self.conv_state.at(0..self.conv_state.len).fill(0.0);
            self.ssm_state.at(0..self.ssm_state.len).fill(0.0);
            if hw > 0 {
                // V^T rows are v_stride apart (n_ctx + OXE_KV_VPAD), not n_ctx
                for r in 0..self.vcache.len / self.v_stride {
                    self.vcache.at(r * self.v_stride..r * self.v_stride + hw).fill(0);
                }
            }
            // int8 V rows (per position; empty in f16 mode): zero as a fresh cache
            self.vcache8.at(0..self.vcache8.len).fill(0);
        }
        self.pos.store(0, Relaxed);
        self.cur_slot.store(0, Relaxed);
        self.pend[2].store(0, Relaxed);
    }

    fn slot_ranges(&self) -> (Range<usize>, Range<usize>) {
        let s = self.cur_slot.load(Relaxed);
        (s * self.conv_len_all..(s + 1) * self.conv_len_all, s * self.ssm_len_all..(s + 1) * self.ssm_len_all)
    }

    /// Save the recurrent state (conv + delta-net S of every layer) and the position. The KV
    /// cache is not copied: attention only reads positions < pos, so restoring a snapshot is
    /// valid as long as KV positions below its `pos` were not rewritten since.
    pub fn save_recurrent(&self) -> RecurrentSnapshot {
        let (cr, sr) = self.slot_ranges();
        // SAFETY: called between runs by the owner thread.
        unsafe { RecurrentSnapshot { pos: self.pos(), conv: self.conv_state.at(cr).to_vec(), ssm: self.ssm_state.at(sr).to_vec() } }
    }

    /// As [`Engine::save_recurrent`], reusing `old`'s buffers (an evicted checkpoint) so no
    /// 62.8 MiB allocation and page-fault pass is paid per checkpoint.
    pub fn save_recurrent_into(&self, old: RecurrentSnapshot) -> RecurrentSnapshot {
        let (cr, sr) = self.slot_ranges();
        let RecurrentSnapshot { mut conv, mut ssm, .. } = old;
        conv.clear();
        ssm.clear();
        // SAFETY: called between runs by the owner thread.
        unsafe {
            conv.extend_from_slice(self.conv_state.at(cr));
            ssm.extend_from_slice(self.ssm_state.at(sr));
        }
        RecurrentSnapshot { pos: self.pos(), conv, ssm }
    }

    /// Bytes [`Engine::write_state`] writes at the current position (capacity hint).
    pub fn state_bytes(&self) -> usize {
        let d = &self.plan.desc;
        let kv_len = d.n_head_kv * self.n_ctx * d.head_dim;
        let att = self.kcache.len / kv_len;
        att * 2 * d.n_head_kv * d.head_dim * self.pos() * 2 + 4 * (self.conv_len_all + self.ssm_len_all)
    }

    /// Note that KV positions < `end` are (about to be) written.
    pub(crate) fn note_kv(&self, end: usize) {
        self.hw.fetch_max(end, Relaxed);
    }

    /// Zero V^T of the target's attention layers at positions [a, b) (finite, as a fresh cache).
    fn zero_v(&self, a: usize, b: usize) {
        if b <= a {
            return;
        }
        let d = &self.plan.desc;
        if self.kv.v8() {
            // int8 V rows (scale + quants) of the rejected positions
            let rb = kv8::row_bytes(d.head_dim);
            for ai in 0..d.n_attention() {
                for kvh in 0..d.n_head_kv {
                    let at = self.v8row(ai, kvh, a);
                    // SAFETY: between runs; rows of the main attention layers.
                    unsafe { self.vcache8.at(at..at + (b - a) * rb).fill(0) };
                }
            }
            return;
        }
        let rows = d.n_attention() * d.n_head_kv * d.head_dim;
        for r in 0..rows {
            // SAFETY: between runs; rows of the main attention layers.
            unsafe { self.vcache.at(r * self.v_stride + a..r * self.v_stride + b).fill(0) };
        }
    }

    /// Go back to a saved position (see [`Engine::save_recurrent`]). V^T entries at positions
    /// >= the snapshot's are zeroed: they get probability 0 in attention anyway, but zero
    /// keeps them finite, as a fresh cache is.
    pub fn restore_recurrent(&self, s: &RecurrentSnapshot) {
        let (cr, sr) = self.slot_ranges();
        assert!(s.conv.len() == cr.len() && s.ssm.len() == sr.len(), "snapshot from another model");
        // SAFETY: called between runs by the owner thread.
        unsafe {
            self.conv_state.at(cr).copy_from_slice(&s.conv);
            self.ssm_state.at(sr).copy_from_slice(&s.ssm);
        }
        let cur = self.pos();
        if cur > s.pos {
            self.zero_v(s.pos, cur);
        }
        self.pos.store(s.pos, Relaxed);
        self.pend[2].store(0, Relaxed);
    }

    /// Shape of the persistent state ([`Engine::write_state`]): a state written by one engine
    /// can be read by another whose layout string is equal (n_ctx may differ).
    /// Element offset of V^T row `row` (= kv head * head_dim + dim) of attention layer `ai`.
    #[inline]
    fn vrow(&self, ai: usize, row: usize) -> usize {
        ai * self.v_len + row * self.v_stride
    }

    pub fn state_layout(&self) -> String {
        let d = &self.plan.desc;
        // f16: the string of every earlier build (existing prefix stores and session files stay
        // valid); int8 types name themselves (kv=k8|v8|kv8), so no file is read across types
        let kv = if self.kv.any8() { format!("kv={}", self.kv.name()) } else { "f16kv".to_string() };
        format!("att={} kv_heads={} head_dim={} conv={} ssm={} {kv}", self.n_att, d.n_head_kv, d.head_dim, self.conv_len_all, self.ssm_len_all)
    }

    /// Write the sequence state of positions < pos to `w` (the prefix store): per attention
    /// layer (MTP layer included when the engine has one) K rows then V^T rows, f16 bits, then
    /// the recurrent state (conv, S) of the current slot, f32; all little-endian, raw.
    pub fn write_state(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        fn bytes<T>(s: &[T]) -> &[u8] {
            // SAFETY: plain-old-data slices (u16 / f32) viewed as bytes.
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
        }
        let d = &self.plan.desc;
        let (n, hd, kvw) = (self.pos(), d.head_dim, d.n_head_kv * d.head_dim);
        let kv_len = d.n_head_kv * self.n_ctx * hd;
        let rb = kv8::row_bytes(hd);
        let (cr, sr) = self.slot_ranges();
        // SAFETY: called between runs by the owner thread.
        unsafe {
            for ai in 0..self.n_att {
                for h in 0..d.n_head_kv {
                    if self.kv.k8() {
                        let at = self.k8row(ai, h, 0);
                        w.write_all(self.kcache8.at(at..at + n * rb))?;
                    } else {
                        let at = ai * kv_len + h * self.n_ctx * hd;
                        w.write_all(bytes(self.kcache.at(at..at + n * hd)))?;
                    }
                }
                if self.kv.v8() {
                    for h in 0..d.n_head_kv {
                        let at = self.v8row(ai, h, 0);
                        w.write_all(self.vcache8.at(at..at + n * rb))?;
                    }
                } else {
                    for j in 0..kvw {
                        let at = self.vrow(ai, j);
                        w.write_all(bytes(self.vcache.at(at..at + n)))?;
                    }
                }
            }
            w.write_all(bytes(self.conv_state.at(cr)))?;
            w.write_all(bytes(self.ssm_state.at(sr)))?;
        }
        Ok(())
    }

    /// Read a state written by [`Engine::write_state`] at `n_pos` positions (same
    /// [`Engine::state_layout`]); the position becomes `n_pos`. On error the state is
    /// undefined: reset the engine.
    pub fn read_state(&self, r: &mut dyn std::io::Read, n_pos: usize) -> std::io::Result<()> {
        fn bytes<T>(s: &mut [T]) -> &mut [u8] {
            // SAFETY: plain-old-data slices (u16 / f32) viewed as bytes.
            unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
        }
        if n_pos > self.n_ctx {
            return Err(std::io::Error::other("state longer than the context"));
        }
        // rows < n_pos may be written even if the read fails midway: reset must zero them (W4-4 d)
        self.note_kv(n_pos);
        let d = &self.plan.desc;
        let (hd, kvw) = (d.head_dim, d.n_head_kv * d.head_dim);
        let kv_len = d.n_head_kv * self.n_ctx * hd;
        let rb = kv8::row_bytes(hd);
        let (cr, sr) = self.slot_ranges();
        let cur = self.pos();
        // SAFETY: called between runs by the owner thread.
        unsafe {
            for ai in 0..self.n_att {
                for h in 0..d.n_head_kv {
                    if self.kv.k8() {
                        let at = self.k8row(ai, h, 0);
                        r.read_exact(self.kcache8.at(at..at + n_pos * rb))?;
                    } else {
                        let at = ai * kv_len + h * self.n_ctx * hd;
                        r.read_exact(bytes(self.kcache.at(at..at + n_pos * hd)))?;
                    }
                }
                if self.kv.v8() {
                    for h in 0..d.n_head_kv {
                        let at = self.v8row(ai, h, 0);
                        r.read_exact(self.vcache8.at(at..at + n_pos * rb))?;
                    }
                } else {
                    for j in 0..kvw {
                        let at = self.vrow(ai, j);
                        r.read_exact(bytes(self.vcache.at(at..at + n_pos)))?;
                    }
                }
            }
            r.read_exact(bytes(self.conv_state.at(cr)))?;
            r.read_exact(bytes(self.ssm_state.at(sr)))?;
        }
        if cur > n_pos {
            self.zero_v(n_pos, cur);
        }
        self.note_kv(n_pos);
        self.pos.store(n_pos, Relaxed);
        self.pend[2].store(0, Relaxed);
        Ok(())
    }

    /// Copy of the sequence state (KV of positions < pos, recurrent state of the current
    /// slot) in llama.cpp's state-file order (see [`StateDump`]).
    pub fn dump_state(&self) -> StateDump {
        let d = &self.plan.desc;
        let n_pos = self.pos();
        let hd = d.head_dim;
        let kvw = d.n_head_kv * hd;
        let kv_len = d.n_head_kv * self.n_ctx * hd;
        let conv_len = (d.ssm_conv_kernel - 1) * d.ssm_conv_channels();
        let ssm_len = d.ssm_n_vheads * d.ssm_head_v_dim() * d.ssm_head_v_dim();
        let (cr, sr) = self.slot_ranges();
        // SAFETY: called between runs by the owner thread.
        unsafe {
            let kc = self.kcache.at(0..self.kcache.len);
            let vc = self.vcache.at(0..self.vcache.len);
            let mut k = Vec::new();
            let mut v = Vec::new();
            // int8 K / V parts are left empty here (see [`Engine::kv_layer_bytes`])
            for ai in 0..d.n_attention() {
                let mut kk = Vec::with_capacity(n_pos * kvw);
                for p in 0..n_pos {
                    for h in 0..d.n_head_kv {
                        if self.kv.k8() {
                            break;
                        }
                        let at = ai * kv_len + (h * self.n_ctx + p) * hd;
                        kk.extend_from_slice(&kc[at..at + hd]);
                    }
                }
                let mut vv = Vec::with_capacity(n_pos * kvw);
                for j in 0..kvw {
                    if self.kv.v8() {
                        break;
                    }
                    let at = self.vrow(ai, j);
                    vv.extend_from_slice(&vc[at..at + n_pos]);
                }
                k.push(kk);
                v.push(vv);
            }
            let cs = self.conv_state.at(cr);
            let ss = self.ssm_state.at(sr);
            let n_rec = d.n_recurrent();
            StateDump {
                n_pos,
                k,
                v,
                conv: (0..n_rec).map(|r| cs[r * conv_len..(r + 1) * conv_len].to_vec()).collect(),
                ssm: (0..n_rec).map(|r| ss[r * ssm_len..(r + 1) * ssm_len].to_vec()).collect(),
            }
        }
    }

    /// Full logits of the last step (lane 0; empty unless `cfg.logits`).
    pub fn logits(&self) -> &[f32] {
        self.logits_lane(0)
    }

    /// Full logits of lane `ln` of the last run.
    pub fn logits_lane(&self, ln: usize) -> &[f32] {
        // SAFETY: read between runs.
        unsafe { self.logits.at(ln, 0..if self.cfg.logits { self.plan.desc.n_vocab } else { 0 }) }
    }

    /// Normalised final hidden state (`h_nextn`) of lane `ln` of the last run: the target's
    /// output_norm(x) after a step/verify, the MTP's shared_head_norm(x) after an MTP run.
    pub fn hidden(&self, ln: usize) -> &[f32] {
        // SAFETY: read between runs.
        unsafe { self.hout.at(ln, 0..self.plan.desc.n_embd) }
    }

    fn rope_init(&self, ln: usize, pos: usize) {
        let d = &self.plan.desc;
        // SAFETY: no run in flight; lane ln's rope cache of head_dim floats.
        unsafe {
            ffi::oxe_mrope_cache_init(
                pos as i64,
                d.rope_sections.as_ptr(),
                1,
                d.n_rot as i32,
                d.n_rot as i64,
                d.rope_freq_base,
                1.0,
                0.0,
                1.0,
                32.0,
                1.0,
                d.n_ctx_train as i32,
                0,
                self.rope.ptr_at(ln, 0),
            );
        }
    }

    fn gather_argmax(&self, ln: usize) -> u32 {
        // SAFETY: all writers finished (join barrier).
        let am = unsafe { self.argmax.at(ln, 0..self.plan.nthreads * 16) };
        // threads own ascending row ranges: the first strict maximum over threads in order is
        // the first strict maximum in row order; nothing above -inf (all -inf or NaN) -> row 0,
        // as sampling::argmax
        let mut best = (f32::NEG_INFINITY, u32::MAX);
        for t in 0..self.plan.nthreads {
            let v = am[t * 16];
            if v.1 != u32::MAX && v.0 > best.0 {
                best = v;
            }
        }
        if best.1 == u32::MAX {
            0
        } else {
            best.1
        }
    }

    /// Run one token at the current position on all workers; returns the greedy argmax
    /// (lowest index among equal maxima, like a first-max scan over the vocabulary).
    pub fn step<P: Team + ?Sized>(&self, pool: &P, token: u32) -> u32 {
        let g = self.verify(pool, &[token]);
        self.commit(1);
        g[0]
    }

    /// Run `tokens` at positions pos..pos+n as one batch-invariant run (every lane computes
    /// exactly what a one-token `step` at its position would). Returns the argmax per lane.
    /// The position does not move until [`Engine::commit`].
    pub fn verify<P: Team + ?Sized>(&self, pool: &P, tokens: &[u32]) -> [u32; MAX_LANES] {
        assert_eq!(pool.nthreads(), self.plan.nthreads);
        let n = tokens.len();
        assert!(n >= 1 && n <= self.n_slots, "verify of {n} tokens (engine has {} lanes)", self.n_slots);
        let pos = self.pos.load(Relaxed);
        assert!(pos + n <= self.n_ctx, "context full ({} positions)", self.n_ctx);
        self.note_kv(pos + n);
        for (ln, _) in tokens.iter().enumerate() {
            self.rope_init(ln, pos + ln);
        }
        self.pend[0].store(pos, Relaxed);
        self.pend[1].store(self.cur_slot.load(Relaxed), Relaxed);
        self.pend[2].store(n, Relaxed);
        self.pend[3].store(0, Relaxed);
        let run = Run { phases: &self.plan.phases, toks: tokens, pos0: pos, head_mask: (1u32 << n) - 1, head_norm: self.plan.out_norm, mtp: false, seed: false };
        pool.run_job(&|ctx: &mut Ctx| self.thread_run(ctx, &run));
        let mut out = [u32::MAX; MAX_LANES];
        for (ln, o) in out.iter_mut().enumerate().take(n) {
            *o = self.gather_argmax(ln);
        }
        out
    }

    /// Keep the first `m` tokens of the last `verify` (m >= 1): the position becomes
    /// pos0 + m and the recurrent state is the one after lane m-1 (slot index change). May be
    /// called again with a smaller `m` until the next run (to drop tokens a caller did not use).
    pub fn commit(&self, m: usize) {
        let (pos0, cur0, n, prev) = (self.pend[0].load(Relaxed), self.pend[1].load(Relaxed), self.pend[2].load(Relaxed), self.pend[3].load(Relaxed));
        assert!(n > 0, "commit without a pending verify");
        let hi = if prev == 0 { n } else { prev };
        assert!(m >= 1 && m <= hi, "commit({m}) of {hi} lanes");
        self.zero_v(pos0 + m, pos0 + hi);
        self.pos.store(pos0 + m, Relaxed);
        self.cur_slot.store((cur0 + m - 1) % self.n_slots, Relaxed);
        self.pend[3].store(m, Relaxed);
    }

    /// Run the MTP head over `toks` at positions pos0.. with input hidden states `hs` (one
    /// per lane). Writes the MTP KV at those positions; `hidden(ln)` is its h_nextn. With
    /// `head_last`, returns the argmax of the last lane's logits (else u32::MAX).
    pub fn mtp_run<P: Team + ?Sized>(&self, pool: &P, toks: &[u32], pos0: usize, hs: &[&[f32]], head_last: bool) -> u32 {
        let m = self.mtp.as_ref().expect("engine has no MTP head");
        let n = toks.len();
        assert!(n >= 1 && n <= self.lanes && hs.len() == n);
        assert!(pos0 + n <= self.n_ctx);
        self.note_kv(pos0 + n);
        let ne = self.plan.desc.n_embd;
        for (ln, h) in hs.iter().enumerate() {
            // SAFETY: no run in flight.
            unsafe { self.hin.at(ln, 0..ne).copy_from_slice(&h[..ne]) };
            self.rope_init(ln, pos0 + ln);
        }
        let run = Run {
            phases: &m.phases,
            toks,
            pos0,
            head_mask: if head_last { 1u32 << (n - 1) } else { 0 },
            head_norm: m.w.head_norm,
            mtp: true,
            seed: false,
        };
        pool.run_job(&|ctx: &mut Ctx| self.thread_run(ctx, &run));
        if head_last {
            self.gather_argmax(n - 1)
        } else {
            u32::MAX
        }
    }

    /// Seed the MTP KV at positions pos0.. with the pairs (toks[i], hs[i]): only what later
    /// drafts read from these positions (eh_proj entry, attn_norm, K and V of the MTP layer),
    /// with the same code and arithmetic as [`Engine::mtp_run`], so the cache rows are the bits a
    /// full MTP run writes. No Q, attention, MoE or head. `hidden()` is not updated.
    pub fn mtp_seed<P: Team + ?Sized>(&self, pool: &P, toks: &[u32], pos0: usize, hs: &[&[f32]]) {
        let m = self.mtp.as_ref().expect("engine has no MTP head");
        let n = toks.len();
        assert!(n >= 1 && n <= self.lanes && hs.len() == n);
        assert!(pos0 + n <= self.n_ctx);
        self.note_kv(pos0 + n);
        let ne = self.plan.desc.n_embd;
        for (ln, h) in hs.iter().enumerate() {
            // SAFETY: no run in flight.
            unsafe { self.hin.at(ln, 0..ne).copy_from_slice(&h[..ne]) };
            self.rope_init(ln, pos0 + ln);
        }
        let run = Run { phases: &m.phases, toks, pos0, head_mask: 0, head_norm: m.w.head_norm, mtp: true, seed: true };
        pool.run_job(&|ctx: &mut Ctx| self.thread_run(ctx, &run));
    }

    pub fn profile_sum(&self) -> Vec<Prof> {
        // SAFETY: called by the owner thread between runs.
        self.threads.iter().map(|c| unsafe { (&*c.get())[0].prof.clone() }).collect()
    }
    pub fn reset_profile(&self) {
        for c in &self.threads {
            // SAFETY: as above.
            unsafe { (&mut *c.get())[0].prof = Prof::default() };
        }
    }

    #[inline]
    fn tr(&self, name: &str, il: Option<usize>, total: usize, off: usize, data: &[f32]) {
        if let Some(t) = &self.trace {
            t.put(name, il, total, off, data);
        }
    }

    fn thread_run(&self, ctx: &mut Ctx, run: &Run) {
        let tid = ctx.tid;
        // SAFETY: ThreadStates of `tid` are used only by this worker.
        let all = unsafe { &mut *self.threads[tid].get() };
        let tss = &mut all[..run.toks.len()];
        let prof = self.cfg.profile;
        if run.mtp {
            self.mtp_entry(tss, tid, run.toks);
            ctx.barrier();
        }
        if run.seed {
            // MTP KV seeding: the AttnQkv phase without Q, then the designated K writes
            let (Work::AttnQkv { k_rows, v_rows, .. }, Work::AttnCore { qheads }) = (&run.phases[0].work[tid], &run.phases[1].work[tid]) else {
                panic!("MTP phases: AttnQkv, AttnCore first")
            };
            let l = run.phases[0].layer;
            let lt = self.layer(l);
            for (ln, ts) in tss.iter_mut().enumerate() {
                self.layer_entry(ts, l, run.toks[ln], ln);
                self.norm(ts, None, lt.attn_norm);
            }
            self.attn_kv(tss, l, run.pos0, k_rows.clone(), v_rows.clone());
            ctx.barrier();
            self.write_k(tss, l, run.pos0, qheads.clone());
            return;
        }
        for phase in run.phases {
            let t0 = if prof { ticks() } else { 0 };
            let w = &phase.work[tid];
            let l = phase.layer;
            if prof {
                // counters of work outside the phases (MTP entry, prefill) are dropped
                prof_take(&PROF_BYTES);
                prof_take(&PROF_IWAIT);
                prof_take(&PROF_SUB0);
                self.prof_phase_bytes(phase.kind, w, l, run.pos0, tss.len());
            }
            match (phase.kind, w) {
                (PhaseKind::RecMix, Work::RecMix { kheads }) => {
                    for (ln, ts) in tss.iter_mut().enumerate() {
                        self.layer_entry(ts, l, run.toks[ln], ln);
                    }
                    self.rec_mix(tss, l, tid, kheads.clone());
                }
                (PhaseKind::RecOut, Work::RecOut { rows }) => self.mixer_out(tss, l, tid, rows.clone()),
                (PhaseKind::AttnQkv, Work::AttnQkv { qheads, k_rows, v_rows }) => {
                    for (ln, ts) in tss.iter_mut().enumerate() {
                        self.layer_entry(ts, l, run.toks[ln], ln);
                    }
                    self.attn_qkv(tss, l, tid, run.pos0, qheads.clone(), k_rows.clone(), v_rows.clone());
                }
                (PhaseKind::AttnCore, Work::AttnCore { qheads }) => self.attn_core(ctx, tss, l, run.pos0, qheads.clone()),
                (PhaseKind::AttnOut, Work::AttnOut { rows }) => self.mixer_out(tss, l, tid, rows.clone()),
                (PhaseKind::MoeRouter, Work::MoeRouter { rows, sh_rows }) => self.moe_router(tss, l, tid, rows.clone(), sh_rows.clone()),
                (PhaseKind::MoeGateUp, Work::MoeGateUp { units }) => self.moe_gate_up(tss, l, tid, units.clone()),
                (PhaseKind::MoeDown, Work::MoeDown { rows }) => self.moe_down(tss, l, rows.clone()),
                (PhaseKind::Head, Work::Head { rows }) => self.head(tss, tid, rows.clone(), run.head_mask, run.head_norm, run.mtp),
                (k, w) => panic!("plan mismatch: {k:?} with {w:?}"),
            }
            if prof {
                let t1 = ticks();
                if phase.barrier_after {
                    ctx.barrier();
                }
                let t2 = ticks();
                let k = phase.kind.index();
                let iw = prof_take(&PROF_IWAIT).min(t1 - t0);
                let p = &mut tss[0].prof;
                p.phase_work[k] += t1 - t0 - iw;
                p.phase_wait[k] += t2 - t1 + iw;
                p.phase_bytes[k] += prof_take(&PROF_BYTES);
                if phase.kind == PhaseKind::RecMix {
                    let g = prof_take(&PROF_SUB0).min(t1 - t0 - iw);
                    p.sub_work[0] += g;
                    p.sub_work[1] += t1 - t0 - iw - g;
                }
            } else if phase.barrier_after {
                ctx.barrier();
            }
        }
        tss[0].prof.tokens += 1;
    }

    /// OD-1: a barrier inside a phase; its wait is counted as wait, not work, when profiling.
    #[inline]
    fn ibar(&self, ctx: &mut Ctx) {
        if self.cfg.profile {
            let t = ticks();
            ctx.barrier();
            prof_add(&PROF_IWAIT, ticks() - t);
        } else {
            ctx.barrier();
        }
    }

    /// OD-1: bytes of `rows` weight rows (of one expert) read once.
    #[inline]
    fn prof_w(&self, w: &WeightRef, rows: usize) {
        if self.cfg.profile {
            let rb = w.bytes / w.n_expert.max(1) / w.rows.max(1);
            prof_add(&PROF_BYTES, (rows * rb) as u64);
        }
    }

    /// OD-1: non-weight bytes of this thread's share of a phase (KV rows, delta-net state).
    fn prof_phase_bytes(&self, kind: PhaseKind, w: &Work, l: usize, pos0: usize, lanes: usize) {
        let d = &self.plan.desc;
        match (kind, w) {
            (PhaseKind::AttnCore, Work::AttnCore { qheads }) => {
                // K and V of positions < pos0 + lanes, each row once for its query-head group
                let hd = d.head_dim;
                let kb = if self.kv.k8() { kv8::row_bytes(hd) } else { 2 * hd };
                let vb = if self.kv.v8() { kv8::row_bytes(hd) } else { 2 * hd };
                let total = d.n_head_kv * (pos0 + lanes) * (kb + vb);
                prof_add(&PROF_BYTES, (total * qheads.len() / d.n_head.max(1)) as u64);
            }
            (PhaseKind::RecMix, Work::RecMix { kheads }) => {
                // delta-net state (f32 hv x hv per value head) read per lane
                let (nk, nv, hv) = (d.ssm_n_kheads, d.ssm_n_vheads, d.ssm_head_v_dim());
                let vh = kheads.len() * (nv / nk.max(1));
                prof_add(&PROF_BYTES, (vh * hv * hv * 4 * lanes) as u64);
            }
            _ => {}
        }
        let _ = l;
    }

    /// Layer input: the token embedding (layer 0) or the previous layer's output.
    fn layer_entry(&self, ts: &mut ThreadState, l: usize, token: u32, ln: usize) {
        if l == 0 {
            self.embed(ts, token);
            if self.trace.is_some() {
                self.tr("model.input_embed", None, ts.xin.len(), 0, &ts.xin.clone());
            }
        } else {
            // SAFETY: x2 complete (MoeDown barrier) and read-only until the next MoeDown.
            ts.xin.copy_from_slice(unsafe { self.x2.at(ln, 0..self.plan.desc.n_embd) });
        }
    }

    fn embed(&self, ts: &mut ThreadState, token: u32) {
        let r = self.ws.get(self.plan.embd);
        let row = self.ws.embedding_row(self.plan, token as usize);
        kernels_sys::dequantize_row(r.qtype(), row, &mut ts.xin);
    }

    /// xn = rms_norm(src) * w; invalidates the activation cache.
    fn norm(&self, ts: &mut ThreadState, src: Option<&[f32]>, wid: usize) {
        let w = self.w(wid).f32_slice();
        let eps = self.plan.desc.rms_eps;
        match src {
            Some(s) => ops::rms_norm_mul(s, w, &mut ts.xn, eps),
            None => ops::rms_norm_mul(&ts.xin, w, &mut ts.xn, eps),
        }
        ts.acts.invalidate();
    }

    /// out[0..rows.len()) = W[rows] . x for expert `e`; `act` must be x quantised for W's type
    /// (ignored for F32 weights, which use ggml_vec_dot_f32 on `x`).
    #[inline]
    fn mv(&self, w: &WeightRef, e: usize, rows: Range<usize>, out: *mut f32, x: *const f32, act: *const u8) {
        if rows.is_empty() {
            return;
        }
        if w.is_f32() {
            for (i, r) in rows.enumerate() {
                // SAFETY: row r of expert e of a k-wide f32 tensor; x has k floats; out has rows.len() floats.
                unsafe {
                    let row = w.f32().add((e * w.rows + r) * w.k);
                    *out.add(i) = ffi::oxe_vec_dot_f32(w.k as i32, row, x);
                }
            }
        } else {
            let q = w.qtype();
            // SAFETY: rows within the tensor (plan coverage, multiples of the row group); the
            // activation holds k values in q's activation format; out has rows.len() floats.
            unsafe {
                if self.dec_vl & VL_Q6K_ONE != 0 && q == QType::Q6K && kernels_sys::gemv_lanes_raw(q, w.k, &[out], w.rows_ptr(e, rows.start), &[act], rows.len()) {
                    return;
                }
                gemv_raw(q, w.k, out, w.rows_ptr(e, rows.start), act, rows.len())
            }
        }
    }

    /// `mv` for `n` lanes over the same weight rows. With n > 1 the rows are walked in
    /// chunks (whole row groups) and every lane runs on a chunk while it is in cache. Each
    /// output row is one independent dot product, so the chunking never changes a value.
    #[inline]
    fn mv_lanes(&self, w: &WeightRef, e: usize, rows: Range<usize>, n: usize, outs: &[*mut f32], xs: &[*const f32], acts: &[*const u8]) {
        self.prof_w(w, rows.len());
        if n == 1 {
            self.mv(w, e, rows, outs[0], xs[0], acts[0]);
            return;
        }
        let vl = self.dec_vl & VL_MV != 0 && !w.is_f32();
        if vl && n <= kernels_sys::VL_MAX_LANES {
            // OD-6b: the rows are read and unpacked once for all lanes (bitwise the GEMV per lane).
            // SAFETY: as mv() for every lane; the lanes' outputs are distinct buffers.
            if unsafe { kernels_sys::gemv_lanes_raw(w.qtype(), w.k, outs, w.rows_ptr(e, rows.start), acts, rows.len()) } {
                return;
            }
        }
        let ch = lane_chunk(w);
        let mut r = rows.start;
        while r < rows.end {
            let c = ch.min(rows.end - r);
            let mut l0 = 0;
            while l0 < n {
                // more than VL_MAX_LANES lanes: lane groups of up to 4 on the chunk while it is in cache
                let g = if vl { (n - l0).min(kernels_sys::VL_MAX_LANES) } else { 1 };
                let mut go = [std::ptr::null_mut(); kernels_sys::VL_MAX_LANES];
                for (i, o) in go.iter_mut().enumerate().take(g) {
                    // SAFETY: outs[ln] has rows.len() floats.
                    *o = unsafe { outs[l0 + i].add(r - rows.start) };
                }
                // SAFETY: as above, for rows r..r+c.
                if g == 1 || !unsafe { kernels_sys::gemv_lanes_raw(w.qtype(), w.k, &go[..g], w.rows_ptr(e, r), &acts[l0..l0 + g], c) } {
                    for i in 0..g {
                        self.mv(w, e, r..r + c, go[i], xs[l0 + i], acts[l0 + i]);
                    }
                }
                l0 += g;
            }
            r += c;
        }
    }

    /// `mv_lanes` with every lane's activation taken from its ts.xn.
    fn mv_xn_lanes(&self, tss: &mut [ThreadState], w: &WeightRef, e: usize, rows: Range<usize>, out: impl Fn(usize, &mut ThreadState) -> *mut f32) {
        let n = tss.len();
        let mut outs = [std::ptr::null_mut(); MAX_LANES];
        let mut xs = [std::ptr::null(); MAX_LANES];
        let mut acts = [std::ptr::null(); MAX_LANES];
        for (ln, ts) in tss.iter_mut().enumerate() {
            acts[ln] = if w.is_f32() { std::ptr::null() } else { ts.acts.get(w.qtype(), &ts.xn) };
            xs[ln] = ts.xn.as_ptr();
            outs[ln] = out(ln, ts);
        }
        self.mv_lanes(w, e, rows, n, &outs[..n], &xs[..n], &acts[..n]);
    }

    // ---------------------------------------------------------------- MTP entry

    /// x2[lane] = eh_proj . [enorm(embd(tok)); hnorm(h)] (rows of this thread).
    fn mtp_entry(&self, tss: &mut [ThreadState], tid: usize, toks: &[u32]) {
        let m = self.mtp.as_ref().unwrap();
        let d = &self.plan.desc;
        let ne = d.n_embd;
        let (en, hn) = (self.w(m.w.enorm).f32_slice(), self.w(m.w.hnorm).f32_slice());
        for (ln, ts) in tss.iter_mut().enumerate() {
            self.embed(ts, toks[ln]);
            ops::rms_norm_mul(&ts.xin, en, &mut ts.cat[..ne], d.rms_eps);
            // SAFETY: hin written before the run, read-only during it.
            let h = unsafe { self.hin.at(ln, 0..ne) };
            ops::rms_norm_mul(h, hn, &mut ts.cat[ne..2 * ne], d.rms_eps);
            ts.acts.invalidate();
        }
        let w = *self.w(m.w.eh_proj);
        let rows = m.entry_rows[tid].clone();
        let n = tss.len();
        let mut outs = [std::ptr::null_mut(); MAX_LANES];
        let mut xs = [std::ptr::null(); MAX_LANES];
        let mut acts = [std::ptr::null(); MAX_LANES];
        for (ln, ts) in tss.iter_mut().enumerate() {
            acts[ln] = if w.is_f32() { std::ptr::null() } else { ts.acts.get(w.qtype(), &ts.cat) };
            xs[ln] = ts.cat.as_ptr();
            outs[ln] = self.x2.ptr_at(ln, rows.start);
        }
        self.mv_lanes(&w, 0, rows, n, &outs[..n], &xs[..n], &acts[..n]);
        for ts in tss.iter_mut() {
            ts.acts.invalidate();
        }
    }

    // ---------------------------------------------------------------- delta-net

    fn rec_mix(&self, tss: &mut [ThreadState], l: usize, tid: usize, kheads: Range<usize>) {
        let tg0 = if self.cfg.profile { ticks() } else { 0 };
        let d = &self.plan.desc;
        let lt = self.layer(l);
        let Mixer::Recurrent { qkv, z, beta, alpha, conv1d, a, dt_bias, norm, .. } = lt.mixer else { unreachable!() };
        for ts in tss.iter_mut() {
            self.norm(ts, None, lt.attn_norm);
        }
        if tid == 0 && self.trace.is_some() {
            self.tr("attn_norm", Some(l), d.n_embd, 0, &tss[0].xn.clone());
        }
        if kheads.is_empty() {
            return;
        }
        let (ds, nk, nv, hv) = (d.ssm_d_state, d.ssm_n_kheads, d.ssm_n_vheads, d.ssm_head_v_dim());
        let vpk = nv / nk;
        let eps = d.rms_eps;
        let nch = d.ssm_conv_channels();
        // channel ranges this thread owns: q of its key heads, k of its key heads, v of the
        // value heads kh + m*nk (llama.cpp: value head v reads key head v % nk)
        let mut ranges: [Range<usize>; 2 + 8] = Default::default();
        ranges[0] = kheads.start * ds..kheads.end * ds;
        ranges[1] = nk * ds + kheads.start * ds..nk * ds + kheads.end * ds;
        assert!(vpk <= 8);
        for m in 0..vpk {
            let (v0, v1) = (kheads.start + m * nk, kheads.end + m * nk);
            ranges[2 + m] = 2 * nk * ds + v0 * hv..2 * nk * ds + v1 * hv;
        }
        let nr = 2 + vpk;
        let wqkv = *self.w(qkv);
        let wz = *self.w(z);
        for r in ranges[..nr].iter().cloned() {
            self.mv_xn_lanes(tss, &wqkv, 0, r.clone(), |_, ts| ts.qkv[r.start..].as_mut_ptr());
        }
        for m in 0..vpk {
            let zr = (kheads.start + m * nk) * hv..(kheads.end + m * nk) * hv;
            self.mv_xn_lanes(tss, &wz, 0, zr.clone(), |_, ts| ts.z[zr.start..].as_mut_ptr());
        }
        // beta / alpha rows of our value heads (whole row groups for quantised weights)
        for (wid, dst_is_beta) in [(beta, true), (alpha, false)] {
            let w = *self.w(wid);
            for m in 0..vpk {
                let (v0, v1) = (kheads.start + m * nk, kheads.end + m * nk);
                let rows = if w.is_f32() {
                    v0..v1
                } else {
                    let g = w.qtype().nr();
                    (v0 / g * g)..(v1.div_ceil(g) * g).min(nv)
                };
                let s = rows.start;
                self.mv_xn_lanes(tss, &w, 0, rows, |_, ts| if dst_is_beta { ts.beta[s..].as_mut_ptr() } else { ts.alpha[s..].as_mut_ptr() });
            }
        }
        if self.cfg.profile {
            prof_add(&PROF_SUB0, ticks() - tg0);
        }
        // per lane, in order: conv (+ state update), delta rule. Lane 0 updates the current
        // slot in place; lane i copies its rows from lane i-1's slot into its own first.
        let ri = self.rec_index[l];
        let kc = d.ssm_conv_kernel;
        let conv_len = (kc - 1) * nch;
        let ssm_len = nv * hv * hv;
        let wconv = self.w(conv1d).f32_slice();
        let (wa, wdt) = (self.w(a).f32_slice(), self.w(dt_bias).f32_slice());
        let wn = self.w(norm).f32_slice();
        let cur = self.cur_slot.load(Relaxed);
        for (ln, ts) in tss.iter_mut().enumerate() {
            let dst = (cur + ln) % self.n_slots;
            let src = if ln == 0 { dst } else { (cur + ln - 1) % self.n_slots };
            let cbase = dst * self.conv_len_all + ri * conv_len;
            let cs = unsafe { self.conv_state.at(cbase..cbase + conv_len) };
            if src != dst {
                let sbase = src * self.conv_len_all + ri * conv_len;
                for r in ranges[..nr].iter() {
                    // SAFETY: conv-state rows of our channels (both slots) are owned by this thread.
                    let from = unsafe { self.conv_state.at(sbase + r.start * (kc - 1)..sbase + r.end * (kc - 1)) };
                    cs[r.start * (kc - 1)..r.end * (kc - 1)].copy_from_slice(from);
                }
            }
            for r in ranges[..nr].iter().cloned() {
                // SAFETY: r within the channel count; state rows r of kc-1 floats; weights rows of kc.
                unsafe {
                    ffi::oxe_ssm_conv_step_f32(
                        cs[r.start * (kc - 1)..].as_mut_ptr(),
                        ts.qkv[r.start..].as_ptr(),
                        wconv[r.start * kc..].as_ptr(),
                        r.len() as i32,
                        kc as i32,
                        ts.conv[r.start..].as_mut_ptr(),
                    );
                }
                let (a0, b0) = (r.start, r.end);
                let tmp = &mut ts.tmp[..b0 - a0];
                ops::silu(&ts.conv[a0..b0], tmp);
                ts.conv[a0..b0].copy_from_slice(tmp);
            }
            if self.trace.is_some() {
                for r in ranges[..nr].iter().cloned() {
                    self.tr("linear_attn_qkv_mixed", Some(l), nch, r.start, &ts.qkv[r.clone()]);
                    self.tr("conv_output_silu", Some(l), nch, r.start, &ts.conv[r.clone()]);
                }
                for m in 0..vpk {
                    let zr = (kheads.start + m * nk) * hv..(kheads.end + m * nk) * hv;
                    self.tr("z", Some(l), d.ssm_d_inner, zr.start, &ts.z[zr.clone()]);
                }
            }
            // l2 norm of q and k of our key heads
            for g in kheads.clone() {
                ops::gdn_l2_norm(&mut ts.conv[g * ds..(g + 1) * ds], eps);
                ops::gdn_l2_norm(&mut ts.conv[nk * ds + g * ds..nk * ds + (g + 1) * ds], eps);
            }
            if self.trace.is_some() {
                self.tr("q_conv_predelta", Some(l), nk * ds, ranges[0].start, &ts.conv[ranges[0].clone()]);
                self.tr("k_conv_predelta", Some(l), nk * ds, ranges[0].start, &ts.conv[ranges[1].clone()]);
            }
            // gates: g = softplus(alpha + dt_bias) * ssm_a ; beta = sigmoid(beta)
            for m in 0..vpk {
                for vh in kheads.start + m * nk..kheads.end + m * nk {
                    let ab = [ts.alpha[vh] + wdt[vh]];
                    let mut sp = [0f32];
                    ops::softplus(&ab, &mut sp);
                    ts.gate[vh] = sp[0] * wa[vh];
                    ts.beta_s[vh] = ops::sigmoid1(ts.beta[vh]);
                    if self.trace.is_some() {
                        self.tr("beta", Some(l), nv, vh, &[ts.beta[vh]]);
                        self.tr("alpha", Some(l), nv, vh, &[ts.alpha[vh]]);
                        self.tr("gate", Some(l), nv, vh, &[ts.gate[vh]]);
                        self.tr("beta_sigmoid", Some(l), nv, vh, &[ts.beta_s[vh]]);
                    }
                }
            }
            // delta rule per value head, then the gated RMSNorm with z
            for m in 0..vpk {
                for vh in kheads.start + m * nk..kheads.end + m * nk {
                    let kh = vh % nk;
                    let hb = ri * ssm_len + vh * hv * hv;
                    // SAFETY: the state of value head vh (every slot) is owned by this thread.
                    let s = unsafe { self.ssm_state.at(dst * self.ssm_len_all + hb..dst * self.ssm_len_all + hb + hv * hv) };
                    // the old state: this slot (lane 0), or the previous lane's slot; OD-6b bit 8 reads
                    // it from there in the step instead of copying it over first
                    let mut s_in = s.as_ptr();
                    if src != dst {
                        let from = unsafe { self.ssm_state.at(src * self.ssm_len_all + hb..src * self.ssm_len_all + hb + hv * hv) };
                        if self.dec_vl & VL_DN_FROM != 0 {
                            s_in = from.as_ptr();
                        } else {
                            s.copy_from_slice(from);
                        }
                    }
                    // SAFETY: q/k/v slices of hv/ds floats inside conv; scratch sized hv; s_in is the
                    // slot itself or another slot's hv * hv state of this head (owned by this thread).
                    unsafe {
                        let (q, k, v) = (ts.conv[kh * ds..].as_ptr(), ts.conv[nk * ds + kh * ds..].as_ptr(), ts.conv[2 * nk * ds + vh * hv..].as_ptr());
                        let (g, be, dl, at) = (ts.gate[vh], ts.beta_s[vh], ts.delta.as_mut_ptr(), ts.gdn_out.as_mut_ptr());
                        if std::ptr::eq(s_in, s.as_ptr()) {
                            // in place: the unchanged step functions
                            let step = if self.dec_fused_dn { ffi::oxe_gated_delta_net_step_f32_x4 } else { ffi::oxe_gated_delta_net_step_f32 };
                            step(s.as_mut_ptr(), q, k, v, g, be, hv as i64, dl, at);
                        } else {
                            let step = if self.dec_fused_dn { ffi::oxe_gated_delta_net_step_f32_x4_from } else { ffi::oxe_gated_delta_net_step_f32_from };
                            step(s.as_mut_ptr(), s_in, q, k, v, g, be, hv as i64, dl, at);
                        }
                    }
                    if self.trace.is_some() {
                        self.tr("attn_output", Some(l), nv * hv, vh * hv, &ts.gdn_out.clone());
                    }
                    ops::rms_norm_mul(&ts.gdn_out, wn, &mut ts.tmpn[..hv], eps);
                    ops::silu(&ts.z[vh * hv..(vh + 1) * hv], &mut ts.zs);
                    // SAFETY: O rows of value head vh are owned by this thread.
                    let o = unsafe { self.o.at(ln, vh * hv..(vh + 1) * hv) };
                    for j in 0..hv {
                        o[j] = ts.tmpn[j] * ts.zs[j];
                    }
                }
            }
        }
    }

    /// ssm_out / attn_output rows of this thread, plus the residual: x1 = out + x.
    fn mixer_out(&self, tss: &mut [ThreadState], l: usize, tid: usize, rows: Range<usize>) {
        let d = &self.plan.desc;
        let (wid, width, name) = match self.layer(l).mixer {
            Mixer::Recurrent { out, .. } => (out, d.ssm_d_inner, "linear_attn_out"),
            Mixer::Attention { out, .. } => (out, d.n_head * d.head_dim, "attn_output"),
        };
        let w = *self.w(wid);
        let n = tss.len();
        let mut outs = [std::ptr::null_mut(); MAX_LANES];
        let mut xs = [std::ptr::null(); MAX_LANES];
        let mut acts = [std::ptr::null(); MAX_LANES];
        for (ln, ts) in tss.iter_mut().enumerate() {
            // SAFETY: O is complete (previous barrier) and read-only in this phase.
            let o = unsafe { self.o.at(ln, 0..width) };
            ts.acts.invalidate();
            acts[ln] = if w.is_f32() { std::ptr::null() } else { ts.acts.get(w.qtype(), o) };
            xs[ln] = o.as_ptr();
            outs[ln] = ts.tmp.as_mut_ptr();
        }
        self.mv_lanes(&w, 0, rows.clone(), n, &outs[..n], &xs[..n], &acts[..n]);
        for (ln, ts) in tss.iter_mut().enumerate() {
            ts.acts.invalidate();
            // SAFETY: this thread owns rows of x1.
            let x1 = unsafe { self.x1.at(ln, rows.clone()) };
            for (i, r) in rows.clone().enumerate() {
                x1[i] = ts.tmp[i] + ts.xin[r];
            }
            if self.trace.is_some() {
                self.tr(name, Some(l), d.n_embd, rows.start, &ts.tmp[..rows.len()].to_vec());
                self.tr("attn_residual", Some(l), d.n_embd, rows.start, &x1.to_vec());
                if tid == 0 {
                    // SAFETY: as above.
                    self.tr("attn_out_input", Some(l), width, 0, unsafe { self.o.at(ln, 0..width) });
                }
            }
        }
    }

    // ---------------------------------------------------------------- attention

    #[allow(clippy::too_many_arguments)]
    fn attn_qkv(&self, tss: &mut [ThreadState], l: usize, tid: usize, pos0: usize, qheads: Range<usize>, k_rows: Range<usize>, v_rows: Range<usize>) {
        let d = &self.plan.desc;
        let lt = self.layer(l);
        let Mixer::Attention { q, q_norm, .. } = lt.mixer else { unreachable!() };
        for ts in tss.iter_mut() {
            self.norm(ts, None, lt.attn_norm);
        }
        if tid == 0 && self.trace.is_some() {
            self.tr("attn_norm", Some(l), d.n_embd, 0, &tss[0].xn.clone());
        }
        let hd = d.head_dim;
        let eps = d.rms_eps;
        let qr = qheads.start * 2 * hd..qheads.end * 2 * hd;
        let wq = *self.w(q);
        self.mv_xn_lanes(tss, &wq, 0, qr.clone(), |_, ts| ts.qg[qr.start..].as_mut_ptr());
        let wqn = self.w(q_norm).f32_slice();
        for (ln, ts) in tss.iter_mut().enumerate() {
            // SAFETY: rope cache written before the run, read-only.
            let cache = unsafe { self.rope.at(ln, 0..hd) };
            for h in qheads.clone() {
                let (a0, b0) = (h * 2 * hd, h * 2 * hd + hd);
                ops::rms_norm_mul(&ts.qg[a0..b0], wqn, &mut ts.q[h * hd..(h + 1) * hd], eps);
                if self.trace.is_some() {
                    self.tr("Qcur_normed", Some(l), d.n_head * hd, h * hd, &ts.q[h * hd..(h + 1) * hd].to_vec());
                }
                // SAFETY: q head of hd floats; cache of n_rot floats.
                unsafe { ffi::oxe_rope_neox_inplace(ts.q[h * hd..].as_mut_ptr(), d.n_rot as i32, cache.as_ptr()) };
                if self.attn_tiled || self.kv.any8() {
                    let c = self.acol(h, ln);
                    // SAFETY: column (lane ln, head h) is written only by the owner of query head h
                    // and read by the attention threads after the AttnQkv barrier.
                    unsafe {
                        ops::fp32_to_fp16(&ts.q[h * hd..(h + 1) * hd], self.aq16.at(c * hd..(c + 1) * hd));
                        ops::sigmoid(&ts.qg[h * 2 * hd + hd..(h + 1) * 2 * hd], self.agate.at(c * hd..(c + 1) * hd));
                        if self.kv.k8() {
                            // fused int8 scores: the query as Q8_0 (llama.cpp's q8_0 KV: the vec_dot_type)
                            let rb = kv8::row_bytes(hd);
                            kv8::quant_row(&ts.q[h * hd..(h + 1) * hd], self.aq8.at(c * rb..(c + 1) * rb));
                        }
                    }
                }
            }
            if self.trace.is_some() {
                self.tr("Qcur_full", Some(l), 2 * d.n_head * hd, qr.start, &ts.qg[qr.clone()].to_vec());
                self.tr("Qcur", Some(l), d.n_head * hd, qheads.start * hd, &ts.q[qheads.start * hd..qheads.end * hd].to_vec());
            }
        }
        self.attn_kv(tss, l, pos0, k_rows, v_rows);
    }

    /// Raw K rows of every lane into kraw (normalised and roped after the barrier, see
    /// [`Engine::write_k`]) and V rows straight into the (transposed) V cache at pos0 + lane.
    fn attn_kv(&self, tss: &mut [ThreadState], l: usize, pos0: usize, k_rows: Range<usize>, v_rows: Range<usize>) {
        let d = &self.plan.desc;
        let hd = d.head_dim;
        let Mixer::Attention { k, v, .. } = self.layer(l).mixer else { unreachable!() };
        // raw K rows (normalised and roped by the attention threads after the barrier)
        let wk = *self.w(k);
        self.mv_xn_lanes(tss, &wk, 0, k_rows.clone(), |ln, _| self.kraw.ptr_at(ln, k_rows.start));
        // V rows straight into the (transposed) V cache at `pos`
        if !v_rows.is_empty() {
            let wv = *self.w(v);
            if self.kv.v8() {
                // int8 V: raw rows (quantised whole after the barrier, see `store_v8`)
                self.mv_xn_lanes(tss, &wv, 0, v_rows.clone(), |ln, _| self.vraw.ptr_at(ln, v_rows.start));
                if self.trace.is_some() {
                    for ln in 0..tss.len() {
                        // SAFETY: this thread's rows of vraw.
                        self.tr("Vcur", Some(l), d.n_head_kv * hd, v_rows.start, unsafe { self.vraw.at(ln, v_rows.clone()) });
                    }
                }
                return;
            }
            self.mv_xn_lanes(tss, &wv, 0, v_rows.clone(), |_, ts| ts.tmp.as_mut_ptr());
            let ai = self.att_index[l];
            let mut h16 = [0u16; 1];
            for (ln, ts) in tss.iter_mut().enumerate() {
                let pos = pos0 + ln;
                for (i, r) in v_rows.clone().enumerate() {
                    let (kvh, dim) = (r / hd, r % hd);
                    ops::fp32_to_fp16(&ts.tmp[i..i + 1], &mut h16);
                    // SAFETY: this thread owns V rows `v_rows` at position `pos`.
                    unsafe { *self.vcache.ptr_at(self.vrow(ai, kvh * hd + dim) + pos) = h16[0] };
                }
                if self.trace.is_some() {
                    self.tr("Vcur", Some(l), d.n_head_kv * hd, v_rows.start, &ts.tmp[..v_rows.len()].to_vec());
                }
            }
        }
    }

    /// The designated threads (owners of query head kvh*group) normalise, rope and write the K
    /// rows of every lane at pos0 + lane (kraw complete: after the AttnQkv barrier).
    fn write_k(&self, tss: &mut [ThreadState], l: usize, pos0: usize, qheads: Range<usize>) {
        let d = &self.plan.desc;
        let Mixer::Attention { k_norm, .. } = self.layer(l).mixer else { unreachable!() };
        let hd = d.head_dim;
        let group = d.n_head / d.n_head_kv;
        let ai = self.att_index[l];
        let kv_len = d.n_head_kv * self.n_ctx * hd;
        let wkn = self.w(k_norm).f32_slice();
        for kvh in 0..d.n_head_kv {
            if !qheads.contains(&(kvh * group)) {
                continue;
            }
            for (ln, ts) in tss.iter_mut().enumerate() {
                // SAFETY: kraw complete (AttnQkv barrier); rope cache read-only.
                let kr = unsafe { self.kraw.at(ln, kvh * hd..(kvh + 1) * hd) };
                let cache = unsafe { self.rope.at(ln, 0..hd) };
                ops::rms_norm_mul(kr, wkn, &mut ts.kn, d.rms_eps);
                // SAFETY: hd floats; cache of n_rot floats.
                unsafe { ffi::oxe_rope_neox_inplace(ts.kn.as_mut_ptr(), d.n_rot as i32, cache.as_ptr()) };
                if self.kv.k8() {
                    self.k8_from(ts, kvh, Some(self.k8row(ai, kvh, pos0 + ln)));
                } else {
                    ops::fp32_to_fp16(&ts.kn, &mut ts.k16[..hd]);
                    let at = ai * kv_len + (kvh * self.n_ctx + pos0 + ln) * hd;
                    // SAFETY: designated writer of this row.
                    unsafe { self.kcache.at(at..at + hd).copy_from_slice(&ts.k16[..hd]) };
                }
                if self.kv.v8() {
                    self.store_v8(ln, ai, kvh, pos0 + ln);
                }
            }
        }
    }

    /// Column of (query head h, lane ln) in the tiled-attention buffers: [kv head][lane][h in group].
    #[inline]
    fn acol(&self, h: usize, ln: usize) -> usize {
        let group = self.plan.desc.n_head / self.plan.desc.n_head_kv;
        ((h / group) * self.lanes + ln) * group + h % group
    }

    fn attn_core(&self, ctx: &mut Ctx, tss: &mut [ThreadState], l: usize, pos0: usize, qheads: Range<usize>) {
        // the same decision on every thread (same run); both paths are bit-identical
        if self.tiled_at(pos0 + tss.len() - 1) {
            return self.attn_core_tiled(ctx, tss, l, pos0, qheads);
        }
        let d = &self.plan.desc;
        let lt = self.layer(l);
        let Mixer::Attention { k_norm, .. } = lt.mixer else { unreachable!() };
        let hd = d.head_dim;
        let group = d.n_head / d.n_head_kv;
        let ai = self.att_index[l];
        let kv_len = d.n_head_kv * self.n_ctx * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let wkn = self.w(k_norm).f32_slice();
        if tss.len() == 1 {
            let ts = &mut tss[0];
            let pos = pos0;
            let n_kv = padded_n_kv(pos, self.n_ctx);
            // SAFETY: rope cache read-only during the run.
            let cache = unsafe { self.rope.at(0, 0..hd) };
            let mut cur_kvh = usize::MAX;
            for h in qheads.clone() {
                let kvh = h / group;
                if kvh != cur_kvh {
                    cur_kvh = kvh;
                    // SAFETY: kraw complete (AttnQkv barrier), read-only here.
                    let kr = unsafe { self.kraw.at(0, kvh * hd..(kvh + 1) * hd) };
                    ops::rms_norm_mul(kr, wkn, &mut ts.kn, d.rms_eps);
                    if self.trace.is_some() && qheads.contains(&(kvh * group)) {
                        self.tr("Kcur_normed", Some(l), d.n_head_kv * hd, kvh * hd, &ts.kn.clone());
                    }
                    // SAFETY: hd floats; cache of n_rot floats.
                    unsafe { ffi::oxe_rope_neox_inplace(ts.kn.as_mut_ptr(), d.n_rot as i32, cache.as_ptr()) };
                    ops::fp32_to_fp16(&ts.kn, &mut ts.k16);
                    if qheads.contains(&(kvh * group)) {
                        // designated writer of this KV head's cache row at `pos`
                        let at = ai * kv_len + (kvh * self.n_ctx + pos) * hd;
                        // SAFETY: only the thread owning query head kvh*group writes this row.
                        unsafe { self.kcache.at(at..at + hd).copy_from_slice(&ts.k16[..hd]) };
                        if self.trace.is_some() {
                            self.tr("Kcur", Some(l), d.n_head_kv * hd, kvh * hd, &ts.kn.clone());
                        }
                    }
                }
                let kp = ts.k16.as_ptr();
                self.attn_head(ts, l, h, kvh, ai, kv_len, pos, n_kv, scale, 0, kp);
            }
            return;
        }
        // several lanes: the designated writers put every lane's K row into the cache, then
        // (after a barrier) each lane attends over the cache rows <= its position. The K
        // values are the same bits a one-token step computes locally.
        self.write_k(tss, l, pos0, qheads.clone());
        self.ibar(ctx);
        for (ln, ts) in tss.iter_mut().enumerate() {
            let pos = pos0 + ln;
            let n_kv = padded_n_kv(pos, self.n_ctx);
            for h in qheads.clone() {
                let kvh = h / group;
                let k_last = self.kcache.ptr_at(ai * kv_len + (kvh * self.n_ctx + pos) * hd) as *const u16;
                self.attn_head(ts, l, h, kvh, ai, kv_len, pos, n_kv, scale, ln, k_last);
            }
        }
    }

    /// Tiled attention (M3-ATTN.md): the same per-element arithmetic as [`Self::attn_head`]
    /// (every score and every output is one `ggml_vec_dot_f16`, the softmax row is
    /// `soft_max_f32` over the padded n_kv), but split over ALL threads by KV position (scores)
    /// and by V^T row (outputs), with each K / V^T row loaded once for all query heads of its
    /// KV head and all lanes. Three steps separated by two barriers inside the phase:
    /// [1] scores K^T q, [2] softmax rows -> f16 probabilities, [3] V^T p (+ sigmoid gate) -> O.
    fn attn_core_tiled(&self, ctx: &mut Ctx, tss: &mut [ThreadState], l: usize, pos0: usize, qheads: Range<usize>) {
        let d = &self.plan.desc;
        let lt = self.layer(l);
        let Mixer::Attention { k_norm, .. } = lt.mixer else { unreachable!() };
        let (tid, nth) = (ctx.tid, ctx.nthreads);
        let hd = d.head_dim;
        let nkv = d.n_head_kv;
        let group = d.n_head / nkv;
        let nl = tss.len();
        let ai = self.att_index[l];
        let kv_len = nkv * self.n_ctx * hd;
        let n_ctx = self.n_ctx;
        let scale = 1.0 / (hd as f32).sqrt();
        let wkn = self.w(k_norm).f32_slice();
        let last = pos0 + nl - 1;
        // K rows of this run's positions: computed by every thread (identical bits, as in the
        // per-head path), kept locally for the scores and stored by the designated writer (the
        // owner of query head kvh*group). Nobody reads these cache rows during this run.
        for (ln, ts) in tss.iter_mut().enumerate() {
            // SAFETY: rope cache read-only during the run.
            let cache = unsafe { self.rope.at(ln, 0..hd) };
            for kvh in 0..nkv {
                // SAFETY: kraw complete (AttnQkv barrier), read-only here.
                let kr = unsafe { self.kraw.at(ln, kvh * hd..(kvh + 1) * hd) };
                ops::rms_norm_mul(kr, wkn, &mut ts.kn, d.rms_eps);
                // SAFETY: hd floats; cache of n_rot floats.
                unsafe { ffi::oxe_rope_neox_inplace(ts.kn.as_mut_ptr(), d.n_rot as i32, cache.as_ptr()) };
                let owner = qheads.contains(&(kvh * group));
                if self.kv.k8() {
                    // int8: the local copy is what the cache row holds, f16(deq(quant(k)))
                    self.k8_from(ts, kvh, if owner { Some(self.k8row(ai, kvh, pos0 + ln)) } else { None });
                } else {
                    ops::fp32_to_fp16(&ts.kn, &mut ts.k16[kvh * hd..(kvh + 1) * hd]);
                    if owner {
                        let at = ai * kv_len + (kvh * n_ctx + pos0 + ln) * hd;
                        // SAFETY: designated writer of this row; no reader in this run.
                        unsafe { self.kcache.at(at..at + hd).copy_from_slice(&ts.k16[kvh * hd..(kvh + 1) * hd]) };
                    }
                }
                if self.kv.v8() && owner {
                    // read in step [3], after two barriers
                    self.store_v8(ln, ai, kvh, pos0 + ln);
                }
            }
        }
        #[cfg(test)]
        if let Some(f) = self.attn_ref {
            return self.attn_core_ref(ctx, tss, f, l, pos0, qheads);
        }
        // [1] scores: rows (kvh, p) for p <= last, split evenly over the threads
        let rows = last + 1;
        let my = rows * nkv * tid / nth..rows * nkv * (tid + 1) / nth;
        const RB: usize = 64;
        for kvh in 0..nkv {
            let a = my.start.max(kvh * rows).min((kvh + 1) * rows) - kvh * rows;
            let b = my.end.max(kvh * rows).min((kvh + 1) * rows) - kvh * rows;
            let c0 = self.acol(kvh * group, 0);
            let kbase = ai * kv_len + kvh * n_ctx * hd;
            let fused_k = self.kv.k8() && self.kv8_fused;
            if fused_k {
                let rb = kv8::row_bytes(hd);
                // fused int8: Q8_0 K rows x Q8_0 query columns (ggml vec_dot_q8_0_q8_0 per element)
                // SAFETY: cache rows < pos0 complete and read-only; aq8 complete (AttnQkv barrier);
                // score columns of rows a..b are written only by this thread; local rows are ours.
                unsafe {
                    if a < b.min(pos0) {
                        ffi::oxe_dot_q8_tile(
                            hd as i32,
                            self.kcache8.ptr_at(self.k8row(ai, kvh, a)),
                            rb as i64,
                            (b.min(pos0) - a) as i32,
                            self.aq8.ptr_at(c0 * rb),
                            rb as i64,
                            (nl * group) as i32,
                            self.akq.ptr_at(c0 * self.acs + a),
                            1,
                            self.acs as i64,
                        );
                    }
                    // this run's rows: the local Q8_0 copy (the bytes the cache row holds)
                    for p in a.max(pos0)..b {
                        let i = p - pos0;
                        ffi::oxe_dot_q8_tile(
                            hd as i32,
                            tss[i].kq8[kvh * rb..].as_ptr(),
                            rb as i64,
                            1,
                            self.aq8.ptr_at((c0 + i * group) * rb),
                            rb as i64,
                            ((nl - i) * group) as i32,
                            self.akq.ptr_at((c0 + i * group) * self.acs + p),
                            1,
                            self.acs as i64,
                        );
                    }
                }
                continue;
            }
            if self.kv.k8() && a < b.min(pos0) {
                // int8 cache rows (< pos0), first version: dequantised in chunks into this
                // thread's f16 scratch
                // SAFETY: as below; the scratch is this thread's.
                unsafe {
                    kv8::scores_q8(
                        self.kcache8.ptr_at(self.k8row(ai, kvh, 0)),
                        hd,
                        a,
                        b.min(pos0),
                        self.aq16.ptr_at(c0 * hd),
                        hd,
                        nl * group,
                        self.akq.ptr_at(c0 * self.acs + a),
                        self.acs,
                        &mut tss[0].kv8s,
                    );
                }
            }
            // cache rows (< pos0): every lane attends to them
            let mut p = if self.kv.k8() { b.min(pos0).max(a) } else { a };
            while p < b.min(pos0) {
                let pe = (p + RB).min(b.min(pos0));
                // SAFETY: cache rows < pos0 complete and read-only; q16 complete (AttnQkv
                // barrier); score columns of rows p..pe are written only by this thread.
                unsafe {
                    ffi::oxe_dot_f16_tile(
                        hd as i32,
                        self.kcache.ptr_at(kbase + p * hd),
                        hd as i64,
                        (pe - p) as i32,
                        self.aq16.ptr_at(c0 * hd),
                        hd as i64,
                        (nl * group) as i32,
                        self.akq.ptr_at(c0 * self.acs + p),
                        1,
                        self.acs as i64,
                        std::ptr::null(),
                    );
                }
                p = pe;
            }
            // this run's rows: row pos0+i is seen by lanes i..nl
            for p in a.max(pos0)..b {
                let i = p - pos0;
                // SAFETY: as above; the row is this thread's local copy.
                unsafe {
                    ffi::oxe_dot_f16_tile(
                        hd as i32,
                        tss[i].k16[kvh * hd..].as_ptr(),
                        hd as i64,
                        1,
                        self.aq16.ptr_at((c0 + i * group) * hd),
                        hd as i64,
                        ((nl - i) * group) as i32,
                        self.akq.ptr_at((c0 + i * group) * self.acs + p),
                        1,
                        self.acs as i64,
                        std::ptr::null(),
                    );
                }
            }
        }
        self.ibar(ctx);
        // [2] softmax rows (lane, head) -> f16 probabilities on [0, n32) (+0 beyond pos)
        let n32 = (last + 1).div_ceil(32) * 32;
        let nrow = nl * d.n_head;
        let ts0 = &mut tss[0];
        for r in nrow * tid / nth..nrow * (tid + 1) / nth {
            let (ln, h) = (r / d.n_head, r % d.n_head);
            let pos = pos0 + ln;
            let c = self.acol(h, ln);
            // SAFETY: scores of column c complete (barrier); probabilities of column c are
            // written only by this thread; wp / mask are this thread's scratch (>= n_ctx).
            unsafe {
                ffi::oxe_attn_softmax_row_f16(
                    self.akq.ptr_at(c * self.acs),
                    pos as i32,
                    padded_n_kv(pos, n_ctx) as i32,
                    scale,
                    ts0.wp.as_mut_ptr(),
                    ts0.mask.as_mut_ptr(),
                    self.ap16.ptr_at(c * self.acs),
                    n32 as i32,
                );
            }
            if let Some(st) = &self.att_stats {
                // SAFETY: column c's probabilities were just written by this thread.
                let pr = unsafe { std::slice::from_raw_parts(self.ap16.ptr_at(c * self.acs) as *const u16, n32) };
                let (mut z, mut t) = (0u64, 0u64);
                let full = (pos + 1) / 32; // blocks entirely at positions <= pos
                for b in pr.chunks_exact(32).take(full) {
                    let m = b.iter().map(|&x| x & 0x7fff).max().unwrap_or(0);
                    z += (m == 0) as u64;
                    t += (m < 0x1400) as u64; // < 2^-10
                }
                st[0].fetch_add(full as u64, Relaxed);
                st[1].fetch_add(z, Relaxed);
                st[2].fetch_add(t, Relaxed);
            }
        }
        self.ibar(ctx);
        // [3] outputs: V^T rows (kvh, dim) split evenly over the threads; all lanes and heads of
        //     the KV head per row; O = out * sigmoid(gate)
        if self.kv.v8() && self.kv8_fused {
            // fused int8 V: work items (KV head, 256-position chunk) split over the threads, so each
            // position's V row is read once per KV head; every item writes its partial [col][dim];
            // after a barrier each output is the f32 sum of its partials in chunk order (+0 first):
            // the same bits for any thread count, batching or replay.
            let ncol = nl * group;
            let nch = n32.div_ceil(kv8::PV_CHUNK);
            let all = n_ctx / kv8::PV_CHUNK;
            let cap = self.lanes * group * hd;
            let items = nkv * nch;
            for it in items * tid / nth..items * (tid + 1) / nth {
                let (kvh, ch) = (it / nch, it % nch);
                let (q0, q1) = (ch * kv8::PV_CHUNK, ((ch + 1) * kv8::PV_CHUNK).min(n32));
                let c0 = self.acol(kvh * group, 0);
                // SAFETY: V rows < pos0 complete since earlier runs, rows of this run quantised before
                // barrier [1], rows beyond pos finite with +0 probabilities; probabilities complete
                // (barrier); partial slot (kvh, ch) is written only by this item.
                unsafe {
                    kv8::pv_chunk(
                        self.kv8_pv,
                        self.vcache8.ptr_at(self.v8row(ai, kvh, 0)),
                        hd,
                        q0,
                        q1,
                        self.ap16.ptr_at(c0 * self.acs),
                        self.acs,
                        ncol,
                        self.vpart.ptr_at((kvh * all + ch) * cap),
                        &mut ts0.kv8s,
                    );
                }
            }
            self.ibar(ctx);
            for r in nkv * hd * tid / nth..nkv * hd * (tid + 1) / nth {
                let (kvh, dd) = (r / hd, r % hd);
                let c0 = self.acol(kvh * group, 0);
                for col in 0..ncol {
                    let mut acc = 0f32;
                    for ch in 0..nch {
                        // SAFETY: partials complete (barrier), read-only.
                        acc += unsafe { *self.vpart.ptr_at((kvh * all + ch) * cap + col * hd + dd) };
                    }
                    let (ln, h) = (col / group, kvh * group + col % group);
                    // SAFETY: gate complete; O element (lane ln, head h, dim dd) owned by this thread.
                    unsafe { *self.o.ptr_at(ln, h * hd + dd) = acc * *self.agate.ptr_at((c0 + col) * hd + dd) };
                }
            }
            return;
        }
        let mut nf = [0i32; MAX_LANES * 16];
        for (c, v) in nf.iter_mut().enumerate().take(nl * group) {
            *v = padded_n_kv(pos0 + c / group, n_ctx) as i32;
        }
        let my = nkv * hd * tid / nth..nkv * hd * (tid + 1) / nth;
        let ncol = nl * group;
        const DB: usize = 32;
        for kvh in 0..nkv {
            let a = my.start.max(kvh * hd).min((kvh + 1) * hd) - kvh * hd;
            let b = my.end.max(kvh * hd).min((kvh + 1) * hd) - kvh * hd;
            let c0 = self.acol(kvh * group, 0);
            if self.kv.v8() {
                // int8 V rows: per Q8_0 block of dims (at most 32 of them), positions [0, n32)
                let mut r = a;
                while r < b {
                    let re = b.min((r / kv8::QK + 1) * kv8::QK);
                    // SAFETY: V rows < pos0 complete since earlier runs, rows of this run quantised
                    // before barrier [1], rows beyond hold finite values (zero or quantised) with +0
                    // probabilities; the accumulators and scratch are this thread's.
                    unsafe {
                        kv8::outputs_q8(
                            self.vcache8.ptr_at(self.v8row(ai, kvh, 0)),
                            hd,
                            r,
                            re,
                            n32,
                            self.ap16.ptr_at(c0 * self.acs),
                            self.acs,
                            ncol,
                            ts0.kv8acc.as_mut_ptr(),
                            &mut ts0.kv8s,
                        );
                    }
                    for col in 0..ncol {
                        let (ln, g) = (col / group, col % group);
                        let h = kvh * group + g;
                        let gc = c0 + col;
                        for dd in r..re {
                            // SAFETY: as in the f16 path below.
                            unsafe { *self.o.ptr_at(ln, h * hd + dd) = ts0.kv8acc[(dd - r) * ncol + col] * *self.agate.ptr_at(gc * hd + dd) };
                        }
                    }
                    r = re;
                }
                continue;
            }
            let vbase = self.vrow(ai, kvh * hd);
            let mut r = a;
            while r < b {
                let re = (r + DB).min(b);
                let out = &mut ts0.tmp[..DB * ncol];
                // SAFETY: V^T rows < pos0 complete since earlier runs, rows of this run written
                // before the AttnQkv barrier, rows beyond read only for the tail rule (finite);
                // probabilities complete (barrier).
                unsafe {
                    ffi::oxe_dot_f16_tile(
                        n32 as i32,
                        self.vcache.ptr_at(vbase + r * self.v_stride),
                        self.v_stride as i64,
                        (re - r) as i32,
                        self.ap16.ptr_at(c0 * self.acs),
                        self.acs as i64,
                        ncol as i32,
                        out.as_mut_ptr(),
                        ncol as i64,
                        1,
                        nf.as_ptr(),
                    );
                }
                for col in 0..ncol {
                    let (ln, g) = (col / group, col % group);
                    let h = kvh * group + g;
                    let gc = c0 + col;
                    for dd in r..re {
                        // SAFETY: gate complete (AttnQkv barrier); O element (lane ln, head h,
                        // dim dd) is written only by the owner of V^T row (kvh, dd).
                        unsafe { *self.o.ptr_at(ln, h * hd + dd) = out[(dd - r) * ncol + col] * *self.agate.ptr_at(gc * hd + dd) };
                    }
                }
                r = re;
            }
        }
    }

    /// Test hook ([`Engine::set_attn_ref`]): every query head of this thread through the f32
    /// reference `f` over the stored K / V (dequantised), with the tiled path's barrier count.
    #[cfg(test)]
    fn attn_core_ref(&self, ctx: &mut Ctx, tss: &mut [ThreadState], f: AttnRef, l: usize, pos0: usize, qheads: Range<usize>) {
        let d = &self.plan.desc;
        let hd = d.head_dim;
        let group = d.n_head / d.n_head_kv;
        let ai = self.att_index[l];
        let scale = 1.0 / (hd as f32).sqrt();
        self.ibar(ctx); // this run's K / V rows complete
        for (ln, ts) in tss.iter_mut().enumerate() {
            let pos = pos0 + ln;
            for h in qheads.clone() {
                let (k, v) = self.kv_head_f32(ai, h / group, pos + 1);
                let mut q = ts.q[h * hd..(h + 1) * hd].to_vec();
                if self.kv.k8() && self.kv8_fused {
                    // the fused scores use the query as Q8_0: the reference sees the same values
                    let mut qq = vec![0u8; kv8::row_bytes(hd)];
                    kv8::quant_row(&q.clone(), &mut qq);
                    kv8::deq_f32(&qq, hd, &mut q);
                }
                f(&q, &k, &v, pos + 1, hd, scale, &mut ts.aout);
                ops::sigmoid(&ts.qg[h * 2 * hd + hd..(h + 1) * 2 * hd], &mut ts.gsig);
                // SAFETY: O rows of our query heads.
                let o = unsafe { self.o.at(ln, h * hd..(h + 1) * hd) };
                for j in 0..hd {
                    o[j] = ts.aout[j] * ts.gsig[j];
                }
            }
        }
        self.ibar(ctx);
        if self.kv.v8() && self.kv8_fused {
            self.ibar(ctx); // the fused V reduction's barrier (same count as the kernel path)
        }
    }

    /// One query head over the cache (positions < pos from the cache, `pos` from `k_last`).
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn attn_head(&self, ts: &mut ThreadState, l: usize, h: usize, kvh: usize, ai: usize, kv_len: usize, pos: usize, n_kv: usize, scale: f32, ln: usize, k_last: *const u16) {
        let d = &self.plan.desc;
        let hd = d.head_dim;
        let l_trace = self.trace.is_some();
        let kbase = ai * kv_len + kvh * self.n_ctx * hd;
        let vbase = self.vrow(ai, kvh * hd);
        // SAFETY: cache rows < pos are complete and read-only; row pos of K is k_last, row pos
        // of V^T was written by the V owners before the AttnQkv barrier; all scratch buffers
        // hold n_ctx >= n_kv entries.
        unsafe {
            ffi::oxe_attn_head_f16(
                ts.q[h * hd..].as_ptr(),
                self.kcache.ptr_at(kbase),
                hd as i64,
                k_last,
                self.vcache.ptr_at(vbase),
                self.v_stride as i64,
                n_kv as i32,
                pos as i32,
                hd as i32,
                scale,
                ts.q16.as_mut_ptr(),
                ts.kq.as_mut_ptr(),
                ts.mask.as_mut_ptr(),
                ts.wp.as_mut_ptr(),
                ts.p16.as_mut_ptr(),
                ts.aout.as_mut_ptr(),
            );
        }
        ops::sigmoid(&ts.qg[h * 2 * hd + hd..(h + 1) * 2 * hd], &mut ts.gsig);
        // SAFETY: O rows of our query heads.
        let o = unsafe { self.o.at(ln, h * hd..(h + 1) * hd) };
        for j in 0..hd {
            o[j] = ts.aout[j] * ts.gsig[j];
        }
        if l_trace {
            self.tr("kq_soft_max", Some(l), n_kv * d.n_head, h * n_kv, &ts.kq[..n_kv].to_vec());
            self.tr("attn_pregate", Some(l), d.n_head * hd, h * hd, &ts.aout.clone());
            self.tr("gate_sigmoid", Some(l), d.n_head * hd, h * hd, &ts.gsig.clone());
            self.tr("attn_gated", Some(l), d.n_head * hd, h * hd, &o.to_vec());
        }
    }

    // ---------------------------------------------------------------- MoE

    fn moe_router(&self, tss: &mut [ThreadState], l: usize, tid: usize, rows: Range<usize>, sh_rows: Range<usize>) {
        let d = &self.plan.desc;
        let lt = self.layer(l);
        for (ln, ts) in tss.iter_mut().enumerate() {
            // SAFETY: x1 complete (mixer barrier); read-only until the next mixer phase.
            let x1 = unsafe { self.x1.at(ln, 0..d.n_embd) };
            self.norm(ts, Some(x1), lt.post_norm);
        }
        if tid == 0 && self.trace.is_some() {
            self.tr("attn_post_norm", Some(l), d.n_embd, 0, &tss[0].xn.clone());
        }
        let wr = *self.w(lt.router);
        // this thread owns router logits `rows`
        self.mv_xn_lanes(tss, &wr, 0, rows.clone(), |ln, _| self.router.ptr_at(ln, rows.start));
        // shared-expert gate (redundant on every thread)
        let gs = self.w(lt.shexp_gate).f32_slice();
        for ts in tss.iter_mut() {
            let sg = ops::dot_f32(&gs[..d.n_embd], &ts.xn);
            ts.sh_gate = ops::sigmoid1(sg);
        }
        // shared expert gate/up rows -> h_shared slice
        if !sh_rows.is_empty() {
            let (wg, wu) = (*self.w(lt.gate_shexp), *self.w(lt.up_shexp));
            let n = sh_rows.len();
            self.mv_xn_lanes(tss, &wg, 0, sh_rows.clone(), |_, ts| ts.g.as_mut_ptr());
            self.mv_xn_lanes(tss, &wu, 0, sh_rows.clone(), |_, ts| ts.u.as_mut_ptr());
            for (ln, ts) in tss.iter_mut().enumerate() {
                // SAFETY: this thread owns h_shared rows `sh_rows`.
                let hs = unsafe { self.hsh.at(ln, sh_rows.clone()) };
                ops::swiglu(&ts.g[..n], &ts.u[..n], hs);
            }
        }
        if self.trace.is_some() {
            // SAFETY: own rows.
            self.tr("ffn_moe_logits", Some(l), d.n_expert, rows.start, unsafe { self.router.at(0, rows.clone()) });
            if tid == 0 {
                self.tr("shared_expert_gate_sigmoid", Some(l), 1, 0, &[tss[0].sh_gate]);
            }
        }
    }

    /// Routing (redundant, identical on every thread): softmax, top-k, normalised weights.
    fn route(&self, ts: &mut ThreadState, ln: usize) {
        let d = &self.plan.desc;
        // SAFETY: router logits complete (MoeRouter barrier), read-only here.
        let lg = unsafe { self.router.at(ln, 0..d.n_expert) };
        ops::soft_max(lg, &mut ts.probs, &mut ts.pwp);
        ops::argsort_top_k(&ts.probs, d.n_expert_used, &mut ts.ids, &mut ts.iscr);
        for j in 0..d.n_expert_used {
            ts.w[j] = ts.probs[ts.ids[j] as usize];
        }
        // ggml_sum_rows -> ggml_clamp(6.103515625e-5, INF) -> ggml_div
        let s = ops::sum(&ts.w);
        let s = s.min(f32::INFINITY).max(6.103515625e-5);
        for j in 0..d.n_expert_used {
            ts.w[j] /= s;
        }
    }

    /// Slot of expert `e` in lane `ts`'s top-k, if routed there.
    #[inline]
    fn slot_of(ts: &ThreadState, e: u32) -> Option<usize> {
        ts.ids.iter().position(|&x| x as u32 == e)
    }

    fn moe_gate_up(&self, tss: &mut [ThreadState], l: usize, tid: usize, units: Range<usize>) {
        let d = &self.plan.desc;
        let lt = self.layer(l);
        let n = tss.len();
        for (ln, ts) in tss.iter_mut().enumerate() {
            self.route(ts, ln);
        }
        if self.trace.is_some() && tid == 0 {
            let ts = &tss[0];
            self.tr("ffn_moe_probs", Some(l), d.n_expert, 0, &ts.probs.clone());
            let ids: Vec<f32> = ts.ids.iter().map(|&i| i as f32).collect();
            self.tr("ffn_moe_topk", Some(l), d.n_expert_used, 0, &ids);
            self.tr("ffn_moe_weights_norm", Some(l), d.n_expert_used, 0, &ts.w.clone());
        }
        let bpe = d.ff_exp / QK_K;
        let (wg, wu) = (*self.w(lt.gate_exps), *self.w(lt.up_exps));
        let down_at = self.w(lt.down_exps).qtype().act();
        let hq_blk = down_at.bytes(QK_K);
        // units: (expert, 256-row block). One lane: the plan's slot-major units. Several
        // lanes: the union of the lanes' experts (lane-major first use), split evenly; each
        // unit runs for every lane that routed to that expert, while its rows are in cache.
        if n > 1 {
            let k = d.n_expert_used;
            let (t0, rest) = tss.split_at_mut(1);
            let t0 = &mut t0[0];
            t0.uni_e.clear();
            t0.uni_u.clear();
            for ln in 0..n {
                for j in 0..k {
                    let e = if ln == 0 { t0.ids[j] } else { rest[ln - 1].ids[j] } as u32;
                    if !t0.uni_e.contains(&e) {
                        t0.uni_e.push(e);
                        for b in 0..bpe {
                            t0.uni_u.push((e, b as u32));
                        }
                    }
                }
            }
        }
        let n_units = if n == 1 { units.len() } else { tss[0].uni_u.len() };
        let nt = self.plan.nthreads;
        // same ranges as plan::split(n_units, nt)[tid], without the allocation
        let my = if n == 1 { units.clone() } else { n_units * tid / nt..n_units * (tid + 1) / nt };
        for u in my {
            let mut users = [(0usize, 0usize); MAX_LANES];
            let mut nu = 0;
            let (e, b) = if n == 1 {
                let (j, b) = (u / bpe, u % bpe);
                users[0] = (0, j);
                nu = 1;
                (tss[0].ids[j] as u32, b)
            } else {
                let (e, b) = tss[0].uni_u[u];
                for ln in 0..n {
                    if let Some(j) = Self::slot_of(&tss[ln], e) {
                        users[nu] = (ln, j);
                        nu += 1;
                    }
                }
                (e, b as usize)
            };
            let users = &users[..nu];
            let rows = b * QK_K..(b + 1) * QK_K;
            let mut og = [std::ptr::null_mut(); MAX_LANES];
            let mut ou = [std::ptr::null_mut(); MAX_LANES];
            let mut xs = [std::ptr::null(); MAX_LANES];
            let mut ag = [std::ptr::null(); MAX_LANES];
            let mut au = [std::ptr::null(); MAX_LANES];
            for (i, &(ln, _)) in users.iter().enumerate() {
                let ts = &mut tss[ln];
                ag[i] = if wg.is_f32() { std::ptr::null() } else { ts.acts.get(wg.qtype(), &ts.xn) };
                au[i] = if wu.is_f32() { std::ptr::null() } else { ts.acts.get(wu.qtype(), &ts.xn) };
                xs[i] = ts.xn.as_ptr();
                og[i] = ts.g.as_mut_ptr();
                ou[i] = ts.u.as_mut_ptr();
            }
            self.mv_lanes(&wg, e as usize, rows.clone(), nu, &og[..nu], &xs[..nu], &ag[..nu]);
            self.mv_lanes(&wu, e as usize, rows.clone(), nu, &ou[..nu], &xs[..nu], &au[..nu]);
            for &(ln, j) in users {
                let ts = &tss[ln];
                let off = j * d.ff_exp + b * QK_K;
                // SAFETY: unit (lane, j, b) owns h[off..off+256] and its activation block.
                let hb = unsafe { self.h.at(ln, off..off + QK_K) };
                ops::swiglu(&ts.g[..QK_K], &ts.u[..QK_K], hb);
                let qoff = (j * bpe + b) * hq_blk;
                // SAFETY: as above.
                let qb = unsafe { self.hq.at(ln, qoff..qoff + hq_blk) };
                quantize_act(down_at, hb, qb);
            }
        }
        // quantise the shared expert's h blocks (complete since the MoeRouter barrier)
        let sh_at = self.w(lt.down_shexp).qtype().act();
        let hsh_blk = sh_at.bytes(QK_K);
        for b in 0..d.ff_shexp / QK_K {
            if nt - 1 - (b % nt) == tid {
                for ln in 0..n {
                    // SAFETY: h_shared read-only here; block b's activation is owned by this thread.
                    let hs = unsafe { self.hsh.at(ln, b * QK_K..(b + 1) * QK_K) };
                    let qb = unsafe { self.hsh_q.at(ln, b * hsh_blk..(b + 1) * hsh_blk) };
                    quantize_act(sh_at, hs, qb);
                }
            }
        }
    }

    fn moe_down(&self, tss: &mut [ThreadState], l: usize, rows: Range<usize>) {
        let d = &self.plan.desc;
        let lt = self.layer(l);
        let k = d.n_expert_used;
        let nr = rows.len();
        let n = tss.len();
        if nr == 0 {
            return;
        }
        let wd = *self.w(lt.down_exps);
        let bpe = d.ff_exp / QK_K;
        let hq_blk = wd.qtype().act().bytes(QK_K);
        let nulls = [std::ptr::null(); MAX_LANES];
        if n == 1 {
            let ts = &mut tss[0];
            for j in 0..k {
                let e = ts.ids[j] as usize;
                let act = self.hq.ptr_at(0, j * bpe * hq_blk) as *const u8;
                let out = ts.dtmp[j * d.n_embd..].as_mut_ptr();
                // SAFETY: h activations complete (MoeGateUp barrier); see mv().
                self.prof_w(&wd, rows.len());
                self.mv(&wd, e, rows.clone(), out, std::ptr::null(), act);
            }
        } else {
            for x in 0..tss[0].uni_e.len() {
                let e = tss[0].uni_e[x];
                let mut outs = [std::ptr::null_mut(); MAX_LANES];
                let mut acts = [std::ptr::null(); MAX_LANES];
                let mut nu = 0;
                for ln in 0..n {
                    if let Some(j) = Self::slot_of(&tss[ln], e) {
                        outs[nu] = tss[ln].dtmp[j * d.n_embd..].as_mut_ptr();
                        acts[nu] = self.hq.ptr_at(ln, j * bpe * hq_blk) as *const u8;
                        nu += 1;
                    }
                }
                self.mv_lanes(&wd, e as usize, rows.clone(), nu, &outs[..nu], &nulls[..nu], &acts[..nu]);
            }
        }
        let wsd = *self.w(lt.down_shexp);
        let mut outs = [std::ptr::null_mut(); MAX_LANES];
        let mut acts = [std::ptr::null(); MAX_LANES];
        for (ln, ts) in tss.iter_mut().enumerate() {
            outs[ln] = ts.dtmp[k * d.n_embd..].as_mut_ptr();
            acts[ln] = self.hsh_q.ptr_at(ln, 0) as *const u8;
        }
        self.mv_lanes(&wsd, 0, rows.clone(), n, &outs[..n], &nulls[..n], &acts[..n]);
        for (ln, ts) in tss.iter_mut().enumerate() {
            // SAFETY: x1 read-only in this phase; this thread owns rows of x2.
            let x1 = unsafe { self.x1.at(ln, rows.clone()) };
            let x2 = unsafe { self.x2.at(ln, rows.clone()) };
            let sg = ts.sh_gate;
            for i in 0..nr {
                // ffn_moe_weighted (experts * weights) then the sequential ggml_add chain
                let mut m = ts.dtmp[i] * ts.w[0];
                for j in 1..k {
                    let p = ts.dtmp[j * d.n_embd + i] * ts.w[j];
                    m += p;
                }
                let s = ts.dtmp[k * d.n_embd + i] * sg; // ffn_shexp_gated
                let f = m + s; // ffn_out
                x2[i] = f + x1[i]; // post_moe / l_out
            }
            if self.trace.is_some() {
                let mut moe = vec![0f32; nr];
                let mut sh = vec![0f32; nr];
                let mut fo = vec![0f32; nr];
                for i in 0..nr {
                    let mut m = ts.dtmp[i] * ts.w[0];
                    for j in 1..k {
                        m += ts.dtmp[j * d.n_embd + i] * ts.w[j];
                    }
                    moe[i] = m;
                    sh[i] = ts.dtmp[k * d.n_embd + i];
                    fo[i] = m + sh[i] * sg;
                }
                self.tr("ffn_moe_out", Some(l), d.n_embd, rows.start, &moe);
                self.tr("ffn_shexp", Some(l), d.n_embd, rows.start, &sh);
                self.tr("ffn_out", Some(l), d.n_embd, rows.start, &fo);
                self.tr("l_out", Some(l), d.n_embd, rows.start, &x2.to_vec());
            }
        }
    }

    // ---------------------------------------------------------------- head

    /// `mtp`: an MTP draft run; with a restricted draft head (WP-D3b) the rows are that head's
    /// and the argmax maps back through its id list (no logits row: drafts need only the argmax).
    fn head(&self, tss: &mut [ThreadState], tid: usize, rows: Range<usize>, mask: u32, norm: Tid, mtp: bool) {
        let d = &self.plan.desc;
        for (ln, ts) in tss.iter_mut().enumerate() {
            // SAFETY: x2 of the last layer complete (MoeDown barrier).
            ts.xin.copy_from_slice(unsafe { self.x2.at(ln, 0..d.n_embd) });
            self.norm(ts, None, norm);
            if tid == 0 {
                // SAFETY: thread 0 writes h_nextn; read only between runs.
                unsafe { self.hout.at(ln, 0..d.n_embd).copy_from_slice(&ts.xn) };
            }
        }
        if tid == 0 && self.trace.is_some() {
            self.tr("result_norm", None, d.n_embd, 0, &tss[0].xn.clone());
        }
        let draft = if mtp { self.mtp.as_ref().and_then(|m| m.w.draft.as_ref()) } else { None };
        let w = match draft {
            Some(dh) => dh.w,
            None => *self.w(self.plan.output),
        };
        let mut best = [(f32::NEG_INFINITY, u32::MAX); MAX_LANES];
        let mut r = rows.start;
        while r < rows.end && mask != 0 {
            let c = self.plan.head_chunk.min(rows.end - r);
            self.prof_w(&w, c);
            // OD-6b: the chunk's rows for every lane of the mask at once (lane groups of up to 4)
            let mut done = 0u32;
            if self.dec_vl & VL_HEAD != 0 && mask & (mask - 1) != 0 {
                let mut go = [std::ptr::null_mut(); kernels_sys::VL_MAX_LANES];
                let mut ga = [std::ptr::null(); kernels_sys::VL_MAX_LANES];
                let (mut g, mut gm) = (0, 0u32);
                let last = 31 - mask.leading_zeros() as usize;
                for (ln, ts) in tss.iter_mut().enumerate() {
                    if mask & (1 << ln) == 0 {
                        continue;
                    }
                    ga[g] = ts.acts.get(w.qtype(), &ts.xn);
                    go[g] = ts.tmp.as_mut_ptr();
                    gm |= 1 << ln;
                    g += 1;
                    if g == kernels_sys::VL_MAX_LANES || ln == last {
                        // SAFETY: as mv() for every lane of the group; each lane writes its own ts.tmp (>= head_chunk floats).
                        if g > 1 && unsafe { kernels_sys::gemv_lanes_raw(w.qtype(), w.k, &go[..g], w.rows_ptr(0, r), &ga[..g], c) } {
                            done |= gm;
                        }
                        g = 0;
                        gm = 0;
                    }
                }
            }
            for (ln, ts) in tss.iter_mut().enumerate() {
                if mask & (1 << ln) == 0 {
                    continue;
                }
                if done & (1 << ln) == 0 {
                    let act = ts.acts.get(w.qtype(), &ts.xn);
                    self.mv(&w, 0, r..r + c, ts.tmp.as_mut_ptr(), ts.xn.as_ptr(), act);
                }
                let b = &mut best[ln];
                // first strict maximum above -inf in row order (NaN never wins): with the
                // thread-ordered gather this is exactly the server's sampling::argmax
                for i in 0..c {
                    if ts.tmp[i] > b.0 {
                        *b = (ts.tmp[i], (r + i) as u32);
                    }
                }
                if self.cfg.logits && draft.is_none() {
                    // SAFETY: this thread owns logits rows r..r+c.
                    unsafe { self.logits.at(ln, r..r + c).copy_from_slice(&ts.tmp[..c]) };
                }
                if self.trace.is_some() && draft.is_none() {
                    self.tr("result_output", None, d.n_vocab, r, &ts.tmp[..c].to_vec());
                }
            }
            r += c;
        }
        for (ln, b) in best.iter().enumerate().take(tss.len()) {
            // restricted draft head: local row -> token id (ids ascending, so the first maximum
            // is still the lowest id)
            let b = match draft {
                Some(dh) if b.1 != u32::MAX => (b.0, dh.ids[b.1 as usize]),
                _ => *b,
            };
            // SAFETY: argmax slot tid of lane ln is ours.
            unsafe { self.argmax.at(ln, tid * 16..tid * 16 + 1)[0] = b };
        }
    }
}
