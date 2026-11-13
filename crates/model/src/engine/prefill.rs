//! M3.2: batched prefill, bit-identical to llama.cpp's own batched prefill (native-layout build,
//! KleidiAI off, repack on, FA off, f16 KV) with the same ubatch size.
//!
//! A prompt is processed in ubatches of `ubatch` tokens (production: 256). Per ubatch every op
//! mirrors what llama.cpp's graph computes for `n_tokens = N` (see docs/lab-notes/M3-PREFILL.md):
//!
//! - Dense projections (repack `forward_mul_mat`): the activation rows are quantised as
//!   4-row interleaved groups (`ggml_quantize_mat_t`) for the first N - N%4 rows, which go
//!   through the i8mm GEMM (`gemm<>`); the remaining rows are quantised with `from_float` and go
//!   through the GEMV. F32 weights use `ggml_vec_dot_f32` per element: ggml-cpu.c undefines
//!   GGML_USE_LLAMAFILE when i8mm is enabled, so llama.cpp never takes the tinyBLAS path here.
//! - Routed experts (repack `forward_mul_mat_id`): every row is quantised with `from_float`;
//!   per expert, the routed rows (token-major, then slot) go 4 at a time through the MUL_MAT_ID
//!   gemm (`gemm_mmid`, bitwise equal to the GEMV for every type used); llama.cpp sends the
//!   remaining cnt % 4 rows through the GEMV, oxe pads them into one more tile (same bits).
//! - Delta-net layers: causal conv and the fused gated delta rule run token by token (the fused
//!   `GDN_CH` op of llama.cpp is the same sequential recurrence as decode).
//! - Attention (FA off): K^T Q and V^T P are `ggml_vec_dot_f16` per element (no llamafile, see
//!   above), so each (token, head) is the decode computation over the ubatch's 256-padded n_kv
//!   with that token's causal mask.
//! - Last layer of the last ubatch: only the output row goes through the residual, the FFN and
//!   the head (llama.cpp's `crop_before_nextn` with logits for the last token only), so that
//!   row's FFN is the decode (N = 1) computation.
//!
//! Work split: one `run_job` per ubatch; each step is split over the team (statically by rows /
//! channels, dynamically by weight-column units and experts) and ends with a barrier. No
//! output element depends on how the work is split, so the result is thread-count invariant.

use super::kv8::{self, Kv8Scratch, KvType};
use super::pfattn;
use super::*;
use crate::plan::Tid;
use kernels_sys::{gemm_mmid_raw, gemm_raw, interleave_rows4_raw, quantize_mat4_raw, quantize_row_raw};

/// moe-prefill kernel switches (bit mask; default [`PF_TILES_DEFAULT`]; 0 = the vendored llama.cpp kernels). Every bit
/// selects a kernel that is bitwise equal to the one it replaces, so the mask never changes a
/// result (gate: tools/run_prefill_verify.sh with `OXE_PF_TILES` set).
///   bit 0 (1): dense Q4_K projections through the multi-token tile GEMM (`gemm_tile_raw`)
///   bit 1 (2): tiled attention: V^T . p blocked by position (`oxe_dot_f16_tile_pb`,
///              `OXE_PF_ATTN_PB` positions per block, default 512)
///   bit 3 (8): routed experts: a single leftover row (and two for Q4_K) goes through the GEMV
///              instead of a padded 4-row tile (the GEMV is what llama.cpp runs for these rows;
///              one GEMV row costs ~2 tile rows, a padded tile 4: kernel timings in M3-MOEPF.md)
///   bit 4 (16): routed experts (Q3_K, Q4_K, Q5_K): an expert's rows packed in pairs and run in
///              tiles of up to 8 tokens x 4 weight rows with the weights decoded once per 64-row
///              chunk (`mmid_tile_raw`); per row count as `mmid_use_tile` decides (3-4 rows of
///              Q3_K / Q4_K stay on the 4-row tile, 1 row on the GEMV with bit 3)
///   bit 5 (32): tiled attention: 8 query tokens per unit (64 columns share every K / V^T row
///              pass) instead of 4; read when the prefill scratch is (re)built
///   bit 6 (64): delta-net recurrence through the fused 4-row step (`oxe_gated_delta_net_step_f32_x4`:
///              two passes over the 64 KB state per token instead of four, 16 dot chains at once)
///   bit 7 (128): F32 projections (the router; Qwen3.6's alpha / beta) in work units of 8 weight rows
///              instead of 64: the unit's weights (64 KB) stay in L1 while the tokens pass, and the
///              step gets 32 router units for the team instead of 4 slow ones (032 profile: this
///              step waits as long as it works). Scheduling only: same vec_dot per element
///   bit 8 (256): Q6_K through the tiles as well: dense projections (shared experts) with the
///              multi-token tile GEMM, routed Q6_K experts with the pair tiles when bit 4 (16) is set
///   bit 9 (512): barrier-wait diet in the delta-net layers (032 profile: 8.2 of 50.4 thread-ms per
///              token is waiting). (a) The z projection leaves the projection phase and becomes extra
///              work units of the delta-rule phase, whose 32 token-serial heads cannot be split
///              evenly over the team; the SiLU(z) gate is applied in the output-quantise phase.
///              (b) The causal conv runs in 64 dynamically scheduled units of 128 channels instead
///              of 32 static ones of 256. Scheduling only: every element sees the same operations
///   bit 10 (1024): the causal conv + SiLU of a delta-net layer runs inside the qkv projection's work
///              units (a unit is 64 weight rows = 64 conv channels; the conv of a channel needs only
///              that channel's projection over the tokens), so the conv phase and its barrier go
///              away and the unit's outputs are still in cache. Same operations per element
///   bit 11 (2048): the row-wise steps (norms, quantise, gates, routing, swiglu, sums, K / V writes)
///              hand out their rows in small chunks from a shared counter instead of one fixed
///              share per thread. Under sustained all-core load the three clusters run at 3.41 /
///              2.97 / 2.51 GHz (LEARNINGS T30-T33): with fixed shares every such step waits for the
///              cluster-2 cores. Same operations per row, whoever runs it
///   bit 2 (4): tiled attention: score / probability scratch columns `ATT_PAD` elements apart
///              from the n_ctx stride (the 32 column streams no longer share an L1 set)
/// Set by `OXE_PF_TILES=<mask>` in the environment (read once) or [`set_pf_tiles`].
static PF_TILES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
pub const PF_TILE_DENSE: u32 = 1;
pub const PF_ATTN_PB: u32 = 2;
pub const PF_ATTN_PAD: u32 = 4;
pub const PF_MMID_GEMV1: u32 = 8;
pub const PF_MMID_TILE: u32 = 16;
pub const PF_ATTN_TT8: u32 = 32;
pub const PF_GDN_X4: u32 = 64;
pub const PF_F32_CU8: u32 = 128;
pub const PF_Q6K_TILE: u32 = 256;
pub const PF_WAIT_DIET: u32 = 512;
pub const PF_CONV_FUSE: u32 = 1024;
pub const PF_DYN_SPLIT: u32 = 2048;
/// Extra elements between the attention scratch columns with `PF_ATTN_PAD` (13 cache lines of
/// f32, 6.5 of f16: an odd offset in lines for both).
const ATT_PAD: usize = 208;

/// Query tokens per tiled-attention unit under the current mask (at most 64 columns per unit:
/// the f16 tile kernel's column chunk and the `nf` array).
pub(super) fn att_tt_want(d: &ModelDesc, mask: u32) -> usize {
    let group = d.n_head / d.n_head_kv;
    if mask & PF_ATTN_TT8 != 0 && 8 * group <= 64 {
        8
    } else {
        ATT_TT
    }
}

/// Whether an expert with `cnt` routed rows goes through the pair tiles (`PF_MMID_TILE`). Kernel
/// timings 2026-11-10 15:40 (one thread, GMAC/s of real rows, pair tiles vs today's 4-row tiles):
/// 2 rows x1.2-1.4, 3-4 rows x0.92-0.95 (Q3_K, Q4_K) / x1.06 (Q5_K), 5-6 rows x1.4-1.7,
/// 8 rows x1.15-1.4, 16 rows x1.3-1.6, 64 rows x1.5-1.9. A single row is fastest through the GEMV
/// (`PF_MMID_GEMV1`), otherwise through a 1-pair tile. `OXE_PF_MMID_MIN=n` forces the tiles for
/// every expert with at least n rows (tests).
fn mmid_use_tile(q: kernels_sys::QType, cnt: usize, mask: u32) -> bool {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let o = *MIN.get_or_init(|| std::env::var("OXE_PF_MMID_MIN").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0));
    if o > 0 {
        return cnt >= o;
    }
    // Q6_K (16:36): the tiles win at every count, 1 row included (27 vs 24.7 GMAC/s GEMV), 3-4 rows x1.10
    match cnt {
        1 => mask & PF_MMID_GEMV1 == 0 || q == kernels_sys::QType::Q6K,
        3 | 4 => matches!(q, kernels_sys::QType::Q5K | kernels_sys::QType::Q6K),
        _ => true,
    }
}

fn attn_pb() -> usize {
    static PB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PB.get_or_init(|| {
        let v = std::env::var("OXE_PF_ATTN_PB").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(512);
        (v.max(32) / 32) * 32
    })
}

/// Selects the moe-prefill kernels (see `PF_TILES`); overrides the environment.
pub fn set_pf_tiles(mask: u32) {
    PF_TILES.store(mask & 0x7fff_ffff, Relaxed);
}

/// llama.cpp ubatches run through each layer together (`OXE_PF_GROUP`, default 0 = by size; [`set_pf_group`]).
/// Exact: per token nothing changes (every full ubatch is a multiple of 4 rows, so each dense row
/// keeps its 4-row GEMM group; the routed-expert kernels are per row; each token keeps its own
/// ubatch's padded n_kv and causal mask; conv and delta net run the tokens in order). It gives the
/// routed-expert tiles G times the rows per expert (their fixed decode cost per expert unit is
/// ~3 token rows: 8 rows -> 119 GMAC/s, 32 rows -> 146 for Q3_K) and 1/G of the per-step waits, at
/// G times the scratch (~60 MB per 256 tokens). Used only when the prefill starts on a multiple of
/// 256 with ubatch % 256 == 0 (so no token's n_kv covers a position a later ubatch of the group
/// has written: the V^T tail rule would see other values there), with the tiled attention and
/// without tracing; otherwise one ubatch at a time. Since F92 the start-alignment part is relaxed
/// when enabled ([`pf_group_unaligned`], opt-in): those tail values are causally masked and add exact zeros.
///
/// Setting 0 = by size (the default since MoE candidate 2; `OXE_PF_GROUP=0|auto` or unset): the
/// group of a prefill call follows its number of new tokens, see [`pf_group_for`]. 1..16 = fixed.
static PF_GROUP: AtomicUsize = AtomicUsize::new(usize::MAX);
pub fn set_pf_group(g: usize) {
    PF_GROUP.store(g.min(16), Relaxed);
}
/// F92: calls that start OFF a multiple of 256 (server turns continuing after a decoded reply) are
/// grouped too when `OXE_PF_GROUP_UNALIGNED=1` (release main: default OFF, as in production MoE
/// candidate 2 bbc2990, until QC-F92 completes its control arm; unset or 0 = the aligned-only rule). Exact: a
/// sub-ubatch's padded n_kv may now cover positions a later sub-ubatch of the same pass has
/// already written, but those positions are causally masked (probability exactly 0) for every
/// token of the earlier sub-ubatch, so finite values there add exact zeros, as the zeros or stale
/// rows they replace did. Model-free proof: `f92_unaligned_group_is_bit_identical` (logits, state,
/// decode rows; prefixes 37 / 129 / 300 / 511, 2-3 threads, masks 1021 / 4095, V pad 0 / 32).
static PF_GROUP_UNALIGNED: AtomicUsize = AtomicUsize::new(usize::MAX);
pub fn set_pf_group_unaligned(on: bool) {
    PF_GROUP_UNALIGNED.store(on as usize, Relaxed);
}
pub fn pf_group_unaligned() -> bool {
    let v = PF_GROUP_UNALIGNED.load(Relaxed);
    if v != usize::MAX {
        return v != 0;
    }
    let on = std::env::var("OXE_PF_GROUP_UNALIGNED").map(|s| s.trim() != "0").unwrap_or(false);
    PF_GROUP_UNALIGNED.store(on as usize, Relaxed);
    on
}
pub fn pf_group() -> usize {
    let v = PF_GROUP.load(Relaxed);
    if v != usize::MAX {
        return v;
    }
    let g = std::env::var("OXE_PF_GROUP").ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(0).min(16);
    PF_GROUP.store(g, Relaxed);
    g
}

/// Thresholds of the by-size group (new tokens of one prefill call): below T1 group 1, from T1
/// group 2, from T2 group 4. Defaults T1 = 100 (provisional, G1 of batch 1), T2 = 0 = never
/// (`OXE_PF_GROUP_T1`, `OXE_PF_GROUP_T2`; [`set_pf_group_thresholds`]). Every setting gives the same
/// state (groups are exact). Note: a call of at most one ubatch runs as one pass whatever the group,
/// so thresholds at or below the ubatch size change nothing.
pub const PF_GROUP_T1_DEFAULT: usize = 100;
static PF_GROUP_T: [AtomicUsize; 2] = [AtomicUsize::new(usize::MAX), AtomicUsize::new(usize::MAX)];
pub fn set_pf_group_thresholds(t1: usize, t2: usize) {
    PF_GROUP_T[0].store(t1.min(usize::MAX - 1), Relaxed);
    PF_GROUP_T[1].store(t2.min(usize::MAX - 1), Relaxed);
}
pub fn pf_group_thresholds() -> (usize, usize) {
    let get = |i: usize, key: &str, def: usize| {
        let v = PF_GROUP_T[i].load(Relaxed);
        if v != usize::MAX {
            return v;
        }
        let t = std::env::var(key).ok().and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(def).min(usize::MAX - 1);
        PF_GROUP_T[i].store(t, Relaxed);
        t
    };
    (get(0, "OXE_PF_GROUP_T1", PF_GROUP_T1_DEFAULT), get(1, "OXE_PF_GROUP_T2", 0))
}
/// The group a prefill call of `n_new` tokens asks for under the current setting (before the
/// alignment rule of [`set_pf_group`]'s doc, which can still turn it into 1).
pub fn pf_group_for(n_new: usize) -> usize {
    match pf_group() {
        0 => {
            let (t1, t2) = pf_group_thresholds();
            group_by_size(n_new, t1, t2)
        }
        g => g,
    }
}
/// The by-size rule itself (see [`pf_group_thresholds`]).
pub fn group_by_size(n_new: usize, t1: usize, t2: usize) -> usize {
    if t2 != 0 && n_new >= t2 {
        4
    } else if n_new >= t1 {
        2
    } else {
        1
    }
}

/// Rows-per-expert histogram of the routed MoE steps since the last reset (process-wide; MAGIC-PLAN
/// MP-3): `hist[c]` = number of (layer pass, expert) pairs that got c routed rows, c capped at 64;
/// index 0 = experts not used in a pass. Passes over fewer than 16 rows are not counted (the last
/// layer's single output row). Costs 256 increments per layer pass on thread 0.
static ROWS_HIST: [std::sync::atomic::AtomicU64; 65] = [const { std::sync::atomic::AtomicU64::new(0) }; 65];
pub fn prefill_rows_hist() -> Vec<u64> {
    ROWS_HIST.iter().map(|c| c.load(Relaxed)).collect()
}
pub fn reset_prefill_rows_hist() {
    for c in &ROWS_HIST {
        c.store(0, Relaxed);
    }
}

/// The default kernel mask (MoE candidate 2; batch 1 G1 / G2: exact against the oracle on Ornith
/// and Qwen): every bit up to 512 except the position-blocked V^T.p (2). `OXE_PF_TILES=0` = the
/// reference kernels.
pub const PF_TILES_DEFAULT: u32 = 1021;

/// The active moe-prefill kernel mask.
pub fn pf_tiles() -> u32 {
    let v = PF_TILES.load(Relaxed);
    if v != u32::MAX {
        return v;
    }
    let m = std::env::var("OXE_PF_TILES").ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(PF_TILES_DEFAULT) & 0x7fff_ffff;
    PF_TILES.store(m, Relaxed);
    m
}

/// llama.cpp's production `--ubatch-size`.
pub const UBATCH: usize = 256;
/// Weight rows per dense GEMM work unit (a multiple of every row group).
const CU: usize = 64;
/// Columns per MUL_MAT_ID unit of the down projection (gate/up units take a whole expert).
const DOWN_CU: usize = 512;
/// Query tokens per attention work unit.
const ATT_TB: usize = 16;
/// Query tokens per tiled-attention work unit (x 8 heads of a KV head = 32 columns that share
/// every K / V^T row load; docs/lab-notes/M3-ATTN.md).
const ATT_TT: usize = 4;
/// Extra floats between the routed down-projection output rows (`dn`): with n_embd 2048 the 8
/// expert rows of a token would sit exactly 8 KB apart and share two L1 sets with the shared
/// expert, the residual and the output row in the weighted sum (aliasing sweep, hit 2). One
/// cache line per row moves each of them to its own set. Layout only.
const DN_PAD: usize = 16;
/// Columns of one MUL_MAT_ID gemm call (repack.cpp `chunk_cols`).
const MMID_COLS: usize = 256;

/// Profiled prefill phases (`cfg.profile`). Each id is the barrier that ends the phase's work:
/// the work of a phase is the time since the thread's previous barrier, its wait the time inside
/// the barrier. Shared steps (norm + quantise, mixer output) take their caller's id.
pub const PF_PHASES: &[&str] = &[
    "embed",                  // PH_EMBED: embedding rows, end of the embed step
    "hid_norm",               // PH_HID_NORM: output norm of every row (all_rows)
    "rec_norm_quant",         // PH_REC_NORM_QUANT (pf_norm_act of a delta-net layer)
    "rec_proj_dense",         // PH_REC_PROJ_DENSE: qkv / z / beta / alpha projections
    "rec_conv_silu",          // PH_REC_CONV_SILU: causal conv + SiLU
    "rec_l2_gates",           // PH_REC_L2_GATES: l2 norm of q / k, gates and beta
    "rec_delta_rule",         // PH_REC_DELTA_RULE: gated delta rule + gated RMSNorm with z
    "rec_out_quant",          // PH_REC_OUT_QUANT: quantise O (ssm_out)
    "rec_out_dense",          // PH_REC_OUT_DENSE: ssm_out projection + residual
    "att_norm_quant",         // PH_ATT_NORM_QUANT (pf_norm_act of an attention layer)
    "att_proj_dense",         // PH_ATT_PROJ_DENSE: Q(+gate), K, V projections
    "att_prep",               // PH_ATT_PREP: q/k norm, RoPE, K/V cache writes, f16 Q
    "att_core_tiled",         // PH_ATT_CORE_TILED: tiled attention (K^T Q, softmax, V^T P)
    "att_core",               // PH_ATT_CORE: per-(head, token block) attention
    "att_out_quant",          // PH_ATT_OUT_QUANT: quantise the gated attention output
    "att_out_dense",          // PH_ATT_OUT_DENSE: attn_output projection + residual
    "moe_norm_quant",         // PH_MOE_NORM_QUANT: post-attention norm + activations
    "moe_router_shared_dense", // PH_MOE_ROUTER_SHARED: router, shared gate/up, shared gate
    "moe_route_swiglu",       // PH_MOE_ROUTE_SWIGLU: routing per token, shared swiglu
    "moe_units_build",        // PH_MOE_UNITS_BUILD: expert row lists and work units (thread 0)
    "moe_gate_up",            // PH_MOE_GATE_UP: routed gate/up (MUL_MAT_ID) + shared down
    "moe_swiglu_quant",       // PH_MOE_SWIGLU_QUANT: routed swiglu + from_float of down rows
    "moe_down",               // PH_MOE_DOWN: routed down (MUL_MAT_ID)
    "moe_sum_residual",       // PH_MOE_SUM_RESIDUAL: expert sum, shared expert, residual
    "head_norm",              // PH_HEAD_NORM: output norm of the output row (thread 0)
    "head",                   // PH_HEAD: head GEMV + argmax (work after the last barrier)
    "trace",                  // PH_TRACE: trace-only barriers and snapshots
];
const PH_EMBED: usize = 0;
const PH_HID_NORM: usize = 1;
const PH_REC_NORM_QUANT: usize = 2;
const PH_REC_PROJ_DENSE: usize = 3;
const PH_REC_CONV_SILU: usize = 4;
const PH_REC_L2_GATES: usize = 5;
const PH_REC_DELTA_RULE: usize = 6;
const PH_REC_OUT_QUANT: usize = 7;
const PH_REC_OUT_DENSE: usize = 8;
const PH_ATT_NORM_QUANT: usize = 9;
const PH_ATT_PROJ_DENSE: usize = 10;
const PH_ATT_PREP: usize = 11;
const PH_ATT_CORE_TILED: usize = 12;
const PH_ATT_CORE: usize = 13;
const PH_ATT_OUT_QUANT: usize = 14;
const PH_ATT_OUT_DENSE: usize = 15;
const PH_MOE_NORM_QUANT: usize = 16;
const PH_MOE_ROUTER_SHARED: usize = 17;
const PH_MOE_ROUTE_SWIGLU: usize = 18;
const PH_MOE_UNITS_BUILD: usize = 19;
const PH_MOE_GATE_UP: usize = 20;
const PH_MOE_SWIGLU_QUANT: usize = 21;
const PH_MOE_DOWN: usize = 22;
const PH_MOE_SUM_RESIDUAL: usize = 23;
const PH_HEAD_NORM: usize = 24;
const PH_HEAD: usize = 25;
const PH_TRACE: usize = 26;
const PF_N: usize = 27;
const _: () = assert!(PF_PHASES.len() == PF_N);

/// Sub-accumulators of the MoE gate/up step (`PfProf::sub`): routed MUL_MAT_ID units and the
/// shared down units. Only the first `PF_SUBS.len()` entries of `PfProf::sub` are used.
pub const PF_SUBS: &[&str] = &["routed_gate_up", "shared_down"];
const SUB_GATE_UP: usize = 0;
const SUB_SHARED_DOWN: usize = 1;
const _: () = assert!(PF_SUBS.len() <= 4);

/// Per-thread prefill profile, summed over the profiled prefills since the last reset (ticks).
#[repr(align(128))]
#[derive(Clone, Copy, Default)]
struct PfProf {
    work: [u64; PF_N],
    wait: [u64; PF_N],
    sub: [u64; 4],
    /// tick at this thread's last barrier exit (or the start of pf_thread)
    last: u64,
}

/// Prefill profile (`Engine::prefill_profile`): per phase, the thread-seconds of work and of
/// barrier wait summed over all threads; `sub_s` likewise for `PF_SUBS`.
#[derive(Clone, Debug, Default)]
pub struct PfProfile {
    pub names: Vec<&'static str>,
    pub work_s: Vec<f64>,
    pub wait_s: Vec<f64>,
    pub sub_names: Vec<&'static str>,
    pub sub_s: Vec<f64>,
    pub threads: usize,
}

/// Shared scratch buffer: written in disjoint ranges by the workers (see the step comments).
struct Sb<T> {
    ptr: *mut T,
    len: usize,
    _v: Vec<T>,
}
// SAFETY: accesses follow the step ownership rules documented at each use.
unsafe impl<T> Send for Sb<T> {}
unsafe impl<T> Sync for Sb<T> {}
impl<T: Clone + Default> Sb<T> {
    fn new(n: usize) -> Sb<T> {
        let mut v = vec![T::default(); n.max(1)];
        Sb { ptr: v.as_mut_ptr(), len: n, _v: v }
    }
}
impl<T> Sb<T> {
    /// SAFETY: the caller owns `r` for the current step (or only reads it).
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn s(&self, r: Range<usize>) -> &mut [T] {
        debug_assert!(r.start <= r.end && r.end <= self.len, "{r:?} of {}", self.len);
        std::slice::from_raw_parts_mut(self.ptr.add(r.start), r.end - r.start)
    }
    #[inline]
    fn p(&self, i: usize) -> *mut T {
        debug_assert!(i <= self.len);
        // SAFETY: within the allocation.
        unsafe { self.ptr.add(i) }
    }
}

#[repr(align(128))]
struct PfThread {
    /// tiled attention: scores / f16 probabilities of the unit's columns (col * n_ctx, lazily
    /// touched up to the prefix length) and the unit's outputs [dim][col]
    akq: Vec<f32>,
    ap16: Vec<u16>,
    aot: Vec<f32>,
    /// accumulators of the position-blocked V^T . p (`PF_ATTN_PB`): [dim][col][4] x 16 bytes
    ast: Vec<u128>,
    tile: Vec<u8>,
    tmp: Vec<f32>,
    /// routed-expert pair tiles (`PF_MMID_TILE`): packed pairs (u32 for the alignment), row and
    /// output pointers (as usize: the scratch must stay Send)
    xpack: Vec<u32>,
    xrows: Vec<usize>,
    xout: Vec<usize>,
    kq: Vec<f32>,
    p16: Vec<u16>,
    wp: Vec<f32>,
    mask: Vec<f32>,
    aout: Vec<f32>,
    gsig: Vec<f32>,
    delta: Vec<f32>,
    gout: Vec<f32>,
    tmpn: Vec<f32>,
    zs: Vec<f32>,
    iscr: Vec<i32>,
    /// int8 KV reads (O-13; empty in f16 mode)
    kv8s: Kv8Scratch,
    /// PREFILL-LONG fa2 unit scratch (O(chunk); empty unless the scratch was built for fa2)
    fa2: pfattn::Scratch,
    step: usize,
}

/// Prefill scratch (allocated on first use, ~60-80 MB at ubatch 256).
pub(super) struct Pf {
    ub: usize,
    nth: usize,
    /// query tokens per tiled-attention unit (`ATT_TT`, or 8 with `PF_ATTN_TT8`)
    att_tt: usize,
    /// the kernel mask of the prefill in flight, read once per `prefill` call: every worker must
    /// see the same mask (bit 128 changes the work-unit grid), whatever `set_pf_tiles` does meanwhile
    mask: std::sync::atomic::AtomicU32,
    /// llama.cpp's ubatch size of the prefill in flight (a run may hold several: `OXE_PF_GROUP`)
    lub: AtomicUsize,
    /// PF_DYN_SPLIT of the prefill in flight (off when tracing)
    dyn_split: std::sync::atomic::AtomicBool,
    /// positions the context-sized thread scratch (akq / ap16 tiles, kq / p16 / wp / mask rows)
    /// holds: the engine's n_ctx, or the fa2 cutoff (PFA_MIN_KV, or PREFILL-POLICY's auto cutoff;
    /// at most n_ctx) when built for fa2
    cap: usize,
    /// built for PREFILL-LONG fa2 (f16 KV, no trace / f32 reference): attention units whose last
    /// token's padded n_kv exceeds `cap` (the cutoff) run `pfattn::Tile`, the others the tiled kernel
    fa2: bool,
    /// APPEND-PREFILL: the fa2 units of the prefill in flight run the fa2x kernel: 0 = no (fa2),
    /// else `fold << 16 | ch` ([`pfattn::XParams`]); stored once per `prefill` call
    fa2x: AtomicUsize,
    /// THREADS-BY-PHASE: threads (tid < this) that take attention-core units in the prefill in flight
    attn_nth: AtomicUsize,
    x: Sb<f32>,
    x1: Sb<f32>,
    xn: Sb<f32>,
    act_k: Sb<u8>,
    act_0: Sb<u8>,
    actr: Sb<u8>,
    qkv: Sb<f32>,
    conv: Sb<f32>,
    z: Sb<f32>,
    alpha: Sb<f32>,
    beta: Sb<f32>,
    gate: Sb<f32>,
    beta_s: Sb<f32>,
    o: Sb<f32>,
    kr: Sb<f32>,
    vr: Sb<f32>,
    qf: Sb<f32>,
    /// f16 q for the tiled attention, [kv head][token][head in group][dim]
    q16: Sb<u16>,
    /// fused int8 K: the same columns as Q8_0 rows (row_bytes(head_dim) each)
    q8: Sb<u8>,
    rope: Sb<f32>,
    router: Sb<f32>,
    probs: Sb<f32>,
    ids: Sb<i32>,
    w: Sb<f32>,
    sg: Sb<f32>,
    shg: Sb<f32>,
    shu: Sb<f32>,
    shh: Sb<f32>,
    sh_act: Sb<u8>,
    g: Sb<f32>,
    u: Sb<f32>,
    hq: Sb<u8>,
    #[cfg(feature="f32-reference")]
    href: Sb<f32>,
    dn: Sb<f32>,
    sdn: Sb<f32>,
    off: Sb<u32>,
    order: Sb<u32>,
    units: Sb<(u32, u32, u32)>,
    n_units: [sched::Padded<AtomicUsize>; 2],
    argmax: Sb<(f32, u32)>,
    /// output-normed hidden rows of the ubatch (prefill_nextn)
    hid: Sb<f32>,
    /// one counter per dynamic step, each on its own line pair (a fast thread starting step
    /// i + 1 must not invalidate the line the stragglers of step i still fetch_add on)
    ctrs: Vec<sched::Padded<AtomicUsize>>,
    th: Vec<UnsafeCell<PfThread>>,
    /// profile of each thread (`cfg.profile`; zero and unused otherwise)
    prof: Vec<UnsafeCell<PfProf>>,
}
// SAFETY: PfThread cells are used only by their worker; shared buffers follow the step rules.
unsafe impl Sync for Pf {}

/// Profile cell of thread `tid`.
#[allow(clippy::mut_from_ref)]
fn pf_prof(pf: &Pf, tid: usize) -> &mut PfProf {
    // SAFETY: cell `tid` is used only by worker `tid` during a run and by the owner between runs
    // (the engine's `pf` lock is held for both).
    unsafe { &mut *pf.prof[tid].get() }
}

/// The work counter of this thread's next dynamic step (every thread runs the same sequence
/// of steps, so step i means the same work list on every thread).
fn step_ctr<'p>(pf: &'p Pf, ts: &mut PfThread) -> &'p AtomicUsize {
    let c = &pf.ctrs[ts.step].0;
    ts.step += 1;
    c
}

/// This thread's indices of 0..n in one row-wise step: its fixed share, or (PF_DYN_SPLIT) chunks of
/// `grain` indices taken from the step's shared counter until none is left.
struct Share<'p> {
    ctr: Option<&'p AtomicUsize>,
    n: usize,
    grain: usize,
    cur: usize,
    end: usize,
}
impl Iterator for Share<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.cur >= self.end {
            let c = self.ctr?.fetch_add(self.grain, Relaxed);
            if c >= self.n {
                self.ctr = None;
                return None;
            }
            self.cur = c;
            self.end = (c + self.grain).min(self.n);
        }
        self.cur += 1;
        Some(self.cur - 1)
    }
}
/// See [`Share`]. Takes one step counter in both modes (every thread calls it in the same order).
fn share<'p>(pf: &'p Pf, ts: &mut PfThread, n: usize, grain: usize, tid: usize) -> Share<'p> {
    let ctr = step_ctr(pf, ts);
    if pf.dyn_split.load(Relaxed) {
        Share { ctr: Some(ctr), n, grain: grain.max(1), cur: 0, end: 0 }
    } else {
        let r = split(n, tid, pf.nth);
        Share { ctr: None, n, grain: 1, cur: r.start, end: r.end }
    }
}

fn split(n: usize, tid: usize, nth: usize) -> Range<usize> {
    n * tid / nth..n * (tid + 1) / nth
}

/// LONG-CTX: context-sized bytes of one prefill thread's scratch (`PfThread` akq / ap16 tiles
/// and kq / p16 / wp / mask rows) at capacity `n_ctx` (>= head_chunk, ff_exp, n_expert, as at
/// every real context); the same expressions as [`Pf::new`]. PREFILL-LONG: with fa2 units beyond
/// `cut` (PFA_MIN_KV, or the PREFILL-POLICY auto cutoff) the rows and tiles hold at most `cut`
/// positions and the fa2 unit scratch (O(chunk)) is added, so the result stops growing with the
/// context beyond the cutoff.
pub(super) fn pf_ctx_bytes(d: &ModelDesc, n_ctx: usize, att_tt: usize, cut: Option<usize>) -> usize {
    let fa2 = cut.is_some();
    let cap = pf_scratch_cap(n_ctx, cut);
    let group = d.n_head / d.n_head_kv;
    let tile = att_tt * group * (cap + ATT_PAD);
    let f = if fa2 { pfattn::Scratch::bytes(att_tt * group, d.head_dim) } else { 0 };
    tile * (4 + 2) + cap * 4 + (cap + d.head_dim) * 2 + cap * 4 + cap * 4 + f
}

/// Positions of the context-sized prefill thread scratch (see `Pf::cap`).
/// Under fa2 (`cut` Some) the capacity is the cutoff (or n_ctx when that is smaller: then no unit
/// exceeds it):
/// `Pf::cap` is what the attention compares a unit's n_kv with.
pub(super) fn pf_scratch_cap(n_ctx: usize, cut: Option<usize>) -> usize {
    cut.map_or(n_ctx, |c| n_ctx.min(c))
}

fn max_act(k: usize) -> usize {
    ActType::Q8K.bytes(k.div_ceil(QK_K) * QK_K).max(ActType::Q8_0.bytes(k.div_ceil(32) * 32))
}

impl Pf {
    /// `n_ctx`: positions of the context-sized thread scratch ([`pf_scratch_cap`]); `fa2`: also
    /// the fa2 unit scratch (the caller passes the capped size then).
    #[allow(clippy::too_many_arguments)]
    fn new(d: &ModelDesc, ub: usize, nth: usize, n_ctx: usize, head_chunk: usize, mask: u32, kv: KvType, reference:bool, fa2: bool) -> Pf {
        #[cfg(not(feature="f32-reference"))]
        let _=reference;
        let att_tt = att_tt_want(d, mask);
        let att_w = d.n_head * d.head_dim;
        let nch = d.ssm_conv_channels();
        let ku = d.n_expert_used;
        let wide = d.n_embd.max(d.ssm_d_inner).max(att_w).max(d.ff_shexp).max(d.ff_exp);
        let hv = d.ssm_head_v_dim();
        Pf {
            ub,
            nth,
            att_tt,
            mask: std::sync::atomic::AtomicU32::new(0),
            lub: AtomicUsize::new(UBATCH),
            dyn_split: std::sync::atomic::AtomicBool::new(false),
            cap: n_ctx,
            fa2,
            fa2x: AtomicUsize::new(0),
            attn_nth: AtomicUsize::new(nth),
            x: Sb::new(ub * d.n_embd),
            x1: Sb::new(ub * d.n_embd),
            xn: Sb::new(ub * d.n_embd),
            act_k: Sb::new(ub * ActType::Q8K.bytes(wide)),
            act_0: Sb::new(ub * ActType::Q8_0.bytes(wide)),
            actr: Sb::new(ub * max_act(d.n_embd)),
            qkv: Sb::new(ub * nch.max(2 * att_w)),
            conv: Sb::new(ub * nch),
            z: Sb::new(ub * d.ssm_d_inner),
            alpha: Sb::new(ub * d.ssm_n_vheads),
            beta: Sb::new(ub * d.ssm_n_vheads),
            gate: Sb::new(ub * d.ssm_n_vheads),
            beta_s: Sb::new(ub * d.ssm_n_vheads),
            o: Sb::new(ub * d.ssm_d_inner.max(att_w)),
            kr: Sb::new(ub * d.n_head_kv * d.head_dim),
            vr: Sb::new(ub * d.n_head_kv * d.head_dim),
            qf: Sb::new(ub * att_w),
            q16: Sb::new(ub * att_w),
            q8: Sb::new(if kv.k8() { ub * d.n_head * kv8::row_bytes(d.head_dim) } else { 0 }),
            rope: Sb::new(ub * d.head_dim),
            router: Sb::new(ub * d.n_expert),
            probs: Sb::new(ub * d.n_expert),
            ids: Sb::new(ub * ku),
            w: Sb::new(ub * ku),
            sg: Sb::new(ub),
            shg: Sb::new(ub * d.ff_shexp),
            shu: Sb::new(ub * d.ff_shexp),
            shh: Sb::new(ub * d.ff_shexp),
            sh_act: Sb::new(ub * max_act(d.ff_shexp)),
            g: Sb::new(ub * ku * d.ff_exp),
            u: Sb::new(ub * ku * d.ff_exp),
            hq: Sb::new(ub * ku * max_act(d.ff_exp)),
            #[cfg(feature="f32-reference")]
            href: Sb::new(if reference {ub*ku*d.ff_exp}else{0}),
            dn: Sb::new(ub * ku * (d.n_embd + DN_PAD)),
            sdn: Sb::new(ub * d.n_embd),
            off: Sb::new(d.n_expert + 1),
            order: Sb::new(ub * ku),
            units: Sb::new(d.n_expert * (2 + d.n_embd.div_ceil(DOWN_CU))),
            n_units: [sched::Padded(AtomicUsize::new(0)), sched::Padded(AtomicUsize::new(0))],
            argmax: Sb::new(nth * 16),
            hid: Sb::new(ub * d.n_embd),
            ctrs: (0..4096).map(|_| sched::Padded(AtomicUsize::new(0))).collect(),
            th: (0..nth)
                .map(|_| {
                    UnsafeCell::new(PfThread {
                        akq: vec![0.0; att_tt * (d.n_head / d.n_head_kv) * (n_ctx + ATT_PAD)],
                        ap16: vec![0; att_tt * (d.n_head / d.n_head_kv) * (n_ctx + ATT_PAD)],
                        ast: vec![0; att_tt * (d.n_head / d.n_head_kv) * d.head_dim * 4],
                        aot: vec![0.0; att_tt * (d.n_head / d.n_head_kv) * d.head_dim],
                        tile: vec![0; 4 * max_act(wide)],
                        tmp: vec![0.0; 4 * MMID_COLS],
                        xpack: Vec::new(),
                        xrows: Vec::new(),
                        xout: Vec::new(),
                        kq: vec![0.0; n_ctx.max(head_chunk).max(d.ff_exp)],
                        p16: vec![0; n_ctx + d.head_dim],
                        wp: vec![0.0; n_ctx.max(d.n_expert)],
                        mask: vec![0.0; n_ctx],
                        aout: vec![0.0; d.head_dim],
                        gsig: vec![0.0; d.head_dim],
                        delta: vec![0.0; hv],
                        gout: vec![0.0; hv],
                        tmpn: vec![0.0; hv.max(d.head_dim)],
                        zs: vec![0.0; hv],
                        iscr: vec![0; d.n_expert],
                        kv8s: if kv.any8() { Kv8Scratch::new(d.head_dim, att_tt * (d.n_head / d.n_head_kv)) } else { Kv8Scratch::empty() },
                        fa2: if fa2 { pfattn::Scratch::new(att_tt * (d.n_head / d.n_head_kv), d.head_dim) } else { pfattn::Scratch::empty() },
                        step: 0,
                    })
                })
                .collect(),
            prof: (0..nth).map(|_| UnsafeCell::new(PfProf::default())).collect(),
        }
    }
}

/// A dense projection: out[t*os + c] = W[c] . x[t] (+ add[t*os + c]).
#[derive(Clone, Copy)]
struct Dense {
    w: WeightRef,
    #[cfg(feature = "npu-prefill")]
    npu: bool,
    /// f32 input rows (stride w.k) for F32 weights
    xin: *const f32,
    /// activation rows in the forward_mul_mat layout for quantised weights
    act: *const u8,
    out: *mut f32,
    os: usize,
    add: *const f32,
    /// full 4-row groups go through the multi-token tile GEMM (same bits; `PF_TILE_DENSE`)
    tile: bool,
    /// weight rows per work unit (`CU`; 8 for F32 weights with `PF_F32_CU8`)
    cu: usize,
}

/// PF_CONV_FUSE needs the qkv projection in units that are multiples of 16 channels (the SiLU's
/// NEON path) and not F32-split differently: quantised weights in `CU`-row units.
fn d_qkv_units_ok(op: &Dense) -> bool {
    d_qkv_units_ok_w(&op.w)
}
fn d_qkv_units_ok_w(w: &WeightRef) -> bool {
    !w.is_f32() && CU % 16 == 0
}

impl Dense {
    /// A copy of the op descriptor (raw pointers and a weight reference; no data).
    fn clone_op(&self) -> Dense {
        *self
    }
    fn units(&self) -> usize {
        self.w.rows.div_ceil(self.cu)
    }
}

/// Which rows of the ubatch a step works on: rows r0..r0+n of the (x, x1) residual buffers.
#[derive(Clone, Copy)]
struct Rows {
    r0: usize,
    n: usize,
}

impl<'a> Engine<'a> {
    /// Batched prefill of `tokens` at the current position, in ubatches of `ubatch` tokens.
    /// Leaves the KV cache and the recurrent state as llama.cpp's batched prefill does and
    /// returns the greedy argmax of the last token's logits (the full row is in
    /// [`Engine::logits`] when `cfg.logits`). `None` for an empty prompt.
    pub fn prefill<P: Team + ?Sized>(&self, pool: &P, tokens: &[u32], ubatch: usize) -> Option<u32> {
        self.prefill_ext(pool, tokens, ubatch, None)
    }

    /// [`Engine::prefill`] that also returns, in `hidden` (resized to tokens.len() * n_embd),
    /// the output-normed hidden state of every prompt position (llama.cpp's `h_nextn`, the
    /// MTP seed). This is llama.cpp's `embeddings_nextn` (unmasked) graph: the last layer's
    /// residual and FFN run on every row (`crop_after_nextn`) instead of the output row only,
    /// so the last layer of the last token is computed as in a llama-server with MTP drafting,
    /// and its logits can differ from [`Engine::prefill`]'s in the last bits. KV and recurrent
    /// state are the same either way.
    pub fn prefill_ext<P: Team + ?Sized>(&self, pool: &P, tokens: &[u32], ubatch: usize, hidden: Option<&mut Vec<f32>>) -> Option<u32> {
        self.prefill_impl(pool, tokens, ubatch, hidden, false, false)
    }

    /// M3.1 (MTP seeding): `hidden` as [`Engine::prefill_ext`], but the last token's logits
    /// are [`Engine::prefill`]'s bit for bit: after the all-rows last layer has produced the
    /// h_nextn rows, the last layer's MoE of the output row is recomputed on its own
    /// (crop_before_nextn, from the unchanged x1 row), so MTP on gives the same greedy output
    /// as MTP off. Only the last row's h_nextn is the all-rows one (an MTP input: drafts only).
    pub fn prefill_nextn<P: Team + ?Sized>(&self, pool: &P, tokens: &[u32], ubatch: usize, hidden: &mut Vec<f32>) -> Option<u32> {
        self.prefill_impl(pool, tokens, ubatch, Some(hidden), true, false)
    }

    /// Per-phase prefill profile (`cfg.profile`), summed over the threads and the prefills since
    /// the last [`Engine::reset_prefill_profile`]. `None` without profiling or before a prefill.
    pub fn prefill_profile(&self) -> Option<PfProfile> {
        if !self.cfg.profile {
            return None;
        }
        let mut work = [0u64; PF_N];
        let mut wait = [0u64; PF_N];
        let mut sub = [0u64; 4];
        let mut threads = 0;
        for slot in &self.pf {
            let guard = slot.lock().unwrap();
            let Some(pf) = guard.as_ref() else { continue };
            threads = threads.max(pf.nth);
            for tid in 0..pf.nth {
                let p = pf_prof(pf, tid);
                for i in 0..PF_N {
                    work[i] += p.work[i];
                    wait[i] += p.wait[i];
                }
                for i in 0..4 {
                    sub[i] += p.sub[i];
                }
            }
        }
        if threads == 0 {
            return None;
        }
        let hz = tick_hz() as f64;
        Some(PfProfile {
            names: PF_PHASES.to_vec(),
            work_s: work.iter().map(|&v| v as f64 / hz).collect(),
            wait_s: wait.iter().map(|&v| v as f64 / hz).collect(),
            sub_names: PF_SUBS.to_vec(),
            sub_s: sub[..PF_SUBS.len()].iter().map(|&v| v as f64 / hz).collect(),
            threads,
        })
    }

    /// Clears the prefill profile (see [`Engine::prefill_profile`]).
    /// Token rows of the prefill scratch in use (the largest pass so far: ubatch x group, capped at
    /// the call length); tests read it to see that a group really applied.
    pub fn prefill_scratch_rows(&self) -> usize {
        self.pf.iter().filter_map(|slot| slot.lock().unwrap().as_ref().map(|p| p.ub)).max().unwrap_or(0)
    }

    /// PREFILL-LONG: whether batched prefill builds its scratch for fa2 ([`pfattn`]): the
    /// engine's mode (`OXE_PF_ATTN`, [`Engine::set_pf_attn`]) selects fa2 at this n_ctx, f16 KV
    /// (int8 KV keeps its own kernels), no trace and no f32 reference (both read whole score rows).
    pub(super) fn pf_fa2(&self) -> bool {
        self.pf_path().fa2()
    }

    /// PREFILL-POLICY: the resolved path (see [`Engine::pf_attn_path`]).
    pub(super) fn pf_path(&self) -> pfattn::PfPath {
        #[cfg(feature = "f32-reference")]
        if self.f32_reference.is_some() {
            return pfattn::PfPath::TILED;
        }
        if self.kv.any8() || self.trace.is_some() {
            return pfattn::PfPath::TILED;
        }
        self.pf_attn.path(self.n_ctx, self.pf_auto)
    }

    /// APPEND-PREFILL: the fa2x kernel parameters when the fa2 units run fa2x
    /// (`OXE_PF_ATTN=fa2x` / [`Engine::set_pf_attn`] with [`PfAttn::Fa2x`], and [`Engine::pf_fa2`]).
    pub(super) fn pf_fa2x(&self) -> Option<pfattn::XParams> {
        (self.pf_path().kernel == pfattn::PfKernel::Fa2x).then_some(self.pf_xparams)
    }

    /// Context-dependent bytes one thread of the prefill scratch holds right now (the buffers
    /// `CtxMem::prefill_per_thread` accounts for), largest over the two scratch slots; None before
    /// the first batched prefill. Tests hold it equal to [`Engine::ctx_mem`].
    pub fn prefill_ctx_bytes_allocated(&self) -> Option<usize> {
        self.pf
            .iter()
            .filter_map(|slot| {
                let g = slot.lock().unwrap();
                let pf = g.as_ref()?;
                // SAFETY: the `pf` lock is held and no run is in flight; lengths only.
                let t = unsafe { &*pf.th[0].get() };
                Some(t.akq.len() * 4 + t.ap16.len() * 2 + t.kq.len() * 4 + t.p16.len() * 2 + t.wp.len() * 4 + t.mask.len() * 4 + t.fa2.bytes_allocated())
            })
            .max()
    }

    pub fn reset_prefill_profile(&self) {
        for slot in &self.pf {
            let mut guard = slot.lock().unwrap();
            if let Some(pf) = guard.as_mut() {
                for c in &pf.prof {
                    // SAFETY: the `pf` lock is held and no run is in flight.
                    unsafe { *c.get() = PfProf::default() };
                }
            }
        }
    }

    /// [`Engine::prefill`] for a prompt segment whose logits nobody reads (the server's anchor /
    /// query / head segments: the last prompt token is decoded separately afterwards). KV cache
    /// and recurrent state end up exactly as with `prefill`; the last layer's FFN for the output
    /// row, the output norm and the vocabulary head (one 0.3 G multiply-accumulate GEMV per call)
    /// are skipped, and [`Engine::logits`] is left as it was.
    pub fn prefill_state_only<P: Team + ?Sized>(&self, pool: &P, tokens: &[u32], ubatch: usize) {
        self.prefill_impl(pool, tokens, ubatch, None, false, true);
    }

    fn prefill_impl<P: Team + ?Sized>(&self, pool: &P, tokens: &[u32], ubatch: usize, mut hidden: Option<&mut Vec<f32>>, redo_last: bool, state_only: bool) -> Option<u32> {
        if tokens.is_empty() {
            return None;
        }
        let ubatch = ubatch.max(1);
        let nth = pool.nthreads();
        let pos0 = self.pos.load(Relaxed);
        assert!(pos0 + tokens.len() <= self.n_ctx, "context full ({} positions)", self.n_ctx);
        self.note_kv(pos0 + tokens.len());
        let mut guard = self.pf[(nth != self.plan.nthreads) as usize].lock().unwrap();
        // several llama.cpp ubatches per pass (see PF_GROUP for why this is exact and when it applies)
        let aligned = ubatch % 256 == 0 && (pos0 % 256 == 0 || pf_group_unaligned());
        let group = if aligned && self.attn_tiled && self.trace.is_none() { pf_group_for(tokens.len()) } else { 1 };
        let gsz = ubatch * group;
        let ub_cap = gsz.min(tokens.len());
        let mask = pf_tiles(); // one sample for the whole call
        let path = self.pf_path();
        let fa2 = path.fa2();
        let cap = pf_scratch_cap(self.n_ctx, path.cut_opt());
        let fits = matches!(&*guard, Some(p) if p.ub >= ub_cap && p.nth == nth && p.att_tt == att_tt_want(&self.plan.desc, mask) && p.fa2 == fa2 && p.cap == cap);
        if !fits {
            #[cfg(feature="f32-reference")]
            let reference=self.f32_reference.is_some();
            #[cfg(not(feature="f32-reference"))]
            let reference=false;
            // drop the old scratch first: at 256K two of them would not fit side by side
            *guard = None;
            *guard = Some(Box::new(Pf::new(&self.plan.desc, ub_cap, nth, cap, self.plan.head_chunk, mask, self.kv, reference, fa2)));
        }
        let pf: &Pf = guard.as_ref().unwrap();
        pf.mask.store(mask, Relaxed);
        pf.fa2x.store(self.pf_fa2x().map_or(0, |xp| xp.fold << 16 | xp.ch), Relaxed);
        pf.attn_nth.store(if self.pf_attn_threads == 0 { nth } else { self.pf_attn_threads.min(nth) }, Relaxed);
        pf.lub.store(ubatch, Relaxed);
        pf.dyn_split.store(mask & PF_DYN_SPLIT != 0 && self.trace.is_none(), Relaxed);
        let de = self.plan.desc.n_embd;
        if let Some(h) = hidden.as_mut() {
            h.clear();
            h.resize(tokens.len() * de, 0.0);
        }
        let all_rows = hidden.is_some();
        let mut best = None;
        let mut p = pos0;
        for (i, chunk) in tokens.chunks(gsz).enumerate() {
            // state_only: no output row at all (as a non-final ubatch)
            let last = (i + 1) * gsz >= tokens.len() && !state_only;
            best = self.prefill_ubatch(pool, pf, chunk, p, last, all_rows, redo_last);
            if let Some(h) = hidden.as_mut() {
                let o = (p - pos0) * de;
                // SAFETY: run finished; hid rows 0..chunk.len() complete.
                h[o..o + chunk.len() * de].copy_from_slice(unsafe { pf.hid.s(0..chunk.len() * de) });
            }
            p += chunk.len();
            self.pos.store(p, Relaxed);
        }
        self.pend[2].store(0, Relaxed); // M3.1: no verify to commit after a prefill
        best
    }

    fn prefill_ubatch<P: Team + ?Sized>(&self, pool: &P, pf: &Pf, toks: &[u32], p0: usize, last: bool, all_rows: bool, redo_last: bool) -> Option<u32> {
        let d = &self.plan.desc;
        let n = toks.len();
        // rope caches of the ubatch positions; dynamic-step counters
        // SAFETY: no run in flight.
        unsafe {
            for t in 0..n {
                ffi::oxe_mrope_cache_init(
                    (p0 + t) as i64,
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
                    pf.rope.p(t * d.head_dim),
                );
            }
        }
        for c in &pf.ctrs {
            c.store(0, Relaxed);
        }
        for t in &pf.th {
            // SAFETY: no run in flight.
            unsafe { (*t.get()).step = 0 };
        }
        #[cfg(any(test, feature = "f32-reference"))]
        self.pf_count_pass(n);
        pool.run_job(&|ctx: &mut Ctx| self.pf_thread(ctx, pf, toks, p0, last, all_rows, redo_last));
        if !last {
            return None;
        }
        // SAFETY: all writers finished (join barrier).
        let am = unsafe { pf.argmax.s(0..pf.nth * 16) };
        let mut best = (f32::NEG_INFINITY, u32::MAX);
        for t in 0..pf.nth {
            let v = am[t * 16];
            if v.1 != u32::MAX && (v.0 > best.0 || best.1 == u32::MAX) {
                best = v;
            }
        }
        Some(best.1)
    }


    #[allow(clippy::too_many_arguments)]
    fn pf_thread(&self, ctx: &mut Ctx, pf: &Pf, toks: &[u32], p0: usize, last: bool, all_rows: bool, redo_last: bool) {
        let d = &self.plan.desc;
        let tid = ctx.tid;
        // SAFETY: PfThread `tid` is used only by this worker.
        let ts = unsafe { &mut *pf.th[tid].get() };
        let n = toks.len();
        let de = d.n_embd;
        if self.cfg.profile {
            pf_prof(pf, tid).last = ticks();
        }
        // embeddings (rows split by thread)
        for t in share(pf, ts, n, 4, tid) {
            let r = self.ws.get(self.plan.embd);
            let row = self.ws.embedding_row(self.plan, toks[t] as usize);
            // SAFETY: row t of x is owned by this thread in this step.
            kernels_sys::dequantize_row(r.qtype(), row, unsafe { pf.x.s(t * de..(t + 1) * de) });
        }
        self.bar(ctx, pf, PH_EMBED);
        let nl = d.n_layer;
        for l in 0..nl {
            let lt = &self.plan.layers[l];
            match lt.mixer {
                Mixer::Recurrent { .. } => self.pf_recurrent(ctx, pf, ts, l, n),
                Mixer::Attention { .. } => self.pf_attention(ctx, pf, ts, l, n, p0),
            }
            if l + 1 < nl || all_rows {
                self.pf_moe(ctx, pf, ts, l, Rows { r0: 0, n });
            } else if last {
                // crop_before_nextn: only the output row continues (logits for the last token)
                self.pf_moe(ctx, pf, ts, l, Rows { r0: n - 1, n: 1 });
            }
        }
        if all_rows {
            // h_nextn: output norm of every row
            let w = self.ws.get(self.plan.out_norm).f32_slice();
            for t in share(pf, ts, n, 4, tid) {
                // SAFETY: row t of hid owned; x read-only.
                unsafe { ops::rms_norm_mul(pf.x.s(t * de..(t + 1) * de), w, pf.hid.s(t * de..(t + 1) * de), d.rms_eps) };
            }
            self.bar(ctx, pf, PH_HID_NORM);
            if last && redo_last {
                // M3.1: the output row's last-layer MoE as Engine::prefill computes it
                self.pf_moe(ctx, pf, ts, nl - 1, Rows { r0: n - 1, n: 1 });
            }
        }
        if last {
            self.pf_head(ctx, pf, ts, n - 1);
        }
        if self.cfg.profile {
            // work after the last barrier of this run (the head GEMV)
            let t = ticks();
            let p = pf_prof(pf, tid);
            p.work[PH_HEAD] += t - p.last;
        }
    }

    #[inline]
    fn ptr(&self, l: usize, name: &str, total: usize, off: usize, data: &[f32]) {
        if let Some(t) = &self.trace {
            t.put(name, Some(l), total, off, data);
        }
    }

    /// The barrier that ends prefill phase `ph`. Without profiling it is `ctx.barrier()`; with it,
    /// the time since this thread's previous barrier is work of `ph` and the time inside the
    /// barrier is its wait.
    #[inline]
    fn bar(&self, ctx: &mut Ctx, pf: &Pf, ph: usize) {
        if !self.cfg.profile {
            ctx.barrier();
            return;
        }
        let p = pf_prof(pf, ctx.tid);
        let t1 = ticks();
        p.work[ph] += t1 - p.last;
        ctx.barrier();
        let t2 = ticks();
        p.wait[ph] += t2 - t1;
        p.last = t2;
    }

    // ------------------------------------------------------------ shared building blocks

    /// RMS-norm rows of `src` (x or x1) into xn and quantise them for the activation types in
    /// `ats` (forward_mul_mat layout) and, if `routed`, per row with `from_float` into actr.
    /// Work split by 4-row groups. Ends with a barrier of phase `ph`.
    #[allow(clippy::too_many_arguments)]
    fn pf_norm_act(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, src: &Sb<f32>, rows: Rows, wnorm: Tid, ats: &[ActType], routed: Option<ActType>, l: usize, name: &str, ph: usize) {
        let d = &self.plan.desc;
        let de = d.n_embd;
        let w = self.ws.get(wnorm).f32_slice();
        let ng = rows.n.div_ceil(4);
        let full = rows.n - rows.n % 4;
        for g in share(pf, ts, ng, 1, ctx.tid) {
            let (a, b) = (4 * g, (4 * g + 4).min(rows.n));
            for i in a..b {
                // SAFETY: rows a..b of xn are owned by this thread; src rows are read-only here.
                unsafe {
                    let s = src.s((rows.r0 + i) * de..(rows.r0 + i + 1) * de);
                    ops::rms_norm_mul(s, w, pf.xn.s(i * de..(i + 1) * de), d.rms_eps);
                }
            }
            for &at in ats {
                let buf = if at == ActType::Q8K { &pf.act_k } else { &pf.act_0 };
                let rb = at.bytes(de);
                // SAFETY: activation rows a..b are owned by this thread.
                unsafe {
                    if b - a == 4 && b <= full {
                        quantize_mat4_raw(at, pf.xn.p(a * de), buf.p(a * rb), de);
                    } else {
                        for i in a..b {
                            quantize_row_raw(at, pf.xn.p(i * de), buf.p(i * rb), de);
                        }
                    }
                }
            }
            if let Some(at) = routed {
                let rb = at.bytes(de);
                for i in a..b {
                    // SAFETY: as above.
                    unsafe { quantize_row_raw(at, pf.xn.p(i * de), pf.actr.p(i * rb), de) };
                }
            }
            if self.trace.is_some() {
                // SAFETY: own rows.
                let v = unsafe { pf.xn.s(a * de..b * de) }.to_vec();
                self.ptr(l, name, rows.n * de, a * de, &v);
            }
        }
        self.bar(ctx, pf, ph);
    }

    /// Activation types the quantised weights in `ws` consume.
    fn acts_of(&self, ws: &[Tid]) -> Vec<ActType> {
        let mut v = Vec::new();
        for &id in ws {
            let w = self.ws.get(id);
            if !w.is_f32() {
                let at = w.qtype().act();
                if !v.contains(&at) {
                    v.push(at);
                }
            }
        }
        v
    }

    fn act_buf<'p>(&self, pf: &'p Pf, at: ActType) -> &'p Sb<u8> {
        if at == ActType::Q8K {
            &pf.act_k
        } else {
            &pf.act_0
        }
    }

    /// One unit (weight rows c0..c1) of a dense projection over `n` activation rows.
    fn dense_unit(&self, op: &Dense, n: usize, c0: usize, c1: usize) {
        let w = &op.w;
        let k = w.k;
        #[cfg(feature="npu-prefill")]
        if self.npu_prefill.as_ref().is_some_and(|p|unsafe{p.cpu_f32(w.ptr,false,n,c0,c1,op.xin,op.out.add(c0),op.os,if op.add.is_null(){op.add}else{op.add.add(c0)})}){return;}
        #[cfg(feature="f32-reference")]
        if let Some(reference)=&self.f32_reference {
            if !w.is_f32(){
                for c in c0..c1 {reference.with_row(w.ptr,0,c,|row|{
                    for t in 0..n{
                        // SAFETY: the unit owns these complete output columns;
                        // inputs are f32 and immutable after the norm barrier.
                        unsafe{
                            let mut y=ffi::oxe_vec_dot_f32(k as i32,row.as_ptr(),op.xin.add(t*k));
                            if !op.add.is_null(){y+=*op.add.add(t*op.os+c);}
                            *op.out.add(t*op.os+c)=y;
                        }
                    }
                });}return;
            }
        }
        #[cfg(feature = "npu-prefill")]
        if op.npu {
            debug_assert!(c0==0 && c1==w.rows);
            // SAFETY: this op is one indivisible work unit; complete inputs are
            // read-only until the phase barrier, and it alone owns all outputs.
            if unsafe{self.npu_prefill.as_ref().expect("prepared registry").run(w.ptr,n,op.xin,op.out,op.os,op.add)} {return;}
            // A latched/failed accelerator keeps this SAME one-unit shape.
            // The CPU overwrites the entire projection before any consumer.
        }
        // SAFETY: the unit owns out columns c0..c1 of all n rows; inputs are read-only.
        unsafe {
            if w.is_f32() {
                // ggml_compute_forward_mul_mat, vec_dot path (F32: one row per vec_dot)
                for t in 0..n {
                    for c in c0..c1 {
                        *op.out.add(t * op.os + c) = ffi::oxe_vec_dot_f32(k as i32, w.f32().add(c * k), op.xin.add(t * k));
                    }
                }
            } else {
                let q = w.qtype();
                let rb = q.act().bytes(k);
                let full = n - n % 4;
                if full > 0 {
                    if op.tile {
                        kernels_sys::gemm_tile_raw(q, k, op.out.add(c0), op.os, w.rows_ptr(0, c0), op.act, full, c1 - c0);
                    } else {
                        gemm_raw(q, k, op.out.add(c0), op.os, w.rows_ptr(0, c0), op.act, full, c1 - c0);
                    }
                }
                for t in full..n {
                    gemv_raw(q, k, op.out.add(t * op.os + c0), w.rows_ptr(0, c0), op.act.add(t * rb), c1 - c0);
                }
            }
            if !op.add.is_null() {
                for t in 0..n {
                    for c in c0..c1 {
                        let o = op.out.add(t * op.os + c);
                        *o += *op.add.add(t * op.os + c);
                    }
                }
            }
        }
    }

    /// Run the dense projections `ops` (dynamic units of CU weight rows) over n rows. No barrier.
    fn pf_dense(&self, pf: &Pf, ts: &mut PfThread, ops: &[Dense], n: usize) {
        let ctr = step_ctr(pf, ts);
        let next = || ctr.fetch_add(1, Relaxed);
        let total: usize = ops.iter().map(|o| o.units()).sum();
        loop {
            let mut u = next();
            if u >= total {
                break;
            }
            for op in ops {
                if u < op.units() {
                    let c0 = u * op.cu;
                    let c1 = (c0 + op.cu).min(op.w.rows);
                    self.dense_unit(op, n, c0, c1);
                    break;
                }
                u -= op.units();
            }
        }
    }

    fn dense(&self, pf: &Pf, n: usize, w: Tid, xin: *const f32, out: *mut f32, os: usize, add: *const f32) -> Dense {
        let w = *self.ws.get(w);
        let act = if w.is_f32() { std::ptr::null() } else { self.act_buf(pf, w.qtype().act()).p(0) as *const u8 };
        let mask = pf.mask.load(Relaxed);
        let tile = !w.is_f32()
            && kernels_sys::gemm_tile_has(w.qtype())
            && if w.qtype() == kernels_sys::QType::Q6K { mask & PF_Q6K_TILE != 0 } else { mask & PF_TILE_DENSE != 0 };
        let cu = if w.is_f32() && mask & PF_F32_CU8 != 0 { 8 } else { CU };
        #[cfg(feature = "npu-prefill")]
        let npu=self.npu_prefill.as_ref().is_some_and(|p|p.selected(w.ptr,n));
        #[cfg(feature = "npu-prefill")]
        let cu=if npu {w.rows}else{cu};
        #[cfg(not(feature = "npu-prefill"))]
        let _=n;
        Dense { w, xin, act, out, os, add, tile, cu,
            #[cfg(feature = "npu-prefill")]
            npu,
        }
    }

    // ------------------------------------------------------------ delta-net layer

    fn pf_recurrent(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, l: usize, n: usize) {
        let d = &self.plan.desc;
        let lt = &self.plan.layers[l];
        let Mixer::Recurrent { qkv, z, beta, alpha, conv1d, a, dt_bias, norm, out } = lt.mixer else { unreachable!() };
        let (tid, nth) = (ctx.tid, pf.nth);
        let (ds, nk, nv, hv) = (d.ssm_d_state, d.ssm_n_kheads, d.ssm_n_vheads, d.ssm_head_v_dim());
        let nch = d.ssm_conv_channels();
        let di = d.ssm_d_inner;
        let eps = d.rms_eps;
        let trace = self.trace.is_some();

        // [1] attn_norm + activations of qkv / z / beta / alpha
        let ats = self.acts_of(&[qkv, z, beta, alpha]);
        self.pf_norm_act(ctx, pf, ts, &pf.x, Rows { r0: 0, n }, lt.attn_norm, &ats, None, l, "attn_norm", PH_REC_NORM_QUANT);

        // [2] projections (with PF_WAIT_DIET the z projection moves to phase [5])
        let pmask = pf.mask.load(Relaxed);
        let diet = pmask & PF_WAIT_DIET != 0 && !trace;
        let fuse_conv = pmask & PF_CONV_FUSE != 0 && !trace;
        let xn = pf.xn.p(0) as *const f32;
        let zop = self.dense(pf, n, z, xn, pf.z.p(0), di, std::ptr::null());
        // causal conv (+ state) and SiLU of channels c0..c1 over the n tokens (phase [3], or inside
        // the qkv units with PF_CONV_FUSE)
        let ri = self.rec_index[l];
        // M3.1: recurrent state of the current slot (MTP verify slots)
        let cur = self.cur_slot.load(Relaxed);
        let (cb, sb) = (cur * self.conv_len_all, cur * self.ssm_len_all);
        let kc = d.ssm_conv_kernel;
        let conv_len = (kc - 1) * nch;
        let wconv = self.ws.get(conv1d).f32_slice();
        let conv_chunk = |c0: usize, c1: usize| {
            // SAFETY: conv-state rows and conv output columns c0..c1 are owned by the caller's unit;
            // the qkv columns c0..c1 of all tokens are complete (same unit, or the barrier).
            unsafe {
                let cs = self.conv_state.at(cb + ri * conv_len + c0 * (kc - 1)..cb + ri * conv_len + c1 * (kc - 1));
                for t in 0..n {
                    ffi::oxe_ssm_conv_step_f32(cs.as_mut_ptr(), pf.qkv.p(t * nch + c0), wconv[c0 * kc..].as_ptr(), (c1 - c0) as i32, kc as i32, pf.conv.p(t * nch + c0));
                    // SiLU in place (ggml_vec_silu_f32 is element-wise; chunks that are multiples of
                    // 16 channels keep every element on the same NEON path as llama.cpp's full row)
                    let row = pf.conv.s(t * nch + c0..t * nch + c1);
                    ffi::oxe_silu_f32((c1 - c0) as i32, row.as_mut_ptr(), row.as_ptr());
                }
            }
        };
        {
            let d_qkv = self.dense(pf, n, qkv, xn, pf.qkv.p(0), nch, std::ptr::null());
            let d_beta = self.dense(pf, n, beta, xn, pf.beta.p(0), nv, std::ptr::null());
            let d_alpha = self.dense(pf, n, alpha, xn, pf.alpha.p(0), nv, std::ptr::null());
            let d_z = self.dense(pf, n, z, xn, pf.z.p(0), di, std::ptr::null());
            let with_z = [d_qkv, d_z, d_beta, d_alpha];
            let ops: &[Dense] = if diet { &with_z[..1] } else { &with_z[..2] };
            // (order: qkv first, then z unless it moved, then beta, alpha)
            let tail = [with_z[2].clone_op(), with_z[3].clone_op()];
            if fuse_conv && d_qkv_units_ok(&with_z[0]) {
                // PF_CONV_FUSE: the pf_dense loop with the conv of a qkv unit's channels right after it
                let all: Vec<&Dense> = ops.iter().chain(tail.iter()).collect();
                let ctr = step_ctr(pf, ts);
                let total: usize = all.iter().map(|o| o.units()).sum();
                loop {
                    let mut u = ctr.fetch_add(1, Relaxed);
                    if u >= total {
                        break;
                    }
                    for (oi, op) in all.iter().enumerate() {
                        if u < op.units() {
                            let c0 = u * op.cu;
                            let c1 = (c0 + op.cu).min(op.w.rows);
                            self.dense_unit(op, n, c0, c1);
                            if oi == 0 {
                                conv_chunk(c0, c1);
                            }
                            break;
                        }
                        u -= op.units();
                    }
                }
            } else {
                let all: Vec<Dense> = ops.iter().chain(tail.iter()).map(|o| o.clone_op()).collect();
                self.pf_dense(pf, ts, &all, n);
            }
        }
        self.bar(ctx, pf, PH_REC_PROJ_DENSE);
        if trace && tid == 0 {
            // SAFETY: complete and read-only after the barrier.
            unsafe {
                self.ptr(l, "linear_attn_qkv_mixed", n * nch, 0, pf.qkv.s(0..n * nch));
                self.ptr(l, "z", n * di, 0, pf.z.s(0..n * di));
                self.ptr(l, "beta", n * nv, 0, pf.beta.s(0..n * nv));
                self.ptr(l, "alpha", n * nv, 0, pf.alpha.s(0..n * nv));
            }
        }
        if self.trace.is_some() {
            self.bar(ctx, pf, PH_TRACE); // trace snapshots must not race with the next step
        }

        // [3] causal conv (+ state) and SiLU, by channel chunks, token by token (done inside the qkv
        //     units with PF_CONV_FUSE: no work and no barrier here then)
        // (chunks are multiples of 16 channels, so ggml_vec_silu_f32 takes the same NEON path for
        // every element as on llama.cpp's full row; PF_WAIT_DIET: 128-channel units, dynamic)
        let conv_fused = fuse_conv && d_qkv_units_ok_w(self.ws.get(qkv));
        if !conv_fused {
            let cc = if diet { 128 } else { 256 };
            let conv_ctr = step_ctr(pf, ts);
            let conv_units = nch.div_ceil(cc);
            let mut conv_static = split(conv_units, tid, nth);
            loop {
                let u = if diet {
                    conv_ctr.fetch_add(1, Relaxed)
                } else {
                    match conv_static.next() {
                        Some(u) => u,
                        None => break,
                    }
                };
                if u >= conv_units {
                    break;
                }
                conv_chunk(u * cc, ((u + 1) * cc).min(nch));
            }
            self.bar(ctx, pf, PH_REC_CONV_SILU);
        }
        if trace && tid == 0 {
            // SAFETY: complete after the barrier.
            unsafe { self.ptr(l, "conv_output_silu", n * nch, 0, pf.conv.s(0..n * nch)) };
        }
        if self.trace.is_some() {
            self.bar(ctx, pf, PH_TRACE); // trace snapshots must not race with the next step
        }

        // [4] l2 norm of q, k (per token, key head); gates and beta (per token, value head)
        let (wa, wdt) = (self.ws.get(a).f32_slice(), self.ws.get(dt_bias).f32_slice());
        for t in share(pf, ts, n, 2, tid) {
            // SAFETY: token t's rows are owned by this thread in this step.
            unsafe {
                for g in 0..nk {
                    ops::gdn_l2_norm(pf.conv.s(t * nch + g * ds..t * nch + (g + 1) * ds), eps);
                    ops::gdn_l2_norm(pf.conv.s(t * nch + nk * ds + g * ds..t * nch + nk * ds + (g + 1) * ds), eps);
                }
                for vh in 0..nv {
                    let ab = [*pf.alpha.p(t * nv + vh) + wdt[vh]];
                    let mut sp = [0f32];
                    ops::softplus(&ab, &mut sp);
                    *pf.gate.p(t * nv + vh) = sp[0] * wa[vh];
                    *pf.beta_s.p(t * nv + vh) = ops::sigmoid1(*pf.beta.p(t * nv + vh));
                }
            }
        }
        self.bar(ctx, pf, PH_REC_L2_GATES);
        if trace && tid == 0 {
            // SAFETY: complete after the barrier.
            unsafe {
                let mut q = Vec::with_capacity(n * nk * ds);
                let mut k = Vec::with_capacity(n * nk * ds);
                for t in 0..n {
                    q.extend_from_slice(pf.conv.s(t * nch..t * nch + nk * ds));
                    k.extend_from_slice(pf.conv.s(t * nch + nk * ds..t * nch + 2 * nk * ds));
                }
                self.ptr(l, "q_conv_predelta", q.len(), 0, &q);
                self.ptr(l, "k_conv_predelta", k.len(), 0, &k);
                self.ptr(l, "gate", n * nv, 0, pf.gate.s(0..n * nv));
                self.ptr(l, "beta_sigmoid", n * nv, 0, pf.beta_s.s(0..n * nv));
            }
        }
        if self.trace.is_some() {
            self.bar(ctx, pf, PH_TRACE); // trace snapshots must not race with the next step
        }

        // [5] gated delta rule per value head over the tokens, then the gated RMSNorm with z
        let ssm_len = nv * hv * hv;
        let wn = self.ws.get(norm).f32_slice();
        let ctr = step_ctr(pf, ts);
        let next = || ctr.fetch_add(1, Relaxed);
        let gdn_x4 = pf.mask.load(Relaxed) & PF_GDN_X4 != 0;
        let z_units = if diet { zop.units() } else { 0 };
        loop {
            let vh = next();
            if vh >= nv + z_units {
                break;
            }
            if vh >= nv {
                // PF_WAIT_DIET: a unit of the z projection (threads that got fewer heads take these)
                let c0 = (vh - nv) * zop.cu;
                self.dense_unit(&zop, n, c0, (c0 + zop.cu).min(zop.w.rows));
                continue;
            }
            let kh = vh % nk;
            // SAFETY: the state of value head vh and O columns of vh are owned by this unit.
            unsafe {
                let s = self.ssm_state.at(sb + ri * ssm_len + vh * hv * hv..sb + ri * ssm_len + (vh + 1) * hv * hv);
                for t in 0..n {
                    let base = t * nch;
                    let step = if gdn_x4 { ffi::oxe_gated_delta_net_step_f32_x4 } else { ffi::oxe_gated_delta_net_step_f32 };
                    step(
                        s.as_mut_ptr(),
                        pf.conv.p(base + kh * ds),
                        pf.conv.p(base + nk * ds + kh * ds),
                        pf.conv.p(base + 2 * nk * ds + vh * hv),
                        *pf.gate.p(t * nv + vh),
                        *pf.beta_s.p(t * nv + vh),
                        hv as i64,
                        ts.delta.as_mut_ptr(),
                        ts.gout.as_mut_ptr(),
                    );
                    if trace {
                        self.ptr(l, "attn_output", n * nv * hv, (t * nv + vh) * hv, &ts.gout);
                    }
                    let o = pf.o.s(t * di + vh * hv..t * di + (vh + 1) * hv);
                    if diet {
                        // z is still being computed in this phase: the gate is applied in [6]
                        ops::rms_norm_mul(&ts.gout, wn, o, eps);
                    } else {
                        ops::rms_norm_mul(&ts.gout, wn, &mut ts.tmpn[..hv], eps);
                        ops::silu(pf.z.s(t * di + vh * hv..t * di + (vh + 1) * hv), &mut ts.zs);
                        for j in 0..hv {
                            o[j] = ts.tmpn[j] * ts.zs[j];
                        }
                    }
                }
            }
        }
        self.bar(ctx, pf, PH_REC_DELTA_RULE);

        // [6] ssm_out (+ residual) -> x1
        // (with PF_WAIT_DIET the SiLU(z) gate of [5] is applied there, row group by row group, before the quantise)
        self.pf_mixer_out(ctx, pf, ts, l, n, out, di, "linear_attn_out", PH_REC_OUT_QUANT, PH_REC_OUT_DENSE, if diet { Some((nv, hv)) } else { None });
    }

    /// Quantise O, project with `wout`, add the layer input: x1 = W.o + x (all n rows). The
    /// quantise and the projection end with barriers of phases `phq` and `phd`.
    #[allow(clippy::too_many_arguments)]
    fn pf_mixer_out(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, l: usize, n: usize, wout: Tid, width: usize, name: &str, phq: usize, phd: usize, zgate: Option<(usize, usize)>) {
        let de = self.plan.desc.n_embd;
        let w = *self.ws.get(wout);
        if !w.is_f32() || zgate.is_some() {
            // per 4-row group: (PF_WAIT_DIET) o *= SiLU(z) per (token, value head) exactly as phase [5]
            // does it, then the forward_mul_mat quantise of the group
            let at = if w.is_f32() { None } else { Some(w.qtype().act()) };
            let full = n - n % 4;
            for g in share(pf, ts, n.div_ceil(4), 1, ctx.tid) {
                let (a, b) = (4 * g, (4 * g + 4).min(n));
                // SAFETY: rows a..b of o and of the activation buffer are owned by this unit; z complete (barrier).
                unsafe {
                    if let Some((nv, hv)) = zgate {
                        for t in a..b {
                            for vh in 0..nv {
                                ops::silu(pf.z.s(t * width + vh * hv..t * width + (vh + 1) * hv), &mut ts.zs);
                                let o = pf.o.s(t * width + vh * hv..t * width + (vh + 1) * hv);
                                for j in 0..hv {
                                    o[j] *= ts.zs[j];
                                }
                            }
                        }
                    }
                    if let Some(at) = at {
                        let buf = self.act_buf(pf, at);
                        let rb = at.bytes(width);
                        if b - a == 4 && b <= full {
                            quantize_mat4_raw(at, pf.o.p(a * width), buf.p(a * rb), width);
                        } else {
                            for i in a..b {
                                quantize_row_raw(at, pf.o.p(i * width), buf.p(i * rb), width);
                            }
                        }
                    }
                }
            }
        }
        self.bar(ctx, pf, phq);
        let op = self.dense(pf, n, wout, pf.o.p(0), pf.x1.p(0), de, pf.x.p(0));
        self.pf_dense(pf, ts, &[op], n);
        self.bar(ctx, pf, phd);
        if self.trace.is_some() && ctx.tid == 0 {
            // SAFETY: complete after the barrier.
            unsafe { self.ptr(l, "attn_residual", n * de, 0, pf.x1.s(0..n * de)) };
            let _ = name; // the projection output itself is not kept (fused with the residual)
        }
        if self.trace.is_some() {
            self.bar(ctx, pf, PH_TRACE); // trace snapshots must not race with the next step
        }
    }

    // ------------------------------------------------------------ attention layer

    fn pf_attention(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, l: usize, n: usize, p0: usize) {
        let d = &self.plan.desc;
        let lt = &self.plan.layers[l];
        let Mixer::Attention { q, k, v, q_norm, k_norm, out } = lt.mixer else { unreachable!() };
        let tid = ctx.tid;
        let hd = d.head_dim;
        let nh = d.n_head;
        let nkv = d.n_head_kv;
        let group = nh / nkv;
        let att_w = nh * hd;
        let kvw = nkv * hd;
        let eps = d.rms_eps;
        let trace = self.trace.is_some();

        // [1] norm + activations
        let ats = self.acts_of(&[q, k, v]);
        self.pf_norm_act(ctx, pf, ts, &pf.x, Rows { r0: 0, n }, lt.attn_norm, &ats, None, l, "attn_norm", PH_ATT_NORM_QUANT);

        // [2] Q(+gate), K, V projections
        let xn = pf.xn.p(0) as *const f32;
        let ops = [
            self.dense(pf, n, q, xn, pf.qkv.p(0), 2 * att_w, std::ptr::null()),
            self.dense(pf, n, k, xn, pf.kr.p(0), kvw, std::ptr::null()),
            self.dense(pf, n, v, xn, pf.vr.p(0), kvw, std::ptr::null()),
        ];
        self.pf_dense(pf, ts, &ops, n);
        self.bar(ctx, pf, PH_ATT_PROJ_DENSE);

        // [3] q/k norm, RoPE, K/V cache writes, f16 Q (head-major) per token
        let ai = self.att_index[l];
        let kv_len = nkv * self.n_ctx * hd;
        let (wqn, wkn) = (self.ws.get(q_norm).f32_slice(), self.ws.get(k_norm).f32_slice());
        for t in share(pf, ts, n, 2, tid) {
            let pos = p0 + t;
            // SAFETY: token t's rows (q16, K/V cache at pos) are owned by this thread; the rope
            // cache is read-only during the run.
            unsafe {
                let cache = pf.rope.s(t * hd..(t + 1) * hd);
                for h in 0..nh {
                    let src = pf.qkv.s(t * 2 * att_w + h * 2 * hd..t * 2 * att_w + h * 2 * hd + hd);
                    let qn = &mut ts.tmpn[..hd];
                    ops::rms_norm_mul(src, wqn, qn, eps);
                    if trace {
                        self.ptr(l, "Qcur_normed", n * att_w, t * att_w + h * hd, qn);
                    }
                    ffi::oxe_rope_neox_inplace(qn.as_mut_ptr(), d.n_rot as i32, cache.as_ptr());
                    if trace {
                        self.ptr(l, "Qcur", n * att_w, t * att_w + h * hd, qn);
                    }
                    pf.qf.s(t * att_w + h * hd..t * att_w + (h + 1) * hd).copy_from_slice(qn);
                    let c = ((h / group) * pf.ub + t) * group + h % group;
                    ops::fp32_to_fp16(qn, pf.q16.s(c * hd..(c + 1) * hd));
                    if self.kv.k8() {
                        let rb = kv8::row_bytes(hd);
                        kv8::quant_row(qn, pf.q8.s(c * rb..(c + 1) * rb));
                    }
                }
                for kvh in 0..nkv {
                    let src = pf.kr.s(t * kvw + kvh * hd..t * kvw + (kvh + 1) * hd);
                    let kn = &mut ts.tmpn[..hd];
                    ops::rms_norm_mul(src, wkn, kn, eps);
                    if trace {
                        self.ptr(l, "Kcur_normed", n * kvw, t * kvw + kvh * hd, kn);
                    }
                    ffi::oxe_rope_neox_inplace(kn.as_mut_ptr(), d.n_rot as i32, cache.as_ptr());
                    #[cfg(feature="f32-reference")]
                    if let Some(reference)=&self.f32_reference {reference.kv.put_k(ai,kvh,pos,kn);}
                    if trace {
                        self.ptr(l, "Kcur", n * kvw, t * kvw + kvh * hd, kn);
                    }
                    if self.kv.k8() {
                        // int8 K: the same quantisation of the same f32 row as decode's writer
                        let at = self.k8row(ai, kvh, pos);
                        kv8::quant_row(kn, self.kcache8.at(at..at + kv8::row_bytes(hd)));
                    } else {
                        let at = ai * kv_len + (kvh * self.n_ctx + pos) * hd;
                        ops::fp32_to_fp16(kn, self.kcache.at(at..at + hd));
                    }
                }
                if self.kv.v8() {
                    // int8 V: one Q8_0 row per (KV head, position), from the f32 projection
                    for kvh in 0..nkv {
                        let at = self.v8row(ai, kvh, pos);
                        kv8::quant_row(pf.vr.s(t * kvw + kvh * hd..t * kvw + (kvh + 1) * hd), self.vcache8.at(at..at + kv8::row_bytes(hd)));
                    }
                }
            }
        }
        // V^T cache write, rows outer and this thread's tokens inner: a V^T row's positions of
        // consecutive tokens share cache lines (32 positions each), so every line is written
        // whole instead of 512 lines being touched once per token (aliasing sweep, hit 1). The
        // same single-element conversion per (token, row) as before.
        if !self.kv.v8() {
            // (blocks of 32 positions = one cache line of a V^T row)
            for blk in share(pf, ts, n.div_ceil(32), 1, tid) {
                let tr = blk * 32..((blk + 1) * 32).min(n);
                let mut h16 = [0u16; 1];
                for r in 0..kvw {
                    // SAFETY: positions p0 + tr of every V^T row are owned by this thread; vr read-only.
                    unsafe {
                        let row = self.vcache.ptr_at(self.vrow(ai, r) + p0 + tr.start);
                        for (i, t) in tr.clone().enumerate() {
                            #[cfg(feature="f32-reference")]
                            if let Some(reference)=&self.f32_reference {reference.kv.put_v(ai,r,p0+t,*pf.vr.p(t*kvw+r));}
                            ops::fp32_to_fp16(&[*pf.vr.p(t * kvw + r)], &mut h16);
                            *row.add(i) = h16[0];
                        }
                    }
                }
            }
        }
        self.bar(ctx, pf, PH_ATT_PREP);
        if trace && tid == 0 {
            // SAFETY: complete after the barrier.
            unsafe {
                self.ptr(l, "Qcur_full", n * 2 * att_w, 0, pf.qkv.s(0..n * 2 * att_w));
                self.ptr(l, "Vcur", n * kvw, 0, pf.vr.s(0..n * kvw));
            }
        }
        if self.trace.is_some() {
            self.bar(ctx, pf, PH_TRACE); // trace snapshots must not race with the next step
        }

        // [4] attention per (query head, token block): per token the decode kernel
        //     (vec_dot_f16 K^T q over positions <= pos, softmax over n_kv with the mask,
        //     probabilities to f16, vec_dot_f16 with V^T over n_kv)
        let n_kv = padded_n_kv(p0 + n - 1, self.n_ctx);
        let scale = 1.0 / (hd as f32).sqrt();
        #[cfg(feature="f32-reference")]
        if let Some(reference)=&self.f32_reference {
            // Full f32 K/Q/V/probabilities. qf already holds normalized, roped
            // f32 queries; do not read q16, p16 or either f16 KV cache here.
            let ctr=step_ctr(pf,ts);
            loop{
                let u=ctr.fetch_add(1,Relaxed);if u>=n*nh{break;}
                let (t,h)=(u/nh,u%nh);let positions=p0+t+1;let kvh=h/group;
                unsafe{
                    for pos in 0..positions{
                        ts.kq[pos]=ffi::oxe_vec_dot_f32(hd as i32,reference.kv.krow(ai,kvh,pos),pf.qf.p(t*att_w+h*hd))*scale;
                    }
                    ops::soft_max(&ts.kq[..positions],&mut ts.wp[..positions],&mut ts.mask[..positions]);
                    for dd in 0..hd{ts.aout[dd]=ffi::oxe_vec_dot_f32(positions as i32,reference.kv.vcol(ai,kvh,dd),ts.wp.as_ptr());}
                    let gsrc=pf.qkv.s(t*2*att_w+h*2*hd+hd..t*2*att_w+(h+1)*2*hd);
                    ops::sigmoid(gsrc,&mut ts.gsig);
                    let out=pf.o.s(t*att_w+h*hd..t*att_w+(h+1)*hd);
                    for dd in 0..hd{out[dd]=ts.aout[dd]*ts.gsig[dd];}
                }
            }
            self.bar(ctx,pf,PH_ATT_CORE);
            return self.pf_mixer_out(ctx,pf,ts,l,n,out,att_w,"attn_output",PH_ATT_OUT_QUANT,PH_ATT_OUT_DENSE,None);
        }
        // O-117 quality runner (bench / tests): the f32-attention reference for every row
        #[cfg(any(test, feature = "f32-reference"))]
        if self.pf_ref {
            self.pf_attention_ref(pf, ts, l, n, p0);
            self.bar(ctx, pf, PH_ATT_CORE);
            return self.pf_mixer_out(ctx, pf, ts, l, n, out, att_w, "attn_output", PH_ATT_OUT_QUANT, PH_ATT_OUT_DENSE, None);
        }
        // PREFILL-LONG: fa2 units exist only when a token of this pass reaches beyond the cutoff
        // (`pf.cap` under fa2: PFA_MIN_KV or the auto cutoff; pf.fa2 implies f16 KV, no trace, no
        // f32 reference); the rest stays on today's kernels
        let fa2 = pf.fa2 && padded_n_kv(p0 + n - 1, self.n_ctx) > pf.cap;
        if fa2 || (self.attn_tiled && !trace) || self.kv.any8() {
            // n_kv of the llama.cpp ubatch (of `lub` tokens, from p0) that holds token t
            let lub = pf.lub.load(Relaxed).max(1);
            let n_ctx = self.n_ctx;
            self.pf_attention_tiled(pf, ts, l, n, p0, tid, fa2, |t| padded_n_kv(p0 + ((t / lub + 1) * lub).min(n) - 1, n_ctx));
            self.bar(ctx, pf, PH_ATT_CORE_TILED);
            #[cfg(any(test, feature = "f32-reference"))]
            self.pf_attn_check(ctx, pf, ts, l, n, p0);
            return self.pf_mixer_out(ctx, pf, ts, l, n, out, att_w, "attn_output", PH_ATT_OUT_QUANT, PH_ATT_OUT_DENSE, None);
        }
        let nb = n.div_ceil(ATT_TB);
        let ctr = step_ctr(pf, ts);
        // THREADS-BY-PHASE: only the first attn_nth threads take units (same bits: dynamic units)
        let mine = tid < pf.attn_nth.load(Relaxed);
        loop {
            let u = if mine { ctr.fetch_add(1, Relaxed) } else { usize::MAX };
            if u >= nh * nb {
                break;
            }
            let (h, b) = (u / nb, u % nb);
            let (t0, t1) = (b * ATT_TB, ((b + 1) * ATT_TB).min(n));
            #[cfg(any(test, feature = "f32-reference"))]
            self.pf_count_unit(PFU_PER_HEAD);
            let kvh = h / group;
            let kbase = ai * kv_len + kvh * self.n_ctx * hd;
            let vbase = self.vrow(ai, kvh * hd);
            for t in t0..t1 {
                let pos = p0 + t;
                // SAFETY: K/V cache rows < p0 + n are complete (step [3]) and read-only here;
                // q rows read-only; scratch is this thread's; O columns of head h, row t owned.
                unsafe {
                    ffi::oxe_attn_head_f16(
                        pf.qf.p(t * att_w + h * hd),
                        self.kcache.ptr_at(kbase),
                        hd as i64,
                        self.kcache.ptr_at(kbase + pos * hd),
                        self.vcache.ptr_at(vbase),
                        self.v_stride as i64,
                        n_kv as i32,
                        pos as i32,
                        hd as i32,
                        scale,
                        ts.p16.as_mut_ptr().add(n_kv),
                        ts.kq.as_mut_ptr(),
                        ts.mask.as_mut_ptr(),
                        ts.wp.as_mut_ptr(),
                        ts.p16.as_mut_ptr(),
                        ts.aout.as_mut_ptr(),
                    );
                    if trace {
                        self.ptr(l, "kq_soft_max", nh * n * n_kv, (h * n + t) * n_kv, &ts.kq[..n_kv]);
                    }
                    let gsrc = pf.qkv.s(t * 2 * att_w + h * 2 * hd + hd..t * 2 * att_w + (h + 1) * 2 * hd);
                    ops::sigmoid(gsrc, &mut ts.gsig);
                    let o = pf.o.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                    for dd in 0..hd {
                        o[dd] = ts.aout[dd] * ts.gsig[dd];
                    }
                    if trace {
                        self.ptr(l, "attn_pregate", n * att_w, t * att_w + h * hd, &ts.aout[..hd]);
                        self.ptr(l, "attn_gated", n * att_w, t * att_w + h * hd, o);
                    }
                }
            }
        }
        self.bar(ctx, pf, PH_ATT_CORE);
        #[cfg(any(test, feature = "f32-reference"))]
        self.pf_attn_check(ctx, pf, ts, l, n, p0);

        // [5] attn_output (+ residual)
        self.pf_mixer_out(ctx, pf, ts, l, n, out, att_w, "attn_output", PH_ATT_OUT_QUANT, PH_ATT_OUT_DENSE, None);
    }

    /// Tiled prefill attention (docs/lab-notes/M3-ATTN.md). Units = (KV head, block of ATT_TT tokens),
    /// dynamically scheduled; per unit the 8 x ATT_TT query columns share every K row (scores)
    /// and every V^T row (outputs). Each score / output is one `ggml_vec_dot_f16` with the
    /// reference's arithmetic, each softmax row is `soft_max_f32` over the token's padded n_kv
    /// (`n_kv_of(t)`), so the result is bitwise the per-(token, head) path. No barrier.
    ///
    /// PREFILL-LONG (`fa2`): a unit whose last token's padded n_kv exceeds the cutoff (`pf.cap`) runs the
    /// fa2 kernel ([`pfattn::Tile`], O(chunk) scratch) instead; the others are unchanged. Only
    /// threads with tid < `attn_nth` take units (THREADS-BY-PHASE; any value gives the same bits).
    #[allow(clippy::too_many_arguments)]
    fn pf_attention_tiled(&self, pf: &Pf, ts: &mut PfThread, l: usize, n: usize, p0: usize, tid: usize, fa2: bool, n_kv_of: impl Fn(usize) -> usize) {
        let d = &self.plan.desc;
        let hd = d.head_dim;
        let nh = d.n_head;
        let nkv = d.n_head_kv;
        let group = nh / nkv;
        let att_w = nh * hd;
        let n_ctx = self.n_ctx;
        let ai = self.att_index[l];
        let kv_len = nkv * n_ctx * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        // PF_WAIT_DIET: a short ubatch gets smaller token blocks until every thread has a unit
        let mut tt = pf.att_tt;
        if pf.mask.load(Relaxed) & PF_WAIT_DIET != 0 {
            while tt > 1 && n.div_ceil(tt) * nkv < pf.nth {
                tt /= 2;
            }
        }
        let nb = n.div_ceil(tt);
        let ctr = step_ctr(pf, ts);
        let mine = tid < pf.attn_nth.load(Relaxed);
        let mut nf = [0i32; 64];
        // moe-prefill switches (same bits either way): scratch column stride, V^T . p blocking.
        // The stride is the scratch capacity (n_ctx, or PFA_MIN_KV under fa2: tiled units then
        // never reach beyond it); only addresses depend on it.
        let mask = pf.mask.load(Relaxed);
        let cs = if mask & PF_ATTN_PAD != 0 { pf.cap + ATT_PAD } else { pf.cap };
        let pb = if mask & PF_ATTN_PB != 0 { attn_pb() } else { 0 };
        loop {
            let u = if mine { ctr.fetch_add(1, Relaxed) } else { usize::MAX };
            if u >= nkv * nb {
                break;
            }
            let (kvh, b) = (u / nb, u % nb);
            let (t0, t1) = (b * tt, ((b + 1) * tt).min(n));
            let ncol = (t1 - t0) * group;
            if fa2 && n_kv_of(t1 - 1) > pf.cap {
                #[cfg(any(test, feature = "f32-reference"))]
                self.pf_count_unit(if pf.fa2x.load(Relaxed) == 0 { PFU_FA2 } else { PFU_FA2X });
                self.pf_fa2_unit(pf, ts, ai, kvh, t0, t1, p0);
                continue;
            }
            #[cfg(any(test, feature = "f32-reference"))]
            self.pf_count_unit(PFU_TILED);
            debug_assert!(n_kv_of(t1 - 1) <= pf.cap && p0 + t1 <= pf.cap, "tiled unit beyond the prefill scratch");
            let rows = p0 + t1; // K rows 0..=p0+t1-1 (rows beyond a token's pos are masked)
            let n32 = rows.div_ceil(32) * 32;
            let kbase = ai * kv_len + kvh * n_ctx * hd;
            let vbase = self.vrow(ai, kvh * hd);
            let c0 = (kvh * pf.ub + t0) * group;
            // SAFETY: K/V cache rows < p0 + n complete (step [3]) and read-only; q16 complete;
            // V^T rows read beyond n32 only by the tail rule (finite cache entries); the unit's
            // scratch is this thread's; O elements (t0..t1, heads of kvh) are owned by the unit.
            unsafe {
                if self.kv.k8() && self.kv8_fused {
                    let rb = kv8::row_bytes(hd);
                    ffi::oxe_dot_q8_tile(hd as i32, self.kcache8.ptr_at(self.k8row(ai, kvh, 0)), rb as i64, rows as i32, pf.q8.p(c0 * rb), rb as i64, ncol as i32, ts.akq.as_mut_ptr(), 1, cs as i64);
                } else if self.kv.k8() {
                    kv8::scores_q8(self.kcache8.ptr_at(self.k8row(ai, kvh, 0)), hd, 0, rows, pf.q16.p(c0 * hd), hd, ncol, ts.akq.as_mut_ptr(), cs, &mut ts.kv8s);
                } else {
                    ffi::oxe_dot_f16_tile(
                        hd as i32,
                        self.kcache.ptr_at(kbase),
                        hd as i64,
                        rows as i32,
                        pf.q16.p(c0 * hd),
                        hd as i64,
                        ncol as i32,
                        ts.akq.as_mut_ptr(),
                        1,
                        cs as i64,
                        std::ptr::null(),
                    );
                }
                for col in 0..ncol {
                    let t = t0 + col / group;
                    nf[col] = n_kv_of(t) as i32;
                    ffi::oxe_attn_softmax_row_f16(
                        ts.akq.as_ptr().add(col * cs),
                        (p0 + t) as i32,
                        nf[col],
                        scale,
                        ts.wp.as_mut_ptr(),
                        ts.mask.as_mut_ptr(),
                        ts.ap16.as_mut_ptr().add(col * cs),
                        n32 as i32,
                    );
                }
                if self.kv.v8() && self.kv8_fused {
                    // fused int8 V: 256-position chunk partials summed in chunk order, as decode
                    let aot = &mut ts.aot[..hd * ncol];
                    aot.fill(0.0);
                    let mut q0 = 0;
                    while q0 < n32 {
                        let q1 = (q0 + kv8::PV_CHUNK).min(n32);
                        let pacc = ts.kv8s.pacc.as_mut_ptr();
                        kv8::pv_chunk(self.kv8_pv, self.vcache8.ptr_at(self.v8row(ai, kvh, 0)), hd, q0, q1, ts.ap16.as_ptr(), cs, ncol, pacc, &mut ts.kv8s);
                        for c in 0..ncol {
                            for dd in 0..hd {
                                aot[dd * ncol + c] += ts.kv8s.pacc[c * hd + dd];
                            }
                        }
                        q0 = q1;
                    }
                } else if self.kv.v8() {
                    // int8 V rows of positions [0, n32): all head dims, one Q8_0 block at a time
                    for bi in 0..hd / kv8::QK {
                        let (r0, r1) = (bi * kv8::QK, (bi + 1) * kv8::QK);
                        kv8::outputs_q8(
                            self.vcache8.ptr_at(self.v8row(ai, kvh, 0)),
                            hd,
                            r0,
                            r1,
                            n32,
                            ts.ap16.as_ptr(),
                            cs,
                            ncol,
                            ts.aot.as_mut_ptr().add(r0 * ncol),
                            &mut ts.kv8s,
                        );
                    }
                } else if pb > 0 && n32 > pb {
                    ffi::oxe_dot_f16_tile_pb(
                        n32 as i32,
                        self.vcache.ptr_at(vbase),
                        self.v_stride as i64,
                        hd as i32,
                        ts.ap16.as_ptr(),
                        cs as i64,
                        ncol as i32,
                        ts.aot.as_mut_ptr(),
                        ncol as i64,
                        1,
                        nf.as_ptr(),
                        pb as i32,
                        ts.ast.as_mut_ptr() as *mut u8,
                    );
                } else {
                    ffi::oxe_dot_f16_tile(
                        n32 as i32,
                        self.vcache.ptr_at(vbase),
                        self.v_stride as i64,
                        hd as i32,
                        ts.ap16.as_ptr(),
                        cs as i64,
                        ncol as i32,
                        ts.aot.as_mut_ptr(),
                        ncol as i64,
                        1,
                        nf.as_ptr(),
                    );
                }
                for col in 0..ncol {
                    let (t, h) = (t0 + col / group, kvh * group + col % group);
                    let gsrc = pf.qkv.s(t * 2 * att_w + h * 2 * hd + hd..t * 2 * att_w + (h + 1) * 2 * hd);
                    ops::sigmoid(gsrc, &mut ts.gsig);
                    let o = pf.o.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                    for dd in 0..hd {
                        o[dd] = ts.aot[dd * ncol + col] * ts.gsig[dd];
                    }
                }
            }
        }
    }

    /// PREFILL-LONG: one fa2 unit (KV head `kvh`, tokens t0..t1 of the pass at p0): the
    /// [`pfattn::Tile`] kernel over K / V^T of the cache (rows < p0 + t1 complete after step [3]),
    /// then the sigmoid gate as in the tiled path.
    #[allow(clippy::too_many_arguments)]
    fn pf_fa2_unit(&self, pf: &Pf, ts: &mut PfThread, ai: usize, kvh: usize, t0: usize, t1: usize, p0: usize) {
        let d = &self.plan.desc;
        let hd = d.head_dim;
        let nkv = d.n_head_kv;
        let group = d.n_head / nkv;
        let att_w = d.n_head * hd;
        let n_ctx = self.n_ctx;
        let ncol = (t1 - t0) * group;
        let c0 = (kvh * pf.ub + t0) * group;
        let tile = pfattn::Tile {
            hd,
            ncol,
            group,
            pos0: p0 + t0,
            scale: 1.0 / (hd as f32).sqrt(),
            k: self.kcache.ptr_at(ai * nkv * n_ctx * hd + kvh * n_ctx * hd),
            vt: self.vcache.ptr_at(self.vrow(ai, kvh * hd)),
            v_stride: self.v_stride,
            n_ctx,
            q16: pf.q16.p(c0 * hd),
        };
        let x = pf.fa2x.load(Relaxed);
        // SAFETY: K / V^T cache rows < p0 + t1 complete (step [3]) and read-only; V^T rows beyond
        // them finite up to n_ctx (zero pages or earlier rows); q16 columns of the unit complete;
        // the fa2 scratch is this thread's; O elements (t0..t1, heads of kvh) owned by the unit.
        unsafe {
            if x == 0 {
                tile.run(&mut ts.fa2);
            } else {
                tile.run_x(&mut ts.fa2, pfattn::XParams { fold: x >> 16, ch: x & 0xffff });
            }
            for col in 0..ncol {
                let (t, h) = (t0 + col / group, kvh * group + col % group);
                let gsrc = pf.qkv.s(t * 2 * att_w + h * 2 * hd + hd..t * 2 * att_w + (h + 1) * 2 * hd);
                ops::sigmoid(gsrc, &mut ts.gsig);
                let o = pf.o.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                for (dd, x) in ts.fa2.out(col).iter().enumerate() {
                    o[dd] = x * ts.gsig[dd];
                }
                #[cfg(test)]
                if self.pf_degrade != 0.0 {
                    o.iter_mut().for_each(|x| *x *= 1.0 + self.pf_degrade);
                }
            }
        }
    }

    // ------------------------------------------------------------ MoE FFN

    fn pf_moe(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, l: usize, rows: Rows) {
        let d = &self.plan.desc;
        let lt = &self.plan.layers[l];
        let tid = ctx.tid;
        let n = rows.n;
        let de = d.n_embd;
        let ne = d.n_expert;
        let ku = d.n_expert_used;
        let ffe = d.ff_exp;
        let ffs = d.ff_shexp;
        let trace = self.trace.is_some();
        let prof = self.cfg.profile;
        let wge = *self.ws.get(lt.gate_exps);
        let wue = *self.ws.get(lt.up_exps);
        let wde = *self.ws.get(lt.down_exps);
        let rat = wge.qtype().act();
        assert_eq!(rat, wue.qtype().act(), "gate/up experts with different activation types");

        // [1] post-attention norm; activations for the shared expert (forward_mul_mat layout)
        //     and per row for the routed experts (forward_mul_mat_id: from_float)
        let ats = self.acts_of(&[lt.gate_shexp, lt.up_shexp]);
        self.pf_norm_act(ctx, pf, ts, &pf.x1, rows, lt.post_norm, &ats, Some(rat), l, "attn_post_norm", PH_MOE_NORM_QUANT);

        // [2] router logits, shared expert gate (vec_dot: 1 row), shared gate/up projections
        let xn = pf.xn.p(0) as *const f32;
        let ops = [
            self.dense(pf, n, lt.router, xn, pf.router.p(0), ne, std::ptr::null()),
            self.dense(pf, n, lt.gate_shexp, xn, pf.shg.p(0), ffs, std::ptr::null()),
            self.dense(pf, n, lt.up_shexp, xn, pf.shu.p(0), ffs, std::ptr::null()),
        ];
        #[cfg(feature="npu-prefill")]
        let shared_overlap=!trace&&self.npu_prefill.as_ref().is_some_and(|p|p.shared_overlap([
            self.ws.get(lt.gate_shexp).ptr,self.ws.get(lt.up_shexp).ptr,self.ws.get(lt.down_shexp).ptr],n));
        #[cfg(not(feature="npu-prefill"))]
        let shared_overlap=false;
        self.pf_dense(pf, ts, if shared_overlap{&ops[..1]}else{&ops}, n);
        let gs = self.ws.get(lt.shexp_gate).f32_slice();
        for t in share(pf, ts, n, 8, tid) {
            // SAFETY: row t of xn read-only; sg[t] owned.
            unsafe {
                let sgv = ops::dot_f32(&gs[..de], pf.xn.s(t * de..(t + 1) * de));
                *pf.sg.p(t) = ops::sigmoid1(sgv);
            }
        }
        self.bar(ctx, pf, PH_MOE_ROUTER_SHARED);

        // [3] routing per token; shared expert swiglu + its down activations
        for t in share(pf, ts, n, 4, tid) {
            // SAFETY: token t's routing rows are owned by this thread.
            unsafe {
                let lg = pf.router.s(t * ne..(t + 1) * ne);
                let probs = pf.probs.s(t * ne..(t + 1) * ne);
                ops::soft_max(lg, probs, &mut ts.wp[..ne]);
                let ids = pf.ids.s(t * ku..(t + 1) * ku);
                ops::argsort_top_k(probs, ku, ids, &mut ts.iscr);
                let w = pf.w.s(t * ku..(t + 1) * ku);
                for j in 0..ku {
                    w[j] = probs[ids[j] as usize];
                }
                let s = ops::sum(w);
                let s = s.min(f32::INFINITY).max(6.103515625e-5);
                for x in w.iter_mut() {
                    *x /= s;
                }
                if !shared_overlap{ops::swiglu(pf.shg.s(t * ffs..(t + 1) * ffs), pf.shu.s(t * ffs..(t + 1) * ffs), pf.shh.s(t * ffs..(t + 1) * ffs));}
            }
        }
        self.bar(ctx, pf, PH_MOE_ROUTE_SWIGLU);
        let wds = *self.ws.get(lt.down_shexp);
        let sh_at = if wds.is_f32() { None } else { Some(wds.qtype().act()) };
        if let Some(at) = sh_at.filter(|_|!shared_overlap) {
            // shared down activations: forward_mul_mat layout over the n rows of shh
            let rb = at.bytes(ffs);
            let full = n - n % 4;
            for g in share(pf, ts, n.div_ceil(4), 1, tid) {
                let (a, b) = (4 * g, (4 * g + 4).min(n));
                // SAFETY: rows a..b owned by this thread.
                unsafe {
                    if b - a == 4 && b <= full {
                        quantize_mat4_raw(at, pf.shh.p(a * ffs), pf.sh_act.p(a * rb), ffs);
                    } else {
                        for i in a..b {
                            quantize_row_raw(at, pf.shh.p(i * ffs), pf.sh_act.p(i * rb), ffs);
                        }
                    }
                }
            }
        }
        if tid == 0 {
            // group the routed rows by expert, token-major then slot (forward_mul_mat_id order),
            // and the work units, largest experts first
            // SAFETY: only thread 0 writes these; the others read them after the barrier.
            unsafe {
                let off = pf.off.s(0..ne + 1);
                off.fill(0);
                let ids = pf.ids.s(0..n * ku);
                for &e in ids.iter() {
                    off[e as usize + 1] += 1;
                }
                if n >= 16 {
                    for e in 0..ne {
                        ROWS_HIST[(off[e + 1] as usize).min(64)].fetch_add(1, Relaxed);
                    }
                }
                for e in 0..ne {
                    off[e + 1] += off[e];
                }
                let mut fill: Vec<u32> = off[..ne].to_vec();
                let order = pf.order.s(0..n * ku);
                for (s, &e) in ids.iter().enumerate() {
                    order[fill[e as usize] as usize] = s as u32;
                    fill[e as usize] += 1;
                }
                let mut ex: Vec<(u32, u32)> = (0..ne).filter(|&e| off[e + 1] > off[e]).map(|e| (off[e + 1] - off[e], e as u32)).collect();
                ex.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                let units = pf.units.s(0..pf.units.len);
                let mut nu = 0;
                // experts with many routed rows are split by columns so they do not finish last
                for &(cnt, e) in &ex {
                    if cnt >= 16 && ffe % 16 == 0 {
                        units[nu] = (e, 0, (ffe / 2) as u32);
                        units[nu + 1] = (e, (ffe / 2) as u32, ffe as u32);
                        nu += 2;
                    } else {
                        units[nu] = (e, 0, ffe as u32);
                        nu += 1;
                    }
                }
                pf.n_units[0].store(nu, Relaxed);
                let mut nd = 0;
                for c0 in (0..de).step_by(DOWN_CU) {
                    for &(_, e) in &ex {
                        units[nu + nd] = (e, c0 as u32, (c0 + DOWN_CU).min(de) as u32);
                        nd += 1;
                    }
                }
                pf.n_units[1].store(nd, Relaxed);
            }
        }
        self.bar(ctx, pf, PH_MOE_UNITS_BUILD);
        if trace && tid == 0 {
            // SAFETY: complete after the barrier.
            unsafe {
                self.ptr(l, "ffn_moe_logits", n * ne, 0, pf.router.s(0..n * ne));
                self.ptr(l, "ffn_moe_probs", n * ne, 0, pf.probs.s(0..n * ne));
                let ids: Vec<f32> = pf.ids.s(0..n * ku).iter().map(|&i| i as f32).collect();
                self.ptr(l, "ffn_moe_topk", n * ku, 0, &ids);
                self.ptr(l, "ffn_moe_weights_norm", n * ku, 0, pf.w.s(0..n * ku));
                let sg = pf.sg.s(0..n).to_vec();
                self.ptr(l, "shared_expert_gate_sigmoid", n, 0, &sg);
            }
        }
        if self.trace.is_some() {
            self.bar(ctx, pf, PH_TRACE); // trace snapshots must not race with the next step
        }

        // [4] routed gate/up (MUL_MAT_ID), shared down projection
        let nu = pf.n_units[0].load(Relaxed);
        let sd = self.dense(pf, n, lt.down_shexp, pf.shh.p(0), pf.sdn.p(0), de, std::ptr::null());
        let sd = Dense { act: if sh_at.is_some() { pf.sh_act.p(0) as *const u8 } else { std::ptr::null() }, ..sd };
        {
            let ctr = step_ctr(pf, ts);
        let next = || ctr.fetch_add(1, Relaxed);
            let shared_units=if shared_overlap{1}else{sd.units()};
            let total = nu + shared_units;
            loop {
                let u = next();
                if u >= total {
                    break;
                }
                if shared_overlap&&u==0 {
                    #[cfg(feature="npu-prefill")]
                    self.pf_shared_overlap(pf,n,&ops[1..],&sd,ffs,sh_at);
                }else if u < nu+usize::from(shared_overlap) {
                    #[cfg(feature="npu-prefill")]
                    if shared_overlap{self.npu_prefill.as_ref().unwrap().note_expert_unit();}
                    // SAFETY: units read-only after the barrier.
                    let ei=u-usize::from(shared_overlap);
                    let (e, c0, c1) = unsafe { *pf.units.p(ei) };
                    let rb = rat.bytes(de);
                    let t0 = if prof { ticks() } else { 0 };
                    self.mmid_unit(pf, ts, &[(&wge, pf.g.p(0)), (&wue, pf.u.p(0))], e as usize, c0 as usize, c1 as usize, pf.actr.p(0), rb, ffe, ku, false);
                    if prof {
                        let t1 = ticks();
                        pf_prof(pf, tid).sub[SUB_GATE_UP] += t1 - t0;
                    }
                } else {
                    let c0 = (u - nu) * sd.cu;
                    let t0 = if prof { ticks() } else { 0 };
                    self.dense_unit(&sd, n, c0, (c0 + sd.cu).min(sd.w.rows));
                    if prof {
                        let t1 = ticks();
                        pf_prof(pf, tid).sub[SUB_SHARED_DOWN] += t1 - t0;
                    }
                }
            }
        }
        self.bar(ctx, pf, PH_MOE_GATE_UP);

        // [5] swiglu + from_float of every routed row (down activations)
        let dat = wde.qtype().act();
        let hrb = dat.bytes(ffe);
        for s in share(pf, ts, n * ku, 16, tid) {
            // SAFETY: routed row s is owned by this thread.
            unsafe {
                let h = &mut ts.kq[..ffe];
                ops::swiglu(pf.g.s(s * ffe..(s + 1) * ffe), pf.u.s(s * ffe..(s + 1) * ffe), h);
                    #[cfg(feature="f32-reference")]
                    if self.f32_reference.is_some(){pf.href.s(s*ffe..(s+1)*ffe).copy_from_slice(h);}
                quantize_row_raw(dat, h.as_ptr(), pf.hq.p(s * hrb), ffe);
            }
        }
        self.bar(ctx, pf, PH_MOE_SWIGLU_QUANT);

        // [6] routed down (MUL_MAT_ID)
        let nd = pf.n_units[1].load(Relaxed);
        {
            let ctr = step_ctr(pf, ts);
        let next = || ctr.fetch_add(1, Relaxed);
            loop {
                let u = next();
                if u >= nd {
                    break;
                }
                // SAFETY: units read-only after the barrier.
                let (e, c0, c1) = unsafe { *pf.units.p(nu + u) };
                self.mmid_unit(pf, ts, &[(&wde, pf.dn.p(0))], e as usize, c0 as usize, c1 as usize, pf.hq.p(0), hrb, de + DN_PAD, 1, true);
            }
        }
        self.bar(ctx, pf, PH_MOE_DOWN);

        // [7] weighted expert sum (ggml_add chain in slot order) + gated shared expert + residual
        for t in share(pf, ts, n, 4, tid) {
            // SAFETY: row t of x (the next layer input) is owned by this thread; the rest read-only.
            unsafe {
                let w = pf.w.s(t * ku..(t + 1) * ku);
                let sg = *pf.sg.p(t);
                let x1 = pf.x1.s((rows.r0 + t) * de..(rows.r0 + t + 1) * de);
                let sdn = pf.sdn.s(t * de..(t + 1) * de);
                let ds = de + DN_PAD;
                let dn = pf.dn.s(t * ku * ds..(t + 1) * ku * ds);
                let x = pf.x.s((rows.r0 + t) * de..(rows.r0 + t + 1) * de);
                for c in 0..de {
                    let mut m = dn[c] * w[0];
                    for j in 1..ku {
                        let p = dn[j * ds + c] * w[j];
                        m += p;
                    }
                    let s = sdn[c] * sg;
                    let f = m + s;
                    x[c] = f + x1[c];
                }
                if trace {
                    let mut moe = vec![0f32; de];
                    for c in 0..de {
                        let mut m = dn[c] * w[0];
                        for j in 1..ku {
                            m += dn[j * ds + c] * w[j];
                        }
                        moe[c] = m;
                    }
                    self.ptr(l, "ffn_moe_out", n * de, t * de, &moe);
                    self.ptr(l, "ffn_shexp", n * de, t * de, sdn);
                    self.ptr(l, "l_out", n * de, t * de, x);
                }
            }
        }
        self.bar(ctx, pf, PH_MOE_SUM_RESIDUAL);
    }

    /// Independent shared FFN branch, owned by one existing worker while the
    /// rest compute routed experts. All output consumers follow PH_MOE_GATE_UP.
    #[cfg(feature="npu-prefill")]
    fn pf_shared_overlap(&self,pf:&Pf,n:usize,gate_up:&[Dense],down:&Dense,ffs:usize,at:Option<ActType>){
        let _activity=self.npu_prefill.as_ref().unwrap().begin_shared_overlap();
        for op in gate_up{self.dense_unit(op,n,0,op.w.rows);}
        // SAFETY: this indivisible unit exclusively owns shared scratch. xn and
        // original dense activation buffers remain immutable through this layer.
        unsafe{
            for t in 0..n{ops::swiglu(pf.shg.s(t*ffs..(t+1)*ffs),pf.shu.s(t*ffs..(t+1)*ffs),pf.shh.s(t*ffs..(t+1)*ffs));}
            // Keep the exact native activation layout for a latched CPU fallback.
            // Experts use actr/hq, never this private shared-down buffer.
            if let Some(at)=at{
                let rb=at.bytes(ffs);let full=n-n%4;
                for a in (0..full).step_by(4){quantize_mat4_raw(at,pf.shh.p(a*ffs),pf.sh_act.p(a*rb),ffs);}
                for a in full..n{quantize_row_raw(at,pf.shh.p(a*ffs),pf.sh_act.p(a*rb),ffs);}
            }
        }
        self.dense_unit(down,n,0,down.w.rows);
    }

    /// One MUL_MAT_ID unit: expert `e`, weight rows c0..c1, for every routed row of the expert.
    /// `mats` are (weight, output base); output row of routed row s is at `s * out_w` (`per_row`
    /// false: s = token*ku + slot indexes the routed act rows by token; true: by s).
    #[allow(clippy::too_many_arguments)]
    fn mmid_unit(&self, pf: &Pf, ts: &mut PfThread, mats: &[(&WeightRef, *mut f32)], e: usize, c0: usize, c1: usize, act: *mut u8, rb: usize, out_w: usize, ku: usize, per_row: bool) {
        // SAFETY: the expert's row list is read-only; the unit owns output columns c0..c1 of
        // its routed rows; the tile/tmp scratch is this thread's.
        unsafe {
            let off = pf.off.s(0..pf.off.len);
            let list = pf.order.s(off[e] as usize..off[e + 1] as usize);
            #[cfg(feature="f32-reference")]
            if let Some(reference)=&self.f32_reference {
                for &(w,out) in mats{for c in c0..c1{
                    reference.with_row(w.ptr,e,c,|row|{
                        for &s in list.iter(){
                            let input=if per_row {pf.href.p(s as usize*w.k)}else{pf.xn.p((s as usize/ku)*w.k)};
                            *out.add(s as usize*out_w+c)=ffi::oxe_vec_dot_f32(w.k as i32,row.as_ptr(),input);
                        }
                    });
                }}return;
            }
            let cnt = list.len();
            let row_of = |s: u32| -> usize { if per_row { s as usize } else { s as usize / ku } };
            let k = mats[0].0.k;
            let q = mats[0].0.qtype();
            let at = q.act();
            let mut i = 0;
            let mask = pf.mask.load(Relaxed);
            let tile_ok = |t: kernels_sys::QType| kernels_sys::mmid_tile_has(t) && (t != kernels_sys::QType::Q6K || mask & PF_Q6K_TILE != 0);
            if mask & PF_MMID_TILE != 0 && mats.iter().all(|m| tile_ok(m.0.qtype())) && mmid_use_tile(q, cnt, mask) {
                // moe-prefill: all rows of the expert through the pair tiles (same bits per row)
                let need = cnt.div_ceil(2) * (k / kernels_sys::QK_K) * kernels_sys::PF_XBLK_BYTES / 4;
                if ts.xpack.len() < need {
                    ts.xpack.resize(need, 0);
                }
                ts.xrows.clear();
                ts.xrows.extend(list.iter().map(|&s| act.add(row_of(s) * rb) as usize));
                // SAFETY: usize and *const u8 have the same layout.
                let rows = std::slice::from_raw_parts(ts.xrows.as_ptr() as *const *const u8, cnt);
                kernels_sys::mmid_tile_pack_raw(rows, k, ts.xpack.as_mut_ptr() as *mut u8);
                for &(w, out) in mats {
                    ts.xout.clear();
                    ts.xout.extend(list.iter().map(|&s| out.add(s as usize * out_w + c0) as usize));
                    let orow = std::slice::from_raw_parts(ts.xout.as_ptr() as *const *mut f32, cnt);
                    let ok = kernels_sys::mmid_tile_raw(w.qtype(), k, orow, w.rows_ptr(e, c0), ts.xpack.as_ptr() as *const u8, c1 - c0);
                    debug_assert!(ok);
                }
                return;
            }
            while i + 4 <= cnt {
                let rp: [*const u8; 4] = std::array::from_fn(|j| act.add(row_of(list[i + j]) * rb) as *const u8);
                interleave_rows4_raw(at, &rp, ts.tile.as_mut_ptr(), k);
                for &(w, out) in mats {
                    let mut cc = c0;
                    while cc < c1 {
                        let nc = MMID_COLS.min(c1 - cc);
                        gemm_mmid_raw(w.qtype(), k, ts.tmp.as_mut_ptr(), nc, w.rows_ptr(e, cc), ts.tile.as_ptr(), 4, nc);
                        for j in 0..4 {
                            let dst = out.add(list[i + j] as usize * out_w + cc);
                            std::ptr::copy_nonoverlapping(ts.tmp.as_ptr().add(j * nc), dst, nc);
                        }
                        cc += nc;
                    }
                }
                i += 4;
            }
            // Remainder rows (cnt % 4). llama.cpp runs them through the GEMV; for every expert
            // type of both models gemm_mmid's row j is bitwise the GEMV of row j (unit test
            // gemm_kernels_vs_generic_and_gemv), so padding the tile with copies of the last
            // row and keeping only the real rows is the same result at 1/4 of the weight
            // decoding per row.
            let rem = cnt - i;
            if rem > 0 && mask & PF_MMID_GEMV1 != 0 && (rem == 1 || (rem == 2 && q == kernels_sys::QType::Q4K)) {
                // llama.cpp's own path for the cnt % 4 rows: one GEMV per row
                for j in 0..rem {
                    let a = act.add(row_of(list[i + j]) * rb) as *const u8;
                    for &(w, out) in mats {
                        gemv_raw(w.qtype(), k, out.add(list[i + j] as usize * out_w + c0), w.rows_ptr(e, c0), a, c1 - c0);
                    }
                }
            } else if rem > 0 {
                let rp: [*const u8; 4] = std::array::from_fn(|j| act.add(row_of(list[i + j.min(rem - 1)]) * rb) as *const u8);
                interleave_rows4_raw(at, &rp, ts.tile.as_mut_ptr(), k);
                for &(w, out) in mats {
                    let mut cc = c0;
                    while cc < c1 {
                        let nc = MMID_COLS.min(c1 - cc);
                        gemm_mmid_raw(w.qtype(), k, ts.tmp.as_mut_ptr(), nc, w.rows_ptr(e, cc), ts.tile.as_ptr(), 4, nc);
                        for j in 0..rem {
                            let dst = out.add(list[i + j] as usize * out_w + cc);
                            std::ptr::copy_nonoverlapping(ts.tmp.as_ptr().add(j * nc), dst, nc);
                        }
                        cc += nc;
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------ head

    /// Output norm and head of x row `r` (one row: the decode computation).
    fn pf_head(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, r: usize) {
        let d = &self.plan.desc;
        let de = d.n_embd;
        let tid = ctx.tid;
        let w = *self.ws.get(self.plan.output);
        let at = w.qtype().act();
        if tid == 0 {
            // SAFETY: thread 0 writes xn row 0 and its activation; read after the barrier.
            unsafe {
                ops::rms_norm_mul(pf.x.s(r * de..(r + 1) * de), self.ws.get(self.plan.out_norm).f32_slice(), pf.xn.s(0..de), d.rms_eps);
                quantize_row_raw(at, pf.xn.p(0), pf.act_k.p(0), de);
                if self.trace.is_some() {
                    if let Some(t) = &self.trace {
                        t.put("result_norm", None, de, 0, pf.xn.s(0..de));
                    }
                }
            }
        }
        self.bar(ctx, pf, PH_HEAD_NORM);
        let g = w.qtype().nr();
        let rows = split(d.n_vocab / g, tid, pf.nth);
        let rows = rows.start * g..rows.end * g;
        let mut best = (f32::NEG_INFINITY, u32::MAX);
        let chunk = self.plan.head_chunk;
        let mut r0 = rows.start;
        while r0 < rows.end {
            let nn = chunk.min(rows.end - r0);
            let out = &mut ts.kq[..nn];
            // SAFETY: rows r0..r0+nn of the head; activation complete after the barrier.
            #[cfg(feature="f32-reference")]
            let is_reference=if let Some(reference)=&self.f32_reference{
                for i in 0..nn{reference.with_row(w.ptr,0,r0+i,|row|{
                    // SAFETY: normalized f32 row complete; this worker owns head rows.
                    out[i]=unsafe{ffi::oxe_vec_dot_f32(de as i32,row.as_ptr(),pf.xn.p(0))};
                });}true
            }else{false};
            #[cfg(not(feature="f32-reference"))]
            let is_reference=false;
            if !is_reference{unsafe { gemv_raw(w.qtype(), de, out.as_mut_ptr(), w.rows_ptr(0, r0), pf.act_k.p(0), nn) }};
            for i in 0..nn {
                if best.1 == u32::MAX || out[i] > best.0 {
                    best = (out[i], (r0 + i) as u32);
                }
            }
            if self.cfg.logits {
                // SAFETY: this thread owns logits rows r0..r0+nn.
                unsafe { self.logits.at(0, r0..r0 + nn).copy_from_slice(out) };
            }
            if let Some(t) = &self.trace {
                t.put("result_output", None, d.n_vocab, r0, out);
            }
            r0 += nn;
        }
        // SAFETY: argmax slot tid is ours.
        unsafe { pf.argmax.s(tid * 16..tid * 16 + 1)[0] = best };
    }
}

// ------------------------------------------------------------ O-117 quality runner hooks
// (bench / tests only: `cfg(test)` or the `f32-reference` feature; never in a production build)

/// O-117: the positions whose batched-prefill attention outputs are compared with the
/// f32-attention reference on the same inputs, and the per-position (sum of squared errors, sum
/// of squared reference values) over every attention layer and query head.
#[cfg(any(test, feature = "f32-reference"))]
pub(super) struct PfCheck {
    positions: Vec<usize>,
    acc: Mutex<std::collections::BTreeMap<usize, (f64, f64)>>,
}

/// O-117: batched-prefill evidence (only counted in bench / test builds).
#[cfg(any(test, feature = "f32-reference"))]
#[derive(Default)]
pub(super) struct PfCounters {
    passes: AtomicU64,
    rows: AtomicU64,
    max_rows: AtomicU64,
    /// attention units: tiled kernel, per-(head, block) kernel, fa2, fa2x, f32-attention reference
    units: [AtomicU64; 5],
}

/// O-117: snapshot of the batched-prefill counters since [`Engine::reset_pf_counts`].
#[cfg(any(test, feature = "f32-reference"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PfCounts {
    /// batched passes (one pool job each: a ubatch, or a group of ubatches)
    pub passes: u64,
    /// token rows through batched passes
    pub rows: u64,
    /// most rows of one pass (> 1: genuinely batched)
    pub max_rows: u64,
    /// attention units per kernel: (KV head, token block) of the tiled kernel
    pub units_tiled: u64,
    /// (query head, token block) of the per-head prefill kernel (no tiled attention / trace)
    pub units_per_head: u64,
    pub units_fa2: u64,
    pub units_fa2x: u64,
    /// (token, KV head) of the f32-attention reference ([`Engine::set_pf_ref`])
    pub units_ref: u64,
}

#[cfg(any(test, feature = "f32-reference"))]
const PFU_TILED: usize = 0;
#[cfg(any(test, feature = "f32-reference"))]
const PFU_PER_HEAD: usize = 1;
#[cfg(any(test, feature = "f32-reference"))]
const PFU_FA2: usize = 2;
#[cfg(any(test, feature = "f32-reference"))]
const PFU_FA2X: usize = 3;
#[cfg(any(test, feature = "f32-reference"))]
const PFU_REF: usize = 4;

/// O-117: the vocabulary logits of one output-normed hidden row (a [`Engine::prefill_ext`] row)
/// through the production head: activation quantisation of the row and the head GEMV, the same
/// kernels and inputs as the batched prefill's own head (`pf_head`), split over `pool`'s threads
/// (every output row is one independent dot product: the bits do not depend on the split).
#[cfg(any(test, feature = "f32-reference"))]
pub fn head_logits<P: Team + ?Sized>(plan: &crate::plan::Plan, ws: &crate::weights::WeightStore, pool: &P, xn: &[f32], out: &mut [f32]) {
    let (de, nv) = (plan.desc.n_embd, plan.desc.n_vocab);
    assert!(xn.len() == de && out.len() >= nv, "head_logits: one n_embd row into n_vocab logits");
    // a plain-data copy of the head's reference (pointer as usize): shared by the workers
    let w: crate::weights::WeightRef = *ws.get(plan.output);
    let qt = w.qtype();
    let at = qt.act();
    // (u64 words: the quantised blocks start with an f32 / f16 scale, keep them aligned)
    let mut act = vec![0u64; at.bytes(de).div_ceil(8)];
    // SAFETY: de floats in, at.bytes(de) bytes out.
    unsafe { quantize_row_raw(at, xn.as_ptr(), act.as_mut_ptr() as *mut u8, de) };
    let g = qt.nr();
    let nth = pool.nthreads();
    let (outp, actp) = (out.as_mut_ptr() as usize, act.as_ptr() as usize);
    let chunk = plan.head_chunk;
    pool.run_job(&|ctx: &mut Ctx| {
        let rows = split(nv / g, ctx.tid, nth);
        let (mut r0, end) = (rows.start * g, rows.end * g);
        while r0 < end {
            let nn = chunk.min(end - r0);
            // SAFETY: this thread owns out[r0..r0 + nn]; the activation row is read-only.
            unsafe { gemv_raw(qt, de, (outp as *mut f32).add(r0), w.rows_ptr(0, r0), actp as *const u8, nn) };
            r0 += nn;
        }
    });
}

#[cfg(any(test, feature = "f32-reference"))]
impl<'a> Engine<'a> {
    /// O-117: batched prefill attention through the f32-attention reference (f64 accumulation
    /// over the roped f32 query and the stored f16 K / V, [`crate::fused_ref::ref_attn_f64`]'s
    /// arithmetic) for every row; f16 KV only. Takes precedence over tiled / fa2 / fa2x.
    pub fn set_pf_ref(&mut self, on: bool) {
        assert!(!on || !self.kv.any8(), "the prefill f32-attention reference reads the f16 KV cache");
        self.pf_ref = on;
    }

    /// O-117: compare the gated batched-prefill attention outputs of these absolute positions with
    /// the f32-attention reference on the same inputs (None = off). Resets the accumulators.
    pub fn set_pf_check(&mut self, positions: Option<Vec<usize>>) {
        assert!(positions.is_none() || !self.kv.any8(), "the prefill attention check reads the f16 KV cache");
        self.pf_check = positions.map(|mut p| {
            p.sort_unstable();
            p.dedup();
            PfCheck { positions: p, acc: Mutex::new(Default::default()) }
        });
    }

    /// O-117: (position, sum of squared errors, sum of squared reference values) of every checked
    /// position a batched prefill reached since the last call; resets.
    pub fn take_pf_check(&self) -> Vec<(usize, f64, f64)> {
        self.pf_check.as_ref().map_or_else(Vec::new, |c| std::mem::take(&mut *c.acc.lock().unwrap()).into_iter().map(|(p, (e, r))| (p, e, r)).collect())
    }

    /// O-117: the engine's weights (the quality runner's [`head_logits`] of hidden rows).
    pub fn weight_store(&self) -> &'a crate::weights::WeightStore {
        self.ws
    }

    /// O-117: the batched-prefill counters since the last reset.
    pub fn pf_counts(&self) -> PfCounts {
        let c = &self.pf_ctr;
        let u = |i: usize| c.units[i].load(Relaxed);
        PfCounts {
            passes: c.passes.load(Relaxed),
            rows: c.rows.load(Relaxed),
            max_rows: c.max_rows.load(Relaxed),
            units_tiled: u(PFU_TILED),
            units_per_head: u(PFU_PER_HEAD),
            units_fa2: u(PFU_FA2),
            units_fa2x: u(PFU_FA2X),
            units_ref: u(PFU_REF),
        }
    }

    pub fn reset_pf_counts(&self) {
        let c = &self.pf_ctr;
        for a in [&c.passes, &c.rows, &c.max_rows].into_iter().chain(c.units.iter()) {
            a.store(0, Relaxed);
        }
    }

    /// Tests: degrade the fa2 / fa2x prefill units by scaling their outputs by (1 + eps).
    #[cfg(test)]
    pub(crate) fn set_pf_degrade(&mut self, eps: f32) {
        self.pf_degrade = eps;
    }

    #[inline]
    fn pf_count_unit(&self, kind: usize) {
        self.pf_ctr.units[kind].fetch_add(1, Relaxed);
    }

    fn pf_count_pass(&self, n: usize) {
        let c = &self.pf_ctr;
        c.passes.fetch_add(1, Relaxed);
        c.rows.fetch_add(n as u64, Relaxed);
        c.max_rows.fetch_max(n as u64, Relaxed);
    }

    /// O-117: the f32-attention reference of one query head `q` (roped f32) at attention layer
    /// `ai`, KV head `kvh`, over the stored f16 K / V^T rows of positions < `npos`: f64 scores,
    /// softmax and V accumulation in position order, output rounded to f32. Bit for bit
    /// [`crate::fused_ref::ref_attn_f64`] over the widened rows ([`Engine::kv_head_f32`]), read in
    /// place (no O(n_ctx) f32 copy per unit). `s`: scratch of at least `npos`.
    pub(crate) fn pf_ref_head(&self, ai: usize, kvh: usize, q: &[f32], npos: usize, s: &mut [f64], out: &mut [f32]) {
        let d = &self.plan.desc;
        let hd = d.head_dim;
        let scale = 1.0 / (hd as f32).sqrt();
        let kb = ai * d.n_head_kv * self.n_ctx * hd + kvh * self.n_ctx * hd;
        let mut m = f64::NEG_INFINITY;
        // SAFETY: K rows / V^T entries of positions < npos are complete (written before the
        // attention barrier of this pass, or by an earlier pass) and read-only here.
        unsafe {
            for p in 0..npos {
                let kr = self.kcache.at(kb + p * hd..kb + (p + 1) * hd);
                let mut a = 0f64;
                for j in 0..hd {
                    a += q[j] as f64 * kv8::h2f(kr[j]) as f64;
                }
                s[p] = a * scale as f64;
                m = m.max(s[p]);
            }
            let mut sum = 0f64;
            for x in s[..npos].iter_mut() {
                *x = (*x - m).exp();
                sum += *x;
            }
            for j in 0..hd {
                let at = self.vrow(ai, kvh * hd + j);
                let vr = self.vcache.at(at..at + npos);
                let mut acc = 0f64;
                for p in 0..npos {
                    acc += s[p] * kv8::h2f(vr[p]) as f64;
                }
                out[j] = (acc / sum) as f32;
            }
        }
    }

    /// O-117 ([`Engine::set_pf_ref`]): every (token, KV head) of the pass through the
    /// f32-attention reference, gated as the kernels gate (dynamic units, no barrier).
    fn pf_attention_ref(&self, pf: &Pf, ts: &mut PfThread, l: usize, n: usize, p0: usize) {
        let d = &self.plan.desc;
        let (hd, nh, nkv) = (d.head_dim, d.n_head, d.n_head_kv);
        let (group, att_w, ai) = (nh / nkv, nh * hd, self.att_index[l]);
        let ctr = step_ctr(pf, ts);
        let mut s = vec![0f64; p0 + n];
        let mut out = vec![0f32; hd];
        let mut gate = vec![0f32; hd];
        loop {
            let u = ctr.fetch_add(1, Relaxed);
            if u >= n * nkv {
                break;
            }
            let (t, kvh) = (u / nkv, u % nkv);
            self.pf_count_unit(PFU_REF);
            for h in kvh * group..(kvh + 1) * group {
                // SAFETY: q / gate rows complete after step [3]; O elements (t, h) owned by this unit.
                unsafe {
                    let q = pf.qf.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                    self.pf_ref_head(ai, kvh, q, p0 + t + 1, &mut s, &mut out);
                    ops::sigmoid(pf.qkv.s(t * 2 * att_w + h * 2 * hd + hd..t * 2 * att_w + (h + 1) * 2 * hd), &mut gate);
                    let o = pf.o.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                    for j in 0..hd {
                        o[j] = out[j] * gate[j];
                    }
                }
            }
        }
    }

    /// O-117 ([`Engine::set_pf_check`]): after the attention core's barrier, the gated outputs of
    /// the checked positions of this pass vs the f32-attention reference over the same q and
    /// stored KV; ends with a barrier (only when a check is set: the same on every thread).
    fn pf_attn_check(&self, ctx: &mut Ctx, pf: &Pf, ts: &mut PfThread, l: usize, n: usize, p0: usize) {
        let Some(chk) = &self.pf_check else { return };
        let d = &self.plan.desc;
        let (hd, nh, nkv) = (d.head_dim, d.n_head, d.n_head_kv);
        let (group, att_w, ai) = (nh / nkv, nh * hd, self.att_index[l]);
        let lo = chk.positions.partition_point(|&p| p < p0);
        let hi = chk.positions.partition_point(|&p| p < p0 + n);
        let mine = &chk.positions[lo..hi];
        let ctr = step_ctr(pf, ts);
        let mut s = vec![0f64; p0 + n];
        let mut out = vec![0f32; hd];
        let mut gate = vec![0f32; hd];
        loop {
            let u = ctr.fetch_add(1, Relaxed);
            if u >= mine.len() * nkv {
                break;
            }
            let (pos, kvh) = (mine[u / nkv], u % nkv);
            let t = pos - p0;
            let (mut e2, mut r2) = (0f64, 0f64);
            for h in kvh * group..(kvh + 1) * group {
                // SAFETY: q, gate and O rows complete after the core barrier; read only.
                unsafe {
                    let q = pf.qf.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                    self.pf_ref_head(ai, kvh, q, pos + 1, &mut s, &mut out);
                    ops::sigmoid(pf.qkv.s(t * 2 * att_w + h * 2 * hd + hd..t * 2 * att_w + (h + 1) * 2 * hd), &mut gate);
                    let o = pf.o.s(t * att_w + h * hd..t * att_w + (h + 1) * hd);
                    for j in 0..hd {
                        let r = (out[j] * gate[j]) as f64;
                        e2 += (o[j] as f64 - r).powi(2);
                        r2 += r * r;
                    }
                }
            }
            let mut g = chk.acc.lock().unwrap();
            let a = g.entry(pos).or_insert((0.0, 0.0));
            a.0 += e2;
            a.1 += r2;
        }
        self.bar(ctx, pf, PH_ATT_CORE);
    }
}
