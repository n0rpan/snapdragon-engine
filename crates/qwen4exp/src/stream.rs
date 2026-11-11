//! The engine's streaming expert op for the graph port (`csrc/flash_graph.cpp`, expert callback).
//!
//! The routed experts of a decode (or MTP verify) ubatch come from M0's expert cache (per-layer
//! CLOCK pools over one anonymous arena) filled by M0's io_uring IO thread from the R4 sidecar,
//! instead of the shard mappings. The graph calls, per layer:
//!   FETCH (one thread): cache lookups of the selected experts (misses queued to the IO thread,
//!          demand class), and the activation x quantised once per token (production from_float);
//!   UP / GATE (all threads): rows split across threads in 16-row units; each unit waits for its
//!          expert's slot to be READY, then R4 gemv (IQ2_S / IQ3_XXS) or production vec_dot;
//!   (production ggml_swiglu_split in the graph);
//!   DOWN (all threads): act rows quantised per thread (production from_float), production vec_dot.
//! Exactness: production's mul_mat_id computes every output row as one vec_dot of the weight row
//! with the quantised activation row, so any row split / order gives the same bits; the R4 gemv is
//! bitwise equal to vec_dot (vendor/ggml-flash test) and checked in-process (`flash-m1 stream`).
//!
//! Any n_tokens works (MTP verify batches of 2-4) as long as the distinct experts of one layer fit
//! that layer's pool: all lookups of one call carry the same stamp, so none of them evicts another.

use crate::cache::{Cache, Entry, Stats};
use crate::ffi;
use crate::io::{self, IoConfig, IoQueue, IoTuning, Req, ReqClass};
use crate::model::{Shards, FF_EXP, N_EMBD, N_LAYER, TOP_K};
use crate::sidecar::{self, Layout};
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::*};
use std::sync::Arc;
use std::time::Instant;

pub const FG_EXP_FETCH: c_int = 0;
pub const FG_EXP_GATE: c_int = 1;
pub const FG_EXP_UP: c_int = 2;
pub const FG_EXP_DOWN: c_int = 3;
pub const FG_EXP_BATCH: c_int = 4;
/// graph-build query (flash_graph.h): write the per-token op's token cap into dst[0]
pub const FG_EXP_CAP: c_int = 5;

#[repr(C)]
pub struct FgExpertCall {
    pub kind: c_int,
    pub il: c_int,
    pub ith: c_int,
    pub nth: c_int,
    pub n_tokens: c_int,
    pub ids: *const i32,
    pub ids_nb1: usize,
    pub src: *const f32,
    pub src_nb1: usize,
    pub src_nb2: usize,
    pub dst: *mut f32,
    pub dst_nb1: usize,
    pub dst_nb2: usize,
}

extern "C" {
    fn fg_vec_dot_type(t: i32) -> i32;
    fn fg_row_size(t: i32, n: i64) -> usize;
    fn fg_from_float(vdt: i32, x: *const f32, y: *mut c_void, n: i64);
    fn fg_vec_dot(t: i32, n: c_int, s: *mut f32, x: *const c_void, y: *const c_void);
    fn fg_vec_swiglu(n: c_int, y: *mut f32, g: *const f32, u: *const f32);
}

/// Token cap of an expert-op kind: the 4-node per-token op (FETCH/UP/GATE/DOWN) is sized for MTP-verify
/// widths (MAX_TOKENS); the batched op (FG_EXP_BATCH, prefill ubatches) has none.
fn max_tokens(kind: c_int) -> Option<usize> {
    if kind == FG_EXP_BATCH {
        None
    } else {
        Some(MAX_TOKENS)
    }
}

#[derive(Clone, Copy)]
struct LayerK {
    gate_t: i32,
    gate_vdt: i32,
    gate_row: usize,
    gate_bytes: usize,
    r4: Option<c_int>,
    down_t: i32,
    down_vdt: i32,
    down_row: usize,
    xq_row: usize,
    aq_row: usize,
}

/// Default expert-cache quota weights (slot counts per layer ~ these): distinct experts per layer in the fixed353
/// prefill (P2 trace; layer 47 = mean of 0-46), = tools/flash-quota-weights.txt. With LFU the Flash default since
/// ticket 035 (20 GiB: -5.5% mean ms/token, lossless) [M].
pub const FLASH_QUOTA_WEIGHTS: [f64; 48] = [342.0, 270.0, 249.0, 138.0, 211.0, 245.0, 266.0, 263.0, 186.0, 172.0, 232.0, 193.0, 246.0, 242.0, 195.0, 121.0, 135.0, 198.0, 242.0, 200.0, 211.0, 236.0, 285.0, 183.0, 196.0, 176.0, 222.0, 196.0, 242.0, 233.0, 203.0, 160.0, 169.0, 227.0, 235.0, 202.0, 228.0, 227.0, 277.0, 192.0, 191.0, 242.0, 249.0, 202.0, 243.0, 226.0, 224.0, 218.0];

const MAX_THREADS: usize = 64;
/// a compute thread gives up waiting for one expert read after this long (an SSD read takes ~2.5 ms)
const WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
const MAX_TOKENS: usize = 16;
const UP_ROWS: usize = 16;
const DOWN_ROWS: usize = 32;

struct Shared {
    entries: Vec<Entry>,
    xq: Vec<u8>,
    stamp: u64,
    // batched (FG_EXP_BATCH) plan, built by thread 0
    b_experts: Vec<u32>,           // distinct experts of the ubatch, first-seen order
    b_pairs: Vec<Vec<u32>>,        // expert -> (token * TOP_K + slot) pairs using it
    b_wave_entries: [Vec<Entry>; 2], // entries of wave w at [w % 2] (aligned with the wave's experts)
    b_xq: Vec<u8>,                 // quantised x per token
    b_gu: Vec<f32>,                // gate | up per pair [pair][2][FF]
    b_aq: Vec<u8>,                 // quantised act per pair
    b_last: Vec<[u64; 8]>,         // PREFILL IO: per layer, the experts the last batched op used (bitset); None before
    b_have_last: Vec<bool>,
    b_ra: usize,                   // W4-I: read-ahead in effect for the current pass (decided at layer 0 when auto)
    b_wave_max: usize,             // experts per wave of the current op (thread 0 publishes it before the first barrier)
}

/// Spin barrier for the graph threads inside one custom op (all `nth` threads call the op).
/// One value on its own 128-byte line (aliasing sweep: no false sharing between per-thread / barrier fields).
#[derive(Default)]
#[repr(align(128))]
struct Line<T>(T);
impl<T> std::ops::Deref for Line<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Default)]
struct SpinBarrier {
    count: Line<AtomicU64>,
    gen: Line<AtomicU64>,
}
impl SpinBarrier {
    fn wait(&self, nth: usize) {
        let g = self.gen.load(Acquire);
        if self.count.fetch_add(1, AcqRel) as usize + 1 == nth {
            self.count.store(0, Relaxed);
            self.gen.fetch_add(1, Release);
        } else {
            while self.gen.load(Acquire) == g {
                std::hint::spin_loop();
            }
        }
    }
}

#[derive(Default)]
pub struct StreamCounters {
    /// time compute threads spent spinning on PENDING slots (sum over threads), ns
    pub wait_ns: AtomicU64,
    /// thread 0's wait only (wall-clock proxy), ns
    pub wait0_ns: AtomicU64,
    pub fetch_ns: AtomicU64,
    pub calls: AtomicU64,
    /// PREFILL IO: experts issued by the layer read-ahead
    pub readahead: AtomicU64,
    /// W4-I: passes that started warm (read-ahead used) / cold (read-ahead off) under `pf_readahead_auto`
    pub ra_warm_passes: AtomicU64,
    pub ra_cold_passes: AtomicU64,
    /// FP-9: thread 0's wall time per stage of the batched expert op, ns: [plan + x quantise, gate/up, swiglu + act
    /// quantise, down]; each includes that stage's waits (SSD wait is also counted in wait0_ns)
    pub batch_stage_ns: [AtomicU64; 4],
}

pub struct Stream {
    pub cache: Cache,
    pub io: Arc<IoQueue>,
    /// the IO configuration in effect (`IoTuning::describe`), for receipts
    pub io_desc: String,
    io_thread: Option<std::thread::JoinHandle<()>>,
    lk: Vec<LayerK>,
    sh: UnsafeCell<Shared>,
    scratch: Vec<Line<UnsafeCell<Vec<u8>>>>,
    pending: Vec<Line<UnsafeCell<Vec<u32>>>>,
    barrier: SpinBarrier,
    pub counters: StreamCounters,
    /// do the units whose expert is resident first, the misses as they land (row-wise ops: same bits)
    pub resident_first: bool,
    /// batched prefill op: i8mm gemm4 over groups of 4 tokens on the R4 layers (bitwise = gemv)
    pub batch_gemm: bool,
    /// PREFILL: gate/up of the R4 layers (IQ2_S, IQ3_XXS) in 8-token SMMLA tiles, each 32-row chunk decoded once per expert
    /// (csrc/pf_expert.c); bitwise equal to the gemv / gemm4 path. Default off.
    pub batch_tile: bool,
    /// PREFILL IO (c)+(b): the batched op takes the experts already in the layer's pool first (first-seen order), then
    /// the missing ones in file order. Every resident expert is then used before a miss can evict it (today the
    /// first-seen order lets misses evict residents the same op needs later: a warm 2048-token pass re-read 36.6 of
    /// 36.6 GiB, ticket 011). Order only: values unchanged. Default off.
    pub pf_order: bool,
    /// F123 (`--flash-pf-order 2`, with `pf_order` on): both groups (residents, then misses) by the op's LAST token that
    /// uses the expert, ascending (ties: first-seen). Same reads as `pf_order` (every resident still comes before every
    /// miss), but the op's last waves, which are what the pool holds when the op ends, are the experts of the prompt's
    /// final tokens instead of the highest-id misses: the decode right after a short prefill keeps its hot experts
    /// (062: residents-first + id sort cost +4.4% decode after a 75-token prefill). Order only: values unchanged.
    /// Default off. Replay: order_sim_f123.py in the private lab notes.
    pub pf_order_last: bool,
    /// PREFILL IO (a): after layer L's batched op, read ahead up to this many experts of layer L+1 at prefetch
    /// priority while L finishes and L+1's dense / attention part runs: into empty slots, or over residents the last
    /// op of that layer did not use (targets = the experts it did use). 0 = off. Default 64, under `pf_readahead_auto`.
    pub pf_readahead: usize,
    /// W4-I (G4: read-ahead +12.9% on a repeated pass, -2.4% and +1.12 GiB of reads on a cold one): with auto on, the
    /// read-ahead is used only for a pass that STARTS WARM: at the pass's first batched op (layer 0) the share of the
    /// op's experts already in the pool (of min(needed, pool slots)) is at least `pf_ra_warm`; otherwise 0 for the
    /// whole pass. With auto off, `pf_readahead` applies to every pass. Default on.
    pub pf_readahead_auto: bool,
    pub pf_ra_warm: f64,
    /// route trace (`enable_trace`): (phase 0 batch / 1 per-token op, layer, token in call, the TOP_K ids), thread 0 only
    trace: UnsafeCell<Option<Vec<(u8, u8, u16, u16, [u16; TOP_K])>>>,
    error: std::sync::Mutex<Option<String>>,
    /// QC-A (FLASH-LOOKAHEAD-DESIGN): `--stage-junk N` = N never-demanded Prefetch reads per token (pure waste)
    junk: UnsafeCell<Option<Junk>>,
    /// expert lookahead (`--lookahead FILE`, lookahead.rs): speculate layer l+1 at layer l's FETCH; None = off
    lookahead: UnsafeCell<Option<Box<crate::lookahead::Lookahead>>>,
}
unsafe impl Sync for Stream {}

/// Ring of read buffers for `--stage-junk`: reads land here, are never mapped into the cache, never demanded.
pub struct Junk {
    pub per_token: usize,
    arena: crate::uring::Arena,
    slot: usize,
    state: Vec<std::sync::atomic::AtomicU8>,
    next: usize,
    rng: u64,
    pub issued: u64,
    /// ring slot still in flight -> read not issued
    pub skipped: u64,
}

const JUNK_RING: usize = 32;
unsafe impl Send for Stream {}

pub struct StreamConfig {
    pub cache_bytes: usize,
    pub qd: u32,
    pub io_cpu: Option<usize>,
    pub prefault: bool,
}

impl Stream {
    pub fn new(s: &Shards, sidecar_path: &Path, cfg: &StreamConfig) -> Result<Stream, String> {
        Stream::new_with(s, sidecar_path, cfg, None)
    }

    /// As `new`; `layer_weights` (48 values, e.g. tools/flash-quota-weights.txt) makes the slot COUNTS per layer
    /// proportional to them (floor 64, same byte budget); None = equal slot counts.
    pub fn new_with(s: &Shards, sidecar_path: &Path, cfg: &StreamConfig, layer_weights: Option<&[f64]>) -> Result<Stream, String> {
        let lay = Layout::new(s, true);
        sidecar::check(s, &lay, sidecar_path)?;
        // bytes per layer ~ weight x slot size => slot counts ~ weight; floor 64
        if layer_weights.is_some_and(|w| w.len() != N_LAYER || w.iter().any(|&x| !(x > 0.0))) {
            return Err(format!("quota weights: need {N_LAYER} positive values"));
        }
        let w: Vec<f64> = (0..N_LAYER).map(|l| layer_weights.map_or(1.0, |w| w[l]) * lay.geom[l].slot as f64).collect();
        let quota = Cache::quotas(&lay, cfg.cache_bytes, &w, 64);
        let fd = crate::uring::open_direct(sidecar_path).map_err(|e| format!("{}: {e}", sidecar_path.display()))?;
        Stream::from_layout(lay, fd, &quota, cfg)
    }

    /// The stream over a given layout, open sidecar and slot counts (`new_with` after its checks; the model-free tests
    /// build a synthetic layout and sidecar).
    fn from_layout(lay: Layout, fd: i32, quota: &[usize], cfg: &StreamConfig) -> Result<Stream, String> {
        let mut lk = Vec::new();
        for l in 0..N_LAYER {
            let g = lay.geom[l];
            let gate_t = g.gate_w.ggml_type();
            let down_t = g.down_w.ggml_type();
            // SAFETY: plain queries of production ggml type traits.
            unsafe {
                let gate_vdt = fg_vec_dot_type(gate_t);
                let down_vdt = fg_vec_dot_type(down_t);
                lk.push(LayerK {
                    gate_t,
                    gate_vdt,
                    gate_row: g.gate_w.row_bytes(N_EMBD),
                    gate_bytes: g.gate_bytes,
                    r4: if lay.r4 { g.gate_w.r4() } else { None },
                    down_t,
                    down_vdt,
                    down_row: g.down_w.row_bytes(FF_EXP),
                    xq_row: fg_row_size(gate_vdt, N_EMBD as i64),
                    aq_row: fg_row_size(down_vdt, FF_EXP as i64),
                });
            }
            assert_eq!(lk[l].gate_row * FF_EXP, g.gate_bytes, "layer {l}: gate geometry");
            assert_eq!(lk[l].down_row * N_EMBD, g.down_bytes, "layer {l}: down geometry");
        }
        let cache = Cache::new(lay, quota, fd, false, cfg.prefault).map_err(|e| e.to_string())?;
        let io = Arc::new(IoQueue::new());
        // F-IO: env OXF_IO (io.rs module docs), e.g. "cpus=8,9,10,11;split=4;promote=1"; unset = the Flash default
        // (IoTuning::FLASH_DEFAULT: one IO thread, whole reads, ~1 s of spin before parking); "default" = the old parking
        let tuning = IoTuning::flash_from_env()?;
        let io_cfg = IoConfig { qd: cfg.qd, prefetch_qd: 16, cpu: cfg.io_cpu };
        let io_desc = tuning.describe(&io_cfg);
        if tuning != IoTuning::default() {
            eprintln!("oxf-io: {io_desc}");
        }
        let io_thread = io::spawn_tuned(io.clone(), io_cfg, tuning);
        let max_xq = lk.iter().map(|k| k.xq_row).max().unwrap();
        let max_aq = lk.iter().map(|k| k.aq_row).max().unwrap();
        Ok(Stream {
            cache,
            io,
            io_desc,
            io_thread: Some(io_thread),
            lk,
            sh: UnsafeCell::new(Shared {
                entries: Vec::with_capacity(MAX_TOKENS * TOP_K),
                xq: vec![0u8; MAX_TOKENS * max_xq],
                stamp: 0,
                b_experts: Vec::new(),
                b_pairs: vec![Vec::new(); crate::model::N_EXPERT],
                b_wave_entries: [Vec::new(), Vec::new()],
                b_xq: Vec::new(),
                b_gu: Vec::new(),
                b_aq: Vec::new(),
                b_last: vec![[0u64; 8]; N_LAYER],
                b_have_last: vec![false; N_LAYER],
                b_ra: 0,
                b_wave_max: 1,
            }),
            scratch: (0..MAX_THREADS).map(|_| Line(UnsafeCell::new(vec![0u8; max_aq]))).collect(),
            pending: (0..MAX_THREADS).map(|_| Line(UnsafeCell::new(Vec::new()))).collect(),
            barrier: SpinBarrier::default(),
            counters: StreamCounters::default(),
            resident_first: false,
            batch_gemm: true,
            batch_tile: false,
            pf_order: true,
            pf_order_last: false,
            pf_readahead: 64,
            pf_readahead_auto: true,
            pf_ra_warm: 0.5,
            trace: UnsafeCell::new(None),
            error: std::sync::Mutex::new(None),
            junk: UnsafeCell::new(None),
            lookahead: UnsafeCell::new(None),
        })
    }

    /// Hot-set warm-up (session start): queue prefetch-class reads of `list` (from [`Stream::hot_list`] of
    /// an earlier process), at most quota - 16 per layer, in list order. Returns the reads issued. Does not
    /// wait: a demand lookup that finds one still pending re-issues it at demand priority (cache.rs).
    ///
    /// # Safety
    /// Control thread only, not during a graph compute.
    pub unsafe fn warm(&self, list: &[(u16, u16)]) -> usize {
        let sh = &mut *self.sh.get();
        sh.stamp += 1;
        let st = sh.stamp;
        let mut per = vec![0usize; N_LAYER];
        let mut issued = 0;
        for &(l, e) in list {
            let (l, e) = (l as usize, e as usize);
            if l >= N_LAYER || e >= crate::model::N_EXPERT || per[l] + 16 >= self.cache.quota[l] {
                continue;
            }
            per[l] += 1;
            issued += self.cache.lookup(&self.io, l, e, st, ReqClass::Prefetch).1 as usize;
        }
        issued
    }

    /// QC-A: stage `n` never-demanded expert reads per token, spread over the layers (0 = off, the default).
    pub fn enable_stage_junk(&mut self, n: usize) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        let slot = (0..N_LAYER).map(|l| self.cache.layout.geom[l].slot).max().unwrap();
        let arena = crate::uring::Arena::new(JUNK_RING * slot, false).map_err(|e| e.to_string())?;
        *self.junk.get_mut() = Some(Junk {
            per_token: n,
            arena,
            slot,
            state: (0..JUNK_RING).map(|_| std::sync::atomic::AtomicU8::new(crate::cache::EMPTY)).collect(),
            next: 0,
            rng: 0x9e37_79b9_7f4a_7c15,
            issued: 0,
            skipped: 0,
        });
        Ok(())
    }
    /// Expert lookahead on (one-token decodes only): `max` speculative reads per layer at most. Off by default.
    /// `max_pending`: a layer speculates nothing while more speculative reads than this are still in flight.
    pub fn enable_lookahead(&mut self, table: &Path, max: usize, max_pending: usize) -> Result<(), String> {
        // one demand call protects its 10 routed experts + up to MAX_SPEC speculated ones = 18 slots; the pool must
        // leave victims beside them (F-IO review 3c: >= 32; production pools have >= 64)
        let need = 32;
        debug_assert!(need >= TOP_K + crate::lookahead::MAX_SPEC + TOP_K);
        let qmin = self.cache.quota.iter().copied().min().unwrap_or(0);
        if qmin < need {
            return Err(format!("lookahead: the smallest layer pool has {qmin} slots, need >= {need}"));
        }
        let mut la = crate::lookahead::Lookahead::load(table, max)?;
        la.max_pending = max_pending;
        *self.lookahead.get_mut() = Some(Box::new(la));
        Ok(())
    }
    /// Lookahead counters so far (None = off); the hit / unused counts are `Stats::spec_*` of the cache.
    pub fn lookahead_stats(&self) -> Option<crate::lookahead::LaStats> {
        // SAFETY: plain counters read outside a decode.
        unsafe { (*self.lookahead.get()).as_ref().map(|la| la.stats) }
    }
    /// (issued, skipped) junk reads so far
    pub fn junk_counts(&self) -> Option<(u64, u64)> {
        // SAFETY: plain counters read outside a decode.
        unsafe { (*self.junk.get()).as_ref().map(|j| (j.issued, j.skipped)) }
    }

    /// Shared per-layer hook (lookahead staging / F-IO early issue): thread 0 of the per-token expert op, right after
    /// layer `l`'s demand lookups were queued, with the routed ids of every token of the call (`n_tokens x TOP_K`).
    /// Anything it issues must be ReqClass::Prefetch and must not change which bytes a demand lookup gets.
    ///
    /// # Safety
    /// Thread 0 of an expert call only (control-thread discipline of cache.rs).
    unsafe fn on_routes(&self, l: usize, n_tokens: usize, ids: &[i32]) {
        if let Some(la) = (*self.lookahead.get()).as_mut() {
            // decode only: a wider call (prefill chunk, MTP verify) has several tokens per layer
            if n_tokens == 1 {
                if let Some(t) = la.note(l, ids) {
                    let t0 = Instant::now();
                    let mut scores = [0i32; crate::model::N_EXPERT];
                    la.score_noted(t, &mut scores);
                    let mut best = [(0u16, 0i32); crate::lookahead::MAX_SPEC];
                    let (n, above) = la.pick(t, &scores, la.max, |e| self.cache.resident(t, e), &mut best);
                    la.stats.scored += 1;
                    la.stats.above_thr += above as u64;
                    // back-pressure: forget the speculative reads that landed; skip this layer while too many fly
                    la.in_flight.retain(|&(fl, fe)| self.cache.pending(fl as usize, fe as usize));
                    la.stats.max_in_flight = la.stats.max_in_flight.max(la.in_flight.len() as u64);
                    if n > 0 && la.in_flight.len() > la.max_pending {
                        la.stats.skipped_pressure_layers += 1;
                        la.stats.skipped_pressure_reads += n as u64;
                    } else if n > 0 {
                        // layer t's FETCH of this token is the next demand call: its stamp protects the speculated slots
                        let next_stamp = (*self.sh.get()).stamp + 1;
                        // the chosen set never evicts its own members
                        let mut keep = [0u64; crate::model::N_EXPERT / 64];
                        for &(e, _) in &best[..n] {
                            keep[e as usize / 64] |= 1 << (e as usize % 64);
                        }
                        for &(e, _) in &best[..n] {
                            match self.cache.speculate(&self.io, t, e as usize, &keep, next_stamp) {
                                1 => {
                                    la.stats.issued += 1;
                                    la.stats.issued_per_layer[t] += 1;
                                    la.in_flight.push((t as u16, e));
                                }
                                2 => {
                                    la.stats.no_slot += 1;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    la.stats.score_ns += t0.elapsed().as_nanos() as u64;
                }
            } else if l == 0 {
                // a wider call starts: nothing noted for it
                la.note_reset();
            }
        }
        if let Some(j) = (*self.junk.get()).as_mut() {
            if n_tokens != 1 {
                return; // decode only
            }
            let n = (l + 1) * j.per_token / N_LAYER - l * j.per_token / N_LAYER;
            let tl = (l + 1) % N_LAYER;
            for _ in 0..n {
                let i = j.next;
                if j.state[i].load(Acquire) == crate::cache::PENDING {
                    j.skipped += 1;
                    continue;
                }
                j.next = (i + 1) % JUNK_RING;
                // a random expert of the next layer that is not resident (its later demand still misses as before)
                let mut e;
                let mut tries = 0;
                loop {
                    j.rng ^= j.rng << 13;
                    j.rng ^= j.rng >> 7;
                    j.rng ^= j.rng << 17;
                    e = (j.rng % 512) as usize;
                    tries += 1;
                    if !self.cache.resident(tl, e) || tries > 64 {
                        break;
                    }
                }
                j.state[i].store(crate::cache::PENDING, Release);
                j.issued += 1;
                self.io.push(Req {
                    class: ReqClass::Prefetch,
                    fd: self.cache.sidecar_fd,
                    buf: j.arena.ptr.add(i * j.slot),
                    len: self.cache.layout.geom[tl].slot as u32,
                    off: self.cache.layout.offset(tl, e),
                    state: &j.state[i],
                });
            }
        }
    }

    /// Start recording every routed (layer, token, experts) the expert op sees (see `write_trace`).
    pub fn enable_trace(&mut self) {
        *self.trace.get_mut() = Some(Vec::new());
    }

    /// thread 0 of an expert call: record the call's routes
    unsafe fn trace_call(&self, phase: u8, c: &FgExpertCall) {
        if let Some(t) = (*self.trace.get()).as_mut() {
            for tok in 0..c.n_tokens as usize {
                let ids = (c.ids as *const u8).add(tok * c.ids_nb1) as *const i32;
                t.push((phase, c.il as u8, tok as u16, c.n_tokens as u16, std::array::from_fn(|j| *ids.add(j) as u16)));
            }
        }
    }

    /// Write the recorded routes as the P2 trace CSV (`turn,phase,step,layer,slot,expert`; tools/eviction_replay.py
    /// reads it). Steps count tokens through the run: a layer-0 call starts the next n_tokens steps.
    pub fn write_trace(&mut self, path: &Path) -> std::io::Result<usize> {
        use std::io::Write;
        let Some(t) = self.trace.get_mut().as_ref() else { return Ok(0) };
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "turn,phase,step,layer,slot,expert")?;
        for (ph, step, l, ids) in trace_rows(t) {
            for (j, e) in ids.iter().enumerate() {
                writeln!(f, "0,{ph},{step},{l},{j},{e}")?;
            }
        }
        Ok(t.len())
    }

    /// The cache's resident experts, rank-major (see `Cache::hot_list`).
    ///
    /// # Safety
    /// Control thread only, not during a graph compute.
    pub unsafe fn hot_list(&self) -> Vec<(u16, u16)> {
        self.cache.hot_list()
    }

    fn fail(&self, m: String) {
        let mut e = self.error.lock().unwrap();
        if e.is_none() {
            *e = Some(m);
        }
    }

    /// The first expert-read error since the last call (a failed or timed-out read during a decode), if any: the
    /// logits of that decode are invalid. Includes the IO thread's own message.
    pub fn take_error(&self) -> Option<String> {
        let io = self.io.take_error();
        let mine = self.error.lock().unwrap().take();
        match (mine, io) {
            (None, None) => None,
            (a, b) => Some([a, b].into_iter().flatten().collect::<Vec<_>>().join("; ")),
        }
    }

    pub fn stats(&self) -> Stats {
        self.cache.peek_stats()
    }

    /// F38: how many tokens the per-token expert op may take: all tokens x TOP_K experts of a layer are looked up under
    /// one stamp (none of them may be evicted by another), so they must fit the smallest pool in use minus a margin of
    /// 4 slots. The SAME rule as the server's `batch_min_tokens_for` (flash_engine.rs), so this engine guard is a
    /// safety net for flash-m3 / T2 and never changes the server's routing. Wider ubatches use the batched op.
    pub fn per_token_cap(&self) -> usize {
        let min = (0..N_LAYER).map(|l| self.cache.slots(l)).min().unwrap_or(0);
        (min.saturating_sub(4) / TOP_K).clamp(1, MAX_TOKENS)
    }

    /// The callback to hand to `GraphCtx::set_expert_fn` with `self` as userdata.
    pub fn callback() -> crate::graph::ExpertFn {
        stream_expert_cb
    }

    fn wait(&self, e: &Entry, ith: usize) {
        if e.ready() {
            return;
        }
        let t = Instant::now();
        let mut n = 0u32;
        while !e.ready() {
            // a failed read (FAILED after a retry) or a stuck one: give up on this unit, record the error; the op
            // finishes with invalid rows and the caller of the decode turns the error into a failed request
            if e.failed() {
                self.fail("expert read failed (see the IO error)".into());
                break;
            }
            n = n.wrapping_add(1);
            if n & 0xffff == 0 && t.elapsed() > WAIT_TIMEOUT {
                self.fail(format!("expert read not complete after {:?}", WAIT_TIMEOUT));
                break;
            }
            std::hint::spin_loop();
        }
        let ns = t.elapsed().as_nanos() as u64;
        self.counters.wait_ns.fetch_add(ns, Relaxed);
        if ith == 0 {
            self.counters.wait0_ns.fetch_add(ns, Relaxed);
        }
    }

    /// # Safety
    /// Called by the graph with a valid call record (see `fg_expert_call`).
    unsafe fn run(&self, c: &FgExpertCall) {
        if c.kind == FG_EXP_CAP {
            *c.dst = self.per_token_cap() as f32;
            return;
        }
        let l = c.il as usize;
        let k = self.lk[l];
        let nt = c.n_tokens as usize;
        if let Some(m) = max_tokens(c.kind) {
            // the per-token 4-node op holds one ubatch's entries/xq in fixed buffers; the batched (wave) op
            // takes any ubatch (ticket 002: a 156-token prefill hit this check before the kind dispatch)
            assert!(nt <= m, "expert op kind {}: {nt} tokens > {m}", c.kind);
        }
        let ith = c.ith as usize;
        let nth = c.nth as usize;
        match c.kind {
            FG_EXP_FETCH => {
                // ggml-cpu calls a custom op on EVERY graph thread whatever its n_tasks (n_tasks only
                // sizes the plan); thread 0 does the lookups, the node barrier orders the rest
                if ith != 0 {
                    return;
                }
                let t0 = Instant::now();
                self.trace_call(1, c);
                let sh = &mut *self.sh.get();
                // W4-C (F-IO decision): every expert of the call is protected under one stamp; the graph routes wider
                // ubatches to the batched op (per_token_cap), so this only fires on a routing bug, with a clear message
                // instead of the cache's "no evictable slot"
                assert!(
                    nt * TOP_K + 4 <= self.cache.slots(l),
                    "per-token expert op: {nt} tokens x {TOP_K} experts + 4 > {} slots in layer {l} (per_token_cap {})",
                    self.cache.slots(l),
                    self.per_token_cap()
                );
                sh.stamp += 1;
                sh.entries.clear();
                for t in 0..nt {
                    let ids = (c.ids as *const u8).add(t * c.ids_nb1) as *const i32;
                    for j in 0..TOP_K {
                        let e = *ids.add(j);
                        assert!((0..512).contains(&e), "expert id {e}");
                        let (ent, _miss) = self.cache.lookup(&self.io, l, e as usize, sh.stamp, ReqClass::Demand);
                        sh.entries.push(ent);
                    }
                    let x = (c.src as *const u8).add(t * c.src_nb2) as *const f32;
                    fg_from_float(k.gate_vdt, x, sh.xq.as_mut_ptr().add(t * k.xq_row) as *mut c_void, N_EMBD as i64);
                }
                if (*self.junk.get()).is_some() || (*self.lookahead.get()).is_some() {
                    let ids: Vec<i32> = (0..nt)
                        .flat_map(|t| (0..TOP_K).map(move |j| *((c.ids as *const u8).add(t * c.ids_nb1) as *const i32).add(j)))
                        .collect();
                    self.on_routes(l, nt, &ids);
                }
                self.counters.fetch_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
                self.counters.calls.fetch_add(1, Relaxed);
            }
            FG_EXP_GATE | FG_EXP_UP => {
                let sh = &*self.sh.get();
                let part = if c.kind == FG_EXP_GATE { 0 } else { 1 };
                let per = FF_EXP / UP_ROWS;
                let units = nt * TOP_K * per;
                let (u0, u1) = (units * ith / nth, units * (ith + 1) / nth);
                let order = self.unit_order(ith, u0, u1, |u| sh.entries[u / per].ready());
                for u in order {
                    let (tk, ch) = (u / per, u % per);
                    let (t, j) = (tk / TOP_K, tk % TOP_K);
                    let ent = &sh.entries[tk];
                    self.wait(ent, ith);
                    let w = ent.ptr.add(part * k.gate_bytes);
                    let xq = sh.xq.as_ptr().add(t * k.xq_row);
                    let dst = (c.dst as *mut u8).add(t * c.dst_nb2 + j * c.dst_nb1) as *mut f32;
                    let (r0, r1) = (ch * UP_ROWS, (ch + 1) * UP_ROWS);
                    if let Some(ty) = k.r4 {
                        ffi::ggml_cpu_expert_r4_gemv(ty, N_EMBD as i64, dst.add(r0), w.add(r0 * k.gate_row), xq, (r1 - r0) as i64);
                    } else {
                        for r in r0..r1 {
                            fg_vec_dot(k.gate_t, N_EMBD as c_int, dst.add(r), w.add(r * k.gate_row) as *const c_void, xq as *const c_void);
                        }
                    }
                }
            }
            FG_EXP_DOWN => {
                let sh = &*self.sh.get();
                let scratch = &mut *self.scratch[ith].get();
                let per = N_EMBD / DOWN_ROWS;
                let units = nt * TOP_K * per;
                let (u0, u1) = (units * ith / nth, units * (ith + 1) / nth);
                let mut cur_tk = usize::MAX;
                let order = self.unit_order(ith, u0, u1, |u| sh.entries[u / per].ready());
                for u in order {
                    let (tk, ch) = (u / per, u % per);
                    let (t, j) = (tk / TOP_K, tk % TOP_K);
                    if tk != cur_tk {
                        let a = (c.src as *const u8).add(t * c.src_nb2 + j * c.src_nb1) as *const f32;
                        fg_from_float(k.down_vdt, a, scratch.as_mut_ptr() as *mut c_void, FF_EXP as i64);
                        cur_tk = tk;
                    }
                    let ent = &sh.entries[tk];
                    self.wait(ent, ith);
                    let w = ent.ptr.add(2 * k.gate_bytes);
                    let dst = (c.dst as *mut u8).add(t * c.dst_nb2 + j * c.dst_nb1) as *mut f32;
                    for r in ch * DOWN_ROWS..(ch + 1) * DOWN_ROWS {
                        fg_vec_dot(k.down_t, FF_EXP as c_int, dst.add(r), w.add(r * k.down_row) as *const c_void, scratch.as_ptr() as *const c_void);
                    }
                }
            }
            FG_EXP_BATCH => self.run_batch(c),
            _ => panic!("expert op: kind {}", c.kind),
        }
    }
}

impl Stream {
    /// The units `u0..u1` of one thread: in order, or (resident_first) the ready ones first, then the
    /// rest in order. Each unit writes its own rows, so the order never changes a value.
    fn unit_order(&self, ith: usize, u0: usize, u1: usize, ready: impl Fn(usize) -> bool) -> Vec<usize> {
        if !self.resident_first {
            return (u0..u1).collect();
        }
        // SAFETY: per-thread scratch, only thread `ith` touches slot `ith`.
        let pend = unsafe { &mut *self.pending[ith].get() };
        pend.clear();
        let mut v = Vec::with_capacity(u1 - u0);
        for u in u0..u1 {
            if ready(u) {
                v.push(u);
            } else {
                pend.push(u as u32);
            }
        }
        v.extend(pend.iter().map(|&u| u as usize));
        v
    }

    /// FG_EXP_BATCH: a large ubatch. The distinct experts are streamed through the cache in waves of
    /// at most half the layer's pool (the next wave's reads are queued while the current one computes);
    /// per wave: gate/up rows (R4 gemv / vec_dot), production swiglu + from_float per pair, down rows.
    /// Every output row is one vec_dot of one weight row with one quantised activation row, exactly as
    /// production's mul_mat_id, so the result equals the mapped-expert graph bit for bit.
    ///
    /// # Safety
    /// As `run`.
    unsafe fn run_batch(&self, c: &FgExpertCall) {
        let l = c.il as usize;
        let k = self.lk[l];
        let nt = c.n_tokens as usize;
        let (ith, nth) = (c.ith as usize, c.nth as usize);
        let shp = self.sh.get();
        let ff = FF_EXP;
        // FP-9: thread 0's wall per stage of the op (each stage ends at a barrier)
        let mut tlap = Instant::now();
        let mut lap = |i: usize, on: bool| {
            if on {
                let now = Instant::now();
                self.counters.batch_stage_ns[i].fetch_add((now - tlap).as_nanos() as u64, Relaxed);
                tlap = now;
            }
        };
        if ith == 0 {
            let t0 = Instant::now();
            self.trace_call(0, c);
            let sh = &mut *shp;
            for &e in &sh.b_experts {
                sh.b_pairs[e as usize].clear();
            }
            sh.b_experts.clear();
            for t in 0..nt {
                let ids = (c.ids as *const u8).add(t * c.ids_nb1) as *const i32;
                for j in 0..TOP_K {
                    let e = *ids.add(j) as usize;
                    assert!(e < 512, "expert id {e}");
                    if sh.b_pairs[e].is_empty() {
                        sh.b_experts.push(e as u32);
                    }
                    sh.b_pairs[e].push((t * TOP_K + j) as u32);
                }
            }
            sh.b_xq.resize(nt * k.xq_row, 0);
            sh.b_gu.resize(nt * TOP_K * 2 * ff, 0.0);
            sh.b_aq.resize(nt * TOP_K * k.aq_row, 0);
            // W4-I read-ahead policy: decided at the first batched op of a pass (layer 0), kept for the pass
            if !self.pf_readahead_auto {
                sh.b_ra = self.pf_readahead;
            } else if l == 0 {
                let need = sh.b_experts.len().min(self.cache.quota[l]).max(1);
                let res = sh.b_experts.iter().filter(|&&e| self.cache.resident(l, e as usize)).count();
                let warm = res as f64 >= self.pf_ra_warm * need as f64;
                sh.b_ra = if warm { self.pf_readahead } else { 0 };
                if warm { &self.counters.ra_warm_passes } else { &self.counters.ra_cold_passes }.fetch_add(1, Relaxed);
            }
            // two waves are protected at a time; with the layer read-ahead on, up to b_ra more slots may hold reads in
            // flight that this op does not need (never evictable while pending): leave room for them
            // (a layer's read-ahead never exceeds a quarter of its pool: small pools keep useful waves)
            sh.b_wave_max = ((self.cache.quota[l].saturating_sub(4 + sh.b_ra.min(self.cache.quota[l] / 4))) / 2).max(1);
            let wave_max = sh.b_wave_max;
            // PREFILL IO: residents first, then the misses in file order; remember the set for the next read-ahead
            if self.pf_order && self.pf_order_last {
                // pair index = token * TOP_K + slot, pushed in token order: the last pair names the last token
                let pairs = &sh.b_pairs;
                order_experts_last_seen(&mut sh.b_experts, |e| self.cache.resident(l, e as usize), |e| pairs[e as usize].last().map_or(0, |&p| p as usize / TOP_K));
            } else if self.pf_order {
                order_experts(&mut sh.b_experts, |e| self.cache.resident(l, e as usize));
            }
            let mut bits = [0u64; 8];
            for &e in &sh.b_experts {
                bits[e as usize / 64] |= 1 << (e % 64);
            }
            sh.b_last[l] = bits;
            sh.b_have_last[l] = true;
            // queue wave 0
            sh.stamp += 1;
            let st = sh.stamp;
            let n0 = sh.b_experts.len().min(wave_max);
            let mut ents = std::mem::take(&mut sh.b_wave_entries[0]);
            ents.clear();
            for &e in &sh.b_experts[..n0] {
                ents.push(self.cache.lookup_protect(&self.io, l, e as usize, st, st, ReqClass::Demand).0);
            }
            sh.b_wave_entries[0] = ents;
            self.counters.fetch_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            self.counters.calls.fetch_add(1, Relaxed);
        }
        self.barrier.wait(nth);
        let wave_max = (*shp).b_wave_max;
        // quantise x per token
        {
            let sh = &*shp;
            for t in (ith..nt).step_by(nth) {
                let x = (c.src as *const u8).add(t * c.src_nb2) as *const f32;
                fg_from_float(k.gate_vdt, x, sh.b_xq.as_ptr().add(t * k.xq_row) as *mut c_void, N_EMBD as i64);
            }
        }
        let n_exp = (*shp).b_experts.len();
        let n_waves = n_exp.div_ceil(wave_max);
        for w in 0..n_waves {
            let (e0, e1) = (w * wave_max, ((w + 1) * wave_max).min(n_exp));
            if ith == 0 && w + 1 < n_waves {
                // queue the next wave, keeping this one's slots (stamp window)
                let sh = &mut *shp;
                let cur = sh.stamp;
                sh.stamp += 1;
                let st = sh.stamp;
                let (n0, n1) = ((w + 1) * wave_max, ((w + 2) * wave_max).min(n_exp));
                let mut ents = std::mem::take(&mut sh.b_wave_entries[(w + 1) % 2]);
                ents.clear();
                for i in n0..n1 {
                    let e = sh.b_experts[i] as usize;
                    ents.push(self.cache.lookup_protect(&self.io, l, e, st, cur, ReqClass::Demand).0);
                }
                sh.b_wave_entries[(w + 1) % 2] = ents;
            }
            self.barrier.wait(nth); // x quantised, this wave's entries and the next wave's queue published
            lap(0, ith == 0);
            let sh = &*shp;
            let ents = &sh.b_wave_entries[w % 2];
            let exps = &sh.b_experts[e0..e1];
            // gate/up. Units are chunk-major per expert: (expert, 16-row chunk, group of <= 4 of its pairs), so a
            // thread's contiguous range reuses each 16-row weight chunk across all the expert's tokens. A full
            // group on an R4 layer is one i8mm gemm4 call (bitwise equal to the gemv per (row, token): vendor
            // contract + ffi test r4_gemm4_bitwise_vs_vec_dot); other groups go pair by pair as before.
            let gemm = self.batch_gemm && k.r4.is_some();
            let gsz = if gemm { 4 } else { 1 };
            let per = ff / UP_ROWS;
            let ng = |i: usize| sh.b_pairs[exps[i] as usize].len().div_ceil(gsz);
            let mut ust = Vec::with_capacity(exps.len() + 1);
            ust.push(0usize);
            for i in 0..exps.len() {
                ust.push(ust[i] + per * ng(i));
            }
            let units = ust[exps.len()];
            let (u0, u1) = (units * ith / nth, units * (ith + 1) / nth);
            // PREFILL: R4 layers (IQ2_S, IQ3_XXS) through the 8-token tiles instead (same bits); the unit loop below then has no units
            let tile = self.batch_tile && k.r4.is_some();
            if tile {
                self.gate_up_tile(&k, sh, ents, exps, ith, nth);
            }
            let (u0, u1) = if tile { (0, 0) } else { (u0, u1) };
            let at = |u: usize| ust.partition_point(|&x| x <= u) - 1;
            for u in self.unit_order(ith, u0, u1, |u| ents[at(u)].ready()) {
                let wi = at(u);
                let (n_g, loc) = (ng(wi), u - ust[wi]);
                let (ch, gi) = (loc / n_g, loc % n_g);
                let ent = &ents[wi];
                self.wait(ent, ith);
                let pl = &sh.b_pairs[exps[wi] as usize];
                let grp = &pl[gi * gsz..((gi + 1) * gsz).min(pl.len())];
                let (r0, r1) = (ch * UP_ROWS, (ch + 1) * UP_ROWS);
                for part in 0..2 {
                    let w8 = ent.ptr.add(part * k.gate_bytes);
                    let dst = |p: u32| (sh.b_gu.as_ptr() as *mut f32).add((p as usize * 2 + part) * ff);
                    let xq = |p: u32| sh.b_xq.as_ptr().add(p as usize / TOP_K * k.xq_row);
                    if grp.len() == 4 {
                        let s4: [*mut f32; 4] = std::array::from_fn(|t| dst(grp[t]).add(r0));
                        let y4: [*const u8; 4] = std::array::from_fn(|t| xq(grp[t]));
                        ffi::ggml_cpu_expert_r4_gemm4(k.r4.unwrap(), N_EMBD as i64, s4.as_ptr(), w8.add(r0 * k.gate_row), y4.as_ptr(), (r1 - r0) as i64);
                        continue;
                    }
                    for &p in grp {
                        if let Some(ty) = k.r4 {
                            ffi::ggml_cpu_expert_r4_gemv(ty, N_EMBD as i64, dst(p).add(r0), w8.add(r0 * k.gate_row), xq(p), (r1 - r0) as i64);
                        } else {
                            for r in r0..r1 {
                                fg_vec_dot(k.gate_t, N_EMBD as c_int, dst(p).add(r), w8.add(r * k.gate_row) as *const c_void, xq(p) as *const c_void);
                            }
                        }
                    }
                }
            }
            self.barrier.wait(nth);
            lap(1, ith == 0);
            // swiglu (production row kernel) + quantise act, per pair
            {
                let mut act = vec![0f32; ff];
                let np: usize = exps.iter().map(|&e| sh.b_pairs[e as usize].len()).sum();
                let mut it = exps.iter().flat_map(|&e| sh.b_pairs[e as usize].iter().copied());
                for i in 0..np {
                    let p = it.next().unwrap() as usize;
                    if i % nth != ith {
                        continue;
                    }
                    let g = sh.b_gu.as_ptr().add(p * 2 * ff);
                    fg_vec_swiglu(ff as c_int, act.as_mut_ptr(), g, g.add(ff));
                    fg_from_float(k.down_vdt, act.as_ptr(), sh.b_aq.as_ptr().add(p * k.aq_row) as *mut c_void, ff as i64);
                }
            }
            self.barrier.wait(nth);
            lap(2, ith == 0);
            // down: chunk-major as well, (expert, 32-row chunk, pair)
            let per = N_EMBD / DOWN_ROWS;
            let mut ust = Vec::with_capacity(exps.len() + 1);
            ust.push(0usize);
            for i in 0..exps.len() {
                ust.push(ust[i] + per * sh.b_pairs[exps[i] as usize].len());
            }
            let units = ust[exps.len()];
            let (u0, u1) = (units * ith / nth, units * (ith + 1) / nth);
            // PREFILL: IQ4_NL down rows through the 8-token tiles instead (same bits); the unit loop below then has no units
            let dtile = self.batch_tile && k.down_t == ffi::W::Iq4Nl.ggml_type();
            if dtile {
                self.down_tile(c, &k, sh, ents, exps, ith, nth);
            }
            let (u0, u1) = if dtile { (0, 0) } else { (u0, u1) };
            let mut wi = if u0 < units { ust.partition_point(|&x| x <= u0) - 1 } else { 0 };
            for u in u0..u1 {
                while u >= ust[wi + 1] {
                    wi += 1;
                }
                let pl = &sh.b_pairs[exps[wi] as usize];
                let loc = u - ust[wi];
                let (ch, p) = (loc / pl.len(), pl[loc % pl.len()]);
                let ent = &ents[wi];
                self.wait(ent, ith);
                let (t, j) = (p as usize / TOP_K, p as usize % TOP_K);
                let wd = ent.ptr.add(2 * k.gate_bytes);
                let aq = sh.b_aq.as_ptr().add(p as usize * k.aq_row);
                let dst = (c.dst as *mut u8).add(t * c.dst_nb2 + j * c.dst_nb1) as *mut f32;
                for r in ch * DOWN_ROWS..(ch + 1) * DOWN_ROWS {
                    fg_vec_dot(k.down_t, ff as c_int, dst.add(r), wd.add(r * k.down_row) as *const c_void, aq as *const c_void);
                }
            }
            self.barrier.wait(nth); // this wave's slots are free to be evicted by the wave after next
            lap(3, ith == 0);
        }
        // PREFILL IO (a): read ahead for the next layer (control thread; this layer's op is done, the next layer's has
        // not started: nobody uses its slots)
        // (not for the last layer: its FFN runs on the output row only, through the per-token op)
        let ra = if l + 1 < N_LAYER { (*shp).b_ra.min(self.cache.quota[l + 1] / 4) } else { 0 };
        if ith == 0 && ra > 0 && l + 2 < N_LAYER {
            let sh = &*shp;
            // targets: what the last op of layer l+1 used (no history: any expert, empty slots only)
            let (keep, want) = if sh.b_have_last[l + 1] { (sh.b_last[l + 1], sh.b_last[l + 1]) } else { ([u64::MAX; 8], [u64::MAX; 8]) };
            let mut issued = 0;
            for e in 0..crate::model::N_EXPERT {
                if issued >= ra {
                    break;
                }
                if want[e / 64] >> (e % 64) & 1 == 0 {
                    continue;
                }
                match self.cache.prefetch_guarded(&self.io, l + 1, e, &keep) {
                    1 => issued += 1,
                    2 => break,
                    _ => {}
                }
            }
            self.counters.readahead.fetch_add(issued as u64, Relaxed);
        }
    }
}

impl Stream {
    /// W4-I per-phase IO core: move the running IO worker threads (named `oxf-io`, `oxf-io1`, ...) to `cpus`
    /// (worker i -> cpus[i % len]); e.g. another core for a batched prefill than for decode. Placement only. Returns
    /// the number of threads moved. SQPOLL kernel threads, if any, are not touched.
    pub fn set_io_cpus(&self, cpus: &[usize]) -> usize {
        if cpus.is_empty() {
            return 0;
        }
        let mut moved = 0;
        let Ok(dir) = std::fs::read_dir("/proc/self/task") else { return 0 };
        for ent in dir.flatten() {
            let Ok(tid) = ent.file_name().to_string_lossy().parse::<i32>() else { continue };
            let Ok(comm) = std::fs::read_to_string(ent.path().join("comm")) else { continue };
            let comm = comm.trim();
            let Some(rest) = comm.strip_prefix("oxf-io") else { continue };
            let wi: usize = if rest.is_empty() { 0 } else if let Ok(w) = rest.parse() { w } else { continue };
            // SAFETY: cpu_set_t is plain data; sched_setaffinity on a thread of this process.
            unsafe {
                let mut set: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_SET(cpus[wi % cpus.len()], &mut set);
                if libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0 {
                    moved += 1;
                }
            }
        }
        moved
    }
}

/// PREFILL IO: `v` (distinct experts, first-seen order) -> the residents in first-seen order, then the others by
/// ascending id (= ascending sidecar offset within the layer).
fn order_experts(v: &mut Vec<u32>, resident: impl Fn(u32) -> bool) {
    let (mut res, mut miss): (Vec<u32>, Vec<u32>) = v.iter().copied().partition(|&e| resident(e));
    miss.sort_unstable();
    res.append(&mut miss);
    *v = res;
}

/// F123: `v` (distinct experts, first-seen order) -> the residents, then the others, each group by ascending `last`
/// (the op's last token that uses the expert; stable: ties keep first-seen order). Every resident still precedes every
/// miss (in its wave too: a resident is stamped before any miss of that wave picks a victim), so the reads are those of
/// `order_experts`; the op's final waves are the experts the prompt's last tokens used.
fn order_experts_last_seen(v: &mut Vec<u32>, resident: impl Fn(u32) -> bool, last: impl Fn(u32) -> usize) {
    let (mut res, mut miss): (Vec<u32>, Vec<u32>) = v.iter().copied().partition(|&e| resident(e));
    res.sort_by_key(|&e| last(e));
    miss.sort_by_key(|&e| last(e));
    res.append(&mut miss);
    *v = res;
}

/// Tile units of one expert: chunk `ch` (of `per`) has weight position `base + ch * n` (n = the expert's pair count,
/// > 0). Returns the chunks whose position lies in the thread's weight range `[lo, hi)`. Over the threads' ranges
/// (a partition of the total weight) every chunk of every expert is returned exactly once.
fn tile_chunks(base: usize, n: usize, per: usize, lo: usize, hi: usize) -> (usize, usize) {
    let ch0 = if lo <= base { 0 } else { (lo - base).div_ceil(n).min(per) };
    let ch1 = if hi <= base { 0 } else { (hi - base).div_ceil(n).min(per) };
    (ch0, ch1)
}

/// Per-thread buffers of [`Stream::gate_up_tile`].
#[derive(Default)]
struct TileScratch {
    xc: Vec<i8>,          // packed X groups of the current expert
    xd: Vec<f32>,         // their block scales
    s: Vec<*mut f32>,     // output row pointers per packed token
    w: Vec<u32>,          // decoded weight chunk (codes, sub-block scales, block scales)
    done: Vec<bool>,      // experts of the wave this thread has finished
}

thread_local! {
    static TILE: std::cell::RefCell<TileScratch> = std::cell::RefCell::new(TileScratch::default());
}

impl Stream {
    /// Gate and up rows of one wave on an R4 layer (IQ2_S or IQ3_XXS). Units are (expert, 32-row chunk), weighted by the expert's
    /// pair count and split over the threads by weight; a unit decodes its weight chunk once and runs all of the
    /// expert's tokens against it in 8-token tiles. The thread packs an expert's quantised tokens once (8-token
    /// interleaved groups). Each output row is written by exactly one unit; values equal the gemv per (row, token)
    /// bit for bit (ffi test `pf_expert_rows_bitwise`).
    ///
    /// # Safety
    /// As `run_batch` (called between its barriers: `b_xq` complete, `b_gu` rows owned per unit).
    unsafe fn gate_up_tile(&self, k: &LayerK, sh: &Shared, ents: &[Entry], exps: &[u32], ith: usize, nth: usize) {
        const TR: usize = 32;
        let (ff, nb) = (FF_EXP, N_EMBD / 256);
        let per = ff / TR;
        let np = |i: usize| sh.b_pairs[exps[i] as usize].len();
        let mut ust = Vec::with_capacity(exps.len() + 1);
        ust.push(0usize);
        for i in 0..exps.len() {
            ust.push(ust[i] + per * np(i));
        }
        let total = ust[exps.len()];
        let (lo, hi) = (total * ith / nth, total * (ith + 1) / nth);
        TILE.with(|cell| {
            let ts = &mut *cell.borrow_mut();
            let iq2 = k.r4 == Some(ffi::GGML_TYPE_IQ2_S);
            let rows_fn = if iq2 { ffi::pf_expert_iq2s_rows } else { ffi::pf_expert_iq3_rows };
            ts.w.resize(if iq2 { ffi::pf_expert_iq2s_scratch(N_EMBD as i64, TR as i64) } else { ffi::pf_expert_iq3_scratch(N_EMBD as i64, TR as i64) }.div_ceil(4), 0);
            ts.done.clear();
            ts.done.resize(exps.len(), false);
            for pass in 0..2 {
                for wi in 0..exps.len() {
                    let n = np(wi);
                    // this thread's chunks of the expert: units whose weight position falls in [lo, hi)
                    let (ch0, ch1) = tile_chunks(ust[wi], n, per, lo, hi);
                    if ch0 >= ch1 || ts.done[wi] {
                        continue;
                    }
                    let ent = &ents[wi];
                    if pass == 0 && self.resident_first && !ent.ready() {
                        continue; // resident experts first; the rest in the second pass
                    }
                    ts.done[wi] = true;
                    self.wait(ent, ith);
                    let pl = &sh.b_pairs[exps[wi] as usize];
                    let ng = n.div_ceil(8);
                    ts.xc.resize(ng * N_EMBD * 8, 0);
                    ts.xd.resize(ng * nb * 8, 0.0);
                    ts.s.clear();
                    ts.s.resize(ng * 8, std::ptr::null_mut());
                    for g in 0..ng {
                        let m = (n - g * 8).min(8);
                        let vy: [*const u8; 8] =
                            std::array::from_fn(|t| if t < m { sh.b_xq.as_ptr().add(pl[g * 8 + t] as usize / TOP_K * k.xq_row) } else { std::ptr::null() });
                        ffi::pf_expert_pack_x8(N_EMBD as i64, m as c_int, vy.as_ptr(), ts.xc.as_mut_ptr().add(g * N_EMBD * 8), ts.xd.as_mut_ptr().add(g * nb * 8));
                    }
                    for part in 0..2 {
                        let w8 = ent.ptr.add(part * k.gate_bytes);
                        for ch in ch0..ch1 {
                            let r0 = ch * TR;
                            for (i, &p) in pl.iter().enumerate() {
                                ts.s[i] = (sh.b_gu.as_ptr() as *mut f32).add((p as usize * 2 + part) * ff + r0);
                            }
                            rows_fn(N_EMBD as i64, w8.add(r0 * k.gate_row), TR as i64, ts.xc.as_ptr(), ts.xd.as_ptr(), ng as i64, ts.s.as_ptr(), ts.w.as_mut_ptr() as *mut c_void);
                        }
                    }
                }
            }
        });
    }
}

impl Stream {
    /// Down rows of one wave on an IQ4_NL layer: as [`Stream::gate_up_tile`] with units (expert, 32-row chunk) over the
    /// 2560 output rows and the expert's quantised activations (q8_0, per pair) as the tokens. Bitwise equal to the
    /// vec_dot per (row, pair) (ffi test `pf_expert_down_bitwise`). No resident-first pass: every slot of the wave was
    /// already used by gate/up.
    ///
    /// # Safety
    /// As `run_batch` (called between its barriers: `b_aq` complete, output rows owned per unit).
    unsafe fn down_tile(&self, c: &FgExpertCall, k: &LayerK, sh: &Shared, ents: &[Entry], exps: &[u32], ith: usize, nth: usize) {
        const TR: usize = 32;
        let (ff, nb) = (FF_EXP, FF_EXP / 32);
        let per = N_EMBD / TR;
        let np = |i: usize| sh.b_pairs[exps[i] as usize].len();
        let mut ust = Vec::with_capacity(exps.len() + 1);
        ust.push(0usize);
        for i in 0..exps.len() {
            ust.push(ust[i] + per * np(i));
        }
        let total = ust[exps.len()];
        let (lo, hi) = (total * ith / nth, total * (ith + 1) / nth);
        TILE.with(|cell| {
            let ts = &mut *cell.borrow_mut();
            ts.w.resize(ffi::pf_expert_iq4nl_scratch(ff as i64, TR as i64).div_ceil(4), 0);
            for wi in 0..exps.len() {
                let n = np(wi);
                let (ch0, ch1) = tile_chunks(ust[wi], n, per, lo, hi);
                if ch0 >= ch1 {
                    continue;
                }
                let ent = &ents[wi];
                self.wait(ent, ith);
                let pl = &sh.b_pairs[exps[wi] as usize];
                let ng = n.div_ceil(8);
                ts.xc.resize(ng * ff * 8, 0);
                ts.xd.resize(ng * nb * 8, 0.0);
                ts.s.clear();
                ts.s.resize(ng * 8, std::ptr::null_mut());
                for g in 0..ng {
                    let m = (n - g * 8).min(8);
                    let vy: [*const u8; 8] = std::array::from_fn(|t| if t < m { sh.b_aq.as_ptr().add(pl[g * 8 + t] as usize * k.aq_row) } else { std::ptr::null() });
                    ffi::pf_expert_pack_x8_q8_0(ff as i64, m as c_int, vy.as_ptr(), ts.xc.as_mut_ptr().add(g * ff * 8), ts.xd.as_mut_ptr().add(g * nb * 8));
                }
                let wd = ent.ptr.add(2 * k.gate_bytes);
                for ch in ch0..ch1 {
                    let r0 = ch * TR;
                    for (i, &p) in pl.iter().enumerate() {
                        let (t, j) = (p as usize / TOP_K, p as usize % TOP_K);
                        ts.s[i] = ((c.dst as *mut u8).add(t * c.dst_nb2 + j * c.dst_nb1) as *mut f32).add(r0);
                    }
                    ffi::pf_expert_iq4nl_rows(ff as i64, wd.add(r0 * k.down_row), TR as i64, ts.xc.as_ptr(), ts.xd.as_ptr(), ng as i64, ts.s.as_ptr(), ts.w.as_mut_ptr() as *mut c_void);
                }
            }
        });
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.io.stop();
        if let Some(h) = self.io_thread.take() {
            h.join().ok();
        }
    }
}

/// Route-trace records (call phase, layer, row in call, rows in call, ids) -> (phase, step, layer, ids). A layer-0 call
/// starts a pass of n0 steps; a later layer's call with fewer rows covers the LAST rows of that pass (the prefill
/// computes its last layer for the output row only, through the per-token op) and takes the pass's phase. Before this
/// fix that row was written as "phase 1, step 0, layer 47" and the lookahead tools refused the file.
fn trace_rows(t: &[(u8, u8, u16, u16, [u16; TOP_K])]) -> Vec<(u8, usize, u8, [u16; TOP_K])> {
    let (mut base, mut next, mut n0, mut ph0) = (0usize, 0usize, 1usize, 0u8);
    let mut out = Vec::with_capacity(t.len());
    for &(ph, l, tok, n, ids) in t {
        if l == 0 && tok == 0 {
            base = next;
            n0 = n.max(1) as usize;
            ph0 = if ph == 0 { 0 } else { 1 };
        }
        let off = n0.saturating_sub(n as usize);
        let step = base + off + tok as usize;
        next = next.max(step + 1);
        out.push((ph0, step, l, ids));
    }
    out
}

#[cfg(test)]
mod trace_tests {
    use super::*;
    #[test]
    fn prefill_last_layer_row_lands_on_the_last_prefill_step() {
        let ids = [7u16; TOP_K];
        let mut t = Vec::new();
        // prefill pass of 5 rows (batched op, phase 0) for layers 0..46, layer 47 for the output row only (phase 1 call)
        for l in 0..47u8 {
            for tok in 0..5u16 {
                t.push((0u8, l, tok, 5u16, ids));
            }
        }
        t.push((1, 47, 0, 1, ids));
        // two decode tokens
        for _ in 0..2 {
            for l in 0..48u8 {
                t.push((1, l, 0, 1, ids));
            }
        }
        let r = trace_rows(&t);
        let last_pf = r.iter().find(|x| x.2 == 47).unwrap();
        assert_eq!((last_pf.0, last_pf.1), (0, 4), "prefill layer-47 row: phase 0, last prefill step");
        let dec: Vec<usize> = r.iter().filter(|x| x.0 == 1 && x.2 == 0).map(|x| x.1).collect();
        assert_eq!(dec, vec![5, 6]);
        assert!(r.iter().all(|x| x.1 <= 6));
    }
}

unsafe extern "C" fn stream_expert_cb(call: *const FgExpertCall, ud: *mut c_void) {
    let s = &*(ud as *const Stream);
    s.run(&*call);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic IQ2_S / IQ4_NL layout (every layer alike) + a sparse sidecar with random experts 0..n_exp of layers
    /// 0 and 1, on the target directory's file system (O_DIRECT; on this machine dm-crypt). None if O_DIRECT is refused.
    fn synth_stream(tag: &str, n_exp: usize, quota: usize) -> Option<(Stream, std::path::PathBuf)> {
        use crate::ffi::W;
        use std::os::unix::fs::FileExt;
        let (gw, dw) = (W::Iq2S, W::Iq4Nl);
        let (gate_bytes, down_bytes) = (gw.row_bytes(N_EMBD) * FF_EXP, dw.row_bytes(FF_EXP) * N_EMBD);
        let record = 2 * gate_bytes + down_bytes;
        let slot = record.div_ceil(4096) * 4096;
        let g = crate::model::ExpertGeom { gate_w: gw, down_w: dw, gate_bytes, down_bytes, record, slot };
        let base: Vec<u64> = (0..N_LAYER).map(|l| (l * slot * crate::model::N_EXPERT) as u64).collect();
        let total = (N_LAYER * slot * crate::model::N_EXPERT) as u64;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target").join(format!("pf-stream-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("sidecar.bin");
        let f = std::fs::File::create(&p).unwrap();
        f.set_len(total).unwrap();
        let mut x = 0x9e37_79b9_7f4a_7c15u64; // the same experts in every stream
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut rec = vec![0u8; slot];
        for l in 0..2 {
            for e in 0..n_exp {
                for b in rec[..record].iter_mut() {
                    *b = rnd() as u8;
                }
                // sane f16 block scales: R4 IQ2_S super-blocks (328 B: 4 x u16 d first), IQ4_NL blocks (18 B: d first)
                for part in 0..2 {
                    for (i, blk) in rec[part * gate_bytes..(part + 1) * gate_bytes].chunks_exact_mut(328).enumerate() {
                        for r in 0..4 {
                            blk[2 * r..2 * r + 2].copy_from_slice(&(0x2000u16 + ((i * 4 + r) as u16 % 512)).to_le_bytes());
                        }
                    }
                }
                for (i, blk) in rec[2 * gate_bytes..record].chunks_exact_mut(18).enumerate() {
                    blk[..2].copy_from_slice(&(0x2000u16 + (i as u16 % 512)).to_le_bytes());
                }
                f.write_all_at(&rec, base[l] + (e * slot) as u64).unwrap();
            }
        }
        f.sync_all().unwrap();
        drop(f);
        let Ok(fd) = crate::uring::open_direct(&p) else {
            std::fs::remove_dir_all(&dir).ok();
            return None;
        };
        let lay = Layout { geom: vec![g; N_LAYER], base, total, r4: true };
        let cfg = StreamConfig { cache_bytes: 0, qd: 32, io_cpu: None, prefault: false };
        // the tests start from the plain op (first-seen order, no read-ahead) and switch things on explicitly
        let mut st = Stream::from_layout(lay, fd, &vec![quota; N_LAYER], &cfg).unwrap();
        st.pf_order = false;
        st.pf_readahead = 0;
        st.pf_readahead_auto = false;
        Some((st, dir))
    }

    /// F38: on a tiny pool the per-token cap keeps tokens x TOP_K within half the pool, the cap query answers through
    /// the callback, and the fallback (the batched op) takes a 16-token chunk on that pool with the same bits as on a
    /// pool that holds everything.
    #[test]
    fn per_token_cap_guards_small_pools() {
        const NE: usize = 64;
        let nt = 16usize;
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ids = Vec::new();
        for _ in 0..nt {
            let mut row: Vec<i32> = Vec::new();
            while row.len() < TOP_K {
                let e = (rnd() % NE as u64) as i32;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            ids.extend(row);
        }
        let xs: Vec<f32> = (0..N_EMBD * nt).map(|_| ((rnd() % 2001) as f32 - 1000.0) / 1000.0).collect();
        let Some((big, d0)) = synth_stream("capbig", NE, 160) else { return };
        let want = batch_op(&big, 1, &ids, &xs);
        assert_eq!(big.per_token_cap(), 15, "160 slots: (160 - 4) / 10 = 15 tokens");
        drop(big);
        std::fs::remove_dir_all(&d0).ok();
        let Some((small, d1)) = synth_stream("capsmall", NE, 24) else { return };
        let cap = small.per_token_cap();
        assert_eq!(cap, 2, "24 slots: (24 - 4) / 10 = 2 tokens");
        assert!(cap * TOP_K + 4 <= 24);
        // the query through the callback
        let mut out = 0f32;
        let q = FgExpertCall { kind: FG_EXP_CAP, il: 0, ith: 0, nth: 1, n_tokens: 0, ids: std::ptr::null(), ids_nb1: 0,
                               src: std::ptr::null(), src_nb1: 0, src_nb2: 0, dst: &mut out, dst_nb1: 0, dst_nb2: 0 };
        unsafe { small.run(&q) };
        assert_eq!(out as usize, cap);
        // the fallback: 16 tokens (160 lookups) through 24 slots in waves, same bits
        let got = batch_op(&small, 1, &ids, &xs);
        assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "batched op on a tiny pool differs");
        drop(small);
        std::fs::remove_dir_all(&d1).ok();
    }

    /// One decode token's expert ops on layer `l` (FETCH, GATE, UP, DOWN; single graph thread): gate | up | down outputs.
    fn token_ops(st: &Stream, l: usize, ids: &[i32], x: &[f32]) -> Vec<f32> {
        assert_eq!((ids.len(), x.len()), (TOP_K, N_EMBD));
        let mut gate = vec![f32::NAN; FF_EXP * TOP_K];
        let mut up = vec![f32::NAN; FF_EXP * TOP_K];
        let mut down = vec![f32::NAN; N_EMBD * TOP_K];
        let call = |kind, src: *const f32, src_nb1: usize, src_nb2: usize, dst: *mut f32, dst_nb1: usize, dst_nb2: usize| FgExpertCall {
            kind, il: l as c_int, ith: 0, nth: 1, n_tokens: 1, ids: ids.as_ptr(), ids_nb1: TOP_K * 4, src, src_nb1, src_nb2, dst, dst_nb1, dst_nb2,
        };
        unsafe {
            st.run(&call(FG_EXP_FETCH, x.as_ptr(), N_EMBD * 4, N_EMBD * 4, std::ptr::null_mut(), 0, 0));
            st.run(&call(FG_EXP_GATE, x.as_ptr(), N_EMBD * 4, N_EMBD * 4, gate.as_mut_ptr(), FF_EXP * 4, FF_EXP * TOP_K * 4));
            st.run(&call(FG_EXP_UP, x.as_ptr(), N_EMBD * 4, N_EMBD * 4, up.as_mut_ptr(), FF_EXP * 4, FF_EXP * TOP_K * 4));
            // the graph feeds silu(gate) * up; any fixed function of the two does for a bit comparison
            let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| (g * 1e-3).tanh() + (u * 1e-3).tanh()).collect();
            st.run(&call(FG_EXP_DOWN, act.as_ptr(), FF_EXP * 4, FF_EXP * TOP_K * 4, down.as_mut_ptr(), N_EMBD * 4, N_EMBD * TOP_K * 4));
        }
        assert!(st.take_error().is_none());
        gate.extend(up);
        gate.extend(down);
        gate
    }

    /// LOOKAHEAD, end to end without the model: one-token decodes over layers 0 and 1 give the SAME BITS with the
    /// lookahead on and off, on a pool small enough that speculation evicts residents; the table makes layer 0's
    /// experts predict layer 1's exactly, so speculated reads are demanded (hits, pending or ready) and some are
    /// evicted unused; the reads really go through the IO thread (O_DIRECT sidecar).
    #[test]
    fn lookahead_per_token_ops_same_bits() {
        use crate::lookahead as la;
        use crate::model::N_EXPERT;
        const NE: usize = 64;
        let implied = |s: usize| (s * 5 + 1) % NE; // a bijection on 0..64
        // table: only target layer 1 speculates, from distance 1 (layer 0)
        let mut tb = Vec::with_capacity(la::LAT_BYTES);
        tb.extend_from_slice(b"LAT2");
        for v in [2u32, N_LAYER as u32, N_EXPERT as u32, la::N_DIST as u32, 16] {
            tb.extend_from_slice(&v.to_le_bytes());
        }
        for t in 0..N_LAYER {
            tb.extend_from_slice(&(if t == 1 { 100i32 } else { i32::MAX }).to_le_bytes());
        }
        tb.resize(tb.len() + 2 * N_LAYER * N_EXPERT, 0);
        let tab0 = tb.len();
        tb.resize(la::LAT_BYTES, 0);
        for s in 0..NE {
            tb[tab0 + ((la::N_DIST + 0) * N_EXPERT + s) * N_EXPERT + implied(s)] = 120;
        }
        let mut x = 0x0bad_5eed_1234_5678u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let n_tok = 12usize;
        let mut ids0: Vec<Vec<i32>> = Vec::new();
        for _ in 0..n_tok {
            let mut row: Vec<i32> = Vec::new();
            while row.len() < TOP_K {
                let e = (rnd() % NE as u64) as i32;
                if !row.contains(&e) {
                    row.push(e);
                }
            }
            ids0.push(row);
        }
        let xs: Vec<Vec<f32>> = (0..2 * n_tok).map(|_| (0..N_EMBD).map(|_| ((rnd() % 2001) as f32 - 1000.0) / 1000.0).collect()).collect();
        let run = |tag: &str, on: Option<usize>| -> Option<(Vec<Vec<f32>>, Stats, Option<la::LaStats>)> {
            let (mut st, dir) = synth_stream(tag, NE, 32)?;
            st.cache.policy = crate::cache::Policy::Lfu;
            if let Some(max) = on {
                let p = dir.join("la.lat");
                std::fs::write(&p, &tb).unwrap();
                st.enable_lookahead(&p, max, 64).unwrap();
            }
            let mut out = Vec::new();
            for t in 0..n_tok {
                out.push(token_ops(&st, 0, &ids0[t], &xs[2 * t]));
                let ids1: Vec<i32> = ids0[t].iter().map(|&s| implied(s as usize) as i32).collect();
                out.push(token_ops(&st, 1, &ids1, &xs[2 * t + 1]));
            }
            let r = (out, st.stats(), st.lookahead_stats());
            drop(st);
            std::fs::remove_dir_all(&dir).ok();
            Some(r)
        };
        let Some((want, s_off, la_off)) = run("laoff", None) else { return };
        // a pool below the minimum is refused, not run into "no evictable slot"
        if let Some((mut small, d)) = synth_stream("lasmall", NE, 24) {
            let p = d.join("la.lat");
            std::fs::write(&p, &tb).unwrap();
            assert!(small.enable_lookahead(&p, 2, 2).is_err());
            drop(small);
            std::fs::remove_dir_all(&d).ok();
        }
        assert!(la_off.is_none() && s_off.spec_issued == 0);
        for max in [2usize, 8] {
            let Some((got, s_on, la_on)) = run(&format!("laon{max}"), Some(max)) else { return };
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!(a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits()), "lookahead max {max}: op {i} differs");
            }
            let la_on = la_on.unwrap();
            eprintln!("lookahead max {max}: scored {} above {} issued {} no_slot {}; spec hits ready {} pending {} unused {}; demand misses {} (off {})",
                      la_on.scored, la_on.above_thr, la_on.issued, la_on.no_slot, s_on.spec_hit_ready, s_on.spec_hit_pending,
                      s_on.spec_evicted_unused, s_on.demand_misses, s_off.demand_misses);
            // every decode token scores layers 1 and 2 (the hook of layers 0 and 1); layer 1 always has 10 candidates
            assert_eq!(la_on.scored as usize, 2 * n_tok);
            assert_eq!(la_on.above_thr as usize, 10 * n_tok);
            assert_eq!(la_on.issued, s_on.spec_issued);
            assert_eq!(la_on.issued_per_layer.iter().sum::<u64>(), la_on.issued);
            assert_eq!(la_on.issued_per_layer[1], la_on.issued, "only target layer 1 speculates in this table");
            assert_eq!(la_on.skipped_pressure_layers, 0, "max_pending 64 never skips here");
            assert!(s_on.spec_issued > 0 && s_on.spec_issued as usize <= max * n_tok);
            // a speculated expert of layer 1 is demanded by the same token right after: all of them are hits
            assert_eq!(s_on.spec_hit_ready + s_on.spec_hit_pending, s_on.spec_issued);
            assert_eq!(s_on.spec_evicted_unused, 0);
            // each speculated hit replaces one demand miss; a resident pushed out by a speculation may miss later
            assert!(s_on.demand_misses < s_off.demand_misses && s_on.demand_misses + s_on.spec_issued >= s_off.demand_misses);
            assert_eq!(s_on.demand_lookups, s_off.demand_lookups);
        }
    }

    /// One batched expert op (single graph thread) on layer `l`: the down outputs [n_embd, TOP_K, nt].
    fn batch_op(st: &Stream, l: usize, ids: &[i32], x: &[f32]) -> Vec<f32> {
        let nt = ids.len() / TOP_K;
        let mut dst = vec![f32::NAN; N_EMBD * TOP_K * nt];
        let c = FgExpertCall {
            kind: FG_EXP_BATCH,
            il: l as c_int,
            ith: 0,
            nth: 1,
            n_tokens: nt as c_int,
            ids: ids.as_ptr(),
            ids_nb1: TOP_K * 4,
            src: x.as_ptr(),
            src_nb1: N_EMBD * 4,
            src_nb2: N_EMBD * 4,
            dst: dst.as_mut_ptr(),
            dst_nb1: N_EMBD * 4,
            dst_nb2: N_EMBD * TOP_K * 4,
        };
        unsafe { st.run(&c) };
        assert!(st.take_error().is_none());
        dst
    }

    /// PREFILL, end to end without the model: the batched expert op gives the SAME BITS whatever the processing order
    /// and IO timing: 8-token tiles on / off, residents-first order on a cold and on a warm pool, the layer read-ahead
    /// (layer 1's op right after layer 0's, i.e. while the read-ahead reads are in flight), and token order permuted
    /// (each token's rows must not depend on which other tokens share its experts' groups). 48 experts through pools
    /// of 40 slots: several waves and evictions per op.
    #[test]
    fn batched_op_is_order_and_io_independent() {
        const NE: usize = 48;
        let nt = 37usize;
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ids = vec![0i32; nt * TOP_K];
        for t in 0..nt {
            let mut pick: Vec<i32> = Vec::new();
            while pick.len() < TOP_K {
                // skewed: a few experts get most tokens (full 8-token groups), the others partial groups
                let e = if rnd() % 3 == 0 { (rnd() % 6) as i32 } else { (rnd() % NE as u64) as i32 };
                if !pick.contains(&e) {
                    pick.push(e);
                }
            }
            ids[t * TOP_K..(t + 1) * TOP_K].copy_from_slice(&pick);
        }
        let xs: Vec<f32> = (0..nt * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
        let Some((base, d0)) = synth_stream("base", NE, 40) else { return };
        let want0 = batch_op(&base, 0, &ids, &xs);
        let want1 = batch_op(&base, 1, &ids, &xs);
        assert!(want0.iter().all(|v| v.is_finite()) && want0.iter().any(|v| *v != 0.0) && want0 != want1);
        drop(base);
        std::fs::remove_dir_all(&d0).ok();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        for (name, tile, order, ra, last) in [
            ("tile", true, false, 0usize, false),
            ("order", false, true, 0, false),
            ("readahead", false, false, 28, false),
            ("order+readahead", false, true, 28, false),
            ("all", true, true, 28, false),
            ("order-last", false, true, 0, true), // F123 (--flash-pf-order 2)
            ("order-last+readahead+tile", true, true, 28, true),
        ] {
            let Some((mut st, dir)) = synth_stream(name, NE, 40) else { return };
            st.batch_tile = tile;
            st.pf_order = order;
            st.pf_order_last = last;
            st.pf_readahead = ra;
            st.pf_readahead_auto = false; // explicit read-ahead on every pass, also the cold one
            // layer 0 (cold), layer 1 right after it (read-ahead in flight), then both again (warm pools, history)
            for pass in 0..2 {
                let got0 = batch_op(&st, 0, &ids, &xs);
                let got1 = batch_op(&st, 1, &ids, &xs);
                if bits(&got0) != bits(&want0) {
                    // which (token, slot) rows differ, and their experts
                    let mut bad: Vec<(usize, usize, i32, usize)> = Vec::new();
                    for t in 0..nt {
                        for j in 0..TOP_K {
                            let o = (t * TOP_K + j) * N_EMBD;
                            let n = (0..N_EMBD).filter(|&i| got0[o + i].to_bits() != want0[o + i].to_bits()).count();
                            if n > 0 {
                                bad.push((t, j, ids[t * TOP_K + j], n));
                            }
                        }
                    }
                    eprintln!("{name} pass {pass}: {} differing rows (token, slot, expert, floats): {:?}", bad.len(), &bad[..bad.len().min(24)]);
                }
                assert!(bits(&got0) == bits(&want0), "{name}: layer 0 differs (pass {pass})");
                assert!(bits(&got1) == bits(&want1), "{name}: layer 1 differs (pass {pass})");
            }
            // tokens in reverse order: every token's rows unchanged
            let rids: Vec<i32> = (0..nt).rev().flat_map(|t| ids[t * TOP_K..(t + 1) * TOP_K].to_vec()).collect();
            let rxs: Vec<f32> = (0..nt).rev().flat_map(|t| xs[t * N_EMBD..(t + 1) * N_EMBD].to_vec()).collect();
            let got = batch_op(&st, 0, &rids, &rxs);
            let unrev: Vec<f32> = (0..nt).rev().flat_map(|t| got[t * TOP_K * N_EMBD..(t + 1) * TOP_K * N_EMBD].to_vec()).collect();
            assert!(bits(&unrev) == bits(&want0), "{name}: token order changed a row");
            if ra > 0 {
                assert!(st.counters.readahead.load(Relaxed) > 0, "{name}: read-ahead issued nothing");
            }
            drop(st);
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// EVIDENCE for the IO owners, not part of the suite (`--ignored`): the same double read exists on the hot-set
    /// warm-up path (`Stream::warm` = prefetch-class reads; a demand hit on a pending one re-issues the read into the
    /// same slot, cache.rs "demand_reissued"). Warm 30 experts of layer 1 (pool of 64), then run the op at once: if this fails, the
    /// session-start path can compute on a slot that a second read is still rewriting.
    /// F-IO: now the REGRESSION test of "one read per slot" (was the prefill lane's ignored evidence test: 6 of 6
    /// wrong with the re-issue). The demand hits on pending warm-up reads must be promoted or awaited, never re-read.
    #[test]
    fn warm_then_batched_op_single_read() {
        const NE: usize = 48;
        let nt = 37usize;
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let ids: Vec<i32> = (0..nt * TOP_K).map(|i| ((i / TOP_K * 7 + i % TOP_K * 5) % NE) as i32).collect();
        let xs: Vec<f32> = (0..nt * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
        let Some((base, d0)) = synth_stream("wbase", NE, 64) else { return };
        let want = batch_op(&base, 1, &ids, &xs);
        drop(base);
        std::fs::remove_dir_all(&d0).ok();
        let mut pending_hits = 0u64;
        let mut fails = 0;
        for rep in 0..6 {
            let Some((st, dir)) = synth_stream(&format!("warm{rep}"), NE, 64) else { return };
            let list: Vec<(u16, u16)> = (0..30u16).map(|e| (1u16, e)).collect();
            let issued = unsafe { st.warm(&list) };
            let got = batch_op(&st, 1, &ids, &xs);
            let same = got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits());
            let cs = unsafe { *st.cache.stats.get() };
            let reissued = cs.demand_reissued;
            eprintln!("warm rep {rep}: warm issued {issued}, demand_reissued {reissued}, promoted {} awaited {}, op output identical {same}", cs.demand_promoted, cs.demand_awaited);
            assert_eq!(reissued, 0, "a pending prefetched slot was read a second time");
            pending_hits += cs.demand_promoted + cs.demand_awaited;
            fails += !same as usize;
            drop(st);
            std::fs::remove_dir_all(&dir).ok();
        }
        assert_eq!(fails, 0, "{fails} of 6 runs computed on a slot under a second read");
        assert!(pending_hits > 0, "the test never hit a pending warm-up read: it proves nothing");
    }

    /// W4-I: with the default policy (order on, read-ahead 64 auto) a cold pass issues no read-ahead, a repeated pass
    /// does, and both give the plain stream's bits. Also: `set_io_cpus` moves the IO thread and reads still complete.
    #[test]
    fn readahead_auto_cold_off_warm_on() {
        const NE: usize = 48;
        let nt = 29usize;
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let ids: Vec<i32> = (0..nt * TOP_K).map(|i| ((i / TOP_K * 7 + i % TOP_K * 5) % NE) as i32).collect();
        let xs: Vec<f32> = (0..nt * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
        let Some((base, d0)) = synth_stream("abase", NE, 64) else { return };
        let want: Vec<Vec<f32>> = (0..2).map(|l| batch_op(&base, l, &ids, &xs)).collect();
        drop(base);
        std::fs::remove_dir_all(&d0).ok();
        let Some((mut st, dir)) = synth_stream("auto", NE, 64) else { return };
        st.pf_order = true;
        st.pf_readahead_auto = true;
        st.pf_readahead = 12; // W4-I policy; pools of 64 in this test
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
        // cold pass: layers 0, 1
        assert!(same(&batch_op(&st, 0, &ids, &xs), &want[0]) && same(&batch_op(&st, 1, &ids, &xs), &want[1]));
        assert_eq!(st.counters.readahead.load(Relaxed), 0, "read-ahead on a cold pass");
        assert_eq!((st.counters.ra_cold_passes.load(Relaxed), st.counters.ra_warm_passes.load(Relaxed)), (1, 0));
        // repeated pass: warm at layer 0 -> read-ahead for layer 1 (its history = the experts used above)
        let moved = st.set_io_cpus(&[0]);
        assert!(moved >= 1, "IO thread not found");
        assert!(same(&batch_op(&st, 0, &ids, &xs), &want[0]) && same(&batch_op(&st, 1, &ids, &xs), &want[1]));
        assert_eq!(st.counters.ra_warm_passes.load(Relaxed), 1);
        // all 48 experts of layer 1 fit its pool of 64 and are resident: nothing to read ahead is also correct; the
        // counter only has to be consistent with the pool (issued <= 12)
        assert!(st.counters.readahead.load(Relaxed) <= 12);
        assert!(st.counters.batch_stage_ns.iter().all(|c| c.load(Relaxed) > 0), "FP-9 stage clocks");
        drop(st);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// W4-I / FP-8, model-free read count: the same 48 tokens through one layer as ONE op vs TWO ops of 24 (two ubatches).
    /// One pass reads every distinct expert once; two ubatches read again what the first one's later waves evicted.
    /// Pool of 24 slots for ~45 distinct experts (as the real 174-230 slots for ~400): counts are deterministic.
    #[test]
    fn one_pass_reads_fewer_experts_than_two_ubatches() {
        const NE: usize = 48;
        let nt = 48usize;
        let mut x = 0x1234_5678_9abc_def1u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut ids = vec![0i32; nt * TOP_K];
        for t in 0..nt {
            let mut pick: Vec<i32> = Vec::new();
            while pick.len() < TOP_K {
                let e = (rnd() % NE as u64) as i32;
                if !pick.contains(&e) {
                    pick.push(e);
                }
            }
            ids[t * TOP_K..(t + 1) * TOP_K].copy_from_slice(&pick);
        }
        let xs: Vec<f32> = (0..nt * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
        let slot = |st: &Stream| st.cache.layout.geom[0].slot as u64;
        let mut reads = Vec::new();
        let mut outs: Vec<Vec<f32>> = Vec::new();
        for (name, split, order) in [("one", 1usize, false), ("two", 2, false), ("two+order", 2, true)] {
            let Some((mut st, dir)) = synth_stream(name, NE, 24) else { return };
            st.pf_order = order;
            let per = nt / split;
            let mut out = Vec::new();
            for c in 0..split {
                out.extend(batch_op(&st, 0, &ids[c * per * TOP_K..(c + 1) * per * TOP_K], &xs[c * per * N_EMBD..(c + 1) * per * N_EMBD]));
            }
            let n = st.stats().read_bytes / slot(&st);
            eprintln!("reads ({name}): {n} experts for {nt} tokens = {:.2} per token", n as f64 / nt as f64);
            reads.push(n);
            outs.push(out);
            drop(st);
            std::fs::remove_dir_all(&dir).ok();
        }
        let distinct = (0..NE as i32).filter(|e| ids.contains(e)).count() as u64;
        assert_eq!(reads[0], distinct, "one pass reads each distinct expert once");
        assert!(reads[1] > reads[0], "two ubatches re-read: {} vs {}", reads[1], reads[0]);
        assert!(reads[2] <= reads[1], "the order never costs reads: {} vs {}", reads[2], reads[1]);
        // and the rows do not depend on the split (the expert op is per (row, token))
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert!(bits(&outs[0]) == bits(&outs[1]) && bits(&outs[0]) == bits(&outs[2]));
    }

    /// F123: residents, then misses, each by last use ascending, ties in first-seen order; same set.
    #[test]
    fn order_experts_last_seen_groups_by_last_use() {
        // first-seen order 40 7 300 12 511 0 99; residents 7 300 99; last uses below
        let last = |e: u32| match e {
            40 => 9,
            7 => 3,
            300 => 1,
            12 => 9,
            511 => 2,
            0 => 5,
            99 => 3,
            _ => unreachable!(),
        };
        let mut v = vec![40u32, 7, 300, 12, 511, 0, 99];
        super::order_experts_last_seen(&mut v, |e| e == 300 || e == 7 || e == 99, last);
        assert_eq!(v, vec![300, 7, 99, 511, 0, 40, 12]);
        let mut w: Vec<u32> = Vec::new();
        super::order_experts_last_seen(&mut w, |_| true, |_| 0);
        assert!(w.is_empty());
    }

    /// F123, model-free IO: a warm pool, then one batched op over more distinct experts than the pool holds. Order 2
    /// (`pf_order_last`) reads exactly what order 1 reads (W4-I's saving kept) and fewer than the plain order, and
    /// leaves every expert of the op's last tokens resident (what the next decode token needs most); order 1 leaves
    /// the highest-id misses instead. Same output bits in all three.
    #[test]
    fn order_last_keeps_reads_and_the_last_tokens_experts() {
        const NE: usize = 120;
        let nt = 40usize;
        let mut x = 0x0bad_5eed_1234_5679u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let pick = |rnd: &mut dyn FnMut() -> u64, lo: usize, n: usize| {
            let mut p: Vec<i32> = Vec::new();
            while p.len() < TOP_K {
                let e = (lo + (rnd() % n as u64) as usize) as i32;
                if !p.contains(&e) {
                    p.push(e);
                }
            }
            p
        };
        // warm-up op: 16 tokens over experts 0..48 (they become the residents)
        let warm: Vec<i32> = (0..16).flat_map(|_| pick(&mut rnd, 0, 48)).collect();
        // the prompt: 40 tokens over all 120 experts (~110 distinct, pool 40)
        // (its last two tokens only over 48..120: experts the warm-up never loaded, so they are misses of this op)
        let ids: Vec<i32> = (0..nt).flat_map(|t| if t + 2 < nt { pick(&mut rnd, 0, NE) } else { pick(&mut rnd, 48, NE - 48) }).collect();
        let xw: Vec<f32> = (0..16 * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
        let xs: Vec<f32> = (0..nt * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect();
        let tail: Vec<i32> = ids[(nt - 2) * TOP_K..].to_vec(); // experts of the last two tokens
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let mut rows = Vec::new();
        for (name, order, last) in [("plain", false, false), ("order1", true, false), ("order2", true, true)] {
            let Some((mut st, dir)) = synth_stream(&format!("f123-{name}"), NE, 40) else { return };
            st.pf_order = order;
            st.pf_order_last = last;
            batch_op(&st, 0, &warm, &xw);
            let r0 = st.stats().read_bytes;
            let out = batch_op(&st, 0, &ids, &xs);
            let reads = (st.stats().read_bytes - r0) / st.cache.layout.geom[0].slot as u64;
            let kept = tail.iter().filter(|&&e| unsafe { st.cache.resident(0, e as usize) }).count();
            eprintln!("F123 {name}: op reads {reads} experts, last two tokens' experts resident after the op {kept} of {}", tail.len());
            rows.push((reads, kept, bits(&out)));
            drop(st);
            std::fs::remove_dir_all(&dir).ok();
        }
        let (plain, o1, o2) = (&rows[0], &rows[1], &rows[2]);
        assert!(plain.2 == o1.2 && plain.2 == o2.2, "the order changed the op's output bits");
        assert_eq!(o2.0, o1.0, "order 2 must read exactly what order 1 reads");
        assert!(o1.0 < plain.0, "residents first must save reads here: {} vs {}", o1.0, plain.0);
        assert_eq!(o2.1, tail.len(), "order 2 leaves every expert of the last two tokens resident");
        assert!(o1.1 < tail.len(), "order 1 should have evicted some of the last tokens' experts (else the test proves nothing)");
    }

    /// PREFILL IO: residents first in first-seen order, then the misses by id; same set.
    #[test]
    fn order_experts_residents_first() {
        let mut v = vec![40u32, 7, 300, 12, 511, 0, 99];
        super::order_experts(&mut v, |e| e == 300 || e == 7 || e == 99);
        assert_eq!(v, vec![7, 300, 99, 0, 12, 40, 511]);
        let mut w: Vec<u32> = Vec::new();
        super::order_experts(&mut w, |_| true);
        assert!(w.is_empty());
    }

    /// PREFILL: the weight-based split of the tile units covers every (expert, chunk) exactly once and is balanced.
    #[test]
    fn tile_chunks_cover_every_unit_once() {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for case in 0..200 {
            let n_exp = 1 + (rnd() % 90) as usize;
            let per = [20usize, 80, 1, 7][case % 4];
            let pairs: Vec<usize> = (0..n_exp).map(|_| 1 + (rnd() % if case % 3 == 0 { 3 } else { 400 }) as usize).collect();
            let mut ust = vec![0usize];
            for i in 0..n_exp {
                ust.push(ust[i] + per * pairs[i]);
            }
            let total = ust[n_exp];
            for nth in [1usize, 2, 3, 8, 11, 12, 64] {
                let mut seen = vec![0u8; n_exp * per];
                let mut weight = vec![0usize; nth];
                for ith in 0..nth {
                    let (lo, hi) = (total * ith / nth, total * (ith + 1) / nth);
                    for wi in 0..n_exp {
                        let (c0, c1) = super::tile_chunks(ust[wi], pairs[wi], per, lo, hi);
                        assert!(c0 <= c1.max(c0) && c1 <= per);
                        for ch in c0..c1 {
                            seen[wi * per + ch] += 1;
                            weight[ith] += pairs[wi];
                        }
                    }
                }
                assert!(seen.iter().all(|&c| c == 1), "case {case} nth {nth}");
                // balance: no thread exceeds its share by more than one unit of the heaviest expert
                let maxp = *pairs.iter().max().unwrap();
                assert!(weight.iter().all(|&w| w <= total / nth + 1 + maxp), "case {case} nth {nth}");
            }
        }
    }

    /// ticket 002 regression: a >16-token prefill ubatch goes to the batched op, which must not be capped;
    /// the per-token kinds keep the MAX_TOKENS cap (fixed buffers).
    #[test]
    fn token_caps_per_kind() {
        assert_eq!(max_tokens(FG_EXP_BATCH), None);
        for k in [FG_EXP_FETCH, FG_EXP_UP, FG_EXP_GATE, FG_EXP_DOWN] {
            assert_eq!(max_tokens(k), Some(MAX_TOKENS));
        }
        // the graph sends ubatches of >= batch_min_tokens (17 on the fast path) to FG_EXP_BATCH, so every
        // ubatch the per-token op sees fits its cap
        assert!(17 > MAX_TOKENS);
    }
}
