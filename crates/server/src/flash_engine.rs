//! Adapter: the Flash (qwen4exp) engine behind the server's [`Engine`] trait (`--engine flash`,
//! cargo feature `flash`).
//!
//! - Forward pass: the graph port over the production ggml objects (`qwen4exp::graph`).
//!   `exact` (M1): production-identical (fixed353 token-identical at 4 decode / 6 prefill threads):
//!   prefill with the routed experts from the shard mappings, production flash attention.
//!   fast (M3, default): fixed-chunk attention (thread-count independent, decode == verify), shared
//!   expert ahead of the routed experts, resident-first expert units, streamed batched prefill (the
//!   prompt's experts go through the cache, no shard page faults).
//! - Prefill in ubatches of `--ubatch-size` tokens (production n_ubatch 4096). Decode (1 token): the
//!   streaming expert op (M0 cache + io_uring IO thread + R4 sidecar).
//! - Persistent prefix store: `state_id` / `save_state` / `load_state` = token history + KV/indexer
//!   rows of the used cells + the recurrent state (~118 MB + ~30 KB/token) + logits + the expert
//!   cache's hot set ((layer, expert) ids, rank-major). Loading queues the hot set as prefetch reads
//!   (WP-S2 hot-set restore) without waiting; demand misses overtake them (re-issued at demand priority).
//! - Snapshots: the recurrent state (GDN conv/ssm + PLE history) + position; KV and the indexer
//!   cache stay in place (cells are positions).
//! - Hot-set file (`--flash-hotset`, [`Engine::warm_hint`] / [`Engine::apply_warm_hint`]): the same
//!   (layer, expert) list on its own, written by the server after requests and at SIGTERM and queued
//!   at process start, before the first request arrives (the prefix-store entry only knows the
//!   experts of its system prompt at save time; this list is the cache as last used).
//!   A state's hot list is only queued while the cache is cold (fewer than half as many resident
//!   experts as the list names): on a warm resident server (parked-session resume, another agent's
//!   prefix) it would evict experts in use.
//!   Warm-up READS (the hot-set queue at start and on state resume) happen only with `--flash-warmup 1`
//!   (`FlashOptions::warmup`, default off); the hot-set file is still written.
//! - Model, dense arena and cache are leaked for the life of the process.

use crate::engine::{Engine, EngineError, Result};
use qwen4exp::dense::{Dense, DenseMode};
use qwen4exp::graph::{FgOptions, GraphCtx, GraphModel};
use qwen4exp::model::{Shards, N_VOCAB};
use qwen4exp::stream::{Stream, StreamConfig};
use std::path::Path;

pub struct FlashEngine {
    ctx: GraphCtx,
    st: &'static Stream,
    threads: usize,
    threads_batch: usize,
    ubatch: usize,
    /// W4-I FP-8: merge full ubatch chunks into passes of up to this many tokens (0 = off)
    pf_max_pass: usize,
    /// W4-I: CPUs of the IO thread(s) during a batched prefill (empty = leave them), and the decode CPU to return to
    io_cpu_prefill: Vec<usize>,
    io_cpu_decode: Option<usize>,
    /// W5-C long-prompt guard (`--flash-pf-admit`, default on): split passes whose new memory would not fit
    pf_admit: bool,
    /// harness allowance of the guard's budget (GiB)
    admit_harness_gib: f64,
    /// largest pass estimate run so far (bytes; its compute buffers stay allocated)
    pf_high_water: u64,
    /// O-27 PSI thresholds (percent avg10; 0 = off)
    psi_some: f64,
    psi_full: f64,
    n_ctx: usize,
    logits: Vec<f32>,
    opts: FgOptions,
    exact: bool,
    /// F4: the running request's cancel flag (registered on the graph only during multi-token prefill calls)
    cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// hot-set warm-up reads allowed (`--flash-warmup 1`)
    warmup: bool,
}

pub struct FlashSnapshot {
    pos: usize,
    rec: Vec<u8>,
}

pub struct FlashOptions {
    pub threads: usize,
    pub threads_batch: usize,
    pub ubatch: usize,
    pub n_ctx: usize,
    pub cache_gib: f64,
    pub sidecar: Option<std::path::PathBuf>,
    pub io_cpu: Option<usize>,
    /// M1 exact path (production-identical) instead of the M3 fast path
    pub exact: bool,
    /// expert-cache eviction: LFU with aging instead of CLOCK (`--flash-cache-policy lfu`)
    pub cache_lfu: bool,
    /// slot counts per layer ~ these 48 weights (`--flash-quota-weights FILE`); None = equal
    pub quota_weights: Option<Vec<f64>>,
    /// G4 prefill switches (`--flash-attn-tile`, `--flash-dense-gemm`, `--flash-expert-tile`): applied on the fast path
    /// only; the exact path keeps them off
    pub attn_tile: bool,
    pub dense_gemm: bool,
    pub expert_tile: bool,
    /// hot-set warm-up reads (`--flash-warmup 1`): at process start and on state resume
    pub warmup: bool,
    /// expert lookahead table (`--flash-lookahead FILE`, LAT2; FLASH-LOOKAHEAD-DESIGN.md); None = off (default)
    pub lookahead: Option<std::path::PathBuf>,
    /// speculative reads per layer at most (`--flash-la-max`, default 2)
    pub la_max: usize,
    /// skip a layer's speculation while more speculative reads than this are in flight (`--flash-la-pending`, default 2)
    pub la_pending: usize,
    /// W4-I prefill policy (see qwen4exp::stream / pf_plan): batched op takes residents first, misses in file order
    pub pf_order: bool,
    /// F123 (`--flash-pf-order 2`): with `pf_order`, residents then misses each by the op's last token using them (the
    /// op's last waves = the prompt's final experts: no decode cost after a short prefill, same reads)
    pub pf_order_last: bool,
    /// None = auto (64 experts read ahead per layer on passes that start warm, 0 on cold ones); Some(n) = n on every pass
    pub pf_readahead: Option<usize>,
    /// FP-8: merge full ubatch chunks into passes of up to this many tokens (0 = off)
    pub pf_max_pass: usize,
    /// CPUs of the IO thread(s) during a batched prefill (empty = leave them on `io_cpu`)
    pub io_cpu_prefill: Vec<usize>,
    /// `--flash-cache-gib auto` (W5-C / F99): `cache_gib` is ignored and 18 or 16 GiB is chosen from MemAvailable
    pub cache_auto: bool,
    /// harness allowance for the admission (`--flash-admit-harness-gib`; None = the F99 trip's 1.40 GiB)
    pub admit_harness_gib: Option<f64>,
    /// W5-C long-prompt guard (`--flash-pf-admit 0|1`, default 1)
    pub pf_admit: bool,
    /// S-c fused decode phases bitmask (`--flash-fuse N`, 63 = all; default 0; fast path only, bitwise)
    pub fuse: i32,
    /// O-27 PSI guard (`--flash-psi-some`, `--flash-psi-full`, percent avg10; 0 = off): memory pressure caps passes
    pub psi_some: f64,
    pub psi_full: f64,
}

/// `--flash-cache-gib auto` / `--flash-cache-pref`: the decision for a load now (MemAvailable read here), and its log line.
pub fn cache_auto_now(n_ctx: usize, harness_gib: Option<f64>, pref_gib: Option<f64>) -> (qwen4exp::pf_plan::CacheDecision, String) {
    cache_decide_now(n_ctx, harness_gib, pref_gib, qwen4exp::pf_plan::DESKTOP_FLOOR_GIB)
}

/// As [`cache_auto_now`], with the opt-in swap-tolerant rule for an explicit 20 / 18 preference (MemTotal-based,
/// `desktop_floor_gib` = the active-desktop resident floor); auto / 16 / 14 are unchanged.
pub fn cache_decide_now(n_ctx: usize, harness_gib: Option<f64>, pref_gib: Option<f64>, desktop_floor_gib: f64) -> (qwen4exp::pf_plan::CacheDecision, String) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let avail = qwen4exp::pf_plan::mem_available_bytes().map_or(f64::NAN, gib);
    let total = qwen4exp::pf_plan::mem_total_bytes().map_or(f64::NAN, gib);
    let swap_free = qwen4exp::pf_plan::swap_free_bytes().map_or(f64::NAN, gib);
    let inp = qwen4exp::pf_plan::AdmitInputs::measured(avail, n_ctx, harness_gib);
    let (d, seen) = qwen4exp::pf_plan::cache_select_swap_tolerant(&inp, pref_gib, total, desktop_floor_gib, swap_free);
    let mut line = d.log_line(&inp);
    if !seen.is_empty() {
        let parts: Vec<String> = seen.iter().map(|(t, b)| format!("{t:.0}: budget {b:+.2}")).collect();
        line += &format!(
            "; swap-tolerant (explicit pref, MemTotal {total:.2} - cache - dense/base - model growth - desktop floor {desktop_floor_gib:.2}; SwapFree {swap_free:.2} must cover the RAM overflow + {:.1}): [{}]{}",
            qwen4exp::pf_plan::SWAP_MARGIN_GIB,
            parts.join("; "),
            if d.reason == qwen4exp::pf_plan::CacheReason::PreferredSwapTolerant {
                format!(" -> cache {:.0} GiB (idle desktop pages may go to swap; engine scope swap.max=0)", d.cache_gib.unwrap_or(0.0))
            } else {
                " -> no tier, MemAvailable rule above".to_string()
            }
        );
    }
    // F127-3: one stable token the controller can read from the first start-up line (before /props exists)
    line += &format!(" [swap-guard mode={}]", d.swap_guard_mode());
    (d, line)
}

/// `OXE_FLASH_PF_PROTECT_HOT=1` (F123 next; validated at start by startup_check: 0 or 1).
pub fn pf_protect_hot_from_env() -> bool {
    std::env::var("OXE_FLASH_PF_PROTECT_HOT").is_ok_and(|v| v.trim() == "1")
}

fn hot_bytes(hot: &[(u16, u16)]) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + 4 * hot.len());
    b.extend_from_slice(&(hot.len() as u64).to_le_bytes());
    for &(l, x) in hot {
        b.extend_from_slice(&l.to_le_bytes());
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

/// Read a hot list written by [`hot_bytes`].
fn read_hot(r: &mut dyn std::io::Read) -> Result<Vec<(u16, u16)>> {
    let e = |e: std::io::Error| EngineError(format!("hot set: {e}"));
    let mut u = [0u8; 8];
    r.read_exact(&mut u).map_err(e)?;
    let nh = u64::from_le_bytes(u) as usize;
    if nh > qwen4exp::model::N_LAYER * qwen4exp::model::N_EXPERT {
        return Err(EngineError("hot set: bad size".into()));
    }
    let mut hb = vec![0u8; nh * 4];
    r.read_exact(&mut hb).map_err(e)?;
    Ok(hb.chunks(4).map(|c| (u16::from_le_bytes([c[0], c[1]]), u16::from_le_bytes([c[2], c[3]]))).collect())
}

/// Slots of a layer pool the per-token expert op must leave free (same margin as the wave op's sizing).
const POOL_MARGIN: usize = 4;
/// The per-token (4-node) expert op takes at most this many tokens (qwen4exp `MAX_TOKENS`).
const PER_TOKEN_OP_MAX: usize = 16;

/// `fg_options.batch_min_tokens` for a cache whose smallest layer pool has `min_quota` slots.
///
/// The per-token expert op looks up all `n_tokens * TOP_K` experts of a chunk under one stamp, so they are all
/// protected from eviction until the chunk is done: a chunk naming more distinct experts than the pool holds
/// has no victim left and the cache panics. The batched (wave) op protects at most two waves of
/// `(quota - 4) / 2` experts. So only chunks of up to `(min_quota - POOL_MARGIN) / TOP_K` tokens may use the
/// per-token op; longer ones use the wave op. Never above 17 (the per-token op's 16-token limit) and never
/// below 2 (a single token is the decode step: TOP_K experts, and a pool has at least 16 slots).
pub fn batch_min_tokens_for(min_quota: usize) -> i32 {
    let fit = min_quota.saturating_sub(POOL_MARGIN) / qwen4exp::model::TOP_K;
    (fit + 1).clamp(2, PER_TOKEN_OP_MAX + 1) as i32
}

impl FlashEngine {
    /// `model`: the directory of the GGUF shards, or any shard in it.
    pub fn open(model: &Path, o: &FlashOptions) -> std::result::Result<FlashEngine, String> {
        // W5-C / F99: MemAvailable is read before anything of the model is loaded
        let cache_gib = if o.cache_auto {
            let (d, line) = cache_auto_now(o.n_ctx, o.admit_harness_gib, None);
            eprintln!("oxe-server: {line}");
            match d.cache_gib {
                Some(c) => c,
                // F122: below the 14 GiB tier the load would cross the host floor: refuse (exit 74 = admission)
                None => {
                    eprintln!("oxe-server: flash: ADMISSION REFUSED (--flash-cache-gib auto: not even 14 GiB fits)");
                    std::process::exit(74)
                }
            }
        } else {
            o.cache_gib
        };
        let dir = if model.is_dir() { model.to_path_buf() } else { model.parent().ok_or("model path")?.to_path_buf() };
        let s: &'static Shards = Box::leak(Box::new(Shards::open(&dir)?));
        let d: &'static Dense = Box::leak(Box::new(Dense::load(s, DenseMode::VecDot)?));
        // the expert shard mappings are only used by the exact path's mapped-expert prefill
        let gm: &'static GraphModel = Box::leak(Box::new(GraphModel::new(s, d, o.exact)?));
        let sidecar = o.sidecar.clone().unwrap_or_else(qwen4exp::sidecar::default_path);
        let cfg = StreamConfig { cache_bytes: (cache_gib * (1u64 << 30) as f64) as usize, qd: 32, io_cpu: o.io_cpu, prefault: true };
        let mut stb = Box::new(Stream::new_with(s, &sidecar, &cfg, o.quota_weights.as_deref())?);
        stb.resident_first = !o.exact;
        if o.cache_lfu {
            stb.cache.policy = qwen4exp::cache::Policy::Lfu;
        }
        // G4 prefill compute stack: fast path only (the exact path keeps all three off, as before)
        stb.batch_tile = o.expert_tile && !o.exact;
        qwen4exp::graph::set_dense_gemm(o.dense_gemm && !o.exact);
        qwen4exp::graph::set_attn_tile(o.attn_tile && !o.exact);
        eprintln!(
            "oxe-server: flash: prefill switches attn-tile {}, dense-gemm {}, expert-tile {}; hot-set warm-up {}",
            o.attn_tile && !o.exact,
            o.dense_gemm && !o.exact,
            o.expert_tile && !o.exact,
            if o.warmup { "on" } else { "off (--flash-warmup 0)" }
        );
        // decode only (one-token calls); the exact path keeps production's read order
        if let (Some(p), false) = (&o.lookahead, o.exact) {
            stb.enable_lookahead(p, o.la_max, o.la_pending)?;
        }
        // W4-I prefill policy, set before the stream is shared (fast path only, like the G4 switches)
        stb.pf_order = o.pf_order && !o.exact;
        stb.pf_order_last = o.pf_order_last && !o.exact;
        // F123 next: hot-resident protection inside an overflowing batched op (opt-in until its ABBA after a short
        // prefill; default off). Environment switch until the server has a flag for it (main.rs is lane B's).
        stb.pf_protect_hot = stb.pf_order && pf_protect_hot_from_env();
        if stb.pf_protect_hot {
            eprintln!("oxe-server: flash: hot-resident protection in batched ops ON (OXE_FLASH_PF_PROTECT_HOT=1; F123)");
        }
        match if o.exact { Some(0) } else { o.pf_readahead } {
            None => {
                stb.pf_readahead = 64;
                stb.pf_readahead_auto = true;
            }
            Some(n) => {
                stb.pf_readahead = n;
                stb.pf_readahead_auto = false;
            }
        }
        let st: &'static Stream = Box::leak(stb);
        let mut ctx = GraphCtx::try_new(gm, o.n_ctx as u32)?;
        // the per-token expert op needs every expert of a chunk resident at once: chunks that could exceed the
        // smallest layer pool go to the batched (wave) op, which sizes its waves from the pool (QC-052: a 6 GiB
        // cache panicked "no evictable slot" on the first multi-token prefill; 18-21 GiB pools are 137-155 slots
        // in their smallest layer, below the 160 experts a 16-token chunk can name)
        let min_quota = st.cache.quota.iter().copied().min().unwrap_or(0);
        let bmin = batch_min_tokens_for(min_quota);
        eprintln!("oxe-server: flash: smallest expert pool {min_quota} slots/layer; batched expert op from {bmin} tokens");
        let opts = if o.exact { FgOptions::default() } else { FgOptions { fixed_attn: 1, shexp_first: 1, batch_min_tokens: bmin, rs_view: 1, fuse: o.fuse, ..FgOptions::default() } };
        if o.fuse != 0 && !o.exact {
            eprintln!("oxe-server: flash: fused decode phases --flash-fuse {} (bitwise; fast path only)", o.fuse);
        }
        ctx.set_options(&opts);
        Ok(FlashEngine {
            ctx,
            st,
            threads: o.threads.max(1),
            threads_batch: o.threads_batch.max(1),
            ubatch: o.ubatch.max(1),
            pf_max_pass: o.pf_max_pass,
            io_cpu_prefill: o.io_cpu_prefill.clone(),
            io_cpu_decode: o.io_cpu,
            pf_admit: o.pf_admit,
            admit_harness_gib: o.admit_harness_gib.unwrap_or(qwen4exp::pf_plan::ADMIT_HARNESS_GIB),
            pf_high_water: 0,
            psi_some: o.psi_some,
            psi_full: o.psi_full,
            n_ctx: o.n_ctx,
            logits: vec![0.0; N_VOCAB],
            opts,
            exact: o.exact,
            cancel: None,
            warmup: o.warmup,
        })
    }

    fn eval(&mut self, toks: &[i32], streaming: bool, threads: usize) -> Result<()> {
        // SAFETY: the stream is leaked ('static).
        unsafe {
            if streaming {
                self.ctx.set_expert_fn(Some(Stream::callback()), self.st as *const Stream as *mut std::ffi::c_void);
            } else {
                self.ctx.set_expert_fn(None, std::ptr::null_mut());
            }
        }
        self.ctx.decode(toks, threads, &mut self.logits, None).map_err(EngineError)?;
        // an expert read failed or timed out during this decode: its logits are invalid; fail the request, not the
        // server (the failed slots are re-read by the next lookup)
        match self.st.take_error() {
            Some(e) => Err(EngineError(format!("flash: {e}"))),
            None => Ok(()),
        }
    }
}

impl Engine for FlashEngine {
    type Snapshot = FlashSnapshot;

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn n_vocab(&self) -> usize {
        N_VOCAB
    }
    fn n_past(&self) -> usize {
        self.ctx.pos()
    }
    fn reset(&mut self) {
        self.ctx.clear();
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Err(EngineError("empty prefill".into()));
        }
        if self.ctx.pos() + tokens.len() > self.n_ctx {
            return Err(EngineError("context full".into()));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= N_VOCAB) {
            return Err(EngineError(format!("token {t} out of vocab")));
        }
        let toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        if toks.len() == 1 {
            self.eval(&toks, true, self.threads)?;
        } else {
            // F4: a cancel stops this call between graph nodes; the recurrent state is then rolled back to the call's
            // start (snapshot ~118 MB, ~25 ms against seconds of prefill), so engine and cache stay in step
            let snap = match &self.cancel {
                Some(f) if !f.load(std::sync::atomic::Ordering::Relaxed) => {
                    self.ctx.set_abort(Some(f.clone()));
                    Some(self.ctx.save_recurrent())
                }
                _ => None,
            };
            let streamed = self.opts.batch_min_tokens > 0;
            // W4-I: FP-8 plan (pf_max_pass 0 = plain chunks of ubatch; else full chunks merged as memory allows) and
            // the prefill IO core for the batched part
            let plan = if self.pf_max_pass == 0 || self.exact {
                qwen4exp::pf_plan::plan_ubatches(toks.len(), self.ubatch, 0)
            } else {
                let budget = qwen4exp::pf_plan::mem_available_bytes().map_or(0, |b| b.saturating_sub(3 << 29));
                let n_past = self.ctx.pos();
                let admitted = qwen4exp::pf_plan::max_pass_tokens(self.ubatch, self.pf_max_pass, n_past, budget, qwen4exp::graph::attn_tile_mode() == 2);
                qwen4exp::pf_plan::plan_ubatches(toks.len(), self.ubatch, admitted)
            };
            // W5-C long-prompt guard: a pass whose NEW memory (above the high water already allocated) exceeds
            // MemAvailable - harness allowance - floor is split into even halves (>= 256 tokens; lossless class, QC-FP8)
            let plan = if self.pf_admit && !self.exact {
                let reserve = ((self.admit_harness_gib + qwen4exp::pf_plan::ADMIT_FLOOR_GIB) * (1u64 << 30) as f64) as u64;
                let budget = qwen4exp::pf_plan::mem_available_bytes().map_or(0, |b| b.saturating_sub(reserve));
                let fused = qwen4exp::graph::attn_tile_mode() == 2;
                let n_past = self.ctx.pos();
                let mut admitted = qwen4exp::pf_plan::admit_passes(&plan, n_past, budget, self.pf_high_water, fused);
                // O-27: memory pressure (e.g. the desktop being paged out next to a 20 GiB cache): smaller passes,
                // less new scratch; logged; never touches other processes
                if let Some((some10, full10)) = qwen4exp::pf_plan::memory_psi() {
                    if let Some(cap) = qwen4exp::pf_plan::psi_pass_cap(some10, full10, self.psi_some, self.psi_full) {
                        let capped = qwen4exp::pf_plan::cap_passes(&admitted, cap);
                        if capped != admitted {
                            eprintln!(
                                "oxe-server: flash: PSI guard: memory some avg10 {some10:.2} full avg10 {full10:.2} -> passes capped at {cap} ({} -> {} passes)",
                                admitted.len(),
                                capped.len()
                            );
                            admitted = capped;
                        }
                    }
                }
                if admitted != plan {
                    eprintln!(
                        "oxe-server: flash: long-prompt guard: {} passes {:?} -> {} passes (n_past {n_past}, budget {:.2} GiB)",
                        plan.len(),
                        &plan[..plan.len().min(8)],
                        admitted.len(),
                        budget as f64 / (1u64 << 30) as f64
                    );
                }
                let mut np = n_past;
                for &n in &admitted {
                    self.pf_high_water = self.pf_high_water.max(qwen4exp::pf_plan::pass_bytes_est(n, np, fused));
                    np += n;
                }
                admitted
            } else {
                plan
            };
            let batched = toks.len() >= 17 && !self.io_cpu_prefill.is_empty();
            if batched {
                self.st.set_io_cpus(&self.io_cpu_prefill);
            }
            let mut at = 0;
            let mut res = Ok(());
            for n in plan {
                let ch = &toks[at..at + n];
                at += n;
                // fast: every chunk streams (>= batch_min_tokens: the wave node; shorter: the 4-node op,
                // which takes <= 16 tokens); exact: mapped experts except single tokens
                let stream = streamed || ch.len() == 1;
                res = self.eval(ch, stream, self.threads_batch);
                if res.is_err() {
                    break;
                }
            }
            if snap.is_some() {
                self.ctx.set_abort(None);
            }
            // W4-I: the IO thread(s) back on the decode core, also after an error / cancel
            if batched {
                if let Some(c) = self.io_cpu_decode {
                    self.st.set_io_cpus(&[c]);
                }
            }
            if let Err(e) = res {
                if let (Some((pos, rec)), true) = (&snap, e.0.contains(qwen4exp::graph::DECODE_ABORTED)) {
                    self.ctx.restore_recurrent(*pos, rec).map_err(EngineError)?;
                    return Err(EngineError(format!("{} flash prefill aborted at the cancel; state back at {pos}", crate::engine::CANCELLED_MARK)));
                }
                return Err(e);
            }
        }
        Ok(&self.logits)
    }
    fn set_cancel(&mut self, flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        self.cancel = flag;
    }
    fn snapshot(&mut self) -> Result<FlashSnapshot> {
        let (pos, rec) = self.ctx.save_recurrent();
        Ok(FlashSnapshot { pos, rec })
    }
    fn state_id(&self) -> Option<String> {
        Some(format!("flash-state-v2 n_ctx={} exact={} opts={:?}", self.n_ctx, self.exact, self.opts))
    }
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> Result<()> {
        let e = |e: std::io::Error| EngineError(format!("save state: {e}"));
        // FP-2: streamed region by region (same bytes as save_state_vec): a streaming caller (state store, session
        // cache, PrefixStore::save_stream) never holds the ~285 MB anchor state in RAM
        self.ctx.save_compact_to(w).map_err(e)?;
        for c in self.logits.chunks(16384) {
            w.write_all(&c.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(e)?;
        }
        // SAFETY: engine thread, no compute in progress.
        let hot = unsafe { self.st.hot_list() };
        w.write_all(&hot_bytes(&hot)).map_err(e)
    }
    /// FLASH-PREFIX-1: the bytes of [`Engine::save_state`] in ONE buffer of the exact size (the ~285 MB anchor state at
    /// 5.5K tokens is built once; `save_state` into a Vec holds it twice for a moment, in the middle of a cold prefill).
    fn save_state_vec(&mut self) -> Result<Vec<u8>> {
        // SAFETY: engine thread, no compute in progress.
        let hot = hot_bytes(&unsafe { self.st.hot_list() });
        let mut out = Vec::with_capacity(self.state_size_hint() + hot.len());
        self.ctx.save_compact_append(&mut out);
        out.reserve_exact(N_VOCAB * 4 + hot.len());
        for x in &self.logits {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out.extend_from_slice(&hot);
        Ok(out)
    }
    fn state_size_hint(&self) -> usize {
        // the hot list (<= 8 + 4 * 48 * 512 bytes) is not included: callers reserve it on top or let the Vec grow once
        8 + 4 * self.ctx.pos() + self.ctx.compact_bytes() + N_VOCAB * 4
    }
    fn snapshot_bytes(s: &FlashSnapshot) -> usize {
        s.rec.len()
    }
    fn warm_hint(&mut self) -> Option<Vec<u8>> {
        // SAFETY: engine thread, no compute in progress.
        let hot = unsafe { self.st.hot_list() };
        (!hot.is_empty()).then(|| hot_bytes(&hot))
    }
    fn apply_warm_hint(&mut self, b: &[u8]) -> Result<String> {
        if !self.warmup {
            return Ok("hot-set warm-up off (--flash-warmup 0): hint not read".into());
        }
        let mut r = b;
        let hot = read_hot(&mut r)?;
        if !r.is_empty() {
            return Err(EngineError("hot set: trailing bytes".into()));
        }
        // SAFETY: engine thread, no compute in progress.
        let issued = unsafe { self.st.warm(&hot) };
        Ok(format!("hot-set restore: {issued} expert reads queued ({} listed)", hot.len()))
    }
    fn load_state(&mut self, r: &mut dyn std::io::Read, n_pos: usize) -> Result<()> {
        let e = |e: std::io::Error| EngineError(format!("load state: {e}"));
        // every length is checked against this context's layout BEFORE any allocation (a corrupt or crafted
        // file must miss + reset, not request a huge buffer)
        let expect = self.ctx.compact_bytes_at(n_pos, self.n_ctx).ok_or_else(|| EngineError(format!("load state: {n_pos} positions > n_ctx {}", self.n_ctx)))?;
        let mut u = [0u8; 8];
        r.read_exact(&mut u).map_err(e)?;
        let n = u64::from_le_bytes(u);
        if n != expect as u64 {
            return Err(EngineError(format!("load state: compact length {n} != {expect} for {n_pos} positions of this context")));
        }
        let n = expect;
        let mut tb = vec![0u8; n_pos * 4];
        r.read_exact(&mut tb).map_err(e)?;
        let toks: Vec<i32> = tb.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut b = vec![0u8; n];
        r.read_exact(&mut b).map_err(e)?;
        let mut lb = vec![0u8; N_VOCAB * 4];
        r.read_exact(&mut lb).map_err(e)?;
        self.ctx.clear();
        self.ctx.load_compact(&toks, &b).map_err(EngineError)?;
        for (x, c) in self.logits.iter_mut().zip(lb.chunks(4)) {
            *x = f32::from_le_bytes(c.try_into().unwrap());
        }
        let hot = read_hot(r)?;
        if !self.warmup {
            // read (the state format carries it) but not queued: no warm-up reads without --flash-warmup 1
            return Ok(());
        }
        // SAFETY: engine thread, no compute in progress.
        let resident = unsafe { self.st.hot_list() }.len();
        if resident * 2 < hot.len() {
            // SAFETY: as above.
            let issued = unsafe { self.st.warm(&hot) };
            eprintln!("oxe-server: flash: hot-set restore: {issued} expert reads queued ({} listed)", hot.len());
        } else {
            eprintln!("oxe-server: flash: hot-set restore skipped: cache warm ({resident} resident, {} listed)", hot.len());
        }
        Ok(())
    }
    fn restore(&mut self, s: &FlashSnapshot) -> Result<()> {
        if s.pos > self.ctx.pos() {
            return Err(EngineError("restore of a snapshot ahead of the current position".into()));
        }
        self.ctx.restore_recurrent(s.pos, &s.rec).map_err(EngineError)
    }
}

#[cfg(test)]
mod tests {
    //! Model-free: tiny expert pools against large chunks (QC-052: "layer 0: no evictable slot" at 6 GiB).
    use super::*;
    use qwen4exp::cache::{Cache, READY};
    use qwen4exp::ffi::W;
    use qwen4exp::io::{IoQueue, ReqClass};
    use qwen4exp::model::{ExpertGeom, N_EXPERT, N_LAYER, TOP_K};
    use qwen4exp::sidecar::Layout;
    use std::sync::atomic::Ordering::Release;

    /// A cache of `quota` 4 KiB slots per layer over a synthetic layout; no sidecar, no IO thread.
    fn tiny_cache(quota: usize) -> Cache {
        let g = ExpertGeom { gate_w: W::Q8_0, down_w: W::Q8_0, gate_bytes: 1024, down_bytes: 1024, record: 3072, slot: 4096 };
        let base = (0..N_LAYER).map(|l| (l * 4096 * N_EXPERT) as u64).collect();
        let layout = Layout { geom: vec![g; N_LAYER], base, total: (N_LAYER * 4096 * N_EXPERT) as u64, r4: false };
        Cache::new(layout, &vec![quota; N_LAYER], -1, true, false).unwrap()
    }
    /// One lookup as the engine does it, then the read "completes" (there is no IO thread here).
    unsafe fn look(c: &Cache, io: &IoQueue, e: usize, stamp: u64, from: u64) {
        let (ent, _) = c.lookup_protect(io, 0, e, stamp, from, ReqClass::Demand);
        (*ent.state).store(READY, Release);
    }
    /// The per-token op: every expert of an `nt`-token chunk under one stamp (worst case: all distinct).
    unsafe fn per_token_chunk(c: &Cache, io: &IoQueue, first: usize, nt: usize, stamp: u64) {
        for i in 0..nt * TOP_K {
            look(c, io, (first + i) % N_EXPERT, stamp, stamp);
        }
    }

    #[test]
    fn batch_min_tokens_keeps_a_chunk_inside_the_pool() {
        for q in 16..=N_EXPERT {
            let b = batch_min_tokens_for(q) as usize;
            assert!((2..=17).contains(&b), "quota {q}: {b}");
            // the largest chunk the per-token op still gets fits the pool; a single token always does
            assert!((b - 1) * TOP_K + POOL_MARGIN <= q || b == 2, "quota {q}: {b}");
            // and it is not smaller than it has to be
            assert!(b == 17 || b * TOP_K + POOL_MARGIN > q, "quota {q}: {b}");
        }
        // the sizes in use [E from the built-in quota weights]: 6 GiB ~64, 18 GiB ~137, 20 GiB ~149, 21 GiB ~155
        assert_eq!(batch_min_tokens_for(64), 7);
        assert_eq!(batch_min_tokens_for(137), 14);
        assert_eq!(batch_min_tokens_for(149), 15);
        assert_eq!(batch_min_tokens_for(164), 17);
    }

    /// With the capacity-derived limit the per-token op never runs out of victims, chunk after chunk, on any
    /// small pool (LFU, the production policy, and CLOCK).
    #[test]
    fn per_token_chunks_within_the_limit_never_run_out_of_slots() {
        for lfu in [true, false] {
            for q in [16usize, 24, 64, 66, 137, 149] {
                let mut c = tiny_cache(q);
                if lfu {
                    c.policy = qwen4exp::cache::Policy::Lfu;
                }
                let io = IoQueue::new();
                let nt = batch_min_tokens_for(q) as usize - 1;
                for chunk in 0..40 {
                    // SAFETY: single thread, synthetic cache.
                    unsafe { per_token_chunk(&c, &io, chunk * 37, nt, 1 + chunk as u64) };
                }
            }
        }
    }

    /// The bug: a 16-token chunk (the old fixed limit) on a 64-slot pool names 160 experts under one stamp.
    #[test]
    #[should_panic(expected = "no evictable slot")]
    fn sixteen_token_chunk_on_a_64_slot_pool_has_no_victim() {
        let mut c = tiny_cache(64);
        c.policy = qwen4exp::cache::Policy::Lfu;
        let io = IoQueue::new();
        // SAFETY: single thread, synthetic cache.
        unsafe { per_token_chunk(&c, &io, 0, 16, 1) };
    }

    /// W4-2 (review): a slot whose read FAILED is read again by the next lookup. That lookup must stamp
    /// the slot for the current chunk like any other hit; otherwise, once the re-read completes, a later
    /// miss of the SAME chunk may evict it while the kernels still use it. CLOCK, 16 slots: every slot is
    /// full from chunk 1; in chunk 2 expert 0 is re-read, then one new expert misses. Expert 0 must still
    /// be resident in the same slot afterwards.
    #[test]
    fn reread_of_a_failed_slot_is_protected_for_the_current_chunk() {
        use qwen4exp::cache::FAILED;
        let g = ExpertGeom { gate_w: W::Q8_0, down_w: W::Q8_0, gate_bytes: 1024, down_bytes: 1024, record: 3072, slot: 4096 };
        let base = (0..N_LAYER).map(|l| (l * 4096 * N_EXPERT) as u64).collect();
        let layout = Layout { geom: vec![g; N_LAYER], base, total: (N_LAYER * 4096 * N_EXPERT) as u64, r4: false };
        // no_io = false: the FAILED path only exists with IO; reads are queued and never run (no IO thread)
        let c = Cache::new(layout, &vec![16; N_LAYER], -1, false, false).unwrap();
        let io = IoQueue::new();
        // SAFETY: single thread, synthetic cache, no IO thread.
        unsafe {
            let mut first = None;
            for e in 0..16 {
                let (ent, miss) = c.lookup_protect(&io, 0, e, 1, 1, ReqClass::Demand);
                assert!(miss);
                (*ent.state).store(READY, Release);
                if e == 0 {
                    first = Some(ent);
                }
            }
            let first = first.unwrap();
            (*first.state).store(FAILED, Release); // the read of expert 0 failed
            // chunk 2: expert 0 again (re-read, completes), then a new expert that needs a victim
            let (again, _) = c.lookup_protect(&io, 0, 0, 2, 2, ReqClass::Demand);
            assert_eq!(again.ptr, first.ptr);
            (*again.state).store(READY, Release);
            let (new, miss) = c.lookup_protect(&io, 0, 100, 2, 2, ReqClass::Demand);
            assert!(miss);
            (*new.state).store(READY, Release);
            assert_ne!(new.ptr, first.ptr, "the re-read slot of the current chunk was evicted");
            let (still, miss) = c.lookup_protect(&io, 0, 0, 2, 2, ReqClass::Demand);
            assert!(!miss, "expert 0 is no longer resident in the chunk that re-read it");
            assert_eq!(still.ptr, first.ptr);
        }
    }

    /// The batched op's schedule (stream.rs `run_batch`: waves of `(quota - 4) / 2` experts, the next wave
    /// queued while the current one is protected) on tiny pools: a whole ubatch that names every expert of
    /// the layer goes through, repeatedly.
    #[test]
    fn wave_schedule_fits_tiny_pools() {
        for q in [16usize, 24, 64, 66] {
            let mut c = tiny_cache(q);
            c.policy = qwen4exp::cache::Policy::Lfu;
            let io = IoQueue::new();
            let wave = ((q - 4) / 2).max(1);
            let mut stamp = 0u64;
            for _ubatch in 0..3 {
                let experts: Vec<usize> = (0..N_EXPERT).collect();
                let mut cur = 0u64;
                for (w, part) in experts.chunks(wave).enumerate() {
                    stamp += 1;
                    let from = if w == 0 { stamp } else { cur };
                    for &e in part {
                        // SAFETY: single thread, synthetic cache.
                        unsafe { look(&c, &io, e, stamp, from) };
                    }
                    cur = stamp;
                }
            }
        }
    }
}
