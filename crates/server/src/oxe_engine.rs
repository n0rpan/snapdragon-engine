//! Adapter: the M2 decode engine (`crates/model`) behind the server's [`Engine`] trait.
//!
//! - Weights: the file-backed repacked cache (`model::weights`), plan and weights leaked for
//!   the life of the process (`model::load::open_static`).
//! - Threads: a [`sched::DetachedPool`], so the server may call from any blocking thread; the
//!   pool's own leader thread is worker 0 (pinned to the first cpu of `cpus`).
//! - Prefill of 2+ tokens is the batched prefill (`Engine::prefill`, M3.2: bit-identical to
//!   llama.cpp's batched prefill with the same ubatch size, logits of the last token only),
//!   optionally on its own pool (`--oxe-prefill-cpus`, e.g. more cores: prefill is compute
//!   bound). A single token is a decode step. Every call returns the full last-token logits.
//!   With MTP on, `Engine::prefill_nextn` (same KV, state and logits as `Engine::prefill`)
//!   also returns every position's h_nextn, which seeds the MTP KV (catch-up).
//! - Snapshots hold the recurrent state + position (+ the MTP's h of the last position);
//!   the KV cache stays in place.
//! - M3.1 MTP (`--spec-type draft-mtp --spec-draft-n-max N`, default per model, see
//!   [`default_mtp_depth`]): greedy requests go through [`Engine::decode_spec`]; output is
//!   bit-identical to MTP off.

use crate::engine::{Engine, EngineError, Result};
use model::engine::{Engine as Core, EngineConfig, EngineOptions, RecurrentSnapshot};
use model::mtp::{MtpDriver, MtpWeights};
use model::weights::LoadOptions;
use sched::DetachedPool;
use std::path::Path;
use std::time::Duration;

pub struct OxeEngine {
    eng: Core<'static>,
    pool: DetachedPool,
    /// prefill team when it differs from the decode cpus
    prefill_pool: Option<DetachedPool>,
    ubatch: usize,
    n_vocab: usize,
    /// MTP speculation (None = off)
    mtp: Option<MtpDriver>,
    /// identity of the MTP head (its weight cache file name: source sha + layout), for state_id
    mtp_id: Option<String>,
    /// MTP depth the engine was built for (verify lanes - 1); per-request depths are <= this
    mtp_max_depth: usize,
    /// tokens committed by the last lookup verify (for spec_rewind), and (drafted, accepted)
    lookup_last: Option<usize>,
    lookup_counts: (u64, u64),
}

/// Per-model MTP default when no --spec-type flag is given (M3.1 speed results, Codex
/// 2026-11-09): Qwen3.6 depth 2 (+30 % decode); Ornith off (its MTP head chains badly: no
/// gain); any other model off.
pub fn default_mtp_depth(model: &Path, head: Option<&Path>) -> usize {
    let name = model.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    if name.contains("ornith") {
        // Ornith's own MTP head chains badly (64.9% / 15.1%, WP-D5a); with Qwen3.6's head as a
        // sidecar it reaches 2.22-2.38 tokens/verify at depth 2 (ticket 014). The depth is
        // ORNITH_GRAFT_DEPTH, confirmed by 014's speed screen before any promotion.
        if head.is_some() {
            ORNITH_GRAFT_DEPTH
        } else {
            0
        }
    } else if name.contains("qwen3.6") || name.contains("qwen36") {
        2
    } else {
        0
    }
}

/// Ornith's MTP depth with a sidecar head and no tune table: 1 (ticket 018 [M]: the restricted
/// graft head at depth 1 ~47 vs 44 tok/s on the short screen, deeper drafts slower). A tune table
/// (`--mtp-tune`) overrides it per context bucket.
pub const ORNITH_GRAFT_DEPTH: usize = 1;

/// The MTP head source: `explicit` if given, else `<model stem>.mtp-head.gguf` next to the model if
/// it exists, else None (the model's own blk.{n_layer} block).
pub fn mtp_head_path(model: &Path, explicit: Option<&Path>) -> Option<std::path::PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    let side = model.with_extension("mtp-head.gguf");
    side.exists().then_some(side)
}

pub struct OxeSnapshot {
    rec: RecurrentSnapshot,
    /// target h_nextn of position pos-1 (MTP draft input)
    h: Option<Vec<f32>>,
}

impl OxeEngine {
    /// Load `model` (building the weight cache on first use) for decode on `cpus`.
    pub fn open(model: &Path, cpus: &[usize], n_ctx: usize) -> std::result::Result<OxeEngine, String> {
        Self::open_full(model, cpus, None, model::engine::UBATCH, n_ctx, 0)
    }

    /// As [`OxeEngine::open`], with batched prefill on `prefill_cpus` (default: the decode
    /// cpus) in ubatches of `ubatch` tokens (llama.cpp's `--ubatch-size`; 256 in production).
    pub fn open_with(model: &Path, cpus: &[usize], prefill_cpus: Option<&[usize]>, ubatch: usize, n_ctx: usize) -> std::result::Result<OxeEngine, String> {
        Self::open_full(model, cpus, prefill_cpus, ubatch, n_ctx, 0)
    }

    /// As [`OxeEngine::open_with`], with MTP speculation of `depth` draft tokens (0 = off).
    pub fn open_full(model: &Path, cpus: &[usize], prefill_cpus: Option<&[usize]>, ubatch: usize, n_ctx: usize, depth: usize) -> std::result::Result<OxeEngine, String> {
        Self::open_head(model, None, cpus, prefill_cpus, ubatch, n_ctx, depth)
    }

    /// As [`OxeEngine::open_full`], with the MTP block read from `head` (a sidecar GGUF) instead of
    /// the model file. `OXE_MTP_FROM` overrides both (experiments).
    pub fn open_head(model: &Path, head: Option<&Path>, cpus: &[usize], prefill_cpus: Option<&[usize]>, ubatch: usize, n_ctx: usize, depth: usize) -> std::result::Result<OxeEngine, String> {
        // OXE_WEIGHTS=anon (--oxe-weights anon): chunked copy into a THP-advised anonymous arena;
        // OXE_MLOCK=1 (--mlock): lock the arena, or fail to start
        let backing = match std::env::var("OXE_WEIGHTS").as_deref() {
            Ok("anon") => model::weights::Backing::Anon,
            Ok("file") | Err(_) => model::weights::Backing::File,
            Ok(x) => return Err(format!("--oxe-weights {x}: file|anon")),
        };
        let opt = LoadOptions { threads: cpus.len().max(1), backing, mlock: std::env::var_os("OXE_MLOCK").is_some(), ..LoadOptions::default() };
        let (plan, ws) = model::load::open_static(model, cpus.len(), n_ctx, &opt)?;
        // OXE_IDLE_SPIN_US (diagnostic): how long idle decode workers spin before sleeping
        // (default 200 us, as `oxe bench`); between tokens the server's own work decides
        // whether the team sleeps
        let spin = std::env::var("OXE_IDLE_SPIN_US").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(200);
        let pool = DetachedPool::new(cpus, Duration::from_micros(spin)).map_err(|e| format!("thread pool: {e}"))?;
        let prefill_pool = match prefill_cpus {
            Some(pc) if pc != cpus => Some(DetachedPool::new(pc, Duration::from_micros(200)).map_err(|e| format!("prefill thread pool: {e}"))?),
            _ => None,
        };
        let cfg = EngineConfig { n_ctx, logits: true, trace: false, profile: false };
        let mut mtp_id = None;
        let (eng, mtp) = if depth > 0 {
            // OXE_MTP_FROM=<gguf> (experiment, WP-D5): the MTP block from another GGUF of the same
            // shape; output is unchanged (exact verify), only acceptance can differ
            let src = std::env::var_os("OXE_MTP_FROM").map(std::path::PathBuf::from).or_else(|| head.map(Path::to_path_buf)).unwrap_or_else(|| model.to_path_buf());
            // W4-4 b: a sidecar head is hashed from its own bytes (OXE_MODEL_SHA256 names the model)
            let head_opt = LoadOptions { sha_from_env: src == model, ..opt.clone() };
            let mut mwl = MtpWeights::load(&src, plan, &head_opt)?;
            // restricted-vocabulary draft head (WP-D3b): the sidecar's stored vocabulary, or
            // OXE_MTP_DRAFT_VOCAB=<ranking> [OXE_MTP_DRAFT_K=<k>] for experiments
            let rk = std::env::var_os("OXE_MTP_DRAFT_VOCAB").map(std::path::PathBuf::from);
            let k = std::env::var("OXE_MTP_DRAFT_K").ok().and_then(|v| v.parse().ok());
            if let Some(rows) = mwl.apply_draft_vocab(&src, model, rk.as_deref(), k)? {
                eprintln!("oxe-server: MTP draft head restricted to {rows} rows");
            }
            let mw: &'static MtpWeights = Box::leak(Box::new(mwl));
            // own head: "true" (unchanged state id, existing prefix-store entries stay valid);
            // a foreign head: its weight cache name (source sha + layout)
            mtp_id = Some(if src == model {
                "true".to_string()
            } else {
                let c = &mw.store.info.cache_path;
                format!("head:{}", Path::new(c).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| c.clone()))
            });
            let eng = Core::with_options(plan, ws, cfg, EngineOptions { verify_lanes: depth + 1, mtp: Some(mw), kv_type: None })?;
            let d = MtpDriver::new(&eng, depth)?;
            (eng, Some(d))
        } else {
            // prompt lookup (--spec-type lookup sets OXE_LOOKUP_LANES = drafts + 1): a plain engine
            // with verify lanes; plain steps run the same one-lane code as Core::new
            match std::env::var("OXE_LOOKUP_LANES").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&l| l > 1) {
                Some(l) => (Core::with_options(plan, ws, cfg, EngineOptions { verify_lanes: l.min(model::engine::MAX_LANES), mtp: None, kv_type: None })?, None),
                None => (Core::new(plan, ws, cfg)?, None),
            }
        };
        Ok(OxeEngine { n_vocab: plan.desc.n_vocab, eng, pool, prefill_pool, ubatch: ubatch.max(1), mtp, mtp_id, mtp_max_depth: depth, lookup_last: None, lookup_counts: (0, 0) })
    }
}

/// `OXE_SMALL_PREFILL=N` (default 0 = off): prompt segments of at most N tokens run on the decode
/// team instead of the larger prefill team (MAGIC-PLAN MP-5: a 24-token tool turn spends 17% of
/// its thread time in barrier waits on 12 threads; the result does not depend on the team).
fn small_prefill_tokens() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("OXE_SMALL_PREFILL").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0))
}
fn quiet_prefill() -> bool {
    static Q: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *Q.get_or_init(|| std::env::var("OXE_PREFILL_QUIET").map(|v| v.trim() == "1").unwrap_or(false))
}

/// The team for a prompt segment of `n_tokens` (fields, not `&self`: the caller also borrows `mtp`).
fn team_for<'a>(prefill_pool: &'a Option<DetachedPool>, pool: &'a DetachedPool, n_tokens: usize) -> &'a DetachedPool {
    match prefill_pool {
        Some(p) if n_tokens > small_prefill_tokens() => p,
        _ => pool,
    }
}

impl OxeEngine {
    /// The MTP head identity used in the session identity ("true" for the model's own head,
    /// "head:<weight cache name>" for a sidecar), None without MTP.
    pub fn mtp_identity(&self) -> Option<&str> {
        self.mtp_id.as_deref()
    }
}

impl Engine for OxeEngine {
    type Snapshot = OxeSnapshot;

    fn n_ctx(&self) -> usize {
        self.eng.n_ctx
    }
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }
    fn n_past(&self) -> usize {
        self.eng.pos()
    }
    fn reset(&mut self) {
        self.eng.reset();
        if let Some(d) = &mut self.mtp {
            d.sync(&self.eng, None);
        }
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.lookup_last = None;
        if tokens.is_empty() {
            return Err(EngineError("empty prefill".into()));
        }
        if self.eng.pos() + tokens.len() > self.eng.n_ctx {
            return Err(EngineError("context full".into()));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.n_vocab) {
            return Err(EngineError(format!("token {t} out of vocab")));
        }
        let team = team_for(&self.prefill_pool, &self.pool, tokens.len());
        match &mut self.mtp {
            Some(d) if tokens.len() == 1 => {
                let mut out = Vec::new();
                d.step_plain(&self.eng, &self.pool, tokens[0], &mut out);
            }
            Some(d) => {
                d.prefill_batched(&self.eng, team, &self.pool, tokens, self.ubatch);
            }
            None if tokens.len() == 1 => {
                self.eng.step(&self.pool, tokens[0]);
            }
            None => {
                self.eng.prefill(team, tokens, self.ubatch);
            }
        }
        Ok(self.eng.logits())
    }
    /// MAGIC-PLAN MP-5: with `OXE_PREFILL_QUIET=1` a multi-token segment whose logits are not read
    /// skips the output head (`Core::prefill_state_only`: same KV and recurrent state). MTP engines
    /// keep the normal path (their prefill also seeds the MTP head).
    fn prefill_quiet(&mut self, tokens: &[u32]) -> Result<()> {
        if self.mtp.is_some() || tokens.len() < 2 || !quiet_prefill() {
            return self.prefill(tokens).map(|_| ());
        }
        self.lookup_last = None;
        if self.eng.pos() + tokens.len() > self.eng.n_ctx {
            return Err(EngineError("context full".into()));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.n_vocab) {
            return Err(EngineError(format!("token {t} out of vocab")));
        }
        let team = team_for(&self.prefill_pool, &self.pool, tokens.len());
        self.eng.prefill_state_only(team, tokens, self.ubatch);
        Ok(())
    }
    fn snapshot(&mut self) -> Result<OxeSnapshot> {
        Ok(OxeSnapshot { rec: self.eng.save_recurrent(), h: self.mtp.as_ref().map(|d| d.h_prev().to_vec()) })
    }
    fn snapshot_reuse(&mut self, old: OxeSnapshot) -> Result<OxeSnapshot> {
        let rec = self.eng.save_recurrent_into(old.rec);
        let h = match (self.mtp.as_ref(), old.h) {
            (Some(d), Some(mut v)) => {
                v.clear();
                v.extend_from_slice(d.h_prev());
                Some(v)
            }
            (Some(d), None) => Some(d.h_prev().to_vec()),
            (None, _) => None,
        };
        Ok(OxeSnapshot { rec, h })
    }
    fn state_size_hint(&self) -> usize {
        // + MTP rows when on (h per position): a hint only
        self.eng.state_bytes() + self.mtp.as_ref().map_or(0, |_| self.eng.pos() * 4 * self.eng.plan.desc.n_embd)
    }
    fn restore(&mut self, s: &OxeSnapshot) -> Result<()> {
        if s.rec.pos > self.eng.pos() {
            return Err(EngineError("restore of a snapshot ahead of the current position".into()));
        }
        self.eng.restore_recurrent(&s.rec);
        if let Some(d) = &mut self.mtp {
            d.sync(&self.eng, s.h.as_deref());
        }
        Ok(())
    }
    fn snapshot_bytes(s: &OxeSnapshot) -> usize {
        s.rec.bytes() + 4 * s.h.as_ref().map_or(0, Vec::len)
    }
    fn state_id(&self) -> Option<String> {
        // the MTP KV depends on which head wrote it: its weight cache name carries the source sha
        Some(format!("oxe-state-v1 {} mtp={}", self.eng.state_layout(), self.mtp_id.as_deref().unwrap_or("false")))
    }
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> Result<()> {
        let e = |e: std::io::Error| EngineError(format!("save state: {e}"));
        self.eng.write_state(w).map_err(e)?;
        if let Some(d) = &self.mtp {
            let (q, toks, h, hl) = d.export_state();
            w.write_all(&(q as u64).to_le_bytes()).map_err(e)?;
            w.write_all(&(toks.len() as u64).to_le_bytes()).map_err(e)?;
            for t in toks {
                w.write_all(&t.to_le_bytes()).map_err(e)?;
            }
            for x in h.iter().chain(&hl) {
                w.write_all(&x.to_le_bytes()).map_err(e)?;
            }
        }
        Ok(())
    }
    fn load_state(&mut self, r: &mut dyn std::io::Read, n_pos: usize) -> Result<()> {
        let e = |e: std::io::Error| EngineError(format!("load state: {e}"));
        self.eng.read_state(r, n_pos).map_err(e)?;
        if let Some(d) = &mut self.mtp {
            let mut u = [0u8; 8];
            r.read_exact(&mut u).map_err(e)?;
            let q = u64::from_le_bytes(u) as usize;
            r.read_exact(&mut u).map_err(e)?;
            let n = u64::from_le_bytes(u) as usize;
            if n > n_pos {
                return Err(EngineError("load state: bad MTP header".into()));
            }
            let ne = self.eng.plan.desc.n_embd;
            let mut b = vec![0u8; n * 4 + (n + 1) * ne * 4];
            r.read_exact(&mut b).map_err(e)?;
            let toks = b[..n * 4].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let f: Vec<f32> = b[n * 4..].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let (h, hl) = f.split_at(n * ne);
            d.import_state(&self.eng, q, toks, h.to_vec(), hl).map_err(EngineError)?;
        }
        Ok(())
    }
    fn decode_spec(&mut self, token: u32) -> Result<Option<(Vec<u32>, Vec<Vec<f32>>)>> {
        let Some(d) = &mut self.mtp else { return Ok(None) };
        if token as usize >= self.n_vocab {
            return Err(EngineError(format!("token {token} out of vocab")));
        }
        if self.eng.pos() + 1 > self.eng.n_ctx {
            return Err(EngineError("context full".into()));
        }
        let mut out = Vec::new();
        d.step(&self.eng, &self.pool, token, &mut out);
        let rows = (0..out.len()).map(|i| self.eng.logits_lane(i).to_vec()).collect();
        out.pop(); // the bonus token is not in the state yet
        Ok(Some((out, rows)))
    }
    fn decode_argmax(&mut self, token: u32) -> Result<u32> {
        self.lookup_last = None;
        if token as usize >= self.n_vocab {
            return Err(EngineError(format!("token {token} out of vocab")));
        }
        if self.eng.pos() + 1 > self.eng.n_ctx {
            return Err(EngineError("context full".into()));
        }
        // the head's argmax (model::Engine::step): first maximum in row order, the same
        // values the logits row holds, so it equals sampling::argmax of that row
        Ok(match &mut self.mtp {
            Some(d) => {
                let mut out = Vec::with_capacity(1);
                d.step_plain(&self.eng, &self.pool, token, &mut out);
                out[0]
            }
            None => self.eng.step(&self.pool, token),
        })
    }
    fn decode_spec_argmax(&mut self, token: u32) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        let Some(d) = &mut self.mtp else { return Ok(None) };
        if token as usize >= self.n_vocab {
            return Err(EngineError(format!("token {token} out of vocab")));
        }
        if self.eng.pos() + 1 > self.eng.n_ctx {
            return Err(EngineError("context full".into()));
        }
        let mut out = Vec::new();
        d.step(&self.eng, &self.pool, token, &mut out);
        // out[i] = argmax of committed row i (accepted drafts equal the argmax of the row
        // before them; the last is the bonus token)
        let g = out.clone();
        out.pop(); // the bonus token is not in the state yet
        Ok(Some((out, g)))
    }
    fn spec_rewind(&mut self, keep: usize) -> Result<()> {
        if let Some(n) = self.lookup_last.take() {
            // a lookup verify: keep the first `keep` of its `n` committed tokens (commit may be
            // repeated with a smaller count until the next run)
            if keep < n {
                self.eng.commit(keep);
            }
            return Ok(());
        }
        if let Some(d) = &mut self.mtp {
            d.rewind(&self.eng, keep);
        }
        Ok(())
    }
    fn decode_verify(&mut self, token: u32, drafts: &[u32]) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        // external drafts are verified on the plain engine only (with MTP on, the MTP driver owns
        // speculation and its catch-up queue)
        if self.mtp.is_some() || self.eng.verify_lanes() < 2 || drafts.is_empty() {
            return Ok(None);
        }
        if token as usize >= self.n_vocab || drafts.iter().any(|&d| d as usize >= self.n_vocab) {
            return Err(EngineError("token out of vocab".into()));
        }
        let k = drafts.len().min(self.eng.verify_lanes() - 1).min(self.eng.n_ctx.saturating_sub(self.eng.pos() + 1));
        if k == 0 {
            return Ok(None);
        }
        let mut vt = [0u32; model::engine::MAX_LANES];
        vt[0] = token;
        vt[1..=k].copy_from_slice(&drafts[..k]);
        let g = self.eng.verify(&self.pool, &vt[..=k]);
        let mut a = 0;
        while a < k && drafts[a] == g[a] {
            a += 1;
        }
        self.eng.commit(a + 1);
        self.lookup_last = Some(a + 1);
        self.lookup_counts.0 += k as u64;
        self.lookup_counts.1 += a as u64;
        Ok(Some((drafts[..a].to_vec(), g[..=a].to_vec())))
    }
    fn spec_max_depth(&self) -> usize {
        self.mtp_max_depth
    }
    fn spec_set_depth(&mut self, depth: usize) {
        if let Some(d) = &mut self.mtp {
            d.depth = depth.clamp(1, self.mtp_max_depth.max(1));
        }
    }
    fn spec_counters(&self) -> (u64, u64) {
        if self.mtp.is_none() {
            return self.lookup_counts;
        }
        self.mtp.as_ref().map(|d| (d.stats.drafted, d.stats.accepted)).unwrap_or((0, 0))
    }
}
