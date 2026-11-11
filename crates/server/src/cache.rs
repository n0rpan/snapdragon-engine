//! Prefix cache over a single engine sequence.
//!
//! Tracks which tokens are in the engine state and a few snapshots (checkpoints). For a new
//! prompt it reuses the longest prefix that is reachable: either the live state (when it is
//! a prefix of the prompt) or the deepest checkpoint inside the common prefix. At least one
//! prompt token is always evaluated so the request gets fresh logits.
//!
//! Recurrent (delta-net) state cannot be rewound, so only positions with a snapshot can be
//! resumed. The generation loop (`generate.rs`) takes them at:
//! - `prompt_len - 1` (a repeated prompt, or one that diverges inside the reply, hits after
//!   one token) and at the end of each generation;
//! - for chat requests, at ubatch-grid positions just before the first user message (the
//!   system prompt + tools "anchor", shared by every conversation with the same agent) and,
//!   when the request ends with a user message, just before the generation prompt (a
//!   "query" checkpoint: the next user message re-renders the assistant turns of this query
//!   without their `<think>` blocks, Qwen3.x/Ornith templates, so the next prompt diverges
//!   right there).
//!
//! - for chat requests, at the ubatch-grid floor of each message start in the newly
//!   evaluated part of the prompt ("turn" checkpoints, WP-P1: an edit or retry of an
//!   earlier message resumes there instead of at the anchor).
//!
//! Grid positions are multiples of the prefill ubatch counted from where the prefill call
//! started, so splitting the prefill there changes no ubatch boundary: the request's
//! numerics are exactly those of an unsplit prefill. Eviction keeps anchors, then query
//! checkpoints, then turn checkpoints (at most `max_turn`), then the most recently used.
//!
//! Parked sessions (WP-P1): when a request would throw away a large part of the live
//! session (another agent's system prompt, a title request, a subagent), the whole state
//! (`Engine::save_state`: KV + recurrent + MTP) is first copied to RAM together with the
//! session's checkpoints, which stay valid relative to that KV. A later request that
//! continues the parked session loads it back (a byte copy, no prefill) and, if it diverges
//! inside it, restores one of its checkpoints on top. A parked state is bit-identical to
//! the live state it came from, so resuming it computes exactly what the uninterrupted
//! session would have. RAM is capped (`park_budget`, LRU).

use crate::engine::{Engine, Result};

/// Snapshot priority (eviction order: lowest first, then least recently used).
pub const PRIO_TAIL: u8 = 0;
pub const PRIO_TURN: u8 = 1;
pub const PRIO_QUERY: u8 = 2;
pub const PRIO_ANCHOR: u8 = 3;

struct Snap<S> {
    pos: usize,
    s: S,
    used: u64,
    prio: u8,
    clean: bool,
    bytes: usize,
}

/// A session moved out of the engine: its whole state and its checkpoints.
struct Parked<S> {
    tokens: Vec<u32>,
    state: Vec<u8>,
    snaps: Vec<Snap<S>>,
    clean: bool,
    prompt_len: usize,
    used: u64,
}

impl<S> Parked<S> {
    fn bytes(&self) -> usize {
        self.state.len() + self.snaps.iter().map(|s| s.bytes).sum::<usize>()
    }
}

#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub parks: u64,
    pub unparks: u64,
    pub park_evictions: u64,
    pub park_ms: f64,
    pub unpark_ms: f64,
}

pub struct PrefixCache<S> {
    /// tokens currently in the engine state (KV positions 0..len)
    pub tokens: Vec<u32>,
    /// every position <= tokens.len()
    snaps: Vec<Snap<S>>,
    pub max_snapshots: usize,
    clock: u64,
    /// The live state is bit-identical to a prefill of `tokens` from position 0 in
    /// ubatch-sized chunks (true after a reset, after restoring a clean snapshot, and while
    /// prefills continue on the ubatch grid; false after any decode step).
    pub clean: bool,
    /// tokens of `tokens` that came from prompts (the rest is generated text of the last
    /// request); parking is decided on lost prompt tokens
    pub prompt_len: usize,
    parked: Vec<Parked<S>>,
    /// RAM for parked sessions in bytes (0: parking off)
    pub park_budget: usize,
    /// park the live session when a request would drop at least this many of its prompt tokens
    pub park_min: usize,
    /// turn checkpoints kept at most (the least recently used one goes first)
    pub max_turn: usize,
    pub stats: CacheStats,
    /// keep parked sessions evicted from RAM in `spilled` (the worker writes them to the
    /// session cache on disk)
    pub spill: bool,
    pub spilled: Vec<(Vec<u32>, Vec<u8>)>,
    /// where the last `prepare` found its prefix: none|live|checkpoint|parked|parked-checkpoint
    /// (the worker overrides it with store|session after a disk load)
    pub last_src: &'static str,
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Checkpoints of a session written to the disk cache besides its full state: the deepest
/// query/anchor checkpoint (a new user message re-renders the previous query, so the next
/// prompt diverges just after it) and the deepest turn/tail one (an edit or a reply that
/// re-renders differently). At most two, below the session's length, deepest first.
fn persist_points<S>(snaps: Vec<Snap<S>>, n: usize) -> Vec<(usize, S)> {
    let mut hi: Option<Snap<S>> = None;
    let mut lo: Option<Snap<S>> = None;
    for s in snaps.into_iter().filter(|s| s.pos < n) {
        let slot = if s.prio >= PRIO_QUERY { &mut hi } else { &mut lo };
        if slot.as_ref().is_none_or(|b| s.pos > b.pos) {
            *slot = Some(s);
        }
    }
    let mut v: Vec<(usize, S)> = hi.into_iter().chain(lo).map(|s| (s.pos, s.s)).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.dedup_by(|a, b| a.0 == b.0);
    v
}

/// Deepest checkpoint position in `snaps` that is `<= limit` (0 if none).
fn deepest<S>(snaps: &[Snap<S>], limit: usize) -> usize {
    snaps.iter().filter(|s| s.pos <= limit).map(|s| s.pos).max().unwrap_or(0)
}

impl<S> PrefixCache<S> {
    pub fn new(max_snapshots: usize) -> Self {
        PrefixCache {
            tokens: Vec::new(),
            snaps: Vec::new(),
            max_snapshots,
            clock: 0,
            clean: true,
            prompt_len: 0,
            parked: Vec::new(),
            park_budget: 0,
            park_min: 512,
            max_turn: 4,
            stats: CacheStats::default(),
            spill: false,
            spilled: Vec::new(),
            last_src: "none",
        }
    }

    pub fn snapshot_positions(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.snaps.iter().map(|s| s.pos).collect();
        v.sort_unstable();
        v
    }

    /// (position, priority) of every live checkpoint, by position.
    pub fn snapshot_info(&self) -> Vec<(usize, u8)> {
        let mut v: Vec<(usize, u8)> = self.snaps.iter().map(|s| (s.pos, s.prio)).collect();
        v.sort_unstable();
        v
    }

    /// Token counts of the parked sessions (most recently used last).
    pub fn parked_lengths(&self) -> Vec<usize> {
        let mut v: Vec<(u64, usize)> = self.parked.iter().map(|p| (p.used, p.tokens.len())).collect();
        v.sort_unstable();
        v.into_iter().map(|x| x.1).collect()
    }

    pub fn parked_bytes(&self) -> usize {
        self.parked.iter().map(Parked::bytes).sum()
    }

    /// Bring the engine to the longest reusable prefix of `prompt`; returns the number of
    /// prompt tokens already in the state (`n_cached`). `use_cache=false` starts from scratch.
    pub fn prepare<E: Engine<Snapshot = S>>(&mut self, eng: &mut E, prompt: &[u32], use_cache: bool) -> Result<usize> {
        self.clock += 1;
        self.last_src = "none";
        if !use_cache || prompt.is_empty() {
            self.reset(eng);
            return Ok(0);
        }
        debug_assert_eq!(eng.n_past(), self.tokens.len());
        let lim = prompt.len() - 1;
        let lcp = common_prefix(&self.tokens, prompt);
        let limit = lcp.min(lim);
        // live state is a prefix of the prompt (and leaves >= 1 token to evaluate)
        if limit == self.tokens.len() {
            self.last_src = if limit > 0 { "live" } else { "none" };
            return Ok(limit);
        }
        // deepest checkpoint within the reusable prefix, live or parked
        let live = deepest(&self.snaps, limit);
        let mut best_parked: Option<(usize, usize)> = None;
        for (i, p) in self.parked.iter().enumerate() {
            let l = common_prefix(&p.tokens, prompt).min(lim);
            let pos = if l == p.tokens.len() { l } else { deepest(&p.snaps, l) };
            if pos > best_parked.map(|b| b.1).unwrap_or(0) {
                best_parked = Some((i, pos));
            }
        }
        // take the parked session out first: parking the live one may evict others
        let from_parked = best_parked.filter(|b| b.1 > live).map(|(i, pos)| (self.parked.remove(i), pos));
        // keep the session this request would drop, if it is worth it
        self.park(eng, if from_parked.is_some() { 0 } else { live })?;
        if let Some((p, pos)) = from_parked {
            let full = pos == p.tokens.len();
            let n = self.unpark(eng, p, pos)?;
            self.last_src = if n == 0 { "none" } else if full { "parked" } else { "parked-checkpoint" };
            return Ok(n);
        }
        let n = self.restore_live(eng, live)?;
        self.last_src = if n > 0 { "checkpoint" } else { "none" };
        Ok(n)
    }

    /// Restore the live checkpoint at `pos` (0: start over).
    fn restore_live<E: Engine<Snapshot = S>>(&mut self, eng: &mut E, pos: usize) -> Result<usize> {
        if pos == 0 {
            self.reset(eng);
            return Ok(0);
        }
        let clock = self.clock;
        let s = self.snaps.iter_mut().find(|s| s.pos == pos).expect("checkpoint at pos");
        s.used = clock;
        eng.restore(&s.s)?;
        self.clean = s.clean;
        self.tokens.truncate(pos);
        self.prompt_len = self.prompt_len.min(pos);
        // the next write goes to `pos`: checkpoints past it become invalid
        self.snaps.retain(|s| s.pos <= pos);
        Ok(pos)
    }

    /// Park the live session before the engine goes back to `keep`, when that drops at
    /// least `park_min` prompt tokens. Checkpoints at or below `keep` stay live (their KV
    /// is shared), the others move with the parked state.
    fn park<E: Engine<Snapshot = S>>(&mut self, eng: &mut E, keep: usize) -> Result<()> {
        if self.park_budget == 0 || self.prompt_len < keep + self.park_min.max(1) {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let mut state = Vec::with_capacity(eng.state_size_hint());
        if let Err(e) = eng.save_state(&mut state) {
            eprintln!("oxe-server: park: {}", e.0);
            return Ok(());
        }
        let (stay, go): (Vec<_>, Vec<_>) = std::mem::take(&mut self.snaps).into_iter().partition(|s| s.pos <= keep);
        self.snaps = stay;
        let p = Parked { tokens: self.tokens.clone(), state, snaps: go, clean: self.clean, prompt_len: self.prompt_len, used: self.clock };
        if p.bytes() > self.park_budget {
            return Ok(());
        }
        // a parked session that is a prefix of this one is superseded by it
        self.parked.retain(|q| !(q.tokens.len() <= p.tokens.len() && p.tokens.starts_with(&q.tokens)));
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("oxe-server: parked a session of {} tokens ({} MiB) in {ms:.0} ms", p.tokens.len(), p.bytes() >> 20);
        self.stats.parks += 1;
        self.stats.park_ms += ms;
        self.parked.push(p);
        self.fit_budget();
        Ok(())
    }

    /// Drop least recently used parked sessions until they fit `park_budget`.
    fn fit_budget(&mut self) {
        while self.parked_bytes() > self.park_budget {
            let Some((i, _)) = self.parked.iter().enumerate().min_by_key(|(_, p)| p.used) else { return };
            let p = self.parked.remove(i);
            self.stats.park_evictions += 1;
            if self.spill {
                self.spilled.push((p.tokens, p.state));
            }
        }
    }

    /// Load parked session `p` and resume it at `pos` (its length or one of its checkpoints).
    fn unpark<E: Engine<Snapshot = S>>(&mut self, eng: &mut E, mut p: Parked<S>, pos: usize) -> Result<usize> {
        let t0 = std::time::Instant::now();
        let n = p.tokens.len();
        if let Err(e) = eng.load_state(&mut &p.state[..], n) {
            eprintln!("oxe-server: unpark: {}", e.0);
            self.reset(eng);
            return Ok(0);
        }
        self.snaps.clear();
        self.tokens = p.tokens.clone();
        self.prompt_len = p.prompt_len;
        self.clean = p.clean;
        let got = if pos == n {
            self.snaps = p.snaps;
            n
        } else {
            // a branch inside the parked session: its checkpoints up to `pos` come along, the
            // parked copy stays for its own continuation
            let (here, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut p.snaps).into_iter().partition(|s| s.pos <= pos);
            p.snaps = keep;
            p.used = self.clock;
            self.snaps = here;
            self.parked.push(p);
            self.fit_budget();
            self.restore_live(eng, pos)?
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("oxe-server: resumed a parked session at {got} of {n} tokens in {ms:.0} ms");
        self.stats.unparks += 1;
        self.stats.unpark_ms += ms;
        Ok(got)
    }

    /// Take every parked session out for the session cache: (tokens, state bytes, the
    /// checkpoints worth persisting, deepest first; see [`persist_points`]).
    pub fn drain_parked(&mut self) -> Vec<(Vec<u32>, Vec<u8>, Vec<(usize, S)>)> {
        std::mem::take(&mut self.parked)
            .into_iter()
            .map(|p| {
                let n = p.tokens.len();
                (p.tokens, p.state, persist_points(p.snaps, n))
            })
            .collect()
    }

    /// The live session's checkpoints worth persisting (deepest first); the cache keeps none.
    pub fn take_live_points(&mut self) -> Vec<(usize, S)> {
        let n = self.tokens.len();
        persist_points(std::mem::take(&mut self.snaps), n)
    }

    pub fn reset<E: Engine<Snapshot = S>>(&mut self, eng: &mut E) {
        eng.reset();
        self.tokens.clear();
        self.snaps.clear();
        self.clean = true;
        self.prompt_len = 0;
    }

    /// The engine state was replaced wholesale (a prefix-store load): it now holds `tokens`,
    /// computed elsewhere, so no in-memory checkpoint matches its KV any more.
    pub fn adopt(&mut self, tokens: &[u32], clean: bool) {
        self.tokens.clear();
        self.tokens.extend_from_slice(tokens);
        self.snaps.clear();
        self.clean = clean;
        self.prompt_len = tokens.len();
    }

    /// Record tokens that were just appended to the engine state.
    pub fn appended(&mut self, toks: &[u32]) {
        self.tokens.extend_from_slice(toks);
    }

    /// Take a checkpoint at the current position (LRU of the end-of-turn kind).
    pub fn checkpoint<E: Engine<Snapshot = S>>(&mut self, eng: &mut E) -> Result<()> {
        let clean = self.clean;
        self.checkpoint_with(eng, PRIO_TAIL, clean)
    }

    /// Take a checkpoint at the current position with an eviction priority. When full, the
    /// lowest-priority, least recently used checkpoint goes (possibly the new one's peer).
    pub fn checkpoint_with<E: Engine<Snapshot = S>>(&mut self, eng: &mut E, prio: u8, clean: bool) -> Result<()> {
        if self.max_snapshots == 0 || (prio == PRIO_TURN && self.max_turn == 0) {
            return Ok(());
        }
        let pos = self.tokens.len();
        if pos == 0 {
            return Ok(());
        }
        self.clock += 1;
        if let Some(s) = self.snaps.iter_mut().find(|s| s.pos == pos) {
            s.prio = s.prio.max(prio);
            s.used = self.clock;
            return Ok(());
        }
        // an evicted checkpoint's buffers are reused for the new one
        let mut spare: Option<S> = None;
        if prio == PRIO_TURN && self.snaps.iter().filter(|s| s.prio == PRIO_TURN).count() >= self.max_turn {
            let (i, _) = self.snaps.iter().enumerate().filter(|(_, s)| s.prio == PRIO_TURN).min_by_key(|(_, s)| s.used).unwrap();
            spare = Some(self.snaps.remove(i).s);
        }
        if self.snaps.len() >= self.max_snapshots {
            // never evict for a lower-priority snapshot than everything kept
            let (i, low) = self.snaps.iter().enumerate().min_by_key(|(_, s)| (s.prio, s.used)).map(|(i, s)| (i, s.prio)).unwrap();
            if low > prio {
                return Ok(());
            }
            spare = Some(self.snaps.remove(i).s);
        }
        let s = match spare {
            Some(old) => eng.snapshot_reuse(old)?,
            None => eng.snapshot()?,
        };
        let bytes = E::snapshot_bytes(&s);
        self.snaps.push(Snap { pos, s, used: self.clock, prio, clean, bytes });
        Ok(())
    }
}
