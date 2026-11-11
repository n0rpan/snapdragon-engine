//! The generation loop (llama-server `update_slots` + `process_token` for one slot):
//! prefix-cache restore, prompt prefill, sampling, EOG / stop-string / length stops,
//! UTF-8 hold-back, n_probs, timings.

use crate::cache::PrefixCache;
use crate::engine::{Engine, EngineError};
use crate::sampling::{token_prob, top_probs, Sampler, SamplingParams};
use std::time::Instant;
use tokenizer::{StreamDecoder, Token, Tokenizer};

#[derive(Clone, Debug)]
pub struct GenParams {
    /// -1 = until EOG or context full
    pub n_predict: i64,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    pub n_probs: usize,
    pub cache_prompt: bool,
    /// render special (control) tokens in the text
    pub special: bool,
    pub ignore_eos: bool,
    /// chat requests: token position of the first user message (the system prompt + tools
    /// end there): anchor checkpoint / prefix-store entry below it
    pub first_user: Option<usize>,
    /// per-request speculation depth (`"speculative": false` = 0, `"spec_draft_n_max": n`);
    /// None = the server's MTP tune table for this context, else the engine's depth
    pub spec_depth: Option<usize>,
    /// chat requests whose last message is a user message: query checkpoint before the
    /// generation prompt (see `cache.rs`)
    pub query_start: bool,
    /// chat requests: token position of every message start (turn checkpoints, WP-P1)
    pub msg_starts: Vec<usize>,
    /// set when the client went away (service: drop guard / closed stream); checked between
    /// prefill chunks and per decoded token
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl GenParams {
    pub fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
    }
}

/// Prefill calls are cut every this many ubatches (on the grid, so numerics are unchanged) to
/// check for a cancel: ~3 s at ~180 tok/s; one extra logits row per cut (<0.5 %).
pub const CANCEL_UBATCHES: usize = 2;

/// Ubatches per engine call for a prompt segment with `left` tokens to go: CANCEL_UBATCHES, or,
/// when the by-size prefill group has a second threshold T2 (`--pf-group-t2`, default never) and
/// at least T2 tokens are left, enough for the engine to see a call of T2 tokens (it then runs 4
/// ubatches per pass; the cancel check interval grows with it). Any multiple of the ubatch keeps
/// the numerics: the engine chunks every call by the ubatch from its start.
fn chunk_ubatches(left: usize, u: usize) -> usize {
    let (_, t2) = model::engine::pf_group_thresholds();
    if model::engine::pf_group() == 0 && t2 != 0 && left >= t2 {
        t2.div_ceil(u.max(1)).max(4)
    } else {
        CANCEL_UBATCHES.max(1)
    }
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            n_predict: -1,
            sampling: SamplingParams::default(),
            stop: Vec::new(),
            n_probs: 0,
            cache_prompt: true,
            special: false,
            ignore_eos: false,
            first_user: None,
            query_start: false,
            spec_depth: None,
            msg_starts: Vec::new(),
            cancel: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopType {
    None,
    Eos,
    Word,
    Limit,
}

impl StopType {
    pub fn as_str(self) -> &'static str {
        match self {
            StopType::Eos => "eos",
            StopType::Word => "word",
            StopType::Limit => "limit",
            StopType::None => "none",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProbEntry {
    pub id: Token,
    pub piece: Vec<u8>,
    pub prob: f32,
}

#[derive(Clone, Debug)]
pub struct TokenProbs {
    pub id: Token,
    pub piece: Vec<u8>,
    pub prob: f32,
    pub top: Vec<ProbEntry>,
}

/// One streamed step.
#[derive(Debug)]
pub struct Partial {
    pub text: String,
    pub token: Token,
    pub n_decoded: usize,
    pub probs: Option<TokenProbs>,
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub cache_n: usize,
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    /// speculative decoding (MTP): draft tokens proposed / accepted in this request
    pub draft_n: u64,
    pub draft_n_accepted: u64,
}

#[derive(Debug)]
pub struct GenResult {
    /// full generated text (stop word removed)
    pub text: String,
    pub tokens: Vec<Token>,
    pub stop: StopType,
    pub stopping_word: String,
    pub truncated: bool,
    pub n_prompt: usize,
    pub n_decoded: usize,
    pub timings: Timings,
    pub probs: Vec<TokenProbs>,
}

#[derive(Debug)]
pub enum GenError {
    /// prompt does not fit: (n_prompt, n_ctx)
    ExceedContext(usize, usize),
    Engine(EngineError),
    Cancelled,
}

impl From<EngineError> for GenError {
    fn from(e: EngineError) -> Self {
        GenError::Engine(e)
    }
}

/// Owns the engine and its prefix cache; one generation at a time.
pub struct Worker<E: Engine> {
    pub engine: E,
    pub cache: PrefixCache<E::Snapshot>,
    /// the engine's prefill ubatch (checkpoint grid; `--ubatch-size`)
    pub ubatch: usize,
    /// persistent prefix store (`--prefix-cache DIR`)
    pub store: Option<crate::prefix_store::PrefixStore>,
    /// turn checkpoints taken per request at most (the newest message starts win)
    pub turn_checkpoints: usize,
    /// on-disk session cache (`--session-cache DIR`): whole conversations, opt-in
    pub sessions: Option<crate::prefix_store::PrefixStore>,
    /// request ledger (`--ledger PATH`): one JSON line of numbers per generation
    pub ledger: Option<std::fs::File>,
    /// MTP auto-tune table (`--mtp-tune FILE`, tools/mtp_tune.py): depth per context bucket
    pub spec_table: Option<SpecTable>,
    /// F82: no speculation at or beyond this context position (`--mtp-max-ctx N`); combined with
    /// the table (the smaller depth wins: 0 here)
    pub mtp_max_ctx: Option<usize>,
    /// prompt-lookup drafting (`--spec-type lookup`, WP-D2): greedy requests without n_probs on an
    /// engine without MTP verify the drafter's proposals (output identical to plain decoding)
    pub lookup: Option<crate::lookup::LookupConfig>,
    /// repetition-loop watchdog (`--repeat-watchdog N`, 0 = off): stop with finish_reason
    /// "length" when the last N generated tokens end in a loop (see [`repeat_loop`])
    pub repeat_watchdog: usize,
    /// Message-boundary prefill schedule (`--prefill-split messages`, the Flash default): the
    /// prefill is split at every message start of the prompt instead of on the ubatch grid
    /// (see [`Worker::plan_messages`]).
    pub msg_split: bool,
    /// warm-up hint file (`--flash-hotset`), see [`WarmFile`]
    pub warm: Option<WarmFile>,
}

/// MTP auto-tune result: the measured best depth per context bucket (0 = speculation off there,
/// a valid result: on a MoE speculation can lose). JSON: {"buckets":[{"max_ctx":N,"depth":d},..]},
/// written by tools/mtp_tune.py from `oxe bench` arms on this machine.
#[derive(Clone, Debug, Default)]
pub struct SpecTable {
    /// (context positions up to, depth), ascending; contexts beyond the last bucket use the last
    pub buckets: Vec<(usize, usize)>,
}

impl SpecTable {
    pub fn load(path: &std::path::Path) -> Result<SpecTable, String> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut b: Vec<(usize, usize)> = j["buckets"]
            .as_array()
            .ok_or("mtp tune table: no buckets")?
            .iter()
            .map(|x| Some((x["max_ctx"].as_u64()? as usize, x["depth"].as_u64()? as usize)))
            .collect::<Option<_>>()
            .ok_or("mtp tune table: bucket needs max_ctx and depth")?;
        b.sort();
        if b.is_empty() {
            return Err("mtp tune table: empty".into());
        }
        Ok(SpecTable { buckets: b })
    }
    pub fn depth_for(&self, n_past: usize) -> usize {
        self.buckets.iter().find(|&&(m, _)| n_past <= m).or(self.buckets.last()).map(|b| b.1).unwrap_or(0)
    }
}

/// The engine's warm-up hint ([`Engine::warm_hint`]) on disk: magic `OXEWRM01`, u32 identity
/// length, identity (the model), the hint. Written to `<path>.pending` and renamed; mode 0600.
/// It names cache contents (Flash: expert ids), never prompt or generated text.
pub struct WarmFile {
    pub path: std::path::PathBuf,
    pub identity: String,
    /// seconds between writes after requests (a forced write ignores it)
    pub min_interval_s: u64,
    last: Option<Instant>,
}

const WARM_MAGIC: &[u8; 8] = b"OXEWRM01";

impl WarmFile {
    pub fn new(path: std::path::PathBuf, identity: String) -> WarmFile {
        WarmFile { path, identity, min_interval_s: 60, last: None }
    }

    /// The stored hint, if the file belongs to this identity.
    pub fn read(&self) -> Option<Vec<u8>> {
        let b = std::fs::read(&self.path).ok()?;
        let id = self.identity.as_bytes();
        let head = 12 + id.len();
        if b.len() < head || &b[..8] != WARM_MAGIC || u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize != id.len() || &b[12..head] != id {
            return None;
        }
        Some(b[head..].to_vec())
    }

    fn write(&mut self, hint: &[u8]) -> std::io::Result<()> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        if let Some(d) = self.path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let mut pending = self.path.clone().into_os_string();
        pending.push(".pending");
        let mut f = std::fs::File::options().write(true).create(true).truncate(true).mode(0o600).open(&pending)?;
        f.write_all(WARM_MAGIC)?;
        f.write_all(&(self.identity.len() as u32).to_le_bytes())?;
        f.write_all(self.identity.as_bytes())?;
        f.write_all(hint)?;
        f.sync_all()?;
        std::fs::rename(&pending, &self.path)?;
        self.last = Some(Instant::now());
        Ok(())
    }
}

/// The period of a loop at the end of `toks` within the last `window` tokens: some period L
/// (1 <= L <= window/4) whose last 4 repetitions are identical (4L tokens), and at least 16
/// tokens long in total, so short legitimate repeats (`}}}}`, a blank line) do not trigger.
pub fn repeat_loop(toks: &[u32], window: usize) -> Option<usize> {
    let n = toks.len();
    if window < 16 || n < 16 {
        return None;
    }
    for l in 1..=window / 4 {
        let span = (4 * l).max(16);
        if span > window || span > n {
            break;
        }
        let tail = &toks[n - span..];
        if tail.iter().enumerate().all(|(i, &t)| i < l || t == tail[i - l]) {
            return Some(l);
        }
    }
    None
}

/// Deepest grid position `base + k*step` that is `<= bound` and `> lo`.
fn grid_floor(base: usize, step: usize, lo: usize, bound: usize) -> Option<usize> {
    if bound <= lo || bound < base {
        return None;
    }
    let p = base + (bound - base) / step * step;
    (p > lo).then_some(p)
}

fn find_stop_full(text: &str, from: usize, stops: &[String]) -> Option<(usize, String)> {
    let mut best: Option<(usize, String)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(i) = text[from..].find(s.as_str()) {
            let p = from + i;
            if best.as_ref().map(|b| p < b.0).unwrap_or(true) {
                best = Some((p, s.clone()));
            }
        }
    }
    best
}

/// What the decode loop holds for the next position: the logits row, or (greedy fast path)
/// only its argmax.
enum NextRow {
    Logits(Vec<f32>),
    Argmax(u32),
}

/// OXE_TIMING_DIAG per-request split of the decode loop (ms).
#[derive(Default)]
struct DiagSplit {
    /// engine calls before the warm-up cut (`oxe bench --warmup`, OXE_TIMING_DIAG_WARMUP, default 16)
    warmup: usize,
    /// after the cut: engine ms, engine calls, tokens, and the loop's start time
    engine_ms_warm: f64,
    n_engine_warm: usize,
    warm_at: Option<(Instant, usize)>,
    engine_ms: f64,
    copy_ms: f64,
    sample_ms: f64,
    probs_ms: f64,
    text_ms: f64,
    emit_ms: f64,
    n_engine: usize,
    n_emit: usize,
}

impl DiagSplit {
    fn engine(&mut self, ms: f64, n_decoded: usize) {
        self.engine_ms += ms;
        self.n_engine += 1;
        if self.n_engine > self.warmup {
            self.engine_ms_warm += ms;
            self.n_engine_warm += 1;
        }
        if self.n_engine == self.warmup {
            self.warm_at = Some((Instant::now(), n_decoded));
        }
    }
}

/// ms since `*t`, and restart `*t`.
fn lap(t: &mut Instant) -> f64 {
    let now = Instant::now();
    let d = now.duration_since(*t).as_secs_f64() * 1e3;
    *t = now;
    d
}

/// does `text` end with a proper prefix of a stop string?
fn ends_with_partial_stop(text: &str, stops: &[String]) -> bool {
    stops.iter().any(|s| {
        let sb = s.as_bytes();
        let tb = text.as_bytes();
        (1..sb.len()).any(|k| k <= tb.len() && tb.ends_with(&sb[..k]))
    })
}

impl<E: Engine> Worker<E> {
    pub fn new(engine: E, max_snapshots: usize) -> Self {
        // turn checkpoints use only the slots beyond the 4 anchor/query/tail checkpoints need
        let turns = 4.min(max_snapshots.saturating_sub(4));
        let mut cache = PrefixCache::new(max_snapshots);
        cache.max_turn = turns;
        Worker { engine, cache, ubatch: model::engine::UBATCH, store: None, turn_checkpoints: turns, sessions: None, ledger: None, repeat_watchdog: 0, spec_table: None, mtp_max_ctx: None, lookup: None, msg_split: false, warm: None }
    }

    /// Process start: hand the stored warm-up hint to the engine (it queues the reads and
    /// returns). Returns the engine's log line, None without a usable file.
    pub fn load_warm(&mut self) -> Option<String> {
        let hint = self.warm.as_ref()?.read()?;
        match self.engine.apply_warm_hint(&hint) {
            Ok(line) => Some(line),
            Err(e) => {
                eprintln!("oxe-server: warm file ignored: {}", e.0);
                None
            }
        }
    }

    /// Write the engine's warm-up hint (after a request: at most every `min_interval_s`;
    /// `force`: shutdown). Returns whether a file was written.
    pub fn save_warm(&mut self, force: bool) -> bool {
        let Some(wf) = &mut self.warm else { return false };
        if !force && wf.last.is_some_and(|t| t.elapsed().as_secs() < wf.min_interval_s) {
            return false;
        }
        let Some(hint) = self.engine.warm_hint() else { return false };
        match wf.write(&hint) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("oxe-server: warm file write failed: {e}");
                false
            }
        }
    }

    /// Message-boundary schedule (`msg_split`): where the prefill of `prompt[n_cached..n-1]`
    /// is split and which splits get a checkpoint.
    ///
    /// The splits are every message start of the prompt (`msg_starts` and `first_user`), a
    /// function of the prompt alone: a cold server, a resident one resuming at the anchor and
    /// a process that loaded the anchor from the prefix store all evaluate the same ubatches
    /// (each message in ubatch-sized chunks from its start), so they compute the same bits.
    /// The anchor is exactly the first user message (not its ubatch-grid floor, which with
    /// Flash's 4096-token ubatch would be 0 for most system prompts), the query checkpoint is
    /// the generation prompt's message start, turn checkpoints are the newest
    /// `turn_checkpoints` other message starts. Returns (canonical, splits): `canonical` =
    /// the state at `n_cached` is itself on this schedule (position 0 or a message start of
    /// a clean state), so everything computed from it is too.
    fn plan_messages(&self, p: &GenParams, n_cached: usize, n: usize) -> (bool, Vec<(usize, Option<u8>)>) {
        let mut b: Vec<usize> = p.msg_starts.iter().copied().chain(p.first_user).filter(|&x| x > 0 && x < n - 1).collect();
        b.sort_unstable();
        b.dedup();
        let canonical = self.cache.clean && (n_cached == 0 || b.binary_search(&n_cached).is_ok());
        let query = if p.query_start { b.last().copied() } else { None };
        let new: Vec<usize> = b.into_iter().filter(|&x| x > n_cached).collect();
        let is_turn = |x: usize| Some(x) != p.first_user && Some(x) != query;
        let n_turn = new.iter().filter(|&&x| is_turn(x)).count();
        let mut skip = n_turn.saturating_sub(self.turn_checkpoints);
        let splits = new
            .into_iter()
            .map(|x| {
                let prio = if Some(x) == p.first_user {
                    Some(crate::cache::PRIO_ANCHOR)
                } else if Some(x) == query {
                    Some(crate::cache::PRIO_QUERY)
                } else if skip > 0 {
                    skip -= 1;
                    None
                } else {
                    Some(crate::cache::PRIO_TURN)
                };
                (x, prio)
            })
            .collect();
        (canonical, splits)
    }

    /// Ubatch-grid schedule (the default): checkpoints only at grid positions, so the split
    /// changes no ubatch boundary (see cache.rs).
    fn plan_grid(&self, p: &GenParams, n_cached: usize, n: usize) -> (bool, Vec<(usize, Option<u8>)>) {
        let u = self.ubatch.max(1);
        let aligned = self.cache.clean && n_cached % u == 0;
        let base = if aligned { 0 } else { n_cached };
        let anchor = p.first_user.and_then(|f| grid_floor(base, u, n_cached, f.min(n - 1)));
        let query = if p.query_start { grid_floor(base, u, n_cached, n.saturating_sub(4).min(n - 1)) } else { None };
        // turn checkpoints: grid floors of the message starts after the anchor, newest first
        let mut turns: Vec<usize> = p.msg_starts.iter().filter(|&&m| p.first_user.map_or(true, |f| m > f)).filter_map(|&m| grid_floor(base, u, n_cached, m.min(n - 1))).collect();
        turns.sort_unstable();
        turns.dedup();
        let skip = turns.len().saturating_sub(self.turn_checkpoints);
        let mut splits: Vec<(usize, u8)> = anchor
            .map(|a| (a, crate::cache::PRIO_ANCHOR))
            .into_iter()
            .chain(query.map(|q| (q, crate::cache::PRIO_QUERY)))
            .chain(turns[skip..].iter().map(|&t| (t, crate::cache::PRIO_TURN)))
            .filter(|&(x, _)| x < n - 1)
            .collect();
        splits.sort();
        splits.dedup_by(|b, a| {
            if a.0 == b.0 {
                a.1 = a.1.max(b.1);
                true
            } else {
                false
            }
        });
        (aligned, splits.into_iter().map(|(x, pr)| (x, Some(pr))).collect())
    }

    /// Load the longest entry of the session cache (`sessions`) or the prefix store that
    /// beats `n_cached`; returns the new n_cached.
    fn load_stored(&mut self, sessions: bool, prompt: &[Token], n_cached: usize) -> usize {
        let st = if sessions { self.sessions.as_ref() } else { self.store.as_ref() };
        let Some(st) = st else { return n_cached };
        let Some(n) = st.best(prompt, prompt.len() - 1).filter(|&n| n > n_cached) else { return n_cached };
        let what = if sessions { "session cache" } else { "prefix store" };
        match st.load(&mut self.engine, prompt, n) {
            Ok(l) => {
                eprintln!("oxe-server: {what}: loaded {} tokens in {:.0} ms", l.n, l.ms);
                self.cache.last_src = if sessions { "session" } else { "store" };
                // prefix-store entries are on the ubatch grid of a cold prefill; sessions are not
                self.cache.adopt(&prompt[..n], !sessions);
                n
            }
            Err(e) => {
                eprintln!("oxe-server: {e}");
                self.cache.reset(&mut self.engine);
                0
            }
        }
    }

    /// Prefill `seg` (which starts on the checkpoint grid `base + k*u`) in calls cut at grid
    /// points every CANCEL_UBATCHES ubatches, checking for a cancel before each call. The
    /// engine chunks each call by `u` from its start, so the ubatch boundaries, and therefore
    /// the numerics, are those of one unsplit call. On a cancel the engine and the cache hold
    /// a consistent prefix of the prompt (a retry resumes from it).
    fn prefill_chunks(&mut self, seg: &[Token], base: usize, u: usize, p: &GenParams) -> Result<(), GenError> {
        let mut i = 0;
        while i < seg.len() {
            if p.cancelled() {
                return Err(GenError::Cancelled);
            }
            let step = u * chunk_ubatches(seg.len() - i, u);
            let cur = self.cache.tokens.len();
            let next = base + ((cur.saturating_sub(base)) / step + 1) * step;
            let take = (next - cur).min(seg.len() - i);
            let chunk = &seg[i..i + take];
            // nobody reads these logits: the last prompt token is evaluated on its own below
            match self.engine.prefill_quiet(chunk) {
                Ok(()) => {}
                // F4: the engine stopped inside this call and rolled back to its start: nothing of the chunk is in
                // the state, so the cache (not yet advanced) and the engine agree
                Err(e) if e.0.starts_with(crate::engine::CANCELLED_MARK) => return Err(GenError::Cancelled),
                Err(e) => return Err(e.into()),
            }
            self.cache.appended(chunk);
            // everything in the state is prompt so far: a cancel here leaves a session that
            // parking keeps (P1), so a retry after another request still resumes it
            self.cache.prompt_len = self.cache.tokens.len();
            i += take;
        }
        Ok(())
    }

    /// Write parked sessions evicted from RAM to the session cache.
    fn spill_sessions(&mut self) {
        let out = std::mem::take(&mut self.cache.spilled);
        if let Some(st) = &mut self.sessions {
            for (t, b) in out {
                if !st.contains(&t) {
                    st.save_async(t, b);
                }
            }
        }
    }

    /// Shutdown (SIGTERM: unload, model switch): write the live session and every parked one
    /// to the session cache and wait for the writes. Besides each session's full state, its
    /// deepest query and turn checkpoints are written as entries of their own (state at that
    /// position, made by the engine: load/restore + save_state), so a prompt that diverges
    /// inside the session (a new user message re-renders the previous query) still resumes
    /// after a reload. Writes one entry at a time (bounded memory). Leaves the engine reset.
    /// Returns the number of entries written.
    pub fn persist_sessions(&mut self) -> usize {
        if self.sessions.is_none() {
            return 0;
        }
        let min = self.sessions.as_ref().unwrap().min_tokens;
        let mut n = 0;
        let mut put = |w: &mut Self, t: Vec<u32>, b: Vec<u8>| {
            let st = w.sessions.as_mut().unwrap();
            if t.len() >= min && !st.contains(&t) {
                st.save_async(t, b);
                st.flush();
                n += 1;
            }
        };
        let err = |e: crate::engine::EngineError| eprintln!("oxe-server: session cache: {}", e.0);
        // the live session: full state, then its checkpoints from the engine as it is
        if self.cache.prompt_len >= self.cache.park_min.max(1) {
            let tokens = self.cache.tokens.clone();
            let points = self.cache.take_live_points();
            let mut b = Vec::with_capacity(self.engine.state_size_hint());
            match self.engine.save_state(&mut b) {
                Ok(()) => put(self, tokens.clone(), b),
                Err(e) => err(e),
            }
            for (pos, s) in points {
                if pos < min {
                    continue;
                }
                let mut b = Vec::new();
                match self.engine.restore(&s).and_then(|_| self.engine.save_state(&mut b)) {
                    Ok(()) => put(self, tokens[..pos].to_vec(), b),
                    Err(e) => err(e),
                }
            }
        }
        // parked sessions: load each, then its checkpoints
        for (tokens, state, points) in self.cache.drain_parked() {
            let want: Vec<_> = points.into_iter().filter(|p| p.0 >= min).collect();
            if !want.is_empty() {
                match self.engine.load_state(&mut &state[..], tokens.len()) {
                    Ok(()) => {
                        for (pos, s) in want {
                            let mut b = Vec::new();
                            match self.engine.restore(&s).and_then(|_| self.engine.save_state(&mut b)) {
                                Ok(()) => put(self, tokens[..pos].to_vec(), b),
                                Err(e) => err(e),
                            }
                        }
                    }
                    Err(e) => err(e),
                }
            }
            put(self, tokens, state);
        }
        for (t, b) in std::mem::take(&mut self.cache.spilled) {
            put(self, t, b);
        }
        self.cache.reset(&mut self.engine);
        n
    }

    /// Bring the engine to `prompt[..n-1]` (prefix cache, disk stores, grid checkpoints, cancel
    /// checks); returns n_cached. The caller evaluates the last token.
    fn prefill_head(&mut self, prompt: &[Token], p: &GenParams) -> Result<usize, GenError> {
        let mut n_cached = self.cache.prepare(&mut self.engine, prompt, p.cache_prompt)?;
        // on-disk session cache, then the persistent prefix store: a longer stored prefix
        // than the in-memory cache offers
        if p.cache_prompt {
            n_cached = self.load_stored(true, prompt, n_cached);
            n_cached = self.load_stored(false, prompt, n_cached);
            self.spill_sessions();
        }
        let n = prompt.len();
        // checkpoint plan for the head prompt[n_cached..n-1] (see cache.rs)
        let (aligned, splits) = if self.msg_split { self.plan_messages(p, n_cached, n) } else { self.plan_grid(p, n_cached, n) };
        // cancel cuts (prefill_chunks): on the grid schedule the checkpoint grid `base + k*u`; on the
        // message schedule each segment's own start (segments start at message starts and the
        // engine chunks a call by `u` from its start), so a cut never moves a ubatch boundary
        let u = self.ubatch.max(1);
        let grid_base = if self.cache.clean && n_cached % u == 0 { 0 } else { n_cached };
        let msg_split = self.msg_split;
        for (pos, prio) in splits {
            let start = self.cache.tokens.len();
            let seg = &prompt[start..pos];
            if !seg.is_empty() {
                self.prefill_chunks(seg, if msg_split { start } else { grid_base }, u, p)?;
            }
            self.cache.clean = aligned;
            let Some(prio) = prio else { continue };
            self.cache.checkpoint_with(&mut self.engine, prio, aligned)?;
            if prio == crate::cache::PRIO_ANCHOR && aligned {
                if let Some(st) = &mut self.store {
                    if pos >= st.min_tokens && !st.contains(&prompt[..pos]) {
                        let mut buf = Vec::new();
                        match self.engine.save_state(&mut buf) {
                            Ok(()) => st.save_async(prompt[..pos].to_vec(), buf),
                            Err(e) => eprintln!("oxe-server: prefix store: {}", e.0),
                        }
                    }
                }
            }
        }
        // evaluate up to the last prompt token, checkpoint at len-1, then the last token
        let start = self.cache.tokens.len();
        let head = &prompt[start..n - 1];
        if !head.is_empty() {
            self.prefill_chunks(head, if msg_split { start } else { grid_base }, u, p)?;
            self.cache.clean = aligned;
            self.cache.checkpoint_with(&mut self.engine, crate::cache::PRIO_TAIL, aligned)?;
        }
        Ok(n_cached)
    }

    /// SX-2 idle re-prefill: prefill `hist` (the conversation in the form the next request will
    /// render it, cut before the next user message) while no request is running, up to the
    /// deepest ubatch-grid point of the request's own prefill (restore point + k*ubatch) at or
    /// below its end, where a grid checkpoint is taken. The next request continues from there
    /// with exactly the ubatch boundaries an on-demand prefill would use, so its state is
    /// bit-identical to the no-idle path (lossless); at most ubatch-1 tokens stay on the
    /// critical path. The raw session is parked by `prepare` as usual (a retry still finds it).
    /// Stops at a chunk boundary when `p.cancel` is raised (a request arrived). Returns the
    /// tokens newly evaluated.
    pub fn prefill_idle(&mut self, hist: &[Token], p: &GenParams) -> Result<usize, GenError> {
        // grid schedule only: the message schedule (Flash) splits at message starts, not on this lattice
        if self.msg_split || hist.len() < 2 || hist.len() >= self.engine.n_ctx() {
            return Ok(0);
        }
        let mut n0 = self.cache.prepare(&mut self.engine, hist, p.cache_prompt)?;
        if p.cache_prompt {
            n0 = self.load_stored(true, hist, n0);
            n0 = self.load_stored(false, hist, n0);
        }
        let u = self.ubatch.max(1);
        // the lattice the request would prefill on: from the restore point (or 0 when aligned)
        let base = if self.cache.clean && n0 % u == 0 { 0 } else { n0 };
        let g = base + (hist.len() - base) / u * u;
        if g <= n0 + u {
            return Ok(0); // less than one ubatch to gain
        }
        // prefill_head stops at its prompt's len-1 and checkpoints there (a grid point)
        let n_cached = self.prefill_head(&hist[..g + 1], p)?;
        Ok(g - n_cached)
    }

    /// Run one generation. `on_partial` returns false to cancel (client went away).
    ///
    /// An engine error (a failed prefill, decode, checkpoint or state load) can leave the engine
    /// ahead of the cache's token list: the failing call may have advanced its position over a
    /// chunk whose results are invalid (Flash: an expert read that failed twice or timed out).
    /// The state is then dropped (engine and cache reset; parked sessions stay), so a retry of
    /// the same prompt starts from an empty engine instead of continuing on the invalid chunk.
    /// A cancel is not an error: it leaves a consistent prefix (see `prefill_chunks`).
    pub fn generate(
        &mut self,
        tok: &Tokenizer,
        prompt: &[Token],
        p: &GenParams,
        on_partial: &mut dyn FnMut(Partial) -> bool,
    ) -> Result<GenResult, GenError> {
        let n_ctx = self.engine.n_ctx();
        if prompt.is_empty() {
            return Err(GenError::Engine(EngineError("empty prompt".into())));
        }
        if prompt.len() >= n_ctx {
            return Err(GenError::ExceedContext(prompt.len(), n_ctx));
        }
        // F4: the engine may stop inside a prefill call on this flag; cleared again whatever happens
        self.engine.set_cancel(p.cancel.clone());
        let r = self.generate_inner(tok, prompt, p, on_partial);
        self.engine.set_cancel(None);
        if let Err(GenError::Engine(e)) = &r {
            eprintln!("oxe-server: engine error, sequence state dropped ({} tokens): {}", self.cache.tokens.len(), e.0);
            self.cache.reset(&mut self.engine);
        }
        r
    }

    fn generate_inner(
        &mut self,
        tok: &Tokenizer,
        prompt: &[Token],
        p: &GenParams,
        on_partial: &mut dyn FnMut(Partial) -> bool,
    ) -> Result<GenResult, GenError> {
        let n_ctx = self.engine.n_ctx();
        let t0 = Instant::now();
        let stats0 = self.cache.stats.clone();
        let n_cached = self.prefill_head(prompt, p)?;
        let n = prompt.len();
        let last = &prompt[n - 1..];
        let mut logits = self.engine.prefill(last)?.to_vec();
        self.cache.appended(last);
        self.cache.prompt_len = n;
        // a single token goes through the decode step, not the batched prefill
        self.cache.clean = false;
        let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        let spec0 = self.engine.spec_counters();
        // MTP speculation (greedy only): rows the engine already computed for tokens it
        // appended speculatively, consumed one by one as the loop reaches them
        let greedy = p.sampling.temperature <= 0.0;
        // greedy fast path: nothing reads the logits row (no n_probs), so take the engine's own
        // argmax (same lowest-id tie-break as sampling::argmax) instead of copying the row
        // and scanning it again
        // OXE_SERVER_LEGACY=1 (A/B diagnostic only): the pre-fast-path loop (row copy + host argmax)
        let legacy = std::env::var_os("OXE_SERVER_LEGACY").is_some();
        let fast = greedy && p.n_probs == 0 && !legacy;
        // speculation depth for this request (MTP auto-tune): request override, else the tune
        // table's depth for the context this request decodes at, else the engine's depth.
        // 0 = no speculation (the MTP state still follows every token through `decode`).
        // F82 context-adaptive MTP: the depth is decided at every new verify from the context the
        // verify starts at (request override, else the tune table's bucket, else the engine's depth;
        // 0 at or beyond --mtp-max-ctx). Drafts only: the output is identical at any depth, and with
        // depth 0 the token goes through `decode`, which keeps the MTP state following every token,
        // so speculation can switch off and on inside one reply.
        let max_d = self.engine.spec_max_depth();
        let spec_table = self.spec_table.clone();
        let mtp_max_ctx = self.mtp_max_ctx;
        let depth_at = |n_past: usize| -> usize {
            if mtp_max_ctx.is_some_and(|m| n_past >= m) {
                return 0;
            }
            p.spec_depth.or_else(|| spec_table.as_ref().map(|t| t.depth_for(n_past))).unwrap_or(max_d).min(max_d)
        };
        let mut cur_depth = usize::MAX; // depth last set on the engine
        let mut next_tok: Option<u32> = None;
        let mut spec_q: std::collections::VecDeque<(u32, NextRow)> = Default::default();
        let mut spec_used = 0usize;
        let mut sampler = Sampler::new(p.sampling.clone());
        let mut dec = StreamDecoder::default();
        let mut generated = String::new();
        let mut n_sent = 0usize;
        let mut tokens = Vec::new();
        // prompt lookup (fast greedy path only; a per-request "speculative": false turns it off too)
        let mut drafter = self.lookup.clone().filter(|_| fast && p.spec_depth != Some(0)).map(|c| {
            let mut d = crate::lookup::LookupDrafter::new(c);
            d.push(prompt);
            d
        });
        let mut probs_all = Vec::new();
        let mut stop = StopType::None;
        let mut stopping_word = String::new();
        let mut truncated = false;
        let mut n_decoded = 0usize;
        // OXE_TIMING_DIAG=1: per-request split of the decode loop (ms): engine calls, logits
        // copies, sampler, text (piece, UTF-8, stop strings), emit (on_partial: JSON + SSE send)
        let diag = std::env::var_os("OXE_TIMING_DIAG").is_some();
        let mut dg = DiagSplit {
            warmup: std::env::var("OXE_TIMING_DIAG_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(16),
            ..Default::default()
        };
        let mut tk = Instant::now();
        loop {
            let t = match next_tok.take() {
                Some(t) => t,
                None => sampler.sample(&logits),
            };
            n_decoded += 1;
            if diag {
                dg.sample_ms += lap(&mut tk);
            }
            let probs = (p.n_probs > 0).then(|| TokenProbs {
                id: t,
                piece: tok.piece(t, p.special),
                prob: token_prob(&logits, t),
                top: top_probs(&logits, p.n_probs)
                    .into_iter()
                    .map(|(id, prob)| ProbEntry { id, piece: tok.piece(id, p.special), prob })
                    .collect(),
            });
            let is_eog = tok.is_eog(t) && !p.ignore_eos;
            tokens.push(t);
            if let Some(d) = drafter.as_mut() {
                d.push(&[t]);
            }
            if let Some(pp) = &probs {
                probs_all.push(pp.clone());
            }
            if diag {
                dg.probs_ms += lap(&mut tk);
            }
            // process_token: append text, stop strings, send the partial
            let piece = tok.piece(t, p.special);
            generated.push_str(&dec.push(&piece));
            let incomplete = dec.has_pending();
            let mut has_next = true;
            if !incomplete {
                let mut send = true;
                if let Some((pos, w)) = find_stop_full(&generated, n_sent.min(generated.len()), &p.stop) {
                    generated.truncate(pos);
                    stop = StopType::Word;
                    stopping_word = w;
                    has_next = false;
                } else if !is_eog && ends_with_partial_stop(&generated[n_sent..], &p.stop) {
                    send = false;
                }
                let text = if send {
                    let s = generated[n_sent.min(generated.len())..].to_string();
                    n_sent = generated.len();
                    s
                } else {
                    String::new()
                };
                if diag {
                    dg.text_ms += lap(&mut tk);
                }
                let go = on_partial(Partial { text, token: t, n_decoded, probs });
                if diag {
                    dg.emit_ms += lap(&mut tk);
                    dg.n_emit += 1;
                }
                if !go || p.cancelled() {
                    // keep the engine state equal to the cache's token list (MTP: drop the
                    // speculative tokens the client never received)
                    if !spec_q.is_empty() {
                        self.engine.spec_rewind(1 + spec_used)?;
                    }
                    return Err(GenError::Cancelled);
                }
            }
            if has_next && self.engine.n_past() - spec_q.len() + 1 >= n_ctx {
                stop = StopType::Limit;
                truncated = true;
                has_next = false;
            }
            if has_next && is_eog {
                stop = StopType::Eos;
                has_next = false;
            }
            if has_next && self.repeat_watchdog > 0 {
                if let Some(l) = repeat_loop(&tokens, self.repeat_watchdog) {
                    eprintln!("oxe-server: repetition watchdog: period-{l} loop over the last {} tokens at {n_decoded} generated; stopping (finish_reason length)", (4 * l).max(16));
                    stop = StopType::Limit;
                    has_next = false;
                }
            }
            if has_next && p.n_predict >= 0 && n_decoded as i64 >= p.n_predict {
                stop = StopType::Limit;
                has_next = false;
            }
            if !has_next {
                break;
            }
            if diag {
                dg.text_ms += lap(&mut tk);
            }
            match spec_q.pop_front() {
                Some((want, row)) if want == t => {
                    spec_used += 1;
                    match row {
                        NextRow::Logits(r) => logits = r,
                        NextRow::Argmax(g) => next_tok = Some(g),
                    }
                }
                popped => {
                    if popped.is_some() {
                        // the sampler left the greedy path: drop the unused speculation
                        self.engine.spec_rewind(1 + spec_used)?;
                        spec_q.clear();
                    }
                    let depth = depth_at(self.engine.n_past());
                    if depth > 0 && depth != cur_depth {
                        self.engine.spec_set_depth(depth);
                        cur_depth = depth;
                    }
                    let spec = if greedy && popped.is_none() && depth > 0 {
                        if fast {
                            self.engine.decode_spec_argmax(t)?.map(|(e, g)| (e, g.into_iter().map(NextRow::Argmax).collect::<Vec<_>>()))
                        } else {
                            self.engine.decode_spec(t)?.map(|(e, r)| (e, r.into_iter().map(NextRow::Logits).collect::<Vec<_>>()))
                        }
                    } else {
                        None
                    };
                    // prompt lookup when the engine has no MTP speculation (decode_spec gave None)
                    let spec = if spec.is_some() {
                        spec
                    } else if let (Some(dr), true) = (drafter.as_mut(), popped.is_none()) {
                        let ds = dr.draft();
                        if ds.is_empty() {
                            None
                        } else {
                            let r = self.engine.decode_verify(t, &ds)?;
                            if let Some((e, _)) = &r {
                                dr.feedback(ds.len(), e.len());
                            }
                            r.map(|(e, g)| (e, g.into_iter().map(NextRow::Argmax).collect::<Vec<_>>()))
                        }
                    } else {
                        None
                    };
                    match spec {
                        Some((extra, rows)) => {
                            let mut rows = rows.into_iter();
                            match rows.next().expect("decode_spec returned no rows") {
                                NextRow::Logits(r) => logits = r,
                                NextRow::Argmax(g) => next_tok = Some(g),
                            }
                            spec_q = extra.into_iter().zip(rows).collect();
                            spec_used = 0;
                            if diag {
                                dg.engine(lap(&mut tk), n_decoded);
                            }
                        }
                        None if fast => {
                            next_tok = Some(self.engine.decode_argmax(t)?);
                            if diag {
                                dg.engine(lap(&mut tk), n_decoded);
                            }
                        }
                        None => {
                            let row = self.engine.decode(t)?;
                            if diag {
                                dg.engine(lap(&mut tk), n_decoded);
                            }
                            logits = row.to_vec();
                            if diag {
                                dg.copy_ms += lap(&mut tk);
                            }
                        }
                    }
                }
            }
            self.cache.appended(&[t]);
            if diag {
                dg.text_ms += lap(&mut tk);
            }
        }
        if !spec_q.is_empty() {
            self.engine.spec_rewind(1 + spec_used)?;
        }
        let spec1 = self.engine.spec_counters();
        let tail = dec.finish();
        if stop != StopType::Word {
            generated.push_str(&tail);
        }
        let loop_ms = t1.elapsed().as_secs_f64() * 1e3;
        // after the warm-up cut, like `oxe bench`: loop ms per engine call and per produced token
        let (loop_ms_warm, tokens_warm) = dg.warm_at.map(|(t, n)| (t.elapsed().as_secs_f64() * 1e3, n_decoded - n)).unwrap_or((0.0, 0));
        // the state now holds prompt + all decoded tokens: checkpoint for the next turn
        let tc = Instant::now();
        self.cache.checkpoint(&mut self.engine)?;
        let checkpoint_ms = tc.elapsed().as_secs_f64() * 1e3;
        let predicted_ms = t1.elapsed().as_secs_f64() * 1e3;
        self.save_warm(false);
        if let Some(f) = &mut self.ledger {
            use std::io::Write;
            let s = &self.cache.stats;
            let line = serde_json::json!({
                "t": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0),
                "n_prompt": prompt.len(), "cache_n": n_cached, "cache_src": self.cache.last_src,
                "prompt_n": prompt.len() - n_cached, "prompt_ms": prompt_ms,
                "predicted_n": n_decoded, "predicted_ms": predicted_ms, "stop": stop.as_str(),
                "draft_n": spec1.0 - spec0.0, "draft_n_accepted": spec1.1 - spec0.1,
                "parks": s.parks - stats0.parks, "unparks": s.unparks - stats0.unparks,
                "park_ms": s.park_ms - stats0.park_ms, "unpark_ms": s.unpark_ms - stats0.unpark_ms,
                "park_evictions": s.park_evictions - stats0.park_evictions,
                "checkpoints": self.cache.snapshot_positions().len(), "parked": self.cache.parked_lengths().len(),
                "parked_mib": self.cache.parked_bytes() >> 20,
            });
            let _ = writeln!(f, "{line}");
        }
        if diag {
            eprintln!(
                "OXE_DIAG {}",
                serde_json::json!({"phase": "request", "prompt_n": prompt.len() - n_cached, "cache_n": n_cached,
                    "generated_n": n_decoded, "prompt_ms": prompt_ms, "decode_ms": predicted_ms, "loop_ms": loop_ms,
                    "checkpoint_ms": checkpoint_ms, "engine_ms": dg.engine_ms, "engine_calls": dg.n_engine,
                    "warmup_calls": dg.warmup, "loop_ms_warm": loop_ms_warm, "engine_ms_warm": dg.engine_ms_warm,
                    "engine_calls_warm": dg.n_engine_warm, "tokens_warm": tokens_warm, "legacy": legacy,
                    "copy_ms": dg.copy_ms, "sample_ms": dg.sample_ms, "probs_ms": dg.probs_ms, "text_ms": dg.text_ms,
                    "emit_ms": dg.emit_ms, "emits": dg.n_emit, "greedy": greedy,
                    "draft_n": self.engine.spec_counters().0 - spec0.0,
                    "request_ms": t0.elapsed().as_secs_f64() * 1e3})
            );
        }
        Ok(GenResult {
            text: generated,
            tokens,
            stop,
            stopping_word,
            truncated,
            n_prompt: prompt.len(),
            n_decoded,
            timings: Timings {
                cache_n: n_cached,
                prompt_n: prompt.len() - n_cached,
                prompt_ms,
                predicted_n: n_decoded,
                predicted_ms,
                draft_n: spec1.0 - spec0.0,
                draft_n_accepted: spec1.1 - spec0.1,
            },
            probs: probs_all,
        })
    }

}
