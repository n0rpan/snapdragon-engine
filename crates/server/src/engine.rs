//! The model interface the server needs. The real implementation will wrap `model`
//! (qwen35moe decode, M2) once it lands; until then [`StubEngine`] stands in.
//!
//! State model: the engine holds one sequence. Its attention KV covers positions
//! `0..n_past`; the recurrent (delta-net) state is that after token `n_past - 1`. Recurrent
//! state cannot be rewound, so going back to an earlier position is only possible through
//! a [`Engine::snapshot`] taken at that position. Restoring a snapshot taken at position
//! `p` is valid as long as KV positions `< p` were not rewritten since, which the server's
//! prefix cache guarantees (it drops snapshots past the point it rewinds to).

use std::fmt;

#[derive(Debug, Clone)]
pub struct EngineError(pub String);
impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for EngineError {}

pub type Result<T> = std::result::Result<T, EngineError>;

/// Prefix of the error an engine returns from a prefill call it aborted on the cancel flag (state rolled back to the
/// call's start); see `Engine::set_cancel`.
pub const CANCELLED_MARK: &str = "[cancelled]";

pub trait Engine: Send + 'static {
    /// Opaque saved state at one position (recurrent state + position; KV is kept in place).
    type Snapshot: Send + 'static;

    /// Maximum sequence length (KV slots).
    fn n_ctx(&self) -> usize;
    fn n_vocab(&self) -> usize;
    /// Number of tokens currently in the state.
    fn n_past(&self) -> usize;
    /// Drop all state (empty sequence).
    fn reset(&mut self);
    /// Append `tokens` (positions `n_past..`) and return the logits after the last one.
    /// `tokens` is never empty.
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]>;
    /// [`Engine::prefill`] for a segment whose logits the caller does not read (every prompt
    /// segment but the last token). Engines may skip the output head; the state afterwards is
    /// the same. Default: plain `prefill`.
    fn prefill_quiet(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill(tokens).map(|_| ())
    }
    /// F4: the request's cancel flag (None = no request running). An engine that can stop inside one prefill call
    /// (Flash) aborts it when the flag rises, rolls its state back to the call's start and returns an error whose
    /// text starts with [`CANCELLED_MARK`]; generate() then reports a cancel with engine and cache in step.
    fn set_cancel(&mut self, _flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {}
    /// W2 opt-in backend teardown while the worker owns its engine exclusively.
    /// Must complete before the process exits; ordinary engines own no devices.
    #[cfg(feature="npu-prefill")]
    fn release_accelerators(&mut self) {}
    /// Append one token and return the logits after it (the decode hot path).
    fn decode(&mut self, token: u32) -> Result<&[f32]> {
        self.prefill(&[token])
    }
    /// Save the state at the current position.
    fn snapshot(&mut self) -> Result<Self::Snapshot>;
    /// Go back to a saved position; `n_past` becomes the snapshot's position.
    fn restore(&mut self, snap: &Self::Snapshot) -> Result<()>;
    /// As [`Engine::snapshot`], reusing an evicted snapshot's buffers when the engine can.
    fn snapshot_reuse(&mut self, old: Self::Snapshot) -> Result<Self::Snapshot> {
        drop(old);
        self.snapshot()
    }
    /// Bytes [`Engine::save_state`] will write (capacity hint; 0 = unknown).
    fn state_size_hint(&self) -> usize {
        0
    }
    /// Memory held by a snapshot (parked-session budget).
    fn snapshot_bytes(_snap: &Self::Snapshot) -> usize {
        0
    }

    // ---- M3.1 speculative decoding hooks (MTP). Engines without speculation keep the
    // defaults and the server decodes one token at a time.

    /// Greedy speculation: append `token` and, when the engine can prove what greedy
    /// sampling picks next, more tokens. Returns `None` if unsupported, else
    /// `(extra, rows)`: `extra` are the additional tokens appended after `token` (each
    /// equal to the argmax of the row before it) and `rows[i]` the logits after the i-th
    /// appended token (`rows.len() == extra.len() + 1`).
    fn decode_spec(&mut self, _token: u32) -> Result<Option<(Vec<u32>, Vec<Vec<f32>>)>> {
        Ok(None)
    }
    /// [`Engine::decode`] for greedy requests that do not need the row: returns the argmax
    /// of the logits after `token` with the tie-break of [`crate::sampling::argmax`] (lowest
    /// id among equal maxima; NaN never wins). Engines whose forward already computes it
    /// override this to skip the row copy and the second scan.
    fn decode_argmax(&mut self, token: u32) -> Result<u32> {
        Ok(crate::sampling::argmax(self.decode(token)?))
    }
    /// [`Engine::decode_spec`] for the same requests: `(extra, argmax of each row)`.
    fn decode_spec_argmax(&mut self, token: u32) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        Ok(self.decode_spec(token)?.map(|(e, rows)| (e, rows.iter().map(|r| crate::sampling::argmax(r)).collect())))
    }
    /// Keep only the first `keep` (>= 1) tokens the last `decode_spec` appended (the
    /// server stopped early). Valid until the next engine call.
    fn spec_rewind(&mut self, _keep: usize) -> Result<()> {
        Ok(())
    }
    /// Verify caller-supplied drafts (prompt lookup): append `token`, then as many of `drafts` as
    /// greedy decoding would produce (the longest prefix where each draft equals the argmax of the
    /// row before it). Returns `(accepted drafts, argmax of every committed row)` or None when the
    /// engine cannot verify external drafts (then the caller decodes one token). `spec_rewind`
    /// keeps a prefix of what was appended, as after `decode_spec`.
    fn decode_verify(&mut self, _token: u32, _drafts: &[u32]) -> Result<Option<(Vec<u32>, Vec<u32>)>> {
        Ok(None)
    }
    /// The deepest speculation the engine was built for (0 = none). Default: no limit known
    /// (engines that only implement `decode_spec` decide themselves).
    fn spec_max_depth(&self) -> usize {
        usize::MAX
    }
    /// Draft depth for the following `decode_spec` calls (1..=spec_max_depth). Depth 0 is the
    /// caller's business: it simply does not call `decode_spec` (the MTP state stays in sync
    /// through `decode`).
    fn spec_set_depth(&mut self, _depth: usize) {}
    /// Cumulative (drafted, accepted) draft tokens (llama-server `draft_n` counters).
    fn spec_counters(&self) -> (u64, u64) {
        (0, 0)
    }

    // ---- persistent prefix store (crate::prefix_store). Engines without it keep the defaults.

    /// Layout identity of [`Engine::save_state`]'s bytes (None: no persistent state). Two
    /// engines with equal ids read each other's states.
    fn state_id(&self) -> Option<String> {
        None
    }
    /// Write the whole sequence state at the current position (KV, recurrent, MTP).
    fn save_state(&mut self, _w: &mut dyn std::io::Write) -> Result<()> {
        Err(EngineError("engine has no persistent state".into()))
    }
    /// The bytes of [`Engine::save_state`] as one buffer (the prefix store's anchor save). Engines with
    /// large states override it to build the buffer once (Flash: ~285 MB at 5.5K tokens).
    fn save_state_vec(&mut self) -> Result<Vec<u8>> {
        let mut b = Vec::with_capacity(self.state_size_hint());
        self.save_state(&mut b)?;
        Ok(b)
    }
    /// Read a state written by [`Engine::save_state`] at `n_pos` positions; `n_past`
    /// becomes `n_pos`. On error the caller resets the engine.
    fn load_state(&mut self, _r: &mut dyn std::io::Read, _n_pos: usize) -> Result<()> {
        Err(EngineError("engine has no persistent state".into()))
    }

    // ---- CACHE-PERFECT P2 segmented state (crate::state_store). Engines without it keep the
    // defaults (kv_row_bytes 0) and the state store stays off for them (Flash v1).

    /// Bytes [`Engine::kv_rows_save`] writes per position; 0 = no segmented state.
    fn kv_row_bytes(&self) -> usize {
        0
    }
    /// Write the attention KV rows (every attention layer, the MTP layer included) of positions
    /// [a, b), b <= n_past. Layout is the engine's; the same engine reads it back.
    fn kv_rows_save(&mut self, _a: usize, _b: usize, _w: &mut dyn std::io::Write) -> Result<()> {
        Err(EngineError("engine has no segmented state".into()))
    }
    /// Read rows of [`Engine::kv_rows_save`] into positions [a, b). `n_past` does not change; the
    /// caller loads rows from position 0 up and finishes with [`Engine::snap_load`].
    fn kv_rows_load(&mut self, _a: usize, _b: usize, _r: &mut dyn std::io::Read) -> Result<()> {
        Err(EngineError("engine has no segmented state".into()))
    }
    /// Write the non-KV part of a checkpoint (recurrent + conv state, MTP driver state) taken at
    /// some position p; its KV rows below p are written separately.
    fn snap_save(&mut self, _s: &Self::Snapshot, _w: &mut dyn std::io::Write) -> Result<()> {
        Err(EngineError("engine has no segmented state".into()))
    }
    /// [`Engine::snap_save`] of the LIVE state (as of [`Engine::snapshot`] now) without building
    /// a snapshot copy first (CACHE-PERFECT P2b, M2: the shutdown write's memory floor).
    fn snap_save_live(&mut self, w: &mut dyn std::io::Write) -> Result<()> {
        let s = self.snapshot()?;
        self.snap_save(&s, w)
    }
    /// Read a [`Engine::snap_save`] taken at `n_pos`, after the KV rows below `n_pos` were loaded:
    /// the engine is then exactly at that checkpoint and `n_past == n_pos`.
    fn snap_load(&mut self, _r: &mut dyn std::io::Read, _n_pos: usize) -> Result<()> {
        Err(EngineError("engine has no segmented state".into()))
    }
    /// Numerics-relevant settings beyond the state layout (attention path, tiling, fused kernels):
    /// part of the state-store identity (Codex P0 review: the EFFECTIVE arithmetic mode).
    fn arith_id(&self) -> String {
        String::new()
    }

    // ---- warm-up hint (`--flash-hotset`): engine caches worth refilling at process start.

    /// A small description of what the engine holds in its caches (Flash: the expert cache's
    /// (layer, expert) list). None: nothing to save.
    fn warm_hint(&mut self) -> Option<Vec<u8>> {
        None
    }
    /// Start refilling the caches from a hint of [`Engine::warm_hint`] (best effort, does not
    /// wait). Never changes the sequence state or any result. Returns a line for the log.
    fn apply_warm_hint(&mut self, _hint: &[u8]) -> Result<String> {
        Ok(String::new())
    }
}

/// A deterministic fake model over the real vocabulary, for server tests and dry runs.
///
/// It "generates" scripted token sequences: each request pops the next script (or repeats
/// the default one); after the script it emits `eos`. Logits are a one-hot-like vector
/// (`10.0` on the scripted token, `0.0` elsewhere, a second-best at `5.0` so n_probs
/// has something to show). It records every call so tests can check the prefix cache.
pub struct StubEngine {
    pub n_ctx: usize,
    pub n_vocab: usize,
    pub eos: u32,
    /// tokens in the "KV"
    pub tokens: Vec<u32>,
    pub scripts: std::collections::VecDeque<Vec<u32>>,
    pub default_script: Vec<u32>,
    cur_script: Vec<u32>,
    script_pos: usize,
    /// next prefill starts a new response (pops the next script)
    fresh: bool,
    logits: Vec<f32>,
    /// log of (kind, n_past before, n_tokens) for tests: kind = "prefill"|"decode"|"restore"|"reset"|"snapshot"
    pub calls: Vec<(&'static str, usize, usize)>,
    /// tokens processed in total by prefill/decode
    pub n_evaluated: usize,
}

/// Stub snapshot: the token prefix it was taken at (checked on restore).
pub struct StubSnapshot {
    tokens: Vec<u32>,
    script_pos: usize,
}

impl StubEngine {
    pub fn new(n_ctx: usize, n_vocab: usize, eos: u32, default_script: Vec<u32>) -> StubEngine {
        StubEngine {
            n_ctx,
            n_vocab,
            eos,
            tokens: Vec::new(),
            scripts: Default::default(),
            default_script,
            cur_script: Vec::new(),
            script_pos: 0,
            fresh: true,
            logits: vec![0.0; n_vocab],
            calls: Vec::new(),
            n_evaluated: 0,
        }
    }

    fn set_logits(&mut self, next: u32) {
        for x in self.logits.iter_mut() {
            *x = 0.0;
        }
        self.logits[next as usize] = 10.0;
        let second = if next as usize + 1 < self.n_vocab { next + 1 } else { 0 };
        if second != next {
            self.logits[second as usize] = 5.0;
        }
    }

    fn next_scripted(&self) -> u32 {
        self.cur_script.get(self.script_pos).copied().unwrap_or(self.eos)
    }
}

impl Engine for StubEngine {
    type Snapshot = StubSnapshot;

    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }
    fn n_past(&self) -> usize {
        self.tokens.len()
    }
    fn reset(&mut self) {
        self.calls.push(("reset", self.tokens.len(), 0));
        self.tokens.clear();
        self.fresh = true;
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Err(EngineError("empty prefill".into()));
        }
        if self.tokens.len() + tokens.len() > self.n_ctx {
            return Err(EngineError("context full".into()));
        }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.n_vocab) {
            return Err(EngineError(format!("token {t} out of vocab")));
        }
        self.calls.push(("prefill", self.tokens.len(), tokens.len()));
        self.n_evaluated += tokens.len();
        self.tokens.extend_from_slice(tokens);
        // a new prompt (first prefill after a reset/restore/decode): start the next script
        if self.fresh {
            self.cur_script = self.scripts.pop_front().unwrap_or_else(|| self.default_script.clone());
            self.script_pos = 0;
            self.fresh = false;
        }
        let n = self.next_scripted();
        self.set_logits(n);
        Ok(&self.logits)
    }
    fn decode(&mut self, token: u32) -> Result<&[f32]> {
        if self.tokens.len() + 1 > self.n_ctx {
            return Err(EngineError("context full".into()));
        }
        self.calls.push(("decode", self.tokens.len(), 1));
        self.n_evaluated += 1;
        self.tokens.push(token);
        self.fresh = true;
        if self.cur_script.get(self.script_pos) == Some(&token) {
            self.script_pos += 1;
        }
        let n = self.next_scripted();
        self.set_logits(n);
        Ok(&self.logits)
    }
    fn snapshot(&mut self) -> Result<StubSnapshot> {
        self.calls.push(("snapshot", self.tokens.len(), 0));
        Ok(StubSnapshot { tokens: self.tokens.clone(), script_pos: self.script_pos })
    }
    /// Exactly the bytes `save_state` writes (LONG-CTX disk-park threshold tests).
    fn state_size_hint(&self) -> usize {
        4 * self.tokens.len() + 8
    }
    fn snapshot_bytes(s: &StubSnapshot) -> usize {
        4 * s.tokens.len() + 8
    }
    fn state_id(&self) -> Option<String> {
        Some(format!("stub vocab={}", self.n_vocab))
    }
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> Result<()> {
        let e = |e: std::io::Error| EngineError(e.to_string());
        for &t in &self.tokens {
            w.write_all(&t.to_le_bytes()).map_err(e)?;
        }
        w.write_all(&(self.script_pos as u64).to_le_bytes()).map_err(e)
    }
    fn warm_hint(&mut self) -> Option<Vec<u8>> {
        (self.n_evaluated > 0).then(|| (self.n_evaluated as u64).to_le_bytes().to_vec())
    }
    fn apply_warm_hint(&mut self, hint: &[u8]) -> Result<String> {
        self.calls.push(("warm", 0, hint.len()));
        Ok(format!("stub warm hint: {} bytes", hint.len()))
    }
    fn load_state(&mut self, r: &mut dyn std::io::Read, n_pos: usize) -> Result<()> {
        let e = |e: std::io::Error| EngineError(e.to_string());
        let mut b = vec![0u8; n_pos * 4 + 8];
        r.read_exact(&mut b).map_err(e)?;
        self.calls.push(("load", self.tokens.len(), n_pos));
        self.tokens = b[..n_pos * 4].chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        self.script_pos = u64::from_le_bytes(b[n_pos * 4..].try_into().unwrap()) as usize;
        self.fresh = true;
        Ok(())
    }
    /// "KV rows" of the stub = the token ids at those positions.
    fn kv_row_bytes(&self) -> usize {
        4
    }
    fn kv_rows_save(&mut self, a: usize, b: usize, w: &mut dyn std::io::Write) -> Result<()> {
        if a > b || b > self.tokens.len() {
            return Err(EngineError("stub: rows beyond n_past".into()));
        }
        for &t in &self.tokens[a..b] {
            w.write_all(&t.to_le_bytes()).map_err(|e| EngineError(e.to_string()))?;
        }
        Ok(())
    }
    fn kv_rows_load(&mut self, a: usize, b: usize, r: &mut dyn std::io::Read) -> Result<()> {
        if a > b || b > self.n_ctx {
            return Err(EngineError("stub: rows beyond n_ctx".into()));
        }
        let mut buf = vec![0u8; (b - a) * 4];
        r.read_exact(&mut buf).map_err(|e| EngineError(e.to_string()))?;
        self.tokens.truncate(a);
        self.tokens.extend(buf.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())));
        self.calls.push(("rows", a, b - a));
        Ok(())
    }
    fn snap_save(&mut self, s: &StubSnapshot, w: &mut dyn std::io::Write) -> Result<()> {
        let e = |e: std::io::Error| EngineError(e.to_string());
        w.write_all(&(s.tokens.len() as u64).to_le_bytes()).map_err(e)?;
        w.write_all(&(s.script_pos as u64).to_le_bytes()).map_err(e)
    }
    fn snap_load(&mut self, r: &mut dyn std::io::Read, n_pos: usize) -> Result<()> {
        let e = |e: std::io::Error| EngineError(e.to_string());
        let mut u = [0u8; 16];
        r.read_exact(&mut u).map_err(e)?;
        let pos = u64::from_le_bytes(u[..8].try_into().unwrap()) as usize;
        if pos != n_pos || self.tokens.len() < n_pos {
            return Err(EngineError("stub: snapshot position does not match its rows".into()));
        }
        self.tokens.truncate(n_pos);
        self.script_pos = u64::from_le_bytes(u[8..].try_into().unwrap()) as usize;
        self.fresh = true;
        self.calls.push(("snap", n_pos, 0));
        Ok(())
    }
    fn restore(&mut self, s: &StubSnapshot) -> Result<()> {
        // the KV for positions < pos must be what the snapshot saw
        if self.tokens.len() < s.tokens.len() || self.tokens[..s.tokens.len()] != s.tokens[..] {
            return Err(EngineError("stub: restore of a snapshot whose KV prefix was overwritten".into()));
        }
        self.calls.push(("restore", self.tokens.len(), s.tokens.len()));
        self.tokens.truncate(s.tokens.len());
        self.script_pos = s.script_pos;
        self.fresh = true;
        Ok(())
    }
}
