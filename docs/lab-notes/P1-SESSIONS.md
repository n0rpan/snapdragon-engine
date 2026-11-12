# WP-P1: turn checkpoints, parked sessions, session persistence, 32K (design)

> **Historical working note.** The branches and worktrees named here were merged into `main` and then deleted; the commit hashes cited still resolve in `main`'s history. Current state: [README](README.md).

Branch `claude/ctx-sessions` from main 707a0ec. Owner Opus-C. Gate: G-exact + no OOM at 28 GiB (Qwen, 32K).
Tags: [M] measured, [E] estimate, [S] read from source.

## What already exists (main 707a0ec) [S]

- `cache.rs` PrefixCache: ONE live sequence + up to `--ctx-checkpoints` (default 4) recurrent-only
  snapshots (62.8 MiB each, M2.md). Restoring one is valid only while KV positions below it are untouched.
- `generate.rs` checkpoints: anchor (system+tools, grid floor of the first user message), query (before the
  generation prompt when the last message is `user`), tail at `len-1`, tail at end of generation.
- `prefix_store.rs`: anchors (>= 1024 tokens) on disk, KV + recurrent, keyed by identity + tokens.

## The gaps (why whole re-prefills still happen)

1. **Agent switch destroys the session.** Any request whose prompt diverges early (title or summary
   request, a `task` subagent with its own system prompt, plan <-> build) makes `prepare()` reset or
   restore below the live session; every snapshot above that point is dropped, the KV is overwritten.
   Back in the main session, only the anchor comes back (from disk); the whole conversation is
   re-prefilled: a 10-16K-token session at ~180 tok/s = 55-90 s [E].
2. **Edits and retries deeper than the last turn.** Only the last query/tail checkpoints exist (4 slots).
   Reverting to message k (OpenCode undo/revert, compaction keeps the head) restarts at the anchor.
3. **Unload / model switch** loses everything but anchors.
4. **16K context**: overflow forces OpenCode compaction (a full re-prefill of the summary + tail).

## Design

**A. Turn-boundary checkpoints (in memory).** For chat requests the service passes the token position of
every message start (`<|im_start|>` positions; the template's message delimiter) to `generate`. While
prefilling the uncached head, a checkpoint goes at the ubatch-grid floor of each message start (same rule as
anchor/query: grid relative to where this prefill started, so the split changes no ubatch boundary and the
request's numerics are those of the unsplit prefill). New priority `PRIO_TURN` (between TAIL and QUERY),
deduplicated per grid point. Only the newest `--turn-checkpoints` (default 6) are taken per request, so a
long fresh prompt does not take 60 snapshots. Default `--ctx-checkpoints` rises 4 -> 12 (0.75 GiB [E]).

**B. Parked sessions (in memory).** A sequence state = KV rows 0..pos + recurrent + MTP state
(`Engine::save_state`), i.e. 20 KiB/token + 62.8 MiB [S: 10 attn layers x 2 kv heads x 256 x K,V f16]
(Qwen MTP: +2 KiB/token + h rows). Before a request would destroy the live session (its reusable prefix
`r` is `< live_len - park_min` (512)), the worker **parks** it: saves the state into RAM together with its
checkpoints (they stay valid relative to the parked KV). For every request the worker picks the deepest
reusable point among: live state, in-memory checkpoints, parked sessions (their position or one of their
checkpoints within the common prefix), the prefix store. Unparking = `load_state` from RAM (memcpy,
~0.1-0.3 s for 16K [E]) and, if the hit is a parked checkpoint, `restore` on top. Budget
`--session-ram-bytes` (default 2 GiB), LRU; evicted sessions spill to disk (C) when enabled.
This is what gives "one prefix per agent": each agent's conversation is its own parked session, and a
fresh conversation with a known system prompt still resumes from the anchor (store or parked anchor
checkpoint).

**C. Session persistence (disk, opt-in).** `--session-cache DIR` (off by default; it holds conversation
content as KV, unlike the anchor-only prefix store). Same file format, identity check and LRU size cap as
`prefix_store.rs`, in its own directory (`--session-cache-max-bytes`, 8 GiB). Written on SIGTERM/SIGINT
(the controller's unload and model switch; it waits 10 s, the server waits at most 5 s for a running
generation) and when a parked session is evicted from RAM. Read at request time like the prefix store.
Privacy: dir 0700, files 0600, TTL (`--session-cache-ttl-hours`, 24) pruned at start; never in git; point it
at a tmpfs to keep sessions RAM-only. v1 stores the state at the session's length only (continuations);
its checkpoints stay in RAM.

**D. 32K.** No engine change: `--ctx-size 32768` already works; KV is `n_ctx`-sized at load. Cost vs 16K:
+320 MiB KV (Qwen MTP +32 MiB), prefill/decode attention scratch +~50 MiB [E: ATT_TT 4 x 8 x n_ctx x 6 B
per thread]; parked sessions grow with their length, not with n_ctx. Profile change in the lab controller
(`context_size` 32768 for Ornith; Qwen only if RSS <= 28 GiB, O-6) + OpenCode `limit.context`.
Codex ticket: RSS at 32K both models, and short-context speed unchanged (oxe bench 1K/8K, ABBA).

## Exactness (G-exact)

- Checkpoint placement only splits prefills on the ubatch grid: outputs identical to main for the same
  request sequence (stub tests on call logs; real-model test: identical token ids, checkpoints on vs off).
- Park/unpark is a byte copy: `save_state` after unpark == `save_state` before park (real-model test
  compares the bytes); the following request's token ids equal those of an un-interrupted session.
- Disk sessions: identity + full token compare before load (as the prefix store); corrupt -> discarded.

## Stub tests (no model)

turn checkpoints at grid floors; retry of turn k-2 resumes at its checkpoint; agent switch A -> B -> A
resumes A at its full length (no re-prefill: prompt_n = new tokens only); park budget eviction order;
session survives restart via `--session-cache`; identity mismatch and corruption fall back to cold.

## Status

A, B, C implemented on `claude/ctx-sessions` with stub tests (`parked_sessions_logic`,
`agent_switch_resumes_parked_session`, `turn_checkpoints_resume_edit`, `session_cache_survives_restart`)
and an ignored real-model G-exact test (`parked_sessions_real_model`). Production args (`--ctx-checkpoints 4`,
no `--session-ram-bytes`, no `--session-cache`) behave as main. Run tickets: the P1 run queue in the private lab notes.
