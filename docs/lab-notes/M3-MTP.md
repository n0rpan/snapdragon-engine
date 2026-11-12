# M3.1: MTP speculative decoding (exact)

> **Historical working note.** The branches and worktrees named here were merged into `main` and then deleted; the commit hashes cited still resolve in `main`'s history. Current state: [README](README.md).

**Update, 2026-11-10: rebased on M3.2 (batched prefill, 9833695, plus `prefill-mmid-pad`
267bb7c/04c1667) and merged to main.**
- **Codex's timing:** Qwen3.6 goes from 33.07 to **43.05 tok/s at depth 2 (+30 %)**, with depth 3
  flat. Ornith is flat at depth 1.
- **llama.cpp confirms Ornith's weak chaining:** 75.7 % first draft, 15.9 % second.
- **Server defaults are now per model:** Qwen3.6 draft-mtp depth 2, Ornith off. Flags override.
- **MTP prompts use the batched prefill:** `Engine::prefill_nextn`, which returns every position's
  h_nextn. Its logits equal `Engine::prefill`'s, see 1.2.
- **Gates re-run on the rebased code** (section 2.1): all identical / exact.

**Status, 2026-11-09:** implemented on branch `feature/mtp`. **Exactness gate passed on Ornith-1.5-35B-A3B
and Qwen3.6-35B-A3B.** For 3 prompts × 256 greedy tokens, MTP depth 1, 2 and 3 give the same token ids
as MTP off, and every committed logit row is **bit-identical** to the MTP-off row at that position
(1,536 rows per model). The MTP-off greedy run in turn equals llama.cpp's oracle continuation on all 6
prompts, and `oxe verify` still gives 308/308 bit-exact rows. **No timing benchmark was run** (Codex
lane, commands below). Instrumentation numbers here (time split) come from correctness runs and are
indicative only.

Headline numbers:
- Qwen3.6: MTP acceptance is 82-89 % at depth 1, and **2.9-3.25 tokens per verify at depth 3**.
- Ornith: first-draft acceptance is 52-71 %, but its head chains badly (second draft about 13 %
  conditional), so tokens per verify stay at **1.5-1.76 at every depth**.
- Expected gain is ~1.3-1.4x for Qwen3.6 and about break-even for Ornith (section 5).

## 1. What was built

| Piece | Where | What |
|---|---|---|
| Lanes | `crates/model/src/engine.rs` | One run carries up to `MAX_LANES = 8` consecutive tokens. Every lane runs the one-token arithmetic. |
| Verify / commit | `Engine::verify(&[t0..tn])`, `Engine::commit(m)` | Batch-invariant verify of n tokens; keep the first m. |
| Rollback | same | KV: position reset, and V^T of rejected positions zeroed (as a fresh cache). Recurrent: slot index (below). No state copies. |
| MTP head | `crates/model/src/mtp.rs` `MtpWeights` | `blk.{n_layer}.*` plus `nextn.{eh_proj,enorm,hnorm,shared_head_norm}`, in its own cache file `<sha>-<engine-id>-mtp.bin` (Ornith 0.85 GiB, Qwen 0.50 GiB). Built in about 1 s on first use. |
| MTP run | `Engine::mtp_run` | Entry (`eh_proj . [enorm(embd(tok)); hnorm(h)]`) -B- attention block (own KV slot) -B- MoE -B- `shared_head_norm` + shared output head. |
| Driver | `mtp.rs` `MtpDriver` | Catch-up, draft chain, verify, greedy accept, `rewind`, `MtpStats`. |
| CLI | `oxe mtp-check`, `oxe bench --spec-draft-n-max N` (alias `--draft N`) | Exactness gate; speed arm. |
| Server | `oxe-server --spec-type draft-mtp --spec-draft-n-max N` (also `--mtp`, `--draft N`) | Greedy requests speculate. `timings.draft_n` / `draft_n_accepted` follow llama-server. |

Existing code paths keep their behaviour. `Engine::new` is the old engine (one lane, one slot), and
`step()` is `verify(&[t]) + commit(1)`, which runs the one-lane code (same calls as M2, no extra
barrier).

### 1.1 Batch-invariant verify (exactness by construction)

The requirement is that lane i of `verify` computes exactly what `step` computes at position
`pos0+i`. Each op does the following:
- **Dense GEMVs** (`mv_lanes`): the thread's row range is walked in chunks of about 24 KiB of weights
  (whole row groups), and every lane runs the same ggml gemv kernel on each chunk while it is in L1.
  An output row is one independent dot product (the kernels loop over 8- or 4-row groups), so the
  chunking cannot change a value. M2 already relied on this: the thread partition never changes a
  result.
- **Norms, rope, conv, delta rule, softmax, routing**: per lane, same code and same inputs. The rope
  cache is per lane.
- **Delta-net state**: lanes run in order inside RecMix (the same thread owns the same heads for all
  lanes). Lane 0 updates the current slot in place. Lane i copies its own heads' rows from slot
  `cur+i-1` into slot `cur+i` (64 KiB per head, cache-hot) and steps there. `commit(m)` sets
  `cur = cur+m-1`, so rollback is an index change. There are `verify_lanes = depth+1` slots
  (63 MiB each for Ornith).
- **Attention**: in a one-lane run nothing changes. With n > 1, each KV head's designated thread
  writes the K rows of all lanes, then an extra barrier follows, and each lane attends over the cache
  with its own row passed as `k_last`. These are the same K bits the one-token path computes locally.
  Masked positions get weight exactly 0, and rejected V^T is zeroed on commit, so the KV contents
  beyond a lane's position cannot leak into it.
- **MoE**: the union of the lanes' experts (first use, lane-major) is split evenly over the threads.
  Each (expert, 256-row block) unit runs gate/up for every lane routed to it, while its rows are hot.
  MoeDown loops over the union, and the weighted combine is per lane in llama.cpp's slot order.
  With one lane this is exactly the plan's old unit order.
- **Head**: chunked over rows with lanes inner. The argmax is a first-max scan per lane in row order
  (same tie-break).

### 1.2 MTP semantics (matches llama.cpp `common_speculative_impl_draft_mtp`, PR #22673)

Read from `src/models/qwen35moe.cpp` `graph_mtp` and `common/speculative.cpp`:
- `h` = the target's `h_nextn` = `output_norm(x_last_layer)` (the LM head input). For chained
  drafts it is the MTP's own `h_nextn` = `shared_head_norm(post_ffn)`.
- MTP KV position p always ends up holding the target pair `(x_p, h_{p-1})`, with `h_{-1} = 0`
  (llama's `pending_h` starts at zeros). The driver keeps these pairs pending and flushes them as
  catch-up lanes (no head) when `mtp_lanes` accumulate. The rest ride along in the next draft run.
- Draft: one MTP run with lanes = [pending catch-up pairs..., (x, h_{P-1}) at P], head on the last
  lane only, gives d1. Then single-lane runs (d_i, h_mtp) at P+i give d_{i+1}. Greedy, p_min 0
  (llama's default), `n_max` = depth.
- Verify: [x, d1..dk] through the target. a = the longest prefix with d_i == argmax(lane i-1).
  Commit a+1 tokens; the bonus is argmax(lane a). The pairs of d_1..d_a with the target's h are
  queued for catch-up.
- Prefill with MTP on (rebased on M3.2): `MtpDriver::prefill_batched` calls the new
  `Engine::prefill_nextn` (in `engine/prefill.rs`). It is M3.2's `prefill_ext`, so the batched
  prefill also returns every position's `h_nextn`, with one change: after the all-rows last layer
  has produced the `h_nextn` rows, the output row's last-layer MoE is recomputed on its own from the
  unchanged `x1` row (`crop_before_nextn`, exactly what `Engine::prefill` does). The KV, the
  recurrent state and the last-token logits are therefore `Engine::prefill`'s bit for bit, so MTP on
  equals MTP off. `prefill_ext` itself keeps llama-server-with-MTP's behaviour, where the last row
  can differ in the last bits. All prompt pairs are then caught up in MTP runs of up to 8 lanes
  (`MAX_LANES`) on the decode team. The token-wise `MtpDriver::prefill` (verify runs of depth+1
  tokens) remains for `oxe bench`, whose plain arm prefills token by token.
- BF16 tensors (Qwen3.6's MTP `ffn_gate_inp` and `ffn_gate_inp_shexp`) are converted to f32 and
  multiplied with the f32 activation. llama.cpp rounds the activation to bf16 there, so a rare draft
  may differ from llama's. The verified output cannot differ.

### 1.3 Server

The `Engine` trait gains three defaulted hooks (`decode_spec`, `spec_rewind`, `spec_counters`). For
greedy requests (`temperature <= 0`), `generate.rs` calls `decode_spec(t)` and gets
`(extra tokens, logit rows)`. It then consumes the queued rows one by one as the loop reaches them,
and `cache.appended` stays per token. On a stop (EOG, stop word, limit, cancel), or if the sampler
ever leaves the queued path, it calls `spec_rewind(1 + consumed)`, which recommits fewer lanes of the
last verify. The engine state then equals the plain run's, which the prefix cache needs. Snapshots
carry the MTP `h` of the last position. Sampled (temperature > 0) requests decode one token at a
time, and no exactness is claimed for them.

Server defaults (`server::oxe_engine::default_mtp_depth`, chosen from the GGUF file name):
- Qwen3.6: draft-mtp depth 2 (Codex: +30 %).
- Ornith: MTP off (flat at depth 1; llama.cpp confirms its head chains badly).
- Any other model: off.

Flags override the default:
- `--spec-type none` turns MTP off.
- `--spec-type draft-mtp`, `--mtp` or `--draft N` turn it on.
- `--spec-draft-n-max N` sets the depth.
Note that a controller profile which passes `--spec-type draft-mtp` for Ornith turns it on.

Server smoke run (Qwen3.6, `--spec-draft-n-max 2`): `/completion` of `def fibonacci(n):` for 64
tokens, and "The capital of France is" for 16 tokens. **The text is identical to MTP off**, with
`draft_n` 44 and `draft_n_accepted` 43.

## 2. Exactness results

`oxe mtp-check --refs <3 oracle dirs> --n 256 --depths 1,2,3`, 8 threads (cpus 0-7). Per prompt it
runs plain greedy (an `Engine::new` engine), then each depth on a fresh MTP engine. It compares ids,
the prefill's last row, and every row a verify committed. Raw JSON is in
`results/mtp-check-{ornith,qwen36}.json` (git-ignored).

| Model | Prompt | Prompt tokens | Depth | ids identical | Bit-identical rows | Acceptance | Accepted at depth 1/2/3, of steps | Tokens/verify | Time split draft / verify / catch-up % |
|---|---|---:|---:|---|---:|---:|---|---:|---|
| ornith | p1-code | 53 | 1 | yes | 256/256 | 71.1 % | 106 of 149 | 1.71 | 13 / 87 / 0.4 |
| ornith | p1-code | 53 | 2 | yes | 256/256 | 37.9 % | 97/13 of 145 | 1.76 | 18 / 81 / 0.3 |
| ornith | p1-code | 53 | 3 | yes | 256/256 | 25.3 % | 97/13 of 145 | 1.76 | 19 / 80 / 0.1 |
| ornith | p2-prose | 64 | 1 | yes | 256/256 | 58.4 % | 94 of 161 | 1.58 | 13 / 87 / 0.5 |
| ornith | p2-prose | 64 | 2 | yes | 256/256 | 30.7 % | 86/11 of 158 | 1.61 | 18 / 82 / 0.4 |
| ornith | p2-prose | 64 | 3 | yes | 256/256 | 21.5 % | 86/11/3 of 155 | 1.65 | 19 / 81 / 0.2 |
| ornith | p3-chat | 41 | 1 | yes | 256/256 | 51.8 % | 87 of 168 | 1.52 | 13 / 87 / 0.3 |
| ornith | p3-chat | 41 | 2 | yes | 256/256 | 28.7 % | 80/13 of 162 | 1.57 | 17 / 83 / 0.2 |
| ornith | p3-chat | 41 | 3 | yes | 256/256 | 19.1 % | 80/13 of 162 | 1.57 | 19 / 81 / 0.1 |
| qwen36 | p1-code | 53 | 1 | yes | 256/256 | 85.5 % | 118 of 138 | 1.86 | 11 / 89 / 0.3 |
| qwen36 | p1-code | 53 | 2 | yes | 256/256 | 79.8 % | 87/71 of 99 | 2.60 | 16 / 84 / 0.3 |
| qwen36 | p1-code | 53 | 3 | yes | 256/256 | 72.8 % | 67/59/51 of 81 | 3.19 | 19 / 81 / 0.2 |
| qwen36 | p2-prose | 64 | 1 | yes | 256/256 | 88.9 % | 120 of 135 | 1.89 | 11 / 89 / 0.4 |
| qwen36 | p2-prose | 64 | 2 | yes | 256/256 | 81.4 % | 84/74 of 97 | 2.63 | 16 / 84 / 0.3 |
| qwen36 | p2-prose | 64 | 3 | yes | 256/256 | 75.1 % | 69/59/50 of 79 | 3.25 | 18 / 81 / 0.3 |
| qwen36 | p3-chat | 41 | 1 | yes | 256/256 | 82.1 % | 115 of 140 | 1.82 | 11 / 89 / 0.2 |
| qwen36 | p3-chat | 41 | 2 | yes | 256/256 | 75.5 % | 85/69 of 102 | 2.51 | 16 / 84 / 0.2 |
| qwen36 | p3-chat | 41 | 3 | yes | 256/256 | 62.5 % | 74/53/40 of 89 | 2.88 | 18 / 81 / 0.2 |

The plain (MTP-off) greedy ids equal llama.cpp's oracle continuation for all 6 prompts. The
regression `oxe verify --ref ornith-p1-code` still gives `greedy_identical: true` and 308/308
bit-exact rows.

**Ornith's head chains badly.** P(d2 accepted | d1 accepted) is 11-16 %, against 79-85 % for
Qwen3.6, and the third draft is almost never accepted. The Qwen3.6 numbers show the chain itself
works (same code), so this is the model: Ornith is a fine-tune, and its `blk.40` head was most likely
not retrained with it. llama.cpp's Ornith MTP acceptance has never been measured (the hardware
answers list it as pending), so this is the first number.

Unit tests (`cargo test --release -p model -p server`, synthetic model, no real weights):
- `mini_verify_lanes_are_batch_invariant_with_rollback`: at 1, 2 and 3 threads, verify runs of 1-4
  lanes with random rejected tails, partial commits, and commit-then-shrink. Every kept lane's
  logits, `h` and argmax equal the plain steps bitwise.
- `mini_mtp_greedy_is_identical_to_plain`: depths 1-3, with drafts forced right or wrong (test hook)
  so every acceptance length occurs, and with rewinds. Ids and rows are identical. The mini model has
  an MTP block with Q8_0 experts (like Ornith) and BF16 routers (like Qwen).
- `speculative_loop_matches_plain` (server): a speculating stub through `generate` at depths 1-3, with
  EOS, stop word and n_predict stops. Tokens, text, stop type and the engine state after the request
  equal the plain loop's.

### 2.1 After the rebase on M3.2 (2026-11-10)

`oxe mtp-check` now prefills with the batched prefill on both sides: `Engine::prefill` for MTP off,
and `MtpDriver::prefill_batched` (`Engine::prefill_nextn` plus MTP seeding) for MTP on. Ubatch is
256, on cpus 0-7. Every prompt and depth gives identical ids and 256/256 bit-identical rows.

| Model | Prompt | Depth | ids identical | Bit-identical rows | Acceptance | Tokens/verify |
|---|---|---:|---|---:|---:|---:|
| ornith | ornith-p1-code | 1 | yes | 256/256 | 72.3 % | 1.72 |
| ornith | ornith-p1-code | 2 | yes | 256/256 | 43.8 % | 1.88 |
| ornith | ornith-p1-code | 3 | yes | 256/256 | 29.2 % | 1.88 |
| ornith | ornith-p2-prose | 1 | yes | 256/256 | 55.5 % | 1.55 |
| ornith | ornith-p2-prose | 2 | yes | 256/256 | 30.7 % | 1.61 |
| ornith | ornith-p2-prose | 3 | yes | 256/256 | 20.5 % | 1.61 |
| ornith | ornith-p3-chat | 1 | yes | 256/256 | 67.8 % | 1.68 |
| ornith | ornith-p3-chat | 2 | yes | 256/256 | 38.3 % | 1.77 |
| ornith | ornith-p3-chat | 3 | yes | 256/256 | 25.5 % | 1.77 |
| qwen36 | qwen36-p1-code | 1 | yes | 256/256 | 85.5 % | 1.86 |
| qwen36 | qwen36-p1-code | 2 | yes | 256/256 | 79.8 % | 2.60 |
| qwen36 | qwen36-p1-code | 3 | yes | 256/256 | 72.8 % | 3.19 |
| qwen36 | qwen36-p2-prose | 1 | yes | 256/256 | 88.9 % | 1.89 |
| qwen36 | qwen36-p2-prose | 2 | yes | 256/256 | 81.4 % | 2.63 |
| qwen36 | qwen36-p2-prose | 3 | yes | 256/256 | 75.1 % | 3.25 |
| qwen36 | qwen36-p3-chat | 1 | yes | 256/256 | 82.1 % | 1.82 |
| qwen36 | qwen36-p3-chat | 2 | yes | 256/256 | 75.5 % | 2.51 |
| qwen36 | qwen36-p3-chat | 3 | yes | 256/256 | 62.5 % | 2.88 |

The other gates on the rebased tree:
- **`tools/run_prefill_verify.sh`**, all six fixtures (ornith and qwen36, pf64, pf512 and pf2048):
  last logits exact, state identical, decode rows exact 7/7, greedy identical.
- **M2 `oxe verify`**, p1-code: Ornith 308/308 and Qwen3.6 308/308 bit-exact rows, greedy identical.
- **Unit tests** (`cargo test -p model -p server -p kernels-sys`): all pass, including the new
  `mini_mtp_with_batched_prefill_is_identical_to_plain`. That test alternates batched prompts (ubatch
  8, partial last ubatch) with speculative decoding under forced acceptance, so the recurrent slot
  index rotates before the second prompt. It also covers rewinds at stop points. Ids, every row and
  the final `dump_state` equal the plain engine's.

The plain greedy ids no longer equal the token-by-token llama.cpp oracle
(`plain_matches_oracle_greedy: false`). This is expected: the reference prompt is now prefilled in
batches, and llama.cpp's own batched prefill differs from its decode path (M2.md, M3-PREFILL.md).
`--prefill tokenwise` reproduces the previous setup (section 2).

### 2.2 Server command lines (integrated MTP + batched prefill)

`$MODELS` is the directory that holds the GGUF files.

```bash
M=$MODELS
# Ornith: MTP off (its default), batched prefill on 12 cores, decode on 8
./target/release/oxe-server --engine oxe --model $M/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --oxe-cpus 0-7 --oxe-prefill-cpus 0-11 --ctx-size 16384 --port 8093
# Qwen3.6: MTP depth 2 (its default; written out explicitly here)
./target/release/oxe-server --engine oxe --model $M/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf \
  --oxe-cpus 0-7 --oxe-prefill-cpus 0-11 --ctx-size 16384 --port 8093 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

On startup the server prints `oxe-server: MTP speculative decoding on, 2 draft tokens` (or `off`).
`--spec-type none` forces MTP off, and `--spec-draft-n-max N` changes the depth. Per request,
`timings.draft_n` / `draft_n_accepted` show the speculation.

### 2.3 Seeding speed-up and multi-turn exactness (branch `mtp-seed-fast`, 2026-11-10)

**Seeding.** The MTP block is a full-attention layer. The only thing later drafts read from a
seeded position is its K/V in the MTP cache. `Engine::mtp_seed` therefore runs just the entry
(`eh_proj . [enorm(embd); hnorm(h)]`), `attn_norm`, the K/V projections, k_norm/rope and the
designated K writes. It skips Q, attention, o-proj, the MoE (previously the union of up to 64
experts per 8-lane run) and the head norm. It uses the same code as `mtp_run` (`attn_kv` and
`write_k` are split out of the attention phases), so the cache rows are the bits a full run writes.

The evidence is that `mtp-check`'s acceptance figures are unchanged to the last digit on both
models. The driver also seeds the decode-time catch-up pairs this way, then drafts with a one-lane
run.

Seeding cost, from the `OXE_MTP_LOG=1` instrumentation of correctness runs (indicative):
- about 51-85 ms per ~1,000 prompt tokens (~0.06 ms/token), previously ~0.6-0.7 ms/token;
- **estimate for the 5,544-token OpenCode prompt: ~0.35 s instead of 3.9 s.**

In the multi-turn server run, MTP-on prompt time was within ±150 ms of MTP off on 4.5-6 s prompts
(single samples). What remains is inherent: `prefill_nextn` runs the last layer's MoE over every
prompt row, because every position needs h_nextn; llama.cpp pays the same. For the other ~2.6 s in
Codex's 39.0 to 45.5 s, the old seeding is the main suspect: catch-up pairs also rode in every
draft run with the full layer. The next real-task run with `OXE_MTP_LOG=1` will show it.

**Multi-turn exactness.** I reproduced an OpenCode-like transcript against the real server
(Qwen3.6, MTP depth 2 vs `--spec-type none`). It has 4 turns, tools, tool calls and results,
`cache_prompt`, and a variant that re-sends the model's own replies.
- With temperature 0, every turn's reply is identical between MTP on and off, with identical
  cache reuse and completion counts. Fixed-seed sampling is identical too, because sampled
  requests never speculate.
- Differences appear only:
  1. **When no temperature is sent.** The server then uses llama-server's default of 0.8 with a
     time-based seed, so two runs differ regardless of MTP.
  2. **In tool-call ids**, which are random per response, as in llama-server.
- The new server test `mtp_multiturn_greedy_is_identical_to_plain` (`#[ignore]`, real model via
  `OXE_TEST_MODEL`) passes on Qwen3.6. It has 5 turns, each prompt = previous prompt + reply + new
  text, with stop-word and limit stops and rewinds.

**One real bug found and fixed.** When a streaming client disconnected mid-generation,
`generate` returned `Cancelled` without rewinding the speculative tokens. The engine was then up
to depth tokens ahead of the prefix cache, and the next request appended at the wrong position
and diverged from MTP off. It now calls `spec_rewind` first. A stub test cancels at every point
and checks that the state and the next request match the plain loop.

If Codex's trace still differs, the deciding check is: find the first request whose body hash
differs between the two runs, and compare the responses of the requests before it. An exactness
bug needs byte-identical bodies with different replies. Real tool output (times, paths, durations)
or random tool-call ids in the history make later bodies differ legitimately.

## 3. Instrumentation

`MtpStats` holds: verify steps, drafted, accepted, `accepted_at[i]`, tokens out, MTP runs and lanes,
plain steps, and wall time split into draft / verify / catch-up. These are printed as JSON by
`oxe mtp-check` and `oxe bench --spec-draft-n-max N` (`mtp` object). The server reports
`timings.draft_n` / `draft_n_accepted` per request.

## 4. Codex speed run (MTP off vs depth 1/2/3, both models)

The branch is `feature/mtp` (`git -C <repo> fetch && git checkout
feature/mtp`, or the worktree `<repo>`). All runs need the lease held
for their whole duration, with llama-server stopped. The first MTP run of a model builds its `-mtp`
cache (about 1 s).

```bash
cd <repo> && CARGO_BUILD_JOBS=2 cargo build --release
CPUS=0,1,2,4,5,6,8,9
L=${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock}
O=~/.cache/local-ai/snapdragon-engine/oracle
ORN=$MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf
QWEN=$MODELS/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf
# a real code prompt: acceptance depends on the text (the default bench prompt is random ids)
ids() { python3 -c "import json;print(','.join(map(str,json.load(open('$O/$1/manifest.json'))['tokens'])))"; }
P_ORN=$(ids ornith-p1-code); P_QWEN=$(ids qwen36-p1-code)
mkdir -p results
for d in 0 1 2 3; do                     # 0 = MTP off (the plain bench)
  X=""; [ $d -gt 0 ] && X="--spec-draft-n-max $d"
  flock -n $L ./target/release/oxe bench --model $ORN  --cpu-list $CPUS --ctx 2048 --warmup 16 --tokens 128 --prompt-ids $P_ORN  $X --out results/bench-ornith-mtp$d.json
  flock -n $L ./target/release/oxe bench --model $QWEN --cpu-list $CPUS --ctx 2048 --warmup 16 --tokens 128 --prompt-ids $P_QWEN $X --out results/bench-qwen36-mtp$d.json
done
# ABBA: repeat the loop in reverse order (3 2 1 0) for the second half.
```

`tools/run_bench.sh ... --spec-draft-n-max N` also works (random-id prompt, M2's fixed settings), but
the acceptance on a random prompt is not representative.

What to read in each JSON:
- `ms_per_token.median` (MTP off) vs `ms_per_token.effective` (MTP on: wall time / tokens produced in
  the timed region).
- `generated_hash`: **must be identical for d = 0, 1, 2, 3 of one model** (exactness under timing
  conditions).
- `mtp.acceptance`, `mtp.tokens_per_verify`, `mtp.time_split`, `ms_per_verify_step`.

For the end-to-end view (prefix cache, server loop, logits copies), run `oxe-server --engine oxe
--oxe-cpus $CPUS --spec-type draft-mtp --spec-draft-n-max N`. It is A/B against the same server
without the flags, comparing `timings.predicted_per_token_ms` of `/completion` greedy requests.

## 5. Measured (Codex, 2026-11-10) and expected gain

Measured: Qwen3.6 goes from 33.07 to 43.05 tok/s at depth 2 (**1.30x**), and depth 3 is flat against
depth 2. Ornith is flat at depth 1. Both are below the upper-bound model below, which ignores
overheads (Qwen depth 2 was estimated at 1.41x). Hence the server defaults: Qwen3.6 depth 2,
Ornith off.

### Byte model (written before the measurement)

Decode is bandwidth-bound. Per-token unique weight bytes (from the fixtures):

| | dense (incl. routers, shared) | routed experts (top-8) | output head | MTP block (dense + top-8) |
|---|---:|---:|---:|---:|
| Ornith | 914 MB | 466 MB | 417 MB | 43 + 27 MB |
| Qwen3.6 | 1,597 MB | 612 MB | 417 MB | 42 + 15 MB |

The cost model [E] assumes the following. A verify of n tokens reads the dense part and the head
once. The experts scale with the expected union of n top-8 sets out of 256, assuming independence
(15.8 / 23.3 / 30.7 experts for n = 2/3/4). Each draft reads the MTP block plus the full output head.
Overheads are ignored: barriers (+9 per MTP run, +1 per attention layer in a multi-lane verify),
compute, and the lane-copy of delta-net heads. The measured T (tokens per verify) comes from section 2.

| Model | Depth | Verify cost | Draft cost | T measured | **Speed-up [E]** |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 | 1 | 1.23 | 0.20 | 1.86 | **1.30x** |
| Qwen3.6 | 2 | 1.45 | 0.38 | 2.58 | **1.41x** |
| Qwen3.6 | 3 | 1.66 | 0.56 | 3.11 | **1.40x** |
| Ornith | 1 | 1.25 | 0.30 | 1.60 | 1.03x |
| Ornith | 2 | 1.50 | 0.57 | 1.65 | 0.80x |
| Ornith | 3 | 1.74 | 0.84 | 1.66 | 0.64x |

Costs are in units of one plain decode token.

Recommendation [E]:
- **Qwen3.6: depth 2** (depth 3 brings no further gain because its verify and draft costs grow).
- **Ornith: MTP off**, or depth 1 at best (break-even), until its head improves or drafting gets
  cheaper (section 6, item 1).

These are upper bounds, since the model ignores overheads. The cross-check against the measured time
split is consistent: drafts take 11-19 % of the speculative wall time. Codex's numbers decide.

## 6. Open issues

1. **The draft head is the dominant draft cost.** It reads the full 0.42 GB output head per draft
   token, while the MTP block itself is only 60-70 MB. A pruned or cached-vocabulary draft head
   (M3-M4-PLAN §1.2 option) would cut the draft cost about 6x. It is lossless for exactness (verify
   is exact) and lossy only for acceptance. This is the main lever for Ornith and for depth 3.
2. **Ornith's MTP head chains at about 13 %.** This is likely a fine-tune/head mismatch. It needs
   llama.cpp's Ornith MTP acceptance on the same prompts to confirm it is the model
   (`--spec-type draft-mtp --spec-draft-n-max 3`, read `draft_n_accepted`). That is a Codex lease run.
3. **Cache reuse across lanes is assumed, not measured.** Lane chunks are sized for L1 (24 KiB), and
   MoE units rely on L1/L2 for the users of one expert. If verify(n) shows more than the byte model's
   cost, `perf stat` of L2 misses on a verify run is the first check.
4. **The server copies logits.** Each speculative step copies depth+1 rows of 1 MB, the same per-token
   cost as the existing `decode(...).to_vec()`. A borrowed-row API would remove it.
5. **Done (2026-11-10): integration with batched prefill.** A cost remains: seeding the MTP KV
   after a long prompt runs the MTP block over every position (8 lanes per run, decode team). That
   is roughly 150 MB of weight reads per 8 positions, so about 1-2 s for a 7K-token prompt [E],
   unmeasured. A batched (GEMM) MTP layer in `prefill.rs` would remove it.
   The original text: MTP catch-up needs the target's `h_nextn`
   (output-normed hidden) of every prompt position. Today the driver prefills through verify runs,
   which is exact and reads weights once per depth+1 tokens. When the batched prefill lands, it
   should expose those rows (`h_nextn`) so that `MtpDriver` can catch up from them. The hook is the
   pair queue (`note`); a small `catch_up(tokens, h_rows)` entry point is enough. Without it,
   prefill-then-MTP still works, but drafts degrade for the prompt.
6. **Not implemented:** `--spec-draft-p-min` (llama's default is 0, which is what this does), sampled
   (temperature > 0) speculation with rejection sampling, and n_probs from speculative rows beyond
   what greedy requests return (they come from the exact rows, so they are already correct).
7. **Memory:** depth+1 recurrent slots (63 MiB each for Ornith; depth 2 adds 126 MiB), the MTP head
   (0.85 / 0.50 GiB of clean file cache), one extra KV layer, and per-lane scratch. The `-mtp` cache
   files are not cleaned up automatically (same rule as M2's main cache).
