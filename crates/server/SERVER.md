# oxe-server

An OpenAI-compatible HTTP server shaped like llama-server, so the controller, OpenCode and
dsh can talk to the Rust engine without changes. It is the M4b "standalone" route from
`M3-M4-PLAN.md`, built early so it is ready when the engine is.

Status (2026-11-09): tokenizer, chat template, tool-call parsing, HTTP API and the prefix
cache are done and tested against llama.cpp itself. The model is still a stub
(`StubEngine`); wiring the M2 engine is the remaining step (section 4).

## 1. Layout

| Crate | What | Ported from (llama.cpp @ 954de24, the production source) |
|---|---|---|
| `crates/tokenizer` | GGUF byte-level BPE, qwen2/qwen35 pre-tokenizers, special-token partitioning, detokenize, streaming UTF-8 hold-back | `src/llama-vocab.cpp`, `src/unicode.cpp`, tables generated from `src/unicode-data.cpp` (`tools/gen_unicode.py`) |
| `crates/chat` | template rendering (minijinja + llama.cpp jinja semantics), OpenAI request normalization, Qwen3-Coder tool-call parser with streaming diffs | `common/chat.cpp`, `common/jinja/value.cpp`, `common/parsers/qwen3-coder.cpp`, `common/chat-peg-parser.cpp`, `tools/server/server-common.cpp` (`oaicompat_chat_params_parse`) |
| `crates/server` | `Engine` trait + stub, prefix cache, sampler, generation loop, endpoints, hyper front end, `oxe-server` binary | `tools/server/server-context.cpp`, `server-task.cpp`, `server-chat.cpp` |
| `crates/server/oracle` | `llama-oracle`: a small C++ program linked against llama.cpp's own shared libs (vocab-only load, no weights) that answers tokenize / apply-template / parse-stream questions with llama.cpp's code. Generates the test fixtures. | |

## 2. Running it

`$MODELS` is the directory that holds the GGUF files.

```sh
CARGO_BUILD_JOBS=1 nice -n 19 cargo build --release -p server
target/release/oxe-server --model $MODELS/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --alias ornith --host 127.0.0.1 --port 8093 --ctx-size 16384 \
  --chat-template-kwargs '{"enable_thinking":false}' --engine stub [--stub-reply TEXT]
```

The stub replies with `--stub-reply` (tokenized), then EOS, and it uses no model memory, so
it is safe to run without the inference lease. llama-server flags that the controller
passes (`--threads`, `--jinja`, `-b/-ub`, `--spec-type`, `--reasoning-budget`, ...) are
accepted and ignored (`IGNORED` in `src/main.rs`); unknown flags are an error.

### Access (v0.1.1)

The port is for local clients. `oxe-server` sends no CORS headers, refuses (403) a request whose
`Host` is not `127.0.0.1`, `localhost` or `[::1]` with the server's port (add names with
`--allow-host NAME`; a specific non-loopback `--host` address is accepted as a Host), and refuses
any request whose `Origin` is not loopback or that carries `Sec-Fetch-Site: cross-site`. That
stops web pages (including DNS rebinding) from using the engine. Against other local users, set
a key: `--api-key-file PATH` or `OXE_API_KEY_FILE=PATH` (one key per line, file mode 0600); every
route, `/health` included, then needs `Authorization: Bearer KEY` (or `X-Api-Key: KEY`), else 401.
The lab controller creates a fresh key per launch. Sampling values out of range (`min_p` or
`top_p` outside 0..1, `top_k` beyond 32 bits, non-finite numbers, penalties outside their range,
`n_probs` / `top_logprobs` above 100) are a 400. A streaming client that takes no data for
`--stream-stall-timeout` seconds (30) is dropped and its generation cancelled.

Endpoints (llama-server shapes, key order included):

| Endpoint | Status |
|---|---|
| `POST /v1/chat/completions`, `/chat/completions` | stream and non-stream; `messages` (system/developer/user/assistant/tool, `tool_calls`, `tool_call_id`, typed content parts), `tools`, `tool_choice` (auto/none/required, string form), `parallel_tool_calls`, `chat_template_kwargs`, `reasoning_format`, `reasoning_effort`, `max_tokens`/`max_completion_tokens`/`n_predict`, `temperature`/`top_k`/`top_p`/`min_p`/`seed`, `stop`, `cache_prompt`, `stream_options.include_usage`, `logprobs`/`top_logprobs` (non-stream). SSE: role chunk, one chunk per diff (`reasoning_content` / `content` / `tool_calls[i]` fragments), finish chunk with `finish_reason` stop/tool_calls/length, optional usage chunk with `choices: []`, `timings` on the last chunk, `data: [DONE]`. |
| `POST /completion`, `/completions` | native shape; prompt as string / token ids / mixed; `n_predict`, `n_probs` (+`post_sampling_probs` key names), `return_tokens`, `stop`, stream (no `[DONE]`, like llama-server) |
| `GET /health`, `/v1/health` | `{"status":"ok"}` |
| `GET /v1/models`, `/models`, `GET /props` | shape-compatible |
| `POST /tokenize` (`add_special`, `parse_special`, `with_pieces`), `/detokenize`, `/apply-template` | yes |
| `GET /slots` | with `--slots` |
| `POST /slots/0?action=save/restore` | 501 for now (see gaps) |
| errors | `{"error":{"code","message","type"}}`, 400 `exceed_context_size_error` with `n_prompt_tokens`/`n_ctx`, 404 `not_found_error` |

Prefix cache: the engine holds one sequence. Before each request the server reuses the
longest reachable prefix of the new prompt: the live state if it is a prefix, else the
deepest checkpoint inside the common prefix (attention KV is kept in place; the recurrent
state comes from the checkpoint). This is what makes agentic turns cheap: turn N+1 re-sends
turn N's prompt + reply + tool result, so only the new tail is prefilled
(`usage.prompt_tokens_details.cached_tokens` and `timings.cache_n` report it).

Checkpoints (`--ctx-checkpoints`, default 4; ~63 MiB each), see `cache.rs`:
- `prompt_len - 1` and the end of every generation (a repeated prompt, or one whose re-sent
  assistant turn differs from the generated one, resumes there);
- chat requests: the **anchor**, the deepest ubatch-grid position <= the first user message
  (system prompt + tools; a new conversation with the same agent resumes there), and, when the
  request ends with a user message, the **query** checkpoint, the deepest grid position
  <= `prompt_len - 4`. Why: the Qwen3.x / Ornith templates render an assistant turn with
  `<think>\n...\n</think>\n\n` only while it follows the last user message. When the next
  user message arrives, every assistant turn of the previous query is re-rendered without
  it, so the new prompt differs from the cached tokens right after `<|im_start|>assistant\n`
  of the query's first reply: 4 tokens before the end of that query's first prompt with
  `enable_thinking:false` (`<think>`, `\n\n`, `</think>`, `\n\n`), 2 with thinking on. No
  checkpoint existed below that point, so such requests fell back to 0 cached tokens (the
  `cache_n 0` of the toy transcript's 4th turn). Tool-result turns are unaffected (the
  history keeps the think block and the reply matches token for token).
- Eviction: lowest priority first (anchor > query > the rest), then least recently used.

Grid positions are multiples of `--ubatch-size` counted from where this request's prefill
started (from 0 when the state is "clean", i.e. was itself prefilled from 0 on the grid).
Splitting a prefill at a grid position changes no ubatch boundary, so a request computes
exactly what an unsplit prefill computes; output for a given request sequence is
deterministic. Resuming from any checkpoint changes later ubatch boundaries relative to a
cold prefill of the whole prompt, as in llama-server (its checkpoints sit at user-message
starts and `n_tokens - 4 - n_ubatch` / `n_tokens - 4`); resuming from a clean grid
checkpoint (anchor, query of a cold request, prefix-store entry) is bit-identical to a cold
prefill.

llama-server comparison: it checkpoints at the start of the last user message (tool results
are user messages in these templates), at `n - 4 - n_ubatch` and `n - 4` (`--batch-size 256`
makes the first `n - 256`), and with `--ctx-checkpoints 4` keeps in practice only its very
first checkpoint (the first user message start) plus the current request's ones. So on a new
user message after a tool chain it resumes from the first user message of the conversation and
re-prefills the whole chain; real OpenCode tasks (one user message, then tool turns) reuse
the cache on every request in both servers.

Persistent prefix store (`--prefix-cache DIR|default`, off unless given; default directory
`~/.cache/local-ai/snapdragon-engine/prefix`, `--prefix-cache-max-bytes` (4 GiB, LRU by mtime),
`--prefix-cache-min-tokens` (1024)), `prefix_store.rs`: when a chat request creates its anchor
from a clean state, the whole engine state at the anchor (KV rows of every attention layer
incl. the MTP layer, recurrent state, MTP driver state) is written in the background to
`<sha256>.bin`. Only system prompt + tool tokens are stored, never user turns or tool output.
A later request, also after a restart, whose prompt starts with a stored prefix loads it if
it is longer than what the in-memory cache offers (log line `prefix store: loaded N tokens in
M ms`). Identity = model (`OXE_MODEL_SHA256`, as the controller exports it, else path + size +
mtime), SHA-256 of the server binary, the engine state layout (MTP on/off) and the ubatch; the
file repeats identity and token ids, both compared in full. Bad or truncated entries are
deleted and the request continues cold. Size: ~22 KiB per token (11 attention layers x K+V
f16) + 63 MiB recurrent state, ~180 MiB for a 5.5K-token OpenCode prefix. The controller's
`/slots` save/restore protocol is not needed: pass `--prefix-cache default` in the oxe args.

## 3. Tests (all with the stub engine; no model weights are read)

```sh
flock -n ${LOCAL_AI_LEASE:-${XDG_STATE_HOME:-$HOME/.local/state}/local-ai/inference.lock} true &&   # don't build while Codex holds the lease
CARGO_BUILD_JOBS=1 nice -n 19 cargo test --release -p tokenizer -p chat -p server
```

| Test | What it proves |
|---|---|
| tokenizer `llama_vocab_test_vectors_qwen35` | llama.cpp's own `ggml-vocab-qwen35` vectors: 50/50 |
| tokenizer `llama_oracle_fixture_ornith` | 279 strings (code, unicode, combining marks, whitespace runs, special tokens, slices of real source/docs) x {parse_special+add_special, plain} = 558 cases identical to llama.cpp `common_tokenize` on the Ornith vocab, plus detokenize round trip |
| chat `llama_fixtures_ornith`, `llama_fixtures_qwen36` | 88 + 88 fixtures produced by llama.cpp's `oaicompat_chat_params_parse` + `common_chat_session`: rendered prompts and generation prompts byte-equal (system/no system, developer, typed parts, multi-turn tool calls with nested/float/unicode arguments, reasoning history, tools with/without params, thinking on/off); model outputs fed token by token give identical per-token diffs and final messages (text, tool call + text, parallel calls, every schema type incl. untyped/anyOf/enum/nullable, reordered required args, missing args, unknown function, truncation, text after calls, reasoning split, tool_choice none, parallel off, reasoning_format none) |
| server `chat_nonstream_shape`, `stream_usage_chunk_and_done` | response/chunk key order, usage/timings, SSE framing, `[DONE]` |
| server `chat_stream_tool_call_roundtrip` | streamed tool call accumulates to the exact arguments string; sending it back as an assistant `tool_calls` + `tool` message renders, and the second turn hits the prefix cache (only the new tail is evaluated) |
| server `completion_n_probs_and_limits` | `/completion` n_probs shape (router), `post_sampling_probs`, `n_predict` limit, context overflow error, native stream |
| server `multiturn_tool_calls_reuse_cache` | OpenCode-like chat (system + tools, tool call, tool result, answer + new user message): turn 2 reuses prompt + reply, the new user message diverges exactly at the `<think>` 4 tokens before the end of the first prompt and resumes from the query checkpoint (was 0), a new conversation resumes from the anchor |
| server `prefix_store_survives_restart` | the anchor is written to disk, a new server over the same directory resumes from it with the same reply; another identity does not see it; a truncated entry is discarded and the request runs cold |
| server `prefix_store_real_model` (`#[ignore]`, `OXE_TEST_MODEL`) | real model: grid splits change no token, a stored-prefix resume after "restart" gives the cold run's token ids, every turn after the first reuses the cache, MTP on = off |
| server `prefix_cache_logic` | repeated prompt, continuation, divergence, unrelated prompt, `cache_prompt:false`; the stub rejects any restore whose KV prefix was overwritten |
| server `stop_strings_and_limits`, `helper_endpoints` | stop words (truncation, hold-back), cancellation, tokenize/detokenize/apply-template/models/404/400 |

Regenerating fixtures (needs only the llama.cpp build tree; vocab-only loads, seconds):

```sh
crates/server/oracle/build.sh <llama.cpp checkout> /tmp/llama-oracle
python3 -I crates/tokenizer/tools/make_corpus.py <llama>/models/ggml-vocab-qwen35.gguf.inp <text files...> > corpus.jsonl
python3 -I crates/tokenizer/tools/oracle_fixture.py /tmp/llama-oracle <model.gguf> corpus.jsonl crates/tokenizer/tests/fixtures/qwen35_llama_tokenize.jsonl
python3 -I crates/chat/tools/make_fixtures.py /tmp/llama-oracle <model.gguf> crates/chat/tests/fixtures/ornith.jsonl
```

## 4. Connecting the real engine (after M2)

Implement `server::engine::Engine` for the model crate and add an `--engine oxe` arm in
`src/main.rs` (next to `stub`). Sketch:

```rust
pub struct OxeEngine {
    eng: model::engine::Engine<'static>,   // plan + weights leaked: the server owns them for life
    pool: sched::Pool,
    logits: Vec<f32>,
}
impl server::engine::Engine for OxeEngine {
    type Snapshot = model::RecurrentSnapshot;     // NEW in model: conv_state + ssm_state + pos
    fn n_ctx(&self) -> usize { self.eng.n_ctx }
    fn n_vocab(&self) -> usize { self.eng.plan.desc.n_vocab }
    fn n_past(&self) -> usize { self.eng.pos() }
    fn reset(&mut self) { self.eng.reset() }
    fn prefill(&mut self, toks: &[u32]) -> Result<&[f32]> {
        for &t in toks { self.eng.step(&self.pool, t); }   // until M3.2 batched prefill
        Ok(self.eng.logits())                            // EngineConfig { logits: true, .. }
    }
    fn snapshot(&mut self) -> Result<Self::Snapshot> { Ok(self.eng.save_recurrent()) }
    fn restore(&mut self, s: &Self::Snapshot) -> Result<()> { self.eng.restore_recurrent(s); Ok(()) }
}
```

What the model crate needs for this (not in M1 yet):
1. `save_recurrent()` / `restore_recurrent()`: copy `conv_state` + `ssm_state` (62.8 MiB for
   Ornith) and set `pos`. KV stays in place: attention only reads positions `< pos`, so
   rewinding `pos` is enough for the KV part. 4 checkpoints = ~250 MiB.
2. Owning construction (`Engine<'a>` borrows `Plan`/`WeightStore`; leaking them is fine for a
   server process, or an owning wrapper).
3. Logits every step (`EngineConfig.logits = true`) for sampling and `n_probs`. Greedy-only
   requests could use `step()`'s argmax directly; add `fn decode_argmax` to the trait if the
   full-logits copy shows up in the profile.
4. Batched prefill (M3.2) behind the same `prefill(&[u32])` call. Token-by-token prefill is
   correct but ~30 tok/s, too slow for 7K-token OpenCode/dsh system prompts on a cold cache.
5. MTP (M3.1) fits behind `decode` later (draft + verify inside the engine, same outputs).

## 5. Verification once the engine is wired (the few tests the owner asked for)

All three run under the inference lease (Codex lane), one model in memory at a time.

(a) **Identical greedy output vs llama-server on 3 fixed prompts.** Start llama-server and
oxe-server on the same GGUF and ctx, then:
`python3 crates/server/tools/compare_greedy.py --ref http://127.0.0.1:8093 --oxe http://127.0.0.1:8094 --n 256`.
It checks the rendered prompt bytes, `/completion` greedy token ids (`return_tokens`) and the
chat message / tool calls for 3 prompts (code, prose, tool call). Use the native-layout
llama.cpp build as the reference (M1.md): the production build uses KleidiAI for Q6_K/Q8_0,
which is a different computation, so token identity against production is not guaranteed;
against production, report first-divergence position and use the M2 KLD gate instead.

(b) **One tool-call round trip in OpenCode.** Point the controller's backend at oxe-server
(or run OpenCode against it directly on a scratch dir), ask for a task that needs one file
read; check that the tool call executes, the tool result goes back, and the second request
shows `cached_tokens > 0` (prefix cache) in the server log/usage.

(c) **The protected coding task once** (the 13-case run is not needed): one run through
the controller with the production profile, compare pass/fail and wall time with the
llama-server baseline.

## 6. Known gaps (none affects greedy, well-formed outputs)

- No grammar-constrained sampling. llama-server activates a lazy tool-call grammar after
  `<tool_call>`; with greedy decoding on a good model the argmax already follows the format,
  but a malformed call that llama.cpp's grammar would have prevented is passed through
  (and parsed the same way llama.cpp parses it). `tool_choice: required` and
  `response_format`/`json_schema` are not enforced either.
- Reasoning budget (`--reasoning-budget`) is not enforced (irrelevant with
  `enable_thinking:false`, the production setting).
- Sampling with temperature > 0 follows llama.cpp's chain order but not its RNG; only greedy
  is bit-comparable.
- `POST /slots/0?action=save|restore` (controller prefix store) returns 501; the server's
  own `--prefix-cache` replaces it (section 2).
- Not implemented: trailing-assistant prefill (`continue_final_message`), `/v1/completions`
  OpenAI shape, `/infill`, embeddings, multimodal content, logprobs in chat streams,
  `/v1/responses`, Anthropic `/v1/messages`.
- `serde_json` is built with `preserve_order` in these crates (llama-server key order). A
  whole-workspace build unifies the feature, so other crates' JSON output then keeps
  insertion order instead of sorted keys.
