//! Server tests with the stub engine and the real Ornith tokenizer/template.

use crate::cache::PrefixCache;
use crate::engine::{Engine, StubEngine};
use crate::generate::{GenParams, StopType, Worker};
use crate::sampling::SamplingParams;
use crate::service::{handle, Config, Reply, State};
use serde_json::{json, Value as Json};
use std::path::PathBuf;
use std::sync::Arc;

/// The Ornith GGUF for tokenizer/template tests. Every test that needs it is `#[ignore]`d, so a
/// plain `cargo test` (CI) reports it as ignored instead of passing it vacuously; running one with
/// `--include-ignored` where the model is missing fails instead of returning early.
pub(crate) fn model() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".local/share/ai-stack/models/llm/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf");
    assert!(p.exists(), "model test run without its model: {} is missing", p.display());
    Some(p)
}

pub(crate) fn greedy() -> GenParams {
    GenParams { sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() }
}

fn state(scripts: &[&str], n_ctx: usize) -> Option<Arc<State<StubEngine>>> {
    state_with(scripts, n_ctx, 4, model::engine::UBATCH, None)
}

/// As [`state`] with `max_snapshots` checkpoints, a prefill `ubatch` (checkpoint grid) and
/// an optional prefix-store directory.
fn state_with(scripts: &[&str], n_ctx: usize, max_snapshots: usize, ubatch: usize, store: Option<&std::path::Path>) -> Option<Arc<State<StubEngine>>> {
    let tok = tokenizer::Tokenizer::from_gguf(model()?).unwrap();
    let mut eng = StubEngine::new(n_ctx, tok.n_vocab(), tok.eos.unwrap(), tok.encode("default reply", false, true));
    for s in scripts {
        eng.scripts.push_back(tok.encode(s, false, true));
    }
    let mut kw = serde_json::Map::new();
    kw.insert("enable_thinking".into(), Json::Bool(false));
    let cfg = Config {
        model_name: "ornith".into(),
        model_path: "/m.gguf".into(),
        chat_opts: chat::ServerChatOptions { chat_template_kwargs: kw, enable_thinking: false, ..Default::default() },
        fingerprint: "oxe-test".into(),
        slots: true,
        idle_reprefill: false,
        engine_status: None,
    };
    let mut w = Worker::new(eng, max_snapshots);
    w.ubatch = ubatch;
    if let Some(dir) = store {
        let id = crate::prefix_store::identity(std::path::Path::new("/m.gguf"), &w.engine.state_id().unwrap(), ubatch);
        w.store = Some(crate::prefix_store::PrefixStore::open(dir, id, 1 << 30, 64).unwrap());
    }
    Some(Arc::new(State::new(cfg, tok, w).unwrap()))
}

async fn call(st: &Arc<State<StubEngine>>, method: &str, path: &str, body: Json) -> (u16, Json, Vec<String>) {
    let b = if body.is_null() { Vec::new() } else { serde_json::to_vec(&body).unwrap() };
    match handle(st.clone(), method, path, "", &b).await {
        Reply::Json(c, j) => (c, j, vec![]),
        Reply::Stream(mut rx) => {
            let mut frames = Vec::new();
            // every wait in tests is bounded: a stream that never ends fails instead of hanging
            loop {
                match tokio::time::timeout(std::time::Duration::from_secs(60), rx.recv()).await {
                    Ok(Some(f)) => frames.push(f),
                    Ok(None) => break,
                    Err(_) => panic!("stream did not end within 60 s (a sender kept alive?)"),
                }
            }
            (200, Json::Null, frames)
        }
    }
}

fn keys(v: &Json) -> Vec<&str> {
    v.as_object().unwrap().keys().map(|s| s.as_str()).collect()
}

const TOOLS: &str = r#"[{"type":"function","function":{"name":"read_file","description":"read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"]}}}]"#;
const CALL: &str = "I'll look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/a \"b\".txt\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n</tool_call>";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn chat_nonstream_shape() {
    let Some(st) = state(&["Hello there!"], 4096) else { return };
    let (c, j, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages":[{"role":"user","content":"hi"}],"temperature":0})).await;
    assert_eq!(c, 200);
    assert_eq!(keys(&j), ["choices", "created", "model", "system_fingerprint", "object", "usage", "id", "timings"]);
    assert_eq!(j["object"], "chat.completion");
    assert!(j["id"].as_str().unwrap().starts_with("chatcmpl-") && j["id"].as_str().unwrap().len() == 41);
    let ch = &j["choices"][0];
    assert_eq!(keys(ch), ["finish_reason", "index", "message"]);
    assert_eq!(ch["finish_reason"], "stop");
    assert_eq!(ch["message"], json!({"role":"assistant","content":"Hello there!"}));
    let u = &j["usage"];
    assert_eq!(keys(u), ["completion_tokens", "prompt_tokens", "total_tokens", "prompt_tokens_details"]);
    // 3 text tokens + EOS (llama-server counts the EOG token too)
    assert_eq!(u["completion_tokens"], 4);
    assert_eq!(
        keys(&j["timings"]),
        // + the CACHE-PERFECT resume source (design sec. 7, P2b)
        ["cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second", "resume_src", "resume_ms"]
    );
    assert_eq!(j["timings"]["resume_src"], "cold");
}

fn parse_frames(frames: &[String]) -> (Vec<Json>, bool) {
    let mut out = Vec::new();
    let mut done = false;
    for f in frames {
        assert!(f.starts_with("data: ") && f.ends_with("\n\n"), "bad SSE frame {f:?}");
        let body = &f[6..f.len() - 2];
        assert!(!body.contains('\n'), "frame must be one line");
        if body == "[DONE]" {
            done = true;
            continue;
        }
        assert!(!done, "data after [DONE]");
        out.push(serde_json::from_str(body).unwrap());
    }
    (out, done)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn chat_stream_tool_call_roundtrip() {
    let Some(st) = state(&[CALL, "The file says hi."], 8192) else { return };
    let tools: Json = serde_json::from_str(TOOLS).unwrap();
    let msgs = json!([{"role":"system","content":"sys"},{"role":"user","content":"read it"}]);
    let (_, _, frames) = call(&st, "POST", "/v1/chat/completions", json!({"messages": msgs, "tools": tools, "stream": true, "temperature": 0})).await;
    let (chunks, done) = parse_frames(&frames);
    assert!(done);
    // first chunk is the role delta
    assert_eq!(chunks[0]["choices"][0]["delta"], json!({"role":"assistant","content":null}));
    for c in &chunks {
        assert_eq!(keys(c)[..6], ["choices", "created", "id", "model", "system_fingerprint", "object"]);
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], chunks[0]["id"]);
    }
    // accumulate deltas like an OpenAI client
    let mut content = String::new();
    let mut name = String::new();
    let mut args = String::new();
    let mut id = String::new();
    for c in &chunks[1..chunks.len() - 1] {
        let d = &c["choices"][0]["delta"];
        assert_eq!(c["choices"][0]["finish_reason"], Json::Null);
        if let Some(s) = d["content"].as_str() {
            content += s;
        }
        if let Some(tc) = d["tool_calls"].as_array() {
            assert_eq!(tc.len(), 1);
            assert_eq!(tc[0]["index"], 0);
            if let Some(i) = tc[0]["id"].as_str() {
                assert!(id.is_empty());
                id = i.to_string();
                assert_eq!(tc[0]["type"], "function");
            }
            if let Some(n) = tc[0]["function"]["name"].as_str() {
                name += n;
            }
            if let Some(a) = tc[0]["function"]["arguments"].as_str() {
                args += a;
            }
        }
    }
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(last["choices"][0]["delta"], json!({}));
    assert!(last.get("timings").is_some());
    assert!(last.get("usage").is_none(), "usage only with include_usage");
    assert_eq!(content, "I'll look.");
    assert_eq!(name, "read_file");
    assert_eq!(id.len(), 32);
    assert_eq!(args, r#"{"path":"/tmp/a \"b\".txt","limit":20}"#);

    // second turn: the client sends the call back with the tool result
    let n_eval_before = st.worker.lock().unwrap().engine.n_evaluated;
    let msgs2 = json!([
        {"role":"system","content":"sys"},{"role":"user","content":"read it"},
        {"role":"assistant","content":content,"tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":args}}]},
        {"role":"tool","tool_call_id":id,"content":"hi"}
    ]);
    let (c, j, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages": msgs2, "tools": tools, "temperature": 0})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["choices"][0]["message"]["content"], "The file says hi.");
    assert_eq!(j["choices"][0]["finish_reason"], "stop");
    // the re-rendered prompt shares the system+tools+user+reply prefix with turn 1
    let cached = j["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    let prompt = j["usage"]["prompt_tokens"].as_u64().unwrap();
    assert!(cached > 250 && cached < prompt, "cached {cached} of {prompt}");
    let n_eval = st.worker.lock().unwrap().engine.n_evaluated - n_eval_before;
    // only the uncached prompt tail + the decoded tokens (the final EOS is not decoded)
    assert_eq!(n_eval as u64, prompt - cached + j["usage"]["completion_tokens"].as_u64().unwrap() - 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn stream_usage_chunk_and_done() {
    let Some(st) = state(&["ok"], 4096) else { return };
    let (_, _, frames) = call(
        &st,
        "POST",
        "/chat/completions",
        json!({"messages":[{"role":"user","content":"x"}],"stream":true,"stream_options":{"include_usage":true}}),
    )
    .await;
    let (chunks, done) = parse_frames(&frames);
    assert!(done);
    let n = chunks.len();
    assert_eq!(chunks[n - 2]["choices"][0]["finish_reason"], "stop");
    assert!(chunks[n - 2].get("timings").is_none());
    assert_eq!(chunks[n - 1]["choices"], json!([]));
    assert_eq!(keys(&chunks[n - 1]), ["choices", "created", "id", "model", "system_fingerprint", "object", "usage", "timings"]);
    assert_eq!(chunks[n - 1]["usage"]["completion_tokens"], 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn completion_n_probs_and_limits() {
    let Some(st) = state(&["A B C D", "A", "x"], 4096) else { return };
    let (c, j, _) = call(&st, "POST", "/completion", json!({"prompt":"Answer: ","n_predict":1,"n_probs":3,"temperature":0})).await;
    assert_eq!(c, 200);
    assert_eq!(j["stop_type"], "limit");
    assert_eq!(j["tokens_predicted"], 1);
    assert_eq!(j["content"], "A");
    let cp = &j["completion_probabilities"];
    assert_eq!(cp.as_array().unwrap().len(), 1);
    assert_eq!(keys(&cp[0]), ["id", "token", "bytes", "logprob", "top_logprobs"]);
    assert_eq!(cp[0]["token"], "A");
    let top = cp[0]["top_logprobs"].as_array().unwrap();
    assert_eq!(top.len(), 3);
    assert_eq!(keys(&top[0]), ["id", "token", "bytes", "logprob"]);
    assert_eq!(top[0]["id"], cp[0]["id"]);
    assert!(top[0]["logprob"].as_f64().unwrap() > top[1]["logprob"].as_f64().unwrap());
    // post_sampling_probs renames the keys
    let (_, j, _) =
        call(&st, "POST", "/completion", json!({"prompt":"Answer: ","n_predict":1,"n_probs":2,"post_sampling_probs":true,"temperature":0})).await;
    assert!(j["completion_probabilities"][0].get("top_probs").is_some());
    assert!(j["completion_probabilities"][0].get("prob").is_some());
    // too long prompt
    let long = "word ".repeat(5000);
    let (c, j, _) = call(&st, "POST", "/completion", json!({"prompt": long})).await;
    assert_eq!(c, 400);
    assert_eq!(j["error"]["type"], "exceed_context_size_error");
    // streamed native completion: partial objects, final has stop:true, no [DONE]
    let (_, _, frames) = call(&st, "POST", "/completion", json!({"prompt":"q","stream":true,"temperature":0})).await;
    let (chunks, done) = parse_frames(&frames);
    assert!(!done);
    assert_eq!(chunks.last().unwrap()["stop"], true);
    assert!(chunks[..chunks.len() - 1].iter().all(|c| c["stop"] == false));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn helper_endpoints() {
    let Some(st) = state(&[], 4096) else { return };
    assert_eq!(call(&st, "GET", "/health", Json::Null).await.1, json!({"status":"ok"}));
    let (_, m, _) = call(&st, "GET", "/v1/models", Json::Null).await;
    assert_eq!(m["data"][0]["id"], "ornith");
    assert_eq!(m["object"], "list");
    let (_, t, _) = call(&st, "POST", "/tokenize", json!({"content":"<|im_start|>hi"})).await;
    let ids: Vec<u32> = t["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    assert_eq!(ids[0], 248045);
    let (_, d, _) = call(&st, "POST", "/detokenize", json!({"tokens": ids})).await;
    assert_eq!(d["content"], "<|im_start|>hi");
    let (_, p, _) = call(&st, "POST", "/tokenize", json!({"content":"hi","with_pieces":true})).await;
    assert_eq!(p["tokens"][0]["piece"], "hi");
    let (_, a, _) = call(&st, "POST", "/apply-template", json!({"messages":[{"role":"user","content":"hi"}]})).await;
    assert_eq!(a["prompt"], "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let (c, e, _) = call(&st, "GET", "/nope", Json::Null).await;
    assert_eq!(c, 404);
    assert_eq!(e["error"]["type"], "not_found_error");
    let (c, e, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages":"x"})).await;
    assert_eq!(c, 400);
    assert_eq!(keys(&e["error"]), ["code", "message", "type"]);
    let (c, _, _) = call(&st, "GET", "/props", Json::Null).await;
    assert_eq!(c, 200);
}

#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn stop_strings_and_limits() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let script = tok.encode("one two STOP three", false, true);
    let mut w = Worker::new(StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script), 4);
    let prompt = tok.encode("hello", false, true);
    let mut gp = greedy();
    gp.stop = vec!["STOP".into()];
    let mut sent = String::new();
    let r = w
        .generate(&tok, &prompt, &gp, &mut |p| {
            sent += &p.text;
            true
        })
        .unwrap();
    assert_eq!(r.stop, StopType::Word);
    assert_eq!(r.stopping_word, "STOP");
    assert_eq!(r.text, "one two ");
    assert_eq!(sent, "one two ");
    // n_predict limit
    let mut gp = greedy();
    gp.n_predict = 2;
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(r.stop, StopType::Limit);
    assert_eq!(r.n_decoded, 2);
    // cancellation from the sink
    let r = w.generate(&tok, &prompt, &greedy(), &mut |_| false);
    assert!(matches!(r, Err(crate::generate::GenError::Cancelled)));
}

#[test]
fn prefix_cache_logic() {
    // tokens are arbitrary ids; the stub validates every restore against its KV
    let mut eng = StubEngine::new(1000, 100, 99, vec![]);
    let mut c: PrefixCache<<StubEngine as Engine>::Snapshot> = PrefixCache::new(3);
    let run = |eng: &mut StubEngine, c: &mut PrefixCache<_>, prompt: &[u32], gen: &[u32]| -> usize {
        let n = c.prepare(eng, prompt, true).unwrap();
        let rest = &prompt[n..];
        if rest.len() > 1 {
            eng.prefill(&rest[..rest.len() - 1]).unwrap();
            c.appended(&rest[..rest.len() - 1]);
            c.checkpoint(eng).unwrap();
        }
        eng.prefill(&rest[rest.len() - 1..]).unwrap();
        c.appended(&rest[rest.len() - 1..]);
        for &g in gen {
            eng.decode(g).unwrap();
            c.appended(&[g]);
        }
        c.checkpoint(eng).unwrap();
        assert_eq!(eng.tokens, c.tokens);
        n
    };
    let p1: Vec<u32> = (1..=50).collect();
    assert_eq!(run(&mut eng, &mut c, &p1, &[60, 61, 62]), 0);
    // identical prompt: restore the checkpoint at len-1
    assert_eq!(run(&mut eng, &mut c, &p1, &[60]), 49);
    // agentic continuation (prompt + reply + more) continues the live state, no restore
    let mut p2 = p1.clone();
    p2.extend([60, 70, 71, 72]);
    let restores = eng.calls.iter().filter(|c| c.0 == "restore").count();
    assert_eq!(run(&mut eng, &mut c, &p2, &[80]), 51);
    assert_eq!(eng.calls.iter().filter(|c| c.0 == "restore").count(), restores);
    // diverges after the first reply token: deepest checkpoint inside the common prefix
    // (end of the second generation = p1 + [60])
    let mut p3 = p1.clone();
    p3.extend([60, 90, 91]);
    let n = run(&mut eng, &mut c, &p3, &[]);
    assert_eq!(n, 51, "checkpoints at {:?}", c.snapshot_positions());
    // diverges inside p1: no checkpoint below 10 -> start over
    let mut p5: Vec<u32> = (1..=9).collect();
    p5.extend([77, 78]);
    assert_eq!(run(&mut eng, &mut c, &p5, &[]), 0);
    // completely different prompt: start over
    let p4: Vec<u32> = (10..20).collect();
    assert_eq!(run(&mut eng, &mut c, &p4, &[]), 0);
    // cache_prompt=false always starts from scratch
    assert_eq!(c.prepare(&mut eng, &p4, false).unwrap(), 0);
    assert_eq!(eng.n_past(), 0);
    assert!(c.snapshot_positions().len() <= 3);
}

// ------------------------------------------------------------------ M3.1 speculation in the loop

/// A stub that speculates like the MTP adapter: `decode_spec` appends the token and up to
/// `depth` greedy continuations (scripted logits), `spec_rewind` drops the unused tail.
struct SpecStub {
    inner: StubEngine,
    depth: usize,
    before: Option<crate::engine::StubSnapshot>,
    appended: Vec<u32>,
    rewinds: usize,
    /// number of `decode_spec` calls, and the context position (`n_past`) of each
    calls: usize,
    at: Vec<usize>,
}

impl Engine for SpecStub {
    type Snapshot = <StubEngine as Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
    fn decode_spec(&mut self, t: u32) -> crate::engine::Result<Option<(Vec<u32>, Vec<Vec<f32>>)>> {
        self.calls += 1;
        self.at.push(self.inner.n_past());
        self.before = Some(self.inner.snapshot()?);
        let mut rows = vec![self.inner.decode(t)?.to_vec()];
        let mut extra = Vec::new();
        self.appended = vec![t];
        while extra.len() < self.depth && self.inner.n_past() + 1 < self.inner.n_ctx {
            let d = crate::sampling::argmax(rows.last().unwrap());
            rows.push(self.inner.decode(d)?.to_vec());
            extra.push(d);
            self.appended.push(d);
        }
        Ok(Some((extra, rows)))
    }
    fn spec_rewind(&mut self, keep: usize) -> crate::engine::Result<()> {
        self.rewinds += 1;
        let s = self.before.take().unwrap();
        self.inner.restore(&s)?;
        for &t in &self.appended[..keep] {
            self.inner.decode(t)?;
        }
        Ok(())
    }
}

#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn speculative_loop_matches_plain() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let script = tok.encode("one two three four five six STOP seven eight nine", false, true);
    let prompt = tok.encode("hello", false, true);
    let cases: Vec<GenParams> = {
        let mut v = vec![greedy()];
        let mut g = greedy();
        g.stop = vec!["STOP".into()];
        v.push(g);
        let mut g = greedy();
        g.n_predict = 3;
        v.push(g);
        v
    };
    for gp in &cases {
        let mut plain = Worker::new(StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone()), 4);
        let want = plain.generate(&tok, &prompt, gp, &mut |_| true).unwrap();
        for depth in 1..=3 {
            let inner = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone());
            let mut w = Worker::new(SpecStub { inner, depth, before: None, appended: vec![], rewinds: 0, calls: 0, at: vec![] }, 4);
            let got = w.generate(&tok, &prompt, gp, &mut |_| true).unwrap();
            assert_eq!(got.tokens, want.tokens, "depth {depth}");
            assert_eq!(got.text, want.text);
            assert_eq!(got.stop, want.stop);
            assert_eq!(w.engine.n_past(), plain.engine.n_past(), "depth {depth}: state length after the request");
            assert_eq!(w.engine.inner.tokens, plain.engine.tokens, "depth {depth}: state tokens");
        }
    }
    // a client that disconnects mid-stream: the state must equal the plain run's (no
    // speculative tokens left behind), and the next request must match too
    for cancel_at in 1..6 {
        let mut plain = Worker::new(StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone()), 4);
        let mut k = 0;
        let r = plain.generate(&tok, &prompt, &greedy(), &mut |_| { k += 1; k < cancel_at });
        assert!(matches!(r, Err(crate::generate::GenError::Cancelled)));
        let want = plain.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap();
        for depth in 1..=3 {
            let inner = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone());
            let mut w = Worker::new(SpecStub { inner, depth, before: None, appended: vec![], rewinds: 0, calls: 0, at: vec![] }, 4);
            let mut k = 0;
            let r = w.generate(&tok, &prompt, &greedy(), &mut |_| { k += 1; k < cancel_at });
            assert!(matches!(r, Err(crate::generate::GenError::Cancelled)));
            assert_eq!(w.engine.n_past(), w.cache.tokens.len(), "cancel at {cancel_at}, depth {depth}: engine ahead of the cache");
            let got = w.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap();
            assert_eq!(got.tokens, want.tokens, "cancel at {cancel_at}, depth {depth}: next request differs");
        }
    }
}

/// Greedy fast path (no logits row: `decode_argmax` / `decode_spec_argmax`) vs the row path
/// (forced by n_probs > 0): the same streamed partials (text, token, count), the same result,
/// and the same engine state, plain and speculative, with stop words, limits and cancels.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn greedy_fast_path_streams_identically() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let script = tok.encode("one two three \u{e9}t\u{e9} \u{65e5}\u{672c} four five six STOP seven eight nine", false, true);
    let prompt = tok.encode("hello", false, true);
    type Stream = Vec<(String, u32, usize)>;
    fn run<E: Engine>(w: &mut Worker<E>, tok: &tokenizer::Tokenizer, prompt: &[u32], gp: &GenParams, cancel_at: usize) -> (Stream, Option<(Vec<u32>, String, StopType)>) {
        let mut st = Vec::new();
        let r = w.generate(tok, prompt, gp, &mut |pt| {
            st.push((pt.text.clone(), pt.token, pt.n_decoded));
            st.len() < cancel_at
        });
        (st, r.ok().map(|r| (r.tokens, r.text, r.stop)))
    }
    let mut cases = vec![greedy()];
    let mut g = greedy();
    g.stop = vec!["STOP".into(), "\u{65e5}x".into()];
    cases.push(g);
    let mut g = greedy();
    g.n_predict = 4;
    cases.push(g);
    let mut n = 0;
    for gp in &cases {
        let mut slow = gp.clone();
        slow.n_probs = 2;
        for cancel_at in [usize::MAX, 1, 3, 6] {
            for depth in 0..=3 {
                let mk = || StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone());
                let (a, b, sa, sb) = if depth == 0 {
                    let (mut wa, mut wb) = (Worker::new(mk(), 4), Worker::new(mk(), 4));
                    let (x, y) = (run(&mut wa, &tok, &prompt, gp, cancel_at), run(&mut wb, &tok, &prompt, &slow, cancel_at));
                    (x, y, wa.engine.tokens.clone(), wb.engine.tokens.clone())
                } else {
                    let sp = |inner| SpecStub { inner, depth, before: None, appended: vec![], rewinds: 0, calls: 0, at: vec![] };
                    let (mut wa, mut wb) = (Worker::new(sp(mk()), 4), Worker::new(sp(mk()), 4));
                    let (x, y) = (run(&mut wa, &tok, &prompt, gp, cancel_at), run(&mut wb, &tok, &prompt, &slow, cancel_at));
                    (x, y, wa.engine.inner.tokens.clone(), wb.engine.inner.tokens.clone())
                };
                assert_eq!(a, b, "depth {depth}, cancel at {cancel_at}, stop {:?}", gp.stop);
                assert_eq!(sa, sb, "state, depth {depth}, cancel at {cancel_at}");
                n += 1;
            }
        }
    }
    eprintln!("greedy fast path: {n} runs identical to the row path");
}

/// M3.1 server-level exactness on a REAL model (needs the inference lease; run with
/// `OXE_TEST_MODEL=<gguf> [OXE_TEST_CPUS=0-7] cargo test --release -p server -- --ignored mtp_multiturn`):
/// an N-turn transcript where every prompt extends the previous prompt + reply + new text
/// (prefix-cache reuse, split-last checkpoint, rewinds at stop words / limits), greedy, run
/// once with MTP off and once with MTP depth 2; every turn's token ids must be identical.
#[test]
#[ignore]
fn mtp_multiturn_greedy_is_identical_to_plain() {
    let Ok(model) = std::env::var("OXE_TEST_MODEL") else {
        panic!("OXE_TEST_MODEL not set: this ignored test needs a real model");
    };
    let cpus = sched::parse_cpu_list(&std::env::var("OXE_TEST_CPUS").unwrap_or_else(|_| "0-7".into())).unwrap();
    let tok = tokenizer::Tokenizer::from_gguf(&model).unwrap();
    let sys = "You are a coding agent. Use tools. Be concise.\n".repeat(30);
    let turns = [
        "User: The add() function in calc.py returns a - b. Read the file and fix it.\nAssistant:",
        "\nTool result: def add(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\nAssistant:",
        "\nTool result: edit applied.\nUser: Now write pytest tests for add and mul.\nAssistant:",
        "\nUser: Explain in one sentence how to run them.\nAssistant:",
        "\nUser: List three edge cases worth testing, as a numbered list.\nAssistant:",
    ];
    let run = |depth: usize| -> Vec<Vec<u32>> {
        let eng = crate::oxe_engine::OxeEngine::open_full(std::path::Path::new(&model), &cpus, None, model::engine::UBATCH, 4096, depth).unwrap();
        let mut w = Worker::new(eng, 4);
        let mut prompt: Vec<u32> = tok.encode(&sys, true, true);
        let mut out = Vec::new();
        for (i, t) in turns.iter().enumerate() {
            prompt.extend(tok.encode(t, false, true));
            let mut gp = greedy();
            gp.n_predict = [40, 64, 96, 48, 80][i];
            if i % 2 == 1 {
                gp.stop = vec!["\n\n".into(), "User:".into()];
            }
            let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
            eprintln!("depth {depth} turn {i}: {} tokens, stop {:?}, cache {} prompt {}, draft {}/{}", r.tokens.len(), r.stop, r.timings.cache_n, r.timings.prompt_n, r.timings.draft_n, r.timings.draft_n_accepted);
            // the next prompt re-sends the reply, as an agent does
            prompt.extend(tok.encode(&r.text, false, true));
            out.push(r.tokens);
        }
        out
    };
    let off = run(0);
    let on = run(2);
    for (i, (a, b)) in on.iter().zip(&off).enumerate() {
        assert_eq!(a, b, "turn {i}: MTP on vs off token ids differ");
    }
}


// ------------------------------------------------------------------ agentic prefix reuse

#[test]
fn sha256_vectors() {
    use crate::prefix_store::sha256_hex;
    assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    let long = vec![b'a'; 1000];
    let mut h = crate::prefix_store::Sha256::default();
    for c in long.chunks(7) {
        h.update(c);
    }
    assert_eq!(h.hex(), sha256_hex(&long));
    assert_eq!(sha256_hex(&long), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
}

fn agent_tools() -> Json {
    json!([
      {"type": "function", "function": {"name": "read", "description": "Read a file from the workspace", "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "absolute path"}}, "required": ["filePath"]}}},
      {"type": "function", "function": {"name": "edit", "description": "Replace text in a file", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}}, "required": ["filePath", "oldString", "newString"]}}}])
}

fn agent_system() -> String {
    format!("You are opencode, an interactive CLI coding agent.\n{}", (0..12).map(|i| format!("Rule {i}: keep changes minimal and test them.")).collect::<Vec<_>>().join("\n"))
}

const READ_CALL: &str = "<tool_call>\n<function=read>\n<parameter=filePath>\n/p/calc.py\n</parameter>\n</function>\n</tool_call>";

/// OpenCode-like transcript: tool call, tool result, answer, then a new user message. The
/// Qwen3.x/Ornith templates render the assistant turns of the previous query without their
/// `<think>` blocks once a new user message follows, so that request diverges from the
/// cached tokens 4 tokens before the end of the query's first prompt (the
/// `<think>\n\n</think>\n\n` of the generation prompt). Before the query checkpoint this
/// fell back to 0 cached tokens.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn multiturn_tool_calls_reuse_cache() {
    let Some(st) = state_with(&[READ_CALL, "Fixed: add() now returns a + b.", "Run pytest."], 16384, 4, 16, None) else { return };
    // the query-checkpoint mechanism, measured without ECHO (with it turn 3 keeps every token)
    st.echo.lock().unwrap().cfg.enabled = false;
    let tools = agent_tools();
    let sys = agent_system();
    let req = |msgs: &Json| json!({"messages": msgs, "tools": tools, "temperature": 0});
    let mut msgs = json!([{"role":"system","content":sys},{"role":"user","content":"The add() function in calc.py is wrong."}]);
    let (_, j1, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
    let n1 = j1["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    let tc = &j1["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tc["function"]["name"], "read");
    // turn 2: the call goes back with its result
    msgs.as_array_mut().unwrap().extend([j1["choices"][0]["message"].clone(), json!({"role":"tool","tool_call_id":tc["id"],"content":"def add(a, b):\n    return a - b\n"})]);
    let (_, j2, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
    let c2 = j2["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert!(c2 >= n1, "turn 2 reuses prompt + reply: cache_n {c2}, first prompt {n1}");
    // turn 3: answer + a new user message re-renders turn 1's assistant message without <think>
    msgs.as_array_mut().unwrap().extend([j2["choices"][0]["message"].clone(), json!({"role":"user","content":"Now write a test."})]);
    let (_, tk, _) = call(&st, "POST", "/apply-template", json!({"messages": msgs, "tools": tools})).await;
    let p3 = st.tok.encode(tk["prompt"].as_str().unwrap(), true, true);
    let (_, tk1, _) = call(&st, "POST", "/apply-template", json!({"messages": [msgs[0].clone(), msgs[1].clone()], "tools": tools})).await;
    let p1 = st.tok.encode(tk1["prompt"].as_str().unwrap(), true, true);
    assert_eq!(p1.len(), n1);
    let div = p1.iter().zip(&p3).take_while(|(a, b)| a == b).count();
    assert_eq!(div, n1 - 4, "first difference: the generation prompt's empty think block");
    assert_eq!(String::from_utf8_lossy(&st.tok.piece(p1[div], true)), "<think>");
    let (_, j3, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
    let c3 = j3["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert!(c3 > 0 && c3 <= div && div - c3 < 16, "turn 3 resumes from the query checkpoint: cache_n {c3}, divergence {div}");
    assert_eq!(j3["choices"][0]["message"]["content"], "Run pytest.");
    // a new conversation with the same system prompt and tools resumes from the anchor
    let first_user = p1.windows(2).position(|w| w == &st.user_delim.as_ref().unwrap()[..]).unwrap();
    let (_, j4, _) = call(&st, "POST", "/v1/chat/completions", req(&json!([msgs[0].clone(), {"role":"user","content":"Something else."}]))).await;
    let c4 = j4["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert!(c4 > 0 && c4 <= first_user && first_user - c4 < 16, "new conversation: cache_n {c4}, first user message at {first_user}");
}

pub(crate) fn scratch_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("oxe-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// Save the system+tools prefix, "restart" (a new server over the same directory) and
/// resume from it: the restored request gets cache_n = the anchor and the same reply.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn prefix_store_survives_restart() {
    let dir = scratch_dir("store");
    let tools = agent_tools();
    let sys = agent_system();
    let body = json!({"messages": [{"role":"system","content":sys},{"role":"user","content":"Hello."}], "tools": tools, "temperature": 0});
    let Some(st) = state_with(&["Hi there."], 16384, 4, 16, Some(&dir)) else { return };
    let (_, a, _) = call(&st, "POST", "/v1/chat/completions", body.clone()).await;
    assert_eq!(a["timings"]["cache_n"], 0);
    st.worker.lock().unwrap().store.as_mut().unwrap().flush();
    let files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|d| d.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(files[0].ends_with(".bin") && files[0].len() == 68);
    // the stored tokens end before the first user message
    let prompt = st.tok.encode(&{
        let (_, tk, _) = call(&st, "POST", "/apply-template", body.clone()).await;
        tk["prompt"].as_str().unwrap().to_string()
    }, true, true);
    let first_user = prompt.windows(2).position(|w| w == &st.user_delim.as_ref().unwrap()[..]).unwrap();
    drop(st);
    // restart: same directory, fresh engine
    let Some(st2) = state_with(&["Hi there."], 16384, 4, 16, Some(&dir)) else { return };
    assert_eq!(st2.worker.lock().unwrap().store.as_ref().unwrap().n_entries(), 1);
    let (_, b, _) = call(&st2, "POST", "/v1/chat/completions", body.clone()).await;
    let c = b["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert_eq!(c, first_user / 16 * 16, "restored the anchor");
    assert_eq!(b["choices"][0]["message"], a["choices"][0]["message"]);
    let w = st2.worker.lock().unwrap();
    assert!(w.engine.calls.iter().any(|c| c.0 == "load" && c.2 == first_user / 16 * 16));
    assert_eq!(w.engine.tokens[..c], prompt[..c]);
    drop(w);
    // a different identity (other ubatch) does not see the entry
    let Some(st3) = state_with(&[], 16384, 4, 32, Some(&dir)) else { return };
    assert_eq!(st3.worker.lock().unwrap().store.as_ref().unwrap().n_entries(), 0);
    // a corrupt entry is discarded, and the request still succeeds from scratch
    let f = std::fs::read_dir(&dir).unwrap().flatten().next().unwrap().path();
    let len = std::fs::metadata(&f).unwrap().len();
    std::fs::OpenOptions::new().write(true).open(&f).unwrap().set_len(len - 3).unwrap();
    let Some(st4) = state_with(&["Hi there."], 16384, 4, 16, Some(&dir)) else { return };
    let (_, d, _) = call(&st4, "POST", "/v1/chat/completions", body.clone()).await;
    assert_eq!(d["timings"]["cache_n"], 0);
    assert_eq!(d["choices"][0]["message"], a["choices"][0]["message"]);
    st4.worker.lock().unwrap().store.as_mut().unwrap().flush();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Real model (`OXE_TEST_MODEL=<gguf> [OXE_TEST_CPUS=0-7] cargo test --release -p server --
/// --ignored prefix_store_real`): an OpenCode-like chat transcript (system + tools, tool
/// call, tool result, new user message) through `Worker::generate`, greedy.
/// - The prefix store entry saved by a cold run, loaded by a new engine, gives the same
///   token ids as the cold run (bit-exact resume on the ubatch grid).
/// - Every turn reuses the cache (cache_n > 0 after the first), MTP on and off give the
///   same token ids.
#[test]
#[ignore]
fn prefix_store_real_model() {
    let Ok(model) = std::env::var("OXE_TEST_MODEL") else {
        panic!("OXE_TEST_MODEL not set: this ignored test needs a real model");
    };
    let cpus = sched::parse_cpu_list(&std::env::var("OXE_TEST_CPUS").unwrap_or_else(|_| "0-7".into())).unwrap();
    let tok = tokenizer::Tokenizer::from_gguf(&model).unwrap();
    let bos = tok.bos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
    let eos = tok.eos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
    let tmpl = chat::ChatTemplate::new(tok.chat_template.as_deref().unwrap(), &bos, &eos).unwrap();
    let mut kw = serde_json::Map::new();
    kw.insert("enable_thinking".into(), Json::Bool(false));
    let opts = chat::ServerChatOptions { chat_template_kwargs: kw, enable_thinking: false, ..Default::default() };
    let delim = tok.encode("<|im_start|>user", false, true);
    let sys = format!("{}\n{}", agent_system(), "Prefer small, verifiable steps; explain what you changed.\n".repeat(40));
    let render = |msgs: &Json| -> (Vec<u32>, Option<usize>, bool) {
        let body = json!({"messages": msgs, "tools": agent_tools()});
        let inp = chat::chat_inputs(&body, &opts, &tmpl).unwrap();
        let p = tok.encode(&chat::apply(&tmpl, &inp).unwrap().prompt, true, true);
        let fu = p.windows(delim.len()).position(|w| w == &delim[..]);
        let qs = msgs.as_array().unwrap().last().unwrap()["role"] == "user";
        (p, fu, qs)
    };
    let turns: Vec<Json> = vec![
        json!([{"role":"system","content":sys},{"role":"user","content":"The add() function in calc.py is wrong. Look at the file first."}]),
        json!([{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"filePath\": \"/home/user/project/calc.py\"}"}}]},
               {"role":"tool","tool_call_id":"call_1","content":"def add(a, b):\n    return a - b\n"}]),
        json!([{"role":"assistant","content":"Fixed: add() now returns a + b."},{"role":"user","content":"Now write a pytest test file for add()."}]),
    ];
    let dir = scratch_dir("real-store");
    let open = |depth: usize| {
        let eng = crate::oxe_engine::OxeEngine::open_full(std::path::Path::new(&model), &cpus, None, model::engine::UBATCH, 8192, depth).unwrap();
        let mut w = Worker::new(eng, 4);
        let id = crate::prefix_store::identity(std::path::Path::new(&model), &w.engine.state_id().unwrap(), w.ubatch);
        w.store = Some(crate::prefix_store::PrefixStore::open(&dir, id, 4 << 30, 1024).unwrap());
        w
    };
    let run = |w: &mut Worker<crate::oxe_engine::OxeEngine>, label: &str, hints: bool| -> Vec<(Vec<u32>, usize)> {
        let mut msgs: Vec<Json> = Vec::new();
        let mut out = Vec::new();
        for t in &turns {
            msgs.extend(t.as_array().unwrap().iter().cloned());
            let (p, fu, qs) = render(&Json::Array(msgs.clone()));
            let mut gp = greedy();
            gp.n_predict = 48;
            if hints {
                gp.first_user = fu;
                gp.query_start = qs;
            }
            let r = w.generate(&tok, &p, &gp, &mut |_| true).unwrap();
            eprintln!("{label}: prompt {} cache_n {} ({:.0} ms prompt) -> {} tokens", p.len(), r.timings.cache_n, r.timings.prompt_ms, r.tokens.len());
            out.push((r.tokens, r.timings.cache_n));
        }
        w.store.as_mut().unwrap().flush();
        out
    };
    // cold (saves the anchor), then a new engine that loads it
    // no checkpoint hints (one unsplit prefill per request, no store writes): the grid splits
    // of the hinted runs must not change a single token
    let plain = {
        let eng = crate::oxe_engine::OxeEngine::open_full(std::path::Path::new(&model), &cpus, None, model::engine::UBATCH, 8192, 0).unwrap();
        let mut w = Worker::new(eng, 4);
        let id = crate::prefix_store::identity(std::path::Path::new(&model), "unused", w.ubatch);
        w.store = Some(crate::prefix_store::PrefixStore::open(&scratch_dir("real-plain"), id, 1 << 30, 1 << 30).unwrap());
        run(&mut w, "off plain", false)
    };
    let cold = run(&mut open(0), "off cold", true);
    assert_eq!(plain[0].0, cold[0].0, "grid split changed the first reply");
    let warm = run(&mut open(0), "off warm", true);
    assert!(warm[0].1 >= 1024, "first request resumed from the stored prefix: cache_n {}", warm[0].1);
    for (i, ((a, ca), (b, _))) in cold.iter().zip(&warm).enumerate() {
        assert_eq!(a, b, "turn {i}: stored-prefix resume differs from the cold run");
        if i > 0 {
            assert!(*ca > 0, "turn {i}: no cache reuse");
        }
    }
    // MTP on: its own entry (state layout differs); same tokens
    let mtp_cold = run(&mut open(2), "mtp cold", true);
    let mtp_warm = run(&mut open(2), "mtp warm", true);
    for (i, ((a, _), ((b, _), (c, _)))) in cold.iter().zip(mtp_cold.iter().zip(&mtp_warm)).enumerate() {
        assert_eq!(a, b, "turn {i}: MTP on differs from off");
        assert_eq!(a, c, "turn {i}: MTP warm differs from off");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------ WP-P1: parked sessions, turn checkpoints

/// Cache-level: two sessions alternate (agent A, agent B, A again). A's whole state comes
/// back from RAM without re-evaluating a token of it; a branch inside a parked session
/// resumes at that session's checkpoint; the RAM budget evicts the least recently used.
#[test]
fn parked_sessions_logic() {
    let mut eng = StubEngine::new(1000, 400, 399, vec![]);
    let mut c: PrefixCache<<StubEngine as Engine>::Snapshot> = PrefixCache::new(8);
    c.park_budget = 1 << 20;
    c.park_min = 8;
    // mimics Worker::generate: prepare, prefill to len-1 + checkpoint, last token, decode, checkpoint
    let run = |eng: &mut StubEngine, c: &mut PrefixCache<_>, prompt: &[u32], gen: &[u32]| -> usize {
        let n = c.prepare(eng, prompt, true).unwrap();
        let rest = &prompt[n..];
        if rest.len() > 1 {
            eng.prefill(&rest[..rest.len() - 1]).unwrap();
            c.appended(&rest[..rest.len() - 1]);
            c.checkpoint(eng).unwrap();
        }
        eng.prefill(&rest[rest.len() - 1..]).unwrap();
        c.appended(&rest[rest.len() - 1..]);
        c.prompt_len = prompt.len();
        for &g in gen {
            eng.decode(g).unwrap();
            c.appended(&[g]);
        }
        c.checkpoint(eng).unwrap();
        assert_eq!(eng.tokens, c.tokens);
        n
    };
    let a1: Vec<u32> = (1..=50).collect();
    assert_eq!(run(&mut eng, &mut c, &a1, &[60, 61]), 0);
    // agent B: A (52 tokens) is parked
    let b1: Vec<u32> = (200..=230).collect();
    assert_eq!(run(&mut eng, &mut c, &b1, &[240]), 0);
    assert_eq!(c.parked_lengths(), vec![52]);
    // A continues: the parked state comes back whole, only the new tokens are evaluated
    let mut a2 = a1.clone();
    a2.extend([60, 61, 70, 71, 72]);
    let ev = eng.n_evaluated;
    assert_eq!(run(&mut eng, &mut c, &a2, &[]), 52);
    assert_eq!(eng.n_evaluated - ev, 3);
    assert!(eng.calls.iter().any(|x| x.0 == "load" && x.2 == 52));
    assert_eq!(c.parked_lengths(), vec![32], "B is parked now");
    assert_eq!(c.stats.unparks, 1);
    // B continues likewise
    let mut b2 = b1.clone();
    b2.extend([240, 250]);
    assert_eq!(run(&mut eng, &mut c, &b2, &[]), 32);
    // a branch inside parked A (a retry of A's first request): resumes at A's checkpoint 49,
    // and the parked copy of A stays for its own continuation
    let mut a3 = a1[..49].to_vec();
    a3.extend([90, 91]);
    assert_eq!(run(&mut eng, &mut c, &a3, &[]), 49);
    assert!(c.parked_lengths().contains(&55), "{:?}", c.parked_lengths());
    // budget: barely more than what is parked now -> parking the live session evicts the LRU one
    c.park_budget = c.parked_bytes() + 50;
    let other: Vec<u32> = (300..=340).collect();
    let before = c.stats.park_evictions;
    assert_eq!(run(&mut eng, &mut c, &other, &[]), 0);
    assert!(c.stats.park_evictions > before);
    assert!(c.parked_bytes() <= c.park_budget);
    // parking off: a switch keeps nothing
    c.park_budget = 0;
    let n_parked = c.parked_lengths().len();
    let fresh: Vec<u32> = (341..=380).collect();
    assert_eq!(run(&mut eng, &mut c, &fresh, &[]), 0);
    assert_eq!(c.parked_lengths().len(), n_parked);
}

/// OpenCode-like: the build agent's session, then a title request (another system prompt, no
/// tools), then the build agent's next turn. Parked: the next turn re-evaluates only the
/// tool result, not the whole conversation.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn agent_switch_resumes_parked_session() {
    let Some(st) = state_with(&[READ_CALL, "A title", "Fixed it."], 16384, 8, 16, None) else { return };
    let ledger = scratch_dir("ledger").with_extension("jsonl");
    {
        let mut w = st.worker.lock().unwrap();
        w.cache.park_budget = 1 << 30;
        w.cache.park_min = 16;
        w.ledger = Some(std::fs::OpenOptions::new().create(true).append(true).open(&ledger).unwrap());
    }
    let tools = agent_tools();
    let sys = agent_system();
    let req = |msgs: &Json| json!({"messages": msgs, "tools": tools, "temperature": 0});
    let mut msgs = json!([{"role":"system","content":sys},{"role":"user","content":"The add() function in calc.py is wrong."}]);
    let (_, j1, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
    let n1 = j1["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    let live = st.worker.lock().unwrap().cache.tokens.len();
    // the title agent
    let (_, jt, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages":[{"role":"system","content":"You are a title generator. Reply with a short title only."},{"role":"user","content":"The add() function in calc.py is wrong."}],"temperature":0})).await;
    assert_eq!(jt["choices"][0]["message"]["content"], "A title");
    assert_eq!(st.worker.lock().unwrap().cache.parked_lengths(), vec![live]);
    // build agent, turn 2: resumes the parked session at its full length
    let tc = j1["choices"][0]["message"]["tool_calls"][0].clone();
    msgs.as_array_mut().unwrap().extend([j1["choices"][0]["message"].clone(), json!({"role":"tool","tool_call_id":tc["id"],"content":"def add(a, b):\n    return a - b\n"})]);
    let (_, j2, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
    let c2 = j2["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert!(c2 >= n1, "turn 2 after the title request reuses prompt + reply: cache_n {c2}, first prompt {n1}");
    assert_eq!(j2["choices"][0]["message"]["content"], "Fixed it.");
    let w = st.worker.lock().unwrap();
    assert_eq!(w.cache.stats.unparks, 1);
    assert_eq!(w.cache.stats.parks, 2, "the title session is parked in turn");
    drop(w);
    // the ledger: one line per generation, numbers and enum labels only
    let lines: Vec<Json> = std::fs::read_to_string(&ledger).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0]["cache_src"], "none");
    assert_eq!(lines[1]["parks"], 1, "{}", lines[1]);
    assert_eq!(lines[2]["cache_src"], "parked");
    assert_eq!(lines[2]["unparks"], 1);
    assert_eq!(lines[2]["cache_n"].as_u64().unwrap() as usize, c2);
    assert!(!std::fs::read_to_string(&ledger).unwrap().contains("calc.py"), "no content in the ledger");
    let _ = std::fs::remove_file(&ledger);
}

/// Turn checkpoints: a fresh 10-message conversation takes checkpoints at the grid floors of
/// the newest message starts; an edit of the last user message resumes at the floor of
/// its start instead of the anchor.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn turn_checkpoints_resume_edit() {
    let Some(st) = state_with(&["Done.", "Done again."], 16384, 8, 16, None) else { return };
    let sys = agent_system();
    let para = |i: usize| (0..12).map(|k| format!("Point {i}.{k}: the parser must accept ports 1 to 65535.")).collect::<Vec<_>>().join(" ");
    let mut msgs = vec![json!({"role":"system","content":sys})];
    for i in 0..4 {
        msgs.push(json!({"role":"user","content":para(i)}));
        msgs.push(json!({"role":"assistant","content":format!("Noted {i}.")}));
    }
    msgs.push(json!({"role":"user","content":"Now fix it."}));
    let (_, j1, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages": msgs, "temperature": 0})).await;
    assert_eq!(j1["choices"][0]["message"]["content"], "Done.");
    let (_, tk, _) = call(&st, "POST", "/apply-template", json!({"messages": msgs})).await;
    let prompt = st.tok.encode(tk["prompt"].as_str().unwrap(), true, true);
    let im = st.user_delim.as_ref().unwrap()[0];
    let starts: Vec<usize> = prompt.iter().enumerate().filter(|(_, &t)| t == im).map(|(i, _)| i).collect();
    let last_user = starts[starts.len() - 2];
    let info = st.worker.lock().unwrap().cache.snapshot_info();
    let turns: Vec<usize> = info.iter().filter(|x| x.1 == crate::cache::PRIO_TURN).map(|x| x.0).collect();
    assert!(!turns.is_empty() && turns.len() <= 4, "{info:?}");
    assert!(turns.contains(&(last_user / 16 * 16)), "turn checkpoint below the last user message: {info:?}, start {last_user}");
    // edit the last user message
    *msgs.last_mut().unwrap() = json!({"role":"user","content":"Now fix it, and add a test."});
    let (_, j2, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages": msgs, "temperature": 0})).await;
    assert_eq!(j2["timings"]["cache_n"].as_u64().unwrap() as usize, last_user / 16 * 16);
    assert_eq!(j2["choices"][0]["message"]["content"], "Done again.");
}

/// Session cache: the build agent's conversation is written at shutdown and a new server
/// (unload, model switch) continues it from disk: only the new tool result is evaluated.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn session_cache_survives_restart() {
    let dir = scratch_dir("sessions");
    let open = |st: &Arc<State<StubEngine>>| {
        let mut w = st.worker.lock().unwrap();
        let id = crate::prefix_store::identity(std::path::Path::new("/m.gguf"), &w.engine.state_id().unwrap(), 16);
        w.sessions = Some(crate::prefix_store::PrefixStore::open(&dir, id, 1 << 30, 16).unwrap());
        w.cache.park_min = 16;
        w.cache.park_budget = 1 << 30;
        w.cache.spill = true;
    };
    let tools = agent_tools();
    let sys = agent_system();
    let req = |msgs: &Json| json!({"messages": msgs, "tools": tools, "temperature": 0});
    let mut msgs = json!([{"role":"system","content":sys},{"role":"user","content":"The add() function in calc.py is wrong."}]);
    let Some(st) = state_with(&[READ_CALL], 16384, 8, 16, None) else { return };
    open(&st);
    let (_, j1, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
    let live = st.worker.lock().unwrap().cache.tokens.len();
    assert!(st.worker.lock().unwrap().persist_sessions() >= 1);
    drop(st);
    let Some(st2) = state_with(&["Fixed it."], 16384, 8, 16, None) else { return };
    open(&st2);
    let tc = j1["choices"][0]["message"]["tool_calls"][0].clone();
    msgs.as_array_mut().unwrap().extend([j1["choices"][0]["message"].clone(), json!({"role":"tool","tool_call_id":tc["id"],"content":"def add(a, b):\n    return a - b\n"})]);
    let (_, j2, _) = call(&st2, "POST", "/v1/chat/completions", req(&msgs)).await;
    assert_eq!(j2["timings"]["cache_n"].as_u64().unwrap() as usize, live, "resumed the whole session from disk");
    assert_eq!(j2["choices"][0]["message"]["content"], "Fixed it.");
    assert!(st2.worker.lock().unwrap().engine.calls.iter().any(|c| c.0 == "load" && c.2 == live));
    // files are private
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
    for f in std::fs::read_dir(&dir).unwrap().flatten() {
        assert_eq!(f.metadata().unwrap().permissions().mode() & 0o777, 0o600);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Real model, WP-P1 G-exact (`OXE_TEST_MODEL=<gguf> [OXE_TEST_CPUS=0-7] [OXE_TEST_MTP=2]
/// cargo test --release -p server -- --ignored parked_sessions_real`):
/// - reference: the 3-turn agent transcript uninterrupted, 4 checkpoints (no turn checkpoints);
/// - test: 8 checkpoints (turn checkpoints on) + parking, with a title request (other system
///   prompt) before turns 2 and 3. Every turn's token ids equal the reference's; turns 2-3
///   resume the parked session at least as deep as the uninterrupted session does; a
///   full-length unpark gives a state
///   byte-identical to the one that was parked.
#[test]
#[ignore]
fn parked_sessions_real_model() {
    let Ok(model) = std::env::var("OXE_TEST_MODEL") else {
        panic!("OXE_TEST_MODEL not set: this ignored test needs a real model");
    };
    let cpus = sched::parse_cpu_list(&std::env::var("OXE_TEST_CPUS").unwrap_or_else(|_| "0-7".into())).unwrap();
    let depth: usize = std::env::var("OXE_TEST_MTP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let tok = tokenizer::Tokenizer::from_gguf(&model).unwrap();
    let bos = tok.bos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
    let eos = tok.eos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
    let tmpl = chat::ChatTemplate::new(tok.chat_template.as_deref().unwrap(), &bos, &eos).unwrap();
    let mut kw = serde_json::Map::new();
    kw.insert("enable_thinking".into(), Json::Bool(false));
    let opts = chat::ServerChatOptions { chat_template_kwargs: kw, enable_thinking: false, ..Default::default() };
    let delim = tok.encode("<|im_start|>user", false, true);
    let sys = format!("{}\n{}", agent_system(), "Prefer small, verifiable steps; explain what you changed.\n".repeat(40));
    let render = |msgs: &Json, tools: bool| -> (Vec<u32>, GenParams) {
        let body = if tools { json!({"messages": msgs, "tools": agent_tools()}) } else { json!({"messages": msgs}) };
        let inp = chat::chat_inputs(&body, &opts, &tmpl).unwrap();
        let p = tok.encode(&chat::apply(&tmpl, &inp).unwrap().prompt, true, true);
        let mut gp = greedy();
        gp.n_predict = 48;
        gp.first_user = p.windows(delim.len()).position(|w| w == &delim[..]);
        gp.query_start = msgs.as_array().unwrap().last().unwrap()["role"] == "user";
        gp.msg_starts = p.iter().enumerate().filter(|(_, &t)| t == delim[0]).map(|(i, _)| i).collect();
        (p, gp)
    };
    let turns: Vec<Json> = vec![
        json!([{"role":"system","content":sys},{"role":"user","content":"The add() function in calc.py is wrong. Look at the file first."}]),
        json!([{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read","arguments":"{\"filePath\": \"/home/user/project/calc.py\"}"}}]},
               {"role":"tool","tool_call_id":"call_1","content":"def add(a, b):\n    return a - b\n"}]),
        json!([{"role":"assistant","content":"Fixed: add() now returns a + b."},{"role":"user","content":"Now write a pytest test file for add()."}]),
    ];
    let title = json!([{"role":"system","content":"You are a title generator. Reply with a title of at most 8 words."},{"role":"user","content":"The add() function in calc.py is wrong."}]);
    let open = |ckpt: usize| {
        let eng = crate::oxe_engine::OxeEngine::open_full(std::path::Path::new(&model), &cpus, None, model::engine::UBATCH, 8192, depth).unwrap();
        Worker::new(eng, ckpt)
    };
    // (token ids, cache_n) of the uninterrupted session
    let reference: Vec<(Vec<u32>, usize)> = {
        let mut w = open(4);
        let mut msgs: Vec<Json> = Vec::new();
        turns
            .iter()
            .map(|t| {
                msgs.extend(t.as_array().unwrap().iter().cloned());
                let (p, gp) = render(&Json::Array(msgs.clone()), true);
                let r = w.generate(&tok, &p, &gp, &mut |_| true).unwrap();
                (r.tokens, r.timings.cache_n)
            })
            .collect()
    };
    let mut w = open(8);
    w.cache.park_budget = 2 << 30;
    w.cache.park_min = 64;
    let mut msgs: Vec<Json> = Vec::new();
    for (i, t) in turns.iter().enumerate() {
        msgs.extend(t.as_array().unwrap().iter().cloned());
        let (p, gp) = render(&Json::Array(msgs.clone()), true);
        if i > 0 {
            // the session as it was before the title request
            let mut before = Vec::new();
            w.engine.save_state(&mut before).unwrap();
            let session = w.cache.tokens.clone();
            let (tp, tg) = render(&title, false);
            let r = w.generate(&tok, &tp, &tg, &mut |_| true).unwrap();
            eprintln!("title: prompt {} cache_n {} -> {} tokens", tp.len(), r.timings.cache_n, r.tokens.len());
            assert_eq!(w.cache.parked_lengths().len(), 1, "the agent session is parked");
            // unpark at full length (a prompt that continues the parked session): the state
            // must be byte-identical to the one that was parked
            let mut probe = session.clone();
            probe.push(p[0]);
            let n = w.cache.prepare(&mut w.engine, &probe, true).unwrap();
            assert_eq!(n, session.len(), "turn {i}: full-length unpark");
            let mut after = Vec::new();
            w.engine.save_state(&mut after).unwrap();
            assert!(before == after, "turn {i}: unparked state differs from the parked one ({} vs {} bytes)", before.len(), after.len());
        }
        let r = w.generate(&tok, &p, &gp, &mut |_| true).unwrap();
        eprintln!("parked run turn {i}: prompt {} cache_n {} ({:.0} ms prompt), checkpoints {:?}", p.len(), r.timings.cache_n, r.timings.prompt_ms, w.cache.snapshot_info());
        assert_eq!(r.tokens, reference[i].0, "turn {i}: token ids differ from the uninterrupted session");
        // The scripted transcript re-renders fixed assistant messages, not the model's own
        // replies, so a prompt can diverge inside (or before) the parked reply; the resume point
        // is then the deepest checkpoint below the divergence (len-1, query or anchor). Resume
        // equality is defined against the uninterrupted session on the same schedule.
        if i > 0 {
            assert!(r.timings.cache_n >= reference[i].1, "turn {i}: cache_n {} < uninterrupted {}", r.timings.cache_n, reference[i].1);
        }
    }
    eprintln!("park stats: {:?}", w.cache.stats);
}

// ------------------------------------------------------------------ msg_split: message-boundary prefill schedule

/// Stub worker for the message-boundary schedule: `msg_split` on, a 4096-token ubatch (so the
/// grid would have no position inside these short prompts, and any split is the schedule's).
pub(crate) fn stub_worker(tok: &tokenizer::Tokenizer, max_snapshots: usize) -> Worker<StubEngine> {
    let eng = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), tok.encode("default reply", false, true));
    let mut w = Worker::new(eng, max_snapshots);
    w.ubatch = 4096;
    w.msg_split = true;
    w
}

fn msg_params(first_user: Option<usize>, query: bool, starts: &[usize]) -> GenParams {
    GenParams { first_user, query_start: query, msg_starts: starts.to_vec(), ..greedy() }
}

/// (n_past before, n_tokens) of the prefill calls logged from index `from` on.
pub(crate) fn prefills(calls: &[(&'static str, usize, usize)], from: usize) -> Vec<(usize, usize)> {
    calls[from..].iter().filter(|c| c.0 == "prefill").map(|c| (c.1, c.2)).collect()
}

/// A prefix store for `w`'s identity; 8 tokens minimum so the 17-token anchor qualifies.
fn msg_store(w: &Worker<StubEngine>, dir: &std::path::Path) -> crate::prefix_store::PrefixStore {
    let id = crate::prefix_store::identity(std::path::Path::new("/m.gguf"), &w.engine.state_id().unwrap(), w.ubatch);
    crate::prefix_store::PrefixStore::open(dir, id, 1 << 30, 8).unwrap()
}

/// A 40-token prompt with messages at 0, 17 (first user) and 30 (last user, the query).
/// With `msg_split` the prefill is split at every message start and the anchor is exactly at
/// the first user message (17; the grid floor with a 4096-token ubatch would be 0). Without
/// `msg_split` the same prompt is one unsplit prefill and gets no anchor.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn msg_split_anchor_is_first_user() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let prompt: Vec<u32> = (100..140).collect();
    let gp = msg_params(Some(17), true, &[0, 17, 30]);
    let mut w = stub_worker(&tok, 8);
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(prefills(&w.engine.calls, 0), vec![(0, 17), (17, 13), (30, 9), (39, 1)]);
    let info = w.cache.snapshot_info();
    assert!(info.contains(&(17, crate::cache::PRIO_ANCHOR)), "{info:?}");
    assert!(info.contains(&(30, crate::cache::PRIO_QUERY)), "{info:?}");
    assert!(info.contains(&(39, crate::cache::PRIO_TAIL)), "{info:?}");
    // end of generation: prompt + the decoded tokens but the last one (never fed to the engine)
    assert!(info.contains(&(40 + r.n_decoded - 1, crate::cache::PRIO_TAIL)), "{info:?}");

    // old behaviour: one unsplit prefill to n-1 and no anchor. At ubatch 4096 the grid has no
    // position inside the prompt, so there is nothing to checkpoint before the tail.
    let mut g = stub_worker(&tok, 8);
    g.msg_split = false;
    g.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(prefills(&g.engine.calls, 0), vec![(0, 39), (39, 1)]);
    assert!(!g.cache.snapshot_info().iter().any(|x| x.1 == crate::cache::PRIO_ANCHOR));
}

/// The anchor of request 1 is restored for a second question with the same system prompt; the
/// prefill resumes at the anchor (cache_n 17) and keeps the message-boundary splits.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn msg_split_resident_second_question() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let p1: Vec<u32> = (100..140).collect();
    let gp = msg_params(Some(17), true, &[0, 17, 30]);
    let mut w = stub_worker(&tok, 8);
    w.generate(&tok, &p1, &gp, &mut |_| true).unwrap();
    // the same system prompt and the first two tokens of the question, then other tokens
    let mut p2 = p1[..19].to_vec();
    p2.extend(300..321);
    assert_eq!(p2.len(), 40);
    let mark = w.engine.calls.len();
    let r = w.generate(&tok, &p2, &gp, &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, 17);
    assert!(w.engine.calls[mark..].iter().any(|c| c.0 == "restore" && c.2 == 17));
    assert_eq!(prefills(&w.engine.calls, mark), vec![(17, 13), (30, 9), (39, 1)]);
}

/// A process that loads the anchor from the prefix store computes the same prefill schedule
/// from 17 on as a cold process does for the same prompt (restored == computed).
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn msg_split_cold_and_store_loaded_schedules_match() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let dir = scratch_dir("msg-store");
    let p1: Vec<u32> = (100..140).collect();
    // the same system prompt (17 tokens), another question
    let mut p2 = p1[..17].to_vec();
    p2.extend(300..323);
    assert_eq!(p2.len(), 40);
    let gp = msg_params(Some(17), true, &[0, 17, 30]);
    // worker A: cold, stores the anchor
    let mut a = stub_worker(&tok, 8);
    let st = msg_store(&a, &dir);
    a.store = Some(st);
    a.generate(&tok, &p1, &gp, &mut |_| true).unwrap();
    a.store.as_mut().unwrap().flush();
    assert_eq!(a.store.as_ref().unwrap().n_entries(), 1);
    // worker B: a new engine on the same store loads the anchor, then computes the question
    let mut b = stub_worker(&tok, 8);
    let st = msg_store(&b, &dir);
    b.store = Some(st);
    let rb = b.generate(&tok, &p2, &gp, &mut |_| true).unwrap();
    assert_eq!(rb.timings.cache_n, 17);
    assert!(b.engine.calls.iter().any(|c| c.0 == "load" && c.1 == 0 && c.2 == 17));
    let from_b = prefills(&b.engine.calls, 0);
    // worker C: cold, no store
    let mut c = stub_worker(&tok, 8);
    c.generate(&tok, &p2, &gp, &mut |_| true).unwrap();
    let from_c = prefills(&c.engine.calls, 0);
    assert_eq!(from_c[0], (0, 17));
    assert_eq!(from_c[1..].to_vec(), from_b);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Turn checkpoints: six message starts after the anchor are all split points, but only the two
/// newest of the non-anchor, non-query ones keep a PRIO_TURN checkpoint.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn msg_split_turn_checkpoints_newest_win() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let prompt: Vec<u32> = (100..180).collect();
    // messages at 0, 10 (first user), 20, 30, 40, 50, 60, 70 (last user, the query); n-1 = 79
    let gp = msg_params(Some(10), true, &[0, 10, 20, 30, 40, 50, 60, 70]);
    let mut w = stub_worker(&tok, 12);
    w.turn_checkpoints = 2;
    w.cache.max_turn = 2;
    w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(
        prefills(&w.engine.calls, 0),
        vec![(0, 10), (10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (60, 10), (70, 9), (79, 1)]
    );
    let info = w.cache.snapshot_info();
    let turns: Vec<usize> = info.iter().filter(|x| x.1 == crate::cache::PRIO_TURN).map(|x| x.0).collect();
    assert_eq!(turns, vec![50, 60], "{info:?}");
    assert!(info.contains(&(10, crate::cache::PRIO_ANCHOR)), "{info:?}");
    assert!(info.contains(&(70, crate::cache::PRIO_QUERY)), "{info:?}");
}

/// A prompt that resumes at a tail checkpoint (n_cached 39, not a message start) is not
/// canonical: the anchor it takes at 45 is not written to the prefix store (store entries
/// must come from a state prefilled on the schedule of a cold process). Only request 1's
/// anchor is stored.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn msg_split_noncanonical_resume_saves_no_store_entry() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let dir = scratch_dir("msg-noncanon");
    let mut w = stub_worker(&tok, 8);
    let st = msg_store(&w, &dir);
    w.store = Some(st);
    let p1: Vec<u32> = (100..140).collect();
    w.generate(&tok, &p1, &msg_params(Some(17), true, &[0, 17, 30]), &mut |_| true).unwrap();
    w.store.as_mut().unwrap().flush();
    assert_eq!(w.store.as_ref().unwrap().n_entries(), 1, "request 1 stores its anchor");
    // request 2 shares the first 39 tokens (request 1's tail checkpoint, not a message start),
    // then a user message starts at 45. The anchor taken at 45 is not stored.
    let mut p2 = p1[..39].to_vec();
    p2.extend(400..421);
    assert_eq!(p2.len(), 60);
    let r = w.generate(&tok, &p2, &msg_params(Some(45), false, &[0, 17, 30, 45]), &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, 39);
    assert!(w.cache.snapshot_info().contains(&(45, crate::cache::PRIO_ANCHOR)), "{:?}", w.cache.snapshot_info());
    w.store.as_mut().unwrap().flush();
    assert_eq!(w.store.as_ref().unwrap().n_entries(), 1, "no entry from a non-canonical state");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Warm-up hint file: written at the end of a request (then at most every 60 s unless forced),
/// handed to the engine by a new process with the same identity, invisible to another
/// identity, and rejected when truncated or garbage. Mode 0600.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn warm_file_roundtrip() {
    use crate::generate::WarmFile;
    use std::os::unix::fs::PermissionsExt;
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let dir = scratch_dir("warm");
    let path = dir.join("hint.bin");
    let prompt: Vec<u32> = (100..110).collect();
    let mut w = stub_worker(&tok, 8);
    w.warm = Some(WarmFile::new(path.clone(), "id-A".into()));
    assert!(w.load_warm().is_none(), "no file yet");
    w.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap();
    assert!(path.exists(), "generate saves the hint at its end");
    // within the 60 s interval a request does not write again; a forced save does
    assert!(!w.save_warm(false));
    assert!(w.save_warm(true));
    assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
    // a new process with the same identity hands the hint to its engine
    let mut v = stub_worker(&tok, 8);
    v.warm = Some(WarmFile::new(path.clone(), "id-A".into()));
    assert!(v.load_warm().is_some());
    assert!(v.engine.calls.iter().any(|c| c.0 == "warm" && c.1 == 0 && c.2 == 8));
    // another identity reads nothing
    let mut o = stub_worker(&tok, 8);
    o.warm = Some(WarmFile::new(path.clone(), "id-B".into()));
    assert!(o.load_warm().is_none());
    assert!(o.engine.calls.is_empty());
    // a truncated (inside the header) or garbage file reads nothing
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..10]).unwrap();
    let mut t = stub_worker(&tok, 8);
    t.warm = Some(WarmFile::new(path.clone(), "id-A".into()));
    assert!(t.load_warm().is_none());
    std::fs::write(&path, b"not a warm file at all").unwrap();
    let mut u = stub_worker(&tok, 8);
    u.warm = Some(WarmFile::new(path.clone(), "id-A".into()));
    assert!(u.load_warm().is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

// ------------------------------------------------------------------ ENGINE-AUDIT F1 / F7

/// A stub that raises a cancel flag on its `at`-th prefill or decode call (the client leaves).
pub(crate) struct CancelStub {
    pub(crate) inner: StubEngine,
    pub(crate) flag: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) at: usize,
    pub(crate) calls: usize,
}

impl Engine for CancelStub {
    type Snapshot = <StubEngine as Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        if self.calls == self.at {
            self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        if self.calls == self.at {
            self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
    fn state_id(&self) -> Option<String> {
        self.inner.state_id()
    }
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> crate::engine::Result<()> {
        self.inner.save_state(w)
    }
    fn load_state(&mut self, r: &mut dyn std::io::Read, n: usize) -> crate::engine::Result<()> {
        self.inner.load_state(r, n)
    }
}

/// Cancel mid-prefill (non-streamed request, no partials yet): generate stops at the next
/// chunk boundary with a consistent state (engine tokens == cache tokens == a prompt prefix
/// on the grid); a retry resumes from it and gives the same reply as an uncancelled run.
/// Cancel mid-decode: stops at once, state consistent.
/// F4: an engine that stops INSIDE a prefill call (Flash's abort hook): the stub raises the flag during its `at`-th
/// prefill call, rolls itself back to that call's start and returns the CANCELLED_MARK error, as FlashEngine does.
pub(crate) struct MidCallCancelStub {
    pub(crate) inner: StubEngine,
    pub(crate) flag: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) at: usize,
    pub(crate) calls: usize,
    /// what generate registered (Some during a request, None after)
    pub(crate) registered: Option<Arc<std::sync::atomic::AtomicBool>>,
}

impl Engine for MidCallCancelStub {
    type Snapshot = <StubEngine as Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn set_cancel(&mut self, f: Option<Arc<std::sync::atomic::AtomicBool>>) {
        self.registered = f;
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        if self.calls == self.at && t.len() > 1 {
            assert!(self.registered.is_some(), "generate registers the cancel flag before prefilling");
            self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
            // the abort hit mid-call and the engine rolled back to the call's start: its state is as before the call
            // (FlashEngine restores the recurrent snapshot; the StubEngine's own snapshot/restore does not cover its
            // script cursor, so the stub models the rolled-back state by not touching it)
            return Err(crate::engine::EngineError(format!("{} stub abort", crate::engine::CANCELLED_MARK)));
        }
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
    fn state_id(&self) -> Option<String> {
        self.inner.state_id()
    }
    fn save_state(&mut self, w: &mut dyn std::io::Write) -> crate::engine::Result<()> {
        self.inner.save_state(w)
    }
    fn load_state(&mut self, r: &mut dyn std::io::Read, n: usize) -> crate::engine::Result<()> {
        self.inner.load_state(r, n)
    }
}

/// F4: a cancel inside a prefill call is a Cancelled result (not an engine error that drops the session), engine and
/// cache agree on a prompt prefix that ends at a call boundary, the flag is unregistered afterwards, and a retry
/// resumes and replies as an uncancelled run.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn cancel_inside_a_prefill_call_rolls_back_to_the_call_start() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let u = 16;
    let reference = {
        let mut w = Worker::new(StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), 4);
        w.ubatch = u;
        w.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap().tokens
    };
    let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let eng = MidCallCancelStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), flag: flag.clone(), at: 3, calls: 0, registered: None };
    let mut w = Worker::new(eng, 4);
    w.ubatch = u;
    let mut gp = greedy();
    gp.cancel = Some(flag.clone());
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true);
    assert!(matches!(r, Err(crate::generate::GenError::Cancelled)), "a mid-call abort is a cancel");
    assert!(w.engine.registered.is_none(), "flag unregistered after the request");
    let n = w.cache.tokens.len();
    assert_eq!(w.engine.inner.tokens, w.cache.tokens, "engine and cache agree after a mid-call cancel");
    assert_eq!(n, 2 * u * crate::generate::CANCEL_UBATCHES, "the 3rd call rolled back: two calls kept");
    assert_eq!(w.cache.tokens[..], prompt[..n]);
    flag.store(false, std::sync::atomic::Ordering::Relaxed);
    w.engine.at = usize::MAX;
    let r2 = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(r2.timings.cache_n, n);
    assert_eq!(r2.tokens, reference);
}

/// Cancel mid-prefill (non-streamed request, no partials yet): generate stops at the next
/// chunk boundary with a consistent state (engine tokens == cache tokens == a prompt prefix
/// on the grid); a retry resumes from it and gives the same reply as an uncancelled run.
/// Cancel mid-decode: stops at once, state consistent.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn cancel_mid_prefill_and_decode() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let u = 16;
    let reference = {
        let mut w = Worker::new(StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), 4);
        w.ubatch = u;
        w.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap().tokens
    };
    let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let eng = CancelStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), flag: flag.clone(), at: 3, calls: 0 };
    let mut w = Worker::new(eng, 4);
    w.ubatch = u;
    let mut gp = greedy();
    gp.cancel = Some(flag.clone());
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true);
    assert!(matches!(r, Err(crate::generate::GenError::Cancelled)));
    let n = w.cache.tokens.len();
    assert_eq!(w.engine.inner.tokens, w.cache.tokens, "engine and cache agree after a cancel");
    assert_eq!(n, 3 * u * crate::generate::CANCEL_UBATCHES, "stopped after the 3rd chunk");
    assert_eq!(w.cache.tokens[..], prompt[..n]);
    // the retry resumes the partial prefill and replies as the uncancelled run did
    flag.store(false, std::sync::atomic::Ordering::Relaxed);
    w.engine.at = usize::MAX;
    let r2 = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(r2.timings.cache_n, n);
    assert_eq!(r2.tokens, reference);
    // cancel during decode: the 2nd decode step raises the flag
    let calls = w.engine.calls;
    w.engine.at = calls + 2 + 1;
    let mut p3 = prompt.clone();
    p3.push(7);
    let r3 = w.generate(&tok, &p3, &gp, &mut |_| true);
    assert!(matches!(r3, Err(crate::generate::GenError::Cancelled)));
    assert_eq!(w.engine.inner.tokens, w.cache.tokens);
    assert!(w.cache.tokens.len() <= p3.len() + 2);
}

/// F7: constrained output is refused with 400 instead of being ignored.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn structured_output_is_refused() {
    let Some(st) = state(&["x"], 4096) else { return };
    for extra in [json!({"response_format": {"type": "json_object"}}), json!({"json_schema": {"type": "object"}}), json!({"grammar": "root ::= \"a\""})] {
        let mut body = json!({"messages":[{"role":"user","content":"hi"}]});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let (c, j, _) = call(&st, "POST", "/v1/chat/completions", body).await;
        assert_eq!(c, 400, "{j}");
        assert!(j["error"]["message"].as_str().unwrap().contains("not supported"));
    }
    // response_format text is the default and stays accepted
    let (c, _, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages":[{"role":"user","content":"hi"}], "response_format": {"type": "text"}})).await;
    assert_eq!(c, 200);
    let (c, _, _) = call(&st, "POST", "/completion", json!({"prompt": "hi", "grammar": "root ::= \"a\""})).await;
    assert_eq!(c, 400);
}

/// Ticket 031 failure (Codex): after an idle unload the resumed session reused 0 tokens when
/// the next request was a new user message: the template re-renders the previous query
/// without its empty think block, so that prompt diverges inside the saved session and the
/// full-length entry is no prefix of it. The query checkpoint is now persisted too: reuse
/// after the restart equals the uninterrupted session's.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn session_cache_resumes_after_user_message() {
    let tools = agent_tools();
    let _ = agent_system();
    let req = |msgs: &Json| json!({"messages": msgs, "tools": tools, "temperature": 0});
    let open = |st: &Arc<State<StubEngine>>, dir: &std::path::Path| {
        let mut w = st.worker.lock().unwrap();
        let id = crate::prefix_store::identity(std::path::Path::new("/m.gguf"), &w.engine.state_id().unwrap(), 16);
        w.sessions = Some(crate::prefix_store::PrefixStore::open(dir, id, 1 << 30, 16).unwrap());
        w.cache.park_min = 16;
        w.cache.park_budget = 1 << 30;
        w.cache.spill = true;
    };
    let scripts = [READ_CALL, "Fixed it.", "Run pytest."];
    // turns 1-2: tool call, tool result -> answer; turn 3: a new user message
    let run_two = |st: Arc<State<StubEngine>>| async move {
        let mut msgs = json!([{"role":"system","content":agent_system()},{"role":"user","content":"The add() function in calc.py is wrong."}]);
        let (_, j1, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages": msgs, "tools": agent_tools(), "temperature": 0})).await;
        let tc = j1["choices"][0]["message"]["tool_calls"][0].clone();
        msgs.as_array_mut().unwrap().extend([j1["choices"][0]["message"].clone(), json!({"role":"tool","tool_call_id":tc["id"],"content":"def add(a, b):\n    return a - b\n"})]);
        let (_, j2, _) = call(&st, "POST", "/v1/chat/completions", json!({"messages": msgs, "tools": agent_tools(), "temperature": 0})).await;
        msgs.as_array_mut().unwrap().extend([j2["choices"][0]["message"].clone(), json!({"role":"user","content":"Now write a test."})]);
        msgs
    };
    // reference: uninterrupted
    let Some(r) = state_with(&scripts, 16384, 8, 16, None) else { return };
    // the session cache's checkpoints, measured without ECHO (its records are RAM only here)
    r.echo.lock().unwrap().cfg.enabled = false;
    let msgs = run_two(r.clone()).await;
    let (_, jr, _) = call(&r, "POST", "/v1/chat/completions", req(&msgs)).await;
    let c_ref = jr["timings"]["cache_n"].as_u64().unwrap();
    assert!(c_ref > 0);
    // test: same turns, persist (idle unload), new process, turn 3
    let dir = scratch_dir("sessions-user");
    let Some(st) = state_with(&scripts, 16384, 8, 16, None) else { return };
    st.echo.lock().unwrap().cfg.enabled = false;
    open(&st, &dir);
    // the same conversation (tool-call ids are random, and the template does not render them)
    let msgs2 = run_two(st.clone()).await;
    let n = st.worker.lock().unwrap().persist_sessions();
    assert!(n >= 2, "full state + its query checkpoint: {n} entries");
    drop(st);
    let Some(st2) = state_with(&["Run pytest."], 16384, 8, 16, None) else { return };
    st2.echo.lock().unwrap().cfg.enabled = false;
    open(&st2, &dir);
    let (_, j3, _) = call(&st2, "POST", "/v1/chat/completions", req(&msgs2)).await;
    let c3 = j3["timings"]["cache_n"].as_u64().unwrap();
    assert_eq!(c3, c_ref, "reuse after the restart equals the uninterrupted session's");
    assert_eq!(j3["choices"][0]["message"], jr["choices"][0]["message"]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn repetition_watchdog_detects_loops_only() {
    use crate::generate::repeat_loop;
    let mut t: Vec<u32> = (0..40).collect();
    assert_eq!(repeat_loop(&t, 64), None);
    // period 3 repeated: needs 16 tokens (6 reps) at period 3
    for _ in 0..6 {
        t.extend([7, 8, 9]);
    }
    assert_eq!(repeat_loop(&t, 64), Some(3));
    assert_eq!(repeat_loop(&t, 0), None);
    // a short run of one token (e.g. closing braces) below 16 does not trigger
    let mut u: Vec<u32> = (0..40).collect();
    u.extend([5; 15]);
    assert_eq!(repeat_loop(&u, 64), None);
    u.push(5);
    assert_eq!(repeat_loop(&u, 64), Some(1));
    // a long period needs 4 repetitions inside the window
    let mut v: Vec<u32> = (100..140).collect();
    let block: Vec<u32> = (0..12).collect();
    for _ in 0..3 {
        v.extend(&block);
    }
    assert_eq!(repeat_loop(&v, 64), None);
    v.extend(&block);
    assert_eq!(repeat_loop(&v, 64), Some(12));
    assert_eq!(repeat_loop(&v, 40), None); // 48 > window
}

/// Codex 039 A: cancel mid-prefill, then another (short) request, then the retry. The
/// partial prefill must survive the intervening request (it is parked) and the retry resume it.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn cancelled_partial_prefill_survives_another_request() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let u = 16;
    let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let eng = CancelStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script.clone()), flag: flag.clone(), at: 5, calls: 0 };
    let mut w = Worker::new(eng, 8);
    w.ubatch = u;
    w.cache.park_budget = 1 << 30;
    w.cache.park_min = 64;
    let mut gp = greedy();
    gp.cancel = Some(flag.clone());
    assert!(matches!(w.generate(&tok, &prompt, &gp, &mut |_| true), Err(crate::generate::GenError::Cancelled)));
    let partial = w.cache.tokens.len();
    assert!(partial >= 64, "partial prefill of {partial} tokens");
    // the user's next, unrelated short request
    flag.store(false, std::sync::atomic::Ordering::Relaxed);
    w.engine.at = usize::MAX;
    let short: Vec<u32> = (0..20u32).map(|i| 7000 + i).collect();
    w.generate(&tok, &short, &gp, &mut |_| true).unwrap();
    assert_eq!(w.cache.parked_lengths(), vec![partial], "the partial session was parked");
    // the retry resumes the partial prefill
    let r = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
    assert_eq!(r.timings.cache_n, partial);
    assert_eq!(w.cache.stats.unparks, 1);
}

#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn mtp_tune_table_and_per_request_off_switch() {
    use crate::generate::SpecTable;
    let dir = std::env::temp_dir().join(format!("oxe-tune-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("t.json");
    std::fs::write(&f, r#"{"buckets":[{"max_ctx":32768,"depth":0},{"max_ctx":4096,"depth":2},{"max_ctx":12288,"depth":1}]}"#).unwrap();
    let t = SpecTable::load(&f).unwrap();
    assert_eq!(t.buckets, vec![(4096, 2), (12288, 1), (32768, 0)]);
    assert_eq!((t.depth_for(10), t.depth_for(4096), t.depth_for(4097), t.depth_for(20000), t.depth_for(99999)), (2, 2, 1, 0, 0));
    std::fs::remove_dir_all(&dir).ok();
    // per-request switch parsing
    let gp = crate::api::gen_params(&json!({"prompt":"x","speculative":false}), false).unwrap();
    assert_eq!(gp.spec_depth, Some(0));
    let gp = crate::api::gen_params(&json!({"prompt":"x","spec_draft_n_max":1}), false).unwrap();
    assert_eq!(gp.spec_depth, Some(1));
    let gp = crate::api::gen_params(&json!({"prompt":"x","speculative":{"n_max":3}}), false).unwrap();
    assert_eq!(gp.spec_depth, Some(3));
    // speculation off for a request: identical output, no speculative calls
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let script = tok.encode("one two three four five six seven", false, true);
    let prompt = tok.encode("hello", false, true);
    let mut plain = Worker::new(StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone()), 4);
    let want = plain.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap();
    for (depth, table) in [(Some(0), None), (None, Some(SpecTable { buckets: vec![(1 << 20, 0)] }))] {
        let inner = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone());
        let mut w = Worker::new(SpecStub { inner, depth: 2, before: None, appended: vec![], rewinds: 0, calls: 0, at: vec![] }, 4);
        w.spec_table = table;
        let mut gp = greedy();
        gp.spec_depth = depth;
        let got = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
        assert_eq!(got.tokens, want.tokens);
        assert_eq!(got.timings.draft_n, 0, "speculation must be off");
        assert!(w.engine.appended.is_empty(), "decode_spec was called");
    }
}

/// A stub that verifies external drafts like OxeEngine::decode_verify (prompt lookup, WP-D2 c).
struct LookupStub {
    inner: StubEngine,
    before: Option<crate::engine::StubSnapshot>,
    appended: Vec<u32>,
    drafted: u64,
}

impl Engine for LookupStub {
    type Snapshot = <StubEngine as Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
    fn decode_verify(&mut self, t: u32, drafts: &[u32]) -> crate::engine::Result<Option<(Vec<u32>, Vec<u32>)>> {
        self.before = Some(self.inner.snapshot()?);
        self.drafted += drafts.len() as u64;
        let mut g = vec![crate::sampling::argmax(self.inner.decode(t)?)];
        self.appended = vec![t];
        let mut extra = Vec::new();
        for &d in drafts {
            if d != *g.last().unwrap() || self.inner.n_past() + 1 >= self.inner.n_ctx {
                break;
            }
            g.push(crate::sampling::argmax(self.inner.decode(d)?));
            extra.push(d);
            self.appended.push(d);
        }
        Ok(Some((extra, g)))
    }
    fn spec_rewind(&mut self, keep: usize) -> crate::engine::Result<()> {
        let s = self.before.take().unwrap();
        self.inner.restore(&s)?;
        for &t in &self.appended[..keep] {
            self.inner.decode(t)?;
        }
        Ok(())
    }
}

/// Prompt lookup: identical tokens, text, stop and engine state as plain greedy decoding, with
/// drafts actually verified (a repetitive script), stop words, limits and cancels.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn lookup_drafting_matches_plain() {
    let Some(p) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(p).unwrap();
    let script = tok.encode("a b c d e f a b c d e f a b c d e f STOP a b c d e f a b c", false, true);
    let prompt = tok.encode("a b c d e f", false, true);
    let mut cases = vec![greedy()];
    let mut g = greedy();
    g.stop = vec!["STOP".into()];
    cases.push(g);
    let mut g = greedy();
    g.n_predict = 9;
    cases.push(g);
    let mut drafted = 0;
    for gp in &cases {
        for cancel_at in [usize::MAX, 4, 11] {
            let mut plain = Worker::new(StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone()), 4);
            let mut k = 0;
            let want = plain.generate(&tok, &prompt, gp, &mut |_| { k += 1; k < cancel_at });
            let inner = StubEngine::new(1024, tok.n_vocab(), tok.eos.unwrap(), script.clone());
            let mut w = Worker::new(LookupStub { inner, before: None, appended: vec![], drafted: 0 }, 4);
            w.lookup = Some(crate::lookup::LookupConfig { kmin: 2, kmax: 4, n_max: 3, backoff: 0 });
            let mut k = 0;
            let got = w.generate(&tok, &prompt, gp, &mut |_| { k += 1; k < cancel_at });
            match (&want, &got) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.tokens, b.tokens);
                    assert_eq!(a.text, b.text);
                    assert_eq!(a.stop, b.stop);
                }
                (Err(_), Err(_)) => {}
                _ => panic!("plain and lookup disagree on cancel"),
            }
            assert_eq!(w.engine.inner.tokens, plain.engine.tokens, "engine state differs (cancel at {cancel_at})");
            drafted += w.engine.drafted;
        }
    }
    assert!(drafted > 0, "no drafts were verified");
}

/// Night combine (039 x 018): a cancel raised while MTP speculation is in flight (the flag goes
/// up inside `decode_spec`) stops after that verify, rewinds the unconsumed speculative tokens,
/// leaves engine == cache == prompt + a prefix of the uncancelled reply, and the next request
/// replies as an uncancelled run would.
struct SpecCancelStub {
    spec: SpecStub,
    flag: Arc<std::sync::atomic::AtomicBool>,
    at: usize,
    calls: usize,
}

impl Engine for SpecCancelStub {
    type Snapshot = <StubEngine as Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.spec.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.spec.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.spec.n_past()
    }
    fn reset(&mut self) {
        self.spec.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.spec.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.spec.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.spec.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.spec.restore(s)
    }
    fn spec_max_depth(&self) -> usize {
        self.spec.spec_max_depth()
    }
    fn decode_spec(&mut self, t: u32) -> crate::engine::Result<Option<(Vec<u32>, Vec<Vec<f32>>)>> {
        self.calls += 1;
        if self.calls == self.at {
            self.flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        self.spec.decode_spec(t)
    }
    fn spec_rewind(&mut self, keep: usize) -> crate::engine::Result<()> {
        self.spec.spec_rewind(keep)
    }
}

#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn cancel_during_mtp_speculation_is_consistent() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("one two three four five six seven eight nine ten eleven twelve", false, true);
    let prompt = tok.encode("hello there", false, true);
    let reference = {
        let mut w = Worker::new(StubEngine::new(1024, tok.n_vocab(), eos, script.clone()), 4);
        w.generate(&tok, &prompt, &greedy(), &mut |_| true).unwrap().tokens
    };
    for depth in 1..=3 {
        for at in 1..4 {
            let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let spec = SpecStub { inner: StubEngine::new(1024, tok.n_vocab(), eos, script.clone()), depth, before: None, appended: vec![], rewinds: 0, calls: 0, at: vec![] };
            let mut w = Worker::new(SpecCancelStub { spec, flag: flag.clone(), at, calls: 0 }, 4);
            let mut gp = greedy();
            gp.cancel = Some(flag.clone());
            let mut sent = 0usize;
            let r = w.generate(&tok, &prompt, &gp, &mut |_| {
                sent += 1;
                true
            });
            assert!(matches!(r, Err(crate::generate::GenError::Cancelled)), "depth {depth}, at {at}");
            // stopped at the first token after the verify that raised the flag
            assert!(sent >= at + 1, "depth {depth}, at {at}: {sent} tokens emitted before the stop");
            assert_eq!(w.engine.n_past(), w.cache.tokens.len(), "depth {depth}, at {at}: engine ahead of the cache");
            assert_eq!(w.engine.spec.inner.tokens, w.cache.tokens, "depth {depth}, at {at}: engine and cache agree");
            let k = w.cache.tokens.len();
            assert_eq!(w.cache.tokens[..prompt.len()], prompt[..]);
            assert_eq!(w.cache.tokens[prompt.len()..], reference[..k - prompt.len()], "a prefix of the uncancelled reply");
            // the next request works and replies as the uncancelled run
            flag.store(false, std::sync::atomic::Ordering::Relaxed);
            w.engine.at = usize::MAX;
            let r2 = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
            assert_eq!(r2.tokens, reference, "depth {depth}, at {at}: next request differs");
        }
    }
}

/// SX-2: with --idle-reprefill, after a turn that ends the query the server prefills the history
/// form while idle; the next user turn then reuses everything before its new user message, where
/// without it the request resumes at the previous query's checkpoint and re-prefills that query.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
async fn idle_reprefill_serves_the_next_user_turn() {
    let run = |idle: bool| async move {
        let st = state_with(&[READ_CALL, "Fixed it.", "Run pytest."], 16384, 8, 16, None).unwrap();
        // the flag lives in the config: rebuild the state with it set
        let st = {
            let w = Arc::try_unwrap(st).ok().unwrap();
            let mut cfg = w.cfg;
            cfg.idle_reprefill = idle;
            let tok = Arc::try_unwrap(w.tok).ok().unwrap();
            Arc::new(State::new(cfg, tok, w.worker.into_inner().unwrap()).unwrap())
        };
        // SX-2 measured without ECHO (with it the next user turn hits without any idle work)
        st.echo.lock().unwrap().cfg.enabled = false;
        let req = |msgs: &Json| json!({"messages": msgs, "tools": agent_tools(), "temperature": 0});
        let mut msgs = json!([{"role":"system","content":agent_system()},{"role":"user","content":"The add() function in calc.py is wrong."}]);
        let (_, j1, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
        let tc = j1["choices"][0]["message"]["tool_calls"][0].clone();
        msgs.as_array_mut().unwrap().extend([j1["choices"][0]["message"].clone(), json!({"role":"tool","tool_call_id":tc["id"],"content":"def add(a, b):\n    return a - b\n"})]);
        let (_, j2, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
        assert_eq!(j2["choices"][0]["finish_reason"], "stop");
        if idle {
            // wait (bounded) for the idle job
            let t0 = std::time::Instant::now();
            while st.idle_done.load(std::sync::atomic::Ordering::SeqCst) < 1 {
                assert!(t0.elapsed().as_secs() < 30, "idle job did not finish");
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        }
        // where turn 2 ended = after its last decode (the idle job only restores and prefills).
        // `calls.len()` read right after the reply raced the idle job: under load the job had
        // already run, its prefills fell before the mark and the boundary lists differed (P2 review).
        let calls0 = st.worker.lock().unwrap().engine.calls.iter().rposition(|c| c.0 == "decode").map_or(0, |i| i + 1);
        msgs.as_array_mut().unwrap().extend([j2["choices"][0]["message"].clone(), json!({"role":"user","content":"Now write a test."})]);
        let (_, tk, _) = call(&st, "POST", "/apply-template", json!({"messages": msgs, "tools": agent_tools()})).await;
        let p3 = st.tok.encode(tk["prompt"].as_str().unwrap(), true, true);
        let d = st.user_delim.clone().unwrap();
        let last_user = p3.windows(d.len()).rposition(|w| w == &d[..]).unwrap();
        let (_, j3, _) = call(&st, "POST", "/v1/chat/completions", req(&msgs)).await;
        assert_eq!(j3["choices"][0]["message"]["content"], "Run pytest.");
        // ubatch boundaries (ubatch 16) of every multi-token prefill since turn 2 ended: the idle
        // job's plus turn 3's, or turn 3's alone without SX-2
        let w = st.worker.lock().unwrap();
        // up to turn 3's last decode: with SX-2 on, turn 3 itself ends with "stop" and spawns the
        // next idle job, which may already have run when the lock is taken here; its prefills are
        // not turn 3's (v0.1.3: the remaining race of this test, now deterministic)
        let end = w.engine.calls.iter().rposition(|c| c.0 == "decode").map_or(w.engine.calls.len(), |i| i + 1).max(calls0);
        let mut bounds: Vec<usize> = w.engine.calls[calls0..end]
            .iter()
            .filter(|c| c.0 == "prefill" && c.2 > 1)
            .flat_map(|c| (c.1..c.1 + c.2).step_by(16))
            .collect();
        bounds.sort_unstable();
        (j3["timings"]["cache_n"].as_u64().unwrap() as usize, last_user, bounds)
    };
    if model().is_none() {
        return;
    }
    let (c_off, last_user, b_off) = run(false).await;
    let (c_on, last_user2, b_on) = run(true).await;
    assert_eq!(last_user, last_user2);
    // lossless: the engine sees the same ubatch boundaries with and without the idle job
    assert_eq!(b_on, b_off, "ubatch boundaries differ: idle {b_on:?} vs on-demand {b_off:?}");
    assert!(c_off < last_user, "without SX-2 the query is re-prefilled: cache_n {c_off} < {last_user}");
    assert!(c_on > c_off && c_on + 16 > last_user, "with SX-2 the request resumes at the grid point below its new user message: cache_n {c_on}, user message at {last_user}");
}

/// W4-4 (c): a store prunes its own entries, plus orphans of the SAME model from an older
/// build (another identity) once they pass the TTL. Another model's entries and files that are
/// not entries (no magic, bad header) are never deleted, by the size cap or the TTL.
#[test]
fn store_prune_leaves_foreign_files_alone() {
    let dir = scratch_dir("shared-prune");
    std::fs::create_dir_all(&dir).unwrap();
    let foreign = dir.join("0000000000000000000000000000000000000000000000000000000000000000.bin");
    std::fs::write(&foreign, vec![7u8; 4096]).unwrap();
    let bogus = dir.join("other.bin");
    std::fs::write(&bogus, b"OXEPFX01not-this-identity").unwrap();
    let ours = "model=sha256:m1 server=sha256:new state=s ubatch=16";
    let old_build = "model=sha256:m1 server=sha256:old state=s ubatch=16";
    let other_model = "model=sha256:m2 server=sha256:new state=s ubatch=16";
    for (id, toks) in [(old_build, vec![1u32, 2, 3]), (other_model, vec![4u32, 5, 6])] {
        let mut st = crate::prefix_store::PrefixStore::open(&dir, id.into(), 1 << 30, 1).unwrap();
        st.save_async(toks, vec![1u8; 64]);
        st.flush();
    }
    // own entries, then a cap smaller than one entry: own entries go, nothing else
    let mut st = crate::prefix_store::PrefixStore::open(&dir, ours.into(), 1 << 30, 1).unwrap();
    for k in 0..3u32 {
        st.save_async(vec![k, k + 1, k + 2, 9], vec![1u8; 2048]);
        st.flush();
    }
    drop(st);
    let n_bin = || std::fs::read_dir(&dir).unwrap().flatten().filter(|d| d.path().extension().map_or(false, |e| e == "bin")).count();
    assert_eq!(n_bin(), 2 + 2 + 3);
    let mut st = crate::prefix_store::PrefixStore::open(&dir, ours.into(), 100, 1).unwrap();
    assert_eq!(st.n_entries(), 0, "own entries pruned to the cap");
    assert_eq!(n_bin(), 4, "the size cap touches only this identity's entries");
    // TTL: own entries and the same model's old-build orphan go; other model + non-entries stay
    st.save_async(vec![9, 9, 9], vec![1u8; 16]);
    st.flush();
    std::thread::sleep(std::time::Duration::from_millis(20));
    let gone = st.prune_older_than(std::time::Duration::from_millis(1));
    assert_eq!(gone, 2, "own entry + same-model orphan");
    assert!(foreign.exists() && bogus.exists(), "non-entries survive");
    assert_eq!(n_bin(), 3, "the other model's entry survives");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Context-adaptive MTP (`mtp_max_ctx`, `spec_table`) turns speculation off and on inside one
/// reply. Output, stop and engine state must equal plain decoding, and `decode_spec` may only
/// run at the context positions the policy allows.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn ctx_adaptive_mtp_switch_mid_reply_matches_plain() {
    use crate::generate::SpecTable;
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty STOP twentyone twentytwo",
        false,
        true,
    );
    let prompt = tok.encode("hello there", false, true);
    let p0 = prompt.len();
    let mut stop_gp = greedy();
    stop_gp.stop = vec!["STOP".into()];
    let mut limit_gp = greedy();
    limit_gp.n_predict = 25;
    let cases = [greedy(), stop_gp, limit_gp];
    let table = || SpecTable { buckets: vec![(p0 + 5, 0), (p0 + 15, 2), (1 << 20, 0)] };
    // (mtp_max_ctx, spec_table, context positions where decode_spec may run, must speculate)
    let policies: Vec<(Option<usize>, Option<SpecTable>, std::ops::Range<usize>, bool)> = vec![
        (Some(p0 + 7), None, p0..p0 + 7, true),
        (None, Some(table()), p0 + 6..p0 + 16, true),
        (Some(p0 + 7), Some(table()), p0 + 6..p0 + 7, false),
        (Some(p0), None, 0..0, false),
    ];
    let mk = |mtp_max_ctx: Option<usize>, spec_table: Option<SpecTable>| {
        let inner = StubEngine::new(1024, tok.n_vocab(), eos, script.clone());
        let mut w = Worker::new(SpecStub { inner, depth: 2, before: None, appended: vec![], rewinds: 0, calls: 0, at: vec![] }, 4);
        w.mtp_max_ctx = mtp_max_ctx;
        w.spec_table = spec_table;
        w
    };
    for gp in &cases {
        let mut plain = Worker::new(StubEngine::new(1024, tok.n_vocab(), eos, script.clone()), 4);
        let want = plain.generate(&tok, &prompt, gp, &mut |_| true).unwrap();
        for (pi, (mtp_max_ctx, spec_table, allowed, must)) in policies.iter().enumerate() {
            let mut w = mk(*mtp_max_ctx, spec_table.clone());
            let got = w.generate(&tok, &prompt, gp, &mut |_| true).unwrap();
            assert_eq!(got.tokens, want.tokens, "policy {pi}: tokens");
            assert_eq!(got.text, want.text, "policy {pi}: text");
            assert_eq!(got.stop, want.stop, "policy {pi}: stop");
            assert_eq!(w.engine.inner.tokens, plain.engine.tokens, "policy {pi}: engine state");
            assert!(
                w.engine.at.iter().all(|x| allowed.contains(x)),
                "policy {pi}: decode_spec at {:?}, allowed {:?}",
                w.engine.at,
                allowed
            );
            if *must {
                assert!(w.engine.calls > 0, "policy {pi}: speculation never ran");
            }
            if allowed.is_empty() {
                assert_eq!(w.engine.calls, 0, "policy {pi}: speculation with mtp_max_ctx at the prompt");
            }
            // a client that disconnects mid-reply: the state must be consistent, and the next
            // request must reply as the uncancelled run
            for cancel_at in [2usize, 6, 12] {
                let mut w = mk(*mtp_max_ctx, spec_table.clone());
                let mut k = 0;
                let r = w.generate(&tok, &prompt, gp, &mut |_| {
                    k += 1;
                    k < cancel_at
                });
                assert!(matches!(r, Err(crate::generate::GenError::Cancelled)), "policy {pi}, cancel at {cancel_at}");
                assert_eq!(w.engine.n_past(), w.cache.tokens.len(), "policy {pi}, cancel at {cancel_at}: engine ahead of the cache");
                let again = w.generate(&tok, &prompt, gp, &mut |_| true).unwrap();
                assert_eq!(again.tokens, want.tokens, "policy {pi}, cancel at {cancel_at}: next request differs");
                assert_eq!(w.engine.inner.tokens, plain.engine.tokens, "policy {pi}, cancel at {cancel_at}: engine state");
            }
        }
    }
}

// ------------------------------------------------------------------ W4 review (model-free)

/// An engine whose `at`-th prefill call fails AFTER it advanced its state: the Flash adapter's
/// `eval` (the decode ran to the end with invalid expert rows after a failed / timed-out expert
/// read, the position moved, then `take_error` turns it into Err).
struct FailAfterStub {
    inner: StubEngine,
    at: usize,
    calls: usize,
}

impl Engine for FailAfterStub {
    type Snapshot = <StubEngine as Engine>::Snapshot;
    fn n_ctx(&self) -> usize {
        self.inner.n_ctx()
    }
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }
    fn n_past(&self) -> usize {
        self.inner.n_past()
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn prefill(&mut self, t: &[u32]) -> crate::engine::Result<&[f32]> {
        self.calls += 1;
        if self.calls == self.at {
            self.inner.prefill(t)?;
            return Err(crate::engine::EngineError("flash: expert read failed (see the IO error)".into()));
        }
        self.inner.prefill(t)
    }
    fn decode(&mut self, t: u32) -> crate::engine::Result<&[f32]> {
        self.inner.decode(t)
    }
    fn snapshot(&mut self) -> crate::engine::Result<Self::Snapshot> {
        self.inner.snapshot()
    }
    fn restore(&mut self, s: &Self::Snapshot) -> crate::engine::Result<()> {
        self.inner.restore(s)
    }
}

/// W4 review finding: an engine error in the middle of a prefill (Flash: a failed expert read)
/// must not leave the engine ahead of the cache's token list. Today the request fails with 500,
/// the server stays up, and the retry of the same prompt is treated as a live continuation: it
/// prefills on top of the failed chunk (wrong positions, invalid KV rows) and answers 200.
#[test]
#[ignore = "needs the Ornith GGUF (tokenizer/template) in ~/.local/share/ai-stack/models/llm; run with --include-ignored where it exists"]
fn engine_error_mid_prefill_leaves_engine_and_cache_in_step() {
    let Some(m) = model() else { return };
    let tok = tokenizer::Tokenizer::from_gguf(m).unwrap();
    let eos = tok.eos.unwrap();
    let script = tok.encode("a reply of a few tokens", false, true);
    let prompt: Vec<u32> = (0..1000u32).map(|i| 100 + i % 5000).collect();
    let eng = FailAfterStub { inner: StubEngine::new(4096, tok.n_vocab(), eos, script), at: 3, calls: 0 };
    let mut w = Worker::new(eng, 4);
    w.ubatch = 16;
    let r = w.generate(&tok, &prompt, &greedy(), &mut |_| true);
    assert!(matches!(r, Err(crate::generate::GenError::Engine(_))));
    let (eng_n, cache_n) = (w.engine.n_past(), w.cache.tokens.len());
    eprintln!("after the failed request: engine n_past {eng_n}, cache tokens {cache_n}");
    // the retry
    let r = w.generate(&tok, &prompt, &greedy(), &mut |_| true);
    let n = w.engine.inner.tokens.len().min(prompt.len());
    let same = w.engine.inner.tokens[..n] == prompt[..n];
    eprintln!("retry: ok {}, engine holds the prompt: {same}, engine n_past {} vs prompt {}", r.is_ok(), w.engine.n_past(), prompt.len());
    assert_eq!(eng_n, cache_n, "engine ahead of the cache after a failed prefill");
    assert!(same, "the retry prefilled on top of the failed chunk");
}

/// Model-free: the Flash switch defaults (G4 prefill stack on, warm-up reads off) and the flags that flip them, as
/// `main` parses them (`FlashSwitches::parse_flag`).
#[test]
fn flash_switch_defaults_and_flags() {
    use crate::flash_switches::FlashSwitches;
    let d = FlashSwitches::default();
    assert!(d.attn_tile && d.dense_gemm && d.expert_tile, "G4 prefill switches must default on: {d:?}");
    assert!(!d.warmup, "warm-up reads must default off: {d:?}");
    let mut s = FlashSwitches::default();
    assert_eq!(s.parse_flag("--flash-dense-gemm", "0"), Ok(true));
    assert_eq!(s.parse_flag("--flash-warmup", "1"), Ok(true));
    assert_eq!(s, FlashSwitches { attn_tile: true, dense_gemm: false, expert_tile: true, warmup: true });
    assert_eq!(s.parse_flag("--flash-attn-tile", "0"), Ok(true));
    assert_eq!(s.parse_flag("--flash-expert-tile", "0"), Ok(true));
    assert!(!s.attn_tile && !s.expert_tile);
    assert_eq!(s.parse_flag("--flash-attn-tile", "1"), Ok(true));
    assert!(s.attn_tile);
    // other flags are not ours; bad values are errors
    assert_eq!(s.parse_flag("--flash-hotset", "default"), Ok(false));
    assert!(s.parse_flag("--flash-warmup", "yes").is_err());
    assert!(s.parse_flag("--flash-dense-gemm", "").is_err());
    assert!(s.warmup && !s.dense_gemm, "a rejected value must not change the switch");
}

// ------------------------------------------------------------------ LONG-CTX: no full-state RAM copies

/// `save_stream` writes the same file bytes as `save_async` of the buffered state (40,002
/// tokens: beyond 32K), and the entry loads back exactly; nothing buffers the state.
#[test]
fn long_ctx_stream_save_equals_buffered_save() {
    let toks: Vec<u32> = (0..40_002u32).map(|i| (i * 7 + 3) % 400).collect();
    let mut eng = StubEngine::new(100_000, 400, 399, vec![]);
    eng.prefill(&toks).unwrap();
    let (d1, d2) = (scratch_dir("lc-async"), scratch_dir("lc-stream"));
    let mut a = crate::prefix_store::PrefixStore::open(&d1, "id".into(), 1 << 30, 16).unwrap();
    let mut b = crate::prefix_store::PrefixStore::open(&d2, "id".into(), 1 << 30, 16).unwrap();
    a.save_async(toks.clone(), eng.save_state_vec().unwrap());
    a.flush();
    let len = b.save_stream(&toks, &mut |w| eng.save_state(w).map_err(|e| e.0)).unwrap();
    assert_eq!(len as usize, eng.state_size_hint());
    let file = |d: &std::path::Path| {
        let p = std::fs::read_dir(d).unwrap().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "bin")).unwrap();
        std::fs::read(p).unwrap()
    };
    assert!(file(&d1) == file(&d2), "streamed entry differs from the buffered one");
    assert!(b.contains(&toks));
    // a reopened store indexes it, and it loads into a fresh engine
    let b2 = crate::prefix_store::PrefixStore::open(&d2, "id".into(), 1 << 30, 16).unwrap();
    let mut e2 = StubEngine::new(100_000, 400, 399, vec![]);
    let mut prompt = toks.clone();
    prompt.push(5);
    assert_eq!(b2.best(&prompt, prompt.len() - 1), Some(toks.len()));
    b2.load(&mut e2, &prompt, toks.len()).unwrap();
    assert_eq!(e2.tokens, toks);
    let _ = std::fs::remove_dir_all(&d1);
    let _ = std::fs::remove_dir_all(&d2);
}

/// Disk parking (`--park-disk-min-bytes`): a 40K-token session that another agent's request
/// displaces is streamed to the session cache with its checkpoint, never copied into RAM; a small
/// session still parks in RAM; the long one resumes from disk with its full length; the engine and
/// cache stay consistent. A RAM park whose state cannot fit the budget builds no copy.
#[test]
fn long_ctx_disk_park_streams_large_sessions() {
    let dir = scratch_dir("lc-park");
    let mut w = Worker::new(StubEngine::new(100_000, 400, 399, vec![]), 8);
    w.sessions = Some(crate::prefix_store::PrefixStore::open(&dir, "id".into(), 1 << 30, 16).unwrap());
    w.cache.park_budget = 1 << 30;
    w.cache.park_min = 8;
    w.cache.spill = true;
    w.cache.park_disk_min = 100_000; // stub state = 4 B/token: 40K tokens park to disk, 31 to RAM
    let run = |w: &mut Worker<StubEngine>, prompt: &[u32], gen: &[u32]| -> usize {
        w.park_large(prompt);
        let n = w.cache.prepare(&mut w.engine, prompt, true).unwrap();
        let rest = &prompt[n..];
        if rest.len() > 1 {
            w.engine.prefill(&rest[..rest.len() - 1]).unwrap();
            w.cache.appended(&rest[..rest.len() - 1]);
            w.cache.checkpoint(&mut w.engine).unwrap();
        }
        w.engine.prefill(&rest[rest.len() - 1..]).unwrap();
        w.cache.appended(&rest[rest.len() - 1..]);
        w.cache.prompt_len = prompt.len();
        for &g in gen {
            w.engine.decode(g).unwrap();
            w.cache.appended(&[g]);
        }
        w.cache.checkpoint(&mut w.engine).unwrap();
        assert_eq!(w.engine.tokens, w.cache.tokens);
        n
    };
    let a1: Vec<u32> = (0..40_000u32).map(|i| (i * 13 + 1) % 300).collect();
    run(&mut w, &a1, &[310, 311]);
    let a_full: Vec<u32> = w.cache.tokens.clone();
    // agent B displaces A: A goes to disk (full state + its prompt checkpoint), not to RAM
    let b1: Vec<u32> = (320..=350).collect();
    assert_eq!(run(&mut w, &b1, &[360]), 0);
    assert_eq!(w.cache.stats.disk_parks, 1);
    assert!(w.cache.parked_lengths().is_empty(), "the long session was copied into RAM");
    let st = w.sessions.as_ref().unwrap();
    assert!(st.contains(&a_full), "full session not on disk");
    assert!(st.contains(&a1[..a1.len() - 1]), "prompt checkpoint not on disk");
    // A continues: B (small) parks in RAM as before; A's whole state is on disk for the worker's
    // session-cache lookup
    let mut a2 = a_full.clone();
    a2.extend([370, 371, 372]);
    run(&mut w, &a2, &[]);
    assert_eq!(w.cache.stats.disk_parks, 1);
    assert_eq!(w.cache.parked_lengths(), vec![b1.len() + 1]);
    let st = w.sessions.as_ref().unwrap();
    assert_eq!(st.best(&a2, a2.len() - 1), Some(a_full.len()));
    let mut e2 = StubEngine::new(100_000, 400, 399, vec![]);
    st.load(&mut e2, &a2, a_full.len()).unwrap();
    assert_eq!(e2.tokens, a_full);
    // RAM parking with a budget below the state: no park (and no copy built for it)
    let mut c: PrefixCache<<StubEngine as Engine>::Snapshot> = PrefixCache::new(8);
    let mut eng = StubEngine::new(100_000, 400, 399, vec![]);
    c.park_budget = 1000;
    c.park_min = 8;
    eng.prefill(&a1).unwrap();
    c.appended(&a1);
    c.prompt_len = a1.len();
    c.prepare(&mut eng, &b1, true).unwrap();
    assert_eq!((c.stats.parks, c.parked_bytes()), (0, 0));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `save_stream` past the 64 MiB write-behind window: a 150 MiB state streams in odd-sized
/// writes; the patched header length and checksum match the data that landed in the file.
#[test]
fn long_ctx_stream_save_write_behind_150mib() {
    let dir = scratch_dir("lc-wb");
    let mut st = crate::prefix_store::PrefixStore::open(&dir, "id".into(), 1 << 40, 1).unwrap();
    let toks: Vec<u32> = (0..100).collect();
    let total = 150usize << 20;
    let chunk: Vec<u8> = (0..1_048_573usize).map(|i| (i * 31 % 251) as u8).collect();
    let len = st
        .save_stream(&toks, &mut |w| {
            let mut left = total;
            while left > 0 {
                let n = left.min(chunk.len());
                w.write_all(&chunk[..n]).map_err(|e| e.to_string())?;
                left -= n;
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(len as usize, total);
    let p = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|x| x == "bin")).unwrap();
    let f = std::fs::read(&p).unwrap();
    let hdr = 8 + 4 + 2 + 8 + 4 * toks.len() + 16;
    assert_eq!(f.len(), hdr + total);
    let rd = |o: usize| u64::from_le_bytes(f[o..o + 8].try_into().unwrap());
    assert_eq!(rd(hdr - 16), total as u64);
    assert_eq!(rd(hdr - 8), crate::prefix_store::check64(&f[hdr..]));
    let _ = std::fs::remove_dir_all(&dir);
}
