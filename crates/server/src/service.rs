//! Endpoint logic, independent of the HTTP library: a request (method, path, JSON body) gives
//! either a complete JSON response or a stream of SSE strings.

use crate::api::{self, ChatMeta};
use crate::engine::Engine;
use crate::generate::{GenError, GenParams, GenResult, Partial, Worker};
use chat::{ChatTemplate, ParserConfig, ServerChatOptions, Session};
use serde_json::{json, Value as Json};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokenizer::{Token, Tokenizer};
use tokio::sync::mpsc;

pub struct Config {
    pub model_name: String,
    pub model_path: String,
    pub chat_opts: ServerChatOptions,
    pub fingerprint: String,
    /// allow /slots POST (save/restore are not implemented yet)
    pub slots: bool,
    /// SX-2: prefill the next request's history form while the client is idle (`--idle-reprefill`)
    pub idle_reprefill: bool,
    /// engine facts for `/props` (Flash: `flash_cache` = the cache tier decision: chosen, reason, preference, margins)
    pub engine_status: Option<Json>,
}

pub struct State<E: Engine> {
    pub cfg: Config,
    pub tok: Arc<Tokenizer>,
    pub tmpl: Option<ChatTemplate>,
    pub worker: Mutex<Worker<E>>,
    pub n_ctx: usize,
    pub n_vocab: usize,
    /// token ids that open a user message in the rendered prompt (ChatML `<|im_start|>user`;
    /// llama-server's message-span delimiter), None if the template has none
    pub user_delim: Option<Vec<Token>>,
    /// SX-2: cancel flag of the pending/running idle re-prefill
    pub idle: Mutex<Option<Arc<AtomicBool>>>,
    /// requests started so far (an idle job that sees a newer request does not start)
    pub reqs: std::sync::atomic::AtomicU64,
    /// idle jobs finished (tests)
    pub idle_done: std::sync::atomic::AtomicU64,
    /// S8: a stream whose client takes no frame for this long is dropped and its generation
    /// cancelled (`--stream-stall-timeout`, env `OXE_STREAM_STALL_SECS`, default 30 s)
    pub stream_stall: std::time::Duration,
}

pub enum Reply {
    Json(u16, Json),
    /// SSE stream: each item is a complete `data: ...\n\n` frame
    Stream(mpsc::Receiver<String>),
}

fn bad(msg: impl AsRef<str>) -> Reply {
    Reply::Json(400, api::error_json(400, msg.as_ref()))
}

impl<E: Engine> State<E> {
    pub fn new(cfg: Config, tok: Tokenizer, worker: Worker<E>) -> Result<State<E>, String> {
        let bos = tok.bos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
        let eos = tok.eos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
        let tmpl = match &tok.chat_template {
            Some(src) => Some(ChatTemplate::new(src, &bos, &eos).map_err(|e| format!("chat template: {e:#}"))?),
            None => None,
        };
        let n_ctx = worker.engine.n_ctx();
        let n_vocab = worker.engine.n_vocab();
        let user_delim = tok.chat_template.as_deref().filter(|t| t.contains("<|im_start|>")).map(|_| tok.encode("<|im_start|>user", false, true)).filter(|d| d.len() >= 2);
        Ok(State {
            cfg,
            tok: Arc::new(tok),
            tmpl,
            worker: Mutex::new(worker),
            n_ctx,
            n_vocab,
            user_delim,
            idle: Mutex::new(None),
            reqs: Default::default(),
            idle_done: Default::default(),
            stream_stall: std::time::Duration::from_secs_f64(
                std::env::var("OXE_STREAM_STALL_SECS").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0).unwrap_or(30.0),
            ),
        })
    }

    /// A request is starting: an idle re-prefill stops at its next chunk boundary (its partial
    /// state is a prefix of the conversation and stays reusable) or does not start at all.
    pub fn preempt_idle(&self) -> u64 {
        if let Some(f) = self.idle.lock().unwrap().take() {
            f.store(true, Ordering::Relaxed);
        }
        self.reqs.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn models(&self) -> Json {
        json!({
            "models": [{
                "name": self.cfg.model_name, "model": self.cfg.model_name, "modified_at": "", "size": "", "digest": "",
                "type": "model", "description": "", "tags": [""], "capabilities": ["completion"], "parameters": "",
                "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""], "parameter_size": "", "quantization_level": ""}
            }],
            "object": "list",
            "data": [{
                "id": self.cfg.model_name, "aliases": [self.cfg.model_name], "tags": [], "object": "model",
                "created": api::now_secs(), "owned_by": "llamacpp",
                "meta": {"vocab_type": 2, "n_vocab": self.n_vocab, "n_ctx": self.n_ctx, "n_ctx_train": self.n_ctx}
            }]
        })
    }

    fn props(&self) -> Json {
        json!({
            "default_generation_settings": {"params": api::generation_settings(&GenParams::default()), "n_ctx": self.n_ctx},
            "total_slots": 1,
            "model_alias": self.cfg.model_name,
            "model_path": self.cfg.model_path,
            "modalities": {"vision": false, "video": false, "audio": false},
            "chat_template": self.tmpl.as_ref().map(|t| t.source.clone()).unwrap_or_default(),
            "bos_token": self.tmpl.as_ref().map(|t| t.bos_token.clone()).unwrap_or_default(),
            "eos_token": self.tmpl.as_ref().map(|t| t.eos_token.clone()).unwrap_or_default(),
            "build_info": self.cfg.fingerprint,
            "is_sleeping": false,
            "engine_status": self.cfg.engine_status.clone().unwrap_or(Json::Null),
        })
    }

    /// Tokenize a `/completion` prompt: string, token array, or mixed array (first string
    /// part gets add_special).
    fn tokenize_prompt(&self, p: &Json, add_special: bool) -> Result<Vec<Token>, String> {
        match p {
            Json::String(s) => Ok(self.tok.encode(s, add_special, true)),
            Json::Array(a) => {
                let mut out = Vec::new();
                let mut first = true;
                for x in a {
                    match x {
                        // ids above u32 used to wrap onto valid ids (`as`)
                        Json::Number(n) => out.push(n.as_u64().and_then(|v| Token::try_from(v).ok()).ok_or("bad token")?),
                        Json::String(s) => {
                            out.extend(self.tok.encode(s, add_special && first, true));
                            first = false;
                        }
                        _ => return Err("prompt array must contain strings or token ids".into()),
                    }
                }
                Ok(out)
            }
            _ => Err("\"prompt\" must be a string or an array".into()),
        }
    }

    fn check_tokens(&self, toks: &[Token]) -> Result<(), String> {
        if let Some(t) = toks.iter().find(|&&t| t as usize >= self.n_vocab) {
            return Err(format!("token id {t} out of vocab"));
        }
        if toks.is_empty() {
            return Err("empty prompt".into());
        }
        Ok(())
    }
}

/// Route a request. Blocking work (generation) is done in `spawn_blocking`.
pub async fn handle<E: Engine>(st: Arc<State<E>>, method: &str, path: &str, query: &str, body: &[u8]) -> Reply {
    // exact paths like llama-server (REVIEW S4: a trimmed `/v1/chat/completions/` ran as a chat
    // completion while the controller did not treat it as inference)
    let parse = || -> Result<Json, Reply> {
        if body.is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_slice(body).map_err(|e| bad(format!("invalid JSON body: {e}")))
    };
    match (method, path) {
        ("GET", "/health") | ("GET", "/v1/health") => Reply::Json(200, json!({"status": "ok"})),
        ("GET", "/v1/models") | ("GET", "/models") => Reply::Json(200, st.models()),
        ("GET", "/props") => Reply::Json(200, st.props()),
        ("GET", "/slots") => {
            if !st.cfg.slots {
                return Reply::Json(501, api::error_json(501, "This server does not support slots endpoint."));
            }
            let busy = st.worker.try_lock().is_err();
            Reply::Json(200, json!([{"id": 0, "n_ctx": st.n_ctx, "speculative": false, "is_processing": busy}]))
        }
        ("POST", p) if p.starts_with("/slots") => {
            let _ = query;
            Reply::Json(501, api::error_json(501, "slot save/restore is not supported by oxe-server yet"))
        }
        ("POST", "/tokenize") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            let Some(content) = b.get("content").and_then(Json::as_str) else {
                return Reply::Json(200, json!({"tokens": []}));
            };
            let add_special = b.get("add_special").and_then(Json::as_bool).unwrap_or(false);
            let parse_special = b.get("parse_special").and_then(Json::as_bool).unwrap_or(true);
            let toks = st.tok.encode(content, add_special, parse_special);
            if b.get("with_pieces").and_then(Json::as_bool).unwrap_or(false) {
                let v: Vec<Json> = toks
                    .iter()
                    .map(|&t| {
                        let p = st.tok.piece(t, true);
                        let piece = match String::from_utf8(p.clone()) {
                            Ok(s) => Json::String(s),
                            Err(_) => p.iter().map(|&b| Json::from(b)).collect(),
                        };
                        json!({"id": t, "piece": piece})
                    })
                    .collect();
                Reply::Json(200, json!({"tokens": v}))
            } else {
                Reply::Json(200, json!({"tokens": toks}))
            }
        }
        ("POST", "/detokenize") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            let toks: Vec<Token> =
                b.get("tokens").and_then(Json::as_array).map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| Token::try_from(v).unwrap_or(Token::MAX))).collect()).unwrap_or_default();
            if let Err(e) = st.check_tokens(&toks).or_else(|e| if toks.is_empty() { Ok(()) } else { Err(e) }) {
                return bad(e);
            }
            Reply::Json(200, json!({"content": st.tok.decode(&toks, true)}))
        }
        ("POST", "/apply-template") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            match prepare_chat(&st, &b) {
                Ok(pc) => Reply::Json(200, json!({"prompt": pc.applied.prompt})),
                Err(r) => r,
            }
        }
        ("POST", "/completion") | ("POST", "/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            completion(st, b).await
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            chat_completion(st, b).await
        }
        _ => Reply::Json(404, json!({"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}})),
    }
}

struct PreparedChat {
    applied: chat::Applied,
    inputs: chat::ChatInputs,
}

fn prepare_chat<E: Engine>(st: &State<E>, body: &Json) -> Result<PreparedChat, Reply> {
    let Some(tmpl) = &st.tmpl else {
        return Err(Reply::Json(500, api::error_json(500, "model has no chat template")));
    };
    let inputs = chat::chat_inputs(body, &st.cfg.chat_opts, tmpl).map_err(|e| bad(e.0))?;
    let applied = chat::apply(tmpl, &inputs).map_err(|e| bad(format!("chat template error: {e:#}")))?;
    Ok(PreparedChat { applied, inputs })
}

fn gen_error_reply(e: GenError) -> Reply {
    match e {
        GenError::ExceedContext(n, c) => Reply::Json(400, api::exceed_context_json(n, c)),
        GenError::Engine(e) => Reply::Json(500, api::error_json(500, &e.0)),
        GenError::Cancelled => Reply::Json(500, api::error_json(500, "cancelled")),
    }
}

fn gen_error_json(e: GenError) -> Json {
    match gen_error_reply(e) {
        Reply::Json(_, j) => j,
        _ => unreachable!(),
    }
}

/// Run the generation on a blocking thread. For streams the first event decides between an
/// HTTP error and a 200 SSE stream, like llama-server (which pre-fetches the first result).
async fn run_blocking<E: Engine, F>(st: Arc<State<E>>, prompt: Vec<Token>, mut gp: GenParams, mut on_partial: F) -> Result<GenResult, GenError>
where
    F: FnMut(Partial) -> bool + Send + 'static,
{
    /// Sets the cancel flag if this future is dropped before the generation returns: hyper
    /// drops a non-streamed request's future when its client disconnects.
    struct DropCancel(Arc<AtomicBool>, bool);
    impl Drop for DropCancel {
        fn drop(&mut self) {
            if !self.1 {
                self.0.store(true, Ordering::Relaxed);
            }
        }
    }
    let flag = gp.cancel.get_or_insert_with(|| Arc::new(AtomicBool::new(false))).clone();
    let mut guard = DropCancel(flag, false);
    let r = tokio::task::spawn_blocking(move || {
        let mut w = st.worker.lock().unwrap_or_else(|p| p.into_inner());
        if gp.cancelled() {
            return Err(GenError::Cancelled); // the client left while queued for the worker
        }
        w.generate(&st.tok, &prompt, &gp, &mut on_partial)
    })
    .await
    .unwrap_or_else(|e| Err(GenError::Engine(crate::engine::EngineError(format!("worker panicked: {e}")))));
    guard.1 = true;
    r
}

/// SX-2 idle speculative re-prefill. After a chat turn that ended the query (`stop`, no tool
/// call), the next request will be this conversation + the reply + a new user message, and the
/// template renders the earlier assistant turns differently once a user message follows (the
/// empty think blocks are dropped), so that request would re-prefill the whole query. Render that
/// history form now (messages + reply + an empty user turn, cut before the user message) and
/// prefill it while the client reads. Lossless: the tokens are the next prompt's prefix; the
/// raw session is parked (a retry still finds it); any request preempts the job.
fn spawn_idle_reprefill<E: Engine>(st: &Arc<State<E>>, b: &Json, reply: Json, req_no: u64) {
    if !st.cfg.idle_reprefill {
        return;
    }
    let Some(d) = st.user_delim.clone() else { return };
    let mut body = b.clone();
    let Some(msgs) = body.get_mut("messages").and_then(Json::as_array_mut) else { return };
    msgs.push(reply);
    msgs.push(json!({"role": "user", "content": ""}));
    let Ok(pc) = prepare_chat(st, &body) else { return };
    let toks = st.tok.encode(&pc.applied.prompt, true, true);
    let Some(cut) = toks.windows(d.len()).rposition(|w| w == &d[..]) else { return };
    let hist = toks[..cut].to_vec();
    let Ok(mut gp) = api::gen_params(&body, true) else { return };
    gp.first_user = hist.windows(d.len()).position(|w| w == &d[..]);
    gp.msg_starts = hist.iter().enumerate().filter(|(_, &t)| t == d[0]).map(|(i, _)| i).collect();
    gp.query_start = false;
    let flag = Arc::new(AtomicBool::new(false));
    gp.cancel = Some(flag.clone());
    *st.idle.lock().unwrap() = Some(flag);
    let st2 = st.clone();
    tokio::task::spawn_blocking(move || {
        {
            let mut w = st2.worker.lock().unwrap_or_else(|p| p.into_inner());
            // a newer request started (or is queued): leave the worker to it
            if gp.cancelled() || st2.reqs.load(Ordering::SeqCst) != req_no {
                drop(w);
            } else {
                match w.prefill_idle(&hist, &gp) {
                    Ok(n) if n > 0 => eprintln!("oxe-server: idle re-prefill: {n} tokens of the history form"),
                    Ok(_) => {}
                    Err(GenError::Cancelled) => eprintln!("oxe-server: idle re-prefill: preempted by a request"),
                    Err(e) => eprintln!("oxe-server: idle re-prefill failed: {e:?}"),
                }
            }
        }
        st2.idle_done.fetch_add(1, Ordering::SeqCst);
    });
}

/// S8 (REVIEW-v0.1.1): send one SSE frame from the generation thread; false = stop generating.
/// A client that keeps its socket open but stops reading fills the channel; `blocking_send`
/// then held the worker (and every later request) for ever. Now, after `stall` without room
/// for the next frame, the stream is given up: the generation is cancelled like a disconnect.
pub(crate) fn send_frame(tx: &mpsc::Sender<String>, frame: String, stall: std::time::Duration) -> bool {
    use tokio::sync::mpsc::error::TrySendError;
    let t0 = std::time::Instant::now();
    let mut frame = frame;
    loop {
        match tx.try_send(frame) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => return false,
            Err(TrySendError::Full(f)) => {
                if t0.elapsed() >= stall {
                    eprintln!("oxe-server: stream client took no data for {:.0} s: stream dropped, generation cancelled", stall.as_secs_f64());
                    return false;
                }
                frame = f;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

/// The async side of [`send_frame`] (final frames after the worker is free): false when the
/// client is gone or has taken nothing for `stall`.
async fn send_final(tx: &mpsc::Sender<String>, frame: String, stall: std::time::Duration) -> bool {
    matches!(tokio::time::timeout(stall, tx.send(frame)).await, Ok(Ok(())))
}

/// Streams: cancel when the client's end of the SSE channel is dropped (the client went
/// away), also while the prompt is still being prefilled.
fn cancel_on_close(gp: &mut GenParams, tx: &mpsc::Sender<String>) {
    let flag = Arc::new(AtomicBool::new(false));
    gp.cancel = Some(flag.clone());
    // A weak handle: holding a Sender here would keep the channel open after the response
    // ends (the client would wait for the end of the stream forever).
    let weak = tx.downgrade();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_millis(100));
        loop {
            tick.tick().await;
            match weak.upgrade() {
                None => return, // the response finished: every sender is gone
                Some(tx) => {
                    if tx.is_closed() {
                        flag.store(true, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    });
}

/// Structured-output fields that this server does not implement: refuse them instead of
/// silently ignoring them (ENGINE-AUDIT F7).
fn unsupported_constraint(b: &Json) -> Option<String> {
    for k in ["response_format", "grammar", "json_schema"] {
        match b.get(k) {
            None | Some(Json::Null) => {}
            Some(Json::Object(o)) if k == "response_format" && o.get("type").and_then(Json::as_str) == Some("text") => {}
            Some(_) => return Some(format!("\"{k}\" (constrained/structured output) is not supported by oxe-server; remove it or use tools")),
        }
    }
    None
}

async fn completion<E: Engine>(st: Arc<State<E>>, b: Json) -> Reply {
    st.preempt_idle();
    if let Some(m) = unsupported_constraint(&b) {
        return bad(m);
    }
    let mut gp = match api::gen_params(&b, false) {
        Ok(g) => g,
        Err(e) => return bad(e),
    };
    let prompt_json = b.get("prompt").cloned().unwrap_or(Json::String(String::new()));
    let prompt = match st.tokenize_prompt(&prompt_json, true) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    if let Err(e) = st.check_tokens(&prompt) {
        return bad(e);
    }
    if prompt.len() >= st.n_ctx {
        return Reply::Json(400, api::exceed_context_json(prompt.len(), st.n_ctx));
    }
    let prompt_text = match &prompt_json {
        Json::String(s) => s.clone(),
        _ => st.tok.decode(&prompt, true),
    };
    let stream = b.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let post = b.get("post_sampling_probs").and_then(Json::as_bool).unwrap_or(false);
    let return_tokens = b.get("return_tokens").and_then(Json::as_bool).unwrap_or(false);
    let model = st.cfg.model_name.clone();
    if !stream {
        return match run_blocking(st.clone(), prompt, gp.clone(), |_| true).await {
            Ok(r) => Reply::Json(200, api::completion_final(&model, &prompt_text, &r, false, &gp, return_tokens, post)),
            Err(e) => gen_error_reply(e),
        };
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let n_prompt = prompt.len();
    let tx2 = tx.clone();
    let st2 = st.clone();
    let stall = st.stream_stall;
    tokio::spawn(async move {
        let res = run_blocking(st2, prompt, gp.clone(), move |p: Partial| {
            let j = api::completion_partial(&p.text, p.n_decoded, n_prompt, p.probs.as_ref(), post);
            send_frame(&tx2, api::sse(&j), stall)
        })
        .await;
        let frame = match res {
            Ok(r) => api::sse(&api::completion_final(&model, &prompt_text, &r, true, &gp, return_tokens, post)),
            Err(GenError::Cancelled) => return,
            Err(e) => api::sse(&gen_error_json(e)),
        };
        let _ = send_final(&tx, frame, stall).await;
    });
    Reply::Stream(rx)
}

async fn chat_completion<E: Engine>(st: Arc<State<E>>, b: Json) -> Reply {
    let req_no = st.preempt_idle();
    if let Some(m) = unsupported_constraint(&b) {
        return bad(m);
    }
    let pc = match prepare_chat(&st, &b) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let mut gp = match api::gen_params(&b, true) {
        Ok(g) => g,
        Err(e) => return bad(e),
    };
    gp.special = false;
    let prompt = st.tok.encode(&pc.applied.prompt, true, true);
    if let Err(e) = st.check_tokens(&prompt) {
        return bad(e);
    }
    if prompt.len() >= st.n_ctx {
        return Reply::Json(400, api::exceed_context_json(prompt.len(), st.n_ctx));
    }
    gp.first_user = st.user_delim.as_ref().and_then(|d| prompt.windows(d.len()).position(|w| w == &d[..]));
    if let Some(d) = &st.user_delim {
        gp.msg_starts = prompt.iter().enumerate().filter(|(_, &t)| t == d[0]).map(|(i, _)| i).collect();
    }
    gp.query_start = b.get("messages").and_then(Json::as_array).and_then(|m| m.last()).and_then(|m| m.get("role")).and_then(Json::as_str) == Some("user");
    let tmpl = st.tmpl.as_ref().unwrap();
    let pcfg = ParserConfig {
        format: pc.applied.format,
        tools: pc.inputs.tools.clone(),
        tool_choice: pc.inputs.tool_choice,
        parallel_tool_calls: pc.inputs.parallel_tool_calls,
        reasoning_format: pc.inputs.reasoning_format,
        supports_thinking: tmpl.supports_thinking,
        // a valid call with its parameters in another order is still a call (llama.cpp's
        // grammar drops it: required parameters first)
        lenient: true,
    };
    let meta = Arc::new(ChatMeta {
        id: format!("chatcmpl-{}", chat::random_string()),
        model: st.cfg.model_name.clone(),
        created: api::now_secs(),
        fingerprint: st.cfg.fingerprint.clone(),
    });
    let stream = b.get("stream").and_then(Json::as_bool).unwrap_or(false);
    let include_usage = b.pointer("/stream_options/include_usage").and_then(Json::as_bool).unwrap_or(false);
    let want_probs = gp.n_probs > 0;
    let gen_prompt = pc.applied.generation_prompt.clone();
    if !stream {
        let r = match run_blocking(st.clone(), prompt, gp, |_| true).await {
            Ok(r) => r,
            Err(e) => return gen_error_reply(e),
        };
        let mut sess = Session::new(pcfg, gen_prompt, Box::new(chat::random_string));
        sess.incremental = std::env::var_os("OXE_SERVER_LEGACY").is_none();
        if let Err(e) = sess.feed(&r.text, true) {
            return Reply::Json(500, api::error_json(500, &e));
        }
        if api::finish_reason(r.stop, !sess.msg().tool_calls.is_empty()) == "stop" {
            spawn_idle_reprefill(&st, &b, sess.msg().to_json(), req_no);
        }
        return Reply::Json(200, api::chat_response(&meta, sess.msg(), &r, want_probs, false));
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let mut sess0 = Session::new(pcfg, gen_prompt, Box::new(chat::random_string));
    // OXE_SERVER_LEGACY=1 (A/B diagnostic only): full re-parse every token
    sess0.incremental = std::env::var_os("OXE_SERVER_LEGACY").is_none();
    let sess = Arc::new(Mutex::new(sess0));
    let st2 = st.clone();
    let stall = st.stream_stall;
    tokio::spawn(async move {
        let tx2 = tx.clone();
        let m2 = meta.clone();
        let s2 = sess.clone();
        let res = run_blocking(st2, prompt, gp, move |p: Partial| {
            let mut frames = Vec::new();
            if p.n_decoded == 1 {
                frames.push(api::chat_chunk(&m2, json!({"role": "assistant", "content": null}), None));
            }
            let diffs = s2.lock().unwrap().feed(&p.text, false);
            match diffs {
                Ok(ds) => {
                    for d in ds {
                        frames.push(api::chat_chunk(&m2, d.to_delta_json(), None));
                    }
                }
                Err(e) => {
                    let _ = send_frame(&tx2, api::sse(&api::error_json(500, &e)), stall);
                    return false;
                }
            }
            for f in frames {
                if !send_frame(&tx2, api::sse(&f), stall) {
                    return false;
                }
            }
            true
        })
        .await;
        let r = match res {
            Ok(r) => r,
            Err(GenError::Cancelled) => return,
            Err(e) => {
                let _ = send_final(&tx, api::sse(&gen_error_json(e)), stall).await;
                return;
            }
        };
        let (diffs, has_calls) = {
            let mut s = sess.lock().unwrap();
            let d = s.feed("", true);
            let has = !s.msg().tool_calls.is_empty();
            (d, has)
        };
        let mut frames = Vec::new();
        match diffs {
            Ok(ds) => {
                for d in ds {
                    frames.push(api::chat_chunk(&meta, d.to_delta_json(), None));
                }
            }
            Err(e) => {
                let _ = send_final(&tx, api::sse(&api::error_json(500, &e)), stall).await;
                return;
            }
        }
        frames.push(api::chat_chunk(&meta, json!({}), Some(api::finish_reason(r.stop, has_calls))));
        if api::finish_reason(r.stop, has_calls) == "stop" {
            let reply = sess.lock().unwrap().msg().to_json();
            spawn_idle_reprefill(&st, &b, reply, req_no);
        }
        if include_usage {
            frames.push(api::chat_usage_chunk(&meta, &r));
        }
        if let Some(Json::Object(last)) = frames.last_mut() {
            last.insert("timings".into(), api::timings_json(&r.timings));
        }
        for f in frames {
            if !send_final(&tx, api::sse(&f), stall).await {
                return;
            }
        }
        let _ = send_final(&tx, "data: [DONE]\n\n".to_string(), stall).await;
    });
    Reply::Stream(rx)
}
