//! Endpoint logic, independent of the HTTP library: a request (method, path, JSON body) gives
//! either a complete JSON response or a stream of SSE strings.

use crate::api::{self, ChatMeta};
use crate::engine::Engine;
use crate::generate::{GenError, GenParams, GenResult, Partial, Progress, Worker};
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
    /// LONG-APPEND (v0.1.3 #7): SSE comment interval while a streamed request waits for the worker
    /// or prefills (`--sse-keepalive SECS`, env `OXE_SSE_KEEPALIVE_SECS`, default 15; 0 = off)
    pub sse_keepalive: std::time::Duration,
    /// CACHE-PERFECT ECHO records (see `echo.rs`); config from `OXE_ECHO*` (the `--echo*` flags)
    pub echo: Mutex<crate::echo::EchoStore>,
    /// token ids that open an assistant turn (`<|im_start|>assistant`), None if the template has none
    pub asst_delim: Option<Vec<Token>>,
    /// the turn end id (`<|im_end|>`)
    pub im_end: Option<Token>,
    /// CACHE-PERFECT P2: the worker has a state store (writes after each reply)
    pub state_cache: bool,
    /// cancel flag of the pending/running state-store write
    pub persist: Mutex<Option<Arc<AtomicBool>>>,
    /// state-store writes finished (tests)
    pub persist_done: std::sync::atomic::AtomicU64,
    /// v0.1.3: held by a post-reply state write through all its phases (the off-lock middle one
    /// included). The cache endpoints take it first, so they see a store with no write in flight;
    /// generation requests never take it. Writes queue behind each other in reply order.
    pub persist_gate: Mutex<()>,
    /// cancel flag of the running generation (SIGTERM cancels it so the bounded shutdown write runs)
    pub running: Mutex<Option<Arc<AtomicBool>>>,
    /// cancel flag of the pending/running preload (P2b; any request raises it)
    pub preload: Mutex<Option<Arc<AtomicBool>>>,
    /// the echo records' disk directory while the disk budget is 0 (Codex P2/P3 review 2)
    pub echo_dir_off: Mutex<Option<std::path::PathBuf>>,
    /// the last preload's status for `GET /state-cache` (idle|running|done|cancelled|failed|used|none|live)
    pub preload_status: Mutex<Json>,
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
        let state_cache = worker.states.is_some();
        let n_vocab = worker.engine.n_vocab();
        let user_delim = tok.chat_template.as_deref().filter(|t| t.contains("<|im_start|>")).map(|_| tok.encode("<|im_start|>user", false, true)).filter(|d| d.len() >= 2);
        let asst_delim = tok.chat_template.as_deref().filter(|t| t.contains("<|im_start|>")).map(|_| tok.encode("<|im_start|>assistant", false, true)).filter(|d| d.len() >= 2);
        let im_end = tok.token_id("<|im_end|>");
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
            sse_keepalive: sse_keepalive_from_env(),
            echo: Mutex::new(crate::echo::EchoStore::new(crate::echo::EchoConfig::from_env())),
            asst_delim,
            im_end,
            state_cache,
            persist: Mutex::new(None),
            persist_done: Default::default(),
            persist_gate: Mutex::new(()),
            running: Mutex::new(None),
            preload: Mutex::new(None),
            preload_status: Mutex::new(json!({"status": "idle"})),
            echo_dir_off: Mutex::new(None),
        })
    }

    /// A request is starting: an idle re-prefill stops at its next chunk boundary (its partial
    /// state is a prefix of the conversation and stays reusable) or does not start at all.
    pub fn preempt_idle(&self) -> u64 {
        if let Some(f) = self.idle.lock().unwrap().take() {
            f.store(true, Ordering::Relaxed);
        }
        if let Some(f) = self.persist.lock().unwrap().take() {
            f.store(true, Ordering::Relaxed);
        }
        if let Some(f) = self.preload.lock().unwrap().take() {
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

    /// Tokenize a rendered chat prompt and find its message starts and assistant turn spans from
    /// the template's own delimiter offsets (CACHE-PERFECT M1, `chat::request::template_marks`):
    /// delimiter text inside a message becomes delimiter tokens too, so it is never counted. The
    /// tokens are exactly `encode(prompt, true, true)`.
    fn render(&self, pc: &PreparedChat) -> Rendered {
        let text = &pc.applied.prompt;
        let marks = match (&self.tmpl, &self.asst_delim) {
            (Some(t), Some(_)) => chat::request::template_marks(t, &pc.inputs, text).ok().flatten(),
            _ => None,
        };
        let Some((starts, ends)) = marks else {
            let tokens = self.tok.encode(text, true, true);
            let (msg_starts, first_user) = match &self.user_delim {
                Some(d) => (tokens.iter().enumerate().filter(|(_, &t)| t == d[0]).map(|(i, _)| i).collect(), tokens.windows(d.len()).position(|w| w == &d[..])),
                None => (Vec::new(), None),
            };
            let last_user = self.user_delim.as_ref().and_then(|d| tokens.windows(d.len()).rposition(|w| w == &d[..]));
            let user_starts = match &self.user_delim {
                Some(d) => tokens.windows(d.len()).enumerate().filter(|(_, w)| *w == &d[..]).map(|(i, _)| i).collect(),
                None => Vec::new(),
            };
            return Rendered { tokens, asst: None, msg_starts, first_user, last_user, user_starts };
        };
        let mut cuts: Vec<usize> = starts.iter().chain(&ends).copied().collect();
        cuts.sort_unstable();
        cuts.dedup();
        let (tokens, at) = self.tok.encode_cuts(text, true, true, &cuts);
        let tok_at = |b: usize| cuts.binary_search(&b).ok().and_then(|i| at[i]);
        let is = |b: usize, role: &str| text[b + chat::request::IM_START.len()..].starts_with(role);
        let msg_starts: Vec<usize> = starts.iter().filter_map(|&b| tok_at(b)).collect();
        let first_user = starts.iter().find(|&&b| is(b, "user")).and_then(|&b| tok_at(b));
        let last_user = starts.iter().rev().find(|&&b| is(b, "user")).and_then(|&b| tok_at(b));
        let user_starts: Vec<usize> = starts.iter().filter(|&&b| is(b, "user")).filter_map(|&b| tok_at(b)).collect();
        let mut asst = Some(Vec::new());
        let n_starts = starts.len();
        for (k, &b) in starts.iter().enumerate().filter(|(_, &b)| is(b, "assistant")) {
            let next = starts.get(k + 1).copied().unwrap_or(text.len());
            match ends.iter().find(|&&e| e > b).filter(|&&e| e < next) {
                Some(&e) => match (tok_at(b), tok_at(e), asst.as_mut()) {
                    (Some(s), Some(t), Some(v)) => v.push((s, t)),
                    _ => asst = None,
                },
                // the generation prompt: the last opener, no turn end after it
                None if k + 1 == n_starts && pc.inputs.add_generation_prompt => {}
                None => asst = None,
            }
        }
        if starts.len() != msg_starts.len() {
            asst = None;
        }
        Rendered { tokens, asst, msg_starts, first_user, last_user, user_starts }
    }

    /// v0.1.3 pre-think point of a fed chat prompt: thinking on (the generation prompt ends in an
    /// open `<think>`), the position just before that `<think>`, when the prompt ends with exactly
    /// the tokens of that tail. None otherwise (thinking off, another template).
    fn pre_think(&self, prompt: &[Token], generation_prompt: &str) -> Option<usize> {
        let at = generation_prompt.rfind("<think>")?;
        let tail = &generation_prompt[at..];
        if tail.contains("</think>") {
            return None;
        }
        let t = self.tok.encode(tail, false, true);
        (!t.is_empty() && t.len() < prompt.len() && prompt.ends_with(&t)).then(|| prompt.len() - t.len())
    }

    /// ECHO: the prompt to feed for a rendered chat prompt (see `echo.rs`).
    fn splice(&self, r: &Rendered, inputs: &chat::ChatInputs) -> (Vec<Token>, crate::echo::SpliceReport) {
        match (&self.asst_delim, self.im_end) {
            (Some(_), Some(_)) => self.echo.lock().unwrap_or_else(|p| p.into_inner()).splice(&r.tokens, &inputs.messages, &inputs.reasoning_sent, r.asst.as_deref()),
            _ => (r.tokens.clone(), Default::default()),
        }
    }

    /// ECHO: keep a finished turn (see `echo.rs`).
    fn echo_record(&self, prompt: &[Token], msg: &chat::ChatMsg, r: &GenResult, session: Option<&str>) {
        let (Some(a), Some(e)) = (&self.asst_delim, self.im_end) else { return };
        let eog = r.stop == crate::generate::StopType::Eos && r.tokens.last().is_some_and(|&t| self.tok.is_eog(t));
        self.echo.lock().unwrap_or_else(|p| p.into_inner()).record(prompt, a, e, msg, &r.tokens, eog, session);
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
    handle_session(st, method, path, query, body, None).await
}

/// [`handle`] with the conversation's session hint (`X-Oxe-Session`, forwarded by the
/// controller). A hint only: echo records are found by their token chain, never by this id.
pub async fn handle_session<E: Engine>(st: Arc<State<E>>, method: &str, path: &str, query: &str, body: &[u8], session: Option<String>) -> Reply {
    handle_request(st, method, path, query, body, session, None).await
}

/// [`handle_session`] with the controller's turn label (`X-Oxe-Turn-Kind`: chat | compaction |
/// post-compaction; v0.1.3). A label only places checkpoints (a compaction request's cut); a wrong
/// one (an instruction quoted in a normal turn) costs one checkpoint slot, never a wrong state.
pub async fn handle_request<E: Engine>(st: Arc<State<E>>, method: &str, path: &str, query: &str, body: &[u8], session: Option<String>, turn_kind: Option<String>) -> Reply {
    handle_policy(st, method, path, query, body, session, turn_kind, false).await
}

/// [`handle_request`] with the controller's state policy (`X-Oxe-State-Policy: ephemeral`, LAN
/// share: a request from another device). An ephemeral generation has RAM-only, request-scoped
/// state (see [`GenParams::ephemeral`]): no echo splice or record, no session hint, no post-reply
/// state write, no idle re-prefill, no prefix reuse; the local live session is held and restored.
#[allow(clippy::too_many_arguments)]
pub async fn handle_policy<E: Engine>(st: Arc<State<E>>, method: &str, path: &str, query: &str, body: &[u8], session: Option<String>, turn_kind: Option<String>, ephemeral: bool) -> Reply {
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
        // CACHE-PERFECT P2 state store (the controller's /_local/cache routes forward here)
        ("GET", "/state-cache") => {
            let st2 = st.clone();
            let r = tokio::task::spawn_blocking(move || {
                let _gate = st2.persist_gate.lock().unwrap_or_else(|p| p.into_inner());
                let w = st2.worker.lock().unwrap_or_else(|p| p.into_inner());
                w.states.as_ref().map(|s| {
                    let mut u = s.usage();
                    u["preload"] = st2.preload_status.lock().unwrap().clone();
                    u
                })
            })
            .await
            .ok()
            .flatten();
            match r {
                Some(j) => Reply::Json(200, j),
                None => Reply::Json(404, api::error_json(404, "state cache is off")),
            }
        }
        // P2b preload (T1-WARM contract): restore a session's newest snapshot into the idle engine
        // in the background; any request cancels it within one block (~5 ms) and it never queues
        // ahead of one (it only runs when it gets the worker at once and no request came since)
        ("POST", "/state-cache/preload") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            if !st.state_cache {
                return Reply::Json(404, api::error_json(404, "state cache is off"));
            }
            let hint = b.get("session").and_then(Json::as_str).map(str::to_string);
            let req_no = st.reqs.load(Ordering::SeqCst);
            let flag = Arc::new(AtomicBool::new(false));
            {
                let w = match st.worker.try_lock() {
                    Ok(w) => w,
                    Err(_) => return Reply::Json(200, json!({"status": "busy"})),
                };
                if !w.cache.tokens.is_empty() || w.preloaded.is_some() {
                    return Reply::Json(200, json!({"status": "live"}));
                }
                if w.states.as_ref().is_none_or(|s| s.budget == 0 || s.newest_plan(None).is_none()) {
                    return Reply::Json(200, json!({"status": "none"}));
                }
            }
            *st.preload.lock().unwrap() = Some(flag.clone());
            *st.preload_status.lock().unwrap() = json!({"status": "running"});
            let st2 = st.clone();
            tokio::task::spawn_blocking(move || {
                let done = |status: &str, pos: usize, ms: f64| *st2.preload_status.lock().unwrap() = json!({"status": status, "pos": pos, "ms": ms});
                // never wait for the worker: a request (or a write) holds it, so the preload yields
                let Ok(mut w) = st2.worker.try_lock() else { return done("cancelled", 0, 0.0) };
                if flag.load(Ordering::Relaxed) || st2.reqs.load(Ordering::SeqCst) != req_no {
                    return done("cancelled", 0, 0.0);
                }
                let (status, pos, ms) = w.preload(hint.as_deref(), &flag);
                done(status, pos, ms);
            });
            Reply::Json(202, json!({"status": "started"}))
        }
        ("POST", "/state-cache/budget") | ("POST", "/state-cache/clear") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            let clear = path.ends_with("/clear");
            let st2 = st.clone();
            let r = tokio::task::spawn_blocking(move || -> Result<Json, (u16, String)> {
                // after the state write in flight (if any), then refused while a request runs
                let _gate = st2.persist_gate.lock().unwrap_or_else(|p| p.into_inner());
                let mut w = st2.worker.try_lock().map_err(|_| (409, "a request is running".to_string()))?;
                let w = &mut *w;
                if w.states.is_none() {
                    return Err((404, "state cache is off".into()));
                }
                let mut echo = st2.echo.lock().unwrap_or_else(|p| p.into_inner());
                if !clear {
                    let bytes = b.get("bytes").and_then(Json::as_u64).ok_or((400, "\"bytes\" must be a non-negative integer".to_string()))?;
                    let s = w.states.as_mut().unwrap();
                    let was_off = s.budget == 0;
                    let freed = s.set_budget(bytes);
                    // Codex P2/P3 review 2: budget 0 = no conversation state goes to disk at all: the
                    // segmented store, the echo records' files and the legacy prefix / session stores
                    // (parks fall back to RAM). Raising it again brings all of them back.
                    if bytes == 0 && !was_off {
                        *st2.echo_dir_off.lock().unwrap() = echo.set_disk(None);
                        w.legacy_off = Some((w.store.take(), w.sessions.take()));
                    } else if bytes > 0 && was_off {
                        if let Some(d) = st2.echo_dir_off.lock().unwrap().take() {
                            echo.set_disk(Some(d));
                        }
                        if let Some((p, se)) = w.legacy_off.take() {
                            w.store = p;
                            w.sessions = se;
                        }
                    }
                    let s = w.states.as_ref().unwrap();
                    return Ok(json!({"budget_bytes": s.budget, "used_bytes": s.used_bytes(), "freed_bytes": freed, "legacy_stores": if w.legacy_off.is_some() { "off" } else { "on" }}));
                }
                let scope = b.get("scope").and_then(Json::as_str).unwrap_or("all").to_string();
                let s = w.states.as_mut().unwrap();
                let other_model = b.get("model").and_then(Json::as_str).filter(|m| *m != s.ident.model_key());
                if scope == "session" {
                    // Codex P2/P3 review 1: purge the session from every tier, keep what others need
                    let sid = b.get("session").and_then(Json::as_str).ok_or((400, "session scope needs a session id".to_string()))?.to_string();
                    let m = s.session_match(&sid).ok_or((400, "unknown session".to_string()))?;
                    let freed = s.clear("session", Some(&sid)).map_err(|e| (400, e))?;
                    let owns = |t: &[u32]| m.owns(t);
                    let mut ram = 0;
                    let pre = w.preloaded.as_ref().is_some_and(|p| p.plan.sid == sid);
                    if pre || (!w.cache.tokens.is_empty() && owns(&w.cache.tokens)) {
                        w.cache.reset(&mut w.engine);
                        w.live_blocks.clear();
                        w.preloaded = None;
                        ram += 1;
                    }
                    ram += w.cache.drop_parked_where(&owns);
                    let mut legacy = 0;
                    for ps in [w.store.as_mut(), w.sessions.as_mut()].into_iter().flatten() {
                        legacy += ps.remove_where(&owns);
                    }
                    if let Some((p, se)) = w.legacy_off.as_mut() {
                        for ps in [p.as_mut(), se.as_mut()].into_iter().flatten() {
                            legacy += ps.remove_where(&owns);
                        }
                    }
                    let s = w.states.as_ref().unwrap();
                    // the echo records of its hint go when no fork of it (same hint) is left
                    let echo_n = match &m.hint {
                        Some(h) if !s.hint_in_use(h) => echo.drop_hint_hash(h),
                        _ => 0,
                    };
                    return Ok(json!({"freed_bytes": freed, "used_bytes": s.used_bytes(), "ram_states": ram, "legacy_entries": legacy, "echo_records": echo_n}));
                }
                let freed = match (scope.as_str(), other_model) {
                    ("model", Some(key)) => {
                        // another model's directory under the root (only one in this store's format)
                        if key.len() != 16 || !key.bytes().all(|c| c.is_ascii_hexdigit()) {
                            return Err((400, "bad model key".into()));
                        }
                        let d = s.root.join(key);
                        let n = crate::state_store::dir_bytes(&d);
                        if d.join("identity.json").is_file() && std::fs::remove_dir_all(&d).is_ok() { n } else { 0 }
                    }
                    _ => s.clear(&scope, None).map_err(|e| (400, e))?,
                };
                let used = s.used_bytes();
                let mut legacy = 0;
                if other_model.is_none() {
                    // the RAM side of the same scope: live state, parked sessions, echo records, and the
                    // legacy disk stores' entries of this engine (Codex P2/P3 review 3)
                    w.cache.reset(&mut w.engine);
                    let _ = w.cache.drain_parked();
                    w.live_blocks.clear();
                    w.preloaded = None;
                    echo.clear();
                    let all = |_: &[u32]| true;
                    for ps in [w.store.as_mut(), w.sessions.as_mut()].into_iter().flatten() {
                        legacy += ps.remove_where(&all);
                    }
                    if let Some((p, se)) = w.legacy_off.as_mut() {
                        for ps in [p.as_mut(), se.as_mut()].into_iter().flatten() {
                            legacy += ps.remove_where(&all);
                        }
                    }
                }
                Ok(json!({"freed_bytes": freed, "used_bytes": used, "legacy_entries": legacy}))
            })
            .await
            .unwrap_or_else(|_| Err((500, "state cache task failed".into())));
            match r {
                Ok(j) => Reply::Json(200, j),
                Err((c, m)) => Reply::Json(c, api::error_json(c, &m)),
            }
        }
        ("POST", "/apply-template") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            match prepare_chat(&st, &b) {
                // CACHE-PERFECT contract checks (tools/cache_contract.py): `"return_tokens": true` adds
                // the prompt's token ids exactly as a chat request tokenizes it, `"echo": true` the ids
                // after the echo splice and its report; with `--engine stub` no weights are loaded
                Ok(pc) => {
                    let mut out = json!({"prompt": pc.applied.prompt});
                    if b.get("return_tokens").and_then(Json::as_bool).unwrap_or(false) {
                        let r = st.render(&pc);
                        out["generation_prompt"] = Json::String(pc.applied.generation_prompt.clone());
                        out["msg_starts"] = json!(r.msg_starts);
                        out["assistant_spans"] = json!(r.asst);
                        if b.get("echo").and_then(Json::as_bool).unwrap_or(false) {
                            let (sp, rep) = st.splice(&r, &pc.inputs);
                            out["echo_tokens"] = json!(sp);
                            out["echo"] = rep.to_json();
                        }
                        out["tokens"] = json!(r.tokens);
                    }
                    Reply::Json(200, out)
                }
                Err(r) => r,
            }
        }
        ("POST", "/completion") | ("POST", "/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            // as chat: an ephemeral (LAN share) request keeps no session and writes no state
            completion(st, b, if ephemeral { None } else { session }, ephemeral).await
        }
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") => {
            let b = match parse() {
                Ok(b) => b,
                Err(r) => return r,
            };
            if ephemeral {
                chat_completion(st, b, None, None, true).await
            } else {
                chat_completion(st, b, session, turn_kind, false).await
            }
        }
        _ => Reply::Json(404, json!({"error": {"message": "File Not Found", "type": "not_found_error", "code": 404}})),
    }
}

/// A rendered chat prompt: tokens and the message layout from the template's delimiter offsets.
struct Rendered {
    tokens: Vec<Token>,
    /// token range [`<|im_start|>`, `<|im_end|>`) of each history assistant turn; None = unknown layout
    asst: Option<Vec<(usize, usize)>>,
    /// every message start (template `<|im_start|>`)
    msg_starts: Vec<usize>,
    first_user: Option<usize>,
    last_user: Option<usize>,
    /// every user-message start
    user_starts: Vec<usize>,
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

/// Echo fields go into `timings` only for requests with assistant turns in their history (the
/// llama-server shape of a first turn stays as it was).
fn add_echo_timings(resp: &mut Json, echo: &Option<Json>) {
    if let (Some(Json::Object(t)), Some(Json::Object(e))) = (resp.get_mut("timings"), echo) {
        for (k, v) in e {
            t.insert(k.clone(), v.clone());
        }
    }
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
        // SIGTERM cancels the running generation (the bounded shutdown write needs the worker)
        *st.running.lock().unwrap() = gp.cancel.clone();
        let r = w.generate(&st.tok, &prompt, &gp, &mut on_partial);
        *st.running.lock().unwrap() = None;
        if w.last_resume.0 != "cold" && r.as_ref().is_ok_and(|r| r.timings.preload_ms.is_some()) {
            *st.preload_status.lock().unwrap() = json!({"status": "used"});
        }
        r
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
    if st.user_delim.is_none() {
        return;
    }
    let mut body = b.clone();
    let Some(msgs) = body.get_mut("messages").and_then(Json::as_array_mut) else { return };
    msgs.push(reply);
    msgs.push(json!({"role": "user", "content": ""}));
    let Ok(pc) = prepare_chat(st, &body) else { return };
    let r = st.render(&pc);
    let (toks, rep) = st.splice(&r, &pc.inputs);
    let Some(cut) = r.last_user.map(|c| rep.map_pos(c)) else { return };
    let hist = toks[..cut.min(toks.len())].to_vec();
    let Ok(mut gp) = api::gen_params(&body, true) else { return };
    gp.first_user = r.first_user.map(|c| rep.map_pos(c)).filter(|&c| c < cut);
    gp.msg_starts = r.msg_starts.iter().map(|&c| rep.map_pos(c)).filter(|&c| c < cut).collect();
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

/// CACHE-PERFECT P2: write the live state to the state store after the reply (off the request
/// path, in the client's idle gap). A newer request preempts it between blocks; what was written
/// stays valid, the next reply writes the rest.
fn spawn_persist<E: Engine>(st: &Arc<State<E>>, session: Option<String>, req_no: u64) {
    if !st.state_cache {
        return;
    }
    let flag = Arc::new(AtomicBool::new(false));
    *st.persist.lock().unwrap() = Some(flag.clone());
    let st2 = st.clone();
    tokio::task::spawn_blocking(move || {
        let _gate = st2.persist_gate.lock().unwrap_or_else(|p| p.into_inner());
        let t0 = std::time::Instant::now();
        // v0.1.3 (item 6): phase 1 copies the head snapshot and the new blocks under the lock ...
        let staged = {
            let mut w = st2.worker.lock().unwrap_or_else(|p| p.into_inner());
            if flag.load(Ordering::Relaxed) || st2.reqs.load(Ordering::SeqCst) != req_no {
                None
            } else {
                let s = w.stage_persist(session.as_deref());
                if s.is_none() {
                    // nothing to stage (or more than the stage holds): the on-lock writer, as before
                    let n = w.persist_states(session.as_deref(), Some(&flag));
                    if n > 0 {
                        eprintln!("oxe-server: state cache: {n} files written in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                s
            }
        };
        if let Some(s) = staged {
            // ... phase 2 writes them without it (a request that arrives now runs at once; the
            // write completes behind it), phase 3 commits the manifest under the lock again
            let written = s.write();
            let mut w = st2.worker.lock().unwrap_or_else(|p| p.into_inner());
            let r = w.commit_staged(s, written, session.as_deref(), Some(&flag));
            if r.head.is_some() {
                eprintln!("oxe-server: state cache: {} files written in {:.0} ms (head off the worker lock)", r.files(), t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        st2.persist_done.fetch_add(1, Ordering::SeqCst);
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

/// LONG-APPEND default SSE keepalive interval (seconds).
pub const SSE_KEEPALIVE_SECS: f64 = 15.0;

/// `OXE_SSE_KEEPALIVE_SECS` (0 = off; unset or unparsable: [`SSE_KEEPALIVE_SECS`]).
pub fn sse_keepalive_from_env() -> std::time::Duration {
    let s = std::env::var("OXE_SSE_KEEPALIVE_SECS").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|v| v.is_finite() && *v >= 0.0).unwrap_or(SSE_KEEPALIVE_SECS);
    std::time::Duration::from_secs_f64(s)
}

/// LONG-APPEND (v0.1.3 #7): the SSE comment the keepalive sends ("queued" before the worker takes
/// the request, then the prefill position).
pub fn keepalive_comment(pr: &Progress, elapsed: std::time::Duration) -> String {
    let (done, total) = (pr.done.load(Ordering::Relaxed), pr.total.load(Ordering::Relaxed));
    if total == 0 {
        format!(": queued {:.0}s\n\n", elapsed.as_secs_f64())
    } else {
        format!(": prefill {done}/{total} {:.0}s\n\n", elapsed.as_secs_f64())
    }
}

/// LONG-APPEND (v0.1.3 #7): a streamed request sends nothing while it waits for the worker or
/// prefills, which at depth can take many minutes (28.5K tokens at ~83K: > 600 s). Every `every`
/// this sends one SSE comment line (`: prefill <done>/<total> <s>s`): SSE parsers ignore comments
/// (WHATWG EventSource, eventsource-parser, Effect's Sse parser in OpenCode, the OpenAI SDKs), so
/// clients see no change, but byte-level idle-read timeouts (sockets, proxies, the lab controller)
/// are reset and the controller can show progress. Event-level timeouts are NOT reset (OpenCode
/// 2.2.7's `chunkTimeout`, default 900 s, counts only `data:` events): those clients need a
/// generous setting (the lab's OpenCode overlay sets 3600 s). Stops at the first decoded token (`decoding`), when the response
/// ends (a weak sender: it never keeps the stream open) or when the client is gone; never blocks
/// the generation (a full channel just skips a tick).
fn spawn_keepalive(tx: &mpsc::Sender<String>, pr: Arc<Progress>, every: std::time::Duration) {
    if every.is_zero() {
        return;
    }
    let weak = tx.downgrade();
    let t0 = std::time::Instant::now();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
        loop {
            tick.tick().await;
            if pr.decoding.load(Ordering::Relaxed) {
                return;
            }
            let Some(tx) = weak.upgrade() else { return };
            if let Err(mpsc::error::TrySendError::Closed(_)) = tx.try_send(keepalive_comment(&pr, t0.elapsed())) {
                return;
            }
        }
    });
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

async fn completion<E: Engine>(st: Arc<State<E>>, b: Json, session: Option<String>, ephemeral: bool) -> Reply {
    let req_no = st.preempt_idle();
    if let Some(m) = unsupported_constraint(&b) {
        return bad(m);
    }
    let mut gp = match api::gen_params(&b, false) {
        Ok(g) => g,
        Err(e) => return bad(e),
    };
    gp.ephemeral = ephemeral;
    gp.session = session.clone();
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
            Ok(r) => {
                // CACHE-PERFECT P2 after the reply, as chat (the write runs off the request path)
                if !ephemeral {
                    spawn_persist(&st, session.clone(), req_no);
                }
                Reply::Json(200, api::completion_final(&model, &prompt_text, &r, false, &gp, return_tokens, post))
            }
            Err(e) => gen_error_reply(e),
        };
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let pr = Arc::new(Progress::default());
    gp.progress = Some(pr.clone());
    spawn_keepalive(&tx, pr.clone(), st.sse_keepalive);
    let n_prompt = prompt.len();
    let tx2 = tx.clone();
    let st2 = st.clone();
    let st3 = st.clone();
    let session3 = session.clone();
    let stall = st.stream_stall;
    tokio::spawn(async move {
        let res = run_blocking(st2, prompt, gp.clone(), move |p: Partial| {
            pr.decoding.store(true, Ordering::Relaxed);
            let j = api::completion_partial(&p.text, p.n_decoded, n_prompt, p.probs.as_ref(), post);
            send_frame(&tx2, api::sse(&j), stall)
        })
        .await;
        let frame = match res {
            Ok(r) => {
                if !ephemeral {
                    spawn_persist(&st3, session3, req_no);
                }
                api::sse(&api::completion_final(&model, &prompt_text, &r, true, &gp, return_tokens, post))
            }
            Err(GenError::Cancelled) => return,
            Err(e) => api::sse(&gen_error_json(e)),
        };
        let _ = send_final(&tx, frame, stall).await;
    });
    Reply::Stream(rx)
}

async fn chat_completion<E: Engine>(st: Arc<State<E>>, b: Json, session: Option<String>, turn_kind: Option<String>, ephemeral: bool) -> Reply {
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
    let rendered = st.render(&pc);
    // ECHO: earlier assistant turns as their exact generated ids (cache keeps every token); an
    // ephemeral request never looks at this machine's echo records (its prompt is the plain render)
    let (prompt, echo) = if ephemeral { (rendered.tokens.clone(), Default::default()) } else { st.splice(&rendered, &pc.inputs) };
    gp.ephemeral = ephemeral;
    let echo_json = (echo.blocks > 0).then(|| echo.to_json());
    gp.echo = echo_json.clone();
    if let Err(e) = st.check_tokens(&prompt) {
        return bad(e);
    }
    if prompt.len() >= st.n_ctx {
        return Reply::Json(400, api::exceed_context_json(prompt.len(), st.n_ctx));
    }
    gp.first_user = rendered.first_user.map(|c| echo.map_pos(c));
    gp.msg_starts = rendered.msg_starts.iter().map(|&c| echo.map_pos(c)).collect();
    gp.user_starts = rendered.user_starts.iter().map(|&c| echo.map_pos(c)).collect();
    gp.pre_think = st.pre_think(&prompt, &pc.applied.generation_prompt);
    gp.turn_kind = turn_kind.filter(|k| matches!(k.as_str(), "chat" | "compaction" | "post-compaction"));
    drop(rendered);
    gp.query_start = b.get("messages").and_then(Json::as_array).and_then(|m| m.last()).and_then(|m| m.get("role")).and_then(Json::as_str) == Some("user");
    gp.session = session.clone();
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
    let prompt_kept = prompt.clone();
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
        if !ephemeral {
            st.echo_record(&prompt_kept, sess.msg(), &r, session.as_deref());
            spawn_persist(&st, session.clone(), req_no);
        }
        if !ephemeral && api::finish_reason(r.stop, !sess.msg().tool_calls.is_empty()) == "stop" {
            spawn_idle_reprefill(&st, &b, sess.msg().to_json(), req_no);
        }
        let mut resp = api::chat_response(&meta, sess.msg(), &r, want_probs, false);
        add_echo_timings(&mut resp, &echo_json);
        return Reply::Json(200, resp);
    }
    let (tx, rx) = mpsc::channel::<String>(256);
    cancel_on_close(&mut gp, &tx);
    let pr = Arc::new(Progress::default());
    gp.progress = Some(pr.clone());
    spawn_keepalive(&tx, pr.clone(), st.sse_keepalive);
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
            pr.decoding.store(true, Ordering::Relaxed);
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
        if !ephemeral {
            let msg = sess.lock().unwrap().msg().clone();
            st.echo_record(&prompt_kept, &msg, &r, session.as_deref());
            spawn_persist(&st, session.clone(), req_no);
        }
        if !ephemeral && api::finish_reason(r.stop, has_calls) == "stop" {
            let reply = sess.lock().unwrap().msg().to_json();
            spawn_idle_reprefill(&st, &b, reply, req_no);
        }
        if include_usage {
            frames.push(api::chat_usage_chunk(&meta, &r));
        }
        if let Some(Json::Object(last)) = frames.last_mut() {
            last.insert("timings".into(), api::timings_json(&r.timings));
        }
        if let Some(last) = frames.last_mut() {
            add_echo_timings(last, &echo_json);
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
