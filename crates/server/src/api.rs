//! Request parameters and response JSON in llama-server's shapes (key order included:
//! serde_json is built with preserve_order).

use crate::generate::{GenParams, GenResult, StopType, Timings, TokenProbs};
use crate::sampling::SamplingParams;
use serde_json::{json, Map, Value as Json};

pub fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Error body `{"error":{"code","message","type"}}` and its HTTP status.
pub fn error_json(status: u16, message: &str) -> Json {
    let ty = match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        501 => "not_supported_error",
        503 => "unavailable_error",
        _ => "server_error",
    };
    json!({"error": {"code": status, "message": message, "type": ty}})
}

pub fn exceed_context_json(n_prompt: usize, n_ctx: usize) -> Json {
    json!({"error": {
        "code": 400,
        "message": format!("request ({n_prompt} tokens) exceeds the available context size ({n_ctx} tokens), try increasing it"),
        "type": "exceed_context_size_error",
        "n_prompt_tokens": n_prompt,
        "n_ctx": n_ctx,
    }})
}

fn num<T: Copy>(body: &Json, keys: &[&str], f: impl Fn(&Json) -> Option<T>, def: T) -> T {
    for k in keys {
        if let Some(v) = body.get(*k) {
            if let Some(x) = f(v) {
                return x;
            }
        }
    }
    def
}

/// Sampling / stopping parameters common to /completion and /v1/chat/completions
/// (`params_from_json_cmpl`).
pub fn gen_params(body: &Json, chat: bool) -> Result<GenParams, String> {
    let d = SamplingParams::default();
    let mut p = GenParams::default();
    p.n_predict = num(body, &["n_predict", "max_completion_tokens", "max_tokens"], Json::as_i64, -1);
    if let Some(v) = body.get("n_predict").and_then(Json::as_i64) {
        p.n_predict = v;
    }
    p.sampling = SamplingParams {
        temperature: num(body, &["temperature"], |v| v.as_f64().map(|x| x as f32), d.temperature),
        top_k: num(body, &["top_k"], |v| v.as_i64().map(|x| x as i32), d.top_k),
        top_p: num(body, &["top_p"], |v| v.as_f64().map(|x| x as f32), d.top_p),
        min_p: num(body, &["min_p"], |v| v.as_f64().map(|x| x as f32), d.min_p),
        seed: num(body, &["seed"], |v| v.as_i64().map(|x| x as u64), d.seed),
    };
    p.stop = match body.get("stop") {
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(a)) => a.iter().filter_map(Json::as_str).filter(|s| !s.is_empty()).map(str::to_string).collect(),
        _ => Vec::new(),
    };
    p.cache_prompt = body.get("cache_prompt").and_then(Json::as_bool).unwrap_or(true);
    p.ignore_eos = body.get("ignore_eos").and_then(Json::as_bool).unwrap_or(false);
    // per-request speculation switch (the controller's off-switch): "speculative": false, or a
    // depth via "spec_draft_n_max" (llama-server's flag name) / "speculative": {"n_max": n}
    p.spec_depth = match body.get("speculative") {
        Some(Json::Bool(false)) => Some(0),
        Some(Json::Object(o)) => o.get("n_max").and_then(Json::as_u64).map(|n| n as usize),
        _ => None,
    }
    .or_else(|| body.get("spec_draft_n_max").and_then(Json::as_u64).map(|n| n as usize));
    if chat {
        if body.get("logprobs").and_then(Json::as_bool).unwrap_or(false) {
            p.n_probs = body.get("top_logprobs").and_then(Json::as_u64).unwrap_or(20) as usize;
        } else if body.get("top_logprobs").map(|v| !v.is_null()).unwrap_or(false) {
            return Err("top_logprobs requires logprobs to be set to true".into());
        }
        if let Some(n) = body.get("n_probs").and_then(Json::as_u64) {
            p.n_probs = n as usize;
        }
    } else {
        p.n_probs = num(body, &["n_probs", "logprobs"], Json::as_u64, 0) as usize;
    }
    Ok(p)
}

pub fn timings_json(t: &Timings) -> Json {
    let per_tok = |ms: f64, n: usize| if n > 0 { ms / n as f64 } else { 0.0 };
    let per_s = |ms: f64, n: usize| if ms > 0.0 { 1e3 / ms * n as f64 } else { 0.0 };
    let mut j = timings_base(t, per_tok, per_s);
    if t.draft_n > 0 {
        // llama-server's speculative counters
        j["draft_n"] = json!(t.draft_n);
        j["draft_n_accepted"] = json!(t.draft_n_accepted);
    }
    j
}

fn timings_base(t: &Timings, per_tok: impl Fn(f64, usize) -> f64, per_s: impl Fn(f64, usize) -> f64) -> Json {
    json!({
        "cache_n": t.cache_n,
        "prompt_n": t.prompt_n,
        "prompt_ms": t.prompt_ms,
        "prompt_per_token_ms": per_tok(t.prompt_ms, t.prompt_n),
        "prompt_per_second": per_s(t.prompt_ms, t.prompt_n),
        "predicted_n": t.predicted_n,
        "predicted_ms": t.predicted_ms,
        "predicted_per_token_ms": per_tok(t.predicted_ms, t.predicted_n),
        "predicted_per_second": per_s(t.predicted_ms, t.predicted_n),
    })
}

pub fn usage_json(r: &GenResult) -> Json {
    json!({
        "completion_tokens": r.n_decoded,
        "prompt_tokens": r.n_prompt,
        "total_tokens": r.n_decoded + r.n_prompt,
        "prompt_tokens_details": {"cached_tokens": r.timings.cache_n},
    })
}

fn logprob(p: f32) -> f32 {
    if p == 0.0 {
        f32::MIN
    } else {
        p.ln()
    }
}

fn piece_str(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// `completion_token_output::probs_vector_to_json` (post_sampling_probs=false shape;
/// with post_sampling the keys become prob/top_probs).
pub fn probs_json(v: &[TokenProbs], post_sampling: bool) -> Json {
    let (pk, tk) = if post_sampling { ("prob", "top_probs") } else { ("logprob", "top_logprobs") };
    let val = |p: f32| if post_sampling { Json::from(p) } else { Json::from(logprob(p)) };
    Json::Array(
        v.iter()
            .map(|t| {
                let mut o = Map::new();
                o.insert("id".into(), t.id.into());
                o.insert("token".into(), piece_str(&t.piece).into());
                o.insert("bytes".into(), t.piece.iter().map(|&b| Json::from(b)).collect());
                o.insert(pk.into(), val(t.prob));
                o.insert(
                    tk.into(),
                    t.top
                        .iter()
                        .map(|e| {
                            let mut x = Map::new();
                            x.insert("id".into(), e.id.into());
                            x.insert("token".into(), piece_str(&e.piece).into());
                            x.insert("bytes".into(), e.piece.iter().map(|&b| Json::from(b)).collect());
                            x.insert(pk.into(), val(e.prob));
                            Json::Object(x)
                        })
                        .collect(),
                );
                Json::Object(o)
            })
            .collect(),
    )
}

pub fn finish_reason(stop: StopType, has_tool_calls: bool) -> &'static str {
    match stop {
        StopType::Word | StopType::Eos => {
            if has_tool_calls {
                "tool_calls"
            } else {
                "stop"
            }
        }
        _ => "length",
    }
}

pub struct ChatMeta {
    pub id: String,
    pub model: String,
    pub created: u64,
    pub fingerprint: String,
}

pub fn chat_response(m: &ChatMeta, msg: &chat::ChatMsg, r: &GenResult, n_probs: bool, post_sampling: bool) -> Json {
    let mut choice = Map::new();
    choice.insert("finish_reason".into(), finish_reason(r.stop, !msg.tool_calls.is_empty()).into());
    choice.insert("index".into(), 0.into());
    choice.insert("message".into(), msg.to_json());
    if n_probs && !r.probs.is_empty() {
        choice.insert("logprobs".into(), json!({"content": probs_json(&r.probs, post_sampling)}));
    }
    json!({
        "choices": [Json::Object(choice)],
        "created": m.created,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "object": "chat.completion",
        "usage": usage_json(r),
        "id": m.id,
        "timings": timings_json(&r.timings),
    })
}

pub fn chat_chunk(m: &ChatMeta, delta: Json, finish: Option<&str>) -> Json {
    json!({
        "choices": [{"finish_reason": finish, "index": 0, "delta": delta}],
        "created": m.created,
        "id": m.id,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "object": "chat.completion.chunk",
    })
}

pub fn chat_usage_chunk(m: &ChatMeta, r: &GenResult) -> Json {
    json!({
        "choices": [],
        "created": m.created,
        "id": m.id,
        "model": m.model,
        "system_fingerprint": m.fingerprint,
        "object": "chat.completion.chunk",
        "usage": usage_json(r),
    })
}

/// native /completion final object (`to_json_non_oaicompat`)
pub fn completion_final(model: &str, prompt: &str, r: &GenResult, stream: bool, gp: &GenParams, return_tokens: bool, post_sampling: bool) -> Json {
    let mut o = Map::new();
    o.insert("index".into(), 0.into());
    o.insert("content".into(), if stream { "".into() } else { r.text.clone().into() });
    o.insert(
        "tokens".into(),
        if return_tokens && !stream { r.tokens.iter().map(|&t| Json::from(t)).collect() } else { Json::Array(vec![]) },
    );
    o.insert("id_slot".into(), 0.into());
    o.insert("stop".into(), true.into());
    o.insert("model".into(), model.into());
    o.insert("tokens_predicted".into(), r.n_decoded.into());
    o.insert("tokens_evaluated".into(), r.n_prompt.into());
    o.insert("generation_settings".into(), generation_settings(gp));
    o.insert("prompt".into(), prompt.into());
    o.insert("has_new_line".into(), r.text.contains('\n').into());
    o.insert("truncated".into(), r.truncated.into());
    o.insert("stop_type".into(), r.stop.as_str().into());
    o.insert("stopping_word".into(), r.stopping_word.clone().into());
    o.insert("tokens_cached".into(), (r.n_prompt + r.n_decoded.saturating_sub(1)).into());
    o.insert("timings".into(), timings_json(&r.timings));
    if !stream && !r.probs.is_empty() {
        o.insert("completion_probabilities".into(), probs_json(&r.probs, post_sampling));
    }
    Json::Object(o)
}

pub fn generation_settings(gp: &GenParams) -> Json {
    json!({
        "n_predict": gp.n_predict,
        "seed": gp.sampling.seed,
        "temperature": gp.sampling.temperature,
        "top_k": gp.sampling.top_k,
        "top_p": gp.sampling.top_p,
        "min_p": gp.sampling.min_p,
        "max_tokens": gp.n_predict,
        "n_probs": gp.n_probs,
        "stop": gp.stop,
        "ignore_eos": gp.ignore_eos,
        "cache_prompt": gp.cache_prompt,
    })
}

/// native /completion streamed partial
pub fn completion_partial(text: &str, n_decoded: usize, n_prompt: usize, probs: Option<&TokenProbs>, post_sampling: bool) -> Json {
    let mut o = Map::new();
    o.insert("index".into(), 0.into());
    o.insert("content".into(), text.into());
    o.insert("tokens".into(), Json::Array(vec![]));
    o.insert("stop".into(), false.into());
    o.insert("id_slot".into(), 0.into());
    o.insert("tokens_predicted".into(), n_decoded.into());
    o.insert("tokens_evaluated".into(), n_prompt.into());
    if let Some(p) = probs {
        o.insert("completion_probabilities".into(), probs_json(std::slice::from_ref(p), post_sampling));
    }
    Json::Object(o)
}

pub fn sse(v: &Json) -> String {
    format!("data: {}\n\n", serde_json::to_string(v).unwrap())
}
