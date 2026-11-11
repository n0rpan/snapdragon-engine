//! v0.1.1 hotfix tests (REVIEW-v0.1.1 B1, B2, S8 and the request-path panic sweep). Model-free:
//! a synthetic byte-level tokenizer with a ChatML template and the stub engine, no GGUF needed.

use crate::engine::StubEngine;
use crate::generate::{GenParams, Worker};
use crate::http::{self, Access};
use crate::service::{handle, Config, Reply, State};
use serde_json::{json, Value as Json};
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokenizer::gguf_meta::Meta;

const TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// 256 byte tokens (GPT-2 byte mapping), `<|im_start|>` = 256, `<|im_end|>` = 257 (EOS).
fn synthetic_tokenizer() -> tokenizer::Tokenizer {
    let table = tokenizer::unicode::byte_to_cpt_table();
    let mut toks: Vec<Meta> = table.iter().map(|c| Meta::Str(c.to_string())).collect();
    let mut types: Vec<Meta> = vec![Meta::Int(1); 256];
    for s in ["<|im_start|>", "<|im_end|>"] {
        toks.push(Meta::Str(s.into()));
        types.push(Meta::Int(3));
    }
    let mut md = std::collections::HashMap::new();
    md.insert("tokenizer.ggml.model".to_string(), Meta::Str("gpt2".into()));
    md.insert("tokenizer.ggml.pre".to_string(), Meta::Str("qwen2".into()));
    md.insert("tokenizer.ggml.tokens".to_string(), Meta::Array(toks));
    md.insert("tokenizer.ggml.token_type".to_string(), Meta::Array(types));
    md.insert("tokenizer.ggml.merges".to_string(), Meta::Array(vec![]));
    md.insert("tokenizer.ggml.eos_token_id".to_string(), Meta::Int(257));
    md.insert("tokenizer.chat_template".to_string(), Meta::Str(TEMPLATE.into()));
    tokenizer::Tokenizer::from_metadata(&md).unwrap()
}

/// A server state whose stub engine answers `reply` (then EOS) to every request.
fn synthetic_state(reply: &str, n_ctx: usize, stall: Option<Duration>) -> Arc<State<StubEngine>> {
    let tok = synthetic_tokenizer();
    let eng = StubEngine::new(n_ctx, tok.n_vocab(), tok.eos.unwrap(), tok.encode(reply, false, false));
    let cfg = Config {
        model_name: "synthetic".into(),
        model_path: "/m.gguf".into(),
        chat_opts: Default::default(),
        fingerprint: "oxe-test".into(),
        slots: true,
        idle_reprefill: false,
        engine_status: None,
    };
    let mut st = State::new(cfg, tok, Worker::new(eng, 4)).unwrap();
    if let Some(s) = stall {
        st.stream_stall = s;
    }
    Arc::new(st)
}

fn chat_body(extra: Json) -> Json {
    let mut b = json!({"messages": [{"role": "user", "content": "hi"}]});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

async fn status_of(st: &Arc<State<StubEngine>>, path: &str, body: Json) -> u16 {
    let b = serde_json::to_vec(&body).unwrap();
    match tokio::time::timeout(Duration::from_secs(30), handle(st.clone(), "POST", path, "", &b)).await.expect("request hung") {
        Reply::Json(c, _) => c,
        Reply::Stream(mut rx) => {
            while tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("stream hung").is_some() {}
            200
        }
    }
}

// ------------------------------------------------------------------------------------- B2

/// `min_p` 2 with temperature 1 emptied the candidate list and `unwrap` panicked (abort in
/// release, `worker panicked` 500 in tests). It is a 400 now, for chat and /completion, stream
/// or not.
#[tokio::test(flavor = "multi_thread")]
async fn min_p_above_one_is_a_400_not_an_abort() {
    let st = synthetic_state("ok", 4096, None);
    for stream in [false, true] {
        let b = chat_body(json!({"temperature": 1, "min_p": 2, "stream": stream}));
        assert_eq!(status_of(&st, "/v1/chat/completions", b).await, 400, "chat stream={stream}");
        let b = json!({"prompt": "hi", "temperature": 1, "min_p": 2, "stream": stream});
        assert_eq!(status_of(&st, "/completion", b).await, 400, "completion stream={stream}");
    }
    // the server still answers, and in-range sampling works
    let b = chat_body(json!({"temperature": 1, "min_p": 0.1, "top_p": 0.9, "top_k": 5}));
    assert_eq!(status_of(&st, "/v1/chat/completions", b).await, 200);
}

#[test]
fn sampling_ranges_are_checked() {
    let bad = [
        json!({"min_p": 1.5}),
        json!({"min_p": -0.1}),
        json!({"top_p": 1.01}),
        json!({"top_p": -1}),
        json!({"temperature": 1e39}),
        json!({"top_k": 1u64 << 40}),
        json!({"top_k": 1.5}),
        json!({"min_p": "0.1"}),
        json!({"presence_penalty": 3}),
        json!({"frequency_penalty": -2.5}),
        json!({"repeat_penalty": -1}),
        json!({"typical_p": 2}),
        json!({"seed": "x"}),
        json!({"logprobs": true, "top_logprobs": 1_000_000_000u64}),
        json!({"n_probs": 1_000_000}),
    ];
    for b in bad {
        assert!(crate::api::gen_params(&b, true).is_err(), "{b} must be refused");
    }
    let good = [
        json!({}),
        json!({"min_p": 0, "top_p": 1, "temperature": 0}),
        json!({"min_p": 1, "top_p": 0}),
        json!({"temperature": -1}), // llama-server: <= 0 is greedy
        json!({"temperature": 2.5, "top_k": -1}),
        json!({"top_k": null, "min_p": null}),
        json!({"presence_penalty": 1.5, "frequency_penalty": -2, "repeat_penalty": 1.1}),
        json!({"seed": -1}),
        json!({"logprobs": true, "top_logprobs": 20}),
    ];
    for b in good {
        assert!(crate::api::gen_params(&b, true).is_ok(), "{b} must be accepted");
    }
}

/// REVIEW S4: routes are exact (llama-server); a trailing slash or a doubled slash is a 404.
#[tokio::test(flavor = "multi_thread")]
async fn non_canonical_paths_are_404() {
    let st = synthetic_state("ok", 4096, None);
    for p in ["/v1/chat/completions/", "//v1/chat/completions", "/completion/"] {
        assert_eq!(status_of(&st, p, chat_body(json!({"n_predict": 1}))).await, 404, "{p}");
    }
    assert_eq!(status_of(&st, "/v1/chat/completions", chat_body(json!({"n_predict": 1}))).await, 200);
}

/// Token ids above u32 used to wrap (`as`) onto valid ids and pass the vocabulary check.
#[tokio::test(flavor = "multi_thread")]
async fn token_ids_above_u32_are_refused() {
    let st = synthetic_state("ok", 4096, None);
    let wrapped = (1u64 << 32) + 97; // == 'a' after `as u32`
    assert_eq!(status_of(&st, "/completion", json!({"prompt": [wrapped]})).await, 400);
    assert_eq!(status_of(&st, "/detokenize", json!({"tokens": [wrapped]})).await, 400);
    assert_eq!(status_of(&st, "/completion", json!({"prompt": [97], "n_predict": 2})).await, 200);
}

/// Idle re-prefill on a history whose length is an exact multiple of the ubatch sliced
/// `hist[..len + 1]` (panic = abort in release).
#[test]
fn idle_reprefill_on_an_aligned_history_does_not_slice_past_the_end() {
    let mut w = Worker::new(StubEngine::new(1000, 300, 299, vec![]), 4);
    w.ubatch = 4;
    let hist: Vec<u32> = (0..16).collect();
    let n = w.prefill_idle(&hist, &GenParams::default()).unwrap();
    assert!(n > 0 && n < 16, "prefilled {n}");
}

// ------------------------------------------------------------------------------------- S8

/// A streaming client that keeps its socket open and stops reading filled the 256-frame channel;
/// `blocking_send` then held the worker mutex for ever and every later request waited. Now the
/// stream is dropped after the stall timeout and the next request completes.
#[tokio::test(flavor = "multi_thread")]
async fn stalled_stream_client_does_not_block_the_worker() {
    let st = synthetic_state(&"a".repeat(2000), 8192, Some(Duration::from_millis(500)));
    let b = serde_json::to_vec(&chat_body(json!({"stream": true, "temperature": 0}))).unwrap();
    let rx = match handle(st.clone(), "POST", "/v1/chat/completions", "", &b).await {
        Reply::Stream(rx) => rx, // held open, never read
        Reply::Json(c, j) => panic!("expected a stream, got {c} {j}"),
    };
    let t0 = Instant::now();
    let b = chat_body(json!({"temperature": 0, "n_predict": 4}));
    let second = tokio::time::timeout(Duration::from_secs(20), status_of(&st, "/v1/chat/completions", b)).await;
    assert_eq!(second.ok(), Some(200), "second request blocked behind a stalled stream");
    assert!(t0.elapsed() < Duration::from_secs(10));
    drop(rx);
}

#[test]
fn send_frame_gives_up_on_a_full_channel() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(1);
    let stall = Duration::from_millis(100);
    assert!(crate::service::send_frame(&tx, "a".into(), stall));
    let t0 = Instant::now();
    assert!(!crate::service::send_frame(&tx, "b".into(), stall), "full channel must time out");
    assert!(t0.elapsed() >= stall && t0.elapsed() < Duration::from_secs(5));
    assert_eq!(rx.try_recv().ok().as_deref(), Some("a"));
    assert!(crate::service::send_frame(&tx, "c".into(), stall), "room again");
    drop(rx);
    assert!(!crate::service::send_frame(&tx, "d".into(), stall), "closed channel");
}

// ------------------------------------------------------------------------------------- B1

/// One raw HTTP/1.1 request; returns (status, lower-case header block, body).
fn raw(port: u16, req: &str) -> (u16, String, String) {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s.write_all(req.as_bytes()).unwrap();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    (status, head.to_ascii_lowercase(), body.to_string())
}

fn get(port: u16, path: &str, headers: &[&str]) -> (u16, String, String) {
    let mut r = format!("GET {path} HTTP/1.1\r\nConnection: close\r\n");
    for h in headers {
        r += h;
        r += "\r\n";
    }
    raw(port, &(r + "\r\n"))
}

fn post(port: u16, path: &str, headers: &[&str], body: &Json) -> (u16, String, String) {
    let b = body.to_string();
    let mut r = format!("POST {path} HTTP/1.1\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n", b.len());
    for h in headers {
        r += h;
        r += "\r\n";
    }
    raw(port, &(r + "\r\n" + &b))
}

/// Start the HTTP front end on an ephemeral loopback port (default access rules, or `access`).
fn start(st: Arc<State<StubEngine>>, access: Option<Access>) -> (u16, tokio::runtime::Runtime) {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let l = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let port = l.local_addr().unwrap().port();
    match access {
        None => rt.spawn(http::serve(st, l)),
        Some(mut a) => {
            a.port = port;
            rt.spawn(http::serve_with(st, l, a))
        }
    };
    (port, rt)
}

/// Wildcard CORS, foreign Host (DNS rebinding) and cross-site Origin used to get 200 with
/// `Access-Control-Allow-Origin: *`; now 403 and never a CORS header.
#[test]
fn foreign_host_and_cross_site_origin_are_refused_without_cors() {
    let (p, _rt) = start(synthetic_state("ok", 4096, None), None);
    let host = format!("Host: 127.0.0.1:{p}");
    let (c, h, _) = get(p, "/props", &[&host]);
    assert_eq!(c, 200);
    assert!(!h.contains("access-control-"), "no CORS headers: {h}");
    for hh in [format!("Host: localhost:{p}"), format!("Host: [::1]:{p}"), format!("Host: LOCALHOST:{p}")] {
        assert_eq!(get(p, "/health", &[&hh]).0, 200, "{hh}");
    }
    // a local page of the controller (another loopback port) is same-site
    assert_eq!(get(p, "/props", &[&host, "Origin: http://127.0.0.1:8092"]).0, 200);
    for bad in [
        vec!["Host: x.example".to_string()],
        vec![format!("Host: x.example:{p}")],
        vec![format!("Host: 127.0.0.1:{}", p.wrapping_add(1))],
        vec![format!("Host: localhost.x.example:{p}")],
        vec![host.clone(), "Origin: https://x.example".into()],
        vec![host.clone(), "Origin: null".into()],
        vec![host.clone(), "Origin: http://127.0.0.1.x.example".into()],
        vec![host.clone(), "Origin: http://localhost@cafe.example".into()],
        vec![host.clone(), "Sec-Fetch-Site: cross-site".into()],
    ] {
        let hs: Vec<&str> = bad.iter().map(String::as_str).collect();
        let (c, h, _) = get(p, "/props", &hs);
        assert_eq!(c, 403, "{bad:?}");
        assert!(!h.contains("access-control-"), "{bad:?}: {h}");
    }
    // a cross-site simple POST (no preflight) never reaches the engine
    let (c, h, _) = post(p, "/v1/chat/completions", &[&host, "Origin: https://x.example"], &chat_body(json!({"n_predict": 1})));
    assert_eq!(c, 403);
    assert!(!h.contains("access-control-"));
    // a preflight gets no CORS grant
    let (c, h, _) = raw(p, &format!("OPTIONS /v1/chat/completions HTTP/1.1\r\n{host}\r\nOrigin: https://x.example\r\nAccess-Control-Request-Method: POST\r\nConnection: close\r\n\r\n"));
    assert_eq!(c, 403);
    assert!(!h.contains("access-control-"));
    let (c, h, _) = raw(p, &format!("OPTIONS /props HTTP/1.1\r\n{host}\r\nConnection: close\r\n\r\n"));
    assert_eq!(c, 204);
    assert!(!h.contains("access-control-"));
}

#[test]
fn api_key_guards_every_route() {
    let keys = http::parse_key_file("\n  test-key-not-secret-0123456789  \nsecond-key\n").unwrap();
    let (p, _rt) = start(synthetic_state("ok", 4096, None), Some(Access { keys, ..Default::default() }));
    let host = format!("Host: 127.0.0.1:{p}");
    // the controller sends the key on its health polls too: no unauthenticated route at all
    assert_eq!(get(p, "/health", &[&host]).0, 401);
    assert_eq!(get(p, "/v1/health", &[&host]).0, 401);
    assert_eq!(get(p, "/health", &[&host, "Authorization: Bearer test-key-not-secret-0123456789"]).0, 200);
    let (c, h, body) = get(p, "/props", &[&host]);
    assert_eq!(c, 401);
    assert!(h.contains("www-authenticate: bearer"), "{h}");
    assert!(body.contains("Invalid API Key"));
    for bad in ["Authorization: Bearer wrong", "Authorization: Bearer test-key-not-secret-012345678", "Authorization: Basic czNjcmV0", "X-Api-Key: nope", "Authorization: Bearer "] {
        assert_eq!(get(p, "/props", &[&host, bad]).0, 401, "{bad}");
        assert_eq!(get(p, "/v1/models", &[&host, bad]).0, 401, "{bad}");
    }
    for good in ["Authorization: Bearer test-key-not-secret-0123456789", "authorization: bearer test-key-not-secret-0123456789", format!("X-Api-Key: {}", ["second", "key"].join("-")).as_str()] {
        assert_eq!(get(p, "/props", &[&host, good]).0, 200, "{good}");
    }
    let body = chat_body(json!({"n_predict": 2, "temperature": 0}));
    assert_eq!(post(p, "/v1/chat/completions", &[&host], &body).0, 401);
    assert_eq!(post(p, "/v1/chat/completions", &[&host, "Authorization: Bearer test-key-not-secret-0123456789"], &body).0, 200);
    // the key does not open the door for a foreign Host
    assert_eq!(get(p, "/props", &["Host: x.example", "Authorization: Bearer test-key-not-secret-0123456789"]).0, 403);
}

#[test]
fn key_file_and_compare_rules() {
    assert!(http::parse_key_file("").is_err());
    assert!(http::parse_key_file(" \n\t\n").is_err());
    assert_eq!(http::parse_key_file("k1\r\nk2").unwrap(), vec![b"k1".to_vec(), b"k2".to_vec()]);
    assert!(http::ct_eq(b"abc", b"abc"));
    assert!(!http::ct_eq(b"abc", b"abd"));
    assert!(!http::ct_eq(b"abc", b"abcd"));
    assert!(!http::ct_eq(b"", b"a"));
    assert!(http::ct_eq(b"", b""));
}

#[test]
fn access_rules_unit() {
    use hyper::header::{HeaderMap, HeaderValue};
    let a = Access { port: 8093, extra_hosts: vec!["192.0.2.5".into()], keys: vec![] };
    let check = |pairs: &[(&'static str, &str)]| {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(*k, HeaderValue::from_str(v).unwrap());
        }
        a.check("GET", "/props", &h)
    };
    assert!(check(&[]).is_ok(), "no Host (HTTP/1.0 tools) is allowed");
    assert!(check(&[("host", "127.0.0.1:8093")]).is_ok());
    assert!(check(&[("host", "[::1]:8093")]).is_ok());
    assert!(check(&[("host", "192.0.2.5:8093")]).is_ok());
    assert!(check(&[("host", "127.0.0.1")]).is_err(), "port-less only on port 80");
    assert!(check(&[("host", "127.0.0.1:8093"), ("host", "x.example:8093")]).is_err());
    assert!(check(&[("host", "[::1]x:8093")]).is_err());
    assert!(check(&[("origin", "http://localhost:3000")]).is_ok());
    assert!(check(&[("origin", "https://[::1]")]).is_ok());
    assert!(check(&[("origin", "http://localhost:3000/path")]).is_err());
    assert!(check(&[("origin", "file://")]).is_err());
    assert!(check(&[("sec-fetch-site", "same-origin")]).is_ok());
    assert!(check(&[("sec-fetch-site", "Cross-Site")]).is_err());
    let a80 = Access { port: 80, ..Default::default() };
    let mut h = HeaderMap::new();
    h.insert("host", HeaderValue::from_static("localhost"));
    assert!(a80.check("GET", "/", &h).is_ok());
}
