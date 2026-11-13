//! v0.1.3 LONG-APPEND (steering 11-12 evening #7), model-free: a fake slow engine (the stub with a
//! per-token prefill sleep). A streamed request that prefills for a long time sends SSE comments
//! (`: queued <s>s`, `: prefill <done>/<total> <s>s`) every `sse_keepalive` until its first token,
//! the HTTP head goes out at once (before the prefill ends), the data frames are unchanged, the
//! keepalive stops at the first token and never keeps a finished stream open; 0 turns it off.

use crate::engine::StubEngine;
use crate::generate::{Progress, Worker};
use crate::service::{handle, keepalive_comment, Config, Reply, State};
use crate::tests_hotfix::{start, synthetic_tokenizer};
use serde_json::{json, Value as Json};
use std::io::{Read, Write};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A stub server whose prefill takes `per_token` per token, with keepalive `every`.
fn slow_state(per_token: Duration, every: Duration) -> Arc<State<StubEngine>> {
    let tok = synthetic_tokenizer();
    let mut eng = StubEngine::new(4096, tok.n_vocab(), tok.eos.unwrap(), tok.encode("done", false, false));
    eng.prefill_sleep_per_token = per_token;
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
    st.sse_keepalive = every;
    Arc::new(st)
}

fn long_chat(stream: bool) -> Json {
    json!({"messages": [{"role": "user", "content": "x".repeat(300)}], "stream": stream, "max_tokens": 8})
}

/// Every frame of a streamed reply (in order).
async fn frames(st: &Arc<State<StubEngine>>, path: &str, body: &Json) -> Vec<String> {
    let b = serde_json::to_vec(body).unwrap();
    let Reply::Stream(mut rx) = handle(st.clone(), "POST", path, "", &b).await else { panic!("not a stream") };
    let mut out = Vec::new();
    while let Some(f) = tokio::time::timeout(Duration::from_secs(30), rx.recv()).await.expect("stream hung") {
        out.push(f);
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn a_long_streamed_prefill_sends_progress_comments_then_the_unchanged_stream() {
    // ~300 prompt tokens x 4 ms = ~1.2 s of prefill; a comment every 100 ms
    let st = slow_state(Duration::from_millis(4), Duration::from_millis(100));
    for path in ["/v1/chat/completions", "/completion"] {
        let body = if path == "/completion" { json!({"prompt": "x".repeat(300), "stream": true, "n_predict": 8}) } else { long_chat(true) };
        let f = frames(&st, path, &body).await;
        let first_data = f.iter().position(|x| x.starts_with("data: ")).expect("no data frame");
        let comments: Vec<&String> = f.iter().filter(|x| x.starts_with(':')).collect();
        assert!(comments.len() >= 4, "{path}: {} keepalive comments over a ~1.2 s prefill: {f:?}", comments.len());
        assert!(f[..first_data].iter().all(|x| x.starts_with(": ") && x.ends_with("\n\n")), "{path}: only SSE comments before the first data frame");
        // (one tick may race the first token: it read `decoding` just before it was set)
        assert!(f[first_data..].iter().filter(|x| x.starts_with(':')).count() <= 1, "{path}: the keepalive must stop at the first token: {f:?}");
        // progress: monotone, ends near the prompt length
        let prog: Vec<(usize, usize)> = comments
            .iter()
            .filter_map(|c| c.strip_prefix(": prefill ")?.split_whitespace().next()?.split_once('/').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())))
            .collect();
        assert!(prog.len() >= 3, "{path}: {comments:?}");
        assert!(prog.windows(2).all(|w| w[0].0 <= w[1].0 && w[0].1 == w[1].1), "{path}: {prog:?}");
        // `done` moves per prefill chunk; the chunk size follows the process-wide PF group
        // thresholds other tests may change (one 299-token chunk then shows 0 until the end)
        assert!(prog.iter().all(|&(d, t)| d <= t && t >= 300), "{path}: {prog:?}");
        if path == "/v1/chat/completions" {
            assert_eq!(f.last().unwrap(), "data: [DONE]\n\n");
        }
    }
    // keepalive 0 = off: the v0.1.2 stream, no comment
    let st0 = slow_state(Duration::from_millis(2), Duration::ZERO);
    let f = frames(&st0, "/v1/chat/completions", &long_chat(true)).await;
    assert!(f.iter().all(|x| x.starts_with("data: ")), "{f:?}");
    // non-streamed: unchanged JSON
    let b = serde_json::to_vec(&long_chat(false)).unwrap();
    match handle(st.clone(), "POST", "/v1/chat/completions", "", &b).await {
        Reply::Json(code, v) => assert!(code == 200 && v["choices"][0]["message"].is_object(), "{code} {v}"),
        Reply::Stream(_) => panic!("non-streamed request got a stream"),
    }
}

/// The HTTP head (200, text/event-stream) is on the wire long before the prefill ends, and the
/// comments reach the socket while it runs: a client's header / idle-read timeout sees bytes.
#[test]
fn over_http_the_head_and_comments_arrive_during_the_prefill() {
    let st = slow_state(Duration::from_millis(8), Duration::from_millis(100)); // ~2.4 s prefill
    let (port, _rt) = start(st, None);
    let b = long_chat(true).to_string();
    let req = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len());
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    let t0 = Instant::now();
    s.write_all(req.as_bytes()).unwrap();
    let mut got = Vec::new();
    let mut buf = [0u8; 4096];
    let (mut t_head, mut t_comment) = (None, None);
    loop {
        let n = s.read(&mut buf).unwrap_or(0);
        if n == 0 {
            break;
        }
        got.extend_from_slice(&buf[..n]);
        let text = String::from_utf8_lossy(&got);
        if t_head.is_none() && text.contains("\r\n\r\n") {
            t_head = Some(t0.elapsed());
        }
        if t_comment.is_none() && text.contains(": prefill ") {
            t_comment = Some(t0.elapsed());
        }
    }
    let total = t0.elapsed();
    let text = String::from_utf8_lossy(&got).to_string();
    assert!(text.starts_with("HTTP/1.1 200") && text.to_ascii_lowercase().contains("content-type: text/event-stream"), "{text}");
    let (h, c) = (t_head.expect("no head"), t_comment.expect("no prefill comment"));
    assert!(total > Duration::from_millis(1500), "the fake engine was not slow: {total:?}");
    assert!(h < total / 3 && c < total / 2, "head {h:?} / first comment {c:?} vs whole reply {total:?}");
    assert!(text.contains("data: [DONE]"), "{text}");
}

#[test]
fn keepalive_comment_format() {
    let p = Progress::default();
    assert_eq!(keepalive_comment(&p, Duration::from_secs(3)), ": queued 3s\n\n");
    p.total.store(28_500, Ordering::Relaxed);
    p.done.store(12_288, Ordering::Relaxed);
    assert_eq!(keepalive_comment(&p, Duration::from_secs_f64(361.4)), ": prefill 12288/28500 361s\n\n");
    assert_eq!(crate::service::SSE_KEEPALIVE_SECS, 15.0);
}
