//! LAN share (lab O-102, security review M2): `X-Oxe-State-Policy: ephemeral`, set by the controller
//! for a request from another device. Model-free (stub engine, synthetic tokenizer); every store
//! lives in its own temp directory.
//!
//! Covered: an ephemeral request reuses no prefix (live, parked, checkpoint or disk: no timing
//! oracle), reads and writes no state-store, echo or warm file, leaves nothing for a later request
//! to match, and the local live session is back exactly as it was (held in RAM; without a RAM
//! budget it resumes from the state store instead); the header is parsed strictly.

use crate::generate::GenParams;
use crate::http::state_policy;
use crate::service::{handle_policy, Reply};
use crate::state_store::BLOCK;
use crate::tests_state::{body, chat, files, sys, tmp};
use crate::tests_state_p2b::{gen, server_echo, toks, worker};
use hyper::header::{HeaderMap, HeaderValue};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, SystemTime};

/// Every file under `dir` with its size and modification time: "nothing was written".
fn listing(dir: &Path) -> Vec<(std::path::PathBuf, u64, SystemTime)> {
    let mut v: Vec<_> = files(dir, "").into_iter().map(|p| {
        let m = std::fs::metadata(&p).unwrap();
        (p, m.len(), m.modified().unwrap())
    }).collect();
    v.sort();
    v
}

fn local_params() -> GenParams {
    GenParams { n_predict: 3, first_user: Some(BLOCK - 10), session: Some("local".into()), ..Default::default() }
}

fn remote_params() -> GenParams {
    // the controller never sends a session hint with the policy; a stray one must not matter
    GenParams { ephemeral: true, ..local_params() }
}

#[test]
fn ephemeral_request_reuses_nothing_writes_nothing_and_the_local_session_comes_back() {
    let dir = tmp("lan-eph-held");
    let mut w = worker(&dir);
    w.cache.park_budget = 1 << 30;
    w.persist_every = 256; // a long ephemeral prefill would write progress if it were not ephemeral
    let local = toks(700, 1);
    gen(&mut w, &local, &local_params());
    w.persist_states(Some("local"), None); // the post-reply write
    let disk = listing(&dir);
    assert!(!disk.is_empty(), "the local conversation is on disk");
    let (kv, live, parked) = (w.engine.tokens.clone(), w.cache.tokens.clone(), w.cache.parked_lengths());

    // a remote request whose prompt shares the local prefix (and the disk blocks)
    let remote = toks(1200, 1);
    assert_eq!(remote[..700], local[..]);
    for round in 0..2 {
        let r = gen(&mut w, &remote, &remote_params());
        assert_eq!(r.timings.cache_n, 0, "round {round}: no prefix reused (no timing oracle)");
        assert_eq!(r.timings.resume_src, "cold");
        assert_eq!(r.timings.prompt_n, remote.len());
        assert_eq!(listing(&dir), disk, "round {round}: nothing written to the state store");
        // the local session is live again, bit for bit; the parked sessions were not touched
        assert_eq!(w.engine.tokens, kv);
        assert_eq!(w.cache.tokens, live);
        assert_eq!(w.cache.parked_lengths(), parked);
    }
    // the local conversation continues from RAM, as if nothing had happened
    let mut next = live.clone();
    next.extend([5, 6, 7, 8]);
    let r = gen(&mut w, &next, &local_params());
    assert_eq!(r.timings.cache_n, live.len());
    assert_eq!(r.timings.resume_src, "live");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_a_ram_budget_the_local_session_resumes_from_the_state_store() {
    let dir = tmp("lan-eph-disk");
    let mut w = worker(&dir);
    assert_eq!(w.cache.park_budget, 0);
    let local = toks(700, 2);
    gen(&mut w, &local, &local_params());
    w.persist_states(Some("local"), None);
    let live = w.cache.tokens.clone();
    let disk = listing(&dir);
    let r = gen(&mut w, &toks(900, 2), &remote_params());
    assert_eq!(r.timings.cache_n, 0);
    assert_eq!(listing(&dir), disk, "the local state was on disk already: the park write wrote nothing");
    assert!(w.engine.tokens.is_empty() && w.cache.tokens.is_empty(), "the remote state is dropped");
    let mut next = live.clone();
    next.extend([5, 6, 7, 8]);
    let r = gen(&mut w, &next, &local_params());
    assert_eq!(r.timings.resume_src, "disk");
    assert!(r.timings.cache_n >= BLOCK, "resumed from the store: {}", r.timings.cache_n);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn ephemeral_chat_skips_echo_and_post_reply_writes() {
    let tok = crate::tests_echo::synth_tokenizer();
    let dir = tmp("lan-eph-chat");
    let st = server_echo(&dir, vec![tok.encode("Answer one.", false, true)], 1 << 40);
    // held in RAM: no park write of the local session (its optional checkpoints may still be pending)
    st.worker.lock().unwrap().cache.park_budget = 1 << 30;
    let first = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"}]);
    chat(&st, first).await;
    let records = st.echo.lock().unwrap().n_records();
    assert!(records > 0);
    let disk = listing(&dir);
    let (persists, reqs) = (st.persist_done.load(Ordering::SeqCst), st.reqs.load(Ordering::SeqCst));
    let live = st.worker.lock().unwrap().cache.tokens.clone();

    // the same conversation, continued from another device
    let msgs = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"},
                      {"role":"assistant","content":"Answer one."},{"role":"user","content":"second"}]);
    let b = serde_json::to_vec(&body(msgs)).unwrap();
    let j = match handle_policy(st.clone(), "POST", "/v1/chat/completions", "", &b, Some("hint-1".into()), None, true).await {
        Reply::Json(200, j) => j,
        Reply::Json(c, j) => panic!("{c}: {j}"),
        Reply::Stream(_) => panic!("stream"),
    };
    assert_eq!(j["timings"]["cache_n"], 0, "{j}");
    assert!(j["timings"].get("echo_blocks").is_none(), "no echo lookup: {j}");
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(st.reqs.load(Ordering::SeqCst), reqs + 1);
    assert_eq!(st.persist_done.load(Ordering::SeqCst), persists, "no post-reply state write");
    assert_eq!(st.echo.lock().unwrap().n_records(), records, "no echo record");
    assert_eq!(listing(&dir), disk, "no file written");
    assert_eq!(st.worker.lock().unwrap().cache.tokens, live, "the local session is live again");

    // the completion route too
    let b = serde_json::to_vec(&json!({"prompt": sys(), "n_predict": 2, "temperature": 0})).unwrap();
    for _ in 0..2 {
        match handle_policy(st.clone(), "POST", "/completion", "", &b, None, None, true).await {
            Reply::Json(200, j) => assert_eq!(j["timings"]["cache_n"], 0, "{j}"),
            Reply::Json(c, j) => panic!("{c}: {j}"),
            Reply::Stream(_) => panic!("stream"),
        }
    }
    assert_eq!(listing(&dir), disk);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn state_policy_header_is_strict() {
    let h = |vals: &[&'static str]| {
        let mut m = HeaderMap::new();
        for v in vals {
            m.append("x-oxe-state-policy", HeaderValue::from_static(v));
        }
        state_policy(&m)
    };
    assert_eq!(h(&[]), Ok(false));
    assert_eq!(h(&["ephemeral"]), Ok(true));
    assert_eq!(h(&["Ephemeral"]), Ok(true));
    assert!(h(&["persist"]).is_err());
    assert!(h(&[""]).is_err());
    assert!(h(&["ephemeral", "ephemeral"]).is_err());
}
