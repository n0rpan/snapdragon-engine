//! CACHE-PERFECT P2 state store tests: model-free (synthetic byte-level tokenizer, Qwen3.x
//! template fixture, stub engine whose "KV rows" are the token ids and whose "recurrent state" is
//! its script position). Every store lives in its own temp directory; nothing under HOME.
//! The bitwise round trip on a real (tiny synthetic) hybrid model with recurrent and MTP state is
//! `model::tests::state_blocks_round_trip_bitwise`.

use crate::engine::{Engine, StubEngine};
use crate::generate::Worker;
use crate::service::{handle, handle_session, Config, Reply, State};
use crate::state_store::{chain, Identity, StateStore, BLOCK};
use crate::tests_echo::synth_tokenizer;
use serde_json::{json, Value as Json};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub(crate) fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("oxe-state-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

pub(crate) fn ident(model: &str, arith: &str) -> Identity {
    let eng = StubEngine::new(16, 300, 0, vec![]);
    Identity::new(model, &eng.state_id().unwrap(), 16, arith, "build-a", false)
}

pub(crate) fn open(dir: &Path, id: Identity, budget: u64) -> StateStore {
    let mut s = StateStore::open(dir, id, budget, 7 * 24 * 3600, true).unwrap();
    s.min_tokens = 256;
    s.min_free_bytes = 0;
    s
}

pub(crate) fn server(dir: &Path, scripts: Vec<Vec<u32>>, id: Identity, budget: u64) -> Arc<State<StubEngine>> {
    let tok = synth_tokenizer();
    let mut eng = StubEngine::new(1 << 16, tok.n_vocab(), tok.eos.unwrap(), tok.encode("x", false, true));
    eng.scripts.extend(scripts);
    let cfg = Config {
        model_name: "synth".into(),
        model_path: "/synth.gguf".into(),
        chat_opts: chat::ServerChatOptions { enable_thinking: false, ..Default::default() },
        fingerprint: "oxe-test".into(),
        slots: false,
        idle_reprefill: false,
        engine_status: None,
    };
    let mut w = Worker::new(eng, 8);
    w.ubatch = 16;
    w.states = Some(open(dir, id, budget));
    Arc::new(State::new(cfg, tok, w).unwrap())
}

/// A system prompt of ~700 tokens (byte tokens): the static prefix every session shares.
pub(crate) fn sys() -> String {
    (0..20).map(|i| format!("Rule {i:02}: keep the change small and test it. ")).collect()
}

pub(crate) fn body(msgs: Json) -> Json {
    json!({"messages": msgs, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false, "preserve_thinking": true}})
}

pub(crate) async fn chat(st: &Arc<State<StubEngine>>, msgs: Json) -> Json {
    let n0 = st.persist_done.load(Ordering::SeqCst);
    let r = match handle_session(st.clone(), "POST", "/v1/chat/completions", "", &serde_json::to_vec(&body(msgs)).unwrap(), Some("hint-1".into())).await {
        Reply::Json(c, j) => {
            assert_eq!(c, 200, "{j}");
            j
        }
        Reply::Stream(_) => panic!("stream"),
    };
    // the state-store write runs after the reply: wait for it
    let t0 = std::time::Instant::now();
    while st.persist_done.load(Ordering::SeqCst) == n0 {
        assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    r
}

pub(crate) async fn call(st: &Arc<State<StubEngine>>, method: &str, path: &str, b: Json) -> (u16, Json) {
    match handle(st.clone(), method, path, "", &serde_json::to_vec(&b).unwrap()).await {
        Reply::Json(c, j) => (c, j),
        Reply::Stream(_) => panic!("stream"),
    }
}

pub(crate) fn kv(st: &State<StubEngine>) -> Vec<u32> {
    st.worker.lock().unwrap().engine.tokens.clone()
}

pub(crate) fn files(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return v };
    for d in rd.flatten() {
        let p = d.path();
        if p.is_dir() {
            v.extend(files(&p, ext));
        } else if p.to_string_lossy().ends_with(ext) {
            v.push(p);
        }
    }
    v
}

#[test]
fn chain_hash_vectors_and_snapshot_keys() {
    let id = ident("sha256:m", "");
    let t: Vec<u32> = (0..(3 * BLOCK + 10) as u32).collect();
    let hs = chain(&id.hash, &t);
    assert_eq!(hs.len(), 3, "full blocks only");
    // a chain is a function of the identity and every earlier token
    assert_eq!(chain(&id.hash, &t[..2 * BLOCK]), hs[..2]);
    let mut t2 = t.clone();
    t2[5] = 99_999;
    let hs2 = chain(&id.hash, &t2);
    assert!(hs2.iter().zip(&hs).all(|(a, b)| a != b), "a change in block 0 changes every later hash");
    assert_ne!(chain(&ident("sha256:m", "OXE_ATTN=tiled").hash, &t)[0], hs[0], "another identity, another chain");
    // a fixed vector (the format is shared with the controller's GC: names must not drift)
    let v = chain(&"0".repeat(64), &vec![7u32; BLOCK]);
    assert_eq!(v[0], {
        let mut s = crate::prefix_store::Sha256::default();
        s.update("0".repeat(64).as_bytes());
        s.update(&vec![7u32; BLOCK].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>());
        s.hex()
    });
    let k1 = crate::state_store::snap_key(&id.hash, &hs, &t, 2 * BLOCK + 5);
    assert_ne!(k1, crate::state_store::snap_key(&id.hash, &hs, &t, 2 * BLOCK + 6));
    assert_ne!(k1, crate::state_store::snap_key(&id.hash, &hs2, &t2, 2 * BLOCK + 5));
}

#[tokio::test(flavor = "multi_thread")]
async fn sessions_share_the_static_prefix_and_resume_after_a_restart() {
    let tok = synth_tokenizer();
    let dir = tmp("share");
    let a = server(&dir, vec![tok.encode("Answer one.", false, true), tok.encode("Answer two.", false, true)], ident("sha256:m", ""), 1 << 40);
    let m1 = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"}]);
    chat(&a, m1.clone()).await;
    let after1 = kv(&a);
    let m2 = json!([{"role":"system","content":sys()},{"role":"user","content":"another conversation"}]);
    chat(&a, m2.clone()).await;
    let (n_manifests, listed): (usize, usize) = {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        (s.manifests.len(), s.manifests.values().map(|m| m.blocks.len()).sum())
    };
    assert_eq!(n_manifests, 2, "one manifest per session");
    let on_disk = files(&dir, ".kv").len();
    assert!(on_disk < listed, "the shared static prefix is stored once: {on_disk} block files for {listed} listed blocks");
    // GET /state-cache
    let (c, u) = call(&a, "GET", "/state-cache", Json::Null).await;
    assert_eq!(c, 200);
    assert_eq!(u["sessions"].as_array().unwrap().len(), 2);
    assert!(u["used_bytes"].as_u64().unwrap() > 0);
    drop(a);

    // restart: a fresh engine (empty KV); conversation 1 continues
    let b = server(&dir, vec![tok.encode("Answer three.", false, true), tok.encode("Answer four.", false, true)], ident("sha256:m", ""), 1 << 40);
    let m1b = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"},{"role":"assistant","content":"Answer one."},{"role":"user","content":"more"}]);
    let r = chat(&b, m1b).await;
    assert_eq!(r["timings"]["cache_n"].as_u64().unwrap() as usize, after1.len(), "resumed at the head: every token of turn 1, no prefill");
    let fed = kv(&b);
    assert_eq!(fed[..after1.len()], after1[..], "the restored state is the uninterrupted one");
    assert!(b.worker.lock().unwrap().engine.calls.iter().any(|c| c.0 == "snap"), "loaded from the state store");
    // a NEW conversation with the same system prompt: longest match = the shared anchor snapshot
    let m3 = json!([{"role":"system","content":sys()},{"role":"user","content":"a third conversation"}]);
    let c0 = b.worker.lock().unwrap().engine.calls.len();
    {
        // drop the live state so only the disk can serve the prefix
        let mut w = b.worker.lock().unwrap();
        let w = &mut *w;
        w.cache.reset(&mut w.engine);
    }
    let r3 = chat(&b, m3).await;
    let cache_n = r3["timings"]["cache_n"].as_u64().unwrap() as usize;
    let sys_tokens = tok.encode(&format!("<|im_start|>system\n{}<|im_end|>\n", sys()), false, true).len();
    assert!(cache_n >= BLOCK && cache_n <= sys_tokens, "resumed from the shared anchor: cache_n {cache_n}, system {sys_tokens}");
    assert!(b.worker.lock().unwrap().engine.calls[c0..].iter().any(|c| c.0 == "rows"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn truncated_or_corrupt_files_are_rejected_and_the_request_still_answers() {
    let tok = synth_tokenizer();
    let dir = tmp("crash");
    let a = server(&dir, vec![tok.encode("Answer one.", false, true)], ident("sha256:m", ""), 1 << 40);
    let m1 = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"}]);
    chat(&a, m1).await;
    drop(a);
    // a crash mid-write: a stale .pending, a truncated block, a flipped byte in the head snapshot
    let blocks = files(&dir, ".kv");
    let snaps = files(&dir, ".st");
    assert!(!blocks.is_empty() && !snaps.is_empty());
    std::fs::write(blocks[0].with_extension("kv.pending"), b"partial").unwrap();
    let b0 = std::fs::read(&blocks[0]).unwrap();
    std::fs::write(&blocks[0], &b0[..b0.len() - 7]).unwrap();
    let b = server(&dir, vec![tok.encode("Answer three.", false, true)], ident("sha256:m", ""), 1 << 40);
    assert!(files(&dir, ".pending").is_empty(), "start-up sweeps .pending files");
    let m1b = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"},{"role":"assistant","content":"Answer one."},{"role":"user","content":"more"}]);
    let r = chat(&b, m1b.clone()).await;
    assert_eq!(r["choices"][0]["message"]["content"], "Answer three.", "the request is answered (cold), never from a bad state");
    // the truncated block was deleted on load, and this reply's write put an intact copy back
    assert_eq!(std::fs::read(&blocks[0]).unwrap(), b0, "the bad block was replaced by an intact one");
    assert_eq!(b.worker.lock().unwrap().states.as_ref().unwrap().stats.load_errors, 1);
    // checksum: flip one payload byte of a snapshot
    drop(b);
    for s in files(&dir, ".st") {
        let mut d = std::fs::read(&s).unwrap();
        let i = d.len() - 1;
        d[i] ^= 0x40;
        std::fs::write(&s, d).unwrap();
    }
    let c = server(&dir, vec![tok.encode("Answer four.", false, true)], ident("sha256:m", ""), 1 << 40);
    let r = chat(&c, m1b).await;
    assert_eq!(r["choices"][0]["message"]["content"], "Answer four.");
    assert!(c.worker.lock().unwrap().states.as_ref().unwrap().stats.load_errors >= 1, "a flipped bit is a load error, not a silent wrong state");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test(flavor = "multi_thread")]
async fn another_identity_is_never_used_and_older_identities_are_orphaned() {
    let tok = synth_tokenizer();
    let dir = tmp("ident");
    let a = server(&dir, vec![tok.encode("Answer one.", false, true)], ident("sha256:m", ""), 1 << 40);
    chat(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"q"}])).await;
    drop(a);
    let old_key = ident("sha256:m", "").model_key().to_string();
    // another model in the same root: kept (counted in the budget), never read
    let other = server(&dir, vec![tok.encode("Other.", false, true)], ident("sha256:other", ""), 1 << 40);
    let r = chat(&other, json!([{"role":"system","content":sys()},{"role":"user","content":"q"},{"role":"assistant","content":"Answer one."},{"role":"user","content":"more"}])).await;
    assert_eq!(r["timings"]["cache_n"].as_u64().unwrap(), 0, "another model's state is never used");
    assert!(dir.join(&old_key).is_dir(), "another model's directory stays");
    drop(other);
    // the same model under another arithmetic mode (or numerics epoch): the old directory is an
    // orphan nobody can read: removed at start, and nothing of it is used
    let c = server(&dir, vec![tok.encode("Tiled.", false, true)], ident("sha256:m", "OXE_ATTN=tiled"), 1 << 40);
    assert!(!dir.join(&old_key).exists(), "older identity of the same model removed");
    let r = chat(&c, json!([{"role":"system","content":sys()},{"role":"user","content":"q"},{"role":"assistant","content":"Answer one."},{"role":"user","content":"more"}])).await;
    assert_eq!(r["timings"]["cache_n"].as_u64().unwrap(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Persist sessions straight through the worker (no HTTP): `n` sessions of different lengths.
pub(crate) fn fill(dir: &Path, budget: u64, lens: &[usize]) -> Worker<StubEngine> {
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![]), 8);
    w.ubatch = 16;
    w.states = Some(open(dir, ident("sha256:m", ""), budget));
    let shared: Vec<u32> = (0..BLOCK as u32).map(|x| x % 250).collect();
    for (i, &n) in lens.iter().enumerate() {
        let mut toks = shared.clone();
        toks.extend((0..n - BLOCK).map(|k| ((k * 7 + i * 13) % 250) as u32 + 1));
        w.cache.reset(&mut w.engine);
        w.engine.prefill(&toks).unwrap();
        w.cache.appended(&toks);
        w.last_first_user = Some(BLOCK - 10);
        assert!(w.persist_states(Some(&format!("s{i}")), None) > 0);
    }
    w
}

pub(crate) fn age(w: &mut Worker<StubEngine>, secs: u64) {
    let s = w.states.as_mut().unwrap();
    let now = crate::state_store::now_s();
    let sids: Vec<String> = s.manifests.keys().cloned().collect();
    for sid in sids {
        let mut m = s.manifests[&sid].clone();
        m.atime = now - secs;
        s.put_manifest(m).unwrap();
    }
}

#[test]
fn eviction_by_value_keeps_shared_blocks_and_pinned_sessions_and_a_lowered_budget_applies_at_once() {
    let dir = tmp("evict");
    let mut w = fill(&dir, 1 << 40, &[600, 3000, 1200]);
    age(&mut w, 3600);
    {
        // the short session is also the oldest: lowest value (cost / bytes x recency)
        let s = w.states.as_mut().unwrap();
        let sid = s.manifests.values().find(|m| m.n_tokens == 600).unwrap().sid.clone();
        let mut m = s.manifests[&sid].clone();
        m.atime -= 100 * 3600;
        s.put_manifest(m).unwrap();
    }
    let s = w.states.as_mut().unwrap();
    assert_eq!(s.manifests.len(), 3);
    let used = s.used_bytes();
    // lower the budget a little: one session goes, the lowest value (cost per byte): the shortest
    let freed = s.set_budget(used - 1);
    assert!(freed > 0 && s.used_bytes() <= used - 1);
    assert_eq!(s.manifests.len(), 2);
    assert!(s.manifests.values().all(|m| m.n_tokens != 600), "the lowest-value session went first");
    // every kept block has its prefix: block 0 (shared) is still there for the kept sessions
    for m in s.manifests.values() {
        for (h, c) in &m.blocks {
            assert!(files(&dir, &format!("{h}.{c}.kv")).len() == 1, "a kept session's block is on disk");
        }
    }
    // a pinned session (used in the last 10 minutes) is kept even over budget
    let now = crate::state_store::now_s();
    let sid = s.manifests.keys().next().unwrap().clone();
    let mut m = s.manifests[&sid].clone();
    m.atime = now;
    s.put_manifest(m).unwrap();
    s.set_budget(0);
    assert!(s.manifests.contains_key(&sid), "pinned");
    assert_eq!(s.manifests.len(), 1);
    // clear(model) removes everything of this model, the root budget is then free
    let freed = s.clear("model", None).unwrap();
    assert!(freed > 0 && s.manifests.is_empty() && files(&dir, ".kv").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn ttl_expires_sessions() {
    let dir = tmp("ttl");
    let mut w = fill(&dir, 1 << 40, &[600, 700]);
    age(&mut w, 8 * 24 * 3600);
    // re-open applies the TTL (7 days)
    drop(w);
    let s2 = open(&dir, ident("sha256:m", ""), 1 << 40);
    assert!(s2.manifests.is_empty(), "sessions unused for longer than the TTL are gone");
    assert!(files(&dir, ".kv").is_empty() && files(&dir, ".st").is_empty(), "and their files");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn edits_drop_the_snapshots_above_the_edit_and_rewrite_only_new_blocks() {
    let dir = tmp("edit");
    let mut w = fill(&dir, 1 << 40, &[1100]);
    let written = w.states.as_ref().unwrap().stats.blocks_written;
    // the session continues: only the new blocks are written
    let more: Vec<u32> = (0..300).map(|k| (k % 200) as u32 + 3).collect();
    w.engine.prefill(&more).unwrap();
    w.cache.appended(&more);
    w.persist_states(Some("s0"), None);
    let s = w.states.as_ref().unwrap();
    assert_eq!(s.stats.blocks_written - written, (1400 / BLOCK - 1100 / BLOCK) as u64, "blocks already on disk are not written again");
    // an edit at position 600: the engine re-computes from there (low water); the head snapshot of
    // the new branch replaces the old one and no snapshot above the edit survives
    let n = w.cache.tokens.len();
    let mut toks = w.cache.tokens[..600].to_vec();
    toks.extend((0..500).map(|k| (k % 100) as u32 + 150));
    w.cache.reset(&mut w.engine);
    w.engine.prefill(&toks).unwrap();
    w.cache.appended(&toks);
    w.persist_states(Some("s0"), None);
    let s = w.states.as_ref().unwrap();
    // P2b: the edited branch is a fork with a manifest of its own; the old branch stays intact
    let hs = chain(&s.ident.hash, &toks);
    let m = s.manifests.values().find(|m| m.blocks.iter().map(|b| b.0.clone()).collect::<Vec<_>>() == hs).expect("the new branch's manifest lists its blocks");
    assert_eq!(m.n_tokens, 1100);
    assert!(m.snaps.iter().all(|x| x.pos <= 1100), "no snapshot above the edit in the new branch");
    assert_ne!(n, 1100);
    assert!(s.manifests.values().any(|o| o.n_tokens == n && o.snaps.iter().any(|x| x.pos == n)), "the old branch's head is not replaced");
    let _ = std::fs::remove_dir_all(&dir);
}

/// P2 review: every turn replaces the head snapshot; the old one is swept at once (unswept, the
/// orphans filled the budget and whole sessions were evicted before them). And a block the live
/// list names but a sweep removed (a write cut before its manifest) is written again, so no
/// manifest ever names a missing block.
#[test]
fn replaced_head_snapshots_are_swept_and_missing_live_blocks_are_rewritten() {
    let dir = tmp("orphans");
    let mut w = fill(&dir, 1 << 40, &[1100]);
    // P2b: replaced heads stay as resume points until thinning drops them: run past MAX_SNAPS
    // turns so dropped snapshots exist (mutation-checked: without the sweep on every enforce
    // their files stay and this fails)
    for r in 0..(crate::state_store::MAX_SNAPS as u32 + 6) {
        let more: Vec<u32> = (0..40).map(|k| (k + r) % 200 + 3).collect();
        w.engine.prefill(&more).unwrap();
        w.cache.appended(&more);
        w.persist_states(Some("s0"), None);
    }
    let listed: usize = w.states.as_ref().unwrap().manifests.values().map(|m| m.snaps.len()).sum();
    assert!(listed <= crate::state_store::MAX_SNAPS, "thinned: {listed}");
    assert_eq!(files(&dir, ".st").len(), listed, "no snapshot file without a manifest entry");
    let gone = w.live_blocks[1].0.clone();
    let victim = files(&dir, ".kv").into_iter().find(|p| p.to_string_lossy().contains(&gone)).unwrap();
    std::fs::remove_file(victim).unwrap();
    let more: Vec<u32> = (0..300).map(|k| k % 200 + 5).collect();
    w.engine.prefill(&more).unwrap();
    w.cache.appended(&more);
    w.persist_states(Some("s0"), None);
    let s = w.states.as_ref().unwrap();
    let m = s.manifests.values().max_by_key(|m| m.n_tokens).unwrap();
    assert_eq!(m.blocks.len(), w.cache.tokens.len() / BLOCK);
    assert!(m.blocks.iter().all(|(h, c)| s.has_block(h, c)), "every listed block is on disk");
    assert!(m.snaps.iter().any(|x| x.class == "head" && x.pos == w.cache.tokens.len()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// P2 review: the identity names every arithmetic knob (the first list missed the two that
/// `--pf-group-t1/-t2` set) and the kernel path (target, CPU features, libm probe).
#[test]
fn the_identity_names_every_arithmetic_knob_and_the_kernel_path() {
    std::env::set_var("OXE_PF_GROUP_T1", "77");
    std::env::set_var("OXE_KV8_FUSED", "0");
    let a = crate::state_store::arith_env();
    std::env::remove_var("OXE_PF_GROUP_T1");
    std::env::remove_var("OXE_KV8_FUSED");
    assert!(a.contains("OXE_PF_GROUP_T1=77") && a.contains("OXE_KV8_FUSED=0"), "{a}");
    let k = crate::state_store::kernel_path();
    assert_eq!(k, crate::state_store::kernel_path(), "stable within a process");
    assert!(k.starts_with("kpath=") && k.contains("libm="), "{k}");
    assert_ne!(ident("sha256:m", &k).hash, ident("sha256:m", "").hash);
}

#[tokio::test(flavor = "multi_thread")]
async fn budget_and_clear_endpoints() {
    let tok = synth_tokenizer();
    let dir = tmp("endpoints");
    let a = server(&dir, vec![tok.encode("Answer one.", false, true)], ident("sha256:m", ""), 1 << 40);
    chat(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"q"}])).await;
    let (c, j) = call(&a, "POST", "/state-cache/budget", json!({"bytes": 1})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["budget_bytes"], 1);
    // the session just written is pinned (10 min): kept even at a 1-byte budget
    assert!(j["used_bytes"].as_u64().unwrap() > 1);
    let (c, _) = call(&a, "POST", "/state-cache/budget", json!({"bytes": -5})).await;
    assert_eq!(c, 400);
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "all"})).await;
    assert_eq!(c, 200, "{j}");
    assert!(j["freed_bytes"].as_u64().unwrap() > 0);
    assert!(files(&dir, ".kv").is_empty() && files(&dir, ".st").is_empty());
    assert!(kv(&a).is_empty(), "the RAM side of the scope is dropped too");
    let (c, _) = call(&a, "POST", "/state-cache/clear", json!({"scope": "nonsense"})).await;
    assert_eq!(c, 400);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_home_rule_refuses_other_paths() {
    let r = StateStore::open(Path::new("/var/tmp/oxe-state-refused"), ident("sha256:m", ""), 1 << 30, 60, false);
    assert!(r.err().unwrap().contains("outside the home"));
    assert!(!Path::new("/var/tmp/oxe-state-refused").exists());
    assert!(StateStore::open(Path::new("relative/dir"), ident("sha256:m", ""), 1 << 30, 60, true).is_err());
}

#[test]
fn thinning_keeps_anchor_head_and_recent_points() {
    use crate::state_store::{thin, MAX_SNAPS};
    let n = 200_000;
    let mut pos: Vec<(usize, bool)> = (1..100).map(|i| (i * 2000, false)).collect();
    pos.push((700, true));
    pos.push((n, true));
    let k = thin(pos, n);
    assert!(k.len() <= MAX_SNAPS);
    assert!(k.contains(&700) && k.contains(&n));
    assert!(k.contains(&198_000), "recent points kept");
}
