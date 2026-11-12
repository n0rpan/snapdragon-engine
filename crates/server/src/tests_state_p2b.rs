//! CACHE-PERFECT P2b state store tests (model-free, stub engine; every store in its own temp
//! directory, nothing under HOME): M1 prefill progress, M2 bounded shutdown write, forks with their
//! own manifests, a kill at every write step, the preload route and resume_src, the echo file
//! naming shared with the controller, and (see the next commit) the clear / budget-off semantics
//! across every tier.

use crate::engine::{Engine, StubEngine};
use crate::generate::{GenParams, PersistMode, Worker};
use crate::service::{handle_session, Reply, State};
use crate::state_store::{hint_hash, BLOCK};
use crate::tests_echo::synth_tokenizer;
use crate::tests_state::{call, files, ident, open, server, sys, tmp};
use serde_json::{json, Value as Json};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) fn worker(dir: &Path) -> Worker<StubEngine> {
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7, 8, 9]), 8);
    w.ubatch = 16;
    w.states = Some(open(dir, ident("sha256:m", ""), 1 << 40));
    w
}

pub(crate) fn toks(n: usize, seed: usize) -> Vec<u32> {
    // a shared first block (the "system prompt"), then per-seed tokens
    (0..n).map(|k| if k < BLOCK { (k % 250) as u32 + 1 } else { ((k * 7 + seed * 13) % 250) as u32 + 1 }).collect()
}

fn params(session: &str) -> GenParams {
    GenParams { n_predict: 3, first_user: Some(BLOCK - 10), session: Some(session.into()), ..Default::default() }
}

pub(crate) fn gen(w: &mut Worker<StubEngine>, prompt: &[u32], p: &GenParams) -> crate::generate::GenResult {
    let tok = synth_tokenizer();
    w.generate(&tok, prompt, p, &mut |_| true).unwrap()
}

/// Every manifest of the store at `dir` loads (every snapshot, its blocks): the store is consistent.
fn assert_consistent(dir: &Path) -> usize {
    let mut s = open(dir, ident("sha256:m", ""), 1 << 40);
    let plans: Vec<_> = s
        .manifests
        .values()
        .flat_map(|m| m.snaps.iter().map(move |x| crate::state_store::Plan { sid: m.sid.clone(), blocks: m.blocks[..x.pos / BLOCK].to_vec(), snap: x.clone() }))
        .collect();
    let mut e = StubEngine::new(1 << 16, 300, 0, vec![]);
    for p in &plans {
        s.load(&mut e, p).unwrap_or_else(|err| panic!("manifest {} snapshot at {}: {err}", p.sid, p.snap.pos));
    }
    assert_eq!(s.stats.load_errors, 0);
    plans.len()
}

/// M1: a long prefill writes its blocks and a head snapshot every `persist_every` tokens, so a kill
/// before the reply (and before any post-reply write) loses at most that much: a restart resumes
/// within `persist_every` of the prompt end.
#[test]
fn m1_long_prefill_persists_progress_every_n_tokens() {
    let dir = tmp("p2b-m1");
    let mut w = worker(&dir);
    w.persist_every = 1024;
    let prompt = toks(5000, 1);
    gen(&mut w, &prompt, &params("s1"));
    // no post-reply write ran (the worker's generate alone): only the progress writes
    let s = w.states.as_ref().unwrap();
    assert_eq!(s.manifests.len(), 1, "one lineage, one manifest");
    let m = s.manifests.values().next().unwrap();
    let head = m.snaps.iter().map(|x| x.pos).max().unwrap();
    assert!(head + 1024 >= prompt.len() - 1 && head < prompt.len(), "progress head at {head} of {}", prompt.len());
    assert!(m.snaps.len() >= 3, "older progress heads stay as resume points (thinned): {:?}", m.snaps.iter().map(|x| x.pos).collect::<Vec<_>>());
    assert_eq!(m.hint.as_deref(), Some(hint_hash("s1").as_str()));
    drop(w);
    // "kill" + restart: the same prompt resumes from the last progress head
    let mut w2 = worker(&dir);
    let r = gen(&mut w2, &prompt, &params("s1"));
    assert_eq!(r.timings.cache_n, head, "resumed at the last progress point");
    assert_eq!(r.timings.resume_src, "disk");
    assert_eq!(w2.engine.tokens[..head], prompt[..head]);
    assert_consistent(&dir);
    // off: persist_every 0 writes nothing during the prefill
    let dir2 = tmp("p2b-m1-off");
    let mut w3 = worker(&dir2);
    w3.persist_every = 0;
    gen(&mut w3, &prompt, &params("s1"));
    assert!(w3.states.as_ref().unwrap().manifests.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

/// M2: the shutdown write is incremental (only blocks not on disk + the head), bounded by a
/// deadline and a memory floor, and says what it saved; after a complete write it writes nothing.
#[test]
fn m2_shutdown_write_is_incremental_bounded_and_floored() {
    let dir = tmp("p2b-m2");
    let mut w = worker(&dir);
    w.persist_every = 1024;
    let prompt = toks(5000, 2);
    gen(&mut w, &prompt, &params("s2"));
    let written0 = w.states.as_ref().unwrap().stats.blocks_written;
    let t = Instant::now() + Duration::from_secs(5);
    let r = w.persist(None, PersistMode::Shutdown(t), None);
    let n = w.cache.tokens.len();
    assert_eq!(r.head, Some(n), "{r:?}");
    let new_blocks = w.states.as_ref().unwrap().stats.blocks_written - written0;
    assert!(new_blocks as usize <= 1024 / BLOCK + 1, "only the tail since the last progress write: {new_blocks} blocks");
    // the RAM checkpoints are saved too (deepest first, inside the deadline): v0.1.2 thinking restart
    assert!(r.snaps >= 1, "{r:?}");
    assert!(r.line().contains(&format!("head at {n} tokens")), "{}", r.line());
    // nothing changed: the next shutdown write is a lookup
    let r2 = w.persist(None, PersistMode::Shutdown(t), None);
    assert_eq!((r2.head, r2.note.as_deref()), (None, Some("already on disk")));
    // a new turn, then a deadline already passed: nothing written, the previous manifest stays
    let mut p2 = w.cache.tokens.clone();
    p2.extend(toks(700, 9)[BLOCK..].iter());
    gen(&mut w, &p2, &GenParams { session: Some("s2".into()), ..params("s2") });
    let r3 = w.persist(None, PersistMode::Shutdown(Instant::now() - Duration::from_millis(1)), None);
    assert!(r3.head.is_none() && r3.blocks == 0 && r3.note.as_deref().unwrap_or("").starts_with("stopped"), "{r3:?}");
    // memory floor: MemAvailable below it: skipped and said so, never a big allocation
    let s = w.states.as_mut().unwrap();
    s.min_mem_bytes = 1 << 40;
    s.mem_available = || Some(1 << 20);
    let r4 = w.persist(None, PersistMode::Shutdown(Instant::now() + Duration::from_secs(5)), None);
    assert!(r4.head.is_none() && r4.note.as_deref().unwrap_or("").contains("save floor"), "{r4:?}");
    assert_consistent(&dir);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Bigger item 1: a fork (or retry / edit) past the session id's blocks gets a manifest of its
/// own; the sibling's head is never replaced, and each branch keeps continuing its own manifest.
#[test]
fn forks_get_their_own_manifest_and_never_replace_a_siblings_head() {
    let dir = tmp("p2b-fork");
    let mut w = worker(&dir);
    let base = toks(1200, 3);
    let mut a = base.clone();
    a.extend((0..300).map(|k| (k % 50) as u32 + 100));
    let mut b = base.clone();
    b.extend((0..300).map(|k| (k % 50) as u32 + 160));
    let put = |w: &mut Worker<StubEngine>, t: &[u32]| {
        w.cache.reset(&mut w.engine);
        w.engine.prefill(t).unwrap();
        w.cache.appended(t);
        w.last_first_user = Some(BLOCK - 10);
        w.persist(Some("h"), PersistMode::Reply, None)
    };
    let ra = put(&mut w, &a);
    let rb = put(&mut w, &b);
    assert_ne!(ra.sid, rb.sid, "the fork has its own manifest");
    let s = w.states.as_ref().unwrap();
    assert_eq!(s.manifests.len(), 2);
    let ma = &s.manifests[ra.sid.as_ref().unwrap()];
    assert!(ma.snaps.iter().any(|x| x.class == "head" && x.pos == a.len()), "branch A's head is intact");
    // branch A continues: same manifest, its head moves
    let mut a2 = a.clone();
    a2.extend((0..100).map(|k| k as u32 + 1));
    let ra2 = put(&mut w, &a2);
    assert_eq!(ra2.sid, ra.sid, "a branch continues its own manifest");
    let s = w.states.as_ref().unwrap();
    assert_eq!(s.manifests.len(), 2);
    assert!(s.manifests[rb.sid.as_ref().unwrap()].snaps.iter().any(|x| x.pos == b.len()), "B untouched");
    // both branches resume from disk after a restart
    drop(w);
    let st = open(&dir, ident("sha256:m", ""), 1 << 40);
    for t in [&a2, &b] {
        let mut q = t.to_vec();
        q.push(5);
        assert_eq!(st.lookup(&q, q.len() - 1, 0).unwrap().snap.pos, t.len());
    }
    assert_consistent(&dir);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Design sec. 8 "kill at each write step": a kill between ANY two file-system steps of a turn's
/// write (payload written / fsync'd / renamed, manifest, sweep removals; post-reply and prefill
/// progress modes) leaves a store that loads cleanly at restart, and the next request answers,
/// resuming at least from the previous complete turn.
#[test]
fn kill_at_each_write_step_leaves_a_consistent_store() {
    for mode in [PersistMode::Reply, PersistMode::Progress] {
        // count the steps of an unkilled turn-2 write
        let total = {
            let dir = tmp("p2b-kill-count");
            let (mut w, t2) = kill_setup(&dir);
            let k0 = w.states.as_ref().unwrap().kill.taken();
            w.persist(Some("k"), mode, None);
            let n = w.states.as_ref().unwrap().kill.taken() - k0;
            let _ = std::fs::remove_dir_all(&dir);
            drop(t2);
            n
        };
        assert!(total >= 8, "{mode:?}: {total} steps");
        for k in 0..=total {
            let dir = tmp(&format!("p2b-kill-{k}"));
            let (mut w, t2) = kill_setup(&dir);
            let turn1 = w.states.as_ref().unwrap().manifests.values().map(|m| m.n_tokens).max().unwrap();
            {
                let s = w.states.as_mut().unwrap();
                s.kill.at = Some(s.kill.taken() + k);
            }
            w.persist(Some("k"), mode, None);
            drop(w);
            // restart: start-up sweeps; every manifest left must load
            let n_snaps = assert_consistent(&dir);
            assert!(n_snaps > 0, "kill at step {k}: the previous turn's state survives");
            assert!(files(&dir, ".pending").is_empty(), "kill at step {k}: start-up swept the .pending files");
            let mut w2 = worker(&dir);
            let mut q = t2.clone();
            q.extend([40, 41, 42]);
            let r = gen(&mut w2, &q, &params("k"));
            assert!(r.timings.cache_n >= turn1, "kill at step {k} ({mode:?}): resumed {} < turn 1 {turn1}", r.timings.cache_n);
            assert_eq!(w2.engine.tokens[..q.len()], q[..], "the request answered on the right state");
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

fn kill_setup(dir: &Path) -> (Worker<StubEngine>, Vec<u32>) {
    let mut w = worker(dir);
    w.persist_every = 0;
    let t1 = toks(900, 5);
    w.engine.prefill(&t1).unwrap();
    w.cache.appended(&t1);
    w.last_first_user = Some(BLOCK - 10);
    assert!(w.persist(Some("k"), PersistMode::Reply, None).head.is_some());
    // turn 2: new blocks, a RAM checkpoint, a new head
    let mut t2 = t1.clone();
    t2.extend(toks(1000, 6)[BLOCK..].iter());
    let more = t2[t1.len()..t1.len() + 300].to_vec();
    w.engine.prefill(&more).unwrap();
    w.cache.appended(&more);
    w.cache.checkpoint(&mut w.engine).unwrap();
    let rest = t2[t1.len() + 300..].to_vec();
    w.engine.prefill(&rest).unwrap();
    w.cache.appended(&rest);
    (w, t2)
}

pub(crate) async fn chat_h(st: &Arc<State<StubEngine>>, msgs: Json, hint: &str) -> Json {
    let n0 = st.persist_done.load(Ordering::SeqCst);
    let b = crate::tests_state::body(msgs);
    let r = match handle_session(st.clone(), "POST", "/v1/chat/completions", "", &serde_json::to_vec(&b).unwrap(), Some(hint.into())).await {
        Reply::Json(c, j) => {
            assert_eq!(c, 200, "{j}");
            j
        }
        Reply::Stream(_) => panic!("stream"),
    };
    let t0 = Instant::now();
    while st.persist_done.load(Ordering::SeqCst) == n0 {
        assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    r
}

/// A server whose echo records live on disk next to the state store (as main.rs attaches them).
pub(crate) fn server_echo(dir: &Path, scripts: Vec<Vec<u32>>, budget: u64) -> Arc<State<StubEngine>> {
    let st = server(dir, scripts, ident("sha256:m", ""), budget);
    {
        let w = st.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        let mut es = st.echo.lock().unwrap();
        es.set_identity(&s.ident.text);
        es.attach(&s.dir.join("echo")).unwrap();
    }
    st
}

pub(crate) fn model_dir(st: &State<StubEngine>) -> std::path::PathBuf {
    st.worker.lock().unwrap().states.as_ref().unwrap().dir.clone()
}

/// Bigger item 2: one echo file naming scheme. A session's echo file is
/// `echo/<hint_hash(hint)>.jsonl`, the manifest's `hint` field names it, and evicting the last
/// manifest with that hint removes it.
#[tokio::test(flavor = "multi_thread")]
async fn echo_files_are_named_by_the_manifest_hint_and_go_with_the_last_manifest() {
    let tok = synth_tokenizer();
    let dir = tmp("p2b-echo");
    let a = server_echo(&dir, vec![tok.encode("Answer one.", false, true)], 1 << 40);
    chat_h(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"first question"}]), "conv-1").await;
    let md = model_dir(&a);
    let h = hint_hash("conv-1");
    assert!(md.join("echo").join(format!("{h}.jsonl")).is_file(), "echo file named by the hint hash");
    let sid = {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        let m = s.manifests.values().next().unwrap();
        assert_eq!(m.hint.as_deref(), Some(h.as_str()), "the manifest's hint field is the echo file stem");
        m.sid.clone()
    };
    // eviction of the last manifest with that hint takes the echo file along
    {
        let mut w = a.worker.lock().unwrap();
        let s = w.states.as_mut().unwrap();
        let mut m = s.manifests[&sid].clone();
        m.atime -= 3 * 3600;
        s.put_manifest(m).unwrap();
        s.set_budget(1);
        assert!(s.manifests.is_empty());
    }
    assert!(!md.join("echo").join(format!("{h}.jsonl")).exists(), "the evicted session's echo file is gone");
    let _ = std::fs::remove_dir_all(&dir);
}

/// P2b preload + resume_src: after a restart a preload puts the newest session's state into the
/// engine in the background; the request that continues it uses it (resume_src ram, preload_ms),
/// a request for anything else drops it. A busy worker gets "busy", an engine already holding a
/// conversation "live", a cancelled preload leaves the engine reset.
#[tokio::test(flavor = "multi_thread")]
async fn preload_restores_the_newest_session_and_yields_to_requests() {
    let tok = synth_tokenizer();
    let dir = tmp("p2b-preload");
    let a = server_echo(&dir, vec![tok.encode("Answer one.", false, true)], 1 << 40);
    let m1 = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"}]);
    let r = chat_h(&a, m1, "conv-p").await;
    assert_eq!(r["timings"]["resume_src"], "cold");
    let head = crate::tests_state::kv(&a).len();
    drop(a);
    let b = server_echo(&dir, vec![tok.encode("Answer two.", false, true)], 1 << 40);
    let (c, j) = call(&b, "POST", "/state-cache/preload", json!({"session": "conv-p"})).await;
    assert_eq!((c, j["status"].as_str()), (202, Some("started")), "{j}");
    let t0 = Instant::now();
    loop {
        let (_, u) = call(&b, "GET", "/state-cache", Json::Null).await;
        if u["preload"]["status"] == "done" {
            assert_eq!(u["preload"]["pos"].as_u64().unwrap() as usize, head);
            break;
        }
        assert!(t0.elapsed().as_secs() < 10, "preload did not finish: {u}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let (_, j) = call(&b, "POST", "/state-cache/preload", json!({})).await;
    assert_eq!(j["status"], "live", "a second preload never replaces a loaded state");
    let m2 = json!([{"role":"system","content":sys()},{"role":"user","content":"first question"},{"role":"assistant","content":"Answer one."},{"role":"user","content":"more"}]);
    let r = chat_h(&b, m2, "conv-p").await;
    assert_eq!(r["timings"]["cache_n"].as_u64().unwrap() as usize, head);
    assert_eq!(r["timings"]["resume_src"], "ram", "{}", r["timings"]);
    assert_eq!(r["timings"]["preload_src"], "disk");
    assert!(r["timings"]["preload_ms"].as_f64().is_some());
    // busy: the worker is held (a request runs)
    {
        let _g = b.worker.lock().unwrap();
        let (_, j) = futures_block(call(&b, "POST", "/state-cache/preload", json!({})));
        assert_eq!(j["status"], "busy");
    }
    // a preload whose request flag is already up loads nothing and leaves the engine reset
    {
        let mut w = b.worker.lock().unwrap();
        let w = &mut *w;
        w.cache.reset(&mut w.engine);
        let f = AtomicBool::new(true);
        let (status, _, _) = w.preload(None, &f);
        assert_eq!(status, "cancelled");
        assert_eq!(w.engine.n_past(), 0);
        assert!(w.preloaded.is_none());
    }
    // a preloaded state that the next request does not continue is dropped, never mixed in
    {
        let mut w = b.worker.lock().unwrap();
        let f = AtomicBool::new(false);
        assert_eq!(w.preload(None, &f).0, "done");
        let other = toks(400, 77);
        let r = gen(&mut w, &other, &GenParams { n_predict: 2, ..Default::default() });
        assert_eq!((r.timings.cache_n, r.timings.resume_src), (0, "cold"));
        assert_eq!(w.engine.tokens[..other.len()], other[..]);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Run a future to completion on a fresh current-thread runtime from inside a blocking section.
fn futures_block<F: std::future::Future + Send>(f: F) -> F::Output
where
    F::Output: Send,
{
    std::thread::scope(|s| s.spawn(|| tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(f)).join().unwrap())
}


/// Codex P2/P3 review: start-up never reads a manifest or an echo file unbounded: an oversized
/// manifest is dropped unread (a miss, never a guess), an oversized echo file is skipped.
#[test]
fn oversized_manifests_and_echo_files_are_not_read() {
    let dir = tmp("p2b-bounded");
    let s = open(&dir, ident("sha256:m", ""), 1 << 40);
    let big = s.dir.join("sessions").join(format!("{}.json", "a".repeat(32)));
    let f = std::fs::File::create(&big).unwrap();
    f.set_len(crate::state_store::MAX_MANIFEST_BYTES + 1).unwrap();
    drop(s);
    let s = open(&dir, ident("sha256:m", ""), 1 << 40);
    assert!(s.manifests.is_empty() && !big.exists(), "an oversized manifest is refused and removed");
    let echo = s.dir.join("echo");
    let f = std::fs::File::create(echo.join(format!("{}.jsonl", "b".repeat(32)))).unwrap();
    f.set_len(crate::echo::MAX_ECHO_FILE_BYTES + 1).unwrap();
    let mut es = crate::echo::EchoStore::new(Default::default());
    es.set_identity(&s.ident.text);
    assert_eq!(es.attach(&echo).unwrap(), 0, "skipped unread");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Codex review of 63eac16 (M2): the real SIGTERM dispatcher. With the state cache on, a slow
/// legacy whole-state writer (the session cache) is never started, so the whole sequence ends
/// within the deadline; without the state cache the legacy sequence runs as before. Mutation-
/// checked: the old dispatcher (legacy writer whenever the clock is before the deadline) fails.
#[test]
fn shutdown_dispatch_never_starts_the_unbounded_legacy_writer_with_the_state_cache_on() {
    let dir = tmp("p2b-dispatch");
    let mut w = worker(&dir);
    w.persist_every = 1024;
    let prompt = toks(3000, 4);
    gen(&mut w, &prompt, &params("d"));
    let called = std::cell::Cell::new(0);
    let mut slow = |_: &mut Worker<StubEngine>| {
        called.set(called.get() + 1);
        std::thread::sleep(Duration::from_secs(3));
        7
    };
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_millis(2500);
    let log = w.shutdown_writes(true, true, t0, deadline, &mut slow);
    assert_eq!(called.get(), 0, "the legacy writer never starts with the state cache on: {log:?}");
    assert!(t0.elapsed() < Duration::from_millis(2500), "the whole sequence ends inside the deadline: {:?}", t0.elapsed());
    assert!(log[0].starts_with("state cache: shutdown write: ") && log[0].contains("head at"), "{log:?}");
    assert!(log.iter().any(|l| l.starts_with("session cache: not written at shutdown")), "{log:?}");
    // without the state cache: the legacy sequence as before
    let mut w2 = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![]), 8);
    let mut fast = |_: &mut Worker<StubEngine>| {
        called.set(called.get() + 1);
        3
    };
    let log = w2.shutdown_writes(false, true, Instant::now(), Instant::now() + Duration::from_secs(1), &mut fast);
    assert_eq!(called.get(), 1);
    assert!(log.iter().any(|l| l.starts_with("session cache: wrote 3 sessions")), "{log:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Codex review of 63eac16 (nonblocking): a direct preload at a live budget 0 reads nothing.
#[test]
fn preload_at_budget_zero_reads_nothing() {
    let dir = tmp("p2b-preload0");
    let mut w = worker(&dir);
    w.persist_every = 1024;
    gen(&mut w, &toks(3000, 8), &params("z"));
    let w = &mut w;
    w.cache.reset(&mut w.engine);
    w.states.as_mut().unwrap().budget = 0;
    assert_eq!(w.preload(None, &AtomicBool::new(false)).0, "off");
    assert_eq!(w.engine.n_past(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// v0.1.2 gate finding (Qwen 82K warm restart 4.61 s against 3 s): the segmented store resumed
/// 78,876 tokens, then the legacy session cache loaded 82,716 on top (two restores). Now ONE
/// restore, chosen before the engine is touched by estimated restore + recompute cost: the legacy
/// entry wins when it is deeper (skipping it would re-prefill ~3.9K tokens at 82K), the segmented
/// snapshot when it is deeper or equal, and a restore that costs more than it saves loses to
/// staying with the RAM state. (The old code ran both restores: `restores` 2.)
#[test]
fn one_restore_per_request_the_deepest_source() {
    // (segmented head, legacy session entry) for a 1500-token conversation
    for (seg, leg) in [(1100usize, 1500usize), (1500, 1100), (1300, 1300)] {
        let dir = tmp(&format!("p2b-choose-{seg}-{leg}"));
        let prompt = toks(1500, 11);
        {
            let mut w = worker(&dir);
            w.persist_every = 0;
            w.engine.prefill(&prompt[..seg]).unwrap();
            w.cache.appended(&prompt[..seg]);
            w.last_first_user = Some(BLOCK - 10);
            assert_eq!(w.persist(Some("q"), PersistMode::Reply, None).head, Some(seg));
            w.engine.reset();
            w.engine.prefill(&prompt[..leg]).unwrap();
            let mut ps = crate::prefix_store::PrefixStore::open(&dir.join("legacy-sessions"), "legacy-id".into(), 1 << 30, 64).unwrap();
            let eng = &mut w.engine;
            ps.save_stream(&prompt[..leg], &mut |out| eng.save_state(out).map_err(|e| e.0)).unwrap();
        }
        let mut w = worker(&dir);
        w.sessions = Some(crate::prefix_store::PrefixStore::open(&dir.join("legacy-sessions"), "legacy-id".into(), 1 << 30, 64).unwrap());
        let mut q = prompt.clone();
        q.extend([3, 4, 5]);
        let r = gen(&mut w, &q, &params("q"));
        let (want_n, want_src, load_call) = if leg > seg { (leg, "session", "load") } else { (seg, "state", "snap") };
        assert_eq!(r.timings.cache_n, want_n, "seg {seg} leg {leg}: the deeper source serves");
        assert_eq!((w.cache.last_src, r.timings.resume_src), (want_src, "disk"), "seg {seg} leg {leg}");
        assert_eq!(w.restores, 1, "seg {seg} leg {leg}: exactly one restore");
        assert!(w.engine.calls.iter().any(|c| c.0 == load_call));
        let other = if load_call == "load" { "snap" } else { "load" };
        assert!(!w.engine.calls.iter().any(|c| c.0 == other), "seg {seg} leg {leg}: the other source is never loaded");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Deepest exact state first, whatever the estimated cost (v0.1.2 thinking-restart identity: a
/// shallower state + a batched re-prefill is not the decoded state); cost decides only between
/// sources of equal depth; a source no deeper than the live / RAM state is not listed.
#[test]
fn restore_plan_prefers_the_deepest_exact_state_and_uses_cost_only_at_equal_depth() {
    use crate::generate::RestoreSrc::{Session, State};
    let dir = tmp("p2b-price");
    let prompt = toks(1500, 12);
    let mut w = worker(&dir);
    w.persist_every = 0;
    w.engine.prefill(&prompt[..1200]).unwrap();
    w.cache.appended(&prompt[..1200]);
    w.last_first_user = Some(BLOCK - 10);
    w.persist(Some("q"), PersistMode::Reply, None);
    let legacy = |w: &mut Worker<StubEngine>, n: usize, name: &str| {
        let mut ps = crate::prefix_store::PrefixStore::open(&dir.join(name), "legacy-id".into(), 1 << 30, 64).unwrap();
        w.engine.reset();
        w.engine.prefill(&prompt[..n]).unwrap();
        let eng = &mut w.engine;
        ps.save_stream(&prompt[..n], &mut |out| eng.save_state(out).map_err(|e| e.0)).unwrap();
        w.sessions = Some(ps);
    };
    let mut q = prompt.clone();
    q.push(9);
    // the legacy entry 10 positions deeper: first even when its restore is absurdly slow
    legacy(&mut w, 1210, "deeper");
    w.restore_bps = (1e12, 1.0);
    assert_eq!(w.plan_restore(&q, 0), vec![Session, State]);
    // segmented deeper: first even when the legacy store is far faster
    legacy(&mut w, 1100, "shallower");
    w.restore_bps = (1.0, 1e12);
    assert_eq!(w.plan_restore(&q, 0), vec![State, Session]);
    // equal depth: the cheaper source first, either way
    legacy(&mut w, 1200, "equal");
    w.restore_bps = (1.0, 1e12);
    assert_eq!(w.plan_restore(&q, 0), vec![Session, State]);
    w.restore_bps = (1e12, 1.0);
    assert_eq!(w.plan_restore(&q, 0), vec![State, Session]);
    // the live / RAM state already as deep: nothing on disk is restored over it
    assert!(w.plan_restore(&q, 1200).is_empty());
    assert_eq!(w.plan_restore(&q, 1150), vec![State, Session], "deeper than RAM: restored, whatever it costs");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Mirror of the v0.1.2 small-lifecycle failure (Codex, 575b520): a thinking turn and its
/// follow-up came back to back, so both post-reply writes were cut by the next request; the
/// follow-up's exact resume point (its tail checkpoint, after the reply's decoded state) lived
/// only in RAM and the SIGTERM write saved the head alone. The restart re-prefilled from the
/// 9216 anchor and the output changed. Now the shutdown write saves the RAM checkpoints too, and
/// the restart resumes the follow-up exactly where the uninterrupted run was.
#[test]
fn thinking_follow_up_after_a_restart_resumes_the_exact_ram_checkpoint() {
    let dir = tmp("p2b-thinking");
    let mut w = worker(&dir);
    w.persist_every = 0;
    let p1 = toks(1300, 21);
    gen(&mut w, &p1, &GenParams { n_predict: 4, ..params("t") });
    // the follow-up continues the live state (reply included) with a new user turn
    let turn_end = w.cache.tokens.len();
    assert_ne!(turn_end % BLOCK, 0, "the turn ends inside a block (partial-block rows in its snapshot)");
    let mut p2 = w.cache.tokens.clone();
    p2.extend((0..20).map(|k| k as u32 + 30));
    let ctl = gen(&mut w, &p2, &GenParams { n_predict: 4, ..params("t") });
    assert_eq!((ctl.timings.resume_src, ctl.timings.cache_n), ("live", p2.len() - 20));
    let ctl_state = w.engine.tokens.clone();
    // no post-reply write ran (cut by the next request); SIGTERM: the bounded shutdown write
    let r = w.persist(None, PersistMode::Shutdown(Instant::now() + Duration::from_secs(5)), None);
    assert!(r.head.is_some() && r.snaps >= 1, "head and the RAM checkpoints: {r:?}");
    let saved: Vec<usize> = w.states.as_ref().unwrap().manifests.values().flat_map(|m| m.snaps.iter().map(|x| x.pos)).collect();
    assert!(saved.contains(&turn_end), "a snapshot at the EXACT turn end {turn_end}, not the last block boundary: {saved:?}");
    assert!(saved.contains(&(p2.len() - 1)), "and at the follow-up's tail: {saved:?}");
    drop(w);
    let mut w2 = worker(&dir);
    let res = gen(&mut w2, &p2, &GenParams { n_predict: 4, ..params("t") });
    assert_eq!(res.timings.cache_n, p2.len() - 1, "resumed at the follow-up's tail checkpoint, nothing re-prefilled but the last token");
    assert_eq!(res.timings.resume_src, "disk");
    assert_eq!(w2.restores, 1);
    assert_eq!(w2.engine.tokens[..p2.len()], ctl_state[..p2.len()], "the resumed state is the uninterrupted one");
    // another follow-up of the same turn (an edit / retry of the user message): it resumes at the
    // exact turn end, not at the last 256 boundary
    let mut p3 = p2[..turn_end].to_vec();
    p3.extend((0..20).map(|k| k as u32 + 90));
    let mut w3 = worker(&dir);
    let r3 = gen(&mut w3, &p3, &GenParams { n_predict: 4, ..params("t") });
    assert_eq!((r3.timings.cache_n, r3.timings.resume_src), (turn_end, "disk"));
    let _ = std::fs::remove_dir_all(&dir);
}
