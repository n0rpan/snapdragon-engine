//! v0.1.3 store-depth tests: the segmented state store is at least as deep as the legacy session
//! cache after every reply, a cut post-reply write and a SIGTERM, across restarts; the eviction
//! order keeps the deep recent state; the legacy writes can go off. Model-free (stub engine,
//! synthetic tokenizer) plus the tiny synthetic hybrid model (MTP off and on, bitwise). Every
//! store lives in its own temp directory; nothing under HOME, no token ids written elsewhere.
//!
//! v0.1.2 gate (Qwen, 82.8K warm restart): the segmented store resumed 78,876 tokens, the legacy
//! session cache 82,716. Cause: the post-reply write put the head on disk and then the optional
//! RAM points SHALLOWEST first; the next request (~0.1 s later) cut it after one file (a 74K turn
//! point), so the 82.7K points never reached the store, while the legacy park streamed the two
//! deepest points. A SIGTERM write also skipped the points such a cut left ("already on disk").

use crate::engine::{Engine, StubEngine};
use crate::generate::{legacy_session_writes, GenParams, PersistMode, Worker};
use crate::sampling::SamplingParams;
use crate::service::{handle_request, Reply, State};
use crate::state_store::{now_s, StateStore, BLOCK};
use crate::tests_echo::synth_tokenizer;
use crate::tests_state::{ident, open, tmp};
use crate::tests_state_p2b::{gen, toks, worker};
use crate::tests_state_v013::{body, srv};
use serde_json::{json, Value as Json};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn greedy(n_predict: i64) -> GenParams {
    GenParams { n_predict, sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() }
}

/// A long first turn with user-turn points (RAM checkpoints below the prompt end).
fn turn_params() -> GenParams {
    let mut gp = greedy(3);
    gp.first_user = Some(BLOCK - 10);
    gp.msg_starts = vec![0, BLOCK - 10, 600, 900, 1200, 1400];
    gp.user_starts = vec![BLOCK - 10, 600, 1200, 1400];
    gp.session = Some("a".into());
    gp
}

/// The deepest RAM checkpoint strictly below the live head (what the legacy park would stream first).
fn deepest_ram_point(w: &Worker<StubEngine>) -> usize {
    let n = w.cache.tokens.len();
    w.cache.snapshot_info().iter().map(|x| x.0).filter(|&p| p < n).max().unwrap()
}

/// Root cause, part 1: a request that arrives after ONE optional snapshot file cuts the post-reply
/// write; that one file is the deepest point (the prompt's last token, a retry / warm-restart
/// resume point), so a replay of the same prompt after a restart resumes there, as deep as the
/// legacy park's first entry. Shallowest first (v0.1.2) it resumed at the shallow turn point.
#[test]
fn a_cut_post_reply_write_keeps_the_deepest_point() {
    let dir = tmp("depth-cut");
    let mut w = worker(&dir);
    w.persist_every = 0;
    let prompt = toks(1500, 4);
    gen(&mut w, &prompt, &turn_params());
    let deep = deepest_ram_point(&w);
    assert!(deep >= prompt.len() - 16, "a RAM point near the prompt end: {deep}");
    let n = w.cache.tokens.len();
    w.opt_write_limit = 1;
    let r = w.persist(Some("a"), PersistMode::Reply, None);
    assert_eq!((r.head, r.snaps), (Some(n), 1), "{r:?}");
    {
        let st = w.states.as_ref().unwrap();
        let m = st.manifests.values().find(|m| m.n_tokens == n).unwrap();
        let below: Vec<usize> = m.snaps.iter().map(|x| x.pos).filter(|&p| p < n && p > BLOCK).collect();
        assert_eq!(below, vec![deep], "the one optional file written is the deepest point");
    }
    drop(w);
    // restart; the same request again (a retry, the gate's warm-restart replay)
    let mut w2 = worker(&dir);
    w2.persist_every = 0;
    let r2 = gen(&mut w2, &prompt, &turn_params());
    assert_eq!((r2.timings.cache_n, r2.timings.resume_src), (deep, "disk"), "resumed at the deepest point");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Root cause, part 2: SIGTERM right after a reply whose optional writes were cut (the post-reply
/// write's flag is raised by the signal): the shutdown write now puts the points it left on disk,
/// deepest first, inside its deadline; v0.1.2 returned "already on disk" (head listed).
#[test]
fn shutdown_writes_the_points_a_cut_reply_write_left() {
    let dir = tmp("depth-shutdown");
    let mut w = worker(&dir);
    w.persist_every = 0;
    let prompt = toks(1500, 5);
    gen(&mut w, &prompt, &turn_params());
    let deep = deepest_ram_point(&w);
    w.opt_write_limit = 0;
    let r = w.persist(Some("a"), PersistMode::Reply, None);
    assert_eq!(r.snaps, 0, "every optional file cut");
    w.opt_write_limit = usize::MAX;
    let t0 = Instant::now();
    let log = w.shutdown_writes(true, false, t0, t0 + Duration::from_secs(5), &mut |_| 0);
    assert!(log[0].contains("snapshots"), "{log:?}");
    // a second shutdown write has nothing left
    let r3 = w.persist(None, PersistMode::Shutdown(Instant::now() + Duration::from_secs(5)), None);
    assert_eq!(r3.note.as_deref(), Some("already on disk"));
    drop(w);
    let mut w2 = worker(&dir);
    w2.persist_every = 0;
    let r2 = gen(&mut w2, &prompt, &turn_params());
    assert_eq!((r2.timings.cache_n, r2.timings.resume_src), (deep, "disk"));
    // a deadline already too close: no optional file is started (the head and manifest stay bounded)
    let dir2 = tmp("depth-shutdown-late");
    let mut w3 = worker(&dir2);
    w3.persist_every = 0;
    gen(&mut w3, &prompt, &turn_params());
    let r4 = w3.persist(None, PersistMode::Shutdown(Instant::now() + Duration::from_millis(100)), None);
    assert_eq!((r4.head.is_some(), r4.snaps), (true, 0), "{r4:?}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

/// One chat request through the HTTP layer (hint "h"); waits for its post-reply state write.
async fn chat(st: &Arc<State<StubEngine>>, b: Vec<u8>) -> Json {
    let n0 = st.persist_done.load(Ordering::SeqCst);
    let r = match handle_request(st.clone(), "POST", "/v1/chat/completions", "", &b, Some("h".into()), None).await {
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

/// A stub server on the store at `dir` with its echo records next to it (as main.rs attaches them).
fn server_at(thinking: bool, scripts: &[&str], dir: &Path) -> Arc<State<StubEngine>> {
    let s = srv(thinking, scripts, Some(dir));
    let (d, id) = {
        let w = s.worker.lock().unwrap();
        let st = w.states.as_ref().unwrap();
        (st.dir.join("echo"), st.ident.text.clone())
    };
    {
        let mut es = s.echo.lock().unwrap();
        es.set_identity(&id);
        es.attach(&d).unwrap();
    }
    s
}

/// The acceptance shape: a multi-turn chat, the engine restarted between every two turns; each turn
/// resumes from disk at the previous turn's full context (prompt + completion - at most 1), thinking
/// off and on.
#[tokio::test(flavor = "multi_thread")]
async fn every_turn_after_a_restart_resumes_at_the_previous_full_context() {
    let long_sys: String = (0..30).map(|i| format!("Rule {i:02}: keep the change small and test it. ")).collect();
    for thinking in [false, true] {
        let dir = tmp(&format!("depth-turns-{thinking}"));
        let replies: Vec<String> = (0..4).map(|i| if thinking { format!("step {i} plan\n</think>\n\nAnswer {i} done.") } else { format!("Answer {i} done.") }).collect();
        let mut msgs = vec![json!({"role":"system","content":long_sys})];
        let mut prev_full: Option<usize> = None;
        for (turn, reply) in replies.iter().enumerate() {
            let s = server_at(thinking, &[reply.as_str()], &dir);
            msgs.push(json!({"role":"user","content":format!("question number {turn}")}));
            let r = chat(&s, body(&Json::Array(msgs.clone()), thinking, true)).await;
            let t = &r["timings"];
            let (c, p, g) = (t["cache_n"].as_u64().unwrap() as usize, t["prompt_n"].as_u64().unwrap() as usize, t["predicted_n"].as_u64().unwrap() as usize);
            if let Some(full) = prev_full {
                assert!(c + 1 >= full && c <= full, "thinking {thinking} turn {turn}: resumed {c}, previous full context {full}");
                assert_eq!(t["resume_src"], "disk", "thinking {thinking} turn {turn}");
            }
            prev_full = Some(c + p + g);
            // the head on disk is the full context (the reply's last token is not evaluated)
            {
                let w = s.worker.lock().unwrap();
                let st = w.states.as_ref().unwrap();
                let deepest = st.manifests.values().flat_map(|m| m.snaps.iter().map(|x| x.pos)).max().unwrap();
                assert_eq!(deepest + 1, c + p + g, "thinking {thinking} turn {turn}: the stored head");
            }
            let mut m = r["choices"][0]["message"].clone();
            m.as_object_mut().unwrap().retain(|k, _| k == "role" || k == "content" || k == "reasoning_content");
            msgs.push(m);
            drop(s); // the restart: a new server on the same store
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---- the tiny synthetic hybrid model -------------------------------------------------------------

fn mini_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("oxe-depth-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cache")).unwrap();
    std::fs::set_permissions(dir.join("cache"), std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    model::testkit::mini_model_ext(&dir.join("mini-mtp.gguf"), true);
    dir
}

fn mini_worker(dir: &Path, depth: usize, store: Option<&Path>) -> Worker<crate::oxe_engine::OxeEngine> {
    let n = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let eng = crate::oxe_engine::OxeEngine::open_cache(&dir.join("mini-mtp.gguf"), None, &[n - 1], None, 64, 2048, depth, Some(dir.join("cache"))).unwrap();
    let mut w = Worker::new(eng, 8);
    w.ubatch = 64;
    w.persist_every = 0;
    if let Some(root) = store {
        let id = crate::state_store::Identity::new("sha256:mini", &w.engine.state_id().unwrap(), 64, &w.engine.arith_id(), "b", false);
        let mut st = StateStore::open(root, id, 1 << 40, 3600, true).unwrap();
        st.min_tokens = 0;
        st.min_free_bytes = 0;
        st.min_mem_bytes = 0;
        w.states = Some(st);
    }
    w
}

fn state_bytes<E: Engine>(e: &mut E) -> Vec<u8> {
    let mut b = Vec::new();
    e.save_state(&mut b).unwrap();
    b
}

/// On real numerics, MTP off and on, thinking off and on (a pre-think point before the reply): a
/// 3-turn session with a restart after every reply. Each turn resumes from disk at prompt +
/// completion - 1 of the turn before, and its tokens, state bytes and draft counters equal the
/// uninterrupted run.
#[test]
fn restarts_resume_at_the_full_context_bitwise_on_the_mini_model() {
    let dir = mini_dir("turns");
    let tok = synth_tokenizer();
    for depth in [0usize, 2] {
        for thinking in [false, true] {
            let root = dir.join(format!("store-{depth}-{thinking}"));
            let mut live = mini_worker(&dir, depth, None);
            let mut prompt: Vec<u32> = (0..400).map(|i| (i * 13 + 7) % 64).collect();
            let mut prev_full = 0usize;
            for turn in 0..3 {
                let mut gp = greedy(8);
                if thinking {
                    gp.pre_think = Some(prompt.len() - 3);
                }
                let mut w = mini_worker(&dir, depth, Some(&root));
                let d = w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
                let ds = state_bytes(&mut w.engine);
                assert!(w.persist(None, PersistMode::Reply, None).head.is_some());
                let l = live.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
                let ls = state_bytes(&mut live.engine);
                if turn > 0 {
                    assert_eq!(d.timings.cache_n, prev_full - 1, "depth {depth} thinking {thinking} turn {turn}: resumed depth");
                    assert_eq!(d.timings.resume_src, "disk");
                    assert_eq!(l.timings.cache_n, d.timings.cache_n);
                }
                assert_eq!(d.tokens, l.tokens, "depth {depth} thinking {thinking} turn {turn}: tokens");
                assert!(ds == ls, "depth {depth} thinking {thinking} turn {turn}: state bytes");
                assert_eq!((d.timings.draft_n, d.timings.draft_n_accepted), (l.timings.draft_n, l.timings.draft_n_accepted));
                prev_full = prompt.len() + d.tokens.len();
                // the next turn: this prompt + the reply (as generated) + a new user message
                prompt.extend(&d.tokens[..d.tokens.len() - 1]);
                prompt.extend((0..7).map(|k| ((turn * 11 + k * 5) % 64) as u32));
                drop(w);
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---- eviction --------------------------------------------------------------------------------------

/// Sessions through a stub worker, each with RAM points at 1/4 and 1/2 of it (optional snapshots).
fn fill(root: &Path, lens: &[(usize, u64)]) -> Worker<StubEngine> {
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![]), 8);
    w.ubatch = 16;
    w.states = Some(open(root, ident("sha256:m", ""), 1 << 40));
    let mut sids = Vec::new();
    for (i, &(n, _)) in lens.iter().enumerate() {
        let t = toks(n, 70 + i);
        w.cache.reset(&mut w.engine);
        for (a, b) in [(0, n / 4), (n / 4, n / 2), (n / 2, n)] {
            w.engine.prefill(&t[a..b]).unwrap();
            w.cache.appended(&t[a..b]);
            if b < n {
                w.cache.checkpoint(&mut w.engine).unwrap();
            }
        }
        w.last_first_user = Some(BLOCK - 10);
        sids.push(w.persist(Some(&format!("e{i}")), PersistMode::Reply, None).sid.unwrap());
    }
    let s = w.states.as_mut().unwrap();
    let now = now_s();
    for (sid, &(_, age)) in sids.iter().zip(lens) {
        let mut m = s.manifests[sid].clone();
        m.atime = now - age;
        s.put_manifest(m).unwrap();
    }
    w
}

/// A deep recent session (unpinned, 20 min) and a shallow old one (3 days): a budget that needs more
/// than the old session's optional points drops the OLD SESSION whole; the deep one keeps its head
/// and every point (v0.1.2 thinned every unpinned session first, the deep one too).
#[test]
fn deep_recent_state_survives_over_shallow_old_state() {
    let root = tmp("depth-evict");
    let mut w = fill(&root, &[(1200, 3 * 86400), (6000, 1200)]);
    let s = w.states.as_mut().unwrap();
    let deep = s.manifests.values().find(|m| m.n_tokens == 6000).unwrap().clone();
    let old = s.manifests.values().find(|m| m.n_tokens == 1200).unwrap().clone();
    assert!(deep.snaps.len() >= 3 && old.snaps.len() >= 3);
    let old_opt: u64 = old.snaps.iter().filter(|x| x.pos != old.n_tokens && x.class != "anchor").map(|x| x.bytes).sum();
    let old_head = old.snaps.iter().find(|x| x.pos == old.n_tokens).unwrap().bytes;
    let used = s.used_bytes();
    // more than the old session's optional points free (a rewritten manifest shrinks too), less
    // than the whole old session
    s.set_budget(used - old_opt - old_head / 2);
    assert!(s.used_bytes() <= s.budget);
    assert!(!s.manifests.contains_key(&old.sid), "the shallow old session goes");
    let d = s.manifests.get(&deep.sid).expect("the deep recent session stays");
    assert_eq!(d.snaps, deep.snaps, "with its head and every point");
    let _ = std::fs::remove_dir_all(&root);
}

/// A small overshoot on one deep session: one optional point goes (the one that saves the least
/// re-prefill: the shallowest), never the head or the session; under a big one the session goes.
#[test]
fn a_small_overshoot_drops_the_cheapest_point_never_the_head() {
    let root = tmp("depth-evict-small");
    let mut w = fill(&root, &[(6000, 1200)]);
    let s = w.states.as_mut().unwrap();
    let m0 = s.manifests.values().next().unwrap().clone();
    let opt: Vec<usize> = m0.snaps.iter().filter(|x| x.pos != m0.n_tokens && x.class != "anchor").map(|x| x.pos).collect();
    assert!(opt.len() >= 2, "{opt:?}");
    s.set_budget(s.used_bytes() - 1);
    let m1 = &s.manifests[&m0.sid];
    assert_eq!(m1.snaps.len(), m0.snaps.len() - 1, "one point");
    assert!(m1.snaps.iter().any(|x| x.pos == m0.n_tokens), "the head stays");
    let gone: Vec<usize> = m0.snaps.iter().map(|x| x.pos).filter(|p| !m1.snaps.iter().any(|y| y.pos == *p)).collect();
    assert_eq!(gone, vec![*opt.iter().min().unwrap()], "the point that saves the least goes");
    s.set_budget(1);
    assert!(s.manifests.is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

// ---- the legacy session cache: read only with the state cache ---------------------------------------

#[test]
fn legacy_session_writes_flag() {
    assert_eq!(legacy_session_writes("auto", true, true), Some(false), "auto: off with the state cache + park write");
    assert_eq!(legacy_session_writes("auto", false, true), Some(true));
    assert_eq!(legacy_session_writes("auto", true, false), Some(true), "no park write: the legacy park stays");
    assert_eq!(legacy_session_writes("on", true, true), Some(true));
    assert_eq!(legacy_session_writes("off", false, true), Some(false));
    assert_eq!(legacy_session_writes("maybe", true, true), None);
}

/// With legacy writes off a session another request displaces is not streamed to the session cache
/// (the park write put it into the state store, at its full depth); a restart resumes it from the
/// state store just as deep. On: the legacy park writes as before.
#[test]
fn legacy_writes_off_parks_into_the_state_store_only() {
    for legacy in [false, true] {
        let dir = tmp(&format!("depth-legacy-{legacy}"));
        let mut w = worker(&dir);
        w.persist_every = 0;
        w.legacy_writes = legacy;
        let ldir = tmp(&format!("depth-legacy-{legacy}-sessions"));
        w.sessions = Some(crate::prefix_store::PrefixStore::open(&ldir, "legacy-id".into(), 1 << 30, 64).unwrap());
        w.cache.park_budget = 1 << 30;
        w.cache.park_min = 8;
        w.cache.spill = legacy;
        w.cache.park_disk_min = 1;
        let a = toks(1500, 6);
        gen(&mut w, &a, &turn_params());
        let a_full = w.cache.tokens.clone();
        // another conversation displaces A (no post-reply write ran for A: the park write does it)
        gen(&mut w, &toks(1300, 7), &greedy(3));
        let ps = w.sessions.as_mut().unwrap();
        ps.flush();
        assert_eq!(ps.n_entries() > 0, legacy, "legacy {legacy}: session cache entries {}", ps.n_entries());
        let st = w.states.as_ref().unwrap();
        assert!(st.manifests.values().any(|m| m.n_tokens == a_full.len()), "A's full depth is in the state store");
        drop(w);
        let mut w2 = worker(&dir);
        let mut q = a_full.clone();
        q.extend([201, 202, 203]);
        let r = gen(&mut w2, &q, &greedy(2));
        assert_eq!((r.timings.cache_n, r.timings.resume_src), (a_full.len(), "disk"));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&ldir);
    }
}
