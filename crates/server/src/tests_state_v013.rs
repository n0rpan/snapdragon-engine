//! v0.1.3 lane B (CACHE-DEEP) tests: model-free (stub engine, synthetic tokenizer, the Qwen3.x
//! template fixture) plus the tiny synthetic hybrid model (`model::testkit`, no real weights) for
//! the bitwise resume checks. Every store and model file lives in its own temp directory; nothing
//! under HOME, no prompt text or token ids written anywhere but those directories.
//!
//! Covered: the pre-think and compaction-cut snapshot classes (stub: where a request resumes;
//! mini model: the resumed state and output are byte-identical to an uninterrupted run, MTP off and
//! on, from RAM and from disk), user-message turn points in the last 16K, up to 12 disk snapshots
//! per session with the gap-aware thinning, the park write (a session being left keeps its resume
//! points on disk), echo file compaction (the splice after a restart is unchanged), the global
//! budget across identities (prefix closure, the cap, thinning first), the off-lock head write (a
//! request runs during it; a sweep in between keeps its files), kills around the off-lock write,
//! old-format files refused, review-carried echo fixes, and mutation checks of the sweep tests.

use crate::cache::{PRIO_CUT, PRIO_PRETHINK, PRIO_TURN};
use crate::echo::{EchoConfig, EchoStore};
use crate::engine::{Engine, StubEngine};
use crate::generate::{GenParams, PersistMode, Worker};
use crate::sampling::SamplingParams;
use crate::service::{handle_policy, handle_request, Config, Reply, State};
use crate::state_store::{thin_max, Plan, StateStore, BLOCK};
use crate::tests_echo::synth_tokenizer;
use crate::tests_state::{files, ident, open, tmp};
use crate::tests_state_p2b::{gen, toks, worker};
use serde_json::{json, Value as Json};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn greedy(n_predict: i64) -> GenParams {
    GenParams { n_predict, sampling: SamplingParams { temperature: 0.0, ..Default::default() }, ..Default::default() }
}

/// Every snapshot of every manifest of the store at `root` (identity `id`) loads: blocks and
/// snapshot on disk, checksums right. Returns how many plans were loaded.
fn consistent(root: &Path, id: crate::state_store::Identity) -> usize {
    let mut s = open(root, id, 1 << 40);
    let plans: Vec<Plan> = s.manifests.values().flat_map(|m| m.snaps.iter().map(move |x| Plan { sid: m.sid.clone(), blocks: m.blocks[..x.pos / BLOCK].to_vec(), snap: x.clone() })).collect();
    let mut e = StubEngine::new(1 << 16, 300, 0, vec![]);
    for p in &plans {
        s.load(&mut e, p).unwrap_or_else(|err| panic!("manifest {} snapshot at {}: {err}", p.sid, p.snap.pos));
    }
    assert_eq!(s.stats.load_errors, 0);
    plans.len()
}

/// A stub server over the synthetic tokenizer, thinking on or off, 8 checkpoint slots, ubatch 16,
/// an optional state store.
pub(crate) fn srv(thinking: bool, scripts: &[&str], store: Option<&Path>) -> Arc<State<StubEngine>> {
    let tok = synth_tokenizer();
    let mut eng = StubEngine::new(1 << 16, tok.n_vocab(), tok.eos.unwrap(), tok.encode("x", false, true));
    for s in scripts {
        eng.scripts.push_back(tok.encode(s, false, true));
    }
    let cfg = Config {
        model_name: "synth".into(),
        model_path: "/synth.gguf".into(),
        chat_opts: chat::ServerChatOptions { enable_thinking: thinking, ..Default::default() },
        fingerprint: "oxe-test".into(),
        slots: false,
        idle_reprefill: false,
        engine_status: None,
    };
    let mut w = Worker::new(eng, 8);
    w.ubatch = 16;
    if let Some(d) = store {
        w.states = Some(open(d, ident("sha256:m", ""), 1 << 40));
    }
    Arc::new(State::new(cfg, tok, w).unwrap())
}

pub(crate) fn body(msgs: &Json, thinking: bool, preserve: bool) -> Vec<u8> {
    serde_json::to_vec(&json!({"messages": msgs, "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": preserve}})).unwrap()
}

/// One chat request (hint "h", optional turn kind); waits for the post-reply state write when the
/// server has a state store.
async fn chat(st: &Arc<State<StubEngine>>, b: Vec<u8>, kind: Option<&str>) -> Json {
    let n0 = st.persist_done.load(Ordering::SeqCst);
    let r = match handle_request(st.clone(), "POST", "/v1/chat/completions", "", &b, Some("h".into()), kind.map(str::to_string)).await {
        Reply::Json(c, j) => {
            assert_eq!(c, 200, "{j}");
            j
        }
        Reply::Stream(_) => panic!("stream"),
    };
    if st.state_cache {
        let t0 = Instant::now();
        while st.persist_done.load(Ordering::SeqCst) == n0 {
            assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }
    r
}

fn kv(st: &State<StubEngine>) -> Vec<u32> {
    st.worker.lock().unwrap().engine.tokens.clone()
}

const SYS: &str = "You are a test system. Keep answers short.";

/// Item 1: thinking on, and the next request's history drops this turn's reasoning (no think
/// block: preserve_thinking false, echo off): it resumes exactly at the pre-think point, i.e.
/// right after `<|im_start|>assistant\n`, not at the grid point below it. Off (0 slots): it does not.
#[tokio::test(flavor = "multi_thread")]
async fn pre_think_checkpoint_serves_a_turn_whose_reasoning_was_stripped() {
    for slots in [2usize, 0] {
        let s = srv(true, &["plan it\n</think>\n\nDone", "ok"], None);
        s.echo.lock().unwrap().cfg.enabled = false;
        s.worker.lock().unwrap().cache.max_pre = slots;
        let m1 = json!([{"role":"system","content":SYS},{"role":"user","content":"do"}]);
        let r1 = chat(&s, body(&m1, true, false), None).await;
        assert_eq!(r1["choices"][0]["message"]["content"], "Done");
        let n1 = (r1["timings"]["cache_n"].as_u64().unwrap() + r1["timings"]["prompt_n"].as_u64().unwrap()) as usize;
        let think = s.tok.encode("<think>\n", false, true).len();
        let pt = n1 - think;
        if slots > 0 {
            let info = s.worker.lock().unwrap().cache.snapshot_info();
            assert!(info.contains(&(pt, PRIO_PRETHINK)), "{info:?}");
        }
        let m2 = json!([{"role":"system","content":SYS},{"role":"user","content":"do"},{"role":"assistant","content":"Done"},{"role":"user","content":"next"}]);
        let r2 = chat(&s, body(&m2, true, false), None).await;
        let c2 = r2["timings"]["cache_n"].as_u64().unwrap() as usize;
        if slots > 0 {
            assert_eq!(c2, pt, "resumed at the pre-think point");
        } else {
            assert!(c2 < pt, "without the class the request re-prefills from below it: {c2} vs {pt}");
        }
        // the fed prompt is the plain render (uninterrupted run = a cold server fed the same request)
        let cold = srv(true, &["ok"], None);
        cold.echo.lock().unwrap().cfg.enabled = false;
        chat(&cold, body(&m2, true, false), None).await;
        let (a, b) = (kv(&s), kv(&cold));
        let n2 = c2 + r2["timings"]["prompt_n"].as_u64().unwrap() as usize;
        assert_eq!(a[..n2], b[..n2], "the resumed request fed exactly the cold request's prompt");
    }
}

/// Thinking off: no pre-think point is taken (the class is for reasoning turns only).
#[tokio::test(flavor = "multi_thread")]
async fn no_pre_think_point_with_thinking_off() {
    let s = srv(false, &["Hello"], None);
    let m1 = json!([{"role":"system","content":SYS},{"role":"user","content":"hi"}]);
    chat(&s, body(&m1, false, true), None).await;
    assert!(!s.worker.lock().unwrap().cache.snapshot_info().iter().any(|x| x.1 == PRIO_PRETHINK));
}

/// Item 2: a compaction request (X-Oxe-Turn-Kind: compaction) takes a checkpoint exactly at its cut
/// (its instruction message's start), persisted as class "cut"; after another conversation took
/// the engine, a retry of it re-prefills nothing before the cut (from disk). Without the label no
/// cut point exists.
#[tokio::test(flavor = "multi_thread")]
async fn compaction_request_reprefills_nothing_before_the_cut() {
    let long_sys: String = (0..30).map(|i| format!("Rule {i:02}: keep the change small and test it. ")).collect();
    let instr = "Summarize the conversation above into a structured summary.";
    for labelled in [true, false] {
        let dir = tmp(&format!("v013-cut-{labelled}"));
        let s = srv(false, &["Answer one with some words.", "Summary of it.", "Other.", "Summary again."], Some(&dir));
        let m1 = json!([{"role":"system","content":long_sys},{"role":"user","content":"question one"}]);
        chat(&s, body(&m1, false, true), None).await;
        let mc = json!([{"role":"system","content":long_sys},{"role":"user","content":"question one"},{"role":"assistant","content":"Answer one with some words."},{"role":"user","content":instr}]);
        let kind = labelled.then_some("compaction");
        let rc = chat(&s, body(&mc, false, true), kind).await;
        // the cut: the start of the instruction message in the fed prompt
        let cut = {
            let w = s.worker.lock().unwrap();
            let d = s.user_delim.clone().unwrap();
            w.engine.tokens.windows(d.len()).rposition(|x| x == &d[..]).unwrap()
        };
        assert!(rc["timings"]["cache_n"].as_u64().unwrap() as usize <= cut);
        let has_cut = s.worker.lock().unwrap().cache.snapshot_info().contains(&(cut, PRIO_CUT));
        assert_eq!(has_cut, labelled, "a cut checkpoint exactly at the cut only for a labelled compaction");
        {
            let w = s.worker.lock().unwrap();
            let st = w.states.as_ref().unwrap();
            let on_disk = st.manifests.values().any(|m| m.snaps.iter().any(|x| x.class == "cut" && x.pos == cut));
            assert_eq!(on_disk, labelled, "the cut is persisted as class cut");
        }
        // another conversation takes the engine (its RAM checkpoints go)
        let mo = json!([{"role":"system","content":"Another agent entirely, with other rules."},{"role":"user","content":"x"}]);
        chat(&s, body(&mo, false, true), None).await;
        // the compaction is retried: nothing before the cut is prefilled again
        let r = chat(&s, body(&mc, false, true), kind).await;
        let c = r["timings"]["cache_n"].as_u64().unwrap() as usize;
        assert!(c >= cut, "labelled={labelled}: resumed at {c}, cut at {cut}");
        if labelled {
            assert_eq!(r["timings"]["resume_src"], "disk");
        }
        assert!(consistent(&dir, ident("sha256:m", "")) > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Item 2: user-message starts in the last 16K come first among the turn checkpoints (bounded by
/// the turn slots); window 0 keeps the old rule (the newest message starts).
#[test]
fn user_starts_in_the_window_come_first_among_turn_points() {
    let tok = synth_tokenizer();
    let prompt: Vec<u32> = (0..1000).map(|k| (k % 200) as u32 + 1).collect();
    let mut gp = greedy(1);
    gp.first_user = Some(20);
    gp.msg_starts = vec![0, 20, 100, 200, 300, 400, 500, 600, 700, 800];
    gp.user_starts = vec![20, 300, 600];
    for window in [16 << 10, 0] {
        let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![7]), 8);
        w.ubatch = 16;
        w.user_window = window;
        w.generate(&tok, &prompt, &gp, &mut |_| true).unwrap();
        let turns: Vec<usize> = w.cache.snapshot_info().iter().filter(|x| x.1 == PRIO_TURN).map(|x| x.0).collect();
        if window > 0 {
            assert!(turns.contains(&288) && turns.contains(&592), "the user starts' grid floors: {turns:?}");
            assert_eq!(turns.len(), 4);
        } else {
            assert_eq!(turns, vec![496, 592, 688, 800], "newest message starts");
        }
    }
}

/// Item 3: up to `max` disk snapshots; the gap-aware thinning keeps the anchor, the head, the two
/// newest points, and the worst gap (the re-prefill after an old edit) near 32K tokens at 128K and
/// 256K contexts.
#[test]
fn thinning_bounds_the_worst_re_prefill() {
    for (n, step) in [(131_072usize, 1024usize), (131_072, 256), (262_144, 1024), (65_536, 512), (40_000, 300)] {
        let mut pos: Vec<(usize, bool)> = (1..n / step).map(|i| (i * step, false)).collect();
        pos.push((700, true));
        pos.push((n, true));
        let k = thin_max(pos, n, 12);
        assert!(k.len() <= 12, "{n}: {}", k.len());
        assert!(k.contains(&700) && k.contains(&n));
        let newest = (n / step - 1) * step;
        assert!(k.contains(&newest) && k.contains(&(newest - step)), "{n}: the two newest points stay");
        let mut v = k.clone();
        v.push(0);
        v.sort_unstable();
        let gap = v.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        assert!(gap <= (32 << 10) + step, "{n}/{step}: worst gap {gap}");
    }
    // a smaller cap is honoured; must-keep points survive any cap
    let pos: Vec<(usize, bool)> = (1..200).map(|i| (i * 500, i == 3 || i == 199)).collect();
    let k = thin_max(pos, 99_500, 4);
    assert!(k.len() <= 4 && k.contains(&1500) && k.contains(&99_500), "{k:?}");
}

/// A long session with RAM turn points (no post-reply write ran: a request came first).
fn session_with_points(w: &mut Worker<StubEngine>, seed: usize) -> Vec<u32> {
    let prompt = toks(1500, seed);
    let mut gp = greedy(3);
    gp.first_user = Some(BLOCK - 10);
    gp.msg_starts = vec![0, BLOCK - 10, 600, 900, 1200];
    gp.user_starts = vec![BLOCK - 10, 600, 1200];
    gp.session = Some("a".into());
    gen(w, &prompt, &gp);
    prompt
}

/// Item 3: a session being left (the next request does not continue it) is written to the state
/// store first with its RAM checkpoints; after a restart an edit at an older user turn resumes at
/// that turn point from disk. With `--state-park-write off` nothing is written.
#[test]
fn a_session_being_left_keeps_its_resume_points_on_disk() {
    for on in [true, false] {
        let dir = tmp(&format!("v013-park-{on}"));
        let mut w = worker(&dir);
        w.persist_every = 0;
        w.park_write = on;
        let a = session_with_points(&mut w, 1);
        // another conversation (shares only the static first block)
        gen(&mut w, &toks(1400, 9), &greedy(3));
        let s = w.states.as_ref().unwrap();
        let m = s.manifests.values().find(|m| m.n_tokens >= a.len());
        if !on {
            assert!(m.is_none(), "park write off: the left session is not written");
            let _ = std::fs::remove_dir_all(&dir);
            continue;
        }
        let m = m.expect("the left session is on disk");
        assert!(m.snaps.len() >= 4, "head + its turn points: {:?}", m.snaps.iter().map(|x| (x.pos, x.class.clone())).collect::<Vec<_>>());
        drop(w);
        // restart, then an edit after the user turn at 900: resumes at its grid floor (896)
        let mut w2 = worker(&dir);
        let mut q = a[..1000].to_vec();
        q.extend((0..50).map(|k| (k % 40) as u32 + 200));
        let r = gen(&mut w2, &q, &greedy(2));
        assert_eq!(r.timings.cache_n, 896, "resumed at the turn point below the edit");
        assert_eq!(r.timings.resume_src, "disk");
        assert!(consistent(&dir, ident("sha256:m", "")) > 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Item 4: echo files are compacted: a retry loop (the same turn again and again) leaves one line,
/// and a store that reads the compacted file back splices exactly the ids the first one did.
#[test]
fn echo_compaction_keeps_the_splice() {
    let dir = tmp("v013-echo-compact");
    let cfg = EchoConfig { compact_bytes: 2048, ..Default::default() };
    let mut es = EchoStore::new(cfg.clone());
    es.set_identity("id-1");
    es.attach(&dir).unwrap();
    let prompt: Vec<u32> = vec![1, 2, 3, 10, 11, 13];
    let msg = chat::ChatMsg { content: "Done".into(), reasoning_content: "plan".into(), ..Default::default() };
    for _ in 0..200 {
        assert!(es.record(&prompt, &[10, 11], 12, &msg, &[20, 21, 22, 12], true, Some("s1")));
    }
    assert!(es.compactions > 0, "the file was compacted");
    let f = files(&dir, ".jsonl");
    assert_eq!(f.len(), 1);
    let size = std::fs::metadata(&f[0]).unwrap().len();
    assert!(size < 2 * 2048, "bounded: {size} bytes");
    let text = std::fs::read_to_string(&f[0]).unwrap();
    assert!(text.lines().count() < 12, "only live lines (+ appends since the last rewrite)");
    let rendered: Vec<u32> = vec![1, 2, 3, 10, 11, 99, 98, 12, 5, 6];
    let msgs = json!([{"role":"user","content":"q"},{"role":"assistant","content":"Done"}]);
    let (a, ra) = es.splice(&rendered, &msgs, &[None, None], Some(&[(3, 7)]));
    assert_eq!(ra.hits, 1);
    assert_eq!(a, vec![1, 2, 3, 10, 11, 13, 20, 21, 22, 12, 5, 6]);
    let mut es2 = EchoStore::new(cfg);
    es2.set_identity("id-1");
    assert!(es2.attach(&dir).unwrap() >= 1);
    let (b, rb) = es2.splice(&rendered, &msgs, &[None, None], Some(&[(3, 7)]));
    assert_eq!((b, rb.hits), (a, 1), "the splice after a restart is the one before");
    // distinct live records are never compacted away
    let before = es2.compactions;
    for i in 0..20u32 {
        let m = chat::ChatMsg { content: format!("Done {i}"), ..Default::default() };
        es2.record(&prompt, &[10, 11], 12, &m, &[20 + i, 12], true, Some("s1"));
    }
    let n_lines = std::fs::read_to_string(&f[0]).unwrap().lines().count();
    assert!(n_lines >= 21 || es2.compactions > before, "live records stay on disk");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Sessions straight through a worker of identity `id` under `root`; each with a mid-session RAM
/// checkpoint (an optional snapshot on disk) and an old atime (not pinned). Returns the worker.
fn fill_id(root: &Path, id: crate::state_store::Identity, lens: &[usize], seed: usize, age_s: u64) -> Worker<StubEngine> {
    let mut w = Worker::new(StubEngine::new(1 << 16, 300, 0, vec![]), 8);
    w.ubatch = 16;
    w.states = Some(open(root, id, 1 << 40));
    for (i, &n) in lens.iter().enumerate() {
        let t = toks(n, seed * 31 + i);
        w.cache.reset(&mut w.engine);
        w.engine.prefill(&t[..n / 2]).unwrap();
        w.cache.appended(&t[..n / 2]);
        w.cache.checkpoint(&mut w.engine).unwrap();
        w.engine.prefill(&t[n / 2..]).unwrap();
        w.cache.appended(&t[n / 2..]);
        w.last_first_user = Some(BLOCK - 10);
        assert!(w.persist(Some(&format!("s{seed}-{i}")), PersistMode::Reply, None).head.is_some());
    }
    let s = w.states.as_mut().unwrap();
    let now = crate::state_store::now_s();
    for sid in s.manifests.keys().cloned().collect::<Vec<_>>() {
        let mut m = s.manifests[&sid].clone();
        m.atime = now - age_s;
        s.put_manifest(m).unwrap();
    }
    w
}

/// Every manifest of every identity under `root` has all its blocks and snapshots on disk.
fn closure_holds(root: &Path) -> usize {
    let mut n = 0;
    for d in std::fs::read_dir(root).unwrap().flatten().filter(|d| d.path().is_dir()) {
        for m in files(&d.path().join("sessions"), ".json") {
            let j: Json = serde_json::from_slice(&std::fs::read(&m).unwrap()).unwrap();
            for b in j["blocks"].as_array().unwrap() {
                let (h, c) = (b[0].as_str().unwrap(), b[1].as_str().unwrap());
                assert!(d.path().join("blocks").join(&h[..2]).join(format!("{h}.{c}.kv")).is_file(), "a kept session's block is missing");
            }
            for s in j["snaps"].as_array().unwrap() {
                let (k, c) = (s["key"].as_str().unwrap(), s["c"].as_str().unwrap());
                assert!(d.path().join("snaps").join(format!("{k}.{c}.st")).is_file(), "a kept snapshot is missing");
            }
            n += 1;
        }
    }
    n
}

/// Item 5: one budget across identities. The loaded engine (A) evicts B's sessions when they are
/// worth least (older), optional snapshots before whole sessions; every kept session keeps its
/// blocks (prefix closure) and the bytes end under the cap. Scope "own" never touches B.
#[test]
fn the_budget_evicts_across_identities_keeping_prefix_closure_and_the_cap() {
    for global in [true, false] {
        let root = tmp(&format!("v013-global-{global}"));
        let ida = ident("sha256:mA", "");
        let idb = ident("sha256:mB", "");
        let wb = fill_id(&root, idb.clone(), &[1200, 1300, 1400], 2, 300 * 3600);
        drop(wb);
        let mut wa = fill_id(&root, ida.clone(), &[1200, 1300], 1, 2 * 3600);
        let s = wa.states.as_mut().unwrap();
        s.global_budget = global;
        let b_dir = root.join(idb.model_key());
        let b_bytes0 = crate::state_store::dir_bytes(&b_dir);
        let used = s.used_bytes();
        let target = used - b_bytes0 / 2;
        s.set_budget(target);
        let b_bytes1 = crate::state_store::dir_bytes(&b_dir);
        if global {
            assert!(s.used_bytes() <= target, "under the cap: {} > {target}", s.used_bytes());
            assert!(b_bytes1 < b_bytes0, "B (older, worth least) gave bytes");
            assert_eq!(s.manifests.len(), 2, "A's newer sessions are kept while B pays");
        } else {
            assert_eq!(b_bytes1, b_bytes0, "scope own: B untouched");
        }
        closure_holds(&root);
        assert!(consistent(&root, ida.clone()) > 0 || !global);
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// Item 5: under a small overshoot only optional snapshots go: the session stays with its head.
#[test]
fn the_budget_thins_optional_snapshots_before_whole_sessions() {
    let root = tmp("v013-thin-first");
    let mut w = fill_id(&root, ident("sha256:m", ""), &[1500], 3, 3600);
    let s = w.states.as_mut().unwrap();
    let m0 = s.manifests.values().next().unwrap().clone();
    assert!(m0.snaps.len() >= 2, "head + an optional point: {:?}", m0.snaps.iter().map(|x| x.pos).collect::<Vec<_>>());
    let used = s.used_bytes();
    s.set_budget(used - 1);
    let m1 = s.manifests.get(&m0.sid).expect("the session itself stays");
    // store-depth: one value order; an optional point goes (as many as the overshoot needs), the head stays
    assert!(m1.snaps.len() < m0.snaps.len(), "an optional point went");
    assert!(m1.snaps.iter().any(|x| x.pos == m0.n_tokens), "the head stays");
    closure_holds(&root);
    let _ = std::fs::remove_dir_all(&root);
}

/// Item 6: the post-reply head write runs off the worker lock. With a slow disk (the write sleeps
/// 1.5 s first) a request that arrives during the write answers at once; afterwards both turns are
/// on disk and the store is consistent.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_does_not_wait_for_the_head_write() {
    let dir = tmp("v013-offlock");
    let s = srv(false, &["First answer here.", "Second answer."], Some(&dir));
    let inflight = {
        let mut w = s.worker.lock().unwrap();
        let st = w.states.as_mut().unwrap();
        st.stage_delay_ms = 1500;
        st.inflight.clone()
    };
    let long_sys: String = (0..30).map(|i| format!("Rule {i:02}: keep the change small and test it. ")).collect();
    let m1 = json!([{"role":"system","content":long_sys},{"role":"user","content":"one"}]);
    let n0 = s.persist_done.load(Ordering::SeqCst);
    match handle_request(s.clone(), "POST", "/v1/chat/completions", "", &body(&m1, false, true), Some("h".into()), None).await {
        Reply::Json(200, _) => {}
        _ => panic!("turn 1"),
    }
    let t0 = Instant::now();
    while inflight.lock().unwrap().is_empty() {
        assert!(t0.elapsed().as_secs() < 10, "the off-lock write did not start");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let m2 = json!([{"role":"system","content":long_sys},{"role":"user","content":"one"},{"role":"assistant","content":"First answer here."},{"role":"user","content":"two"}]);
    let t1 = Instant::now();
    match handle_request(s.clone(), "POST", "/v1/chat/completions", "", &body(&m2, false, true), Some("h".into()), None).await {
        Reply::Json(200, j) => assert_eq!(j["choices"][0]["message"]["content"], "Second answer."),
        _ => panic!("turn 2"),
    }
    let waited = t1.elapsed();
    assert!(waited < Duration::from_millis(1000), "the request waited {waited:?} for the head write");
    while s.persist_done.load(Ordering::SeqCst) < n0 + 2 {
        assert!(t0.elapsed().as_secs() < 30, "the writes did not finish");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(inflight.lock().unwrap().is_empty(), "every staged file released");
    let n_kv = kv(&s).len();
    {
        let w = s.worker.lock().unwrap();
        let st = w.states.as_ref().unwrap();
        assert!(st.manifests.values().any(|m| m.snaps.iter().any(|x| x.class == "head" && x.pos == n_kv)), "turn 2's head is on disk");
    }
    assert!(files(&dir, ".pending").is_empty());
    assert!(consistent(&dir, ident("sha256:m", "")) > 0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The off-lock write with a sweep between its file writes and its commit (another request's
/// eviction pass): the staged files survive and the committed manifest loads. `mutant` 3 = sweeps
/// ignore the in-flight list (must break it: mutation check).
fn offlock_sweep_scenario(mutant: u8) -> Result<(), String> {
    let dir = tmp(&format!("v013-offlock-sweep-{mutant}"));
    let mut w = worker(&dir);
    w.persist_every = 0;
    let p = toks(1300, 4);
    gen(&mut w, &p, &greedy(3));
    let s = w.stage_persist(Some("h")).ok_or("nothing staged")?;
    let bytes = s.write();
    {
        let st = w.states.as_mut().unwrap();
        st.mutant = mutant;
        st.sweep();
        st.mutant = 0;
    }
    // a request ran meanwhile (the live state is not the staged one): the commit names the staged
    // files and repairs nothing, so a file a sweep took shows
    w.cache.epoch += 1;
    let r = w.commit_staged(s, bytes, Some("h"), None);
    let out = (|| {
        if r.head.is_none() {
            return Err(format!("not committed: {r:?}"));
        }
        let st = w.states.as_ref().unwrap();
        for m in st.manifests.values() {
            for x in &m.snaps {
                if !st.has_snap(&x.key, &x.c) {
                    return Err(format!("snapshot at {} missing", x.pos));
                }
            }
            for (h, c) in &m.blocks {
                if files(&dir, &format!("{h}.{c}.kv")).is_empty() {
                    return Err("block missing".into());
                }
            }
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn a_sweep_during_an_off_lock_write_keeps_its_files_mutation_checked() {
    offlock_sweep_scenario(0).unwrap();
    assert!(offlock_sweep_scenario(3).is_err(), "a sweep that ignores the in-flight list must fail this check");
}

/// Kills around the off-lock write: after its files are written but before the commit (no
/// manifest names them), and in the middle of a file (a torn `.pending`). A restart sweeps them,
/// the previous turn loads, and the next request answers on the right state.
#[test]
fn kills_around_the_off_lock_write_leave_a_consistent_store() {
    for torn in [false, true] {
        let dir = tmp(&format!("v013-offlock-kill-{torn}"));
        let mut w = worker(&dir);
        w.persist_every = 0;
        let t1 = toks(900, 5);
        gen(&mut w, &t1, &greedy(3));
        assert!(w.persist(Some("k"), PersistMode::Reply, None).head.is_some());
        let mut t2 = w.cache.tokens.clone();
        t2.extend(toks(1300, 6)[BLOCK..].iter().take(400));
        gen(&mut w, &t2, &greedy(3));
        let s = w.stage_persist(Some("k")).expect("staged");
        if torn {
            // killed inside the head file: half a `.pending` stays
            let p = s.dir.join("snaps").join(format!("{}.s.st.pending", s.head_key));
            std::fs::write(&p, &s.head_payload[..s.head_payload.len() / 2]).unwrap();
        } else {
            s.write().unwrap();
        }
        drop(s);
        drop(w);
        // restart: start-up sweeps; the turn-1 manifest loads
        assert!(consistent(&dir, ident("sha256:m", "")) > 0);
        assert!(files(&dir, ".pending").is_empty(), "start-up swept the .pending files");
        let st = open(&dir, ident("sha256:m", ""), 1 << 40);
        let named: usize = st.manifests.values().map(|m| m.snaps.len()).sum();
        assert_eq!(files(&dir, ".st").len(), named, "no snapshot file without a manifest entry");
        drop(st);
        let mut w2 = worker(&dir);
        let mut q = t2.clone();
        q.extend([40, 41, 42]);
        let r = gen(&mut w2, &q, &greedy(2));
        assert!(r.timings.cache_n >= t1.len(), "resumed at least from turn 1: {}", r.timings.cache_n);
        assert_eq!(w2.engine.tokens[..q.len()], q[..], "the request answered on the right state");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Old formats are refused cleanly, never mis-read: a manifest of another version is removed at
/// open, a snapshot file of format 1 fails its load (the request answers from below it), an echo
/// line of another version is skipped.
#[test]
fn old_format_files_are_refused_cleanly() {
    let dir = tmp("v013-old-format");
    let mut w = worker(&dir);
    w.persist_every = 0;
    let t = toks(1100, 7);
    gen(&mut w, &t, &greedy(3));
    assert!(w.persist(Some("o"), PersistMode::Reply, None).head.is_some());
    let mdir = w.states.as_ref().unwrap().dir.clone();
    drop(w);
    // snapshot files of format 1 (the version field after the magic)
    for f in files(&mdir.join("snaps"), ".st") {
        let mut b = std::fs::read(&f).unwrap();
        b[8..12].copy_from_slice(&1u32.to_le_bytes());
        std::fs::write(&f, b).unwrap();
    }
    // a manifest of a future version next to the real one
    let fake = mdir.join("sessions").join(format!("{}.json", "ab".repeat(16)));
    std::fs::write(&fake, json!({"version": 99, "sid": "ab".repeat(16)}).to_string()).unwrap();
    let mut w2 = worker(&dir);
    assert!(!fake.exists(), "a manifest of another version is removed unread");
    let mut q = t.clone();
    q.extend([9, 9, 9]);
    let r = gen(&mut w2, &q, &greedy(2));
    assert_eq!(r.timings.cache_n, 0, "the old-format snapshot was refused: cold, never a wrong state");
    assert_eq!(w2.engine.tokens[..q.len()], q[..]);
    assert!(w2.states.as_ref().unwrap().stats.load_errors > 0);
    // an echo line of another version is skipped
    let edir = tmp("v013-old-echo");
    std::fs::write(edir.join("x.jsonl"), "{\"v\":2,\"id\":\"z\"}\t0000000000000000\n").unwrap();
    let mut es = EchoStore::new(EchoConfig::default());
    es.set_identity("id");
    assert_eq!(es.attach(&edir).unwrap(), 0);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&edir);
}

/// The P2 sweep tests, mutation-checked (P2R-MUT): the scenario of
/// `replaced_head_snapshots_are_swept_and_missing_live_blocks_are_rewritten` as a check that
/// returns its first failure. `mutant` 1 = no sweep on every enforce, 2 = a swept live block counts
/// as on disk. A broken sweep must fail it.
fn orphan_scenario(mutant: u8) -> Result<(), String> {
    let dir = tmp(&format!("v013-mut-{mutant}"));
    let mut w = crate::tests_state::fill(&dir, 1 << 40, &[1100]);
    w.states.as_mut().unwrap().mutant = mutant;
    let out = (|| {
        for r in 0..(crate::state_store::MAX_SNAPS as u32 + 6) {
            let more: Vec<u32> = (0..40).map(|k| (k + r) % 200 + 3).collect();
            w.engine.prefill(&more).unwrap();
            w.cache.appended(&more);
            w.persist_states(Some("s0"), None);
        }
        let listed: usize = w.states.as_ref().unwrap().manifests.values().map(|m| m.snaps.len()).sum();
        if files(&dir, ".st").len() != listed {
            return Err(format!("{} snapshot files, {listed} listed", files(&dir, ".st").len()));
        }
        let gone = w.live_blocks[1].0.clone();
        let victim = files(&dir, ".kv").into_iter().find(|p| p.to_string_lossy().contains(&gone)).unwrap();
        std::fs::remove_file(victim).unwrap();
        let more: Vec<u32> = (0..300).map(|k| k % 200 + 5).collect();
        w.engine.prefill(&more).unwrap();
        w.cache.appended(&more);
        w.persist_states(Some("s0"), None);
        let s = w.states.as_ref().unwrap();
        let m = s.manifests.values().max_by_key(|m| m.n_tokens).unwrap();
        if m.blocks.iter().any(|(h, c)| files(&dir, &format!("{h}.{c}.kv")).is_empty()) {
            return Err("a listed block is missing".into());
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn the_sweep_tests_fail_on_a_broken_sweep() {
    orphan_scenario(0).unwrap();
    assert!(orphan_scenario(1).is_err(), "no sweep on every enforce must be caught");
    assert!(orphan_scenario(2).is_err(), "a swept live block counted as on disk must be caught");
}

/// Review-carried: tool arguments whose numbers do not survive a parse exactly are compared as
/// text (an integer past 64 bits that differs in its last digit is a miss, never a false hit);
/// small decimals still compare as values; reasoning under another field name must match.
#[test]
fn echo_compares_huge_numbers_exactly_and_reasoning_under_other_names() {
    let prompt: Vec<u32> = vec![1, 2, 3, 10, 11, 13];
    let rendered: Vec<u32> = vec![1, 2, 3, 10, 11, 99, 98, 12, 5, 6];
    let call = |args: &str| chat::ToolCall { name: "f".into(), arguments: args.into(), id: "c".into() };
    let splice_with = |rec: &chat::ChatMsg, sent: Json| {
        let mut es = EchoStore::new(EchoConfig::default());
        es.set_identity("id");
        assert!(es.record(&prompt, &[10, 11], 12, rec, &[20, 21, 12], true, None));
        let msgs = json!([{"role":"user","content":"q"}, sent]);
        es.splice(&rendered, &msgs, &[None, None], Some(&[(3, 7)])).1
    };
    let big = chat::ChatMsg { tool_calls: vec![call(r#"{"n": 123456789012345678901234567890}"#)], ..Default::default() };
    let tc = |a: &str| json!({"role":"assistant","content":"","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":a}}]});
    assert_eq!(splice_with(&big, tc(r#"{"n": 123456789012345678901234567890}"#)).hits, 1, "the same text hits");
    let r = splice_with(&big, tc(r#"{"n": 123456789012345678901234567891}"#));
    assert_eq!((r.hits, r.miss_class()), (0, Some(crate::echo::MISS_TOOL)), "another huge number is a miss");
    let small = chat::ChatMsg { tool_calls: vec![call(r#"{"x": 2.50, "y": 7}"#)], ..Default::default() };
    assert_eq!(splice_with(&small, tc(r#"{"y":7,"x":2.5}"#)).hits, 1, "re-serialised small numbers still hit");
    let rs = chat::ChatMsg { content: "Done".into(), reasoning_content: "plan".into(), ..Default::default() };
    assert_eq!(splice_with(&rs, json!({"role":"assistant","content":"Done","reasoning":"plan"})).hits, 1);
    let r = splice_with(&rs, json!({"role":"assistant","content":"Done","reasoning":"other"}));
    assert_eq!((r.hits, r.miss_class()), (0, Some(crate::echo::MISS_REASONING)));
    let r = splice_with(&rs, json!({"role":"assistant","content":"Done","thinking":{"text":"plan"}}));
    assert_eq!(r.hits, 0, "a reasoning object cannot be compared: a miss");
}

// ---- the tiny synthetic hybrid model: bitwise resume -------------------------------------------

fn mini_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("oxe-v013-{tag}-{}", std::process::id()));
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

/// (generated tokens, state bytes after, (drafted, accepted))
fn run_mini(w: &mut Worker<crate::oxe_engine::OxeEngine>, prompt: &[u32], gp: &GenParams) -> (Vec<u32>, Vec<u8>, (u64, u64), usize) {
    let tok = synth_tokenizer();
    let r = w.generate(&tok, prompt, gp, &mut |_| true).unwrap();
    (r.tokens.clone(), state_bytes(&mut w.engine), (r.timings.draft_n, r.timings.draft_n_accepted), r.timings.cache_n)
}

/// Items 1 and 2 on real numerics (MTP off and on): a request that diverges exactly at a pre-think
/// or a cut point resumes there from RAM, and (cut) from disk after a restart; its generated
/// tokens, its state bytes (incl. the MTP driver) and its draft counters equal a run that never
/// cached anything but splits its prefill at the same point (same ubatch boundaries). And an
/// echo-shaped continuation (prompt + the generated ids) resumed from the disk head after a
/// restart equals the uninterrupted continuation (the MTP echo case).
#[test]
fn new_snapshot_classes_resume_bitwise_on_the_mini_model() {
    let dir = mini_dir("classes");
    let p1: Vec<u32> = (0..400).map(|i| (i * 13 + 7) % 64).collect();
    // diverges at 314: past the first block (a persisted class), inside a 64-token ubatch; a short
    // tail (up to 320, below the next lattice point 384) is one engine call in every run
    const AT: usize = 314;
    let mut p2: Vec<u32> = p1[..AT].to_vec();
    p2.extend([(p1[AT] + 1) % 64, 3, 9, 27, 17, 51, 40]);
    for depth in [0usize, 2] {
        for class in ["pre-think", "cut"] {
            let root = dir.join(format!("store-{depth}-{class}"));
            let mut gp1 = greedy(8);
            match class {
                "pre-think" => gp1.pre_think = Some(AT),
                _ => {
                    gp1.turn_kind = Some("compaction".into());
                    gp1.user_starts = vec![AT];
                }
            }
            let mut w1 = mini_worker(&dir, depth, Some(&root));
            run_mini(&mut w1, &p1, &gp1);
            assert!(w1.persist(None, PersistMode::Reply, None).head.is_some());
            let got = run_mini(&mut w1, &p2, &greedy(8));
            assert_eq!(got.3, AT, "depth {depth} {class}: resumed at the point (RAM)");
            // the reference: a cold run with an exact split at AT
            let mut gpr = greedy(8);
            gpr.pre_think = Some(AT);
            let mut w0 = mini_worker(&dir, depth, None);
            let want = run_mini(&mut w0, &p2, &gpr);
            assert_eq!(want.3, 0);
            assert_eq!(got.0, want.0, "depth {depth} {class}: tokens");
            assert!(got.1 == want.1, "depth {depth} {class}: state bytes differ");
            assert_eq!(got.2, want.2, "depth {depth} {class}: drafted / accepted");
            if depth > 0 {
                assert!(want.2 .0 > 0, "MTP drafted");
            }
            if class == "cut" {
                // from disk after a restart: the cut is persisted
                let mut w3 = mini_worker(&dir, depth, Some(&root));
                let d = run_mini(&mut w3, &p2, &greedy(8));
                assert_eq!(d.3, AT, "depth {depth}: resumed at the cut from disk");
                assert_eq!((&d.0, d.2), (&want.0, want.2));
                assert!(d.1 == want.1, "depth {depth}: disk resume state bytes differ");
            }
        }
        // the echo-shaped continuation: turn 2 = turn 1's prompt + its generated ids + new tokens
        let root = dir.join(format!("store-{depth}-echo"));
        let mut wa = mini_worker(&dir, depth, Some(&root));
        let t1 = run_mini(&mut wa, &p1, &greedy(8));
        assert!(wa.persist(None, PersistMode::Reply, None).head.is_some());
        let mut q = p1.clone();
        q.extend(&t1.0[..t1.0.len() - 1]);
        q.extend([5, 6, 7]);
        let live = run_mini(&mut wa, &q, &greedy(8));
        assert_eq!(live.3, p1.len() + t1.0.len() - 1, "depth {depth}: every generated token reused");
        let mut wb = mini_worker(&dir, depth, Some(&root));
        let disk = run_mini(&mut wb, &q, &greedy(8));
        assert_eq!(disk.3, live.3, "depth {depth}: the disk head is the live state");
        assert_eq!((&disk.0, disk.2), (&live.0, live.2), "depth {depth}: echo continuation tokens / drafts");
        assert!(disk.1 == live.1, "depth {depth}: echo continuation state bytes");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The PRIO_TURN import is used by the window test; keep the type of PRIO_TURN checked here too.
#[test]
fn class_ranks() {
    assert_eq!(crate::cache::rank(PRIO_PRETHINK), PRIO_TURN);
    assert_eq!(crate::cache::rank(PRIO_CUT), crate::cache::PRIO_ANCHOR);
}

/// Lane A input: the prefill attention path and the fused cutoffs are in the identity, so a state
/// made under one path is never resumed under another (the env side; the engine's own auto path:
/// tests_v013_int).
#[test]
fn the_prefill_attention_path_is_in_the_identity() {
    // over a reader, not the process environment: the engines of parallel tests read OXE_PF_ATTN
    // at construction (lane A), so a process-wide set_var here would race them
    let a = crate::state_store::arith_env_with(|_| None);
    let set = [("OXE_PF_ATTN", "fa2"), ("OXE_ATTN_FUSED_MIN_KV", "8192"), ("OXE_ATTN_FUSED_1LANE_MIN_KV", "4096")];
    let b = crate::state_store::arith_env_with(|k| set.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string()));
    assert!(b.contains("OXE_PF_ATTN=fa2") && b.contains("OXE_ATTN_FUSED_MIN_KV=8192") && b.contains("OXE_ATTN_FUSED_1LANE_MIN_KV=4096"), "{b}");
    assert_ne!(ident("sha256:m", &a).hash, ident("sha256:m", &b).hash);
}

/// v0.1.3 raw-completion persist: the native `/completion` API (JSON and SSE) writes the state
/// after the reply, as the chat API does; an ephemeral (LAN share) request writes nothing.
#[tokio::test(flavor = "multi_thread")]
async fn native_completion_persists_after_the_reply() {
    let dir = tmp("v013-raw-persist");
    let long: String = (0..30).map(|i| format!("Rule {i:02}: keep the change small and test it. ")).collect();
    let req = |stream: bool| serde_json::to_vec(&json!({"prompt": long, "temperature": 0, "stream": stream})).unwrap();
    let wait_persist = |s: Arc<State<StubEngine>>, n0: u64| async move {
        let t0 = Instant::now();
        while s.persist_done.load(Ordering::SeqCst) == n0 {
            assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    };

    // JSON reply: one post-reply write, its head at the engine's position
    let s = srv(false, &["First raw answer.", "Second raw answer."], Some(&dir));
    let n0 = s.persist_done.load(Ordering::SeqCst);
    match handle_request(s.clone(), "POST", "/completion", "", &req(false), Some("raw".into()), None).await {
        Reply::Json(200, _) => {}
        _ => panic!("json completion"),
    }
    wait_persist(s.clone(), n0).await;
    let n_kv = kv(&s).len();
    {
        let w = s.worker.lock().unwrap();
        let st = w.states.as_ref().unwrap();
        assert!(st.manifests.values().any(|m| m.snaps.iter().any(|x| x.class == "head" && x.pos == n_kv)), "the raw reply's head is on disk");
    }
    assert!(files(&dir, ".pending").is_empty());
    assert!(consistent(&dir, ident("sha256:m", "")) > 0);

    // SSE reply: the write happens before the final frame, so it is counted when the stream ends
    let n1 = s.persist_done.load(Ordering::SeqCst);
    let mut rx = match handle_request(s.clone(), "POST", "/completion", "", &req(true), Some("raw".into()), None).await {
        Reply::Stream(rx) => rx,
        Reply::Json(c, j) => panic!("stream expected, got {c} {j}"),
    };
    let mut last = String::new();
    while let Some(f) = rx.recv().await {
        last = f;
    }
    assert!(last.contains("stop") || last.contains("content"), "final frame: {last}");
    wait_persist(s.clone(), n1).await;
    assert!(consistent(&dir, ident("sha256:m", "")) > 0);
    let _ = std::fs::remove_dir_all(&dir);

    // ephemeral (LAN share): no state write after the reply
    let dir2 = tmp("v013-raw-ephemeral");
    let s2 = srv(false, &["Ephemeral answer."], Some(&dir2));
    let n2 = s2.persist_done.load(Ordering::SeqCst);
    match handle_policy(s2.clone(), "POST", "/completion", "", &req(false), Some("raw".into()), None, true).await {
        Reply::Json(200, _) => {}
        _ => panic!("ephemeral completion"),
    }
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(s2.persist_done.load(Ordering::SeqCst), n2, "an ephemeral completion writes no state");
    let _ = std::fs::remove_dir_all(&dir2);
}
