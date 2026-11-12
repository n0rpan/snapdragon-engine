//! CACHE-PERFECT P2b, Codex P2/P3 review 1-3 (model-free, stub engine, temp dirs only): a session
//! clear purges every tier, budget 0 turns off every conversation disk writer (state store, echo
//! files, legacy prefix store / session cache) and a clear(all) covers the legacy stores too.

use crate::engine::Engine;
use crate::state_store::hint_hash;
use crate::tests_echo::synth_tokenizer;
use crate::tests_state::{call, files, sys, tmp};
use crate::tests_state_p2b::{chat_h, model_dir, server_echo, toks};
use serde_json::json;
use std::path::Path;

/// Codex P2/P3 review 1: clearing one session purges it from every tier (manifest, echo file and
/// RAM echo records, the live and parked RAM state, legacy store entries) so nothing can bring it
/// back; the other session (and the blocks it shares) still restores after a restart.
#[tokio::test(flavor = "multi_thread")]
async fn session_clear_purges_every_tier_and_keeps_the_other_session() {
    let tok = synth_tokenizer();
    let dir = tmp("p2b-clear");
    let a = server_echo(&dir, vec![tok.encode("Answer A.", false, true), tok.encode("Answer B.", false, true)], 1 << 40);
    let ma = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation A asks this"}]);
    let mb = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation B asks that"}]);
    chat_h(&a, mb.clone(), "conv-b").await;
    chat_h(&a, ma.clone(), "conv-a").await; // A is the live state
    let a_tokens = crate::tests_state::kv(&a);
    let md = model_dir(&a);
    let (sid_a, n_echo_b) = {
        let w = a.worker.lock().unwrap();
        let s = w.states.as_ref().unwrap();
        let sid = s.manifests.values().find(|m| m.hint.as_deref() == Some(hint_hash("conv-a").as_str())).unwrap().sid.clone();
        (sid, a.echo.lock().unwrap().n_session_records("conv-b"))
    };
    assert!(n_echo_b >= 1);
    assert!(md.join("echo").join(format!("{}.jsonl", hint_hash("conv-a"))).is_file());
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "session", "session": sid_a})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["ram_states"], 1, "A's live RAM state dropped: {j}");
    assert!(j["echo_records"].as_u64().unwrap() >= 1, "{j}");
    assert!(!md.join("echo").join(format!("{}.jsonl", hint_hash("conv-a"))).exists(), "A's echo file gone");
    assert_eq!(a.echo.lock().unwrap().n_session_records("conv-a"), 0, "A's RAM echo records gone");
    assert_eq!(a.echo.lock().unwrap().n_session_records("conv-b"), n_echo_b, "B's echo records stay");
    assert!(crate::tests_state::kv(&a).is_empty(), "A's live state is gone");
    assert!(md.join("echo").join(format!("{}.jsonl", hint_hash("conv-b"))).is_file(), "B's echo file stays");
    drop(a);
    // restart: A cannot be restored (no resurrection), B still can; the shared system blocks stay
    let b = server_echo(&dir, vec![tok.encode("x", false, true), tok.encode("y", false, true)], 1 << 40);
    let ma2 = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation A asks this"},{"role":"assistant","content":"Answer A."},{"role":"user","content":"more"}]);
    let r = chat_h(&b, ma2, "conv-a").await;
    let n = r["timings"]["cache_n"].as_u64().unwrap() as usize;
    assert!(n < a_tokens.len(), "A's own state is gone: resumed only {n} (shared prefix at most)");
    {
        let mut w = b.worker.lock().unwrap();
        let w = &mut *w;
        w.cache.reset(&mut w.engine);
        let _ = w.cache.drain_parked();
    }
    let mb2 = json!([{"role":"system","content":sys()},{"role":"user","content":"conversation B asks that"},{"role":"assistant","content":"Answer B."},{"role":"user","content":"more"}]);
    let r = chat_h(&b, mb2, "conv-b").await;
    assert_eq!(r["timings"]["resume_src"], "disk", "B restores from disk: {}", r["timings"]);
    assert!(r["timings"]["cache_n"].as_u64().unwrap() as usize > n);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Codex P2/P3 review 2 and 3: budget 0 means no conversation state goes to disk from then on:
/// no state files, no echo lines, no legacy prefix-store / session-cache entries; raising the
/// budget brings them back. A clear(all) also clears the legacy stores' entries.
#[tokio::test(flavor = "multi_thread")]
async fn budget_zero_turns_off_every_disk_writer_and_clear_covers_the_legacy_stores() {
    let tok = synth_tokenizer();
    let dir = tmp("p2b-off");
    let a = server_echo(&dir, vec![tok.encode("One.", false, true), tok.encode("Two.", false, true), tok.encode("Three.", false, true)], 1 << 40);
    // a legacy prefix store next to it (as --prefix-cache)
    let legacy = dir.join("legacy");
    {
        let mut w = a.worker.lock().unwrap();
        let mut ps = crate::prefix_store::PrefixStore::open(&legacy, "legacy-id".into(), 1 << 30, 64).unwrap();
        ps.save_async(toks(300, 1), vec![1, 2, 3, 4]);
        ps.flush();
        w.store = Some(ps);
    }
    let (c, j) = call(&a, "POST", "/state-cache/budget", json!({"bytes": 0})).await;
    assert_eq!(c, 200, "{j}");
    assert_eq!(j["legacy_stores"], "off");
    let snapshot = |d: &Path| {
        let mut v: Vec<(std::path::PathBuf, u64)> = files(d, "").into_iter().map(|p| {
            let n = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            (p, n)
        }).collect();
        v.sort();
        v
    };
    let before = snapshot(&dir);
    let r = chat_h(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"while off"}]), "conv-off").await;
    assert_eq!(r["choices"][0]["message"]["content"], "One.");
    assert_eq!(snapshot(&dir), before, "budget 0: nothing on disk changed (state, echo, legacy)");
    assert!(a.echo.lock().unwrap().n_session_records("conv-off") >= 1, "RAM echo still works");
    {
        let w = a.worker.lock().unwrap();
        assert!(w.store.is_none() && w.legacy_off.is_some(), "the legacy store is off while the budget is 0");
    }
    // back on: every writer returns
    let (_, j) = call(&a, "POST", "/state-cache/budget", json!({"bytes": 1u64 << 40})).await;
    assert_eq!(j["legacy_stores"], "on");
    chat_h(&a, json!([{"role":"system","content":sys()},{"role":"user","content":"while on"}]), "conv-on").await;
    assert!(model_dir(&a).join("echo").join(format!("{}.jsonl", hint_hash("conv-on"))).is_file(), "echo on disk again");
    assert!(a.worker.lock().unwrap().store.is_some());
    // clear all: the legacy store's entries go too
    let (c, j) = call(&a, "POST", "/state-cache/clear", json!({"scope": "all"})).await;
    assert_eq!(c, 200, "{j}");
    assert!(j["legacy_entries"].as_u64().unwrap() >= 1, "{j}");
    assert_eq!(a.worker.lock().unwrap().store.as_ref().unwrap().n_entries(), 0);
    assert!(files(&legacy, ".bin").is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
