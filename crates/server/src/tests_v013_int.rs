//! v0.1.3 integration (cross-lane) tests:
//! - lane B x http.rs: the HTTP front end forwards `X-Oxe-Turn-Kind`, so a labelled compaction
//!   takes its cut checkpoint in production (lane B's own tests call `handle_request` directly);
//! - lane A x lane B: the state identity keys on the prefill attention path the engine actually
//!   takes (`Engine::pf_attn_fa2()`), so a tiled-made state is never resumed by an fa2 engine and
//!   vice versa, with no `OXE_PF_ATTN` variable involved;
//! - orphan sweep (decision 2026-11-12): only stale identities (another numerics epoch, state
//!   ABI, block size or store version) of the same model are removed at open; same-epoch variants
//!   (deep <-> max, tiled <-> fa2, other arith knobs) are kept for budget / TTL eviction.

use crate::cache::PRIO_CUT;
use crate::engine::Engine;
use crate::oxe_engine::OxeEngine;
use crate::state_store::{chain, snap_key, Identity, Manifest, SnapRef, StateStore, BLOCK};
use crate::tests_state::tmp;
use crate::tests_state_v013::{body, srv};
use serde_json::json;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

fn post(port: u16, b: &[u8], headers: &[String]) -> (u16, String) {
    let mut r = format!("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n", b.len());
    for h in headers {
        r += h;
        r += "\r\n";
    }
    r += "\r\n";
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    s.write_all(r.as_bytes()).unwrap();
    s.write_all(b).unwrap();
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    let text = String::from_utf8_lossy(&out).into_owned();
    let (head, rest) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    (head.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0), rest.to_string())
}

#[test]
fn http_forwards_the_turn_kind_so_a_compaction_takes_its_cut() {
    let long_sys: String = (0..30).map(|i| format!("Rule {i:02}: keep the change small and test it. ")).collect();
    let instr = "Summarize the conversation above into a structured summary.";
    for labelled in [true, false] {
        let dir = tmp(&format!("v013-int-http-cut-{labelled}"));
        let s = srv(false, &["Answer one with some words.", "Summary of it."], Some(&dir));
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let l = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let port = l.local_addr().unwrap().port();
        rt.spawn(crate::http::serve(s.clone(), l));
        let send = |msgs: &serde_json::Value, kind: Option<&str>| {
            let n0 = s.persist_done.load(Ordering::SeqCst);
            let mut h = vec!["X-Oxe-Session: h".to_string()];
            if let Some(k) = kind {
                h.push(format!("X-Oxe-Turn-Kind: {k}"));
            }
            let (code, txt) = post(port, &body(msgs, false, true), &h);
            assert_eq!(code, 200, "{txt}");
            let t0 = Instant::now();
            while s.persist_done.load(Ordering::SeqCst) == n0 {
                assert!(t0.elapsed().as_secs() < 30, "persist did not finish");
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        send(&json!([{"role":"system","content":long_sys},{"role":"user","content":"question one"}]), None);
        let mc = json!([{"role":"system","content":long_sys},{"role":"user","content":"question one"},{"role":"assistant","content":"Answer one with some words."},{"role":"user","content":instr}]);
        send(&mc, labelled.then_some("compaction"));
        let cut = {
            let w = s.worker.lock().unwrap();
            let d = s.user_delim.clone().unwrap();
            w.engine.tokens.windows(d.len()).rposition(|x| x == &d[..]).unwrap()
        };
        let has_cut = s.worker.lock().unwrap().cache.snapshot_info().contains(&(cut, PRIO_CUT));
        assert_eq!(has_cut, labelled, "through HTTP: a cut checkpoint only for a labelled compaction");
        drop(rt);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn engine(model: &Path, cache: &Path, fa2: bool) -> OxeEngine {
    let n = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let mut e = OxeEngine::open_cache(model, None, &[n - 1], None, 64, 2048, 0, Some(cache.to_path_buf())).unwrap();
    // explicit both ways: the test must not depend on an OXE_PF_ATTN in the environment
    e.set_pf_attn(if fa2 { model::engine::PfAttn::Fa2 } else { model::engine::PfAttn::Tiled });
    e
}

#[test]
fn a_tiled_state_is_never_resumed_by_an_fa2_engine_and_vice_versa() {
    let dir = std::env::temp_dir().join(format!("oxe-v013-int-pfattn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cache")).unwrap();
    std::fs::set_permissions(dir.join("cache"), std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    let model = dir.join("mini.gguf");
    model::testkit::mini_model_ext(&model, false);
    let a: Vec<u32> = (0..(2 * BLOCK + 60) as u32).map(|i| (i * 13 + 7) % 64).collect();
    let ident = |e: &OxeEngine| Identity::new("sha256:mini", &e.state_id().unwrap(), 64, &e.arith_id(), "b", false);
    let (tiled, fa2) = (engine(&model, &dir.join("cache"), false), engine(&model, &dir.join("cache"), true));
    assert!(!tiled.arith_id().contains("pf_attn=fa2") && fa2.arith_id().contains("pf_attn=fa2"), "{} / {}", tiled.arith_id(), fa2.arith_id());
    assert_ne!(ident(&tiled).hash, ident(&fa2).hash);
    // APPEND-PREFILL: fa2x is its own numerics class (fold / chunk in the identity)
    let mut fx = engine(&model, &dir.join("cache"), true);
    fx.set_pf_attn(model::engine::PfAttn::Fa2x);
    let xp = model::engine::pfattn::XParams::from_env();
    assert!(fx.arith_id().contains(&format!("pf_attn=fa2x.f{}.c{}", xp.fold, xp.ch)), "{}", fx.arith_id());
    assert!(!fa2.arith_id().contains("pf_attn=fa2x"));
    assert!(ident(&fx).hash != ident(&fa2).hash && ident(&fx).hash != ident(&tiled).hash);
    drop((tiled, fa2, fx));
    for maker_fa2 in [false, true] {
        let root = dir.join(format!("store-{maker_fa2}"));
        let mut src = engine(&model, &dir.join("cache"), maker_fa2);
        src.prefill(&a).unwrap();
        let mut st = StateStore::open(&root, ident(&src), 1 << 40, 3600, true).unwrap();
        st.min_free_bytes = 0;
        st.min_mem_bytes = 0;
        let hs = chain(&st.ident.hash, &a);
        let mut buf = Vec::new();
        let blocks: Vec<(String, String)> = (0..2).map(|i| (hs[i].clone(), st.write_block(&mut src, &hs[i], i, &mut buf).unwrap())).collect();
        let n = a.len();
        let hk = snap_key(&st.ident.hash, &hs, &a, n);
        let (hc, hb) = st.write_snap(&mut src, &hk, n, None).unwrap();
        let head = SnapRef { pos: n, class: "turn".into(), key: hk, c: hc, clean: false, bytes: hb };
        st.put_manifest(Manifest { sid: "0".repeat(32), hint: None, created: 0, atime: crate::state_store::now_s(), n_tokens: n, blocks, snaps: vec![head] }).unwrap();
        let mut q = a.clone();
        q.push(1);
        // the maker's own kind of engine finds it ...
        let same = engine(&model, &dir.join("cache"), maker_fa2);
        let st_same = StateStore::open(&root, ident(&same), 1 << 40, 3600, true).unwrap();
        assert_eq!(st_same.lookup(&q, q.len() - 1, 0).map(|p| p.snap.pos), Some(n), "maker fa2={maker_fa2}: same path resumes");
        // ... the other path never does
        let other = engine(&model, &dir.join("cache"), !maker_fa2);
        let st_other = StateStore::open(&root, ident(&other), 1 << 40, 3600, true).unwrap();
        assert!(st_other.lookup(&q, q.len() - 1, 0).is_none(), "maker fa2={maker_fa2}: a state of the other prefill path was offered");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_deep_max_switch_keeps_both_and_a_stale_epoch_or_abi_is_removed() {
    use crate::tests_state::{ident, open};
    let dir = tmp("v013-int-orphan");
    let deep = ident("sha256:m", "");
    let max = ident("sha256:m", " pf_attn=fa2");
    let other_model = ident("sha256:other", "");
    let key = |i: &Identity| i.model_key().to_string();
    // an identity directory is created at open; switch deep -> max -> deep
    drop(open(&dir, deep.clone(), 1 << 40));
    drop(open(&dir, other_model.clone(), 1 << 40));
    drop(open(&dir, max.clone(), 1 << 40));
    assert!(dir.join(key(&deep)).is_dir() && dir.join(key(&max)).is_dir(), "deep -> max keeps deep");
    drop(open(&dir, deep.clone(), 1 << 40));
    assert!(dir.join(key(&deep)).is_dir() && dir.join(key(&max)).is_dir(), "max -> deep keeps max");
    // a numerics-epoch bump: the old directory can never be read again -> removed
    let rewrite = |i: &Identity, f: &dyn Fn(&mut serde_json::Value)| {
        let p = dir.join(key(i)).join("identity.json");
        let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        f(&mut v);
        std::fs::write(&p, v.to_string()).unwrap();
    };
    rewrite(&max, &|v| v["numerics_epoch"] = json!(model::engine::NUMERICS_EPOCH.wrapping_sub(1)));
    drop(open(&dir, deep.clone(), 1 << 40));
    assert!(!dir.join(key(&max)).exists(), "an older numerics epoch is removed");
    // a state ABI change (another state layout) and another store version: removed too
    drop(open(&dir, max.clone(), 1 << 40));
    rewrite(&max, &|v| {
        let t = v["identity"].as_str().unwrap().replacen("state=", "state=old-layout ", 1);
        v["identity"] = json!(t);
    });
    drop(open(&dir, deep.clone(), 1 << 40));
    assert!(!dir.join(key(&max)).exists(), "another state ABI is removed");
    drop(open(&dir, max.clone(), 1 << 40));
    rewrite(&max, &|v| v["version"] = json!(0));
    drop(open(&dir, deep.clone(), 1 << 40));
    assert!(!dir.join(key(&max)).exists(), "another store version is removed");
    // another model is never touched
    assert!(dir.join(key(&other_model)).is_dir(), "another model's directory stays");
    let _ = std::fs::remove_dir_all(&dir);
}

/// v0.1.3 PREFILL-POLICY (#2): the identity keys on the path `auto` RESOLVES to. A 32K engine:
/// auto under the legacy policy is tiled (the tiled identity, no suffix); auto with min-kv 8192 and
/// the fa2 kernel is the explicit fa2 path (same identity: same bits); auto with min-kv 16384 adds
/// the cutoff (`.k16384`), per kernel; explicit modes ignore the auto policy.
#[test]
fn the_identity_keys_on_the_resolved_auto_path() {
    use model::engine::{AutoPolicy, PfAttn};
    let dir = std::env::temp_dir().join(format!("oxe-v013-pfpolicy-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cache")).unwrap();
    std::fs::set_permissions(dir.join("cache"), std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    let model = dir.join("mini.gguf");
    model::testkit::mini_model_ctx(&model, false, 65_536);
    let n = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    let mk = |mode: PfAttn, auto: AutoPolicy| {
        let mut e = OxeEngine::open_cache(&model, None, &[n - 1], None, 64, 32_768, 0, Some(dir.join("cache"))).unwrap();
        e.set_pf_attn(mode);
        e.set_pf_auto(auto);
        e
    };
    let ident = |e: &OxeEngine| Identity::new("sha256:mini", &e.state_id().unwrap(), 64, &e.arith_id(), "b", false).hash;
    let k16 = |k: PfAttn| AutoPolicy::new(Some(16_384), k).unwrap();
    let tiled = mk(PfAttn::Tiled, AutoPolicy::LEGACY);
    let auto_leg = mk(PfAttn::Auto, AutoPolicy::LEGACY);
    assert!(!auto_leg.arith_id().contains("pf_attn="), "{}", auto_leg.arith_id());
    assert_eq!(ident(&auto_leg), ident(&tiled), "legacy auto at 32K is the tiled path");
    assert!(auto_leg.pf_attn_summary().contains("prefill attention tiled"), "{}", auto_leg.pf_attn_summary());
    let fa2 = mk(PfAttn::Fa2, k16(PfAttn::Fa2x));
    assert!(fa2.arith_id().ends_with(" pf_attn=fa2"), "explicit fa2 ignores the auto policy: {}", fa2.arith_id());
    let auto8 = mk(PfAttn::Auto, AutoPolicy::new(Some(8_192), PfAttn::Fa2).unwrap());
    assert_eq!(ident(&auto8), ident(&fa2), "auto resolving to fa2 at PFA_MIN_KV is the fa2 path");
    let a16 = mk(PfAttn::Auto, k16(PfAttn::Fa2));
    assert!(a16.arith_id().ends_with(" pf_attn=fa2.k16384"), "{}", a16.arith_id());
    let x16 = mk(PfAttn::Auto, k16(PfAttn::Fa2x));
    let xp = model::engine::pfattn::XParams::from_env();
    assert!(x16.arith_id().ends_with(&format!(" pf_attn=fa2x.f{}.c{}.k16384", xp.fold, xp.ch)), "{}", x16.arith_id());
    assert!(x16.pf_attn_summary().contains("prefill attention fa2x beyond n_kv 16384"), "{}", x16.pf_attn_summary());
    let fx = mk(PfAttn::Fa2x, AutoPolicy::LEGACY);
    let hs = [ident(&tiled), ident(&fa2), ident(&a16), ident(&x16), ident(&fx)];
    for i in 0..hs.len() {
        for j in i + 1..hs.len() {
            assert_ne!(hs[i], hs[j], "paths {i} and {j} share an identity");
        }
    }
    drop((tiled, auto_leg, fa2, auto8, a16, x16, fx));
    let _ = std::fs::remove_dir_all(&dir);
}
