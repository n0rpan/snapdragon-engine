//! CACHE-PERFECT P2b: the `OxeEngine` snapshot glue (snap_save / snap_save_live / snap_load,
//! kv_rows_*) and the MTP driver through the state store, on the tiny synthetic hybrid model
//! (`model::testkit`, no real weights; the weight cache goes to a temp dir, never HOME).
//!
//! G3: a resume from disk with MTP on is exact INCLUDING the draft state: the snapshot carries
//! the driver's q, the pending catch-up pairs and h_prev, so the resumed engine has the same
//! `save_state` bytes (which hold the MTP driver export) as the uninterrupted one, decodes the same
//! tokens with the same draft / accept counters, and a RAM restore of the same checkpoint equals
//! the disk load.

use crate::engine::Engine;
use crate::oxe_engine::OxeEngine;
use crate::state_store::{chain, snap_key, Identity, Manifest, Plan, SnapRef, StateStore, BLOCK};
use std::path::{Path, PathBuf};

fn cpus() -> Vec<usize> {
    let n = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    vec![n - 1]
}

fn engine(model: &Path, cache: &Path, depth: usize) -> OxeEngine {
    OxeEngine::open_cache(model, None, &cpus(), None, 64, 2048, depth, Some(cache.to_path_buf())).unwrap()
}

/// Greedy decode of `steps` engine calls from token `t`: the tokens, the final state bytes and the
/// (drafted, accepted) counters since the start.
fn run(e: &mut OxeEngine, mut t: u32, steps: usize) -> (Vec<u32>, Vec<u8>, (u64, u64)) {
    let c0 = e.spec_counters();
    let mut out = Vec::new();
    for _ in 0..steps {
        out.push(t);
        match e.decode_spec_argmax(t).unwrap() {
            Some((extra, g)) => {
                out.extend(&extra);
                t = *g.last().unwrap();
            }
            None => t = e.decode_argmax(t).unwrap(),
        }
    }
    let mut st = Vec::new();
    e.save_state(&mut st).unwrap();
    let c1 = e.spec_counters();
    (out, st, (c1.0 - c0.0, c1.1 - c0.1))
}

fn state(e: &mut OxeEngine) -> Vec<u8> {
    let mut b = Vec::new();
    e.save_state(&mut b).unwrap();
    b
}

#[test]
fn oxe_snapshot_glue_and_mtp_driver_resume_exactly_from_the_state_store() {
    let dir: PathBuf = std::env::temp_dir().join(format!("oxe-p2b-oxestate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cache")).unwrap();
    std::fs::set_permissions(dir.join("cache"), std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    let model = dir.join("mini-mtp.gguf");
    model::testkit::mini_model_ext(&model, true);
    // the mini model's sha is taken from the environment (model tests do the same)
    std::env::set_var("OXE_MODEL_SHA256", "1".repeat(64));
    let a: Vec<u32> = (0..700).map(|i| (i * 13 + 7) % 64).collect();
    for depth in [0usize, 2] {
        let mut src = engine(&model, &dir.join("cache"), depth);
        // a turn checkpoint at 600 (taken while prefilling, as cache.rs does), the head at 700
        src.prefill(&a[..600]).unwrap();
        let turn = src.snapshot().unwrap();
        src.prefill(&a[600..]).unwrap();
        assert_eq!(src.n_past(), 700);
        let root = dir.join(format!("store-{depth}"));
        let id = Identity::new("sha256:mini", &src.state_id().unwrap(), 64, &src.arith_id(), "b", false);
        let mut st = StateStore::open(&root, id, 1 << 40, 3600, true).unwrap();
        st.min_free_bytes = 0;
        st.min_mem_bytes = 0;
        let hs = chain(&st.ident.hash, &a);
        let mut buf = Vec::new();
        let blocks: Vec<(String, String)> = (0..2).map(|i| (hs[i].clone(), st.write_block(&mut src, &hs[i], i, &mut buf).unwrap())).collect();
        // head: straight from the live buffers (snap_save_live, the shutdown path); turn: a RAM checkpoint
        let hk = snap_key(&st.ident.hash, &hs, &a, 700);
        let (hc, hb) = st.write_snap(&mut src, &hk, 700, None).unwrap();
        let tk = snap_key(&st.ident.hash, &hs, &a, 600);
        let (tc, tb) = st.write_snap(&mut src, &tk, 600, Some(&turn)).unwrap();
        let snap = |pos: usize, key: &str, c: &str, bytes: u64| SnapRef { pos, class: "turn".into(), key: key.into(), c: c.into(), clean: false, bytes };
        let (head_ref, turn_ref) = (snap(700, &hk, &hc, hb), snap(600, &tk, &tc, tb));
        st.put_manifest(Manifest { sid: "0".repeat(32), hint: None, created: 0, atime: crate::state_store::now_s(), n_tokens: 700, blocks: blocks.clone(), snaps: vec![turn_ref.clone(), head_ref.clone()] }).unwrap();
        // the uninterrupted references: from the head, and a RAM restore of the turn checkpoint
        let want_head_state = state(&mut src);
        let want_head = run(&mut src, 5, 12);
        // (back to 700 for the turn reference: reload the engine from its own head)
        let mut ram = engine(&model, &dir.join("cache"), depth);
        st.load(&mut ram, &Plan { sid: "0".repeat(32), blocks: blocks.clone(), snap: head_ref.clone() }).unwrap();
        ram.restore(&turn).unwrap();
        let want_turn_state = state(&mut ram);
        let want_turn = run(&mut ram, 5, 12);
        for (r, want_state, want) in [(&head_ref, &want_head_state, &want_head), (&turn_ref, &want_turn_state, &want_turn)] {
            let mut dst = engine(&model, &dir.join("cache"), depth);
            // a used engine: the load must not depend on what was there
            dst.prefill(&a[..300].iter().map(|x| (x + 1) % 64).collect::<Vec<_>>()).unwrap();
            let plan = Plan { sid: "0".repeat(32), blocks: blocks[..r.pos / BLOCK].to_vec(), snap: r.clone() };
            st.load(&mut dst, &plan).unwrap();
            assert_eq!(dst.n_past(), r.pos);
            assert!(state(&mut dst) == *want_state, "depth {depth} pos {}: state bytes (incl. the MTP driver export) differ", r.pos);
            let got = run(&mut dst, 5, 12);
            assert_eq!(got.0, want.0, "depth {depth} pos {}: tokens", r.pos);
            assert!(got.1 == want.1, "depth {depth} pos {}: state after decoding", r.pos);
            assert_eq!(got.2, want.2, "depth {depth} pos {}: MTP drafted / accepted (the draft state) differ", r.pos);
            if depth > 0 {
                assert!(want.2 .0 > 0, "MTP drafted");
            }
        }
        // a snapshot of the other MTP setting is refused (never a silent wrong draft state)
        let mut other = engine(&model, &dir.join("cache"), if depth == 0 { 2 } else { 0 });
        let plan = Plan { sid: "0".repeat(32), blocks: blocks.clone(), snap: head_ref.clone() };
        assert!(st.load(&mut other, &plan).is_err());
    }
    std::env::remove_var("OXE_MODEL_SHA256");
    let _ = std::fs::remove_dir_all(&dir);
}
