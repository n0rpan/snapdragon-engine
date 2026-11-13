//! v0.1.3 PREFILL-LONG (model-free, mini model from testkit): the fa2 prefill attention in the
//! engine (`OXE_PF_ATTN`, [`Engine::set_pf_attn`]), its O(chunk) scratch, THREADS-BY-PHASE
//! (`OXE_PF_ATTN_THREADS`), the O-70 fused decode cutoffs / lane policy / OXE_ATTN_STATS, the V^T
//! THP switch (LC-8), the chunked MTP batched prefill (LC-6) and a numerics golden at a deep,
//! tiled position (CP2B-GOLD).
//!
//! The mini model's only attention layer is its last layer: the KV cache and the recurrent state
//! after a prefill do not depend on attention outputs (they must be bit-identical between kernels),
//! while every prompt row's hidden state (prefill_ext) and the logits do. Deep contexts are
//! synthetic states read with `read_state` (see the parent module).
//! The kernel itself is tested against exact f64 attention at production shapes in
//! `tests/pfattn_kernel.rs`.

use super::{long_engine, long_parts, rec_bytes, rel_diff, synth_state, synth_state_shift};
use crate::engine::pfattn::{self, AutoPolicy, PfAttn, PfKernel, PfPath, PFA_MIN_KV};
use crate::engine::{ctx_mem_with, ctx_mem_with_auto, Engine, KvType, ATTN_FUSED_1LANE_MIN_KV, ATTN_FUSED_MIN_KV};
use crate::plan::Plan;
use crate::tests::{bits_eq, fixture, header, kv_all, same_state};
use crate::weights::WeightStore;

const N_CTX: usize = 131_072;

fn engine_mode<'a>(plan: &'a Plan, ws: &'a WeightStore, n_ctx: usize, kv: KvType, mode: PfAttn) -> Engine<'a> {
    let mut e = long_engine(plan, ws, n_ctx, kv, 0, None);
    e.set_pf_attn(mode);
    e
}

/// Loads `blob` (n_pos positions) into `e`, prefills `toks` in ubatches of `ub` with every row's
/// hidden state; returns (hidden rows, logits after the last token).
fn deep_prefill(e: &Engine, pool: &sched::Pool, blob: &[u8], n_pos: usize, toks: &[u32], ub: usize) -> (Vec<f32>, Vec<f32>) {
    e.read_state(&mut &blob[..], n_pos).unwrap();
    let mut h = Vec::new();
    e.prefill_ext(pool, toks, ub, Some(&mut h)).unwrap();
    assert_eq!(e.pos(), n_pos + toks.len());
    (h, e.logits().to_vec())
}

fn toks(n: usize, a: u32, b: u32) -> Vec<u32> {
    (0..n as u32).map(|i| (i * a + b) % 64).collect()
}

/// fa2 vs today's tiled prefill at n_kv 8K / 40K / 70K: KV and recurrent state bit-identical,
/// every hidden row and the logits within the O-70 class bound used for the 70K decode against
/// the f32 reference (0.05 relative), and far closer than the same tiled prefill over an
/// off-by-one context (V shifted by one position), so the comparison can see a wrong row.
#[test]
fn pf_fa2_matches_tiled_at_8k_40k_70k() {
    let (dir, plan, ws, _, pool) = long_parts("pfa2deep", 2, false);
    let tk = toks(300, 7, 3);
    for n_pos in [8_000usize, 40_000, 70_001] {
        let t = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Tiled);
        let rec = rec_bytes(&t, &pool);
        let blob = synth_state(&t, n_pos, &rec);
        let (ht, lt) = deep_prefill(&t, &pool, &blob, n_pos, &tk, 256);
        let f = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Fa2);
        assert!(f.pf_attn_fa2() && !t.pf_attn_fa2());
        let (hf, lf) = deep_prefill(&f, &pool, &blob, n_pos, &tk, 256);
        assert!(hf.iter().chain(&lf).all(|x| x.is_finite()), "n_pos {n_pos}: non-finite fa2 output");
        assert!(same_state(&t.dump_state(), &f.dump_state()), "n_pos {n_pos}: fa2 changed the KV / recurrent state");
        let (dh, dl) = (rel_diff(&hf, &ht), rel_diff(&lf, &lt));
        // sensitivity control: the tiled prefill over V misaligned by one position
        let z = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Tiled);
        let (hz, lz) = deep_prefill(&z, &pool, &synth_state_shift(&z, n_pos, &rec, 1), n_pos, &tk, 256);
        let (oh, ol) = (rel_diff(&hz, &ht), rel_diff(&lz, &lt));
        eprintln!("pf-fa2 n_pos {n_pos}: fa2 vs tiled hidden {dh:.2e} logits {dl:.2e}; off-by-one context hidden {oh:.2e} logits {ol:.2e}");
        assert!(dh < 0.05 && dl < 0.05, "n_pos {n_pos}: fa2 vs tiled hidden {dh} logits {dl}");
        assert!(ol > 5.0 * (dl + 1e-3), "n_pos {n_pos}: the comparison cannot see an off-by-one context (logits {ol} vs fa2 {dl})");
        assert!(oh > 5.0 * dh, "n_pos {n_pos}: hidden rows cannot see an off-by-one context ({oh} vs fa2 {dh})");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Short contexts never change: with fa2 selected, (a) a 700-token prompt from position 0 and its
/// decode are bit-identical to the tiled mode (the scratch is the capped one, only its stride
/// differs), at n_ctx 16384 and at 262144 (`auto` selects fa2 there); (b) a prompt crossing the
/// cutoff (7,600 + 1,000 tokens): the rows whose ubatch n_kv is <= PFA_MIN_KV are bit-identical,
/// the deeper rows fa2-class, the state bit-identical.
#[test]
fn pf_fa2_keeps_today_below_8k_bitwise() {
    let (dir, plan, ws, _, pool) = long_parts("pfa2short", 2, false);
    let ne = plan.desc.n_embd;
    let tk = toks(700, 13, 7);
    for (n_ctx, mode) in [(16_384usize, PfAttn::Fa2), (262_144, PfAttn::Auto)] {
        let run = |m: PfAttn| {
            let e = engine_mode(&plan, &ws, n_ctx, KvType::F16, m);
            let mut h = Vec::new();
            let mut x = e.prefill_ext(&pool, &tk, 256, Some(&mut h)).unwrap();
            let mut rows = vec![e.logits().to_vec()];
            for _ in 0..3 {
                x = e.step(&pool, x);
                rows.push(e.logits().to_vec());
            }
            (e.pf_attn_fa2(), h, rows, e.dump_state())
        };
        let (fa, hf, rf, sf) = run(mode);
        let (ft, ht, rt, st) = run(PfAttn::Tiled);
        assert!(fa && !ft, "n_ctx {n_ctx}: mode {mode:?} must select fa2");
        assert!(bits_eq(&hf, &ht), "n_ctx {n_ctx}: hidden rows differ below the fa2 cutoff");
        for (i, (a, b)) in rf.iter().zip(&rt).enumerate() {
            assert!(bits_eq(a, b), "n_ctx {n_ctx}: logits row {i} differs below the fa2 cutoff");
        }
        assert!(same_state(&sf, &st), "n_ctx {n_ctx}: state differs below the fa2 cutoff");
    }
    // (b) crossing: ubatches at 7600 (n_kv 7936), 7856 (8192): tiled; from 8112 (8448): fa2
    let n_pos = 7_600;
    let tk = toks(1_000, 11, 5);
    let t = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Tiled);
    let rec = rec_bytes(&t, &pool);
    let blob = synth_state(&t, n_pos, &rec);
    let (ht, _) = deep_prefill(&t, &pool, &blob, n_pos, &tk, 256);
    let f = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Fa2);
    let (hf, _) = deep_prefill(&f, &pool, &blob, n_pos, &tk, 256);
    let cut = 512;
    // (the ubatch's last token sets its n_kv)
    assert!(crate::engine::padded_n_kv(n_pos + cut - 1, N_CTX) <= PFA_MIN_KV && crate::engine::padded_n_kv(n_pos + cut + 255, N_CTX) > PFA_MIN_KV);
    assert!(bits_eq(&hf[..cut * ne], &ht[..cut * ne]), "rows with n_kv <= PFA_MIN_KV must keep the tiled bits");
    assert!(!bits_eq(&hf[cut * ne..], &ht[cut * ne..]), "rows beyond PFA_MIN_KV did not run fa2");
    assert!(rel_diff(&hf[cut * ne..], &ht[cut * ne..]) < 0.05);
    assert!(same_state(&t.dump_state(), &f.dump_state()));
    std::fs::remove_dir_all(&dir).ok();
}

/// fa2 is thread-count invariant (1 and 3 threads) and THREADS-BY-PHASE (attention units on 1 or
/// 2 of 3 threads) changes no bit, for fa2 and for today's tiled kernel.
#[test]
fn pf_fa2_and_attn_threads_are_thread_invariant() {
    let n_pos = 9_000;
    let tk = toks(300, 5, 1);
    let mut want: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut want_tiled: Option<(Vec<f32>, Vec<f32>)> = None;
    for nt in [1usize, 3] {
        let (dir, plan, ws, _, pool) = long_parts(&format!("pfa2thr{nt}"), nt, false);
        for (mode, attn_threads) in [(PfAttn::Fa2, 0usize), (PfAttn::Fa2, 1), (PfAttn::Fa2, 2), (PfAttn::Tiled, 0), (PfAttn::Tiled, 1)] {
            let mut e = engine_mode(&plan, &ws, N_CTX, KvType::F16, mode);
            e.set_pf_attn_threads(attn_threads);
            let rec = rec_bytes(&e, &pool);
            let blob = synth_state(&e, n_pos, &rec);
            let got = deep_prefill(&e, &pool, &blob, n_pos, &tk, 256);
            let slot = if mode == PfAttn::Fa2 { &mut want } else { &mut want_tiled };
            if let Some((h, l)) = slot.as_ref() {
                assert!(bits_eq(&got.0, h) && bits_eq(&got.1, l), "{mode:?}: {nt} threads, attention on {attn_threads}: result differs");
            } else {
                *slot = Some(got);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// The prefill scratch per thread is O(chunk) with fa2: the same at 131072 and 262144, equal to
/// the PFA_MIN_KV-capped tiled scratch plus the fa2 unit scratch; `auto` is tiled at 131072 and
/// fa2 at 262144; int8 KV keeps its own (context-sized) scratch. The engine's allocation equals
/// the accounting after a deep prefill. Ornith / Qwen3.6 shapes (GGUF header fixtures) printed.
#[test]
fn pf_fa2_scratch_is_o_chunk() {
    let (dir, plan, ws, _, pool) = long_parts("pfa2mem", 2, false);
    let d = &plan.desc;
    let pp = |n: usize, kv: KvType, m: PfAttn| ctx_mem_with(d, n, kv, 1, 1, 1, m).prefill_per_thread;
    let f128 = pp(131_072, KvType::F16, PfAttn::Fa2);
    let f256 = pp(262_144, KvType::F16, PfAttn::Fa2);
    assert_eq!(f128, f256, "fa2 prefill scratch grows with the context");
    let t8 = pp(PFA_MIN_KV, KvType::F16, PfAttn::Tiled);
    assert!(f256 > t8 && f256 - t8 <= pfattn::Scratch::bytes(pfattn::PFA_MAX_COLS, d.head_dim), "fa2 {f256} vs capped tiled {t8}");
    let t256 = pp(262_144, KvType::F16, PfAttn::Tiled);
    assert!(t256 > 16 * f256, "tiled {t256} vs fa2 {f256} at 262144");
    assert_eq!(pp(131_072, KvType::F16, PfAttn::Auto), pp(131_072, KvType::F16, PfAttn::Tiled));
    assert_eq!(pp(262_144, KvType::F16, PfAttn::Auto), f256);
    assert_eq!(pp(262_144, KvType::KV8, PfAttn::Fa2), pp(262_144, KvType::KV8, PfAttn::Tiled));
    let tk = toks(300, 3, 2);
    for (n_ctx, mode) in [(N_CTX, PfAttn::Tiled), (N_CTX, PfAttn::Fa2), (262_144, PfAttn::Fa2), (262_144, PfAttn::Auto)] {
        let e = engine_mode(&plan, &ws, n_ctx, KvType::F16, mode);
        let rec = rec_bytes(&e, &pool);
        let blob = synth_state(&e, 9_000, &rec);
        // OXE_PF_TILES may be switched by other tests in this process (att_tt 4 / 8): accept
        // the accounting sampled just before or just after the prefill that built the scratch
        let before = e.ctx_mem().prefill_per_thread;
        deep_prefill(&e, &pool, &blob, 9_000, &tk, 256);
        let after = e.ctx_mem().prefill_per_thread;
        let got = e.prefill_ctx_bytes_allocated().unwrap();
        assert!(got == before || got == after, "n_ctx {n_ctx} {mode:?}: allocated {got} vs accounted {before} / {after}");
        assert_eq!(after, ctx_mem_with(d, n_ctx, KvType::F16, e.n_att(), e.lanes(), e.verify_lanes(), mode).prefill_per_thread);
    }
    for key in ["ornith", "qwen36"] {
        let rd = crate::desc::ModelDesc::from_header(&header(&fixture(key))).unwrap();
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        for n in [131_072usize, 262_144] {
            let t = ctx_mem_with(&rd, n, KvType::F16, 11, 8, 2, PfAttn::Tiled).prefill_per_thread;
            let f = ctx_mem_with(&rd, n, KvType::F16, 11, 8, 2, PfAttn::Fa2).prefill_per_thread;
            eprintln!("pf-fa2 {key} n_ctx {n}: prefill scratch per thread tiled {:.1} MiB, fa2 {:.1} MiB; 12 threads: -{:.2} GiB", t as f64 / 1048576.0, f as f64 / 1048576.0, gib(12 * (t - f)));
            assert!(f * 10 < t, "{key} {n}: fa2 {f} vs tiled {t}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A 32K f16 engine with 4 verify lanes, fused on / off and the given fused cutoffs.
fn fused_engine<'a>(plan: &'a Plan, ws: &'a WeightStore, fused: bool, any: usize, one: usize) -> Engine<'a> {
    let mut e = long_engine(plan, ws, 32_768, KvType::F16, 4, None);
    e.set_attn_fused(fused);
    e.set_attn_fused_min_kv(any, one);
    e
}

/// O70-KEEP / O70-POLICY: the fused decode attention has its own cutoff (independent of the tiled
/// one) and single-lane runs stay tiled below ATTN_FUSED_1LANE_MIN_KV while verify runs go fused;
/// the barrier count follows the path; OXE_ATTN_STATS counts in fused mode (same block totals as
/// the tiled path, zero blocks never more than total).
#[test]
fn o70_fused_cutoffs_lane_policy_and_stats() {
    let (dir, plan, ws, _, pool) = long_parts("o70pol", 2, false);
    let n_att = plan.desc.n_attention();
    let base = plan.barriers_per_token();
    let mk = |fused: bool, any: usize, one: usize| fused_engine(&plan, &ws, fused, any, one);
    let rec = rec_bytes(&mk(false, 0, 0), &pool);
    for n_pos in [6_000usize, 20_000] {
        let blob = synth_state(&mk(false, 0, 0), n_pos, &rec);
        let load = |e: &Engine| e.read_state(&mut &blob[..], n_pos).unwrap();
        // tiled reference step
        let mut t = mk(false, ATTN_FUSED_MIN_KV, ATTN_FUSED_1LANE_MIN_KV);
        t.set_attn_stats(true);
        load(&t);
        assert_eq!(t.barriers_per_token(), base + 2 * n_att);
        t.step(&pool, 7);
        let lt = t.logits().to_vec();
        // fused with the default policy
        let mut p = mk(true, ATTN_FUSED_MIN_KV, ATTN_FUSED_1LANE_MIN_KV);
        p.set_attn_stats(true);
        load(&p);
        let one_lane_fused = n_pos >= ATTN_FUSED_1LANE_MIN_KV;
        assert_eq!(p.barriers_per_token(), base + if one_lane_fused { n_att } else { 2 * n_att }, "n_pos {n_pos}");
        p.step(&pool, 7);
        if one_lane_fused {
            assert!(!bits_eq(p.logits(), &lt), "n_pos {n_pos}: a single lane above the cutoff did not run fused");
            assert!(rel_diff(p.logits(), &lt) < 0.05);
        } else {
            assert!(bits_eq(p.logits(), &lt), "n_pos {n_pos}: a single lane below 16K must stay tiled");
        }
        // stats: the same block totals as the tiled path (one column per query head)
        let (st, sp) = (t.attn_stats().unwrap(), p.attn_stats().unwrap());
        let total = (plan.desc.n_head * ((n_pos + 1) / 32) * n_att) as u64;
        assert_eq!(st.0, total, "tiled stats total");
        assert_eq!(sp.0, total, "n_pos {n_pos}: fused stats total");
        assert!(sp.1 <= sp.2 && sp.2 <= sp.0, "{sp:?}");
        eprintln!("o70 stats n_pos {n_pos}: tiled {st:?} fused-path {sp:?} (fused={one_lane_fused})");
        // verify runs are fused at both depths: lane 0 equals a 1-lane run with the lane rule off
        let v = mk(true, ATTN_FUSED_MIN_KV, ATTN_FUSED_1LANE_MIN_KV);
        load(&v);
        v.verify(&pool, &[7, 9]);
        let lane0 = v.logits_lane(0).to_vec();
        let f1 = mk(true, ATTN_FUSED_MIN_KV, ATTN_FUSED_MIN_KV);
        load(&f1);
        f1.step(&pool, 7);
        assert!(bits_eq(f1.logits(), &lane0), "n_pos {n_pos}: verify lane 0 is not the fused 1-lane result");
        assert!(!bits_eq(&lane0, &lt), "n_pos {n_pos}: the verify run did not take the fused path");
        // own cutoff: fused below the tiled cutoff, and off above it when its cutoff is higher
        let mut lo = mk(true, 0, 0);
        lo.set_attn_tiled_min_kv(1 << 30);
        load(&lo);
        assert_eq!(lo.barriers_per_token(), base + n_att, "fused must not ride the tiled cutoff");
        lo.step(&pool, 7);
        assert!(bits_eq(lo.logits(), f1.logits()));
        let hi = mk(true, 1 << 30, 1 << 30);
        load(&hi);
        assert_eq!(hi.barriers_per_token(), base + 2 * n_att);
        hi.step(&pool, 7);
        assert!(bits_eq(hi.logits(), &lt), "above its own cutoff the run must be tiled");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// VmFlags of the /proc/self/smaps mapping that contains `addr`.
fn vma_flags(addr: usize) -> Option<String> {
    let s = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let mut inside = false;
    for line in s.lines() {
        if let Some((range, _)) = line.split_once(' ') {
            if let Some((a, b)) = range.split_once('-') {
                if let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) {
                    inside = a <= addr && addr < b;
                    continue;
                }
            }
        }
        if inside && line.starts_with("VmFlags:") {
            return Some(format!("{line} "));
        }
    }
    None
}

/// LC-8: OXE_KV_V_THP=0 ([`crate::engine::set_kv_v_thp`]) advises the V^T region MADV_NOHUGEPAGE
/// (VmFlags `nh`, not `hg`), the default keeps it THP (`hg`); results are the same bits.
#[test]
fn lc8_v_region_thp_switch() {
    let (dir, plan, ws, _, pool) = long_parts("lc8", 1, false);
    let tk = toks(300, 9, 4);
    let on = long_engine(&plan, &ws, N_CTX, KvType::F16, 0, None);
    let (a, len) = on.v_region();
    let f_on = vma_flags(a + len / 2).expect("smaps entry of the V^T region");
    crate::engine::set_kv_v_thp(false);
    let off = long_engine(&plan, &ws, N_CTX, KvType::F16, 0, None);
    crate::engine::set_kv_v_thp(true);
    let (a2, len2) = off.v_region();
    let f_off = vma_flags(a2 + len2 / 2).expect("smaps entry of the V^T region");
    eprintln!("lc8: default [{}] / OXE_KV_V_THP=0 [{}]", f_on.trim(), f_off.trim());
    if f_on.contains(" hg ") {
        assert!(f_off.contains(" nh ") && !f_off.contains(" hg "), "V^T region still THP: {f_off}");
    } else {
        eprintln!("lc8: MADV_HUGEPAGE not visible on this kernel; only the result identity is checked");
    }
    on.prefill(&pool, &tk, 256).unwrap();
    off.prefill(&pool, &tk, 256).unwrap();
    assert!(bits_eq(on.logits(), off.logits()) && same_state(&on.dump_state(), &off.dump_state()));
    std::fs::remove_dir_all(&dir).ok();
}

/// LC-6: MtpDriver::prefill_batched in chunks (ubatch-aligned) gives the same logits, state (MTP
/// layer KV included) and following speculative steps, bit for bit, as one call.
#[test]
fn lc6_mtp_prefill_batched_chunks_are_bitwise() {
    let (dir, plan, ws, mw, pool) = long_parts("lc6", 2, true);
    let mw = mw.unwrap();
    let tk = toks(700, 13, 5);
    let run = |chunk: usize, ub: usize| {
        let e = long_engine(&plan, &ws, 4096, KvType::F16, 2, Some(&mw));
        let mut d = crate::mtp::MtpDriver::new(&e, 1).unwrap();
        d.prefill_chunk = chunk;
        let mut x = d.prefill_batched(&e, &pool, &pool, &tk, ub);
        let mut rows = vec![e.logits().to_vec()];
        let mut ids = vec![x];
        let mut out = Vec::new();
        for _ in 0..6 {
            d.step(&e, &pool, x, &mut out);
            for i in 0..out.len() {
                rows.push(e.logits_lane(i).to_vec());
            }
            ids.extend_from_slice(&out);
            x = *out.last().unwrap();
        }
        assert_eq!(d.stats.prefill_tokens, tk.len() as u64);
        (ids, rows, e.dump_state(), kv_all(&e, 2))
    };
    for ub in [256usize, 9] {
        let want = run(0, ub);
        for chunk in [ub, 300, 512] {
            let got = run(chunk, ub);
            assert_eq!(got.0, want.0, "ub {ub} chunk {chunk}: ids differ");
            assert!(got.1.len() == want.1.len() && got.1.iter().zip(&want.1).all(|(a, b)| bits_eq(a, b)), "ub {ub} chunk {chunk}: logits differ");
            assert!(same_state(&got.2, &want.2) && got.3 == want.3, "ub {ub} chunk {chunk}: state differs");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// CP2B-GOLD: numerics golden at a deep position: a 9,000-position synthetic context (beyond
/// ATTN_TILED_MIN_KV and PFA_MIN_KV), a 300-token tiled prefill (today's default kernel; fa2 is
/// opt-in and outside the epoch) and 4 tiled decode steps, every KV type; logits and state hashed.
/// Same key / epoch rules as `numerics_epoch_golden_state` (an unknown kernel path fails).
#[test]
fn numerics_epoch_golden_deep_tiled() {
    const EPOCH: u32 = crate::engine::NUMERICS_EPOCH;
    // (Check64 of the kernel path, epoch, golden)
    const GOLDEN: &[(&str, u32, &str)] = &[
        // Snapdragon X Elite laptop (Oryon) and the lab-arm64 helper (Apple M2, Asahi): the same kernel
        // path key as numerics_epoch_golden_state (recorded from ci-arm64 run 43242235946)
        ("4dbfb70103822fda", EPOCH, "dd6faf757a18be29"),
    ];
    let (dir, plan, ws, _, pool) = long_parts("golddeep", 2, false);
    let tk = toks(300, 13, 7);
    let n_pos = 9_000;
    struct Fnv(u64);
    impl Fnv {
        fn update(&mut self, b: &[u8]) {
            for &x in b {
                self.0 = (self.0 ^ x as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
        let e = engine_mode(&plan, &ws, 16_384, kv, PfAttn::Tiled);
        let rec = rec_bytes(&e, &pool);
        e.read_state(&mut &synth_state(&e, n_pos, &rec)[..], n_pos).unwrap();
        let mut tok = e.prefill(&pool, &tk, 256).unwrap();
        for x in e.logits() {
            h.update(&x.to_bits().to_le_bytes());
        }
        for _ in 0..4 {
            tok = e.step(&pool, tok);
            for x in e.logits() {
                h.update(&x.to_bits().to_le_bytes());
            }
        }
        let mut st = Vec::new();
        e.write_state(&mut st).unwrap();
        h.update(&st);
    }
    let got = format!("{:016x}", h.0);
    let kp = crate::engine::kernel_path();
    let key = {
        let mut c = crate::check::Check64::default();
        c.update(kp.as_bytes());
        format!("{:016x}", c.finish())
    };
    std::fs::remove_dir_all(&dir).ok();
    eprintln!("numerics-golden-deep: key {key} epoch {EPOCH} state {got} kernel_path [{kp}]");
    match GOLDEN.iter().find(|g| g.0 == key) {
        Some(&(_, epoch, want)) => {
            assert_eq!(epoch, EPOCH, "NUMERICS_EPOCH was bumped: record the new deep golden for kernel path {key} (got {got})");
            assert_eq!(got, want, "deep-context numerics changed on kernel path {key} [{kp}]: bump NUMERICS_EPOCH (engine.rs) and both goldens together");
        }
        None if std::env::var("OXE_NUMERICS_GOLDEN_UNLISTED").as_deref() == Ok("print") => {
            eprintln!("numerics_epoch_golden_deep_tiled: kernel path {key} is not allowlisted (OXE_NUMERICS_GOLDEN_UNLISTED=print): not checked");
        }
        None => panic!("kernel path {key} [{kp}] is not allowlisted: add (\"{key}\", {EPOCH}, \"{got}\") to GOLDEN in numerics_epoch_golden_deep_tiled (one commit)"),
    }
}

/// v0.1.3 APPEND-PREFILL: fa2x in the engine (mini model: head_dim 128, 2 query heads per KV
/// head). At n_kv 8K / 40K / 70K: state bit-identical to tiled, hidden rows / logits within the
/// fa2 bound of tiled and not the fa2 bits (the fa2x kernel ran); bit-identical for 1 and 3
/// threads and with attention units on 1 of 3 threads; rows below PFA_MIN_KV keep the tiled bits.
#[test]
fn pf_fa2x_engine_matches_tiled_thread_invariant_and_keeps_short_rows() {
    let tk = toks(300, 7, 3);
    let mut want: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for nt in [1usize, 3] {
        let (dir, plan, ws, _, pool) = long_parts(&format!("pfa2x{nt}"), nt, false);
        for (i, n_pos) in [8_000usize, 40_000, 70_001].into_iter().enumerate() {
            let t = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Tiled);
            let rec = rec_bytes(&t, &pool);
            let blob = synth_state(&t, n_pos, &rec);
            let mut x = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Fa2x);
            assert!(x.pf_attn_fa2() && x.pf_attn_fa2x().is_some() && x.pf_attn_kernel() == "fa2x");
            if nt == 3 && i == 1 {
                x.set_pf_attn_threads(1);
            }
            let (hx, lx) = deep_prefill(&x, &pool, &blob, n_pos, &tk, 256);
            assert!(hx.iter().chain(&lx).all(|v| v.is_finite()), "n_pos {n_pos}: non-finite fa2x output");
            if nt == 1 {
                let (ht, lt) = deep_prefill(&t, &pool, &blob, n_pos, &tk, 256);
                assert!(same_state(&t.dump_state(), &x.dump_state()), "n_pos {n_pos}: fa2x changed the KV / recurrent state");
                let (dh, dl) = (rel_diff(&hx, &ht), rel_diff(&lx, &lt));
                eprintln!("pf-fa2x n_pos {n_pos}: fa2x vs tiled hidden {dh:.2e} logits {dl:.2e}");
                assert!(dh < 0.05 && dl < 0.05, "n_pos {n_pos}: fa2x vs tiled hidden {dh} logits {dl}");
                let f = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Fa2);
                assert_eq!(f.pf_attn_kernel(), "fa2");
                let (hf, _) = deep_prefill(&f, &pool, &blob, n_pos, &tk, 256);
                assert!(!bits_eq(&hf, &hx), "n_pos {n_pos}: fa2x gave the fa2 bits (kernel not run?)");
                want.push((hx, lx));
            } else {
                assert!(bits_eq(&hx, &want[i].0) && bits_eq(&lx, &want[i].1), "n_pos {n_pos}: fa2x differs between 1 and 3 threads");
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
    // crossing the cutoff from 7,600: the rows of ubatches with n_kv <= PFA_MIN_KV keep the tiled bits
    let (dir, plan, ws, _, pool) = long_parts("pfa2xcut", 2, false);
    let ne = plan.desc.n_embd;
    let n_pos = 7_600;
    let tk = toks(1_000, 11, 5);
    let t = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Tiled);
    let blob = synth_state(&t, n_pos, &rec_bytes(&t, &pool));
    let (ht, _) = deep_prefill(&t, &pool, &blob, n_pos, &tk, 256);
    let x = engine_mode(&plan, &ws, N_CTX, KvType::F16, PfAttn::Fa2x);
    let (hx, _) = deep_prefill(&x, &pool, &blob, n_pos, &tk, 256);
    let cut = 512;
    assert!(bits_eq(&hx[..cut * ne], &ht[..cut * ne]), "fa2x: rows with n_kv <= PFA_MIN_KV must keep the tiled bits");
    assert!(!bits_eq(&hx[cut * ne..], &ht[cut * ne..]) && rel_diff(&hx[cut * ne..], &ht[cut * ne..]) < 0.05);
    assert!(same_state(&t.dump_state(), &x.dump_state()));
    std::fs::remove_dir_all(&dir).ok();
}

/// v0.1.3 PREFILL-POLICY (#2): `auto` with a min-kv cutoff in the engine. A 32K engine with
/// min-kv 16384 resolves to the auto kernel beyond 16384 (fa2 or fa2x), its scratch accounting is
/// the cutoff-capped one, and a prompt crossing the cutoff keeps the tiled bits for every ubatch
/// with padded n_kv <= 16384 while the deeper rows are the explicit fa2x / fa2 class; the default
/// policy keeps a 32K auto engine tiled, bit for bit (behaviour unchanged until AP-1).
#[test]
fn pf_auto_policy_min_kv_in_the_engine() {
    let (dir, plan, ws, _, pool) = long_parts("pfauto", 2, false);
    let (d, ne) = (&plan.desc, plan.desc.n_embd);
    let n_ctx = 32_768;
    let n_pos = 15_600;
    let tk = toks(1_000, 13, 1);
    let t = engine_mode(&plan, &ws, n_ctx, KvType::F16, PfAttn::Tiled);
    let blob = synth_state(&t, n_pos, &rec_bytes(&t, &pool));
    let (ht, lt) = deep_prefill(&t, &pool, &blob, n_pos, &tk, 256);
    // ubatches end at 15855 / 16111 / 16367 (padded 15872 / 16128 / 16384 <= 16384): 768 tiled rows
    let cut_rows = 768;
    // default policy: auto at 32K is tiled, bit for bit
    let mut a = engine_mode(&plan, &ws, n_ctx, KvType::F16, PfAttn::Auto);
    a.set_pf_auto(AutoPolicy::LEGACY);
    assert_eq!(a.pf_attn_path(), PfPath::TILED);
    let (ha, la) = deep_prefill(&a, &pool, &blob, n_pos, &tk, 256);
    assert!(bits_eq(&ha, &ht) && bits_eq(&la, &lt), "legacy auto at 32K must be the tiled bits");
    for kernel in [PfAttn::Fa2x, PfAttn::Fa2] {
        let pol = AutoPolicy::new(Some(16_384), kernel).unwrap();
        let mut e = engine_mode(&plan, &ws, n_ctx, KvType::F16, PfAttn::Auto);
        e.set_pf_auto(pol);
        let want_k = if kernel == PfAttn::Fa2x { PfKernel::Fa2x } else { PfKernel::Fa2 };
        assert_eq!(e.pf_attn_path(), PfPath { kernel: want_k, cut: 16_384 });
        assert_eq!(e.pf_attn_kernel(), kernel.name());
        assert_eq!(e.pf_attn_fa2x().is_some(), kernel == PfAttn::Fa2x);
        let (he, le) = deep_prefill(&e, &pool, &blob, n_pos, &tk, 256);
        assert!(he.iter().chain(&le).all(|v| v.is_finite()));
        assert!(same_state(&t.dump_state(), &e.dump_state()), "{kernel:?}: auto changed the KV / recurrent state");
        assert!(bits_eq(&he[..cut_rows * ne], &ht[..cut_rows * ne]), "{kernel:?}: rows with n_kv <= the cutoff must keep the tiled bits");
        let (dh, dl) = (rel_diff(&he[cut_rows * ne..], &ht[cut_rows * ne..]), rel_diff(&le, &lt));
        eprintln!("pf-auto {kernel:?} min-kv 16384: deep rows vs tiled hidden {dh:.2e} logits {dl:.2e}");
        assert!(!bits_eq(&he[cut_rows * ne..], &ht[cut_rows * ne..]) && dh < 0.05 && dl < 0.05);
        // the scratch is the cutoff-capped one, as accounted
        let acc = ctx_mem_with_auto(d, n_ctx, KvType::F16, e.n_att(), e.lanes(), e.verify_lanes(), PfAttn::Auto, pol).prefill_per_thread;
        let got = e.prefill_ctx_bytes_allocated().unwrap();
        let (before, after) = (acc, e.ctx_mem().prefill_per_thread);
        assert!(got == before || got == after, "{kernel:?}: allocated {got} vs accounted {before} / {after}");
        assert!(acc < ctx_mem_with_auto(d, n_ctx, KvType::F16, e.n_att(), e.lanes(), e.verify_lanes(), PfAttn::Tiled, pol).prefill_per_thread);
        // int8 KV keeps its own kernels whatever the policy
        let mut k8 = engine_mode(&plan, &ws, n_ctx, KvType::KV8, PfAttn::Auto);
        k8.set_pf_auto(pol);
        assert_eq!(k8.pf_attn_path(), PfPath::TILED);
    }
    // an auto engine whose n_ctx cannot exceed the cutoff resolves to tiled
    let mut s16 = engine_mode(&plan, &ws, 16_384, KvType::F16, PfAttn::Auto);
    s16.set_pf_auto(AutoPolicy::new(Some(16_384), PfAttn::Fa2x).unwrap());
    assert_eq!(s16.pf_attn_path(), PfPath::TILED);
    std::fs::remove_dir_all(&dir).ok();
}
