//! v0.1.3 lane G, model-free (child of `stream`; uses its test helpers):
//! - F87: the chat prefill schedule sends message segments of < 15 tokens through the per-token (4-node) expert op
//!   and longer ones through the batched (wave) op; both must give the same bits for the same tokens. Plus the
//!   dense-matmul evidence: a w-column `ggml_mul_mat` vs 1-column calls (reported, see `graph::selftest_mm_cols`).
//! - F123 next: hot-resident protection inside an overflowing batched op (`pf_protect_hot`): same bits, same reads,
//!   the protected hot experts are still resident after the op.
//! - F127-3 / F118: the swap-guard mode in the cache decision, the F118 admission terms pinned.
use super::tests::{batch_op, synth_stream};
use super::*;

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

fn pick(rnd: &mut dyn FnMut() -> u64, lo: usize, n: usize) -> Vec<i32> {
    let mut p: Vec<i32> = Vec::new();
    while p.len() < TOP_K {
        let e = (lo + (rnd() % n as u64) as usize) as i32;
        if !p.contains(&e) {
            p.push(e);
        }
    }
    p
}

/// The per-token (4-node) expert op over `nt` tokens as the graph runs it: FETCH, GATE, UP, the production swiglu
/// (the same row kernel the batched op uses), DOWN. Returns the down outputs [n_embd, TOP_K, nt].
fn per_token_op(st: &Stream, l: usize, ids: &[i32], x: &[f32]) -> Vec<f32> {
    let nt = ids.len() / TOP_K;
    assert_eq!(x.len(), nt * N_EMBD);
    let mut gate = vec![f32::NAN; FF_EXP * TOP_K * nt];
    let mut up = vec![f32::NAN; FF_EXP * TOP_K * nt];
    let mut act = vec![0f32; FF_EXP * TOP_K * nt];
    let mut down = vec![f32::NAN; N_EMBD * TOP_K * nt];
    let call = |kind, src: *const f32, src_nb1: usize, src_nb2: usize, dst: *mut f32, dst_nb1: usize, dst_nb2: usize| FgExpertCall {
        kind,
        il: l as c_int,
        ith: 0,
        nth: 1,
        n_tokens: nt as c_int,
        ids: ids.as_ptr(),
        ids_nb1: TOP_K * 4,
        src,
        src_nb1,
        src_nb2,
        dst,
        dst_nb1,
        dst_nb2,
    };
    // SAFETY: single graph thread; buffers sized for nt tokens x TOP_K.
    unsafe {
        st.run(&call(FG_EXP_FETCH, x.as_ptr(), N_EMBD * 4, N_EMBD * 4, std::ptr::null_mut(), 0, 0));
        st.run(&call(FG_EXP_GATE, x.as_ptr(), N_EMBD * 4, N_EMBD * 4, gate.as_mut_ptr(), FF_EXP * 4, FF_EXP * TOP_K * 4));
        st.run(&call(FG_EXP_UP, x.as_ptr(), N_EMBD * 4, N_EMBD * 4, up.as_mut_ptr(), FF_EXP * 4, FF_EXP * TOP_K * 4));
        for p in 0..nt * TOP_K {
            let o = p * FF_EXP;
            fg_vec_swiglu(FF_EXP as c_int, act.as_mut_ptr().add(o), gate.as_ptr().add(o), up.as_ptr().add(o));
        }
        st.run(&call(FG_EXP_DOWN, act.as_ptr(), FF_EXP * 4, FF_EXP * TOP_K * 4, down.as_mut_ptr(), N_EMBD * 4, N_EMBD * TOP_K * 4));
    }
    assert!(st.take_error().is_none());
    down
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// F87: for every chunk width the per-token op takes (1..=15 on a 160-slot pool), the per-token op and the batched op
/// (8-token tiles on and off, the 4-pair gemm on and off) give the same bits for the same tokens.
#[test]
fn f87_per_token_and_batched_expert_ops_are_bitwise_equal() {
    const NE: usize = 64;
    let Some((mut st, dir)) = synth_stream("f87", NE, 160) else { return };
    let cap = st.per_token_cap();
    assert_eq!(cap, 15);
    let mut rnd = rng(0x5eed_f087_0000_0001);
    let mut checked = 0;
    for nt in 1..=cap {
        let ids: Vec<i32> = (0..nt).flat_map(|_| pick(&mut rnd, 0, NE)).collect();
        let xs: Vec<f32> = (0..nt * N_EMBD).map(|i| if i % 977 == 0 { 300.0 } else { ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect();
        let want = per_token_op(&st, 1, &ids, &xs);
        assert!(want.iter().all(|v| v.is_finite()) && want.iter().any(|v| *v != 0.0));
        for (tile, gemm) in [(false, false), (false, true), (true, true), (true, false)] {
            st.batch_tile = tile;
            st.batch_gemm = gemm;
            let got = batch_op(&st, 1, &ids, &xs);
            let diff = bits(&got).iter().zip(bits(&want)).filter(|(a, b)| *a != b).count();
            assert_eq!(diff, 0, "nt {nt} tile {tile} gemm {gemm}: {diff} floats differ between the batched and the per-token op");
            checked += 1;
        }
    }
    eprintln!("F87 expert ops: per-token == batched, bitwise, at widths 1..={cap} x 4 batched variants ({checked} checks)");
    drop(st);
    std::fs::remove_dir_all(&dir).ok();
}

/// F87, dense matmuls (evidence; ggml's mul_mat is the known width-dependent component, flash_split_test.inc): the
/// table of columns equal to their 1-column result per width 1..16 for Q6_K and Q8_0 is printed for the record.
/// Asserted: a 1-column call is reproducible and every difference is small (rounding class, not a wrong kernel).
#[test]
fn f87_mul_mat_columns_vs_one_column_evidence() {
    let t = crate::graph::selftest_mm_cols(4, 16);
    for (name, row) in ["q6_K", "q8_0"].iter().zip(t.iter()) {
        let line: Vec<String> = row.iter().enumerate().map(|(w, (same, d))| format!("w{}:{}/{} ({d:.1e})", w + 1, same, w + 1)).collect();
        eprintln!("F87 mul_mat {name} columns bitwise vs 1-column: {}", line.join(" "));
        assert_eq!(row[0].0, 1, "{name}: a 1-column call must reproduce itself");
        for (w, (_, d)) in row.iter().enumerate() {
            assert!(d.is_finite() && *d < 1.0, "{name} width {}: max |diff| {d}", w + 1);
        }
    }
}

/// F123 next, synthetic replay: hot experts (LFU count >= 2 from earlier ops), then one batched op over far more
/// distinct experts than the pool holds. With `pf_protect_hot` the op reads exactly what order 1 reads, gives the
/// same bits, and every protected hot expert it used is still resident afterwards (the next decode token's
/// experts); without it the count is reported for comparison.
#[test]
fn f123_protect_hot_keeps_hot_residents_same_reads_same_bits() {
    const NE: usize = 120;
    const Q: usize = 40;
    let mut rnd = rng(0x0f12_3a11_5eed_0003);
    // "decode": 3 ops over the hot experts 0..10 (each op +1 LFU count per expert), then one op over 10..40
    let hot_ops: Vec<Vec<i32>> = (0..3).map(|_| (0..4).flat_map(|_| pick(&mut rnd, 0, 10)).collect()).collect();
    let cold_op: Vec<i32> = (0..8).flat_map(|_| pick(&mut rnd, 10, 30)).collect();
    // the prompt: 40 tokens over all 120 experts (overflows the 40-slot pool several times)
    let nt = 40usize;
    let ids: Vec<i32> = (0..nt).flat_map(|_| pick(&mut rnd, 0, NE)).collect();
    let xr = |rnd: &mut dyn FnMut() -> u64, n: usize| (0..n * N_EMBD).map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).collect::<Vec<f32>>();
    let xh: Vec<Vec<f32>> = hot_ops.iter().map(|o| xr(&mut rnd, o.len() / TOP_K)).collect();
    let xc = xr(&mut rnd, cold_op.len() / TOP_K);
    let xs = xr(&mut rnd, nt);
    let used: std::collections::BTreeSet<i32> = ids.iter().copied().collect();
    let hot_used: Vec<i32> = (0..10).filter(|e| used.contains(e)).collect();
    let mut rows = Vec::new();
    for (name, protect) in [("order1", false), ("order1+protect", true)] {
        let Some((mut st, dir)) = synth_stream(&format!("f123p-{name}"), NE, Q) else { return };
        st.cache.policy = crate::cache::Policy::Lfu;
        st.pf_order = true;
        st.pf_protect_hot = protect;
        for (o, x) in hot_ops.iter().zip(&xh) {
            batch_op(&st, 0, o, x);
        }
        batch_op(&st, 0, &cold_op, &xc);
        // SAFETY: control thread, no op running.
        let hot_before = hot_used.iter().filter(|&&e| unsafe { st.cache.resident(0, e as usize) && st.cache.lfu_count(0, e as usize) >= 2 }).count();
        let r0 = st.stats().read_bytes;
        let out = batch_op(&st, 0, &ids, &xs);
        let reads = (st.stats().read_bytes - r0) / st.cache.layout.geom[0].slot as u64;
        let kept = hot_used.iter().filter(|&&e| unsafe { st.cache.resident(0, e as usize) }).count();
        let protected = st.pf_protect_hot_n.load(Relaxed) as usize;
        eprintln!("F123 {name}: op reads {reads} experts; hot experts used by the op {} (resident + count >= 2 before: {hot_before}); resident after the op {kept}; protected {protected}", hot_used.len());
        rows.push((reads, kept, protected, hot_before, bits(&out)));
        drop(st);
        std::fs::remove_dir_all(&dir).ok();
    }
    let (o1, pr) = (&rows[0], &rows[1]);
    assert!(o1.4 == pr.4, "protection changed the op's output bits");
    assert_eq!(pr.0, o1.0, "protection must not change the op's reads");
    assert!(pr.3 > 0, "the replay must start with hot residents (else it proves nothing)");
    assert_eq!(pr.2, pr.3.min(Q / 4), "every hot resident the op uses is protected (cap: a quarter of the pool)");
    assert!(pr.1 >= pr.2, "a protected hot expert was evicted: {} resident of {} protected", pr.1, pr.2);
    assert!(pr.1 >= o1.1, "protection kept fewer hot experts than order 1: {} vs {}", pr.1, o1.1);
}

/// F123: without `pf_order` the switch does nothing (protection is defined on the residents-first order only), and a
/// non-overflowing op protects nothing.
#[test]
fn f123_protect_hot_only_in_overflowing_order1_ops() {
    let Some((mut st, dir)) = synth_stream("f123p-small", 120, 40) else { return };
    st.cache.policy = crate::cache::Policy::Lfu;
    st.pf_order = true;
    st.pf_protect_hot = true;
    let mut rnd = rng(7);
    let small: Vec<i32> = (0..3).flat_map(|_| pick(&mut rnd, 0, 12)).collect();
    let x: Vec<f32> = vec![0.25; 3 * N_EMBD];
    for _ in 0..3 {
        batch_op(&st, 0, &small, &x);
    }
    assert_eq!(st.pf_protect_hot_n.load(Relaxed), 0, "12 experts fit a 40-slot pool: nothing to protect");
    st.pf_order = false;
    let big: Vec<i32> = (0..30).flat_map(|_| pick(&mut rnd, 0, 120)).collect();
    batch_op(&st, 0, &big, &vec![0.25; 30 * N_EMBD]);
    assert_eq!(st.pf_protect_hot_n.load(Relaxed), 0, "pf_order off: the switch is inert");
    drop(st);
    std::fs::remove_dir_all(&dir).ok();
}

/// F127-3: the decision carries the controller's guard mode; F118 terms pinned (16 GiB at 16K needs 26.125 GiB).
#[test]
fn swap_guard_mode_and_f118_terms() {
    use crate::pf_plan::*;
    let inp = AdmitInputs::measured(26.2, 16384, None);
    assert!((inp.need_gib(16.0) - 26.125).abs() < 1e-9, "F118: 16 + 4.825 + 3.50 + 0.80 + 1.0");
    assert!((inp.need_gib(14.0) - 24.125).abs() < 1e-9);
    let d = cache_select(&inp, None);
    assert_eq!((d.cache_gib, d.swap_guard_mode()), (Some(16.0), "memavailable"));
    assert!(d.status_json().contains("\"swap_guard\":\"memavailable\""));
    let d = cache_select(&AdmitInputs::measured(10.0, 16384, None), None);
    assert_eq!((d.cache_gib, d.swap_guard_mode()), (None, "refuse"));
    // an explicit 20 admitted by the MemTotal rule (budget >= 0, swap covers the overflow)
    let (d, _) = cache_select_swap_tolerant(&AdmitInputs::measured(25.5, 16384, None), Some(20.0), 30.72, 2.0, 16.0);
    assert_eq!((d.cache_gib, d.swap_guard_mode()), (Some(20.0), "swap-tolerant"));
    assert!(d.status_json().contains("\"swap_guard\":\"swap-tolerant\""));
}
