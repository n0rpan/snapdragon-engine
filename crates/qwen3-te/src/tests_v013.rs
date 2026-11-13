//! v0.1.3 lane H (TE-CPU-674): the production CPU arm's speed-ups are bitwise the reference paths.

use super::*;
use crate::attn_gqa::attention_gqa;

fn rand_vec(r: &mut Rng, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| r.f() * s).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// (heads, kv_heads, hd, n, n_real): GQA groups 1/2/4/8 take the grouped kernel, 3 the fallback;
/// hd 20 has a non-multiple-of-8 tail; 32/8/128 is the Qwen3-4B shape.
const SHAPES: [(usize, usize, usize, usize, usize); 6] =
    [(4, 2, 16, 37, 37), (8, 2, 16, 41, 13), (32, 8, 128, 70, 23), (4, 4, 24, 9, 5), (8, 1, 20, 33, 30), (6, 2, 16, 17, 11)];

#[test]
fn gqa_attention_is_bitwise_the_reference() {
    let mut r = Rng(0x7e5);
    for &(heads, kvh, hd, n, n_real) in &SHAPES {
        let q = rand_vec(&mut r, n * heads * hd, 2.0);
        let k = rand_vec(&mut r, n * kvh * hd, 2.0);
        let v = rand_vec(&mut r, n * kvh * hd, 1.0);
        // real tokens first, then pad (klein), plus one masked hole inside the real part
        let mut mask: Vec<bool> = (0..n).map(|i| i < n_real).collect();
        if n_real > 3 {
            mask[2] = false;
        }
        let mut want = vec![0f32; n * heads * hd];
        cpu::attention(&q, &k, &v, &mask, n, heads, kvh, hd, &mut want, 3);
        for threads in [1, 2, 5, 16] {
            let mut got = vec![f32::NAN; n * heads * hd];
            attention_gqa(&q, &k, &v, &mask, n, heads, kvh, hd, &mut got, threads);
            assert!(bits(&got) == bits(&want), "heads {heads} kv {kvh} hd {hd} n {n} threads {threads}: bits differ");
        }
    }
}

#[test]
fn gqa_attention_large_logits_match_reference() {
    // big scores stress the max subtraction / exp path; still bitwise
    let mut r = Rng(0xbeef);
    let (heads, kvh, hd, n) = (8, 2, 32, 25);
    let q = rand_vec(&mut r, n * heads * hd, 30.0);
    let k = rand_vec(&mut r, n * kvh * hd, 30.0);
    let v = rand_vec(&mut r, n * kvh * hd, 1.0);
    let mask = vec![true; n];
    let (mut a, mut b) = (vec![0f32; n * heads * hd], vec![0f32; n * heads * hd]);
    cpu::attention(&q, &k, &v, &mask, n, heads, kvh, hd, &mut a, 2);
    attention_gqa(&q, &k, &v, &mask, n, heads, kvh, hd, &mut b, 7);
    assert!(bits(&a) == bits(&b));
}

#[test]
fn ggml_shared_activation_projection_is_bitwise_per_projection() {
    let st = Store::new(tiny_cfg(), 47);
    let w = st.weights();
    let c = &w.cfg;
    let mut r = Rng(0x51);
    for threads in [1, 3] {
        let g = crate::ggml::GgmlLinears::new(&w, threads).unwrap();
        // 43 rows: 10 interleaved groups + a 3-row gemv tail; 4: groups only; 2: tail only
        for n in [43usize, 4, 2] {
            let x = rand_vec(&mut r, n * c.hidden, 1.5);
            let xf = rand_vec(&mut r, n * c.ff, 1.5);
            for (l, kinds, inp) in [
                (0, vec![Kind::Q, Kind::K, Kind::V], &x),
                (3, vec![Kind::Gate, Kind::Up], &x),
                (2, vec![Kind::Down], &xf),
            ] {
                let mut shared = vec![Vec::new(); kinds.len()];
                g.project(l, &kinds, inp, n, &mut shared).unwrap();
                for (i, &kd) in kinds.iter().enumerate() {
                    let mut one = Vec::new();
                    crate::ggml::gemm_one(&g, l, kd, inp, n, &mut one).unwrap();
                    assert!(bits(&shared[i]) == bits(&one), "layer {l} {kd:?} n {n} threads {threads}");
                }
            }
        }
    }
}

#[test]
fn ggml_encoder_with_grouped_attention_is_thread_invariant() {
    // whole graph (cpu-ggml linears with shared activations + grouped attention)
    let st = Store::new(tiny_cfg(), 53);
    let w = st.weights();
    let spec = tiny_spec();
    let e = input(29, &spec, 40);
    let g = crate::ggml::GgmlLinears::new(&w, 2).unwrap();
    let a = run(&w, &spec, &e, &g, 2);
    let b = run(&w, &spec, &e, &g, 6);
    assert!(bits(&a) == bits(&b), "thread count changed bits through the grouped attention");
    assert!(a.iter().all(|v| v.is_finite()));
}
