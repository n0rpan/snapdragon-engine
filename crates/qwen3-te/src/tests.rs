//! Model-free tests: synthetic small Qwen3 weights, an independent f64 forward written separately
//! from `encoder.rs`, the oxn CPU emulator for the NPU path.

use crate::config::{KleinSpec, Qwen3Config};
use crate::cpu;
use crate::encoder::{CpuLinears, CpuMode, Encoder, Linears};
use crate::metrics;
use crate::npu::{NpuLinears, NpuOpts};
use crate::prompt::{self, Encoded, PromptError, Tokenize};
use crate::weights::{Kind, Layer, Mat, TeWeights};
use kernels_sys::QType;
use oxn::linear::WeightRows;
use oxn::wfmt::WFmt;
use std::borrow::Cow;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Reference Q8_0 quantizer (ggml `quantize_row_q8_0_ref` layout: fp16 d, 32 x i8).
fn q8_bytes(x: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(x.len() / 32 * 34);
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&oxn::tiles::f32_to_half(d).to_le_bytes());
        out.extend(b.iter().map(|v| (v * id).round() as i8 as u8));
    }
    out
}

fn tiny_cfg() -> Qwen3Config {
    Qwen3Config { hidden: 64, n_layers: 5, n_heads: 4, n_kv_heads: 2, head_dim: 16, ff: 96, eps: 1e-6, rope_theta: 10000.0, vocab: 40 }
}

fn tiny_spec() -> KleinSpec {
    KleinSpec { out_layers: vec![1, 3, 4], min_len: 40, ..KleinSpec::klein() }
}

/// Owner of the synthetic Q8_0 bytes; `weights()` borrows them.
struct Store {
    cfg: Qwen3Config,
    embd: Vec<f32>,
    mats: Vec<Vec<Vec<u8>>>,
    norms: Vec<[Vec<f32>; 4]>,
}

impl Store {
    fn new(cfg: Qwen3Config, seed: u64) -> Store {
        let mut r = Rng(seed);
        let embd = (0..cfg.vocab * cfg.hidden).map(|_| r.f()).collect();
        let mut mats = Vec::new();
        let mut norms = Vec::new();
        for _ in 0..cfg.n_layers {
            let mut lm = Vec::new();
            for kd in Kind::ALL {
                let (o, k) = crate::weights::expected_shape(&cfg, kd);
                let s = 1.0 / (k as f32).sqrt();
                let w: Vec<f32> = (0..o * k).map(|_| r.f() * s * 1.5).collect();
                lm.push(q8_bytes(&w));
            }
            mats.push(lm);
            let mut nv = |n: usize| (0..n).map(|_| 1.0 + 0.2 * r.f()).collect::<Vec<f32>>();
            norms.push([nv(cfg.hidden), nv(cfg.head_dim), nv(cfg.head_dim), nv(cfg.hidden)]);
        }
        Store { cfg, embd, mats, norms }
    }

    fn weights(&self) -> TeWeights<'_> {
        let c = &self.cfg;
        let layers = (0..c.n_layers)
            .map(|l| Layer {
                attn_norm: self.norms[l][0].clone(),
                q_norm: self.norms[l][1].clone(),
                k_norm: self.norms[l][2].clone(),
                ffn_norm: self.norms[l][3].clone(),
                mats: Kind::ALL
                    .iter()
                    .enumerate()
                    .map(|(i, &kd)| {
                        let (o, k) = crate::weights::expected_shape(c, kd);
                        Mat::Quant { ty: QType::Q8_0, o, k, bytes: &self.mats[l][i] }
                    })
                    .collect(),
            })
            .collect();
        TeWeights { cfg: c.clone(), embd: Mat::F32 { o: c.vocab, k: c.hidden, data: Cow::Borrowed(&self.embd) }, layers }
    }
}

fn input(n_real: usize, spec: &KleinSpec, vocab: usize) -> Encoded {
    let ids: Vec<u32> = (0..n_real).map(|i| ((i * 7 + 3) % (vocab - 1)) as u32).collect();
    let mut s = spec.clone();
    s.pad_id = (vocab - 1) as u32;
    prompt::pad(ids, &s)
}

/// Independent forward in f64: plain loops, `-inf` masking, `powf` RoPE angles.
fn naive(w: &TeWeights, spec: &KleinSpec, e: &Encoded) -> Vec<f32> {
    let c = &w.cfg;
    let (n, hdim, hd) = (e.tokens.len(), c.hidden, c.head_dim);
    let row = |m: &Mat, r: usize| {
        let mut v = vec![0f32; m.k()];
        m.read_row(r, &mut v).unwrap();
        v.into_iter().map(|x| x as f64).collect::<Vec<f64>>()
    };
    let lin = |m: &Mat, x: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let rows: Vec<Vec<f64>> = (0..m.o()).map(|r| row(m, r)).collect();
        x.iter().map(|xt| rows.iter().map(|wr| wr.iter().zip(xt).map(|(a, b)| a * b).sum()).collect()).collect()
    };
    let norm = |x: &[f64], g: &[f32]| -> Vec<f64> {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let s = 1.0 / (ms + c.eps as f64).sqrt();
        x.iter().zip(g).map(|(v, &g)| v * s * g as f64).collect()
    };
    let rope = |v: &mut [f64], p: usize| {
        let half = hd / 2;
        for i in 0..half {
            let th = p as f64 * (c.rope_theta as f64).powf(-2.0 * i as f64 / hd as f64);
            let (a, b) = (v[i], v[i + half]);
            v[i] = a * th.cos() - b * th.sin();
            v[i + half] = a * th.sin() + b * th.cos();
        }
    };
    let mut x: Vec<Vec<f64>> = e.tokens.iter().map(|&t| row(&w.embd, t as usize)).collect();
    let mut outs = Vec::new();
    for l in 0..spec.layers_needed() {
        let lw = &w.layers[l];
        let h: Vec<Vec<f64>> = x.iter().map(|r| norm(r, &lw.attn_norm)).collect();
        let mut q = lin(lw.mat(Kind::Q), &h);
        let mut k = lin(lw.mat(Kind::K), &h);
        let v = lin(lw.mat(Kind::V), &h);
        for t in 0..n {
            for hh in 0..c.n_heads {
                let s = &mut q[t][hh * hd..(hh + 1) * hd];
                let nv = norm(s, &lw.q_norm);
                s.copy_from_slice(&nv);
                rope(s, t);
            }
            for hh in 0..c.n_kv_heads {
                let s = &mut k[t][hh * hd..(hh + 1) * hd];
                let nv = norm(s, &lw.k_norm);
                s.copy_from_slice(&nv);
                rope(s, t);
            }
        }
        let g = c.n_heads / c.n_kv_heads;
        let mut att = vec![vec![0f64; c.q_dim()]; n];
        for t in 0..n {
            for hh in 0..c.n_heads {
                let kh = hh / g;
                let sc: Vec<f64> = (0..n)
                    .map(|j| {
                        if j > t || !e.mask[j] {
                            f64::NEG_INFINITY
                        } else {
                            (0..hd).map(|d| q[t][hh * hd + d] * k[j][kh * hd + d]).sum::<f64>() / (hd as f64).sqrt()
                        }
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                let z: f64 = ex.iter().sum();
                for d in 0..hd {
                    att[t][hh * hd + d] = (0..n).map(|j| ex[j] / z * v[j][kh * hd + d]).sum();
                }
            }
        }
        let o = lin(lw.mat(Kind::O), &att);
        for t in 0..n {
            for d in 0..hdim {
                x[t][d] += o[t][d];
            }
        }
        let h2: Vec<Vec<f64>> = x.iter().map(|r| norm(r, &lw.ffn_norm)).collect();
        let gt = lin(lw.mat(Kind::Gate), &h2);
        let up = lin(lw.mat(Kind::Up), &h2);
        let a: Vec<Vec<f64>> = gt.iter().zip(&up).map(|(g, u)| g.iter().zip(u).map(|(g, u)| g / (1.0 + (-g).exp()) * u).collect()).collect();
        let d = lin(lw.mat(Kind::Down), &a);
        for t in 0..n {
            for j in 0..hdim {
                x[t][j] += d[t][j];
            }
        }
        if spec.out_layers.contains(&(l + 1)) {
            outs.push(x.clone());
        }
    }
    let mut r = Vec::with_capacity(n * hdim * outs.len());
    for t in 0..n {
        for o in &outs {
            r.extend(o[t].iter().map(|&v| v as f32));
        }
    }
    r
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut s) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += ((x - y) as f64).powi(2);
        s += (*y as f64).powi(2);
    }
    (d / s).sqrt()
}

fn run(w: &TeWeights, spec: &KleinSpec, e: &Encoded, lin: &dyn Linears, threads: usize) -> Vec<f32> {
    Encoder::new(w, spec.clone(), threads).unwrap().encode(lin, e).unwrap().0
}

#[test]
fn cpu_f32_graph_matches_independent_f64_forward() {
    let st = Store::new(tiny_cfg(), 7);
    let w = st.weights();
    let spec = tiny_spec();
    let e = input(13, &spec, w.cfg.vocab);
    let got = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 3, mode: CpuMode::F32 }, 3);
    let want = naive(&w, &spec, &e);
    assert_eq!(got.len(), 40 * 3 * 64);
    let r = rel(&got, &want);
    assert!(r < 2e-6, "rel {r}");
}

#[test]
fn output_does_not_depend_on_thread_count() {
    let st = Store::new(tiny_cfg(), 8);
    let w = st.weights();
    let spec = tiny_spec();
    let e = input(11, &spec, w.cfg.vocab);
    for mode in [CpuMode::F32, CpuMode::Q8] {
        let a = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 1, mode }, 1);
        let b = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 5, mode }, 5);
        assert!(a == b, "{mode:?}: thread count changed bits");
    }
}

#[test]
fn real_rows_ignore_padding_and_pad_rows_are_distinct() {
    let st = Store::new(tiny_cfg(), 9);
    let w = st.weights();
    let lin = CpuLinears { w: &w, threads: 2, mode: CpuMode::F32 };
    let s40 = tiny_spec();
    let s64 = KleinSpec { min_len: 64, ..tiny_spec() };
    let (a, b) = (run(&w, &s40, &input(9, &s40, 40), &lin, 2), run(&w, &s64, &input(9, &s64, 40), &lin, 2));
    let ctx = 3 * 64;
    assert_eq!(&a[..9 * ctx], &b[..9 * ctx], "real rows changed with the pad count");
    // pads attend to the real tokens but carry their own position: not all identical
    assert_ne!(&a[20 * ctx..21 * ctx], &a[30 * ctx..31 * ctx]);
    // every pad row of the 40-run matches the same position in the 64-run (causal)
    assert_eq!(&a[..40 * ctx], &b[..40 * ctx]);
}

#[test]
fn layer_slots_are_the_requested_blocks() {
    let st = Store::new(tiny_cfg(), 10);
    let w = st.weights();
    let lin = CpuLinears { w: &w, threads: 2, mode: CpuMode::F32 };
    let e = input(10, &tiny_spec(), 40);
    let full = run(&w, &tiny_spec(), &e, &lin, 2);
    let only3 = run(&w, &KleinSpec { out_layers: vec![3], ..tiny_spec() }, &e, &lin, 2);
    for t in 0..40 {
        assert_eq!(&full[t * 192 + 64..t * 192 + 128], &only3[t * 64..(t + 1) * 64]);
    }
}

#[test]
fn q8_emulation_is_close_to_f32_and_exact_against_integer_reference() {
    let st = Store::new(tiny_cfg(), 11);
    let w = st.weights();
    let spec = tiny_spec();
    let e = input(17, &spec, 40);
    let f = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 2, mode: CpuMode::F32 }, 2);
    let q = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 2, mode: CpuMode::Q8 }, 2);
    let r = rel(&q, &f);
    assert!(r > 0.0 && r < 0.03, "q8 vs f32 rel {r}");
    // one GEMM against a per-element integer reference
    let m = w.layers[0].mat(Kind::Up);
    let mut rng = Rng(5);
    let x: Vec<f32> = (0..3 * m.k()).map(|_| rng.f() * 3.0).collect();
    let mut y = vec![0f32; 3 * m.o()];
    cpu::gemm_q8(m, &x, 3, &mut y, 2).unwrap();
    let nb = m.k() / 32;
    let wq = m.q8_0().unwrap();
    for t in 0..3 {
        let mut xd = vec![0f32; nb];
        let mut xq = vec![0i8; m.k()];
        cpu::quantize_q8_0(&x[t * m.k()..(t + 1) * m.k()], &mut xd, &mut xq);
        for o in [0, 17, m.o() - 1] {
            let mut acc = 0f32;
            for b in 0..nb {
                let blk = &wq[(o * nb + b) * 34..(o * nb + b + 1) * 34];
                let dw = oxn::tiles::half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let is: i32 = (0..32).map(|i| blk[2 + i] as i8 as i32 * xq[b * 32 + i] as i32).sum();
                acc += is as f32 * (dw * xd[b]);
            }
            assert_eq!(acc, y[t * m.o() + o]);
        }
    }
}

#[test]
fn quantize_q8_0_rounds_ties_to_even_and_handles_zero() {
    let mut x = vec![0f32; 32];
    let (mut d, mut q) = (vec![0f32; 1], vec![0i8; 32]);
    cpu::quantize_q8_0(&x, &mut d, &mut q);
    assert_eq!((d[0], q.iter().all(|&v| v == 0)), (0.0, true));
    x[0] = 127.0;
    x[1] = 2.5;
    x[2] = -3.5;
    cpu::quantize_q8_0(&x, &mut d, &mut q);
    assert_eq!((d[0], q[0], q[1], q[2]), (1.0, 127, 2, -4));
}

fn npu_opts(weights: WFmt, budget: usize) -> NpuOpts {
    NpuOpts { budget_bytes: budget, weights, kc_max: 0, pack_threads: 2, io_threads: 3 }
}

#[test]
fn npu_emulator_fp16_and_w8_match_cpu_reference() {
    let st = Store::new(tiny_cfg(), 12);
    let w = st.weights();
    let spec = tiny_spec();
    // 45 real tokens: n = 45, one full 32-token tile plus a 13-token tail
    let e = input(45, &spec, 40);
    let want = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 2, mode: CpuMode::F32 }, 2);
    for fmt in [WFmt::Fp16, WFmt::W8] {
        let be = Box::new(oxn::emu::Emu::new(8 << 20, 2));
        let lin = NpuLinears::new(&w, be, npu_opts(fmt, 64 << 20), 2).unwrap();
        assert_eq!(lin.placed_count(), 5 * 7, "{fmt:?}: every linear of the 5 loaded blocks placed");
        let got = run(&w, &spec, &e, &lin, 2);
        let r = rel(&got, &want);
        assert!(r < 1e-2, "{fmt:?}: rel {r}");
        let rep = metrics::report(&got, &want, 45, 45, 64, &spec.out_layers).unwrap();
        let (ok, why) = metrics::gate(&rep, 0.9999, 1e-2);
        assert!(ok, "{fmt:?}: {why:?}");
        let s = lin.summary();
        assert_eq!(s["cpu_calls"], 0);
        assert_eq!(s["npu_calls"], 4 * 7);
    }
}

#[test]
fn npu_budget_overflow_and_fp16_range_fall_back_to_cpu_counted() {
    let st = Store::new(tiny_cfg(), 13);
    let mut w = st.weights();
    let spec = tiny_spec();
    let e = input(12, &spec, 40);
    // tiny budget: some linears stay on the CPU; results still agree with the reference
    let want = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 2, mode: CpuMode::F32 }, 2);
    let be = Box::new(oxn::emu::Emu::new(8 << 20, 1));
    let lin = NpuLinears::new(&w, be, npu_opts(WFmt::Fp16, 40 << 10), 2).unwrap();
    assert!(lin.placed_count() > 0 && lin.placed_count() < 28, "placed {}", lin.placed_count());
    let got = run(&w, &spec, &e, &lin, 2);
    assert!(rel(&got, &want) < 1e-2);
    assert!(lin.summary()["cpu_calls"].as_u64().unwrap() > 0);
    drop(lin);
    // an ffn_norm gain of 1e5 pushes gate/up inputs past the fp16 range in block 0
    w.layers[0].ffn_norm.iter_mut().for_each(|g| *g = 1e5);
    let be = Box::new(oxn::emu::Emu::new(8 << 20, 1));
    let lin = NpuLinears::new(&w, be, npu_opts(WFmt::Fp16, 64 << 20), 2).unwrap();
    let _ = Encoder::new(&w, spec.clone(), 2).unwrap().encode(&lin, &e);
    assert!(lin.summary()["range_fallbacks"].as_u64().unwrap() >= 2);
}

struct FakeTok;
impl Tokenize for FakeTok {
    /// one id per byte (special strings become one id), so piece boundaries are visible
    fn encode_piece(&self, s: &str) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        let mut rest = s;
        while !rest.is_empty() {
            if let Some(sp) = prompt::SD_SPECIALS.iter().find(|sp| rest.starts_with(*sp)) {
                out.push(1000 + prompt::SD_SPECIALS.iter().position(|x| x == sp).unwrap() as u32);
                rest = &rest[sp.len()..];
            } else {
                out.push(rest.as_bytes()[0] as u32);
                rest = &rest[1..];
            }
        }
        Ok(out)
    }
}

#[test]
fn klein_tokens_template_pieces_padding_and_refusals() {
    let spec = KleinSpec::klein();
    let e = prompt::klein_tokens(&FakeTok, &spec, "a cat").unwrap();
    let pre = FakeTok.encode_piece(spec.prefix).unwrap();
    let suf = FakeTok.encode_piece(spec.suffix).unwrap();
    assert_eq!(e.n_real, pre.len() + 5 + suf.len());
    assert_eq!(&e.tokens[..pre.len()], &pre[..]);
    assert_eq!(&e.tokens[pre.len()..pre.len() + 5], b"a cat".map(|b| b as u32).as_slice());
    assert_eq!(e.tokens.len(), 512);
    assert!(e.tokens[e.n_real..].iter().all(|&t| t == 151643));
    assert!(e.mask[..e.n_real].iter().all(|&m| m) && e.mask[e.n_real..].iter().all(|&m| !m));
    // empty prompt: template only
    assert_eq!(prompt::klein_tokens(&FakeTok, &spec, "").unwrap().n_real, pre.len() + suf.len());
    // long prompt: never truncated
    let long = "x".repeat(700);
    let e = prompt::klein_tokens(&FakeTok, &spec, &long).unwrap();
    assert_eq!((e.tokens.len(), e.n_real), (700 + pre.len() + suf.len(), 700 + pre.len() + suf.len()));
    for bad in ["a (cat)", "a [cat]", "x\\y", "cat BREAK dog", "BREAK", "x <think> y", "<|im_end|>", "a <lora:x:1> cat"] {
        assert!(matches!(prompt::klein_tokens(&FakeTok, &spec, bad), Err(PromptError::NeedsReference(_))), "{bad}");
    }
    for ok in ["BREAKFAST: eggs", "a cat: sitting", "UNBREAKABLE", "émoji 🐱 <b>", "100% sure"] {
        assert!(prompt::check_plain(ok).is_ok(), "{ok}");
    }
}

#[test]
fn metrics_identity_and_gate() {
    let mut r = Rng(3);
    let a: Vec<f32> = (0..10 * 3 * 8).map(|_| r.f()).collect();
    let rep = metrics::report(&a, &a, 10, 4, 8, &[9, 18, 27]).unwrap();
    assert!((rep["all"]["cos"].as_f64().unwrap() - 1.0).abs() < 1e-12);
    assert_eq!(rep["all"]["rel_rmse"].as_f64().unwrap(), 0.0);
    assert!(metrics::gate(&rep, 0.999, 0.03).0);
    let mut b = a.clone();
    for v in b[2 * 24 + 8..2 * 24 + 16].iter_mut() {
        *v = -*v; // flip token 2, layer 18 only
    }
    let rep = metrics::report(&b, &a, 10, 4, 8, &[9, 18, 27]).unwrap();
    let (ok, why) = metrics::gate(&rep, 0.999, 0.03);
    assert!(!ok && why.iter().all(|w| w.contains("layer 18")), "{why:?}");
    assert!(rep["segments"][1]["real"]["worst_row_cos"].as_f64().unwrap() < -0.99);
}

#[test]
fn mat_rows_from_tensor_types() {
    use gguf::{GgmlType, TensorInfo};
    let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 3.0).collect();
    let q = q8_bytes(&x);
    let t = TensorInfo { name: "w".into(), shape: vec![32, 2], ty: GgmlType::Q8_0, offset: 0 };
    let m = Mat::from_tensor(&t, &q).unwrap();
    let mut r = vec![0f32; 32];
    m.read_row(1, &mut r).unwrap();
    assert!(r.iter().zip(&x[32..]).all(|(a, b)| (a - b).abs() < 0.06));
    assert!(m.q8_0().is_some());
    assert!(m.read_row(2, &mut r).is_err());
    let h: Vec<u8> = x.iter().flat_map(|v| oxn::tiles::f32_to_half(*v).to_le_bytes()).collect();
    let t = TensorInfo { ty: GgmlType::F16, ..t };
    let m = Mat::from_tensor(&t, &h).unwrap();
    m.read_row(0, &mut r).unwrap();
    assert_eq!(&r[..], &x[..32]);
    assert!(Mat::from_tensor(&t, &h[..10]).is_err());
}

/// Real tokenizer (Codex): `QWEN3_TE_GGUF=.../Qwen3-4B-Q8_0.gguf cargo test -p qwen3-te -- --ignored`.
/// The template pieces must give the Qwen3 ids sd.cpp's built-in Qwen2 tokenizer gives.
#[test]
#[ignore]
fn real_tokenizer_template_ids() {
    let p = std::env::var("QWEN3_TE_GGUF").expect("QWEN3_TE_GGUF");
    let tok = tokenizer::Tokenizer::from_gguf(&p).unwrap();
    let spec = KleinSpec::klein();
    assert_eq!(tok.encode_piece(spec.prefix).unwrap(), vec![151644, 872, 198]);
    assert_eq!(tok.encode_piece(spec.suffix).unwrap(), vec![151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]);
    let g = gguf::Gguf::open(&p).unwrap();
    let c = Qwen3Config::from_header(&g.header).unwrap();
    assert_eq!(c, Qwen3Config::qwen3_4b());
    let w = TeWeights::from_gguf(&g, 27).unwrap();
    assert_eq!(w.layers.len(), 27);
    assert_eq!(w.linear_fp16_bytes(), 27 * 2 * (2 * 2560 * 4096 + 2 * 2560 * 1024 + 3 * 2560 * 9728) as u64);
}

/// Minimal GGUF v3 writer: string/u32/f32 metadata, F32 and Q8_0 tensors.
fn gguf_file(path: &std::path::Path, kv: &[(&str, KvVal)], tensors: &[(String, Vec<u64>, u32, Vec<u8>)]) {
    let mut b: Vec<u8> = Vec::new();
    let s = |b: &mut Vec<u8>, x: &str| {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    };
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((kv.len() as u64).to_le_bytes());
    for (k, v) in kv {
        s(&mut b, k);
        match v {
            KvVal::U32(x) => {
                b.extend(4u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            KvVal::F32(x) => {
                b.extend(6u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            KvVal::Str(x) => {
                b.extend(8u32.to_le_bytes());
                s(&mut b, x);
            }
        }
    }
    let mut off = 0u64;
    let mut offs = Vec::new();
    for (name, dims, ty, data) in tensors {
        s(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(ty.to_le_bytes());
        b.extend(off.to_le_bytes());
        offs.push(off);
        off = (off + data.len() as u64).div_ceil(32) * 32;
    }
    while b.len() % 32 != 0 {
        b.push(0);
    }
    let base = b.len();
    for ((_, _, _, data), o) in tensors.iter().zip(offs) {
        b.resize(base + o as usize, 0);
        b.extend(data);
    }
    std::fs::write(path, b).unwrap();
}

enum KvVal {
    U32(u32),
    F32(f32),
    Str(&'static str),
}

#[test]
fn gguf_loader_reads_names_shapes_and_config_like_llama_cpp_files() {
    let c = tiny_cfg();
    let st = Store::new(c.clone(), 21);
    let f32b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut t = vec![("token_embd.weight".to_string(), vec![c.hidden as u64, c.vocab as u64], 0u32, f32b(&st.embd))];
    for l in 0..c.n_layers {
        for (i, kd) in Kind::ALL.iter().enumerate() {
            let (o, k) = crate::weights::expected_shape(&c, *kd);
            t.push((format!("blk.{l}.{}.weight", kd.gguf()), vec![k as u64, o as u64], 8, st.mats[l][i].clone()));
        }
        for (j, nm) in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"].iter().enumerate() {
            t.push((format!("blk.{l}.{nm}.weight"), vec![st.norms[l][j].len() as u64], 0, f32b(&st.norms[l][j])));
        }
    }
    let kv = [
        ("general.architecture", KvVal::Str("qwen3")),
        ("qwen3.block_count", KvVal::U32(c.n_layers as u32)),
        ("qwen3.embedding_length", KvVal::U32(c.hidden as u32)),
        ("qwen3.feed_forward_length", KvVal::U32(c.ff as u32)),
        ("qwen3.attention.head_count", KvVal::U32(c.n_heads as u32)),
        ("qwen3.attention.head_count_kv", KvVal::U32(c.n_kv_heads as u32)),
        ("qwen3.attention.key_length", KvVal::U32(c.head_dim as u32)),
        ("qwen3.rope.freq_base", KvVal::F32(c.rope_theta)),
        ("qwen3.attention.layer_norm_rms_epsilon", KvVal::F32(c.eps)),
    ];
    let dir = std::env::temp_dir().join(format!("qwen3-te-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("tiny.gguf");
    gguf_file(&p, &kv, &t);
    let g = gguf::Gguf::open(&p).unwrap();
    assert_eq!(Qwen3Config::from_header(&g.header).unwrap(), c);
    let wf = TeWeights::from_gguf(&g, 4).unwrap();
    assert!(TeWeights::from_gguf(&g, 6).is_err());
    let wm = st.weights();
    let spec = tiny_spec();
    let e = input(14, &spec, c.vocab);
    let a = run(&wf, &spec, &e, &CpuLinears { w: &wf, threads: 2, mode: CpuMode::F32 }, 2);
    let b = run(&wm, &spec, &e, &CpuLinears { w: &wm, threads: 2, mode: CpuMode::F32 }, 2);
    assert!(a == b, "GGUF-loaded weights differ from the in-memory ones");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn o46_gate_passes_closer_candidates_only() {
    let mut r = Rng(4);
    let refv: Vec<f32> = (0..12 * 3 * 8).map(|_| r.f()).collect();
    let noisy = |amp: f32, r: &mut Rng| refv.iter().map(|v| v + amp * r.f()).collect::<Vec<f32>>();
    let inst = noisy(0.05, &mut r);
    let good = noisy(0.01, &mut r);
    let bad = noisy(0.2, &mut r);
    let rep = |x: &[f32]| metrics::report(x, &refv, 12, 5, 8, &[9, 18, 27]).unwrap();
    assert!(metrics::gate_vs_installed(&rep(&good), &rep(&inst), 1.0).0);
    let (ok, why) = metrics::gate_vs_installed(&rep(&bad), &rep(&inst), 1.0);
    assert!(!ok && why.len() == 6, "{why:?}");
    assert!(metrics::gate_vs_installed(&rep(&inst), &rep(&inst), 1.0).0, "equal distance passes");
}

#[test]
fn ggml_backend_matches_q8_emulation_and_is_thread_invariant() {
    let st = Store::new(tiny_cfg(), 31);
    let w = st.weights();
    let spec = tiny_spec();
    // 43 real tokens: 10 full 4-row groups plus a 3-row gemv tail
    let e = input(43, &spec, 40);
    let q = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 2, mode: CpuMode::Q8 }, 2);
    let f = run(&w, &spec, &e, &CpuLinears { w: &w, threads: 2, mode: CpuMode::F32 }, 2);
    let g1 = crate::ggml::GgmlLinears::new(&w, 1).unwrap();
    let g5 = crate::ggml::GgmlLinears::new(&w, 5).unwrap();
    let a = run(&w, &spec, &e, &g1, 1);
    let b = run(&w, &spec, &e, &g5, 5);
    assert!(a == b, "ggml: thread count changed bits");
    let rq = rel(&a, &q);
    assert!(rq < 1e-4, "ggml vs q8 emulation rel {rq}");
    assert!(rel(&a, &f) < 0.03);
    assert_eq!(g1.bytes(), st.mats.iter().flatten().map(|m| m.len()).sum::<usize>() - st.mats[4].iter().map(|m| m.len()).sum::<usize>() + st.mats[4].iter().map(|m| m.len()).sum::<usize>());
}

#[test]
fn encoder_identity_names_arith_file_and_size() {
    let dir = std::env::temp_dir().join(format!("qwen3-te-id-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("te.gguf");
    std::fs::write(&p, [0u8; 123]).unwrap();
    assert_eq!(crate::encoder_identity(&p, "cpu-ggml").unwrap(), format!("{};cpu-ggml;te.gguf;123", crate::ENCODER_VERSION));
    assert!(crate::encoder_identity(&p, "gpu").is_err());
    assert_ne!(crate::encoder_identity(&p, "npu-fp16").unwrap(), crate::encoder_identity(&p, "npu-w8").unwrap());
    std::fs::remove_dir_all(&dir).ok();
}

// v0.1.3 lane H tests live in their own file (conflict rule: new tests in new files)
#[path = "tests_v013.rs"]
mod v013;
