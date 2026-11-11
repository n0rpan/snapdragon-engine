//! Model-free tests for the whole-model reference. Tiny random klein-shaped models built as a
//! `MemSource` (the same tensor names as the klein GGUF).

use super::*;
use image_dit::fast_model::{FastModel, MemSource};
use image_dit::fast::FastConfig;
use image_dit::types::rel_rmse;
use image_dit::weights::quant_q8_0;

struct Rng(u64);

impl Rng {
    /// Uniform in [-1, 1).
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

#[derive(Clone, Copy)]
struct Tiny {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    mlp: usize,
    in_ch: usize,
    ctx: usize,
    out_ch: usize,
    n_double: usize,
    n_single: usize,
}

impl Tiny {
    /// The brief's shapes: hidden 64, heads 2, head_dim 32, mlp 192, in 16, context 48, out 16.
    fn standard(n_double: usize, n_single: usize) -> Tiny {
        Tiny { hidden: 64, heads: 2, head_dim: 32, mlp: 192, in_ch: 16, ctx: 48, out_ch: 16, n_double, n_single }
    }
}

fn put_f(src: &mut MemSource, name: &str, v: Vec<f32>, rows: usize, k: usize) {
    src.f.insert(name.to_string(), (v, rows, k));
}

fn put_q(src: &mut MemSource, name: &str, r: &mut Rng, rows: usize, k: usize) {
    let v = r.vec(rows * k, 1.0 / (k as f32).sqrt());
    src.q8.insert(name.to_string(), (quant_q8_0(&v, rows, k), rows, k));
}

fn put_norm(src: &mut MemSource, name: &str, r: &mut Rng, len: usize) {
    let v: Vec<f32> = (0..len).map(|_| 1.0 + 0.3 * r.next()).collect();
    put_f(src, name, v, 1, len);
}

/// Like `put_q` with every value multiplied by `g`.
fn put_qg(src: &mut MemSource, name: &str, r: &mut Rng, rows: usize, k: usize, g: f32) {
    let v: Vec<f32> = r.vec(rows * k, 1.0 / (k as f32).sqrt()).iter().map(|x| x * g).collect();
    src.q8.insert(name.to_string(), (quant_q8_0(&v, rows, k), rows, k));
}

fn scaled(v: Vec<f32>, g: f32) -> Vec<f32> {
    v.iter().map(|x| x * g).collect()
}

/// Gain applied to the modulation matrices and to the block output projections, so that the
/// blocks change the output by a large fraction (the default random init makes them small).
const GAIN: f32 = 8.0;

fn tiny_source(c: Tiny, seed: u64) -> MemSource {
    tiny_source_gain(c, seed, GAIN)
}

fn tiny_source_gain(c: Tiny, seed: u64, g: f32) -> MemSource {
    let mut r = Rng(seed.wrapping_mul(2654435761).wrapping_add(12345));
    let mut src = MemSource::default();
    let h = c.hidden;
    let m = c.mlp;
    let hd = c.head_dim;
    let fi = |k: usize| 1.0 / (k as f32).sqrt();
    let v = r.vec(h * c.in_ch, fi(c.in_ch));
    put_f(&mut src, "img_in.weight", v, h, c.in_ch);
    let v = r.vec(h * c.ctx, fi(c.ctx));
    put_f(&mut src, "txt_in.weight", v, h, c.ctx);
    let v = r.vec(h * TIME_DIM, fi(TIME_DIM));
    put_f(&mut src, "time_in.in_layer.weight", v, h, TIME_DIM);
    let v = r.vec(h * h, fi(h));
    put_f(&mut src, "time_in.out_layer.weight", v, h, h);
    for name in ["double_stream_modulation_img.lin.weight", "double_stream_modulation_txt.lin.weight"] {
        let v = scaled(r.vec(6 * h * h, fi(h)), g);
        put_f(&mut src, name, v, 6 * h, h);
    }
    let v = scaled(r.vec(3 * h * h, fi(h)), g);
    put_f(&mut src, "single_stream_modulation.lin.weight", v, 3 * h, h);
    let v = scaled(r.vec(2 * h * h, fi(h)), g);
    put_f(&mut src, "final_layer.adaLN_modulation.1.weight", v, 2 * h, h);
    let v = r.vec(c.out_ch * h, fi(h));
    put_f(&mut src, "final_layer.linear.weight", v, c.out_ch, h);
    for i in 0..c.n_double {
        for s in ["img", "txt"] {
            put_q(&mut src, &format!("double_blocks.{i}.{s}_attn.qkv.weight"), &mut r, 3 * h, h);
            put_qg(&mut src, &format!("double_blocks.{i}.{s}_attn.proj.weight"), &mut r, h, h, g);
            put_q(&mut src, &format!("double_blocks.{i}.{s}_mlp.0.weight"), &mut r, 2 * m, h);
            put_qg(&mut src, &format!("double_blocks.{i}.{s}_mlp.2.weight"), &mut r, h, m, g);
            put_norm(&mut src, &format!("double_blocks.{i}.{s}_attn.norm.query_norm.scale"), &mut r, hd);
            put_norm(&mut src, &format!("double_blocks.{i}.{s}_attn.norm.key_norm.scale"), &mut r, hd);
        }
    }
    for i in 0..c.n_single {
        put_q(&mut src, &format!("single_blocks.{i}.linear1.weight"), &mut r, 3 * h + 2 * m, h);
        put_qg(&mut src, &format!("single_blocks.{i}.linear2.weight"), &mut r, h, h + m, g);
        put_norm(&mut src, &format!("single_blocks.{i}.norm.query_norm.scale"), &mut r, hd);
        put_norm(&mut src, &format!("single_blocks.{i}.norm.key_norm.scale"), &mut r, hd);
    }
    src
}

/// Random RoPE table `[n][half][4]` from random angles: `[cos, -sin, sin, cos]`.
fn random_pe(r: &mut Rng, n: usize, half: usize) -> Vec<f32> {
    let mut pe = Vec::with_capacity(n * half * 4);
    for _ in 0..n * half {
        let a = 3.0 * r.next();
        let (s, c) = a.sin_cos();
        pe.extend_from_slice(&[c, -s, s, c]);
    }
    pe
}

fn random_input(c: Tiny, n_txt: usize, n_img: usize, seed: u64, timestep: f32) -> ModelInput {
    let mut r = Rng(seed ^ 0x9e37_79b9);
    let n = n_txt + n_img;
    ModelInput {
        n_img,
        n_txt,
        img: r.vec(n_img * c.in_ch, 1.0),
        txt: r.vec(n_txt * c.ctx, 1.0),
        timestep,
        pe: random_pe(&mut r, n, c.head_dim / 2),
    }
}

#[test]
fn threads_do_not_change_the_result_and_txt_cache_is_exact() {
    let c = Tiny::standard(1, 2);
    let mut m1 = RefModel::from_source(&tiny_source(c, 1), 1).unwrap();
    let mut m3 = RefModel::from_source(&tiny_source(c, 1), 3).unwrap();
    let input = random_input(c, 5, 7, 2, 0.7);
    let a = m1.forward(&input);
    let b = m3.forward(&input);
    assert_eq!(a, b, "threads 1 and 3 must agree bitwise");
    assert_eq!(a.len(), 7 * c.out_ch);
    // second call hits the txt cache and must give the same numbers
    assert_eq!(m3.forward(&input), b, "cached txt embedding changed the result");
    // a different context must not reuse the old embedding
    let mut other = random_input(c, 5, 7, 2, 0.7);
    other.txt[0] += 0.5;
    assert_ne!(m3.forward(&other), b, "changed txt must change the output");
}

#[test]
fn taps_record_the_single_block_inputs() {
    let c = Tiny::standard(1, 2);
    let mut m = RefModel::from_source(&tiny_source(c, 3), 2).unwrap();
    let input = random_input(c, 5, 7, 4, 0.3);
    let mut taps = Taps::recording();
    let out = m.forward_with_taps(&input, &mut taps);
    let t0 = taps.single0_in.expect("tap 0");
    let t1 = taps.single1_in.expect("tap 1");
    assert_eq!(t0.len(), 12 * c.hidden);
    assert_eq!(t1.len(), 12 * c.hidden);
    assert_ne!(t0, t1);
    assert_eq!(out, m.forward(&input));
}

#[test]
fn timestep_embedding_matches_hand_values_dim8() {
    // t = 0.5, time_factor 1000 -> arg = 500 * freq_j, freq = 10000^(-j/4) = 1, 0.1, 0.01, 0.001
    let got = timestep_embedding(0.5, 8, 10000.0, 1000.0);
    let args = [500.0f64, 50.0, 5.0, 0.5];
    for j in 0..4 {
        assert!((got[j] as f64 - args[j].cos()).abs() < 1e-4, "cos slot {j}: {} vs {}", got[j], args[j].cos());
        assert!((got[j + 4] as f64 - args[j].sin()).abs() < 1e-4, "sin slot {j}");
    }
    // odd dim gets a trailing zero
    let odd = timestep_embedding(0.5, 7, 10000.0, 1000.0);
    assert_eq!(odd.len(), 7);
    assert_eq!(odd[6], 0.0);
}

#[test]
fn modulation_chunk_order_is_shift_scale_gate_twice() {
    let h = 4;
    // element k*h + r = 100*k + r
    let v: Vec<f32> = (0..6 * h).map(|i| (100 * (i / h) + i % h) as f32).collect();
    let m = Mods6::from_vec(&v, h);
    assert_eq!(m.shift1, vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(m.scale1[0], 100.0);
    assert_eq!(m.gate1[0], 200.0);
    assert_eq!(m.shift2[0], 300.0);
    assert_eq!(m.scale2[0], 400.0);
    assert_eq!(m.gate2[3], 503.0);
    let s = Mods3::from_vec(&v[..3 * h], h);
    assert_eq!((s.shift[0], s.scale[0], s.gate[0]), (0.0, 100.0, 200.0));
}

// ---- naive f64 double block (independent of the library code above) -----------------------

fn naive_ln_mod(x: &[f64], shift: &[f32], scale: &[f32]) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let d = (var + 1e-6).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let y = (v - mean) / d;
            (y + y * scale[i] as f64) + shift[i] as f64
        })
        .collect()
}

/// `rows` tokens of width `inn` through `w` (`[out][in]`) in f64.
fn naive_lin(w: &[f32], inn: usize, out: usize, x: &[f64], rows: usize) -> Vec<f64> {
    let mut y = vec![0f64; rows * out];
    for t in 0..rows {
        for r in 0..out {
            let mut s = 0f64;
            for i in 0..inn {
                s += w[r * inn + i] as f64 * x[t * inn + i];
            }
            y[t * out + r] = s;
        }
    }
    y
}

fn naive_stream_qkv(d: Dims, s: &StreamF32, x: &[f64], shift: &[f32], scale: &[f32], pe: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (h, hd) = (d.hidden, d.head_dim);
    let n = x.len() / h;
    let mut xm = Vec::with_capacity(n * h);
    for t in 0..n {
        xm.extend(naive_ln_mod(&x[t * h..(t + 1) * h], shift, scale));
    }
    let y = naive_lin(&s.qkv, h, 3 * h, &xm, n);
    let (mut q, mut k, mut v) = (vec![0f64; n * h], vec![0f64; n * h], vec![0f64; n * h]);
    for t in 0..n {
        for j in 0..h {
            q[t * h + j] = y[t * 3 * h + j];
            k[t * h + j] = y[t * 3 * h + h + j];
            v[t * h + j] = y[t * 3 * h + 2 * h + j];
        }
        for hh in 0..d.heads {
            for buf in [(&mut q, &s.q_scale), (&mut k, &s.k_scale)] {
                let (vals, sc) = buf;
                let base = t * h + hh * hd;
                let ms = (0..hd).map(|j| vals[base + j] * vals[base + j]).sum::<f64>() / hd as f64;
                let den = (ms + 1e-6).sqrt();
                for j in 0..hd {
                    vals[base + j] = vals[base + j] / den * sc[j] as f64;
                }
                // rope on pairs
                for p in 0..hd / 2 {
                    let pe_i = (t * (hd / 2) + p) * 4;
                    let (a, b) = (vals[base + 2 * p], vals[base + 2 * p + 1]);
                    let (p0, p1, p2, p3) = (pe[pe_i] as f64, pe[pe_i + 1] as f64, pe[pe_i + 2] as f64, pe[pe_i + 3] as f64);
                    vals[base + 2 * p] = p0 * a + p1 * b;
                    vals[base + 2 * p + 1] = p2 * a + p3 * b;
                }
            }
        }
    }
    (q, k, v)
}

fn naive_double(d: Dims, b: &DoubleBlockF32, img: &[f32], txt: &[f32], mi: &Mods6, mt: &Mods6, pe: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let (h, hd, m) = (d.hidden, d.head_dim, d.mlp);
    let nt = txt.len() / h;
    let ni = img.len() / h;
    let n = nt + ni;
    let txt64: Vec<f64> = txt.iter().map(|&v| v as f64).collect();
    let img64: Vec<f64> = img.iter().map(|&v| v as f64).collect();
    let (qt, kt, vt) = naive_stream_qkv(d, &b.txt, &txt64, &mt.shift1, &mt.scale1, &pe[..nt * (hd / 2) * 4]);
    let (qi, ki, vi) = naive_stream_qkv(d, &b.img, &img64, &mi.shift1, &mi.scale1, &pe[nt * (hd / 2) * 4..]);
    let q: Vec<f64> = [qt, qi].concat();
    let k: Vec<f64> = [kt, ki].concat();
    let v: Vec<f64> = [vt, vi].concat();
    // joint attention, naive
    let mut attn = vec![0f64; n * h];
    let sc = 1.0 / (hd as f64).sqrt();
    for t in 0..n {
        for hh in 0..d.heads {
            let base = hh * hd;
            let logits: Vec<f64> = (0..n)
                .map(|tk| (0..hd).map(|j| q[t * h + base + j] * k[tk * h + base + j]).sum::<f64>() * sc)
                .collect();
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = logits.iter().map(|l| (l - mx).exp()).collect();
            let z: f64 = ex.iter().sum();
            for j in 0..hd {
                attn[t * h + base + j] = (0..n).map(|tk| ex[tk] / z * v[tk * h + base + j]).sum::<f64>();
            }
        }
    }
    let post = |s: &StreamF32, x: &[f64], a: &[f64], mm: &Mods6, rows: usize| -> Vec<f64> {
        let p = naive_lin(&s.proj, h, h, a, rows);
        let mut x: Vec<f64> = x.iter().enumerate().map(|(i, &xv)| xv + mm.gate1[i % h] as f64 * p[i]).collect();
        let mut xm = Vec::with_capacity(rows * h);
        for t in 0..rows {
            xm.extend(naive_ln_mod(&x[t * h..(t + 1) * h], &mm.shift2, &mm.scale2));
        }
        let m0 = naive_lin(&s.mlp0, h, 2 * m, &xm, rows);
        let mut act = vec![0f64; rows * m];
        for t in 0..rows {
            for j in 0..m {
                let g = m0[t * 2 * m + j];
                act[t * m + j] = g / (1.0 + (-g).exp()) * m0[t * 2 * m + m + j];
            }
        }
        let m2 = naive_lin(&s.mlp2, m, h, &act, rows);
        for (i, xv) in x.iter_mut().enumerate() {
            *xv += mm.gate2[i % h] as f64 * m2[i];
        }
        x
    };
    let txt_out = post(&b.txt, &txt64, &attn[..nt * h], mt, nt);
    let img_out = post(&b.img, &img64, &attn[nt * h..], mi, ni);
    (img_out, txt_out)
}

#[test]
fn double_block_matches_naive_f64() {
    let d = Dims { hidden: 32, heads: 2, head_dim: 16, mlp: 96 };
    let (h, hd, m) = (d.hidden, d.head_dim, d.mlp);
    let mut r = Rng(77);
    let stream = |r: &mut Rng| StreamF32 {
        qkv: r.vec(3 * h * h, 1.0 / (h as f32).sqrt()),
        proj: r.vec(h * h, 1.0 / (h as f32).sqrt()),
        mlp0: r.vec(2 * m * h, 1.0 / (h as f32).sqrt()),
        mlp2: r.vec(h * m, 1.0 / (m as f32).sqrt()),
        q_scale: (0..hd).map(|_| 1.0 + 0.3 * r.next()).collect(),
        k_scale: (0..hd).map(|_| 1.0 + 0.3 * r.next()).collect(),
    };
    let blk = DoubleBlockF32 { img: stream(&mut r), txt: stream(&mut r) };
    let mi = Mods6::from_vec(&r.vec(6 * h, 0.5), h);
    let mt = Mods6::from_vec(&r.vec(6 * h, 0.5), h);
    let (n_txt, n_img) = (5, 7);
    let img = r.vec(n_img * h, 1.0);
    let txt = r.vec(n_txt * h, 1.0);
    let pe = random_pe(&mut r, n_txt + n_img, hd / 2);

    let (lib_img, lib_txt) = double_block(d, &blk, img.clone(), txt.clone(), &mi, &mt, &pe, 3);
    let (nv_img, nv_txt) = naive_double(d, &blk, &img, &txt, &mi, &mt, &pe);
    let nv_img: Vec<f32> = nv_img.iter().map(|&v| v as f32).collect();
    let nv_txt: Vec<f32> = nv_txt.iter().map(|&v| v as f32).collect();
    let e_img = rel_rmse(&lib_img, &nv_img);
    let e_txt = rel_rmse(&lib_txt, &nv_txt);
    println!("double block rel rmse: img {e_img:e} txt {e_txt:e}");
    assert!(e_img < 1e-5, "img rel rmse {e_img}");
    assert!(e_txt < 1e-5, "txt rel rmse {e_txt}");
}

// ---- cross-check: RefModel (this crate) against FastModel (lead's int8 path) ----------------

fn clone_src(s: &MemSource) -> MemSource {
    MemSource { q8: s.q8.clone(), f: s.f.clone() }
}

/// The same source with every block output projection zeroed (attn proj, mlp.2, single linear2):
/// each block then adds exactly 0 to the residual stream, so the output is embedders + final layer.
fn no_block_source(src: &MemSource) -> MemSource {
    let mut out = clone_src(src);
    let names: Vec<String> = out
        .q8
        .keys()
        .filter(|k| k.ends_with("_attn.proj.weight") || k.ends_with("_mlp.2.weight") || k.ends_with("linear2.weight"))
        .cloned()
        .collect();
    for n in names {
        let (_, rows, k) = out.q8[&n].clone();
        let zeros = vec![0f32; rows * k];
        out.q8.insert(n, (quant_q8_0(&zeros, rows, k), rows, k));
    }
    out
}

fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// pe with every rotation angle negated: `[c, -s, s, c]` becomes `[c, s, -s, c]`.
fn negate_pe(input: &ModelInput) -> ModelInput {
    let pe: Vec<f32> = input
        .pe
        .chunks_exact(4)
        .flat_map(|p| [p[0], -p[1], -p[2], p[3]])
        .collect();
    ModelInput {
        n_img: input.n_img,
        n_txt: input.n_txt,
        img: input.img.clone(),
        txt: input.txt.clone(),
        timestep: input.timestep,
        pe,
    }
}

#[derive(Debug)]
struct Case {
    /// rel rmse of fast vs ref on the full output
    full_e: f64,
    /// rel rmse of the reference output vs the no-block output: how much the blocks matter
    contrib: f64,
    /// rel rmse of (fast - noblock) vs (ref - noblock): the error relative to the blocks' contribution
    err_delta: f64,
}

/// `src_ref` drives the reference (and its no-block baseline), `src_fast` the fast model;
/// `in_ref` / `in_fast` are the inputs for each.
fn run_case(src_ref: &MemSource, src_fast: &MemSource, in_ref: &ModelInput, in_fast: &ModelInput, cfg: FastConfig, pool: &sched::Pool) -> Case {
    let mut rm = RefModel::from_source(src_ref, 3).expect("ref model");
    let want = rm.forward(in_ref);
    let mut nm = RefModel::from_source(&no_block_source(src_ref), 3).expect("no-block model");
    let nob = nm.forward(in_ref);
    let mut fm = FastModel::from_source(src_fast, pool, cfg).expect("fast model");
    let got = fm.forward(in_fast);
    Case {
        full_e: rel_rmse(&got, &want),
        contrib: rel_rmse(&want, &nob),
        err_delta: rel_rmse(&sub(&got, &nob), &sub(&want, &nob)),
    }
}

fn standard_case(c: Tiny, label: &str) -> Case {
    let cfg = FastConfig { chunk1: 32, chunk2: 32, batch_groups: 2, lin2_row_block: 16, key_chunk: 16, ..FastConfig::default() };
    let pool = sched::Pool::new(&[0, 1, 2], std::time::Duration::from_millis(2)).expect("pool");
    let src = tiny_source(c, 9);
    let input = random_input(c, 13, 19, 5, 0.7);
    let k = run_case(&src, &src, &input, &input, cfg, &pool);
    println!(
        "cross-check {label}: full rel rmse {:.4e}, blocks contribute {:.4e}, err relative to contribution {:.4e}",
        k.full_e, k.contrib, k.err_delta
    );
    k
}

fn assert_agreement(c: Tiny, label: &str) {
    let k = standard_case(c, label);
    assert!(k.contrib > 0.3, "{label}: blocks must matter (contribution {})", k.contrib);
    assert!(k.full_e < 0.08, "{label}: fast vs ref full rel rmse {}", k.full_e);
    assert!(k.err_delta < 0.08, "{label}: error relative to contribution {}", k.err_delta);
}

#[test]
fn fast_model_agrees_with_reference_on_tiny_model() {
    assert_agreement(Tiny::standard(1, 2), "full (1 double + 2 single)");
}

#[test]
fn fast_model_agrees_with_reference_double_only() {
    assert_agreement(Tiny::standard(1, 0), "double only");
}

#[test]
fn fast_model_agrees_with_reference_single_only() {
    assert_agreement(Tiny::standard(0, 2), "single only");
}

/// F-IMG-ATTN: f16/f32 attention in the fast model (int8 linears) agrees with the reference, and
/// f32 attention is closer to it than int8 attention (the attention is then exact up to float
/// noise, so only the linears' int8 error is left).
#[test]
fn fast_model_float_attention_agrees_and_is_closer() {
    use image_dit::attn_float::AttnPrec;
    let c = Tiny::standard(1, 2);
    let pool = sched::Pool::new(&[0, 1, 2], std::time::Duration::from_millis(2)).expect("pool");
    let src = tiny_source(c, 9);
    let input = random_input(c, 13, 19, 5, 0.7);
    let mut errs = Vec::new();
    for prec in [AttnPrec::Int8, AttnPrec::F16, AttnPrec::F32] {
        let cfg = FastConfig { chunk1: 32, chunk2: 32, batch_groups: 2, lin2_row_block: 16, key_chunk: 16, attn_prec: prec };
        let k = run_case(&src, &src, &input, &input, cfg, &pool);
        println!("attn-prec {}: full rel rmse {:.4e}, err relative to contribution {:.4e}", prec.name(), k.full_e, k.err_delta);
        assert!(k.full_e < 0.08 && k.err_delta < 0.08, "{prec:?}: {k:?}");
        errs.push(k.err_delta);
    }
    assert!(errs[2] < errs[0], "f32 attention must be closer to the reference than int8: {errs:?}");
    assert!(errs[1] < errs[0], "f16 attention must be closer to the reference than int8: {errs:?}");
}

// Negative controls: each perturbation of the fast side must make the error relative to the
// blocks' contribution large, otherwise the agreement tests above would not be sensitive.

#[test]
fn negative_control_negated_rope_angles_is_detected() {
    let c = Tiny::standard(1, 2);
    let cfg = FastConfig { chunk1: 32, chunk2: 32, batch_groups: 2, lin2_row_block: 16, key_chunk: 16, ..FastConfig::default() };
    let pool = sched::Pool::new(&[0, 1, 2], std::time::Duration::from_millis(2)).expect("pool");
    let src = tiny_source(c, 9);
    let input = random_input(c, 13, 19, 5, 0.7);
    let k = run_case(&src, &src, &input, &negate_pe(&input), cfg, &pool);
    println!("negative control (negated pe angles): err relative to contribution {:.4e}", k.err_delta);
    assert!(k.err_delta > 0.2, "{k:?}");
}

#[test]
fn negative_control_swapped_qk_norm_scales_is_detected() {
    let c = Tiny::standard(1, 2);
    let cfg = FastConfig { chunk1: 32, chunk2: 32, batch_groups: 2, lin2_row_block: 16, key_chunk: 16, ..FastConfig::default() };
    let pool = sched::Pool::new(&[0, 1, 2], std::time::Duration::from_millis(2)).expect("pool");
    let src = tiny_source(c, 9);
    let mut bad = clone_src(&src);
    let qn = "double_blocks.0.img_attn.norm.query_norm.scale";
    let kn = "double_blocks.0.img_attn.norm.key_norm.scale";
    // Swap the two scales and make them strongly different from the reference: the norm scales
    // are about 1 +- 0.3, so a plain swap alone changes the attention only weakly.
    let q: Vec<f32> = bad.f[qn].0.iter().map(|v| 4.0 * v).collect();
    let kv: Vec<f32> = bad.f[kn].0.iter().map(|v| 0.25 * v).collect();
    bad.f.get_mut(qn).unwrap().0 = kv;
    bad.f.get_mut(kn).unwrap().0 = q;
    let input = random_input(c, 13, 19, 5, 0.7);
    let k = run_case(&src, &bad, &input, &input, cfg, &pool);
    println!("negative control (swapped q/k norm scales, double block 0 img): err relative to contribution {:.4e}", k.err_delta);
    assert!(k.err_delta > 0.2, "{k:?}");
}

#[test]
fn negative_control_different_linear2_is_detected() {
    let c = Tiny::standard(1, 2);
    let cfg = FastConfig { chunk1: 32, chunk2: 32, batch_groups: 2, lin2_row_block: 16, key_chunk: 16, ..FastConfig::default() };
    let pool = sched::Pool::new(&[0, 1, 2], std::time::Duration::from_millis(2)).expect("pool");
    let src = tiny_source(c, 9);
    let mut bad = clone_src(&src);
    let name = "single_blocks.0.linear2.weight";
    let (_, rows, k_in) = bad.q8[name].clone();
    let mut r = Rng(4242);
    let v: Vec<f32> = r.vec(rows * k_in, 1.0 / (k_in as f32).sqrt()).iter().map(|x| x * GAIN).collect();
    bad.q8.insert(name.to_string(), (quant_q8_0(&v, rows, k_in), rows, k_in));
    let input = random_input(c, 13, 19, 5, 0.7);
    let k = run_case(&src, &bad, &input, &input, cfg, &pool);
    println!("negative control (different single block 0 linear2): err relative to contribution {:.4e}", k.err_delta);
    assert!(k.err_delta > 0.2, "{k:?}");
}

/// Structural check behind "can text-token work be cached across sampling steps?": with the same
/// text context, the text rows entering the single blocks change when only the timestep changes
/// (the time-dependent modulation sits in front of every text linear) and when only the image
/// latent changes (the joint attention of the double blocks mixes image into text). So nothing
/// after `txt_in` is step-invariant by construction; caching it would be an approximation whose
/// size only real weights can tell (see `oxi txt-stats`, `txt_rows_step_change`).
#[test]
fn text_stream_is_not_step_invariant() {
    let c = Tiny::standard(1, 2);
    let src = tiny_source(c, 77);
    let (n_txt, n_img) = (13usize, 19usize);
    let a = random_input(c, n_txt, n_img, 5, 0.9);
    let mut b = random_input(c, n_txt, n_img, 5, 0.3); // same txt, img, pe; other timestep
    b.txt = a.txt.clone();
    b.img = a.img.clone();
    b.pe = a.pe.clone();
    let mut d = random_input(c, n_txt, n_img, 6, 0.9); // same txt, pe, timestep; other image latent
    d.txt = a.txt.clone();
    d.pe = a.pe.clone();
    assert_ne!(d.img, a.img);
    let mut m = RefModel::from_source(&src, 2).expect("ref model");
    let tap = |m: &mut RefModel, inp: &ModelInput| -> Vec<f32> {
        let mut t = Taps::recording();
        m.forward_with_taps(inp, &mut t);
        let x = t.single0_in.expect("tap recorded");
        x[..n_txt * c.hidden].to_vec()
    };
    let (ta, tb, td) = (tap(&mut m, &a), tap(&mut m, &b), tap(&mut m, &d));
    let by_time = rel_rmse(&tb, &ta);
    let by_image = rel_rmse(&td, &ta);
    println!("text rows entering single block 0: change by timestep {by_time:.3e}, by image latent {by_image:.3e}");
    assert!(by_time > 1e-3, "text rows did not change with the timestep: {by_time:e}");
    assert!(by_image > 1e-3, "text rows did not change with the image latent: {by_image:e}");
    // and the same input twice is identical (the test measures the model, not noise)
    assert_eq!(tap(&mut m, &a), ta);
}

// ---------------------------------------------------------------------------------------------
// W5-G: arithmetic-parity mode and per-stage taps
// ---------------------------------------------------------------------------------------------

#[test]
fn all_taps_are_consistent_and_do_not_change_the_result() {
    let c = Tiny::standard(2, 2);
    let mut m = RefModel::from_source(&tiny_source(c, 11), 2).unwrap();
    let input = random_input(c, 5, 7, 4, 0.6);
    let plain = m.forward(&input);
    let mut taps = Taps::all();
    let out = m.forward_with_taps(&input, &mut taps);
    assert_eq!(out, plain, "taps must not change the forward");
    let h = c.hidden;
    assert_eq!(taps.img_emb.as_ref().unwrap().len(), 7 * h);
    assert_eq!(taps.txt_emb.as_ref().unwrap().len(), 5 * h);
    assert_eq!(taps.vec.as_ref().unwrap().len(), h);
    assert_eq!(taps.mod_single.as_ref().unwrap().len(), 3 * h);
    let dbl = taps.double_out.as_ref().unwrap();
    assert_eq!(dbl.len(), 2);
    // the last double block's (txt, img) is exactly what enters single block 0
    let (li, lt) = &dbl[1];
    let s0 = taps.single0_in.as_ref().unwrap();
    assert_eq!(&[lt.clone(), li.clone()].concat(), s0);
    // single block 0's output is single block 1's input; its intermediates have the right shapes
    let (o, st) = taps.single0.as_ref().unwrap();
    assert_eq!(&o.out, taps.single1_in.as_ref().unwrap());
    assert_eq!(st.x_mod.len(), 12 * h);
    assert_eq!(st.q.len(), 12 * h);
    assert_eq!(o.mlp.len(), 12 * c.mlp);
    assert_eq!(taps.final_in.as_ref().unwrap().len(), 7 * h);
    // time_vectors gives the tapped embedding-path vectors
    let tv = m.time_vectors(input.timestep);
    assert_eq!(&tv.vec, taps.vec.as_ref().unwrap());
    assert_eq!(&tv.mod_single, taps.mod_single.as_ref().unwrap());
}

#[test]
fn parity_mode_rounds_activations_and_stays_close() {
    let c = Tiny::standard(1, 2);
    let mut m = RefModel::from_source(&tiny_source(c, 12), 2).unwrap();
    let input = random_input(c, 5, 7, 9, 0.5);
    let f32_out = m.forward(&input);
    m.set_arith(Arith::SdParity);
    let mut taps = Taps::all();
    let par = m.forward_with_taps(&input, &mut taps);
    let e = rel_rmse(&par, &f32_out);
    // Q8_0 activations (7-bit mantissa per 32 values) move the output a little, never a lot
    assert!(e > 0.0, "parity mode must change the arithmetic");
    assert!(e < 0.05, "parity mode drifted {e}");
    // the x_mod tap is the f32 LayerNorm+modulate output (sd.cpp dumps it before quantising), so
    // in single block 0 it differs from the f32 forward's only through the upstream double block
    let (_, st) = taps.single0.as_ref().unwrap();
    assert_eq!(st.x_mod.len(), 12 * c.hidden);
    assert!(st.x_mod.chunks_exact(32).any(|blk| {
        let d = blk.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
        d > 0.0 && blk.iter().any(|v| ((v / d) - (v / d).round()).abs() > 1e-2)
    }), "x_mod tap must be the unrounded f32 values");
    // switching back gives the f32 reference bitwise (txt cache dropped on the switch)
    m.set_arith(Arith::F32);
    assert_eq!(m.forward(&input), f32_out);
}

#[test]
fn parity_single_block_component_matches_the_forward_tap() {
    // the component probe (single block alone on a given input) equals the in-forward tap
    let c = Tiny::standard(1, 2);
    let mut m = RefModel::from_source(&tiny_source(c, 13), 3).unwrap();
    m.set_arith(Arith::SdParity);
    let input = random_input(c, 5, 7, 2, 0.4);
    let mut taps = Taps::all();
    let _ = m.forward_with_taps(&input, &mut taps);
    let ms = taps.mod_single.clone().unwrap();
    let h = c.hidden;
    let inp = BlockInput {
        n_tokens: 12,
        x: taps.single0_in.clone().unwrap(),
        pe: input.pe.clone(),
        shift: ms[..h].to_vec(),
        scale: ms[h..2 * h].to_vec(),
        gate: ms[2 * h..].to_vec(),
    };
    let (o, st) = m.single_block(0, &inp).unwrap();
    let (to, tst) = taps.single0.as_ref().unwrap();
    assert_eq!(o.out, to.out);
    assert_eq!(st.q, tst.q);
}

#[test]
fn perturbation_control_is_off_by_default_and_small_when_on() {
    let c = Tiny::standard(1, 2);
    let mut m = RefModel::from_source(&tiny_source(c, 14), 2).unwrap();
    let input = random_input(c, 5, 7, 3, 0.5);
    let base = m.forward(&input);
    m.set_perturb_ulp(0);
    assert_eq!(m.forward(&input), base);
    m.set_perturb_ulp(1);
    let p = m.forward(&input);
    let e = rel_rmse(&p, &base);
    assert!(e > 0.0 && e < 1e-4, "1-ulp perturbation moved the f32 output by {e}");
}

#[test]
fn perturb_moves_each_value_by_at_most_n_ulps() {
    let x: Vec<f32> = (0..1000).map(|i| (i as f32 - 500.0) * 0.37).collect();
    let p = crate::drift::perturb(&x, 2);
    let mut moved = 0;
    for (a, b) in x.iter().zip(&p) {
        let d = (a.to_bits() as i64 - b.to_bits() as i64).abs();
        assert!(d <= 2, "{a} -> {b}");
        moved += (d > 0) as usize;
    }
    assert!(moved > 500, "only {moved} values moved");
    assert_eq!(p[500], 0.0, "zero stays zero");
}
