//! Production-shape microbench of the DiT block LINEARS on the model path (`FastModel`): klein
//! dims, random Q8_0 weights, 512 text tokens + 1024 / 4096 image tokens (the 1536- and
//! 4608-token shapes). No model file. Reports per stage: wall time, effective TOPS of the linear
//! stages against the int8 roofline, and the load imbalance of the static plan.
//!
//! Timing-sensitive: run with ALL build locks held (BUILD-RULES 10), e.g.
//! `tools/dit_quiet.sh cargo test --release -p image-dit -- --ignored --nocapture prod_shape_model`
//! Env: `OXI_DIT_TOKENS=1536,4608` (default both), `OXI_FIXTURE_THREADS` (12), `OXI_DIT_DOUBLE` (1),
//! `OXI_DIT_SINGLE` (1), `OXI_DIT_REPS` (4), `OXI_DIT_CHUNK` (256).

use crate::fast::FastConfig;
use crate::fast_model::{FastModel, LinTune, MemSource, ModelTimes};
use crate::model::{DitBackend, ModelInput};
use crate::prod_fixture::{random_q8, Rng};
use crate::types::Dims;
use std::time::Duration;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn put_f(src: &mut MemSource, name: &str, rows: usize, k: usize, amp: f32, rng: &mut Rng) {
    let v: Vec<f32> = (0..rows * k).map(|_| rng.unit() * amp).collect();
    src.f.insert(name.to_string(), (v, rows, k));
}

fn put_q(src: &mut MemSource, name: String, rows: usize, k: usize, rng: &mut Rng) {
    src.q8.insert(name, (random_q8(rows, k, rng), rows, k));
}

const IN_CH: usize = 128;
const CTX: usize = 64; // klein's is 7680; the text embedder is not what this fixture measures
const OUT_CH: usize = 128;

pub(crate) fn klein_source(n_double: usize, n_single: usize, rng: &mut Rng) -> MemSource {
    let d = Dims::KLEIN_4B;
    let (h, hd) = (d.hidden, d.head_dim);
    let mlp = d.lin2_in() - h;
    let mut src = MemSource::default();
    let fi = |k: usize| 1.0 / (k as f32).sqrt();
    put_f(&mut src, "img_in.weight", h, IN_CH, fi(IN_CH), rng);
    put_f(&mut src, "txt_in.weight", h, CTX, fi(CTX), rng);
    put_f(&mut src, "time_in.in_layer.weight", h, 256, fi(256), rng);
    put_f(&mut src, "time_in.out_layer.weight", h, h, fi(h), rng);
    put_f(&mut src, "double_stream_modulation_img.lin.weight", 6 * h, h, fi(h), rng);
    put_f(&mut src, "double_stream_modulation_txt.lin.weight", 6 * h, h, fi(h), rng);
    put_f(&mut src, "single_stream_modulation.lin.weight", 3 * h, h, fi(h), rng);
    put_f(&mut src, "final_layer.adaLN_modulation.1.weight", 2 * h, h, fi(h), rng);
    put_f(&mut src, "final_layer.linear.weight", OUT_CH, h, fi(h), rng);
    let ones = vec![1.0f32; hd];
    for i in 0..n_double {
        for s in ["img", "txt"] {
            put_q(&mut src, format!("double_blocks.{i}.{s}_attn.qkv.weight"), 3 * h, h, rng);
            put_q(&mut src, format!("double_blocks.{i}.{s}_attn.proj.weight"), h, h, rng);
            put_q(&mut src, format!("double_blocks.{i}.{s}_mlp.0.weight"), 2 * mlp, h, rng);
            put_q(&mut src, format!("double_blocks.{i}.{s}_mlp.2.weight"), h, mlp, rng);
            src.f.insert(format!("double_blocks.{i}.{s}_attn.norm.query_norm.scale"), (ones.clone(), 1, hd));
            src.f.insert(format!("double_blocks.{i}.{s}_attn.norm.key_norm.scale"), (ones.clone(), 1, hd));
        }
    }
    for i in 0..n_single {
        put_q(&mut src, format!("single_blocks.{i}.linear1.weight"), 3 * h + 2 * mlp, h, rng);
        put_q(&mut src, format!("single_blocks.{i}.linear2.weight"), h, h + mlp, rng);
        src.f.insert(format!("single_blocks.{i}.norm.query_norm.scale"), (ones.clone(), 1, hd));
        src.f.insert(format!("single_blocks.{i}.norm.key_norm.scale"), (ones.clone(), 1, hd));
    }
    src
}

pub(crate) fn klein_input(n_txt: usize, n_img: usize, rng: &mut Rng) -> ModelInput {
    let d = Dims::KLEIN_4B;
    let n = n_txt + n_img;
    let mut pe = Vec::with_capacity(n * (d.head_dim / 2) * 4);
    for _ in 0..n * (d.head_dim / 2) {
        let a = rng.unit() * std::f32::consts::PI;
        pe.extend_from_slice(&[a.cos(), -a.sin(), a.sin(), a.cos()]);
    }
    ModelInput {
        n_img,
        n_txt,
        img: (0..n_img * IN_CH).map(|_| rng.unit()).collect(),
        txt: (0..n_txt * CTX).map(|_| rng.unit()).collect(),
        timestep: 0.6,
        pe,
    }
}

fn hash(v: &[f32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for x in v {
        h = (h ^ x.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
#[ignore]
fn prod_shape_model_linears() {
    let d = Dims::KLEIN_4B;
    let (h, mlp) = (d.hidden, d.lin2_in() - d.hidden);
    let threads = env_usize("OXI_FIXTURE_THREADS", 12);
    let (nd, ns) = (env_usize("OXI_DIT_DOUBLE", 1), env_usize("OXI_DIT_SINGLE", 1));
    let reps = env_usize("OXI_DIT_REPS", 4);
    let chunk = env_usize("OXI_DIT_CHUNK", 256);
    let tokens: Vec<usize> = std::env::var("OXI_DIT_TOKENS").unwrap_or_else(|_| "1536,4608".into()).split(',').filter_map(|s| s.parse().ok()).collect();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let src = klein_source(nd, ns, &mut rng);
    let cpus: Vec<usize> = (0..threads).collect();
    let pool = sched::Pool::new(&cpus, Duration::from_millis(2)).expect("pool");
    let cfg = FastConfig { chunk1: chunk, chunk2: chunk, ..FastConfig::default() };
    let t0 = std::time::Instant::now();
    let mut model = FastModel::from_source(&src, &pool, cfg).expect("model");
    let pack = t0.elapsed().as_secs_f64();
    drop(src);
    let base_tune = model.tune;
    // OXI_DIT_TUNE = ';'-separated variants of "unit_batch,plain_batch,plain_rows" (empty field = default).
    // The variants are interleaved per repetition (v0 v1 .. v0 v1 ..), so they see the same thermal state.
    let variants: Vec<LinTune> = std::env::var("OXI_DIT_TUNE")
        .unwrap_or_default()
        .split(';')
        .map(|var| {
            let f: Vec<Option<usize>> = var.split(',').map(|s| s.trim().parse().ok()).collect();
            let get = |i: usize, d: usize| f.get(i).copied().flatten().unwrap_or(d);
            LinTune {
                unit_batch: get(0, base_tune.unit_batch),
                plain_batch: get(1, base_tune.plain_batch),
                plain_rows: get(2, base_tune.plain_rows),
            }
        })
        .collect();
    for &n in &tokens {
        let n_txt = 512usize;
        let n_img = n - n_txt;
        let inp = klein_input(n_txt, n_img, &mut rng);
        let first = model.forward(&inp);
        assert!(first.iter().all(|v| v.is_finite()), "non-finite output");
        let mut all: Vec<Vec<ModelTimes>> = vec![Vec::new(); variants.len()];
        for _ in 0..reps {
            for (vi, tune) in variants.iter().enumerate() {
                model.tune = *tune;
                let out = model.forward(&inp);
                assert!(out.iter().zip(&first).all(|(a, b)| a.to_bits() == b.to_bits()), "output not repeatable / depends on the tuning");
                all[vi].push(model.last_times);
            }
        }
        // linear MACs: every stream token passes 3h*h + h*h + 2mlp*h + h*mlp in a double block,
        // every token (3h + 2mlp)*h + h*(h + mlp) in a single block
        let unit_mac = n as f64 * ((nd * (3 * h * h + 2 * mlp * h) + ns * (3 * h + 2 * mlp) * h) as f64);
        let plain_mac = n as f64 * ((nd * (h * h + h * mlp) + ns * h * (h + mlp)) as f64);
        let att_mac = (nd + ns) as f64 * 2.0 * (n as f64) * (n as f64) * h as f64;
        println!("dit model fixture tokens={n} chunk={chunk} threads={threads} blocks={nd}d+{ns}s pack={pack:.1}s hash={:016x}", hash(&first));
        let nt = threads as f64;
        for (vi, tune) in variants.iter().enumerate() {
            let all = &all[vi];
            let best = |f: &dyn Fn(&ModelTimes) -> f64| all.iter().map(f).fold(f64::MAX, f64::min);
            let med = |f: &dyn Fn(&ModelTimes) -> f64| {
                let mut v: Vec<f64> = all.iter().map(f).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[(v.len() - 1) / 2]
            };
            let lin = |t: &ModelTimes| t.units + t.plain;
            let blk = |t: &ModelTimes| t.double_blocks + t.single_blocks;
            let tb = all.iter().min_by(|a, b| lin(a).partial_cmp(&lin(b)).unwrap()).expect("reps");
            println!(
                "  tune {},{},{} best-of-{reps}: linears {:.4} s = {:.2} TOPS (units {:.2}, plain {:.2} TOPS), attention {:.4} s = {:.2} TOPS, blocks {:.4} s | median: linears {:.4} s = {:.2} TOPS, attention {:.4}, blocks {:.4} | quant {:.4} alloc {:.4} imbalance u {:.3} p {:.3} busy/wall u {:.3} p {:.3}",
                tune.unit_batch,
                tune.plain_batch,
                tune.plain_rows,
                lin(tb),
                2.0 * (unit_mac + plain_mac) / lin(tb) / 1e12,
                2.0 * unit_mac / best(&|t| t.units) / 1e12,
                2.0 * plain_mac / best(&|t| t.plain) / 1e12,
                best(&|t| t.attention),
                2.0 * att_mac / best(&|t| t.attention) / 1e12,
                best(&blk),
                med(&lin),
                2.0 * (unit_mac + plain_mac) / med(&lin) / 1e12,
                med(&|t| t.attention),
                med(&blk),
                tb.quant,
                tb.alloc,
                tb.units_busy_max * nt / tb.units_busy,
                tb.plain_busy_max * nt / tb.plain_busy,
                tb.units_busy / nt / tb.units,
                tb.plain_busy / nt / tb.plain,
            );
        }
    }
}
