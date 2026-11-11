//! Production-shape fixture for the single-stream block (klein dims, 1536 and 4608 tokens) with
//! random Q8_0 weights: no model file. It exists so that address-dependent effects (cache
//! aliasing at production strides, thread-boundary sharing) and gross speed regressions are
//! visible without the model, and it checks that the output is finite and repeatable bit for bit.
//! It does NOT check numerics against a reference (the f32 reference at these shapes takes
//! minutes; the model-free numeric checks live in the small-shape tests and `oxi-block selftest`).
//!
//! Heavy (12 pinned threads, about 1 GiB): run explicitly in a DEV window,
//! `cargo test --release -p image-dit -- --ignored --nocapture prod_shape`
//! (`OXI_DIT_FIXTURE_4608=1` adds the 4608-token case).

use crate::backend::SingleBlockBackend;
use crate::fast::{FastBlock, FastConfig};
use crate::types::{BlockInput, Dims};
use std::time::Duration;

pub(crate) struct Rng(pub u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub(crate) fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Random Q8_0 bytes: per 32-block a small positive half-float scale and 32 random codes.
pub(crate) fn random_q8(rows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let blocks = rows * k / 32;
    let mut out = Vec::with_capacity(blocks * 34);
    for _ in 0..blocks {
        // halves 0x1000..0x13ff: about 4.9e-4 .. 9.8e-4, so rows have a sane norm at k = 3072
        let d: u16 = 0x1000 | (rng.next() & 0x3ff) as u16;
        out.extend_from_slice(&d.to_le_bytes());
        let (a, b, c, e) = (rng.next(), rng.next(), rng.next(), rng.next());
        for w in [a, b, c, e] {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

fn input(dims: &Dims, n: usize, rng: &mut Rng) -> BlockInput {
    let h = dims.hidden;
    let mut pe = Vec::with_capacity(n * (dims.head_dim / 2) * 4);
    for _ in 0..n * (dims.head_dim / 2) {
        let a = rng.unit() * std::f32::consts::PI;
        pe.extend_from_slice(&[a.cos(), -a.sin(), a.sin(), a.cos()]);
    }
    BlockInput {
        n_tokens: n,
        x: (0..n * h).map(|_| rng.unit() + rng.unit()).collect(),
        pe,
        shift: (0..h).map(|_| rng.unit() * 0.1).collect(),
        scale: (0..h).map(|_| rng.unit() * 0.1).collect(),
        gate: (0..h).map(|_| rng.unit() * 0.3).collect(),
    }
}

#[test]
#[ignore]
fn prod_shape_block_1536_and_4608() {
    let dims = Dims::KLEIN_4B;
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let w1 = random_q8(dims.lin1_out(), dims.hidden, &mut rng);
    let w2 = random_q8(dims.hidden, dims.lin2_in(), &mut rng);
    let ones = vec![1.0f32; dims.head_dim];
    let threads = std::env::var("OXI_FIXTURE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(12usize);
    let cpus: Vec<usize> = (0..threads).collect();
    let pool = sched::Pool::new(&cpus, Duration::from_millis(2)).expect("pool");
    let mut sizes = vec![1536usize];
    if std::env::var("OXI_DIT_FIXTURE_4608").is_ok() {
        sizes.push(4608);
    }
    for chunk in [256usize, 32] {
        let cfg = FastConfig { chunk1: chunk, chunk2: chunk, ..FastConfig::default() };
        let t0 = std::time::Instant::now();
        let mut block = FastBlock::new(dims, &w1, &w2, &ones, &ones, cfg, &pool);
        let pack = t0.elapsed().as_secs_f64();
        for &n in &sizes {
            let inp = input(&dims, n, &mut rng);
            let (a, _) = block.forward(&inp, false);
            let mut best = f64::MAX;
            let mut best_t = None;
            let mut same = true;
            for _ in 0..3 {
                let (b, t) = block.forward(&inp, false);
                same &= a.out.iter().zip(&b.out).all(|(x, y)| x.to_bits() == y.to_bits());
                if t.total < best {
                    best = t.total;
                    best_t = Some(t);
                }
            }
            assert!(a.out.iter().all(|v| v.is_finite()), "non-finite output");
            assert!(same, "output not repeatable");
            let t = best_t.expect("three runs");
            let lin = n as f64 * (dims.lin1_out() * dims.hidden + dims.hidden * dims.lin2_in()) as f64;
            let att = 2.0 * (n as f64) * (n as f64) * dims.hidden as f64;
            println!(
                "dit block fixture tokens={n} chunk={chunk} threads={threads} pack={pack:.2}s: total {:.3} s (norm_quant {:.3}, linear1 {:.3}, attention {:.3}, requant {:.3}, linear2 {:.3}); effective TOPS block {:.2}, linear1 {:.2}, linear2 {:.2}, attention {:.2}",
                t.total,
                t.norm_quant,
                t.linear1,
                t.attention,
                t.requant,
                t.linear2,
                2.0 * (lin + att) / t.total / 1e12,
                2.0 * n as f64 * (dims.lin1_out() * dims.hidden) as f64 / t.linear1 / 1e12,
                2.0 * n as f64 * (dims.hidden * dims.lin2_in()) as f64 / t.linear2 / 1e12,
                2.0 * att / t.attention / 1e12,
            );
        }
    }
}
