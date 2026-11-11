//! Unit tests for the reference block, the weight helpers and the dump reader.
//! Tiny shapes, deterministic inputs, no model files.

use super::{attn_stage, dot, layer_norm_modulate, rms_norm_scale, rope, silu, single_block_ref};
use crate::dump::{read_block_dump, read_tag};
use crate::types::{rel_rmse, BlockInput, BlockOutput, BlockWeightsF32, Dims};
use crate::weights::{dequant_q8_0, f32_to_f16, quant_q8_0};
use int8_gemm::f16_to_f32;
use std::path::{Path, PathBuf};

fn tiny() -> Dims {
    Dims { hidden: 16, heads: 2, head_dim: 8, mlp: 24 }
}

/// xorshift64, no external crates.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [-a, a).
    fn sym(&mut self, a: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / 16_777_216.0;
        (2.0 * u - 1.0) * a
    }
    fn vec(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| self.sym(a)).collect()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Per (token, pair) `[cos, -sin, sin, cos]` from random angles.
fn rand_pe(rng: &mut Rng, n: usize, half: usize) -> Vec<f32> {
    let mut pe = Vec::with_capacity(n * half * 4);
    for _ in 0..n * half {
        let a = rng.sym(std::f32::consts::PI);
        let (s, c) = a.sin_cos();
        pe.extend_from_slice(&[c, -s, s, c]);
    }
    pe
}

fn rand_problem(rng: &mut Rng, d: Dims, n: usize) -> (BlockWeightsF32, BlockInput) {
    let w = BlockWeightsF32 {
        dims: d,
        w1: rng.vec(d.lin1_out() * d.hidden, 0.2),
        w2: rng.vec(d.hidden * d.lin2_in(), 0.2),
        q_scale: (0..d.head_dim).map(|_| 1.0 + rng.sym(0.2)).collect(),
        k_scale: (0..d.head_dim).map(|_| 1.0 + rng.sym(0.2)).collect(),
    };
    let input = BlockInput {
        n_tokens: n,
        x: rng.vec(n * d.hidden, 1.0),
        pe: rand_pe(rng, n, d.head_dim / 2),
        shift: rng.vec(d.hidden, 0.1),
        scale: rng.vec(d.hidden, 0.1),
        gate: rng.vec(d.hidden, 0.5),
    };
    (w, input)
}

fn flat32(rows: &[Vec<f64>]) -> Vec<f32> {
    rows.iter().flat_map(|r| r.iter().map(|&x| x as f32)).collect()
}

/// Independent scalar reference: plain f64 loops, no helpers from the parent module.
fn naive_block(w: &BlockWeightsF32, inp: &BlockInput) -> BlockOutput {
    let d = w.dims;
    let (h, hd, mlp, n) = (d.hidden, d.head_dim, d.mlp, inp.n_tokens);
    let half = hd / 2;
    let lin1 = 3 * h + 2 * mlp;
    let lin2 = h + mlp;

    // LayerNorm (no affine, population variance) then modulation.
    let mut xm = vec![vec![0f64; h]; n];
    for t in 0..n {
        let mut mean = 0f64;
        for i in 0..h {
            mean += inp.x[t * h + i] as f64;
        }
        mean /= h as f64;
        let mut var = 0f64;
        for i in 0..h {
            let dv = inp.x[t * h + i] as f64 - mean;
            var += dv * dv;
        }
        var /= h as f64;
        for i in 0..h {
            let yv = (inp.x[t * h + i] as f64 - mean) / (var + 1e-6).sqrt();
            xm[t][i] = yv + yv * inp.scale[i] as f64 + inp.shift[i] as f64;
        }
    }

    // linear1
    let mut y = vec![vec![0f64; lin1]; n];
    for t in 0..n {
        for r in 0..lin1 {
            let mut s = 0f64;
            for i in 0..h {
                s += w.w1[r * h + i] as f64 * xm[t][i];
            }
            y[t][r] = s;
        }
    }

    // q, k: RMS norm per head then RoPE; v: raw.
    let mut q = vec![vec![0f64; h]; n];
    let mut k = vec![vec![0f64; h]; n];
    let mut v = vec![vec![0f64; h]; n];
    for t in 0..n {
        for hh in 0..d.heads {
            let base = hh * hd;
            let mut msq = 0f64;
            let mut msk = 0f64;
            for dd in 0..hd {
                msq += y[t][base + dd] * y[t][base + dd];
                msk += y[t][h + base + dd] * y[t][h + base + dd];
            }
            let rq = (msq / hd as f64 + 1e-6).sqrt();
            let rk = (msk / hd as f64 + 1e-6).sqrt();
            let mut qn = vec![0f64; hd];
            let mut kn = vec![0f64; hd];
            for dd in 0..hd {
                qn[dd] = y[t][base + dd] / rq * w.q_scale[dd] as f64;
                kn[dd] = y[t][h + base + dd] / rk * w.k_scale[dd] as f64;
            }
            for i in 0..half {
                let p = |j: usize| inp.pe[(t * half + i) * 4 + j] as f64;
                let (q0, q1) = (qn[2 * i], qn[2 * i + 1]);
                q[t][base + 2 * i] = p(0) * q0 + p(1) * q1;
                q[t][base + 2 * i + 1] = p(2) * q0 + p(3) * q1;
                let (k0, k1) = (kn[2 * i], kn[2 * i + 1]);
                k[t][base + 2 * i] = p(0) * k0 + p(1) * k1;
                k[t][base + 2 * i + 1] = p(2) * k0 + p(3) * k1;
            }
        }
        for c in 0..h {
            v[t][c] = y[t][2 * h + c];
        }
    }

    // Attention per head over all tokens.
    let mut attn = vec![vec![0f64; h]; n];
    for tq in 0..n {
        for hh in 0..d.heads {
            let base = hh * hd;
            let mut logits = vec![0f64; n];
            for tk in 0..n {
                let mut s = 0f64;
                for dd in 0..hd {
                    s += q[tq][base + dd] * k[tk][base + dd];
                }
                logits[tk] = s / (hd as f64).sqrt();
            }
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = logits.iter().map(|&l| (l - mx).exp()).collect();
            let den: f64 = ex.iter().sum();
            for dd in 0..hd {
                let mut acc = 0f64;
                for tk in 0..n {
                    acc += ex[tk] / den * v[tk][base + dd];
                }
                attn[tq][base + dd] = acc;
            }
        }
    }

    // SwiGLU: first half is the gate.
    let mut mm = vec![vec![0f64; mlp]; n];
    for t in 0..n {
        for j in 0..mlp {
            let a = y[t][3 * h + j];
            let act = a / (1.0 + (-a).exp());
            mm[t][j] = act * y[t][3 * h + mlp + j];
        }
    }

    // linear2 over concat(attn, m), gate and residual.
    let mut z = vec![vec![0f64; h]; n];
    let mut out = vec![vec![0f64; h]; n];
    for t in 0..n {
        let mut cat = Vec::with_capacity(lin2);
        cat.extend_from_slice(&attn[t]);
        cat.extend_from_slice(&mm[t]);
        for r in 0..h {
            let mut s = 0f64;
            for c in 0..lin2 {
                s += w.w2[r * lin2 + c] as f64 * cat[c];
            }
            z[t][r] = s;
            out[t][r] = inp.x[t * h + r] as f64 + inp.gate[r] as f64 * s;
        }
    }

    BlockOutput {
        out: flat32(&out),
        attn: flat32(&attn),
        mlp: flat32(&mm),
        z: flat32(&z),
    }
}

#[test]
fn layer_norm_zero_mean_unit_variance_and_modulation() {
    let mut rng = Rng::new(3);
    let x = rng.vec(16, 3.0);
    let zeros = vec![0f32; 16];
    let mut out = vec![0f32; 16];
    layer_norm_modulate(&x, &zeros, &zeros, &mut out);
    let mean = out.iter().map(|&v| v as f64).sum::<f64>() / 16.0;
    let var = out.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / 16.0;
    assert!(mean.abs() < 1e-5, "mean {mean}");
    assert!((var - 1.0).abs() < 1e-4, "var {var}");

    // scale = 1, shift = 1: y + y*1 + 1 = 2y + 1.
    let ones = vec![1f32; 16];
    let mut out2 = vec![0f32; 16];
    layer_norm_modulate(&x, &ones, &ones, &mut out2);
    for i in 0..16 {
        assert!((out2[i] - (2.0 * out[i] + 1.0)).abs() < 1e-5, "index {i}");
    }
}

#[test]
fn rms_norm_gives_unit_mean_square() {
    let mut x = vec![3.0f32, 4.0];
    rms_norm_scale(&mut x, &[1.0, 1.0]);
    let ms = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / 2.0;
    assert!((ms - 1.0).abs() < 1e-5, "mean square {ms}");
}

#[test]
fn rope_identity_and_quarter_turn() {
    let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
    rope(&mut x, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0]);

    // cos = 0, sin = 1: [a, b] -> [-b, a].
    let mut y = vec![1.0f32, 2.0, 3.0, 4.0];
    rope(&mut y, &[0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0]);
    assert_eq!(y, vec![-2.0, 1.0, -4.0, 3.0]);
}

#[test]
fn rope_preserves_pair_norms_for_rotations() {
    let mut rng = Rng::new(21);
    let mut x = rng.vec(8, 2.0);
    let before: Vec<f64> = x.chunks_exact(2).map(|p| (p[0] as f64).hypot(p[1] as f64)).collect();
    let pe = rand_pe(&mut rng, 1, 4);
    rope(&mut x, &pe);
    for (i, p) in x.chunks_exact(2).enumerate() {
        let after = (p[0] as f64).hypot(p[1] as f64);
        assert!((after - before[i]).abs() < 1e-5, "pair {i}: {after} vs {}", before[i]);
    }
}

#[test]
fn silu_values() {
    assert_eq!(silu(0.0), 0.0);
    assert!((silu(50.0) - 50.0).abs() < 1e-3);
    assert!(silu(-50.0).abs() < 1e-6);
}

#[test]
fn dot_matches_simple_sum() {
    let mut rng = Rng::new(9);
    for len in [1usize, 7, 8, 9, 17, 40] {
        let a = rng.vec(len, 1.0);
        let b = rng.vec(len, 1.0);
        let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
        let got = dot(&a, &b) as f64;
        assert!((got - want).abs() < 1e-5, "len {len}: {got} vs {want}");
    }
}

#[test]
fn attention_equal_keys_average_values() {
    let d = tiny();
    let (h, n) = (d.hidden, 4usize);
    let mut rng = Rng::new(11);
    let q = rng.vec(n * h, 1.0);
    let krow = rng.vec(h, 1.0);
    let k: Vec<f32> = (0..n).flat_map(|_| krow.iter().copied()).collect();
    let v = rng.vec(n * h, 1.0);
    let out = attn_stage(d, &q, &k, &v, n, 0, n);
    for tq in 0..n {
        for c in 0..h {
            let mean = (0..n).map(|tk| v[tk * h + c] as f64).sum::<f64>() / n as f64;
            assert!((out[tq * h + c] as f64 - mean).abs() < 1e-5, "tq {tq} c {c}");
        }
    }
}

#[test]
fn attention_peaked_logit_selects_one_value() {
    let d = tiny();
    let (h, hd, n) = (d.hidden, d.head_dim, 4usize);
    let mut rng = Rng::new(12);
    let v = rng.vec(n * h, 1.0);
    let mut q = vec![0f32; n * h];
    let mut k = vec![0f32; n * h];
    // Head 0 of every query asks for dim 0; only key 2 has a large dot product with it.
    for tq in 0..n {
        q[tq * h] = 10.0;
    }
    k[2 * h] = 10.0;
    let out = attn_stage(d, &q, &k, &v, n, 0, n);
    for tq in 0..n {
        for dd in 0..hd {
            let got = out[tq * h + dd];
            let want = v[2 * h + dd];
            assert!((got - want).abs() < 1e-4, "tq {tq} dim {dd}: {got} vs {want}");
        }
    }
}

#[test]
fn reference_matches_naive_and_is_thread_invariant() {
    let d = tiny();
    let n = 5;
    let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15);
    let (w, inp) = rand_problem(&mut rng, d, n);
    let r1 = single_block_ref(&w, &inp, 1);
    let r3 = single_block_ref(&w, &inp, 3);
    let nv = naive_block(&w, &inp);

    let checks = [
        ("out", &r1.out, &nv.out),
        ("attn", &r1.attn, &nv.attn),
        ("mlp", &r1.mlp, &nv.mlp),
        ("z", &r1.z, &nv.z),
    ];
    for (name, got, want) in checks {
        let e = rel_rmse(got, want);
        assert!(e < 1e-5, "{name}: rel_rmse {e}");
    }

    assert_eq!(bits(&r1.out), bits(&r3.out), "out differs across thread counts");
    assert_eq!(bits(&r1.attn), bits(&r3.attn), "attn differs across thread counts");
    assert_eq!(bits(&r1.mlp), bits(&r3.mlp), "mlp differs across thread counts");
    assert_eq!(bits(&r1.z), bits(&r3.z), "z differs across thread counts");
}

#[test]
fn f16_conversion_values_and_round_trip() {
    assert_eq!(f32_to_f16(0.0), 0x0000);
    assert_eq!(f32_to_f16(1.0), 0x3c00);
    assert_eq!(f32_to_f16(-2.0), 0xc000);
    assert_eq!(f32_to_f16(65504.0), 0x7bff);
    assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
    assert_eq!(f32_to_f16(1e10), 0x7c00);

    let table = [0.0f32, 1.0, -2.0, 0.1, 65504.0, 1e-5, 1e-7, -3.0e-5, 1e10, -1e10];
    for &x in &table {
        let back = f16_to_f32(f32_to_f16(x));
        if x.abs() > 65504.0 {
            assert!(back.is_infinite() && back.signum() == x.signum(), "x={x} back={back}");
            continue;
        }
        let tol = if x.abs() < 6.103_515_6e-5 {
            2f32.powi(-25)
        } else {
            x.abs() * 2f32.powi(-11)
        };
        assert!((back - x).abs() <= tol, "x={x} back={back}");
    }
}

#[test]
fn q8_quant_dequant_error_within_half_step() {
    let (rows, k) = (3usize, 64usize);
    let mut rng = Rng::new(5);
    let mut x = rng.vec(rows * k, 2.0);
    // First block is all zeros: d must be 0 and the codes zero.
    for v in x.iter_mut().take(32) {
        *v = 0.0;
    }
    let q = quant_q8_0(&x, rows, k);
    assert_eq!(q.len(), rows * (k / 32) * 34);
    let back = dequant_q8_0(&q, rows, k);
    assert_eq!(back.len(), x.len());
    for b in 0..rows * (k / 32) {
        let d = f16_to_f32(u16::from_le_bytes([q[b * 34], q[b * 34 + 1]]));
        for j in 0..32 {
            let i = b * 32 + j;
            let err = (back[i] - x[i]).abs();
            assert!(err <= d / 2.0 * (1.0 + 1e-5) + 1e-6, "elem {i}: err {err}, d {d}");
        }
    }
    assert!(back[..32].iter().all(|&v| v == 0.0));
}

fn scratch_dir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("image-dit-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn write_tensor(dir: &Path, tag: &str, ty: &str, ne: [usize; 4], vals: &[f32]) {
    let name = format!("{tag}.call0.{ty}.{}x{}x{}x{}.bin", ne[0], ne[1], ne[2], ne[3]);
    let mut bytes = Vec::new();
    for &v in vals {
        if ty == "f16" {
            bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        } else {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join(name), bytes).unwrap();
}

/// Writes a complete, valid block dump set for `d` and `n` tokens, except the tag `skip`
/// (so a test can write that one tag with a wrong shape instead). Returns (x, expected_out).
fn write_valid_set(dir: &Path, d: Dims, n: usize, rng: &mut Rng, skip: &str) -> (Vec<f32>, Vec<f32>) {
    let half = d.head_dim / 2;
    let x = rng.vec(n * d.hidden, 1.0);
    let eo = rng.vec(n * d.hidden, 1.0);
    let pe = rand_pe(rng, n, half);
    let shift = rng.vec(d.hidden, 0.1);
    let scale = rng.vec(d.hidden, 0.1);
    let gate = rng.vec(d.hidden, 0.5);
    if skip != "dump.single.0.in" {
        write_tensor(dir, "dump.single.0.in", "f32", [d.hidden, n, 1, 1], &x);
    }
    if skip != "dump.single.1.in" {
        write_tensor(dir, "dump.single.1.in", "f32", [d.hidden, n, 1, 1], &eo);
    }
    if skip != "dump.pe" {
        write_tensor(dir, "dump.pe", "f32", [2, 2, half, n], &pe);
    }
    if skip != "dump.single.mod.shift" {
        write_tensor(dir, "dump.single.mod.shift", "f32", [d.hidden, 1, 1, 1], &shift);
    }
    if skip != "dump.single.mod.scale" {
        write_tensor(dir, "dump.single.mod.scale", "f32", [d.hidden, 1, 1, 1], &scale);
    }
    if skip != "dump.single.mod.gate" {
        write_tensor(dir, "dump.single.mod.gate", "f32", [d.hidden, 1, 1, 1], &gate);
    }
    (x, eo)
}

#[test]
fn dump_block_round_trip() {
    let dir = scratch_dir("dump-roundtrip");
    let d = tiny();
    let n = 5;
    let mut rng = Rng::new(7);
    let (x, eo) = write_valid_set(&dir, d, n, &mut rng, "");

    let t = read_tag(&dir, "dump.single.0.in").unwrap();
    assert_eq!(t.ne, [d.hidden, n, 1, 1]);

    let bd = read_block_dump(&dir, &d).unwrap();
    assert_eq!(bd.input.n_tokens, n);
    assert_eq!(bd.input.x, x);
    assert_eq!(bd.expected_out, eo);
    assert_eq!(bd.input.pe.len(), n * (d.head_dim / 2) * 4);
    assert_eq!(bd.input.shift.len(), d.hidden);
    assert_eq!(bd.input.scale.len(), d.hidden);
    assert_eq!(bd.input.gate.len(), d.hidden);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dump_f16_is_converted() {
    let dir = scratch_dir("dump-f16");
    write_tensor(&dir, "dump.probe", "f16", [4, 1, 1, 1], &[1.0, -2.0, 0.5, 0.25]);
    let t = read_tag(&dir, "dump.probe").unwrap();
    assert_eq!(t.ne, [4, 1, 1, 1]);
    assert_eq!(t.data, vec![1.0, -2.0, 0.5, 0.25]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dump_rejects_bad_shapes_and_missing_tags() {
    let d = tiny();
    let n = 5;
    let mut rng = Rng::new(8);

    // Wrong hidden size on the input tensor.
    let dir = scratch_dir("dump-bad-x");
    write_valid_set(&dir, d, n, &mut rng, "dump.single.0.in");
    write_tensor(&dir, "dump.single.0.in", "f32", [15, n, 1, 1], &rng.vec(15 * n, 1.0));
    let err = read_block_dump(&dir, &d).err().expect("expected an error");
    assert!(err.contains("dump.single.0.in") && err.contains("15"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);

    // Wrong element count for a modulation vector (ne reported).
    let dir = scratch_dir("dump-bad-shift");
    write_valid_set(&dir, d, n, &mut rng, "dump.single.mod.shift");
    write_tensor(&dir, "dump.single.mod.shift", "f32", [15, 1, 1, 1], &rng.vec(15, 0.1));
    let err = read_block_dump(&dir, &d).err().expect("expected an error");
    assert!(err.contains("dump.single.mod.shift") && err.contains("15"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);

    // pe with the wrong half-width (3 pairs instead of head_dim/2 = 4).
    let dir = scratch_dir("dump-bad-pe");
    write_valid_set(&dir, d, n, &mut rng, "dump.pe");
    write_tensor(&dir, "dump.pe", "f32", [2, 2, 3, n], &rng.vec(2 * 2 * 3 * n, 1.0));
    let err = read_block_dump(&dir, &d).err().expect("expected an error");
    assert!(err.contains("dump.pe"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);

    // Missing tag.
    let dir = scratch_dir("dump-missing");
    let err = read_tag(&dir, "dump.single.0.in").err().expect("expected an error");
    assert!(err.contains("no dump file"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn arith_f32_is_the_plain_reference_and_parity_differs_slightly() {
    use crate::arith::Arith;
    use crate::reference::{single_block_ref_arith, single_block_ref_taps};
    let d = Dims { hidden: 64, heads: 2, head_dim: 32, mlp: 96 };
    let mut r = Rng::new(0x5eed_1234);
    let (w, inp) = rand_problem(&mut r, d, 9);
    let plain = single_block_ref(&w, &inp, 2);
    let f = single_block_ref_arith(&w, &inp, 2, Arith::F32);
    assert_eq!(plain.out, f.out);
    let (p, taps) = single_block_ref_taps(&w, &inp, 2, Arith::SdParity);
    let delta = |o: &[f32]| o.iter().zip(&inp.x).map(|(a, b)| a - b).collect::<Vec<f32>>();
    let e = crate::types::rel_rmse(&delta(&p.out), &delta(&f.out));
    assert!(e > 0.0 && e < 0.05, "parity block delta error {e}");
    assert_eq!(taps.x_mod.len(), 9 * 64);
    assert_eq!(taps.k.len(), 9 * 64);
}
