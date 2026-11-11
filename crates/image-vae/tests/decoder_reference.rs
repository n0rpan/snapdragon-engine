//! Decoder tests, model-free, against an independent naive f64 implementation that
//! works directly on the tensor map (names as in the FLUX.2 safetensors file).
//!
//! Also checks:
//! * un-patchify against a literal emulation of the ggml reshape/permute sequence
//!   from sd.cpp `AutoEncoderKLModel::decode` (index-valued tensors);
//! * thread-count invariance (bitwise);
//! * the file path (safetensors written to disk and read back).

use std::collections::HashMap;

use image_vae::decoder::{unpatchify, Tensors, VaeDecoder};
use image_vae::SafeTensors;

// ---------------------------------------------------------------------------
// ggml emulation for the un-patchify test
// ---------------------------------------------------------------------------

/// ggml_permute semantics: `result.ne[axes[k]] = ne[k]`; the element at source
/// coordinates `s` moves to coordinates `t` with `t[axes[k]] = s[k]`. ne[0] is fastest.
fn ggml_permute(data: &[usize], ne: [usize; 4], axes: [usize; 4]) -> (Vec<usize>, [usize; 4]) {
    let mut newne = [0usize; 4];
    for k in 0..4 {
        newne[axes[k]] = ne[k];
    }
    let mut out = vec![0usize; data.len()];
    for s3 in 0..ne[3] {
        for s2 in 0..ne[2] {
            for s1 in 0..ne[1] {
                for s0 in 0..ne[0] {
                    let s = [s0, s1, s2, s3];
                    let mut t = [0usize; 4];
                    for k in 0..4 {
                        t[axes[k]] = s[k];
                    }
                    let src = s0 + ne[0] * (s1 + ne[1] * (s2 + ne[2] * s3));
                    let dst = t[0] + newne[0] * (t[1] + newne[1] * (t[2] + newne[2] * t[3]));
                    out[dst] = data[src];
                }
            }
        }
    }
    (out, newne)
}

/// sd.cpp `ggml_ext_torch_permute`: torch-style axes -> ggml axes
/// (`ggml_axes[i] = j` such that `torch_axes[j] == i`).
fn torch_to_ggml_axes(torch: [usize; 4]) -> [usize; 4] {
    let mut g = [0usize; 4];
    for i in 0..4 {
        g[i] = torch.iter().position(|&t| t == i).expect("permutation");
    }
    g
}

#[test]
fn unpatchify_matches_ggml_reshape_permute_sequence() {
    let (c, h, w) = (3usize, 4usize, 5usize);
    let p = 2usize;
    let n_in = c * p * p * h * w;
    // Index-valued latent: element value = its linear index in [c*4][h][w] order.
    let idx: Vec<usize> = (0..n_in).collect();

    // ggml view of z is ne = [w, h, C*p*p, N=1]; step 1 reshapes it (memory unchanged).
    let mut data = idx.clone();
    // 1. reshape_4d(z, w*h, p*p, C, N)
    let mut ne = [w * h, p * p, c, 1];
    // 2. cont(permute(z, 1, 0, 2, 3))  -> [N, C, h*w, p*p] in torch order
    let (d, n2) = ggml_permute(&data, ne, torch_to_ggml_axes([1, 0, 2, 3]));
    assert_eq!(n2, [p * p, w * h, c, 1], "shape after permute 1");
    data = d;
    // 3. reshape_4d(z, p, p, w, h*C*N)
    ne = [p, p, w, h * c];
    // 4. cont(permute(z, 0, 2, 1, 3))
    let (d, n2) = ggml_permute(&data, ne, torch_to_ggml_axes([0, 2, 1, 3]));
    data = d;
    ne = n2;
    assert_eq!(ne, [p, w, p, h * c], "shape before final reshape");
    // 5. reshape_4d(z, W, H, C, N) with W = w*p, H = h*p
    ne = [p * w, p * h, c, 1];
    assert_eq!(ne.iter().product::<usize>(), data.len());

    let z_f32: Vec<f32> = idx.iter().map(|&i| i as f32).collect();
    let got = unpatchify(&z_f32, c, h, w);
    let want: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    assert_eq!(got, want);
}

// ---------------------------------------------------------------------------
// Random tensor maps with the sd.cpp decoder structure
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

#[derive(Clone, Copy)]
struct Cfg {
    /// Latent channels after un-patchify (z_channels in sd.cpp).
    z: usize,
    /// Mid channels (= ch * mults[last]).
    mid: usize,
    /// ch_mult list (levels), e.g. [1, 2, 4].
    mults: &'static [usize],
    /// ResnetBlocks per level (num_res_blocks + 1).
    blocks: usize,
    groups: usize,
}

impl Cfg {
    fn base_ch(&self) -> usize {
        self.mid / self.mults[self.mults.len() - 1]
    }
}

const CFG_A: Cfg = Cfg {
    z: 2,
    mid: 16,
    mults: &[1, 2],
    blocks: 3,
    groups: 4,
};
const CFG_B: Cfg = Cfg {
    z: 4,
    mid: 32,
    mults: &[1, 2, 4],
    blocks: 2,
    groups: 8,
};
/// Wide enough for the Winograd form (channels >= 32) and the int8 path (channels a multiple of 64).
const CFG_WIDE: Cfg = Cfg {
    z: 4,
    mid: 128,
    mults: &[1, 2],
    blocks: 2,
    groups: 8,
};

struct Gen {
    rng: Lcg,
    t: Tensors,
}

impl Gen {
    fn conv(&mut self, name: &str, cout: usize, cin: usize, k: usize) {
        let fan_in = (cin * k * k) as f32;
        let scale = (3.0 / fan_in).sqrt();
        let w = self.rng.vec(cout * cin * k * k, scale);
        let b = self.rng.vec(cout, 0.1);
        self.t.insert(format!("{name}.weight"), (vec![cout, cin, k, k], w));
        self.t.insert(format!("{name}.bias"), (vec![cout], b));
    }
    fn norm(&mut self, name: &str, c: usize) {
        let w: Vec<f32> = self.rng.vec(c, 0.2).into_iter().map(|v| 1.0 + v).collect();
        let b = self.rng.vec(c, 0.1);
        self.t.insert(format!("{name}.weight"), (vec![c], w));
        self.t.insert(format!("{name}.bias"), (vec![c], b));
    }
    fn resblock(&mut self, name: &str, cin: usize, cout: usize) {
        self.norm(&format!("{name}.norm1"), cin);
        self.conv(&format!("{name}.conv1"), cout, cin, 3);
        self.norm(&format!("{name}.norm2"), cout);
        self.conv(&format!("{name}.conv2"), cout, cout, 3);
        if cin != cout {
            self.conv(&format!("{name}.nin_shortcut"), cout, cin, 1);
        }
    }
    fn attn(&mut self, name: &str, c: usize) {
        self.norm(&format!("{name}.norm"), c);
        for p in ["q", "k", "v", "proj_out"] {
            self.conv(&format!("{name}.{p}"), c, c, 1);
        }
    }
}

fn make_tensors(cfg: Cfg, seed: u64) -> Tensors {
    let mut g = Gen {
        rng: Lcg(seed),
        t: HashMap::new(),
    };
    let ch = cfg.base_ch();
    let n = cfg.mults.len();
    g.conv("decoder.post_quant_conv", cfg.z, cfg.z, 1);
    g.conv("decoder.conv_in", cfg.mid, cfg.z, 3);
    g.resblock("decoder.mid.block_1", cfg.mid, cfg.mid);
    g.attn("decoder.mid.attn_1", cfg.mid);
    g.resblock("decoder.mid.block_2", cfg.mid, cfg.mid);
    let mut chs = cfg.mid;
    for i in (0..n).rev() {
        let out = ch * cfg.mults[i];
        for j in 0..cfg.blocks {
            g.resblock(&format!("decoder.up.{i}.block.{j}"), chs, out);
            chs = out;
        }
        if i != 0 {
            g.conv(&format!("decoder.up.{i}.upsample.conv"), chs, chs, 3);
        }
    }
    g.norm("decoder.norm_out", chs);
    g.conv("decoder.conv_out", 3, chs, 3);
    g.t
}

// ---------------------------------------------------------------------------
// Naive f64 reference (independent of src/decoder.rs)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RT {
    c: usize,
    h: usize,
    w: usize,
    d: Vec<f64>,
}

impl RT {
    fn zeros(c: usize, h: usize, w: usize) -> RT {
        RT {
            c,
            h,
            w,
            d: vec![0.0; c * h * w],
        }
    }
    fn at(&self, c: usize, y: usize, x: usize) -> f64 {
        self.d[(c * self.h + y) * self.w + x]
    }
}

struct Reference<'a> {
    t: &'a Tensors,
    groups: usize,
}

impl Reference<'_> {
    fn get(&self, name: &str) -> (&[usize], &[f32]) {
        let (s, v) = self.t.get(name).unwrap_or_else(|| panic!("missing {name}"));
        (s.as_slice(), v.as_slice())
    }

    fn conv(&self, base: &str, x: &RT) -> RT {
        let (ws, w) = self.get(&format!("{base}.weight"));
        let (_, b) = self.get(&format!("{base}.bias"));
        let (cout, cin, k) = (ws[0], ws[1], ws[2]);
        assert_eq!(cin, x.c, "{base}: cin");
        assert_eq!(ws[2], ws[3]);
        let p = (k / 2) as isize;
        let mut out = RT::zeros(cout, x.h, x.w);
        for co in 0..cout {
            for y in 0..x.h {
                for xx in 0..x.w {
                    let mut acc = b[co] as f64;
                    for ci in 0..cin {
                        for ky in 0..k {
                            for kx in 0..k {
                                let iy = y as isize + ky as isize - p;
                                let ix = xx as isize + kx as isize - p;
                                if iy < 0 || ix < 0 || iy >= x.h as isize || ix >= x.w as isize {
                                    continue;
                                }
                                let wv = w[((co * cin + ci) * k + ky) * k + kx] as f64;
                                acc += wv * x.at(ci, iy as usize, ix as usize);
                            }
                        }
                    }
                    out.d[(co * x.h + y) * x.w + xx] = acc;
                }
            }
        }
        out
    }

    fn gn(&self, base: &str, x: &RT) -> RT {
        let (_, w) = self.get(&format!("{base}.weight"));
        let (_, b) = self.get(&format!("{base}.bias"));
        let cg = x.c / self.groups;
        let n = cg * x.h * x.w;
        let mut out = RT::zeros(x.c, x.h, x.w);
        for g in 0..self.groups {
            let mut vals = Vec::with_capacity(n);
            for c in g * cg..(g + 1) * cg {
                for y in 0..x.h {
                    for xx in 0..x.w {
                        vals.push(x.at(c, y, xx));
                    }
                }
            }
            let mean = vals.iter().sum::<f64>() / n as f64;
            let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
            for (k, c) in (g * cg..(g + 1) * cg).enumerate() {
                for y in 0..x.h {
                    for xx in 0..x.w {
                        let v = vals[k * x.h * x.w + y * x.w + xx];
                        out.d[(c * x.h + y) * x.w + xx] =
                            (v - mean) / (var + 1e-6).sqrt() * w[c] as f64 + b[c] as f64;
                    }
                }
            }
        }
        out
    }

    fn silu(x: &RT) -> RT {
        let mut o = x.clone();
        for v in o.d.iter_mut() {
            *v /= 1.0 + (-*v).exp();
        }
        o
    }

    fn add(a: &RT, b: &RT) -> RT {
        let mut o = a.clone();
        for (x, y) in o.d.iter_mut().zip(&b.d) {
            *x += y;
        }
        o
    }

    fn resblock(&self, base: &str, x: &RT) -> RT {
        let h = self.conv(&format!("{base}.conv1"), &Self::silu(&self.gn(&format!("{base}.norm1"), x)));
        let h = self.conv(&format!("{base}.conv2"), &Self::silu(&self.gn(&format!("{base}.norm2"), &h)));
        let sc = if self.t.contains_key(&format!("{base}.nin_shortcut.weight")) {
            self.conv(&format!("{base}.nin_shortcut"), x)
        } else {
            x.clone()
        };
        Self::add(&h, &sc)
    }

    fn attn(&self, base: &str, x: &RT) -> RT {
        let xn = self.gn(&format!("{base}.norm"), x);
        let q = self.conv(&format!("{base}.q"), &xn);
        let k = self.conv(&format!("{base}.k"), &xn);
        let v = self.conv(&format!("{base}.v"), &xn);
        let c = x.c;
        let l = x.h * x.w;
        let scale = 1.0 / (c as f64).sqrt();
        let mut o = RT::zeros(c, x.h, x.w);
        for i in 0..l {
            let mut s: Vec<f64> = (0..l)
                .map(|j| (0..c).map(|cc| q.d[cc * l + i] * k.d[cc * l + j]).sum::<f64>() * scale)
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for v in s.iter_mut() {
                *v = (*v - m).exp();
            }
            let z: f64 = s.iter().sum();
            for cc in 0..c {
                o.d[cc * l + i] = (0..l).map(|j| s[j] / z * v.d[cc * l + j]).sum();
            }
        }
        let p = self.conv(&format!("{base}.proj_out"), &o);
        Self::add(&p, x)
    }

    fn upsample(x: &RT) -> RT {
        let mut o = RT::zeros(x.c, 2 * x.h, 2 * x.w);
        for c in 0..x.c {
            for y in 0..2 * x.h {
                for xx in 0..2 * x.w {
                    o.d[(c * 2 * x.h + y) * 2 * x.w + xx] = x.at(c, y / 2, xx / 2);
                }
            }
        }
        o
    }

    /// Un-patchify in torch style: reshape [C, p, p, h, w] -> permute to [C, h, p, w, p]
    /// -> reshape [C, h*p, w*p] (diffusers `unpatchify_latents`).
    fn unpatchify_torch(z: &[f32], c: usize, h: usize, w: usize) -> RT {
        let p = 2;
        let mut o = RT::zeros(c, h * p, w * p);
        // Flat index in reshaped tensor [C, p, p, h, w]:
        for cc in 0..c {
            for p1 in 0..p {
                for p2 in 0..p {
                    for y in 0..h {
                        for x in 0..w {
                            let flat = (((cc * p + p1) * p + p2) * h + y) * w + x;
                            // permuted index [C, h, p1, w, p2]
                            let oy = y * p + p1;
                            let ox = x * p + p2;
                            o.d[(cc * h * p + oy) * w * p + ox] = z[flat] as f64;
                        }
                    }
                }
            }
        }
        o
    }

    fn decode(&self, z: &[f32], c4: usize, h: usize, w: usize) -> Vec<f64> {
        let x = Self::unpatchify_torch(z, c4 / 4, h, w);
        let x = self.conv("decoder.post_quant_conv", &x);
        let x = self.conv("decoder.conv_in", &x);
        let x = self.resblock("decoder.mid.block_1", &x);
        let x = self.attn("decoder.mid.attn_1", &x);
        let mut x = self.resblock("decoder.mid.block_2", &x);
        // Levels from the tensor map: count "decoder.up.i" entries.
        let n_levels = (0..).take_while(|i| self.t.contains_key(&format!("decoder.up.{i}.block.0.conv1.weight"))).count();
        for i in (0..n_levels).rev() {
            let mut j = 0;
            while self.t.contains_key(&format!("decoder.up.{i}.block.{j}.conv1.weight")) {
                x = self.resblock(&format!("decoder.up.{i}.block.{j}"), &x);
                j += 1;
            }
            if i != 0 {
                x = Self::upsample(&x);
                x = self.conv(&format!("decoder.up.{i}.upsample.conv"), &x);
            }
        }
        let x = self.gn("decoder.norm_out", &x);
        let x = Self::silu(&x);
        self.conv("decoder.conv_out", &x).d
    }
}

fn rel_rmse(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut num = 0f64;
    let mut den = 0f64;
    for (&g, &w) in got.iter().zip(want) {
        num += (g as f64 - w) * (g as f64 - w);
        den += w * w;
    }
    (num / den).sqrt()
}

fn random_latent(n: usize, seed: u64) -> Vec<f32> {
    Lcg(seed).vec(n, 1.0)
}

fn check_config(cfg: Cfg, seed: u64, h: usize, w: usize) {
    let t = make_tensors(cfg, seed);
    let latent = random_latent(4 * cfg.z * h * w, seed ^ 0x5555);
    let reference = Reference {
        t: &t,
        groups: cfg.groups,
    };
    let want = reference.decode(&latent, 4 * cfg.z, h, w);

    let dec = VaeDecoder::from_tensors(t.clone(), cfg.groups).expect("build decoder");
    assert_eq!(dec.latent_channels(), 4 * cfg.z);
    let (oh, ow) = dec.output_size(h, w);
    let got = dec.decode(&latent, 4 * cfg.z, h, w, 2).expect("decode");
    assert_eq!(got.len(), 3 * oh * ow);
    assert_eq!(want.len(), 3 * oh * ow);
    let e = rel_rmse(&got, &want);
    eprintln!("config z={} mid={} levels={} rel_rmse={e:e}", cfg.z, cfg.mid, cfg.mults.len());
    assert!(e < 1e-5, "relative RMSE {e:e} >= 1e-5");
}

#[test]
fn decoder_matches_f64_reference_small() {
    check_config(CFG_A, 11, 3, 3); // L = 36 > ATTN block of 32 rows: exercises a partial block
}

#[test]
fn decoder_matches_f64_reference_three_levels() {
    check_config(CFG_B, 23, 2, 4);
}

/// The other arithmetic modes against the same f64 reference: the Winograd form of the 3x3
/// convolutions (f32, rounding-level difference) and the channel-major path kept for A/B.
#[test]
fn winograd_and_channel_major_match_f64_reference() {
    use image_vae::decoder::set_conv_mode;
    for (cfg, seed, h, w) in [(CFG_A, 11u64, 3usize, 3usize), (CFG_B, 23, 2, 4), (CFG_WIDE, 31, 2, 3)] {
        let t = make_tensors(cfg, seed);
        let latent = random_latent(4 * cfg.z * h * w, seed ^ 0x5555);
        let want = Reference { t: &t, groups: cfg.groups }.decode(&latent, 4 * cfg.z, h, w);
        let dec = VaeDecoder::from_tensors(t.clone(), cfg.groups).expect("build decoder");
        let direct = dec.decode(&latent, 4 * cfg.z, h, w, 2).expect("decode");
        set_conv_mode("winograd").unwrap();
        let wino = dec.decode(&latent, 4 * cfg.z, h, w, 2).expect("decode winograd");
        let wino5 = dec.decode(&latent, 4 * cfg.z, h, w, 5).expect("decode winograd");
        set_conv_mode("f32").unwrap();
        let (major, _) = dec.decode_timed_channel_major(&latent, 4 * cfg.z, h, w, 2).expect("decode channel-major");
        let (e_d, e_w, e_m) = (rel_rmse(&direct, &want), rel_rmse(&wino, &want), rel_rmse(&major, &want));
        eprintln!("config z={} mid={}: rel_rmse direct {e_d:e}, winograd {e_w:e}, channel-major {e_m:e}", cfg.z, cfg.mid);
        assert!(e_w < 1e-5, "winograd: relative RMSE {e_w:e} >= 1e-5");
        assert!(e_m < 1e-5, "channel-major: relative RMSE {e_m:e} >= 1e-5");
        assert!(wino.iter().zip(&wino5).all(|(a, b)| a.to_bits() == b.to_bits()), "winograd decode depends on the thread count");
        if cfg.mid >= 32 {
            // wide enough for the Winograd form to be taken: it must not be the direct result bit for bit
            assert!(wino.iter().zip(&direct).any(|(a, b)| a.to_bits() != b.to_bits()), "winograd mode did not change anything: is it running?");
        }
    }
}

/// The int8 mode on a config wide enough to take it (channels a multiple of 64): close to f32,
/// independent of the thread count.
#[test]
fn int8_mode_is_close_and_thread_invariant() {
    use image_vae::decoder::set_conv_mode;
    let (cfg, h, w) = (CFG_WIDE, 2usize, 4usize);
    let t = make_tensors(cfg, 77);
    let latent = random_latent(4 * cfg.z * h * w, 5);
    let dec = VaeDecoder::from_tensors(t, cfg.groups).unwrap();
    let f = dec.decode(&latent, 4 * cfg.z, h, w, 2).unwrap();
    set_conv_mode("int8").unwrap();
    let a = dec.decode(&latent, 4 * cfg.z, h, w, 1).unwrap();
    let b = dec.decode(&latent, 4 * cfg.z, h, w, 6).unwrap();
    set_conv_mode("f32").unwrap();
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "int8 decode depends on the thread count");
    let want: Vec<f64> = f.iter().map(|v| *v as f64).collect();
    let e = rel_rmse(&a, &want);
    eprintln!("int8 vs f32 decode on CFG_WIDE: rel_rmse {e:e}");
    if cfg!(target_arch = "aarch64") {
        assert!(e > 1e-6 && e < 0.2, "int8 decode: relative RMSE {e:e} vs f32");
    }
}

#[test]
fn decode_is_thread_invariant_and_pure() {
    for cfg in [CFG_A, CFG_B] {
        let t = make_tensors(cfg, 99);
        let dec = VaeDecoder::from_tensors(t, cfg.groups).unwrap();
        let (h, w) = (3usize, 5usize);
        let latent = random_latent(4 * cfg.z * h * w, 7);
        let base = dec.decode(&latent, 4 * cfg.z, h, w, 1).unwrap();
        for threads in [2usize, 3, 7] {
            let got = dec.decode(&latent, 4 * cfg.z, h, w, threads).unwrap();
            assert!(
                got.len() == base.len() && got.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
                "threads={threads} differs from threads=1"
            );
        }
        let again = dec.decode(&latent, 4 * cfg.z, h, w, 1).unwrap();
        assert!(again.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[test]
fn decode_rejects_bad_inputs() {
    let cfg = CFG_A;
    let dec = VaeDecoder::from_tensors(make_tensors(cfg, 3), cfg.groups).unwrap();
    let latent = random_latent(4 * cfg.z * 2 * 2, 1);
    assert!(dec.decode(&latent, 4 * cfg.z + 4, 2, 2, 1).is_err());
    assert!(dec.decode(&latent[1..], 4 * cfg.z, 2, 2, 1).is_err());
    assert!(dec.decode(&[], 4 * cfg.z, 0, 0, 1).is_err());
}

#[test]
fn from_tensors_rejects_wrong_group_count() {
    // 16 mid channels are not divisible by 3 groups.
    let t = make_tensors(CFG_A, 5);
    assert!(VaeDecoder::from_tensors(t, 3).is_err());
}

/// Writes an F32 safetensors file with the given tensors (sorted names).
fn write_safetensors(path: &std::path::Path, t: &Tensors) {
    let mut names: Vec<&String> = t.keys().collect();
    names.sort();
    let mut header = String::from("{");
    let mut body: Vec<u8> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let (shape, vals) = &t[*name];
        let start = body.len();
        for v in vals {
            body.extend_from_slice(&v.to_le_bytes());
        }
        if i > 0 {
            header.push(',');
        }
        let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
            shape_s.join(","),
            body.len()
        ));
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&body);
    std::fs::write(path, file).unwrap();
}

#[test]
fn safetensors_file_round_trip_decodes_identically() {
    let cfg = CFG_A;
    let t = make_tensors(cfg, 17);
    let mut all = t.clone();
    // Non-decoder tensors must be ignored by the decoder loader.
    all.insert("encoder.conv_in.weight".into(), (vec![2, 2], vec![0.5; 4]));
    let dir = std::env::temp_dir().join(format!("image-vae-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny-vae.safetensors");
    write_safetensors(&path, &all);

    let st = SafeTensors::open(&path).unwrap();
    assert!(st.info("decoder.conv_in.weight").is_some());
    let (shape, vals) = st.tensor_f32("decoder.conv_in.weight").unwrap();
    assert_eq!(shape, t["decoder.conv_in.weight"].0);
    assert_eq!(vals, t["decoder.conv_in.weight"].1);

    let from_file = VaeDecoder::from_safetensors(&st, cfg.groups).unwrap();
    let from_map = VaeDecoder::from_tensors(t, cfg.groups).unwrap();
    let latent = random_latent(4 * cfg.z * 2 * 3, 4);
    let a = from_file.decode(&latent, 4 * cfg.z, 2, 3, 2).unwrap();
    let b = from_map.decode(&latent, 4 * cfg.z, 2, 3, 2).unwrap();
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert_eq!(from_file.output_size(2, 3), (8, 12));
    assert_eq!(a.len(), 3 * 8 * 12);

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
