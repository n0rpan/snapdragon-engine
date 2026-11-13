//! v0.1.3 lane H review (F15 / W4-6) of the image kernels, model-free: the public tensor-map
//! constructor must refuse value vectors that do not match their shapes (the Winograd output
//! transform and the GroupNorm apply read biases / norm parameters through raw pointers sized by
//! the shapes), and extreme latents must decode to finite values in every convolution mode
//! (GroupNorm statistics in f64, int8 scales of huge activations, Winograd transforms).
//! Complements `decoder_reference.rs` (accuracy against an f64 reference, thread invariance).

use std::collections::HashMap;
use std::sync::Mutex;

use image_vae::decoder::{set_conv_mode, Tensors, VaeDecoder};

/// The conv mode is process-wide: the decode test of this binary holds this lock while it switches.
static MODE: Mutex<()> = Mutex::new(());

/// Restores the default mode even when an assertion fails.
struct DefaultMode;
impl Drop for DefaultMode {
    fn drop(&mut self) {
        let _ = set_conv_mode("f32");
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 // [-1, 1)
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

/// sd.cpp decoder structure, wide enough for the Winograd form (channels >= 32) and the int8 path
/// (channels a multiple of 64): z = 4, mid = 128, two levels (ch_mult [1, 2]), two ResnetBlocks
/// per level, 8 groups.
const Z: usize = 4;
const MID: usize = 128;
const MULTS: [usize; 2] = [1, 2];
const BLOCKS: usize = 2;
const GROUPS: usize = 8;

struct Gen {
    rng: Lcg,
    t: Tensors,
}

impl Gen {
    fn conv(&mut self, name: &str, cout: usize, cin: usize, k: usize) {
        let scale = (3.0 / (cin * k * k) as f32).sqrt();
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
}

fn make_tensors(seed: u64) -> Tensors {
    let mut g = Gen { rng: Lcg(seed), t: HashMap::new() };
    let ch = MID / MULTS[MULTS.len() - 1];
    g.conv("decoder.post_quant_conv", Z, Z, 1);
    g.conv("decoder.conv_in", MID, Z, 3);
    g.resblock("decoder.mid.block_1", MID, MID);
    g.norm("decoder.mid.attn_1.norm", MID);
    for p in ["q", "k", "v", "proj_out"] {
        g.conv(&format!("decoder.mid.attn_1.{p}"), MID, MID, 1);
    }
    g.resblock("decoder.mid.block_2", MID, MID);
    let mut chs = MID;
    for i in (0..MULTS.len()).rev() {
        let out = ch * MULTS[i];
        for j in 0..BLOCKS {
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

/// A value vector one element short or long of its shape is an error, not a decoder whose kernels
/// later read past the vector (conv weights and biases, 1x1 and 3x3, norm weights and biases).
#[test]
fn from_tensors_rejects_value_length_mismatch() {
    let base = make_tensors(3);
    assert!(VaeDecoder::from_tensors(base.clone(), GROUPS).is_ok(), "the consistent map must load");
    for (name, longer) in [
        ("decoder.conv_in.bias", false),
        ("decoder.conv_in.weight", true),
        ("decoder.mid.block_1.conv1.weight", false),
        ("decoder.mid.attn_1.q.bias", true),
        ("decoder.mid.attn_1.proj_out.weight", false),
        ("decoder.up.1.upsample.conv.bias", false),
        ("decoder.mid.block_1.norm1.bias", false),
        ("decoder.mid.attn_1.norm.bias", true),
        ("decoder.norm_out.weight", false),
        ("decoder.conv_out.bias", false),
    ] {
        let mut t = base.clone();
        let entry = t.get_mut(name).expect(name);
        if longer {
            entry.1.push(0.0);
        } else {
            entry.1.pop();
        }
        assert!(VaeDecoder::from_tensors(t, GROUPS).is_err(), "{name}: inconsistent value count accepted");
    }
}

/// An all-zero latent and a latent of magnitude 1e20 decode to finite values in the f32,
/// Winograd and int8 modes, bitwise independent of the thread count. At 1e20 the f32 square of
/// an activation (1e40) would overflow: GroupNorm must keep its statistics in f64, and the final
/// GroupNorm brings the output back to O(1).
#[test]
fn extreme_latents_decode_finite_in_every_mode() {
    let _lock = MODE.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = DefaultMode;
    let dec = VaeDecoder::from_tensors(make_tensors(41), GROUPS).expect("build decoder");
    let (h, w) = (2usize, 2usize); // 4x4 at the mid block (Winograd), 8x8 at level 0 (int8 width)
    let c = dec.latent_channels();
    let zero = vec![0.0f32; c * h * w];
    let huge = Lcg(9).vec(c * h * w, 1e20);
    for mode in ["f32", "winograd", "int8"] {
        set_conv_mode(mode).unwrap();
        for (label, z) in [("zero", &zero), ("1e20", &huge)] {
            let a = dec.decode(z, c, h, w, 1).expect("decode");
            let b = dec.decode(z, c, h, w, 3).expect("decode");
            assert!(a.iter().all(|v| v.is_finite()), "{mode} {label}: non-finite output");
            let amax = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(amax < 1e4, "{mode} {label}: output magnitude {amax:e}");
            assert!(a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{mode} {label}: depends on the thread count");
        }
    }
}
