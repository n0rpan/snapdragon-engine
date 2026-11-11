//! Own initial noise for the FLUX.2 latent: deterministic Gaussian tokens from a u64 seed.
//!
//! Generator: SplitMix64 (Steele, Lea, Flood 2014; the constants below) produces 64-bit words;
//! a word becomes a uniform `u` in (0, 1) from its top 53 bits (`(w >> 11 + 0.5) / 2^53`), and
//! Box-Muller turns two uniforms into two independent standard normals in f64, rounded to f32.
//! The sequence is a pure function of the seed, so the same seed gives the same bits on any machine.
//!
//! This is OUR generator. sd.cpp draws its initial noise with its own RNG, so the same seed number
//! gives a different latent here than in sd.cpp: images from `--noise own` will not match sd.cpp
//! images for the same seed. Use a hand-over (`--noise sdcpp`) when a comparison needs sd.cpp's noise.

/// SplitMix64 state.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in the open interval (0, 1).
    pub fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / (1u64 << 53) as f64)
    }
}

/// `n` standard normal samples from `seed` (Box-Muller, pairs are consumed in order).
pub fn gaussian(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let u1 = rng.next_unit();
        let u2 = rng.next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Initial latent tokens `[n_img][channels]`, row-major, channel fastest (the DiT token layout).
pub fn initial_tokens(seed: u64, n_img: usize, channels: usize) -> Vec<f32> {
    gaussian(seed, n_img * channels)
}

// ---------------------------------------------------------------------------------------------
// sd.cpp's own initial noise (IMG-SHIP): same seed -> same latent as sd-cli
// ---------------------------------------------------------------------------------------------

const PHILOX_M: [u32; 2] = [0xD251_1F53, 0xCD9E_8D57];
const PHILOX_W: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];

/// `n` normals exactly as sd.cpp's `PhiloxRNG::randn(n)` right after `manual_seed(seed)` (offset 0),
/// which is the default `--rng cuda` (`src/core/rng_philox.hpp`, a port of the webui torch-cuda
/// imitation). Counter `[offset, 0, i, 0]`, key `[seed lo, seed hi]`, 10 rounds, then a single-output
/// Box-Muller in f32 (`sqrt(-2 ln u) * sin(v)`) with the same f32 constants and libm calls.
pub fn philox_randn(seed: u64, n: usize) -> Vec<f32> {
    let offset: u32 = 0;
    let two_pow32_inv: f32 = 2.328_306_4e-10;
    let two_pow32_inv_2pi: f32 = 2.328_306_4e-10 * 6.283_185_5;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut c = [offset, 0u32, i as u32, 0u32];
        let mut k0 = seed as u32;
        let mut k1 = (seed >> 32) as u32;
        for round in 0..10 {
            let v1 = c[0] as u64 * PHILOX_M[0] as u64;
            let v2 = c[2] as u64 * PHILOX_M[1] as u64;
            c = [(v2 >> 32) as u32 ^ c[1] ^ k0, v2 as u32, (v1 >> 32) as u32 ^ c[3] ^ k1, v1 as u32];
            if round < 9 {
                k0 = k0.wrapping_add(PHILOX_W[0]);
                k1 = k1.wrapping_add(PHILOX_W[1]);
            }
        }
        let x = c[0] as f32;
        let y = c[1] as f32;
        // GCC contracts `x * a + b` into one fused multiply-add on aarch64 (-ffp-contract=fast)
        let u = x.mul_add(two_pow32_inv, two_pow32_inv / 2.0);
        let v = y.mul_add(two_pow32_inv_2pi, two_pow32_inv_2pi / 2.0);
        // sd.cpp calls the unqualified C `log`/`sqrt`/`sin` (double precision) on f32 values; the
        // f32 result is rounded at `float s` and at `float r1`, as below.
        let s = (-2.0f64 * (u as f64).ln()).sqrt() as f32;
        out.push((s as f64 * (v as f64).sin()) as f32);
    }
    out
}

/// sd.cpp's initial latent tokens `[n_img][channels]` for a txt2img run with `seed` on a `(gh, gw)`
/// grid: `randn_like(init_latent)` fills the ggml tensor `[w, h, c, 1]` (w fastest, i.e. NCHW) in
/// linear order, then patchify (patch 1) makes token `row * gw + col` with channel fastest. With the
/// flow schedule's first sigma 1.0 this is exactly the DiT's input at step 0 (`dump.flux.img` call 0).
pub fn sdcpp_initial_tokens(seed: u64, gh: usize, gw: usize, channels: usize) -> Vec<f32> {
    let nchw = philox_randn(seed, channels * gh * gw);
    crate::latent::nchw_to_tokens(&nchw, channels, gh, gw).expect("sizes are consistent by construction")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_variance_over_1e5_samples() {
        let n = 100_000;
        let v = gaussian(12345, n);
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "variance {var}");
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn same_seed_same_bits_different_seed_different_bits() {
        let a = initial_tokens(7, 16, 8);
        let b = initial_tokens(7, 16, 8);
        let c = initial_tokens(8, 16, 8);
        assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert_ne!(a, c);
        assert_eq!(a.len(), 16 * 8);
    }

    #[test]
    fn unit_samples_are_open_interval() {
        let mut r = SplitMix64::new(1);
        for _ in 0..10_000 {
            let u = r.next_unit();
            assert!(u > 0.0 && u < 1.0);
        }
    }

    /// The bits sd-cli itself fed the DiT at step 0, when the seed hand-overs are on this machine
    /// (data files only, no model). Skips quietly elsewhere. Override the root with OXI_NOISE_DUMPS.
    #[test]
    fn sdcpp_noise_matches_sdcli_dumps_bitwise() {
        let root = std::env::var("OXI_NOISE_DUMPS").map(std::path::PathBuf::from).unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".local/share/ai-stack/pictures/imageflow-20261110")
        });
        let mut checked = 0;
        for seed in [42u64, 43, 44] {
            let f = root.join(format!("s{seed}/dump.flux.img.call0.f32.128x1024x1x1.bin"));
            let Ok(bytes) = std::fs::read(&f) else { continue };
            let want: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            let got: Vec<u32> = sdcpp_initial_tokens(seed, 32, 32, 128).iter().map(|x| x.to_bits()).collect();
            assert_eq!(got.len(), want.len());
            let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
            assert_eq!(same, want.len(), "seed {seed}: {same}/{} values bit-identical", want.len());
            checked += 1;
        }
        eprintln!("sdcpp noise: {checked} seed dumps checked bitwise");
    }

    #[test]
    fn philox_is_deterministic_seed_dependent_and_normal() {
        let a = philox_randn(42, 4096);
        assert_eq!(a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), philox_randn(42, 4096).iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        assert_ne!(a, philox_randn(43, 4096));
        // element i depends only on (seed, i): a prefix of a longer draw is the shorter draw
        assert_eq!(&philox_randn(42, 8192)[..4096], &a[..]);
        let n = 100_000;
        let v = philox_randn(7, n);
        let mean = v.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let var = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02 && (var - 1.0).abs() < 0.02, "mean {mean} var {var}");
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn sdcpp_tokens_are_the_nchw_draw_transposed() {
        let (gh, gw, c) = (3, 5, 4);
        let flat = philox_randn(9, c * gh * gw);
        let tok = sdcpp_initial_tokens(9, gh, gw, c);
        for row in 0..gh {
            for col in 0..gw {
                for ch in 0..c {
                    assert_eq!(tok[(row * gw + col) * c + ch].to_bits(), flat[ch * gh * gw + row * gw + col].to_bits());
                }
            }
        }
    }
}
