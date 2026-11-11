//! FLUX.2 VAE decoder in Rust (std + serde_json only), ported from the decode path of
//! sd.cpp (`model/vae/auto_encoder_kl.hpp`, `model/vae/vae.hpp`,
//! `pipeline/diffusion_engine.cpp`, `runtime/preprocessing.hpp`).
//!
//! * [`safetensors`]: minimal reader (F32, F16, BF16).
//! * [`decoder`]: the decoder; layouts are documented in its module docs.
//! * [`png`]: 8-bit RGB writer and RGB/RGBA reader (self-contained inflate).
//!
//! Pixel conversion (sd.cpp `VAE::decode` with `scale_input`, then
//! `preprocessing_float_to_u8`): `v = clamp((x + 1) * 0.5, 0, 1)` in f32, then
//! `0 if v <= 0`, `255 if v >= 1`, else `(v * 255 + 0.5) as u8` (round half up).

use std::path::Path;

pub mod decoder;
pub mod png;
pub mod safetensors;

pub use decoder::VaeDecoder;
pub use safetensors::SafeTensors;

/// 8-bit RGB image, row-major, 3 bytes per pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Loads the decoder from a safetensors file (the load part of [`decode_file`]).
pub fn load_decoder(vae_path: &Path) -> Result<VaeDecoder, String> {
    let st = SafeTensors::open(vae_path)?;
    VaeDecoder::from_safetensors(&st, decoder::DEFAULT_GROUPS)
}

/// Decodes with an already loaded decoder (the decode part of [`decode_file`]).
pub fn decode_with(dec: &VaeDecoder, z_nchw: &[f32], channels: usize, h: usize, w: usize, threads: usize) -> Result<Rgb, String> {
    let (oh, ow) = dec.output_size(h, w);
    let out = dec.decode(z_nchw, channels, h, w, threads)?;
    if out.len() != 3 * oh * ow {
        return Err(format!("decode_file: decoder returned {} values, expected {}", out.len(), 3 * oh * ow));
    }
    Ok(to_rgb(&out, oh, ow))
}

/// Loads the VAE weights from `vae_path` (safetensors), decodes the latent and
/// converts the result to 8-bit RGB.
///
/// `z_nchw` is the de-normalised latent `[channels][h][w]` (channels = 128 for
/// FLUX.2); see the `decoder` module docs. Uses GroupNorm group count 32 (sd.cpp).
/// Returns an image of `output_size(h, w)` (16*w x 16*h for FLUX.2).
pub fn decode_file(
    vae_path: &Path,
    z_nchw: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    threads: usize,
) -> Result<Rgb, String> {
    let dec = load_decoder(vae_path)?;
    decode_with(&dec, z_nchw, channels, h, w, threads)
}

/// Converts raw decoder output `[3][height][width]` to 8-bit RGB with the sd.cpp
/// mapping: `clamp((x + 1) * 0.5, 0, 1)`, then round-half-up to 0..=255.
///
/// Panics if `decoded_chw.len() != 3 * height * width`.
pub fn to_rgb(decoded_chw: &[f32], height: usize, width: usize) -> Rgb {
    let n = height * width;
    assert_eq!(decoded_chw.len(), 3 * n, "to_rgb: expected 3*{height}*{width} values");
    let mut data = vec![0u8; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            let x = decoded_chw[c * n + i];
            let value = (x + 1.0f32) * 0.5f32;
            // std::max(0, std::min(1, value)) as in sd.cpp scale_tensor_to_0_1
            let clamped = if value < 1.0 { value } else { 1.0 };
            let clamped = if 0.0 < clamped { clamped } else { 0.0 };
            data[3 * i + c] = float_to_u8(clamped);
        }
    }
    Rgb {
        width,
        height,
        data,
    }
}

/// sd.cpp `preprocessing_float_to_u8`.
fn float_to_u8(v: f32) -> u8 {
    if v <= 0.0 {
        return 0;
    }
    if v >= 1.0 {
        return 255;
    }
    (v * 255.0f32 + 0.5f32) as u8
}

/// Peak signal-to-noise ratio over all bytes: `10 * log10(255^2 / MSE)`.
/// Returns `f64::INFINITY` for identical images and an error when the sizes differ.
pub fn psnr(a: &Rgb, b: &Rgb) -> Result<f64, String> {
    if a.width != b.width || a.height != b.height {
        return Err(format!(
            "psnr: size mismatch {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        ));
    }
    if a.data.len() != b.data.len() || a.data.len() != a.width * a.height * 3 {
        return Err("psnr: malformed image data".into());
    }
    let mut se = 0f64;
    for (&x, &y) in a.data.iter().zip(&b.data) {
        let d = x as f64 - y as f64;
        se += d * d;
    }
    if se == 0.0 {
        return Ok(f64::INFINITY);
    }
    let mse = se / a.data.len() as f64;
    Ok(10.0 * (255.0f64 * 255.0 / mse).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_to_u8_edges() {
        assert_eq!(float_to_u8(-0.1), 0);
        assert_eq!(float_to_u8(0.0), 0);
        assert_eq!(float_to_u8(1.0), 255);
        assert_eq!(float_to_u8(2.0), 255);
        assert_eq!(float_to_u8(0.5), 128); // 127.5 + 0.5 = 128
        assert_eq!(float_to_u8(0.2), 51); // 51.0 + 0.5 truncates to 51
    }

    #[test]
    fn to_rgb_mapping() {
        // Decoder output -1 -> black, +1 -> white, 0 -> mid grey (128), >1 clamps.
        let decoded = [-1.0f32, 0.0, 1.0, 5.0, /* G */ -1.0, 0.0, 1.0, 5.0, /* B */ -3.0, -1.0, 0.0, 1.0];
        let img = to_rgb(&decoded, 2, 2);
        assert_eq!(img.width, 2);
        assert_eq!(img.height, 2);
        // pixel 0 = (R=-1, G=-1, B=-3)
        assert_eq!(&img.data[0..3], &[0, 0, 0]);
        // pixel 1 = (R=0, G=0, B=-1)
        assert_eq!(&img.data[3..6], &[128, 128, 0]);
        // pixel 2 = (R=1, G=1, B=0)
        assert_eq!(&img.data[6..9], &[255, 255, 128]);
        // pixel 3 = (R=5, G=5, B=1)
        assert_eq!(&img.data[9..12], &[255, 255, 255]);
    }

    #[test]
    fn psnr_known_values() {
        let a = Rgb {
            width: 2,
            height: 1,
            data: vec![0; 6],
        };
        assert_eq!(psnr(&a, &a.clone()).unwrap(), f64::INFINITY);

        // One byte off by 255 out of 6: MSE = 255^2 / 6 -> PSNR = 10 log10(6).
        let mut b = a.clone();
        b.data[0] = 255;
        let p = psnr(&a, &b).unwrap();
        assert!((p - 10.0 * 6f64.log10()).abs() < 1e-9, "{p}");

        // Every byte off by 1: MSE = 1 -> PSNR = 10 log10(65025).
        let c = Rgb {
            width: 2,
            height: 1,
            data: vec![1; 6],
        };
        let p = psnr(&a, &c).unwrap();
        assert!((p - 10.0 * 65025f64.log10()).abs() < 1e-9, "{p}");
    }

    #[test]
    fn psnr_size_mismatch_is_error() {
        let a = Rgb {
            width: 2,
            height: 1,
            data: vec![0; 6],
        };
        let b = Rgb {
            width: 1,
            height: 2,
            data: vec![0; 6],
        };
        assert!(psnr(&a, &b).is_err());
    }
}
