//! Token <-> latent layout for the klein DiT (patch size 1), and the FLUX.2 VAE latent statistics.
//!
//! Tokens are `[n][c]` with token `t = row * w + col` (row-major over h, then w) and channel fastest,
//! which is what `patchify` (`model/diffusion/dit.hpp` 8-30, patch 1, patch_last) produces. The
//! NCHW latent is `[c][h][w]` with `w` fastest, so `nchw[c*h*w + row*w + col] = tokens[(row*w+col)*c + ch]`.
//!
//! `diffusion_to_vae_latents`: `auto_encoder_kl.hpp` `diffusion_to_vae_latents` with
//! `sd_version_uses_flux2_vae`: `(latents * std) / scale_factor + mean`, `scale_factor = 1.0`
//! (`auto_encoder_kl.hpp` ~693-694), tables from `get_latents_mean_std` (~776-816).

/// Channels of the FLUX.2 latent (both the DiT tokens and the VAE latent).
pub const LATENT_CHANNELS: usize = 128;

/// `get_latents_mean_std` mean, 128 entries, copied from `auto_encoder_kl.hpp` lines 783-798.
pub const VAE_LATENT_MEAN: [f32; 128] = [
    -0.0676, -0.0715, -0.0753, -0.0745, 0.0223, 0.0180, 0.0142, 0.0184, -0.0001, -0.0063, -0.0002, -0.0031, -0.0272, -0.0281, -0.0276, -0.0290, -0.0769, -0.0672, -0.0902, -0.0892, 0.0168, 0.0152, 0.0079, 0.0086, 0.0083, 0.0015, 0.0003, -0.0043, -0.0439, -0.0419, -0.0438, -0.0431, -0.0102, -0.0132, -0.0066, -0.0048, -0.0311, -0.0306, -0.0279, -0.0180, 0.0030, 0.0015, 0.0126, 0.0145, 0.0347, 0.0338, 0.0337, 0.0283, 0.0020, 0.0047, 0.0047, 0.0050, 0.0123, 0.0081, 0.0081, 0.0146, 0.0681, 0.0679, 0.0767, 0.0732, -0.0462, -0.0474, -0.0392, -0.0511, -0.0528, -0.0477, -0.0470, -0.0517, -0.0317, -0.0316, -0.0345, -0.0283, 0.0510, 0.0445, 0.0578, 0.0458, -0.0412, -0.0458, -0.0487, -0.0467, -0.0088, -0.0106, -0.0088, -0.0046, -0.0376, -0.0432, -0.0436, -0.0499, 0.0118, 0.0166, 0.0203, 0.0279, 0.0113, 0.0129, 0.0016, 0.0072, -0.0118, -0.0018, -0.0141, -0.0054, -0.0091, -0.0138, -0.0145, -0.0187, 0.0323, 0.0305, 0.0259, 0.0300, 0.0540, 0.0614, 0.0495, 0.0590, -0.0511, -0.0603, -0.0478, -0.0524, -0.0227, -0.0274, -0.0154, -0.0255, -0.0572, -0.0565, -0.0518, -0.0496, 0.0116, 0.0054, 0.0163, 0.0104
];

/// `get_latents_mean_std` std, 128 entries, copied from `auto_encoder_kl.hpp` lines 800-815.
pub const VAE_LATENT_STD: [f32; 128] = [
    1.8029, 1.7786, 1.7868, 1.7837, 1.7717, 1.7590, 1.7610, 1.7479, 1.7336, 1.7373, 1.7340, 1.7343, 1.8626, 1.8527, 1.8629, 1.8589, 1.7593, 1.7526, 1.7556, 1.7583, 1.7363, 1.7400, 1.7355, 1.7394, 1.7342, 1.7246, 1.7392, 1.7304, 1.7551, 1.7513, 1.7559, 1.7488, 1.8449, 1.8454, 1.8550, 1.8535, 1.8240, 1.7813, 1.7854, 1.7945, 1.8047, 1.7876, 1.7695, 1.7676, 1.7782, 1.7667, 1.7925, 1.7848, 1.7579, 1.7407, 1.7483, 1.7368, 1.7961, 1.7998, 1.7920, 1.7925, 1.7780, 1.7747, 1.7727, 1.7749, 1.7526, 1.7447, 1.7657, 1.7495, 1.7775, 1.7720, 1.7813, 1.7813, 1.8162, 1.8013, 1.8023, 1.8033, 1.7527, 1.7331, 1.7563, 1.7482, 1.7610, 1.7507, 1.7681, 1.7613, 1.7665, 1.7545, 1.7828, 1.7726, 1.7896, 1.7999, 1.7864, 1.7760, 1.7613, 1.7625, 1.7560, 1.7577, 1.7783, 1.7671, 1.7810, 1.7799, 1.7201, 1.7068, 1.7265, 1.7091, 1.7793, 1.7578, 1.7502, 1.7455, 1.7587, 1.7500, 1.7525, 1.7362, 1.7616, 1.7572, 1.7444, 1.7430, 1.7509, 1.7610, 1.7634, 1.7612, 1.7254, 1.7135, 1.7321, 1.7226, 1.7664, 1.7624, 1.7718, 1.7664, 1.7457, 1.7441, 1.7569, 1.7530
];

/// VAE `scale_factor` for the FLUX.2 VAE.
pub const VAE_SCALE_FACTOR: f32 = 1.0;

/// `[n][c]` tokens (n = h*w) to NCHW `[c][h][w]`.
pub fn tokens_to_nchw(tokens: &[f32], c: usize, h: usize, w: usize) -> Result<Vec<f32>, String> {
    let n = h * w;
    if tokens.len() != n * c {
        return Err(format!("tokens: {} values, want {n} x {c}", tokens.len()));
    }
    let mut out = vec![0f32; n * c];
    for t in 0..n {
        for ch in 0..c {
            out[ch * n + t] = tokens[t * c + ch];
        }
    }
    Ok(out)
}

/// NCHW `[c][h][w]` to `[n][c]` tokens (inverse of [`tokens_to_nchw`]).
pub fn nchw_to_tokens(nchw: &[f32], c: usize, h: usize, w: usize) -> Result<Vec<f32>, String> {
    let n = h * w;
    if nchw.len() != n * c {
        return Err(format!("latent: {} values, want {c} x {h} x {w}", nchw.len()));
    }
    let mut out = vec![0f32; n * c];
    for t in 0..n {
        for ch in 0..c {
            out[t * c + ch] = nchw[ch * n + t];
        }
    }
    Ok(out)
}

/// Diffusion-space NCHW latent to the VAE decoder's latent: per channel `z = x * std[c] + mean[c]`.
pub fn diffusion_to_vae_latents(nchw: &[f32], c: usize, h: usize, w: usize) -> Result<Vec<f32>, String> {
    if c != LATENT_CHANNELS {
        return Err(format!("FLUX.2 VAE stats exist for {LATENT_CHANNELS} channels, got {c}"));
    }
    let n = h * w;
    if nchw.len() != n * c {
        return Err(format!("latent: {} values, want {c} x {h} x {w}", nchw.len()));
    }
    let mut out = vec![0f32; n * c];
    for ch in 0..c {
        let (m, s) = (VAE_LATENT_MEAN[ch], VAE_LATENT_STD[ch]);
        for p in 0..n {
            out[ch * n + p] = (nchw[ch * n + p] * s) / VAE_SCALE_FACTOR + m;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_have_128_entries_and_spot_values() {
        assert_eq!(VAE_LATENT_MEAN.len(), 128);
        assert_eq!(VAE_LATENT_STD.len(), 128);
        assert_eq!(VAE_LATENT_MEAN[0], -0.0676);
        assert_eq!(VAE_LATENT_MEAN[4], 0.0223);
        assert_eq!(VAE_LATENT_MEAN[127], 0.0104);
        assert_eq!(VAE_LATENT_STD[0], 1.8029);
        assert_eq!(VAE_LATENT_STD[12], 1.8626);
        assert_eq!(VAE_LATENT_STD[127], 1.7530);
        assert_eq!(VAE_SCALE_FACTOR, 1.0);
    }

    #[test]
    fn token_layout_round_trip_and_element_order() {
        let (c, h, w) = (3usize, 2usize, 4usize);
        let tokens: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let nchw = tokens_to_nchw(&tokens, c, h, w).unwrap();
        // token (row 1, col 2) = t 6, channel 2 -> nchw[2][1][2]
        assert_eq!(nchw[2 * h * w + 1 * w + 2], tokens[6 * c + 2]);
        assert_eq!(nchw_to_tokens(&nchw, c, h, w).unwrap(), tokens);
        assert!(tokens_to_nchw(&tokens[1..], c, h, w).is_err());
    }

    #[test]
    fn diffusion_to_vae_applies_std_then_mean() {
        let (h, w) = (1usize, 2usize);
        let x: Vec<f32> = (0..128 * 2).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
        let z = diffusion_to_vae_latents(&x, 128, h, w).unwrap();
        // channel 0, pixel 1
        let want = x[1] * 1.8029 + (-0.0676);
        assert!((z[1] - want).abs() < 1e-6, "{} vs {want}", z[1]);
        // channel 127, pixel 0
        let i = 127 * 2;
        let want = x[i] * 1.7530 + 0.0104;
        assert!((z[i] - want).abs() < 1e-6);
        assert!(diffusion_to_vae_latents(&x[..2], 1, 1, 2).is_err());
    }
}

/// FLUX.2 VAE spatial factor: `get_scale_factor()` is 16 for `sd_version_uses_flux2_vae`
/// (`model/vae/vae.hpp` 175-186). The latent grid of a `width` x `height` image is `(h/16, w/16)`.
pub const VAE_FACTOR: usize = 16;

/// Latent grid `(h, w)` for an image size; errors unless both sides are multiples of 16.
pub fn grid_for(width: usize, height: usize) -> Result<(usize, usize), String> {
    if width == 0 || height == 0 || width % VAE_FACTOR != 0 || height % VAE_FACTOR != 0 {
        return Err(format!("width {width} and height {height} must be positive multiples of {VAE_FACTOR}"));
    }
    Ok((height / VAE_FACTOR, width / VAE_FACTOR))
}
