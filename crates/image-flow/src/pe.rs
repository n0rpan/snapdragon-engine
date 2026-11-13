//! Native FLUX.2 RoPE table (`dump.pe`), so a new prompt needs no sd-cli hand-over.
//!
//! Port of stable-diffusion.cpp 5a1ba5d `model/common/rope.hpp` (`gen_flux_pe` -> `gen_flux_ids` ->
//! `embed_nd`) with the FLUX.2 settings of `model/diffusion/flux.hpp`: theta 2000, axes_dim
//! {32, 32, 32, 32}, patch 1, `txt_arange_dims = {3}`, no reference latents, no circular wrap.
//! - text token i (0..n_txt): ids (0, 0, 0, i);
//! - image token (r, c) of the latent grid (row-major): ids (0, r, c, 0);
//! - per axis a (dim 32, 16 frequencies): `omega_j = 1 / powf(theta, j * step)` with
//!   `step = ((32 - 2) / 32) / 15` in f32 (sd.cpp's `linspace`), angle = id * omega_j;
//! - value block per (token, j): `[cos, -sin, sin, cos]`, axes concatenated (64 = pe_half blocks).
//!
//! Layout `[n_txt + n_img][pe_half][4]`, the `Conditioning::pe` layout. The table depends only on
//! (n_txt, grid): any prompt of the same token count and size shares it.

pub const FLUX2_THETA: f32 = 2000.0;
pub const FLUX2_AXES: [usize; 4] = [32, 32, 32, 32];

/// `rope_frequencies(dim, theta)` of sd.cpp, in f32.
fn frequencies(dim: usize, theta: f32) -> Vec<f32> {
    let half = dim / 2;
    let (start, end) = (0f32, (dim as f32 - 2.0) / dim as f32);
    let step = (end - start) / (half as f32 - 1.0);
    (0..half)
        .map(|i| {
            let scale = if half == 1 { start } else { start + i as f32 * step };
            1.0f32 / theta.powf(scale)
        })
        .collect()
}

/// pe_half of FLUX.2 (64).
pub fn pe_half() -> usize {
    FLUX2_AXES.iter().map(|d| d / 2).sum()
}

/// The FLUX.2 klein RoPE table for `n_txt` text tokens and a `(h, w)` latent grid.
pub fn flux2_pe(n_txt: usize, grid: (usize, usize)) -> Vec<f32> {
    let (h, w) = grid;
    let n = n_txt + h * w;
    let ph = pe_half();
    let omegas: Vec<Vec<f32>> = FLUX2_AXES.iter().map(|&d| frequencies(d, FLUX2_THETA)).collect();
    let mut out = vec![0f32; n * ph * 4];
    for t in 0..n {
        let ids: [f32; 4] = if t < n_txt {
            [0.0, 0.0, 0.0, t as f32]
        } else {
            let p = t - n_txt;
            [0.0, (p / w) as f32, (p % w) as f32, 0.0]
        };
        let mut j0 = 0;
        for (a, om) in omegas.iter().enumerate() {
            for (j, &o) in om.iter().enumerate() {
                let angle = ids[a] * o;
                let (s, c) = (angle.sin(), angle.cos());
                let b = (t * ph + j0 + j) * 4;
                out[b..b + 4].copy_from_slice(&[c, -s, s, c]);
            }
            j0 += om.len();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoff::Handoff;

    #[test]
    fn shape_identity_rows_and_text_axis() {
        let pe = flux2_pe(3, (2, 2));
        assert_eq!(pe.len(), 7 * 64 * 4);
        // text token 0 and image token (0, 0): every angle is 0 -> [1, -0, 0, 1]
        for t in [0usize, 3] {
            for j in 0..64 {
                assert_eq!(&pe[(t * 64 + j) * 4..(t * 64 + j) * 4 + 4], &[1.0, -0.0, 0.0, 1.0]);
            }
        }
        // text token 2: only axis 3 (blocks 48..64) rotates, with omega_0 = 1 -> angle 2
        let b = (2 * 64 + 48) * 4;
        assert_eq!(&pe[b..b + 4], &[2f32.cos(), -(2f32.sin()), 2f32.sin(), 2f32.cos()]);
        assert_eq!(&pe[(2 * 64 + 16) * 4..(2 * 64 + 16) * 4 + 4], &[1.0, -0.0, 0.0, 1.0]);
        // image token (1, 0) = t 5: axis 1 (blocks 16..32) rotates by row 1
        let b = (5 * 64 + 16) * 4;
        assert_eq!(pe[b], 1f32.cos());
        assert_eq!(frequencies(32, 2000.0)[0], 1.0);
    }

    /// Real hand-overs (file reads only, no model): `OXI_PE_HANDOFFS=dir1:dir2 cargo test -p image-flow
    /// --release pe -- --ignored`. The native table must equal sd.cpp's `dump.pe` to f32 rounding of
    /// the trig functions (max abs diff <= 1e-6).
    #[test]
    #[ignore]
    fn matches_sdcpp_dump_pe() {
        let dirs = std::env::var("OXI_PE_HANDOFFS").expect("OXI_PE_HANDOFFS");
        for d in dirs.split(':') {
            let h = Handoff::load(std::path::Path::new(d)).unwrap();
            let side = (h.n_img as f64).sqrt() as usize;
            assert_eq!(side * side, h.n_img, "{d}: square grids only in this check");
            let pe = flux2_pe(h.n_txt, (side, side));
            assert_eq!((h.pe_half, pe.len()), (64, h.pe.len()), "{d}");
            let max = pe.iter().zip(&h.pe).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let exact = pe.iter().zip(&h.pe).filter(|(a, b)| a == b).count();
            eprintln!("{d}: n_txt {} n_img {} max_abs {max:e} exact {exact}/{}", h.n_txt, h.n_img, pe.len());
            assert!(max <= 1e-6, "{d}: max abs {max}");
        }
    }
}
