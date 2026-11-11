//! FLUX.2 sigma schedule: a port of sd.cpp's `Flux2Scheduler` (`runtime/denoiser.hpp`).
//!
//! Source references (image-sd-probe/src/src):
//! - `flux_time_shift(mu, sigma, t) = exp(mu) / (exp(mu) + (1/t - 1)^sigma)`, denoiser.hpp 725-727.
//! - `Flux2Scheduler::compute_empirical_mu`, denoiser.hpp 859-875: constants a1 = 8.73809524e-05,
//!   b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666; for `image_seq_len > 4300`,
//!   `mu = a2 * L + b2`; otherwise `m200 = a2*L + b2`, `m10 = a1*L + b1`, `a = (m200 - m10)/190`,
//!   `b = m200 - 200*a`, `mu = a*num_steps + b`.
//! - `Flux2Scheduler::get_sigmas`, denoiser.hpp 877-902: `n == 0` gives `[1.0]`; for `i in 0..=n`,
//!   `t = 1 - i/n`; `t <= 0` gives 0, `t >= 1` gives 1, otherwise `flux_time_shift(mu, 1, t)`; the
//!   last entry is forced to 0.
//! - `image_seq_len` is `get_image_seq_len(h, w) = (h / vae) * (w / vae)` with the FLUX.2 VAE factor
//!   16 (`pipeline/diffusion_engine.cpp` 2932-2935, `model/vae/vae.hpp` 175-186), passed from
//!   `pipeline/request.cpp` 451-460. For a hand-over this equals the number of image tokens.
//! - The denoiser for FLUX.2 is `FluxFlowDenoiser` (`FLUX_FLOW_PRED`); the scheduler does not depend on
//!   it (`denoiser.hpp` 1185-1187 chooses `Flux2Scheduler` for `FLUX2_SCHEDULER`, the default for klein).
//!
//! All arithmetic is f32 as in the source. NOT YET CHECKED against real dumps: `oxi verify` reports
//! the difference to the dumped sigmas; use `--sigmas stored` until that check has passed.

/// `Flux2Scheduler::compute_empirical_mu`.
pub fn flux2_empirical_mu(image_seq_len: usize, num_steps: u32) -> f32 {
    let a1 = 8.73809524e-05f32;
    let b1 = 1.89833333f32;
    let a2 = 0.00016927f32;
    let b2 = 0.45666666f32;
    let l = image_seq_len as f32;
    if image_seq_len > 4300 {
        return a2 * l + b2;
    }
    let m200 = a2 * l + b2;
    let m10 = a1 * l + b1;
    let a = (m200 - m10) / 190.0f32;
    let b = m200 - 200.0f32 * a;
    a * num_steps as f32 + b
}

/// `flux_time_shift(mu, 1, t)`.
pub fn flux_time_shift(mu: f32, t: f32) -> f32 {
    mu.exp() / (mu.exp() + (1.0f32 / t - 1.0f32))
}

/// Sigmas of `steps` denoising steps: `steps + 1` values, the last one 0 (`Flux2Scheduler::get_sigmas`).
pub fn flux2_sigmas(steps: u32, image_seq_len: usize) -> Vec<f32> {
    if steps == 0 {
        return vec![1.0];
    }
    let n = steps;
    let mu = flux2_empirical_mu(image_seq_len, n);
    let mut sigmas = Vec::with_capacity(n as usize + 1);
    for i in 0..=n {
        let t = 1.0f32 - i as f32 / n as f32;
        if t <= 0.0 {
            sigmas.push(0.0);
        } else if t >= 1.0 {
            sigmas.push(1.0);
        } else {
            sigmas.push(flux_time_shift(mu, t));
        }
    }
    sigmas[n as usize] = 0.0;
    sigmas
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_and_monotone_decrease() {
        for &(steps, l) in &[(1u32, 256usize), (4, 1024), (4, 4096), (8, 5000), (20, 300)] {
            let s = flux2_sigmas(steps, l);
            assert_eq!(s.len(), steps as usize + 1);
            assert_eq!(s[0], 1.0, "first sigma is t = 1 -> 1");
            assert_eq!(*s.last().unwrap(), 0.0, "last sigma is forced to 0");
            for w in s.windows(2) {
                assert!(w[1] < w[0], "not strictly decreasing: {:?}", w);
            }
        }
    }

    #[test]
    fn zero_steps_is_the_source_special_case() {
        assert_eq!(flux2_sigmas(0, 1024), vec![1.0]);
    }

    #[test]
    fn midpoint_matches_the_closed_form() {
        // t = 0.5 gives (1/t - 1) = 1, so sigma = e^mu / (e^mu + 1)
        let mu = flux2_empirical_mu(1024, 4);
        let want = (mu.exp() / (mu.exp() + 1.0)) as f64;
        // 4 steps: sigma index 2 is t = 0.5
        let s = flux2_sigmas(4, 1024);
        assert!((s[2] as f64 - want).abs() < 1e-6, "{} vs {want}", s[2]);
    }

    #[test]
    fn empirical_mu_branches() {
        // above 4300 the step count does not enter
        assert_eq!(flux2_empirical_mu(5000, 1), flux2_empirical_mu(5000, 50));
        // below: linear in steps, and the 200-step and 10-step anchors are reproduced
        let l = 1024usize;
        let m200 = 0.00016927f32 * l as f32 + 0.45666666;
        assert!((flux2_empirical_mu(l, 200) - m200).abs() < 1e-4);
        let m10 = 8.73809524e-05f32 * l as f32 + 1.89833333;
        assert!((flux2_empirical_mu(l, 10) - m10).abs() < 1e-4);
    }

    #[test]
    fn klein_512_px_schedule_values() {
        // 512 x 512 px: image_seq_len = (512/16)^2 = 1024; print for the hardware-batch comparison
        let s = flux2_sigmas(4, 1024);
        println!("klein 4-step sigmas at 1024 tokens: {s:?}");
        assert!(s[1] > s[2] && s[2] > s[3] && s[3] > 0.0);
    }
}
