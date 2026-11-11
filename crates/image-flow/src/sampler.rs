//! Euler sampler in token space for the FLUX.2 flow model, with the update rule of sd.cpp.
//!
//! Rule (sd.cpp, verified against the source):
//! - denoiser for `FLUX_FLOW_PRED` is `FluxFlowDenoiser` (`runtime/denoiser.hpp` ~1442):
//!   `sigma_to_t(s) = s`, so the model is called with `timestep = sigma` (the DiT multiplies by 1000
//!   inside the timestep embedding); `get_scalings(s) = {c_skip 1, c_out -s, c_in 1}`.
//! - `denoise` in `pipeline/diffusion_engine.cpp` (~2439-2880): input `x * c_in = x`; with CFG 1 and
//!   no uncond pass the guider returns the conditional prediction unchanged (`runtime/guidance.cpp`
//!   ~149-170), so `pred = model_out`; `denoised = pred * c_out + x * c_skip = x - sigma * out`.
//! - `sample_euler` (`runtime/denoiser.hpp` ~1840-1858): `d = (x - denoised) / sigma = out`, then
//!   `x += d * (sigma_next - sigma)`.
//!
//! So: `x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * out_i`, `out_i = model(x_i, sigma_i)`, no sign flip,
//! no timestep scaling, `sigma_steps = 0`. The schedule (Flux2Scheduler, ~853-907) is in the dumps.

use crate::cond_cache::Conditioning;
use crate::handoff::Handoff;
use image_dit::model::{DitBackend, ModelInput};

/// Everything the Euler loop needs. Built from a hand-over, or from a cached conditioning plus an
/// initial latent (own noise or a hand-over's img0).
pub struct FlowInputs {
    pub n_img: usize,
    pub n_txt: usize,
    /// `[n_img][in_ch]` initial latent tokens
    pub img0: Vec<f32>,
    /// `[n_txt][ctx]`
    pub txt: Vec<f32>,
    /// `[n_txt + n_img][pe_half][4]`
    pub pe: Vec<f32>,
    /// sigma of each step; the number of steps is `sigmas.len()`
    pub sigmas: Vec<f32>,
    /// sigma after the last step (0 for the FLUX.2 schedule)
    pub sigma_end: f32,
}

impl FlowInputs {
    pub fn from_handoff(h: &Handoff) -> FlowInputs {
        FlowInputs {
            n_img: h.n_img,
            n_txt: h.n_txt,
            img0: h.img0.clone(),
            txt: h.txt.clone(),
            pe: h.pe.clone(),
            sigmas: h.sigmas.clone(),
            sigma_end: h.sigma_end,
        }
    }

    /// From a cached conditioning and an initial latent of `n_img * in_ch` values.
    pub fn from_conditioning(c: &Conditioning, img0: Vec<f32>, sigmas: Vec<f32>) -> Result<FlowInputs, String> {
        if c.n_img == 0 || img0.len() % c.n_img != 0 {
            return Err(format!("initial latent has {} values, not a multiple of {} tokens", img0.len(), c.n_img));
        }
        if sigmas.is_empty() {
            return Err("no sigmas".into());
        }
        Ok(FlowInputs {
            n_img: c.n_img,
            n_txt: c.n_txt,
            img0,
            txt: c.txt.clone(),
            pe: c.pe.clone(),
            sigmas,
            sigma_end: 0.0,
        })
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len()
    }

    /// Sigma of step `i` and the sigma after it.
    pub fn sigma_pair(&self, i: usize) -> (f32, f32) {
        let next = if i + 1 < self.steps() { self.sigmas[i + 1] } else { self.sigma_end };
        (self.sigmas[i], next)
    }
}

/// Runs the Euler loop from a hand-over (see [`run_flow_inputs`]).
pub fn run_flow(model: &mut dyn DitBackend, h: &Handoff, on_step: &mut dyn FnMut(usize, &[f32], &[f32])) -> Vec<f32> {
    run_flow_inputs(model, &FlowInputs::from_handoff(h), on_step)
}

/// Runs the Euler loop from `inp.img0`. `on_step(i, x_before, model_out)` is called for each step.
/// Returns the latent tokens `[n_img][in_ch]` after the last update.
pub fn run_flow_inputs(model: &mut dyn DitBackend, inp: &FlowInputs, on_step: &mut dyn FnMut(usize, &[f32], &[f32])) -> Vec<f32> {
    let mut x = inp.img0.clone();
    for i in 0..inp.steps() {
        let (sigma, sigma_next) = inp.sigma_pair(i);
        let input = ModelInput {
            n_img: inp.n_img,
            n_txt: inp.n_txt,
            img: x.clone(),
            txt: inp.txt.clone(),
            timestep: sigma,
            pe: inp.pe.clone(),
        };
        let v = model.forward(&input);
        assert_eq!(v.len(), x.len(), "model output must have the latent's shape (out_ch == in_ch)");
        on_step(i, &x, &v);
        let dt = sigma_next - sigma;
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * vi;
        }
    }
    x
}

/// Least-squares `c` in `x_after - x_before ~ c * out` (over all elements, f64).
/// Equals `sigma_next - sigma` when the rule holds exactly. NaN if `out` is all zero.
pub fn fit_step(x_before: &[f32], out: &[f32], x_after: &[f32]) -> f64 {
    assert_eq!(x_before.len(), out.len(), "fit_step lengths");
    assert_eq!(x_after.len(), out.len(), "fit_step lengths");
    let (mut num, mut den) = (0f64, 0f64);
    for ((&b, &o), &a) in x_before.iter().zip(out).zip(x_after) {
        let d = a as f64 - b as f64;
        num += d * o as f64;
        den += o as f64 * o as f64;
    }
    if den == 0.0 {
        return f64::NAN;
    }
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoff::Handoff;
    use std::path::PathBuf;

    /// v = -x: the closed form of the Euler loop is x_end = x0 * prod (1 - (s_{i+1} - s_i)).
    struct NegBackend {
        calls: Vec<(f32, Vec<f32>)>,
    }

    impl DitBackend for NegBackend {
        fn name(&self) -> &'static str {
            "neg"
        }
        fn forward(&mut self, input: &ModelInput) -> Vec<f32> {
            self.calls.push((input.timestep, input.img.clone()));
            input.img.iter().map(|v| -v).collect()
        }
    }

    fn handoff(sigmas: Vec<f32>, img0: Vec<f32>, n_img: usize) -> Handoff {
        let steps = sigmas.len();
        Handoff {
            dir: PathBuf::from("/nonexistent"),
            n_img,
            n_txt: 2,
            in_ch: img0.len() / n_img,
            ctx_dim: 4,
            out_ch: img0.len() / n_img,
            steps,
            sigmas,
            sigma_end: 0.0,
            img0,
            txt: vec![0.0; 2 * 4],
            pe: vec![0.0; (2 + n_img) * 4],
            pe_half: 1,
            img_steps: vec![],
            out_steps: vec![None; steps],
            single_in: [vec![None; steps], vec![None; steps]],
            single_hidden: None,
            vae_z: None,
            vae_image: None,
        }
    }

    #[test]
    fn euler_loop_matches_closed_form_for_v_equals_minus_x() {
        let sig = vec![1.0f32, 0.8, 0.5, 0.2];
        let x0: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        let h = handoff(sig.clone(), x0.clone(), 2);
        let mut be = NegBackend { calls: vec![] };
        let mut seen = Vec::new();
        let out = run_flow(&mut be, &h, &mut |i, x, v| seen.push((i, x.to_vec(), v.to_vec())));

        // model is called at the step's own sigma
        let ts: Vec<f32> = be.calls.iter().map(|c| c.0).collect();
        assert_eq!(ts, sig);
        // closed form: x_end = x0 * prod_i (1 - (s_{i+1} - s_i)), with s_steps = 0
        let mut factor = 1f64;
        for i in 0..sig.len() {
            let next = if i + 1 < sig.len() { sig[i + 1] } else { 0.0 };
            factor *= 1.0 - (next as f64 - sig[i] as f64);
        }
        for (o, x) in out.iter().zip(&x0) {
            let want = *x as f64 * factor;
            assert!((*o as f64 - want).abs() < 1e-6, "{o} vs {want}");
        }
        // callback saw the pre-update x and the model output
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0].1, x0);
        assert_eq!(seen[0].2, x0.iter().map(|v| -v).collect::<Vec<f32>>());
    }

    #[test]
    fn fit_step_recovers_the_step_size() {
        let before = vec![1.0f32, 2.0, -1.0];
        let out = vec![0.5f32, -1.0, 2.0];
        let dt = -0.25f32;
        let after: Vec<f32> = before.iter().zip(&out).map(|(b, o)| b + dt * o).collect();
        let c = fit_step(&before, &out, &after);
        assert!((c - dt as f64).abs() < 1e-7, "{c}");
        assert!(fit_step(&before, &[0.0, 0.0, 0.0], &after).is_nan());
    }

    #[test]
    fn fit_step_on_the_euler_loop_gives_sigma_differences() {
        let sig = vec![1.0f32, 0.6, 0.3];
        let h = handoff(sig.clone(), vec![0.3, -0.7, 1.1, 0.2], 2);
        let mut be = NegBackend { calls: vec![] };
        let mut steps = Vec::new();
        let last = run_flow(&mut be, &h, &mut |_, x, v| steps.push((x.to_vec(), v.to_vec())));
        for i in 0..sig.len() {
            let after: &[f32] = if i + 1 < sig.len() { &steps[i + 1].0 } else { &last };
            let c = fit_step(&steps[i].0, &steps[i].1, after);
            let want = h.sigma_pair(i).1 as f64 - h.sigma_pair(i).0 as f64;
            assert!((c - want).abs() < 1e-5, "step {i}: fit {c} want {want}");
        }
    }
}
