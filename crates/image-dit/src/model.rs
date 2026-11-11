//! Whole-DiT boundary (M1): what one denoising forward takes and returns, for any backend.
//!
//! Mirrors sd.cpp `Flux::forward_orig` for FLUX.2 klein (text-to-image, CFG 1, no guidance embed):
//! `img` are the latent tokens before `img_in`, `txt` the text-encoder context before `txt_in`,
//! the result is the final layer's output per image token. Token order inside the blocks is text
//! first, then image; `pe` is given for that concatenated order.

/// Inputs of one forward.
pub struct ModelInput {
    pub n_img: usize,
    pub n_txt: usize,
    /// `[n_img][in_channels]` (klein: 128)
    pub img: Vec<f32>,
    /// `[n_txt][context_dim]` (klein: 7680)
    pub txt: Vec<f32>,
    /// sigma in [0, 1] as sd.cpp passes it to the timestep embedding (which multiplies by 1000)
    pub timestep: f32,
    /// `[n_txt + n_img][head_dim/2][2][2]`, per (token, pair) `[[cos, -sin], [sin, cos]]`
    pub pe: Vec<f32>,
}

/// A device/precision implementation of the whole DiT.
pub trait DitBackend {
    fn name(&self) -> &'static str;
    /// Returns `[n_img][out_channels]` (klein: 128), the model prediction for this step.
    fn forward(&mut self, input: &ModelInput) -> Vec<f32>;
}
