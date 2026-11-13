//! image-flow: the FLUX.2 klein sampling loop around a DiT backend.
//!
//! - [`full_ref`]: the whole DiT in plain f32 (`RefModel`, name `ref-f32`), the reference for the model.
//! - [`handoff`]: reader for the sd.cpp `SDPROBE` hand-over directory (per-step inputs and outputs).
//! - [`sampler`]: Euler loop in token space with sd.cpp's update rule (`run_flow`, `run_flow_inputs`).
//! - [`latent`]: token <-> NCHW latent layout, the FLUX.2 VAE statistics and the latent grid.
//! - [`cond_cache`]: on-disk cache of text conditioning, RoPE and sigmas, keyed by SHA-256 ([`sha256`]).
//! - [`noise`]: own deterministic Gaussian initial noise (not sd.cpp's generator).
//! - [`schedule`]: port of sd.cpp's `Flux2Scheduler` (`flux2_sigmas`).
//!
//! The CLI is `src/bin/oxi.rs` (`generate`, `verify`, `compare`, `cond-put`).

pub mod actstats;
pub mod chancalib;
pub mod cond_cache;
pub mod drift;
pub mod full_ref;
pub mod handoff;
pub mod latent;
pub mod noise;
pub mod pe;
pub mod schedule;
pub mod sampler;
pub mod sha256;
pub mod txtstats;

pub use cond_cache::Conditioning;
pub use full_ref::{ModelDims, RefModel, Taps};
pub use handoff::Handoff;
pub use sampler::FlowInputs;
