//! qwen35moe model side of the engine: descriptor, file-backed repacked weights, static
//! plan, and the decode engine with llama.cpp-exact math.

pub mod check;
pub mod desc;
pub mod engine;
#[cfg(feature="f32-reference")]
pub mod f32_reference;
/// Long-context fused-attention numeric reference (bench / tests only: `oxe fused-ref`).
#[cfg(any(test, feature = "f32-reference"))]
pub mod fused_ref;
pub mod load;
pub mod mtp;
pub mod npu_rows;
#[cfg(feature = "npu-prefill")]
pub mod npu_prefill;
pub mod plan;
#[cfg(any(test, feature = "testkit"))]
pub mod testkit;
pub mod weights;

#[cfg(test)]
mod tests;

#[cfg(feature="npu-prefill")]
pub(crate) mod cpu_recurrent;
