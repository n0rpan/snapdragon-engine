//! Image engine, milestone M0: one FLUX.2 klein single-stream block at real shapes.
//!
//! - [`types`]: shapes and the block's inputs/weights as plain f32 (shared by both paths).
//! - [`weights`]: GGUF tensor mapping for klein (dims from tensor shapes, Q8_0 access, dequantisation).
//! - [`dump`]: reader for the `SDPROBE` tensor dumps written by the patched sd.cpp (run ticket 013).
//! - [`reference`]: straightforward f32 implementation of the block, the yardstick for every error number.
//! - [`arith`]: activation rounding for the sd.cpp arithmetic-parity reference (W5-G).
//! - [`attn_float`]: f32 / f16 attention option of the fast path (F-IMG-ATTN; linears stay int8).
//! - [`fast`]: the int8 SMMLA path on a static per-thread plan (`int8-gemm` + `sched`).
//! - [`backend`]: the block-level backend trait the CPU path implements and an NPU path can implement later.
//!
//! Design: `IMAGE-ENGINE-DESIGN.md` at the repository root.

pub mod arith;
pub mod attn_float;
pub mod backend;
pub mod dump;
pub mod fast;
pub mod fast_model;
pub mod model;
pub mod reference;
pub mod types;
pub mod weights;

#[cfg(test)]
mod prod_fixture;
#[cfg(test)]
mod prod_model_fixture;
