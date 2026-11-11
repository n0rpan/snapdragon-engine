//! image-npu: model-free planning, the device boundary, and the hybrid DiT forward for
//! NPU-resident linears. No device code.
//!
//! - `plan`: window planner, session slots, estimates (see PLAN-NPU.md).
//! - `backend`: `LinearDevice` trait, `MockDevice`, `WindowedRunner`.
//! - `hybrid`: `HybridModel`, the whole DiT with block linears on a device.

pub mod backend;
pub mod hybrid;
pub mod plan;
