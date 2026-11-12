//! Shared accelerator linear packing and execution, re-exported for image callers.
//!
//! The implementation lives in `oxn::linear` so model prefill can use exactly
//! the same allocation, affinity, tiling, auditing and teardown contract.

pub use oxn::linear::*;
