//! Backend boundary of the image engine.
//!
//! The engine's graph is a fixed sequence of blocks; a backend owns a WINDOW of consecutive blocks
//! (their weights prepared in the backend's own format) and runs them on f32 token matrices. The
//! CPU backend keeps every block resident. An NPU backend maps windows that fit its session budget
//! (about 1 GiB measured on this machine: about 7 klein blocks of 130 MB int8 weights each) and
//! slides or holds several sessions; activations cross the boundary as f32 (or f16) so each
//! backend chooses its own activation format (CPU: int8 with chunk scales; HTP: per-row int8
//! weights with 16-bit activations is the usual choice).
//!
//! M0 only implements the CPU side for one single-stream block; the trait is the shape M1 builds on.

use crate::types::{BlockInput, BlockOutput, Dims};

/// What a backend must be told to prepare a block: where its weights are in the model file.
/// Byte ranges instead of slices so a backend can map, copy or upload them as it likes.
#[derive(Clone, Debug)]
pub struct BlockWeightsRef {
    pub dims: Dims,
    /// (file offset, byte length) of the Q8_0 `linear1` and `linear2` tensor data.
    pub linear1: (u64, u64),
    pub linear2: (u64, u64),
    pub q_scale: Vec<f32>,
    pub k_scale: Vec<f32>,
}

/// Per-stage wall time of one block forward, in seconds (thread 0's clock around each parallel stage).
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    pub norm_quant: f64,
    pub linear1: f64,
    pub attention: f64,
    pub requant: f64,
    pub linear2: f64,
    pub total: f64,
}

/// A prepared single-stream block on some device.
pub trait SingleBlockBackend {
    fn name(&self) -> &'static str;
    /// Run the block. `want_intermediates` fills `attn`, `mlp`, `z` of the output (slower).
    fn forward(&mut self, input: &BlockInput, want_intermediates: bool) -> (BlockOutput, StageTimes);
}
