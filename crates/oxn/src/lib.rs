//! oxn: host-side GEMM contract for the NPU-1 oxn HMX kernels.
//!
//! `Backend` is the contract: `Y[T,O] = X[T,K] * W[O,K]^T` in the 32x32 fp16 tile layouts of
//! `npu/oxn/common/oxn_tiles.h` (W as W8 records with `GF_W8`, see `npu/oxn/common/oxn_w8.h`).
//! `emu::Emu` is a pure-Rust implementation for tests. `device::Device` (feature `device`) runs
//! the same contract on the DSP through the C host library in `npu/oxn/host`.

pub mod linear;
pub mod mapping_probe;
pub mod shared_memory;
mod weight_imports;
pub use weight_imports::ImportedWeight;
pub mod tiles;
pub mod wfmt;
pub mod emu;
#[cfg(feature = "device")]
pub mod device;

pub const GF_YCOL: u32 = 0x01;
pub const GF_NO_WFETCH: u32 = 0x02;
pub const GF_NO_YWRITE: u32 = 0x04;
pub const GF_NO_HMX: u32 = 0x08;
pub const GF_W8: u32 = 0x10;

/// A buffer shared with the accelerator. `ptr` stays valid until the backend is dropped.
/// `session` and `slot` name it in the backend: slots count from 0 in each session.
#[derive(Clone, Copy, Debug)]
pub struct Buf {
    pub session: u32,
    pub slot: u32,
    pub ptr: *mut u8,
    pub len: usize,
}
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

#[derive(Clone, Copy, Debug, Default)]
pub struct GemmArgs {
    pub x_slot: u32,
    pub x_off: u32,
    pub w_slot: u32,
    pub w_off: u32,
    pub y_slot: u32,
    pub y_off: u32,
    pub t: u32,
    pub k: u32,
    pub o: u32,
    pub flags: u32,
    pub r: u32,
    pub c: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GemmStats {
    pub wall_s: f64,
    pub dsp_us: u32,
    pub pcycles: u64,
    pub wait_pcycles: u64,
    pub hmx_pcycles: u64,
    /// W8 expansion (dequantisation) cycles on the DSP
    pub deq_pcycles: u64,
    pub steps: u32,
    pub r: u32,
    pub c: u32,
}

pub trait Backend: Send {
    fn name(&self) -> &'static str;
    /// Number of sessions (independent address spaces on the accelerator), 1 or more.
    fn sessions(&self) -> usize;
    /// VTCM bytes of session 0.
    fn vtcm_bytes(&self) -> u32;
    /// Zero-filled buffer in `session`. bytes: multiple of 4096, <= 128 MiB.
    fn alloc(&mut self, session: usize, bytes: usize) -> Result<Buf, String>;
    /// Y[T,O] = X[T,K] * W[O,K]^T in the tile layouts of npu/oxn/common/oxn_tiles.h (W: fp16 tiles, or W8 records with GF_W8).
    /// The three slots in `a` are slots of `session`.
    fn gemm(&mut self, session: usize, a: &GemmArgs) -> Result<GemmStats, String>;
}
