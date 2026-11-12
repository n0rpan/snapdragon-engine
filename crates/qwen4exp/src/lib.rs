//! qwen4exp: Qwen3.8-Flash-Next in the oxe engine. M0 = timing prototype only (see docs/lab-notes/M0-FLASH.md):
//! real weights and kernels, static per-thread plan, expert streaming from an R4 sidecar
//! through an io_uring cache, replayed routing; activations are dummy.
pub mod cache;
pub mod cli;
pub mod dense;
pub mod exec;
pub mod ffi;
pub mod graph;
pub mod io;
pub mod lookahead;
pub mod model;
pub mod mtp;
pub mod mtp_engine;
pub mod mtp_graph;
pub mod mtp_head;
pub mod mtp_vocab;
pub mod ngram;
pub mod pf_plan;
pub mod sidecar;
pub mod stream;
pub mod trace;
pub mod uring;

pub const KERNEL_CFLAGS: &str = env!("OXF_KERNEL_CFLAGS");
pub const ENGINE_COMMIT: &str = env!("OXE_GIT_COMMIT");
