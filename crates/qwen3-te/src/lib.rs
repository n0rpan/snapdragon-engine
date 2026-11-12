//! Native Qwen3 dense text encoder for FLUX.2 klein conditioning.
//!
//! Replaces the probe-sd-cli prompt hand-over (sd-cli start, Qwen3-4B Q8_0 load + encode, DiT load
//! to reach model call 0) with an in-process encoder that produces the same `[n_txt][7680]` context
//! (`dump.flux.txt`). Design and gates: `handoff/TEXT-ENCODER-NPU-DESIGN.md`.
//!
//! - [`config`]: model shape from GGUF and the klein spec (template, padding, layers 9/18/27).
//! - [`prompt`]: template + tokenization + pad/mask, refusing prompts outside the plain subset.
//! - [`weights`]: borrowed GGUF linears (row readers), only the 27 blocks klein needs.
//! - [`encoder`]: the graph, with a [`encoder::Linears`] backend: CPU f32 reference, CPU Q8
//!   (sd.cpp arithmetic emulation), or [`npu::NpuLinears`] (oxn emulator or device).
//! - [`metrics`]: hidden-state agreement and the gate.

pub mod config;
pub mod cpu;
pub mod encoder;
pub mod metrics;
pub mod npu;
pub mod prompt;
pub mod weights;

#[cfg(test)]
mod tests;

use config::KleinSpec;
use encoder::{CpuLinears, CpuMode, Encoder, Linears};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
pub enum TeError {
    /// The prompt is valid for sd.cpp but outside the native subset: use the sd-cli route.
    NeedsReference(String),
    Other(String),
}

impl std::fmt::Display for TeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeError::NeedsReference(s) => write!(f, "needs the reference encoder: {s}"),
            TeError::Other(s) => f.write_str(s),
        }
    }
}

impl From<String> for TeError {
    fn from(s: String) -> Self {
        TeError::Other(s)
    }
}

/// Which arithmetic runs the linears.
pub enum TeBackend {
    CpuF32,
    CpuQ8,
    Npu { backend: Box<dyn oxn::Backend>, opts: npu::NpuOpts },
}

pub struct TeOutput {
    /// `[n][ctx_dim]`
    pub ctx: Vec<f32>,
    pub n: usize,
    pub n_real: usize,
    pub ctx_dim: usize,
    pub tokens: Vec<u32>,
    pub summary: Value,
}

/// Open `gguf`, tokenize `prompt` with its vocabulary, and encode it for klein.
pub fn encode_prompt(gguf: &Path, prompt: &str, backend: TeBackend, threads: usize) -> Result<TeOutput, TeError> {
    let t_all = Instant::now();
    let spec = KleinSpec::klein();
    let t = Instant::now();
    let tok = tokenizer::Tokenizer::from_gguf(gguf).map_err(|e| TeError::Other(format!("tokenizer: {}", e.0)))?;
    let enc = match prompt::klein_tokens(&tok, &spec, prompt) {
        Ok(e) => e,
        Err(prompt::PromptError::NeedsReference(s)) => return Err(TeError::NeedsReference(s)),
        Err(e) => return Err(TeError::Other(e.to_string())),
    };
    let tokenize_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let g = gguf::Gguf::open(gguf).map_err(|e| TeError::Other(e.0))?;
    let w = weights::TeWeights::from_gguf(&g, spec.layers_needed())?;
    let open_s = t.elapsed().as_secs_f64();
    let encd = Encoder::new(&w, spec.clone(), threads)?;
    let cpu;
    let npu_lin;
    let lin: &dyn Linears = match backend {
        TeBackend::CpuF32 | TeBackend::CpuQ8 => {
            let mode = if matches!(backend, TeBackend::CpuF32) { CpuMode::F32 } else { CpuMode::Q8 };
            cpu = CpuLinears { w: &w, threads, mode };
            &cpu
        }
        TeBackend::Npu { backend, opts } => {
            npu_lin = npu::NpuLinears::new(&w, backend, opts, threads)?;
            &npu_lin
        }
    };
    let (ctx, tm) = encd.encode(lin, &enc)?;
    let summary = json!({
        "backend": lin.name(),
        "n": enc.tokens.len(),
        "n_real": enc.n_real,
        "ctx_dim": spec.ctx_dim(&w.cfg),
        "out_layers": spec.out_layers,
        "tokenize_s": tokenize_s,
        "open_s": open_s,
        "encode": tm.json(),
        "linears": lin.summary(),
        "wall_s": t_all.elapsed().as_secs_f64(),
    });
    Ok(TeOutput { ctx, n: enc.tokens.len(), n_real: enc.n_real, ctx_dim: spec.ctx_dim(&w.cfg), tokens: enc.tokens, summary })
}

/// Backend from command-line style options (shared by the `qwen3-te` binary and `oxi`):
/// `te-backend` cpu-f32 (default) | cpu-q8 | npu; for npu: `te-npu` emu|device (default emu),
/// `te-npu-weights` fp16|w8 (default fp16), `te-npu-sessions` 1..4 (default 3),
/// `te-npu-budget-mib` (default 6144), `te-npu-kc` (default 0 = whole K),
/// `te-io-threads` (default 8).
pub fn backend_from_opts(get: &dyn Fn(&str) -> Option<String>) -> Result<TeBackend, String> {
    let num = |k: &str, d: usize| -> Result<usize, String> {
        match get(k) {
            None => Ok(d),
            Some(v) => v.parse().map_err(|_| format!("--{k} must be an integer, got '{v}'")),
        }
    };
    match get("te-backend").as_deref().unwrap_or("cpu-f32") {
        "cpu-f32" => Ok(TeBackend::CpuF32),
        "cpu-q8" => Ok(TeBackend::CpuQ8),
        "npu" => {
            let sessions = num("te-npu-sessions", 3)?;
            if !(1..=4).contains(&sessions) {
                return Err(format!("--te-npu-sessions must be 1..4, got {sessions}"));
            }
            let weights = match get("te-npu-weights").as_deref().unwrap_or("fp16") {
                "fp16" => oxn::wfmt::WFmt::Fp16,
                "w8" => oxn::wfmt::WFmt::W8,
                o => return Err(format!("--te-npu-weights must be fp16 or w8, got '{o}'")),
            };
            let kc_max = num("te-npu-kc", 0)?;
            if kc_max != 0 && kc_max < 32 {
                return Err("--te-npu-kc must be 0 or at least 32".into());
            }
            let budget = num("te-npu-budget-mib", 6144)?.checked_mul(1 << 20).ok_or("--te-npu-budget-mib too large")?;
            let backend: Box<dyn oxn::Backend> = match get("te-npu").as_deref().unwrap_or("emu") {
                "emu" => Box::new(oxn::emu::Emu::new(8 << 20, sessions)),
                "device" => device_backend(sessions)?,
                o => return Err(format!("--te-npu must be emu or device, got '{o}'")),
            };
            let opts = npu::NpuOpts { budget_bytes: budget, weights, kc_max, pack_threads: 0, io_threads: num("te-io-threads", 8)? };
            Ok(TeBackend::Npu { backend, opts })
        }
        o => Err(format!("--te-backend must be cpu-f32, cpu-q8 or npu, got '{o}'")),
    }
}

#[cfg(feature = "npu-device")]
fn device_backend(sessions: usize) -> Result<Box<dyn oxn::Backend>, String> {
    Ok(Box::new(oxn::device::Device::open(sessions)?))
}

#[cfg(not(feature = "npu-device"))]
fn device_backend(_sessions: usize) -> Result<Box<dyn oxn::Backend>, String> {
    Err("built without the npu-device feature".into())
}

/// Write `[n][ctx_dim]` little-endian f32.
pub fn write_f32(path: &Path, v: &[f32]) -> Result<(), String> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() % 4 != 0 {
        return Err(format!("{}: {} bytes is not f32 data", path.display(), b.len()));
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
