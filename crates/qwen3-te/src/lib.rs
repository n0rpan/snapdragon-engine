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
//!   [`ggml::GgmlLinears`] (`cpu-ggml`, sd.cpp's ggml Q8_0 kernels: the production CPU path).
//! - [`metrics`]: hidden-state agreement and the gate.

pub mod attn_gqa;
pub mod config;
pub mod cpu;
pub mod encoder;
pub mod ggml;
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
    /// sd.cpp's ggml Q8_0 kernels (repacked weights): the production CPU path
    CpuGgml,
    Npu { backend: Box<dyn oxn::Backend>, opts: npu::NpuOpts },
}

impl TeBackend {
    /// Arithmetic class, part of the encoder identity (cache namespace).
    pub fn arith(&self) -> String {
        match self {
            TeBackend::CpuF32 => "cpu-f32".into(),
            TeBackend::CpuQ8 => "cpu-q8".into(),
            TeBackend::CpuGgml => "cpu-ggml".into(),
            TeBackend::Npu { opts, .. } => {
                format!("npu-{}", if opts.weights == oxn::wfmt::WFmt::Fp16 { "fp16" } else { "w8" })
            }
        }
    }
}

/// Version of the native graph + prompt handling. Bump on any change that can alter the context.
pub const ENCODER_VERSION: &str = "qwen3-te/1";

/// Encoder identity for cache provenance: graph version, arithmetic class and the TE file
/// (name and size, like the DiT part of the key). Computable without loading the model.
pub fn encoder_identity(te: &Path, arith: &str) -> Result<String, String> {
    let meta = std::fs::metadata(te).map_err(|e| format!("{}: {e}", te.display()))?;
    let name = te.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or("text encoder path has no file name")?;
    if !["cpu-f32", "cpu-q8", "cpu-ggml", "npu-fp16", "npu-w8"].contains(&arith) {
        return Err(format!("unknown encoder arithmetic '{arith}'"));
    }
    Ok(format!("{ENCODER_VERSION};{arith};{name};{}", meta.len()))
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

/// `VmRSS` and `VmHWM` of this process in KiB (Linux; `None` elsewhere).
pub fn rss_kib() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let f = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok());
    Some((f("VmRSS:")?, f("VmHWM:")?))
}

/// Open `gguf`, tokenize `prompt` with its vocabulary, and encode it for klein.
/// `max_n` > 0: a prompt whose padded length exceeds it is refused as `NeedsReference` (the
/// native path is gated at n_txt 512). Every accelerator session is released before return
/// (`summary.release_s`), so the DiT can place its weights afterwards (one at a time).
pub fn encode_prompt(gguf: &Path, prompt: &str, backend: TeBackend, threads: usize, max_n: usize) -> Result<TeOutput, TeError> {
    let t_all = Instant::now();
    let rss0 = rss_kib();
    let spec = KleinSpec::klein();
    let arith = backend.arith();
    let t = Instant::now();
    let tok = tokenizer::Tokenizer::from_gguf(gguf).map_err(|e| TeError::Other(format!("tokenizer: {}", e.0)))?;
    let enc = match prompt::klein_tokens(&tok, &spec, prompt) {
        Ok(e) => e,
        Err(prompt::PromptError::NeedsReference(s)) => return Err(TeError::NeedsReference(s)),
        Err(e) => return Err(TeError::Other(e.to_string())),
    };
    drop(tok);
    if max_n > 0 && enc.tokens.len() > max_n {
        return Err(TeError::NeedsReference(format!("{} tokens, the native path is gated at {max_n}", enc.tokens.len())));
    }
    let tokenize_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let g = gguf::Gguf::open(gguf).map_err(|e| TeError::Other(e.0))?;
    let w = weights::TeWeights::from_gguf(&g, spec.layers_needed())?;
    let open_s = t.elapsed().as_secs_f64();
    let encd = Encoder::new(&w, spec.clone(), threads)?;
    let dma = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (ctx, tm, name, lsum, prep_s) = {
        let t = Instant::now();
        let lin: Box<dyn Linears + '_> = match backend {
            TeBackend::CpuF32 => Box::new(CpuLinears { w: &w, threads, mode: CpuMode::F32 }),
            TeBackend::CpuQ8 => Box::new(CpuLinears { w: &w, threads, mode: CpuMode::Q8 }),
            TeBackend::CpuGgml => Box::new(ggml::GgmlLinears::new(&w, threads)?),
            TeBackend::Npu { backend, opts } => {
                let counted = Box::new(npu::CountingBackend::new(backend, dma.clone()));
                Box::new(npu::NpuLinears::new(&w, counted, opts, threads)?)
            }
        };
        let prep_s = t.elapsed().as_secs_f64();
        let (ctx, tm) = encd.encode(&*lin, &enc)?;
        (ctx, tm, lin.name(), lin.summary(), prep_s)
        // `lin` (accelerator sessions and buffers, or the repacked weights) is dropped here
    };
    let release_s = t_all.elapsed().as_secs_f64() - tokenize_s - open_s - prep_s - tm.total_s;
    let rss1 = rss_kib();
    let summary = json!({
        "backend": name,
        "encoder_identity": encoder_identity(gguf, &arith).ok(),
        "n": enc.tokens.len(),
        "n_real": enc.n_real,
        "ctx_dim": spec.ctx_dim(&w.cfg),
        "out_layers": spec.out_layers,
        "tokenize_s": tokenize_s,
        "open_s": open_s,
        "prepare_s": prep_s,
        "encode": tm.json(),
        "release_s": release_s,
        "linears": lsum,
        "dma_allocated_bytes": dma.load(std::sync::atomic::Ordering::Relaxed),
        "host": {"rss_start_kib": rss0.map(|r| r.0), "rss_end_kib": rss1.map(|r| r.0), "hwm_kib": rss1.map(|r| r.1)},
        "wall_s": t_all.elapsed().as_secs_f64(),
    });
    Ok(TeOutput { ctx, n: enc.tokens.len(), n_real: enc.n_real, ctx_dim: spec.ctx_dim(&w.cfg), tokens: enc.tokens, summary })
}

/// Backend from command-line style options (shared by the `qwen3-te` binary and `oxi`):
/// `te-backend` cpu-f32 (default) | cpu-q8 | cpu-ggml | npu; for npu: `te-npu` emu|device (default emu),
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
        "cpu-ggml" => Ok(TeBackend::CpuGgml),
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
        o => Err(format!("--te-backend must be cpu-f32, cpu-q8, cpu-ggml or npu, got '{o}'")),
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
