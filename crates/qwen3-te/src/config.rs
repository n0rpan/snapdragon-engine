//! Model shape (from GGUF `qwen3.*` metadata) and the FLUX.2 klein conditioning spec.
//!
//! The klein spec is a transcription of stable-diffusion.cpp 5a1ba5d
//! `src/conditioning/conditioner.hpp` (`VERSION_FLUX2_KLEIN` branch of `LLMEmbedder`) and
//! `src/model/te/llm.hpp` (`forward_embeds`, `LLMArch::QWEN3`):
//! - template `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
//! - pad with `<|endoftext|>` (151643) to at least 512 tokens, padding on the right, mask 0 on pads;
//! - attention mask = causal AND key-not-pad; every pad row still gets its own hidden state;
//! - output = the residual stream after layers 9, 18 and 27 (1-based; i.e. after block index 8, 17, 26),
//!   concatenated along the hidden dimension: `[n][3 * hidden]` = `[n][7680]` for Qwen3-4B;
//! - no final norm (the 37th "layer" is not requested), no token-weight rescale for plain prompts,
//!   `prompt_template_encode_start_idx = 0` (nothing is sliced off).

use gguf::Header;

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3Config {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub vocab: usize,
}

impl Qwen3Config {
    /// Qwen3-4B (the klein text encoder) as published.
    pub fn qwen3_4b() -> Qwen3Config {
        Qwen3Config {
            hidden: 2560,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            ff: 9728,
            eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab: 151936,
        }
    }

    pub fn from_header(h: &Header) -> Result<Qwen3Config, String> {
        let arch = h.architecture().unwrap_or("");
        if arch != "qwen3" {
            return Err(format!("text encoder: architecture '{arch}', want qwen3 (dense)"));
        }
        let u = |k: &str| -> Result<usize, String> {
            h.get(&format!("qwen3.{k}"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("text encoder: missing qwen3.{k}"))
        };
        let f = |k: &str, d: f64| -> f32 { h.get(&format!("qwen3.{k}")).and_then(|v| v.as_f64()).unwrap_or(d) as f32 };
        let hidden = u("embedding_length")?;
        let n_heads = u("attention.head_count")?;
        let head_dim = u("attention.key_length").unwrap_or(hidden / n_heads.max(1));
        let vocab = h
            .tensor("token_embd.weight")
            .and_then(|t| t.shape.get(1).copied())
            .ok_or("text encoder: missing token_embd.weight")? as usize;
        let c = Qwen3Config {
            hidden,
            n_layers: u("block_count")?,
            n_heads,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim,
            ff: u("feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon", 1e-6),
            rope_theta: f("rope.freq_base", 1_000_000.0),
            vocab,
        };
        c.check()?;
        Ok(c)
    }

    pub fn check(&self) -> Result<(), String> {
        if self.hidden == 0 || self.n_heads == 0 || self.n_kv_heads == 0 || self.head_dim == 0 || self.ff == 0 {
            return Err(format!("text encoder: zero dimension in {self:?}"));
        }
        if self.n_heads % self.n_kv_heads != 0 {
            return Err(format!("text encoder: {} heads not a multiple of {} kv heads", self.n_heads, self.n_kv_heads));
        }
        if self.head_dim % 2 != 0 {
            return Err("text encoder: odd head_dim".into());
        }
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

/// What klein needs from the encoder.
#[derive(Clone, Debug, PartialEq)]
pub struct KleinSpec {
    /// 1-based layer counts after which the residual stream is taken (sd.cpp `out_layers`).
    pub out_layers: Vec<usize>,
    /// pad to at least this many tokens
    pub min_len: usize,
    pub pad_id: u32,
    pub prefix: &'static str,
    pub suffix: &'static str,
}

impl KleinSpec {
    pub fn klein() -> KleinSpec {
        KleinSpec {
            out_layers: vec![9, 18, 27],
            min_len: 512,
            pad_id: 151643,
            prefix: "<|im_start|>user\n",
            suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        }
    }

    /// Blocks that must run: the deepest requested layer (27 of 36 for klein).
    pub fn layers_needed(&self) -> usize {
        self.out_layers.iter().copied().max().unwrap_or(0)
    }

    pub fn check(&self, c: &Qwen3Config) -> Result<(), String> {
        if self.out_layers.is_empty() || self.out_layers.iter().any(|&l| l == 0 || l > c.n_layers) {
            return Err(format!("klein spec: out_layers {:?} outside 1..={}", self.out_layers, c.n_layers));
        }
        if self.out_layers.windows(2).any(|w| w[0] >= w[1]) {
            return Err("klein spec: out_layers must be strictly increasing".into());
        }
        Ok(())
    }

    pub fn ctx_dim(&self, c: &Qwen3Config) -> usize {
        self.out_layers.len() * c.hidden
    }
}
