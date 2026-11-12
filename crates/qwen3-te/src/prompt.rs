//! klein prompt → token ids + key mask, as sd.cpp builds them (see `config.rs` for the source).
//!
//! sd.cpp tokenizes three pieces SEPARATELY and concatenates the ids: the template prefix, the user
//! text (after its A1111-style emphasis parser) and the template suffix. BPE does not merge across
//! the piece boundaries, so neither do we.
//!
//! Only plain prompts take the native path: a prompt with emphasis/escape syntax (`( ) [ ] \`),
//! the `BREAK` keyword, or a special-token string is refused with [`PromptError::NeedsReference`];
//! the caller then keeps the sd-cli route. For a plain prompt sd.cpp's parser returns the text
//! unchanged with weight 1.0, and `apply_token_weights` is a no-op.

use crate::config::KleinSpec;

/// Special-token strings that sd.cpp's built-in Qwen2 tokenizer splits on
/// (stable-diffusion.cpp 5a1ba5d `tokenizers/qwen2_tokenizer.cpp`). A user prompt that contains one
/// of them would depend on two tokenizers agreeing about special-token partitioning: refused.
pub const SD_SPECIALS: &[&str] = &[
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
    "<tool_response>", "</tool_response>", "<think>", "</think>", "<|boi_token|>", "<|bor_token|>",
    "<|eor_token|>", "<|bot_token|>", "<|tms_token|>",
];

#[derive(Debug, Clone, PartialEq)]
pub enum PromptError {
    /// Valid for sd.cpp but outside the native subset: use the reference (sd-cli) route.
    NeedsReference(String),
    Tokenizer(String),
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptError::NeedsReference(s) => write!(f, "prompt needs the reference encoder: {s}"),
            PromptError::Tokenizer(s) => write!(f, "tokenizer: {s}"),
        }
    }
}

/// Token-id source. The real one is the GGUF BPE tokenizer (`tokenizer` crate, llama.cpp port,
/// `parse_special = true`, no BOS/EOS); tests use a fake.
pub trait Tokenize {
    fn encode_piece(&self, text: &str) -> Result<Vec<u32>, String>;
}

impl Tokenize for tokenizer::Tokenizer {
    fn encode_piece(&self, text: &str) -> Result<Vec<u32>, String> {
        // sd.cpp's Qwen2Tokenizer adds neither BOS nor EOS (tokenizer.h add_*_token = false);
        // add_special = false keeps llama's GGUF flags out of it.
        Ok(self.encode(text, false, true))
    }
}

/// Is `text` in the subset whose sd.cpp parse is the identity with weight 1.0?
pub fn check_plain(text: &str) -> Result<(), PromptError> {
    if let Some(c) = text.chars().find(|c| matches!(c, '(' | ')' | '[' | ']' | '\\')) {
        return Err(PromptError::NeedsReference(format!("emphasis/escape character '{c}'")));
    }
    if has_word(text, "BREAK") {
        return Err(PromptError::NeedsReference("BREAK keyword".into()));
    }
    if text.contains("<lora:") {
        // sd-cli strips LoRA tags from the prompt when a LoRA directory is configured
        return Err(PromptError::NeedsReference("LoRA tag".into()));
    }
    if let Some(s) = SD_SPECIALS.iter().find(|s| text.contains(*s)) {
        return Err(PromptError::NeedsReference(format!("special token string {s}")));
    }
    Ok(())
}

/// `\bWORD\b` with ECMAScript word characters `[A-Za-z0-9_]` (std::regex default grammar).
fn has_word(text: &str, word: &str) -> bool {
    let b = text.as_bytes();
    let isw = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut from = 0;
    while let Some(p) = text[from..].find(word) {
        let s = from + p;
        let e = s + word.len();
        let left = s == 0 || !isw(b[s - 1]);
        let right = e == b.len() || !isw(b[e]);
        if left && right {
            return true;
        }
        from = s + 1;
    }
    false
}

/// The encoder input of one prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// `max(min_len, n_real)` ids: the real ones, then `pad_id`.
    pub tokens: Vec<u32>,
    /// `true` for real tokens, `false` for pads (keys a query may not attend to).
    pub mask: Vec<bool>,
    pub n_real: usize,
}

/// Build the klein encoder input for `user` (the prompt text as typed).
pub fn klein_tokens(tok: &dyn Tokenize, spec: &KleinSpec, user: &str) -> Result<Encoded, PromptError> {
    check_plain(user)?;
    let mut ids = tok.encode_piece(spec.prefix).map_err(PromptError::Tokenizer)?;
    if !user.is_empty() {
        ids.extend(tok.encode_piece(user).map_err(PromptError::Tokenizer)?);
    }
    ids.extend(tok.encode_piece(spec.suffix).map_err(PromptError::Tokenizer)?);
    Ok(pad(ids, spec))
}

/// Right-pad to `min_len` (never truncate: sd.cpp's klein max_length is unbounded).
pub fn pad(mut ids: Vec<u32>, spec: &KleinSpec) -> Encoded {
    let n_real = ids.len();
    let n = n_real.max(spec.min_len);
    ids.resize(n, spec.pad_id);
    let mask = (0..n).map(|i| i < n_real).collect();
    Encoded { tokens: ids, mask, n_real }
}
