//! `ModelDesc`: the qwen35moe architecture read from GGUF metadata and the tensor table.
//! Nothing here is hard-coded per model; the hot loop only sees these numbers.

use gguf::{GgmlType, Header, TensorInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated delta-net (linear attention) mixer.
    Recurrent,
    /// Full softmax attention (every `full_attention_interval`-th layer).
    Attention,
}

#[derive(Clone, Debug)]
pub struct ModelDesc {
    pub arch: String,
    /// Target layers (MTP / nextn layers excluded).
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_exp: usize,
    pub ff_shexp: usize,
    pub ssm_conv_kernel: usize,
    /// Key head dim of the delta net (`ssm.state_size`).
    pub ssm_d_state: usize,
    /// Number of key heads (`ssm.group_count`).
    pub ssm_n_kheads: usize,
    /// Number of value heads (`ssm.time_step_rank`).
    pub ssm_n_vheads: usize,
    /// Value width (`ssm.inner_size`) = n_vheads * head_v_dim.
    pub ssm_d_inner: usize,
    pub rms_eps: f32,
    /// RoPE: rotated dims (`rope.dimension_count`), base, M-RoPE sections, training context.
    pub n_rot: usize,
    pub rope_freq_base: f32,
    pub rope_sections: [i32; 4],
    pub n_ctx_train: usize,
    pub layers: Vec<LayerKind>,
}

impl ModelDesc {
    pub fn ssm_head_v_dim(&self) -> usize {
        self.ssm_d_inner / self.ssm_n_vheads
    }
    /// Channels of the causal conv (q, k, v concatenated) = rows of `attn_qkv`.
    pub fn ssm_conv_channels(&self) -> usize {
        self.ssm_d_inner + 2 * self.ssm_n_kheads * self.ssm_d_state
    }
    /// f32 recurrent state per recurrent layer: conv state + delta-rule S (bytes).
    pub fn recurrent_state_bytes_per_layer(&self) -> usize {
        4 * ((self.ssm_conv_kernel - 1) * self.ssm_conv_channels() + self.ssm_d_state * self.ssm_d_inner)
    }
    /// f16 K+V bytes per cached position per attention layer.
    pub fn kv_bytes_per_pos_per_layer(&self) -> usize {
        2 * 2 * self.n_head_kv * self.head_dim
    }
    pub fn n_recurrent(&self) -> usize {
        self.layers.iter().filter(|k| **k == LayerKind::Recurrent).count()
    }
    pub fn n_attention(&self) -> usize {
        self.layers.iter().filter(|k| **k == LayerKind::Attention).count()
    }

    pub fn from_header(h: &Header) -> Result<ModelDesc, String> {
        let arch = h.architecture().ok_or("no general.architecture")?.to_string();
        if arch != "qwen35moe" {
            return Err(format!("architecture {arch} is not supported (qwen35moe only)"));
        }
        let u = |k: &str| -> Result<usize, String> {
            h.get(&format!("{arch}.{k}")).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or(format!("missing {arch}.{k}"))
        };
        let block_count = u("block_count")?;
        let nextn = u("nextn_predict_layers").unwrap_or(0);
        let n_layer = block_count - nextn;
        let interval = u("full_attention_interval")?;
        let eps = h
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f64())
            .ok_or("missing rms eps")? as f32;
        let output = h.tensor("output.weight").ok_or("no output.weight")?;
        let mut layers = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let rec = h.tensor(&format!("blk.{il}.ssm_a")).is_some();
            let att = h.tensor(&format!("blk.{il}.attn_q.weight")).is_some();
            let kind = match (rec, att) {
                (true, false) => LayerKind::Recurrent,
                (false, true) => LayerKind::Attention,
                _ => return Err(format!("layer {il}: cannot classify (ssm_a {rec}, attn_q {att})")),
            };
            // llama.cpp qwen35moe: recurrent iff (il + 1) % interval != 0
            let expect = if (il + 1) % interval != 0 { LayerKind::Recurrent } else { LayerKind::Attention };
            if kind != expect {
                return Err(format!("layer {il}: tensors say {kind:?}, full_attention_interval says {expect:?}"));
            }
            layers.push(kind);
        }
        let d = ModelDesc {
            n_layer,
            n_embd: u("embedding_length")?,
            n_vocab: output.shape[1] as usize,
            n_head: u("attention.head_count")?,
            n_head_kv: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            ff_exp: u("expert_feed_forward_length")?,
            ff_shexp: u("expert_shared_feed_forward_length")?,
            ssm_conv_kernel: u("ssm.conv_kernel")?,
            ssm_d_state: u("ssm.state_size")?,
            ssm_n_kheads: u("ssm.group_count")?,
            ssm_n_vheads: u("ssm.time_step_rank")?,
            ssm_d_inner: u("ssm.inner_size")?,
            rms_eps: eps,
            n_rot: u("rope.dimension_count")?,
            rope_freq_base: h.get(&format!("{arch}.rope.freq_base")).and_then(|v| v.as_f64()).ok_or("missing rope.freq_base")? as f32,
            rope_sections: {
                let mut s = [0i32; 4];
                match h.get(&format!("{arch}.rope.dimension_sections")) {
                    Some(gguf::Value::Array(a)) => {
                        for (i, v) in a.iter().take(4).enumerate() {
                            s[i] = v.as_f64().ok_or("bad rope section")? as i32;
                        }
                    }
                    _ => return Err("missing rope.dimension_sections (qwen35moe uses IMRoPE)".into()),
                }
                s
            },
            n_ctx_train: u("context_length")?,
            layers,
            arch,
        };
        d.check(h)?;
        Ok(d)
    }

    /// Shape cross-checks between metadata and tensors (what the plan relies on).
    fn check(&self, h: &Header) -> Result<(), String> {
        let shape = |n: &str| h.tensor(n).map(|t| t.shape.clone()).ok_or(format!("missing tensor {n}"));
        let e = self.n_embd as u64;
        for (il, kind) in self.layers.iter().enumerate() {
            let p = |s: &str| format!("blk.{il}.{s}");
            match kind {
                LayerKind::Recurrent => {
                    if shape(&p("attn_qkv.weight"))? != vec![e, self.ssm_conv_channels() as u64] {
                        return Err(format!("layer {il}: attn_qkv shape"));
                    }
                    if shape(&p("ssm_out.weight"))? != vec![self.ssm_d_inner as u64, e] {
                        return Err(format!("layer {il}: ssm_out shape"));
                    }
                }
                LayerKind::Attention => {
                    if shape(&p("attn_q.weight"))? != vec![e, (2 * self.n_head * self.head_dim) as u64] {
                        return Err(format!("layer {il}: attn_q shape (expected q + gate per head)"));
                    }
                }
            }
            if shape(&p("ffn_gate_exps.weight"))? != vec![e, self.ff_exp as u64, self.n_expert as u64] {
                return Err(format!("layer {il}: ffn_gate_exps shape"));
            }
        }
        if self.ssm_n_vheads % self.ssm_n_kheads != 0 || self.n_head % self.n_head_kv != 0 {
            return Err("head counts do not divide".into());
        }
        if self.ff_exp % kernels_sys::QK_K != 0 || self.ff_shexp % kernels_sys::QK_K != 0 {
            return Err("expert FFN width must be a multiple of 256 (one activation block per MoE unit)".into());
        }
        if self.ssm_d_state != self.ssm_head_v_dim() {
            return Err("delta-net head_k_dim != head_v_dim (the fused GDN op requires S_k == S_v)".into());
        }
        if self.n_rot > self.head_dim || self.n_rot % 2 != 0 {
            return Err("bad rope.dimension_count".into());
        }
        Ok(())
    }
}

/// How a tensor contributes to per-token decode traffic (same roles as model_inspect.py).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Read in full every token.
    Dense,
    /// Routed expert tensor: `n_expert_used / n_expert` of it per token.
    Expert,
    /// Token embedding: one row per token.
    Embedding,
    /// MTP / nextn layer: not read during plain decode.
    Mtp,
}

pub fn role(t: &TensorInfo, desc: &ModelDesc) -> Role {
    if let Some(rest) = t.name.strip_prefix("blk.") {
        let il: usize = rest.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(0);
        if il >= desc.n_layer || t.name.contains(".nextn.") {
            return Role::Mtp;
        }
    }
    if t.name.contains("_exps") {
        Role::Expert
    } else if t.name.starts_with("token_embd.") {
        Role::Embedding
    } else {
        Role::Dense
    }
}

/// Is this a matmul weight the kernel set can run, and with which kernel?
pub fn qtype(ty: GgmlType) -> Option<kernels_sys::QType> {
    match ty {
        GgmlType::Q3_K => Some(kernels_sys::QType::Q3K),
        GgmlType::Q4_K => Some(kernels_sys::QType::Q4K),
        GgmlType::Q5_K => Some(kernels_sys::QType::Q5K),
        GgmlType::Q6_K => Some(kernels_sys::QType::Q6K),
        GgmlType::Q8_0 => Some(kernels_sys::QType::Q8_0),
        _ => None,
    }
}
