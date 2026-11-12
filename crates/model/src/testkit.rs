//! Test kit (feature `testkit`, and the crate's own tests): the tiny synthetic qwen35moe GGUF
//! writer, so server-side tests can drive a real `OxeEngine` (recurrent + KV + MTP state)
//! model-free. Not part of the production build.

use gguf::{GgmlType, TensorInfo};
use kernels_sys::QType;

pub fn rng_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}


pub fn fix_scales(q: QType, v: &mut [u8]) {
    let one = [0x00u8, 0x20]; // f16 0.0078125
    for blk in v.chunks_mut(q.block_bytes()) {
        match q {
            QType::Q4K | QType::Q5K => {
                blk[0..2].copy_from_slice(&one);
                blk[2..4].copy_from_slice(&one);
            }
            QType::Q3K => blk[108..110].copy_from_slice(&one),
            QType::Q6K => blk[208..210].copy_from_slice(&one),
            QType::Q8_0 => blk[0..2].copy_from_slice(&one),
        }
    }
}

pub struct GgufWriter {
    pub kv: Vec<u8>,
    pub n_kv: u64,
    pub tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

impl GgufWriter {
    pub fn key(&mut self, k: &str, vtype: u32) {
        self.kv.extend((k.len() as u64).to_le_bytes());
        self.kv.extend(k.as_bytes());
        self.kv.extend(vtype.to_le_bytes());
        self.n_kv += 1;
    }
    pub fn u32(&mut self, k: &str, v: u32) {
        self.key(k, 4);
        self.kv.extend(v.to_le_bytes());
    }
    pub fn f32(&mut self, k: &str, v: f32) {
        self.key(k, 6);
        self.kv.extend(v.to_le_bytes());
    }
    pub fn arr_i32(&mut self, k: &str, v: &[i32]) {
        self.key(k, 9);
        self.kv.extend(5u32.to_le_bytes());
        self.kv.extend((v.len() as u64).to_le_bytes());
        v.iter().for_each(|x| self.kv.extend(x.to_le_bytes()));
    }
    pub fn str(&mut self, k: &str, v: &str) {
        self.key(k, 8);
        self.kv.extend((v.len() as u64).to_le_bytes());
        self.kv.extend(v.as_bytes());
    }
    pub fn tensor(&mut self, name: &str, shape: &[u64], ty: GgmlType, seed: u64) {
        let t = TensorInfo { name: name.into(), shape: shape.to_vec(), ty, offset: 0 };
        let n = t.bytes() as usize;
        let data = if ty == GgmlType(30) {
            // BF16: small finite values
            (0..n / 2).flat_map(|i| ((((((i * 7 + seed as usize) % 19) as f32) * 0.01 - 0.09).to_bits() >> 16) as u16).to_le_bytes()).collect()
        } else { match crate::desc::qtype(ty) {
            Some(q) => {
                let mut v = rng_bytes(n, seed);
                fix_scales(q, &mut v);
                v
            }
            None => (0..n / 4).flat_map(|i| (((i * 7 + seed as usize) % 19) as f32 * 0.01 - 0.09).to_le_bytes()).collect(),
        } };
        self.tensors.push((name.into(), shape.to_vec(), ty, data));
    }
    pub fn write(&self, path: &std::path::Path) {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend((self.tensors.len() as u64).to_le_bytes());
        b.extend(self.n_kv.to_le_bytes());
        b.extend(&self.kv);
        let mut off = 0u64;
        let mut offs = Vec::new();
        for (name, shape, ty, data) in &self.tensors {
            b.extend((name.len() as u64).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend((shape.len() as u32).to_le_bytes());
            shape.iter().for_each(|d| b.extend(d.to_le_bytes()));
            b.extend(ty.0.to_le_bytes());
            b.extend(off.to_le_bytes());
            offs.push(off);
            off = (off + data.len() as u64).div_ceil(32) * 32;
        }
        b.resize(b.len().div_ceil(32) * 32, 0);
        let base = b.len();
        for ((_, _, _, data), o) in self.tensors.iter().zip(offs) {
            b.resize(base + o as usize, 0);
            b.extend(data);
        }
        std::fs::write(path, b).unwrap();
    }
}

/// A 4-layer qwen35moe with tiny dims (3 delta-net + 1 attention, 8 experts top-2,
/// Q3_K / Q4_K / Q5_K experts, Q6_K and Q8_0 shared experts, Q6_K head, F32 beta/alpha).
pub fn mini_model(path: &std::path::Path) {
    mini_model_ext(path, false)
}

/// The mini model (GQA: 4 query heads over 2 KV heads), optionally with an MTP block `blk.4` (attention + MoE with Q8_0 experts
/// like Ornith's head, BF16 router like Qwen3.6's, nextn eh_proj/enorm/hnorm/shared_head_norm).
pub fn mini_model_ext(path: &std::path::Path, mtp: bool) {
    mini_model_ctx(path, mtp, 4096)
}

/// [`mini_model_ext`] with GGUF `context_length` = `ctx_len` (LONG-CTX tests: 262144 like Ornith / Qwen3.6).
pub fn mini_model_ctx(path: &std::path::Path, mtp: bool, ctx_len: u32) {
    let mut w = GgufWriter { kv: vec![], n_kv: 0, tensors: vec![] };
    let a = "qwen35moe";
    w.str("general.architecture", a);
    w.u32("general.alignment", 32);
    for (k, v) in [
        ("block_count", if mtp { 5 } else { 4 }),
        ("nextn_predict_layers", mtp as u32),
        ("full_attention_interval", 4),
        ("embedding_length", 256),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("expert_count", 8),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 256),
        ("expert_shared_feed_forward_length", 256),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 1),
        ("ssm.time_step_rank", 2),
        ("ssm.inner_size", 256),
    ] {
        w.u32(&format!("{a}.{k}"), v);
    }
    w.f32(&format!("{a}.attention.layer_norm_rms_epsilon"), 1e-6);
    w.u32(&format!("{a}.rope.dimension_count"), 64);
    w.u32(&format!("{a}.context_length"), ctx_len);
    w.f32(&format!("{a}.rope.freq_base"), 1e7);
    w.arr_i32(&format!("{a}.rope.dimension_sections"), &[11, 11, 10, 0]);
    let (e, f32_, q3, q4, q6) = (256u64, GgmlType::F32, GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q6_K);
    w.tensor("token_embd.weight", &[e, 64], q4, 1);
    w.tensor("output_norm.weight", &[e], f32_, 2);
    w.tensor("output.weight", &[e, 64], q6, 3);
    for il in 0..4u64 {
        let p = |s: &str| format!("blk.{il}.{s}");
        let s = 100 * (il + 1);
        w.tensor(&p("attn_norm.weight"), &[e], f32_, s);
        w.tensor(&p("post_attention_norm.weight"), &[e], f32_, s + 1);
        if il == 3 {
            w.tensor(&p("attn_q.weight"), &[e, 1024], q4, s + 2);
            w.tensor(&p("attn_k.weight"), &[e, 256], q4, s + 3);
            w.tensor(&p("attn_v.weight"), &[e, 256], q4, s + 4);
            w.tensor(&p("attn_q_norm.weight"), &[128], f32_, s + 5);
            w.tensor(&p("attn_k_norm.weight"), &[128], f32_, s + 6);
            w.tensor(&p("attn_output.weight"), &[512, e], q4, s + 7);
        } else {
            w.tensor(&p("attn_qkv.weight"), &[e, 512], q4, s + 2);
            w.tensor(&p("attn_gate.weight"), &[e, 256], q4, s + 3);
            w.tensor(&p("ssm_beta.weight"), &[e, 2], f32_, s + 4);
            w.tensor(&p("ssm_alpha.weight"), &[e, 2], f32_, s + 5);
            w.tensor(&p("ssm_conv1d.weight"), &[4, 512], f32_, s + 6);
            w.tensor(&p("ssm_a"), &[2], f32_, s + 7);
            w.tensor(&p("ssm_dt.bias"), &[2], f32_, s + 8);
            w.tensor(&p("ssm_norm.weight"), &[128], f32_, s + 9);
            w.tensor(&p("ssm_out.weight"), &[256, e], q4, s + 10);
        }
        let xt = [q3, q4, GgmlType::Q5_K, q4][il as usize];
        w.tensor(&p("ffn_gate_inp.weight"), &[e, 8], f32_, s + 11);
        w.tensor(&p("ffn_gate_inp_shexp.weight"), &[e], f32_, s + 12);
        w.tensor(&p("ffn_gate_exps.weight"), &[e, 256, 8], xt, s + 13);
        w.tensor(&p("ffn_up_exps.weight"), &[e, 256, 8], xt, s + 14);
        w.tensor(&p("ffn_down_exps.weight"), &[256, e, 8], xt, s + 15);
        let st = if il == 1 { GgmlType::Q8_0 } else { q6 };
        w.tensor(&p("ffn_gate_shexp.weight"), &[e, 256], st, s + 16);
        w.tensor(&p("ffn_up_shexp.weight"), &[e, 256], st, s + 17);
        w.tensor(&p("ffn_down_shexp.weight"), &[256, e], q6, s + 18);
    }
    if mtp {
        let p = |s: &str| format!("blk.4.{s}");
        let (q8, bf) = (GgmlType::Q8_0, GgmlType(30));
        w.tensor(&p("attn_norm.weight"), &[e], f32_, 901);
        w.tensor(&p("post_attention_norm.weight"), &[e], f32_, 902);
        w.tensor(&p("attn_q.weight"), &[e, 512], q8, 903);
        w.tensor(&p("attn_k.weight"), &[e, 128], q8, 904);
        w.tensor(&p("attn_v.weight"), &[e, 128], q8, 905);
        w.tensor(&p("attn_q_norm.weight"), &[128], f32_, 906);
        w.tensor(&p("attn_k_norm.weight"), &[128], f32_, 907);
        w.tensor(&p("attn_output.weight"), &[256, e], q8, 908);
        w.tensor(&p("ffn_gate_inp.weight"), &[e, 8], bf, 911);
        w.tensor(&p("ffn_gate_inp_shexp.weight"), &[e], bf, 912);
        w.tensor(&p("ffn_gate_exps.weight"), &[e, 256, 8], q8, 913);
        w.tensor(&p("ffn_up_exps.weight"), &[e, 256, 8], q8, 914);
        w.tensor(&p("ffn_down_exps.weight"), &[256, e, 8], q8, 915);
        w.tensor(&p("ffn_gate_shexp.weight"), &[e, 256], q8, 916);
        w.tensor(&p("ffn_up_shexp.weight"), &[e, 256], q8, 917);
        w.tensor(&p("ffn_down_shexp.weight"), &[256, e], q8, 918);
        w.tensor(&p("nextn.eh_proj.weight"), &[2 * e, e], q8, 919);
        w.tensor(&p("nextn.enorm.weight"), &[e], f32_, 920);
        w.tensor(&p("nextn.hnorm.weight"), &[e], f32_, 921);
        w.tensor(&p("nextn.shared_head_norm.weight"), &[e], f32_, 922);
    }
    w.write(path);
}


