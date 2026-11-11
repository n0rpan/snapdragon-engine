//! The Flash model as stored: 4 GGUF shards (header-only parse), tensor lookup by name across
//! shards, and the per-layer geometry the plan needs. Hyper-parameters are read from the
//! metadata and checked against the values this prototype was written for.

use crate::ffi::W;
use std::path::{Path, PathBuf};

pub const N_LAYER: usize = 48;
pub const N_EMBD: usize = 2560;
pub const N_HC: usize = 4; // hyper-connection streams
pub const HC_RANK: usize = 320;
pub const N_EXPERT: usize = 512;
pub const TOP_K: usize = 10;
pub const FF_EXP: usize = 640;
pub const N_VOCAB: usize = 248320;
// gated delta net
pub const SSM_INNER: usize = 6144; // 48 value heads x 128
pub const SSM_V_HEADS: usize = 48;
pub const SSM_K_HEADS: usize = 16;
pub const SSM_D: usize = 128;
pub const SSM_CONV: usize = 4;
pub const SSM_QKV: usize = 2 * SSM_K_HEADS * SSM_D + SSM_INNER; // 10240
// attention
pub const N_HEAD: usize = 24;
pub const N_HEAD_KV: usize = 2;
pub const HEAD_D: usize = 256;
pub const IDX_HEADS: usize = 4;
pub const IDX_D: usize = 128;
// per-layer embeddings (N-gram table)
pub const PLE_LAYER: usize = 1;
pub const PLE_HEADS: usize = 16;
pub const PLE_DIM: usize = 160; // values per stored row
pub const PLE_ROW_BYTES: usize = 90; // IQ4_NL, 5 blocks

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub shard: usize,
    /// absolute byte offset in the shard file
    pub off: u64,
    pub bytes: u64,
    pub w: W,
    pub shape: Vec<u64>,
}

pub struct Shards {
    pub paths: Vec<PathBuf>,
    pub files: Vec<gguf::Gguf>,
    pub tensors: Vec<Tensor>,
}

impl Shards {
    /// Open every `*-0000N-of-0000M.gguf` in `dir` (or the shard set of the given first file).
    pub fn open(dir: &Path) -> Result<Shards, String> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(format!("no .gguf in {}", dir.display()));
        }
        let mut files = Vec::new();
        let mut tensors = Vec::new();
        for (i, p) in paths.iter().enumerate() {
            let g = gguf::Gguf::open(p).map_err(|e| e.0)?;
            for t in &g.header.tensors {
                let w = W::from_ggml(t.ty.0).ok_or_else(|| format!("{}: unsupported type {}", t.name, t.ty.0))?;
                tensors.push(Tensor {
                    name: t.name.clone(),
                    shard: i,
                    off: g.header.data_offset + t.offset,
                    bytes: t.bytes(),
                    w,
                    shape: t.shape.clone(),
                });
            }
            files.push(g);
        }
        let s = Shards { paths, files, tensors };
        s.check_hparams()?;
        Ok(s)
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn req(&self, name: &str) -> &Tensor {
        self.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    fn meta_u64(&self, key: &str) -> Option<u64> {
        self.files.iter().find_map(|f| f.header.get(key).and_then(|v| v.as_u64()))
    }

    fn check_hparams(&self) -> Result<(), String> {
        let arch = self.files.iter().find_map(|f| f.header.architecture().map(|s| s.to_string()));
        if arch.as_deref() != Some("qwen4exp") {
            return Err(format!("architecture {arch:?}, expected qwen4exp"));
        }
        let want: [(&str, u64); 9] = [
            ("qwen4exp.block_count", N_LAYER as u64),
            ("qwen4exp.embedding_length", N_EMBD as u64),
            ("qwen4exp.expert_count", N_EXPERT as u64),
            ("qwen4exp.expert_used_count", TOP_K as u64),
            ("qwen4exp.expert_feed_forward_length", FF_EXP as u64),
            ("qwen4exp.ssm.inner_size", SSM_INNER as u64),
            ("qwen4exp.hyper_connection.count", N_HC as u64),
            ("qwen4exp.hyper_connection.low_rank", HC_RANK as u64),
            ("qwen4exp.attention.head_count", N_HEAD as u64),
        ];
        for (k, v) in want {
            if self.meta_u64(k) != Some(v) {
                return Err(format!("{k} = {:?}, prototype written for {v}", self.meta_u64(k)));
            }
        }
        Ok(())
    }

    /// The PLE head offsets / vocab sizes (16 heads) from the metadata.
    pub fn ple_heads(&self) -> (Vec<u64>, Vec<u64>) {
        let arr = |key: &str| -> Vec<u64> {
            self.files
                .iter()
                .find_map(|f| match f.header.get(key) {
                    Some(gguf::Value::Array(v)) => Some(v.iter().filter_map(|x| x.as_u64()).collect()),
                    _ => None,
                })
                .unwrap_or_default()
        };
        (arr("qwen4exp.ple.head_offsets"), arr("qwen4exp.ple.head_vocab_sizes"))
    }

    pub fn meta_u64_pub(&self, key: &str) -> Option<u64> {
        self.meta_u64(key)
    }
    pub fn u64_array(&self, key: &str) -> Vec<u64> {
        self.files
            .iter()
            .find_map(|f| match f.header.get(key) {
                Some(gguf::Value::Array(v)) => Some(v.iter().filter_map(|x| x.as_u64()).collect()),
                _ => None,
            })
            .unwrap_or_default()
    }

    pub fn is_attn_layer(&self, l: usize) -> bool {
        self.get(&format!("blk.{l}.attn_q.weight")).is_some()
    }
}

/// Expert projection geometry of one layer.
#[derive(Clone, Copy, Debug)]
pub struct ExpertGeom {
    pub gate_w: W,
    pub down_w: W,
    pub gate_bytes: usize, // per expert, = up bytes
    pub down_bytes: usize,
    /// gate + up + down of one expert
    pub record: usize,
    /// record rounded up to 4 KiB (sidecar stride and cache slot size)
    pub slot: usize,
}

pub fn expert_geom(s: &Shards, l: usize) -> ExpertGeom {
    let g = s.req(&format!("blk.{l}.ffn_gate_exps.weight"));
    let u = s.req(&format!("blk.{l}.ffn_up_exps.weight"));
    let d = s.req(&format!("blk.{l}.ffn_down_exps.weight"));
    assert_eq!(g.w, u.w);
    assert_eq!(g.shape, vec![N_EMBD as u64, FF_EXP as u64, N_EXPERT as u64]);
    assert_eq!(d.shape, vec![FF_EXP as u64, N_EMBD as u64, N_EXPERT as u64]);
    let gate_bytes = g.w.row_bytes(N_EMBD) * FF_EXP;
    let down_bytes = d.w.row_bytes(FF_EXP) * N_EMBD;
    assert_eq!(g.bytes as usize, gate_bytes * N_EXPERT);
    assert_eq!(d.bytes as usize, down_bytes * N_EXPERT);
    let record = 2 * gate_bytes + down_bytes;
    ExpertGeom { gate_w: g.w, down_w: d.w, gate_bytes, down_bytes, record, slot: record.div_ceil(4096) * 4096 }
}
