//! Plan / byte-accounting tests against model_inspect.py fixtures, the weight cache, and
//! end-to-end decode of a tiny synthetic qwen35moe GGUF (determinism across thread counts).

use crate::desc::ModelDesc;
use crate::engine::{Engine, EngineConfig};
use crate::plan::{PhaseKind, Plan};
use crate::weights::{Backing, Layout, LoadOptions, Verify, WeightStore};
use gguf::{GgmlType, Header, TensorInfo, Value};
use kernels_sys::{quantize_act, vec_dot_generic, QType};
use serde_json::Value as J;
use std::path::PathBuf;
use std::time::Duration;

fn fixture(name: &str) -> J {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn to_value(v: &J) -> Value {
    match v {
        J::String(s) => Value::Str(s.clone()),
        J::Bool(b) => Value::Bool(*b),
        J::Array(a) => Value::Array(a.iter().map(to_value).collect()),
        J::Number(n) if n.is_u64() => Value::U64(n.as_u64().unwrap()),
        J::Number(n) if n.is_i64() => Value::I64(n.as_i64().unwrap()),
        J::Number(n) => Value::F64(n.as_f64().unwrap()),
        J::Null | J::Object(_) => panic!("unexpected metadata value {v}"),
    }
}

/// A Header rebuilt from model_inspect.py's fixture (no model file needed).
fn header(fx: &J) -> Header {
    Header {
        version: fx["gguf_version"].as_u64().unwrap() as u32,
        metadata: fx["metadata"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), to_value(v))).collect(),
        tensors: fx["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| TensorInfo {
                name: t[0].as_str().unwrap().into(),
                ty: GgmlType::from_name(t[1].as_str().unwrap()).unwrap(),
                shape: t[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect(),
                offset: t[3].as_u64().unwrap(),
            })
            .collect(),
        data_offset: fx["data_offset"].as_u64().unwrap(),
        header_bytes_read: fx["header_bytes_read"].as_u64().unwrap(),
    }
}

fn check_budget(key: &str) {
    let fx = fixture(key);
    let h = header(&fx);
    let desc = ModelDesc::from_header(&h).unwrap();
    assert_eq!(desc.n_layer, fx["layers"]["target"].as_u64().unwrap() as usize);
    assert_eq!(desc.n_recurrent(), fx["layers"]["recurrent"].as_u64().unwrap() as usize);
    let n_kv = 2048;
    let plan = Plan::build(&h, &desc, 8, n_kv).unwrap();
    let b = plan.budget(&h.tensors);
    // weights: exactly model_inspect.py's active bytes per token
    assert_eq!(b.weights(), fx["decode"]["active_bytes_per_token"].as_u64().unwrap(), "{key} weight bytes/token");
    // recurrent state: read + write of model_inspect's state bytes
    assert_eq!(b.state_rw, 2 * fx["recurrent_state_bytes"].as_u64().unwrap(), "{key} state");
    // KV: model_inspect's f16 bytes per token of context x n_kv, plus one position written
    let kvpt = fx["kv_bytes_per_token_f16"].as_u64().unwrap();
    assert_eq!(b.kv_read, kvpt * n_kv as u64);
    assert_eq!(b.kv_write, kvpt);
    // the tensor table's own sum agrees with the per-tensor bytes model_inspect printed
    for (t, r) in h.tensors.iter().zip(fx["tensors"].as_array().unwrap()) {
        assert_eq!(t.bytes(), r[4].as_u64().unwrap());
    }
    eprintln!(
        "{key}: weights {:.4} GB + state {:.4} + KV@2K {:.4} + embd row {} B = {:.4} GB/token; {} barriers/token",
        b.weights() as f64 / 1e9,
        b.state_rw as f64 / 1e9,
        (b.kv_read + b.kv_write) as f64 / 1e9,
        b.embedding_row,
        b.total() as f64 / 1e9,
        plan.barriers_per_token()
    );
}

fn rng_bytes(n: usize, seed: u64) -> Vec<u8> {
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


fn fix_scales(q: QType, v: &mut [u8]) {
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

#[test]
fn ornith_byte_accounting_matches_model_inspect() {
    check_budget("ornith");
}

#[test]
fn qwen36_byte_accounting_matches_model_inspect() {
    check_budget("qwen36");
}

#[test]
fn barrier_budget_and_partitions() {
    let fx = fixture("ornith");
    let h = header(&fx);
    let desc = ModelDesc::from_header(&h).unwrap();
    assert_eq!((desc.n_rot, desc.rope_sections), (64, [11, 11, 10, 0]));
    for n in [1, 2, 4, 6, 8, 12] {
        let plan = Plan::build(&h, &desc, n, 4096).unwrap();
        plan.check_coverage().unwrap_or_else(|e| panic!("{n} threads: {e}"));
        // 30 x 5 + 10 x 6 + 1 (head join) = 211, independent of the thread count
        assert_eq!(plan.barriers_per_token(), 30 * 5 + 10 * 6 + 1);
    }
    let plan = Plan::build(&h, &desc, 8, 2048).unwrap();
    let per_layer = |l: usize| plan.phases.iter().filter(|p| p.layer == l && p.barrier_after).count();
    assert_eq!(per_layer(0), 5); // delta-net: mix, out, router, gate/up, down
    assert_eq!(per_layer(3), 6); // attention: qkv, core, out, router, gate/up, down
    assert_eq!(plan.phases.last().unwrap().kind, PhaseKind::Head);
}

#[test]
fn kernel_set_covers_both_models() {
    for key in ["ornith", "qwen36"] {
        let h = header(&fixture(key));
        let desc = ModelDesc::from_header(&h).unwrap();
        let plan = Plan::build(&h, &desc, 8, 256).unwrap();
        WeightStore::check_supported(&plan, &h.tensors).unwrap_or_else(|e| panic!("{key}: {e}"));
    }
}

#[test]
fn real_models_first_row_group_repack_matches() {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let mut seen = std::collections::HashSet::new();
    for f in ["Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf", "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"] {
        let p = home.join(".local/share/ai-stack/models/llm").join(f);
        if !p.is_file() {
            eprintln!("skip: no {f}");
            continue;
        }
        let g = gguf::Gguf::open(&p).unwrap();
        for t in &g.header.tensors {
            let Some(q) = crate::desc::qtype(t.ty) else { continue };
            if t.shape.len() < 2 || !seen.insert(q) {
                continue;
            }
            let k = t.shape[0] as usize;
            let rows = q.nr();
            let n = rows * q.row_bytes(k);
            let src = &g.tensor_data(t)[..n];
            let mut dst = vec![0u8; n];
            kernels_sys::repack(q, src, &mut dst, rows, k);
            let x: Vec<f32> = (0..k).map(|i| ((i * 13 % 37) as f32 - 18.0) / 18.0).collect();
            let mut a = vec![0u8; q.act().bytes(k)];
            quantize_act(q.act(), &x, &mut a);
            let mut out = vec![0f32; rows];
            kernels_sys::gemv(q, k, &mut out, &dst, &a);
            for r in 0..rows {
                let want = vec_dot_generic(q, k, &src[r * q.row_bytes(k)..], &a);
                assert!(want.is_finite());
                assert!((out[r] - want).abs() <= 1e-4 * want.abs().max(1.0), "{} {} row {r}: {} vs {want}", t.name, q.name(), out[r]);
            }
            eprintln!("{f}: {} ({}): first {rows} rows OK", t.name, q.name());
        }
    }
}

// ------------------------------------------------------------------ mini GGUF end to end

struct GgufWriter {
    kv: Vec<u8>,
    n_kv: u64,
    tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)>,
}

impl GgufWriter {
    fn key(&mut self, k: &str, vtype: u32) {
        self.kv.extend((k.len() as u64).to_le_bytes());
        self.kv.extend(k.as_bytes());
        self.kv.extend(vtype.to_le_bytes());
        self.n_kv += 1;
    }
    fn u32(&mut self, k: &str, v: u32) {
        self.key(k, 4);
        self.kv.extend(v.to_le_bytes());
    }
    fn f32(&mut self, k: &str, v: f32) {
        self.key(k, 6);
        self.kv.extend(v.to_le_bytes());
    }
    fn arr_i32(&mut self, k: &str, v: &[i32]) {
        self.key(k, 9);
        self.kv.extend(5u32.to_le_bytes());
        self.kv.extend((v.len() as u64).to_le_bytes());
        v.iter().for_each(|x| self.kv.extend(x.to_le_bytes()));
    }
    fn str(&mut self, k: &str, v: &str) {
        self.key(k, 8);
        self.kv.extend((v.len() as u64).to_le_bytes());
        self.kv.extend(v.as_bytes());
    }
    fn tensor(&mut self, name: &str, shape: &[u64], ty: GgmlType, seed: u64) {
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
    fn write(&self, path: &std::path::Path) {
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
fn mini_model(path: &std::path::Path) {
    mini_model_ext(path, false)
}

/// The mini model (GQA: 4 query heads over 2 KV heads), optionally with an MTP block `blk.4` (attention + MoE with Q8_0 experts
/// like Ornith's head, BF16 router like Qwen3.6's, nextn eh_proj/enorm/hnorm/shared_head_norm).
fn mini_model_ext(path: &std::path::Path, mtp: bool) {
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
    w.u32(&format!("{a}.context_length"), 4096);
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


fn mini_paths(tag: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("oxe-mini-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mini.gguf");
    mini_model(&path);
    (dir, path)
}

fn mini_opts(dir: &std::path::Path) -> LoadOptions {
    let cache = dir.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::set_permissions(&cache, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
    LoadOptions { backing: Backing::File, verify: Verify::Full, threads: 2, cache_dir: Some(cache), progress: false, thp_collapse: false, mlock: false, sha_from_env: true }
}

/// The weight cache: built once (arena == reference repack, byte for byte), hit on the
/// second load, rejected and rebuilt when corrupted; anon backing gives the same bytes.
#[test]
fn mini_gguf_weight_cache() {
    let (dir, path) = mini_paths("cache");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let g = gguf::Gguf::open(&path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &desc, 2, 256).unwrap();
    let g2 = gguf::Gguf::open(&path).unwrap();
    let ws = WeightStore::load(g, &plan, &opt).unwrap();
    assert!(!ws.info.cache_hit);
    for id in plan.tensors() {
        let t = &g2.header.tensors[id];
        let r = ws.get(id);
        let src = g2.tensor_data(t);
        // SAFETY: the tensor's arena range (read-only after load).
        let got = unsafe { std::slice::from_raw_parts(r.ptr as *const u8, r.bytes) };
        let mut want = vec![0u8; r.bytes];
        match r.layout {
            Layout::F32 | Layout::Raw(_) => want.copy_from_slice(src),
            Layout::Packed(q) => {
                let eb = r.expert_bytes();
                for e in 0..r.n_expert {
                    kernels_sys::repack(q, &src[e * eb..(e + 1) * eb], &mut want[e * eb..(e + 1) * eb], r.rows, r.k);
                }
            }
        }
        assert!(got == &want[..], "{}: arena differs from the reference repack", t.name);
    }
    let cache_path = ws.info.cache_path.clone();
    let data_off = ws.data_off as u64;
    drop(ws);
    let ws = WeightStore::load(gguf::Gguf::open(&path).unwrap(), &plan, &opt).unwrap();
    assert!(ws.info.cache_hit);
    drop(ws);
    let mut anon = opt.clone();
    anon.backing = Backing::Anon;
    let wa = WeightStore::load(gguf::Gguf::open(&path).unwrap(), &plan, &anon).unwrap();
    assert!(wa.info.cache_hit && wa.info.backing == "anon");
    drop(wa);
    // corrupt one byte in the data region: the full verification must reject it and rebuild
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(&cache_path).unwrap();
        f.seek(SeekFrom::Start(data_off + 100)).unwrap();
        f.write_all(&[0xA5]).unwrap();
    }
    let r = WeightStore::load(gguf::Gguf::open(&path).unwrap(), &plan, &opt);
    assert!(r.is_err(), "corruption must be detected");
    std::fs::remove_dir_all(&dir).ok();
}

/// CPUs for a test's pinned spin-barrier pool: a fresh run of cores per pool, from the top of
/// the cpu list. Two bugs of the old `(0..n).map(|i| i % available_parallelism())` (2026-11-10,
/// found by Codex on the merged candidate): `sched::Pool::new` pins the CALLER to `cpus[0]`, and
/// `available_parallelism()` reports the calling thread's affinity mask, so from a test's second
/// pool on it returned 1 and every worker of the later thread-count arms was pinned onto one CPU
/// (spin barriers on one core: the mini tests took minutes instead of well under a second per
/// arm). And every pool of every test started at CPU 0. Here the CPU count is read once, before
/// any pool of this process has pinned anything.
fn test_cpus(n: usize) -> Vec<usize> {
    static AVAIL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    // first use in the process: a test thread asks for its cpus before its first pool pins it
    let avail = *AVAIL.get_or_init(|| std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1));
    let base = NEXT.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    (0..n).map(|i| avail - 1 - (base + i) % avail).collect()
}

fn run_mini(path: &std::path::Path, opt: &LoadOptions, n: usize, steps: usize) -> (Vec<u32>, Vec<Vec<f32>>, u64) {
    run_mini_attn(path, opt, n, steps, false)
}

fn run_mini_attn(path: &std::path::Path, opt: &LoadOptions, n: usize, steps: usize, legacy: bool) -> (Vec<u32>, Vec<Vec<f32>>, u64) {
    // Pool pins its calling thread to worker 0. Isolate each run so the next
    // thread-count arm inherits the test runner's full CPU affinity, not CPU 0.
    std::thread::scope(|scope| scope.spawn(move || {
        let g = gguf::Gguf::open(path).unwrap();
        let desc = ModelDesc::from_header(&g.header).unwrap();
        let plan = Plan::build(&g.header, &desc, n, 512).unwrap();
        plan.check_coverage().unwrap();
        let ws = WeightStore::load(g, &plan, opt).unwrap();
        let cpus = test_cpus(n); // disjoint cores (moe-prefill 8db5bc2) + scoped thread (0b16801)
        let pool = sched::Pool::new(&cpus, Duration::from_micros(100)).unwrap();
        let mut eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false }).unwrap();
        eng.set_attn_legacy(legacy);
        eng.set_attn_tiled_min_kv(0); // the mini runs are short: always tiled (unless legacy)
        let mut ids = Vec::new();
        let mut logits = Vec::new();
        let b0 = pool.barrier_episodes();
        let mut tok = 1u32;
        for s in 0..steps {
            // a fixed prompt for the first half, then greedy
            let next = eng.step(&pool, tok);
            tok = if s < steps / 2 { (s as u32 * 7 + 3) % 64 } else { next };
            ids.push(next);
            logits.push(eng.logits().to_vec());
        }
        let barriers = (pool.barrier_episodes() - b0) / steps as u64;
        (ids, logits, barriers)
    }).join().unwrap())
}

/// End-to-end decode of the mini model: every thread count gives bit-identical logits and
/// ids (the partitioning must not change any result), the logits are finite and argmax is
/// consistent, and the barrier count matches the plan.
#[test]
fn mini_gguf_decode_is_thread_count_invariant() {
    let (dir, path) = mini_paths("decode");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (ids1, lg1, b1) = run_mini(&path, &opt, 1, 12);
    // plan barriers + 2 inside the (one) tiled attention phase
    assert_eq!(b1, 3 * 5 + 6 + 1 + 2);
    for lg in &lg1 {
        assert!(lg.iter().all(|v| v.is_finite()));
    }
    for (i, lg) in lg1.iter().enumerate() {
        let mut best = 0;
        for (j, &v) in lg.iter().enumerate() {
            if v > lg[best] {
                best = j;
            }
        }
        assert_eq!(ids1[i] as usize, best);
    }
    for n in [2usize, 3, 4] {
        let (ids, lg, b) = run_mini(&path, &opt, n, 12);
        assert_eq!(b, b1, "{n} threads: barriers");
        assert_eq!(ids, ids1, "{n} threads: ids differ");
        for (s, (a, b)) in lg.iter().zip(&lg1).enumerate() {
            let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "{n} threads: logits of step {s} differ from 1 thread");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// M3-ATTN: the tiled attention (split over KV positions and V^T rows, GQA-shared rows, softmax
/// on round_up(pos+1, 4) entries, V^T p cut at round_up(pos+1, 32) + tail rule) gives the same
/// logits bit for bit as the per-head M1 path, over 300 steps (n_kv 256 -> 512, all 4-/32-
/// boundary classes) at 1, 3 and 4 threads.
#[test]
fn mini_gguf_tiled_attention_equals_legacy() {
    let (dir, path) = mini_paths("attn");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (ids0, lg0, _) = run_mini_attn(&path, &opt, 2, 300, true);
    for n in [1usize, 3, 4] {
        let (ids, lg, _) = run_mini_attn(&path, &opt, n, 300, false);
        assert_eq!(ids, ids0, "{n} threads: ids differ");
        for (s, (a, b)) in lg.iter().zip(&lg0).enumerate() {
            let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "{n} threads: logits of step {s} differ from the legacy path");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Engine audit F4: the chunked anonymous (huge-page advised) arena decodes bit-identically to the
/// file-backed one; --mlock either locks (then identical too) or fails with an RLIMIT_MEMLOCK message.
#[test]
fn mini_gguf_anon_chunked_load_is_bit_identical() {
    let (dir, path) = mini_paths("anonload");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (ids0, lg0, _) = run_mini_attn(&path, &opt, 2, 120, false);
    let mut anon = opt.clone();
    anon.backing = Backing::Anon;
    let (ids, lg, _) = run_mini_attn(&path, &anon, 2, 120, false);
    assert_eq!(ids, ids0);
    assert!(lg.iter().zip(&lg0).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())), "anon arena logits differ");
    let mut locked = anon.clone();
    locked.mlock = true;
    let g = gguf::Gguf::open(&path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &desc, 2, 512).unwrap();
    match WeightStore::load(g, &plan, &locked) {
        Ok(ws) => assert!(ws.info.backing.ends_with("+mlock")),
        Err(e) => assert!(e.contains("RLIMIT_MEMLOCK"), "unexpected mlock error: {e}"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The by-size ubatch group (MoE candidate 2 default): below T1 one ubatch per pass, from T1 two,
/// from T2 four; T2 = 0 never asks for four.
#[test]
fn prefill_group_by_size_thresholds() {
    use crate::engine::group_by_size;
    assert_eq!([0, 24, 99].map(|n| group_by_size(n, 100, 0)), [1, 1, 1]);
    assert_eq!([100, 600, 1 << 20].map(|n| group_by_size(n, 100, 0)), [2, 2, 2]);
    assert_eq!([99, 100, 1023, 1024, 5000].map(|n| group_by_size(n, 100, 1024)), [1, 2, 2, 4, 4]);
    assert_eq!(group_by_size(50, 0, 0), 2);
    assert_eq!(group_by_size(50, 300, 40), 4);
}

/// OXE_DEC_ATTN_PAD=1 (decode attention scratch columns off the power-of-two n_ctx stride) and
/// OXE_ATTN_STATS=1 change no bit: tiled decode at n_ctx 512 (a power of two), 1-3 threads.
/// (Other tests constructing engines meanwhile may see the variables: harmless, the results are
/// bit-identical by design, which is what this test checks.)
#[test]
fn mini_gguf_decode_attn_pad_and_stats_are_bit_identical() {
    let (dir, path) = mini_paths("attnpad");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    std::env::set_var("OXE_DEC_ATTN_PAD", "0"); // the reference: unpadded scratch (the pad is the default)
    let (ids0, lg0, _) = run_mini_attn(&path, &opt, 2, 300, false);
    std::env::set_var("OXE_DEC_ATTN_PAD", "1");
    std::env::set_var("OXE_ATTN_STATS", "1");
    for n in [1usize, 3] {
        let (ids, lg, _) = run_mini_attn(&path, &opt, n, 300, false);
        assert_eq!(ids, ids0, "{n} threads: ids differ with the padded scratch");
        for (s, (a, b)) in lg.iter().zip(&lg0).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()), "{n} threads: logits of step {s} differ");
        }
    }
    std::env::remove_var("OXE_DEC_ATTN_PAD");
    std::env::remove_var("OXE_ATTN_STATS");
    std::fs::remove_dir_all(&dir).ok();
}

/// Snapshot / restore of the recurrent state: decode 10 tokens recording logits; restore the
/// snapshot taken after token 4 and replay tokens 4..10: every logit row is bit-identical.
#[test]
fn mini_gguf_snapshot_restore_replays_exactly() {
    let (dir, path) = mini_paths("snap");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let g = gguf::Gguf::open(&path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &desc, 2, 512).unwrap();
    let ws = WeightStore::load(g, &plan, &opt).unwrap();
    let pool = sched::DetachedPool::new(&test_cpus(2), Duration::from_micros(50)).unwrap();
    let eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false }).unwrap();
    let toks: Vec<u32> = (0..10).map(|i| (i * 11 + 5) % 64).collect();
    let mut want = Vec::new();
    let mut snap = None;
    for (i, &t) in toks.iter().enumerate() {
        if i == 4 {
            snap = Some(eng.save_recurrent());
        }
        eng.step(&pool, t);
        want.push(eng.logits().to_vec());
    }
    let s = snap.unwrap();
    assert_eq!(s.pos, 4);
    eng.restore_recurrent(&s);
    assert_eq!(eng.pos(), 4);
    for (i, &t) in toks.iter().enumerate().skip(4) {
        eng.step(&pool, t);
        assert!(eng.logits().iter().zip(&want[i]).all(|(a, b)| a.to_bits() == b.to_bits()), "step {i} differs after restore");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Batched prefill on the mini model.
fn mini_engine_parts(path: &std::path::Path, opt: &LoadOptions, n: usize) -> (Plan, WeightStore, sched::Pool) {
    let g = gguf::Gguf::open(path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &desc, n, 512).unwrap();
    let ws = WeightStore::load(g, &plan, opt).unwrap();
    let cpus = test_cpus(n);
    let pool = sched::Pool::new(&cpus, Duration::from_micros(100)).unwrap();
    (plan, ws, pool)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_state(a: &crate::engine::StateDump, b: &crate::engine::StateDump) -> bool {
    a.n_pos == b.n_pos
        && a.k == b.k
        && a.v == b.v
        && a.conv.iter().zip(&b.conv).all(|(x, y)| same_bits(x, y))
        && a.ssm.iter().zip(&b.ssm).all(|(x, y)| same_bits(x, y))
}

/// With fewer than 4 tokens every projection is a GEMV on from_float rows and each token's
/// attention is the decode kernel, exactly as llama.cpp's graph for n_tokens < 4; so a
/// prefill of 1..3 tokens must be bit-identical to the same tokens decoded one by one
/// (logits, KV and recurrent state), also when it starts mid-sequence.
#[test]
fn mini_gguf_small_prefill_equals_decode() {
    let (dir, path) = mini_paths("pfsmall");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 3);
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let dec = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    let pf = Engine::new(&plan, &ws, cfg).unwrap();
    let toks: Vec<u32> = vec![5, 17, 33, 2, 60, 9];
    // [1 token] [2 tokens] [3 tokens]
    let mut i = 0;
    for len in [1usize, 2, 3] {
        for &t in &toks[i..i + len] {
            dec.step(&pool, t);
        }
        let am = pf.prefill(&pool, &toks[i..i + len], crate::engine::UBATCH).unwrap();
        i += len;
        assert!(same_bits(dec.logits(), pf.logits()), "prefill of {len} tokens: logits differ from decode");
        assert!(same_state(&dec.dump_state(), &pf.dump_state()), "prefill of {len} tokens: state differs from decode");
        let mut best = 0;
        for (j, &v) in pf.logits().iter().enumerate() {
            if v > pf.logits()[best] {
                best = j;
            }
        }
        assert_eq!(am as usize, best);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Batched prefill over several ubatches (GEMM rows, MUL_MAT_ID tiles, remainders) gives
/// bit-identical logits and state at 1, 2, 3 and 4 threads, and decode continues identically.
#[test]
fn mini_gguf_prefill_is_thread_count_invariant() {
    let (dir, path) = mini_paths("pfthreads");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..23).map(|i| (i * 13 + 7) % 64).collect();
    let mut want: Option<(Vec<f32>, crate::engine::StateDump, Vec<Vec<f32>>)> = None;
    for n in [1usize, 2, 3, 4] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        let eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false }).unwrap();
        // ubatches of 9: 9 + 9 + 5 tokens (4-row groups + remainders, p0 > 0)
        let mut next = eng.prefill(&pool, &toks, 9).unwrap();
        assert_eq!(eng.pos(), toks.len());
        let lg = eng.logits().to_vec();
        assert!(lg.iter().all(|v| v.is_finite()));
        let st = eng.dump_state();
        let mut dec = Vec::new();
        for _ in 0..3 {
            next = eng.step(&pool, next);
            dec.push(eng.logits().to_vec());
        }
        match &want {
            None => want = Some((lg, st, dec)),
            Some((l1, s1, d1)) => {
                assert!(same_bits(&lg, l1), "{n} threads: prefill logits differ");
                assert!(same_state(&st, s1), "{n} threads: prefill state differs");
                for (a, b) in dec.iter().zip(d1) {
                    assert!(same_bits(a, b), "{n} threads: decode after prefill differs");
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// M3-ATTN: tiled prefill attention == the per-(token, head) path: a 700-token prompt in
/// ubatches of 256 (n_kv 256 / 512 / 768, partial last ubatch), then 40 decode steps across the
/// 768 boundary; logits, state and decode rows bit-identical at 1 and 3 threads.
#[test]
fn mini_gguf_tiled_prefill_equals_legacy() {
    let (dir, path) = mini_paths("pfattn");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..700).map(|i| (i * 13 + 7) % 64).collect();
    let mut want: Option<(Vec<f32>, crate::engine::StateDump, Vec<Vec<f32>>)> = None;
    for (n, legacy) in [(2usize, true), (1, false), (3, false)] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        let mut eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 1024, logits: true, trace: false, profile: false }).unwrap();
        eng.set_attn_legacy(legacy);
        eng.set_attn_tiled_min_kv(0);
        let mut next = eng.prefill(&pool, &toks, 256).unwrap();
        let lg = eng.logits().to_vec();
        let st = eng.dump_state();
        let mut dec = Vec::new();
        for _ in 0..40 {
            next = eng.step(&pool, next);
            dec.push(eng.logits().to_vec());
        }
        match &want {
            None => want = Some((lg, st, dec)),
            Some((l1, s1, d1)) => {
                assert!(same_bits(&lg, l1), "{n} threads: prefill logits differ from legacy");
                assert!(same_state(&st, s1), "{n} threads: prefill state differs from legacy");
                for (i, (a, b)) in dec.iter().zip(d1).enumerate() {
                    assert!(same_bits(a, b), "{n} threads: decode row {i} after prefill differs from legacy");
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// moe-prefill: every kernel switch (`OXE_PF_TILES` bits: dense tile GEMM, position-blocked
/// attention, padded attention columns, GEMV for single leftover expert rows) and the profiler
/// leave the prefill bit-identical: a 700-token prompt in ubatches of 256 plus a 37-token
/// extension (odd 4-row groups, GEMV leftovers), then 12 decode steps.
#[test]
fn mini_gguf_prefill_kernel_switches_are_bit_identical() {
    let (dir, path) = mini_paths("pfswitch");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    std::env::set_var("OXE_PF_MMID_MIN", "1"); // every expert through the pair tiles, incl. 1-row ones
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..700).map(|i| (i * 13 + 7) % 64).collect();
    let ext: Vec<u32> = (0..37).map(|i| (i * 29 + 3) % 64).collect();
    let mut want: Option<(Vec<f32>, crate::engine::StateDump, Vec<Vec<f32>>)> = None;
    for (n, mask, profile) in [(2usize, 0u32, false), (2, 1 << 30, false), (2, 1 << 29, false), (3, 1 << 28, false), (2, 4095 | 1 << 29, false), (3, 4095 | 1 << 28 | 1 << 30, false), (3, 511 | 1 << 30, false), (2, 1, false), (3, 2, false), (2, 4, false), (3, 8, false), (2, 16, false), (3, 24, false), (2, 32, false), (3, 38, false), (2, 64, false), (3, 128, false), (2, 256, false), (3, 256 | 16, false), (2, 512, false), (3, 512 | 128 | 64, false), (2, 1024, false), (3, 1024 | 512, false), (2, 2048, false), (3, 2048 | 512, false), (2, 4095, true), (1, 4095, false)] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        // bit 30 of the test's mask: V^T rows 32 elements off the n_ctx stride (OXE_KV_VPAD; decode too)
        crate::engine::set_kv_vpad(if mask & (1 << 30) != 0 { 32 } else { 0 });
        // bits 29 / 28 of the test's mask: 2 / 4 llama.cpp ubatches per pass (OXE_PF_GROUP)
        let group = if mask & (1 << 29) != 0 { 2 } else if mask & (1 << 28) != 0 { 4 } else { 1 };
        let mask = mask & 0xffff;
        let mut eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 1024, logits: true, trace: false, profile }).unwrap();
        crate::engine::set_kv_vpad(0);
        eng.set_attn_tiled_min_kv(0);
        crate::engine::set_pf_tiles(mask);
        crate::engine::set_pf_group(group);
        eng.prefill(&pool, &toks, 256).unwrap();
        let mut next = eng.prefill(&pool, &ext, 256).unwrap();
        crate::engine::set_pf_tiles(0);
        crate::engine::set_pf_group(1);
        let lg = eng.logits().to_vec();
        let st = eng.dump_state();
        if profile {
            let p = eng.prefill_profile().expect("profile");
            assert_eq!(p.threads, n);
            assert!(p.work_s.iter().sum::<f64>() > 0.0);
        }
        let mut dec = Vec::new();
        for _ in 0..12 {
            next = eng.step(&pool, next);
            dec.push(eng.logits().to_vec());
        }
        match &want {
            None => want = Some((lg, st, dec)),
            Some((l1, s1, d1)) => {
                assert!(same_bits(&lg, l1), "mask {mask}, {n} threads: prefill logits differ");
                assert!(same_state(&st, s1), "mask {mask}, {n} threads: prefill state differs");
                for (i, (a, b)) in dec.iter().zip(d1).enumerate() {
                    assert!(same_bits(a, b), "mask {mask}, {n} threads: decode row {i} after prefill differs");
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// F92: a prefill call that starts OFF a multiple of 256 (a server turn continuing after a decoded
/// reply) run with 2 / 4 ubatches per pass must give the bits of one ubatch at a time: logits,
/// state and the decode rows after it. Prefixes 37 / 300 / 129 tokens, calls of 600 and 300 tokens
/// (several full ubatches + a tail), 2 and 3 threads, masks 1021 / 4095, V pad 0 / 32.
#[test]
fn f92_unaligned_group_is_bit_identical() {
    let (dir, path) = mini_paths("f92");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let mut fails = Vec::new();
    for &(pre, len) in &[(37usize, 600usize), (300, 600), (129, 300), (300, 300), (511, 400)] {
        let a: Vec<u32> = (0..pre as u32).map(|i| (i * 13 + 7) % 64).collect();
        let b: Vec<u32> = (0..len as u32).map(|i| (i * 29 + 3) % 64).collect();
        for &(n, mask, vpad) in &[(2usize, 1021u32, 0usize), (3, 1021, 32), (2, 4095, 32), (3, 4095, 0)] {
            let mut want: Option<(Vec<f32>, crate::engine::StateDump, Vec<Vec<f32>>)> = None;
            for group in [1usize, 2, 4] {
                let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
                crate::engine::set_kv_vpad(vpad);
                let eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 1024, logits: true, trace: false, profile: false }).unwrap();
                crate::engine::set_kv_vpad(0);
                let mut eng = eng;
                eng.set_attn_tiled_min_kv(0);
                crate::engine::set_pf_tiles(mask);
                crate::engine::set_pf_group(1);
                eng.prefill(&pool, &a, 256).unwrap();
                crate::engine::set_pf_group(group);
                crate::engine::set_pf_group_unaligned(true); // opt-in (default off on release main); exact, so other tests may share it
                let mut next = eng.prefill(&pool, &b, 256).unwrap();
                // the grouped path really ran: the scratch grew to the group's pass size
                let rows = eng.prefill_scratch_rows();
                assert!(rows >= (256 * group).min(len), "pre {pre} len {len} group {group}: grouping did not apply ({rows} scratch rows)");
                crate::engine::set_pf_group(1);
                crate::engine::set_pf_tiles(0);
                let lg = eng.logits().to_vec();
                let st = eng.dump_state();
                let mut dec = Vec::new();
                for _ in 0..8 {
                    next = eng.step(&pool, next);
                    dec.push(eng.logits().to_vec());
                }
                match &want {
                    None => want = Some((lg, st, dec)),
                    Some((l1, s1, d1)) => {
                        let tag = format!("pre {pre} len {len} threads {n} mask {mask} vpad {vpad} group {group}");
                        if !same_bits(&lg, l1) {
                            fails.push(format!("{tag}: logits"));
                        }
                        if !same_state(&st, s1) {
                            fails.push(format!("{tag}: state"));
                        }
                        if dec.iter().zip(d1).any(|(x, y)| !same_bits(x, y)) {
                            fails.push(format!("{tag}: decode rows"));
                        }
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
    assert!(fails.is_empty(), "unaligned grouping is not bitwise:\n{}", fails.join("\n"));
}

/// Not a check: writes the mini model to `$OXE_MINI_OUT` so the CLI can be exercised model-free
/// (`OXE_MINI_OUT=/path/mini.gguf cargo test -p model -- --ignored mini_gguf_write_for_cli`, then
/// e.g. `OXE_MODEL_SHA256=000..0 oxe bench --model /path/mini.gguf --arms ...`).
#[test]
#[ignore]
fn mini_gguf_write_for_cli() {
    let out = std::env::var("OXE_MINI_OUT").expect("set OXE_MINI_OUT");
    mini_model(std::path::Path::new(&out));
}

/// Layout changes (V^T row pad) must not leak into reset, park (write_state) or restore
/// (read_state, restore_recurrent): after a long prompt and a reset, a shorter prompt gives the
/// state and decode rows of a fresh engine; a state written with one V pad loads into an engine
/// with another and decodes identically; a recurrent restore replays identically.
#[test]
fn mini_gguf_reset_park_restore_with_v_pad() {
    let (dir, path) = mini_paths("pfreset");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let a: Vec<u32> = (0..700).map(|i| (i * 13 + 7) % 64).collect();
    let b: Vec<u32> = (0..300).map(|i| (i * 29 + 3) % 64).collect();
    let cfg = EngineConfig { n_ctx: 1024, logits: true, trace: false, profile: false };
    let decode = |eng: &Engine, pool: &sched::Pool, n: usize| -> Vec<Vec<f32>> {
        let mut tok = 5u32;
        (0..n)
            .map(|_| {
                tok = eng.step(pool, tok);
                eng.logits().to_vec()
            })
            .collect()
    };
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
    // reference: fresh engine, V pad 0 (the llama.cpp-like layout), prompt b, 8 decode steps
    crate::engine::set_kv_vpad(0);
    let fresh = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    fresh.prefill(&pool, &b, 256).unwrap();
    let st0 = fresh.dump_state();
    let mut parked = Vec::new();
    fresh.write_state(&mut parked).unwrap();
    let want = decode(&fresh, &pool, 8);
    for vpad in [0usize, 32, 96] {
        crate::engine::set_kv_vpad(vpad);
        let eng = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        // reset after a longer prompt
        eng.prefill(&pool, &a, 256).unwrap();
        eng.reset();
        eng.prefill(&pool, &b, 256).unwrap();
        assert!(same_state(&eng.dump_state(), &st0), "vpad {vpad}: state after reset + prefill differs from a fresh engine");
        let snap = eng.save_recurrent();
        let got = decode(&eng, &pool, 8);
        for (i, (x, y)) in got.iter().zip(&want).enumerate() {
            assert!(same_bits(x, y), "vpad {vpad}: decode row {i} after reset differs");
        }
        // recurrent restore to the end of the prompt, replay
        eng.restore_recurrent(&snap);
        let again = decode(&eng, &pool, 8);
        for (i, (x, y)) in again.iter().zip(&want).enumerate() {
            assert!(same_bits(x, y), "vpad {vpad}: decode row {i} after restore_recurrent differs");
        }
        // park / restore across layouts: the state written with V pad 0 loads into this layout
        eng.reset();
        eng.read_state(&mut &parked[..], b.len()).unwrap();
        assert!(same_state(&eng.dump_state(), &st0), "vpad {vpad}: state read back differs");
        let loaded = decode(&eng, &pool, 8);
        for (i, (x, y)) in loaded.iter().zip(&want).enumerate() {
            assert!(same_bits(x, y), "vpad {vpad}: decode row {i} after read_state differs");
        }
        // and this layout's own written state is byte-identical to the pad-0 one
        eng.reset();
        eng.read_state(&mut &parked[..], b.len()).unwrap();
        let mut again_parked = Vec::new();
        eng.write_state(&mut again_parked).unwrap();
        assert!(again_parked == parked, "vpad {vpad}: written state bytes differ between layouts");
    }
    crate::engine::set_kv_vpad(crate::engine::KV_VPAD_DEFAULT);
    std::fs::remove_dir_all(&dir).ok();
}

/// MP-5 (small-turn path): `prefill_state_only` leaves exactly the state of `prefill` (the next
/// decode steps are bit-identical), also when the short segment runs on another team than the
/// long one (two scratch slots), with the wait-diet switch on.
#[test]
fn mini_gguf_prefill_state_only_and_small_team() {
    let (dir, path) = mini_paths("pfsmall");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let head: Vec<u32> = (0..300).map(|i| (i * 13 + 7) % 64).collect();
    let small: Vec<u32> = (0..23).map(|i| (i * 29 + 3) % 64).collect();
    let mut want: Option<(crate::engine::StateDump, Vec<Vec<f32>>)> = None;
    for (state_only, other_team, mask) in [(false, false, 0u32), (true, false, 0), (true, true, 0), (true, true, 512 | 8), (false, true, 1023), (true, true, 4095)] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
        let big = sched::Pool::new(&test_cpus(3), Duration::from_micros(100)).unwrap();
        let eng = Engine::new(&plan, &ws, EngineConfig { n_ctx: 1024, logits: true, trace: false, profile: false }).unwrap();
        crate::engine::set_pf_tiles(mask);
        // the long segment on the "prefill team", the short one on the decode team (or the same)
        let (t_long, t_short): (&sched::Pool, &sched::Pool) = if other_team { (&big, &pool) } else { (&pool, &pool) };
        if state_only {
            eng.prefill_state_only(t_long, &head, 256);
            eng.prefill_state_only(t_short, &small, 256);
        } else {
            eng.prefill(t_long, &head, 256).unwrap();
            eng.prefill(t_short, &small, 256).unwrap();
        }
        crate::engine::set_pf_tiles(0);
        let st = eng.dump_state();
        let mut dec = Vec::new();
        let mut tok = 5u32;
        for _ in 0..6 {
            tok = eng.step(&pool, tok);
            dec.push(eng.logits().to_vec());
        }
        match &want {
            None => want = Some((st, dec)),
            Some((s1, d1)) => {
                assert!(same_state(&st, s1), "state_only {state_only} other_team {other_team} mask {mask}: state differs");
                for (i, (a, b)) in dec.iter().zip(d1).enumerate() {
                    assert!(same_bits(a, b), "state_only {state_only} other_team {other_team} mask {mask}: decode row {i} differs");
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// prefill_ext (h_nextn for MTP): every position's output-normed hidden state; the state is
/// the same as with the plain prefill, and the result is thread-count invariant.
#[test]
fn mini_gguf_prefill_hidden_states() {
    let (dir, path) = mini_paths("pfhidden");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..13).map(|i| (i * 5 + 3) % 64).collect();
    let mut first: Option<(Vec<f32>, Vec<f32>)> = None;
    for n in [1usize, 3] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
        let a = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        let b = Engine::new(&plan, &ws, cfg).unwrap();
        let mut h = Vec::new();
        a.prefill_ext(&pool, &toks, 6, Some(&mut h)).unwrap();
        b.prefill(&pool, &toks, 6).unwrap();
        assert_eq!(h.len(), toks.len() * plan.desc.n_embd);
        assert!(h.iter().all(|v| v.is_finite()));
        assert!(same_state(&a.dump_state(), &b.dump_state()), "hidden mode changed the state");
        let lg = a.logits().to_vec();
        match &first {
            None => first = Some((h, lg)),
            Some((h1, l1)) => {
                assert!(same_bits(&h, h1), "{n} threads: hidden states differ");
                assert!(same_bits(&lg, l1), "{n} threads: logits differ");
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------------------------------ M3.1: lanes and MTP

fn mtp_setup(tag: &str, nthreads: usize) -> (PathBuf, PathBuf, Plan, WeightStore, crate::mtp::MtpWeights, sched::Pool) {
    let dir = std::env::temp_dir().join(format!("oxe-mini-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mini-mtp.gguf");
    mini_model_ext(&path, true);
    std::env::set_var("OXE_MODEL_SHA256", "1".repeat(64));
    let opt = mini_opts(&dir);
    let g = gguf::Gguf::open(&path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    assert_eq!(desc.n_layer, 4);
    let plan = Plan::build(&g.header, &desc, nthreads, 512).unwrap();
    let ws = WeightStore::load(g, &plan, &opt).unwrap();
    let mw = crate::mtp::MtpWeights::load(&path, &plan, &opt).unwrap();
    let cpus = test_cpus(nthreads);
    let pool = sched::Pool::new(&cpus, Duration::from_micros(50)).unwrap();
    (dir, path, plan, ws, mw, pool)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// verify(n) == n one-token steps, bit for bit, for every lane; partial commits roll the
/// KV and the recurrent slots back exactly (rejected tails of random tokens), at 1-3 threads.
#[test]
fn mini_verify_lanes_are_batch_invariant_with_rollback() {
    for nt in [1usize, 2, 3] {
        let (dir, _path, plan, ws, mw, pool) = mtp_setup(&format!("lanes{nt}"), nt);
        let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
        let seq: Vec<u32> = (0..60).map(|i| (i * 13 + 7) % 64).collect();
        // reference: plain one-token steps
        let r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        let mut want = Vec::new();
        let mut want_h = Vec::new();
        for &t in &seq {
            r.step(&pool, t);
            want.push(r.logits().to_vec());
            want_h.push(r.hidden(0).to_vec());
        }
        drop(r);
        let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: 4, mtp: Some(&mw), kv_type: None }).unwrap();
        e.set_attn_tiled_min_kv(0); // MTP / verify lanes tiled, the plain reference per-head
        let mut p = 0usize;
        let mut it = 0u32;
        let mut commits = [0usize; 5];
        while p < seq.len() {
            let n = (1 + (it as usize * 7) % 4).min(seq.len() - p);
            let m = 1 + (it as usize * 5 + 3) % n; // tokens kept
            // lanes < m carry the true tokens; the rest random garbage
            let mut toks: Vec<u32> = seq[p..p + n].to_vec();
            for (i, t) in toks.iter_mut().enumerate().skip(m) {
                *t = (it * 31 + i as u32 * 17 + 1) % 64;
            }
            let g = e.verify(&pool, &toks);
            for ln in 0..m {
                assert!(bits_eq(e.logits_lane(ln), &want[p + ln]), "{nt} threads: lane {ln} at pos {} differs", p + ln);
                assert!(bits_eq(e.hidden(ln), &want_h[p + ln]), "{nt} threads: h of lane {ln} differs");
                let lg = e.logits_lane(ln);
                let mut best = 0;
                for (j, &v) in lg.iter().enumerate() {
                    if v > lg[best] {
                        best = j;
                    }
                }
                assert_eq!(g[ln] as usize, best);
            }
            // sometimes commit everything first and then shrink (the server's rewind)
            if it % 3 == 0 && n > m {
                e.commit(n);
                e.commit(m);
            } else {
                e.commit(m);
            }
            commits[m] += 1;
            p += m;
            it += 1;
        }
        assert!(commits[1] > 0 && commits[2] > 0 && commits[3] > 0, "{commits:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// OD-2: the fused 4-row delta-net step in decode (`OXE_DEC_FUSED_DN=1`) is bit-identical to the
/// default step: plain one-token steps, and verify lanes 1-4 with partial commits (rollback of
/// the recurrent slots and the KV), at 1-3 threads; logits and hidden states compared bit for bit.
#[test]
fn mini_decode_fused_delta_net_is_bit_identical() {
    for nt in [1usize, 2, 3] {
        let (dir, _path, plan, ws, mw, pool) = mtp_setup(&format!("fuseddn{nt}"), nt);
        let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
        let seq: Vec<u32> = (0..48).map(|i| (i * 11 + 5) % 64).collect();
        // reference: switch off, plain one-token steps
        let mut r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        r.set_dec_fused_dn(false);
        let mut want = Vec::new();
        let mut want_h = Vec::new();
        for &t in &seq {
            r.step(&pool, t);
            want.push(r.logits().to_vec());
            want_h.push(r.hidden(0).to_vec());
        }
        drop(r);
        // switch on, plain steps
        let mut f = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        f.set_dec_fused_dn(true);
        assert!(f.dec_fused_dn());
        for (i, &t) in seq.iter().enumerate() {
            f.step(&pool, t);
            assert!(bits_eq(f.logits(), &want[i]), "{nt} threads: fused step logits differ at {i}");
            assert!(bits_eq(f.hidden(0), &want_h[i]), "{nt} threads: fused step h differs at {i}");
        }
        drop(f);
        // switch on, MTP verify lanes with rejected (random) tails rolled back
        let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: 4, mtp: Some(&mw), kv_type: None }).unwrap();
        e.set_dec_fused_dn(true);
        let (mut p, mut it) = (0usize, 0u32);
        let mut commits = [0usize; 5];
        while p < seq.len() {
            let n = (1 + (it as usize * 7) % 4).min(seq.len() - p);
            let m = 1 + (it as usize * 5 + 3) % n;
            let mut toks: Vec<u32> = seq[p..p + n].to_vec();
            for (i, t) in toks.iter_mut().enumerate().skip(m) {
                *t = (it * 29 + i as u32 * 13 + 3) % 64;
            }
            e.verify(&pool, &toks);
            for ln in 0..m {
                assert!(bits_eq(e.logits_lane(ln), &want[p + ln]), "{nt} threads: fused lane {ln} at pos {} differs", p + ln);
                assert!(bits_eq(e.hidden(ln), &want_h[p + ln]), "{nt} threads: fused h of lane {ln} differs");
            }
            if it % 3 == 0 && n > m {
                e.commit(n);
                e.commit(m);
            } else {
                e.commit(m);
            }
            commits[m] += 1;
            p += m;
            it += 1;
        }
        assert!(commits[1] > 0 && commits[2] > 0, "{commits:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// OD-1: the decode profile (cfg.profile) changes no bits, counts one token per run, and fills
/// work, bytes and the delta-net sub-steps for plain steps and verify runs of 1-4 lanes.
#[test]
fn mini_decode_profile_is_bit_identical_and_counts() {
    use crate::plan::PhaseKind;
    let nt = 2;
    let (dir, _path, plan, ws, _mw, pool) = mtp_setup("prof", nt);
    let seq: Vec<u32> = (0..24).map(|i| (i * 7 + 2) % 64).collect();
    let mk = |profile: bool| {
        let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile };
        Engine::with_options(&plan, &ws, cfg, crate::engine::EngineOptions { verify_lanes: 4, mtp: None, kv_type: None }).unwrap()
    };
    let (a, b) = (mk(false), mk(true));
    let mut p = 0;
    let mut runs = 0u64;
    for n in [1usize, 2, 3, 4, 1, 2] {
        let toks = &seq[p..p + n];
        a.verify(&pool, toks);
        b.verify(&pool, toks);
        for ln in 0..n {
            assert!(bits_eq(a.logits_lane(ln), b.logits_lane(ln)), "profile changed lane {ln} at {p}");
        }
        a.commit(n);
        b.commit(n);
        p += n;
        runs += 1;
    }
    let pr = b.profile_sum();
    assert_eq!(pr.len(), nt);
    assert_eq!(pr[0].tokens, runs);
    let sum = |f: &dyn Fn(&crate::engine::Prof) -> u64| pr.iter().map(f).sum::<u64>();
    for k in [PhaseKind::RecMix, PhaseKind::AttnCore, PhaseKind::MoeGateUp, PhaseKind::MoeDown, PhaseKind::Head] {
        let i = k.index();
        assert!(sum(&|q| q.phase_work[i]) > 0, "{k:?} has no work");
        assert!(sum(&|q| q.phase_bytes[i]) > 0, "{k:?} has no bytes");
    }
    assert!(sum(&|q| q.sub_work[0]) > 0 && sum(&|q| q.sub_work[1]) > 0);
    let i = PhaseKind::RecMix.index();
    assert!(sum(&|q| q.sub_work[0] + q.sub_work[1]) <= sum(&|q| q.phase_work[i]));
    std::fs::remove_dir_all(&dir).ok();
}

/// Greedy decoding with the MTP driver (depth 1, 2, 3) gives the same token ids and the same
/// logit rows at every accepted position as plain greedy decoding; rewind after a step
/// continues exactly.
#[test]
fn mini_mtp_greedy_is_identical_to_plain() {
    let (dir, _path, plan, ws, mw, pool) = mtp_setup("mtpgreedy", 2);
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let prompt: Vec<u32> = vec![3, 17, 42, 5, 9, 33, 1];
    let n_gen = 48;
    let r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    let mut next = 0;
    for &t in &prompt {
        next = r.step(&pool, t);
    }
    let mut want_ids = vec![next];
    let mut want_rows = vec![r.logits().to_vec()];
    for _ in 1..n_gen {
        next = r.step(&pool, next);
        want_ids.push(next);
        want_rows.push(r.logits().to_vec());
    }
    drop(r);
    for depth in 1..=3 {
        let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: depth + 1, mtp: Some(&mw), kv_type: None }).unwrap();
        e.set_attn_tiled_min_kv(0); // MTP / verify lanes tiled, the plain reference per-head
        let mut d = crate::mtp::MtpDriver::new(&e, depth).unwrap();
        // the random mini MTP never guesses right: substitute the true greedy token at most
        // positions (and a wrong one at every 5th) so every acceptance length occurs
        let plen = prompt.len();
        let mk = |truth: Vec<u32>| -> Box<dyn FnMut(usize, u32) -> u32 + Send> {
            Box::new(move |pos, dr| {
                let i = pos - plen; // index into want_ids of the token at `pos`
                if pos % 5 == 0 || i >= truth.len() {
                    dr ^ 1
                } else {
                    truth[i]
                }
            })
        };
        d.draft_override = Some(mk(want_ids.clone()));
        let x0 = d.prefill(&e, &pool, &prompt);
        assert!(bits_eq(e.logits_lane((prompt.len() - 1) % (depth + 1)), &want_rows[0]), "depth {depth}: prefill logits differ");
        let mut ids = vec![x0];
        let mut out = Vec::new();
        let mut x = x0;
        while ids.len() < n_gen {
            let p = e.pos();
            d.step(&e, &pool, x, &mut out);
            for (i, &t) in out.iter().enumerate() {
                let row = p + i + 1 - prompt.len(); // logits after position p+i
                if row < want_rows.len() {
                    assert!(bits_eq(e.logits_lane(i), &want_rows[row]), "depth {depth}: logits at row {row} differ");
                }
                ids.push(t);
            }
            x = *out.last().unwrap();
        }
        ids.truncate(n_gen);
        assert_eq!(ids, want_ids, "depth {depth}: ids differ");
        eprintln!("mini mtp depth {depth}: {:?}", d.stats);
        assert!(d.stats.steps > 0 && d.stats.drafted >= d.stats.steps);
        assert!(d.stats.accepted_at[depth - 1] > 0 && d.stats.accepted < d.stats.drafted, "depth {depth}: acceptance paths not exercised");
        // rewind: the MTP-on decode stopped after keeping only one token of a step must
        // continue exactly like the plain decode
        e.reset();
        let mut d = crate::mtp::MtpDriver::new(&e, depth).unwrap();
        d.draft_override = Some(mk(want_ids.clone()));
        let mut x = d.prefill(&e, &pool, &prompt);
        let mut ids = vec![x];
        let mut rewound = 0;
        while ids.len() < n_gen {
            d.step(&e, &pool, x, &mut out);
            if out.len() > 1 {
                d.rewind(&e, 1);
                out.truncate(1);
                rewound += 1;
            }
            ids.extend_from_slice(&out);
            x = *out.last().unwrap();
        }
        ids.truncate(n_gen);
        assert_eq!(ids, want_ids, "depth {depth}: ids differ after rewinds");
        assert!(rewound > 0, "depth {depth}: no rewind exercised");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// WP-D3b restricted draft head: (a) a draft vocabulary of every id drafts exactly what the full
/// head drafts (same MTP stats, same ids); (b) a small vocabulary changes only the drafts: the
/// output stays the plain greedy sequence, and every draft token is in the vocabulary.
#[test]
fn mini_mtp_restricted_draft_head() {
    let (dir, path, plan, ws, mw_full, pool) = mtp_setup("draftvocab", 2);
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let prompt: Vec<u32> = vec![3, 17, 42, 5, 9, 33, 1];
    let n_gen = 40;
    let r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    let mut next = 0;
    for &t in &prompt {
        next = r.step(&pool, t);
    }
    let mut want = vec![next];
    for _ in 1..n_gen {
        next = r.step(&pool, next);
        want.push(next);
    }
    drop(r);
    let run = |mw: &crate::mtp::MtpWeights| {
        let e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: 3, mtp: Some(mw), kv_type: None }).unwrap();
        let mut d = crate::mtp::MtpDriver::new(&e, 2).unwrap();
        let drafted = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let dr2 = drafted.clone();
        d.draft_override = Some(Box::new(move |_pos, t| {
            dr2.lock().unwrap().push(t);
            t
        }));
        let mut x = d.prefill(&e, &pool, &prompt);
        let mut ids = vec![x];
        let mut out = Vec::new();
        while ids.len() < n_gen {
            d.step(&e, &pool, x, &mut out);
            ids.extend_from_slice(&out);
            x = *out.last().unwrap();
        }
        ids.truncate(n_gen);
        let v = drafted.lock().unwrap().clone();
        (ids, d.stats.clone(), v)
    };
    let (ids_full, st_full, dr_full) = run(&mw_full);
    assert_eq!(ids_full, want);
    let n_vocab = plan.desc.n_vocab as u32;
    // (a) every id
    let opt = mini_opts(&dir);
    let mut mw_all = crate::mtp::MtpWeights::load(&path, &plan, &opt).unwrap();
    mw_all.set_draft_vocab(&path, &(0..n_vocab).collect::<Vec<_>>()).unwrap();
    let (ids_all, st_all, dr_all) = run(&mw_all);
    assert_eq!(ids_all, want);
    assert_eq!(dr_all, dr_full, "a full draft vocabulary must draft exactly what the full head drafts");
    assert_eq!((st_all.drafted, st_all.accepted), (st_full.drafted, st_full.accepted));
    // (b) a small vocabulary (every 3rd id, unsorted input with duplicates and an out-of-range id)
    let mut small: Vec<u32> = (0..n_vocab).filter(|i| i % 3 == 0).rev().collect();
    small.push(small[0]);
    small.push(n_vocab + 5);
    let mut mw_small = crate::mtp::MtpWeights::load(&path, &plan, &opt).unwrap();
    mw_small.set_draft_vocab(&path, &small).unwrap();
    let (ids_small, _, dr_small) = run(&mw_small);
    assert_eq!(ids_small, want, "a restricted draft head must not change the output");
    assert!(!dr_small.is_empty() && dr_small.iter().all(|t| t % 3 == 0 && *t < n_vocab), "drafts outside the vocabulary");
    std::fs::remove_dir_all(&dir).ok();
}

/// Batched prefill with MTP seeding: `prefill_nextn` keeps `prefill`'s logits and state bit for
/// bit; an MTP engine that alternates batched prompts and speculative decode (recurrent slot
/// index rotated before the second prompt) matches the plain engine (prefill + steps).
#[test]
fn mini_mtp_with_batched_prefill_is_identical_to_plain() {
    let (dir, _path, plan, ws, mw, pool) = mtp_setup("mtpprefill", 2);
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let p1: Vec<u32> = (0..23).map(|i| (i * 11 + 3) % 64).collect();
    let p2: Vec<u32> = (0..17).map(|i| (i * 5 + 1) % 64).collect();
    let n_gen = 20;
    // reference: plain engine
    let r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    let mut want_ids = Vec::new();
    let mut want_rows = Vec::new();
    for p in [&p1, &p2] {
        let mut x = r.prefill(&pool, p, 8).unwrap();
        want_rows.push(r.logits().to_vec());
        want_ids.push(x);
        for _ in 1..n_gen {
            x = r.step(&pool, x);
            want_ids.push(x);
            want_rows.push(r.logits().to_vec());
        }
        r.step(&pool, x); // the last sampled token enters the state before the next prompt
    }
    let want_state = r.dump_state();
    drop(r);
    // prefill_nextn == prefill on a fresh engine
    let a = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    let mut h = Vec::new();
    a.prefill_nextn(&pool, &p1, 8, &mut h).unwrap();
    assert!(bits_eq(a.logits(), &want_rows[0]), "prefill_nextn logits differ from prefill");
    assert_eq!(h.len(), p1.len() * plan.desc.n_embd);
    drop(a);
    // position -> reference token (to force right drafts)
    let mut at = std::collections::HashMap::new();
    let mut pos = 0;
    for (pi, p) in [&p1, &p2].iter().enumerate() {
        pos += p.len();
        for j in 0..n_gen {
            at.insert(pos + j, want_ids[pi * n_gen + j]);
        }
        pos += n_gen;
    }
    for depth in 1..=3 {
        let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: depth + 1, mtp: Some(&mw), kv_type: None }).unwrap();
        e.set_attn_tiled_min_kv(0); // MTP / verify lanes tiled, the plain reference per-head
        let mut d = crate::mtp::MtpDriver::new(&e, depth).unwrap();
        let at = at.clone();
        d.draft_override = Some(Box::new(move |p, dr| if p % 5 == 0 { dr ^ 1 } else { at.get(&p).copied().unwrap_or(dr) }));
        let mut ids = Vec::new();
        let mut out = Vec::new();
        for (pi, p) in [&p1, &p2].iter().enumerate() {
            let base = pi * n_gen;
            let mut x = d.prefill_batched(&e, &pool, &pool, p, 8);
            assert!(bits_eq(e.logits(), &want_rows[base]), "depth {depth}: prompt {pi} logits differ");
            let mut got = vec![x];
            while got.len() < n_gen {
                let start = got.len();
                d.step(&e, &pool, x, &mut out);
                let keep = out.len().min(n_gen - start);
                for i in 0..keep {
                    assert!(bits_eq(e.logits_lane(i), &want_rows[base + start + i]), "depth {depth}: row {} of prompt {pi} differs", start + i);
                    got.push(out[i]);
                }
                if keep < out.len() {
                    d.rewind(&e, keep); // stop inside an accepted run, like the server
                }
                x = *got.last().unwrap();
            }
            let mut o = Vec::new();
            d.step_plain(&e, &pool, x, &mut o);
            ids.extend_from_slice(&got);
        }
        assert_eq!(ids, want_ids, "depth {depth}: ids differ");
        assert!(d.stats.accepted > 0, "depth {depth}: no acceptance");
        assert!(same_state(&e.dump_state(), &want_state), "depth {depth}: final state differs");
    }
    std::fs::remove_dir_all(&dir).ok();
}

// ------------------------------------------------------------------ O-13: int8 (Q8_0) KV cache

use crate::engine::{EngineOptions, KvType, PvMode};

/// An engine with KV type `kv` (logits on, no MTP, one verify lane).
fn kv_engine<'a>(plan: &'a Plan, ws: &'a WeightStore, n_ctx: usize, kv: KvType) -> Engine<'a> {
    let cfg = EngineConfig { n_ctx, logits: true, trace: false, profile: false };
    Engine::with_options(plan, ws, cfg, EngineOptions { verify_lanes: 0, mtp: None, kv_type: Some(kv) }).unwrap()
}

/// Straightforward f32 attention of one query head over K / V rows [position][dim].
fn ref_attn_f32(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, scale: f32, out: &mut [f32]) {
    let mut s = vec![0f32; n];
    let mut m = f32::NEG_INFINITY;
    for p in 0..n {
        let mut a = 0f32;
        for j in 0..hd {
            a += q[j] * k[p * hd + j];
        }
        s[p] = a * scale;
        m = m.max(s[p]);
    }
    let mut sum = 0f32;
    for x in s.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    for j in 0..hd {
        let mut a = 0f32;
        for p in 0..n {
            a += s[p] * v[p * hd + j];
        }
        out[j] = a / sum;
    }
}

/// Stored KV bytes of every attention layer (main layers; plus the MTP layer when `mtp`).
fn kv_all(e: &Engine, n_att: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
    (0..n_att).map(|ai| e.kv_layer_bytes(ai)).collect()
}

fn argmax(lg: &[f32]) -> usize {
    let mut best = 0;
    for (j, &v) in lg.iter().enumerate() {
        if v > lg[best] {
            best = j;
        }
    }
    best
}

/// (2) int8 decode (k8 / v8 / kv8) over 300 steps (two V position blocks, n_kv 256 -> 512)
/// equals a straightforward f32 attention over the SAME stored (dequantised) K / V: the reference
/// engine runs the identical model with only the attention core replaced (the mini model's one
/// attention layer is the last, so both engines store the same K / V bytes). The gap to the
/// reference is the f16 arithmetic of the tiles (f16 q, f16 accumulation, f16 probabilities), so
/// the tolerance is the f16 cache's own gap to the same reference (tiled f16 engine vs reference
/// over its f16 values): an int8 read bug (wrong row, block or transpose) is far above it.
/// Greedy ids equal the reference's (a flip only at a near-tie below that noise).
#[test]
fn mini_int8_kv_decode_matches_f32_reference() {
    let (dir, path) = mini_paths("kv8ref");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
    let n_att = plan.desc.n_attention();
    let steps = 300;
    let mut base = 0f32; // the f16 cache's max relative gap to the f32 reference
    let mut f16_rows: Vec<Vec<f32>> = Vec::new();
    let arms = [(KvType::F16, true), (KvType::K8, true), (KvType::V8, true), (KvType::KV8, true), (KvType::K8, false), (KvType::V8, false), (KvType::KV8, false)];
    for (kv, fused) in arms {
        let mut e = kv_engine(&plan, &ws, 512, kv);
        let mut r = kv_engine(&plan, &ws, 512, kv);
        e.set_kv8_fused(fused);
        r.set_kv8_fused(fused);
        e.set_attn_tiled_min_kv(0);
        r.set_attn_tiled_min_kv(0);
        r.set_attn_ref(Some(ref_attn_f32 as crate::engine::AttnRef));
        assert_eq!(e.kv_type(), kv);
        if kv != KvType::F16 {
            e.set_attn_tiled_min_kv(4096);
            assert!(e.barriers_per_token() == r.barriers_per_token(), "int8 KV must take the tiled attention at every context");
        }
        let (mut worst, mut worst16, mut flips) = (0f32, 0f32, 0);
        let mut tok = 1u32;
        for s in 0..steps {
            let ge = e.step(&pool, tok);
            let gr = r.step(&pool, tok);
            let (a, b) = (e.logits(), r.logits());
            assert!(a.iter().all(|v| v.is_finite()), "{kv:?}: non-finite logits at step {s}");
            let sc = b.iter().fold(1f32, |m, x| m.max(x.abs()));
            let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / sc;
            worst = worst.max(diff);
            assert_eq!(ge as usize, argmax(a));
            if kv == KvType::F16 {
                f16_rows.push(a.to_vec());
            } else {
                let d16 = a.iter().zip(&f16_rows[s]).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / sc;
                worst16 = worst16.max(d16);
                assert!(diff <= 2.0 * base + 1e-3, "{kv:?} step {s}: logits differ from the f32 reference by {diff} (f16 cache: {base})");
            }
            if ge != gr {
                let gap = (b[ge as usize] - b[gr as usize]).abs() / sc;
                assert!(gap <= 2.0 * base.max(diff), "{kv:?} step {s}: greedy id differs from the reference beyond a near-tie (gap {gap})");
                flips += 1;
            }
            // the f16 run's greedy continuation drives every type (same tokens, comparable rows)
            tok = if s < steps / 2 { (s as u32 * 7 + 3) % 64 } else { argmax(&f16_rows[s]) as u32 };
        }
        assert!(kv_all(&e, n_att) == kv_all(&r, n_att), "{kv:?}: reference engine stored other KV bytes");
        if kv == KvType::F16 {
            base = worst;
        }
        eprintln!("KV {kv:?} fused={fused}: max relative logit diff vs f32 reference {worst:.2e} (near-tie flips {flips}), vs the f16 cache {worst16:.2e}");
        assert!(flips <= 1, "{kv:?}: {flips} greedy flips against the reference");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// (3) Write-time quantisation: the stored int8 K / V bytes are a function of the f32 rows only.
/// Token-by-token decode == prefill in pieces of 1-3 tokens (the GEMV path: the same f32 rows):
/// bytes, recurrent state, and the following decode rows bit-identical. Prefill at ubatch 7 / 64 /
/// 256: each int8 row is the quantisation of the row the f16 engine stores (within half a step),
/// and whenever two batchings give the same f16 cache they give the same int8 bytes and decode.
#[test]
fn mini_int8_kv_write_is_batch_invariant() {
    let (dir, path) = mini_paths("kv8write");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 3);
    let d = &plan.desc;
    let n_att = d.n_attention();
    let toks: Vec<u32> = (0..300).map(|i| (i * 13 + 7) % 64).collect();
    let tail = |e: &Engine, mut x: u32| -> Vec<Vec<f32>> {
        let mut v = Vec::new();
        for _ in 0..8 {
            x = e.step(&pool, x);
            v.push(e.logits().to_vec());
        }
        v
    };
    let same_rows = |a: &[Vec<f32>], b: &[Vec<f32>]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_bits(x, y));
    // token by token, both types
    let mut dec_runs = Vec::new();
    for kv in [KvType::F16, KvType::KV8] {
        let e = kv_engine(&plan, &ws, 1024, kv);
        let mut x = 0;
        for &t in &toks {
            x = e.step(&pool, t);
        }
        dec_runs.push((kv_all(&e, n_att), e.dump_state(), x, tail(&e, x)));
    }
    let (want_kv, want_rec, want_x, want_tail) = &dec_runs[1];
    // prefill in pieces of 1, 2, 3 tokens
    let pf = kv_engine(&plan, &ws, 1024, KvType::KV8);
    let (mut i, mut k, mut x) = (0usize, 0usize, 0u32);
    while i < toks.len() {
        let len = (1 + k % 3).min(toks.len() - i);
        x = pf.prefill(&pool, &toks[i..i + len], 256).unwrap();
        i += len;
        k += 1;
    }
    assert_eq!(x, *want_x);
    assert!(kv_all(&pf, n_att) == *want_kv, "small prefills: int8 KV bytes differ from decode");
    assert!(same_state(&pf.dump_state(), want_rec), "small prefills: recurrent state differs from decode");
    assert!(same_rows(&tail(&pf, x), want_tail), "small prefills: decode after differs");
    // ubatch 7 / 64 / 256 (+ decode as a fourth batching)
    let mut runs = vec![(0usize, dec_runs[0].0.clone(), dec_runs[1].0.clone(), dec_runs[1].3.clone())];
    for ub in [7usize, 64, 256] {
        let a = kv_engine(&plan, &ws, 1024, KvType::F16);
        a.prefill(&pool, &toks, ub).unwrap();
        let b = kv_engine(&plan, &ws, 1024, KvType::KV8);
        let xb = b.prefill(&pool, &toks, ub).unwrap();
        // quantised where the f16 engine stores: |deq - f16| <= ~d/2 (+ f16 / scale rounding)
        for ai in 0..n_att {
            for kvh in 0..d.n_head_kv {
                let (k16, v16) = a.kv_head_f32(ai, kvh, toks.len());
                let (k8, v8) = b.kv_head_f32(ai, kvh, toks.len());
                for (x16, x8, what) in [(&k16, &k8, "K"), (&v16, &v8, "V")] {
                    for (bi, (c16, c8)) in x16.chunks(32).zip(x8.chunks(32)).enumerate() {
                        let dd = c16.iter().fold(0f32, |m, v| m.max(v.abs())) / 127.0;
                        for (u, w) in c16.iter().zip(c8) {
                            assert!((u - w).abs() <= 0.6 * dd + 1e-3 * u.abs() + 1e-7, "ub {ub}: {what} block {bi} of layer {ai} head {kvh}: {w} vs f16 {u}");
                        }
                    }
                }
            }
        }
        runs.push((ub, kv_all(&a, n_att), kv_all(&b, n_att), tail(&b, xb)));
    }
    let mut pairs = 0;
    for i in 0..runs.len() {
        for j in i + 1..runs.len() {
            if runs[i].1 == runs[j].1 {
                pairs += 1;
                assert!(runs[i].2 == runs[j].2, "ubatch {} vs {}: same f16 cache, different int8 bytes", runs[i].0, runs[j].0);
                assert!(same_rows(&runs[i].3, &runs[j].3), "ubatch {} vs {}: same KV, different decode", runs[i].0, runs[j].0);
            }
        }
    }
    eprintln!("int8 KV write invariance: {pairs} of 6 batching pairs had identical f16 caches (ubatch 0 = decode)");
    std::fs::remove_dir_all(&dir).ok();
}

/// (4) kv8: snapshot / restore replays exactly; verify lanes with partial commits (rollback of
/// rejected int8 V rows) equal one-token steps bit for bit; MTP (catch-up seeding, drafts, verify,
/// rewinds) and batched prefill + MTP give the plain kv8 greedy output and logits.
#[test]
fn mini_int8_kv_snapshot_and_mtp_rollback_replay() {
    let kv = KvType::KV8;
    // snapshot / restore
    {
        let (dir, path) = mini_paths("kv8snap");
        std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
        let opt = mini_opts(&dir);
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
        let eng = kv_engine(&plan, &ws, 512, kv);
        let toks: Vec<u32> = (0..40).map(|i| (i * 11 + 5) % 64).collect();
        let mut want = Vec::new();
        let mut snap = None;
        for (i, &t) in toks.iter().enumerate() {
            if i == 14 {
                snap = Some(eng.save_recurrent());
            }
            eng.step(&pool, t);
            want.push(eng.logits().to_vec());
        }
        eng.restore_recurrent(&snap.unwrap());
        assert_eq!(eng.pos(), 14);
        for (i, &t) in toks.iter().enumerate().skip(14) {
            eng.step(&pool, t);
            assert!(same_bits(eng.logits(), &want[i]), "kv8: step {i} differs after restore");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
    let (dir, _path, plan, ws, mw, pool) = mtp_setup("kv8mtp", 2);
    let n_att = plan.desc.n_attention();
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let opts = |lanes: usize| EngineOptions { verify_lanes: lanes, mtp: Some(&mw), kv_type: Some(kv) };
    // verify lanes with rollback
    {
        let seq: Vec<u32> = (0..60).map(|i| (i * 13 + 7) % 64).collect();
        let r = kv_engine(&plan, &ws, 512, kv);
        let mut want = Vec::new();
        for &t in &seq {
            r.step(&pool, t);
            want.push(r.logits().to_vec());
        }
        let want_kv = kv_all(&r, n_att);
        drop(r);
        let e = Engine::with_options(&plan, &ws, cfg.clone(), opts(4)).unwrap();
        let (mut p, mut it, mut partial) = (0usize, 0u32, 0);
        while p < seq.len() {
            let n = (1 + (it as usize * 7) % 4).min(seq.len() - p);
            let m = 1 + (it as usize * 5 + 3) % n;
            let mut toks: Vec<u32> = seq[p..p + n].to_vec();
            for (i, t) in toks.iter_mut().enumerate().skip(m) {
                *t = (it * 31 + i as u32 * 17 + 1) % 64;
            }
            e.verify(&pool, &toks);
            for ln in 0..m {
                assert!(same_bits(e.logits_lane(ln), &want[p + ln]), "kv8: lane {ln} at pos {} differs", p + ln);
            }
            if it % 3 == 0 && n > m {
                e.commit(n);
                e.commit(m);
            } else {
                e.commit(m);
            }
            partial += (m < n) as usize;
            p += m;
            it += 1;
        }
        assert!(partial > 0);
        assert!(kv_all(&e, n_att) == want_kv, "kv8: KV bytes after verify / rollback differ from plain steps");
    }
    // MTP greedy (depth 2) with forced acceptances and rewinds
    {
        let prompt: Vec<u32> = vec![3, 17, 42, 5, 9, 33, 1];
        let n_gen = 40;
        let r = kv_engine(&plan, &ws, 512, kv);
        let mut next = 0;
        for &t in &prompt {
            next = r.step(&pool, t);
        }
        let mut want_ids = vec![next];
        let mut want_rows = vec![r.logits().to_vec()];
        for _ in 1..n_gen {
            next = r.step(&pool, next);
            want_ids.push(next);
            want_rows.push(r.logits().to_vec());
        }
        drop(r);
        let depth = 2;
        let e = Engine::with_options(&plan, &ws, cfg.clone(), opts(depth + 1)).unwrap();
        let mut d = crate::mtp::MtpDriver::new(&e, depth).unwrap();
        let plen = prompt.len();
        let truth = want_ids.clone();
        d.draft_override = Some(Box::new(move |pos, dr| {
            let i = pos - plen;
            if pos % 5 == 0 || i >= truth.len() {
                dr ^ 1
            } else {
                truth[i]
            }
        }));
        let mut x = d.prefill(&e, &pool, &prompt);
        let mut ids = vec![x];
        let mut out = Vec::new();
        let mut rewound = 0;
        while ids.len() < n_gen {
            let p = e.pos();
            d.step(&e, &pool, x, &mut out);
            for i in 0..out.len() {
                let row = p + i + 1 - plen;
                if row < want_rows.len() {
                    assert!(same_bits(e.logits_lane(i), &want_rows[row]), "kv8 mtp: logits at row {row} differ");
                }
            }
            if out.len() > 1 && ids.len() % 3 == 0 {
                d.rewind(&e, 1);
                out.truncate(1);
                rewound += 1;
            }
            ids.extend_from_slice(&out);
            x = *out.last().unwrap();
        }
        ids.truncate(n_gen);
        assert_eq!(ids, want_ids, "kv8 mtp: ids differ");
        assert!(d.stats.accepted > 0 && rewound > 0, "kv8 mtp: acceptance / rewind not exercised");
    }
    // batched prefill + MTP seeding
    {
        let p1: Vec<u32> = (0..23).map(|i| (i * 11 + 3) % 64).collect();
        let r = kv_engine(&plan, &ws, 512, kv);
        let mut x = r.prefill(&pool, &p1, 8).unwrap();
        let mut want = vec![x];
        for _ in 1..20 {
            x = r.step(&pool, x);
            want.push(x);
        }
        drop(r);
        let e = Engine::with_options(&plan, &ws, cfg.clone(), opts(3)).unwrap();
        let mut d = crate::mtp::MtpDriver::new(&e, 2).unwrap();
        let mut x = d.prefill_batched(&e, &pool, &pool, &p1, 8);
        let mut ids = vec![x];
        let mut out = Vec::new();
        while ids.len() < 20 {
            d.step(&e, &pool, x, &mut out);
            ids.extend_from_slice(&out);
            x = *out.last().unwrap();
        }
        ids.truncate(20);
        assert_eq!(ids, want, "kv8: batched prefill + MTP ids differ from plain");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// (5) kv8 decode and prefill are thread-count invariant (1-3 threads): logits, ids, KV bytes.
#[test]
fn mini_int8_kv_is_thread_count_invariant() {
    let (dir, path) = mini_paths("kv8threads");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..300).map(|i| (i * 13 + 7) % 64).collect();
    let mut want: Option<(Vec<u32>, Vec<Vec<f32>>, Vec<(Vec<u8>, Vec<u8>)>, Vec<f32>, Vec<(Vec<u8>, Vec<u8>)>)> = None;
    for n in [1usize, 2, 3] {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        let n_att = plan.desc.n_attention();
        // decode
        let e = kv_engine(&plan, &ws, 1024, KvType::KV8);
        let (mut ids, mut rows) = (Vec::new(), Vec::new());
        let mut tok = 1u32;
        for s in 0..120 {
            let g = e.step(&pool, tok);
            tok = if s < 60 { (s as u32 * 7 + 3) % 64 } else { g };
            ids.push(g);
            rows.push(e.logits().to_vec());
        }
        let dkv = kv_all(&e, n_att);
        // prefill (ubatch 64: 4-row groups and remainders)
        let p = kv_engine(&plan, &ws, 1024, KvType::KV8);
        p.prefill(&pool, &toks, 64).unwrap();
        let plg = p.logits().to_vec();
        let pkv = kv_all(&p, n_att);
        match &want {
            None => want = Some((ids, rows, dkv, plg, pkv)),
            Some((i1, r1, k1, l1, pk1)) => {
                assert_eq!(&ids, i1, "{n} threads: kv8 decode ids differ");
                for (s, (a, b)) in rows.iter().zip(r1).enumerate() {
                    assert!(same_bits(a, b), "{n} threads: kv8 decode logits of step {s} differ");
                }
                assert!(&dkv == k1, "{n} threads: kv8 decode KV bytes differ");
                assert!(same_bits(&plg, l1), "{n} threads: kv8 prefill logits differ");
                assert!(&pkv == pk1, "{n} threads: kv8 prefill KV bytes differ");
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// (6) The state layout names the KV type; write_state / read_state round-trip a kv8 state
/// (bytes, recurrent state, and the decode that follows), and other types do not match it.
#[test]
fn mini_int8_kv_state_layout_and_round_trip() {
    let (dir, path) = mini_paths("kv8state");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let (plan, ws, pool) = mini_engine_parts(&path, &opt, 2);
    let n_att = plan.desc.n_attention();
    let layouts: Vec<String> = [KvType::F16, KvType::K8, KvType::V8, KvType::KV8].iter().map(|&t| kv_engine(&plan, &ws, 512, t).state_layout()).collect();
    assert!(layouts[0].ends_with(" f16kv"), "f16 layout changed: {}", layouts[0]);
    assert!(layouts[3].contains("kv=kv8"), "{}", layouts[3]);
    for i in 0..4 {
        for j in i + 1..4 {
            assert_ne!(layouts[i], layouts[j]);
        }
    }
    for kv in [KvType::KV8, KvType::K8, KvType::V8] {
        let a = kv_engine(&plan, &ws, 512, kv);
        let toks: Vec<u32> = (0..37).map(|i| (i * 7 + 2) % 64).collect();
        let x = a.prefill(&pool, &toks, 16).unwrap();
        let mut buf = Vec::new();
        a.write_state(&mut buf).unwrap();
        // a different context size reads it (positions only), after unrelated decode
        let b = kv_engine(&plan, &ws, 1024, kv);
        for t in [9u32, 8, 7, 6, 5, 4, 3, 2, 1, 0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41] {
            b.step(&pool, t);
        }
        assert_eq!(a.state_layout(), b.state_layout());
        b.read_state(&mut &buf[..], toks.len()).unwrap();
        assert_eq!(b.pos(), toks.len());
        assert!(kv_all(&a, n_att) == kv_all(&b, n_att), "{kv:?}: KV bytes differ after read_state");
        assert!(same_state(&a.dump_state(), &b.dump_state()), "{kv:?}: state differs after read_state");
        let (mut xa, mut xb) = (x, x);
        for s in 0..20 {
            xa = a.step(&pool, xa);
            xb = b.step(&pool, xb);
            assert!(same_bits(a.logits(), b.logits()), "{kv:?}: decode step {s} after read_state differs");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Deterministic pseudo-random f32 in [-1, 1).
fn lcg_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn q8_rows(n_rows: usize, hd: usize, stride: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut out = vec![0u8; n_rows * stride];
    for r in 0..n_rows {
        let x: Vec<f32> = lcg_f32(hd, seed + r as u64).iter().map(|v| v * scale * (1.0 + r as f32 * 0.37)).collect();
        // SAFETY: hd floats into hd/32 blocks.
        unsafe { kernels_sys::ffi::quantize_row_q8_0(x.as_ptr(), out[r * stride..].as_mut_ptr(), hd as i64) };
    }
    out
}

/// O-13 kernel: oxe_dot_q8_tile == ggml_vec_dot_q8_0_q8_0_generic bit for bit, every element,
/// with row / column tails (7 x 5, 1 x 1, 4 x 9), padded strides and head dims 128 / 256.
#[test]
fn kv8_dot_q8_tile_is_ggml_vec_dot_q8_0_bitwise() {
    for (hd, nx, ny, pad) in [(256usize, 7usize, 5usize, 0usize), (128, 1, 1, 6), (256, 4, 9, 34), (128, 13, 64, 0)] {
        let rb = hd / 32 * 34;
        let (xs, ys) = (rb + pad, rb + 2 * pad);
        let x = q8_rows(nx, hd, xs, 11, 3.0);
        let y = q8_rows(ny, hd, ys, 977, 0.5);
        let (ox, oy) = (1usize, nx + 3);
        let mut out = vec![f32::NAN; ny * oy];
        // SAFETY: nx rows / ny columns as built; out has (ny-1)*oy + nx elements.
        unsafe { kernels_sys::ffi::oxe_dot_q8_tile(hd as i32, x.as_ptr(), xs as i64, nx as i32, y.as_ptr(), ys as i64, ny as i32, out.as_mut_ptr(), ox as i64, oy as i64) };
        for r in 0..nx {
            for c in 0..ny {
                let mut want = 0f32;
                // SAFETY: one row and one column of hd values.
                unsafe { kernels_sys::ffi::ggml_vec_dot_q8_0_q8_0_generic(hd as i32, &mut want, 0, x[r * xs..].as_ptr(), 0, y[c * ys..].as_ptr(), 0, 1) };
                let got = out[r * ox + c * oy];
                assert_eq!(got.to_bits(), want.to_bits(), "hd {hd} {nx}x{ny}: ({r}, {c}) {got} vs ggml {want}");
            }
        }
    }
}

/// O-13 kernel: oxe_pv_q8 == the scalar definition (dequantise q * d, then an f32 fma chain over
/// increasing positions from +0 with the f16 probabilities) bit for bit, and within 1e-5 of an
/// f64 sum; odd column counts, a chunk not starting at 0.
#[test]
fn kv8_pv_q8_matches_scalar_reference() {
    for (hd, ncol, p0, p1) in [(256usize, 5usize, 37usize, 200usize), (128, 8, 0, 256), (128, 1, 250, 251), (256, 64, 0, 64)] {
        let rb = hd / 32 * 34;
        let v = q8_rows(p1, hd, rb, 5, 2.0);
        let ps = p1 + 13;
        // probabilities: positive f16 values (exact in f32) and some +0
        let mut prob = vec![0u16; ncol * ps];
        let raw = lcg_f32(ncol * ps, 99);
        for (i, p) in prob.iter_mut().enumerate() {
            let x = raw[i].abs() * 0.01;
            let mut h = [0u16];
            kernels_sys::ops::fp32_to_fp16(&[if i % 7 == 0 { 0.0 } else { x }], &mut h);
            *p = h[0];
        }
        let mut out = vec![f32::NAN; ncol * hd];
        let mut pscr = vec![0f32; (p1 - p0) * ncol.div_ceil(4) * 4];
        // SAFETY: rows < p1; probabilities readable on [p0, p1) of every column; buffers sized.
        unsafe { kernels_sys::ffi::oxe_pv_q8(hd as i32, v.as_ptr(), rb as i64, p0 as i32, p1 as i32, prob.as_ptr(), ps as i64, ncol as i32, out.as_mut_ptr(), pscr.as_mut_ptr()) };
        // exact f16 -> f32, decoded by hand (finite values only)
        let h2f = |h: u16| -> f32 {
            let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let e = ((h >> 10) & 0x1f) as i32;
            let m = (h & 0x3ff) as f32;
            if e == 0 {
                s * m * (1.0 / 16777216.0)
            } else {
                s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
            }
        };
        for c in 0..ncol {
            for dd in 0..hd {
                let (blk, j) = (dd / 32, dd % 32);
                let mut acc = 0f32;
                let mut acc64 = 0f64;
                let mut abs64 = 0f64;
                for p in p0..p1 {
                    let b = &v[p * rb + blk * 34..];
                    let dv = h2f(u16::from_le_bytes([b[0], b[1]]));
                    let x = (b[2 + j] as i8) as f32 * dv;
                    let pr = h2f(prob[c * ps + p]);
                    acc = x.mul_add(pr, acc);
                    acc64 += x as f64 * pr as f64;
                    abs64 += (x as f64 * pr as f64).abs();
                }
                let got = out[c * hd + dd];
                assert_eq!(got.to_bits(), acc.to_bits(), "hd {hd} ncol {ncol} [{p0},{p1}): ({c}, {dd}) {got} vs {acc}");
                // f32 summation error bound: n * eps * sum |terms| (relative to |result| it is unbounded
                // under cancellation, which random-sign V values produce)
                let n = (p1 - p0) as f64;
                assert!((got as f64 - acc64).abs() <= n * f32::EPSILON as f64 * abs64 + 1e-12, "f32 sum outside its error bound");
            }
        }
    }
}

/// Random Q8_0 V rows, f16 probabilities (>= 0, some +0, magnitudes over four decades) and one
/// call of each V . p kernel: (reference, fast, int16, int8) outputs.
#[allow(clippy::type_complexity)]
fn pv_kernels(hd: usize, ncol: usize, p0: usize, p1: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rb = hd / 32 * 34;
    let v = q8_rows(p1, hd, rb, seed, 2.0);
    let ps = p1 + 13;
    let mut prob = vec![0u16; ncol * ps];
    let raw = lcg_f32(ncol * ps, seed + 99);
    for (i, p) in prob.iter_mut().enumerate() {
        let x = raw[i].abs() * [0.01f32, 0.001, 0.0001, 0.00001][i % 4];
        let mut h = [0u16];
        kernels_sys::ops::fp32_to_fp16(&[if i % 7 == 0 { 0.0 } else { x }], &mut h);
        *p = h[0];
    }
    let np = p1 - p0;
    // SAFETY: a pure size computation.
    let ib = unsafe { kernels_sys::ffi::oxe_pv_q8_int_scratch(hd as i32, np as i32, ncol as i32) } as usize;
    let mut scr = vec![0f32; (2 * np * ncol.div_ceil(4) * 4).max(ib / 4 + 1)];
    const CANARY: f32 = -77.25;
    let mut outs: Vec<Vec<f32>> = Vec::new();
    for k in 0..4 {
        let mut out = vec![f32::NAN; ncol * hd + 8];
        out[ncol * hd..].fill(CANARY);
        let (hd_, rb_, a, b, ps_, nc) = (hd as i32, rb as i64, p0 as i32, p1 as i32, ps as i64, ncol as i32);
        // SAFETY: rows < p1; probabilities readable on [p0, p1) of every column; buffers sized.
        unsafe {
            match k {
                0 => kernels_sys::ffi::oxe_pv_q8(hd_, v.as_ptr(), rb_, a, b, prob.as_ptr(), ps_, nc, out.as_mut_ptr(), scr.as_mut_ptr()),
                1 => kernels_sys::ffi::oxe_pv_q8_fast(hd_, v.as_ptr(), rb_, a, b, prob.as_ptr(), ps_, nc, out.as_mut_ptr(), scr.as_mut_ptr()),
                _ => kernels_sys::ffi::oxe_pv_q8_int(hd_, v.as_ptr(), rb_, a, b, prob.as_ptr(), ps_, nc, out.as_mut_ptr(), scr.as_mut_ptr() as *mut core::ffi::c_void, if k == 2 { 16 } else { 8 }),
            }
        }
        assert!(out[ncol * hd..].iter().all(|x| *x == CANARY), "kernel {k} wrote past its {ncol} columns");
        out.truncate(ncol * hd);
        outs.push(out);
    }
    let i8o = outs.pop().unwrap();
    let i16o = outs.pop().unwrap();
    let fast = outs.pop().unwrap();
    (outs.pop().unwrap(), fast, i16o, i8o)
}

const PV_SHAPES: [(usize, usize, usize, usize); 10] =
    [(256, 5, 37, 200), (128, 8, 0, 256), (128, 1, 250, 251), (256, 64, 0, 64), (256, 8, 0, 256), (64, 12, 3, 100), (256, 3, 0, 255), (32, 9, 7, 16), (256, 4, 0, 256), (256, 16, 256, 512)];

/// OD-4d: oxe_pv_q8_fast == oxe_pv_q8 bit for bit (every output, odd column counts on both tile
/// widths, chunks not starting at 0 or not a multiple of 8, head dims 32..256).
#[test]
fn kv8_pv_q8_fast_matches_reference_bitwise() {
    for (i, (hd, ncol, p0, p1)) in PV_SHAPES.into_iter().enumerate() {
        let (want, fast, _, _) = pv_kernels(hd, ncol, p0, p1, 5 + i as u64);
        for (j, (a, b)) in fast.iter().zip(&want).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "hd {hd} ncol {ncol} [{p0},{p1}): output {j} {a} vs reference {b}");
        }
    }
}

/// OD-4d: the lossy integer V . p stays within its stated error of the reference: relative L2 over
/// the outputs below 1.5e-3 with 16-bit weights (measured about 4e-4: three f16 roundings) and
/// below 3e-2 with 8-bit weights (measured about 7e-3); all-zero probabilities give +0 outputs.
#[test]
fn kv8_pv_q8_int_within_stated_error() {
    for (i, (hd, ncol, p0, p1)) in PV_SHAPES.into_iter().enumerate() {
        let (want, _, i16o, i8o) = pv_kernels(hd, ncol, p0, p1, 5 + i as u64);
        let nrm: f64 = want.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let err = |o: &[f32]| o.iter().zip(&want).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum::<f64>().sqrt() / nrm;
        let (e16, e8) = (err(&i16o), err(&i8o));
        assert!(e16 < 1.5e-3, "hd {hd} ncol {ncol} [{p0},{p1}): int16 relative error {e16}");
        assert!(e8 < 3e-2, "hd {hd} ncol {ncol} [{p0},{p1}): int8 relative error {e8}");
    }
    let (hd, ncol, np) = (64usize, 3usize, 40usize);
    let v = q8_rows(np, hd, hd / 32 * 34, 3, 1.0);
    let prob = vec![0u16; ncol * np];
    // SAFETY: a pure size computation; then rows < np, zero probabilities, sized buffers.
    let ib = unsafe { kernels_sys::ffi::oxe_pv_q8_int_scratch(hd as i32, np as i32, ncol as i32) } as usize;
    let mut scr = vec![0f32; ib / 4 + 1];
    for bits in [16, 8] {
        let mut out = vec![f32::NAN; ncol * hd];
        unsafe { kernels_sys::ffi::oxe_pv_q8_int(hd as i32, v.as_ptr(), (hd / 32 * 34) as i64, 0, np as i32, prob.as_ptr(), np as i64, ncol as i32, out.as_mut_ptr(), scr.as_mut_ptr() as *mut core::ffi::c_void, bits) };
        assert!(out.iter().all(|x| x.to_bits() == 0), "bits {bits}: zero probabilities must give +0");
    }
}

/// OD-4d on the mini model (int8 KV, fused): the fast V . p kernel gives the reference kernel's
/// logits bit for bit in prefill and decode; the lossy integer kernels are thread-count invariant
/// and stay within 5e-2 of the logit range of the reference. Measured 1.2e-2 (int16) and 9.7e-3
/// (int8): both sit at the int8 cache's own noise floor (1.7e-2 to the f32 reference above), not
/// at the kernel error (3.5e-4 / 7e-3): any perturbation flips some int8 roundings of the K / V
/// rows written later. The kernel error itself is bounded in kv8_pv_q8_int_within_stated_error.
#[test]
fn mini_int8_kv_pv_modes() {
    let (dir, path) = mini_paths("kv8pv");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..280).map(|i| (i * 13 + 7) % 64).collect();
    let run = |mode: PvMode, n: usize| -> Vec<Vec<f32>> {
        let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
        let mut e = kv_engine(&plan, &ws, 1024, KvType::KV8);
        e.set_kv8_pv(mode);
        let mut x = e.prefill(&pool, &toks, 64).unwrap();
        let mut rows = vec![e.logits().to_vec()];
        for _ in 0..30 {
            x = e.step(&pool, x);
            rows.push(e.logits().to_vec());
        }
        rows
    };
    let want = run(PvMode::Ref, 2);
    let fast = run(PvMode::Fast, 2);
    for (i, (a, b)) in fast.iter().zip(&want).enumerate() {
        assert!(same_bits(a, b), "fast V . p: logits row {i} differ from the reference kernel");
    }
    for (mode, tol) in [(PvMode::I16, 5e-2f32), (PvMode::I8, 5e-2)] {
        let a = run(mode, 1);
        let b = run(mode, 3);
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(same_bits(x, y), "{mode:?}: logits row {i} depend on the thread count");
        }
        let mut worst = 0f32;
        for (x, w) in a.iter().zip(&want) {
            let range = w.iter().fold(0f32, |m, v| m.max(v.abs()));
            let gap = x.iter().zip(w).fold(0f32, |m, (p, q)| m.max((p - q).abs()));
            worst = worst.max(gap / range);
        }
        println!("mini {mode:?}: max logit gap / logit range vs the reference kernel = {worst:e}");
        assert!(worst < tol, "{mode:?}: logit gap {worst} of the range");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// O-13: the fused and the first-version int8 reads are both thread-count invariant and both
/// stay close to each other (the fused scores use a Q8_0 query, so not bitwise).
#[test]
fn mini_int8_kv_fused_and_first_version_thread_invariant() {
    let (dir, path) = mini_paths("kv8fused");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let toks: Vec<u32> = (0..280).map(|i| (i * 13 + 7) % 64).collect();
    for fused in [true, false] {
        let mut want: Option<(Vec<f32>, Vec<Vec<f32>>)> = None;
        for n in [1usize, 2, 3] {
            let (plan, ws, pool) = mini_engine_parts(&path, &opt, n);
            let mut e = kv_engine(&plan, &ws, 1024, KvType::KV8);
            e.set_kv8_fused(fused);
            let mut x = e.prefill(&pool, &toks, 64).unwrap();
            let lg = e.logits().to_vec();
            let mut rows = Vec::new();
            for _ in 0..30 {
                x = e.step(&pool, x);
                rows.push(e.logits().to_vec());
            }
            match &want {
                None => want = Some((lg, rows)),
                Some((l1, r1)) => {
                    assert!(same_bits(&lg, l1), "fused={fused}, {n} threads: prefill logits differ");
                    for (i, (a, b)) in rows.iter().zip(r1).enumerate() {
                        assert!(same_bits(a, b), "fused={fused}, {n} threads: decode row {i} differs");
                    }
                }
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// OD-6b: verify runs through the lane GEMVs (`OXE_VL`, bits VL_MV | VL_HEAD, each alone and both)
/// are bit-identical to plain one-token steps: 1-4 lanes with partial commits and rejected tails,
/// and 5-6 lanes (lane groups of up to 4), at 1-3 threads; logits and hidden states bit for bit.
/// The mini model has Q3_K / Q4_K / Q5_K experts, Q4_K dense, Q6_K head and shared experts and one
/// Q8_0 shared expert, so every lane kernel runs.
#[test]
fn mini_verify_lane_kernels_are_bit_identical() {
    use crate::engine::{VL_DN_FROM, VL_HEAD, VL_MV, VL_Q6K_ONE};
    for ty in kernels_sys::QType::ALL {
        assert!(kernels_sys::gemv_lanes_has(ty), "{ty:?}: no lane GEMV in this build (the test would pass on the fallback)");
    }
    for nt in [1usize, 2, 3] {
        let (dir, _path, plan, ws, mw, pool) = mtp_setup(&format!("vlk{nt}"), nt);
        let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
        let seq: Vec<u32> = (0..72).map(|i| (i * 11 + 5) % 64).collect();
        let mut r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        r.set_verify_lane_kernels(0);
        let mut want = Vec::new();
        let mut want_h = Vec::new();
        for &t in &seq {
            r.step(&pool, t);
            want.push(r.logits().to_vec());
            want_h.push(r.hidden(0).to_vec());
        }
        drop(r);
        // plain steps with the one-lane Q6_K kernel (head, shared experts)
        let mut f = Engine::new(&plan, &ws, cfg.clone()).unwrap();
        f.set_verify_lane_kernels(VL_Q6K_ONE);
        for (i, &t) in seq.iter().enumerate() {
            f.step(&pool, t);
            assert!(bits_eq(f.logits(), &want[i]), "{nt} threads: one-lane Q6_K step logits differ at {i}");
            assert!(bits_eq(f.hidden(0), &want_h[i]), "{nt} threads: one-lane Q6_K step h differs at {i}");
        }
        drop(f);
        for mask in [VL_MV, VL_HEAD, VL_MV | VL_HEAD, VL_Q6K_ONE, VL_DN_FROM, 15] {
            // 1-4 lanes on an MTP engine, rejected (random) tails rolled back
            let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: 4, mtp: Some(&mw), kv_type: None }).unwrap();
            e.set_verify_lane_kernels(mask);
            e.set_dec_fused_dn(mask == 15); // the fused 4-row step out of place too
            assert_eq!(e.verify_lane_kernels(), mask);
            let (mut p, mut it) = (0usize, 0u32);
            let mut commits = [0usize; 5];
            while p < seq.len() {
                let n = (1 + (it as usize * 7) % 4).min(seq.len() - p);
                let m = 1 + (it as usize * 5 + 3) % n;
                let mut toks: Vec<u32> = seq[p..p + n].to_vec();
                for (i, t) in toks.iter_mut().enumerate().skip(m) {
                    *t = (it * 29 + i as u32 * 13 + 3) % 64;
                }
                let g = e.verify(&pool, &toks);
                for ln in 0..m {
                    assert!(bits_eq(e.logits_lane(ln), &want[p + ln]), "{nt} threads, mask {mask}: lane {ln} at pos {} differs", p + ln);
                    assert!(bits_eq(e.hidden(ln), &want_h[p + ln]), "{nt} threads, mask {mask}: h of lane {ln} differs");
                    let lg = e.logits_lane(ln);
                    let mut best = 0;
                    for (j, &v) in lg.iter().enumerate() {
                        if v > lg[best] {
                            best = j;
                        }
                    }
                    assert_eq!(g[ln] as usize, best, "{nt} threads, mask {mask}: argmax of lane {ln}");
                }
                if it % 3 == 0 && n > m {
                    e.commit(n);
                    e.commit(m);
                } else {
                    e.commit(m);
                }
                commits[m] += 1;
                p += m;
                it += 1;
            }
            assert!(commits[1] > 0 && commits[2] > 0 && commits[3] > 0, "{commits:?}");
            drop(e);
            // 5 and 6 lanes (lane groups 4 + 1 and 4 + 2), all committed
            let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: 6, mtp: None, kv_type: None }).unwrap();
            e.set_verify_lane_kernels(mask);
            let mut p = 0usize;
            let mut it = 0usize;
            while p < seq.len() {
                let n = (5 + it % 2).min(seq.len() - p);
                e.verify(&pool, &seq[p..p + n]);
                for ln in 0..n {
                    assert!(bits_eq(e.logits_lane(ln), &want[p + ln]), "{nt} threads, mask {mask}: {n} lanes, lane {ln} at pos {} differs", p + ln);
                    assert!(bits_eq(e.hidden(ln), &want_h[p + ln]), "{nt} threads, mask {mask}: {n} lanes, h of lane {ln} differs");
                }
                e.commit(n);
                p += n;
                it += 1;
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Greedy MTP decoding with the speculation switched per step mid-reply: depth 2 for the first 8
/// positions after the prompt, plain steps for the next 12, depth 1 for 12, depth 2 to the end.
/// Run once without and once with a rewind (keep one token) inside an on-stretch. Ids and every
/// committed logit row are bit-identical to plain greedy decoding.
#[test]
fn mini_mtp_switching_depth_mid_reply_is_identical_to_plain() {
    let (dir, _path, plan, ws, mw, pool) = mtp_setup("mtpswitch", 2);
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let prompt: Vec<u32> = vec![3, 17, 42, 5, 9, 33, 1];
    let plen = prompt.len();
    let n_gen = 48;
    let r = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    let mut next = 0;
    for &t in &prompt {
        next = r.step(&pool, t);
    }
    let mut want_ids = vec![next];
    let mut want_rows = vec![r.logits().to_vec()];
    for _ in 1..n_gen {
        next = r.step(&pool, next);
        want_ids.push(next);
        want_rows.push(r.logits().to_vec());
    }
    drop(r);
    for with_rewind in [false, true] {
        let mut e = Engine::with_options(&plan, &ws, cfg.clone(), crate::engine::EngineOptions { verify_lanes: 3, mtp: Some(&mw), kv_type: None }).unwrap();
        e.set_attn_tiled_min_kv(0); // MTP / verify lanes tiled, the plain reference per-head
        let mut d = crate::mtp::MtpDriver::new(&e, 2).unwrap();
        // same test hook as mini_mtp_greedy_is_identical_to_plain: true tokens at most positions,
        // a wrong one at every 5th, so every acceptance length occurs
        let truth = want_ids.clone();
        d.draft_override = Some(Box::new(move |pos: usize, dr: u32| -> u32 {
            let i = pos - plen; // index into truth of the token at `pos`
            if pos % 5 == 0 || i >= truth.len() {
                dr ^ 1
            } else {
                truth[i]
            }
        }));
        let mut x = d.prefill(&e, &pool, &prompt);
        assert!(bits_eq(e.logits_lane((plen - 1) % 3), &want_rows[0]), "rewind {with_rewind}: prefill logits differ");
        let mut ids = vec![x];
        let mut out = Vec::new();
        let mut rewound = false;
        let mut plain = 0usize;
        while ids.len() < n_gen {
            let p = e.pos(); // position of x
            let off = p - plen;
            if (8..20).contains(&off) {
                d.step_plain(&e, &pool, x, &mut out);
                plain += 1;
                let row = p + 1 - plen; // logits after position p
                if row < want_rows.len() {
                    assert!(bits_eq(e.logits_lane(0), &want_rows[row]), "rewind {with_rewind}: plain step at pos {p}: logits differ");
                }
            } else {
                d.depth = if (20..32).contains(&off) { 1 } else { 2 };
                d.step(&e, &pool, x, &mut out);
                for i in 0..out.len() {
                    let row = p + i + 1 - plen; // logits after position p+i
                    if row < want_rows.len() {
                        assert!(bits_eq(e.logits_lane(i), &want_rows[row]), "rewind {with_rewind}: depth {} at pos {p}: logits at row {row} differ", d.depth);
                    }
                }
                if with_rewind && !rewound && out.len() > 1 {
                    d.rewind(&e, 1);
                    out.truncate(1);
                    rewound = true;
                }
            }
            ids.extend_from_slice(&out);
            x = *out.last().unwrap();
        }
        ids.truncate(n_gen);
        assert_eq!(ids, want_ids, "rewind {with_rewind}: ids differ");
        assert!(plain > 0 && d.stats.plain_steps > 0, "rewind {with_rewind}: no plain step exercised");
        assert!(d.stats.steps > 0 && d.stats.accepted > 0, "rewind {with_rewind}: acceptance not exercised: {:?}", d.stats);
        assert!(!with_rewind || rewound, "no rewind exercised");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The MTP weight cache file is named by the model's SHA-256. `sha_from_env = false` ignores
/// OXE_MODEL_SHA256 (set to "1"*64 by mtp_setup) for a file that is not the model, so a copy of the
/// MTP gguf with different bytes gets the real digest; `sha_from_env = true` keeps the override.
#[test]
fn mtp_sidecar_head_sha_ignores_model_sha_env() {
    let (dir, path, plan, _ws, _mw, _pool) = mtp_setup("shaenv", 1);
    // trailing bytes after the last tensor: Gguf::open only checks that every tensor lies inside
    // the file, so the copy still loads, and its bytes (hence its digest) differ
    let copy = dir.join("mini-mtp-copy.gguf");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(&[0u8; 16]);
    std::fs::write(&copy, &bytes).unwrap();
    let cache = dir.join("cache");
    let env_sha = "1".repeat(64);
    let file_name = |m: &crate::mtp::MtpWeights| -> String {
        std::path::Path::new(&m.store.info.cache_path).file_name().unwrap().to_string_lossy().into_owned()
    };
    let mut opt = mini_opts(&dir);
    opt.sha_from_env = false;
    let real = crate::weights::model_sha256_opt(&copy, &cache, false).unwrap();
    assert_ne!(real, env_sha);
    let m = crate::mtp::MtpWeights::load(&copy, &plan, &opt).unwrap();
    let name = file_name(&m);
    assert!(!name.starts_with(&env_sha), "cache file {name} is named by OXE_MODEL_SHA256");
    assert!(name.starts_with(&real), "cache file {name} is not named by the real sha {real}");
    drop(m);
    opt.sha_from_env = true;
    let m = crate::mtp::MtpWeights::load(&copy, &plan, &opt).unwrap();
    let name = file_name(&m);
    assert!(name.starts_with(&env_sha), "cache file {name} is not named by OXE_MODEL_SHA256");
    drop(m);
    std::fs::remove_dir_all(&dir).ok();
}
