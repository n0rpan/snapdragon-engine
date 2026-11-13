//! oxe: decode engine CLI (M1/M2).
//!
//!   oxe gen     --model M (--tokens 1,2,3 | --ref DIR) [--n 64]          greedy ids to stdout (JSON)
//!   oxe trace   --model M (--tokens .. | --ref DIR) --dump-pos 0-3 --out DIR
//!                                                                         named intermediates, dump_ref format
//!   oxe compare --ref DIR --ours DIR [--pos P] [--all]                    per-tensor error table
//!   oxe verify  --model M --ref DIR                                       M2 gate: teacher-forced replay of the
//!                                                                         oracle's prompt + greedy tokens, ids + KLD
//!   oxe bench   --model M [--tokens 128 --warmup 16 --prompt-len 16 | --prompt-ids 1,2,..] [--batched-prompt]
//!                                                                         decode timing with real math (Codex lane)
//!   oxe cache   --model M                                                 build/verify the repacked weight cache only
//!   oxe prefill-verify --model M --ref DIR [--ubatch 256] [--trace-out D]  M3.2 gate: batched prefill vs llama.cpp's
//!                                                                         batched prefill (logits, KV, recurrent state),
//!                                                                         then teacher-forced decode of the oracle's tokens
//!   oxe prefill-bench --model M [--prompt-len 512[,N..]] [--prefix-len 0] [--pf-tiles 0[,M..]] [--pf-group 1[,G..]] [--ubatch 256] [--reps 3] [--profile] [--state-hash] prefill timing, arms interleaved (Codex lane)
//!   (OXE_ATTN=legacy selects the per-head M1 attention path for A/B; OXE_ATTN_TILED_MIN_KV=N the
//!   decode cutoff for the tiled path, default 4096; results are bit-identical either way)
//!   oxe mtp-check --model M --refs DIR1,DIR2,.. [--n 256 --depths 1,2,3 --prefill batched|tokenwise]
//!                                                                         M3.1 gate: greedy with MTP on (each
//!                                                                         depth) vs off: ids + logits of every
//!                                                                         accepted position bit-identical
//!   (bench, gen: --spec-draft-n-max N (alias --draft N) decodes with MTP speculation, depth N;
//!    bench, gen, mtp-check: --mtp-from GGUF takes the MTP block from another GGUF of the same shape;
//!    --draft-vocab RANKING [--draft-k K] restricts the draft head to the first K ids of RANKING)
//!
//!   LONG-CTX: bench (MTP, --batched-prompt) and prefill-bench (--prefix-len) take --state-file F: the
//!   prompt / prefix state is built once (chunked batched prefill), written to F, and read back by
//!   later runs whose engine layout, KV type and token ids match (a 128K context is ~40 min to build,
//!   seconds to read). bench --plain-steps: depth-0 decode on that MTP engine / state. --prompt-ids-file / --prefix-ids-file F: ids separated by commas / whitespace
//!   (a 128K-id --prompt-ids exceeds the 128 KiB argument limit). n_ctx is checked against the GGUF
//!   context_length (262144 for Ornith / Qwen3.6).
//!
//!   oxe fused-ref --candidate prefill-fa2|prefill-fa2x|fused-decode --model M --prefix-len P
//!               (--prefix-ids-file F | --text-file F) --state-file S --out R
//!               [--n 1000 --ref-samples 200 --ubatch 256 --control decode|ubatch:N --continuation-file T
//!                --kld-factor 1.0 --top1-slack-pp 0.5 --ref-ratio-max 1.1 --jump-threshold 1e-3 --dry-run]
//!                                                                         (--features f32-reference, bench only) O-117
//!                                                                         quality runner: candidate vs tiled from one saved
//!                                                                         state, every position, measured control, f32-attention
//!                                                                         reference (FUSED-REF.md; NOT a full-model f32 ref)
//!
//! Common: --cpu-list 0-7 (default) | --threads N (cpus 0..N), --ctx 4096, --weights file|anon,
//! --verify sample|full|none, --cache-dir D, --thp-collapse, --profile, --out FILE (bench JSON).
#[cfg(feature="f32-reference")]
#[path="../f32_proof.rs"]
mod f32_proof;
#[cfg(feature="f32-reference")]
#[path="../fused_ref_cmd.rs"]
mod fused_ref_cmd;
#[cfg(feature="npu-device")]
#[path="../expert_dma.rs"]
mod expert_dma;
#[cfg(feature="npu-device")]
#[path="../expert_price.rs"]
mod expert_price;
use model::engine::{tick_hz, Engine, EngineConfig, EngineOptions, StateDump};
use model::mtp::{MtpDriver, MtpStats, MtpWeights};
use model::load::{memory_report, open};
use model::plan::PhaseKind;
use model::weights::{Backing, LoadOptions, Verify};
use sched::{parse_cpu_list, Pool};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Options that take no value (`--name`). Every boolean the commands read with `Args::flag` must be listed
/// here (scanned by the test below), or the parser would swallow the next argument as its value (`bool_flags_cover_every_flag_read`).
const BOOL_FLAGS: &[&str] = &["profile", "thp-collapse", "all", "logits", "batched-prompt", "mlock", "state-hash", "check-args", "dry-run", "plain-steps"];

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
    flags: Vec<String>,
}

impl Args {
    fn parse() -> Args {
        Args::parse_from(std::env::args().skip(1).collect())
    }

    /// [`Args::parse`] of the arguments after the program name.
    fn parse_from(args: Vec<String>) -> Args {
        let mut it = args.into_iter();
        let cmd = it.next().unwrap_or_else(|| usage());
        let mut kv = HashMap::new();
        let mut flags = Vec::new();
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            let a = &rest[i];
            let Some(k) = a.strip_prefix("--") else { die(&format!("unexpected argument {a}")) };
            if BOOL_FLAGS.contains(&k) {
                flags.push(k.to_string());
                i += 1;
            } else {
                let v = rest.get(i + 1).unwrap_or_else(|| die(&format!("--{k} needs a value"))).clone();
                kv.insert(k.to_string(), v);
                i += 2;
            }
        }
        Args { cmd, kv, flags }
    }
    /// `--check-args` for every command but prefill-bench (which has its own, richer one): print how
    /// the command line was understood and stop before any model is opened. An option that no
    /// command of this binary reads is an error (the parser would silently ignore it and, say, a
    /// mistyped `--weights anon` would run file-backed). Keep KNOWN in step with new options.
    fn check_known(&self) -> ! {
        const KNOWN: &[&str] = &[
            "layer", "shared-experts", "tensor", "npu-overlap", "npu-recurrent", "npu-recurrent-mask", "npu-audit", "teacher-ids", "split-last", "reference-mode", "f32-greedy", "npu-prefill", "npu-go-file", "arms", "secs", "sets", "rows", "work", "cache-dir", "cpu-list", "ctx", "depths", "draft", "draft-k", "draft-vocab", "drop", "dump-pos", "files-from", "gen",
            "idle-spin-us", "json", "kv", "lanes", "load-threads", "model", "mtp-from", "n", "ours", "out", "pf-group", "pf-tiles", "pos",
            "prefill", "prefix-len", "prefix-ids-file", "prompt-ids", "prompt-ids-file", "prompt-len", "state-file", "ref", "refs", "reps", "runs", "score", "spec-draft-n-max", "text-file",
            "threads", "tokens", "trace-out", "ubatch", "verify", "warmup", "weights", "cmd",
            "continuation-file", "ref-samples", "ref-ratio-max", "fused-chunk", "jump-threshold", "candidate", "control", "kld-factor", "top1-slack-pp",
        ];
        let mut unknown: Vec<&String> = self.kv.keys().filter(|k| !KNOWN.contains(&k.as_str())).collect();
        unknown.sort();
        // long values (prompt ids) are shown by length only
        let kv: BTreeMap<&String, String> = self.kv.iter().map(|(k, v)| (k, if v.len() > 80 { format!("<{} bytes>", v.len()) } else { v.clone() })).collect();
        let arms = self.kv.get("arms").map(|spec| parse_arms(spec)); // dies on a bad arm spec
        println!("{}", json!({ "cmd": self.cmd, "options": kv, "flags": self.flags, "unknown_keys": unknown, "arms": arms }));
        if !unknown.is_empty() {
            die("unknown options");
        }
        std::process::exit(0)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.kv.get(k).map(|s| s.as_str())
    }
    fn num(&self, k: &str, d: usize) -> usize {
        self.get(k).map(|v| v.parse().unwrap_or_else(|_| die(&format!("--{k}: not a number")))).unwrap_or(d)
    }
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
    fn cpus(&self) -> Vec<usize> {
        if let Some(n) = self.get("threads") {
            if self.get("cpu-list").is_none() {
                let n: usize = n.parse().unwrap_or_else(|_| die("--threads"));
                return (0..n).collect();
            }
        }
        parse_cpu_list(self.get("cpu-list").unwrap_or("0-7")).unwrap_or_else(|e| die(&e))
    }
    fn load_opts(&self) -> LoadOptions {
        LoadOptions {
            backing: match self.get("weights").unwrap_or("file") {
                "file" => Backing::File,
                "anon" => Backing::Anon,
                w => die(&format!("--weights {w}: file|anon")),
            },
            verify: match self.get("verify").unwrap_or("sample") {
                "sample" => Verify::Sample,
                "full" => Verify::Full,
                "none" => Verify::None,
                v => die(&format!("--verify {v}: sample|full|none")),
            },
            threads: self.num("load-threads", 8),
            cache_dir: self.get("cache-dir").map(PathBuf::from),
            progress: true,
            thp_collapse: self.flag("thp-collapse"),
            mlock: self.flag("mlock"),
            sha_from_env: true,
        }
    }
    fn model(&self) -> PathBuf {
        PathBuf::from(self.get("model").unwrap_or_else(|| die("--model is required")))
    }
}

/// Every subcommand, in usage order. N3 (audit): main() dispatches from this table, so the usage
/// line can no longer list fewer commands than the binary accepts.
const COMMANDS: &[(&str, fn(&Args))] = &[
    ("gen", cmd_gen),
    ("trace", cmd_trace),
    ("compare", cmd_compare),
    ("verify", cmd_verify),
    ("bench", cmd_bench),
    ("verify-bench", cmd_verify_bench),
    ("kv-kld", cmd_kv_kld),
    ("kbench-mt", cmd_kbench_mt),
    ("vocab-freq", cmd_vocab_freq),
    ("cache", cmd_cache),
    ("prefill-verify", cmd_prefill_verify),
    ("prefill-bench", cmd_prefill_bench),
    ("mtp-check", cmd_mtp_check),
    ("expert-price", cmd_expert_price),
    ("expert-dma-check", cmd_expert_dma_check),
    ("f32-export", cmd_f32_export),
    ("f32-inspect-text", cmd_f32_inspect_text),
    ("f32-fixture-tokens", cmd_f32_fixture_tokens),
    ("f32-tensor-map", cmd_f32_tensor_map),
    ("f32-greedy-export", cmd_f32_greedy_export),
    ("fused-ref", cmd_fused_ref),
    ("npu-greedy-check", cmd_npu_greedy_check),
    ("npu-f32-check", cmd_npu_f32_check),
    ("npu-prefill-check", cmd_npu_prefill_check),
];

// Feature-gated subcommands (text-encoder/NPU lane), wrapped so the N3 table stays complete.
fn cmd_expert_price(_a: &Args) {
    #[cfg(feature="npu-device")] if let Err(e)=expert_price::run(_a){die(&e);}
    #[cfg(not(feature="npu-device"))] die("npu-device feature required");
}
fn cmd_expert_dma_check(_a: &Args) {
    #[cfg(feature="npu-device")] if let Err(e)=expert_dma::run(_a){die(&e);}
    #[cfg(not(feature="npu-device"))] die("npu-device feature required");
}
fn cmd_f32_export(_a: &Args) {
    #[cfg(feature="f32-reference")] f32_proof::export(_a);
    #[cfg(not(feature="f32-reference"))] die("build with f32-reference feature");
}
fn cmd_f32_inspect_text(_a: &Args) {
    #[cfg(feature="f32-reference")] f32_proof::inspect_text(_a);
    #[cfg(not(feature="f32-reference"))] die("build with f32-reference feature");
}
fn cmd_f32_fixture_tokens(_a: &Args) {
    #[cfg(feature="f32-reference")] f32_proof::fixture_tokens(_a);
    #[cfg(not(feature="f32-reference"))] die("build with f32-reference feature");
}
fn cmd_f32_tensor_map(_a: &Args) {
    #[cfg(feature="f32-reference")] f32_proof::tensor_map(_a);
    #[cfg(not(feature="f32-reference"))] die("build with f32-reference feature");
}
fn cmd_f32_greedy_export(_a: &Args) {
    #[cfg(feature="f32-reference")] f32_proof::greedy_export(_a);
    #[cfg(not(feature="f32-reference"))] die("build with f32-reference feature");
}
fn cmd_fused_ref(_a: &Args) {
    #[cfg(feature="f32-reference")] fused_ref_cmd::run(_a);
    #[cfg(not(feature="f32-reference"))] die("build with f32-reference feature (bench only)");
}
fn cmd_npu_greedy_check(_a: &Args) {
    #[cfg(all(feature="f32-reference",feature="npu-prefill"))] f32_proof::greedy(_a);
    #[cfg(not(all(feature="f32-reference",feature="npu-prefill")))] die("build with f32-reference,npu-prefill features");
}
fn cmd_npu_f32_check(_a: &Args) {
    #[cfg(all(feature="f32-reference",feature="npu-prefill"))] f32_proof::compare(_a);
    #[cfg(not(all(feature="f32-reference",feature="npu-prefill")))] die("build with f32-reference,npu-prefill features");
}

fn usage_line() -> String {
    let names: Vec<&str> = COMMANDS.iter().map(|(name, _)| *name).collect();
    format!("usage: oxe {} [options]  (see the source header or docs/lab-notes/M2.md)", names.join("|"))
}

fn usage() -> ! {
    eprintln!("{}", usage_line());
    std::process::exit(2)
}

fn die(msg: &str) -> ! {
    eprintln!("oxe: {msg}");
    std::process::exit(2)
}

fn parse_ids(s: &str) -> Vec<u32> {
    s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap_or_else(|_| die("bad token id"))).collect()
}

fn parse_range(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for p in s.split(',').filter(|p| !p.is_empty()) {
        match p.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => v.push(p.parse().unwrap()),
        }
    }
    v
}

fn read_manifest(dir: &Path) -> serde_json::Value {
    let s = std::fs::read_to_string(dir.join("manifest.json")).unwrap_or_else(|e| die(&format!("{}: {e}", dir.display())));
    serde_json::from_str(&s).unwrap_or_else(|e| die(&format!("manifest: {e}")))
}

fn ids_of(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().map(|x| x.as_u64().unwrap() as u32).collect()).unwrap_or_default()
}

/// Prompt tokens from --tokens or the oracle manifest of --ref.
fn prompt(a: &Args) -> Vec<u32> {
    if let Some(t) = a.get("tokens") {
        return parse_ids(t);
    }
    if let Some(r) = a.get("ref") {
        return ids_of(&read_manifest(Path::new(r))["tokens"]);
    }
    die("--tokens or --ref is required")
}

// ------------------------------------------------------------------ tensor files

type Tensors = Vec<(String, Vec<i64>, Vec<f32>)>;

fn read_tensors(p: &Path) -> Tensors {
    let mut b = Vec::new();
    std::fs::File::open(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display()))).read_to_end(&mut b).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    while i + 4 <= b.len() {
        let nl = u32_at(&b, i) as usize;
        i += 4;
        let name = String::from_utf8_lossy(&b[i..i + nl]).into_owned();
        i += nl;
        let ty = u32_at(&b, i);
        let nd = u32_at(&b, i + 4) as usize;
        i += 8;
        let ne: Vec<i64> = (0..nd).map(|k| i64::from_le_bytes(b[i + 8 * k..i + 8 * k + 8].try_into().unwrap())).collect();
        i += 8 * nd;
        let n: i64 = ne.iter().product();
        let mut v = Vec::with_capacity(n as usize);
        for k in 0..n as usize {
            let w = &b[i + 4 * k..i + 4 * k + 4];
            v.push(if ty == 0 { f32::from_le_bytes(w.try_into().unwrap()) } else { i32::from_le_bytes(w.try_into().unwrap()) as f32 });
        }
        i += 4 * n as usize;
        out.push((name, ne, v));
    }
    out
}

fn write_tensors(p: &Path, t: &BTreeMap<String, Vec<f32>>) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(p).unwrap());
    for (name, v) in t {
        f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        f.write_all(name.as_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&4u32.to_le_bytes()).unwrap();
        for d in [v.len() as i64, 1, 1, 1] {
            f.write_all(&d.to_le_bytes()).unwrap();
        }
        for x in v {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

// ------------------------------------------------------------------ helpers

struct Setup {
    loaded: model::load::Loaded,
    pool: Pool,
}

fn setup(a: &Args, n_ctx: usize) -> Setup {
    let cpus = a.cpus();
    let t0 = Instant::now();
    let mut loaded = open(&a.model(), cpus.len(), n_ctx, &a.load_opts()).unwrap_or_else(|e| die(&e));
    if let Some(mode) = a.get("shared-experts") {
        if mode != "dma" { die("--shared-experts only accepts dma; omit for normal file weights"); }
        #[cfg(feature="npu-device")]
        {
            let go=a.get("npu-go-file").unwrap_or_else(||die("shared expert relocation requires --npu-go-file guard handshake"));
            println!("{}",json!({"event":"ready_before_device","pid":std::process::id(),"kind":"CPU allocation only, no DSP session"}));
            std::io::stdout().flush().unwrap();
            let until=Instant::now()+Duration::from_secs(30);
            while !Path::new(go).is_file() {
                if Instant::now()>until {die("shared expert guard handshake timed out; no DMA allocated");}
                std::thread::sleep(Duration::from_millis(20));
            }
            let allocator=oxn::shared_memory::SharedAllocator::open(24<<30).unwrap_or_else(|e|die(&e));
            let start=Instant::now();
            let bytes=loaded.ws.relocate_experts(&loaded.plan,&allocator).unwrap_or_else(|e|die(&e));
            let seconds=start.elapsed().as_secs_f64();loaded.ws.load_seconds+=seconds;
            assert_eq!(bytes,allocator.owned_bytes());
            println!("{}",json!({"event":"shared_experts_ready","owned_dma_bytes":bytes,"dsp_mapped_bytes":0,"tensors":loaded.plan.layers.len()*3,"copy_seconds":seconds,"cpu_layout_unchanged":true}));
            std::io::stdout().flush().unwrap();
        }
        #[cfg(not(feature="npu-device"))] die("shared expert weights need npu-device feature");
    }
    let info = &loaded.ws.info;
    eprintln!(
        "load: {:.2} s ({} cache, {} backing, verified {} tensors in {:.2} s{}), {:.2} GiB weights, {} threads on cpus {:?}",
        t0.elapsed().as_secs_f64(),
        if info.cache_hit { "warm" } else { "built" },
        info.backing,
        info.verified_tensors,
        info.verify_seconds,
        if info.cache_hit { String::new() } else { format!(", build {:.1} s", info.built_seconds) },
        loaded.ws.repacked_bytes as f64 / (1u64 << 30) as f64,
        cpus.len(),
        cpus
    );
    let pool = Pool::new(&cpus, Duration::from_micros(a.num("idle-spin-us", 200) as u64)).unwrap_or_else(|e| die(&format!("pool: {e}")));
    Setup { loaded, pool }
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b
}

/// KL(p_ref || q_ours) over the full vocabulary, in f64.
fn kld(r: &[f32], o: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (lr, lo) = (lse(r), lse(o));
    r.iter().zip(o).map(|(&a, &b)| {
        let lp = a as f64 - lr;
        let lq = b as f64 - lo;
        lp.exp() * (lp - lq)
    })
    .sum::<f64>()
    .max(0.0)
}

fn sha_ids(ids: &[u32]) -> String {
    format!("{:016x}", model::weights::hash_bytes(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()))
}

/// Token ids from a file: decimal ids separated by commas and / or whitespace.
fn read_ids_file(path: &str) -> Vec<u32> {
    let t = std::fs::read_to_string(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    t.split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap_or_else(|_| die(&format!("{path}: '{x}' is not a token id")))).collect()
}

/// The bench prompt: --prompt-ids-file, --prompt-ids, else `plen` fixed pseudo-random ids.
fn bench_prompt(a: &Args, plen: usize) -> Vec<u32> {
    if let Some(f) = a.get("prompt-ids-file") {
        return read_ids_file(f);
    }
    match a.get("prompt-ids") {
        Some(t) => parse_ids(t),
        None => (0..plen as u32).map(|i| (i.wrapping_mul(2654435761) % 150_000) + 1000).collect(),
    }
}

/// LONG-CTX `--state-file`: magic, header line (engine layout + KV type + what built it + the ids'
/// hash), the token after the prompt, the engine state ([`Engine::write_state`]), then `extra`.
const STATE_MAGIC: &[u8; 8] = b"OXEBST01";

fn state_header(eng: &Engine, kind: &str, ids: &[u32]) -> String {
    format!("{} kind={kind} n={} ids={}", eng.state_layout(), ids.len(), sha_ids(ids))
}

/// Read a state file into `eng` when its header equals `header`; returns (first token, reader at
/// the extra part). None (with a note) when absent or for another configuration.
fn state_file_load(path: &str, eng: &Engine, header: &str, n_pos: usize) -> Option<(u32, std::io::BufReader<std::fs::File>)> {
    let f = std::fs::File::open(path).ok()?;
    let mut r = std::io::BufReader::with_capacity(8 << 20, f);
    let mut m = [0u8; 8];
    let mut u = [0u8; 4];
    r.read_exact(&mut m).ok()?;
    r.read_exact(&mut u).ok()?;
    let mut h = vec![0u8; u32::from_le_bytes(u) as usize];
    r.read_exact(&mut h).ok()?;
    if &m != STATE_MAGIC || h != header.as_bytes() {
        eprintln!("state-file {path}: other configuration ({}); rebuilding", String::from_utf8_lossy(&h));
        return None;
    }
    r.read_exact(&mut u).ok()?;
    let first = u32::from_le_bytes(u);
    let t0 = Instant::now();
    eng.reset();
    if let Err(e) = eng.read_state(&mut r, n_pos) {
        eng.reset();
        eprintln!("state-file {path}: read failed ({e}); rebuilding");
        return None;
    }
    eprintln!("state-file {path}: {n_pos} positions read in {:.1} s", t0.elapsed().as_secs_f64());
    Some((first, r))
}

/// Write `eng`'s state (+ `extra`) to `path` (via path.pending, then rename).
fn state_file_save(path: &str, eng: &Engine, header: &str, first: u32, extra: &[u8]) {
    let tmp = format!("{path}.pending");
    let t0 = Instant::now();
    let w = (|| -> std::io::Result<()> {
        let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(&tmp)?);
        w.write_all(STATE_MAGIC)?;
        w.write_all(&(header.len() as u32).to_le_bytes())?;
        w.write_all(header.as_bytes())?;
        w.write_all(&first.to_le_bytes())?;
        eng.write_state(&mut w)?;
        w.write_all(extra)?;
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    match w {
        Ok(()) => eprintln!("state-file {path}: {} positions written in {:.1} s", eng.pos(), t0.elapsed().as_secs_f64()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            eprintln!("state-file {path}: write failed: {e}");
        }
    }
}

/// Positions per batched-prefill call when building a long state (the server prefills in chunks
/// too; bounds the per-call hidden rows of the MTP seeding: 4096 x n_embd f32 = 32 MiB).
const STATE_CHUNK: usize = 4096;

/// MTP depth from --spec-draft-n-max / --draft (0 = off).
fn mtp_depth(a: &Args) -> usize {
    let d = a.get("spec-draft-n-max").or(a.get("draft")).map(|v| v.parse::<usize>().unwrap_or_else(|_| die("--spec-draft-n-max: not a number"))).unwrap_or(0);
    if d > 3 {
        die("--spec-draft-n-max: 0..3");
    }
    d
}

fn load_mtp(a: &Args, s: &Setup) -> MtpWeights {
    let t0 = Instant::now();
    // --mtp-from GGUF: take the MTP block (blk.{n_layer}.*, nextn.*) from another GGUF of the same
    // architecture (WP-D5: Qwen3.6's head on Ornith's trunk). Output stays exact (verify is unchanged);
    // only acceptance can change.
    let mtp_src = a.get("mtp-from").map(PathBuf::from).unwrap_or_else(|| a.model());
    // W4-4 b: a sidecar head is hashed from its own bytes (OXE_MODEL_SHA256 names the model)
    let head_opt = LoadOptions { sha_from_env: mtp_src == a.model(), ..a.load_opts() };
    // F-VAE1: setup() built the pool, so this (worker-0) thread is pinned to one CPU; the head's scoped
    // copy / verify threads would inherit that mask. Load with the original mask (decode timing unchanged).
    let mut m = s.pool.unpinned(|| MtpWeights::load(&mtp_src, &s.loaded.plan, &head_opt)).unwrap_or_else(|e| die(&format!("MTP head: {e}")));
    // --draft-vocab RANKING [--draft-k K]: restricted-vocabulary draft head (WP-D3b); without it,
    // a sidecar head's stored vocabulary (oxe.draft_vocab.ids) is used if present
    let rk = a.get("draft-vocab").map(PathBuf::from);
    let k = a.get("draft-k").map(|v| v.parse::<usize>().unwrap_or_else(|_| die("--draft-k")));
    match m.apply_draft_vocab(&mtp_src, &a.model(), rk.as_deref(), k) {
        Ok(Some(rows)) => eprintln!("load: MTP draft head restricted to {rows} rows"),
        Ok(None) => {}
        Err(e) => die(&format!("draft vocabulary: {e}")),
    }
    eprintln!("load: MTP head {:.2} GiB in {:.2} s ({})", m.bytes() as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64(), m.store.info.cache_path);
    m
}

fn stats_json(s: &MtpStats) -> serde_json::Value {
    let mut at: Vec<u64> = s.accepted_at.to_vec();
    while at.len() > 1 && *at.last().unwrap() == 0 {
        at.pop();
    }
    json!({
        "verify_steps": s.steps, "drafted": s.drafted, "accepted": s.accepted,
        "acceptance": s.acceptance(), "tokens_per_verify": s.tokens_per_step(),
        "accepted_at_depth": at, "tokens_out": s.tokens_out,
        "mtp_runs": s.mtp_runs, "mtp_lanes": s.mtp_lanes, "plain_steps": s.plain_steps,
        "time_s": { "draft": s.draft_s, "verify": s.verify_s, "catchup": s.catchup_s,
                    "prefill": s.prefill_s, "prefill_seed": s.prefill_seed_s },
        "prefill_tokens": s.prefill_tokens,
        "time_split": {
            "draft_pct": 100.0 * s.draft_s / (s.draft_s + s.verify_s + s.catchup_s).max(1e-12),
            "verify_pct": 100.0 * s.verify_s / (s.draft_s + s.verify_s + s.catchup_s).max(1e-12),
            "catchup_pct": 100.0 * s.catchup_s / (s.draft_s + s.verify_s + s.catchup_s).max(1e-12),
        },
    })
}

// ------------------------------------------------------------------ commands

fn cmd_gen(a: &Args) {
    let p = prompt(a);
    let n = a.num("n", 64);
    let s = setup(a, a.num("ctx", 4096).max(p.len() + n + 1));
    let eng = Engine::new(&s.loaded.plan, &s.loaded.ws, EngineConfig { n_ctx: a.num("ctx", 4096).max(p.len() + n + 1), logits: false, trace: false, profile: false })
        .unwrap_or_else(|e| die(&e));
    let t0 = Instant::now();
    let mut next = 0;
    for &t in &p {
        next = eng.step(&s.pool, t);
    }
    let tp = t0.elapsed();
    let mut out = Vec::new();
    let t1 = Instant::now();
    for i in 0..n {
        out.push(next);
        if i + 1 < n {
            next = eng.step(&s.pool, next);
        }
    }
    let td = t1.elapsed();
    eprintln!(
        "gen: prompt {} tokens in {:.2} s (token-by-token), {} generated in {:.2} s ({:.2} ms/token)",
        p.len(),
        tp.as_secs_f64(),
        n,
        td.as_secs_f64(),
        td.as_secs_f64() * 1000.0 / n.max(1) as f64
    );
    let mem: serde_json::Map<String, serde_json::Value> = memory_report().into_iter().map(|(k, v)| (k, json!(v))).collect();
    println!("{}", json!({ "prompt": p, "generated": out, "generated_hash": sha_ids(&out), "memory_kb": mem }));
}

fn cmd_trace(a: &Args) {
    let p = prompt(a);
    let pos: Vec<usize> = parse_range(a.get("dump-pos").unwrap_or("0"));
    let out = PathBuf::from(a.get("out").unwrap_or_else(|| die("--out")));
    std::fs::create_dir_all(&out).unwrap();
    let s = setup(a, a.num("ctx", 4096));
    let eng = Engine::new(&s.loaded.plan, &s.loaded.ws, EngineConfig { n_ctx: a.num("ctx", 4096), logits: true, trace: true, profile: false })
        .unwrap_or_else(|e| die(&e));
    let n_gen = a.num("gen", 0);
    let mut toks = p.clone();
    let mut gen = Vec::new();
    let mut lf = std::io::BufWriter::new(std::fs::File::create(out.join("logits.f32")).unwrap());
    let mut i = 0;
    while i < toks.len() {
        let next = eng.step(&s.pool, toks[i]);
        let tr = eng.trace.as_ref().unwrap().take();
        if pos.contains(&i) {
            let m: BTreeMap<String, Vec<f32>> = tr.into_iter().collect();
            write_tensors(&out.join(format!("pos{i}.tensors")), &m);
        }
        for x in eng.logits() {
            lf.write_all(&x.to_le_bytes()).unwrap();
        }
        if i + 1 == toks.len() && gen.len() < n_gen {
            gen.push(next);
            if gen.len() < n_gen {
                toks.push(next);
            }
        }
        i += 1;
    }
    let m = json!({ "tokens": p, "generated": gen, "dump_pos": pos, "engine": "oxe", "threads": s.pool.nthreads() });
    std::fs::write(out.join("manifest.json"), serde_json::to_string_pretty(&m).unwrap()).unwrap();
    eprintln!("trace: {} positions, wrote {}", toks.len(), out.display());
}

fn cmd_compare(a: &Args) {
    let r = PathBuf::from(a.get("ref").unwrap_or_else(|| die("--ref")));
    let o = PathBuf::from(a.get("ours").unwrap_or_else(|| die("--ours")));
    let pos = a.num("pos", 0);
    // decode dumps are pos<P>.tensors, batched-prefill dumps pos<P>.batch.tensors
    let tf = |d: &Path| {
        let p = d.join(format!("pos{pos}.tensors"));
        if p.exists() { p } else { d.join(format!("pos{pos}.batch.tensors")) }
    };
    let rt = read_tensors(&tf(&r));
    let ot: HashMap<String, Vec<f32>> = read_tensors(&tf(&o)).into_iter().map(|(n, _, v)| (n, v)).collect();
    // oracle: the last tensor of each name wins (names are reused, e.g. Kcur raw then roped)
    let mut order: Vec<String> = Vec::new();
    let mut last: HashMap<String, Vec<f32>> = HashMap::new();
    for (n, _, v) in rt {
        if !last.contains_key(&n) {
            order.push(n.clone());
        }
        last.insert(n, v);
    }
    println!("{:<34} {:>8} {:>8} {:>12} {:>12} {:>10}", "tensor", "n", "exact%", "max_abs", "max_rel", "ref_max");
    let mut n_cmp = 0;
    let mut n_exact = 0;
    let mut first_bad: Option<String> = None;
    for name in &order {
        let Some(ov) = ot.get(name) else { continue };
        let rv = &last[name];
        // our kq_soft_max covers n_kv per head like the oracle; compare the common prefix
        let n = rv.len().min(ov.len());
        if rv.len() != ov.len() && !a.flag("all") {
            println!("{name:<34} size {} vs ours {}", rv.len(), ov.len());
        }
        let mut exact = 0usize;
        let (mut ma, mut rmax) = (0f64, 0f64);
        for i in 0..n {
            let (x, y) = (rv[i], ov[i]);
            if x.to_bits() == y.to_bits() || (x == 0.0 && y == 0.0) {
                exact += 1;
            }
            if x.is_finite() {
                rmax = rmax.max((x as f64).abs());
            }
            ma = ma.max(((x as f64) - (y as f64)).abs());
        }
        let rel = if rmax > 0.0 { ma / rmax } else { ma };
        n_cmp += 1;
        if exact == n {
            n_exact += 1;
        } else if first_bad.is_none() {
            first_bad = Some(name.clone());
        }
        if a.flag("all") || exact != n || name.starts_with("result") || name.starts_with("l_out") {
            println!("{name:<34} {n:>8} {:>7.2}% {ma:>12.3e} {rel:>12.3e} {rmax:>10.3e}", 100.0 * exact as f64 / n.max(1) as f64);
        }
    }
    let missing: Vec<&String> = order.iter().filter(|n| !ot.contains_key(*n)).collect();
    println!(
        "compared {n_cmp} tensors at pos {pos}: {n_exact} bit-exact; first non-exact: {}; {} oracle names not traced by oxe",
        first_bad.unwrap_or_else(|| "none".into()),
        missing.len()
    );
}

fn cmd_verify(a: &Args) {
    let r = PathBuf::from(a.get("ref").unwrap_or_else(|| die("--ref")));
    let m = read_manifest(&r);
    let prompt = ids_of(&m["tokens"]);
    let gen = ids_of(&m["generated"]);
    let n_vocab = m["n_vocab"].as_u64().unwrap_or(0) as usize;
    let logits_path = r.join("logits.f32");
    let mut lf = std::fs::File::open(&logits_path).ok();
    let all: Vec<u32> = prompt.iter().chain(gen.iter()).cloned().collect();
    // positions evaluated by the oracle: every prompt token and every generated token but the last
    let n_eval = prompt.len() + gen.len().saturating_sub(1);
    let s = setup(a, a.num("ctx", 4096).max(n_eval + 1));
    let eng = Engine::new(
        &s.loaded.plan,
        &s.loaded.ws,
        EngineConfig { n_ctx: a.num("ctx", 4096).max(n_eval + 1), logits: true, trace: false, profile: false },
    )
    .unwrap_or_else(|e| die(&e));
    let mut rbuf = vec![0f32; n_vocab];
    let mut bytes = vec![0u8; 4 * n_vocab];
    let (mut kl_sum, mut kl_max, mut top1, mut nk, mut exact_rows) = (0f64, 0f64, 0usize, 0usize, 0usize);
    let mut first_div: Option<usize> = None;
    let mut ours_gen = Vec::new();
    let t0 = Instant::now();
    for (p, &tok) in all.iter().take(n_eval).enumerate() {
        let next = eng.step(&s.pool, tok);
        let ol = eng.logits();
        if p + 1 >= prompt.len() && p + 1 - prompt.len() < gen.len() {
            ours_gen.push(next);
            let want = gen[p + 1 - prompt.len()];
            if next != want && first_div.is_none() {
                first_div = Some(p + 1 - prompt.len());
            }
        }
        if let Some(f) = lf.as_mut() {
            if f.read_exact(&mut bytes).is_ok() {
                for (i, c) in bytes.chunks_exact(4).enumerate() {
                    rbuf[i] = f32::from_le_bytes(c.try_into().unwrap());
                }
                let k = kld(&rbuf, ol);
                kl_sum += k;
                kl_max = kl_max.max(k);
                nk += 1;
                top1 += (argmax(&rbuf) == argmax(ol)) as usize;
                exact_rows += rbuf.iter().zip(ol).all(|(x, y)| x.to_bits() == y.to_bits()) as usize;
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let res = json!({
        "ref": r.display().to_string(),
        "prompt_tokens": prompt.len(),
        "generated_ref": gen.len(),
        "greedy_identical": first_div.is_none() && ours_gen.len() == gen.len(),
        "first_divergence": first_div,
        "positions_with_logits": nk,
        "mean_kld": if nk > 0 { kl_sum / nk as f64 } else { f64::NAN },
        "max_kld": kl_max,
        "top1_agreement": if nk > 0 { top1 as f64 / nk as f64 } else { f64::NAN },
        "bit_exact_logit_rows": exact_rows,
        "threads": s.pool.nthreads(),
        "seconds": secs,
    });
    println!("{}", serde_json::to_string_pretty(&res).unwrap());
}

// ------------------------------------------------------------------ M3.2 prefill

/// Parse llama.cpp's `llama_state_seq_get_data` output for a qwen35moe hybrid memory
/// (llama_kv_cache::state_write then llama_memory_recurrent::state_write, one stream, one seq).
fn parse_llama_state(b: &[u8], n_att: usize, n_rec: usize) -> Result<StateDump, String> {
    struct R<'a> {
        b: &'a [u8],
        i: usize,
    }
    impl R<'_> {
        fn take(&mut self, n: usize) -> Result<&[u8], String> {
            if self.i + n > self.b.len() {
                return Err(format!("state file truncated at {}", self.i));
            }
            self.i += n;
            Ok(&self.b[self.i - n..self.i])
        }
        fn u32(&mut self) -> Result<u32, String> {
            Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
        }
        fn u64(&mut self) -> Result<u64, String> {
            Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
        }
    }
    let mut r = R { b, i: 0 };
    r.u32()?; // io magic (llama_context::state_seq_get_data)
    r.u32()?; // seq id
    let n_stream = r.u32()?;
    if n_stream != 1 {
        return Err(format!("n_stream {n_stream}"));
    }
    let cells = r.u32()? as usize;
    for c in 0..cells {
        let pos = r.u32()? as usize;
        let nseq = r.u32()? as usize;
        r.take(12)?; // llama_kv_cell_ext (IMRoPE: has_cell_ext)
        r.take(4 * nseq)?;
        if pos != c {
            return Err(format!("KV cell {c} holds position {pos} (expected a contiguous sequence)"));
        }
    }
    let v_trans = r.u32()?;
    let n_layer = r.u32()? as usize;
    r.u32()?; // n_rot_k
    r.u32()?; // n_rot_v
    if v_trans != 1 || n_layer != n_att {
        return Err(format!("KV header: v_trans {v_trans}, n_layer {n_layer} (expected 1, {n_att})"));
    }
    let f16s = |x: &[u8]| x.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<u16>>();
    let f32s = |x: &[u8]| x.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>();
    let mut k = Vec::new();
    for _ in 0..n_att {
        let ty = r.u32()?;
        let row = r.u64()? as usize;
        if ty != 1 {
            return Err(format!("K type {ty} (expected f16)"));
        }
        k.push(f16s(r.take(cells * row)?));
    }
    let mut v = Vec::new();
    for _ in 0..n_att {
        let ty = r.u32()?;
        let el = r.u32()? as usize;
        let rows = r.u32()? as usize;
        if ty != 1 || el != 2 {
            return Err(format!("V type {ty} el {el}"));
        }
        v.push(f16s(r.take(rows * cells * el)?));
    }
    let rcells = r.u32()? as usize;
    for _ in 0..rcells {
        r.u32()?; // pos
        let nseq = r.u32()? as usize;
        r.take(4 * nseq)?;
    }
    let s_trans = r.u32()?;
    let _n_layer_r = r.u32()?;
    if s_trans != 0 || rcells != 1 {
        return Err(format!("recurrent header: s_trans {s_trans}, cells {rcells}"));
    }
    let mut conv = Vec::new();
    for _ in 0..n_rec {
        let ty = r.u32()?;
        let row = r.u64()? as usize;
        if ty != 0 {
            return Err(format!("conv state type {ty}"));
        }
        conv.push(f32s(r.take(row)?));
    }
    let mut ssm = Vec::new();
    for _ in 0..n_rec {
        let ty = r.u32()?;
        let row = r.u64()? as usize;
        if ty != 0 {
            return Err(format!("ssm state type {ty}"));
        }
        ssm.push(f32s(r.take(row)?));
    }
    if r.i != b.len() {
        return Err(format!("{} trailing bytes in the state file", b.len() - r.i));
    }
    Ok(StateDump { n_pos: cells, k, v, conv, ssm })
}

/// Bitwise comparison of two state dumps: (equal elements, total, first difference).
fn cmp_state(r: &StateDump, o: &StateDump) -> serde_json::Value {
    fn cmp<T: Copy + PartialEq>(name: &str, a: &[Vec<T>], b: &[Vec<T>], eq: impl Fn(T, T) -> bool, first: &mut Option<String>) -> (usize, usize) {
        let (mut same, mut tot) = (0, 0);
        for (l, (x, y)) in a.iter().zip(b).enumerate() {
            if x.len() != y.len() {
                first.get_or_insert(format!("{name}[{l}]: length {} vs {}", x.len(), y.len()));
                tot += x.len().max(y.len());
                continue;
            }
            for (i, (&p, &q)) in x.iter().zip(y).enumerate() {
                tot += 1;
                if eq(p, q) {
                    same += 1;
                } else if first.is_none() {
                    *first = Some(format!("{name}[{l}][{i}]"));
                }
            }
        }
        (same, tot)
    }
    let mut first = None;
    let k = cmp("k", &r.k, &o.k, |a, b| a == b, &mut first);
    let v = cmp("v", &r.v, &o.v, |a, b| a == b, &mut first);
    let c = cmp("conv", &r.conv, &o.conv, |a: f32, b: f32| a.to_bits() == b.to_bits(), &mut first);
    let s = cmp("ssm", &r.ssm, &o.ssm, |a: f32, b: f32| a.to_bits() == b.to_bits(), &mut first);
    let all = r.n_pos == o.n_pos && k.0 == k.1 && v.0 == v.1 && c.0 == c.1 && s.0 == s.1;
    json!({
        "identical": all,
        "n_pos": [r.n_pos, o.n_pos],
        "k_equal": [k.0, k.1], "v_equal": [v.0, v.1], "conv_equal": [c.0, c.1], "ssm_equal": [s.0, s.1],
        "first_difference": first,
    })
}

fn read_logit_rows(p: &Path, n_vocab: usize) -> Vec<Vec<f32>> {
    let b = std::fs::read(p).unwrap_or_default();
    b.chunks_exact(4 * n_vocab).map(|r| r.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()).collect()
}

fn cmd_prefill_verify(a: &Args) {
    let r = PathBuf::from(a.get("ref").unwrap_or_else(|| die("--ref")));
    let m = read_manifest(&r);
    let prompt = ids_of(&m["tokens"]);
    let gen = ids_of(&m["generated"]);
    let n_vocab = m["n_vocab"].as_u64().unwrap_or(0) as usize;
    if m["batch"] != json!(true) || m["last_only"] != json!(true) {
        die("the oracle must be a --batch --last-only dump (dump_ref)");
    }
    let ubatch = a.num("ubatch", m["n_ubatch"].as_u64().unwrap_or(256) as usize);
    let n_ctx = a.num("ctx", m["n_ctx"].as_u64().unwrap_or(4096) as usize);
    let rows = read_logit_rows(&r.join("logits.f32"), n_vocab);
    let s = setup(a, n_ctx);
    let trace_out = a.get("trace-out").map(PathBuf::from);
    let eng = Engine::new(&s.loaded.plan, &s.loaded.ws, EngineConfig { n_ctx, logits: true, trace: trace_out.is_some(), profile: false })
        .unwrap_or_else(|e| die(&e));
    let t0 = Instant::now();
    let first = eng.prefill(&s.pool, &prompt, ubatch).unwrap_or_else(|| die("empty prompt"));
    let secs = t0.elapsed().as_secs_f64();
    if let Some(dir) = &trace_out {
        std::fs::create_dir_all(dir).unwrap();
        let tr: BTreeMap<String, Vec<f32>> = eng.trace.as_ref().unwrap().take().into_iter().collect();
        write_tensors(&dir.join(format!("pos{}.batch.tensors", prompt.len() - 1)), &tr);
        let mm = json!({ "tokens": prompt, "engine": "oxe-prefill", "ubatch": ubatch });
        std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&mm).unwrap()).unwrap();
    }
    let row_cmp = |i: usize, ours: &[f32]| -> (bool, f64, bool) {
        let rr = &rows[i];
        let exact = rr.iter().zip(ours).all(|(x, y)| x.to_bits() == y.to_bits());
        (exact, kld(rr, ours), argmax(rr) == argmax(ours))
    };
    let (pf_exact, pf_kld, pf_top1) = if rows.is_empty() { (false, f64::NAN, false) } else { row_cmp(0, eng.logits()) };
    // state right after the prefill
    let state = match m["state"].as_str().filter(|p| !p.is_empty()) {
        Some(p) => {
            let b = std::fs::read(p).unwrap_or_else(|e| die(&format!("{p}: {e}")));
            let d = &s.loaded.plan.desc;
            match parse_llama_state(&b, d.n_attention(), d.n_recurrent()) {
                Ok(rs) => cmp_state(&rs, &eng.dump_state()),
                Err(e) => json!({ "error": e }),
            }
        }
        None => json!(null),
    };
    // decode after the prefill, teacher-forced on the oracle's greedy tokens
    let mut dec_exact = 0;
    let mut dec_rows = 0;
    let mut greedy = vec![first];
    for g in 0..gen.len().saturating_sub(1) {
        let next = eng.step(&s.pool, gen[g]);
        greedy.push(next);
        if g + 1 < rows.len() {
            dec_rows += 1;
            dec_exact += row_cmp(g + 1, eng.logits()).0 as usize;
        }
    }
    let res = json!({
        "ref": r.display().to_string(),
        "prompt_tokens": prompt.len(),
        "ubatch": ubatch,
        "threads": s.pool.nthreads(),
        "pf_tiles": model::engine::pf_tiles(), "attn": if eng.attn_tiled() { "tiled" } else { "legacy" },
        "effective": model::engine::effective_switches(),
        "prefill_seconds": secs,
        "last_logits_bit_exact": pf_exact,
        "last_logits_kld": pf_kld,
        "last_top1_agrees": pf_top1,
        "state": state,
        "decode_rows_bit_exact": [dec_exact, dec_rows],
        "greedy_identical": greedy.len() == gen.len() && greedy == gen,
        "greedy_ours": greedy,
        // MP-3: routed rows per (layer pass, expert) on this (real-text) fixture; index = rows, capped at 64
        "pf_group": model::engine::pf_group(),
        "rows_per_expert_hist": model::engine::prefill_rows_hist(),
    });
    println!("{}", serde_json::to_string_pretty(&res).unwrap());
}

/// FNV-1a 64 as an io::Write sink (prefill-bench --state-hash).
struct Fnv(u64);
impl std::io::Write for Fnv {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // 8 bytes per multiply (the word-wise variant): the state is hundreds of MB
        let mut h = self.0;
        let mut c = buf.chunks_exact(8);
        for w in &mut c {
            h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x100000001b3);
        }
        for &b in c.remainder() {
            h = (h ^ b as u64).wrapping_mul(0x100000001b3);
        }
        self.0 = h;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Prefill timing (Codex lane). A fixed pseudo-random prompt on top of an optional untimed prefix.
/// `--prompt-len` and `--pf-tiles` take comma lists: every (prompt length, kernel mask) pair is
/// an arm; all arms share one process, one loaded model and one prefilled prefix, and the masks
/// are interleaved rep by rep (order reversed on odd reps: A B B A), so an arm comparison needs no
/// second model load. `--profile` adds the per-phase profile of each arm (timed reps only).
fn cmd_prefill_bench(a: &Args) {
    let list = |k: &str, d: usize| -> Vec<usize> {
        match a.get(k) {
            Some(v) => v.split(',').map(|x| x.trim().parse().unwrap_or_else(|_| die(&format!("--{k}: not a number list")))).collect(),
            None => vec![d],
        }
    };
    let plens = list("prompt-len", 512);
    // arms = every (kernel mask, ubatch group) pair: --pf-tiles m1,m2 x --pf-group g1,g2
    let groups = list("pf-group", model::engine::pf_group());
    let masks: Vec<(u32, usize)> = list("pf-tiles", model::engine::pf_tiles() as usize).into_iter().flat_map(|m| groups.iter().map(move |&g| (m as u32, g))).collect();
    let ubatch = a.num("ubatch", model::engine::UBATCH);
    let reps = a.num("reps", 3).max(1);
    let profile = a.flag("profile");
    // --state-hash: after every timed rep, FNV-1a 64 of the engine state (KV cache prefix, conv and
    // delta-net state, as Engine::write_state serialises it; outside the timer). The value must be
    // the same for every arm, every rep and every process of one (model, prefix, prompt) shape:
    // the production-shape identity gate of the address-only switches.
    let state_hash = a.flag("state-hash");
    // --prefix-len P: prefill P tokens once (untimed), then time the prompts on top of them
    // (recurrent state restored and V rows zeroed between reps; the prefix KV stays)
    let prefix = a.num("prefix-len", 0);
    if a.flag("check-args") {
        // model-free: show how the command line was understood and stop (run this on every ticket
        // command before filing it; a flag missing from the parser's list eats the next argument)
        let known = ["model", "cpu-list", "threads", "ctx", "ubatch", "reps", "prefix-len", "prefix-ids-file", "prompt-len", "pf-tiles", "pf-group", "tokens", "out", "weights", "cache-dir", "verify", "state-file", "npu-prefill", "npu-go-file", "npu-recurrent", "npu-recurrent-mask", "npu-overlap"];
        let unknown: Vec<&String> = a.kv.keys().filter(|k| !known.contains(&k.as_str())).collect();
        println!("{}", json!({ "prompt_len": plens, "arms_mask_group": masks, "ubatch": ubatch, "reps": reps, "prefix_len": prefix, "profile": profile, "state_hash": state_hash, "npu_prefill": a.get("npu-prefill"), "flags": a.flags, "unknown_keys": unknown }));
        if !unknown.is_empty() {
            die("unknown options");
        }
        return;
    }
    let pmax = *plens.iter().max().unwrap();
    let n_ctx = a.num("ctx", 4096).max(prefix + pmax + 1);
    let s = setup(a, n_ctx);
    let mut eng = Engine::new(&s.loaded.plan, &s.loaded.ws, EngineConfig { n_ctx, logits: false, trace: false, profile }).unwrap_or_else(|e| die(&e));
    maybe_npu_prefill(a,&mut eng);
    let snap = if prefix > 0 {
        eng.reset();
        let pre: Vec<u32> = match a.get("prefix-ids-file") {
            Some(f) => {
                let ids = read_ids_file(f);
                if ids.len() < prefix {
                    die(&format!("--prefix-ids-file has {} ids < --prefix-len {prefix}", ids.len()));
                }
                ids[..prefix].to_vec()
            }
            None => (0..prefix as u32).map(|i| (i.wrapping_mul(2246822519) % 150_000) + 1000).collect(),
        };
        model::engine::set_pf_tiles(masks[0].0);
        model::engine::set_pf_group(masks[0].1);
        // --state-file (LONG-CTX): the prefix built once (chunked, untimed), read back by later runs
        let header = state_header(&eng, "prefill-bench-prefix", &pre);
        let loaded = a.get("state-file").and_then(|f| state_file_load(f, &eng, &header, prefix)).is_some();
        if !loaded {
            let t0 = Instant::now();
            if a.get("state-file").is_some() {
                for c in pre.chunks(STATE_CHUNK) {
                    eng.prefill(&s.pool, c, ubatch).unwrap();
                }
            } else {
                eng.prefill(&s.pool, &pre, ubatch).unwrap();
            }
            eprintln!("prefix: {prefix} tokens in {:.3} s (untimed)", t0.elapsed().as_secs_f64());
            if let Some(f) = a.get("state-file") {
                state_file_save(f, &eng, &header, 0, &[]);
            }
        }
        Some(eng.save_recurrent())
    } else {
        None
    };
    // `attn` below is the decode selection. Bind prefill timing to the resolved prefill
    // kernel and its cutoff/parameters, so a requested fa2/fa2x arm cannot be mislabelled.
    let pf_path = eng.pf_attn_path();
    let prefill_attention = json!({
        "kernel": pf_path.name(),
        "cutoff": pf_path.cut_opt(),
        "fa2x": eng.pf_attn_fa2x().map(|p| json!({ "fold": p.fold, "chunk": p.ch })),
    });
    let mut arms = Vec::new();
    let mut all_same = true;
    let mut state_same = true;
    for &plen in &plens {
        let prompt: Vec<u32> = match a.get("tokens") {
            Some(t) => parse_ids(t),
            None => (0..plen as u32).map(|i| (i.wrapping_mul(2654435761) % 150_000) + 1000).collect(),
        };
        let run = |(mask, group): (u32, usize)| -> (f64, u32) {
            model::engine::set_pf_tiles(mask);
            model::engine::set_pf_group(group);
            match &snap {
                Some(sn) => eng.restore_recurrent(sn),
                None => eng.reset(),
            }
            let t0 = Instant::now();
            let first = eng.prefill(&s.pool, &prompt, ubatch).unwrap();
            (t0.elapsed().as_secs_f64(), first)
        };
        let (dt, first0) = run(masks[0]);
        eprintln!("prefill {} tokens on {prefix}, warm-up (mask {} group {}): {:.3} s = {:.1} tok/s", prompt.len(), masks[0].0, masks[0].1, dt, prompt.len() as f64 / dt);
        let nm = masks.len();
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); nm];
        let mut hashes: Vec<Vec<String>> = vec![Vec::new(); nm];
        let mut hists: Vec<Vec<u64>> = vec![Vec::new(); nm];
        // per arm: (work, wait) per phase and the sub accumulators, summed over its timed reps
        let mut prof: Vec<Option<model::engine::PfProfile>> = vec![None; nm];
        for rep in 0..reps {
            let order: Vec<usize> = if rep % 2 == 0 { (0..nm).collect() } else { (0..nm).rev().collect() };
            for mi in order {
                if profile {
                    eng.reset_prefill_profile();
                }
                model::engine::reset_prefill_rows_hist();
                let (dt, first) = run(masks[mi]);
                all_same &= first == first0;
                times[mi].push(dt);
                hists[mi] = model::engine::prefill_rows_hist();
                if state_hash {
                    let mut h = Fnv(0xcbf29ce484222325);
                    eng.write_state(&mut h).unwrap();
                    hashes[mi].push(format!("{:016x}", h.0));
                    state_same &= hashes[mi].last() == hashes[0].first();
                }
                eprintln!("prefill {} tokens on {prefix}, rep {rep}, mask {} group {}: {:.3} s = {:.1} tok/s", prompt.len(), masks[mi].0, masks[mi].1, dt, prompt.len() as f64 / dt);
                if profile {
                    let pr = eng.prefill_profile().unwrap_or_else(|| die("--profile: no profiled prefill ran"));
                    match &mut prof[mi] {
                        None => prof[mi] = Some(pr),
                        Some(acc) => {
                            for i in 0..pr.work_s.len() {
                                acc.work_s[i] += pr.work_s[i];
                                acc.wait_s[i] += pr.wait_s[i];
                            }
                            for i in 0..pr.sub_s.len() {
                                acc.sub_s[i] += pr.sub_s[i];
                            }
                        }
                    }
                }
            }
        }
        for mi in 0..nm {
            let raw = times[mi].clone();
            let mut sorted = raw.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = sorted[sorted.len() / 2];
            let wall: f64 = raw.iter().sum();
            let mut arm = json!({
                "prompt_tokens": prompt.len(),
                "prefix_tokens": prefix,
                "prefill_attention": prefill_attention.clone(),
                "pf_tiles": masks[mi].0,
                "pf_group": masks[mi].1,
                // what the timed call ran with: the by-size setting resolved, 1 when the start is not on a 256 boundary
                "pf_group_effective": if prefix % 256 == 0 && ubatch % 256 == 0 && eng.attn_tiled() { let (t1, t2) = model::engine::pf_group_thresholds(); if masks[mi].1 == 0 { model::engine::group_by_size(prompt.len(), t1, t2) } else { masks[mi].1 } } else { 1 },
                "threads": s.pool.nthreads(),
                "seconds": raw,
                "median_seconds": med,
                "tok_s_median": prompt.len() as f64 / med,
                "ms_per_token_median": med * 1e3 / prompt.len() as f64,
                "first_token": first0,
            });
            if state_hash {
                arm["state_fnv"] = json!(hashes[mi]);
            }
            arm["rows_per_expert_hist_last_rep"] = json!(hists[mi]);
            if let Some(pr) = &prof[mi] {
                // seconds are thread-seconds (summed over the threads); pct is of the phases' total
                let total: f64 = pr.work_s.iter().zip(&pr.wait_s).map(|(w, t)| w + t).sum();
                let pct = |w: f64, t: f64| if total > 0.0 { 100.0 * (w + t) / total } else { 0.0 };
                let phases: Vec<serde_json::Value> = (0..pr.names.len())
                    .filter(|&i| pr.work_s[i] + pr.wait_s[i] > 0.0)
                    .map(|i| json!({ "name": pr.names[i], "work_s": pr.work_s[i], "wait_s": pr.wait_s[i], "pct_of_thread_time": pct(pr.work_s[i], pr.wait_s[i]) }))
                    .collect();
                let sub: serde_json::Map<String, serde_json::Value> = pr.sub_names.iter().zip(&pr.sub_s).map(|(n, s)| (n.to_string(), json!(s))).collect();
                arm["profile"] = json!({ "threads": pr.threads, "timed_reps": raw.len(), "wall_s_total": wall, "phases": phases, "sub": sub });
                let tot = |i: usize| pr.work_s[i] + pr.wait_s[i];
                let mut order: Vec<usize> = (0..pr.names.len()).filter(|&i| tot(i) > 0.0).collect();
                order.sort_by(|&x, &y| tot(y).partial_cmp(&tot(x)).unwrap());
                eprintln!("profile: {} tokens on {prefix}, mask {} group {}: {} threads, {} timed reps, wall {:.3} s (thread-seconds below)", prompt.len(), masks[mi].0, masks[mi].1, pr.threads, raw.len(), wall);
                eprintln!("  {:<26} {:>9} {:>9} {:>9} {:>7}", "phase", "work_s", "wait_s", "total_s", "pct");
                for i in order {
                    eprintln!("  {:<26} {:>9.4} {:>9.4} {:>9.4} {:>6.2}%", pr.names[i], pr.work_s[i], pr.wait_s[i], tot(i), pct(pr.work_s[i], pr.wait_s[i]));
                }
                for (n, s) in pr.sub_names.iter().zip(&pr.sub_s) {
                    eprintln!("  sub {:<22} {:>9.4}", n, s);
                }
            }
            arms.push(arm);
        }
    }
    let mem: serde_json::Map<String, serde_json::Value> = memory_report().into_iter().map(|(k, v)| (k, json!(v))).collect();
    // the first arm's numbers stay at the top level (the pre-moe-prefill JSON shape)
    let mut res = arms[0].clone();
    res["attn"] = json!(if eng.attn_tiled() { "tiled" } else { "legacy" });
    res["ubatch"] = json!(ubatch);
    res["n_ctx"] = json!(eng.n_ctx);
    res["threads"] = json!(s.pool.nthreads());
    res["effective"] = json!(model::engine::effective_switches());
    #[cfg(feature="npu-prefill")]
    {res["npu_prefill"]=eng.npu_prefill_stats().map(npu_stats_json).unwrap_or(serde_json::Value::Null);}
    res["cpus"] = json!(s.pool.cpus());
    res["first_token_identical_across_arms"] = json!(all_same);
    res["kv_vpad"] = json!(model::engine::kv_vpad());
    res["kv_type"] = json!(eng.kv_type().name());
    res["state_file"] = json!(a.get("state-file"));
    if state_hash {
        res["state_identical_across_arms"] = json!(state_same);
    }
    res["memory_kb"] = json!(mem);
    res["arms"] = json!(arms);
    let out = serde_json::to_string_pretty(&res).unwrap();
    if let Some(p) = a.get("out") {
        std::fs::write(p, &out).unwrap();
    }
    println!("{out}");
    if !all_same {
        die("first token differs between arms");
    }
    if !state_same {
        die("engine state differs between arms");
    }
}

/// `oxe verify-bench --model M [--prompt-ids ..|--prompt-len N] --batched-prompt --lanes 1,2,3,4,5,6,8 --runs 48`:
/// WP-D2 phase b: the cost of one verify run of n lanes (n tokens through the target in one batch-invariant
/// run), the multiplier any drafter (prompt lookup, MTP) pays per verify. Arms interleave (ABBA per round);
/// each run verifies n greedy-continuation tokens and commits all n, so the context advances realistically.
/// Output: ms per run (median), and cost relative to one plain step, per n.
/// `--profile` (OD-1): per lane count, work / barrier-wait thread-ms and bytes per run per phase kind
/// (dense GEMV, router, expert gate/up and down, attention, delta-net GEMV / recurrence, head); the
/// ms of a profiled run are not throughput numbers.
fn cmd_verify_bench(a: &Args) {
    let lanes: Vec<usize> = parse_range(a.get("lanes").unwrap_or("1,2,3,4,5,6,8"));
    let maxl = *lanes.iter().max().unwrap_or(&1);
    if maxl > model::engine::MAX_LANES {
        die("--lanes: at most MAX_LANES");
    }
    let runs = a.num("runs", 48);
    let plen = a.num("prompt-len", 512);
    let prompt = bench_prompt(a, plen);
    let rounds = 2;
    let need = prompt.len() + 16 + rounds * 2 * runs * lanes.iter().sum::<usize>() + 8;
    let n_ctx = a.num("ctx", 4096).max(need);
    let s = setup(a, n_ctx);
    // --profile (OD-1): per-phase work / barrier-wait thread-ms and bytes per run, per lane count
    let profile = a.flag("profile");
    let eng = Engine::with_options(&s.loaded.plan, &s.loaded.ws, EngineConfig { n_ctx, logits: false, trace: false, profile }, EngineOptions { verify_lanes: maxl, mtp: None, kv_type: None })
        .unwrap_or_else(|e| die(&e));
    let mut next = if a.flag("batched-prompt") {
        eng.prefill(&s.pool, &prompt, model::engine::UBATCH).unwrap()
    } else {
        let mut x = 0;
        for &t in &prompt {
            x = eng.step(&s.pool, t);
        }
        x
    };
    for _ in 0..16 {
        next = eng.step(&s.pool, next);
    }
    // the greedy continuation (what a perfect drafter proposes: correlated tokens, realistic expert
    // overlap between lanes), generated once with plain steps, then replayed through verify runs
    let per = (runs / (rounds * 2)).max(1);
    let total: usize = rounds * 2 * per * lanes.iter().sum::<usize>();
    let snap = eng.save_recurrent();
    let mut g = vec![next];
    while g.len() < total + 1 {
        let x = eng.step(&s.pool, *g.last().unwrap());
        g.push(x);
    }
    eng.restore_recurrent(&snap);
    let mut ms: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    let mut profs: BTreeMap<usize, Vec<model::engine::Prof>> = BTreeMap::new();
    let mut order = lanes.clone();
    let mut at = 0;
    for _ in 0..rounds * 2 {
        for &n in &order {
            if profile {
                eng.reset_profile();
            }
            for _ in 0..per {
                let toks = &g[at..at + n];
                let t = Instant::now();
                let out = eng.verify(&s.pool, toks);
                ms.entry(n).or_default().push(t.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(&out[..n - 1], &g[at + 1..at + n], "verify disagrees with the plain greedy steps");
                eng.commit(n);
                at += n;
            }
            if profile {
                let pr = eng.profile_sum();
                let acc = profs.entry(n).or_insert_with(|| vec![model::engine::Prof::default(); pr.len()]);
                for (a, p) in acc.iter_mut().zip(&pr) {
                    for i in 0..a.phase_work.len() {
                        a.phase_work[i] += p.phase_work[i];
                        a.phase_wait[i] += p.phase_wait[i];
                        a.phase_bytes[i] += p.phase_bytes[i];
                    }
                    for j in 0..a.sub_work.len() {
                        a.sub_work[j] += p.sub_work[j];
                    }
                    a.tokens += p.tokens;
                }
            }
        }
        order.reverse(); // ABBA
    }
    let med = |v: &Vec<f64>| {
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };
    let base = ms.get(&1).map(med).unwrap_or(f64::NAN);
    let mut rows: Vec<serde_json::Value> = ms.iter().map(|(n, v)| json!({"lanes": n, "ms_median": med(v), "runs": v.len(), "cost_vs_plain": med(v) / base})).collect();
    for r in &rows {
        eprintln!("verify-bench: {} lanes: {:.3} ms/run, {:.3}x one plain step", r["lanes"], r["ms_median"].as_f64().unwrap(), r["cost_vs_plain"].as_f64().unwrap());
    }
    if profile {
        // profiled timings are not throughput (LEARNINGS T12b): the table sizes phases, the ms above are under the profiler
        let hz = tick_hz() as f64;
        for r in rows.iter_mut() {
            let n = r["lanes"].as_u64().unwrap() as usize;
            if let Some(pr) = profs.get(&n) {
                let (ph, sub) = decode_profile(pr, hz, &format!("verify-bench {n} lanes"));
                r["profile"] = json!(ph);
                r["profile_sub"] = sub;
            }
        }
    }
    let res = json!({"model": a.model().display().to_string(), "prompt_tokens": prompt.len(), "n_ctx": eng.n_ctx, "end_pos": eng.pos(), "arms": rows, "profiled": profile,
        "cpus": s.pool.cpus(), "engine_commit": option_env!("OXE_GIT_COMMIT").unwrap_or("unknown")});
    if let Some(p) = a.get("out") {
        std::fs::write(p, serde_json::to_string_pretty(&res).unwrap()).unwrap();
    }
    println!("{}", serde_json::to_string_pretty(&res).unwrap());
}

/// OD-1 decode phase profile from the engine's per-thread accumulators (summed over `tokens`
/// runs; a verify run of n lanes is one run). Per phase kind: work and barrier-wait ms per run as
/// the mean over threads (`work_ms`, `wait_ms`, the pre-OD-1 fields) and summed over threads
/// (`work_thread_ms`, `wait_thread_ms`), bytes asked for per run and the rate over the phase's
/// mean wall share. Prints a table to stderr headed by `label`.
fn decode_profile(pr: &[model::engine::Prof], hz: f64, label: &str) -> (Vec<serde_json::Value>, serde_json::Value) {
    let nth = pr.len().max(1) as f64;
    let nt = pr.first().map(|p| p.tokens).unwrap_or(0).max(1) as f64;
    let ms = |t: u64| t as f64 / hz * 1000.0 / nt;
    let mut rows = Vec::new();
    let (mut tw, mut tt, mut tb) = (0.0, 0.0, 0.0);
    eprintln!("profile {label}: {} threads, {} runs (per run: thread-ms summed over threads; wall = per-thread mean)", pr.len(), nt);
    eprintln!("  {:<14} {:<26} {:>9} {:>9} {:>7} {:>8} {:>10} {:>7}", "phase", "class", "work_tms", "wait_tms", "idle%", "wall_ms", "MB/run", "GB/s");
    for k in PhaseKind::ALL {
        let i = k.index();
        let work_t: f64 = pr.iter().map(|p| ms(p.phase_work[i])).sum();
        let wait_t: f64 = pr.iter().map(|p| ms(p.phase_wait[i])).sum();
        let bytes: f64 = pr.iter().map(|p| p.phase_bytes[i] as f64).sum::<f64>() / nt;
        let wall = (work_t + wait_t) / nth;
        let gbs = if wall > 0.0 { bytes / (wall * 1e-3) / 1e9 } else { 0.0 };
        let idle = if work_t + wait_t > 0.0 { 100.0 * wait_t / (work_t + wait_t) } else { 0.0 };
        tw += work_t;
        tt += wait_t;
        tb += bytes;
        eprintln!("  {:<14} {:<26} {:>9.3} {:>9.3} {:>6.1}% {:>8.3} {:>10.2} {:>7.1}", k.name(), model::engine::phase_class(k), work_t, wait_t, idle, wall, bytes / 1e6, gbs);
        rows.push(json!({ "phase": k.name(), "class": model::engine::phase_class(k),
            "work_ms": work_t / nth, "wait_ms": wait_t / nth, "idle_pct": idle,
            "work_thread_ms": work_t, "wait_thread_ms": wait_t, "bytes_per_run": bytes, "gb_s": gbs }));
    }
    let wall = (tw + tt) / nth;
    eprintln!("  {:<14} {:<26} {:>9.3} {:>9.3} {:>6.1}% {:>8.3} {:>10.2} {:>7.1}", "total", "", tw, tt, if tw + tt > 0.0 { 100.0 * tt / (tw + tt) } else { 0.0 }, wall, tb / 1e6, if wall > 0.0 { tb / (wall * 1e-3) / 1e9 } else { 0.0 });
    let mut sub = serde_json::Map::new();
    for (j, name) in model::engine::PROF_SUB_NAMES.iter().enumerate() {
        let v: f64 = pr.iter().map(|p| ms(p.sub_work[j])).sum();
        eprintln!("  sub {:<56} {:>9.3} thread-ms/run", name, v);
        sub.insert(name.to_string(), json!(v));
    }
    (rows, serde_json::Value::Object(sub))
}

/// `oxe kbench-mt [--sets "0;0-3;0-7;0-11;0,4,8;0-1,4-5,8-9"] [--secs 1.0]`: MODEL-FREE scaling of the
/// prefill kernels over core sets (MAGIC: where do 12 threads lose against one?). Synthetic weights at
/// the prefill shapes; every thread is pinned (sched::Pool) and takes work units from a shared
/// counter, as the engine does. Workloads, each reported as aggregate and per-thread GMAC/s:
///   l1      dense Q4_K tile on a problem that stays in each thread's L1/L2 (8 rows x 2048 x 64 tokens,
///           private copies): pure compute, shows frequency / heat scaling only
///   dense   one delta-net layer's qkv projection per pass: 8192 x 2048 Q4_K (9.4 MB) x 256 tokens in
///           64-row units, 8 different weight matrices cycled (75 MB: weights stream from DRAM),
///           activations shared read-only (0.6 MB), outputs to a shared 8 MB buffer
///   routed  routed experts: 256 experts of 512 x 2048 Q3_K (115 MB), `--rows` routed rows each (default
///           16), pair tiles, one expert per work unit, rows from a shared activation set
///   routed-cl  the same work, but each expert is split by weight rows over the threads of ONE cluster
///           (4 consecutive cores of the set): the cluster's cores share the expert's weights in their L2
/// Timing only (hold all build locks; DEV windows; not authoritative).
fn cmd_kbench_mt(a: &Args) {
    use kernels_sys::{ActType, QType};
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let secs: f64 = a.get("secs").map(|v| v.parse().unwrap_or(1.0)).unwrap_or(1.0);
    let rows_per_expert = a.num("rows", 16);
    let sets: Vec<Vec<usize>> = a.get("sets").unwrap_or("0;0-3;0-7;0-11;0,4,8;0-1,4-5,8-9").split(';').map(|l| parse_cpu_list(l).unwrap_or_else(|e| die(&format!("--sets: {e}")))).collect();
    let only: Option<Vec<&str>> = a.get("work").map(|w| w.split(',').collect());
    let want = |w: &str| only.as_ref().map(|o| o.contains(&w)).unwrap_or(true);
    // synthetic row-major weight rows with sane f16 scales, repacked
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut h = [0u16; 1];
    kernels_sys::f32_to_f16(&[0.01], &mut h);
    let d16 = h[0].to_le_bytes();
    let mut synth = |ty: QType, nrows: usize, k: usize| -> Vec<u8> {
        let bb = ty.block_bytes();
        let mut v: Vec<u8> = (0..nrows * ty.row_bytes(k)).map(|_| rnd() as u8).collect();
        for blk in v.chunks_mut(bb) {
            match ty {
                QType::Q4K | QType::Q5K => {
                    blk[0..2].copy_from_slice(&d16);
                    blk[2..4].copy_from_slice(&d16);
                }
                QType::Q3K => blk[108..110].copy_from_slice(&d16),
                QType::Q6K => blk[208..210].copy_from_slice(&d16),
                QType::Q8_0 => blk[0..2].copy_from_slice(&d16),
            }
        }
        let mut packed = vec![0u8; v.len()];
        kernels_sys::repack(ty, &v, &mut packed, nrows, k);
        packed
    };
    let k = 2048usize;
    let ntok = 256usize;
    let x: Vec<f32> = (0..ntok * k).map(|i| ((i * 2654435761usize) % 2001) as f32 / 1000.0 - 1.0).collect();
    // dense activations: 4-row interleaved groups; routed: per-row from_float
    let rbk = ActType::Q8K.bytes(k);
    let mut act4 = vec![0u8; ntok * rbk];
    let mut actr = vec![0u8; ntok * rbk];
    for g in (0..ntok).step_by(4) {
        // SAFETY: 4 rows of k floats in, one 4-row group out.
        unsafe { kernels_sys::quantize_mat4_raw(ActType::Q8K, x[g * k..].as_ptr(), act4[g * rbk..].as_mut_ptr(), k) };
    }
    for r in 0..ntok {
        kernels_sys::quantize_act(ActType::Q8K, &x[r * k..(r + 1) * k], &mut actr[r * rbk..(r + 1) * rbk]);
    }
    let (dq, drows, nmat) = (QType::Q4K, 8192usize, 8usize);
    let dense_w: Vec<Vec<u8>> = if want("dense") { (0..nmat).map(|_| synth(dq, drows, k)).collect() } else { Vec::new() };
    let (eq, erows, ne) = (QType::Q3K, 512usize, 256usize);
    let ebytes = eq.row_bytes(k) * erows;
    let experts: Vec<u8> = if want("routed") || want("routed-cl") {
        let one = synth(eq, erows, k);
        let mut all = vec![0u8; ne * ebytes];
        for e in 0..ne {
            all[e * ebytes..(e + 1) * ebytes].copy_from_slice(&one);
        }
        all
    } else {
        Vec::new()
    };
    let l1_w = synth(dq, 8, k);
    struct Sh<T>(*mut T);
    // SAFETY: disjoint writes per work unit (documented at each use).
    unsafe impl<T> Sync for Sh<T> {}
    unsafe impl<T> Send for Sh<T> {}
    let mut dense_out = vec![0f32; ntok * drows];
    let mut routed_out = vec![0f32; ne.max(1) * rows_per_expert * erows];
    let mut results = Vec::new();
    println!("{:<10} {:<22} {:>3} {:>10} {:>10} {:>8}", "work", "cpus", "thr", "GMAC/s", "per thr", "vs 1");
    let mut base: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for cpus in &sets {
        let nt = cpus.len();
        let pool = Pool::new(cpus, Duration::from_micros(200)).unwrap_or_else(|e| die(&format!("pool: {e}")));
        for work in ["l1", "dense", "routed", "routed-cl"] {
            if !want(work) {
                continue;
            }
            let ctr = AtomicUsize::new(0);
            let done = AtomicUsize::new(0); // units of work finished (for the MAC count)
            let (dout, rout) = (Sh(dense_out.as_mut_ptr()), Sh(routed_out.as_mut_ptr()));
            let (dout, rout) = (&dout, &rout);
            let passes = AtomicUsize::new(0);
            let mac_per_unit: f64 = match work {
                "l1" => (8 * k * 64) as f64,
                "dense" => (64 * k * ntok) as f64,
                _ => (erows * k * rows_per_expert) as f64,
            };
            let t0 = Instant::now();
            let mut reps = 0usize;
            // each pool.run is one "pass" (a layer); repeat until `secs` have passed
            while t0.elapsed().as_secs_f64() < secs || reps < 2 {
                let pass = passes.fetch_add(1, Relaxed);
                ctr.store(0, Relaxed);
                pool.run(|ctx| {
                    match work {
                        "l1" => {
                            // private small problem: 8 rows x 64 tokens, repeated
                            let mut out = vec![0f32; 64 * 8];
                            for _ in 0..200 {
                                // SAFETY: 8 rows of l1_w, 64 tokens (16 groups) of act4, private out.
                                unsafe { kernels_sys::gemm_tile_raw(dq, k, out.as_mut_ptr(), 8, l1_w.as_ptr(), act4.as_ptr(), 64, 8) };
                                done.fetch_add(1, Relaxed);
                            }
                        }
                        "dense" => {
                            let w = &dense_w[pass % nmat];
                            loop {
                                let u = ctr.fetch_add(1, Relaxed);
                                if u >= drows / 64 {
                                    break;
                                }
                                // SAFETY: the unit owns output columns u*64..u*64+64 of every token row.
                                unsafe { kernels_sys::gemm_tile_raw(dq, k, dout.0.add(u * 64), drows, w.as_ptr().add(dq.group_offset(u * 64, k)), act4.as_ptr(), ntok, 64) };
                                done.fetch_add(1, Relaxed);
                            }
                        }
                        "routed" => {
                            let mut xq = vec![0u32; rows_per_expert.div_ceil(2) * (k / 256) * kernels_sys::PF_XBLK_BYTES / 4];
                            loop {
                                let e = ctr.fetch_add(1, Relaxed);
                                if e >= ne {
                                    break;
                                }
                                let rp: Vec<*const u8> = (0..rows_per_expert).map(|j| actr[((e * 7 + j * 13) % ntok) * rbk..].as_ptr()).collect();
                                // SAFETY: the unit owns the rows_per_expert output rows of expert e.
                                let orow: Vec<*mut f32> = (0..rows_per_expert).map(|j| unsafe { rout.0.add((e * rows_per_expert + j) * erows) }).collect();
                                unsafe {
                                    kernels_sys::mmid_tile_pack_raw(&rp, k, xq.as_mut_ptr() as *mut u8);
                                    kernels_sys::mmid_tile_raw(eq, k, &orow, experts.as_ptr().add(e * ebytes), xq.as_ptr() as *const u8, erows);
                                }
                                done.fetch_add(1, Relaxed);
                            }
                        }
                        _ => {
                            // routed-cl: clusters of 4 consecutive workers; a cluster takes whole experts, and
                            // its workers take 64-row slices of that expert (weights shared in the cluster's L2)
                            let ncl = nt.div_ceil(4);
                            let cl = ctx.tid / 4;
                            let in_cl = ((cl + 1) * 4).min(nt) - cl * 4;
                            let me = ctx.tid % 4;
                            let mut xq = vec![0u32; rows_per_expert.div_ceil(2) * (k / 256) * kernels_sys::PF_XBLK_BYTES / 4];
                            // static split: expert e belongs to cluster e % ncl; rows slice by worker
                            for e in (cl..ne).step_by(ncl) {
                                let rp: Vec<*const u8> = (0..rows_per_expert).map(|j| actr[((e * 7 + j * 13) % ntok) * rbk..].as_ptr()).collect();
                                let r0 = (erows / 64 * me / in_cl) * 64;
                                let r1 = (erows / 64 * (me + 1) / in_cl) * 64;
                                if r1 > r0 {
                                    // SAFETY: worker `me` owns output columns r0..r1 of the expert's rows.
                                    let orow: Vec<*mut f32> = (0..rows_per_expert).map(|j| unsafe { rout.0.add((e * rows_per_expert + j) * erows + r0) }).collect();
                                    unsafe {
                                        kernels_sys::mmid_tile_pack_raw(&rp, k, xq.as_mut_ptr() as *mut u8);
                                        kernels_sys::mmid_tile_raw(eq, k, &orow, experts.as_ptr().add(e * ebytes + eq.group_offset(r0, k)), xq.as_ptr() as *const u8, r1 - r0);
                                    }
                                }
                                if me == 0 {
                                    done.fetch_add(1, Relaxed);
                                }
                            }
                        }
                    }
                });
                reps += 1;
            }
            let dt = t0.elapsed().as_secs_f64();
            let gmac = done.load(Relaxed) as f64 * mac_per_unit / dt / 1e9;
            let b = *base.entry(work).or_insert(gmac / nt as f64);
            let cl = cpus.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
            println!("{:<10} {:<22} {:>3} {:>10.1} {:>10.1} {:>8.2}", work, cl, nt, gmac, gmac / nt as f64, gmac / nt as f64 / b);
            results.push(json!({ "work": work, "cpus": cpus, "threads": nt, "gmac_s": gmac, "gmac_s_per_thread": gmac / nt as f64, "passes": reps, "seconds": dt }));
        }
    }
    if let Some(p) = a.get("out") {
        std::fs::write(p, serde_json::to_string_pretty(&json!({ "rows_per_expert": rows_per_expert, "results": results })).unwrap()).unwrap();
    }
}

/// Engine construction-time switches an arm of `bench --arms` may set. OXE_VL (F96 b): the lane-GEMV mask is read
/// by `Engine::new` (`dec_vl`), so VL arms share one process and one prefill instead of re-prefilling per process.
const ARM_KEYS: [&str; 6] = ["OXE_DEC_ATTN_PAD", "OXE_DEC_FUSED_DN", "OXE_KV_TYPE", "OXE_KV8_FUSED", "OXE_ATTN_STATS", "OXE_VL"];

/// `name[:KEY=V[,KEY=V..]]|name2..` -> arms (dies on an unknown switch or a malformed entry).
fn parse_arms(spec: &str) -> Vec<(String, Vec<(String, String)>)> {
    spec
        .split('|')
        .map(|arm| {
            let (name, envs) = arm.split_once(':').unwrap_or((arm, ""));
            let envs: Vec<(String, String)> = envs
                .split(',')
                .filter(|e| !e.is_empty())
                .map(|e| {
                    let (k, v) = e.split_once('=').unwrap_or_else(|| die(&format!("--arms: '{e}' is not KEY=VALUE")));
                    if !ARM_KEYS.contains(&k) {
                        die(&format!("--arms: unknown switch {k} (known: {})", ARM_KEYS.join(", ")));
                    }
                    (k.to_string(), v.to_string())
                })
                .collect();
            if name.is_empty() {
                die("--arms: empty arm name");
            }
            (name.to_string(), envs)
        })
        .collect()
}

/// `oxe bench --arms SPEC`: several decode switch arms in ONE process (test economy: the 8K / 16K
/// decode arms of ticket 049 re-prefilled the prompt in every process for ~4 s of timed decode).
/// SPEC = `name[:KEY=V[,KEY=V..]]|name2..`, KEY from the engine's construction-time switches
/// (OXE_DEC_ATTN_PAD, OXE_DEC_FUSED_DN, OXE_KV_TYPE, OXE_KV8_FUSED, OXE_ATTN_STATS, OXE_VL). One engine per
/// arm on the same weights. The prompt is prefilled ONCE per KV type; the other arms of that type
/// load the written state (bit-identical for layout / scheduling switches; an int8 arm prefills
/// itself). Then `--reps` rounds (default 4, the first `--drop` = 1 dropped): every arm restores
/// the prompt-end state, decodes `--warmup` + `--tokens` greedy tokens from the same first token,
/// arms in order on even rounds and reversed on odd ones (A B .. / .. B A). Reports per arm the
/// kept medians, every round's `generated_hash`, and whether the hashes agree within a KV type.
/// `--profile`: the decode phase profile of each arm's last round.
fn cmd_bench_arms(a: &Args, spec: &str) {
    let arms = parse_arms(spec);
    const KEYS: [&str; 6] = ARM_KEYS;
    let tokens = a.num("tokens", 128);
    let warmup = a.num("warmup", 16);
    let reps = a.num("reps", 4).max(1);
    let drop = a.num("drop", 1).min(reps - 1);
    let plen = a.num("prompt-len", 16);
    let profile = a.flag("profile");
    let prompt = bench_prompt(a, plen);
    if prompt.is_empty() {
        die("--arms needs a prompt");
    }
    let n_ctx = a.num("ctx", 4096).max(prompt.len() + warmup + tokens + 1);
    let s = setup(a, n_ctx);
    let plan = &s.loaded.plan;
    let mut engines = Vec::new();
    for (name, envs) in &arms {
        for k in KEYS {
            std::env::remove_var(k);
        }
        for (k, v) in envs {
            std::env::set_var(k, v);
        }
        let eng = Engine::new(plan, &s.loaded.ws, EngineConfig { n_ctx, logits: false, trace: false, profile }).unwrap_or_else(|e| die(&format!("arm {name}: {e}")));
        engines.push(eng);
    }
    for k in KEYS {
        std::env::remove_var(k);
    }
    // prompt-end state per KV type: prefill once, the other arms of the type load it
    let mut states: Vec<(&'static str, Vec<u8>, u32, f64)> = Vec::new(); // (kv, state, first token, prefill s)
    for (i, eng) in engines.iter().enumerate() {
        let kv = eng.kv_type().name();
        match states.iter().find(|st| st.0 == kv) {
            Some(st) => {
                eng.read_state(&mut &st.1[..], prompt.len()).unwrap_or_else(|e| die(&format!("arm {}: read_state: {e}", arms[i].0)));
            }
            None => {
                let t0 = Instant::now();
                let first = eng.prefill(&s.pool, &prompt, model::engine::UBATCH).unwrap();
                let secs = t0.elapsed().as_secs_f64();
                let mut buf = Vec::new();
                eng.write_state(&mut buf).unwrap();
                eprintln!("bench --arms: prefilled {} tokens for kv {kv} in {secs:.1} s (arm {}), state {} MB", prompt.len(), arms[i].0, buf.len() >> 20);
                states.push((kv, buf, first, secs));
            }
        }
    }
    let na = arms.len();
    let mut medians: Vec<Vec<f64>> = vec![Vec::new(); na];
    let mut means: Vec<Vec<f64>> = vec![Vec::new(); na];
    let mut hashes: Vec<Vec<String>> = vec![Vec::new(); na];
    let mut last_ids: Vec<Vec<u32>> = vec![Vec::new(); na];
    for rep in 0..reps {
        let order: Vec<usize> = if rep % 2 == 0 { (0..na).collect() } else { (0..na).rev().collect() };
        for i in order {
            let eng = &engines[i];
            let st = states.iter().find(|st| st.0 == eng.kv_type().name()).unwrap();
            eng.reset();
            eng.read_state(&mut &st.1[..], prompt.len()).unwrap_or_else(|e| die(&format!("arm {}: read_state: {e}", arms[i].0)));
            let mut next = st.2;
            let mut gen = Vec::new();
            for _ in 0..warmup {
                gen.push(next);
                next = eng.step(&s.pool, next);
            }
            if rep + 1 == reps {
                eng.reset_profile();
            }
            let mut ms = Vec::with_capacity(tokens);
            for _ in 0..tokens {
                gen.push(next);
                let t = Instant::now();
                next = eng.step(&s.pool, next);
                ms.push(t.elapsed().as_secs_f64() * 1000.0);
            }
            gen.push(next);
            let mean = ms.iter().sum::<f64>() / ms.len() as f64;
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = ms[ms.len() / 2];
            eprintln!("bench --arms: round {rep}{} arm {:<12} {:.3} ms/token median, hash {}", if rep < drop { " (dropped)" } else { "" }, arms[i].0, med, sha_ids(&gen));
            medians[i].push(med);
            means[i].push(mean);
            hashes[i].push(sha_ids(&gen));
            last_ids[i] = gen;
        }
    }
    let hz = tick_hz() as f64;
    let mut out_arms = Vec::new();
    let mut hash_ok = true;
    for i in 0..na {
        let eng = &engines[i];
        let kv = eng.kv_type().name();
        // within a KV type every arm and round must generate the same ids
        let ref_hash = (0..na).find(|&j| engines[j].kv_type().name() == kv).map(|j| hashes[j][0].clone()).unwrap();
        let same = hashes[i].iter().all(|h| *h == ref_hash);
        hash_ok &= same;
        let mut kept: Vec<f64> = medians[i][drop..].to_vec();
        kept.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = if kept.len() % 2 == 1 { kept[kept.len() / 2] } else { (kept[kept.len() / 2 - 1] + kept[kept.len() / 2]) / 2.0 };
        let mut arm = json!({
            "name": arms[i].0, "env": arms[i].1.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>(), "kv": kv,
            "ms_per_token_median_per_round": medians[i], "ms_per_token_mean_per_round": means[i], "rounds_dropped": drop,
            "ms_per_token_median": med, "tok_s": 1000.0 / med,
            "generated_hash_per_round": hashes[i], "generated_hash": hashes[i][0], "hash_equal_within_kv": same,
            "generated_ids": last_ids[i],
            "attn_col_stride": eng.attn_col_stride(),
            "attn_prob_blocks": eng.attn_stats().map(|(b, z, t)| json!({"blocks": b, "zero": z, "below_2m10": t})),
        });
        if profile {
            let (ph, sub) = decode_profile(&eng.profile_sum(), hz, "bench");
            arm["profile_last_round"] = json!(ph);
            arm["profile_sub_last_round"] = sub;
        }
        out_arms.push(arm);
    }
    let mem: serde_json::Map<String, serde_json::Value> = memory_report().into_iter().map(|(k, v)| (k, json!(v))).collect();
    let res = json!({
        "engine": "oxe", "engine_commit": option_env!("OXE_GIT_COMMIT").unwrap_or("unknown"),
        "effective": model::engine::effective_switches(),
        "model": a.model().display().to_string(), "cpus": s.pool.cpus(), "threads": s.pool.nthreads(),
        "prompt_tokens": prompt.len(), "warmup": warmup, "tokens": tokens, "n_ctx": n_ctx, "rounds": reps, "rounds_dropped": drop,
        "order": "arms in order on even rounds, reversed on odd rounds",
        "prefill": states.iter().map(|st| json!({ "kv": st.0, "seconds": st.3, "state_bytes": st.1.len(), "first_token": st.2 })).collect::<Vec<_>>(),
        "hash_equal_within_kv": hash_ok,
        "arms": out_arms,
        "memory_kb": mem,
        "kernel_cflags": kernels_sys::KERNEL_CFLAGS,
    });
    let text = serde_json::to_string_pretty(&res).unwrap();
    if let Some(p) = a.get("out") {
        std::fs::write(p, &text).unwrap();
    }
    println!("{text}");
    if !hash_ok {
        die("generated ids differ between arms or rounds of one KV type");
    }
}

fn cmd_bench(a: &Args) {
    if let Some(spec) = a.get("arms") {
        return cmd_bench_arms(a, spec);
    }
    if mtp_depth(a) > 0 {
        return cmd_bench_mtp(a);
    }
    let tokens = a.num("tokens", 128);
    let warmup = a.num("warmup", 16);
    let plen = a.num("prompt-len", 16);
    let n_ctx = a.num("ctx", 4096).max(plen + warmup + tokens + 1);
    let profile = a.flag("profile");
    let s = setup(a, n_ctx);
    let plan = &s.loaded.plan;
    let eng = Engine::new(plan, &s.loaded.ws, EngineConfig { n_ctx, logits: false, trace: false, profile }).unwrap_or_else(|e| die(&e));
    // a fixed deterministic prompt (ids from --tokens, else a fixed pseudo-random sequence)
    let prompt = bench_prompt(a, plen);
    let mut next = 0;
    if a.flag("batched-prompt") {
        // batched prefill of the prompt (long-context decode: --prompt-len 16384 --batched-prompt)
        if !prompt.is_empty() {
            next = eng.prefill(&s.pool, &prompt, model::engine::UBATCH).unwrap();
        }
    } else {
        for &t in &prompt {
            next = eng.step(&s.pool, t);
        }
    }
    let mut gen = Vec::new();
    for _ in 0..warmup {
        gen.push(next);
        next = eng.step(&s.pool, next);
    }
    eng.reset_profile();
    let b0 = s.pool.barrier_episodes();
    let mut ms = Vec::with_capacity(tokens);
    for _ in 0..tokens {
        gen.push(next);
        let t = Instant::now();
        next = eng.step(&s.pool, next);
        ms.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    gen.push(next);
    let barriers = (s.pool.barrier_episodes() - b0) as f64 / tokens as f64;
    let mut sorted = ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let p90 = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)];
    let budget = plan.budget(&s.loaded.tensors);
    let hz = tick_hz() as f64;
    let mut phases = Vec::new();
    let mut profile_sub = serde_json::Value::Null;
    if profile {
        let (ph, sub) = decode_profile(&eng.profile_sum(), hz, "bench");
        phases = ph;
        profile_sub = sub;
    }
    let mem: serde_json::Map<String, serde_json::Value> = memory_report().into_iter().map(|(k, v)| (k, json!(v))).collect();
    let info = &s.loaded.ws.info;
    let res = json!({
        "engine": "oxe", "engine_commit": option_env!("OXE_GIT_COMMIT").unwrap_or("unknown"),
        "effective": model::engine::effective_switches(),
        "model": a.model().display().to_string(),
        "cpus": s.pool.cpus(), "threads": s.pool.nthreads(),
        "prompt_tokens": prompt.len(), "warmup": warmup, "tokens": tokens, "n_ctx": eng.n_ctx,
        "ms_per_token": { "median": med, "mean": mean, "min": sorted[0], "p90": p90 },
        "tok_s": 1000.0 / med,
        "barriers_per_token": barriers, "barriers_per_token_plan": eng.barriers_per_token(),
        "attn": if eng.attn_fused() { "fused" } else if eng.attn_tiled() { "tiled" } else { "legacy" }, "attn_tiled_min_kv": eng.attn_tiled_min_kv(),
        "attn_fused_chunk": eng.attn_fused_chunk(),
        "attn_col_stride": eng.attn_col_stride(),
        "attn_prob_blocks": eng.attn_stats().map(|(b, z, t)| json!({"blocks": b, "zero": z, "below_2m10": t,
            "zero_frac": z as f64 / b.max(1) as f64, "below_2m10_frac": t as f64 / b.max(1) as f64})),
        "bytes_per_token": { "weights": budget.weights(), "state_rw": budget.state_rw, "total_at_ctx": budget.total() },
        "generated_ids": gen, "generated_hash": sha_ids(&gen),
        "load": { "seconds": s.loaded.ws.load_seconds, "cache_hit": info.cache_hit, "backing": info.backing,
                  "cache_path": info.cache_path, "built_seconds": info.built_seconds, "shared_expert_owned_bytes": s.loaded.ws.shared_expert_bytes(),
                  "verified_tensors": info.verified_tensors, "weights_bytes": s.loaded.ws.repacked_bytes },
        "memory_kb": mem,
        "profile": phases,
        "profile_sub": profile_sub,
        "kernel_cflags": kernels_sys::KERNEL_CFLAGS,
    });
    let text = serde_json::to_string_pretty(&res).unwrap();
    if let Some(p) = a.get("out") {
        std::fs::write(p, &text).unwrap();
    }
    println!("{text}");
    eprintln!(
        "bench: {:.2} ms/token median (p90 {:.2}), {:.2} tok/s, {:.0} barriers/token, ids hash {}, VmHWM {:.2} GiB",
        med,
        p90,
        1000.0 / med,
        barriers,
        sha_ids(&gen),
        memory_report().iter().find(|(k, _)| k == "VmHWM").map(|(_, v)| *v as f64 / 1048576.0).unwrap_or(0.0)
    );
}

/// `oxe vocab-freq --model M --files-from LIST --out RANKING [--weights-from LIST2:W]`: token
/// frequencies of a local corpus under the model's tokenizer (no weights loaded). LIST holds one
/// file path per line; files that are not UTF-8 are skipped. RANKING: one id per line, most
/// frequent first (ties by id), then every unseen id in id order, so it covers the vocabulary.
/// `--json OUT` writes counts and per-list totals. WP-D3b (restricted draft head).
/// `oxe kv-kld --model M (--text-file F | --prompt-ids ..) --kv k8|v8|kv8 --score 8192:1024,24576:1024 [--ctx 32768]`:
/// O-13 gate. Two engines on the same weights in one process, f16 KV (reference) and the int8 KV type, are fed
/// the same token stream: batched prefill up to each scoring window, then teacher-forced single steps through
/// the window; at every scored position KLD(f16 || int8) of the full next-token distributions (f64) and top-1
/// agreement. Output: mean, p50, p99, max KLD and top-1 per window and overall (JSON to --out).
fn cmd_kv_kld(a: &Args) {
    let kv = model::engine::KvType::parse(a.get("kv").unwrap_or("kv8")).unwrap_or_else(|| die("--kv f16|k8|v8|kv8"));
    let mut windows: Vec<(usize, usize)> = a
        .get("score")
        .unwrap_or("8192:1024")
        .split(',')
        .map(|w| {
            let (s, n) = w.split_once(':').unwrap_or_else(|| die("--score START:LEN,.."));
            (s.parse().unwrap_or_else(|_| die("--score")), n.parse().unwrap_or_else(|_| die("--score")))
        })
        .collect();
    windows.sort();
    let need = windows.iter().map(|&(s, n)| s + n + 1).max().unwrap_or(0);
    let toks: Vec<u32> = match (a.get("text-file"), a.get("prompt-ids")) {
        (Some(f), _) => {
            // Tokenise only as much text as the windows need (the whole 11 MB wikitext file was
            // tokenised before: QC-4 2026-11-10 spent its time before the first KLD line). Start
            // with ~8 bytes per needed token and double until there are enough tokens.
            let t_tok = Instant::now();
            let tk = tokenizer::Tokenizer::from_gguf(a.model()).unwrap_or_else(|e| die(&format!("tokenizer: {e:?}")));
            let text = std::fs::read_to_string(f).unwrap_or_else(|e| die(&format!("{f}: {e}")));
            let mut bytes = (need * 8).max(4096).min(text.len());
            let toks = loop {
                while bytes < text.len() && !text.is_char_boundary(bytes) {
                    bytes += 1;
                }
                let t = tk.encode(&text[..bytes], false, false);
                if t.len() >= need || bytes >= text.len() {
                    break t;
                }
                bytes = (bytes * 2).min(text.len());
            };
            eprintln!("kv-kld: tokenised {} of {} bytes -> {} tokens (need {need}) in {:.2} s", bytes, text.len(), toks.len(), t_tok.elapsed().as_secs_f64());
            toks
        }
        (None, Some(t)) => parse_ids(t),
        _ => die("--text-file F or --prompt-ids"),
    };
    if toks.len() < need {
        die(&format!("text has {} tokens, the windows need {need}", toks.len()));
    }
    let n_ctx = a.num("ctx", 32768).max(need);
    if a.flag("dry-run") {
        // model-free: tokenisation and the plan only
        let steps: usize = windows.iter().map(|w| w.1).sum();
        println!("{}", json!({ "kv": kv.name(), "n_ctx": n_ctx, "tokens_available": toks.len(), "tokens_needed": need, "windows": windows,
            "prefill_tokens_per_engine": need - 1 - steps, "decode_steps_per_engine": steps, "engines": 2 }));
        return;
    }
    let s = setup(a, n_ctx);
    let cfg = EngineConfig { n_ctx, logits: true, trace: false, profile: false };
    let mk = |t| Engine::with_options(&s.loaded.plan, &s.loaded.ws, cfg.clone(), EngineOptions { verify_lanes: 1, mtp: None, kv_type: Some(t) }).unwrap_or_else(|e| die(&e));
    let ea = mk(model::engine::KvType::F16);
    let eb = mk(kv);
    let lsm = |l: &[f32]| -> Vec<f64> {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln() + m;
        l.iter().map(|&x| x as f64 - z).collect()
    };
    let argmax = |l: &[f32]| l.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
    let mut pos = 0usize;
    let mut all: Vec<f64> = Vec::new();
    let mut agree = 0usize;
    let mut per = Vec::new();
    let t0 = Instant::now();
    for &(start, len) in &windows {
        if start > pos {
            // progress lines: a run that hits a time limit shows where it was
            let tp = Instant::now();
            ea.prefill(&s.pool, &toks[pos..start], model::engine::UBATCH);
            eprintln!("kv-kld: f16 engine prefilled {}..{start} in {:.1} s", pos, tp.elapsed().as_secs_f64());
            let tp = Instant::now();
            eb.prefill(&s.pool, &toks[pos..start], model::engine::UBATCH);
            eprintln!("kv-kld: {} engine prefilled {}..{start} in {:.1} s", kv.name(), pos, tp.elapsed().as_secs_f64());
            pos = start;
        }
        let (mut w, mut wa) = (Vec::with_capacity(len), 0usize);
        for _ in 0..len {
            ea.step(&s.pool, toks[pos]);
            eb.step(&s.pool, toks[pos]);
            pos += 1;
            let (la, lb) = (ea.logits(), eb.logits());
            let (pa, pb) = (lsm(la), lsm(lb));
            let kl: f64 = pa.iter().zip(&pb).map(|(&x, &y)| x.exp() * (x - y)).sum();
            w.push(kl.max(0.0));
            wa += (argmax(la) == argmax(lb)) as usize;
        }
        let st = kld_stats(&w);
        eprintln!("kv-kld {}: window {start}+{len}: mean {:.3e} p99 {:.3e} max {:.3e} top1 {:.4}", kv.name(), st["mean"].as_f64().unwrap_or(0.0), st["p99"].as_f64().unwrap_or(0.0), st["max"].as_f64().unwrap_or(0.0), wa as f64 / len as f64);
        per.push(json!({"start": start, "len": len, "kld": st, "top1": wa as f64 / len as f64}));
        all.extend(&w);
        agree += wa;
    }
    let res = json!({"model": a.model().display().to_string(), "kv": kv.name(), "n_ctx": n_ctx, "tokens": toks.len(),
        "windows": per, "overall": {"positions": all.len(), "kld": kld_stats(&all), "top1": agree as f64 / all.len().max(1) as f64},
        "seconds": t0.elapsed().as_secs_f64(), "engine_commit": option_env!("OXE_GIT_COMMIT").unwrap_or("unknown")});
    let text = serde_json::to_string_pretty(&res).unwrap();
    if let Some(p) = a.get("out") {
        std::fs::write(p, &text).unwrap();
    }
    println!("{text}");
}

fn kld_stats(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return json!(null);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| s[((s.len() as f64 - 1.0) * f).round() as usize];
    json!({"mean": s.iter().sum::<f64>() / s.len() as f64, "p50": q(0.5), "p99": q(0.99), "max": s[s.len() - 1]})
}

fn cmd_vocab_freq(a: &Args) {
    let tok = tokenizer::Tokenizer::from_gguf(a.model()).unwrap_or_else(|e| die(&format!("tokenizer: {e:?}")));
    let n_vocab = tok.n_vocab();
    let mut counts = vec![0u64; n_vocab];
    let lists: Vec<String> = a.get("files-from").unwrap_or_else(|| die("--files-from LIST[,LIST:WEIGHT..]")).split(',').map(str::to_string).collect();
    let mut totals = Vec::new();
    for spec in &lists {
        // LIST or LIST:WEIGHT (integer weight per token occurrence, e.g. to balance code and prose)
        let (path, w) = match spec.rsplit_once(':') {
            Some((p, w)) if w.parse::<u64>().is_ok() => (p.to_string(), w.parse::<u64>().unwrap()),
            _ => (spec.clone(), 1),
        };
        let list = std::fs::read_to_string(&path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
        let (mut files, mut skipped, mut ntok) = (0u64, 0u64, 0u64);
        for f in list.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let Ok(text) = std::fs::read_to_string(f) else {
                skipped += 1;
                continue;
            };
            files += 1;
            for t in tok.encode(&text, false, false) {
                if (t as usize) < n_vocab {
                    counts[t as usize] += w;
                    ntok += 1;
                }
            }
        }
        eprintln!("vocab-freq: {path}: {files} files ({skipped} skipped), {ntok} tokens, weight {w}");
        totals.push(json!({"list": path, "files": files, "skipped": skipped, "tokens": ntok, "weight": w}));
    }
    let mut ids: Vec<u32> = (0..n_vocab as u32).collect();
    ids.sort_by(|&x, &y| counts[y as usize].cmp(&counts[x as usize]).then(x.cmp(&y)));
    let seen = counts.iter().filter(|&&c| c > 0).count();
    let out = a.get("out").unwrap_or_else(|| die("--out RANKING"));
    let text: String = ids.iter().map(|i| format!("{i}\n")).collect();
    std::fs::write(out, text).unwrap_or_else(|e| die(&format!("{out}: {e}")));
    eprintln!("vocab-freq: {seen} of {n_vocab} ids seen; ranking -> {out}");
    if let Some(j) = a.get("json") {
        let top: Vec<serde_json::Value> = ids.iter().take(seen).map(|&i| json!([i, counts[i as usize]])).collect();
        std::fs::write(j, serde_json::to_string(&json!({"lists": totals, "seen": seen, "n_vocab": n_vocab, "counts": top})).unwrap()).unwrap();
    }
}

fn cmd_cache(a: &Args) {
    let s = setup(a, 256);
    let info = &s.loaded.ws.info;
    println!(
        "{}",
        json!({ "cache_path": info.cache_path, "cache_hit": info.cache_hit, "model_sha256": info.model_sha256,
                "built_seconds": info.built_seconds, "verify_seconds": info.verify_seconds,
                "verified_tensors": info.verified_tensors, "weights_bytes": s.loaded.ws.repacked_bytes,
                "memory_kb": memory_report().into_iter().map(|(k, v)| (k, json!(v))).collect::<serde_json::Map<_, _>>() })
    );
}

/// `oxe bench` with MTP speculation (depth = --spec-draft-n-max). Same prompt, warm-up and
/// token count as the plain bench; the timed region produces exactly `tokens` greedy tokens
/// (the last verify step may overshoot and is cut), and `generated_hash` must equal the
/// plain bench's hash for the same model (exactness). ms_per_token is the effective
/// wall time per produced token (speculation makes per-token times bursty, so no median).
fn cmd_bench_mtp(a: &Args) {
    let depth = mtp_depth(a);
    let tokens = a.num("tokens", 128);
    let warmup = a.num("warmup", 16);
    let plen = a.num("prompt-len", 16);
    let n_ctx = a.num("ctx", 4096).max(plen + warmup + tokens + depth + 2);
    let s = setup(a, n_ctx);
    let mw = load_mtp(a, &s);
    let plan = &s.loaded.plan;
    let eng = Engine::with_options(plan, &s.loaded.ws, EngineConfig { n_ctx, logits: false, trace: false, profile: false }, EngineOptions { verify_lanes: depth + 1, mtp: Some(&mw), kv_type: None })
        .unwrap_or_else(|e| die(&e));
    let mut drv = MtpDriver::new(&eng, depth).unwrap_or_else(|e| die(&e));
    let prompt = bench_prompt(a, plen);
    let tp = Instant::now();
    // --state-file (LONG-CTX): the prompt state built once, read back by later runs
    let state_file = a.get("state-file");
    if state_file.is_some() && !a.flag("batched-prompt") {
        die("--state-file needs --batched-prompt");
    }
    let header = state_header(&eng, &format!("bench-mtp head={}", a.get("mtp-from").unwrap_or("model")), &prompt);
    let mut loaded = None;
    if let Some(f) = state_file {
        if let Some((first, mut r)) = state_file_load(f, &eng, &header, prompt.len()) {
            let ne = plan.desc.n_embd;
            let rd = |r: &mut dyn Read, n: usize| -> Vec<u8> {
                let mut b = vec![0u8; n];
                r.read_exact(&mut b).unwrap_or_else(|e| die(&format!("state-file: MTP part: {e}")));
                b
            };
            let q = u64::from_le_bytes(rd(&mut r, 8).try_into().unwrap()) as usize;
            let n = u64::from_le_bytes(rd(&mut r, 8).try_into().unwrap()) as usize;
            let toks: Vec<u32> = rd(&mut r, n * 4).chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let f: Vec<f32> = rd(&mut r, (n + 1) * ne * 4).chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let (h, hl) = f.split_at(n * ne);
            drv.import_state(&eng, q, toks, h.to_vec(), hl).unwrap_or_else(|e| die(&format!("state-file: {e}")));
            loaded = Some(first);
        }
    }
    let state_loaded = loaded.is_some();
    // --batched-prompt: the server's prompt path (batched prefill + KV-only MTP seeding); else the
    // token-wise verify-run prefill (slow for long prompts)
    let mut x = if let Some(first) = loaded {
        first
    } else if state_file.is_some() {
        // built in chunks (as the server prefills), then saved
        let mut x = 0;
        for c in prompt.chunks(STATE_CHUNK) {
            x = drv.prefill_batched(&eng, &s.pool, &s.pool, c, model::engine::UBATCH);
        }
        let (q, toks, h, hl) = drv.export_state();
        let mut extra = Vec::new();
        extra.extend((q as u64).to_le_bytes());
        extra.extend((toks.len() as u64).to_le_bytes());
        toks.iter().for_each(|t| extra.extend(t.to_le_bytes()));
        h.iter().chain(&hl).for_each(|v| extra.extend(v.to_le_bytes()));
        state_file_save(state_file.unwrap(), &eng, &header, x, &extra);
        x
    } else if a.flag("batched-prompt") {
        drv.prefill_batched(&eng, &s.pool, &s.pool, &prompt, model::engine::UBATCH)
    } else {
        drv.prefill(&eng, &s.pool, &prompt)
    };
    let prefill_ms = tp.elapsed().as_secs_f64() * 1000.0;
    // --plain-steps (LONG-CTX MTP-by-context pricing): decode depth 0 on the same MTP engine and
    // state (step_plain keeps the MTP catch-up in sync: the server's path beyond --mtp-max-ctx)
    let plain = a.flag("plain-steps");
    let step = |drv: &mut MtpDriver, x: u32, out: &mut Vec<u32>| if plain { drv.step_plain(&eng, &s.pool, x, out) } else { drv.step(&eng, &s.pool, x, out) };
    let mut gen = Vec::new();
    let mut out = Vec::new();
    // warm-up: produce `warmup` tokens (gen holds every token fed to the target, as in the plain bench)
    while gen.len() < warmup {
        gen.push(x);
        step(&mut drv, x, &mut out);
        gen.extend_from_slice(&out[..out.len() - 1]);
        x = *out.last().unwrap();
    }
    drv.stats = MtpStats::default();
    let target = warmup + tokens;
    let g0 = gen.len();
    let t0 = Instant::now();
    let mut step_ms = Vec::new();
    while gen.len() < target {
        gen.push(x);
        let t = Instant::now();
        step(&mut drv, x, &mut out);
        step_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        gen.extend_from_slice(&out[..out.len() - 1]);
        x = *out.last().unwrap();
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let produced = gen.len() - g0;
    // the plain bench's gen has warmup + tokens + 1 entries (the last is the next token)
    gen.push(x);
    gen.truncate(warmup + tokens + 1);
    let eff = total_ms / produced.max(1) as f64;
    let mem: serde_json::Map<String, serde_json::Value> = memory_report().into_iter().map(|(k, v)| (k, json!(v))).collect();
    let info = &s.loaded.ws.info;
    let cm = eng.ctx_mem();
    let ctx_mem_json = json!({ "kv": cm.kv, "kv_per_pos": cm.kv_per_pos, "recurrent": cm.recurrent, "decode_scratch": cm.decode_scratch });
    let res = json!({
        "engine": "oxe", "engine_commit": option_env!("OXE_GIT_COMMIT").unwrap_or("unknown"),
        "effective": model::engine::effective_switches(),
        "model": a.model().display().to_string(),
        "cpus": s.pool.cpus(), "threads": s.pool.nthreads(),
        "spec": { "type": "draft-mtp", "draft_n_max": if plain { 0 } else { depth }, "plain_steps": plain },
        "prompt_tokens": prompt.len(), "prefill_ms": prefill_ms, "warmup": warmup, "tokens": tokens, "n_ctx": eng.n_ctx,
        "state_file": state_file, "state_loaded": state_loaded, "kv_type": eng.kv_type().name(),
        "ctx_mem_bytes": ctx_mem_json,
        "tokens_produced_timed": produced,
        "ms_per_token": { "effective": eff, "median": eff, "mean": eff },
        "tok_s": 1000.0 / eff,
        "ms_per_verify_step": { "mean": step_ms.iter().sum::<f64>() / step_ms.len().max(1) as f64 },
        "mtp": stats_json(&drv.stats),
        "generated_ids": gen, "generated_hash": sha_ids(&gen),
        "load": { "seconds": s.loaded.ws.load_seconds, "cache_hit": info.cache_hit, "backing": info.backing,
                  "cache_path": info.cache_path, "weights_bytes": s.loaded.ws.repacked_bytes, "mtp_bytes": mw.bytes() },
        "memory_kb": mem,
        "kernel_cflags": kernels_sys::KERNEL_CFLAGS,
    });
    let text = serde_json::to_string_pretty(&res).unwrap();
    if let Some(p) = a.get("out") {
        std::fs::write(p, &text).unwrap();
    }
    println!("{text}");
    eprintln!(
        "bench mtp depth {depth}: {:.2} ms/token effective, {:.2} tok/s, acceptance {:.1}%, {:.2} tokens/verify, ids hash {}",
        eff,
        1000.0 / eff,
        100.0 * drv.stats.acceptance(),
        drv.stats.tokens_per_step(),
        sha_ids(&gen)
    );
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// M3.1 exactness gate. Per reference prompt (oracle manifest tokens): plain greedy decode of
/// --n tokens (ids + every logit row), then for each depth the MTP driver from a reset
/// engine: ids identical, and the logit row of every position a verify run committed (and the
/// prefill's last row) bit-identical to the plain row of that position. Also reports whether
/// the plain ids equal the oracle's (llama.cpp) greedy continuation.
fn cmd_mtp_check(a: &Args) {
    let refs: Vec<PathBuf> = a.get("refs").unwrap_or_else(|| die("--refs DIR,DIR,..")).split(',').map(PathBuf::from).collect();
    let n = a.num("n", 256);
    let depths: Vec<usize> = parse_range(a.get("depths").unwrap_or("1,2,3"));
    // prompt: batched prefill (Engine::prefill vs MtpDriver::prefill_batched, default) or
    // token by token (plain steps vs verify runs of depth+1 tokens)
    let batched = match a.get("prefill").unwrap_or("batched") {
        "batched" => true,
        "tokenwise" => false,
        x => die(&format!("--prefill {x}: batched|tokenwise")),
    };
    let ubatch = a.num("ubatch", model::engine::UBATCH);
    let maxd = *depths.iter().max().unwrap_or(&1);
    if maxd == 0 || maxd > 3 {
        die("--depths: 1..3");
    }
    let prompts: Vec<(PathBuf, Vec<u32>, Vec<u32>)> = refs.iter().map(|r| { let m = read_manifest(r); (r.clone(), ids_of(&m["tokens"]), ids_of(&m["generated"])) }).collect();
    let maxp = prompts.iter().map(|p| p.1.len()).max().unwrap_or(1);
    let n_ctx = a.num("ctx", 4096).max(maxp + n + maxd + 2);
    let s = setup(a, n_ctx);
    let mw = load_mtp(a, &s);
    let plan = &s.loaded.plan;
    let cfg = EngineConfig { n_ctx, logits: true, trace: false, profile: false };
    let plain = Engine::new(plan, &s.loaded.ws, cfg.clone()).unwrap_or_else(|e| die(&e));
    let engs: Vec<(usize, Engine)> = depths
        .iter()
        .map(|&d| (d, Engine::with_options(plan, &s.loaded.ws, cfg.clone(), EngineOptions { verify_lanes: d + 1, mtp: Some(&mw), kv_type: None }).unwrap_or_else(|e| die(&e))))
        .collect();
    let mut results = Vec::new();
    let mut all_ok = true;
    for (dir, prompt, oracle_gen) in &prompts {
        plain.reset();
        let mut x = 0;
        if batched {
            x = plain.prefill(&s.pool, prompt, ubatch).unwrap();
        } else {
            for &t in prompt {
                x = plain.step(&s.pool, t);
            }
        }
        let mut ids = vec![x];
        let mut rows = vec![plain.logits().to_vec()];
        while ids.len() < n {
            x = plain.step(&s.pool, x);
            ids.push(x);
            rows.push(plain.logits().to_vec());
        }
        let k = oracle_gen.len().min(n);
        let oracle_match = oracle_gen[..k] == ids[..k];
        let mut per_depth = Vec::new();
        for (d, e) in &engs {
            e.reset();
            let mut drv = MtpDriver::new(e, *d).unwrap_or_else(|er| die(&er));
            let t0 = Instant::now();
            let (mut x, last_lane) = if batched {
                (drv.prefill_batched(e, &s.pool, &s.pool, prompt, ubatch), 0)
            } else {
                (drv.prefill(e, &s.pool, prompt), (prompt.len() - 1) % e.verify_lanes())
            };
            let mut rows_checked = 0usize;
            let mut rows_bad = 0usize;
            let mut first_bad: Option<usize> = None;
            if bits_eq(e.logits_lane(last_lane), &rows[0]) {
                rows_checked += 1;
            } else {
                rows_bad += 1;
                first_bad.get_or_insert(0);
            }
            let mut got = vec![x];
            let mut out = Vec::new();
            while got.len() < n {
                let p = e.pos();
                drv.step(e, &s.pool, x, &mut out);
                for (i, &t) in out.iter().enumerate() {
                    let row = p + i + 1 - prompt.len();
                    if row < rows.len() {
                        if bits_eq(e.logits_lane(i), &rows[row]) {
                            rows_checked += 1;
                        } else {
                            rows_bad += 1;
                            first_bad.get_or_insert(row);
                        }
                    }
                    got.push(t);
                }
                x = *out.last().unwrap();
            }
            got.truncate(n);
            let first_div = got.iter().zip(&ids).position(|(a, b)| a != b);
            let ok = first_div.is_none() && rows_bad == 0;
            all_ok &= ok;
            eprintln!(
                "mtp-check {} depth {d}: ids {} , {rows_checked} rows bit-identical, {rows_bad} differ; acceptance {:.1}%, {:.2} tokens/verify",
                dir.file_name().unwrap().to_string_lossy(),
                if first_div.is_none() { "identical" } else { "DIFFER" },
                100.0 * drv.stats.acceptance(),
                drv.stats.tokens_per_step()
            );
            per_depth.push(json!({
                "depth": d, "ids_identical": first_div.is_none(), "first_divergence": first_div,
                "logit_rows_bit_identical": rows_checked, "logit_rows_differ": rows_bad, "first_bad_row": first_bad,
                "seconds": t0.elapsed().as_secs_f64(), "mtp": stats_json(&drv.stats),
            }));
        }
        results.push(json!({ "ref": dir.display().to_string(), "prompt_tokens": prompt.len(), "n": n,
            "plain_ids_hash": sha_ids(&ids), "plain_matches_oracle_greedy": oracle_match, "oracle_compared": k, "depths": per_depth }));
    }
    let res = json!({ "model": a.model().display().to_string(), "threads": s.pool.nthreads(), "prefill": if batched { format!("batched ubatch {ubatch}") } else { "tokenwise".into() }, "all_identical": all_ok, "prompts": results });
    let text = serde_json::to_string_pretty(&res).unwrap();
    if let Some(p) = a.get("out") {
        std::fs::write(p, &text).unwrap();
    }
    println!("{text}");
    if !all_ok {
        std::process::exit(1);
    }
}

#[cfg(feature="npu-prefill")]
fn npu_stats_json(s:model::npu_prefill::Stats)->serde_json::Value{
    json!({"prepared":s.prepared,"skipped":s.skipped,"calls":s.calls,"rows":s.rows,
        "reserved_dma_bytes":s.reserved_dma_bytes,"occupied_weight_bytes":s.occupied_weight_bytes,
        "host_scratch_bytes":s.host_scratch_bytes,"first_error":s.first_error,"recurrent_mode":s.recurrent_mode,"recurrent_mask":format!("{:016x}",s.recurrent_mask),"cpu_f32_matrices":s.cpu_f32_matrices,"gemm_flops":s.gemm_flops,"overlap_shared":s.overlap_shared,"shared_chain_units":s.shared_chain_units,"shared_chain_s":s.shared_chain_s,"expert_units_started_during_shared":s.expert_units_started_during_shared,"host_phases":{"input_s":s.input_s,"device_run_s":s.device_run_s,"output_s":s.output_s,"commit_s":s.commit_s,"projection_s":s.projection_s,"lock_wait_thread_s":s.lock_wait_s},
        "original_audit":s.original_audit.iter().map(|a|json!({"tensor":a.tensor,"k":a.k,"o":a.o,"samples":a.samples,
            "relative_rmse":(a.squared_error/a.squared_reference.max(1e-30)).sqrt(),"max_error":a.max_error,"max_input":a.max_input,
            "max_output":a.max_output,"max_reference":a.max_reference})).collect::<Vec<_>>(),
        "accelerator":s.accelerator.map(|a|json!({"pack_s":a.pack_s,"gemm_s":a.gemm_s,"dsp_us":a.dsp_us,"calls":a.calls,"allocated_bytes":a.allocated_bytes,"weights_bytes":a.weights_bytes}))})
}

fn maybe_npu_prefill(a:&Args,eng:&mut Engine){
    let Some(mode)=a.get("npu-prefill")else{return};
    #[cfg(not(feature="npu-prefill"))]{let _=(eng,mode);die("build with the npu-prefill feature");}
    #[cfg(feature="npu-prefill")]{
        if !["device","emu"].contains(&mode){die("--npu-prefill must be device or emu");}
        let g=gguf::Gguf::open(a.model()).unwrap_or_else(|e|die(&e.to_string()));
        let dir=model::weights::cache_dir(a.load_opts().cache_dir.as_deref()).unwrap_or_else(|e|die(&e));
        // Check THIS source independently of an inherited target-model SHA env.
        let sha=model::weights::model_sha256_opt(&g.path,&dir,false).unwrap_or_else(|e|die(&e));
        // CPU weights already live in the independent repacked cache. A cold
        // source checksum must not leave a second model-sized clean page cache.
        for t in &g.header.tensors{g.release_range(t,0,t.bytes()as usize);g.drop_cache(t);}
        let backend:Box<dyn oxn::Backend>=if mode=="emu"{
            Box::new(oxn::emu::Emu::new(8<<20,1))
        }else{
            if !cfg!(feature="npu-device"){die("build with the npu-device feature");}
            let go=a.get("npu-go-file").unwrap_or_else(||die("device prefill needs --npu-go-file for guard handshake"));
            println!("{}",json!({"event":"ready_before_device","pid":std::process::id()}));
            std::io::stdout().flush().unwrap();let deadline=Instant::now()+Duration::from_secs(30);
            while !Path::new(go).is_file(){if Instant::now()>deadline{die("guard handshake timeout; no NPU opened");}std::thread::sleep(Duration::from_millis(20));}
            #[cfg(feature="npu-device")]{Box::new(oxn::device::Device::open(1).unwrap_or_else(|e|die(&e)))}
            #[cfg(not(feature="npu-device"))]{unreachable!()}
        };
        let st=eng.enable_npu_prefill(&g,&sha,backend,model::npu_prefill::Config{audit_original:a.get("npu-audit")==Some("1"),
            recurrent:model::npu_prefill::RecurrentMode::parse(a.get("npu-recurrent").unwrap_or("native")).unwrap_or_else(|e|die(&e)),
            overlap_shared:match a.get("npu-overlap").unwrap_or("none"){"none"=>false,"shared"=>true,_=>die("--npu-overlap none|shared")},
            recurrent_mask:a.get("npu-recurrent-mask").map(|s|u64::from_str_radix(s.trim_start_matches("0x"),16).unwrap_or_else(|_|die("hex recurrent mask required"))).unwrap_or(u64::MAX),..Default::default()},&a.cpus()).unwrap_or_else(|e|die(&e));
        eprintln!("oxe: npu-prefill prepared {}",npu_stats_json(st));
        println!("{}",json!({"event":"npu_prepared"}));std::io::stdout().flush().unwrap();
    }
}

fn cmd_npu_prefill_check(a:&Args){
    #[cfg(not(feature="npu-prefill"))]{let _=a;die("build with npu-prefill feature");}
    #[cfg(feature="npu-prefill")]{
        if a.get("npu-prefill").is_none(){die("explicit --npu-prefill device|emu is required");}
        let p=if a.get("ref").is_some()||a.get("tokens").is_some(){prompt(a)}else{
            (0..a.num("prompt-len",64)as u32).map(|i|i.wrapping_mul(2654435761)%150000+1000).collect()
        };
        let n=a.num("n",16);
        if p.len()<32||p.len()>1024||n==0||n>128{die("bounded proof needs32..1024 prompt tokens and1..128 decode rows");}
        let n_ctx=a.num("ctx",2048).max(p.len()+n+1);
        let s=setup(a,n_ctx);
        if p.iter().any(|&t|t as usize>=s.loaded.plan.desc.n_vocab){die("prompt token outside vocabulary");}
        let cfg=EngineConfig{n_ctx,logits:true,trace:false,profile:false};
        let reference=Engine::new(&s.loaded.plan,&s.loaded.ws,cfg.clone()).unwrap_or_else(|e|die(&e));
        let single=Engine::new(&s.loaded.plan,&s.loaded.ws,cfg.clone()).unwrap_or_else(|e|die(&e));
        let mut candidate=Engine::new(&s.loaded.plan,&s.loaded.ws,cfg).unwrap_or_else(|e|die(&e));
        maybe_npu_prefill(a,&mut candidate);
        let ubatch=a.num("ubatch",256);
        let t=Instant::now();let mut next=reference.prefill(&s.pool,&p,ubatch).unwrap();let cpu_prefill_s=t.elapsed().as_secs_f64();
        for &tok in &p{single.step(&s.pool,tok);}
        let t=Instant::now();candidate.prefill(&s.pool,&p,ubatch);let npu_prefill_s=t.elapsed().as_secs_f64();
        let npu_calls_after_prefill=candidate.npu_prefill_stats().unwrap().calls;
        let mut rows=Vec::new();let mut ids=Vec::new();
        for i in 0..n{
            let base=reference.logits();
            if base.is_empty()||base.len()!=candidate.logits().len()||base.len()!=single.logits().len()
                ||base.iter().chain(candidate.logits()).chain(single.logits()).any(|v|!v.is_finite()){
                die("nonfinite or mismatched logit rows");
            }
            let numeric=kld(base,candidate.logits());let noise=kld(base,single.logits());
            rows.push(json!({"row":i,"npu_vs_batch_kld":numeric,"single_vs_batch_noise":noise,
                "npu_top1_matches_batch":argmax(base)==argmax(candidate.logits()),"single_top1_matches_batch":argmax(base)==argmax(single.logits())}));
            ids.push(next);let tok=next;next=reference.step(&s.pool,tok);single.step(&s.pool,tok);candidate.step(&s.pool,tok);
        }
        let mean=|key:&str|rows.iter().map(|r|r[key].as_f64().unwrap()).sum::<f64>()/rows.len()as f64;
        let numeric=mean("npu_vs_batch_kld");let noise=mean("single_vs_batch_noise");
        let st=candidate.npu_prefill_stats().unwrap();
        // The already-reviewed W2 gate is SAME-model, SAME-position CPU noise.
        // Top1 at this bounded count is information only, not a2K-position gate.
        let no_decode_npu=st.calls==npu_calls_after_prefill;
        let mut before=Fnv(0xcbf29ce484222325);candidate.write_state(&mut before).unwrap();
        let passed=numeric.is_finite()&&noise.is_finite()&&numeric<=noise&&st.calls>0&&st.first_error.is_none()&&no_decode_npu;
        let before_decode_release=npu_stats_json(st);
        candidate.release_npu_prefill();
        let mut after=Fnv(0xcbf29ce484222325);candidate.write_state(&mut after).unwrap();
        let release_state_identical=before.0==after.0;
        let mut reference_decode=Vec::new();let mut candidate_decode=Vec::new();
        // Identical post-prefill decode inputs test that releasing the offload
        // cache leaves CPU state usable. These are not a timing comparison.
        for tok in ids.iter().take(2){reference_decode.push(reference.step(&s.pool,*tok));candidate_decode.push(candidate.step(&s.pool,*tok));}
        let passed=passed&&release_state_identical;
        let result=json!({"passed":passed,"kind":"bounded-NPU-prefill-numeric-proof","prompt_tokens":p.len(),"rows":rows,
            "mean_npu_kld":numeric,"mean_cpu_batch_single_noise":noise,"gate_multiplier":1.0,"teacher_ids_hash":sha_ids(&ids),
            "cpu_prefill_s_informational":cpu_prefill_s,"npu_prefill_s_informational":npu_prefill_s,"npu":before_decode_release,
            "release_cleared_registry":candidate.npu_prefill_stats().is_none(),"post_release_reference_ids":reference_decode,"post_release_candidate_ids":candidate_decode,
            "decode_npu_calls_zero":no_decode_npu,"release_state_identical":release_state_identical,"release_state_fnv":format!("{:016x}",before.0),
            "model_sha256":s.loaded.ws.info.model_sha256,"threads":s.pool.nthreads(),"effective":model::engine::effective_switches()});
        let text=serde_json::to_string_pretty(&result).unwrap();
        if let Some(out)=a.get("out"){std::fs::write(out,&text).unwrap();}println!("{text}");
        if !passed{std::process::exit(1);}
    }
}

fn main() {
    sched::cli::exit_zero_on_epipe(); // F-CI9: `| head` ends with status 0, not a core dump
    let a = Args::parse();
    if let Some(mode)=a.get("shared-experts") {
        if a.cmd != "bench" || mode != "dma" { die("experimental --shared-experts dma is limited to bench"); }
    }
    // fused-ref: validate its own values (candidate, control, gate numbers) before the generic check
    #[cfg(feature = "f32-reference")]
    if a.flag("check-args") && a.cmd == "fused-ref" {
        fused_ref_cmd::check_args(&a);
    }
    if a.flag("check-args") && a.cmd != "prefill-bench" {
        a.check_known();
    }
    if a.get("npu-prefill").is_some() && !cfg!(feature="npu-prefill") {
        die("this binary was built without the npu-prefill feature");
    }
    if a.get("npu-prefill").is_some() && !["npu-prefill-check","npu-f32-check","npu-greedy-check","f32-tensor-map","prefill-bench"].contains(&a.cmd.as_str()) {
        die("experimental NPU option is limited to npu-prefill-check and prefill-bench");
    }
    match COMMANDS.iter().find(|(name, _)| *name == a.cmd) {
        Some((_, run)) => run(&a),
        None => usage(),
    }
}

#[cfg(test)]
mod n3_tests {
    use super::*;

    /// Table-driven: every boolean flag any command reads (`flag("..")` in the CLI sources, the
    /// feature-gated command files included) is a parser boolean, and parses as one.
    #[test]
    fn bool_flags_cover_every_flag_read() {
        let sources = [
            include_str!("oxe.rs"),
            include_str!("../fused_ref_cmd.rs"),
            include_str!("../f32_proof.rs"),
            include_str!("../expert_dma.rs"),
            include_str!("../expert_price.rs"),
        ];
        let mut read = Vec::new();
        for src in sources {
            for part in src.split("flag(\"").skip(1) {
                let name = &part[..part.find('"').unwrap()];
                if !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
                    read.push(name.to_string());
                }
            }
        }
        read.sort();
        read.dedup();
        for must in ["plain-steps", "batched-prompt", "check-args", "dry-run", "profile"] {
            assert!(read.iter().any(|r| r == must), "scanner missed flag(\"{must}\")");
        }
        let missing: Vec<&String> = read.iter().filter(|r| !BOOL_FLAGS.contains(&r.as_str())).collect();
        assert!(missing.is_empty(), "flags read but not parsed as booleans: {missing:?}");
        // and the parser treats each as a boolean: the next option is not swallowed
        for f in BOOL_FLAGS {
            let args = Args::parse_from(vec!["bench".into(), format!("--{f}"), "--n".into(), "7".into()]);
            assert!(args.flag(f) && args.get("n") == Some("7") && args.kv.len() == 1, "--{f}");
        }
    }

    #[test]
    fn usage_lists_every_dispatched_command_once() {
        let line = usage_line();
        let listed = line.split_whitespace().nth(2).unwrap();
        let names: Vec<&str> = COMMANDS.iter().map(|(n, _)| *n).collect();
        assert_eq!(listed.split('|').collect::<Vec<_>>(), names);
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
        assert_eq!(names.len(), 24);
    }

    #[test]
    fn bench_arms_accept_oxe_vl() {
        let arms = parse_arms("base|vl:OXE_VL=15|pad0:OXE_DEC_ATTN_PAD=0,OXE_VL=4");
        assert_eq!(arms.len(), 3);
        assert_eq!(arms[1].1, vec![("OXE_VL".to_string(), "15".to_string())]);
        assert_eq!(arms[2].1.len(), 2);
        assert!(ARM_KEYS.contains(&"OXE_VL"));
    }
}
