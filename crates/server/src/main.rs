//! oxe-server CLI. Flags mirror llama-server's where they mean the same thing; llama-only
//! flags that the controller passes are accepted and ignored (listed in IGNORED).

use serde_json::{Map, Value as Json};
use server::engine::{Engine, StubEngine};
use server::generate::Worker;
use server::service::{Config, State};
use std::sync::Arc;

const USAGE: &str = "usage: oxe-server --model PATH [--alias NAME] [--host 127.0.0.1] [--port 8080]
  [--ctx-size N] [--ctx-checkpoints N] [--chat-template-kwargs JSON] [--reasoning-format none|auto|deepseek]
  [--reasoning on|off] [--slots] [--idle-reprefill (SX-2, off by default)] [--engine stub|oxe|flash] [--stub-reply TEXT] [--oxe-cpus 0-7]
  [--oxe-prefill-cpus LIST (default: --oxe-cpus)] [--ubatch-size 256 (oxe prefill ubatch)]
  [--spec-type draft-mtp|none] [--spec-draft-n-max N (1-3)] [--mtp] [--draft N]
  [--flash-threads 4] [--flash-threads-batch 6] [--flash-cache-gib 16|auto (auto: 20 / 18 / 16 / 14 GiB, the largest tier MemAvailable covers, else refuse = exit 74; W5-C, F122, O-27)] [--flash-cache-pref auto|20|18|16|14 (preferred tier; falls back to the largest that fits with a NOTE; implies auto; 20 / 18 use the swap-tolerant MemTotal rule)] [--flash-desktop-floor-gib 3.0] [--flash-psi-some 10 --flash-psi-full 2 (percent avg10; default 0 = off on release main, 064 proposes 10 / 2)] [--flash-admit-harness-gib 0.80 (F118 growth)] [--flash-admit-dry-run] [--flash-pf-admit 0|1 (long-prompt guard, default 1)] [--flash-fuse N (S-c fused phases, 63 = all, default 0)] [--flash-pf-order 0|1|2 (default 1; 2 = F123 residents then misses, each by last use in the prompt)] [--flash-pf-readahead N|auto (default 0)] [--flash-pf-max-pass N|auto (default 0)] [--flash-io-cpu-prefill LIST] [--flash-io-cpu N] [--flash-sidecar PATH] [--flash-lookahead TABLE.lat [--flash-la-max 2] [--flash-la-pending 2]] [--flash-cache-policy lfu|clock] [--flash-quota-weights builtin|equal|FILE] [--flash-exact] (--engine flash: needs the `flash` feature)
  [--prefix-cache DIR|default (off unless given; default = ~/.cache/local-ai/snapdragon-engine/prefix)]
  [--prefix-cache-max-bytes N (4 GiB)] [--prefix-cache-min-tokens N (1024)] [--prefix-cache-save-floor-mib N (flash 1024, else 0: save an entry only while MemAvailable >= N MiB + 2 x the entry)] [--prefix-cache-ttl-h N (0 = off; at start, delete this store's entries and the same model's older-build orphans unused for N hours)]
  [--turn-checkpoints N (4; only checkpoint slots beyond 4 are used for them)]
  [--session-ram-bytes N (0 = off: RAM for parked sessions)] [--park-min-tokens N (512)]
  [--session-cache DIR (off; whole conversations on disk, written on SIGTERM and RAM eviction)]
  [--session-cache-max-bytes N (8 GiB)] [--session-cache-ttl-hours N (24)]
  [--ledger PATH (off; one JSON line of numbers per generation, appended, mode 0600)]
  [--api-key-file PATH (or env OXE_API_KEY_FILE; one key per line; every route, /health included, then needs
   \"Authorization: Bearer KEY\")] [--api-key KEY (visible in argv: prefer the file)]
  [--allow-host NAME (extra accepted Host name; default only 127.0.0.1 / localhost / [::1] with the port)]
  [--stream-stall-timeout SECS (30: a streaming client that takes no data this long is dropped, its generation cancelled)]
  [--repeat-watchdog N (stop a generation that ends in a token loop within the last N tokens; 0 = off)]
  [--mtp-head GGUF (MTP block from a sidecar; default <model>.mtp-head.gguf if present)]
  [--mtp-tune FILE (measured MTP depth per context bucket; default <model>.mtp-tune.json if present)]
  [--pf-group-t1 N (100) --pf-group-t2 N (0 = never): prefill calls of >= T1 new tokens run 2 ubatches per pass,
   >= T2 run 4; same state at every setting. OXE_PF_GROUP=1|2|4 fixes the group, OXE_PF_TILES=<mask> the kernels]
  [--mtp-max-ctx N (no MTP speculation at or beyond context position N; decided per verify)]
  [--spec-type lookup (prompt-lookup drafting, engines without MTP) [--lookup-kmin 4] [--lookup-n 4] [--lookup-backoff 0]]
  (per request: \"speculative\": false turns speculation off, \"spec_draft_n_max\": n sets the depth)
  [--oxe-weights file|anon (anon: weights copied in chunks into a huge-page anonymous arena)] [--mlock (lock weights or fail)]
  [--prefill-split grid|messages (grid; flash: messages = split the prefill at message starts, anchor exactly at the first user message)]
  [--flash-hotset FILE|default (off unless given; default = ~/.cache/local-ai/snapdragon-engine/flash/hotset.bin: expert ids, written after requests and at shutdown)]
  [--flash-warmup 0|1 (0: the hot-set file is written but never read back; 1: re-read the expert hot set at start and on state resume)]
  [--flash-attn-tile 0|1] [--flash-dense-gemm 0|1] [--flash-expert-tile 0|1] (flash prefill compute, fast path; default 1 each)
  (MTP default per model: Qwen3.6 draft-mtp depth 2; Ornith depth 2 only with a sidecar head
   (Qwen3.6's, tools/extract_mtp_head.py), else off; others off; flags override)";

/// llama-server flags accepted for drop-in argv compatibility (flag, takes a value)
const IGNORED: &[(&str, bool)] = &[
    ("--threads", true), ("-t", true), ("--threads-batch", true), ("-tb", true), ("--n-gpu-layers", true), ("-ngl", true),
    ("--jinja", false), ("--parallel", true), ("-np", true), ("--batch-size", true), ("-b", true), ("--flash-attn", true), ("-fa", true),
    ("--cache-ram", true), ("--reasoning-budget", true),
    ("--slot-save-path", true), ("--no-webui", false), ("--metrics", false), ("--no-mmap", false),
    ("--no-warmup", false), ("--log-verbosity", true), ("--perf", false), ("--cont-batching", false), ("--no-cache-prompt", false),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = None;
    let mut alias = None;
    let mut host = "127.0.0.1".to_string();
    let mut api_key_file: Option<String> = None;
    let mut api_keys: Vec<Vec<u8>> = Vec::new();
    let mut allow_hosts: Vec<String> = Vec::new();
    let mut port: u16 = 8080;
    let mut n_ctx: usize = 16384;
    let mut checkpoints: usize = 4;
    let mut kwargs = Map::new();
    let mut reasoning_format = chat::ReasoningFormat::Deepseek;
    let mut enable_thinking = true;
    let mut slots = false;
    let mut idle_reprefill = false;
    let mut engine = "stub".to_string();
    let mut stub_reply = "Hello from the oxe stub engine.".to_string();
    let mut oxe_cpus = "0-7".to_string();
    let mut oxe_prefill_cpus: Option<String> = None;
    let mut ubatch = model::engine::UBATCH;
    let mut ubatch_set = false;
    // --engine flash (cargo feature `flash`): Qwen3.8-Flash-Next, graph port + streaming experts
    let mut flash_threads: usize = 4;
    let mut flash_threads_batch: usize = 6;
    let mut flash_cache_gib: f64 = 16.0;
    let mut flash_cache_gib_set = false;
    let mut flash_exact = false;
    let mut flash_io_cpu: Option<usize> = None;
    // G4 prefill compute switches (default on, fast path) and the warm-up read gate (default off)
    let mut flash_sw = server::flash_switches::FlashSwitches::default();
    // W4-I prefill policy: order (default on), read-ahead (auto | N), FP-8 pass size (0 = off | N | auto = 8192),
    // IO CPUs during a batched prefill (comma list; empty = same as --flash-io-cpu)
    let mut flash_pf_order = true;
    let mut flash_pf_order_last = false; // F123: --flash-pf-order 2
    // G4' (2026-11-11): order on, read-ahead off, FP-8 off, no per-phase IO core
    let mut flash_pf_readahead: Option<usize> = Some(0);
    let mut flash_cache_auto = false;
    let mut flash_admit_harness: Option<f64> = None;
    // O-27: preferred tier (`--flash-cache-pref auto|20|18|16|14`); implies `--flash-cache-gib auto`
    let mut flash_cache_pref: Option<f64> = None;
    let mut flash_cache_pref_set = false;
    // O-27: active-desktop resident floor of the swap-tolerant rule (explicit 20 / 18 preference only)
    let mut flash_desktop_floor: f64 = 3.0;
    // O-27 PSI guard thresholds, percent avg10 (0 = off)
    // release main: off by default like installed Flash 4b6ab92 until ticket 064 qualifies ae55e0c
    let mut flash_psi_some: f64 = 0.0;
    let mut flash_psi_full: f64 = 0.0;
    let mut flash_admit_dry_run = false;
    let mut flash_pf_admit = true;
    let mut flash_fuse: i32 = 0;
    let mut flash_pf_max_pass: usize = 0;
    let mut flash_io_cpu_prefill: Vec<usize> = Vec::new();
    // F-IO: the expert sidecar to read (default ~/.cache/local-ai/snapdragon-engine/flash/experts-r4.bin), e.g. the
    // unencrypted copy /var/lib/local-ai-models/flash/experts-r4.bin (its .json manifest must sit next to it)
    let mut flash_sidecar: Option<std::path::PathBuf> = None;
    // F-LOOKAHEAD: expert lookahead table (LAT2); off unless given
    let mut flash_lookahead: Option<std::path::PathBuf> = None;
    let mut flash_la_max: usize = 2;
    let mut flash_la_pending: usize = 2;
    let mut flash_cache_lfu = true; // LFU + built-in quota weights: the Flash default since ticket 035 (lossless)
    let mut flash_quota_weights: Option<String> = None;
    // M3.1 speculative decoding: llama-server's --spec-type draft-mtp / --spec-draft-n-max.
    // None = the model's default (server::oxe_engine::default_mtp_depth)
    let mut spec_mtp: Option<bool> = None;
    let mut draft_n_max: Option<usize> = None;
    let mut repeat_watchdog: usize = 0;
    let mut cache_k = "f16".to_string();
    let mut cache_v = "f16".to_string();
    // --mtp-head GGUF: the MTP block from a sidecar GGUF (tools/extract_mtp_head.py), e.g. Qwen3.6's
    // head for Ornith; default: `<model>.mtp-head.gguf` next to the model if it exists
    let mut mtp_head: Option<String> = None;
    // --mtp-tune FILE: measured MTP depth per context bucket (tools/mtp_tune.py); default:
    // `<model>.mtp-tune.json` next to the model if present
    let mut mtp_tune: Option<String> = None;
    // F82: no MTP speculation at or beyond this context position (--mtp-max-ctx N; unset = no cap)
    let mut mtp_max_ctx: Option<usize> = None;
    let mut lookup = false;
    let mut lookup_cfg = server::lookup::LookupConfig::default();
    let mut prefix_dir: Option<String> = None;
    let mut prefix_max: u64 = 4 << 30;
    let mut prefix_min: usize = 1024;
    let mut prefix_ttl_h: u64 = 0;
    let mut prefix_save_floor_mib: Option<u64> = None;
    let mut turn_checkpoints: usize = 4;
    let mut session_ram: usize = 0;
    let mut park_min: usize = 512;
    let mut session_dir: Option<String> = None;
    let mut session_max: u64 = 8 << 30;
    let mut session_ttl_h: u64 = 24;
    let mut prefill_split: Option<bool> = None;
    let mut hotset: Option<String> = None;
    let mut ledger: Option<String> = None;
    let mut i = 0;
    let need = |i: usize| -> String {
        args.get(i + 1).cloned().unwrap_or_else(|| {
            eprintln!("missing value for {}\n{USAGE}", args[i]);
            std::process::exit(2)
        })
    };
    while i < args.len() {
        let a = args[i].as_str();
        let mut used = 2;
        match a {
            "--model" | "-m" => model = Some(need(i)),
            "--alias" | "-a" => alias = Some(need(i)),
            "--host" => host = need(i),
            "--port" => port = need(i).parse().expect("--port"),
            "--ctx-size" | "-c" => n_ctx = need(i).parse().expect("--ctx-size"),
            "--ctx-checkpoints" => checkpoints = need(i).parse().expect("--ctx-checkpoints"),
            "--chat-template-kwargs" => match serde_json::from_str::<Json>(&need(i)) {
                Ok(Json::Object(m)) => kwargs = m,
                _ => {
                    eprintln!("--chat-template-kwargs must be a JSON object");
                    std::process::exit(2)
                }
            },
            "--reasoning-format" => {
                reasoning_format = match need(i).as_str() {
                    "none" => chat::ReasoningFormat::None,
                    "auto" => chat::ReasoningFormat::Auto,
                    "deepseek" => chat::ReasoningFormat::Deepseek,
                    "deepseek-legacy" => chat::ReasoningFormat::DeepseekLegacy,
                    x => {
                        eprintln!("unknown reasoning format {x}");
                        std::process::exit(2)
                    }
                }
            }
            "--reasoning" => enable_thinking = !matches!(need(i).as_str(), "off" | "0" | "false"),
            "--slots" => {
                slots = true;
                used = 1
            }
            "--idle-reprefill" => {
                idle_reprefill = true;
                used = 1
            }
            "--engine" => engine = need(i),
            "--prefix-cache" => prefix_dir = Some(need(i)),
            "--prefix-cache-max-bytes" => prefix_max = need(i).parse().expect("--prefix-cache-max-bytes"),
            "--prefix-cache-min-tokens" => prefix_min = need(i).parse().expect("--prefix-cache-min-tokens"),
            "--prefix-cache-ttl-h" => prefix_ttl_h = need(i).parse().expect("--prefix-cache-ttl-h"),
            "--prefix-cache-save-floor-mib" => prefix_save_floor_mib = Some(need(i).parse().expect("--prefix-cache-save-floor-mib")),
            "--turn-checkpoints" => turn_checkpoints = need(i).parse().expect("--turn-checkpoints"),
            "--session-ram-bytes" => session_ram = need(i).parse().expect("--session-ram-bytes"),
            "--park-min-tokens" => park_min = need(i).parse().expect("--park-min-tokens"),
            "--session-cache" => session_dir = Some(need(i)),
            "--ledger" => ledger = Some(need(i)),
            "--api-key-file" => api_key_file = Some(need(i)),
            "--api-key" => {
                eprintln!("oxe-server: --api-key is visible to other local users in the process list; prefer --api-key-file");
                api_keys.push(need(i).trim().as_bytes().to_vec())
            }
            "--allow-host" => allow_hosts.push(need(i).trim().to_ascii_lowercase()),
            // S8: read by State::new (single-threaded here)
            "--stream-stall-timeout" => std::env::set_var("OXE_STREAM_STALL_SECS", need(i).parse::<f64>().ok().filter(|v| *v > 0.0).expect("--stream-stall-timeout").to_string()),
            "--session-cache-max-bytes" => session_max = need(i).parse().expect("--session-cache-max-bytes"),
            "--session-cache-ttl-hours" => session_ttl_h = need(i).parse().expect("--session-cache-ttl-hours"),
            "--prefill-split" => {
                prefill_split = Some(match need(i).as_str() {
                    "messages" => true,
                    "grid" => false,
                    x => {
                        eprintln!("--prefill-split {x}: grid|messages");
                        std::process::exit(2)
                    }
                })
            }
            "--flash-hotset" => hotset = Some(need(i)),
            "--flash-attn-tile" | "--flash-dense-gemm" | "--flash-expert-tile" | "--flash-warmup" => {
                if let Err(e) = flash_sw.parse_flag(a, &need(i)) {
                    eprintln!("oxe-server: {e}");
                    std::process::exit(2)
                }
            }
            "--stub-reply" => stub_reply = need(i),
            "--oxe-cpus" => oxe_cpus = need(i),
            "--oxe-prefill-cpus" => oxe_prefill_cpus = Some(need(i)),
            "--flash-threads" => flash_threads = need(i).parse().unwrap_or(4),
            "--flash-threads-batch" => flash_threads_batch = need(i).parse().unwrap_or(6),
            "--flash-cache-gib" | "--flash-cache" => {
                // fail loudly: a typo must never silently become another cache size (admission is sized for the flag)
                let v = need(i);
                flash_cache_gib_set = true;
                flash_cache_auto = v == "auto";
                flash_cache_gib = match v.parse::<f64>() {
                    _ if v == "auto" => flash_cache_gib,
                    Ok(g) if g.is_finite() && g > 0.0 => g,
                    _ => {
                        eprintln!("oxe-server: --flash-cache-gib {v:?}: not a positive number of GiB");
                        std::process::exit(2);
                    }
                };
            }
            "--flash-io-cpu" => flash_io_cpu = need(i).parse().ok(),
            "--flash-admit-harness-gib" => {
                let v = need(i);
                flash_admit_harness = match v.parse::<f64>() {
                    Ok(g) if g.is_finite() && g >= 0.0 => Some(g),
                    _ => {
                        eprintln!("oxe-server: --flash-admit-harness-gib {v:?}: not a number of GiB >= 0");
                        std::process::exit(2);
                    }
                };
            }
            "--flash-cache-pref" => {
                let v = need(i);
                #[cfg(feature = "flash")]
                {
                    flash_cache_pref = qwen4exp::pf_plan::parse_cache_pref(&v).unwrap_or_else(|e| {
                        eprintln!("oxe-server: {e}");
                        std::process::exit(2)
                    });
                }
                #[cfg(not(feature = "flash"))]
                let _ = v;
                flash_cache_pref_set = true;
            }
            "--flash-desktop-floor-gib" | "--flash-psi-some" | "--flash-psi-full" => {
                let v = need(i);
                let g = match v.parse::<f64>() {
                    Ok(g) if g.is_finite() && g >= 0.0 => g,
                    _ => {
                        eprintln!("oxe-server: {a} {v:?}: a number >= 0");
                        std::process::exit(2);
                    }
                };
                match a {
                    "--flash-desktop-floor-gib" => flash_desktop_floor = g,
                    "--flash-psi-some" => flash_psi_some = g,
                    _ => flash_psi_full = g,
                }
            }
            "--flash-admit-dry-run" => {
                flash_admit_dry_run = true;
                used = 1
            }
            "--flash-pf-admit" => flash_pf_admit = need(i) != "0",
            "--flash-fuse" => {
                let v = need(i);
                flash_fuse = match v.parse::<i32>() {
                    Ok(f) if (0..=63).contains(&f) => f,
                    _ => {
                        eprintln!("oxe-server: --flash-fuse {v:?}: a bitmask 0..63");
                        std::process::exit(2);
                    }
                };
            }
            "--flash-pf-order" => {
                let v = need(i);
                flash_pf_order = v != "0";
                flash_pf_order_last = v == "2";
            }
            "--flash-pf-readahead" => flash_pf_readahead = need(i).parse().ok(), // "auto" (or anything not a number) = auto
            "--flash-pf-max-pass" => flash_pf_max_pass = if need(i) == "auto" { 8192 } else { need(i).parse().unwrap_or(0) },
            "--flash-io-cpu-prefill" => flash_io_cpu_prefill = need(i).split(',').filter_map(|x| x.parse().ok()).collect(),
            "--flash-sidecar" => flash_sidecar = Some(std::path::PathBuf::from(need(i))),
            "--flash-lookahead" => flash_lookahead = Some(std::path::PathBuf::from(need(i))),
            "--flash-la-max" => flash_la_max = need(i).parse().unwrap_or_else(|_| { eprintln!("--flash-la-max: not a number"); std::process::exit(2) }),
            "--flash-la-pending" => flash_la_pending = need(i).parse().unwrap_or_else(|_| { eprintln!("--flash-la-pending: not a number"); std::process::exit(2) }),
            "--flash-cache-policy" => flash_cache_lfu = need(i) != "clock",
            "--flash-quota-weights" => flash_quota_weights = Some(need(i)),
            "--flash-exact" => {
                flash_exact = true;
                used = 1
            }
            "--ubatch-size" | "-ub" => {
                ubatch_set = true;
                ubatch = need(i).parse().unwrap_or_else(|_| {
                    eprintln!("--ubatch-size: not a number");
                    std::process::exit(2)
                })
            }
            "--spec-type" => {
                for t in need(i).split(',') {
                    match t {
                        "draft-mtp" => spec_mtp = Some(true),
                        "none" => spec_mtp = Some(false),
                        // prompt lookup (WP-D2): verified drafts from the request's own tokens
                        "lookup" | "ngram-simple" => lookup = true,
                        x => eprintln!("--spec-type {x}: not supported by oxe-server (draft-mtp, lookup, none); ignored"),
                    }
                }
            }
            "--spec-draft-n-max" => draft_n_max = Some(need(i).parse().expect("--spec-draft-n-max")),
            "--draft" | "--draft-max" | "--draft-n" => {
                draft_n_max = Some(need(i).parse().expect("--draft"));
                spec_mtp = Some(true);
            }
            // llama-server's KV cache types (were ignored): f16 (default) or q8_0 per K and V
            "--cache-type-k" | "-ctk" => cache_k = need(i),
            "--cache-type-v" | "-ctv" => cache_v = need(i),
            "--mtp" => {
                spec_mtp = Some(true);
                used = 1
            }
            // weights: file-backed cache mapping (default) or an anonymous THP arena; --mlock locks
            // the weights or the server does not start (set before the engine opens, single-threaded)
            "--oxe-weights" => std::env::set_var("OXE_WEIGHTS", need(i)),
            "--mlock" => {
                std::env::set_var("OXE_MLOCK", "1");
                used = 1
            }
            "--repeat-watchdog" => repeat_watchdog = need(i).parse().expect("--repeat-watchdog"),
            "--mtp-head" => mtp_head = Some(need(i)),
            "--mtp-tune" => mtp_tune = Some(need(i)),
            // set before the engine reads them (single-threaded here)
            "--pf-group-t1" => std::env::set_var("OXE_PF_GROUP_T1", need(i).parse::<usize>().expect("--pf-group-t1").to_string()),
            "--pf-group-t2" => std::env::set_var("OXE_PF_GROUP_T2", need(i).parse::<usize>().expect("--pf-group-t2").to_string()),
            "--mtp-max-ctx" => mtp_max_ctx = Some(need(i).parse().expect("--mtp-max-ctx")),
            "--lookup-kmin" => lookup_cfg.kmin = need(i).parse().expect("--lookup-kmin"),
            "--lookup-n" => lookup_cfg.n_max = need(i).parse().expect("--lookup-n"),
            "--lookup-backoff" => lookup_cfg.backoff = need(i).parse().expect("--lookup-backoff"),
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            _ => match IGNORED.iter().find(|(f, _)| *f == a) {
                Some((_, takes)) => used = if *takes { 2 } else { 1 },
                None => {
                    eprintln!("unknown argument {a}\n{USAGE}");
                    std::process::exit(2)
                }
            },
        }
        i += used;
    }
    // B1: API key (fails closed: a named but unusable key file stops the server)
    let api_key_file = api_key_file.or_else(|| std::env::var("OXE_API_KEY_FILE").ok().filter(|v| !v.is_empty()));
    if let Some(p) = &api_key_file {
        use std::os::unix::fs::PermissionsExt;
        match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| server::http::parse_key_file(&t)) {
            Ok(k) => api_keys.extend(k),
            Err(e) => {
                eprintln!("oxe-server: API key file {p}: {e}");
                std::process::exit(2)
            }
        }
        if let Ok(m) = std::fs::metadata(p) {
            if m.permissions().mode() & 0o077 != 0 {
                eprintln!("oxe-server: WARNING: API key file {p} is readable by other users (chmod 600)");
            }
        }
    }
    if api_keys.iter().any(|k| k.is_empty()) {
        eprintln!("oxe-server: empty API key");
        std::process::exit(2)
    }
    // a specific non-loopback bind address is a Host name its clients use
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        if !ip.is_loopback() && !ip.is_unspecified() {
            allow_hosts.push(match ip {
                std::net::IpAddr::V6(v6) => format!("[{v6}]"),
                v4 => v4.to_string(),
            });
        }
    }
    let access = server::http::Access { port: 0, extra_hosts: allow_hosts, keys: api_keys };
    if !access.keys.is_empty() {
        eprintln!("oxe-server: API key required on every route");
    }
    // O-27: a preference selects among the auto tiers; a fixed --flash-cache-gib N together with a preference is a
    // contradiction (which one would the user mean?)
    if flash_cache_pref_set {
        if flash_cache_gib_set && !flash_cache_auto {
            eprintln!("oxe-server: --flash-cache-pref needs --flash-cache-gib auto (or no --flash-cache-gib)");
            std::process::exit(2)
        }
        flash_cache_auto = true;
    }
    // W5-C: print the `--flash-cache-gib auto` decision for this host now and exit (no model, QC-ADM)
    if flash_admit_dry_run {
        #[cfg(feature = "flash")]
        {
            let (d, line) = server::flash_engine::cache_decide_now(n_ctx, flash_admit_harness, flash_cache_pref, flash_desktop_floor);
            println!("{line}");
            println!("{}", d.status_json());
            return;
        }
        #[cfg(not(feature = "flash"))]
        {
            eprintln!("--flash-admit-dry-run needs the `flash` feature");
            std::process::exit(2)
        }
    }
    let Some(model) = model else {
        eprintln!("{USAGE}");
        std::process::exit(2)
    };
    // O-27 / F122: the Flash cache tier is decided here, before anything of the model (tokenizer included) is read;
    // the engine then opens the chosen size, and /props reports the decision as engine_status.flash_cache
    #[allow(unused_mut)]
    let mut engine_status: Option<Json> = None;
    #[cfg(feature = "flash")]
    if engine == "flash" && flash_cache_auto {
        let (d, line) = server::flash_engine::cache_decide_now(n_ctx, flash_admit_harness, flash_cache_pref, flash_desktop_floor);
        eprintln!("oxe-server: {line}");
        let st = d.status_json();
        eprintln!("oxe-server: flash cache status {st}");
        engine_status = serde_json::from_str::<Json>(&st).ok().map(|v| serde_json::json!({ "flash_cache": v }));
        match d.cache_gib {
            Some(c) => {
                flash_cache_gib = c;
                flash_cache_auto = false;
            }
            None => {
                eprintln!("oxe-server: flash: ADMISSION REFUSED (--flash-cache-gib auto: not even 14 GiB fits)");
                std::process::exit(74)
            }
        }
    }
    if let Some(Json::Bool(b)) = kwargs.get("enable_thinking") {
        enable_thinking = *b;
    }
    let tok = tokenizer::Tokenizer::from_gguf(&model).unwrap_or_else(|e| {
        eprintln!("tokenizer: {e}");
        std::process::exit(1)
    });
    let model_name = alias.unwrap_or_else(|| std::path::Path::new(&model).file_name().unwrap().to_string_lossy().into_owned());
    let cfg = Config {
        model_name,
        model_path: model.clone(),
        chat_opts: chat::ServerChatOptions { chat_template_kwargs: kwargs, reasoning_format, enable_thinking },
        fingerprint: format!("oxe-{}", env!("CARGO_PKG_VERSION")),
        slots,
        idle_reprefill,
        engine_status,
    };
    let prefix = prefix_dir.map(|d| {
        let dir = if d == "default" {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/local-ai/snapdragon-engine/prefix")
        } else {
            std::path::PathBuf::from(d)
        };
        // FLASH-PREFIX-2: Flash saves only with MemAvailable >= this + 2 x entry (default = the controller's 1 GiB host floor)
        let save_floor_mib = prefix_save_floor_mib.unwrap_or(if engine == "flash" { 1024 } else { 0 });
        PrefixOpts { dir, max_bytes: prefix_max, min_tokens: prefix_min, ttl_h: prefix_ttl_h, save_floor_bytes: save_floor_mib << 20, model: std::path::PathBuf::from(&model) }
    });
    let warm = hotset.filter(|h| h != "off").map(|h| {
        if h == "default" {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/local-ai/snapdragon-engine/flash/hotset.bin")
        } else {
            std::path::PathBuf::from(h)
        }
    });
    let sess = SessionOpts {
        checkpoints,
        turn_checkpoints,
        session_ram,
        park_min,
        dir: session_dir.map(std::path::PathBuf::from),
        max_bytes: session_max,
        ttl_h: session_ttl_h,
        model: std::path::PathBuf::from(&model),
        msg_split: prefill_split.unwrap_or(engine == "flash"),
        warm,
        ledger: ledger.map(std::path::PathBuf::from),
    };
    match engine.as_str() {
        "stub" => {
            let script = tok.encode(&stub_reply, false, true);
            let eng = StubEngine::new(n_ctx, tok.n_vocab(), tok.eos.unwrap_or(0), script);
            serve(cfg, tok, eng, sess, ubatch, prefix, host, port, engine, n_ctx, repeat_watchdog, None, None, None, access)
        }
        "oxe" => {
            let cpus = sched::parse_cpu_list(&oxe_cpus).unwrap_or_else(|e| {
                eprintln!("--oxe-cpus: {e}");
                std::process::exit(2)
            });
            let pcpus = oxe_prefill_cpus.as_deref().map(|l| {
                sched::parse_cpu_list(l).unwrap_or_else(|e| {
                    eprintln!("--oxe-prefill-cpus: {e}");
                    std::process::exit(2)
                })
            });
            let head = server::oxe_engine::mtp_head_path(std::path::Path::new(&model), mtp_head.as_deref().map(std::path::Path::new));
            let model_default = server::oxe_engine::default_mtp_depth(std::path::Path::new(&model), head.as_deref());
            let tune_path = mtp_tune.map(std::path::PathBuf::from).or_else(|| {
                let p = std::path::Path::new(&model).with_extension("mtp-tune.json");
                p.exists().then_some(p)
            });
            let table = tune_path.as_ref().map(|p| {
                server::generate::SpecTable::load(p).unwrap_or_else(|e| {
                    eprintln!("--mtp-tune: {e}");
                    std::process::exit(2)
                })
            });
            let mut depth = match spec_mtp {
                Some(false) => 0,
                Some(true) => draft_n_max.unwrap_or(if model_default > 0 { model_default } else { 2 }),
                None if model_default > 0 => draft_n_max.unwrap_or(model_default),
                None => 0,
            };
            // with a tune table (and MTP not forced off, and a head available) the engine is built
            // for the deepest tuned depth; requests then use their bucket's depth (0 = off)
            if let Some(t) = &table {
                if spec_mtp != Some(false) && (model_default > 0 || head.is_some() || spec_mtp == Some(true)) {
                    depth = draft_n_max.unwrap_or(t.buckets.iter().map(|b| b.1).max().unwrap_or(0));
                }
                eprintln!("oxe-server: MTP tune table {}: {:?} (context <= positions -> depth)", tune_path.as_ref().unwrap().display(), t.buckets);
            }
            if depth > 3 {
                eprintln!("--spec-draft-n-max must be 1..3 for draft-mtp");
                std::process::exit(2)
            }
            eprintln!(
                "oxe-server: MTP speculative decoding {} (model default {model_default})",
                if depth > 0 { format!("on, {depth} draft tokens (greedy requests)") } else { "off".into() }
            );
            if depth > 0 {
                if let Some(h) = &head {
                    eprintln!("oxe-server: MTP head from {}", h.display());
                }
            }
            let use_lookup = lookup && depth == 0;
            if lookup && depth > 0 {
                eprintln!("oxe-server: --spec-type lookup ignored: MTP speculation is on (depth {depth})");
            }
            if use_lookup {
                // set before the engine opens (single-threaded here): the verify lanes it needs
                std::env::set_var("OXE_LOOKUP_LANES", (lookup_cfg.n_max + 1).to_string());
                eprintln!("oxe-server: prompt-lookup drafting on ({:?})", lookup_cfg);
            }
            // O-13 int8 KV: --cache-type-k/-v q8_0 -> the engine's KV type (set before the engine opens)
            let kv = match (cache_k.as_str(), cache_v.as_str()) {
                ("f16", "f16") => "f16",
                ("q8_0", "f16") => "k8",
                ("f16", "q8_0") => "v8",
                ("q8_0", "q8_0") => "kv8",
                (k, v) => {
                    eprintln!("--cache-type-k {k} / --cache-type-v {v}: oxe supports f16 and q8_0");
                    std::process::exit(2)
                }
            };
            if kv != "f16" {
                std::env::set_var("OXE_KV_TYPE", kv);
                eprintln!("oxe-server: KV cache {kv} (K {cache_k}, V {cache_v})");
            }
            eprintln!("oxe-server: engine switches: {}", model::engine::effective_switches());
            let eng = server::oxe_engine::OxeEngine::open_head(std::path::Path::new(&model), head.as_deref(), &cpus, pcpus.as_deref(), ubatch, n_ctx, depth).unwrap_or_else(|e| {
                eprintln!("oxe engine: {e}");
                std::process::exit(1)
            });
            // W4-4 a: one stable, machine-checkable MTP line (the T1 gate checks it)
            eprintln!(
                "oxe-server: mtp depth={depth} head={} head_cache={} max_ctx={} table={}",
                if depth == 0 { "none".to_string() } else { head.as_ref().map(|h| h.display().to_string()).unwrap_or_else(|| "own".into()) },
                eng.mtp_identity().unwrap_or("none"),
                mtp_max_ctx.map(|m| m.to_string()).unwrap_or_else(|| "off".into()),
                tune_path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into())
            );
            serve(cfg, tok, eng, sess, ubatch, prefix, host, port, engine, n_ctx, repeat_watchdog, table, use_lookup.then_some(lookup_cfg), mtp_max_ctx, access)
        }
        #[cfg(feature = "flash")]
        "flash" => {
            // O-21: prompt prefixes stay off the unencrypted weights partition (the sidecar's directory tree)
            let prefix = prefix.filter(|p| match server::prefix_store::inside_dir_of(&p.dir, flash_sidecar.as_deref()) {
                Some(side) => {
                    eprintln!("oxe-server: prefix store disabled: {} is inside the expert sidecar's directory {} (O-21: prompts stay on the encrypted home)", p.dir.display(), side.display());
                    false
                }
                None => true,
            });
            let o = server::flash_engine::FlashOptions {
                threads: flash_threads,
                threads_batch: flash_threads_batch,
                ubatch: if ubatch_set { ubatch } else { 4096 },
                n_ctx,
                cache_gib: flash_cache_gib,
                sidecar: flash_sidecar.clone(),
                lookahead: flash_lookahead.clone(),
                la_max: flash_la_max,
                la_pending: flash_la_pending,
                io_cpu: flash_io_cpu,
                cache_lfu: flash_cache_lfu,
                quota_weights: match flash_quota_weights.as_deref() {
                    None | Some("builtin") => Some(qwen4exp::stream::FLASH_QUOTA_WEIGHTS.to_vec()),
                    Some("equal") => None,
                    Some(p) => Some({
                    let t = std::fs::read_to_string(p).unwrap_or_else(|e| panic!("--flash-quota-weights {p}: {e}"));
                    t.lines()
                        .filter(|l| !l.trim_start().starts_with('#'))
                        .flat_map(|l| l.split_whitespace())
                        .map(|x| x.parse::<f64>().unwrap_or_else(|_| panic!("--flash-quota-weights: bad value {x}")))
                        .collect()
                    }),
                },
                exact: flash_exact,
                attn_tile: flash_sw.attn_tile,
                dense_gemm: flash_sw.dense_gemm,
                expert_tile: flash_sw.expert_tile,
                warmup: flash_sw.warmup,
                pf_order: flash_pf_order,
                pf_order_last: flash_pf_order_last,
                pf_readahead: flash_pf_readahead,
                pf_max_pass: flash_pf_max_pass,
                io_cpu_prefill: flash_io_cpu_prefill.clone(),
                cache_auto: flash_cache_auto,
                admit_harness_gib: flash_admit_harness,
                pf_admit: flash_pf_admit,
                fuse: flash_fuse,
                psi_some: flash_psi_some,
                psi_full: flash_psi_full,
            };
            let eng = server::flash_engine::FlashEngine::open(std::path::Path::new(&model), &o).unwrap_or_else(|e| {
                eprintln!("flash engine: {e}");
                std::process::exit(1)
            });
            serve(cfg, tok, eng, sess, if ubatch_set { ubatch } else { 4096 }, prefix, host, port, engine, n_ctx, repeat_watchdog, None, None, None, access)
        }
        x => {
            eprintln!("unknown engine {x} (stub|oxe{})", if cfg!(feature = "flash") { "|flash" } else { "" });
            std::process::exit(2)
        }
    }
}

struct SessionOpts {
    checkpoints: usize,
    turn_checkpoints: usize,
    session_ram: usize,
    park_min: usize,
    dir: Option<std::path::PathBuf>,
    max_bytes: u64,
    ttl_h: u64,
    model: std::path::PathBuf,
    /// message-boundary prefill schedule (generate.rs `plan_messages`)
    msg_split: bool,
    /// warm-up hint file (`--flash-hotset`)
    warm: Option<std::path::PathBuf>,
    ledger: Option<std::path::PathBuf>,
}

static TERM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

extern "C" fn on_term(_sig: libc::c_int) {
    TERM.store(true, std::sync::atomic::Ordering::SeqCst);
}

struct PrefixOpts {
    dir: std::path::PathBuf,
    max_bytes: u64,
    min_tokens: usize,
    /// `--prefix-cache-ttl-h` (0 = off)
    ttl_h: u64,
    /// `--prefix-cache-save-floor-mib` in bytes (0 = off)
    save_floor_bytes: u64,
    model: std::path::PathBuf,
}

#[allow(clippy::too_many_arguments)]
fn serve<E: Engine>(cfg: Config, tok: tokenizer::Tokenizer, eng: E, sess: SessionOpts, ubatch: usize, prefix: Option<PrefixOpts>, host: String, port: u16, engine: String, n_ctx: usize, repeat_watchdog: usize, spec_table: Option<server::generate::SpecTable>, lookup: Option<server::lookup::LookupConfig>, mtp_max_ctx: Option<usize>, mut access: server::http::Access) {
    let mut w = Worker::new(eng, sess.checkpoints);
    w.ubatch = ubatch;
    w.spec_table = spec_table;
    w.mtp_max_ctx = mtp_max_ctx;
    w.lookup = lookup;
    w.repeat_watchdog = repeat_watchdog;
    if repeat_watchdog > 0 {
        eprintln!("oxe-server: repetition watchdog on (window {repeat_watchdog} tokens)");
    }
    // turn checkpoints only use slots beyond the 4 the anchor/query/tail checkpoints need
    let turns = sess.turn_checkpoints.min(sess.checkpoints.saturating_sub(4));
    w.turn_checkpoints = turns;
    w.cache.max_turn = turns;
    w.cache.park_budget = sess.session_ram;
    w.cache.park_min = sess.park_min;
    w.msg_split = sess.msg_split;
    // stored states are only valid for the schedule that computed them
    let sched = if sess.msg_split { " sched=messages" } else { "" };
    if let Some(path) = &sess.warm {
        // the hint names cache contents of this model; any server build may use it
        let id = server::prefix_store::identity(&sess.model, "warm", 0);
        let id = id.split(" server=").next().unwrap_or("").to_string();
        w.warm = Some(server::generate::WarmFile::new(path.clone(), id));
        match w.load_warm() {
            Some(line) => eprintln!("oxe-server: {} ({line})", path.display()),
            None => eprintln!("oxe-server: {}: no usable warm file yet", path.display()),
        }
    }
    let warm_file = w.warm.is_some();
    if let Some(p) = &sess.ledger {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(p) {
            Ok(f) => w.ledger = Some(f),
            Err(e) => eprintln!("oxe-server: ledger {} disabled: {e}", p.display()),
        }
    }
    let mut session_cache = false;
    if let Some(dir) = &sess.dir {
        match w.engine.state_id() {
            Some(sid) => {
                let id = server::prefix_store::identity(&sess.model, &format!("{sid}{sched}"), ubatch);
                match server::prefix_store::PrefixStore::open(dir, id, sess.max_bytes, sess.park_min.max(1)) {
                    Ok(mut s) => {
                        // TTL: only this identity's sessions (W4-4 c: a shared directory is safe)
                        let gone = s.prune_older_than(std::time::Duration::from_secs(sess.ttl_h * 3600));
                        if gone > 0 {
                            eprintln!("oxe-server: session cache: {gone} sessions older than {} h removed", sess.ttl_h);
                        }
                        eprintln!("oxe-server: session cache {} ({} usable sessions)", dir.display(), s.n_entries());
                        w.sessions = Some(s);
                        w.cache.spill = true;
                        session_cache = true;
                    }
                    Err(e) => eprintln!("oxe-server: session cache disabled: {e}"),
                }
            }
            None => eprintln!("oxe-server: session cache disabled: engine has no persistent state"),
        }
    }
    eprintln!(
        "oxe-server: {} checkpoints ({turns} for turns), parked sessions {}",
        sess.checkpoints,
        if sess.session_ram > 0 { format!("up to {} MiB", sess.session_ram >> 20) } else { "off".into() }
    );
    if let Some(p) = prefix {
        match w.engine.state_id() {
            Some(sid) => {
                let id = server::prefix_store::identity(&p.model, &format!("{sid}{sched}"), ubatch);
                match server::prefix_store::PrefixStore::open(&p.dir, id, p.max_bytes, p.min_tokens) {
                    Ok(mut s) => {
                        s.save_floor_bytes = p.save_floor_bytes;
                        if p.ttl_h > 0 {
                            // own entries + the same model's orphans of older builds (W4-4 c rules)
                            let gone = s.prune_older_than(std::time::Duration::from_secs(p.ttl_h * 3600));
                            if gone > 0 {
                                eprintln!("oxe-server: prefix store: {gone} entries unused for {} h removed", p.ttl_h);
                            }
                        }
                        eprintln!(
                            "oxe-server: prefix store {} ({} usable entries, cap {} MiB, min {} tokens)",
                            p.dir.display(),
                            s.n_entries(),
                            p.max_bytes >> 20,
                            p.min_tokens
                        );
                        w.store = Some(s);
                    }
                    Err(e) => eprintln!("oxe-server: prefix store disabled: {e}"),
                }
            }
            None => eprintln!("oxe-server: prefix store disabled: engine has no persistent state"),
        }
    }
    let st = State::new(cfg, tok, w).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let st = Arc::new(st);
    // SIGTERM/SIGINT (controller unload, model switch): write the sessions, then exit
    unsafe {
        libc::signal(libc::SIGTERM, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t);
        libc::signal(libc::SIGINT, on_term as extern "C" fn(libc::c_int) as libc::sighandler_t);
    }
    let st_term = st.clone();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
        if !TERM.load(std::sync::atomic::Ordering::SeqCst) {
            continue;
        }
        if session_cache || warm_file {
            // wait up to 5 s for a running generation; the controller allows 10 s
            let t0 = std::time::Instant::now();
            loop {
                if let Ok(mut w) = st_term.worker.try_lock() {
                    if w.save_warm(true) {
                        eprintln!("oxe-server: warm file written");
                    }
                    if session_cache {
                        let n = w.persist_sessions();
                        eprintln!("oxe-server: session cache: wrote {n} sessions in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                    }
                    break;
                }
                if t0.elapsed().as_secs() >= 5 {
                    eprintln!("oxe-server: shutdown: worker busy, warm file and sessions not written");
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }
        std::process::exit(0);
    });
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    rt.block_on(async move {
        let l = tokio::net::TcpListener::bind((host.as_str(), port)).await.unwrap_or_else(|e| {
            eprintln!("bind {host}:{port}: {e}");
            std::process::exit(1)
        });
        access.port = l.local_addr().map(|a| a.port()).unwrap_or(port);
        eprintln!("oxe-server listening on http://{host}:{} (engine {engine}, n_ctx {n_ctx})", access.port);
        server::http::serve_with(st, l, access).await.unwrap();
    });
}
