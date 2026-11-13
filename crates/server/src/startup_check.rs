//! Start-up checks of `oxe-server`, run as the first statement of `main` (v0.1.3 lane G):
//!
//! - **CPU features** (F-CI4): the release build targets Oryon (`target-cpu=oryon-1`, ggml
//!   `armv8.6-a+i8mm+bf16+dotprod+fp16`). On a core without one of them the binary used to die
//!   with SIGILL somewhere in the first kernel; now it stops at start with a clear message
//!   (exit [`EXIT_CPU`]).
//! - **OXE_\* environment** (W4-5, F96): the engine reads its knobs with `parse().ok().unwrap_or(default)`,
//!   so a typo (`OXE_PF_TILES=0x3ff`, `OXE_KV_TYPE=kv`, `OXE_DEC_ATTN_PAD=O`) silently became the
//!   default and measured as "no gain". Every known knob is validated here; a malformed value stops
//!   the server (exit 2) with one line per problem. Unknown `OXE_*` names are reported as a warning
//!   (bench / test drivers set their own), and an inherited `OXE_KV_TYPE` is pointed out (the
//!   `--cache-type-k/-v f16` flags do not clear it).
//! - **CLI-mode output on a closed pipe** (F-CI9): `oxe-server --help | head -1` aborted with a core
//!   dump (println! panics on EPIPE; release is panic=abort). `--help` and `--check-args` are
//!   answered here with EPIPE-safe writes and exit 0; other print-and-exit modes
//!   (`--flash-admit-dry-run`) get the [`sched::cli::exit_zero_on_epipe`] hook. A serving
//!   process does NOT get the hook: a closed stdout while serving stays a crash (systemd restarts it).
//! - **`--check-args`** (F96 c): validate the environment and the flag names, print the effective
//!   `OXE_*` configuration as JSON, exit (0 = clean, 2 = a problem). Values of the flags themselves are
//!   parsed by `main` (lane B owns it); the flag NAMES are checked against the usage text here.

use std::io::Write;

/// Exit status when the CPU lacks a feature the build needs.
pub const EXIT_CPU: i32 = 3;

/// The aarch64 features the release build assumes (rustc and ggml flags).
pub const REQUIRED_CPU_FEATURES: [&str; 4] = ["i8mm", "bf16", "dotprod", "fp16"];

/// Known bits of `OXE_PF_TILES` (model::engine prefill: bits 0..=11).
pub const PF_TILES_KNOWN: u32 = 0xfff;
/// Known bits of `OXE_VL` (model::engine VL_MV | VL_HEAD | VL_Q6K_ONE | VL_DN_FROM).
pub const VL_KNOWN: u32 = 0xf;

/// The whole start-up check; call once, first thing in `main`, with the usage text.
pub fn run(usage: &str) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // print-and-exit modes: a closed stdout must end with status 0, not a core dump
    if args.iter().any(|a| a == "--flash-admit-dry-run") {
        sched::cli::exit_zero_on_epipe();
    }
    if args.iter().any(|a| a == "-h" || a == "--help") {
        write_out_and_exit(&format!("{usage}\n"), 0);
    }
    let missing = missing_cpu_features(&cpu_has);
    if !missing.is_empty() {
        eprintln!("{}", cpu_message(&missing));
        std::process::exit(EXIT_CPU);
    }
    let env: Vec<(String, String)> = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.to_string_lossy().into_owned())))
        .filter(|(k, _)| k.starts_with("OXE_"))
        .collect();
    let rep = check_env(&env);
    for w in &rep.warnings {
        eprintln!("oxe-server: WARNING: {w}");
    }
    let check_args = args.iter().any(|a| a == "--check-args");
    if check_args {
        let unknown = flags_not_in_usage(&args, usage);
        let ok = rep.errors.is_empty();
        let out = serde_json::json!({
            "ok": ok,
            "cpu": REQUIRED_CPU_FEATURES.iter().map(|f| (f.to_string(), serde_json::Value::Bool(cpu_has(f)))).collect::<serde_json::Map<_, _>>(),
            "env": rep.effective,
            "errors": rep.errors,
            "warnings": rep.warnings,
            // flag names not in --help: llama-server flags accepted for compatibility, or typos
            "flags_not_in_usage": unknown,
        });
        let mut s = serde_json::to_string_pretty(&out).unwrap_or_default();
        s.push('\n');
        if !ok {
            for e in &rep.errors {
                eprintln!("oxe-server: {e}");
            }
        }
        write_out_and_exit(&s, if ok { 0 } else { 2 });
    }
    if !rep.errors.is_empty() {
        for e in &rep.errors {
            eprintln!("oxe-server: {e}");
        }
        eprintln!("oxe-server: refusing to start with malformed OXE_* values (unset them or fix the value; `--check-args` shows the effective configuration)");
        std::process::exit(2);
    }
}

/// Write `s` to stdout, ignore a closed pipe, exit with `code`.
fn write_out_and_exit(s: &str, code: i32) -> ! {
    let mut o = std::io::stdout().lock();
    let _ = o.write_all(s.as_bytes()).and_then(|_| o.flush());
    std::process::exit(code)
}

/// Runtime CPU feature test (non-aarch64 builds: nothing is required).
pub fn cpu_has(f: &str) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        match f {
            "i8mm" => std::arch::is_aarch64_feature_detected!("i8mm"),
            "bf16" => std::arch::is_aarch64_feature_detected!("bf16"),
            "dotprod" => std::arch::is_aarch64_feature_detected!("dotprod"),
            "fp16" => std::arch::is_aarch64_feature_detected!("fp16"),
            _ => false,
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = f;
        true
    }
}

/// The required features `has` reports missing (in [`REQUIRED_CPU_FEATURES`] order).
pub fn missing_cpu_features(has: &dyn Fn(&str) -> bool) -> Vec<&'static str> {
    REQUIRED_CPU_FEATURES.iter().copied().filter(|f| !has(f)).collect()
}

pub fn cpu_message(missing: &[&str]) -> String {
    format!(
        "oxe-server: this CPU lacks {} (needed: {}). This build targets Snapdragon X (Oryon, Armv8.6-A); \
         on this CPU it would crash with an illegal instruction. Use a build for this CPU.",
        missing.join(", "),
        REQUIRED_CPU_FEATURES.join(", ")
    )
}

/// What [`check_env`] found.
#[derive(Debug, Default)]
pub struct EnvReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    /// name -> {"value": raw, "effective": what the engine will use}
    pub effective: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy)]
enum Kind {
    /// unsigned integer in [lo, hi]
    Uint(u64, u64),
    /// bit mask with only these bits
    Mask(u32),
    /// "0" or "1"
    Bool01,
    /// one of these words (trimmed, lower case)
    Word(&'static [&'static str]),
    /// a positive finite number
    PosFloat,
    /// presence matters (or a path / free text): not validated
    Any,
}

/// Every `OXE_*` variable the engine, server and tools read (`grep -rhoE '"OXE_[A-Z0-9_]+"' crates`), with its form.
const KNOWN: &[(&str, Kind)] = &[
    ("OXE_PF_TILES", Kind::Mask(PF_TILES_KNOWN)),
    ("OXE_VL", Kind::Mask(VL_KNOWN)),
    ("OXE_PF_GROUP", Kind::Uint(0, 16)),
    ("OXE_PF_GROUP_T1", Kind::Uint(0, u64::MAX - 1)),
    ("OXE_PF_GROUP_T2", Kind::Uint(0, u64::MAX - 1)),
    ("OXE_PF_GROUP_UNALIGNED", Kind::Bool01),
    ("OXE_PF_MMID_MIN", Kind::Uint(0, 64)),
    ("OXE_PF_ATTN_PB", Kind::Uint(1, 1 << 20)),
    // v0.1.3 PREFILL-POLICY: prefill attention mode and what `auto` resolves to
    ("OXE_PF_ATTN", Kind::Word(&["auto", "tiled", "fa2", "fa2x", ""])),
    ("OXE_PF_ATTN_AUTO_KERNEL", Kind::Word(&["fa2", "fa2x"])),
    ("OXE_PF_ATTN_AUTO_MIN_KV", Kind::Any),
    ("OXE_SMALL_PREFILL", Kind::Uint(0, u64::MAX)),
    ("OXE_KV_TYPE", Kind::Word(&["f16", "k8", "v8", "kv8", "q8_0"])),
    ("OXE_KV8_PV", Kind::Word(&["fast", "ref", "i16", "int16", "i8", "int8"])),
    ("OXE_KV8_FUSED", Kind::Bool01),
    ("OXE_KV_VPAD", Kind::Uint(0, 4096)),
    ("OXE_ATTN", Kind::Word(&["tiled", "fused", "legacy"])),
    ("OXE_ATTN_TILED_MIN_KV", Kind::Uint(0, u64::MAX)),
    ("OXE_ATTN_FUSED_CH", Kind::Uint(128, 4096)),
    ("OXE_DEC_ATTN_PAD", Kind::Bool01),
    ("OXE_DEC_FUSED_DN", Kind::Bool01),
    ("OXE_IDLE_SPIN_US", Kind::Uint(0, 10_000_000)),
    ("OXE_LOOKUP_LANES", Kind::Uint(0, 64)),
    ("OXE_MTP_DRAFT_K", Kind::Uint(1, u64::MAX)),
    ("OXE_WEIGHTS", Kind::Word(&["file", "anon"])),
    ("OXE_ECHO", Kind::Word(&["0", "1", "on", "off", "true", "false", "yes", "no"])),
    ("OXE_ECHO_MAX_TOKENS", Kind::Uint(0, u64::MAX)),
    ("OXE_ECHO_TTL_SECS", Kind::Uint(0, u64::MAX)),
    ("OXE_STREAM_STALL_SECS", Kind::PosFloat),
    // v0.1.3 LONG-APPEND: SSE comment interval during queue wait / prefill (0 = off)
    ("OXE_SSE_KEEPALIVE_SECS", Kind::Any),
    ("OXE_PREFILL_QUIET", Kind::Bool01),
    ("OXE_TIMING_DIAG_WARMUP", Kind::Uint(0, u64::MAX)),
    // Flash (lane G v0.1.3): F123 hot-resident protection in batched prefill ops (default off)
    ("OXE_FLASH_PF_PROTECT_HOT", Kind::Bool01),
    ("OXE_MLOCK", Kind::Any),
    ("OXE_API_KEY_FILE", Kind::Any),
    ("OXE_MODEL_SHA256", Kind::Any),
    ("OXE_GIT_COMMIT", Kind::Any),
    ("OXE_ATTN_STATS", Kind::Any),
    ("OXE_TIMING_DIAG", Kind::Any),
    ("OXE_MTP_TRACE", Kind::Any),
    ("OXE_MTP_LOG", Kind::Any),
    ("OXE_MTP_FROM", Kind::Any),
    ("OXE_MTP_DRAFT_VOCAB", Kind::Any),
    ("OXE_TOKENIZER_GGUF", Kind::Any),
    ("OXE_SERVER_LEGACY", Kind::Any),
    ("OXE_KV", Kind::Any),
    // tests, benches and build scripts (never read by a serving process; listed so they are not "unknown")
    ("OXE_TEST_MODEL", Kind::Any),
    ("OXE_TEST_CPUS", Kind::Any),
    ("OXE_TEST_MTP", Kind::Any),
    ("OXE_MODEL_QWEN36", Kind::Any),
    ("OXE_MODEL_ORNITH", Kind::Any),
    ("OXE_NPU_TEST_GGUF", Kind::Any),
    ("OXE_MINI_OUT", Kind::Any),
    ("OXE_FUZZ_RECORDING", Kind::Any),
    ("OXE_NUMERICS_GOLDEN_UNLISTED", Kind::Any),
    ("OXE_SCHED_PANIC_CHILD", Kind::Any),
    ("OXE_KERNEL_CFLAGS", Kind::Any),
    ("OXE_KBENCH_THREADS", Kind::Any),
    ("OXE_KBENCH_REPS", Kind::Any),
    ("OXE_KBENCH_CPUS", Kind::Any),
    ("OXE_KBENCH_CASES", Kind::Any),
    ("OXE_KBENCH_LAYERS", Kind::Any),
    ("OXE_KBENCH_CTX", Kind::Any),
    ("OXE_KBENCH_CH", Kind::Any),
];

/// Parse an unsigned integer the way a user writes a mask or a count: decimal, or 0x-hex.
fn parse_u64(s: &str) -> Option<u64> {
    let t = s.trim();
    match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => t.parse().ok(),
    }
}

/// Validate one value; Ok(effective) or Err(why).
fn check_one(name: &str, kind: Kind, v: &str) -> Result<serde_json::Value, String> {
    use serde_json::Value as J;
    let t = v.trim();
    // an empty value is read as unset (the default) by the number and word readers; not by the 0/1 switches
    // (`v != "0"`: empty = ON, which is not the default of OXE_PF_GROUP_UNALIGNED): those must say 0 or 1
    if t.is_empty() && !matches!(kind, Kind::Any | Kind::Bool01) {
        return Ok(J::String("default".into()));
    }
    match kind {
        Kind::Uint(lo, hi) => {
            // decimal only: the engine's readers use str::parse (0x... would be read as the default)
            match t.parse::<u64>() {
                Ok(n) if (lo..=hi).contains(&n) => Ok(J::from(n)),
                Ok(n) => Err(format!("{name}={v:?}: {n} is outside {lo}..={hi}")),
                Err(_) => Err(format!("{name}={v:?}: not a decimal integer")),
            }
        }
        Kind::Mask(known) => {
            let n = match t.parse::<u64>() {
                Ok(n) => n,
                Err(_) if parse_u64(t).is_some() => return Err(format!("{name}={v:?}: write the mask in decimal (the engine reads decimal; {} here)", parse_u64(t).unwrap())),
                Err(_) => return Err(format!("{name}={v:?}: not a decimal bit mask")),
            };
            if n > u32::MAX as u64 || (n as u32) & !known != 0 {
                return Err(format!("{name}={v:?}: unknown bits 0x{:x} (known mask 0x{known:x}); they would be ignored", n & !(known as u64)));
            }
            Ok(J::from(n))
        }
        Kind::Bool01 => match t {
            "0" => Ok(J::Bool(false)),
            "1" => Ok(J::Bool(true)),
            _ => Err(format!("{name}={v:?}: 0 or 1")),
        },
        Kind::Word(ws) => {
            let l = t.to_ascii_lowercase();
            if ws.contains(&l.as_str()) {
                Ok(J::String(l))
            } else {
                Err(format!("{name}={v:?}: one of {}", ws.join("|")))
            }
        }
        Kind::PosFloat => match t.parse::<f64>() {
            Ok(x) if x.is_finite() && x > 0.0 => Ok(J::from(x)),
            _ => Err(format!("{name}={v:?}: a positive number")),
        },
        Kind::Any => Ok(J::String(v.to_string())),
    }
}

/// Validate the `OXE_*` variables of `env` (name, value).
pub fn check_env(env: &[(String, String)]) -> EnvReport {
    let mut r = EnvReport::default();
    let mut env: Vec<&(String, String)> = env.iter().filter(|(k, _)| k.starts_with("OXE_")).collect();
    env.sort();
    for (k, v) in env {
        match KNOWN.iter().find(|(n, _)| n == k) {
            Some((_, kind)) => match check_one(k, *kind, v) {
                Ok(eff) => {
                    r.effective.insert(k.clone(), serde_json::json!({ "value": v, "effective": eff }));
                }
                Err(e) => r.errors.push(e),
            },
            None => r.warnings.push(format!("unknown environment variable {k} (not read by this build; a typo?)")),
        }
        if k == "OXE_KV_TYPE" && !matches!(v.trim().to_ascii_lowercase().as_str(), "f16" | "") {
            r.warnings.push(format!("OXE_KV_TYPE={v} is inherited from the environment: --cache-type-k/-v f16 does not clear it (oxe engine KV type)"));
        }
    }
    r
}

/// Flag names (`-x`, `--name`) that appear in `usage`.
pub fn usage_flags(usage: &str) -> std::collections::BTreeSet<String> {
    usage
        .split(|c: char| c.is_whitespace() || "[](),|=\"".contains(c))
        .filter(|w| w.starts_with('-') && w.len() > 1 && w.as_bytes()[1].is_ascii_alphabetic() || w.starts_with("--") && w.len() > 2)
        .map(|w| w.trim_end_matches(|c: char| !c.is_ascii_alphanumeric()).to_string())
        .filter(|w| w.len() > 1)
        .collect()
}

/// Flags of `args` whose name is not in `usage` (values after a flag are skipped by shape only:
/// anything not starting with '-' is a value; negative numbers are values).
pub fn flags_not_in_usage(args: &[String], usage: &str) -> Vec<String> {
    let known = usage_flags(usage);
    let mut out = Vec::new();
    for a in args {
        let is_flag = a.starts_with('-') && a.len() > 1 && !a[1..].starts_with(|c: char| c.is_ascii_digit() || c == '.');
        if is_flag && a != "--check-args" && !known.contains(a.as_str()) && !out.contains(a) {
            out.push(a.clone());
        }
    }
    out
}
