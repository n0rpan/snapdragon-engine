//! v0.1.3 lane G: oxe-server start-up checks (startup_check.rs), against the built binary.
//! P56: every child runs with a cleared environment and a scratch HOME; no model is read.
use server::startup_check::*;
use std::os::fd::{FromRawFd, OwnedFd};
use std::process::{Command, Stdio};

const BIN: &str = env!("CARGO_BIN_EXE_oxe-server");

fn scratch_home(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("oxe-startup-check-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn cmd(tag: &str, env: &[(&str, &str)], args: &[&str]) -> Command {
    let mut c = Command::new(BIN);
    c.env_clear().env("HOME", scratch_home(tag)).env("PATH", "/usr/bin:/bin").args(args);
    for (k, v) in env {
        c.env(k, v);
    }
    c
}

/// A pipe whose read end is already closed: the first write gets EPIPE.
fn closed_pipe() -> Stdio {
    let mut fds = [0i32; 2];
    // SAFETY: pipe + close on fds this process owns; the write end moves into the Stdio.
    unsafe {
        assert_eq!(libc::pipe(fds.as_mut_ptr()), 0);
        libc::close(fds[0]);
        Stdio::from(OwnedFd::from_raw_fd(fds[1]))
    }
}

#[test]
fn help_into_a_closed_pipe_exits_zero() {
    for flag in ["--help", "-h"] {
        let s = cmd("help-epipe", &[], &[flag]).stdout(closed_pipe()).stderr(Stdio::null()).status().unwrap();
        assert_eq!(s.code(), Some(0), "{flag} | (closed) must exit 0, got {s:?}");
    }
}

#[test]
fn help_prints_the_usage_and_the_capability_marker() {
    let o = cmd("help", &[], &["--help"]).output().unwrap();
    assert!(o.status.success());
    let t = String::from_utf8_lossy(&o.stdout);
    assert!(t.starts_with("usage: oxe-server --model PATH"), "{t:.80}");
    assert!(t.contains("oxe-capability state-preload-v1"));
}

#[test]
fn help_anywhere_wins_over_a_bad_value_and_a_bad_env() {
    let o = cmd("help-any", &[("OXE_PF_TILES", "nonsense")], &["--port", "x", "--help"]).output().unwrap();
    assert_eq!(o.status.code(), Some(0));
}

#[test]
fn check_args_clean_env_prints_effective_config() {
    let o = cmd("ca-clean", &[("OXE_PF_TILES", "13"), ("OXE_VL", "15"), ("OXE_KV_VPAD", "32")], &["--check-args", "--engine", "stub", "--port", "8099"]).output().unwrap();
    let t = String::from_utf8_lossy(&o.stdout);
    assert_eq!(o.status.code(), Some(0), "{t}\n{}", String::from_utf8_lossy(&o.stderr));
    let j: serde_json::Value = serde_json::from_str(&t).unwrap();
    assert_eq!(j["ok"], true);
    assert_eq!(j["env"]["OXE_PF_TILES"]["effective"], 13);
    assert_eq!(j["env"]["OXE_VL"]["effective"], 15);
    assert_eq!(j["flags_not_in_usage"].as_array().unwrap().len(), 0, "{t}");
}

#[test]
fn check_args_into_a_closed_pipe_exits_zero() {
    let s = cmd("ca-epipe", &[], &["--check-args"]).stdout(closed_pipe()).stderr(Stdio::null()).status().unwrap();
    assert_eq!(s.code(), Some(0), "{s:?}");
}

#[test]
fn check_args_lists_flags_not_in_the_usage() {
    let o = cmd("ca-flags", &[], &["--check-args", "--flash-cache-gibb", "16", "--port", "-1"]).output().unwrap();
    let j: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(j["flags_not_in_usage"], serde_json::json!(["--flash-cache-gibb"]));
}

#[test]
fn malformed_env_is_refused_at_start_and_by_check_args() {
    let bad: &[(&str, &str, &str)] = &[
        ("OXE_PF_TILES", "4096", "unknown bits"),
        ("OXE_PF_TILES", "0x3ff", "decimal"),
        ("OXE_PF_GROUP", "two", "not a decimal integer"),
        ("OXE_VL", "16", "unknown bits"),
        ("OXE_KV_TYPE", "kv", "one of"),
        ("OXE_DEC_ATTN_PAD", "O", "0 or 1"),
        ("OXE_KV_VPAD", "5000", "outside"),
        ("OXE_STREAM_STALL_SECS", "-1", "positive"),
    ];
    for (k, v, why) in bad {
        // a normal start: refused before the model path is even looked at (exit 2, one line naming the variable)
        let o = cmd("bad-start", &[(k, v)], &["--model", "/nonexistent/model.gguf"]).output().unwrap();
        let e = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(2), "{k}={v}: {e}");
        assert!(e.contains(k) && e.contains(why) && e.contains("refusing to start"), "{k}={v}: {e}");
        let o = cmd("bad-ca", &[(k, v)], &["--check-args"]).output().unwrap();
        assert_eq!(o.status.code(), Some(2), "{k}={v} --check-args");
        let j: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
        assert_eq!(j["ok"], false);
    }
}

#[test]
fn unknown_names_and_inherited_kv_type_only_warn() {
    let o = cmd("warn", &[("OXE_PF_TILESS", "13"), ("OXE_KV_TYPE", "kv8")], &["--check-args"]).output().unwrap();
    assert_eq!(o.status.code(), Some(0));
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(e.contains("unknown environment variable OXE_PF_TILESS"), "{e}");
    assert!(e.contains("OXE_KV_TYPE=kv8 is inherited"), "{e}");
}

#[test]
fn cpu_feature_check_names_what_is_missing() {
    assert!(missing_cpu_features(&|_| true).is_empty());
    assert_eq!(missing_cpu_features(&|f| f != "i8mm" && f != "bf16"), vec!["i8mm", "bf16"]);
    let m = cpu_message(&["i8mm"]);
    assert!(m.contains("lacks i8mm") && m.contains("illegal instruction"), "{m}");
    // the CI helper (Apple M2) and the Snapdragon have all four
    #[cfg(target_arch = "aarch64")]
    assert!(missing_cpu_features(&cpu_has).is_empty(), "{:?}", missing_cpu_features(&cpu_has));
}

#[test]
fn env_check_empty_values() {
    let env = |k: &str, v: &str| vec![(k.to_string(), v.to_string())];
    // empty = unset for numbers and words
    assert!(check_env(&env("OXE_PF_TILES", "")).errors.is_empty());
    assert!(check_env(&env("OXE_ATTN", "")).errors.is_empty());
    // a 0/1 switch must say which (its reader takes empty as ON)
    assert!(!check_env(&env("OXE_PF_GROUP_UNALIGNED", "")).errors.is_empty());
}

#[test]
fn usage_flags_are_extracted() {
    let f = usage_flags("usage: x --model PATH [--alias NAME] [--slots] [--flash-pf-order 0|1|2 (default 1)] -c\n  OXE_PF_GROUP=1|2|4");
    for k in ["--model", "--alias", "--slots", "--flash-pf-order", "-c"] {
        assert!(f.contains(k), "{k}: {f:?}");
    }
    assert!(!f.iter().any(|x| x.contains("OXE")));
}
