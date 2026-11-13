//! F-CI9: a print into a closed pipe ends a CLI with status 0 (child-process test).
use super::cli::*;

#[test]
fn epipe_message_is_recognised() {
    assert!(is_epipe_print_panic(Some("failed printing to stdout: Broken pipe (os error 32)")));
    assert!(is_epipe_print_panic(Some("failed printing to stderr: Broken pipe (os error 32)")));
    assert!(!is_epipe_print_panic(Some("failed printing to stdout: No space left on device (os error 28)")));
    assert!(!is_epipe_print_panic(Some("index out of bounds")));
    assert!(!is_epipe_print_panic(None));
}

/// Child half: stdout becomes a pipe whose read end is closed, then println! until it fails.
#[test]
fn epipe_child() {
    let Ok(mode) = std::env::var("OXE_SCHED_EPIPE_CHILD") else { return };
    let mut fds = [0i32; 2];
    // SAFETY: plain pipe/close/dup2 on fds this process owns.
    unsafe {
        assert_eq!(libc::pipe(fds.as_mut_ptr()), 0);
        libc::close(fds[0]);
        assert!(libc::dup2(fds[1], 1) >= 0);
        libc::close(fds[1]);
    }
    if mode == "hook" {
        exit_zero_on_epipe();
    }
    for i in 0..100_000 {
        println!("line {i}");
    }
    // only reached if no print failed
    std::process::exit(7);
}

fn run_child(mode: &str) -> std::process::ExitStatus {
    let exe = std::env::current_exe().unwrap();
    std::process::Command::new(exe)
        .args(["--exact", "tests_cli::epipe_child", "--nocapture", "--test-threads", "1"])
        .env("OXE_SCHED_EPIPE_CHILD", mode)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .unwrap()
}

#[test]
fn closed_pipe_exits_zero_with_the_hook() {
    let s = run_child("hook");
    assert_eq!(s.code(), Some(0), "with the hook a closed stdout must end with status 0, got {s:?}");
}

#[test]
fn closed_pipe_fails_without_the_hook() {
    // the control: the same child without the hook does not exit 0 (test harness: the panic fails the test)
    let s = run_child("none");
    assert_ne!(s.code(), Some(0), "control child unexpectedly succeeded: {s:?}");
    assert_ne!(s.code(), Some(7), "control child printed into a closed pipe without an error: {s:?}");
}
