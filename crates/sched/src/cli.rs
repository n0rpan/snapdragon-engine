//! Command-line helpers shared by the engine's binaries (F-CI9).

/// Make a print to a closed stdout end the process with status 0 instead of a core dump.
///
/// Rust ignores SIGPIPE, so `println!` into a closed pipe (`flash-m3 selftest | head -1`) panics with
/// "failed printing to stdout: Broken pipe", and release builds are `panic = "abort"`: a SIGABRT and a
/// core dump for what the shell considers normal. This panic hook turns exactly that panic into
/// `exit(0)` (the reader has what it wanted); every other panic goes to the previous hook unchanged.
/// For print-and-exit command-line tools only: a serving process should not call it (there a closed
/// stdout is a fault worth a restart).
pub fn exit_zero_on_epipe() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if is_epipe_print_panic(info.payload_as_str()) {
            std::process::exit(0);
        }
        prev(info)
    }));
}

/// True for the panic message of a `print!` / `println!` (stdout or stderr) that hit a closed pipe.
pub fn is_epipe_print_panic(msg: Option<&str>) -> bool {
    msg.is_some_and(|m| m.starts_with("failed printing to std") && (m.contains("Broken pipe") || m.contains("os error 32")))
}
