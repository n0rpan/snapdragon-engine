// Record the engine's git commit (plus "-dirty") in the M0 report.
use std::process::Command;

fn main() {
    let rev = Command::new("git").args(["rev-parse", "--short=12", "HEAD"]).output().ok().filter(|o| o.status.success());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    if let Some(o) = rev {
        let mut s = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if dirty {
            s.push_str("-dirty");
        }
        println!("cargo:rustc-env=OXE_GIT_COMMIT={s}");
    }
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
