// The device backend is C (npu/oxn/host/oxn_host.c) built against the open-source FastRPC headers.
// Only compiled when the `device` feature is on; the default build touches no C at all.
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../../npu/oxn/host/oxn_host.c");
    println!("cargo:rerun-if-changed=../../npu/oxn/host/oxn_host.h");
    println!("cargo:rerun-if-changed=../../npu/oxn/common");
    println!("cargo:rerun-if-env-changed=FASTRPC_INC");

    if std::env::var_os("CARGO_FEATURE_DEVICE").is_none() {
        return;
    }
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let inc = std::env::var("FASTRPC_INC").unwrap_or_else(|_| {
        panic!(
            "oxn: the `device` feature needs FASTRPC_INC=<dir> (the open-source qualcomm/fastrpc inc/ dir with remote.h and AEEStdErr.h)"
        )
    });
    cc::Build::new()
        .file(root.join("../../npu/oxn/host/oxn_host.c"))
        .include(root.join("../../npu/oxn/common"))
        .include(&inc)
        .opt_level(2)
        .flag("-std=gnu11")
        .warnings(true)
        .compile("oxn_host");
    println!("cargo:rustc-link-lib=dl");
}
