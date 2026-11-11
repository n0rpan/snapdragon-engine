// Builds the vendored Flash kernel subset (vendor/ggml-flash, see its PROVENANCE.md) with the same
// ggml-cpu flags as crates/kernels-sys: -O3 -DNDEBUG -std=gnu11 -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme.
// GCC FMA contraction (-ffp-contract=fast, the GNU default) is part of the numerics.
use std::path::PathBuf;
use std::process::Command;

const CPU_ARCH: &str = "-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme";

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let v = root.join("vendor/ggml-flash");
    let g = v.join("generated");
    let up = g.join("upstream");
    let mut b = cc::Build::new();
    b.opt_level(3)
        .define("NDEBUG", None)
        .define("_GNU_SOURCE", None)
        .warnings(false)
        .flag("-std=gnu11")
        .flag(CPU_ARCH)
        .flag_if_supported("-fno-fast-math")
        .include(&v)
        .include(&g)
        .include(up.join("include"))
        .include(up.join("src"))
        .include(up.join("src/ggml-cpu"))
        .file(v.join("oxf_kernels.c"))
        .file(up.join("src/ggml-cpu/expert-r4.c"));
    if v.join("compat.c").exists() {
        b.file(v.join("compat.c"));
    }
    b.compile("oxf_flash");

    // PREFILL expert tiles (csrc/pf_expert.c + the tiles of pf_kernels.c), FMA contraction OFF: the fold is explicit
    cc::Build::new()
        .opt_level(3)
        .define("NDEBUG", None)
        .define("_GNU_SOURCE", None)
        .warnings(false)
        .flag("-std=gnu11")
        .flag(CPU_ARCH)
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .include(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc"))
        .include(&v)
        .include(&g)
        .include(up.join("include"))
        .include(up.join("src"))
        .include(up.join("src/ggml-cpu"))
        .file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/pf_expert.c"))
        .file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/pf_expert_iq3.c"))
        .file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/pf_expert_down.c"))
        .file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc/pf_kernels.c"))
        .compile("oxf_pf");

    // The qwen4exp graph port, linked against the PRODUCTION ggml objects (tools/sync_ggml_prod.sh).
    // Compiled with the production ggml-cpu C++ flags; it does no float arithmetic of its own.
    let home = std::env::var("HOME").unwrap_or_default();
    let prod = std::env::var("OXF_GGML_PROD").unwrap_or_else(|_| format!("{home}/.cache/local-ai/snapdragon-engine/ggml-prod"));
    let prod = PathBuf::from(prod);
    if !prod.join("lib/libggml-cpu.a").exists() {
        panic!("{}: production ggml missing, run tools/sync_ggml_prod.sh", prod.display());
    }
    let csrc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc");
    let objs = cc::Build::new()
        .cpp(true)
        .opt_level(3)
        .define("NDEBUG", None)
        .warnings(false)
        .flag("-std=c++17")
        .flag("-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme")
        .flag("-fopenmp")
        .include(prod.join("include"))
        .include(&csrc)
        .file(csrc.join("flash_graph.cpp"))
        .file(csrc.join("flash_dense.cpp"))
        .file(csrc.join("pf_attn.cpp"))
        .file(csrc.join("pf_phase.cpp"))
        .compile_intermediates();
    // prefill tiles (WORKSTREAM PREFILL): C, FMA contraction OFF (the fold order is the contract, see pf_kernels.c)
    let pf_objs = cc::Build::new()
        .opt_level(3)
        .define("NDEBUG", None)
        .warnings(true)
        .flag("-std=gnu11")
        .flag(CPU_ARCH)
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .include(&csrc)
        .file(csrc.join("pf_kernels.c"))
        .compile_intermediates();
    // One relocatable object = the graph port + the production ggml members it needs, with every
    // symbol but the fg_* C API made local. Production ggml is then invisible to the rest of the
    // link (kernels-sys vendors other copies of some ggml kernels for the oxe engine; no clash,
    // no symbol interposition either way). The code is unchanged: same objects, same machine code.
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let all = out.join("fg_prod_all.o");
    let mut ld = Command::new("ld");
    ld.arg("-r").arg("-o").arg(&all);
    for o in objs.iter().chain(pf_objs.iter()) {
        ld.arg(o);
    }
    ld.arg("--start-group").arg(prod.join("lib/libggml-cpu.a")).arg(prod.join("lib/libggml-base.a")).arg("--end-group");
    let st = ld.status().expect("ld -r");
    assert!(st.success(), "ld -r failed");
    let st = Command::new("objcopy").arg("-w").arg("--keep-global-symbol=fg_*").arg(&all).status().expect("objcopy");
    assert!(st.success(), "objcopy failed");
    let lib = out.join("liboxf_flash_graph.a");
    std::fs::remove_file(&lib).ok();
    let st = Command::new("ar").arg("crs").arg(&lib).arg(&all).status().expect("ar");
    assert!(st.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=oxf_flash_graph");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // flash-m0 (timing prototype, dummy activations) still links kernels-sys's vendored repack
    // kernels, which production's libggml-cpu.a also defines; its numerics are not checked, so it
    // may take either copy. The exactness binaries never reference kernels-sys.
    println!("cargo:rustc-link-arg-bin=flash-m0=-Wl,--allow-multiple-definition");
    println!("cargo:rerun-if-changed={}", csrc.display());
    println!("cargo:rerun-if-changed={}", prod.join("MANIFEST").display());
    println!("cargo:rerun-if-changed={}", prod.join("lib").display());
    println!("cargo:rerun-if-env-changed=OXF_GGML_PROD");
    let manifest = std::fs::read_to_string(prod.join("MANIFEST")).unwrap_or_default();
    let cpu_sha = manifest.lines().find(|l| l.ends_with("lib/libggml-cpu.a")).and_then(|l| l.split_whitespace().next()).unwrap_or("unknown");
    println!("cargo:rustc-env=OXF_GGML_PROD_CPU_SHA={cpu_sha}");
    println!("cargo:rerun-if-changed={}", v.display());
    println!("cargo:rustc-env=OXF_KERNEL_CFLAGS=-O3 -DNDEBUG -std=gnu11 {CPU_ARCH}");
    let rev = Command::new("git").args(["rev-parse", "--short=12", "HEAD"]).output().ok().filter(|o| o.status.success());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    let mut s = rev.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_else(|| "unknown".into());
    if dirty {
        s.push_str("-dirty");
    }
    println!("cargo:rustc-env=OXE_GIT_COMMIT={s}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/index");
}
