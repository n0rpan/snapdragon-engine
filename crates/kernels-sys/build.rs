// Builds the vendored ggml subset plus our shims with EXACTLY the flags llama.cpp uses for the
// library each item comes from (read from the pinned build's build.ninja, KleidiAI off):
//   ggml-cpu  (repack kernels, arm quants, vec.cpp, ported ops):
//             -O3 -DNDEBUG -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme
//   ggml-base (ggml-quants.c: quantize_row_q8_K_ref, dequantize_row_*): -O3 -DNDEBUG, no -march
// Scalar float code is FMA-contracted by GCC (-ffp-contract=fast is the GNU default), so the
// flags and the compiler are part of the numerics; see PROVENANCE.md.
use std::path::PathBuf;

const CPU_ARCH: &str = "-march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme";

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let vendor = root.join("vendor/ggml-subset");
    let generated = vendor.join("generated");
    let upstream = generated.join("upstream");
    let csrc = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("csrc");

    let base = |b: &mut cc::Build| {
        b.opt_level(3).define("NDEBUG", None).define("_GNU_SOURCE", None).define("_XOPEN_SOURCE", "600").warnings(false);
        // cc adds its own -march/-mcpu for some targets only when asked; be explicit about none
        b.flag_if_supported("-fno-fast-math");
    };

    // ggml-cpu C items
    let mut c1 = cc::Build::new();
    base(&mut c1);
    c1.include(&vendor).include(&generated).flag(CPU_ARCH).file(generated.join("quants_subset.c")).flag("-std=gnu11").compile("oxe_ggml_c");

    // ggml-cpu C++: repack kernels + interleavers (shim includes repack_subset.cpp)
    let mut cpp = cc::Build::new();
    base(&mut cpp);
    cpp.cpp(true)
        .include(&vendor)
        .include(&generated)
        .include(&csrc)
        .flag(CPU_ARCH)
        .flag("-std=gnu++17")
        .file(csrc.join("oxe_shim.cpp"))
        .file(csrc.join("oxe_native_quant.cpp"))
        .compile("oxe_ggml_cpp");

    // ggml-cpu C++: upstream vec.cpp verbatim + the ported ops (against the upstream headers)
    let mut ops = cc::Build::new();
    base(&mut ops);
    ops.cpp(true)
        .include(upstream.join("src"))
        .include(upstream.join("src/ggml-cpu"))
        .include(upstream.join("include"))
        .flag(CPU_ARCH)
        .flag("-std=gnu++17")
        .file(upstream.join("src/ggml-cpu/vec.cpp"))
        .file(csrc.join("oxe_ops.cpp"))
        .compile("oxe_ggml_ops");

    // ggml-base items (no -march), C. Compiled (and so linked) LAST: oxe_ggml_cpp calls
    // quantize_row_q8_K_ref, and a single-pass static link only resolves it from a later archive
    // (the kernels-sys lib test failed to link in CI otherwise). Link order only; same objects.
    let mut c0 = cc::Build::new();
    base(&mut c0);
    c0.include(&vendor).include(&generated).file(generated.join("quants_base.c")).flag("-std=gnu11").compile("oxe_ggml_base");

    println!("cargo:rerun-if-changed={}", root.join("npu/oxn/common").display());
    println!("cargo:rerun-if-changed={}", vendor.display());
    println!("cargo:rerun-if-changed={}", csrc.display());
    println!("cargo:rustc-env=OXE_KERNEL_CFLAGS=ggml-cpu: -O3 -DNDEBUG {CPU_ARCH}; ggml-base: -O3 -DNDEBUG");
}
