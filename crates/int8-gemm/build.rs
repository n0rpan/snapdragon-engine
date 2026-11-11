// The micro-kernels are C with NEON intrinsics (SMMLA needs +i8mm; std::arch's vmmlaq_s32 is not stable).
// -ffp-contract=off: the float fix-up order is part of the API contract (see src/lib.rs), so the
// compiler must not fuse a multiply and an add unless the code asks for it (vfmaq_f32).
fn main() {
    let dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("csrc");
    cc::Build::new()
        .file(dir.join("i8g_kernels.c"))
        .file(dir.join("i8g_sinks.c"))
        .opt_level(3)
        .define("NDEBUG", None)
        .flag("-std=gnu11")
        .flag("-march=armv8.6-a+dotprod+i8mm+fp16+nosve+nosme")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .warnings(true)
        .compile("oxe_int8_gemm");
    // F-IMG-ATTN float attention: built on its own so +fp16fml (FMLAL) does not touch the int8 kernels' flags.
    cc::Build::new()
        .file(dir.join("i8g_fattn.c"))
        .opt_level(3)
        .define("NDEBUG", None)
        .flag("-std=gnu11")
        .flag("-march=armv8.6-a+dotprod+i8mm+fp16+fp16fml+nosve+nosme")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .warnings(true)
        .compile("oxe_int8_gemm_fattn");
    println!("cargo:rerun-if-changed={}", dir.display());
}
