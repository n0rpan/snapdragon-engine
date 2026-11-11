# snapdragon-engine

Purpose-built Rust inference engines for the **Snapdragon X Elite CPU** under Linux:

| Engine | Models | What is special | Status |
|---|---|---|---|
| **`oxe`** (`oxe-server --engine oxe`) | Ornith-1.5-35B-A3B, Qwen3.6-35B-A3B (`qwen35moe` family: hybrid attention + gated delta-net, routed + shared experts, MTP head) | static per-model plan (~170–210 barriers per token instead of a ~3,900-node graph), repacked int8 `i8mm` kernels, tiled attention, batched prefill, MTP depth 1 with verify lanes, exact prefix/session reuse | production on the test laptop |
| **Flash** (`oxe-server --engine flash`, crate `qwen4exp`) | Qwen3.8-Flash-Next (125B MoE, 512 experts, UD-IQ3_XXS, 83.6 GB) | dense weights in RAM, experts streamed from NVMe through an LFU cache with per-layer quotas, O_DIRECT IO on a dedicated core, layer-major batched prefill, measured memory admission | production on the test laptop |
| **`oxi`** (crates `int8-gemm`, `image-dit`, `image-flow`, `image-vae`) | FLUX.2 klein | f32 reference within its own 1-ulp noise of stable-diffusion.cpp; int8 `SMMLA` linears (4–5× per block) | **experimental**: with f32 attention it passes the 3-seed image quality gate at 512 px; with int8 attention it fails; not deployed, no speed claim yet |

Measured on a Lenovo Yoga Slim 7x (X1E80100, 32 GB), CPU only, 2026-11-10/11: **stock software → today's installed engines**. Receipt ids refer to the lab repository's `benchmarks/results/public/<id>.json` (′ → `-prime`). Where the two sides differ in metric or context the row says so; do not multiply rows.

| Workload | Stock | Today (installed) | Same conditions? | Evidence |
|---|---|---|---|---|
| Flash 125B MoE decode, experts from SSD | BigMoeOnEdge, 12 GiB expert cache: 356.49 ms/token (2.8 tok/s) | Flash engine, 14 GiB cache, prefill order 2: **81.13 ms/token (12.3 tok/s)**, 4.4× | no: cache and metric differ; same metric at 16 GiB: 84.90 ms/token (4.2×) | G3, 063, G3′ |
| Ornith 35B-A3B decode, short prompt | pinned llama.cpp server, stock device tree and threads: 12.73 tok/s | `oxe`, VL15 + MTP depth 1: **54.4 tok/s**, 4.3× | close: different harness and output length (candidate `24caf62`, installed as `bbc2990`) | lab ROOFLINE, G1′ |
| Ornith decode, 8K / 32K | not measured | **55.8 / 34.5 tok/s** (plain on the same binary: 41.3 / 30.4) | today vs plain only; token ids identical | QC-CTX2 |
| Ornith / Qwen3.6 prefill, cold | upstream llama.cpp `llama-bench` pp512: 121.14 / 172.33 tok/s | 2,048 tokens, mask 1021 / group 2: **276.77 / 261.64 tok/s** | no: 512 vs 2,048 tokens; same length vs previous production 198.13 / 177.71 | G1, G2 |
| Qwen3.6 decode | upstream llama.cpp tg128 (empty context): 26.18 tok/s | MTP depth 1: **41.0 tok/s at 2K**, 39.9 at 8K | no: context differs | QC-CTX2-Q |
| Fixed 2K OpenCode request, whole request | stock not measured; previous production Rust: Ornith 13.27 s | **9.82 s** Ornith; Qwen 12.11 → **8.36 s** | yes: same request, identical token hashes | G0′, QC-CTX2, G2′ |
| Images 512 px through Studio (experimental `oxi`) | stable-diffusion.cpp: 77.85 s | **51.53 s** (−33.8%), 45.94 s with a cached prompt, installed live smoke 47.42 s; quality gate passed | yes: same path, seeds, A B B A | IMG-SHIP-1 |

Flash MTP was built and measured (no gain: a 2-row verify costs 1.77× a plain step) and is kept out of `main` under the tag `archive/flash-mtp-w4c`.

Full tables, conditions, method and the list of what did not work are in the companion **lab repository** (<https://github.com/n0rpan/snapdragon-local-ai-lab>): `docs/RESULTS.md`, `docs/ARCHITECTURE.md`, `docs/REPRODUCE.md`. The lab repository also has the controller that serves these engines behind one OpenAI-compatible endpoint, and the harness configs (OpenCode, pi, dsh).

## Build

Linux ARM64, Rust stable, GCC 16 (the vendored ggml kernels are built with `-O3 -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme -fno-fast-math`; bit-exactness against llama.cpp was established with GCC 16). `.cargo/config.toml` sets `target-cpu=oryon-1`; change it for other cores.

```sh
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p server -p oxe-cli                  # oxe
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -p server --features flash  # + Flash
```

**Flash build prerequisite.** The Flash graph port links the *production* ggml static libraries (`libggml-base.a`, `libggml-cpu.a`) of a BigMoeOnEdge build, so its numerics match the engine Flash replaced bit for bit. Build them once, then copy them in:

1. Fetch the public sources and apply the one-line demand-mapping patch exactly as in the lab repository's `docs/FLASH-REBUILD.md` ("Reconstruct in a new directory"): [Helldez/BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) `930b2c9ebd8c2cface6566bd9f11075ac07adfe0` (Apache-2.0) with its llama.cpp fork [Helldez/llama.cpp](https://github.com/Helldez/llama.cpp) `7ac98f4a8089a36c0581b7c6e8973417a8fa175c` (MIT) plus `patches/flash-demand-mapping.patch` (lab repository) as a local commit; the patched llama.cpp tree must be `4f0c6498e9ac5a2b098c99a0d8398dfc738fbdef`. Our private commit of that tree is `04ff7df`; the patch only touches `src/llama-model.cpp`, so the ggml sources are exactly those of `7ac98f4`.
2. Configure with the CMake command of `docs/FLASH-REBUILD.md` ("Build with the qualified CPU configuration": Release, `-mcpu=oryon-1+rng+crc+nopauth+dotprod+i8mm+nosve+nosme`, `GGML_OPENMP=ON`, `GGML_LLAMAFILE=ON` (default), `GGML_CPU_REPACK=OFF`, `GGML_CPU_KLEIDIAI=OFF`, static libraries), then build only the two libraries: `cmake --build "$FLASH_WORK/build" --parallel 2 --target ggml-base ggml-cpu`.
3. `BIGMOE_BUILD="$FLASH_WORK/build" tools/sync_ggml_prod.sh "$FLASH_WORK/parent"` copies them and the ggml headers with a SHA-256 `MANIFEST` into `$OXF_GGML_PROD` (default `~/.cache/local-ai/snapdragon-engine/ggml-prod`). Without them `build.rs` stops with "production ggml missing".

Reference hashes of the production libraries (GCC 16.1.1, CMake 4.4.4, Ninja 1.13.2): `libggml-base.a` `bb0ca3f8…ba067`, `libggml-cpu.a` `adf384be…d01`. Another toolchain gives other hashes; `flash-m3 selftest` and the lab's Flash identity gates are the check, not the hash. Licences: BigMoeOnEdge Apache-2.0, llama.cpp/ggml MIT; nothing from either is redistributed here except the MIT ggml files under `vendor/`. Linking the vendored `vendor/ggml-flash` instead (no separate build) is planned and needs a fresh bit-identity proof.

Model-free tests:

```sh
cargo test --release -p kernels-sys -p model -p sched -p tokenizer -p chat -p server
cargo test --release -p server --lib --features flash
./target/release/flash-m3 selftest
```

## Run

```sh
./target/release/oxe-server --engine oxe --model Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 32768 --oxe-cpus 0-7 --oxe-prefill-cpus 0-11 \
  --spec-type draft-mtp --spec-draft-n-max 1
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

`oxe-server` speaks the llama-server / OpenAI wire format (chat completions, streaming, tool calls) and accepts and ignores most llama-server flags it does not need. Run it without arguments for the full usage. Flash needs its expert sidecar (`flash-m0 sidecar`, see `M0-FLASH.md`) and enough free RAM for the chosen cache tier (lab `docs/ARCHITECTURE.md`, "Memory admission").

Weights are not included: download them by the pinned revisions and SHA-256 in the lab repository's `dependencies/model-artifacts.json`.

## Layout

```
Cargo.toml              workspace; release = lto fat, codegen-units 1, panic abort
vendor/ggml-subset/     verbatim extract of the ggml CPU kernels used + PROVENANCE.md
crates/
  kernels-sys/          cc build of the subset, hand-written FFI, FFI-vs-reference tests
  gguf/                 mmap GGUF reader
  sched/                pinned persistent pool, sense-reversing spin barrier, LPT balancer
  model/                qwen35moe descriptor, load-time repack, static plan, MTP, attention
  tokenizer/ chat/      GGUF BPE tokenizer; chat templates + Qwen3-Coder tool-call parser
  server/               oxe-server (engines: oxe, flash, stub)
  qwen4exp/             Flash graph, expert cache, IO queue, prefill planner, flash-m* tools
  int8-gemm/ image-*/   oxi image engine (experimental)
  oxe-cli/              oxe (gen/trace/compare/verify/bench/cache), barrier-bench, gguf-dump
tools/oracle/           llama.cpp reference dumper used for the bit-exactness gates
M0.md … M3-*.md         milestone write-ups: gates, measurements, design decisions
```

## Correctness

Every change declares a class before it is measured: **lossless** (bit-identical logits/state against the native-layout llama.cpp reference or the previous path), **reorder** (different float order, mean KLD within 2× the reference's own batch noise) or **lossy** (opt-in only). Bit identity is the bug detector; see the milestone docs and the lab's `docs/RESULTS.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the lab repository's `docs/HELP-WANTED.md` (NPU prefill, image int8 attention, Flash expert prediction, shared-expert verify, parser fuzzing, …).

## Licence

MIT, copyright Oscar Nordström (n0rpan) (see `LICENSE`). Vendored code keeps the licence of its source: `vendor/ggml-subset/` and `vendor/ggml-flash/` are MIT, the ggml authors; the image crates port code from stable-diffusion.cpp (MIT, `third_party/stable-diffusion.cpp-LICENSE`); code ported from llama.cpp is named in `THIRD_PARTY_NOTICES`. The SSD-streaming approach for Flash was started from [BigMoeOnEdge](https://github.com/Helldez/BigMoeOnEdge) (Apache-2.0), whose source is not vendored here.
