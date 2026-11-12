# snapdragon-engine

Purpose-built Rust inference engines for the **Snapdragon X Elite CPU** under Linux. They run 35B mixture-of-experts models at interactive speed, and stream a 125B MoE from SSD. Speed numbers link their receipts in the lab repository, and every speedup is checked against exact references.

| Engine | Models | What is special | Status |
|---|---|---|---|
| **`oxe`** (`oxe-server --engine oxe`) | Ornith-1.5-35B-A3B, Qwen3.6-35B-A3B | Static per-model plan (about 170–210 thread barriers per token instead of a ~3,900-node graph), repacked int8 `i8mm` kernels, tiled attention, batched prefill, MTP depth 1 with verify lanes, exact prefix and session reuse | production on the test laptop |
| **Flash** (`oxe-server --engine flash`, crate `qwen4exp`) | Qwen3.8-Flash-Next (125B MoE, 83.6 GB) | Dense weights in RAM, experts streamed from NVMe through an LFU cache, O_DIRECT reads on a dedicated IO core, measured memory admission | production on the test laptop |
| **`oxi`** (image crates) | FLUX.2 klein | f32 reference at PSNR 31.5–34 dB against stable-diffusion.cpp (gate 27.8 dB); int8 linears | **experimental, opt-in**; 512 px fast path only, other sizes stay on stable-diffusion.cpp |

## Hero: stock day one → installed today

Measured on a Lenovo Yoga Slim 7x (X1E80100, 32 GB), CPU only. Stock means the software and device tree we started with. Receipt ids point to the companion lab repository's `benchmarks/results/public/<id>.json`. Where metric or context differ, the row says so.

| Workload | Stock (day one) | Installed today | Same conditions? | Receipt |
|---|---|---|---|---|
| CPU memory reads, 2 GiB | 48.88 GB/s | 115.85 GB/s (2.37×), best recorded median, MEMLAT overlay | yes, median of 3 | [RAM-MEMLAT](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/RAM-MEMLAT.json) |
| Flash 125B MoE decode, experts from SSD | 356.49 ms/token (2.8 tok/s), BigMoeOnEdge, 12 GiB cache | **81.13 ms/token** (12.3 tok/s), 14 GiB cache, measured before the 18 GiB prefix-store install | no: cache differs. Same metric at 16 GiB: 84.90 ms/token (4.2×) | [G3](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/G3.json) · [063](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/063.json) · [G3-prime](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/G3-prime.json) |
| Image 512 px, whole request in Studio | 77.85 s, stable-diffusion.cpp | **51.53 s** matched A/B (−34%); **24.89 s** installed VAE fix (one live smoke) | yes for 51.53 s; 24.89 s is not a matched comparison | [IMG-SHIP-1](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/IMG-SHIP-1.json) · [VAE-1](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/VAE-1.json) |

The Ornith 35B-A3B decode row is not in this table yet, because no like-for-like day-one receipt is public. The one matched pair that is public is memlat off → on, same harness: 12.73 → 17.18 tok/s (+35%, receipt [docs/MEMLAT.md:11](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/docs/MEMLAT.md#current-evidence)). The installed `oxe` figure of 54.4 tok/s is the mean of two VL15 runs with a 53-token prompt ([G1-prime](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/G1-prime.json)). It was measured under different conditions from that pair; the two do not combine into a speedup.

CPU only; no Adreno GPU or NPU is used in any number. The MEMLAT overlay applies to the installed cells only. The full tables, prompt lengths, cache states and the rows that do not match are in the lab repository: [docs/RESULTS.md](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/docs/RESULTS.md).

## Why purpose-built

Generic runtimes are written for every model on every machine. This engine asks what one chip does for two model families.

- **We measured the hardware first.** Memory bandwidth, thread placement and the cost of each graph step shaped the design. The engines were then built for exactly these models and this SoC.
- **Every speedup is gated on quality against exact references.** Each change declares its class before measurement. *Lossless* changes must be bit-identical to the reference. *Reorder* changes must stay within twice the reference's own batch noise, measured as mean KLD. *Lossy* changes are opt-in only. Bit identity is the bug detector.

Honest limits:

- **Decode is memory-bandwidth-bound.** Each token streams the active weights from DRAM, so the ceiling is the memory system, not the arithmetic.
- **Marketing TOPS are not usable throughput.** Spec-sheet compute numbers are not what a decode step can use.
- **One laptop, one firmware.** The numbers say what this machine did and how to check it.

## The memory discovery

The memlat driver is built into the `linux-omarchy` kernel. The Yoga's device tree lacked the SCMI `protocol@80` node, so the firmware's memory-latency and bandwidth scaling was never enabled, and CPU memory reads were capped at about **49 GB/s**. A one-node device-tree overlay adds the node. It is installed as a separate boot entry and made the default in `/boot/limine.conf`; the stock entry is kept. The same read benchmark then reaches a median of **115.85 GB/s** ([RAM-MEMLAT](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/benchmarks/results/public/RAM-MEMLAT.json)). After two deep suspend and resume cycles the same benchmark measured 108.63 GB/s, so 115.85 GB/s is the best recorded median, not a typical one.

Limits, stated plainly:

- The fix is firmware- and laptop-specific. Read [docs/MEMLAT.md](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/docs/MEMLAT.md) before trying it elsewhere.
- Stability under GPU load is not settled and has no public receipt yet. Upstream discussion: [Omarchy issue #18254](https://github.com/omacom/omarchy/issues/18254).

## Download

Each [release](https://github.com/n0rpan/snapdragon-engine/releases) has ready-made Linux aarch64 binaries. **They are built by GitHub Actions from the tagged source** (`.github/workflows/release.yml`, on GitHub's Linux ARM64 runner, in a Fedora container with a pinned Rust and GCC). Nobody builds or uploads them by hand.

| Tarball | Contents |
|---|---|
| `oxe-server-vX.Y.Z-linux-aarch64.tar.gz` | `oxe-server` with the oxe MoE engine (Ornith, Qwen3.6 35B-A3B) |
| `oxe-server-flash-vX.Y.Z-linux-aarch64.tar.gz` | `oxe-server` with the Flash engine, plus `flash-m0` (expert sidecar) and `flash-m3` (selftest) |
| `oxi-vX.Y.Z-linux-aarch64.tar.gz` | `oxi`, the FLUX.2 klein image engine (experimental) |

Each tarball unpacks to one directory named after it, holding `bin/`, a `README.txt` with the source commit, workflow run, toolchain and minimum glibc, and the licences. No model weights are included.

**Supported CPUs.** The binaries target ARMv8.6-A cores with dotprod, i8mm, fp16 and bf16. Measurements were made on the Snapdragon X Elite (X1E80100) only; the X Plus has not been tested. The packaged `README.txt` says cores without i8mm or bf16 stop with "Illegal instruction"; that has not been tested on such a core. Build from source there with your own `target-cpu`.

**Verify a download** in the download directory:

```sh
sha256sum -c SHA256SUMS --ignore-missing
gh attestation verify oxe-server-vX.Y.Z-linux-aarch64.tar.gz --repo n0rpan/snapdragon-engine \
  --signer-workflow n0rpan/snapdragon-engine/.github/workflows/release.yml
```

The second command checks the Sigstore-signed provenance: that this repository's release workflow built the tarball, and from which commit and tag. It needs GitHub CLI 2.49 or newer. `README.txt` inside the tarball lists the SHA-256 of each binary.

**Numerics.** The published numbers come from binaries built on the test laptop. The CI toolchain can produce different last bits. The model-free tests and the Flash selftest pass on every CI build. For bit-identical outputs, build locally and run the lab's gates.

## Run

Get the default model's weights (pinned revision; 17.4 GB), next to the tarball:

```sh
curl -LO https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF/resolve/4c326713c52d13fdb40983a2a09a46cca8e88739/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf   # 17.4 GB
echo "a7373f724cf3693e8a6a5f8209ae9a06ff8071ae2af60659bdbdf8ff6f6c10df  Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf" | sha256sum -c
```

Then unpack and start the server:

```sh
tar xzf oxe-server-vX.Y.Z-linux-aarch64.tar.gz
cd oxe-server-vX.Y.Z-linux-aarch64
./bin/oxe-server --engine oxe --model ../Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf \
  --host 127.0.0.1 --port 8080 --ctx-size 32768 --oxe-cpus 0-7 --oxe-prefill-cpus 0-11 \
  --spec-type draft-mtp --spec-draft-n-max 1
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

The pinned revisions and SHA-256 of every model are listed in the lab repository's [`dependencies/model-artifacts.json`](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/dependencies/model-artifacts.json). The server speaks the llama-server / OpenAI wire format. It answers local clients only. Set `--api-key-file` to require a bearer key from other local users too.

## Build from source

Linux ARM64, Rust stable, GCC 16. `.cargo/config.toml` sets `target-cpu=oryon-1`; change it for other cores.

```sh
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p server -p oxe-cli
CARGO_BUILD_JOBS=2 nice -n 19 cargo build --release -p qwen4exp -p server --features flash
```

The Flash build first needs two ggml static libraries from a pinned public BigMoeOnEdge build. The exact commands are in the lab repository's `docs/FLASH-REBUILD.md`. The model-free tests are:

```sh
cargo test --release -p kernels-sys -p model -p sched -p tokenizer -p chat -p server
./target/release/flash-m3 selftest
```

## How the two repositories fit together

- **This repository** holds the engines, their correctness rules and their milestone write-ups. The write-ups are in [docs/lab-notes/](docs/lab-notes/); the index is [docs/lab-notes/INDEX.md](docs/lab-notes/INDEX.md).
- **[snapdragon-local-ai-lab](https://github.com/n0rpan/snapdragon-local-ai-lab)** is the controller. It serves these engines behind one OpenAI-compatible endpoint on port 8092, with Studio for images, harness configs (OpenCode, pi, dsh), memory guards, benchmarks and every receipt cited above. Start there to use the models, reproduce numbers, or read the history.

## Layout

```
vendor/ggml-subset/   verbatim extract of the ggml CPU kernels used, with PROVENANCE.md
crates/
  kernels-sys/        cc build of the kernel subset, FFI and reference tests
  gguf/               mmap GGUF reader
  sched/              pinned thread pool, spin barrier, LPT balancer
  model/              qwen35moe descriptor, load-time repack, static plan, MTP, attention
  tokenizer/ chat/    GGUF BPE tokenizer; chat templates and tool-call parser
  server/             oxe-server (engines: oxe, flash)
  qwen4exp/           Flash graph, expert cache, IO queue, flash-m* tools
  int8-gemm/ image-*/ oxi image engine (experimental)
  oxe-cli/            oxe tools: gen, trace, compare, verify, bench, cache
```

## Correctness

Every change declares a class before it is measured: **lossless** (bit-identical logits or state against the reference or the previous path), **reorder** (different float order, mean KLD within twice the reference's own batch noise) or **lossy** (opt-in only). Bit identity is the bug detector; the milestone notes in [docs/lab-notes/](docs/lab-notes/) hold the gates and measurements.

## Help wanted

- A Flash expert-prediction improvement at small caches.
- A shared-expert verify step for MoE speculation.
- Image int8 attention calibration.
- Fuzzing the tool-call parser.
- Numbers from other Snapdragon X laptops, with the receipt.

Each task has context and acceptance criteria in the lab's [docs/HELP-WANTED.md](https://github.com/n0rpan/snapdragon-local-ai-lab/blob/main/docs/HELP-WANTED.md). Read [CONTRIBUTING.md](CONTRIBUTING.md) first. Declare the class of a change before you measure it.

## Licence

MIT; see [LICENSE](LICENSE).

Copyright (c) 2026 Oscar Nordström (n0rpan)

Vendored code keeps the licence of its source: `vendor/ggml-subset/` and `vendor/ggml-flash/` are MIT, and the image crates port code from stable-diffusion.cpp (MIT). Code ported from llama.cpp is named in `THIRD_PARTY_NOTICES`. Model weights and vendor runtimes are not included and have their own terms.

Snapdragon and Hexagon are trademarks of Qualcomm Incorporated. This is an independent project, not affiliated with, sponsored by, or endorsed by Qualcomm.
