# Engine lab notes index

Milestone write-ups, working logs and one superseded server note. They record what was gated, measured and tried. They are not the user guide: for building, downloading and running the engines, start at the [README](../../README.md).

The engine notes carry their measurements inline and have no public receipt of their own. The published numbers and their receipts are in the lab repository, `benchmarks/results/public/`. The `PROGRESS-*` working logs stay in the development repository and are not part of the public snapshot, so they are named here without a link. Each row gives the note's last-change date, its outcome in one line, and the receipt it cites.

| Note | Date | Outcome | Receipt |
|---|---|---|---|
| [M0.md](M0.md) | 2026-11-11 | Roofline probe for `oxe`: a timing-only runner for the static per-token plan (about 171 barriers); the design's kill test, superseded by M1. | none (in-note) |
| [M0-FLASH.md](M0-FLASH.md) | 2026-11-11 | Flash timing prototype with real weights, a static plan and SSD expert streaming. Numerics were not checked; superseded by M1-FLASH. | none (in-note) |
| [M1.md](M1.md) | 2026-11-11 | Real `qwen35moe` decode math: the gate passed on Ornith, with every traced tensor and logit row bit-identical to llama.cpp. | none (in-note) |
| [M1-FLASH.md](M1-FLASH.md) | 2026-11-11 | Real Flash forward pass, checked against the production answer; the decode work continued in M3 and its working log is PROGRESS-M3. | none (in-note) |
| [M2.md](M2.md) | 2026-11-10 | Full-token decode bit-identical to the native-layout llama.cpp on both models (gate passed); speed was not timed in this note. | none (in-note) |
| [M3-ATTN.md](M3-ATTN.md) | 2026-11-11 | Long-context tiled attention, bit-identical to llama.cpp FA-off; all gates passed. | none (in-note) |
| [M3-FLASH-IO.md](M3-FLASH-IO.md) | 2026-11-11 | Flash expert read path: what a cache miss costs and the I/O design. Model-free bench plus estimates; no model run for this document. | none (in-note) |
| [M3-MTP.md](M3-MTP.md) | 2026-11-11 | Exact MTP speculative decoding, merged: Qwen3.6 33.07 to 43.05 tok/s at depth 2 (+30%); Ornith flat at depth 1. | none (in-note) |
| [M3-PREFILL.md](M3-PREFILL.md) | 2026-11-11 | Batched prefill bit-identical to llama.cpp's batched prefill at 64, 512 and 2048 tokens on both models (gate passed). | none (in-note) |
| [P1-SESSIONS.md](P1-SESSIONS.md) | 2026-11-11 | Design for turn checkpoints and parked sessions at 32K (gate: exact, no OOM at 28 GiB). The lab-side setup is in the lab note MOE-BATCH1. | none (design) |
| `PROGRESS-M1.md` (working log) | 2026-11-10 | Working log for Flash M1 (real forward pass against production); strategy and checkpoints. | none (in-note) |
| `PROGRESS-M3.md` (working log) | 2026-11-10 | Working log for the Flash fast decode path, starting from 196 ms/token at M1; spawned the PF, T2 and PROD logs. | none (in-note) |
| `PROGRESS-PF.md` (working log) | 2026-11-10 | Working log for the Flash prefill workstream: lossless changes only, one ticket per change. | none (in-note) |
| `PROGRESS-T2.md` (working log) | 2026-11-10 | Working log for MTP speculative decoding in the Rust Flash engine. | none (in-note) |
| `PROGRESS-SV.md` (working log) | 2026-11-10 | Working log for Flash through `oxe-server`; superseded by PROGRESS-PROD. | none (in-note) |
| `PROGRESS-PROD.md` (working log) | 2026-11-10 | Working log for Flash production on the Rust engine; supersedes PROGRESS-SV. | none (in-note) |
| [SERVER.md](SERVER.md) | 2026-11-11 | **Historical.** Early oxe-server integration note; its opening "model is still a stub" is out of date. | none |
