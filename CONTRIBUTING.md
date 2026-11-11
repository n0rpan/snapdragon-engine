# Contributing

The project's contribution guide, measurement rules and task list live in the companion lab repository (<https://github.com/n0rpan/snapdragon-local-ai-lab>): `CONTRIBUTING.md` and `docs/HELP-WANTED.md`. In short:

1. **Declare the class** of a change before measuring it: lossless (bit-identical against the reference), reorder (within measured noise) or lossy (opt-in only).
2. **Interleave arms** (A B B A), drop the warm-up, compare medians; changes under 3% are noise for a decision. This laptop class throttles within ~30 s of all-core load.
3. **Production shapes**: test at n_ctx 16,384/32,768 with real head counts and thread counts.
4. **Keep changes behind a flag**, default off, until measured; a new kernel ships with a model-free bitwise or reference test.
5. **Builds fail loudly**: `set -o pipefail`, check the binary's SHA-256 after building, give every test a timeout.
6. **Privacy**: no weights, no transcripts, no home paths, user or host names, serials or keys in commits or issues.

Before sending a pull request:

```sh
cargo test --release -p kernels-sys -p model -p sched -p tokenizer -p chat -p server
```

plus `-p qwen4exp` / `--features flash` and `flash-m3 selftest` for Flash changes.

Contributions are accepted under the MIT licence. Keep upstream notices in `vendor/` intact and update `vendor/*/PROVENANCE.md` when re-extracting kernels.
