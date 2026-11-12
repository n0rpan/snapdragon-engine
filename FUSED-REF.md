# FUSED-REF: long-context numeric reference for the fused attention

`oxe fused-ref` is the numeric gate for the O-70 fused decode attention (`OXE_ATTN=fused`) at 32K and 64K.
It is a bench command. It is built only with `--features f32-reference` and is never a server setting.
Production builds do not change: the reference hook (`Engine::set_attn_ref`) exists only under `cfg(test)`
or that feature, and it is `None` unless this command sets it.

## Scope statement (for the receipt)

> f32-attention reference with production weights (fused vs tiled attention over the same stored f16 KV
> and the same saved prefix state; weights, matmuls and recurrent layers are the production CPU path);
> NOT a full-model f32 reference.

A full-model f32 reference is not feasible at 32K-64K. A 35B model needs about 140 GB of f32 weights, and
`f32-export` stays limited to a prompt of 512 or less, 256 rows or less, and a context of 4096 or less.
This command therefore does not claim end-to-end exact-f32 numerics. It isolates the one thing that
changes, the attention core, and measures it at the real depth on the real model.

## What it measures

Two engines (f16 KV, no MTP, the tiled path at every depth) load the same saved prefix state:
A is the incumbent (tiled), B is the candidate (fused). Both then step the same fixed continuation, teacher-forced.

**(a) Candidate vs incumbent.** At each of the `--n` positions (default 1000) the command records:

- KL(P_tiled ‖ P_fused), as mean, p50, p90, p99 and max;
- top-1 agreement.

**(b) f32-attention reference.** This runs at `--ref-samples` evenly spaced positions (default 200), and also
at every position where A and B disagree on top-1. At each of these positions, engine A's state is forked: a
recurrent snapshot plus KV truncation. The same step then runs three ways over the same stored KV:

1. f32 attention math (f32 K/V widened from the stored f16, f64 accumulation of scores, softmax and V);
2. fused;
3. tiled.

The per-position logit error is the RMS logit difference over the full vocabulary. KL(ref ‖ arm) is also
recorded. During the fused and tiled forked steps, every attention core is also checked at the attention
level: its gated output is compared with the f32-attention reference computed on the same roped query and the
same stored KV, summed over all layers and heads (rel-L2). Each fork is checked: the tiled step is replayed afterwards and must be bit-identical to the
un-forked one (`fork_replay_exact`).

## Gate (all must hold; exit 1 otherwise)

| check | rule |
|---|---|
| scope | at least 1000 positions and at least 200 reference samples |
| (a) KLD | mean KL(P_tiled ‖ P_fused) ≤ the noise floor. With `--control` the floor is MEASURED on the same state (below; `noise_floor_source: measured`). `--noise-floor-kld X` alone is labelled "provided (not measured)": Ornith 0.027 is a measurement from elsewhere, and Qwen 0.01 is provisional |
| (a) top-1 | agreement ≥ 99.5% |
| wrong top-1 | 0 positions where the continuous fused top-1 differs from the reference top-1 and the reference's top-1 margin is above 5 logits (`--margin`). The tiled count is reported next to it |
| (b) logit error | over the ≥ 200 sampled positions: fused median ≤ `--ref-ratio-max` (default 1.1) × tiled median, and the same for p90 |
| (b) jumps | positions with logit RMS error > 1e-3 (`--jump-threshold`): fused count ≤ tiled count + max(2, 3% of samples) |
| (b) attention | fused attention-level rel-L2 ≤ 1.1 × tiled |
| fork | every fork replays exactly |

Why median, p90 and jump counts, and not the mean: the per-position logit error is heavy-tailed. About 10%
of positions jump to 1e-2..1e-1 RMS after ANY attention change of ~1e-4, in both arms; these are most likely
MoE top-2 routing flips. A mean over a few samples measures where the jumps landed, not the kernel. The first
version of this gate used the mean over 20 samples and reported fused as ~1000× further from the reference
(1e-2 vs 1.3e-5): two jumps had landed on fused. The fused-precision probe (`tests/fprec.rs`, branch
`claude/fused-precision` 968d97a) showed that fused has no defect.

Mini model (model-free test), 200 samples, current kernels:

| | logit RMS p50 | logit RMS p90 | jumps | attention rel-L2 |
|---|---|---|---|---|
| tiled | 1.1e-5 | 1.5e-2 | 21 | 1.43e-4 |
| fused | 9.5e-6 | 5.9e-5 | 17 | 1.20e-4 |

The gate passes. Over all 250 positions: tiled p50 1.05e-5, mean 5.98e-3; fused p50 8.8e-6, mean 5.06e-3.
A deliberately degraded fused path (output × 1.01, `mini_fused_ref_degraded_fused_fails_the_reference_gate`)
fails: attention rel-L2 1.0e-2, logit p50 0.10, 189 jumps.

With a ~10% jump rate, the p90 sits on the edge of the jump tail. Read it together with the jump counts.

## `--control`: the measured noise floor

`--control` measures the incumbent's own numerical noise on the same saved state, before the fused
comparison. Engine A steps the continuation through the decode path (one token per step, as for generated
tokens). Engine B takes the same tokens through the batched prefill path, in chunks of `--control-ubatch`
(default 4), as for prompt tokens. Both use tiled attention and both are teacher-forced from the same state.

At each chunk end the command records KL(P_decode ‖ P_prefill_batch) and top-1 agreement. With n = 1000
and ubatch 4, that is 250 positions. Both engines are then reloaded from the state file and the fused run
proceeds.

The floor is `--control-factor` (default 1.0) × the control's mean KLD. It is reported under `control`, and
the gate is labelled `noise_floor_source: measured`. A KLD gate counts as "measured" only with `--control`.
Cost: one more 1000-step decode pass plus 250 small prefills, about +1.5 min at 32K and +3 min at 64K.

## Commands

Build once (bench binary, no production change; `-j 2`, nice 19, under build.lock):

```sh
R=<engine checkout at claude/gate-tools>
cd $R && CC="sccache cc" CXX="sccache c++" nice -n 19 cargo build --release -p oxe-cli --features f32-reference --bin oxe -j 2
```

Run under the inference lease, one model at a time, with nothing else loaded:

```sh
M=~/.local/share/ai-stack/models/llm/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf   # Qwen: Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf (same flags; --control measures its floor)
ST=<0700 state dir>; S=<result dir>; IDS=<ids file of >=65536 real long agent/code text ids; the first $C are the prefix>
for C in 32768 65536; do
  $R/target/release/oxe fused-ref --model $M --cpu-list 0-7 --prefix-len $C --prefix-ids-file $IDS \
     --state-file $ST/pf-f16-$C.state --n 1000 --ref-samples 200 --control \
     --out $S/fused-ref-ornith-$C.json
done
```

- The state header kind is `prefill-bench-prefix`. A `prefill-bench --prefix-len $C --prefix-ids-file …
  --state-file …` state of the same model and ids is reused as is. A missing state is built once, chunked,
  and saved.
- `--text-file F` (for example wikitext-2 train) can replace `--prefix-ids-file`.
- The continuation defaults to the built-in synthetic mixed fixture
  (`crates/oxe-cli/fixtures/fused-ref-continuation.txt`: chat markup, tool-call JSON, code, logs and prose,
  parsed with special tokens). `--continuation-file T` replaces it.
- `--dry-run` reads the tokenizer only. It prints the plan (token counts, id hashes, n_ctx) without loading
  weights.

## Cost estimate (Ornith; Qwen similar)

These are estimates, not measured.

| | 32K | 64K |
|---|---|---|
| state build (if not reused) | ~3-4 min | ~11 min |
| (a) 1000 positions, 2 engines + a 63 MiB recurrent snapshot per position | ~2 min | ~4 min |
| (b) ~200+ forks (reference step ~0.8 s / ~1.6 s, plus fused and tiled steps) | ~4 min | ~8 min |
| total with a reused state | ~6-8 min | ~12-15 min |
| RAM beyond the mmapped weights | ~2.5 GiB | ~4.5 GiB |

The RAM figure covers 2 KV caches, plus up to 8 × 2 × n·head_dim f32 transient per reference step.

128K is not in scope. Two 128K engines plus the reference transient (~9 GiB) next to ~20 GB of weights would
page the weights on this 30 GB machine.
