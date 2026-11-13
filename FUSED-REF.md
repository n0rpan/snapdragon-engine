# FUSED-REF: the O-117 quality runner (fa2 / fa2x prefill, fused decode)

`oxe fused-ref --candidate prefill-fa2 | prefill-fa2x | fused-decode` is the quality gate O-117 pre-registers
for v0.1.3: clauses (a)-(c) at 32K and 64K. Clause (d) (tool-JSON 100% on one long-session closed-loop
replay at ≥ 48K) is run separately by the closed-loop runner.

It is a bench command. It is built only with `--features f32-reference` and is never a server setting.
Production builds do not change: every hook it uses (`Engine::set_pf_ref`, `set_pf_check`, the prefill
counters, `head_logits`, `set_attn_ref`, `set_attn_check`) exists only under `cfg(test)` or that feature.

## Scope statement (for the receipt)

> O-117 quality runner: teacher-forced from one saved prefix state, every position compared (candidate vs
> tiled incumbent, and a measured control of the incumbent on the same positions); f32-attention reference
> with production weights (f64-accumulated attention over the same q and stored f16 KV; weights, matmuls and
> recurrent layers are the production CPU path); NOT a full-model f32 reference; clause (d) tool-JSON is run
> separately

Each receipt also carries a `scope_label` naming the candidate, the prefix, the row counts and the control.
A full-model f32 reference is not feasible at 32K-64K (a 35B model needs about 140 GB of f32 weights).

## What it runs

All arms start from the SAME saved prefix state (`--state-file`, built once by the tiled prefill and read
back before every arm) and take the same teacher-forced continuation of `--n` tokens (default 1000).

**Prefill candidates (`prefill-fa2`, `prefill-fa2x`): one engine, four arms.**

| arm | what runs |
|---|---|
| incumbent | batched prefill of the continuation, `OXE_PF_ATTN=tiled`, `--ubatch` (default 256, production) |
| candidate | the same batched prefill with `fa2` or `fa2x` (every unit beyond `PFA_MIN_KV` = 8192; at 32K / 64K that is every unit) |
| reference | the same batched prefill with the f32-attention reference as the attention core of every row |
| control | `--control decode` (default): tiled decode, one token per step; or `--control ubatch:N`: tiled batched prefill at another ubatch |

The batched arms use `Engine::prefill_ext`: the production ubatch / grouping, plus every row's output-normed
hidden state. The logits of every row come from the production head (`head_logits`: activation quantisation
and the head GEMV, bit-identical to the batched prefill's own head, tested). So all `--n` positions are
compared, for the candidate AND for the control, at the same positions. This replaces the earlier
`--control`, which compared only the 250 chunk ends of 1000 positions.

**Proof that the batched path ran.** In bench / test builds the engine counts batched passes, rows per pass
and attention units per kernel (tiled, per-head, fa2, fa2x, f32 reference). The receipt lists them per arm
(`path.arms`). The verdict requires all of the following (`path.ok`, problems in `path.problems`):

- every candidate row came from a batched pass;
- the largest candidate pass is more than 1 row;
- the candidate kernel ran units and the other kernel did not;
- the incumbent ran only tiled units;
- the reference ran only reference units;
- the control covered every position.

A scalar-step fa2 run (ubatch 1, or decode) can never be labelled a pass.

**`fused-decode`: two engines.** A (tiled decode) and B (fused decode, cutoffs 0) step the continuation. A is
forked at the sampled positions and at every top-1 disagreement: one step runs three ways over A's stored
state (f32-attention reference, fused, tiled), and the replayed tiled step must equal the un-forked one bit
for bit. The control must be batched (`--control ubatch:N`, default `ubatch:256`). It is the tiled batched
prefill of the same tokens, compared at every position with the decode incumbent.

## The clauses (all must hold, plus scope and path; exit 1 otherwise)

| | rule |
|---|---|
| scope | ≥ 1000 compared rows, control rows at the same positions, ≥ 200 sampled reference positions |
| (a) KLD | mean KL(P_cand ‖ P_tiled) ≤ `--kld-factor` (1.0) × mean KL(P_control ‖ P_tiled). Both use the same direction and the same positions |
| (b) top-1 | agreement(cand, tiled) ≥ agreement(control, tiled) − `--top1-slack-pp` (0.5) pp. The floor is the measured control's (dynamic, not absolute) |
| (c) logit error | over the sampled positions, the RMS logit error vs the reference: candidate median ≤ 1.1 × tiled median, and the same for p90 |
| (c) jumps | positions with RMS error > 1e-3: candidate jumps ≤ tiled jumps + max(2, ceil(3% × sampled positions)). 200 samples: allowance 6. 1000 samples: 30 |
| (c) attention | candidate attention-output rel-L2 ≤ 1.1 × tiled |
| (d) | not run here (closed-loop runner) |

How the attention level of (c) is measured for prefill: in the incumbent's and the candidate's own batched
passes, the gated attention outputs of the sampled rows are compared with the f32-attention reference
computed on THEIR OWN q and stored K/V (all layers and heads). The result is sqrt(Σ err² / Σ ref²).

Jump allowance: the code had max(2, ceil(0.03 × samples)), as Codex assumed. It is kept, with one fix: a
1e-9 guard so that float rounding never adds one. 0.03 × 100 is 3.0000000000000004 in floating point, so
the old code gave 4 instead of 3. Results at 200 and 1000 samples are unchanged.

Why median, p90 and jumps, and not the mean: the per-position logit error is heavy-tailed. About 10% of
positions jump to 1e-2..1e-1 after any ~1e-4 attention change; these are most likely MoE routing flips
(`tests/fprec.rs`).

## Model-free tests (CI)

On the mini model:

- `tests/longctx/fused_ref_o117.rs`: prefill. Covers the full 1000-row shape with the counters, independent
  recomputation of the KLD direction and the head, a degraded fa2 that fails (c), ubatch 1 that is never a
  pass, the in-place f16 reference bit-identical to `ref_attn_f64`, and the gate arithmetic.
- `tests.rs` `mini_fused_ref_*`: fused decode. Covers fork transparency, the 250 + 250 rows, the KLD
  direction, and a degraded fused path that fails.

## Commands (Snapdragon, under the inference lease, one model at a time)

```sh
# build (bench binary only)
cargo build --release -p oxe-cli --features f32-reference --bin oxe
M=<model gguf>; ST=<0700 state dir>; S=<result dir>; IDS=<ids file, >= 65536 real long agent/code ids>
for C in 32768 65536; do
  for K in prefill-fa2 prefill-fa2x; do
    oxe fused-ref --candidate $K --model $M --cpu-list 0-7 --prefix-len $C --prefix-ids-file $IDS \
      --state-file $ST/pf-f16-$C.state --n 1000 --ref-samples 200 --ubatch 256 --control decode \
      --out $S/o117-$K-$C.json
  done
  oxe fused-ref --candidate fused-decode --model $M --cpu-list 0-7 --prefix-len $C --prefix-ids-file $IDS \
    --state-file $ST/pf-f16-$C.state --n 1000 --ref-samples 200 --control ubatch:256 \
    --out $S/o117-fused-decode-$C.json
done
```

- `--dry-run` reads only the tokenizer and prints the plan.
- The state kind is `prefill-bench-prefix`, so a prefill-bench state of the same ids is reused. The state is
  built with the tiled prefill.
- `OXE_PF_FA2X_FOLD` / `OXE_PF_FA2X_CH` tune fa2x as in production.

## Cost estimate (Ornith / Qwen 35B-A3B; estimates, not measured)

| per run, state reused | 32K | 64K |
|---|---|---|
| prefill-fa2 / -fa2x: 3 batched passes + 4-5 state reads | ~1-2 min | ~2-3 min |
| reference pass (f32 attention for all 1000 rows; ~2.7 / 5.4 T f64 MAC) | ~6 min | ~12 min |
| attention checks (200 rows × 2 arms) | ~2.5 min | ~5 min |
| control decode (1000 steps) + 3000 head GEMVs | ~1.5 min | ~2.5 min |
| **prefill candidate total** | **~11-12 min** | **~21-23 min** |
| fused-decode (2 engines, 200+ forks, batched control) | ~7-9 min | ~13-16 min |
| state build, once per context (if not reused) | ~3-4 min | ~11 min |

All three candidates at both contexts come to about 1.5-1.8 h with the states reused.

RAM beyond the mmapped weights: a prefill run uses one engine, ~1.6 GiB at 64K. fused-decode uses two
engines, ~4.5 GiB at 64K.

128K is out of scope.
