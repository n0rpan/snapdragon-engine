# M3-FLASH-IO: the expert read path (F-IO team, 2026-11-10)

> **Historical working note.** The branches and worktrees named here were merged into `main` and then deleted; the commit hashes cited still resolve in `main`'s history. Current state: [README](README.md).

Branch `flash-io` (from flash-m3 a6b2afd = the 2fde61a tree + docs), worktree `<engine checkout>`.
Tags: **[M]** measured here (model-free bench, real sidecar, real route trace), **[M engine]** measured by Codex on the
model, **[D]** arithmetic from [M], **[E]** estimate. No model was run for this document.

## 1. What a miss costs and why

Setup [M, read-only inspection]: the sidecar is on btrfs over dm-crypt (`aes-xts-plain64`), and the mapping is opened
with **`no_read_workqueue`** (`dmsetup table`: `... 3 allow_discards no_read_workqueue sector_size:4096`). The NVMe has
one hardware queue per CPU with its interrupt on that CPU, `max_sectors_kb` 256, scheduler kyber.

Consequence: every 256 KiB bio of a read is decrypted **inline in the completion softirq of the CPU that submitted
it**. The bench confirms it: with one IO thread on core 8, all softirq time of the run is on CPU 8 (0.37 ms per MiB);
with workers on 8, 9, 10 it splits three ways and nothing lands on 0-7. So:

* decrypt never touches the compute cores (good), but
* one IO thread = one core decrypting = at most ~2.3-2.4 GiB/s, and a 2 MiB expert record carries ~0.73 ms of serial
  AES on the critical path, and two misses in one layer are decrypted one after the other.

Anatomy of one read on an awake, spinning IO thread [M, closed loop, depth 1, 1500 reads]:

| read size | 4 KiB | 256 KiB | 512 KiB | 1 MiB | full record (1.5-2.5 MiB, mean 1.9) |
|---|---|---|---|---|---|
| latency p50, ms | 0.078 | 0.402 | 0.529 | 0.747 | 1.092 |

Slope 0.42 ms/MiB, of which decrypt 0.37; the rest (~0.3 ms + ~0.05 ms/MiB) is the device and the submit path. The
device alone moves ~4.4 GiB/s, so the floor for a 2 MiB record is about 0.45 ms device + decrypt / workers.

The engine today is slower than that floor for three separate reasons, each measured:

1. **Serial decrypt** (above).
2. **The IO thread parks between misses.** It spins 20,000 iterations (~0.6 ms) and then sleeps on a condvar; the core
   drops into `cpu-sleep-0` (exit latency 500 us). In decode a layer with a miss comes every ~5 ms, so it parks before
   almost every one (2,261 parks for 2,107 miss layers). Cost: **0.31 ms per miss layer, 5.9 ms/token** (36.3 -> 30.3).
3. **Several misses in one layer queue behind each other's decrypt**: 34.1 misses/token fall in 20.5 layers
   (1.67 per miss layer); layer stall by misses in the layer, one worker: 1: 1.16 ms, 2: ~1.8, 4: ~3.3.

The bench reproduces the engine's own numbers: mean read latency 2.40 ms at 20 GiB in the engine-like arm (engine:
~2.5 ms [M engine, 017]; FD-1 trace at 18 GiB: 2.755 mean, p50 2.12, 45.8 reads/token [M engine]; the replay gives
44.0 misses/token at 18 GiB and 34.1 at 20 GiB, tools/eviction_replay.py 34.2).

## 2. Bench results

Tool: `crates/flash-iobench` (runs the engine's own `io.rs` / `uring.rs`, included by path), runner
`tools/iobench.sh`. Trace: fixed353 (353 decode tokens), cache policy = the engine's LFU + built-in quotas.
"Stall" = time from issuing all of a layer's misses (the FETCH node) until all are READY, summed per token, with
nothing computed meanwhile (so it is an upper bound of what the engine waits; resident-first hides a little of it).
IO workers on cores 8-11, control thread on core 7, cores 0-7 untouched. Both build locks held (quiet machine).

### 2.1 Engine-like replay: 20 GiB, 120 tokens, 1.5 ms of "compute" between layers [M]

| arm | `OXF_IO` | stall ms/token (mean / p50 / p90) | layer stall p50 / p90 / p99 ms | mean read ms |
|---|---|---|---|---|
| **today** (1 thread, parks) | (unset) | **36.13**, rep 36.47 / 33.4 / 61.3 | 1.43 / 2.97 / 6.89 | 2.40 |
| 1 thread, no parking | `spins=1000000` | 30.30 / 27.0 / 53.5 | 1.16 / 2.50 / 5.08 | 2.14 |
| 3 threads, whole reads | `cpus=8,9,10` + spin | 26.31 / 24.2 / 44.1 | 1.16 / 1.95 / 3.69 | 1.59 |
| 2 threads, striped | `cpus=8,9;split=2` + spin | 27.71 / 24.8 / 48.2 | 1.09 / 2.30 / 4.74 | 1.56 |
| **3 threads, striped** | `cpus=8,9,10;split=3` + spin | **23.70**, rep 23.69 / 21.7 / 41.5 | 0.95 / 1.92 / 3.61 | 1.34 |
| 3 threads, striped, default parking | `cpus=8,9,10;split=3` | 29.79 / 26.9 / 51.9 | 1.19 / 2.89 / 4.10 | 1.67 |
| 4 threads, striped | `cpus=8,9,10,11;split=4` + spin | 28.49 / 26.5 / 48.3 | 1.29 / 2.71 / 4.36 | 1.56 |
| today, nodatasum copy | (unset) | 34.97 / 31.6 / 60.8 | 1.39 / 2.98 / 5.59 | 2.36 |
| 3 threads striped, nodatasum copy | as above | 22.27 / 19.8 / 39.5 | 0.87 / 1.84 / 3.63 | 1.26 |

Reading: today -> chosen config = **36.3 -> 23.7 ms/token of pure IO stall (-12.6 ms, -35%)**; the nodatasum copy adds
-1.2 (today's IO) to -1.4 ms (new IO). The fourth worker does not help: core 11 carries the desktop and a part of
0.5 MiB is already mostly device time. Without think time (back-to-back layers, 353 tokens) the same ranking holds:
33.1-33.4 (today) vs 25.7 (3 threads striped).

Quiet-window repeat by Codex (HWCHAR, 20:50, all build slots + the inference lease held, frozen 909fba2; receipt
`<results dir>/hwchar-20261110-203831/fio-data/hwchar.txt`) [M]: today **36.54** -> 1 thread spinning 30.35 ->
3 threads 26.27 -> **3 threads striped 23.51** ms/token; nodatasum 34.78 / 22.12; **18 GiB: 48.48 -> 31.66**. Every row is
within 1% of the table above, so the numbers taken under load (build slots held) stand. One outlier there: nodatasum
with 3 unstriped threads 54.4 (p99 layer stall 24 ms, 65 parks), not reproduced by its neighbours; not used.

18 GiB checks the bench against the engine: FD-1's trace at 18 GiB has 45.8 reads/token, device mean 2.755 ms and an
expert stall of ~52 ms/token (31.4 swiglu wait + ~21 inside the gate op, Opus-B's reading) [M engine]; the replay at
18 GiB has 44.0 misses/token, mean read 2.58-2.63 ms and 48.5-49.9 ms/token of stall. **The bench's "today" arm is the
engine's stall to within ~5%.**

### 2.2 Ring options, 353 tokens, no think time [M]: nothing

| arm | stall ms/token |
|---|---|
| baseline (3 runs) | 33.35, 33.08, 33.08 |
| SQPOLL (kernel submit thread on core 9) | 32.58 |
| DEFER_TASKRUN | 33.10 |
| registered file | 33.08 |
| registered buffer | not available: RLIMIT_MEMLOCK is 8 MiB (ENOMEM) |
| ring depth 4 / 64 (default 32) | 33.16 / 33.04 |

SQPOLL moves the submit, not the decrypt count; depth is irrelevant because a layer never has more than 10 misses
and reads are latency-bound. These stay bench-only options.

### 2.3 Throughput vs depth and workers, closed loop, whole records [M]

| depth | 1 worker: MiB/s, mean ms | 3 workers | 3 workers striped | 4 workers |
|---|---|---|---|---|
| 1 | 1778, 1.11 | - | 2164, 0.91 | (striped) 1808, 1.09 |
| 2 | 2200, 1.77 | 3422, 1.14 | 3455, 1.13 | - |
| 4 | 2339, 3.33 | 4357, 1.79 | 4166, 1.87 | (striped) 3731, 2.09 |
| 8 | 2411, 6.47 | 4446, 3.50 | 4269, 3.65 | 4417, 3.53 |
| 16 | - | - | - | 4448, 7.00 |

One worker saturates at ~2.4 GiB/s (decrypt-bound), three at ~4.4 GiB/s (device-bound). **For the prefill this is
the lever**: its reads are bandwidth-bound, so 3 workers lift the IO ceiling by ~1.85x (FP-4 / the prefill lane's
053). Offset alignment (2 MiB-aligned offsets: 1.099 vs 1.092 ms) and stripe alignment (256 KiB) change nothing.

### 2.4 Speculation: issue layer L+1's misses while layer L runs (oracle = perfect prediction) [M]

3 workers striped, `promote=1`, 1.5 ms think, 120 tokens. "Waste" = extra useless reads issued per layer.

| arm | stall ms/token | note |
|---|---|---|
| no speculation | 23.70 | |
| oracle, 0 waste | **2.78** | the ceiling of any lookahead with this IO: -20.9 ms |
| oracle, 1 wasted read per layer (48 / token) | 13.74 (p50 6.6, p90 41) | noisy run |
| oracle, 2 wasted reads per layer (96 / token) | 12.81 | +0.10 ms per wasted read |
| oracle, 3 wasted reads per layer (144 / token) | 20.30 | +0.12 ms per wasted read |
| oracle, 5 wasted reads per layer (240 / token) | 60.37 | device saturated (4.0 GiB/s): worse than no speculation |
| oracle, 0 waste, old re-issue path (removed since: it was also a correctness bug, section 5) | 5.98, 7.57 | the second read of an in-flight record cost 3-5 ms/token |
| oracle on today's IO (1 thread, parks) | 8.83, 9.62 | |

So a predictor's wasted reads are cheap while the device has headroom and ruinous beyond it: at 1.5 ms per layer
and 4.4 GiB/s there is room for about **3 records per layer in total** (demand + right + wrong). Budget for Opus-B's
predictor: <= ~1.5 wasted reads per layer on average, issued only where the score is high.

### 2.5 The unencrypted copy (O-21: ext4, `/var/lib/local-ai-models/flash/experts-r4.bin`) [M, 21:04-21:08, all slots held]

| sidecar | IO config | stall ms/token, 20 GiB | 18 GiB | one record p50 | max MiB/s |
|---|---|---|---|---|---|
| LUKS | today (1 thread, parks) | 35.9 | 48.5 | 1.13 ms | 2.4k |
| LUKS | 3 striped workers, spinning | 23.5 | 31.7 | 0.89 | 4.4k |
| plain | 1 thread, parks | 23.8, 21.1 | 29.3 | | |
| **plain** | **1 thread, spinning** | **17.3, 17.5** | **23.7** | **0.66** | **4.7k on one thread** |
| plain | 2 threads / 2 striped | 17.7 / 18.4 | - / 25.1 | 0.74 | 4.7k |
| plain | 3 threads / 3 striped | 17.7 / 18.1 | | 0.69 | |
| plain | 1 spinning + oracle lookahead, 0 / 2 wasted reads per layer | 1.5 / 3.5 | | | |

No decrypt softirq, and ext4 completes direct reads inline (16 involuntary switches/s; btrfs + dm-crypt: 350-1000, the
io-wq punts). More threads or striping buy nothing there. Wasted speculative reads are cheap on the plain file
(2 per layer: +2 ms/token, against +10 on LUKS).

## 3. Chosen configuration

**If the plain sidecar is adopted (ticket 056 arm P): `OXF_IO="spins=1000000"`**: one IO thread on core 8 that does not
park; cores 9-11 stay free. Expected [D from M]: LUKS today 36 -> 17.4 ms/token of stall = about -18.5 ms/token (-16%)
at 20 GiB, about -25 at 18 GiB. The rest of this section is the configuration for an ENCRYPTED sidecar (arm B).


```
OXF_IO="cpus=8,9,10;split=3;promote=1;spins=1000000"      # decode: compute on 0-7, IO on 8-10, core 11 left to the OS
```
plus the nodatasum sidecar if 041 confirms it on the model (bench: -1.2 to -1.4 ms/token, i.e. ~1%: below the 3% bar
on its own, so it is "keep the copy only together with this change").

* `spins=1000000` keeps a worker spinning ~30 ms after its last read: no parks inside a decode, parked when the
  server is idle. Cost: three cores at 100% during decode. **Unknown: heat.** The decode replay already runs at
  88-97 C (LEARNINGS T15); if three spinning cores pull the compute clusters' frequency down, the gain shrinks. The
  ticket therefore has an arm with one spinning worker only (bench: -5.9 ms) and one with three.
* Default (OXF_IO unset): one thread, whole reads, 20,000 spins, as before, except that a pending prefetched slot is
  never read twice (section 5).
* Prefill: `cpus=9,10,11` cannot be used with 11 compute threads (no free cores); with 8-9 compute threads + 3 IO
  workers the IO ceiling is 4.4 GiB/s instead of 2.4 [M]. That trade is the prefill lane's A/B (053), not decided here.

### Expected saving on the model [D from M bench + M engine], to be confirmed by ticket 056's A/B

* The bench's "today" stall equals the engine's measured stall at 18 GiB (48.5-49.9 vs ~52 ms/token), so the bench
  difference is the estimate: **-13 ms/token at 20 GiB (36.5 -> 23.5; about -11% of a ~117 ms token) and -17 ms/token at
  18 GiB (48.5 -> 31.7; about -13% of 125.6)**. With the nodatasum copy another -1.4. Evidence class: model-free replay
  of real routes on the real file through the engine's own IO code, repeated in a quiet window; NOT a model A/B.
* It is not the 40-60 ms the 036 post-mortem hoped for: the stall is ~36 ms at 20 GiB to begin with, and the device's
  own ~0.45 ms per record does not parallelise. What is left after this change (23.5 ms) only speculation can hide.
* Not included: heat from three spinning cores (could cost; arm C of 056 is the one-core variant, bench -6.2 ms);
  the resident-first overlap (the engine hides a little of the stall in both arms); the prefill gain (separate A/B).

## 4. Design: issue misses earlier

What exists (flash-m3): the FETCH node runs right after the router's top-k, on thread 0, and queues ALL of the
layer's misses at once at demand class; the shared expert and (resident-first) the units of resident experts run
while the reads fly; then every thread spins on the missing slots. So "all misses at the router step" is done, and
there is no deeper queue to use: 60% of miss layers have exactly one miss.

What remains is the stall measured above: 23.7 ms/token with the new IO (36.3 today), minus what resident-first
hides. Three ways to shrink it, in order of value:

1. **Speculate layer L+1 at layer L's FETCH** (bench ceiling: -20.9 ms/token with a perfect predictor). Mechanism,
   all on thread 0 inside FETCH(L), where the other threads wait anyway:
   * predict L+1's experts (Opus-B's learned lookahead, FD-4; or the router of L+1 applied to the current residual);
   * for each predicted expert that is not resident, issue a **prefetch-class** read into layer L+1's pool. The
     prefill lane's `Cache::prefetch_guarded(io, l, e, keep)` (root-flashpf-43b9232) is the right entry point: it never
     evicts a kept expert and takes the LFU-coldest victim, so a wrong guess replaces the slot that was going next anyway;
   * at FETCH(L+1) the normal demand lookup finds the slot: READY (no stall), or PENDING. With `promote=1` a PENDING
     hit moves still-queued parts to the demand class and otherwise just waits for the parts in flight; without it
     the engine reads the record a second time (bench: 3-5 ms/token lost).
   * IO side is ready: prefetch reads go round-robin over the workers (unsplit, capped at 16 in flight per worker)
     so they do not delay a demand read's parts by more than one 256 KiB bio per worker.
   * Gate for the predictor (replaces s_c / s_w in tools/lookahead_replay.py), measured here: a covered miss saves
     ~0.6 ms/token-layer (23.7 ms / 34 misses = 0.70 ms per miss with the new IO; 1.06 ms with today's IO); a wasted read
     costs ~0.10 ms while total reads stay <= 3 per layer, and the budget is hard: > ~1.5 wasted per layer average
     loses everything. Net [D]: recall 0.5 at 1 wasted read per layer = -11.9 + 4.8 = about -7 ms/token; recall 0.47 with
     42 wasted per token (the router-preview predictor that was called dead at s_c 0.2) = -11 + 4 = about -7 ms as well.
     **That predictor is worth re-running through the replay with these two numbers before training anything bigger.**
2. **Read-ahead arena (prefill lane's proposal).** In prefill the IO idles whenever the layer's pool has no free
   slot. A shared arena of ~256-512 MB outside the pools lets IO run during dense and attention; a slot is then
   adopted by pointer swap. For decode the same arena would hold wrong guesses without evicting anything. It needs
   `Entry.ptr` to be per-slot indirect (today `base + slot * size`), which is a cache.rs change for Opus-B and the
   prefill lane to agree on; the IO layer needs nothing new (a read is just fd, offset, buffer, state word).
   I recommend it for prefill first (bandwidth-bound, 36.6 GiB per pass) and for decode only if (1) shows pollution.
3. **Keep the IO cores awake without spinning** (system setting, needs the machine owner): `cpu-sleep-0` has a 500 us exit latency
   and is the reason parking costs 0.31 ms per miss layer. Disabling that state on cores 8-10 while Flash serves
   (`/sys/devices/system/cpu/cpu{8,9,10}/cpuidle/state1/disable`, or a PM QoS request) would give the no-parking
   latency at near-idle power. Root-only; not done, not tested.

Not worth building (measured or by construction): SQPOLL, registered buffers/files, deeper rings (2.2); a fourth IO
worker (2.1); splitting gate/up/down readiness (an expert's compute is ~0.05 ms, nothing to overlap); per-part
alignment (2.3).

## 5. Code (branch flash-io)

* `crates/qwen4exp/src/io.rs`: `IoTuning` (text form = env `OXF_IO`), up to 4 workers, striped demand reads with a
  join (READY when the last part lands; one failing part fails the request once), `IoQueue::promote`, configurable
  idle spin. Public `Req` / `IoQueue::push` / `spawn` unchanged, so ngram.rs, exec.rs, flash-m0 and the prefill
  lane's `prefetch_guarded` compile as they are.
* `crates/qwen4exp/src/uring.rs`: `RingOpts`, `register_file`, `register_buffer`.
* `crates/qwen4exp/src/cache.rs`: the "demand hit on a pending prefetched slot" branch + two counters.
* **One read per slot (root fix, 8bd5dad).** The old branch read the record a second time at demand priority; the
  first completion flipped READY and the second read rewrote the buffer under the kernels (through dm-crypt +
  O_DIRECT the buffer holds ciphertext until it is decrypted in place): wrong outputs after `Stream::warm` (found by
  the prefill lane, 6 of 6 in its evidence test). Now a demand hit promotes the queued parts and waits for those in
  flight; nothing can switch that off. Regression test `stream::tests::warm_then_batched_op_single_read` (their test,
  un-ignored): 6 of 6 identical with one IO thread and with 3 striped workers, `demand_reissued` 0.
* `crates/qwen4exp/src/stream.rs`: reads `OXF_IO`; `Stream::io_desc`. `flash-m3` JSON: `decode_io.config`, parks, parts,
  promotions, all-thread expert wait.
* Tests (model-free, in `cargo test -p qwen4exp --lib`): stripes cover a read exactly; a striped read over 2-4 workers
  (also with SQPOLL / DEFER_TASKRUN / fixed file) is byte-identical to the file for all four slot sizes; a failing
  part fails the slot once; a queued prefetch is promoted and overtakes the backlog; the tuning text form.
* Semantics change to note: `decode_io.mean_latency_us` now runs from push to READY of the whole request (it was
  SQE submit to completion); with an idle queue the difference is the 0.06 ms queue wait FD-1 measured.
