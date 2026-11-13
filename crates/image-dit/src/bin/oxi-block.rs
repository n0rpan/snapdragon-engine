//! oxi-block: M0 prototype driver for one klein single-stream block.
//!
//!   oxi-block selftest [--threads N]
//!       Model-free: tiny random block, int8 fast path against the f32 reference.
//!   oxi-block bench --gguf FILE [--block I] (--dump DIR | --tokens N) [--threads N] [--cpus LIST]
//!                   [--chunk1 C] [--chunk2 C] [--key-chunk K] [--batch-groups B] [--lin2-row-block R]
//!                   [--attn-prec int8|f16|f32[,..]] [--reps R] [--no-ref] [--parity] [--ref-threads N] [--json OUT]
//!       Real weights. `--dump DIR` uses the SDPROBE tensors of the patched sd.cpp (real activations and
//!       sd.cpp's own block output); `--tokens N` uses synthetic activations (timing only: the error
//!       numbers of synthetic inputs say little about real ones).
//!
//!       `--attn-prec` (F-IMG-ATTN): attention arithmetic of the fast path (default int8; the linears
//!       are int8 in every mode). A comma list runs one ARM per value against ONE reference (and
//!       parity reference): the output is then `{"arms": [<one bench object per value>], ...}`.
//!
//! Prints one JSON object on stdout (and to --json).

use image_dit::backend::{SingleBlockBackend, StageTimes};
use image_dit::fast::{FastBlock, FastConfig};
use image_dit::arith::Arith;
use image_dit::attn_float::AttnPrec;
use image_dit::reference::{single_block_ref, single_block_ref_arith};
use image_dit::types::{rel_rmse, BlockInput, BlockOutput, BlockWeightsF32, Dims};
use image_dit::{dump, weights};
use serde_json::json;
use std::time::{Duration, Instant};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn gauss(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit()) * 0.866
    }
}

fn synthetic_input(dims: &Dims, n: usize, seed: u64) -> BlockInput {
    let mut rng = Rng(seed);
    let h = dims.hidden;
    let x = (0..n * h).map(|_| rng.gauss()).collect();
    let mut pe = Vec::with_capacity(n * (dims.head_dim / 2) * 4);
    for _ in 0..n * (dims.head_dim / 2) {
        let a = rng.unit() * std::f32::consts::PI;
        pe.extend_from_slice(&[a.cos(), -a.sin(), a.sin(), a.cos()]);
    }
    BlockInput {
        n_tokens: n,
        x,
        pe,
        shift: (0..h).map(|_| rng.gauss() * 0.1).collect(),
        scale: (0..h).map(|_| rng.gauss() * 0.1).collect(),
        gate: (0..h).map(|_| rng.gauss() * 0.3).collect(),
    }
}

fn delta(out: &[f32], x: &[f32]) -> Vec<f32> {
    out.iter().zip(x).map(|(o, x)| o - x).collect()
}

fn errors(got: &BlockOutput, want: &BlockOutput, x: &[f32]) -> serde_json::Value {
    let mut v = json!({
        "out": rel_rmse(&got.out, &want.out),
        "delta": rel_rmse(&delta(&got.out, x), &delta(&want.out, x)),
    });
    if !got.attn.is_empty() && !want.attn.is_empty() {
        v["attn"] = json!(rel_rmse(&got.attn, &want.attn));
        v["mlp"] = json!(rel_rmse(&got.mlp, &want.mlp));
        v["z"] = json!(rel_rmse(&got.z, &want.z));
    }
    v
}

fn times_json(t: &StageTimes) -> serde_json::Value {
    json!({"norm_quant": t.norm_quant, "linear1": t.linear1, "attention": t.attention, "requant": t.requant, "linear2": t.linear2, "total": t.total})
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn arg_usize(args: &[String], name: &str, default: usize) -> usize {
    arg(args, name).map(|v| v.parse().unwrap_or_else(|_| panic!("{name} needs a number"))).unwrap_or(default)
}

fn make_pool(args: &[String]) -> sched::Pool {
    let threads = arg_usize(args, "--threads", 12);
    let cpus = match arg(args, "--cpus") {
        Some(list) => sched::parse_cpu_list(&list).expect("--cpus"),
        None => (0..threads).collect(),
    };
    assert_eq!(cpus.len(), threads, "--cpus must list exactly --threads cores");
    sched::Pool::new(&cpus, Duration::from_millis(2)).expect("thread pool")
}

fn config(args: &[String]) -> FastConfig {
    let d = FastConfig::default();
    FastConfig {
        chunk1: arg_usize(args, "--chunk1", d.chunk1),
        chunk2: arg_usize(args, "--chunk2", d.chunk2),
        batch_groups: arg_usize(args, "--batch-groups", d.batch_groups),
        lin2_row_block: arg_usize(args, "--lin2-row-block", d.lin2_row_block),
        key_chunk: arg_usize(args, "--key-chunk", d.key_chunk),
        attn_prec: d.attn_prec,
    }
}

/// `--attn-prec` values (comma list), default `[int8]`.
fn attn_precs(args: &[String]) -> Vec<AttnPrec> {
    match arg(args, "--attn-prec") {
        None => vec![AttnPrec::Int8],
        Some(list) => list.split(',').map(|v| AttnPrec::parse(v.trim()).unwrap_or_else(|e| panic!("{e}"))).collect(),
    }
}

fn selftest(args: &[String]) -> i32 {
    let dims = Dims { hidden: 128, heads: 4, head_dim: 32, mlp: 384 };
    let mut rng = Rng(0x5eed);
    let w1f: Vec<f32> = (0..dims.lin1_out() * dims.hidden).map(|_| rng.gauss() * 0.08).collect();
    let w2f: Vec<f32> = (0..dims.hidden * dims.lin2_in()).map(|_| rng.gauss() * 0.04).collect();
    let w1q = weights::quant_q8_0(&w1f, dims.lin1_out(), dims.hidden);
    let w2q = weights::quant_q8_0(&w2f, dims.hidden, dims.lin2_in());
    let q_scale: Vec<f32> = (0..dims.head_dim).map(|_| 1.0 + rng.unit() * 0.3).collect();
    let k_scale: Vec<f32> = (0..dims.head_dim).map(|_| 1.0 + rng.unit() * 0.3).collect();
    // both paths see the same Q8_0 weights
    let wref = BlockWeightsF32 {
        dims,
        w1: weights::dequant_q8_0(&w1q, dims.lin1_out(), dims.hidden),
        w2: weights::dequant_q8_0(&w2q, dims.hidden, dims.lin2_in()),
        q_scale: q_scale.clone(),
        k_scale: k_scale.clone(),
    };
    let pool = make_pool(&[String::from("--threads"), arg(args, "--threads").unwrap_or_else(|| "3".into())]);
    let mut worst = 0.0f64;
    let mut rows = Vec::new();
    let mut ok = true;
    for &n in &[37usize, 64] {
        let input = synthetic_input(&dims, n, 7 + n as u64);
        let want = single_block_ref(&wref, &input, 2);
        for &(c1, c2, kc, prec) in &[
            (32usize, 32usize, 16usize, AttnPrec::Int8),
            (64, 32, 64, AttnPrec::Int8),
            (128, 128, 8, AttnPrec::Int8),
            (32, 32, 16, AttnPrec::F32),
            (32, 32, 16, AttnPrec::F16),
            (128, 128, 8, AttnPrec::F32),
        ] {
            let cfg = FastConfig { chunk1: c1, chunk2: c2, batch_groups: 2, lin2_row_block: 16, key_chunk: kc, attn_prec: prec };
            let mut fast = FastBlock::new(dims, &w1q, &w2q, &q_scale, &k_scale, cfg, &pool);
            let (got, _) = fast.forward(&input, true);
            let (lean, _) = fast.forward(&input, false);
            let same = got.out.iter().zip(&lean.out).all(|(a, b)| a.to_bits() == b.to_bits());
            let e = errors(&got, &want, &input.x);
            let d = e["delta"].as_f64().unwrap();
            worst = worst.max(d);
            ok &= same && d.is_finite();
            rows.push(json!({"tokens": n, "chunk1": c1, "chunk2": c2, "key_chunk": kc, "attn_prec": prec.name(), "errors": e, "repeatable": same}));
        }
    }
    // int8 x int8 on tiny random data: a loose bound that still catches any structural mistake
    // (a wrong row order, head, RoPE pair or scale gives errors of order 1).
    let limit = 0.05;
    ok &= worst < limit;
    println!("{}", serde_json::to_string_pretty(&json!({"selftest": if ok { "PASS" } else { "FAIL" }, "worst_delta_rel_rmse": worst, "limit": limit, "cases": rows})).unwrap());
    if ok {
        0
    } else {
        1
    }
}

fn bench(args: &[String]) -> i32 {
    let path = arg(args, "--gguf").expect("--gguf FILE");
    let block = arg_usize(args, "--block", 0);
    let g = gguf::Gguf::open(&path).expect("open gguf");
    let raw = weights::single_block_raw(&g, block).expect("block tensors");
    let dims = raw.dims;
    let (input, sd_out, source) = match arg(args, "--dump") {
        Some(dir) => {
            let d = dump::read_block_dump(std::path::Path::new(&dir), &dims).expect("dump");
            (d.input, Some(d.expected_out), format!("dump:{dir}"))
        }
        None => {
            let n = arg_usize(args, "--tokens", 1536);
            (synthetic_input(&dims, n, 42), None, "synthetic".to_string())
        }
    };
    let n = input.n_tokens;
    let base_cfg = config(args);
    let precs = attn_precs(args);
    let pool = make_pool(args);
    let reps = arg_usize(args, "--reps", 5);
    let want_ref = !args.iter().any(|a| a == "--no-ref");
    let want_parity = args.iter().any(|a| a == "--parity");
    let ref_threads = arg_usize(args, "--ref-threads", pool.nthreads());

    // Timing first (as before W5-G: the long reference runs after it), reps interleaved over the
    // arms so a drift in clock or temperature hits every arm alike.
    let mut fasts = Vec::new();
    let mut packs = Vec::new();
    for &prec in &precs {
        let cfg = FastConfig { attn_prec: prec, ..base_cfg };
        let t0 = Instant::now();
        fasts.push(FastBlock::new(dims, raw.linear1_q8, raw.linear2_q8, &raw.q_scale, &raw.k_scale, cfg, &pool));
        packs.push(t0.elapsed().as_secs_f64());
    }
    let mut runs: Vec<Vec<serde_json::Value>> = vec![Vec::new(); precs.len()];
    let mut bests: Vec<Option<StageTimes>> = vec![None; precs.len()];
    let mut lasts: Vec<BlockOutput> = (0..precs.len()).map(|_| BlockOutput::default()).collect();
    for _ in 0..reps {
        for (i, fast) in fasts.iter_mut().enumerate() {
            let (o, t) = fast.forward(&input, false);
            if bests[i].map_or(true, |b| t.total < b.total) {
                bests[i] = Some(t);
            }
            runs[i].push(times_json(&t));
            lasts[i] = o;
        }
    }

    // One reference (and parity reference) for all arms.
    let mut shared = json!({});
    let (want, par) = if want_ref {
        let wref = weights::block_weights_f32(&raw);
        let t0 = Instant::now();
        let want = single_block_ref(&wref, &input, ref_threads);
        shared["reference_seconds"] = json!(t0.elapsed().as_secs_f64());
        let par = if want_parity {
            // W5-G: the reference in sd.cpp arithmetic (Q8_0 activations into linear1/linear2)
            let t0 = Instant::now();
            let par = single_block_ref_arith(&wref, &input, ref_threads, Arith::SdParity);
            shared["parity_reference_seconds"] = json!(t0.elapsed().as_secs_f64());
            shared["parity_vs_ref"] = errors(&par, &want, &input.x);
            Some(par)
        } else {
            None
        };
        (Some(want), par)
    } else {
        (None, None)
    };
    if let (Some(want), Some(sd)) = (&want, &sd_out) {
        let sd_o = BlockOutput { out: sd.clone(), ..Default::default() };
        shared["ref_vs_sdcpp"] = errors(want, &sd_o, &input.x);
        shared["sdcpp_vs_ref"] = errors(&sd_o, want, &input.x);
        if let Some(par) = &par {
            shared["parity_vs_sdcpp"] = errors(par, &sd_o, &input.x);
        }
    }

    let mut arms = Vec::new();
    for (i, fast) in fasts.iter_mut().enumerate() {
        let prec = precs[i];
        let cfg = fast.config();
        let runs = std::mem::take(&mut runs[i]);
        let best = bests[i].expect("reps >= 1");
        let last = &lasts[i];
        let mut totals: Vec<f64> = runs.iter().map(|r| r["total"].as_f64().unwrap()).collect();
        totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = totals[totals.len() / 2];
        let mut att: Vec<f64> = runs.iter().map(|r| r["attention"].as_f64().unwrap()).collect();
        att.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_attn = att[att.len() / 2];

        // MAC accounting (per block forward)
        let lin_macs = n as f64 * (dims.lin1_out() * dims.hidden + dims.hidden * dims.lin2_in()) as f64;
        let attn_macs = 2.0 * (n as f64) * (n as f64) * dims.hidden as f64;
        let tops = |macs: f64, s: f64| 2.0 * macs / s / 1e12;

        let mut out = json!({
            "gguf": path, "block": block, "source": source, "tokens": n,
            "dims": {"hidden": dims.hidden, "heads": dims.heads, "head_dim": dims.head_dim, "mlp": dims.mlp},
            "threads": pool.nthreads(), "cpus": pool.cpus(),
            "config": {"chunk1": cfg.chunk1, "chunk2": cfg.chunk2, "batch_groups": cfg.batch_groups, "lin2_row_block": cfg.lin2_row_block, "key_chunk": cfg.key_chunk, "attn_prec": prec.name()},
            "attn_prec": prec.name(),
            "pack_seconds": packs[i], "packed_weight_bytes": fast.weight_bytes(),
            "runs": runs, "best": times_json(&best), "median_total": median, "median_attention": median_attn,
            "macs": {"linear": lin_macs, "attention": attn_macs},
            "tops_best": {
                "linear1": tops(n as f64 * (dims.lin1_out() * dims.hidden) as f64, best.linear1),
                "linear2": tops(n as f64 * (dims.hidden * dims.lin2_in()) as f64, best.linear2),
                "attention": tops(attn_macs, best.attention),
                "block": tops(lin_macs + attn_macs, best.total),
            },
        });
        if let Some(want) = &want {
            let (got, _) = fast.forward(&input, true);
            // the lean forward (timed above) must give the same output as the tapped one
            out["lean_equals_tapped"] = json!(got.out.iter().zip(&last.out).all(|(a, b)| a.to_bits() == b.to_bits()));
            out["fast_vs_ref"] = errors(&got, want, &input.x);
            if let Some(sd) = &sd_out {
                let sd_o = BlockOutput { out: sd.clone(), ..Default::default() };
                out["fast_vs_sdcpp"] = errors(&got, &sd_o, &input.x);
            }
            if let Some(par) = &par {
                out["fast_vs_parity"] = errors(&got, par, &input.x);
            }
        } else if let Some(sd) = &sd_out {
            let sd_o = BlockOutput { out: sd.clone(), ..Default::default() };
            out["fast_vs_sdcpp"] = errors(last, &sd_o, &input.x);
        }
        for (k, v) in shared.as_object().unwrap() {
            out[k] = v.clone();
        }
        arms.push(out);
    }
    let out = if arms.len() == 1 {
        arms.pop().unwrap()
    } else {
        let summary: Vec<serde_json::Value> = arms
            .iter()
            .map(|a| json!({"attn_prec": a["attn_prec"], "median_total": a["median_total"], "median_attention": a["median_attention"],
                            "fast_vs_parity": a.get("fast_vs_parity"), "fast_vs_sdcpp": a.get("fast_vs_sdcpp"), "fast_vs_ref": a.get("fast_vs_ref")}))
            .collect();
        json!({"summary": summary, "shared": shared, "arms": arms})
    };
    let text = serde_json::to_string_pretty(&out).unwrap();
    println!("{text}");
    if let Some(p) = arg(args, "--json") {
        std::fs::write(p, &text).expect("write --json");
    }
    0
}

fn main() {
    sched::cli::exit_zero_on_epipe(); // G-EPIPE-OXI (F-CI9): `| head` ends with status 0, not a core dump
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(|s| s.as_str()) {
        Some("selftest") => selftest(&args[1..]),
        Some("bench") => bench(&args[1..]),
        _ => {
            eprintln!("usage: oxi-block selftest [--threads N] | oxi-block bench --gguf FILE (--dump DIR | --tokens N) [options]; see the source header");
            2
        }
    };
    std::process::exit(code);
}
