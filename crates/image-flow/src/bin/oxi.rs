//! `oxi`: FLUX.2 klein image generation from a sd.cpp hand-over, with checks against its dumps.
//!
//! ```text
//! oxi generate --gguf F --vae VAE.safetensors --handoff DIR --out OUT.png [--backend ref|fast] [--vae-conv f32|winograd|int8]
//!      [--threads N] [--grid HxW] [--latent-out FILE] [--json FILE] [--chunk1 N --chunk2 N --key-chunk N]
//!      [--attn-prec int8|f16|f32]   (fast backend: attention arithmetic, linears stay int8; F-IMG-ATTN)
//! oxi verify   --gguf F --handoff DIR [--backend ref|fast] [--arith f32|sdcpp] [--threads N] [--json FILE]
//! oxi drift-probe --gguf F --handoff DIR [--step 0] [--scope component|blocks] [--arith f32,sdcpp]
//!      [--perturb-ulp N] [--max-single N] [--threads N] [--json FILE]   (W5-G: components on sd.cpp's own dumped inputs)
//! oxi compare  A.png B.png
//! oxi cond-put --handoff DIR --prompt-file F --model GGUF --width W --height H [--cache DIR]
//!      [--sigmas stored|computed --steps 4]   (computed: call-0-only hand-over, IMG-SHIP)
//! oxi generate --gguf F --vae VAE --cond-prompt-file F --width W --height H --seed N [--steps 4]
//!      [--cache DIR] [--sigmas stored|computed] [--noise own|sdcpp] --out OUT.png   (no sd.cpp run needed)
//! ```
//! Exit code 3 from `generate --cond-prompt-file`: no cached conditioning for the inputs.
//! Output summaries are JSON on stdout (and in `--json FILE` when given).

use image_dit::arith::Arith;
use image_dit::attn_float::AttnPrec;
use image_dit::fast::FastConfig;
use image_dit::fast_model::{FastModel, ModelTimes};
use image_dit::model::{DitBackend, ModelInput};
use image_dit::types::rel_rmse;
use image_flow::cond_cache::{self, Conditioning, KeyInputs};
use image_flow::handoff::Handoff;
use image_flow::latent;
use image_flow::noise;
use image_flow::sampler::{fit_step, run_flow, run_flow_inputs, FlowInputs};
use image_flow::schedule;
use image_flow::{RefModel, Taps};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn main() {
    if let Err(e) = real_main() {
        eprintln!("oxi: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match argv.split_first() {
        Some((c, r)) => (c.as_str(), r),
        None => return Err(usage()),
    };
    let args = Args::parse(rest)?;
    match cmd {
        "generate" => cmd_generate_any(&args),
        "cond-put" => cmd_cond_put(&args),
        "verify" => cmd_verify(&args),
        "drift-probe" => cmd_drift_probe(&args),
        "act-stats" => cmd_act_stats(&args),
        "txt-stats" => cmd_txt_stats(&args),
        "vae-decode" => cmd_vae_decode(&args),
        "compare" => cmd_compare(&args),
        _ => Err(usage()),
    }
}

fn usage() -> String {
    "usage: oxi generate --gguf F --vae VAE --handoff DIR --out OUT.png [--backend ref|fast] [--threads N] \
     [--grid HxW] [--latent-out FILE] [--json FILE]\n       oxi verify --gguf F --handoff DIR [--backend ref|fast] \
     [--threads N] [--json FILE]\n       oxi vae-decode --vae VAE --handoff DIR --out OUT.png [--vae-conv f32|winograd|int8]\n       oxi compare A.png B.png"
        .to_string()
}

// ---------------------------------------------------------------------------------------------
// arguments
// ---------------------------------------------------------------------------------------------

struct Args {
    kv: HashMap<String, String>,
    pos: Vec<String>,
}

impl Args {
    fn parse(rest: &[String]) -> Result<Args, String> {
        let mut kv = HashMap::new();
        let mut pos = Vec::new();
        let mut i = 0;
        while i < rest.len() {
            if let Some(key) = rest[i].strip_prefix("--") {
                let v = rest.get(i + 1).ok_or_else(|| format!("--{key} needs a value"))?;
                kv.insert(key.to_string(), v.clone());
                i += 2;
            } else {
                pos.push(rest[i].clone());
                i += 1;
            }
        }
        Ok(Args { kv, pos })
    }

    fn req(&self, key: &str) -> Result<&str, String> {
        self.kv.get(key).map(|s| s.as_str()).ok_or_else(|| format!("missing --{key}"))
    }

    fn opt(&self, key: &str) -> Option<&str> {
        self.kv.get(key).map(|s| s.as_str())
    }

    fn usize_or(&self, key: &str, default: usize) -> Result<usize, String> {
        match self.opt(key) {
            None => Ok(default),
            Some(s) => s.parse::<usize>().map_err(|_| format!("--{key}: '{s}' is not a number")),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// backends
// ---------------------------------------------------------------------------------------------

/// The chosen DiT, with per-forward timings kept for the report.
enum Backend<'p> {
    Ref(RefModel),
    Fast(FastModel<'p>),
}

struct Timed<'p> {
    inner: Backend<'p>,
    times: Vec<ModelTimes>,
}

impl DitBackend for Timed<'_> {
    fn name(&self) -> &'static str {
        match &self.inner {
            Backend::Ref(m) => m.name(),
            Backend::Fast(m) => m.name(),
        }
    }

    fn forward(&mut self, input: &ModelInput) -> Vec<f32> {
        match &mut self.inner {
            Backend::Ref(m) => m.forward(input),
            Backend::Fast(m) => {
                let v = m.forward(input);
                let t = m.last_times;
                eprintln!(
                    "forward: total {:.3}s (embed {:.3} double {:.3} single {:.3} final {:.3} attention {:.3}) linears {:.3} (unit {:.3} plain {:.3}) quant {:.3}",
                    t.total,
                    t.embed,
                    t.double_blocks,
                    t.single_blocks,
                    t.final_layer,
                    t.attention,
                    t.units + t.plain,
                    t.units,
                    t.plain,
                    t.quant
                );
                self.times.push(t);
                v
            }
        }
    }
}

/// Builds the backend. `pool` must outlive the returned value when `fast` is chosen.
/// `--arith` of the reference backend: f32 (default) or sdcpp (W5-G arithmetic parity).
fn arith_of(s: Option<&str>) -> Result<Arith, String> {
    s.map_or(Ok(Arith::F32), Arith::parse)
}

fn build_backend<'p>(
    gguf: &Path,
    kind: &str,
    threads: usize,
    cfg: FastConfig,
    pool: &'p Option<sched::Pool>,
    arith: Option<&str>,
) -> Result<Timed<'p>, String> {
    if kind != "ref" && arith.is_some() {
        return Err("--arith applies to --backend ref only".into());
    }
    if kind != "fast" && cfg.attn_prec != AttnPrec::Int8 {
        return Err("--attn-prec applies to --backend fast only".into());
    }
    let inner = match kind {
        "ref" => {
            let mut m = RefModel::open(gguf, threads)?;
            m.set_arith(arith_of(arith)?);
            // W5-G self-noise control: OXI_PERTURB_ULP=N moves the token embeddings by up to N ulps
            if let Some(u) = std::env::var("OXI_PERTURB_ULP").ok().and_then(|v| v.parse::<u32>().ok()) {
                eprintln!("oxi: OXI_PERTURB_ULP={u}: token embeddings perturbed (self-noise control)");
                m.set_perturb_ulp(u);
            }
            Backend::Ref(m)
        }
        "fast" => {
            let pool = pool.as_ref().ok_or("fast backend needs a pool")?;
            Backend::Fast(FastModel::open(gguf, pool, cfg)?)
        }
        other => return Err(format!("--backend must be ref or fast, got '{other}'")),
    };
    Ok(Timed { inner, times: Vec::new() })
}

fn make_pool(kind: &str, threads: usize) -> Result<Option<sched::Pool>, String> {
    if kind != "fast" {
        return Ok(None);
    }
    let cpus: Vec<usize> = (0..threads).collect();
    // Idle workers spin this long and then sleep on a futex. Between the parallel stages the main
    // thread does serial work (embedders, modulation, final layer, buffer set-up); with a long spin
    // 11 cores burn power doing nothing there, which costs sustained speed on this machine (it
    // throttles at about 97 C). 200 us covers the gaps between consecutive stages of a block; a
    // wake-up costs tens of microseconds. OXI_IDLE_SPIN_US overrides it (results never depend on it).
    let spin_us = std::env::var("OXI_IDLE_SPIN_US").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(200);
    sched::Pool::new(&cpus, Duration::from_micros(spin_us)).map(Some).map_err(|e| format!("pool: {e}"))
}

/// One VAE decode with its timing split (file load vs decode) and the CPU mask it ran under.
struct VaeReport {
    rgb: image_vae::Rgb,
    conv: String,
    load_s: f64,
    decode_s: f64,
    allowed_cpus: usize,
    threads: usize,
}

impl VaeReport {
    fn json(&self) -> Value {
        json!({"conv": self.conv, "load_seconds": self.load_s, "decode_seconds": self.decode_s,
               "seconds_incl_load": self.load_s + self.decode_s, "allowed_cpus": self.allowed_cpus,
               "threads": self.threads})
    }
}

fn decode_vae(vae: &Path, z: &[f32], gh: usize, gw: usize, threads: usize, conv: &str) -> Result<VaeReport, String> {
    image_vae::decoder::set_conv_mode(conv)?;
    let allowed = sched::allowed_cpus();
    if allowed != 0 && allowed < threads {
        eprintln!("oxi: WARNING: the VAE may use only {allowed} cpu(s) for {threads} threads (inherited CPU mask)");
    }
    let t = Instant::now();
    let dec = image_vae::load_decoder(vae)?;
    let load_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let rgb = image_vae::decode_with(&dec, z, latent::LATENT_CHANNELS, gh, gw, threads)?;
    let decode_s = t.elapsed().as_secs_f64();
    eprintln!(
        "oxi: vae decode ({conv} conv) {:.3} s (includes loading the VAE file: load {load_s:.3} s, decode {decode_s:.3} s, {allowed} cpus allowed)",
        load_s + decode_s
    );
    Ok(VaeReport { rgb, conv: conv.to_string(), load_s, decode_s, allowed_cpus: allowed, threads })
}

fn fast_config(a: &Args) -> Result<FastConfig, String> {
    let mut cfg = FastConfig::default();
    cfg.chunk1 = a.usize_or("chunk1", cfg.chunk1)?;
    cfg.chunk2 = a.usize_or("chunk2", cfg.chunk2)?;
    cfg.key_chunk = a.usize_or("key-chunk", cfg.key_chunk)?;
    if let Some(p) = a.opt("attn-prec") {
        cfg.attn_prec = AttnPrec::parse(p)?;
    }
    Ok(cfg)
}

/// Latent grid `(h, w)`: from `dump.vae.z` when present, else `--grid HxW`.
fn latent_grid(h: &Handoff, a: &Args) -> Result<(usize, usize), String> {
    let grid = match h.vae_grid() {
        Some(g) => g,
        None => {
            let s = a
                .opt("grid")
                .ok_or("the hand-over has no dump.vae.z: pass --grid HxW for the latent grid")?;
            let (hh, ww) = s.split_once('x').ok_or("--grid must look like 64x48")?;
            (
                hh.parse::<usize>().map_err(|_| "--grid height")?,
                ww.parse::<usize>().map_err(|_| "--grid width")?,
            )
        }
    };
    if grid.0 * grid.1 != h.n_img {
        return Err(format!("latent grid {}x{} has {} tokens, hand-over has {}", grid.0, grid.1, grid.0 * grid.1, h.n_img));
    }
    Ok(grid)
}

fn write_json(a: &Args, v: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    println!("{text}");
    if let Some(p) = a.opt("json") {
        std::fs::write(p, text.as_bytes()).map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------------------------

fn cmd_generate(a: &Args) -> Result<(), String> {
    let gguf = PathBuf::from(a.req("gguf")?);
    let vae = PathBuf::from(a.req("vae")?);
    let dir = PathBuf::from(a.req("handoff")?);
    let out = PathBuf::from(a.req("out")?);
    let kind = a.opt("backend").unwrap_or("ref");
    let threads = a.usize_or("threads", 12)?;
    let cfg = fast_config(a)?;

    let h = Handoff::load(&dir)?;
    let (gh, gw) = latent_grid(&h, a)?;
    let pool = make_pool(kind, threads)?;
    let mut be = build_backend(&gguf, kind, threads, cfg, &pool, a.opt("arith"))?;

    let mut step_secs = Vec::with_capacity(h.steps);
    let mut last = Instant::now();
    let t0 = Instant::now();
    let x = run_flow(&mut be, &h, &mut |_, _, _| {
        let now = Instant::now();
        step_secs.push((now - last).as_secs_f64());
        last = now;
    });
    let total = t0.elapsed().as_secs_f64();

    let nchw = latent::tokens_to_nchw(&x, latent::LATENT_CHANNELS, gh, gw)?;
    if let Some(p) = a.opt("latent-out") {
        let bytes: Vec<u8> = nchw.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes).map_err(|e| format!("{p}: {e}"))?;
    }
    let z = latent::diffusion_to_vae_latents(&nchw, latent::LATENT_CHANNELS, gh, gw)?;
    // Free the DiT and its pinned pool before the VAE: Pool::new pinned this thread to cpu 0 and
    // the VAE's worker threads inherit this thread's mask (IMG-VAE: 12 VAE threads on one core,
    // 21.9 s instead of ~3 s). Dropping the pool restores the mask; it also frees the DiT weights.
    let backend_name = be.name();
    let model_times = std::mem::take(&mut be.times);
    drop(be);
    drop(pool);
    let vae_conv = a.opt("vae-conv").unwrap_or("f32");
    let vae_rep = decode_vae(&vae, &z, gh, gw, threads, vae_conv)?;
    let rgb = &vae_rep.rgb;
    image_vae::png::write_png(&out, rgb)?;

    let report = json!({
        "backend": backend_name,
        "attn_prec": cfg.attn_prec.name(),
        "arith": a.opt("arith").unwrap_or("f32"),
        "threads": threads,
        "noise": "sdcpp",
        "cond_key": null,
        "sigmas_source": "dump",
        "steps": h.steps,
        "n_img": h.n_img,
        "n_txt": h.n_txt,
        "grid": [gh, gw],
        "sigmas": h.sigmas,
        "step_seconds": step_secs,
        "total_seconds": total,
        "image": { "width": rgb.width, "height": rgb.height },
        "out": out.display().to_string(),
        "vae": vae_rep.json(),
        "model_times": model_times.iter().map(|t| json!({
            "embed": t.embed, "double_blocks": t.double_blocks, "single_blocks": t.single_blocks,
            "final_layer": t.final_layer, "attention": t.attention, "total": t.total,
        })).collect::<Vec<_>>(),
    });
    write_json(a, &report)
}

// ---------------------------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------------------------

fn input_for(h: &Handoff, img: &[f32], sigma: f32) -> ModelInput {
    ModelInput {
        n_img: h.n_img,
        n_txt: h.n_txt,
        img: img.to_vec(),
        txt: h.txt.clone(),
        timestep: sigma,
        pe: h.pe.clone(),
    }
}

fn cmd_drift_probe(a: &Args) -> Result<(), String> {
    let gguf = PathBuf::from(a.req("gguf")?);
    let dir = PathBuf::from(a.req("handoff")?);
    let threads = a.usize_or("threads", 8)?;
    let step = a.usize_or("step", 0)?;
    let scope = image_flow::drift::Scope::parse(a.opt("scope").unwrap_or("component"))?;
    let ariths = a
        .opt("arith")
        .unwrap_or("f32,sdcpp")
        .split(',')
        .map(Arith::parse)
        .collect::<Result<Vec<_>, _>>()?;
    let h = Handoff::load(&dir)?;
    if step >= h.steps {
        return Err(format!("--step {step} exceeds the {}-step handoff", h.steps));
    }
    let t0 = Instant::now();
    let mut m = RefModel::open(&gguf, threads)?;
    let types = m.embed_types();
    let bo = image_flow::drift::BlocksOpts {
        perturb_ulp: a.usize_or("perturb-ulp", 0)? as u32,
        max_single: a.opt("max-single").map(|v| v.parse::<usize>().map_err(|_| "--max-single: not a number")).transpose()?,
    };
    let mut report = image_flow::drift::run_with(&mut m, &h, &dir, step, scope, &ariths, bo)?;
    report["perturb_ulp"] = json!(bo.perturb_ulp);
    report["threads"] = json!(threads);
    report["embed_types"] = json!(format!("{types:?}"));
    report["total_seconds"] = json!(t0.elapsed().as_secs_f64());
    write_json(a, &report)
}

fn cmd_verify(a: &Args) -> Result<(), String> {
    let gguf = PathBuf::from(a.req("gguf")?);
    let dir = PathBuf::from(a.req("handoff")?);
    let kind = a.opt("backend").unwrap_or("ref");
    let threads = a.usize_or("threads", 12)?;
    let cfg = fast_config(a)?;

    let h = Handoff::load(&dir)?;
    let pool = make_pool(kind, threads)?;
    let mut be = build_backend(&gguf, kind, threads, cfg, &pool, a.opt("arith"))?;
    let steps = h.steps;
    if a.opt("teacher-step").is_some() {
        let i = a.usize_or("teacher-step", 0)?;
        if i >= steps {
            return Err(format!("teacher step {i} exceeds the {steps}-step handoff"));
        }
        let reference = h.out_at(i).ok_or("teacher step is missing dump.flux.out")?;
        let (sigma, _) = h.sigma_pair(i);
        let input = input_for(&h, h.img_at(i), sigma);
        let start = Instant::now();
        let mut taps = Vec::new();
        let got = if let Backend::Ref(model) = &mut be.inner {
            let mut captured = Taps::recording();
            let output = model.forward_with_taps(&input, &mut captured);
            for (block, actual) in [(0, captured.single0_in), (1, captured.single1_in)] {
                if let (Some(actual), Some(expected)) = (actual, h.single_in_at(block, i)) {
                    taps.push(json!({"block": block, "rel_rmse": rel_rmse(&actual, expected)}));
                }
            }
            output
        } else {
            be.forward(&input)
        };
        // W5-G: raw model output (f32 LE, [n_img][out_ch]) for self-noise comparisons between runs
        if let Some(p) = a.opt("out-raw") {
            let bytes: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(p, bytes).map_err(|e| format!("{p}: {e}"))?;
        }
        let report = json!({"backend": kind, "arith": a.opt("arith").unwrap_or("f32"), "threads": threads, "step": i,
            "forward_s": start.elapsed().as_secs_f64(), "rel_rmse": rel_rmse(&got, reference),
            "single_block_input_rmse": taps});
        let text = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
        if let Some(path) = a.opt("json") {
            std::fs::write(path, &text).map_err(|error| error.to_string())?;
        }
        println!("{text}");
        return Ok(());
    }
    let mut checks: Vec<Value> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    // 0. the schedule port against the dumped sigmas (image_seq_len = n_img for a hand-over)
    let computed = schedule::flux2_sigmas(steps as u32, h.n_img);
    let max_abs = h
        .sigmas
        .iter()
        .zip(&computed)
        .map(|(d, c)| (*d as f64 - *c as f64).abs())
        .fold(0f64, f64::max);
    checks.push(json!({
        "check": "flux2_sigmas_vs_dump",
        "max_abs_diff": max_abs,
        "dumped": h.sigmas,
        "computed": &computed[..steps],
    }));

    // 1. free-running loop from the hand-over's noise
    let mut free_before: Vec<Vec<f32>> = Vec::new();
    let free_final = run_flow(&mut be, &h, &mut |_, x, _| free_before.push(x.to_vec()));

    // 2. teacher-forced: each step's input is the dumped one
    let mut tf_out: Vec<Vec<f32>> = Vec::with_capacity(steps);
    for i in 0..steps {
        let (sigma, _) = h.sigma_pair(i);
        tf_out.push(be.forward(&input_for(&h, h.img_at(i), sigma)));
    }
    for i in 0..steps {
        let drift = rel_rmse(&free_before[i], h.img_at(i));
        let mut c = json!({"check": "free_running_x_vs_dump_img", "step": i, "rel_rmse": drift});
        c["note"] = json!("free-running input compared with the dumped input (drift)");
        checks.push(c);
        if let Some(o) = h.out_at(i) {
            checks.push(json!({"check": "model_out_vs_dump_out", "step": i, "rel_rmse": rel_rmse(&tf_out[i], o)}));
            checks.push(json!({"check": "fit_c_dump_out", "step": i,
                "c": fit_step(h.img_at(i), o, next_x(&h, i, &free_final)),
                "sigma_diff": sigma_diff(&h, i)}));
        } else {
            notes.push(format!("step {i}: no dump.flux.out, output comparison skipped"));
        }
        checks.push(json!({"check": "fit_c_model_out", "step": i,
            "c": fit_step(h.img_at(i), &tf_out[i], next_x(&h, i, &free_final)),
            "sigma_diff": sigma_diff(&h, i),
            "next_x": if i + 1 < steps { "dump img" } else { "free-running final x" }}));
    }

    // 3. single-block taps (reference backend only; the fast path does not expose them)
    if let Backend::Ref(m) = &mut be.inner {
        let mut taps = Taps::recording();
        let _ = m.forward_with_taps(&input_for(&h, h.img_at(0), h.sigma_pair(0).0), &mut taps);
        for (b, got) in [(0usize, taps.single0_in), (1, taps.single1_in)] {
            match (h.single_in_at(b, 0), got) {
                (Some(want), Some(got)) if !got.is_empty() => {
                    checks.push(json!({"check": format!("single_{b}_in_vs_dump"), "step": 0, "rel_rmse": rel_rmse(&got, want)}))
                }
                (None, _) => notes.push(format!("dump.single.{b}.in absent: tap check skipped")),
                _ => notes.push(format!("single block {b} not reached: tap check skipped")),
            }
        }
    } else {
        notes.push("taps: fast backend has no single-block taps, skipped".into());
    }

    // 4. free-running final vs the last step's implied result (dump img + dt * dump out)
    if let Some(o) = h.out_at(steps - 1) {
        let dt = h.sigma_pair(steps - 1).1 - h.sigma_pair(steps - 1).0;
        let implied: Vec<f32> = h.img_at(steps - 1).iter().zip(o).map(|(x, v)| x + dt * v).collect();
        checks.push(json!({"check": "free_final_vs_implied_last_step", "rel_rmse": rel_rmse(&free_final, &implied)}));

        // 5. VAE latent: implied final tokens -> NCHW -> diffusion_to_vae vs dump.vae.z
        if let (Some(z), Some((gh, gw))) = (&h.vae_z, h.vae_grid()) {
            let nchw = latent::tokens_to_nchw(&implied, latent::LATENT_CHANNELS, gh, gw)?;
            let vae_z = latent::diffusion_to_vae_latents(&nchw, latent::LATENT_CHANNELS, gh, gw)?;
            checks.push(json!({"check": "vae_z_vs_dump_from_implied_final", "rel_rmse": rel_rmse(&vae_z, &z.data)}));
        }
    } else {
        notes.push("no dump.flux.out for the last step: implied final and VAE checks skipped".into());
    }
    if let (Some(z), Some((gh, gw))) = (&h.vae_z, h.vae_grid()) {
        let nchw = latent::tokens_to_nchw(&free_final, latent::LATENT_CHANNELS, gh, gw)?;
        let vae_z = latent::diffusion_to_vae_latents(&nchw, latent::LATENT_CHANNELS, gh, gw)?;
        checks.push(json!({"check": "vae_z_vs_dump_from_free_final", "rel_rmse": rel_rmse(&vae_z, &z.data)}));
    } else if h.vae_z.is_none() {
        notes.push("dump.vae.z absent: VAE latent check skipped".into());
    }

    let report = json!({
        "backend": be.name(),
        "attn_prec": cfg.attn_prec.name(),
        "threads": threads,
        "steps": steps,
        "n_img": h.n_img,
        "n_txt": h.n_txt,
        "checks": checks,
        "notes": notes,
    });
    write_json(a, &report)
}

/// `generate` with `--cond-prompt-file` runs from the conditioning cache and an own seed;
/// otherwise it runs from a sd.cpp hand-over (`--handoff`).
fn cmd_generate_any(a: &Args) -> Result<(), String> {
    if a.opt("cond-prompt-file").is_some() {
        cmd_generate_cond(a)
    } else {
        cmd_generate(a)
    }
}

/// Inputs of a cache key, read from the command line. The prompt bytes are read here and never printed.
struct CondArgs {
    prompt: Vec<u8>,
    model_name: String,
    model_size: u64,
    gguf: PathBuf,
    width: usize,
    height: usize,
    steps: usize,
    root: PathBuf,
}

fn cond_args(a: &Args, prompt_flag: &str) -> Result<CondArgs, String> {
    let prompt_path = PathBuf::from(a.req(prompt_flag)?);
    let prompt = std::fs::read(&prompt_path).map_err(|e| format!("--{prompt_flag}: {e}"))?;
    let gguf = PathBuf::from(a.req("gguf")?);
    let meta = std::fs::metadata(&gguf).map_err(|e| format!("{}: {e}", gguf.display()))?;
    let model_name = gguf
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or("--gguf has no file name")?;
    let width = a.usize_or("width", 0)?;
    let height = a.usize_or("height", 0)?;
    if width == 0 || height == 0 {
        return Err("--width and --height are required".into());
    }
    let root = a.opt("cache").map(PathBuf::from).unwrap_or_else(cond_cache::default_root);
    Ok(CondArgs {
        prompt,
        model_name,
        model_size: meta.len(),
        gguf,
        width,
        height,
        steps: a.usize_or("steps", 4)?,
        root,
    })
}

fn cmd_cond_put(a: &Args) -> Result<(), String> {
    let mut ca = cond_args(a, "prompt-file")?;
    let dir = PathBuf::from(a.req("handoff")?);
    let h = Handoff::load(&dir)?;
    // --sigmas stored (default): the entry describes the hand-over's own steps.
    // --sigmas computed: a partial hand-over (sd-cli stopped after model call 0, IMG-SHIP); the entry
    // gets --steps (default 4) computed sigmas, checked against every dumped one.
    let partial = match a.opt("sigmas").unwrap_or("stored") {
        "stored" => {
            ca.steps = h.steps;
            false
        }
        "computed" => true,
        other => return Err(format!("--sigmas must be stored or computed, got '{other}'")),
    };
    let kin = KeyInputs {
        prompt: &ca.prompt,
        model_name: &ca.model_name,
        model_size: ca.model_size,
        width: ca.width,
        height: ca.height,
        steps: ca.steps,
    };
    let entry = if partial {
        cond_cache::put_from_partial_handoff(&ca.root, &kin, &h)?
    } else {
        cond_cache::put_from_handoff(&ca.root, &kin, &h)?
    };
    let key = cond_cache::cache_key(&kin);
    write_json(
        a,
        &json!({
            "key": key,
            "entry": entry.display().to_string(),
            "width": ca.width,
            "height": ca.height,
            "steps": ca.steps,
            "n_txt": h.n_txt,
            "n_img": h.n_img,
        }),
    )
}

fn cmd_generate_cond(a: &Args) -> Result<(), String> {
    let ca = cond_args(a, "cond-prompt-file")?;
    latent::grid_for(ca.width, ca.height)?;
    let seed: u64 = a.req("seed")?.parse().map_err(|_| "--seed must be an integer".to_string())?;
    let kin = KeyInputs {
        prompt: &ca.prompt,
        model_name: &ca.model_name,
        model_size: ca.model_size,
        width: ca.width,
        height: ca.height,
        steps: ca.steps,
    };
    let key = cond_cache::cache_key(&kin);
    let cond: Conditioning = match cond_cache::get(&ca.root, &kin)? {
        Some(c) => c,
        None => {
            eprintln!(
                "oxi: no cached conditioning for key {key} (this prompt, model, size and steps). \
                 The sd.cpp hand-over step is needed once: run sd.cpp on this prompt with the same model, \
                 size and steps, then run `oxi cond-put --handoff DIR --prompt-file F --model GGUF \
                 --width W --height H`."
            );
            std::process::exit(3);
        }
    };
    let kind = a.opt("backend").unwrap_or("ref");
    let threads = a.usize_or("threads", 12)?;
    let cfg = fast_config(a)?;
    let vae = PathBuf::from(a.req("vae")?);
    let out = PathBuf::from(a.req("out")?);

    // initial noise from the seed: ours (default, `own`) or sd-cli's Philox draw (`sdcpp`: the same
    // seed gives sd-cli's latent, so the picture matches sd.cpp's for that seed within the image gate)
    let steps = cond.sigmas.len();
    let noise_kind = a.opt("noise").unwrap_or("own");
    let img0 = match noise_kind {
        "own" => noise::initial_tokens(seed, cond.n_img, latent::LATENT_CHANNELS),
        "sdcpp" => noise::sdcpp_initial_tokens(seed, cond.grid.0, cond.grid.1, latent::LATENT_CHANNELS),
        other => return Err(format!("--noise must be own or sdcpp, got '{other}'")),
    };
    let (sigmas, sigma_end, sigmas_source) = match a.opt("sigmas").unwrap_or("stored") {
        "stored" => (cond.sigmas.clone(), 0.0f32, "stored"),
        "computed" => {
            let full = schedule::flux2_sigmas(steps as u32, cond.n_img);
            (full[..steps].to_vec(), full[steps], "computed")
        }
        other => return Err(format!("--sigmas must be stored or computed, got '{other}'")),
    };
    let mut inp = FlowInputs::from_conditioning(&cond, img0, sigmas)?;
    inp.sigma_end = sigma_end;

    let pool = make_pool(kind, threads)?;
    let mut be = build_backend(&ca.gguf, kind, threads, cfg, &pool, a.opt("arith"))?;
    let (gh, gw) = cond.grid;
    let mut step_secs = Vec::with_capacity(steps);
    let mut last = Instant::now();
    let t0 = Instant::now();
    let x = run_flow_inputs(&mut be, &inp, &mut |_, _, _| {
        let now = Instant::now();
        step_secs.push((now - last).as_secs_f64());
        last = now;
    });
    let total = t0.elapsed().as_secs_f64();

    let nchw = latent::tokens_to_nchw(&x, latent::LATENT_CHANNELS, gh, gw)?;
    if let Some(p) = a.opt("latent-out") {
        let bytes: Vec<u8> = nchw.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes).map_err(|e| format!("{p}: {e}"))?;
    }
    let z = latent::diffusion_to_vae_latents(&nchw, latent::LATENT_CHANNELS, gh, gw)?;
    // Free the DiT and its pinned pool before the VAE: Pool::new pinned this thread to cpu 0 and
    // the VAE's worker threads inherit this thread's mask (IMG-VAE: 12 VAE threads on one core,
    // 21.9 s instead of ~3 s). Dropping the pool restores the mask; it also frees the DiT weights.
    let backend_name = be.name();
    let model_times = std::mem::take(&mut be.times);
    drop(be);
    drop(pool);
    let vae_conv = a.opt("vae-conv").unwrap_or("f32");
    let vae_rep = decode_vae(&vae, &z, gh, gw, threads, vae_conv)?;
    let rgb = &vae_rep.rgb;
    image_vae::png::write_png(&out, rgb)?;

    let report = json!({
        "backend": backend_name,
        "attn_prec": cfg.attn_prec.name(),
        "threads": threads,
        "noise": noise_kind,
        "seed": seed,
        "cond_key": key,
        "sigmas_source": sigmas_source,
        "steps": steps,
        "n_img": cond.n_img,
        "n_txt": cond.n_txt,
        "grid": [gh, gw],
        "sigmas": inp.sigmas,
        "step_seconds": step_secs,
        "total_seconds": total,
        "image": { "width": rgb.width, "height": rgb.height },
        "out": out.display().to_string(),
        "vae": vae_rep.json(),
        "model_times": model_times.iter().map(|t| json!({
            "embed": t.embed, "double_blocks": t.double_blocks, "single_blocks": t.single_blocks,
            "final_layer": t.final_layer, "attention": t.attention, "total": t.total,
        })).collect::<Vec<_>>(),
    });
    write_json(a, &report)
}

fn sigma_diff(h: &Handoff, i: usize) -> f64 {
    let (s, n) = h.sigma_pair(i);
    n as f64 - s as f64
}

/// The x after step `i`: the dumped next input, or the free-running final for the last step.
fn next_x<'a>(h: &'a Handoff, i: usize, free_final: &'a [f32]) -> &'a [f32] {
    if i + 1 < h.steps {
        h.img_at(i + 1)
    } else {
        free_final
    }
}

// ---------------------------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------------------------

/// `oxi act-stats --handoff DIR [--json FILE]`: int8 quantisation error of the modulated input of
/// single block 0 at every dumped step, for several scale chunks and channel arrangements (see
/// `image_flow::actstats`). Reads only the hand-over files; no model is loaded.
fn cmd_act_stats(a: &Args) -> Result<(), String> {
    use image_flow::actstats::{channel_rms, magnitude_spread, modulated, quant_error, Arrange};
    use image_flow::handoff::read_tag_call;
    let dir = Path::new(a.req("handoff")?);
    let mut steps = Vec::new();
    for call in 0..16 {
        let (x, shift, scale) = match (read_tag_call(dir, "dump.single.0.in", call)?, read_tag_call(dir, "dump.single.mod.shift", call)?, read_tag_call(dir, "dump.single.mod.scale", call)?) {
            (Some(x), Some(sh), Some(sc)) => (x, sh, sc),
            _ => break,
        };
        let (h, n) = (x.ne[0], x.ne[1]);
        if shift.data.len() != h || scale.data.len() != h || x.data.len() != n * h {
            return Err(format!("call {call}: unexpected dump shapes"));
        }
        let xm = modulated(&x.data, n, h, &shift.data, &scale.data);
        let mag = channel_rms(&xm, n, h);
        let (ratio, top8) = magnitude_spread(&mag);
        let mut rows = Vec::new();
        for chunk in [32usize, 64, 128, 256, 512, 1024, h] {
            if h % chunk != 0 {
                continue;
            }
            let mut row = json!({"chunk": chunk});
            for (name, arr) in [("natural", Arrange::Natural), ("sorted", Arrange::Sorted), ("equalized", Arrange::Equalized)] {
                let (plain, balanced) = quant_error(&xm, n, h, &mag, chunk, arr);
                row[name] = json!({"plain": plain, "balanced": balanced});
            }
            rows.push(row);
        }
        steps.push(json!({"call": call, "tokens": n, "hidden": h, "channel_rms_max_over_median": ratio, "energy_share_top8_channels": top8, "quant_rel_error": rows}));
    }
    if steps.is_empty() {
        return Err("no dump.single.0.in with its modulation dumps in this hand-over".into());
    }
    let text = serde_json::to_string_pretty(&json!({"what": "int8 quantisation of the modulated single-block-0 input, one max-based scale per token per chunk", "steps": steps})).map_err(|e| e.to_string())?;
    println!("{text}");
    if let Some(p) = a.opt("json") {
        std::fs::write(p, &text).map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}

/// `oxi txt-stats --handoff DIR [--json FILE]`: how redundant the dumped text context is (see
/// `image_flow::txtstats`). Reads only the hand-over files; no model is loaded.
fn cmd_txt_stats(a: &Args) -> Result<(), String> {
    let h = Handoff::load(Path::new(a.req("handoff")?))?;
    if h.n_txt == 0 {
        return Err("the hand-over has no text tokens".into());
    }
    let dim = h.txt.len() / h.n_txt;
    let pe_row = h.pe.len() / (h.n_txt + h.n_img);
    let tols = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1];
    let est = image_flow::txtstats::estimate(&h.txt, h.n_txt, dim, h.n_img, &tols);
    let norms: Vec<f64> = (0..h.n_txt).map(|t| h.txt[t * dim..(t + 1) * dim].iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt()).collect();
    // relative distance of each token to its predecessor: shows where the prompt ends and the padding begins
    let step: Vec<f64> = (1..h.n_txt)
        .map(|t| {
            let d: f64 = h.txt[t * dim..(t + 1) * dim].iter().zip(&h.txt[(t - 1) * dim..t * dim]).map(|(x, y)| ((*x - *y) as f64).powi(2)).sum::<f64>().sqrt();
            if norms[t] > 0.0 { d / norms[t] } else { 0.0 }
        })
        .collect();
    // step-to-step change of the text rows (and, for scale, the image rows) entering single block 0:
    // how much a "cache the text stream across steps" approximation would be off, on real data
    let hidden_rows = |x: &[f32]| x.len() / (h.n_txt + h.n_img);
    let mut step_change = Vec::new();
    for i in 1..h.steps {
        if let (Some(p), Some(q)) = (h.single_in_at(0, i - 1), h.single_in_at(0, i)) {
            let hd = hidden_rows(p);
            if hd > 0 && p.len() == q.len() {
                let cut = h.n_txt * hd;
                step_change.push(json!({"steps": [i - 1, i], "txt_rows_rel_change": rel_rmse(&q[..cut], &p[..cut]), "img_rows_rel_change": rel_rmse(&q[cut..], &p[cut..])}));
            }
        }
    }
    let mut sorted = step.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let step_median = if sorted.is_empty() { 0.0 } else { sorted[sorted.len() / 2] };
    let report = json!({
        "n_txt": h.n_txt, "n_img": h.n_img, "ctx_dim": dim,
        "txt_pe_max_abs_spread": image_flow::txtstats::txt_pe_spread(&h.pe, h.n_txt, pe_row),
        "zero_norm_tokens": norms.iter().filter(|n| **n == 0.0).count(),
        "merge": est.iter().map(|e| json!({"tol": e.tol, "n_eff": e.n_eff, "longest_run": e.longest_run,
            "linear_macs_ratio": e.linear_ratio, "attention_macs_ratio": e.attention_ratio})).collect::<Vec<_>>(),
        "rel_step_first_64": &step[..step.len().min(64)],
        "rel_step_last_16": &step[step.len().saturating_sub(16)..],
        "rel_step_median": step_median,
        "txt_rows_step_change_at_single0": step_change,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    println!("{text}");
    if let Some(p) = a.opt("json") {
        std::fs::write(p, &text).map_err(|e| format!("{p}: {e}"))?;
    }
    Ok(())
}

/// `oxi vae-decode --vae VAE --handoff DIR --out OUT.png [--vae-conv f32|winograd|int8] [--threads N]`:
/// decode sd.cpp's OWN dumped latent (`dump.vae.z`) with the Rust decoder. Comparing OUT.png with
/// sd.cpp's PNG of the same run isolates the VAE port from the DiT (same latent in, image out).
fn cmd_vae_decode(a: &Args) -> Result<(), String> {
    let vae = PathBuf::from(a.req("vae")?);
    let out = PathBuf::from(a.req("out")?);
    let h = Handoff::load(Path::new(a.req("handoff")?))?;
    let threads = a.usize_or("threads", 12)?;
    let z = h.vae_z.as_ref().ok_or("the hand-over has no dump.vae.z")?;
    let (gh, gw) = h.vae_grid().ok_or("the hand-over has no dump.vae.z")?;
    let vae_conv = a.opt("vae-conv").unwrap_or("f32");
    let repeat = a.usize_or("repeat", 1)?.max(1);
    let rep = decode_vae(&vae, &z.data, gh, gw, threads, vae_conv)?;
    // --repeat N: N-1 more decodes on a fresh load each time is not wanted (file cache noise), so
    // the extra runs reuse one loaded decoder and report the best warm decode time.
    let mut warm = Vec::new();
    if repeat > 1 {
        let dec = image_vae::load_decoder(&vae)?;
        for _ in 1..repeat {
            let t = Instant::now();
            let again = image_vae::decode_with(&dec, &z.data, latent::LATENT_CHANNELS, gh, gw, threads)?;
            warm.push(t.elapsed().as_secs_f64());
            if again.data != rep.rgb.data {
                return Err("vae-decode: a repeated decode differs from the first (non-deterministic)".into());
            }
        }
    }
    image_vae::png::write_png(&out, &rep.rgb)?;
    let best_warm = warm.iter().cloned().fold(f64::INFINITY, f64::min);
    println!(
        "{}",
        json!({"vae_conv": vae_conv, "grid": [gh, gw], "width": rep.rgb.width, "height": rep.rgb.height,
               "decode_seconds_incl_load": rep.load_s + rep.decode_s, "load_seconds": rep.load_s,
               "decode_seconds": rep.decode_s, "warm_decode_seconds": warm,
               "best_warm_decode_seconds": if warm.is_empty() { Value::Null } else { json!(best_warm) },
               "allowed_cpus": rep.allowed_cpus, "threads": threads})
    );
    Ok(())
}

fn cmd_compare(a: &Args) -> Result<(), String> {
    if a.pos.len() != 2 {
        return Err("usage: oxi compare A.png B.png".into());
    }
    let x = image_vae::png::read_png(Path::new(&a.pos[0]))?;
    let y = image_vae::png::read_png(Path::new(&a.pos[1]))?;
    if x.width != y.width || x.height != y.height || x.data.len() != y.data.len() {
        return Err(format!(
            "size mismatch: {}x{} vs {}x{}",
            x.width, x.height, y.width, y.height
        ));
    }
    let psnr = image_vae::psnr(&x, &y)?;
    let max_abs = x.data.iter().zip(&y.data).map(|(p, q)| (*p as i32 - *q as i32).abs()).max().unwrap_or(0);
    let report = json!({
        "a": a.pos[0], "b": a.pos[1], "width": x.width, "height": x.height,
        "psnr_db": psnr, "max_abs_diff": max_abs,
    });
    println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    Ok(())
}
