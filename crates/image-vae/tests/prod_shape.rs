//! Production-shape decode with RANDOM weights at the real FLUX.2 channel layout.
//! Names and shapes come from tests/fixtures/flux2-vae.tensors.txt (decoder.* only).
//! Ignored by default (heavy). Run with, from the workspace root:
//!   tools/lane_build.sh --mem test --release -p image-vae -j 2 -- --ignored --nocapture prod_shape_decode_512
//! The 1024 px variant additionally needs OXI_VAE_FIXTURE_1024=1.

use std::collections::HashMap;
use std::path::PathBuf;

use image_vae::decoder::{Tensors, VaeDecoder};

fn table_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/flux2-vae.tensors.txt")
}

/// Parses `name<TAB>dtype<TAB>[d0, d1, ...]` lines for `decoder.*` tensors.
fn parse_decoder_table() -> Vec<(String, Vec<usize>)> {
    let text = std::fs::read_to_string(table_path()).expect("read tensor table");
    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split('\t');
        let (Some(name), Some(_dtype), Some(shape)) = (parts.next(), parts.next(), parts.next()) else {
            continue;
        };
        if !(name.starts_with("decoder.")) {
            continue;
        }
        let dims: Vec<usize> = shape
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse().expect("dim"))
            .collect();
        out.push((name.to_string(), dims));
    }
    assert!(!out.is_empty(), "no decoder tensors in table");
    out
}

/// Small random values: conv weights uniform with variance 1/fan_in, biases tiny,
/// GroupNorm weight 1 and bias 0.
fn random_tensors(seed: u64) -> Tensors {
    let mut s = seed;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut t: Tensors = HashMap::new();
    for (name, shape) in parse_decoder_table() {
        let n: usize = shape.iter().product();
        let vals: Vec<f32> = if name.ends_with(".weight") && shape.len() == 4 {
            let fan_in = (shape[1] * shape[2] * shape[3]) as f32;
            let a = (3.0 / fan_in).sqrt();
            (0..n).map(|_| rnd() * a).collect()
        } else if name.contains("norm") && name.ends_with(".weight") {
            vec![1.0; n]
        } else if name.contains("norm") {
            vec![0.0; n]
        } else {
            (0..n).map(|_| rnd() * 0.01).collect()
        };
        t.insert(name, (shape, vals));
    }
    t
}

/// (relative RMSE, PSNR in dB as 8-bit pixels `clamp((v + 1) / 2) * 255`, fraction of values inside [-1, 1]) of `a` against `b`.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut e, mut sig, mut mse_px, mut inside) = (0f64, 0f64, 0f64, 0usize);
    for (a, b) in a.iter().zip(b) {
        e += ((a - b) as f64).powi(2);
        sig += (*b as f64).powi(2);
        let pa = ((*a as f64 + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0;
        let pb = ((*b as f64 + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0;
        mse_px += (pa - pb).powi(2);
        inside += (b.abs() <= 1.0) as usize;
    }
    let n = a.len() as f64;
    let psnr = if mse_px > 0.0 { 10.0 * (255.0f64 * 255.0 / (mse_px / n)).log10() } else { f64::INFINITY };
    ((e / sig).sqrt(), psnr, inside as f64 / n)
}

/// A/B in ONE process: the channel-major decode of 8531151 against the channel-last decode,
/// f32 and int8, interleaved `OXI_VAE_REPS` times (default 2); the table shows the best of each.
fn run(h: usize, threads: usize) {
    use image_vae::decoder::{set_int8_conv, set_winograd, DecodeTimes};
    let t = random_tensors(0x5eed);
    let dec = VaeDecoder::from_tensors(t, 32).expect("build decoder");
    assert_eq!(dec.latent_channels(), 128);
    let mut s = 0x1234_5678u64;
    let z: Vec<f32> = (0..128 * h * h)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let reps: usize = std::env::var("OXI_VAE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    // arms: (name, int8, channel-last, winograd)
    let arms = [
        ("f32 channel-major (8531151)", false, false, false),
        ("f32 channel-last", false, true, false),
        ("int8 channel-major (8531151)", true, false, false),
        ("int8 channel-last", true, true, false),
        ("f32 channel-last winograd", false, true, true),
    ];
    const NA: usize = 5;
    let mut best: Vec<Option<DecodeTimes>> = vec![None; NA];
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); NA];
    let mut all_totals: Vec<Vec<f64>> = vec![Vec::new(); NA];
    // OXI_VAE_ARMS=1,3 runs only those arms (profiling); the comparison lines then cover what ran
    let only: Option<Vec<usize>> = std::env::var("OXI_VAE_ARMS").ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect());
    let runs = |ai: usize| only.as_ref().is_none_or(|o| o.contains(&ai));
    for rep in 0..reps + 1 {
        for (ai, (_, int8, last, wino)) in arms.iter().enumerate() {
            if !runs(ai) {
                continue;
            }
            set_int8_conv(*int8);
            set_winograd(*wino);
            let (out, tm) = if *last { dec.decode_timed(&z, 128, h, h, threads) } else { dec.decode_timed_channel_major(&z, 128, h, h, threads) }.expect("decode");
            set_int8_conv(false);
            set_winograd(false);
            if rep == 0 {
                // warm-up pass (first-use weight packing); also the determinism reference
                outs[ai] = out;
                continue;
            }
            assert!(out.iter().zip(&outs[ai]).all(|(a, b)| a.to_bits() == b.to_bits()), "{}: two runs differ", arms[ai].0);
            all_totals[ai].push(tm.total);
            if best[ai].as_ref().is_none_or(|b| tm.total < b.total) {
                best[ai] = Some(tm);
            }
        }
    }
    println!("prod_shape h={h} (output {}x{}) threads={threads}, best of {reps} interleaved reps after one warm-up pass", 16 * h, 16 * h);
    let ran: Vec<usize> = (0..NA).filter(|&ai| runs(ai)).collect();
    let best: Vec<Option<DecodeTimes>> = best;
    let cell = |ai: usize, f: &dyn Fn(&DecodeTimes) -> f64| best[ai].as_ref().map(|b| format!("{:>12.4}", f(b))).unwrap_or_else(|| format!("{:>12}", "-"));
    println!("  {:<36} {:>12} {:>12} {:>12} {:>12} {:>12}", "stage [s]", "f32 major", "f32 last", "int8 major", "int8 last", "f32 wino");
    let nstages = best[ran[0]].as_ref().expect("ran").stages.len();
    for i in 0..nstages {
        let name = best[ran[0]].as_ref().unwrap().stages[i].0.clone();
        println!("  {name:<36} {} {} {} {} {}", cell(0, &|b| b.stages[i].1), cell(1, &|b| b.stages[i].1), cell(2, &|b| b.stages[i].1), cell(3, &|b| b.stages[i].1), cell(4, &|b| b.stages[i].1));
    }
    println!("  {:<36} {} {} {} {} {}", "TOTAL", cell(0, &|b| b.total), cell(1, &|b| b.total), cell(2, &|b| b.total), cell(3, &|b| b.total), cell(4, &|b| b.total));
    for &ai in &ran {
        println!("  all totals, {}: {:?}", arms[ai].0, all_totals[ai].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
    for ai in [1usize, 3, 4] {
        if let Some(b) = &best[ai] {
            println!("  kinds, {}: {}", arms[ai].0, b.kinds.iter().map(|(k, v)| format!("{k} {v:.3}")).collect::<Vec<_>>().join(", "));
        }
    }
    if ran.len() < NA {
        return;
    }
    for o in &outs {
        assert_eq!(o.len(), 3 * 16 * h * 16 * h);
        assert!(o.iter().all(|v| v.is_finite()), "non-finite output");
    }
    let bitwise = outs[1].iter().zip(&outs[0]).all(|(a, b)| a.to_bits() == b.to_bits());
    let max_abs = outs[1].iter().zip(&outs[0]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let (rel, psnr, inside) = compare(&outs[1], &outs[0]);
    println!("  f32 channel-last vs f32 channel-major: bitwise {bitwise}, rel RMSE {rel:.3e}, max abs diff {max_abs:.3e}, PSNR {psnr:.1} dB ({:.0}% of the values inside [-1, 1])", inside * 100.0);
    assert!(psnr >= 50.0, "channel-last f32 decode is {psnr:.1} dB from the channel-major decode (gate 50 dB)");
    let max_abs = outs[4].iter().zip(&outs[0]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let (rel, psnr, _) = compare(&outs[4], &outs[0]);
    println!("  f32 channel-last winograd vs f32 channel-major: rel RMSE {rel:.3e}, max abs diff {max_abs:.3e}, PSNR {psnr:.1} dB");
    assert!(psnr >= 50.0, "winograd f32 decode is {psnr:.1} dB from the channel-major decode (gate 50 dB)");
    let (rel, psnr, _) = compare(&outs[3], &outs[1]);
    println!("  int8 channel-last vs f32 channel-last: rel RMSE {rel:.4}, PSNR {psnr:.1} dB (random weights)");
    let (rel, psnr, _) = compare(&outs[2], &outs[0]);
    println!("  int8 channel-major vs f32 channel-major: rel RMSE {rel:.4}, PSNR {psnr:.1} dB (random weights)");
    let (rel, psnr, _) = compare(&outs[3], &outs[2]);
    println!("  int8 channel-last vs int8 channel-major: rel RMSE {rel:.4}, PSNR {psnr:.1} dB");
}

#[test]
#[ignore]
fn prod_shape_decode_512() {
    run(32, 12);
}

#[test]
#[ignore]
fn prod_shape_decode_1024() {
    if std::env::var("OXI_VAE_FIXTURE_1024").as_deref() != Ok("1") {
        eprintln!("skipped: set OXI_VAE_FIXTURE_1024=1");
        return;
    }
    run(64, 12);
}
