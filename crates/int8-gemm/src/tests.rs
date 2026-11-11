use super::*;

/// Small deterministic generator (no external crates).
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
    /// Roughly normal (sum of 4 uniforms), occasionally a large outlier.
    fn act(&mut self) -> f32 {
        let g = self.unit() + self.unit() + self.unit() + self.unit();
        if self.next() % 97 == 0 {
            g * 9.0
        } else {
            g
        }
    }
}

fn random_f32(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.act()).collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn rel_rmse(got: &[f32], want: &[f64]) -> f64 {
    let (mut e, mut s) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        e += (*g as f64 - w).powi(2);
        s += w * w;
    }
    (e / s).sqrt()
}

#[test]
fn abi() {
    assert_eq!(unsafe { raw::i8g_kernel_abi() }, 1);
}

#[test]
fn f16_decode() {
    assert_eq!(f16_to_f32(0x3c00), 1.0);
    assert_eq!(f16_to_f32(0xc000), -2.0);
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
    assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
    assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2.0f32.powi(-24));
    assert_eq!(f16_to_f32(0x0400), 2.0f32.powi(-14));
    assert_eq!(f16_to_f32(0x3555), 0.333251953125);
}

#[test]
fn quantize_round_trip_and_layout() {
    let mut rng = Rng(7);
    let (n, k, chunk) = (11, 96, 32);
    let x = random_f32(&mut rng, n * k);
    let p = Packed::from_f32(&x, n, k, chunk);
    assert_eq!(p.groups(), 2);
    for v in 0..n {
        let d = p.dequantize(v);
        for c in 0..k / chunk {
            let xc = &x[v * k + c * chunk..v * k + (c + 1) * chunk];
            let amax = xc.iter().fold(0.0f32, |m, a| m.max(a.abs()));
            assert_eq!(p.scale(v, c), amax / 127.0);
            for i in 0..chunk {
                let err = (d[c * chunk + i] - xc[i]).abs();
                assert!(err <= 0.5001 * p.scale(v, c), "v{v} c{c} i{i} err {err}");
                assert!(p.code(v, c * chunk + i).unsigned_abs() <= 127);
            }
        }
    }
    // padding vectors stay zero
    for v in n..16 {
        for i in 0..k {
            assert_eq!(padded_code(&p, v, i), 0);
        }
    }
}

#[test]
fn zero_chunk_is_safe() {
    let mut x = vec![0.0f32; 64];
    x[40] = 3.0;
    let p = Packed::from_f32(&x, 1, 64, 32);
    assert_eq!(p.scale(0, 0), 0.0);
    assert_eq!(p.code(0, 40), 127);
    assert!(p.dequantize(0).iter().all(|v| v.is_finite()));
}

fn check_bitwise(n: usize, m: usize, k: usize, chunk: usize, seed: u64) {
    let mut rng = Rng(seed);
    let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, chunk);
    let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, chunk);
    for order in [Order::MulAdd, Order::Fma] {
        let (tg, rg) = (0..x.groups(), 0..w.groups());
        let stride = rg.len() * G + 3;
        let len = tg.len() * G * stride;
        let mut a = vec![f32::NAN; len];
        let mut b = vec![f32::NAN; len];
        gemm(&x, &w, tg.clone(), rg.clone(), &mut a, stride, order, false);
        gemm_ref(&x, &w, tg.clone(), rg.clone(), &mut b, stride, order, false);
        for t in 0..tg.len() * G {
            assert_eq!(bits(&a[t * stride..t * stride + rg.len() * G]), bits(&b[t * stride..t * stride + rg.len() * G]), "n{n} m{m} k{k} chunk{chunk} {order:?} t{t}");
            // the gap between rows of y is not touched
            assert!(a[t * stride + rg.len() * G..(t + 1) * stride].iter().all(|v| v.is_nan()));
        }
        // accumulate: second pass adds onto the first, same contract
        gemm(&x, &w, tg.clone(), rg.clone(), &mut a, stride, order, true);
        gemm_ref(&x, &w, tg.clone(), rg.clone(), &mut b, stride, order, true);
        for t in 0..tg.len() * G {
            assert_eq!(bits(&a[t * stride..t * stride + rg.len() * G]), bits(&b[t * stride..t * stride + rg.len() * G]));
        }
    }
}

#[test]
fn smmla_matches_scalar_contract_bitwise() {
    check_bitwise(8, 8, 32, 32, 1);
    check_bitwise(16, 24, 256, 32, 2);
    check_bitwise(13, 19, 256, 64, 3);
    check_bitwise(9, 40, 768, 256, 4);
    check_bitwise(24, 8, 768, 768, 5);
    check_bitwise(5, 5, 3072, 3072, 6);
    check_bitwise(8, 16, 96, 8, 7);
}

#[test]
fn sub_ranges_match_full() {
    let mut rng = Rng(11);
    let (n, m, k, chunk) = (32, 48, 128, 64);
    let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, chunk);
    let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, chunk);
    let mut full = vec![0.0f32; n * m];
    gemm(&x, &w, 0..4, 0..6, &mut full, m, Order::MulAdd, false);
    let mut part = vec![0.0f32; 16 * 24];
    gemm(&x, &w, 1..3, 2..5, &mut part, 24, Order::MulAdd, false);
    for t in 0..16 {
        for r in 0..24 {
            assert_eq!(part[t * 24 + r].to_bits(), full[(t + 8) * m + r + 16].to_bits());
        }
    }
}

#[test]
fn extreme_codes_do_not_overflow() {
    // all codes +-127 (and -128 on the weight side, as a foreign Q8_0 file could hold)
    let (k, chunk) = (4096, 4096);
    let mut x = Packed::zeroed(8, k, chunk);
    let mut w = Packed::zeroed(8, k, chunk);
    for v in 0..8 {
        x.set_codes(v, 0, &vec![if v % 2 == 0 { 127 } else { -127 }; k], &[1.0]);
        w.set_codes(v, 0, &vec![if v < 4 { -128 } else { 127 }; k], &[1.0]);
    }
    let mut y = vec![0.0f32; 64];
    gemm(&x, &w, 0..1, 0..1, &mut y, 8, Order::MulAdd, false);
    for t in 0..8 {
        for r in 0..8 {
            let want = (if t % 2 == 0 { 127i64 } else { -127 }) * (if r < 4 { -128 } else { 127 }) * k as i64;
            assert_eq!(y[t * 8 + r], want as f32);
        }
    }
}

fn q8_0_bytes(rng: &mut Rng, rows: usize, k: usize) -> (Vec<u8>, Vec<f64>) {
    let mut bytes = Vec::new();
    let mut vals = Vec::new();
    for _ in 0..rows * k / 32 {
        // d as an exactly representable half: m * 2^-12, m in 1..1024
        let m = 1 + (rng.next() % 1023) as u16;
        let h: u16 = {
            // m * 2^-12 as f16: normalise
            let lz = m.leading_zeros() - 5; // m < 2^11
            let exp = 15 - 12 + (10 - lz as i32);
            let man = ((m as u32) << lz) & 0x3ff;
            ((exp as u16) << 10) | man as u16
        };
        let d = m as f64 / 4096.0;
        assert_eq!(f16_to_f32(h) as f64, d);
        bytes.extend_from_slice(&h.to_le_bytes());
        for _ in 0..32 {
            let q = (rng.next() % 255) as i32 - 127;
            bytes.push(q as i8 as u8);
            vals.push(d * q as f64);
        }
    }
    (bytes, vals)
}

#[test]
fn q8_0_chunk32_is_verbatim_and_perm_works() {
    let mut rng = Rng(21);
    let (rows, k) = (10, 96);
    let (bytes, vals) = q8_0_bytes(&mut rng, rows, k);
    let p = Packed::from_q8_0(&bytes, rows, k, 32, None);
    for r in 0..rows {
        let d = p.dequantize(r);
        for i in 0..k {
            assert_eq!(d[i] as f64, vals[r * k + i]);
        }
    }
    let perm = [7u32, 0, 9, 3];
    let q = Packed::from_q8_0(&bytes, rows, k, 32, Some(&perm));
    assert_eq!(q.n(), 4);
    for (v, &r) in perm.iter().enumerate() {
        assert_eq!(q.dequantize(v), p.dequantize(r as usize));
    }
}

#[test]
fn q8_0_requant_error_bound() {
    let mut rng = Rng(22);
    let (rows, k) = (8, 256);
    let (bytes, vals) = q8_0_bytes(&mut rng, rows, k);
    for chunk in [64, 128, 256] {
        let p = Packed::from_q8_0(&bytes, rows, k, chunk, None);
        for r in 0..rows {
            let d = p.dequantize(r);
            for i in 0..k {
                let err = (d[i] as f64 - vals[r * k + i]).abs();
                assert!(err <= 0.5001 * p.scale(r, i / chunk) as f64);
            }
        }
    }
}

#[test]
fn gemm_accuracy_vs_f64() {
    // f32 inputs, int8 x int8 GEMM against the f64 product of the ORIGINAL f32 data:
    // the error must be small and must shrink with the chunk size.
    let mut rng = Rng(31);
    let (n, m, k) = (16, 16, 768);
    let xf = random_f32(&mut rng, n * k);
    let wf = random_f32(&mut rng, m * k);
    let mut want = vec![0.0f64; n * m];
    for t in 0..n {
        for r in 0..m {
            want[t * m + r] = (0..k).map(|i| xf[t * k + i] as f64 * wf[r * k + i] as f64).sum();
        }
    }
    let mut last = f64::MAX;
    for chunk in [768, 256, 32] {
        let x = Packed::from_f32(&xf, n, k, chunk);
        let w = Packed::from_f32(&wf, m, k, chunk);
        let mut y = vec![0.0f32; n * m];
        gemm(&x, &w, 0..2, 0..2, &mut y, m, Order::MulAdd, false);
        let e = rel_rmse(&y, &want);
        assert!(e < 0.05, "chunk {chunk}: rel rmse {e}");
        assert!(e < last * 1.05, "chunk {chunk}: {e} not better than {last}");
        last = e;
    }
}

#[test]
fn two_level_matches_reference_and_values() {
    // K-quant shape: f32 scale per 256, integer scale per 16.
    let mut rng = Rng(41);
    let (n, m, k, chunk, sub) = (8, 16, 512, 256, 16);
    let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, chunk);
    let mut w = Packed::zeroed(m, k, chunk);
    let mut sub_scales = vec![0i16; w.groups() * (k / sub) * G];
    for r in 0..m {
        let codes: Vec<i8> = (0..k).map(|_| (rng.next() % 64) as i8 - 32).collect();
        let sup: Vec<f32> = (0..k / chunk).map(|_| 0.001 + rng.unit().abs() * 0.01).collect();
        w.set_codes(r, 0, &codes, &sup);
        for sb in 0..k / sub {
            sub_scales[((r / G) * (k / sub) + sb) * G + r % G] = (rng.next() % 255) as i16 - 127;
        }
    }
    w.sub = Some(SubScales { sub, scales: sub_scales });
    for order in [Order::MulAdd, Order::Fma] {
        let mut a = vec![0.0f32; n * m];
        let mut b = vec![0.0f32; n * m];
        gemm(&x, &w, 0..1, 0..2, &mut a, m, order, false);
        gemm_ref(&x, &w, 0..1, 0..2, &mut b, m, order, false);
        assert_eq!(bits(&a), bits(&b));
        // and against the decoded values in f64
        let mut want = vec![0.0f64; n * m];
        for t in 0..n {
            let xd = x.dequantize(t);
            for r in 0..m {
                let wd = w.dequantize(r);
                want[t * m + r] = (0..k).map(|i| xd[i] as f64 * wd[i] as f64).sum();
            }
        }
        assert!(rel_rmse(&a, &want) < 1e-5);
    }
}

#[test]
fn blocked_driver_covers_everything_once() {
    let mut rng = Rng(51);
    let (n, m, k, chunk) = (37, 24, 64, 64);
    let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, chunk);
    let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, chunk);
    let mut full = vec![0.0f32; x.groups() * G * m];
    gemm(&x, &w, 0..x.groups(), 0..3, &mut full, m, Order::MulAdd, false);
    let mut seen = vec![0u32; x.groups() * G];
    let mut scratch = Vec::new();
    gemm_blocked(&x, &w, 0..x.groups(), 1..3, 2, Order::MulAdd, &mut scratch, |t0, nt, block| {
        assert_eq!(block.len(), nt * 16);
        for t in 0..nt {
            seen[t0 + t] += 1;
            for r in 0..16 {
                assert_eq!(block[t * 16 + r].to_bits(), full[(t0 + t) * m + 8 + r].to_bits());
            }
        }
    });
    assert!(seen.iter().all(|&c| c == 1));
}

#[test]
fn unsigned_kernel_matches_scalar() {
    let mut rng = Rng(61);
    let steps = 5;
    let x: Vec<u8> = (0..steps * 64).map(|_| (rng.next() % 256) as u8).collect();
    let w: Vec<i8> = (0..steps * 64).map(|_| (rng.next() % 256) as u8 as i8).collect();
    let mut out = [0i32; 64];
    unsafe { raw::i8g_udot8x8(x.as_ptr(), w.as_ptr(), steps, out.as_mut_ptr()) };
    for t in 0..8 {
        for r in 0..8 {
            let mut s = 0i32;
            for st in 0..steps {
                for i in 0..8 {
                    s += x[st * 64 + t * 8 + i] as i32 * w[st * 64 + r * 8 + i] as i32;
                }
            }
            assert_eq!(out[t * 8 + r], s);
        }
    }
    // scaled unsigned tile: one chunk, scales 1 -> same integers as floats
    let ones = [1.0f32; 8];
    let mut y = [0.0f32; 64];
    unsafe { raw::i8g_utile8x8(x.as_ptr(), w.as_ptr(), steps, 1, ones.as_ptr(), ones.as_ptr(), y.as_mut_ptr(), 8, 0) };
    for i in 0..64 {
        assert_eq!(y[i], out[i] as f32);
    }
}

#[test]
fn softmax_pack_matches_scalar() {
    let mut rng = Rng(71);
    for &kc in &[8usize, 64, 256] {
        for qi in [0usize, 3, 7] {
            let mut s: Vec<f32> = (0..kc).map(|_| rng.act() * 3.0).collect();
            if kc > 8 {
                s[kc - 1] = f32::NEG_INFINITY; // a padding key
                s[5] = -200.0; // far below the clamp
            }
            let sv: Vec<f32> = (0..kc).map(|_| rng.unit().abs() * 0.05).collect();
            let cmax = unsafe { raw::i8g_row_max(s.as_ptr(), kc) };
            assert_eq!(cmax, s.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            let m = cmax + 0.25; // a running max from an earlier chunk
            // scalar reference in f64
            let e: Vec<f64> = s.iter().map(|&v| ((v - m) as f64).max(-87.0).exp()).collect();
            let want_sum: f64 = e.iter().sum();
            let pv: Vec<f64> = e.iter().zip(&sv).map(|(a, b)| a * *b as f64).collect();
            let pmax = pv.iter().copied().fold(0.0f64, f64::max);
            let mut codes = vec![0xAAu8; (kc / 8) * 64];
            let (mut sum, mut scale) = (0.0f32, 0.0f32);
            let mut work = s.clone();
            unsafe { raw::i8g_softmax_pack(work.as_mut_ptr(), kc, m, sv.as_ptr(), codes.as_mut_ptr(), qi, &mut sum, &mut scale) };
            assert!(((sum as f64 - want_sum) / want_sum).abs() < 1e-4, "sum {sum} vs {want_sum}");
            assert!(((scale as f64 - pmax / 255.0) / (pmax / 255.0)).abs() < 1e-4);
            for j in 0..kc {
                let got = codes[(j / 8) * 64 + qi * 8 + j % 8] as f64;
                let want = pv[j] / pmax * 255.0;
                assert!((got - want).abs() <= 0.5 + 0.05, "kc {kc} qi {qi} j {j}: {got} vs {want}");
            }
            // other queries' bytes are untouched
            for (i, &b) in codes.iter().enumerate() {
                if (i % 64) / 8 != qi {
                    assert_eq!(b, 0xAA);
                }
            }
        }
    }
    // an all-zero value scale gives scale 0 and zero codes
    let mut s = vec![0.0f32; 16];
    let sv = vec![0.0f32; 16];
    let mut codes = vec![7u8; 128];
    let (mut sum, mut scale) = (0.0f32, 1.0f32);
    unsafe { raw::i8g_softmax_pack(s.as_mut_ptr(), 16, 0.0, sv.as_ptr(), codes.as_mut_ptr(), 2, &mut sum, &mut scale) };
    assert_eq!(scale, 0.0);
    assert!((sum - 16.0).abs() < 1e-4);
    assert!(codes[16..24].iter().all(|&b| b == 0) && codes[64 + 16..64 + 24].iter().all(|&b| b == 0));
}

/// Kernel roofline on one core (ignored: timing). L1-resident operands, so this is the SMMLA tile
/// itself; compare with the per-core rate of the block fixture to see how much is memory and glue.
/// `cargo test --release -p int8-gemm -- --ignored --nocapture kernel_peak`
#[test]
#[ignore]
fn kernel_peak_single_core() {
    let mut rng = Rng(99);
    for &(n, m, k, chunk, label) in &[
        (8usize, 8usize, 3072usize, 3072usize, "L1: 1 tile, k=3072, one fix-up"),
        (8, 8, 3072, 256, "L1: 1 tile, k=3072, chunk 256"),
        (8, 8, 3072, 32, "L1: 1 tile, k=3072, chunk 32"),
        (8, 256, 3072, 256, "X in L1, W 786 KB (L2): 32 row groups, chunk 256"),
        (64, 256, 3072, 256, "X 196 KB, W 786 KB (both L2), chunk 256"),
        (8, 8, 128, 128, "attention QK shape: k=128"),
    ] {
        let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, chunk);
        let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, chunk);
        let mut y = vec![0.0f32; n * m];
        let macs_per_pass = (n * m * k) as f64;
        let passes = (4.0e9 / macs_per_pass).ceil() as usize;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..passes {
                gemm(&x, &w, 0..x.groups(), 0..w.groups(), &mut y, m, Order::MulAdd, false);
            }
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let gmac_s = macs_per_pass * passes as f64 / best / 1e9;
        println!("kernel peak [{label}]: {gmac_s:.1} GMAC/s per core = {:.2} TOPS/core, {:.2} SMMLA/ns", 2.0 * gmac_s / 1000.0, gmac_s / 32.0);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}

/// Roofline probe (ignored: timing): is an SDOT tile (16 MAC per instruction) faster or slower per
/// core than the SMMLA tile (32 MAC per instruction)? L1-resident operands, pure integer kernels.
#[test]
#[ignore]
fn kernel_peak_sdot_vs_smmla() {
    let mut rng = Rng(5);
    let steps = 192usize; // 3072 k for SDOT (16 per step), 12 KB per operand
    let x: Vec<i8> = (0..steps * 64).map(|_| (rng.next() % 255) as i32 as i8).collect();
    let w: Vec<i8> = (0..steps * 64).map(|_| (rng.next() % 255) as i32 as i8).collect();
    // correctness of the probe on the first tile
    let mut out = [0i32; 16];
    unsafe { raw::i8g_probe_sdot4x4(x.as_ptr(), w.as_ptr(), steps, out.as_mut_ptr()) };
    for t in 0..4 {
        for r in 0..4 {
            let mut s = 0i32;
            for st in 0..steps {
                for i in 0..16 {
                    s += x[st * 64 + t * 16 + i] as i32 * w[st * 64 + r * 16 + i] as i32;
                }
            }
            assert_eq!(out[t * 4 + r], s);
        }
    }
    let reps = 400_000usize;
    let mut best_sdot = f64::MAX;
    let mut best_smmla = f64::MAX;
    let mut out64 = [0i32; 64];
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            unsafe { raw::i8g_probe_sdot4x4(x.as_ptr(), w.as_ptr(), steps, out.as_mut_ptr()) };
        }
        best_sdot = best_sdot.min(t0.elapsed().as_secs_f64());
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            // same bytes read as an 8x8 SMMLA tile: steps of 8 k, 64 bytes per step
            unsafe { raw::i8g_dot8x8(x.as_ptr(), w.as_ptr(), steps, out64.as_mut_ptr()) };
        }
        best_smmla = best_smmla.min(t0.elapsed().as_secs_f64());
    }
    let sdot_mac = (reps * steps * 16 * 16) as f64; // 16 SDOT x 16 MAC per step
    let smmla_mac = (reps * steps * 16 * 32) as f64; // 16 SMMLA x 32 MAC per step
    println!(
        "kernel peak SDOT 4x4: {:.1} GMAC/s ({:.2} SDOT/ns); SMMLA 8x8 (integer only): {:.1} GMAC/s ({:.2} SMMLA/ns)",
        sdot_mac / best_sdot / 1e9,
        sdot_mac / 16.0 / best_sdot / 1e9,
        smmla_mac / best_smmla / 1e9,
        smmla_mac / 32.0 / best_smmla / 1e9
    );
}

/// Roofline probe (ignored: timing): the same tile kernel on `OXI_PEAK_THREADS` (default 12)
/// threads at once, each with its own operands. Separates what the machine gives with all cores
/// busy (frequency, shared L2 of a 4-core cluster) from what a GEMM driver loses on top.
/// `cargo test --release -p int8-gemm -- --ignored --nocapture kernel_peak_all_cores`
#[test]
#[ignore]
fn kernel_peak_all_cores() {
    let threads: usize = std::env::var("OXI_PEAK_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    // OXI_PEAK_SCALE multiplies the work per case (default about 0.15 s each); OXI_PEAK_CASES picks cases ("0,1").
    let scale: f64 = std::env::var("OXI_PEAK_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let cases: Option<Vec<usize>> = std::env::var("OXI_PEAK_CASES").ok().map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect());
    for (ci, &(n, m, k, chunk, label)) in [
        (8usize, 8usize, 3072usize, 256usize, "L1: 1 tile, k=3072, chunk 256"),
        (32, 128, 3072, 256, "X 98 KB, W 393 KB per thread (qkv unit), chunk 256"),
        (32, 512, 3072, 256, "X 98 KB, W 1.57 MB per thread (mlp unit), chunk 256"),
        (32, 64, 12288, 256, "X 393 KB, W 786 KB per thread (linear2 block), chunk 256"),
    ]
    .iter()
    .enumerate()
    {
        if cases.as_ref().is_some_and(|c| !c.contains(&ci)) {
            continue;
        }
        let start = std::sync::Barrier::new(threads);
        let rates: Vec<f64> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|i| {
                    let start = &start;
                    s.spawn(move || {
                        let mut rng = Rng(1000 + i as u64);
                        let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, chunk);
                        let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, chunk);
                        let mut y = vec![0.0f32; n * m];
                        let macs_per_pass = (n * m * k) as f64;
                        let passes = (3.0e10 * scale / macs_per_pass).ceil() as usize;
                        gemm(&x, &w, 0..x.groups(), 0..w.groups(), &mut y, m, Order::MulAdd, false);
                        start.wait();
                        let t0 = std::time::Instant::now();
                        for _ in 0..passes {
                            gemm(&x, &w, 0..x.groups(), 0..w.groups(), &mut y, m, Order::MulAdd, false);
                        }
                        macs_per_pass * passes as f64 / t0.elapsed().as_secs_f64() / 1e9
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("thread")).collect()
        });
        let sum: f64 = rates.iter().sum();
        let min = rates.iter().cloned().fold(f64::MAX, f64::min);
        println!("kernel peak x{threads} [{label}]: {sum:.0} GMAC/s = {:.2} TOPS, per thread mean {:.1} min {min:.1}", 2.0 * sum / 1000.0, sum / threads as f64);
    }
}

#[test]
fn rows1_is_bitwise_gemm() {
    let mut rng = Rng(4242);
    for &(n, m, k) in &[(8usize, 8usize, 128usize), (24, 40, 128), (13, 250, 32), (16, 64, 8), (40, 24, 256)] {
        let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, k);
        let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, k);
        let stride = w.groups() * G + 5;
        for tg in 0..x.groups() {
            for rg in [0..w.groups(), 1.min(w.groups() - 1)..w.groups(), 0..1] {
                let len = 7 * stride + rg.len() * G;
                let mut a = vec![7.0f32; len];
                let mut b = vec![7.0f32; len];
                gemm(&x, &w, tg..tg + 1, rg.clone(), &mut a, stride, Order::MulAdd, false);
                gemm_rows1(&x, &w, tg, rg.clone(), &mut b, stride);
                assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()), "n {n} m {m} k {k} tg {tg} rg {rg:?}");
            }
        }
    }
}

/// Timing (ignored): the attention q.k shape through the general tile path and through `gemm_rows1`.
#[test]
#[ignore]
fn kernel_peak_qk_rows1() {
    let mut rng = Rng(77);
    let (n, m, k) = (8usize, 256usize, 128usize);
    let x = Packed::from_f32(&random_f32(&mut rng, n * k), n, k, k);
    let w = Packed::from_f32(&random_f32(&mut rng, m * k), m, k, k);
    let mut y = vec![0.0f32; n * m];
    let passes = 20000usize;
    let (mut best_a, mut best_b) = (f64::MAX, f64::MAX);
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        for _ in 0..passes {
            gemm(&x, &w, 0..1, 0..w.groups(), &mut y, m, Order::MulAdd, false);
        }
        best_a = best_a.min(t0.elapsed().as_secs_f64());
        let t0 = std::time::Instant::now();
        for _ in 0..passes {
            gemm_rows1(&x, &w, 0, 0..w.groups(), &mut y, m);
        }
        best_b = best_b.min(t0.elapsed().as_secs_f64());
    }
    let mac = (n * m * k * passes) as f64 / 1e9;
    println!("kernel peak q.k (8 queries x 256 keys, k=128): tile path {:.1} GMAC/s, rows1 {:.1} GMAC/s per core", mac / best_a, mac / best_b);
}
