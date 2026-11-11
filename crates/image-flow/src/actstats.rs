//! Why does a coarser scale chunk cost precision on real activations, and what would fix it without
//! giving the speed back? Measured on a dumped block input (no model needed).
//!
//! Background (ticket 030, real klein activations): the single-block prototype meets the precision
//! gate with one scale per 32 values but not with one per 256 (block delta 1.35% vs 2.3%), and
//! the MLP path, which depends only on linear1's operands, carries most of the difference. Nothing
//! accumulates numerically: every chunk's integer sum is exact and at most 96 float terms are added
//! per output. The loss is resolution: one scale per chunk is set by the largest value in the chunk,
//! so a few large channels fix the step for the 255 others.
//!
//! This module quantises the modulated block input the way the fast path does and reports the error
//! for several chunk sizes and three channel arrangements:
//! - `natural`: channels in file order (what the engine does today);
//! - `sorted`: channels permuted by their typical magnitude, so a chunk holds channels of similar
//!   size (a static column permutation of the weights; free at run time, no weight rescaling);
//! - `equalized`: every channel divided by its typical magnitude before quantisation (the weights'
//!   columns would be multiplied by it; changes the weight side too, not measured here).
//! Two error measures: `plain` = ||xq - x|| / ||x|| (dominated by the large channels), and
//! `balanced` = the same after dividing each channel by its magnitude (every channel counts equally:
//! the better proxy for the error of W x when small channels matter as much as large ones).
//! The weight side and the true output error need the weights: that is a hardware-slot sweep.

/// LayerNorm (no affine, eps 1e-6) + modulation of `n` rows of width `h`, as the block does.
pub fn modulated(x: &[f32], n: usize, h: usize, shift: &[f32], scale: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), n * h);
    let mut out = vec![0f32; n * h];
    for t in 0..n {
        let row = &x[t * h..(t + 1) * h];
        let mean = row.iter().map(|v| *v as f64).sum::<f64>() / h as f64;
        let var = row.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / h as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for k in 0..h {
            let y = ((row[k] as f64 - mean) * inv) as f32;
            out[t * h + k] = (y + y * scale[k]) + shift[k];
        }
    }
    out
}

/// Per-channel magnitude over the tokens: root mean square (0 for an all-zero channel).
pub fn channel_rms(xm: &[f32], n: usize, h: usize) -> Vec<f32> {
    let mut acc = vec![0f64; h];
    for t in 0..n {
        for (a, v) in acc.iter_mut().zip(&xm[t * h..(t + 1) * h]) {
            *a += (*v as f64).powi(2);
        }
    }
    acc.iter().map(|a| (a / n.max(1) as f64).sqrt() as f32).collect()
}

/// Channel arrangement of a quantisation experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arrange {
    Natural,
    Sorted,
    Equalized,
}

/// (plain, balanced) relative error of int8 quantisation with one max-based scale per token per `chunk`
/// channels, under the given arrangement. `mag` is the per-channel magnitude used for sorting,
/// equalising and for the balanced measure.
pub fn quant_error(xm: &[f32], n: usize, h: usize, mag: &[f32], chunk: usize, arr: Arrange) -> (f64, f64) {
    assert!(chunk > 0 && h % chunk == 0);
    let mut order: Vec<usize> = (0..h).collect();
    if arr == Arrange::Sorted {
        order.sort_by(|a, b| mag[*a].partial_cmp(&mag[*b]).unwrap_or(std::cmp::Ordering::Equal));
    }
    let div: Vec<f32> = (0..h).map(|k| if arr == Arrange::Equalized && mag[k] > 0.0 { mag[k] } else { 1.0 }).collect();
    let (mut e_plain, mut s_plain, mut e_bal, mut s_bal) = (0f64, 0f64, 0f64, 0f64);
    let mut buf = vec![0f32; chunk];
    for t in 0..n {
        let row = &xm[t * h..(t + 1) * h];
        for c in 0..h / chunk {
            let idx = &order[c * chunk..(c + 1) * chunk];
            let mut amax = 0f32;
            for (b, &k) in buf.iter_mut().zip(idx) {
                *b = row[k] / div[k];
                amax = amax.max(b.abs());
            }
            let s = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for (b, &k) in buf.iter().zip(idx) {
                let q = (b * inv).round().clamp(-127.0, 127.0) * s * div[k];
                let d = (q - row[k]) as f64;
                e_plain += d * d;
                s_plain += (row[k] as f64).powi(2);
                if mag[k] > 0.0 {
                    let m = mag[k] as f64;
                    e_bal += (d / m).powi(2);
                    s_bal += (row[k] as f64 / m).powi(2);
                }
            }
        }
    }
    ((e_plain / s_plain.max(f64::MIN_POSITIVE)).sqrt(), (e_bal / s_bal.max(f64::MIN_POSITIVE)).sqrt())
}

/// Spread of the channel magnitudes: (largest / median, share of the total energy in the 8 largest channels).
pub fn magnitude_spread(mag: &[f32]) -> (f64, f64) {
    let mut m: Vec<f64> = mag.iter().map(|v| *v as f64).collect();
    m.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = m[m.len() / 2].max(f64::MIN_POSITIVE);
    let total: f64 = m.iter().map(|v| v * v).sum();
    let top8: f64 = m.iter().rev().take(8).map(|v| v * v).sum();
    (m[m.len() - 1] / med, top8 / total.max(f64::MIN_POSITIVE))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, h: usize, big: &[usize], gain: f32) -> Vec<f32> {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut x = vec![0f32; n * h];
        for t in 0..n {
            for k in 0..h {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let u = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                x[t * h + k] = u * if big.contains(&k) { gain } else { 1.0 };
            }
        }
        x
    }

    #[test]
    fn modulation_matches_definition() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let xm = modulated(&x, 1, 4, &[0.5; 4], &[1.0; 4]);
        // LN of (1,2,3,4): mean 2.5, var 1.25
        let sd = (1.25f64 + 1e-6).sqrt();
        for (k, v) in xm.iter().enumerate() {
            let y = (x[k] as f64 - 2.5) / sd;
            assert!((*v as f64 - (2.0 * y + 0.5)).abs() < 1e-5);
        }
    }

    #[test]
    fn outlier_channels_hurt_coarse_chunks_and_sorting_or_equalising_repairs_it() {
        // 64 channels, two of them 100x larger, placed in different chunks of 32
        let (n, h) = (200usize, 64usize);
        let x = rows(n, h, &[3, 40], 100.0);
        let mag = channel_rms(&x, n, h);
        let (ratio, top8) = magnitude_spread(&mag);
        assert!(ratio > 50.0 && top8 > 0.9);
        let (_, nat) = quant_error(&x, n, h, &mag, 32, Arrange::Natural);
        let (_, sorted) = quant_error(&x, n, h, &mag, 32, Arrange::Sorted);
        let (_, eq) = quant_error(&x, n, h, &mag, 32, Arrange::Equalized);
        let (_, fine) = quant_error(&x, n, h, &mag, 8, Arrange::Natural);
        // natural at chunk 32: the small channels share a step set by the outlier -> large balanced error
        assert!(nat > 0.1, "natural {nat}");
        // Sorting puts both outliers in one chunk. Its error can still dominate
        // the aggregate, but sorting should improve it.
        assert!(sorted < nat, "sorted {sorted} vs natural {nat}");
        // equalising removes the problem altogether
        assert!(eq < 0.01, "equalized {eq}");
        assert!(fine < nat, "finer chunks help: {fine} vs {nat}");
        // without outliers all arrangements are the same small error
        let y = rows(n, h, &[], 1.0);
        let mag_y = channel_rms(&y, n, h);
        let (p, b) = quant_error(&y, n, h, &mag_y, 32, Arrange::Natural);
        assert!(p < 0.01 && b < 0.01);
    }
}
