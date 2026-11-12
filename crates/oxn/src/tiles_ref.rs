//! Slow per-element reference implementations of the tile layouts (test-only).
//!
//! Written directly from the layout formulas with plain parameters. The only code
//! shared with tiles.rs is the scalar half conversion pair.

use super::{f32_to_half, half_to_f32};

/// Flat u16 index of activation element (m, kk) for padded T, K and kc.
pub fn ref_act_index(t_pad: usize, k: usize, kc: usize, m: usize, kk: usize) -> usize {
    assert!(m < t_pad && kk < k && kc % 32 == 0 && k % kc == 0);
    let kct = kc / 32;
    let rtiles = t_pad / 32;
    let c = kk / kc;
    let kl = kk % kc;
    let tile = c * rtiles * kct + (m / 32) * kct + kl / 32;
    let e = ((m % 32) / 2) * 64 + 2 * (kl % 32) + (m % 2);
    tile * 1024 + e
}

/// Reference X writer: element by element.
pub fn ref_write_act(dst: &mut [u16], t_pad: usize, k: usize, kc: usize, m: usize, k0: usize, row: &[f32]) {
    for (i, &v) in row.iter().enumerate() {
        dst[ref_act_index(t_pad, k, kc, m, k0 + i)] = f32_to_half(v);
    }
}

/// Reference Y reader: sums chunks in ascending order, first chunk assigns.
#[allow(clippy::too_many_arguments)]
pub fn ref_read_out(
    src: &[u16],
    t_pad: usize,
    o: usize,
    nchunks: usize,
    ycol: bool,
    m: usize,
    n0: usize,
    out: &mut [f32],
) {
    let rtiles = t_pad / 32;
    let ntiles_o = o / 32;
    for (i, slot) in out.iter_mut().enumerate() {
        let n = n0 + i;
        let mut acc = 0f32;
        for c in 0..nchunks {
            let tile = if ycol {
                (n / 32) * rtiles + m / 32
            } else {
                (m / 32) * ntiles_o + n / 32
            };
            let idx = c * rtiles * ntiles_o * 1024 + tile * 1024 + ((m % 32) / 2) * 64 + 2 * (n % 32) + (m % 2);
            let v = half_to_f32(src[idx]);
            acc = if c == 0 { v } else { acc + v };
        }
        *slot = acc;
    }
}

/// Reference W8 packer: starts from zeros and writes element by element.
pub fn ref_pack_w8(o: usize, k: usize, kc: usize, q8: &[u8], perm: Option<&[u32]>) -> Vec<u8> {
    let nb = k / 32;
    let kct = kc / 32;
    let ntiles_o = o / 32;
    let mut dst = vec![0u8; ntiles_o * nb * 1152];
    for n in 0..o {
        let s = perm.map_or(n, |p| p[n] as usize);
        for kk in 0..k {
            let c = kk / kc;
            let kl = kk % kc;
            let rec = c * ntiles_o * kct + (n / 32) * kct + kl / 32;
            let base = rec * 1152;
            let blk = (s * nb + kk / 32) * 34;
            let nl = n % 32;
            let kli = kl % 32;
            dst[base + (kli / 2) * 64 + 2 * nl + (kli % 2)] = q8[blk + 2 + kk % 32];
            dst[base + 1024 + 2 * nl] = q8[blk];
            dst[base + 1024 + 2 * nl + 1] = q8[blk + 1];
        }
    }
    dst
}

/// Reference W8 element read: weight(n, k_idx) = half_to_f32(scale) * q.
pub fn ref_w8_get(w8: &[u8], o: usize, k: usize, kc: usize, n: usize, k_idx: usize) -> f32 {
    assert!(n < o && k_idx < k);
    let kct = kc / 32;
    let ntiles_o = o / 32;
    let c = k_idx / kc;
    let kl = k_idx % kc;
    let rec = c * ntiles_o * kct + (n / 32) * kct + kl / 32;
    let base = rec * 1152;
    let nl = n % 32;
    let kli = kl % 32;
    let scale = half_to_f32(u16::from_le_bytes([w8[base + 1024 + 2 * nl], w8[base + 1025 + 2 * nl]]));
    let q = w8[base + (kli / 2) * 64 + 2 * nl + (kli % 2)] as i8;
    scale * (q as f32)
}
