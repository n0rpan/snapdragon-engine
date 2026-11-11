//! O-13: int8 (Q8_0) KV cache rows and the attention reads over them (first version: correctness
//! first). A K or V row of one (layer, KV head, position) is head_dim values in Q8_0 blocks of 32
//! (f16 scale d = amax/127, q = round(x/d)), quantised once when the row is written, with the
//! vendored ggml `quantize_row_q8_0` (the arm build llama.cpp's own f32 -> q8_0 KV copy runs).
//! Reads dequantise (q * d in f32, exact; then f32 -> f16 as the f16 cache would hold it) into a
//! per-thread f16 scratch in L1-sized chunks and run the existing `oxe_dot_f16_tile` kernels.
//! Every reader of a row sees the same f16 values, whoever wrote it and however it was batched.

use kernels_sys::{ffi, ops};

/// Values per Q8_0 block and bytes per block (f16 scale + 32 int8).
pub const QK: usize = 32;
pub const BLK: usize = 34;
/// K rows dequantised per chunk of the score tile (rows x head_dim f16 + f32: 48 KB at 256 dims).
pub const K_ROWS: usize = 32;
/// Positions per V block of the output tile (32 dims x 256 positions: 16 KB f16 + 32 KB f32).
pub const V_POS: usize = 256;

/// KV cache element type (`OXE_KV_TYPE=f16|k8|v8|kv8`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvType {
    /// f16 K rows and f16 V^T rows (the default; bit-identical to the engine before O-13)
    #[default]
    F16,
    /// Q8_0 K rows, f16 V^T
    K8,
    /// f16 K rows, Q8_0 V rows (per position)
    V8,
    /// Q8_0 K and V rows
    KV8,
}

impl KvType {
    pub fn parse(s: &str) -> Option<KvType> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "" => Some(KvType::F16),
            "k8" => Some(KvType::K8),
            "v8" => Some(KvType::V8),
            "kv8" | "q8_0" => Some(KvType::KV8),
            _ => None,
        }
    }
    /// `OXE_KV_TYPE` (unset: f16).
    pub fn from_env() -> Result<KvType, String> {
        match std::env::var("OXE_KV_TYPE") {
            Err(_) => Ok(KvType::F16),
            Ok(v) => KvType::parse(&v).ok_or_else(|| format!("OXE_KV_TYPE={v}: expected f16, k8, v8 or kv8")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::K8 => "k8",
            KvType::V8 => "v8",
            KvType::KV8 => "kv8",
        }
    }
    #[inline]
    pub fn k8(self) -> bool {
        matches!(self, KvType::K8 | KvType::KV8)
    }
    #[inline]
    pub fn v8(self) -> bool {
        matches!(self, KvType::V8 | KvType::KV8)
    }
    #[inline]
    pub fn any8(self) -> bool {
        self != KvType::F16
    }
}

/// Bytes of one Q8_0 row of `hd` values.
#[inline]
pub fn row_bytes(hd: usize) -> usize {
    hd / QK * BLK
}

/// IEEE half -> f32 (exact; GGML_FP16_TO_FP32).
#[inline]
pub fn h2f(h: u16) -> f32 {
    let s = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            // subnormal: m * 2^-24 (exact)
            let v = m as f32 * (1.0 / 16777216.0);
            return if s != 0 { -v } else { v };
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Quantise `x` (len a multiple of 32) into Q8_0 `out` (len row_bytes(x.len())).
#[inline]
pub fn quant_row(x: &[f32], out: &mut [u8]) {
    debug_assert!(x.len() % QK == 0 && out.len() >= row_bytes(x.len()));
    // SAFETY: x has k floats, out has k/32 blocks.
    unsafe { ffi::quantize_row_q8_0(x.as_ptr(), out.as_mut_ptr(), x.len() as i64) }
}

/// Dequantise one Q8_0 block to f32 (ggml dequantize_row_q8_0: qs[j] * d, exact in f32).
#[inline]
pub fn deq_block(b: &[u8], out: &mut [f32]) {
    let d = h2f(u16::from_le_bytes([b[0], b[1]]));
    for j in 0..QK {
        out[j] = (b[2 + j] as i8) as f32 * d;
    }
}

/// Dequantise `n` values of Q8_0 data to f32.
#[inline]
pub fn deq_f32(src: &[u8], n: usize, out: &mut [f32]) {
    for i in 0..n / QK {
        deq_block(&src[i * BLK..(i + 1) * BLK], &mut out[i * QK..(i + 1) * QK]);
    }
}

/// The f16 values every attention read of a stored Q8_0 row uses: f16(q * d).
#[inline]
pub fn deq_f16(src: &[u8], n: usize, tmp: &mut [f32], out: &mut [u16]) {
    deq_f32(src, n, tmp);
    ops::fp32_to_fp16(&tmp[..n], &mut out[..n]);
}

/// Per-thread scratch of the int8 attention reads.
pub struct Kv8Scratch {
    f32s: Vec<f32>,
    f16s: Vec<u16>,
    part: Vec<f32>,
    /// fused path: one chunk's f32 probabilities [p][col] and one chunk's outputs [col][dim]
    pscr: Vec<f32>,
    pub pacc: Vec<f32>,
}

impl Kv8Scratch {
    pub fn new(hd: usize, max_ncol: usize) -> Kv8Scratch {
        let n = (K_ROWS * hd).max(QK * V_POS);
        let nc = max_ncol.max(1);
        Kv8Scratch {
            f32s: vec![0.0; n],
            f16s: vec![0; n],
            part: vec![0.0; QK * nc],
            // the largest of the V . p kernels' scratches (reference, fast: floats; integer: bytes)
            // SAFETY: a pure size computation.
            pscr: vec![0.0; (2 * PV_CHUNK * nc.div_ceil(4) * 4).max(unsafe { ffi::oxe_pv_q8_int_scratch(hd as i32, PV_CHUNK as i32, nc as i32) } as usize / 4 + 1)],
            pacc: vec![0.0; nc * hd],
        }
    }
    pub fn empty() -> Kv8Scratch {
        Kv8Scratch { f32s: Vec::new(), f16s: Vec::new(), part: Vec::new(), pscr: Vec::new(), pacc: Vec::new() }
    }
}

/// Positions per partial of the fused V . p (absolute boundaries 0, 256, ...): the unit of the
/// decode work split and of the fixed-order f32 reduction (also used by the prefill).
pub const PV_CHUNK: usize = 256;

/// `OXE_KV8_FUSED=0`: the first-version reads (dequantise to f16 scratch + f16 tiles) for A/B.
pub fn fused_from_env() -> bool {
    std::env::var("OXE_KV8_FUSED").map(|v| v.trim() != "0").unwrap_or(true)
}

/// Kernel of the fused V . p (`OXE_KV8_PV=fast|ref|i16|i8`, read at engine construction).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PvMode {
    /// `oxe_pv_q8_fast`: fp32, bit-identical to `Ref` (the default)
    #[default]
    Fast,
    /// `oxe_pv_q8`: the first fused kernel, kept as the reference
    Ref,
    /// `oxe_pv_q8_int` with 16-bit weights: LOSSY (about 3e-4 of the output norm; needs the KLD gate)
    I16,
    /// `oxe_pv_q8_int` with 8-bit weights: LOSSY (up to about 2x the int8 V error; needs the KLD gate)
    I8,
}

impl PvMode {
    pub fn parse(s: &str) -> Option<PvMode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fast" | "" => Some(PvMode::Fast),
            "ref" => Some(PvMode::Ref),
            "i16" | "int16" => Some(PvMode::I16),
            "i8" | "int8" => Some(PvMode::I8),
            _ => None,
        }
    }
    /// True when the outputs equal the reference kernel's bit for bit.
    pub fn lossless(self) -> bool {
        matches!(self, PvMode::Fast | PvMode::Ref)
    }
}

/// `OXE_KV8_PV`: the V . p kernel (unset or unknown: `Fast`). An unknown value and a lossy choice
/// are reported on stderr, so a run's log shows that its outputs are not the reference's.
pub fn pv_mode_from_env() -> PvMode {
    match std::env::var("OXE_KV8_PV") {
        Ok(v) => {
            let m = PvMode::parse(&v).unwrap_or_else(|| {
                eprintln!("oxe: OXE_KV8_PV={v:?} is not fast|ref|i16|i8, using fast");
                PvMode::Fast
            });
            if !m.lossless() {
                eprintln!("oxe: OXE_KV8_PV={m:?}: LOSSY integer V . p for int8 V (not bit-identical to the reference kernel)");
            }
            m
        }
        Err(_) => PvMode::Fast,
    }
}

/// Fused V . p of positions [p0, p1) (one chunk, <= PV_CHUNK): out[c * hd + d] (overwritten).
/// `Fast` and `Ref` give the same bits; `I16` / `I8` are the lossy integer kernels (probabilities
/// must be >= 0, as attention probabilities are).
///
/// # Safety
/// vrows holds rows < p1 (`row_bytes(hd)` apart); probs has ncol columns `ps` apart readable on
/// [p0, p1); out has ncol * hd floats; sc from `Kv8Scratch::new(hd, >= ncol)`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn pv_chunk(mode: PvMode, vrows: *const u8, hd: usize, p0: usize, p1: usize, probs: *const u16, ps: usize, ncol: usize, out: *mut f32, sc: &mut Kv8Scratch) {
    debug_assert!(p1 - p0 <= PV_CHUNK && sc.pscr.len() >= 2 * (p1 - p0) * ncol.div_ceil(4) * 4);
    let (hd_, rb, a, b, ps_, nc) = (hd as i32, row_bytes(hd) as i64, p0 as i32, p1 as i32, ps as i64, ncol as i32);
    match mode {
        PvMode::Fast => ffi::oxe_pv_q8_fast(hd_, vrows, rb, a, b, probs, ps_, nc, out, sc.pscr.as_mut_ptr()),
        PvMode::Ref => ffi::oxe_pv_q8(hd_, vrows, rb, a, b, probs, ps_, nc, out, sc.pscr.as_mut_ptr()),
        PvMode::I16 | PvMode::I8 => {
            debug_assert!(sc.pscr.len() * 4 >= ffi::oxe_pv_q8_int_scratch(hd_, b - a, nc) as usize);
            ffi::oxe_pv_q8_int(hd_, vrows, rb, a, b, probs, ps_, nc, out, sc.pscr.as_mut_ptr() as *mut core::ffi::c_void, if mode == PvMode::I16 { 16 } else { 8 })
        }
    }
}

/// Scores of Q8_0 K rows [p0, p1) of one KV head against `ncol` f16 query columns:
/// out[(p - p0) + c * oy] = vec_dot_f16(f16(deq K[p]), q[c]) (the f16 tile on the dequantised rows).
/// `krows`: row 0 of the KV head (rows `rb` bytes apart).
///
/// # Safety
/// krows holds rows < p1; q has ncol columns of hd halves `qs` apart; out is writable at every
/// (p - p0) + c * oy.
#[allow(clippy::too_many_arguments)]
pub unsafe fn scores_q8(krows: *const u8, hd: usize, p0: usize, p1: usize, q: *const u16, qs: usize, ncol: usize, out: *mut f32, oy: usize, sc: &mut Kv8Scratch) {
    let rb = row_bytes(hd);
    let mut p = p0;
    while p < p1 {
        let pe = (p + K_ROWS).min(p1);
        let n = (pe - p) * hd;
        // rows of one KV head are contiguous: one dequantisation run for the chunk
        let src = std::slice::from_raw_parts(krows.add(p * rb), (pe - p) * rb);
        deq_f16(src, n, &mut sc.f32s, &mut sc.f16s);
        ffi::oxe_dot_f16_tile(hd as i32, sc.f16s.as_ptr(), hd as i64, (pe - p) as i32, q, qs as i64, ncol as i32, out.add(p - p0), 1, oy as i64, std::ptr::null());
        p = pe;
    }
}

/// Outputs over Q8_0 V rows (per position) of one KV head, dims [d0, d1), positions [0, n)
/// (n a multiple of 32): out[(d - d0) * ncol + c] = sum_p f16(deq V[p][d]) * prob[c][p], as f16
/// tiles over position blocks of `V_POS` (absolute boundaries 0, 256, ...) summed in f32 in
/// block order. Positions past a column's own range must hold +0 probabilities (and finite V).
///
/// # Safety
/// vrows holds rows < n (`row_bytes(hd)` apart); probs has ncol columns `ps` apart readable on
/// [0, n); out has (d1 - d0) * ncol floats.
#[allow(clippy::too_many_arguments)]
pub unsafe fn outputs_q8(vrows: *const u8, hd: usize, d0: usize, d1: usize, n: usize, probs: *const u16, ps: usize, ncol: usize, out: *mut f32, sc: &mut Kv8Scratch) {
    let rb = row_bytes(hd);
    if d1 <= d0 {
        return;
    }
    for i in 0..(d1 - d0) * ncol {
        *out.add(i) = 0.0;
    }
    let mut blk = [0f32; QK];
    for bi in d0 / QK..d1.div_ceil(QK) {
        let (a, b) = (d0.max(bi * QK), d1.min((bi + 1) * QK));
        let nd = b - a;
        let mut pb = 0;
        while pb < n {
            let len = (n - pb).min(V_POS);
            // dequantise + transpose: f32s[j * V_POS + (p - pb)] = V[p][bi*32 + j]
            for i in 0..len {
                let src = std::slice::from_raw_parts(vrows.add((pb + i) * rb + bi * BLK), BLK);
                deq_block(src, &mut blk);
                for j in 0..QK {
                    sc.f32s[j * V_POS + i] = blk[j];
                }
            }
            for j in a - bi * QK..b - bi * QK {
                ops::fp32_to_fp16(&sc.f32s[j * V_POS..j * V_POS + len], &mut sc.f16s[j * V_POS..j * V_POS + len]);
            }
            ffi::oxe_dot_f16_tile(
                len as i32,
                sc.f16s.as_ptr().add((a - bi * QK) * V_POS),
                V_POS as i64,
                nd as i32,
                probs.add(pb),
                ps as i64,
                ncol as i32,
                sc.part.as_mut_ptr(),
                ncol as i64,
                1,
                std::ptr::null(),
            );
            for r in 0..nd {
                for c in 0..ncol {
                    *out.add((a - d0 + r) * ncol + c) += sc.part[r * ncol + c];
                }
            }
            pb += len;
        }
    }
}
