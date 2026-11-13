//! Production CPU linears: sd.cpp's own ggml kernels for Q8_0 weights (repacked `q8_0_4x8`
//! weights, `ggml_quantize_mat_q8_0_4x8` activations, `ggml_gemm_q8_0_4x8_q8_0`; the row tail that
//! is not a multiple of 4 takes `gemv`), as sd-cli's CPU backend computes a Q8_0 MUL_MAT with the
//! repack buffer type. Weights are repacked once at construction (2.70 GiB for klein's 27 blocks,
//! anonymous memory, freed with the struct). Columns are split over `threads` workers in whole
//! 4-row groups, so every output element has one fixed summation order (thread-count invariant).

use crate::cpu::{par_ranges, SendPtr};
use crate::encoder::Linears;
use crate::weights::{Kind, TeWeights};
use kernels_sys::{ActType, QType};
use std::collections::HashMap;
use std::time::Instant;

struct Packed {
    o: usize,
    k: usize,
    bytes: Vec<u8>,
}

pub struct GgmlLinears {
    packed: HashMap<(usize, Kind), Packed>,
    threads: usize,
    pub repack_s: f64,
}

impl GgmlLinears {
    /// Repack every linear of `w` (all must be Q8_0 with `o % 4 == 0`, `k % 32 == 0`).
    pub fn new(w: &TeWeights, threads: usize) -> Result<GgmlLinears, String> {
        let t = Instant::now();
        let mut jobs = Vec::new();
        for (l, lw) in w.layers.iter().enumerate() {
            for kd in Kind::ALL {
                let m = lw.mat(kd);
                let q = m.q8_0().ok_or_else(|| format!("cpu-ggml: blk.{l}.{} is not Q8_0", kd.gguf()))?;
                if m.o() % 4 != 0 || m.k() % 32 != 0 {
                    return Err(format!("cpu-ggml: blk.{l}.{} shape {}x{} not 4x32-aligned", kd.gguf(), m.o(), m.k()));
                }
                jobs.push(((l, kd), m.o(), m.k(), q));
            }
        }
        let mut out: Vec<Option<Packed>> = (0..jobs.len()).map(|_| None).collect();
        let op = out.as_mut_ptr() as usize;
        par_ranges(jobs.len(), threads, |r| {
            for i in r {
                let (_, o, k, q) = jobs[i];
                let mut bytes = vec![0u8; q.len()];
                kernels_sys::repack(QType::Q8_0, q, &mut bytes, o, k);
                // SAFETY: slot i is written by this range only.
                unsafe { *(op as *mut Option<Packed>).add(i) = Some(Packed { o, k, bytes }) };
            }
        });
        let packed = jobs.iter().zip(out).map(|(j, p)| (j.0, p.expect("packed"))).collect();
        Ok(GgmlLinears { packed, threads: threads.max(1), repack_s: t.elapsed().as_secs_f64() })
    }

    pub fn bytes(&self) -> usize {
        self.packed.values().map(|p| p.bytes.len()).sum()
    }
}

/// Q8_0 activations of one `n x k` input, laid out as llama.cpp's forward_mul_mat wdata: `n4/4`
/// interleaved 4-row groups, then the `n - n4` tail rows one by one (`from_float`).
pub(crate) struct QAct {
    k: usize,
    n: usize,
    groups: Vec<u8>,
    tail: Vec<u8>,
}

/// Quantize `x` once; every projection that shares this input reuses the result (q/k/v and
/// gate/up), so the bytes and therefore the outputs are exactly those of a per-projection pass.
pub(crate) fn quantize_act(x: &[f32], n: usize, k: usize, threads: usize) -> QAct {
    let at = ActType::Q8_0;
    let rb = at.bytes(k);
    let n4 = n / 4 * 4;
    let mut groups = vec![0u8; n4 * rb];
    let mut tail = vec![0u8; (n - n4) * rb];
    let ap = groups.as_mut_ptr() as usize;
    par_ranges(n4 / 4, threads, |g| {
        for gi in g {
            // SAFETY: group gi owns bytes [gi*4*rb, (gi+1)*4*rb) and reads 4 full rows of x.
            unsafe { kernels_sys::quantize_mat4_raw(at, x[gi * 4 * k..].as_ptr(), (ap as *mut u8).add(gi * 4 * rb), k) };
        }
    });
    for (i, t) in (n4..n).enumerate() {
        // SAFETY: one row of k floats into rb bytes.
        unsafe { kernels_sys::quantize_row_raw(at, x[t * k..].as_ptr(), tail[i * rb..].as_mut_ptr(), k) };
    }
    QAct { k, n, groups, tail }
}

/// `y = x W^T` for one repacked Q8_0 matrix from pre-quantized activations.
fn gemm_q(p: &Packed, a: &QAct, y: &mut [f32], threads: usize) {
    let (o, k, n) = (p.o, p.k, a.n);
    debug_assert_eq!(a.k, k);
    let rb = ActType::Q8_0.bytes(k);
    let n4 = n / 4 * 4;
    let groups = o / 4;
    let yp = SendPtr(y.as_mut_ptr());
    par_ranges(groups, threads, |gs| {
        let yp = yp;
        if gs.is_empty() {
            return;
        }
        let (c0, nc) = (gs.start * 4, gs.len() * 4);
        let w = p.bytes[QType::Q8_0.group_offset(c0, k)..].as_ptr();
        // SAFETY: columns c0..c0+nc of every row belong to this range; sizes as documented in kernels-sys.
        unsafe {
            if n4 > 0 {
                kernels_sys::gemm_raw(QType::Q8_0, k, yp.0.add(c0), o, w, a.groups.as_ptr(), n4, nc);
            }
            for (i, t) in (n4..n).enumerate() {
                kernels_sys::gemv_raw(QType::Q8_0, k, yp.0.add(t * o + c0), w, a.tail[i * rb..].as_ptr(), nc);
            }
        }
    });
}

/// `y = x W^T` for one repacked Q8_0 matrix (quantizes `x` itself; the pre-v0.1.3 per-projection
/// path, kept as the bitwise reference of the shared-activation path).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gemm_one(lin: &GgmlLinears, layer: usize, kind: Kind, x: &[f32], n: usize, y: &mut Vec<f32>) -> Result<(), String> {
    let p = lin.packed.get(&(layer, kind)).ok_or_else(|| format!("cpu-ggml: no blk.{layer}.{}", kind.gguf()))?;
    if x.len() != n * p.k {
        return Err(format!("cpu-ggml: input {} for n {n} k {}", x.len(), p.k));
    }
    y.resize(n * p.o, 0.0);
    let a = quantize_act(x, n, p.k, lin.threads);
    gemm_q(p, &a, y, lin.threads);
    Ok(())
}

impl Linears for GgmlLinears {
    fn name(&self) -> String {
        "cpu-ggml".into()
    }
    fn project(&self, layer: usize, kinds: &[Kind], x: &[f32], n: usize, outs: &mut [Vec<f32>]) -> Result<(), String> {
        // One quantization of the shared input for every projection (q/k/v, gate/up): the
        // activation bytes are a pure function of `x`, so this is bitwise the per-projection path.
        let mut act: Option<QAct> = None;
        for (i, kd) in kinds.iter().enumerate() {
            let p = self.packed.get(&(layer, *kd)).ok_or_else(|| format!("cpu-ggml: no blk.{layer}.{}", kd.gguf()))?;
            if x.len() != n * p.k {
                return Err(format!("cpu-ggml: input {} for n {n} k {}", x.len(), p.k));
            }
            if !matches!(&act, Some(a) if a.k == p.k) {
                act = Some(quantize_act(x, n, p.k, self.threads));
            }
            outs[i].resize(n * p.o, 0.0);
            gemm_q(p, act.as_ref().expect("quantized"), &mut outs[i], self.threads);
        }
        Ok(())
    }
    fn summary(&self) -> serde_json::Value {
        serde_json::json!({"repack_s": self.repack_s, "repacked_bytes": self.bytes()})
    }
}
