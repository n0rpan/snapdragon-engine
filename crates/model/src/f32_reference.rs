//! Original-GGUF f32 reference. No activation quantization, fp16 products,
//! probability rounding or f16 KV. Weight rows are streamed, never all expanded.
//! Private proof path only; sequential prefill/step, no parked-state ABI.
use crate::{
    plan::Plan,
    weights::{Mapping, WeightStore},
};
use gguf::{GgmlType, Gguf};
use kernels_sys::QType;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {static ROW:RefCell<Vec<f32>>=const{RefCell::new(Vec::new())};}
struct Matrix {
    tensor: usize,
    ty: QType,
    k: usize,
    rows: usize,
    experts: usize,
    row_bytes: usize,
}

/// F32 shadows have the same key row / value column organization as the CPU
/// graph. All mutations require the phase's exclusive element ownership.
pub(crate) struct Kv {
    k: Mapping,
    v: Mapping,
    layers: usize,
    nkv: usize,
    hd: usize,
    nctx: usize,
}
impl Kv {
    fn new(layers: usize, nkv: usize, hd: usize, nctx: usize) -> Result<Self, String> {
        let elements = layers
            .checked_mul(nkv)
            .and_then(|x| x.checked_mul(hd))
            .and_then(|x| x.checked_mul(nctx))
            .ok_or("f32 ref KV overflow")?;
        let bytes = elements.checked_mul(4).ok_or("f32 ref KV bytes overflow")?;
        Ok(Self {
            k: Mapping::anon(bytes, false).map_err(|e| e.to_string())?,
            v: Mapping::anon(bytes, false).map_err(|e| e.to_string())?,
            layers,
            nkv,
            hd,
            nctx,
        })
    }
    /// Caller owns this (layer,head,position), and no reader runs before barrier.
    pub(crate) unsafe fn put_k(&self, l: usize, h: usize, p: usize, row: &[f32]) {
        assert!(l < self.layers && h < self.nkv && p < self.nctx && row.len() == self.hd);
        let at = ((l * self.nkv + h) * self.nctx + p) * self.hd;
        std::ptr::copy_nonoverlapping(row.as_ptr(), self.k.base().cast::<f32>().add(at), self.hd);
    }
    /// Caller owns this value column's position; every other writer is disjoint.
    pub(crate) unsafe fn put_v(&self, l: usize, row: usize, p: usize, value: f32) {
        assert!(l < self.layers && row < self.nkv * self.hd && p < self.nctx);
        *self
            .v
            .base()
            .cast::<f32>()
            .add((l * self.nkv * self.hd + row) * self.nctx + p) = value;
    }
    pub(crate) fn krow(&self, l: usize, h: usize, p: usize) -> *const f32 {
        assert!(l < self.layers && h < self.nkv && p < self.nctx);
        self.k
            .base()
            .cast::<f32>()
            .wrapping_add(((l * self.nkv + h) * self.nctx + p) * self.hd)
    }
    pub(crate) fn vcol(&self, l: usize, h: usize, d: usize) -> *const f32 {
        assert!(l < self.layers && h < self.nkv && d < self.hd);
        self.v
            .base()
            .cast::<f32>()
            .wrapping_add(((l * self.nkv + h) * self.hd + d) * self.nctx)
    }
}

pub(crate) struct Reference {
    g: Gguf,
    mats: HashMap<usize, Matrix>,
    pub(crate) kv: Kv,
}
impl Reference {
    pub(crate) fn new(
        g: Gguf,
        sha: &str,
        plan: &Plan,
        ws: &WeightStore,
        nctx: usize,
    ) -> Result<Self, String> {
        if sha.len() != 64 || sha != ws.info.model_sha256 {
            return Err("f32 ref source/CPU identity mismatch".into());
        }
        if nctx > 4096 {
            return Err("f32 reference is bounded to4096 positions".into());
        }
        let mut mats = HashMap::new();
        for id in plan.tensors() {
            let w = ws.get(id);
            if w.is_f32() {
                continue;
            }
            let t = g
                .header
                .tensors
                .get(id)
                .ok_or("f32 ref missing source tensor")?;
            let ty = match t.ty {
                GgmlType::Q3_K => QType::Q3K,
                GgmlType::Q4_K => QType::Q4K,
                GgmlType::Q5_K => QType::Q5K,
                GgmlType::Q6_K => QType::Q6K,
                GgmlType::Q8_0 => QType::Q8_0,
                _ => return Err(format!("f32 ref unsupported original type {:?}", t.ty)),
            };
            if !(2..=3).contains(&t.shape.len())
                || t.shape[0] != w.k as u64
                || t.shape[1] != w.rows as u64
                || t.shape.get(2).copied().unwrap_or(1) != w.n_expert as u64
                || w.k % ty.qk() != 0
            {
                return Err(format!("f32 ref source geometry mismatch {}", t.name));
            }
            let rb = ty.row_bytes(w.k);
            let bytes = rb
                .checked_mul(w.rows)
                .and_then(|x| x.checked_mul(w.n_expert))
                .ok_or("f32 ref matrix bytes overflow")?;
            if g.tensor_data(t).len() != bytes {
                return Err("f32 ref original row bytes mismatch".into());
            }
            mats.insert(
                w.ptr,
                Matrix {
                    tensor: id,
                    ty,
                    k: w.k,
                    rows: w.rows,
                    experts: w.n_expert,
                    row_bytes: rb,
                },
            );
        }
        let kv = Kv::new(
            plan.desc.n_attention(),
            plan.desc.n_head_kv,
            plan.desc.head_dim,
            nctx,
        )?;
        Ok(Self { g, mats, kv })
    }
    /// A row-local callback reuses one TLS buffer per CPU worker. It never reads
    /// the interleaved CPU arena as original GGUF, nor retains a borrowed row.
    pub(crate) fn with_row<R>(
        &self,
        key: usize,
        expert: usize,
        row: usize,
        f: impl FnOnce(&[f32]) -> R,
    ) -> R {
        let m = self
            .mats
            .get(&key)
            .expect("f32 ref uncovered quantized matrix");
        assert!(expert < m.experts && row < m.rows);
        let data = self.g.tensor_data(&self.g.header.tensors[m.tensor]);
        let at = (expert * m.rows + row) * m.row_bytes;
        ROW.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            scratch.resize(m.k, 0.0);
            kernels_sys::dequantize_row(m.ty, &data[at..at + m.row_bytes], &mut scratch);
            assert!(
                scratch.iter().all(|v| v.is_finite()),
                "nonfinite reference weight"
            );
            f(&scratch)
        })
    }
}
impl Drop for Reference {
    fn drop(&mut self) {
        for t in &self.g.header.tensors {
            self.g.release_range(t, 0, t.bytes() as usize);
            self.g.drop_cache(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shadow_kv_keeps_original_f32_not_half_and_preserves_other_positions() {
        let kv = Kv::new(1, 1, 4, 8).unwrap();
        let row = [1.0001234, -0.3000123, 0.000123456, 3.1412345];
        unsafe {
            kv.put_k(0, 0, 3, &row);
            for (i, &x) in row.iter().enumerate() {
                kv.put_v(0, i, 3, x);
            }
        }
        unsafe {
            assert_eq!(std::slice::from_raw_parts(kv.krow(0, 0, 3), 4), row);
            assert_eq!(*kv.krow(0, 0, 2), 0.0);
            for (i, &x) in row.iter().enumerate() {
                assert_eq!(*kv.vcol(0, 0, i).add(3), x);
                assert_eq!(*kv.vcol(0, 0, i).add(2), 0.0);
            }
        }
        assert_ne!(
            row[0],
            oxn::tiles::half_to_f32(oxn::tiles::f32_to_half(row[0]))
        );
    }
    #[test]
    fn q3_original_rows_dequantize_without_repack_or_activation_quantization() {
        let mut bytes = vec![0u8; 110];
        let mut row = vec![f32::NAN; 256];
        kernels_sys::dequantize_row(QType::Q3K, &bytes, &mut row);
        assert!(row.iter().all(|&v| v == 0.0));
        bytes[108..110].copy_from_slice(&0x2000u16.to_le_bytes());
        kernels_sys::dequantize_row(QType::Q3K, &bytes, &mut row);
        assert!(row.iter().all(|v| v.is_finite()));
        assert!(row.iter().any(|&v| v != 0.0));
    }
}
