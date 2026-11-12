//! Original-weight f32 recurrent projection ablation. No activation quantization,
//! no expanded weight arena; one original dequantized row per CPU worker.
use crate::{
    npu_rows::GgufRows,
    plan::{Mixer, Plan},
    weights::WeightStore,
};
use kernels_sys::{ffi, QType};
use oxn::linear::WeightRows;
use std::{cell::RefCell, collections::HashMap};
thread_local! {static ROW:RefCell<Vec<f32>>=const{RefCell::new(Vec::new())};}
struct Matrix {
    tensor: usize,
    ty: QType,
    k: usize,
    o: usize,
    rb: usize,
}
pub(crate) struct Rows {
    g: gguf::Gguf,
    map: HashMap<usize, Matrix>,
}
impl Rows {
    pub(crate) fn new(
        g: &gguf::Gguf,
        p: &Plan,
        ws: &WeightStore,
        mask: u64,
    ) -> Result<Self, String> {
        let mut map = HashMap::new();
        if p.layers.len() > 64 {
            return Err("recurrent ablation supports <=64 layers".into());
        }
        for (l, t) in p.layers.iter().enumerate() {
            if mask & (1u64 << l) == 0 {
                continue;
            }
            let Mixer::Recurrent {
                qkv,
                z,
                beta,
                alpha,
                out,
                ..
            } = t.mixer
            else {
                continue;
            };
            for id in [qkv, z, beta, alpha, out] {
                let w = ws.get(id);
                if w.is_f32() {
                    continue;
                }
                let t = &g.header.tensors[id];
                let source = GgufRows::new(t, g.tensor_data(t))?;
                if source.shape() != (w.rows, w.k) || w.n_expert != 1 {
                    return Err("CPU f32 recurrent source geometry mismatch".into());
                }
                map.insert(
                    w.ptr,
                    Matrix {
                        tensor: id,
                        ty: w.qtype(),
                        k: w.k,
                        o: w.rows,
                        rb: w.qtype().row_bytes(w.k),
                    },
                );
            }
        }
        Ok(Self {
            g: g.remap().map_err(|e| e.to_string())?,
            map,
        })
    }
    pub(crate) fn contains(&self, key: usize) -> bool {
        self.map.contains_key(&key)
    }
    pub(crate) fn count(&self) -> usize {
        self.map.len()
    }
    /// Caller owns n rows at stride os, columns c0..c1. Both output and optional
    /// residual pointers start at c0; inputs are contiguous n*K f32 elements.
    pub(crate) unsafe fn project(
        &self,
        key: usize,
        n: usize,
        c0: usize,
        c1: usize,
        x: *const f32,
        out: *mut f32,
        os: usize,
        add: *const f32,
    ) -> bool {
        let Some(m) = self.map.get(&key) else {
            return false;
        };
        assert!(n > 0 && c0 <= c1 && c1 <= m.o && os >= c1 - c0);
        let data = self.g.tensor_data(&self.g.header.tensors[m.tensor]);
        ROW.with(|r| {
            let mut r = r.borrow_mut();
            r.resize(m.k, 0.);
            for c in c0..c1 {
                kernels_sys::dequantize_row(m.ty, &data[c * m.rb..(c + 1) * m.rb], &mut r);
                for t in 0..n {
                    let mut y = ffi::oxe_vec_dot_f32(m.k as i32, r.as_ptr(), x.add(t * m.k));
                    if !add.is_null() {
                        y += *add.add(t * os + c - c0);
                    }
                    *out.add(t * os + c - c0) = y;
                }
            }
        });
        true
    }
}
impl Drop for Rows {
    fn drop(&mut self) {
        for m in self.map.values() {
            let t = &self.g.header.tensors[m.tensor];
            self.g.release_range(t, 0, t.bytes() as usize);
            self.g.drop_cache(t);
        }
    }
}
