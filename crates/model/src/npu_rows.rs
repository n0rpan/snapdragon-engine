//! Original GGUF rows for prefill-only accelerator packing. CPU repacked bytes
//! are deliberately not accepted: their interleave is a different layout.

use gguf::{GgmlType, TensorInfo};
use kernels_sys::QType;
use oxn::linear::WeightRows;

pub struct GgufRows<'a> {
    data: &'a [u8],
    ty: QType,
    o: usize,
    k: usize,
    row_bytes: usize,
}

impl<'a> GgufRows<'a> {
    /// `data` must be the original bytes of this 2-D tensor from `Gguf::tensor_data`.
    pub fn new(tensor: &TensorInfo, data: &'a [u8]) -> Result<Self, String> {
        let ty = match tensor.ty {
            GgmlType::Q8_0 => QType::Q8_0,
            GgmlType::Q4_K => QType::Q4K,
            GgmlType::Q5_K => QType::Q5K,
            GgmlType::Q6_K => QType::Q6K,
            _ => return Err(format!("npu rows: unsupported source type {:?}", tensor.ty)),
        };
        if tensor.shape.len() != 2 { return Err("npu rows: source must be 2-D".into()); }
        let k = usize::try_from(tensor.shape[0]).map_err(|_| "npu rows: K overflow")?;
        let o = usize::try_from(tensor.shape[1]).map_err(|_| "npu rows: O overflow")?;
        if k == 0 || o == 0 || k % ty.qk() != 0 {
            return Err("npu rows: invalid quantized row shape".into());
        }
        let row_bytes = (k / ty.qk()).checked_mul(ty.block_bytes()).ok_or("npu rows: row size overflow")?;
        let bytes = row_bytes.checked_mul(o).ok_or("npu rows: tensor size overflow")?;
        if bytes != data.len() { return Err("npu rows: source byte length mismatch".into()); }
        Ok(Self { data, ty, o, k, row_bytes })
    }
}

impl WeightRows for GgufRows<'_> {
    fn shape(&self) -> (usize, usize) { (self.o, self.k) }
    fn read_row(&self, row: usize, dst: &mut [f32]) -> Result<(), String> {
        if row >= self.o || dst.len() != self.k { return Err("npu rows: row index or output size mismatch".into()); }
        let start = row * self.row_bytes;
        kernels_sys::dequantize_row(self.ty, &self.data[start..start + self.row_bytes], dst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tensor(ty: GgmlType, shape: Vec<u64>) -> TensorInfo {
        TensorInfo { name: "fixture".into(), ty, shape, offset: 0 }
    }
    #[test]
    fn rejects_experts_bad_lengths_unsupported_types_and_bad_rows() {
        assert!(GgufRows::new(&tensor(GgmlType::Q4_K, vec![256, 32, 8]), &[]).is_err());
        assert!(GgufRows::new(&tensor(GgmlType::Q3_K, vec![256, 32]), &[]).is_err());
        assert!(GgufRows::new(&tensor(GgmlType::Q6_K, vec![32, 32]), &[]).is_err());
        assert!(GgufRows::new(&tensor(GgmlType::Q8_0, vec![32, 32]), &[0; 34]).is_err());
        let data = [0; 34 * 32];
        let rows = GgufRows::new(&tensor(GgmlType::Q8_0, vec![32, 32]), &data).unwrap();
        assert!(rows.read_row(32, &mut [0.0; 32]).is_err());
        assert!(rows.read_row(0, &mut [0.0; 31]).is_err());
        let mut row = [1.0; 32]; rows.read_row(31, &mut row).unwrap();
        assert_eq!(row, [0.0; 32]);
    }
    #[test]
    fn mixed_sources_expose_complete_independent_rows() {
        for (ty, qb, scale_at) in [(GgmlType::Q8_0, 34, 0), (GgmlType::Q4_K, 144, 0), (GgmlType::Q6_K, 210, 208)] {
            let k = if ty == GgmlType::Q8_0 { 32 } else { 256 };
            let mut data = vec![0; qb * 2];
            data[qb..].fill(1);
            data[qb + scale_at..qb + scale_at + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            let rows = GgufRows::new(&tensor(ty, vec![k as u64, 2]), &data).unwrap();
            let mut zero = vec![f32::NAN; k]; let mut nonzero = zero.clone();
            rows.read_row(0, &mut zero).unwrap(); rows.read_row(1, &mut nonzero).unwrap();
            assert!(zero.iter().all(|&x| x == 0.0));
            assert!(nonzero.iter().all(|x| x.is_finite()));
            assert!(nonzero.iter().any(|&x| x != 0.0));
        }
    }
}
