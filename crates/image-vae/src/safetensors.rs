//! Minimal safetensors reader.
//!
//! File layout: 8-byte little-endian u64 header length `N`, then `N` bytes of
//! JSON header (an object mapping tensor name to `{dtype, shape, data_offsets}`,
//! plus an optional `"__metadata__"` object), then the data region. `data_offsets`
//! are relative to the start of the data region.
//!
//! Supported dtypes for `tensor_f32`: F32, F16, BF16 (all little-endian).
//! The whole file is read into memory with `std::fs::read`.

use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range relative to the start of the data region.
    pub data_start: usize,
    pub data_end: usize,
}

pub struct SafeTensors {
    data: Vec<u8>,
    /// Byte offset of the data region inside `data`.
    data_base: usize,
    tensors: BTreeMap<String, TensorInfo>,
    metadata: Option<serde_json::Value>,
}

impl SafeTensors {
    /// Reads and parses the file at `path`.
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        SafeTensors::from_bytes(bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parses an in-memory safetensors image.
    pub fn from_bytes(data: Vec<u8>) -> Result<SafeTensors, String> {
        if data.len() < 8 {
            return Err("safetensors: file shorter than 8 bytes".into());
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&data[..8]);
        let header_len = u64::from_le_bytes(len_bytes);
        let data_base = 8usize
            .checked_add(usize::try_from(header_len).map_err(|_| "safetensors: header too large")?)
            .ok_or("safetensors: header length overflow")?;
        if data_base > data.len() {
            return Err(format!(
                "safetensors: header length {header_len} exceeds file size {}",
                data.len()
            ));
        }
        let header_json = std::str::from_utf8(&data[8..data_base])
            .map_err(|e| format!("safetensors: header is not UTF-8: {e}"))?;
        let root: serde_json::Value = serde_json::from_str(header_json)
            .map_err(|e| format!("safetensors: header JSON: {e}"))?;
        let obj = root
            .as_object()
            .ok_or("safetensors: header is not a JSON object")?;

        let data_len = data.len() - data_base;
        let mut tensors = BTreeMap::new();
        let mut metadata = None;
        for (name, value) in obj {
            if name == "__metadata__" {
                metadata = Some(value.clone());
                continue;
            }
            let dtype = value
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("safetensors: {name}: missing dtype"))?
                .to_string();
            let shape = value
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("safetensors: {name}: missing shape"))?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .map(|d| d as usize)
                        .ok_or_else(|| format!("safetensors: {name}: bad shape entry"))
                })
                .collect::<Result<Vec<usize>, String>>()?;
            let offs = value
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("safetensors: {name}: missing data_offsets"))?;
            let start = offs[0]
                .as_u64()
                .ok_or_else(|| format!("safetensors: {name}: bad data_offsets"))? as usize;
            let end = offs[1]
                .as_u64()
                .ok_or_else(|| format!("safetensors: {name}: bad data_offsets"))? as usize;
            if start > end || end > data_len {
                return Err(format!(
                    "safetensors: {name}: data_offsets [{start},{end}) outside data region of {data_len} bytes"
                ));
            }
            let numel: usize = shape.iter().product();
            let elem = dtype_size(&dtype).ok_or_else(|| {
                format!("safetensors: {name}: unsupported dtype {dtype}")
            })?;
            if numel * elem != end - start {
                return Err(format!(
                    "safetensors: {name}: shape {shape:?} x {elem} bytes != {} bytes",
                    end - start
                ));
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    data_start: start,
                    data_end: end,
                },
            );
        }
        Ok(SafeTensors {
            data,
            data_base,
            tensors,
            metadata,
        })
    }

    /// Tensor names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// The `__metadata__` object, if present.
    pub fn metadata(&self) -> Option<&serde_json::Value> {
        self.metadata.as_ref()
    }

    /// Returns `(shape, values)` for a tensor converted to f32.
    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("safetensors: tensor {name} not found"))?;
        let bytes = &self.data[self.data_base + info.data_start..self.data_base + info.data_end];
        let values: Vec<f32> = match info.dtype.as_str() {
            "F32" => bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            "F16" => bytes
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            "BF16" => bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .collect(),
            other => {
                return Err(format!(
                    "safetensors: tensor {name}: dtype {other} not supported (F32/F16/BF16 only)"
                ))
            }
        };
        Ok((info.shape.clone(), values))
    }
}

fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "F16" | "BF16" | "I16" | "U16" => Some(2),
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => Some(1),
        _ => None,
    }
}

/// IEEE 754 binary16 to binary32 (exact; handles subnormals, inf and NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // Subnormal: value = frac * 2^-24. Normalise into f32.
            let mut e: i32 = -14;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_file(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut header = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut body = Vec::new();
        for (name, dtype, shape, bytes) in entries {
            let start = body.len();
            body.extend_from_slice(bytes);
            let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            header.push_str(&format!(
                ",\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
                shape_s.join(","),
                body.len()
            ));
        }
        header.push('}');
        // Pad header to 8 bytes with spaces as the reference writer does.
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn round_trip_f32_f16_bf16() {
        let vals = [1.5f32, -2.25, 0.0, 3.0e-3];
        let mut f32_bytes = Vec::new();
        for v in vals {
            f32_bytes.extend_from_slice(&v.to_le_bytes());
        }
        // f16 encodings: 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800, 65504 = 0x7BFF
        let f16_bytes: Vec<u8> = [0x3C00u16, 0xC000, 0x3800, 0x7BFF]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        // bf16 of 1.0 = 0x3F80, -2.0 = 0xC000
        let bf16_bytes: Vec<u8> = [0x3F80u16, 0xC000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let file = build_file(&[
            ("a.weight", "F32", &[2, 2], &f32_bytes),
            ("b.half", "F16", &[4], &f16_bytes),
            ("c.bf", "BF16", &[2], &bf16_bytes),
        ]);
        let st = SafeTensors::from_bytes(file).unwrap();
        assert!(st.metadata().is_some());
        let names: Vec<&str> = st.names().collect();
        assert_eq!(names, vec!["a.weight", "b.half", "c.bf"]);
        let (shape, a) = st.tensor_f32("a.weight").unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(a, vals.to_vec());
        let (_, b) = st.tensor_f32("b.half").unwrap();
        assert_eq!(b, vec![1.0, -2.0, 0.5, 65504.0]);
        let (_, c) = st.tensor_f32("c.bf").unwrap();
        assert_eq!(c, vec![1.0, -2.0]);
        assert!(st.tensor_f32("missing").is_err());
    }

    #[test]
    fn rejects_bad_offsets_and_sizes() {
        // Offsets past the end of the data region.
        let mut header = String::from("{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,8]}}");
        while header.len() % 8 != 0 {
            header.push(' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&[0u8; 4]);
        assert!(SafeTensors::from_bytes(file).is_err());

        // Shape disagrees with byte count.
        let file = build_file(&[("y", "F32", &[3], &[0u8; 4])]);
        assert!(SafeTensors::from_bytes(file).is_err());

        // Header length larger than file.
        let mut bogus = 1000u64.to_le_bytes().to_vec();
        bogus.extend_from_slice(b"{}");
        assert!(SafeTensors::from_bytes(bogus).is_err());
    }

    #[test]
    fn f16_special_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
