//! Reader for the raw tensor dumps written by the patched sd.cpp (run ticket 013).
//!
//! File names are `<tag>.call<n>.<type>.<a>x<b>x<c>x<d>.bin`, where `<a>..<d>` are ggml
//! `ne[0..4]` (`ne[0]` is the fastest-varying index). Data is contiguous little-endian; `<type>`
//! is `f32` or `f16`.

use crate::types::{BlockInput, Dims};
use int8_gemm::f16_to_f32;
use std::path::Path;

/// One dumped tensor, converted to f32, in ggml memory order (`ne[0]` fastest).
pub struct DumpTensor {
    pub ne: [usize; 4],
    pub data: Vec<f32>,
}

/// Read the single dump file for `tag` in `dir` (error if none or several match).
pub fn read_tag(dir: &Path, tag: &str) -> Result<DumpTensor, String> {
    let prefix = format!("{tag}.call");
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut hits: Vec<String> = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = ent.file_name().to_string_lossy().into_owned();
        if name.starts_with(&prefix) {
            hits.push(name);
        }
    }
    hits.sort();
    match hits.len() {
        0 => return Err(format!("no dump file for tag '{tag}' in {}", dir.display())),
        1 => {}
        n => {
            return Err(format!(
                "{n} dump files for tag '{tag}' in {}: {hits:?}",
                dir.display()
            ))
        }
    }
    let name = hits[0].clone();
    let bad = || format!("bad dump file name '{name}' (want <tag>.call<n>.<type>.<a>x<b>x<c>x<d>.bin)");

    // Everything after "<tag>.call" is "<n>.<type>.<shape>.bin".
    let body = name[prefix.len()..].strip_suffix(".bin").ok_or_else(bad)?;
    let parts: Vec<&str> = body.split('.').collect();
    if parts.len() != 3 || parts[0].parse::<u64>().is_err() {
        return Err(bad());
    }
    let ty = parts[1];
    let dims: Vec<usize> = parts[2]
        .split('x')
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<usize>, _>>()
        .map_err(|_| bad())?;
    if dims.len() != 4 {
        return Err(bad());
    }
    let ne = [dims[0], dims[1], dims[2], dims[3]];
    let count: usize = ne.iter().product();
    let esize = match ty {
        "f32" => 4usize,
        "f16" => 2usize,
        other => return Err(format!("{name}: unsupported element type '{other}'")),
    };

    let path = dir.join(&name);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != count * esize {
        return Err(format!(
            "{name}: file has {} bytes, ne {ne:?} with {ty} needs {}",
            bytes.len(),
            count * esize
        ));
    }
    let data: Vec<f32> = if ty == "f32" {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect()
    };
    Ok(DumpTensor { ne, data })
}

/// One block's inputs and the sd.cpp output to compare against.
pub struct BlockDump {
    pub input: BlockInput,
    pub expected_out: Vec<f32>,
}

/// Read the single-block dump set (`dump.single.0.in`, `dump.single.1.in`, `dump.pe`,
/// `dump.single.mod.{shift,scale,gate}`) and validate every shape against `dims`.
pub fn read_block_dump(dir: &Path, dims: &Dims) -> Result<BlockDump, String> {
    let h = dims.hidden;
    let x = read_tag(dir, "dump.single.0.in")?;
    if x.ne[0] != h || x.ne[2] != 1 || x.ne[3] != 1 || x.ne[1] == 0 {
        return Err(format!(
            "dump.single.0.in: expected ne [{h}, n_tokens, 1, 1] with n_tokens >= 1, got {:?}",
            x.ne
        ));
    }
    let n = x.ne[1];

    let expected = read_tag(dir, "dump.single.1.in")?;
    if expected.ne != x.ne {
        return Err(format!(
            "dump.single.1.in: ne {:?} differs from dump.single.0.in ne {:?}",
            expected.ne, x.ne
        ));
    }

    let half = dims.head_dim / 2;
    let pe = read_tag(dir, "dump.pe")?;
    let pe_ne = [2, 2, half, n];
    if pe.ne != pe_ne {
        return Err(format!("dump.pe: expected ne {pe_ne:?} (= [2, 2, head_dim/2, n_tokens]), got {:?}", pe.ne));
    }

    let shift = read_vec(dir, "dump.single.mod.shift", h)?;
    let scale = read_vec(dir, "dump.single.mod.scale", h)?;
    let gate = read_vec(dir, "dump.single.mod.gate", h)?;

    Ok(BlockDump {
        input: BlockInput {
            n_tokens: n,
            x: x.data,
            pe: pe.data,
            shift,
            scale,
            gate,
        },
        expected_out: expected.data,
    })
}

/// A modulation vector: element count must equal `len` (ne is reported in the error).
fn read_vec(dir: &Path, tag: &str, len: usize) -> Result<Vec<f32>, String> {
    let t = read_tag(dir, tag)?;
    if t.data.len() != len {
        return Err(format!(
            "{tag}: expected {len} elements (ne [{len}, 1, 1, 1]), got ne {:?} with {} elements",
            t.ne,
            t.data.len()
        ));
    }
    Ok(t.data)
}
