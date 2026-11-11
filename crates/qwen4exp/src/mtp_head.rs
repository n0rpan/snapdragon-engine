//! Loader for the MTP head GGUF of Qwen3.8-Flash-Next: one non-split file holding block 48 (`blk.48.*`,
//! including the nextn tensors). The file is mapped read-only (plain MAP_SHARED, no MADV_RANDOM) and its
//! `blk.48.*` tensors are handed to `GraphModel::mtp_attach` (see `mtp_graph.rs`).

use crate::graph::{FgRegion, FgTensorRef, GraphModel};
use std::ffi::{c_void, CString};
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

const ARCH: &str = "qwen4exp";
const HEAD_PREFIX: &str = "blk.48.";
const HEAD_MARKER: &str = "blk.48.nextn.eh_proj.weight";
const EXPERTS: [&str; 3] = ["blk.48.ffn_gate_exps.weight", "blk.48.ffn_up_exps.weight", "blk.48.ffn_down_exps.weight"];

/// `~/.local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (needs HOME).
pub fn default_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf")
}

/// Shape in ggml order padded with 1s to 4 dims; None if the tensor has more than 4 dims.
fn pad_shape(shape: &[u64]) -> Option<[i64; 4]> {
    if shape.len() > 4 {
        return None;
    }
    let mut ne = [1i64; 4];
    for (i, &x) in shape.iter().enumerate() {
        ne[i] = x as i64;
    }
    Some(ne)
}

/// Whether a tensor belongs to the head block that `mtp_attach` takes.
fn is_head_tensor(name: &str) -> bool {
    name.starts_with(HEAD_PREFIX)
}

/// The MTP head file, mmapped read-only. Unmapped on drop.
///
/// The `blk.48.*` data pointers handed to a `GraphModel` by [`HeadFile::attach`] point into this mapping,
/// so the `HeadFile` must outlive every `GraphModel` it is attached to.
pub struct HeadFile {
    pub path: PathBuf,
    /// (ptr, len) of the whole-file mapping.
    map: (*mut u8, usize),
    /// (name, ggml type id, ne padded to 4 dims, absolute file offset, bytes) for every tensor in the file.
    tensors: Vec<(String, i32, [i64; 4], u64, u64)>,
}
unsafe impl Send for HeadFile {}
unsafe impl Sync for HeadFile {}

impl Drop for HeadFile {
    fn drop(&mut self) {
        // SAFETY: our own mapping, unmapped exactly once.
        unsafe { libc::munmap(self.map.0 as *mut c_void, self.map.1) };
    }
}

impl HeadFile {
    /// Parse the header and map the file. Checks: architecture is "qwen4exp", the nextn marker tensor
    /// exists, every tensor's data lies inside the file, and no tensor has more than 4 dims.
    pub fn open(path: &Path) -> Result<HeadFile, String> {
        let disp = path.display();
        let f = File::open(path).map_err(|e| format!("{disp}: {e}"))?;
        let len = f.metadata().map_err(|e| format!("{disp}: {e}"))?.len() as usize;
        if len == 0 {
            return Err(format!("{disp}: empty file"));
        }
        // SAFETY: read-only shared mapping of a model file we never write; the fd may close after this.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap {disp}: {}", std::io::Error::last_os_error()));
        }
        // From here on `hf` owns the mapping; any early return unmaps it through Drop.
        let mut hf = HeadFile { path: path.to_path_buf(), map: (ptr as *mut u8, len), tensors: Vec::new() };
        let header = gguf::parse_header(hf.bytes()).map_err(|e| format!("{disp}: {e}"))?;
        if header.architecture() != Some(ARCH) {
            return Err(format!("{disp}: architecture {:?}, expected \"{ARCH}\"", header.architecture()));
        }
        if header.tensor(HEAD_MARKER).is_none() {
            return Err(format!("{disp}: missing tensor {HEAD_MARKER} (not an MTP head file?)"));
        }
        for t in &header.tensors {
            let bytes = t.bytes();
            let abs = header.data_offset + t.offset;
            if abs.checked_add(bytes).map_or(true, |end| end > len as u64) {
                return Err(format!("{disp}: {}: data beyond end of file", t.name));
            }
            let ne = pad_shape(&t.shape).ok_or_else(|| format!("{disp}: {}: more than 4 dims", t.name))?;
            hf.tensors.push((t.name.clone(), t.ty.0 as i32, ne, abs, bytes));
        }
        Ok(hf)
    }

    /// The whole file image.
    fn bytes(&self) -> &[u8] {
        // SAFETY: the mapping is `map.1` bytes long and lives until Drop; the borrow keeps `self` alive.
        unsafe { std::slice::from_raw_parts(self.map.0, self.map.1) }
    }

    /// Attach the `blk.48.*` tensors to `m` (region index 0 = this mapping). Names are kept alive
    /// for the duration of the call only. The `HeadFile` must outlive `m`.
    pub fn attach(&self, m: &mut GraphModel) -> Result<(), String> {
        let heads: Vec<&(String, i32, [i64; 4], u64, u64)> = self.tensors.iter().filter(|t| is_head_tensor(&t.0)).collect();
        if heads.is_empty() {
            return Err(format!("{}: no {HEAD_PREFIX}* tensors", self.path.display()));
        }
        let names: Vec<CString> = heads
            .iter()
            .map(|t| CString::new(t.0.as_str()).map_err(|e| format!("{}: {e}", t.0)))
            .collect::<Result<_, _>>()?;
        let refs: Vec<FgTensorRef> = heads
            .iter()
            .zip(&names)
            .map(|(t, c)| FgTensorRef {
                name: c.as_ptr(),
                ty: t.1,
                ne: t.2,
                // SAFETY: `t.3` + `t.4` lies inside the mapping (checked in open).
                data: unsafe { self.map.0.add(t.3 as usize) } as *mut c_void,
                region: 0,
            })
            .collect();
        let region = FgRegion { base: self.map.0 as *mut c_void, size: self.map.1 };
        // SAFETY: every `data` is inside `region`, which is this mapping; the caller keeps this
        // HeadFile alive as long as `m`; `names` outlives the call.
        unsafe { m.mtp_attach(&refs, &region) }
    }

    /// Total bytes of the three routed expert tensors of block 48.
    pub fn expert_bytes(&self) -> u64 {
        self.tensors.iter().filter(|t| EXPERTS.contains(&t.0.as_str())).map(|t| t.4).sum()
    }

    /// madvise(MADV_WILLNEED) over the page-aligned ranges of the three expert tensors (no-op if !willneed).
    pub fn advise_experts(&self, willneed: bool) {
        if !willneed {
            return;
        }
        // SAFETY: sysconf with a valid name has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        for t in self.tensors.iter().filter(|t| EXPERTS.contains(&t.0.as_str())) {
            let start = t.3 / page * page;
            // Rounded up to a page boundary; stays inside the mapping because the mapping covers whole pages.
            let end = (t.3 + t.4).div_ceil(page) * page;
            // SAFETY: advisory on whole pages of our own mapping; start < end <= mapped pages.
            unsafe { libc::madvise(self.map.0.add(start as usize) as *mut c_void, (end - start) as usize, libc::MADV_WILLNEED) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_shape_and_filters_names() {
        assert_eq!(pad_shape(&[2048, 16]), Some([2048, 16, 1, 1]));
        assert_eq!(pad_shape(&[]), Some([1, 1, 1, 1]));
        assert_eq!(pad_shape(&[1, 2, 3, 4, 5]), None);
        assert!(is_head_tensor("blk.48.nextn.eh_proj.weight"));
        assert!(!is_head_tensor("blk.47.attn_q.weight"));
        assert!(!is_head_tensor("output.weight"));
    }
}
