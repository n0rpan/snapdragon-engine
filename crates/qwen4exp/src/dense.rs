//! Dense (always-resident) weights: every non-expert tensor except the N-gram table, read once
//! at start into one anonymous THP arena (buffered pread + FADV_DONTNEED, so the shard pages do
//! not stay in the cgroup's page cache). Optionally Q6_K / Q8_0 matrices are repacked into
//! ggml's 8x8 / 4x8 interleaved layouts (`--dense repack`, kernels-sys GEMVs); the default keeps
//! the file layout and the production NEON `vec_dot` (`--dense vecdot`, the exactness path).

use crate::ffi::{self, W};
use crate::model::{Shards, Tensor};
use crate::uring::Arena;
use kernels_sys::{ActType, QType};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseMode {
    VecDot,
    Repack,
}

/// A row-major matrix: `n` rows of `k` weights.
#[derive(Clone, Copy)]
pub struct Mat {
    pub p: *const u8,
    pub w: W,
    pub k: usize,
    pub n: usize,
    pub repacked: bool,
}
unsafe impl Send for Mat {}
unsafe impl Sync for Mat {}

impl Mat {
    pub fn act(&self) -> Option<ActType> {
        self.w.act()
    }
    pub fn bytes(&self) -> usize {
        self.w.row_bytes(self.k) * self.n
    }
    /// Rows [r0, r1) times the quantised activation `a` (or f32 `x` for F32/BF16).
    ///
    /// # Safety
    /// `a` holds `act().bytes(k)`; `out` holds r1-r0 floats; for repacked matrices r0/r1 are
    /// multiples of the interleave (8 for Q6_K, 4 for Q8_0).
    #[inline]
    pub unsafe fn gemv(&self, r0: usize, r1: usize, a: &[u8], x: &[f32], out: &mut [f32]) {
        if r1 <= r0 {
            return;
        }
        match self.w {
            W::F32 => ffi::gemv_f32(self.p as *const f32, self.k, r0, r1, x, out),
            W::Bf16 => ffi::gemv_bf16(self.p as *const u16, self.k, r0, r1, x, out),
            _ if self.repacked => {
                let qt = if self.w == W::Q6K { QType::Q6K } else { QType::Q8_0 };
                kernels_sys::gemv_raw(qt, self.k, out.as_mut_ptr(), self.p.add(qt.group_offset(r0, self.k)), a.as_ptr(), r1 - r0)
            }
            _ => ffi::gemv_vecdot(self.w, self.p, self.k, r0, r1, a.as_ptr(), out.as_mut_ptr()),
        }
    }
}

pub struct Dense {
    _arena: Arena,
    pub mats: HashMap<String, Mat>,
    pub load_seconds: f64,
    pub arena_bytes: usize,
}

fn is_dense(t: &Tensor) -> bool {
    !t.name.contains("_exps.") && t.name != "per_layer_token_embd.weight"
}

impl Dense {
    pub fn load(s: &Shards, mode: DenseMode) -> Result<Dense, String> {
        let t0 = Instant::now();
        let ts: Vec<&Tensor> = s.tensors.iter().filter(|t| is_dense(t)).collect();
        let total: usize = ts.iter().map(|t| (t.bytes as usize).div_ceil(64) * 64).sum();
        let arena = Arena::new(total, false).map_err(|e| e.to_string())?;
        let files: Vec<File> = s.paths.iter().map(|p| File::open(p).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
        let mut mats = HashMap::new();
        let mut off = 0usize;
        let mut tmp = Vec::new();
        for t in ts {
            let n = t.bytes as usize;
            // SAFETY: inside the arena (sized above).
            let dst = unsafe { std::slice::from_raw_parts_mut(arena.ptr.add(off), n) };
            let k = t.shape[0] as usize;
            let rows = t.shape.get(1).copied().unwrap_or(1) as usize;
            let repack = mode == DenseMode::Repack
                && t.shape.len() == 2
                && t.name != "token_embd.weight"
                && ((t.w == W::Q6K && rows % 8 == 0) || (t.w == W::Q8_0 && rows % 4 == 0));
            let f = &files[t.shard];
            if repack {
                tmp.resize(n, 0);
                f.read_exact_at(&mut tmp, t.off).map_err(|e| format!("{}: {e}", t.name))?;
                let qt = if t.w == W::Q6K { QType::Q6K } else { QType::Q8_0 };
                kernels_sys::repack(qt, &tmp, dst, rows, k);
            } else {
                f.read_exact_at(dst, t.off).map_err(|e| format!("{}: {e}", t.name))?;
            }
            // SAFETY: advisory syscall on our fd.
            unsafe { libc::posix_fadvise(f.as_raw_fd(), t.off as i64, n as i64, libc::POSIX_FADV_DONTNEED) };
            mats.insert(t.name.clone(), Mat { p: dst.as_ptr(), w: t.w, k, n: rows, repacked: repack });
            off += n.div_ceil(64) * 64;
        }
        Ok(Dense { arena_bytes: arena.len, _arena: arena, mats, load_seconds: t0.elapsed().as_secs_f64() })
    }

    /// The arena every dense tensor lives in (base, bytes).
    pub fn region(&self) -> (*mut u8, usize) {
        (self._arena.ptr, self._arena.len)
    }

    pub fn get(&self, name: &str) -> Mat {
        *self.mats.get(name).unwrap_or_else(|| panic!("dense tensor {name} missing"))
    }
    pub fn opt(&self, name: &str) -> Option<Mat> {
        self.mats.get(name).copied()
    }
    pub fn f32(&self, name: &str) -> *const f32 {
        let m = self.get(name);
        assert_eq!(m.w, W::F32, "{name}");
        m.p as *const f32
    }
}
