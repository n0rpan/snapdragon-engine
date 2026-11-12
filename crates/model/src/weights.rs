//! Weight store: every tensor the plan reads, in one arena laid out by `arena_layout`.
//!
//! - Quantised matmul weights (Q3_K/Q4_K/Q5_K/Q6_K 8-row, Q8_0 4-row) are repacked into ggml's
//!   interleaved layouts, per expert for 3-D tensors (`[expert][row-group][block]`).
//! - F32 tensors are copied, the token embedding is copied row-major (raw quantised rows).
//!
//! Backing (M2 item 4): the arena is a FILE. On the first load of a model the repacked arena
//! is written once to `~/.cache/local-ai/snapdragon-engine/<model-sha256>-<engine-id>.bin`
//! (temp file + fsync + rename, then a full checksum verification read from disk); later
//! loads map that file read-only, check its header and a random sample of tensor checksums,
//! and never touch the GGUF again. The source GGUF's pages are released (madvise + fadvise
//! DONTNEED) as each tensor is repacked. `Backing::Anon` copies the cache file into an
//! anonymous THP-advised arena instead (the M0 configuration, for A/B speed runs).

use crate::desc::qtype;
use crate::plan::{Plan, Tid};
use gguf::{GgmlType, Gguf, TensorInfo};
use kernels_sys::QType;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::time::Instant;

/// Bump when the arena layout or any repack format changes.
pub const LAYOUT_VERSION: &str = "w1";
/// llama.cpp commit the vendored kernels/repack come from.
pub const KERNEL_COMMIT: &str = "ee6b9a1";
const MAGIC: &[u8; 8] = b"OXEWC001";
const HUGE: usize = 2 << 20;

/// Engine identity used in the cache key: layout version, kernel commit, build flags.
pub fn engine_id() -> String {
    format!("{LAYOUT_VERSION}-{KERNEL_COMMIT}-{:08x}", hash_bytes(kernels_sys::KERNEL_CFLAGS.as_bytes()) as u32)
}

// ------------------------------------------------------------------ mappings

/// An owned memory mapping (anonymous or file), unmapped on drop.
pub struct Mapping {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping is written only during load (disjoint ranges per item, one writer each)
// and is read-only during decode.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn anon(len: usize, hugepages: bool) -> std::io::Result<Mapping> {
        let len = len.max(1);
        // SAFETY: anonymous private mapping; checked for MAP_FAILED.
        let ptr = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        if hugepages {
            // SAFETY: advisory on our own mapping.
            unsafe { libc::madvise(ptr, len, libc::MADV_HUGEPAGE) };
        }
        Ok(Mapping { ptr: ptr as *mut u8, len })
    }
    fn file(f: &File, len: usize, writable: bool) -> std::io::Result<Mapping> {
        use std::os::unix::io::AsRawFd;
        let prot = if writable { libc::PROT_READ | libc::PROT_WRITE } else { libc::PROT_READ };
        // SAFETY: shared mapping of a file we own; checked for MAP_FAILED.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, f.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mapping { ptr: ptr as *mut u8, len })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn base(&self) -> *mut u8 {
        self.ptr
    }
    fn slice(&self, off: usize, len: usize) -> &[u8] {
        assert!(off + len <= self.len);
        // SAFETY: within the mapping.
        unsafe { std::slice::from_raw_parts(self.ptr.add(off), len) }
    }
    fn advise(&self, advice: i32) {
        // SAFETY: advisory on our own mapping.
        unsafe { libc::madvise(self.ptr as *mut libc::c_void, self.len, advice) };
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly what we mapped.
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}

/// Back-compat alias (M0 name).
pub type AnonArena = Mapping;

// ------------------------------------------------------------------ layout

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Row-major f32.
    F32,
    /// Interleaved, `[expert][row-group][block]`.
    Packed(QType),
    /// Row-major quantised rows (token embedding).
    Raw(QType),
}

impl Layout {
    fn tag(self) -> (u8, u8) {
        let q = |q: QType| QType::ALL.iter().position(|&x| x == q).unwrap() as u8;
        match self {
            Layout::F32 => (0, 0),
            Layout::Packed(t) => (1, q(t)),
            Layout::Raw(t) => (2, q(t)),
        }
    }
    fn from_tag(a: u8, b: u8) -> Option<Layout> {
        let q = *QType::ALL.get(b as usize)?;
        match a {
            0 => Some(Layout::F32),
            1 => Some(Layout::Packed(q)),
            2 => Some(Layout::Raw(q)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WeightRef {
    pub ptr: usize,
    pub layout: Layout,
    /// Row length (ggml ne0).
    pub k: usize,
    /// Rows per expert (ne1).
    pub rows: usize,
    /// Experts (ne2, 1 for 2-D).
    pub n_expert: usize,
    pub bytes: usize,
}

impl WeightRef {
    pub fn expert_bytes(&self) -> usize {
        self.bytes / self.n_expert
    }
    pub fn qtype(&self) -> QType {
        match self.layout {
            Layout::Packed(q) | Layout::Raw(q) => q,
            l => panic!("not a quantised layout: {l:?}"),
        }
    }
    pub fn is_f32(&self) -> bool {
        self.layout == Layout::F32
    }
    /// Pointer to the interleaved block of row `row` (multiple of the type's row group) of expert `e`.
    #[inline]
    pub fn rows_ptr(&self, e: usize, row: usize) -> *const u8 {
        let q = self.qtype();
        // Stays a debug_assert (N1 audit, deliberately): this runs per GEMM tile in prefill;
        // promoting it needs a timing window. Layout and k are checked at load (layout_for).
        debug_assert!(matches!(self.layout, Layout::Packed(_)) && row % q.nr() == 0);
        (self.ptr + e * self.expert_bytes() + q.group_offset(row, self.k)) as *const u8
    }
    #[inline]
    pub fn f32(&self) -> *const f32 {
        // N1 (audit): a real check; a wrong layout here reads quantised bytes as floats.
        assert!(self.layout == Layout::F32, "f32() on a {:?} tensor", self.layout);
        self.ptr as *const f32
    }
    /// F32 tensor as a slice (all experts).
    pub fn f32_slice(&self) -> &[f32] {
        assert!(self.layout == Layout::F32, "f32_slice() on a {:?} tensor", self.layout);
        // N1 (audit): from_raw_parts needs an aligned pointer and a whole number of floats.
        assert!(self.ptr % std::mem::align_of::<f32>() == 0, "F32 tensor at unaligned offset {:#x}", self.ptr);
        assert!(self.bytes % 4 == 0, "F32 tensor of {} bytes is not a whole number of floats", self.bytes);
        // SAFETY: the tensor's own read-only arena range, checked aligned and sized above.
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.bytes / 4) }
    }
}

fn layout_for(t: &TensorInfo, embd: bool) -> Result<Layout, String> {
    if t.ty == GgmlType::F32 {
        return Ok(Layout::F32);
    }
    let q = qtype(t.ty).ok_or_else(|| format!("{}: type {} has no kernel (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/F32)", t.name, t.ty.name()))?;
    if embd {
        if q == QType::Q3K {
            return Err(format!("{}: Q3_K embedding rows are not supported", t.name));
        }
        return Ok(Layout::Raw(q));
    }
    if t.shape.len() < 2 || t.shape[1] % q.nr() as u64 != 0 || t.shape[0] % q.qk() as u64 != 0 {
        return Err(format!("{}: shape {:?} cannot be {}-row repacked", t.name, t.shape, q.nr()));
    }
    Ok(Layout::Packed(q))
}

/// Geometry `k x rows x n_expert` of a tensor, checked against its byte size (REVIEW-v0.1.1 S1):
/// the kernels address rows by this shape, so a size that does not match it would read past
/// the tensor (a crafted file).
fn checked_geometry(t: &TensorInfo) -> Result<(usize, usize, usize, usize), String> {
    let bad = || format!("{}: shape {:?} does not match its {} data bytes", t.name, t.shape, t.bytes());
    let (_, blk, size) = t.ty.info().ok_or_else(bad)?;
    let k = t.shape.first().copied().unwrap_or(0);
    let rows = t.shape.get(1).copied().unwrap_or(1);
    let n_expert = t.shape.get(2).copied().unwrap_or(1);
    if k == 0 || rows == 0 || n_expert == 0 || k % blk != 0 || t.shape.iter().skip(3).any(|&d| d != 1) {
        return Err(bad());
    }
    let want = (k / blk).checked_mul(size).and_then(|r| r.checked_mul(rows)).and_then(|r| r.checked_mul(n_expert));
    let want = match want {
        Some(w) if Some(w) == t.checked_bytes() => w,
        _ => return Err(bad()),
    };
    let u = |x: u64| usize::try_from(x).map_err(|_| bad());
    Ok((u(k)?, u(rows)?, u(n_expert)?, u(want)?))
}

fn make_ref(t: &TensorInfo, embd: bool, ptr: usize) -> Result<WeightRef, String> {
    let (k, rows, n_expert, bytes) = checked_geometry(t)?;
    Ok(WeightRef { ptr, layout: layout_for(t, embd)?, k, rows, n_expert, bytes })
}

/// Arena offsets: tensors >= 2 MiB start on a 2 MiB boundary, smaller ones on 4 KiB.
/// Checked arithmetic: every `(off, bytes)` lies inside the returned total.
fn arena_layout(tensors: &[TensorInfo], ids: &[Tid]) -> Result<(Vec<(Tid, usize)>, usize), String> {
    let over = || "weight arena size overflows".to_string();
    let up = |x: usize, a: usize| x.checked_add(a - 1).map(|v| v / a * a);
    let mut off = 0usize;
    let mut out = Vec::new();
    for &id in ids {
        let b = usize::try_from(tensors[id].checked_bytes().ok_or_else(over)?).map_err(|_| over())?;
        let align = if b >= HUGE { HUGE } else { 4096 };
        off = up(off, align).ok_or_else(over)?;
        out.push((id, off));
        off = off.checked_add(b).ok_or_else(over)?;
    }
    Ok((out, up(off, HUGE).ok_or_else(over)?))
}

// ------------------------------------------------------------------ hashing

/// Fast non-cryptographic 64-bit checksum (4 independent lanes over u64 words).
pub fn hash_bytes(b: &[u8]) -> u64 {
    const K: [u64; 4] = [0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93];
    let mut h = [0x243F6A8885A308D3u64, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89];
    let words = b.len() / 32;
    for i in 0..words {
        for l in 0..4 {
            let w = u64::from_le_bytes(b[i * 32 + l * 8..i * 32 + l * 8 + 8].try_into().unwrap());
            h[l] = (h[l] ^ w).wrapping_mul(K[l]).rotate_left(31);
        }
    }
    let mut x = (b.len() as u64).wrapping_mul(K[0]);
    for &v in &b[words * 32..] {
        x = (x ^ v as u64).wrapping_mul(K[1]).rotate_left(17);
    }
    for (l, v) in h.iter().enumerate() {
        x = (x ^ v).wrapping_mul(K[l]).rotate_left(23);
    }
    x ^ (x >> 29)
}

fn combine(acc: u64, h: u64) -> u64 {
    (acc ^ h).wrapping_mul(0x9E3779B97F4A7C15).rotate_left(27)
}

// ------------------------------------------------------------------ cache file

#[derive(Clone, Debug)]
struct Entry {
    name: String,
    layout: Layout,
    k: u64,
    rows: u64,
    n_expert: u64,
    off: u64,
    bytes: u64,
    hash: u64,
}

#[derive(Clone, Debug)]
struct CacheHeader {
    engine_id: String,
    model_sha: String,
    model_bytes: u64,
    data_off: u64,
    total: u64,
    entries: Vec<Entry>,
}

struct W(Vec<u8>);
impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v)
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes())
    }
    fn s(&mut self, v: &str) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v.as_bytes())
    }
}
struct R<'a>(&'a [u8], usize);
impl R<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        if self.1 + n > self.0.len() {
            return Err("cache header truncated".into());
        }
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn s(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        if n > 1 << 20 {
            return Err("cache header string too long".into());
        }
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "bad utf8 in cache header".into())
    }
}

impl CacheHeader {
    fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.s(&self.engine_id);
        w.s(&self.model_sha);
        w.u64(self.model_bytes);
        w.u64(self.data_off);
        w.u64(self.total);
        w.u64(self.entries.len() as u64);
        for e in &self.entries {
            w.s(&e.name);
            let (a, b) = e.layout.tag();
            w.u8(a);
            w.u8(b);
            for v in [e.k, e.rows, e.n_expert, e.off, e.bytes, e.hash] {
                w.u64(v);
            }
        }
        let h = hash_bytes(&w.0);
        w.u64(h);
        let mut out = (w.0.len() as u64 + 8).to_le_bytes().to_vec();
        out.extend_from_slice(&w.0);
        out
    }
    fn decode(buf: &[u8]) -> Result<CacheHeader, String> {
        if buf.len() < 8 {
            return Err("cache header truncated".into());
        }
        let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        if n < 16 || n > buf.len() {
            return Err("cache header length invalid".into());
        }
        let body = &buf[8..n];
        let (data, hb) = body.split_at(body.len() - 8);
        if hash_bytes(data) != u64::from_le_bytes(hb.try_into().unwrap()) {
            return Err("cache header checksum mismatch".into());
        }
        let mut r = R(data, 0);
        if r.take(8)? != MAGIC {
            return Err("not an oxe weight cache (magic)".into());
        }
        let engine_id = r.s()?;
        let model_sha = r.s()?;
        let model_bytes = r.u64()?;
        let data_off = r.u64()?;
        let total = r.u64()?;
        let ne = r.u64()? as usize;
        let mut entries = Vec::with_capacity(ne);
        for _ in 0..ne {
            let name = r.s()?;
            let (a, b) = (r.u8()?, r.u8()?);
            let layout = Layout::from_tag(a, b).ok_or("bad layout tag")?;
            let mut v = [0u64; 6];
            for x in v.iter_mut() {
                *x = r.u64()?;
            }
            entries.push(Entry { name, layout, k: v[0], rows: v[1], n_expert: v[2], off: v[3], bytes: v[4], hash: v[5] });
        }
        Ok(CacheHeader { engine_id, model_sha, model_bytes, data_off, total, entries })
    }
}

/// The cache directory (created 0700 if missing; must be ours and not group/other-writable).
pub fn cache_dir(override_dir: Option<&Path>) -> Result<PathBuf, String> {
    let dir = match override_dir {
        Some(d) => d.to_path_buf(),
        None => {
            let base = std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
                .ok_or("no HOME")?;
            base.join("local-ai/snapdragon-engine")
        }
    };
    if !dir.exists() {
        std::fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).map_err(|e| e.to_string())?;
    }
    let md = std::fs::symlink_metadata(&dir).map_err(|e| e.to_string())?;
    // SAFETY: trivial libc call.
    let uid = unsafe { libc::getuid() };
    if !md.is_dir() || md.uid() != uid || md.mode() & 0o022 != 0 {
        return Err(format!("unsafe cache dir {} (must be a directory owned by uid {uid}, not group/other-writable)", dir.display()));
    }
    Ok(dir)
}

/// SHA-256 of the model file, computed once with `sha256sum` and remembered in
/// `<cache>/sha256-index.tsv` keyed by (dev, inode, size, mtime, ctime). `OXE_MODEL_SHA256`
/// overrides.
pub fn model_sha256(path: &Path, dir: &Path) -> Result<String, String> {
    model_sha256_opt(path, dir, true)
}

/// [`model_sha256`]; `from_env` false ignores `OXE_MODEL_SHA256` (a file that is not the model).
pub fn model_sha256_opt(path: &Path, dir: &Path, from_env: bool) -> Result<String, String> {
    if let Ok(s) = std::env::var("OXE_MODEL_SHA256").map_err(|_| ()).and_then(|s| if from_env { Ok(s) } else { Err(()) }) {
        if s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()) {
            return Ok(s.to_ascii_lowercase());
        }
        return Err("OXE_MODEL_SHA256 must be 64 hex digits".into());
    }
    let md = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let key = format!("{} {} {} {}.{} {}.{}", md.dev(), md.ino(), md.size(), md.mtime(), md.mtime_nsec(), md.ctime(), md.ctime_nsec());
    let index = dir.join("sha256-index.tsv");
    if let Ok(s) = std::fs::read_to_string(&index) {
        for line in s.lines() {
            if let Some((k, rest)) = line.split_once('\t') {
                if k == key {
                    if let Some(sha) = rest.split('\t').next() {
                        return Ok(sha.to_string());
                    }
                }
            }
        }
    }
    eprintln!("load: computing sha256 of {} (once; cached in {})", path.display(), index.display());
    let out = std::process::Command::new("sha256sum").arg(path).output().map_err(|e| format!("sha256sum: {e}"))?;
    if !out.status.success() {
        return Err("sha256sum failed".into());
    }
    let sha = String::from_utf8_lossy(&out.stdout).split_whitespace().next().unwrap_or("").to_string();
    if sha.len() != 64 {
        return Err("sha256sum: unexpected output".into());
    }
    let mut f = OpenOptions::new().create(true).append(true).mode(0o600).open(&index).map_err(|e| e.to_string())?;
    let _ = writeln!(f, "{key}\t{sha}\t{}", path.display());
    Ok(sha)
}

// ------------------------------------------------------------------ store

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// mmap the cache file read-only (default).
    File,
    /// copy the cache file into an anonymous THP-advised arena.
    Anon,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verify {
    /// header + a random sample of tensor checksums (default on a warm load).
    Sample,
    /// every tensor checksum.
    Full,
    None,
}

#[derive(Clone, Debug)]
pub struct LoadOptions {
    pub backing: Backing,
    pub verify: Verify,
    pub threads: usize,
    pub cache_dir: Option<PathBuf>,
    pub progress: bool,
    /// MADV_HUGEPAGE + MADV_COLLAPSE attempt on the file mapping (needs file THP support).
    pub thp_collapse: bool,
    /// mlock the arena after loading; failure is an error (never silently unlocked)
    pub mlock: bool,
    /// `OXE_MODEL_SHA256` names THIS file (the target model). False for any other file loaded with
    /// these options (a sidecar MTP head): its sha is computed from its own bytes, so its weight
    /// cache and the session identity carry the head's sha (W4-4 b).
    pub sha_from_env: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { backing: Backing::File, verify: Verify::Sample, threads: 8, cache_dir: None, progress: true, thp_collapse: false, mlock: false, sha_from_env: true }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LoadInfo {
    pub cache_path: String,
    pub cache_hit: bool,
    pub built_seconds: f64,
    pub verify_seconds: f64,
    pub verified_tensors: usize,
    pub backing: String,
    pub model_sha256: String,
}

pub struct WeightStore {
    pub refs: Vec<Option<WeightRef>>,
    pub arena: Mapping,
    /// Offset of the data region inside `arena` (header for file backing).
    pub data_off: usize,
    pub embd_row_bytes: usize,
    pub load_seconds: f64,
    pub repacked_bytes: u64,
    pub info: LoadInfo,
    // CPU allocations own native expert bytes independently of any DSP session.
    // No pointer is published until every selected tensor has been staged.
    shared_experts: Vec<(Tid, oxn::shared_memory::SharedMemory)>,
}

impl WeightStore {
    fn ids(plan: &Plan) -> Vec<Tid> {
        let mut ids: Vec<Tid> = plan.tensors();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Check that the kernel set covers every tensor of the plan.
    pub fn check_supported(plan: &Plan, tensors: &[TensorInfo]) -> Result<(), String> {
        for id in Self::ids(plan) {
            layout_for(&tensors[id], id == plan.embd)?;
        }
        Ok(())
    }

    /// Load the plan's tensors: from the cache file if present and valid, else build it
    /// from the GGUF first. The GGUF is closed (unmapped) before this returns.
    pub fn load(g: Gguf, plan: &Plan, opt: &LoadOptions) -> Result<WeightStore, String> {
        Self::check_supported(plan, &g.header.tensors)?;
        Self::load_ids(g, Self::ids(plan), Some(plan.embd), opt, "")
    }

    /// Load an explicit set of tensors (sorted, unique) into their own cache file
    /// `<sha>-<engine-id><tag>.bin` (M3.1: the MTP head uses tag "-mtp"). `embd` is the
    /// token-embedding tensor if it is among `ids` (raw rows), else None.
    pub fn load_ids(g: Gguf, ids: Vec<Tid>, embd: Option<Tid>, opt: &LoadOptions, tag: &str) -> Result<WeightStore, String> {
        let t0 = Instant::now();
        let tensors = g.header.tensors.clone();
        for &id in &ids {
            layout_for(&tensors[id], Some(id) == embd)?;
        }
        let dir = cache_dir(opt.cache_dir.as_deref())?;
        let sha = model_sha256_opt(&g.path, &dir, opt.sha_from_env)?;
        let eid = engine_id();
        let path = dir.join(format!("{sha}-{eid}{tag}.bin"));
        let mut info = LoadInfo { cache_path: path.display().to_string(), model_sha256: sha.clone(), ..Default::default() };
        let (offs, total) = arena_layout(&tensors, &ids)?;
        for &id in &ids {
            checked_geometry(&tensors[id])?;
        }

        let mut hdr = None;
        if path.exists() {
            match open_cache(&path, &eid, &sha, g.file_bytes) {
                Ok(h) => hdr = Some(h),
                Err(e) => {
                    eprintln!("load: cache {} unusable ({e}); rebuilding", path.display());
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        let built = hdr.is_none();
        let (map, hdr) = match hdr {
            Some((f, h)) => {
                drop(g); // warm load: the GGUF is not needed at all
                let len = (h.data_off + h.total) as usize;
                let m = Mapping::file(&f, len, false).map_err(|e| format!("mmap cache: {e}"))?;
                (m, h)
            }
            None => {
                let tb = Instant::now();
                let h = build_cache(&g, &tensors, &ids, embd, &offs, total, &path, &eid, &sha, opt)?;
                drop(g); // unmap the GGUF (its pages were already released per tensor)
                info.built_seconds = tb.elapsed().as_secs_f64();
                let f = open_ro_nofollow(&path)?;
                // drop the written pages so the verification reads what is on disk
                fadvise_dontneed(&f, 0, 0);
                let len = (h.data_off + h.total) as usize;
                let m = Mapping::file(&f, len, false).map_err(|e| format!("mmap cache: {e}"))?;
                (m, h)
            }
        };
        info.cache_hit = !built;
        // entries must match the plan's layout exactly (and the arena its size: every ref below
        // points into [base, base + total))
        if hdr.total != total as u64 {
            return Err("cache arena size does not match the plan".into());
        }
        if hdr.entries.len() != ids.len() {
            return Err("cache entry count does not match the plan".into());
        }
        for (e, &(id, off)) in hdr.entries.iter().zip(&offs) {
            let t = &tensors[id];
            let want = make_ref(t, Some(id) == embd, 0)?;
            if e.name != t.name || e.off as usize != off || e.bytes as usize != want.bytes || e.layout != want.layout || e.n_expert != want.n_expert as u64 {
                return Err(format!("cache entry {} does not match the model/plan", e.name));
            }
        }
        // verification
        let tv = Instant::now();
        let verify = if built { Verify::Full } else { opt.verify };
        let sel: Vec<usize> = match verify {
            Verify::Full => (0..hdr.entries.len()).collect(),
            Verify::None => vec![],
            Verify::Sample => {
                let mut x = (t0.elapsed().as_nanos() as u64 ^ 0x9E3779B97F4A7C15) | 1;
                let mut v: Vec<usize> = (0..8)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        (x % hdr.entries.len() as u64) as usize
                    })
                    .collect();
                v.sort_unstable();
                v.dedup();
                v
            }
        };
        verify_entries(&map, &hdr, &sel, opt.threads)?;
        info.verify_seconds = tv.elapsed().as_secs_f64();
        info.verified_tensors = sel.len();

        let (arena, data_off) = match opt.backing {
            Backing::File => {
                if opt.thp_collapse {
                    map.advise(libc::MADV_HUGEPAGE);
                    map.advise(25 /* MADV_COLLAPSE */);
                }
                info.backing = "file".into();
                (map, hdr.data_off as usize)
            }
            Backing::Anon => {
                // Chunked copy into a THP-advised anonymous arena (ENGINE-AUDIT F4): read the
                // cache file in chunks with pread and drop each chunk's page cache right after,
                // so the peak stays ~ the arena size instead of arena + whole file mapping
                // (27.68 GiB peak for Ornith with the old full-map copy).
                drop(map);
                let f = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                fadvise_dontneed(&f, 0, 0); // the verify sample's pages
                let a = Mapping::anon(hdr.total as usize, true).map_err(|e| format!("anon arena: {e}"))?;
                chunked_read_into(&f, hdr.data_off, a.base(), hdr.total as usize, opt.threads.clamp(1, 4))?;
                info.backing = "anon".into();
                (a, 0)
            }
        };
        if opt.mlock {
            lock_arena(&arena)?;
            info.backing += "+mlock";
        }
        let base = arena.base() as usize + data_off;
        let mut refs = vec![None; tensors.len()];
        for &(id, off) in &offs {
            refs[id] = Some(make_ref(&tensors[id], Some(id) == embd, base + off)?);
        }
        Ok(WeightStore {
            refs,
            arena,
            data_off,
            embd_row_bytes: embd.map(|e| (tensors[e].bytes() / tensors[e].shape[1]) as usize).unwrap_or(0),
            load_seconds: t0.elapsed().as_secs_f64(),
            repacked_bytes: total as u64,
            info,
            shared_experts: Vec::new(),
        })
    }

    /// Exact CPU-owned DMA bytes. Imported DSP views must NOT add these again.
    pub fn shared_expert_bytes(&self) -> usize {
        self.shared_experts.iter().map(|(_, m)| m.len()).sum()
    }

    /// Borrow a sealed native-layout tensor for a bounded DSP window. CPU decode
    /// retains the same object through this WeightStore even after DSP teardown.
    pub fn shared_expert(&self, id: Tid) -> Option<&oxn::shared_memory::SharedMemory> {
        self.shared_experts.binary_search_by_key(&id, |(i, _)| *i).ok().map(|i| &self.shared_experts[i].1)
    }

    /// Experimental storage change only: no DSP session or arithmetic change.
    /// Requires exclusive store access BEFORE constructing any Engine. Publish
    /// all replacement pointers together; on error the original refs stay valid.
    #[cfg(feature = "npu-device")]
    pub fn relocate_experts(&mut self, plan: &Plan, allocator: &oxn::shared_memory::SharedAllocator) -> Result<usize, String> {
        if self.info.backing != "file" || !self.shared_experts.is_empty() {
            return Err("shared experts require an unmodified file-backed store".into());
        }
        let mut ids: Vec<_> = plan.layers.iter().flat_map(|l| [l.gate_exps, l.up_exps, l.down_exps]).collect();
        ids.sort_unstable(); ids.dedup();
        // Hold the already-validated cache inode while advising away source pages.
        // fadvise is advisory and does not alter file contents or invalidate refs.
        let cache = open_ro_nofollow(Path::new(&self.info.cache_path))?;
        let staged = self.stage_shared_backings(&ids, |src| {
            let mut m = allocator.allocate(src.len())?;
            m.as_mut_bytes().copy_from_slice(src);
            Ok(m.seal())
        }, |ptr, bytes| {
            let page = 4096usize;
            let begin = (ptr + page - 1) & !(page - 1);
            let end = (ptr + bytes) & !(page - 1);
            if end > begin {
                // SAFETY: aligned inward, wholly inside the live file mapping.
                unsafe { libc::madvise(begin as *mut libc::c_void, end - begin, libc::MADV_DONTNEED); }
                if let (Ok(off), Ok(len)) = (i64::try_from(begin - self.arena.base() as usize), i64::try_from(end - begin)) {
                    fadvise_dontneed(&cache, off, len);
                }
            }
        })?;
        // Everything is validated and owned. No fallible action after publishing.
        for (id, memory) in &staged {
            self.refs[*id].as_mut().unwrap().ptr = memory.as_bytes().as_ptr() as usize;
        }
        self.shared_experts = staged;
        self.info.backing = "file+shared-experts".into();
        Ok(self.shared_expert_bytes())
    }

    #[cfg(any(feature = "npu-device", test))]
    fn stage_shared_backings<B: AsRef<[u8]>>(&self, ids: &[Tid], mut allocate_copy: impl FnMut(&[u8]) -> Result<B, String>, mut copied: impl FnMut(usize, usize)) -> Result<Vec<(Tid, B)>, String> {
        if ids.is_empty() || ids.windows(2).any(|p| p[0] >= p[1]) {
            return Err("shared expert IDs must be nonempty, unique and sorted".into());
        }
        let start = self.arena.base() as usize;
        let end = start.checked_add(self.arena.len()).ok_or("arena range overflow")?;
        // Preflight ALL bounds/types before allocating or releasing source pages.
        for &id in ids {
            let r = self.refs.get(id).and_then(Option::as_ref).ok_or("expert tensor not loaded")?;
            if r.n_expert < 2 || !matches!(r.layout, Layout::Packed(_)) || r.bytes == 0 || r.bytes % 4096 != 0 || r.bytes > 512 << 20
                || r.ptr < start || r.ptr.checked_add(r.bytes).is_none_or(|e| e > end) {
                return Err("shared expert layout/range/size is not supported".into());
            }
        }
        let mut staged = Vec::with_capacity(ids.len());
        for &id in ids {
            let r = self.refs[id].as_ref().unwrap();
            // SAFETY: checked file arena range; exclusive store, no running engine.
            let source = unsafe { std::slice::from_raw_parts(r.ptr as *const u8, r.bytes) };
            let owned = allocate_copy(source)?;
            if owned.as_ref() != source { return Err("native packed expert copy differs".into()); }
            staged.push((id, owned));
            copied(r.ptr, r.bytes);
        }
        Ok(staged)
    }

    #[inline]
    pub fn get(&self, id: Tid) -> &WeightRef {
        self.refs[id].as_ref().expect("tensor not loaded")
    }

    /// Bytes of embedding row `token`.
    pub fn embedding_row(&self, plan: &Plan, token: usize) -> &[u8] {
        let r = self.get(plan.embd);
        assert!(token < r.rows, "token id {token} out of range");
        // SAFETY: row `token` of the raw embedding inside the read-only arena.
        unsafe { std::slice::from_raw_parts((r.ptr + token * self.embd_row_bytes) as *const u8, self.embd_row_bytes) }
    }
}

fn open_ro_nofollow(path: &Path) -> Result<File, String> {
    let f = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let md = f.metadata().map_err(|e| e.to_string())?;
    // SAFETY: trivial libc call.
    let uid = unsafe { libc::getuid() };
    if !md.is_file() || md.uid() != uid || md.mode() & 0o022 != 0 {
        return Err(format!("unsafe cache file {}", path.display()));
    }
    Ok(f)
}

/// Read `len` bytes of `f` from `file_off` into `dst` (an anonymous mapping of at least `len`
/// bytes) in 64 MiB chunks on `threads` threads, dropping each chunk's page cache after the copy.
fn chunked_read_into(f: &File, file_off: u64, dst: *mut u8, len: usize, threads: usize) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    const CHUNK: usize = 64 << 20;
    let n = len.div_ceil(CHUNK);
    let next = AtomicUsize::new(0);
    let failed = std::sync::Mutex::new(None::<String>);
    let base = dst as usize;
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= n {
                    break;
                }
                let (o, l) = (i * CHUNK, CHUNK.min(len - i * CHUNK));
                // SAFETY: chunk i of the destination mapping; chunks are disjoint and each is
                // written by exactly one thread.
                let buf = unsafe { std::slice::from_raw_parts_mut((base + o) as *mut u8, l) };
                if let Err(e) = f.read_exact_at(buf, file_off + o as u64) {
                    *failed.lock().unwrap() = Some(format!("cache read at {}: {e}", file_off + o as u64));
                    break;
                }
                fadvise_dontneed(f, (file_off + o as u64) as i64, l as i64);
            });
        }
    });
    match failed.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// mlock the whole arena, or explain why it cannot be (RLIMIT_MEMLOCK).
fn lock_arena(m: &Mapping) -> Result<(), String> {
    // SAFETY: locking our own mapping.
    if unsafe { libc::mlock(m.base() as *const libc::c_void, m.len()) } == 0 {
        return Ok(());
    }
    let e = std::io::Error::last_os_error();
    let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: plain getrlimit into a local.
    unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) };
    Err(format!(
        "--mlock: mlock of {:.2} GiB failed: {e} (RLIMIT_MEMLOCK soft {} / hard {} bytes); raise it (systemd LimitMEMLOCK=infinity, or ulimit -l unlimited) or run without --mlock",
        m.len() as f64 / (1u64 << 30) as f64,
        rl.rlim_cur,
        rl.rlim_max
    ))
}

fn fadvise_dontneed(f: &File, off: i64, len: i64) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: advisory syscall on our own fd.
    unsafe { libc::posix_fadvise(f.as_raw_fd(), off, len, libc::POSIX_FADV_DONTNEED) };
}

fn open_cache(path: &Path, eid: &str, sha: &str, model_bytes: u64) -> Result<(File, CacheHeader), String> {
    let mut f = open_ro_nofollow(path)?;
    let mut lenb = [0u8; 8];
    f.read_exact(&mut lenb).map_err(|e| e.to_string())?;
    let n = u64::from_le_bytes(lenb) as usize;
    if !(16..=64 << 20).contains(&n) {
        return Err("cache header length invalid".into());
    }
    let mut buf = vec![0u8; n];
    buf[..8].copy_from_slice(&lenb);
    f.read_exact(&mut buf[8..]).map_err(|e| e.to_string())?;
    let h = CacheHeader::decode(&buf)?;
    if h.engine_id != eid {
        return Err(format!("engine id {} != {eid}", h.engine_id));
    }
    if h.model_sha != sha || h.model_bytes != model_bytes {
        return Err("model identity mismatch".into());
    }
    let flen = f.metadata().map_err(|e| e.to_string())?.len();
    if h.data_off.checked_add(h.total) != Some(flen) {
        return Err(format!("cache file is {flen} bytes, header says {} + {}", h.data_off, h.total));
    }
    Ok((f, h))
}

fn verify_entries(map: &Mapping, h: &CacheHeader, sel: &[usize], threads: usize) -> Result<(), String> {
    let next = AtomicUsize::new(0);
    let bad = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, 16) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= sel.len() {
                    break;
                }
                let e = &h.entries[sel[i]];
                let eb = (e.bytes / e.n_expert) as usize;
                let mut acc = 0u64;
                for x in 0..e.n_expert as usize {
                    acc = combine(acc, hash_bytes(map.slice(h.data_off as usize + e.off as usize + x * eb, eb)));
                }
                if acc != e.hash {
                    bad.lock().unwrap().push(e.name.clone());
                }
            });
        }
    });
    let bad = bad.into_inner().unwrap();
    if !bad.is_empty() {
        return Err(format!("weight cache checksum mismatch in {} tensor(s), e.g. {}; delete the cache file", bad.len(), bad[0]));
    }
    Ok(())
}


#[allow(clippy::too_many_arguments)]
fn build_cache(
    g: &Gguf,
    tensors: &[TensorInfo],
    ids: &[Tid],
    embd_id: Option<Tid>,
    offs: &[(Tid, usize)],
    total: usize,
    path: &Path,
    eid: &str,
    sha: &str,
    opt: &LoadOptions,
) -> Result<CacheHeader, String> {
    let t0 = Instant::now();
    let _ = ids;
    // header size bound: fixed part + entries
    let mut entries: Vec<Entry> = offs
        .iter()
        .map(|&(id, off)| {
            let t = &tensors[id];
            let r = make_ref(t, Some(id) == embd_id, 0)?;
            Ok(Entry { name: t.name.clone(), layout: r.layout, k: r.k as u64, rows: r.rows as u64, n_expert: r.n_expert as u64, off: off as u64, bytes: r.bytes as u64, hash: 0 })
        })
        .collect::<Result<_, String>>()?;
    let probe = CacheHeader { engine_id: eid.into(), model_sha: sha.into(), model_bytes: g.file_bytes, data_off: 0, total: total as u64, entries: entries.clone() };
    let data_off = (probe.encode().len() + 4096).div_ceil(HUGE) * HUGE;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&tmp)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    let cleanup = |e: String| {
        let _ = std::fs::remove_file(&tmp);
        e
    };
    f.set_len((data_off + total) as u64).map_err(|e| cleanup(format!("ftruncate: {e}")))?;
    let map = Mapping::file(&f, data_off + total, true).map_err(|e| cleanup(format!("mmap tmp cache: {e}")))?;
    if opt.progress {
        eprintln!("load: building weight cache {} ({:.2} GiB) with {} threads", path.display(), total as f64 / (1u64 << 30) as f64, opt.threads);
    }
    // work items: (entry index, expert)
    let mut items = Vec::new();
    for (ei, e) in entries.iter().enumerate() {
        for x in 0..e.n_expert as usize {
            items.push((ei, x));
        }
    }
    let hashes: Vec<Vec<AtomicU64>> = entries.iter().map(|e| (0..e.n_expert).map(|_| AtomicU64::new(0)).collect()).collect();
    let remaining: Vec<AtomicUsize> = entries.iter().map(|e| AtomicUsize::new(e.n_expert as usize)).collect();
    let next = AtomicUsize::new(0);
    let done = AtomicU64::new(0);
    let base = map.base() as usize + data_off;
    let entries_ro = &entries;
    std::thread::scope(|s| {
        for _ in 0..opt.threads.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= items.len() {
                    break;
                }
                let (ei, x) = items[i];
                let e = &entries_ro[ei];
                let t = &tensors[offs[ei].0];
                let eb = (e.bytes / e.n_expert) as usize;
                let src = &g.tensor_data(t)[x * eb..(x + 1) * eb];
                // arena_layout keeps every (off, bytes) inside total; a cheap release assert
                assert!(e.off as usize + (x + 1) * eb <= total, "weight cache item outside the arena");
                // SAFETY: each (tensor, expert) item owns a disjoint range of the mapping, inside
                // [data_off, data_off + total) by the assert above.
                let dst = unsafe { std::slice::from_raw_parts_mut((base + e.off as usize + x * eb) as *mut u8, eb) };
                match e.layout {
                    Layout::F32 | Layout::Raw(_) => dst.copy_from_slice(src),
                    Layout::Packed(q) => kernels_sys::repack(q, src, dst, e.rows as usize, e.k as usize),
                }
                hashes[ei][x].store(hash_bytes(dst), Relaxed);
                g.release_range(t, x * eb, eb);
                if remaining[ei].fetch_sub(1, AcqRel) == 1 {
                    g.drop_cache(t);
                }
                let before = done.fetch_add(eb as u64, Relaxed);
                if opt.progress && (before >> 30) != ((before + eb as u64) >> 30) {
                    eprintln!(
                        "load: {:.1}/{:.1} GiB repacked, {:.0} s",
                        (before + eb as u64) as f64 / (1u64 << 30) as f64,
                        total as f64 / (1u64 << 30) as f64,
                        t0.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    for (ei, e) in entries.iter_mut().enumerate() {
        e.hash = hashes[ei].iter().fold(0u64, |acc, h| combine(acc, h.load(Relaxed)));
    }
    let h = CacheHeader { engine_id: eid.into(), model_sha: sha.into(), model_bytes: g.file_bytes, data_off: data_off as u64, total: total as u64, entries };
    let enc = h.encode();
    assert!(enc.len() <= data_off);
    // SAFETY: the header region [0, data_off) of our own writable mapping.
    unsafe { std::ptr::copy_nonoverlapping(enc.as_ptr(), map.base(), enc.len()) };
    // SAFETY: msync of our own mapping.
    let rc = unsafe { libc::msync(map.base() as *mut libc::c_void, map.len(), libc::MS_SYNC) };
    if rc != 0 {
        return Err(cleanup(format!("msync: {}", std::io::Error::last_os_error())));
    }
    drop(map);
    f.sync_all().map_err(|e| cleanup(format!("fsync: {e}")))?;
    std::fs::rename(&tmp, path).map_err(|e| cleanup(format!("rename: {e}")))?;
    if let Some(d) = path.parent() {
        if let Ok(df) = File::open(d) {
            let _ = df.sync_all();
        }
    }
    if opt.progress {
        eprintln!("load: weight cache written in {:.1} s", t0.elapsed().as_secs_f64());
    }
    Ok(h)
}

#[cfg(test)]
mod s1_tests {
    use super::*;

    fn t(shape: &[u64], ty: GgmlType) -> TensorInfo {
        TensorInfo { name: "w".into(), shape: shape.to_vec(), ty, offset: 0 }
    }

    /// REVIEW-v0.1.1 S1: the refs the kernels address by shape must have exactly that many bytes,
    /// and the arena layout must not wrap.
    #[test]
    fn geometry_and_arena_are_checked() {
        // normal 2-D / 3-D tensors pass
        let r = make_ref(&t(&[256, 8], GgmlType::Q4_K), false, 0).unwrap();
        assert_eq!((r.k, r.rows, r.n_expert, r.bytes), (256, 8, 1, 8 * 144));
        let r = make_ref(&t(&[256, 8, 4], GgmlType::Q8_0), false, 0).unwrap();
        assert_eq!(r.bytes, 4 * 8 * (256 / 32) * 34);
        assert!(make_ref(&t(&[64], GgmlType::F32), false, 0).is_ok());
        // a 4th dimension the kernels ignore would make the data larger than the shape they read
        assert!(make_ref(&t(&[64, 1, 1, 2], GgmlType::F32), false, 0).is_err());
        // zero experts (divide by zero in the cache build) and overflowing shapes
        assert!(make_ref(&t(&[64, 8, 0], GgmlType::F32), false, 0).is_err());
        assert!(make_ref(&t(&[1 << 40, 1 << 40], GgmlType::F32), false, 0).is_err());
        // arena: sizes that wrap usize are an error, not a small arena
        let ts = vec![t(&[1 << 61], GgmlType::F32), t(&[1 << 61], GgmlType::F32)];
        assert!(arena_layout(&ts, &[0, 1]).is_err());
        let ts = vec![t(&[1024], GgmlType::F32), t(&[(HUGE / 4) as u64], GgmlType::F32)];
        let (offs, total) = arena_layout(&ts, &[0, 1]).unwrap();
        for (id, off) in offs {
            assert!(off + ts[id].bytes() as usize <= total);
        }
    }
    /// The real model headers (header only, no weights read; skipped when absent) still pass.
    #[test]
    fn real_model_headers_pass_the_geometry_check() {
        let dir = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/ai-stack/models/llm");
        for f in ["Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf", "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf", "Qwen3.8-27B-Q4_K_M.gguf"] {
            let p = dir.join(f);
            if !p.is_file() {
                eprintln!("skip: {f}");
                continue;
            }
            let g = Gguf::open(&p).unwrap();
            for t in &g.header.tensors {
                checked_geometry(t).unwrap_or_else(|e| panic!("{f}: {e}"));
            }
            let ids: Vec<Tid> = (0..g.header.tensors.len()).collect();
            arena_layout(&g.header.tensors, &ids).unwrap();
        }
    }
}

#[cfg(test)]
mod shared_expert_tests {
    use super::*;
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    const BYTES: usize = 36 * 4096; // valid Q4_K [256,512,2], not fake geometry
    fn store() -> WeightStore {
        let arena = Mapping::anon(2*BYTES,false).unwrap();
        unsafe { std::slice::from_raw_parts_mut(arena.base(),2*BYTES).iter_mut().enumerate().for_each(|(i,v)|*v=(i%251)as u8); }
        let refs = (0..2).map(|i|Some(WeightRef{ptr:arena.base()as usize+i*BYTES,layout:Layout::Packed(QType::Q4K),k:256,rows:512,n_expert:2,bytes:BYTES})).collect();
        WeightStore{refs,arena,data_off:0,embd_row_bytes:0,load_seconds:0.,repacked_bytes:(2*BYTES)as u64,info:LoadInfo::default(),shared_experts:vec![]}
    }
    struct Owned { data:Vec<u8>, live:Arc<AtomicUsize> }
    impl Owned { fn new(src:&[u8],live:&Arc<AtomicUsize>)->Self{live.fetch_add(1,Ordering::Relaxed);Self{data:src.to_vec(),live:live.clone()}} }
    impl AsRef<[u8]> for Owned { fn as_ref(&self)->&[u8]{&self.data} }
    impl Drop for Owned {fn drop(&mut self){self.live.fetch_sub(1,Ordering::Relaxed);}}
    #[test]
    fn allocation_failure_rolls_back_all_staging_without_publishing_pointers() {
        let ws=store();let old:Vec<_>=ws.refs.iter().map(|r|r.unwrap().ptr).collect();
        let live=Arc::new(AtomicUsize::new(0));let mut calls=0;let mut advised=0;
        let got=ws.stage_shared_backings(&[0,1],|src|{calls+=1;if calls==2{Err("injected allocation failure".into())}else{Ok(Owned::new(src,&live))}},|_,_|advised+=1);
        assert!(got.is_err());assert_eq!(calls,2);assert_eq!(advised,1);assert_eq!(live.load(Ordering::Relaxed),0);
        assert_eq!(old,ws.refs.iter().map(|r|r.unwrap().ptr).collect::<Vec<_>>());assert_eq!(ws.shared_expert_bytes(),0);
        assert_eq!(ws.arena.slice(0,2*BYTES),(0..2*BYTES).map(|i|(i%251)as u8).collect::<Vec<_>>());
    }
    #[test]
    fn all_ranges_preflight_before_first_allocation_or_advice() {
        for ids in [vec![],vec![0,0],vec![1,0],vec![0,2]]{
            let ws=store();let mut calls=0;
            let got=ws.stage_shared_backings(&ids,|src|{calls+=1;Ok(src.to_vec())},|_,_|panic!("advice before preflight"));
            assert!(got.is_err());assert_eq!(calls,0);
        }
        let mut ws=store();ws.refs[1].as_mut().unwrap().ptr=usize::MAX-4;
        let got=ws.stage_shared_backings(&[0,1],|_:&[u8]|->Result<Vec<u8>,String>{panic!("allocation before range preflight")},|_,_|panic!("advice before preflight"));
        assert!(got.is_err());
    }
    #[test]
    fn wrong_copy_is_rejected_and_success_preserves_exact_native_bytes() {
        let ws=store();let live=Arc::new(AtomicUsize::new(0));let mut advised=0;
        let bad=ws.stage_shared_backings(&[0,1],|src|{let mut o=Owned::new(src,&live);o.data[9]^=1;Ok(o)},|_,_|advised+=1);
        assert!(bad.is_err());assert_eq!(advised,0);assert_eq!(live.load(Ordering::Relaxed),0);
        let good=ws.stage_shared_backings(&[0,1],|src|Ok(Owned::new(src,&live)),|_,bytes|{assert_eq!(bytes,BYTES);advised+=1;}).unwrap();
        assert_eq!(advised,2);assert_eq!(live.load(Ordering::Relaxed),2);
        for (id,m) in &good{assert_eq!(m.as_ref(),ws.arena.slice(id*BYTES,BYTES));}
        drop(good);assert_eq!(live.load(Ordering::Relaxed),0);
    }
}

#[cfg(test)]
mod real_native_reader_tests {
    use super::*;
    #[test]
    #[ignore = "explicit real-weight proof; requires canonical inference lease"]
    fn native_expert_tiles_match_real_original_gguf() {
        let home=PathBuf::from(std::env::var("HOME").unwrap());
        assert!(!crate::load::lease_free(&home.join(".local/state/local-ai/inference.lock")),"hold canonical inference lease before touching real weights");
        let path=home.join(".local/share/ai-stack/models/llm/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf");
        let g=Gguf::open(&path).unwrap();let mut checked=0;
        for (name,ty) in [("blk.0.ffn_gate_exps.weight",QType::Q4K),("blk.5.ffn_down_exps.weight",QType::Q3K)]{
            let tensor=g.header.tensors.iter().find(|t|t.name==name).unwrap();
            assert_eq!(qtype(tensor.ty),Some(ty));assert_eq!(tensor.shape.len(),3);
            let (k,rows,experts)=(tensor.shape[0]as usize,tensor.shape[1]as usize,tensor.shape[2]as usize);
            assert_eq!(experts,256);let raw=g.tensor_data(tensor);let rb=ty.row_bytes(k);let bb=ty.block_bytes();
            for e in [0,1,127,255]{for row0 in [0,rows-32]{for kb in [0,k/512,k/256-1]{
                let mut group=vec![0u8;32*bb];
                for r in 0..32{let at=(e*rows+row0+r)*rb+kb*bb;group[r*bb..(r+1)*bb].copy_from_slice(&raw[at..at+bb]);}
                let mut packed=vec![0u8;group.len()];kernels_sys::repack(ty,&group,&mut packed,32,256);
                let mut floats=vec![0f32;32*256];
                for r in 0..32{kernels_sys::dequantize_row(ty,&group[r*bb..(r+1)*bb],&mut floats[r*256..(r+1)*256]);}
                let mut expected=vec![0u16;floats.len()];kernels_sys::f32_to_f16(&floats,&mut expected);
                for c0 in (0..256).step_by(32){
                    let mut tile=[0u16;1024];kernels_sys::native_expert_tile(ty,&packed,256,32,1,0,0,c0,&mut tile).unwrap();
                    for r in 0..32{for c in 0..32{assert_eq!(tile[(c/2)*64+2*r+c%2],expected[r*256+c0+c],"{name} expert{e} row{} k{}",row0+r,kb*256+c0+c);checked+=1;}}
                }
            }}}
        }
        assert_eq!(checked,393216);println!("W2b native reader real-GGUF fp16 elements checked={checked}, all exact, no model graph/DSP session");
    }
}

#[cfg(test)]
mod n1_tests {
    use super::*;

    fn f32_ref(ptr: usize, bytes: usize) -> WeightRef {
        WeightRef { ptr, layout: Layout::F32, k: bytes / 4, rows: 1, n_expert: 1, bytes }
    }

    #[test]
    fn f32_slice_reads_an_aligned_whole_tensor() {
        let v = [1.0f32, 2.0, 3.0];
        assert_eq!(f32_ref(v.as_ptr() as usize, 12).f32_slice(), &v);
    }

    #[test]
    #[should_panic(expected = "unaligned")]
    fn f32_slice_refuses_an_unaligned_pointer() {
        let v = [0u32; 4];
        let _ = f32_ref(v.as_ptr() as usize + 1, 8).f32_slice();
    }

    #[test]
    #[should_panic(expected = "whole number of floats")]
    fn f32_slice_refuses_a_partial_float() {
        let v = [0u32; 4];
        let _ = f32_ref(v.as_ptr() as usize, 7).f32_slice();
    }

    #[test]
    #[should_panic(expected = "f32() on a")]
    fn f32_pointer_refuses_a_quantised_layout() {
        let w = WeightRef { ptr: 0, layout: Layout::Raw(QType::Q8_0), k: 32, rows: 1, n_expert: 1, bytes: 34 };
        let _ = w.f32();
    }
}
