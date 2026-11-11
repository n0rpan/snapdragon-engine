//! One-time repacked expert sidecar: one 4 KiB-aligned record per (layer, expert), laid out
//! `gate | up | down`, gate/up already in the R4 layout when the type has one (IQ2_S, IQ3_XXS),
//! so a cache miss is ONE aligned O_DIRECT read straight into the cache slot, with no CPU
//! transform and no bounce buffer. Record stride = slot size = record rounded up to 4 KiB.
//! A JSON manifest next to the file records the layout and the source shard identities.

use crate::ffi::{self, W};
use crate::model::{expert_geom, ExpertGeom, Shards, FF_EXP, N_EMBD, N_EXPERT, N_LAYER};
use crate::uring::Arena;
use serde_json::json;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const SIDECAR_VERSION: u64 = 1;

#[derive(Clone, Debug)]
pub struct Layout {
    pub geom: Vec<ExpertGeom>,
    /// byte offset of (layer, expert 0)
    pub base: Vec<u64>,
    pub total: u64,
    pub r4: bool,
}

impl Layout {
    pub fn new(s: &Shards, r4: bool) -> Layout {
        let geom: Vec<ExpertGeom> = (0..N_LAYER).map(|l| expert_geom(s, l)).collect();
        let mut base = Vec::with_capacity(N_LAYER);
        let mut off = 0u64;
        for g in &geom {
            base.push(off);
            off += (g.slot * N_EXPERT) as u64;
        }
        Layout { geom, base, total: off, r4 }
    }
    pub fn offset(&self, l: usize, e: usize) -> u64 {
        self.base[l] + (e * self.geom[l].slot) as u64
    }
}

fn shard_identity(s: &Shards) -> serde_json::Value {
    json!(s
        .paths
        .iter()
        .map(|p| {
            let m = std::fs::metadata(p).ok();
            json!({
                "name": p.file_name().map(|n| n.to_string_lossy().to_string()),
                "bytes": m.as_ref().map(|m| m.len()),
                "mtime": m.and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs()),
            })
        })
        .collect::<Vec<_>>())
}

/// The unencrypted copy of the expert sidecar (O-17 / F-IO: 105.1 -> 90.5 ms/token vs the LUKS copy).
pub const PLAIN_SIDECAR: &str = "/var/lib/local-ai-models/flash/experts-r4.bin";

/// Default expert sidecar: the plain copy when it and its manifest exist, else ~/.cache/... (the LUKS volume).
/// `check` still verifies the manifest's identity against the model, so a stale copy is refused, not used.
pub fn default_path() -> PathBuf {
    let plain = PathBuf::from(PLAIN_SIDECAR);
    if plain.is_file() && manifest_path(&plain).is_file() {
        return plain;
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/local-ai/snapdragon-engine/flash/experts-r4.bin")
}

pub fn manifest_path(out: &Path) -> PathBuf {
    let mut p = out.as_os_str().to_owned();
    p.push(".json");
    PathBuf::from(p)
}

fn manifest(s: &Shards, lay: &Layout) -> serde_json::Value {
    json!({
        "version": SIDECAR_VERSION,
        "kind": "qwen4exp expert sidecar (gate|up|down per record, 4 KiB stride)",
        "r4": lay.r4,
        "total_bytes": lay.total,
        "shards": shard_identity(s),
        "layers": (0..N_LAYER).map(|l| {
            let g = &lay.geom[l];
            json!({"layer": l, "gate_type": format!("{:?}", g.gate_w), "down_type": format!("{:?}", g.down_w),
                   "gate_bytes": g.gate_bytes, "down_bytes": g.down_bytes, "record": g.record, "slot": g.slot, "base": lay.base[l]})
        }).collect::<Vec<_>>(),
    })
}

/// Check an existing sidecar's manifest against the shards and the expected layout.
pub fn check(s: &Shards, lay: &Layout, path: &Path) -> Result<(), String> {
    let m: serde_json::Value = serde_json::from_slice(
        &std::fs::read(manifest_path(path)).map_err(|e| format!("{}: {e}", manifest_path(path).display()))?,
    )
    .map_err(|e| e.to_string())?;
    let want = manifest(s, lay);
    for k in ["version", "r4", "total_bytes", "shards", "layers"] {
        if m[k] != want[k] {
            return Err(format!("sidecar manifest field {k} differs from the current shards/layout; rebuild it"));
        }
    }
    let len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len != lay.total {
        return Err(format!("sidecar is {len} bytes, layout says {}", lay.total));
    }
    Ok(())
}

/// Build one record in `dst` (slot bytes, zero padded) from the three source slices.
pub fn build_record(g: &ExpertGeom, r4: bool, gate: &[u8], up: &[u8], down: &[u8], dst: &mut [u8]) {
    dst[..g.gate_bytes].copy_from_slice(gate);
    dst[g.gate_bytes..2 * g.gate_bytes].copy_from_slice(up);
    dst[2 * g.gate_bytes..g.record].copy_from_slice(down);
    dst[g.record..g.slot].fill(0);
    if r4 {
        if let Some(t) = g.gate_w.r4() {
            for part in 0..2 {
                let p = dst[part * g.gate_bytes..].as_mut_ptr();
                // SAFETY: the part holds FF_EXP rows of N_EMBD weights.
                let ok = unsafe { ffi::ggml_cpu_expert_r4_slice_to_r4(t, p, N_EMBD as i64, FF_EXP as i64) };
                assert!(ok, "R4 transform refused");
            }
        }
    }
}

const CHUNK: usize = 32; // experts per source read

/// Write the sidecar to `out` (via `out.partial`, renamed at the end). Heavy I/O: hold the lease.
pub fn build(s: &Shards, lay: &Layout, out: &Path) -> Result<serde_json::Value, String> {
    let t0 = Instant::now();
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let mut partial = out.as_os_str().to_owned();
    partial.push(".partial");
    let partial = PathBuf::from(partial);
    let wf = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(&partial)
            .map_err(|e| format!("{}: {e}", partial.display()))?
    };
    // preallocate so extents are contiguous where possible
    // SAFETY: plain syscall on our fd.
    unsafe { libc::posix_fallocate(wf.as_raw_fd(), 0, lay.total as i64) };
    let files: Vec<File> = s.paths.iter().map(|p| File::open(p).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let max_slot = lay.geom.iter().map(|g| g.slot).max().unwrap();
    let wbuf = Arena::new(CHUNK * max_slot, false).map_err(|e| e.to_string())?;
    let mut src = [Vec::new(), Vec::new(), Vec::new()];
    let mut written = 0u64;
    for l in 0..N_LAYER {
        let g = lay.geom[l];
        let ts = ["gate", "up", "down"].map(|p| s.req(&format!("blk.{l}.ffn_{p}_exps.weight")).clone());
        for e0 in (0..N_EXPERT).step_by(CHUNK) {
            let n = CHUNK.min(N_EXPERT - e0);
            for (i, t) in ts.iter().enumerate() {
                let per = if i == 2 { g.down_bytes } else { g.gate_bytes };
                src[i].resize(n * per, 0);
                let off = t.off + (e0 * per) as u64;
                files[t.shard].read_exact_at(&mut src[i], off).map_err(|e| format!("{}: {e}", t.name))?;
                // drop the source pages again (cgroup page cache)
                // SAFETY: advisory syscall on our fd.
                unsafe { libc::posix_fadvise(files[t.shard].as_raw_fd(), off as i64, src[i].len() as i64, libc::POSIX_FADV_DONTNEED) };
            }
            // SAFETY: wbuf holds CHUNK * max_slot bytes.
            let dst = unsafe { std::slice::from_raw_parts_mut(wbuf.ptr, n * g.slot) };
            for j in 0..n {
                build_record(
                    &g,
                    lay.r4,
                    &src[0][j * g.gate_bytes..(j + 1) * g.gate_bytes],
                    &src[1][j * g.gate_bytes..(j + 1) * g.gate_bytes],
                    &src[2][j * g.down_bytes..(j + 1) * g.down_bytes],
                    &mut dst[j * g.slot..(j + 1) * g.slot],
                );
            }
            wf.write_all_at(dst, lay.offset(l, e0)).map_err(|e| format!("write: {e}"))?;
            written += dst.len() as u64;
        }
        eprintln!(
            "sidecar: layer {l:2} done, {:.1} GB in {:.0} s ({:.2} GB/s)",
            written as f64 / 1e9,
            t0.elapsed().as_secs_f64(),
            written as f64 / 1e9 / t0.elapsed().as_secs_f64()
        );
    }
    wf.sync_all().map_err(|e| e.to_string())?;
    drop(wf);
    std::fs::rename(&partial, out).map_err(|e| e.to_string())?;
    let m = manifest(s, lay);
    std::fs::write(manifest_path(out), serde_json::to_vec_pretty(&m).unwrap()).map_err(|e| e.to_string())?;
    Ok(json!({"bytes": written, "seconds": t0.elapsed().as_secs_f64()}))
}

/// Re-read `n` pseudo-random records with O_DIRECT and compare with records rebuilt from the shards.
pub fn verify(s: &Shards, lay: &Layout, path: &Path, n: usize) -> Result<usize, String> {
    let fd = crate::uring::open_direct(path).map_err(|e| e.to_string())?;
    let files: Vec<File> = s.paths.iter().map(|p| File::open(p).map_err(|e| e.to_string())).collect::<Result<_, _>>()?;
    let max_slot = lay.geom.iter().map(|g| g.slot).max().unwrap();
    let rb = Arena::new(max_slot, false).map_err(|e| e.to_string())?;
    let mut want = vec![0u8; max_slot];
    let mut x = 0x9e3779b97f4a7c15u64;
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let (l, e) = if i == 0 { (0, 0) } else if i == 1 { (N_LAYER - 1, N_EXPERT - 1) } else { ((x % N_LAYER as u64) as usize, ((x >> 20) % N_EXPERT as u64) as usize) };
        let g = lay.geom[l];
        let mut parts = Vec::new();
        for (k, p) in ["gate", "up", "down"].iter().enumerate() {
            let t = s.req(&format!("blk.{l}.ffn_{p}_exps.weight"));
            let per = if k == 2 { g.down_bytes } else { g.gate_bytes };
            let mut b = vec![0u8; per];
            files[t.shard].read_exact_at(&mut b, t.off + (e * per) as u64).map_err(|e| e.to_string())?;
            parts.push(b);
        }
        build_record(&g, lay.r4, &parts[0], &parts[1], &parts[2], &mut want[..g.slot]);
        // SAFETY: rb is page aligned and holds max_slot bytes.
        let r = unsafe { libc::pread(fd, rb.ptr as *mut libc::c_void, g.slot, lay.offset(l, e) as i64) };
        if r != g.slot as isize {
            return Err(format!("short read at layer {l} expert {e}: {r}"));
        }
        // SAFETY: just read.
        let got = unsafe { std::slice::from_raw_parts(rb.ptr, g.slot) };
        if got != &want[..g.slot] {
            return Err(format!("record (layer {l}, expert {e}) differs from the shards"));
        }
    }
    // SAFETY: our fd.
    unsafe { libc::close(fd) };
    Ok(n)
}

#[allow(dead_code)]
fn _w(_: W) {}
