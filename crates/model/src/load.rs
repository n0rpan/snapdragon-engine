//! Open a model: header -> descriptor -> plan -> weights (cache file backed).

use crate::desc::ModelDesc;
use crate::plan::Plan;
use crate::weights::{LoadOptions, WeightStore};
use gguf::Gguf;
use std::path::Path;

pub struct Loaded {
    pub plan: Plan,
    pub ws: WeightStore,
    pub tensors: Vec<gguf::TensorInfo>,
}

/// `n_ctx` is the KV-cache capacity (positions); `nthreads` the decode thread count.
pub fn open(model: &Path, nthreads: usize, n_ctx: usize, opt: &LoadOptions) -> Result<Loaded, String> {
    let g = Gguf::open(model).map_err(|e| e.to_string())?;
    let desc = ModelDesc::from_header(&g.header)?;
    let plan = Plan::build(&g.header, &desc, nthreads, n_ctx.div_ceil(256) * 256)?;
    plan.check_coverage()?;
    let tensors = g.header.tensors.clone();
    let ws = WeightStore::load(g, &plan, opt)?;
    Ok(Loaded { plan, ws, tensors })
}

/// Open and leak the plan and weights (a server process owns them for its whole life), so
/// an `Engine<'static>` can be built from them.
pub fn open_static(model: &Path, nthreads: usize, n_ctx: usize, opt: &LoadOptions) -> Result<(&'static Plan, &'static WeightStore), String> {
    let l = open(model, nthreads, n_ctx, opt)?;
    Ok((Box::leak(Box::new(l.plan)), Box::leak(Box::new(l.ws))))
}

/// Peak and current memory of this process from /proc/self/status and smaps_rollup (kB).
pub fn memory_report() -> Vec<(String, u64)> {
    let mut out = Vec::new();
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for l in s.lines() {
            for k in ["VmHWM", "VmRSS", "RssAnon", "RssFile"] {
                if let Some(v) = l.strip_prefix(&format!("{k}:")) {
                    out.push((k.to_string(), v.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0)));
                }
            }
        }
    }
    if let Ok(s) = std::fs::read_to_string("/proc/self/smaps_rollup") {
        for l in s.lines() {
            for k in ["AnonHugePages", "FilePmdMapped", "ShmemPmdMapped", "Rss"] {
                if let Some(v) = l.strip_prefix(&format!("{k}:")) {
                    out.push((format!("smaps.{k}"), v.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0)));
                }
            }
        }
    }
    out
}

/// Is the model path's `flock` lease free? (callers that must not run concurrently with a
/// hardware benchmark check this before a real-model run)
pub fn lease_free(path: &Path) -> bool {
    use std::os::unix::io::AsRawFd;
    match std::fs::File::open(path) {
        Ok(f) => {
            // SAFETY: flock on our own fd; released when f is dropped.
            let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            rc == 0
        }
        Err(_) => true,
    }
}

