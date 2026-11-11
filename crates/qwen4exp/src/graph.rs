//! The qwen4exp forward pass as a ggml graph computed by the PRODUCTION ggml objects
//! (`csrc/flash_graph.cpp`, linked against the libggml-base/libggml-cpu that production runs; see
//! `tools/sync_ggml_prod.sh`). This is the exactness reference inside the engine and the prefill
//! path: same ops, same order, same op parameters as BigMoeOnEdge's llama.cpp graph.
//!
//! Weights: the dense arena (`Dense`) for every non-expert tensor; the routed experts and the
//! N-gram table from read-only mappings of the shards (or, with an expert callback, from the
//! engine's streaming cache).

use crate::dense::Dense;
use crate::model::Shards;
use std::ffi::{c_char, c_int, c_void, CString};
use std::fs::File;
use std::os::unix::io::AsRawFd;

#[repr(C)]
pub struct FgRegion {
    pub base: *mut c_void,
    pub size: usize,
}
#[repr(C)]
pub struct FgTensorRef {
    pub name: *const c_char,
    pub ty: i32,
    pub ne: [i64; 4],
    pub data: *mut c_void,
    pub region: i32,
}
#[repr(C)]
pub struct FgPle {
    pub mult: [u64; 3],
    pub head_offsets: [u32; 16],
    pub head_vocab: [u32; 16],
    pub eos: i32,
}
#[repr(C)]
pub struct FgDump {
    pub dir: *const c_char,
    pub names: *const *const c_char,
    pub n_names: c_int,
}
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct FgOptions {
    /// attention in fixed 256-cell blocks merged in order (thread-count and batch independent)
    pub fixed_attn: c_int,
    /// shared expert built right after the expert fetch
    pub shexp_first: c_int,
    /// > 0: ubatches with at least this many tokens use the batched (wave) expert node
    pub batch_min_tokens: c_int,
    /// compute node by node and accumulate time per node kind (diagnostics, slow)
    pub profile: c_int,
    /// non-fresh one-sequence ubatches read the recurrent states in place (bitwise the same, no 3 MB 1-thread copies)
    pub rs_view: c_int,
    /// S-b: cached one-token graphs run by production ggml's op functions under the engine's barrier schedule (same bits)
    pub executor: c_int,
    /// S-c fused phases bitmask: 1 hc-mix, 2 router, 4 shared expert + combine (each bitwise = the nodes it replaces)
    pub fuse: c_int,
}
pub enum FgModel {}
pub enum FgCtx {}
pub enum GgmlTensor {}
pub type ExpertFn = unsafe extern "C" fn(call: *const crate::stream::FgExpertCall, ud: *mut c_void);

extern "C" {
    fn fg_model_new(refs: *const FgTensorRef, n: c_int, regions: *const FgRegion, n_regions: c_int, ple: *const FgPle) -> *mut FgModel;
    fn fg_model_free(m: *mut FgModel);
    fn fg_ctx_new(m: *mut FgModel, n_ctx: u32) -> *mut FgCtx;
    fn fg_ctx_free(c: *mut FgCtx);
    fn fg_ctx_clear(c: *mut FgCtx);
    fn fg_ctx_state_bytes(c: *const FgCtx) -> usize;
    fn fg_ctx_set_expert_fn(c: *mut FgCtx, f: Option<ExpertFn>, ud: *mut c_void);
    fn fg_ctx_set_options(c: *mut FgCtx, o: *const FgOptions);
    fn fg_ctx_profile_text(c: *const FgCtx, buf: *mut c_char, cap: usize) -> usize;
    fn fg_ctx_profile_reset(c: *mut FgCtx);
    fn fg_ctx_timing(c: *mut FgCtx, out5: *mut f64, reset: c_int);
    fn fg_ctx_set_graph_reuse(c: *mut FgCtx, on: c_int);
    fn fg_selftest_exec(nth: c_int) -> c_int;
    fn fg_selftest_fuse_hc(nth: c_int) -> c_int;
    fn fg_selftest_fuse_ffn(nth: c_int) -> c_int;
    fn fg_selftest_fuse_hcpost(nth: c_int) -> c_int;
    fn fg_selftest_fuse_gdn(nth: c_int) -> c_int;
    fn fg_selftest_fuse_attn(nth: c_int) -> c_int;
    fn fg_bench_dense(nth: c_int, copies: c_int, reps: c_int);
    fn fg_ctx_exec_stats(c: *const FgCtx, out3: *mut f64);
    fn fg_ctx_set_trace(c: *mut FgCtx, on: c_int);
    fn fg_ctx_trace_text(c: *const FgCtx, buf: *mut c_char, cap: usize) -> usize;
    fn fg_selftest_attn(n_kv: c_int, n_rows: c_int) -> c_int;
    // PREFILL (csrc/flash_dense.cpp): multi-column dense matmul
    fn fg_pf_set_dense(mode: c_int);
    // FP-9 (csrc/pf_phase.cpp): per-phase clocks of the prefill custom ops
    fn fg_pf_phase_ms(out: *mut f64, calls: *mut std::ffi::c_long, reset: c_int);
    // PREFILL FP-1 (csrc/pf_attn.cpp): fixed-chunk attention part op in register tiles
    fn fg_pf_set_attn(on: c_int);
    fn fg_pf_get_attn() -> c_int;
    fn fg_pf_attn_units(reset: c_int) -> std::ffi::c_long;
    fn fg_pf_selftest_attn(n_kv: c_int, n_rows: c_int, nth: c_int, pattern: c_int, reps: c_int, out2: *mut f64) -> c_int;
    fn fg_pf_stats(out2: *mut std::ffi::c_long, reset: c_int);
    fn fg_selftest_dense(nth: c_int) -> c_int;
    fn fg_selftest_split(nth: c_int) -> c_int;
    fn fg_pf_bench_dense(ty: c_int, k: std::ffi::c_long, r: std::ffi::c_long, t: std::ffi::c_long, nth: c_int, reps: c_int, out3: *mut f64) -> c_int;
    fn fg_ctx_copy_state(dst: *mut FgCtx, src: *const FgCtx) -> c_int;
    fn fg_ctx_pos(c: *const FgCtx) -> u32;
    fn fg_ctx_state_data(c: *mut FgCtx) -> *mut c_void;
    fn fg_ctx_get_tokens(c: *const FgCtx, out: *mut i32);
    fn fg_ctx_set_tokens(c: *mut FgCtx, toks: *const i32, n: u32);
    fn fg_ctx_recurrent_bytes(c: *const FgCtx) -> usize;
    fn fg_ctx_compact_bytes(c: *mut FgCtx, pos: u32) -> usize;
    fn fg_ctx_save_compact(c: *mut FgCtx, buf: *mut c_void);
    fn fg_ctx_load_compact(c: *mut FgCtx, buf: *const c_void, pos: u32) -> c_int;
    fn fg_ctx_save_recurrent(c: *const FgCtx, buf: *mut c_void);
    fn fg_ctx_restore_recurrent(c: *mut FgCtx, buf: *const c_void, pos: u32) -> c_int;
    fn fg_decode(c: *mut FgCtx, tokens: *const i32, n: c_int, n_threads: c_int, logits: *mut f32, dump: *const FgDump) -> c_int;
    fn fg_ctx_set_abort(c: *mut FgCtx, flag: *const u8);
    fn fg_selftest_abort() -> c_int;
}

/// Read-only mapping of one shard.
pub struct Mapped {
    pub ptr: *mut u8,
    pub len: usize,
}
impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our own mapping.
        unsafe { libc::munmap(self.ptr as *mut c_void, self.len) };
    }
}
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

fn map_file(p: &std::path::Path) -> Result<Mapped, String> {
    let f = File::open(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
    // SAFETY: read-only shared mapping of a file we keep no writer to.
    let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, f.as_raw_fd(), 0) };
    if ptr == libc::MAP_FAILED {
        return Err(format!("mmap {}: {}", p.display(), std::io::Error::last_os_error()));
    }
    // SAFETY: advisory.
    unsafe { libc::madvise(ptr, len, libc::MADV_RANDOM) };
    Ok(Mapped { ptr: ptr as *mut u8, len })
}

pub struct GraphModel {
    pub(crate) m: *mut FgModel,
    pub maps: Vec<Mapped>,
    _names: Vec<CString>,
}
unsafe impl Send for GraphModel {}
unsafe impl Sync for GraphModel {}

impl GraphModel {
    /// `map_experts`: give the graph the routed experts from shard mappings (reference / prefill).
    pub fn new(s: &Shards, d: &Dense, map_experts: bool) -> Result<GraphModel, String> {
        let maps: Vec<Mapped> = s.paths.iter().map(|p| map_file(p)).collect::<Result<_, _>>()?;
        let (dbase, dlen) = d.region();
        let mut regions = vec![FgRegion { base: dbase as *mut c_void, size: dlen }];
        for mp in &maps {
            regions.push(FgRegion { base: mp.ptr as *mut c_void, size: mp.len });
        }
        let mut names = Vec::new();
        let mut refs = Vec::new();
        for t in &s.tensors {
            let is_exp = t.name.contains("_exps.");
            let (data, region) = if let Some(m) = d.mats.get(&t.name) {
                (m.p as *mut c_void, 0)
            } else if is_exp && !map_experts {
                continue;
            } else {
                // SAFETY: inside the mapping of its shard.
                (unsafe { maps[t.shard].ptr.add(t.off as usize) } as *mut c_void, 1 + t.shard as i32)
            };
            let mut ne = [1i64; 4];
            for (i, &x) in t.shape.iter().enumerate() {
                ne[i] = x as i64;
            }
            let c = CString::new(t.name.clone()).unwrap();
            refs.push(FgTensorRef { name: c.as_ptr(), ty: t.w.ggml_type(), ne, data, region });
            names.push(c);
        }
        let (offs, voc) = s.ple_heads();
        let mult = s.u64_array("qwen4exp.ple.layer_multipliers");
        if offs.len() < 16 || voc.len() < 16 || mult.len() < 3 {
            return Err("PLE metadata incomplete".into());
        }
        let mut ple = FgPle { mult: [mult[0], mult[1], mult[2]], head_offsets: [0; 16], head_vocab: [0; 16], eos: 0 };
        for h in 0..16 {
            ple.head_offsets[h] = offs[h] as u32;
            ple.head_vocab[h] = voc[h] as u32;
        }
        ple.eos = s.meta_u64_pub("qwen4exp.ple.eos_token_id").ok_or("ple eos")? as i32;
        // SAFETY: refs/regions/ple outlive the call; the model keeps pointers into the arena and the
        // mappings, which GraphModel owns (the arena via the caller's Dense, which must outlive it).
        let m = unsafe { fg_model_new(refs.as_ptr(), refs.len() as c_int, regions.as_ptr(), regions.len() as c_int, &ple) };
        if m.is_null() {
            return Err("fg_model_new failed".into());
        }
        Ok(GraphModel { m, maps, _names: names })
    }
}
impl Drop for GraphModel {
    fn drop(&mut self) {
        // SAFETY: created by fg_model_new.
        unsafe { fg_model_free(self.m) };
    }
}

pub struct GraphCtx {
    pub(crate) c: *mut FgCtx,
    /// serving cancel flag (kept alive while registered)
    abort: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// `GraphCtx::decode` error text of a cancelled decode (positions unchanged; restore the recurrent snapshot or clear
/// before the next decode, which otherwise fails with "fg_decode: -7 (state dirty)").
pub const DECODE_ABORTED: &str = "fg_decode: -6 (aborted)";
unsafe impl Send for GraphCtx {}

pub struct Dump {
    pub dir: String,
    pub names: Vec<String>,
}

impl GraphCtx {
    pub fn new(m: &GraphModel, n_ctx: u32) -> GraphCtx {
        // SAFETY: valid model.
        GraphCtx { c: unsafe { fg_ctx_new(m.m, n_ctx) }, abort: None }
    }
    pub fn clear(&mut self) {
        unsafe { fg_ctx_clear(self.c) }
    }
    /// Profile table (kind, calls, ms) accumulated since the last reset (options.profile).
    pub fn profile(&self) -> Vec<(String, u64, f64)> {
        // SAFETY: two-call size protocol on a valid context.
        let txt = unsafe {
            let n = fg_ctx_profile_text(self.c, std::ptr::null_mut(), 0);
            let mut b = vec![0u8; n];
            fg_ctx_profile_text(self.c, b.as_mut_ptr() as *mut c_char, n);
            String::from_utf8_lossy(&b[..n.saturating_sub(1)]).to_string()
        };
        txt.lines()
            .filter_map(|l| {
                let mut it = l.split('\t');
                Some((it.next()?.to_string(), it.next()?.parse().ok()?, it.next()?.parse().ok()?))
            })
            .collect()
    }
    /// fg_decode stage times since the last reset: [build, alloc, inputs, compute] ms and the call count.
    pub fn timing(&mut self, reset: bool) -> ([f64; 4], f64) {
        let mut o = [0f64; 5];
        unsafe { fg_ctx_timing(self.c, o.as_mut_ptr(), reset as c_int) };
        ([o[0], o[1], o[2], o[3]], o[4])
    }
    /// One-token decodes reuse the previous graph + allocation while the shape is unchanged (default on).
    pub fn set_graph_reuse(&mut self, on: bool) {
        unsafe { fg_ctx_set_graph_reuse(self.c, on as c_int) }
    }
    /// FD-1 work/wait trace of executor runs (see flash_graph.cpp); `trace_text` returns its table.
    pub fn set_trace(&mut self, on: bool) {
        unsafe { fg_ctx_set_trace(self.c, on as c_int) }
    }
    pub fn trace_text(&self) -> String {
        unsafe {
            let n = fg_ctx_trace_text(self.c, std::ptr::null_mut(), 0);
            let mut b = vec![0u8; n];
            fg_ctx_trace_text(self.c, b.as_mut_ptr() as *mut c_char, n);
            String::from_utf8_lossy(&b[..n.saturating_sub(1)]).into_owned()
        }
    }
    /// S-b executor: (graphs scheduled, nodes in the last schedule, barriers in it).
    pub fn exec_stats(&self) -> [f64; 3] {
        let mut o = [0f64; 3];
        unsafe { fg_ctx_exec_stats(self.c, o.as_mut_ptr()) };
        o
    }
    pub fn profile_reset(&mut self) {
        unsafe { fg_ctx_profile_reset(self.c) }
    }
    pub fn set_options(&mut self, o: &FgOptions) {
        // SAFETY: valid context; plain struct copy.
        unsafe { fg_ctx_set_options(self.c, o) }
    }
    /// Snapshot / restore: copy `src`'s whole sequence state into this context (same model, n_ctx).
    pub fn copy_state_from(&mut self, src: &GraphCtx) -> Result<(), String> {
        // SAFETY: two valid contexts.
        match unsafe { fg_ctx_copy_state(self.c, src.c) } {
            0 => Ok(()),
            r => Err(format!("fg_ctx_copy_state: {r}")),
        }
    }
    /// Tokens in the state.
    pub fn pos(&self) -> usize {
        unsafe { fg_ctx_pos(self.c) as usize }
    }
    /// Save the whole sequence state to `path` (header, tokens, `extra` f32s e.g. logits, state).
    pub fn save_file(&mut self, path: &std::path::Path, extra: &[f32]) -> Result<(), String> {
        use std::io::Write;
        let n = self.pos();
        let mut toks = vec![0i32; n];
        let sb = self.state_bytes();
        // SAFETY: buffers sized from the context.
        let st = unsafe {
            fg_ctx_get_tokens(self.c, toks.as_mut_ptr());
            std::slice::from_raw_parts(fg_ctx_state_data(self.c) as *const u8, sb)
        };
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
        let mut hdr = Vec::new();
        hdr.extend_from_slice(b"FGS1");
        for v in [n as u64, extra.len() as u64, sb as u64] {
            hdr.extend_from_slice(&v.to_le_bytes());
        }
        f.write_all(&hdr).map_err(|e| e.to_string())?;
        for t in &toks {
            f.write_all(&t.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for x in extra {
            f.write_all(&x.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        f.write_all(st).map_err(|e| e.to_string())?;
        f.flush().map_err(|e| e.to_string())
    }
    /// Load a state saved by `save_file`; returns (tokens, extra).
    pub fn load_file(&mut self, path: &std::path::Path) -> Result<(Vec<i32>, Vec<f32>), String> {
        let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if b.len() < 28 || &b[..4] != b"FGS1" {
            return Err("not an FGS1 state file".into());
        }
        let u = |i: usize| u64::from_le_bytes(b[4 + 8 * i..12 + 8 * i].try_into().unwrap()) as usize;
        let (n, ne, sb) = (u(0), u(1), u(2));
        if sb != self.state_bytes() || b.len() != 28 + 4 * n + 4 * ne + sb {
            return Err("state file does not match this context".into());
        }
        let toks: Vec<i32> = (0..n).map(|i| i32::from_le_bytes(b[28 + 4 * i..32 + 4 * i].try_into().unwrap())).collect();
        let o = 28 + 4 * n;
        let extra: Vec<f32> = (0..ne).map(|i| f32::from_le_bytes(b[o + 4 * i..o + 4 + 4 * i].try_into().unwrap())).collect();
        // SAFETY: sizes checked above.
        unsafe {
            std::ptr::copy_nonoverlapping(b[o + 4 * ne..].as_ptr(), fg_ctx_state_data(self.c) as *mut u8, sb);
            fg_ctx_set_tokens(self.c, toks.as_ptr(), n as u32);
        }
        Ok((toks, extra))
    }
    /// Compact whole state at the current position: (tokens, KV/indexer rows < pos + recurrent bytes).
    pub fn save_compact(&mut self) -> (Vec<i32>, Vec<u8>) {
        let n = self.pos();
        let mut toks = vec![0i32; n];
        // SAFETY: buffers sized by the context.
        unsafe {
            fg_ctx_get_tokens(self.c, toks.as_mut_ptr());
            let mut b = vec![0u8; fg_ctx_compact_bytes(self.c, n as u32)];
            fg_ctx_save_compact(self.c, b.as_mut_ptr() as *mut c_void);
            (toks, b)
        }
    }
    /// Load a state saved by `save_compact` (the context's other cells are left as they are).
    pub fn load_compact(&mut self, toks: &[i32], b: &[u8]) -> Result<(), String> {
        // SAFETY: size checked against the context's layout.
        unsafe {
            if b.len() != fg_ctx_compact_bytes(self.c, toks.len() as u32) {
                return Err("compact state size does not match this context".into());
            }
            if fg_ctx_load_compact(self.c, b.as_ptr() as *const c_void, toks.len() as u32) != 0 {
                return Err("compact state position beyond n_ctx".into());
            }
            fg_ctx_set_tokens(self.c, toks.as_ptr(), toks.len() as u32);
        }
        Ok(())
    }
    /// Recurrent snapshot at the current position (GDN conv/ssm + PLE history).
    pub fn save_recurrent(&self) -> (usize, Vec<u8>) {
        // SAFETY: buffer sized by the context.
        unsafe {
            let mut b = vec![0u8; fg_ctx_recurrent_bytes(self.c)];
            fg_ctx_save_recurrent(self.c, b.as_mut_ptr() as *mut c_void);
            (self.pos(), b)
        }
    }
    /// Back to a snapshot taken at `pos` (<= the current position).
    pub fn restore_recurrent(&mut self, pos: usize, b: &[u8]) -> Result<(), String> {
        // SAFETY: as save_recurrent.
        unsafe {
            if b.len() != fg_ctx_recurrent_bytes(self.c) {
                return Err("snapshot size".into());
            }
            match fg_ctx_restore_recurrent(self.c, b.as_ptr() as *const c_void, pos as u32) {
                0 => Ok(()),
                r => Err(format!("fg_ctx_restore_recurrent: {r}")),
            }
        }
    }
    pub fn state_bytes(&self) -> usize {
        unsafe { fg_ctx_state_bytes(self.c) }
    }
    /// # Safety
    /// `ud` must stay valid while the context evaluates.
    pub unsafe fn set_expert_fn(&mut self, f: Option<ExpertFn>, ud: *mut c_void) {
        fg_ctx_set_expert_fn(self.c, f, ud)
    }
    pub fn decode(&mut self, tokens: &[i32], n_threads: usize, logits: &mut [f32], dump: Option<&Dump>) -> Result<(), String> {
        let (_cs, _ps, d);
        let dp = if let Some(dm) = dump {
            let dir = CString::new(dm.dir.clone()).unwrap();
            let cs: Vec<CString> = dm.names.iter().map(|n| CString::new(n.clone()).unwrap()).collect();
            let ps: Vec<*const c_char> = cs.iter().map(|c| c.as_ptr()).collect();
            d = FgDump { dir: dir.as_ptr(), names: ps.as_ptr(), n_names: ps.len() as c_int };
            _cs = (dir, cs);
            _ps = ps;
            &d as *const FgDump
        } else {
            std::ptr::null()
        };
        assert!(logits.len() >= crate::model::N_VOCAB);
        // SAFETY: valid context, buffers sized above.
        let r = unsafe { fg_decode(self.c, tokens.as_ptr(), tokens.len() as c_int, n_threads as c_int, logits.as_mut_ptr(), dp) };
        match r {
            0 => Ok(()),
            -6 => Err(DECODE_ABORTED.to_string()),
            -7 => Err("fg_decode: -7 (state dirty: an aborted decode was not rolled back)".to_string()),
            r => Err(format!("fg_decode: {r}")),
        }
    }

    /// Serving cancel: while `flag` is non-zero, a running or new decode stops (between graph nodes) with
    /// `DECODE_ABORTED`. None = off.
    pub fn set_abort(&mut self, flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>) {
        // AtomicBool has bool's layout (one byte, 0 / 1): the C side reads it as a volatile byte between graph nodes
        let p = flag.as_ref().map_or(std::ptr::null(), |a| a.as_ptr() as *const u8);
        // SAFETY: the Arc stays in self.abort while registered.
        unsafe { fg_ctx_set_abort(self.c, p) };
        self.abort = flag;
    }
}
impl Drop for GraphCtx {
    fn drop(&mut self) {
        unsafe { fg_ctx_free(self.c) };
    }
}

/// Serving-cancel self-test (synthetic graph, no model): ggml stops on the flag.
pub fn selftest_abort() -> bool {
    unsafe { fg_selftest_abort() == 0 }
}

/// S-b executor self-test (synthetic graph, no model): bitwise equal to ggml_graph_compute.
pub fn selftest_exec(nth: usize) -> bool {
    unsafe { fg_selftest_exec(nth as c_int) == 0 }
}

/// PREFILL FP-1: process-wide switch of the register-tile attention part op for prefill ubatches (>= 16 rows, mode
/// `--attn online`); bitwise equal to the graph port's own tiled path.
pub fn set_attn_tile(on: bool) {
    // SAFETY: an atomic store.
    unsafe { fg_pf_set_attn(on as c_int) }
}

/// Units (8-row tile x kv head x 256-cell block) computed by the tile path since the last reset.
pub fn attn_tile_units(reset: bool) -> i64 {
    // SAFETY: an atomic load.
    unsafe { fg_pf_attn_units(reset as c_int) as i64 }
}

/// Model-free check of the tile path against the graph port's tiled path on one fixture (patterns 0-2): (bitwise
/// equal and tile path taken, ms off, ms on).
pub fn selftest_attn_tile(n_kv: usize, n_rows: usize, nth: usize, pattern: usize, reps: usize) -> (bool, f64, f64) {
    let mut o = [0f64; 4];
    // SAFETY: self-contained; four doubles.
    let rc = unsafe { fg_pf_selftest_attn(n_kv as c_int, n_rows as c_int, nth as c_int, pattern as c_int, reps as c_int, o.as_mut_ptr()) };
    (rc == 0, o[0], o[1])
}

/// As [`selftest_attn_tile`], all timings: (bitwise incl. the fused node == merge(part), [ms part off, ms part tiles,
/// ms part + merge (tiles), ms fused node]).
pub fn selftest_attn_tile4(n_kv: usize, n_rows: usize, nth: usize, pattern: usize, reps: usize) -> (bool, [f64; 4]) {
    let mut o = [0f64; 4];
    // SAFETY: self-contained; four doubles.
    let rc = unsafe { fg_pf_selftest_attn(n_kv as c_int, n_rows as c_int, nth as c_int, pattern as c_int, reps as c_int, o.as_mut_ptr()) };
    (rc == 0, o)
}

/// W4-I: attention switch as a mode: 0 off, 1 register tiles in the part op, 2 = 1 + the fused part+merge node for
/// prefill ubatches (no partial tensor; needed for passes beyond ~4K tokens).
pub fn set_attn_tile_mode(mode: i32) {
    // SAFETY: an atomic store.
    unsafe { fg_pf_set_attn(mode as c_int) }
}

/// W4-I: the current attention switch mode (0 / 1 / 2, see [`set_attn_tile_mode`]).
pub fn attn_tile_mode() -> i32 {
    // SAFETY: an atomic load.
    unsafe { fg_pf_get_attn() as i32 }
}

/// FP-9: wall ms on thread 0 inside the prefill custom ops since the last reset: [attention tiles, dense pack,
/// dense matmul tiles], and their call counts.
pub fn pf_phase_ms(reset: bool) -> ([f64; 3], [i64; 3]) {
    let mut ms = [0f64; 3];
    let mut calls = [0 as std::ffi::c_long; 3];
    // SAFETY: three doubles and three longs.
    unsafe { fg_pf_phase_ms(ms.as_mut_ptr(), calls.as_mut_ptr(), reset as c_int) };
    (ms, [calls[0] as i64, calls[1] as i64, calls[2] as i64])
}

/// PREFILL: process-wide switch of the multi-column dense matmul (read when a graph is built). Eligible Q6_K / Q8_0
/// matmuls of ubatches of >= 256 tokens are computed in 8x4 SMMLA tiles, bitwise equal to production's mul_mat.
pub fn set_dense_gemm(on: bool) {
    // SAFETY: an atomic store.
    unsafe { fg_pf_set_dense(on as c_int) }
}

/// (matmul nodes built as multi-column, built as ggml fallback while the switch was on) since the last reset.
pub fn dense_gemm_stats(reset: bool) -> [i64; 2] {
    let mut o = [0 as std::ffi::c_long; 2];
    // SAFETY: two longs.
    unsafe { fg_pf_stats(o.as_mut_ptr(), reset as c_int) };
    [o[0] as i64, o[1] as i64]
}

/// Multi-column dense matmul self-test (synthetic, no model): tiles == scalar definitions, op bitwise == ggml_mul_mat.
pub fn selftest_dense(nth: usize) -> bool {
    // SAFETY: self-contained.
    unsafe { fg_selftest_dense(nth as c_int) == 0 }
}

/// Chunked-prefill split evidence (synthetic, no model): prints one "split <component> <pattern>" line per comparison.
/// True when every width-independent component (GDN, ssm_conv, rms_norm) is bitwise; matmul lines are reported only.
pub fn selftest_split(nth: usize) -> bool {
    // SAFETY: self-contained.
    unsafe { fg_selftest_split(nth as c_int) == 0 }
}

/// Model-free timing of one shape (ggml type id): (ms ggml_mul_mat, ms multi-column, bitwise equal).
pub fn bench_dense_gemm(ty: i32, k: usize, r: usize, t: usize, nth: usize, reps: usize) -> (f64, f64, bool) {
    let mut o = [0f64; 3];
    // SAFETY: self-contained; three doubles.
    unsafe { fg_pf_bench_dense(ty, k as _, r as _, t as _, nth as c_int, reps as c_int, o.as_mut_ptr()) };
    (o[0], o[1], o[2] == 1.0)
}

/// S-c fused phases self-tests (synthetic, no model): bitwise equal to the ggml subgraphs.
pub fn selftest_fuse(nth: usize) -> (bool, bool, bool) {
    unsafe { (fg_selftest_fuse_hc(nth as c_int) == 0, fg_selftest_fuse_ffn(nth as c_int) == 0, fg_selftest_fuse_hcpost(nth as c_int) == 0) }
}

/// S-c P3/P4 (GDN pre/post) self-test (synthetic, no model): bitwise equal to build_layer_attn_linear's nodes.
pub fn selftest_fuse_gdn(nth: usize) -> bool {
    unsafe { fg_selftest_fuse_gdn(nth as c_int) == 0 }
}

/// S-c P7 (attention pre/post) self-test (synthetic, no model): bitwise equal to build_layer_attn's nodes.
pub fn selftest_fuse_attn(nth: usize) -> bool {
    unsafe { fg_selftest_fuse_attn(nth as c_int) == 0 }
}

/// Model-free dense matvec bandwidth bench (Flash shapes): ggml graph vs static rows vs raw read.
pub fn bench_dense(nth: usize, copies: usize, reps: usize) {
    unsafe { fg_bench_dense(nth as c_int, copies as c_int, reps as c_int) }
}

/// Fixed-chunk attention self-test (synthetic, no model): true = pass.
pub fn selftest_attn(n_kv: usize, n_rows: usize) -> bool {
    // SAFETY: self-contained.
    unsafe { fg_selftest_attn(n_kv as c_int, n_rows as c_int) == 0 }
}

/// Greedy pick as llama's greedy sampler: the first maximum.
pub fn argmax(l: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..l.len() {
        if l[i] > l[best] {
            best = i;
        }
    }
    best
}
