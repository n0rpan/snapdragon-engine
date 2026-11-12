//! The accelerator path of the fast model: the big linears run as oxn GEMMs on an
//! [`crate::Backend`] (the CPU emulator `crate::emu::Emu`, or the DSP behind the `device` feature).
//!
//! One [`NpuCtx`] per model. Weights are placed once at load ([`NpuCtx::add_linear`]), in 128 MiB
//! weight buffers, as blobs: one per (K chunk, output part). An output part is a run of whole output
//! tiles whose blob fits `max_blob_bytes`. A whole linear lives on one session: a backend session has a
//! 32-bit address space of its own, so the model's weights are spread over `backend.sessions()` of them.
//! Each session has two activation buffers (64 MiB) and one output buffer (128 MiB), reused by every
//! linear of that session: a linear's activation must not be overwritten before its run has finished,
//! and its output must be read out before the next run (the fast model does both).
//!
//! Layouts are those of `crate::tiles`: activations (fp16, K chunks side by side), weight blobs in the
//! format of [`NpuConfig::weights`] (W8 records or fp16 tiles), outputs column-major (`GF_YCOL`) with
//! one partial sum per K chunk, summed by `NpuOut::row`.

use std::collections::BTreeMap;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::tiles::{self, ActLayout, OutLayout};
use crate::wfmt::{self, WFmt};
use crate::{Backend, Buf, GemmArgs, GF_YCOL};

const ACT_BYTES: usize = 64 << 20;
const OUT_BYTES: usize = 128 << 20;
const WEIGHT_BYTES: usize = 128 << 20;
/// Alignment of every offset handed to the backend (oxn GEMM_ALIGN).
const ALIGN: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct NpuConfig {
    /// Max total bytes of weights placed on the backend (the weight bytes of every placed linear, in `weights`
    /// format), split evenly over the sessions: each session takes at most `budget_bytes / sessions`.
    pub budget_bytes: usize,
    /// K chunk limit: kc is the largest multiple of 32 that divides K and is <= kc_max. 0 = no K chunking (kc = K).
    pub kc_max: usize,
    /// Output tiles in column order (`GF_YCOL`). Must be true: [`NpuCtx::new`] rejects false, because the output
    /// parts of a linear are written as column-major tile runs.
    pub ycol: bool,
    /// Threads that pack a linear's weights into its blobs (0 = pool CPU count when provided,
    /// otherwise `std::thread::available_parallelism()`).
    pub pack_threads: usize,
    /// Token-row split (0 = off): with `p > 0` an accelerator linear also keeps its CPU copy and each
    /// call computes about `p` percent of its token rows on the CPU, at the same time as the accelerator.
    pub cpu_rows_pct: u32,
    /// Weight format of the blobs: W8 records (`GF_W8`) or fp16 tiles.
    pub weights: WFmt,
    /// Largest weight blob in bytes (0 = 128 MiB, the weight buffer size). A linear whose blobs would be larger is
    /// split into output parts of whole output tiles.
    pub max_blob_bytes: usize,
    /// Accuracy audit (see [`NpuCtx::audit_rows`]): the audited outputs are compared with an exact f32 reference.
    /// Changes no output.
    pub audit: bool,
}

/// A logical, row-major weight source. Implementations may dequantize one GGUF row
/// at a time; accelerator packing never needs a full f32 tensor copy. The source
/// must stay immutable while packing workers read distinct rows concurrently.
pub trait WeightRows: Sync {
    fn shape(&self) -> (usize, usize);
    /// Fill every element of `dst`, or return an error.
    fn read_row(&self, row: usize, dst: &mut [f32]) -> Result<(), String>;
}

/// One weight blob of a linear: the strips of output tiles `ot0 .. ot0 + nt` of K chunk `chunk`.
#[derive(Clone, Copy, Debug)]
pub struct NpuBlob {
    /// slot of the weight buffer, in the linear's session
    pub slot: u32,
    /// byte offset in the slot
    pub off: u32,
    pub chunk: u32,
    pub ot0: u32,
    pub nt: u32,
    /// start of the blob in the shared buffer (for the audit and for tests)
    pub ptr: *const u8,
}

// SAFETY: the pointer is into a backend buffer that lives as long as the backend (owned by NpuCtx) and is only read.
unsafe impl Send for NpuBlob {}
unsafe impl Sync for NpuBlob {}

/// A linear placed on the backend: one blob per (K chunk, output part), all on one session.
#[derive(Clone, Debug)]
pub struct NpuLinear {
    /// the tensor name (for the audit)
    pub name: String,
    pub o: usize,
    pub k: usize,
    pub kc: usize,
    /// the session that holds the blobs, activations and outputs of this linear
    pub session: usize,
    pub blobs: Vec<NpuBlob>,
}

/// Counters. The per-forward ones (`calls`, `gemm_s`, `dsp_us`, `pcycles`, `hmx_pcycles`, `deq_cycles`,
/// `unpack_s`, `nonfinite`, `max_abs`, `npu_rows`, `cpu_rows`, `split_cpu_s`) are reset by
/// `NpuCtx::reset_step_stats`; the rest are for the whole model.
#[derive(Clone, Copy, Debug, Default)]
pub struct NpuStats {
    /// backend GEMM calls (one per weight blob of a linear)
    pub calls: u64,
    /// wall time of the backend GEMM calls, seconds
    pub gemm_s: f64,
    /// DSP-side time reported by the backend, microseconds
    pub dsp_us: u64,
    /// DSP cycles of the GEMM calls (total, the sum of `GemmStats::pcycles`)
    pub pcycles: u64,
    /// DSP cycles of the HMX part of the GEMM calls
    pub hmx_pcycles: u64,
    /// DSP cycles of the W8 expansion of the GEMM calls
    pub deq_cycles: u64,
    /// CPU time delivering the outputs to the sinks, seconds
    pub unpack_s: f64,
    /// wall time spent placing linears (packing and copying into the weight buffers), seconds
    pub pack_s: f64,
    /// Largest worker count used for a weight blob (not the pool caller's inherited CPU count).
    pub pack_threads: usize,
    pub weights_bytes: usize,
    /// Actual reserved shared-buffer bytes, including activation/output buffers and weight slack.
    pub allocated_bytes: usize,
    /// Reserved weight-buffer capacity (distinct from occupied `weights_bytes`).
    pub weight_capacity_bytes: usize,
    /// linears placed on the accelerator
    pub linears: usize,
    /// linears offered to the accelerator that did not fit (shape or budget): they stay on the CPU
    pub cpu_linears: usize,
    /// sessions of the backend
    pub sessions: usize,
    /// weight format of the blobs
    pub weights: WFmt,
    /// the configured K chunk limit (0 = no K chunking)
    pub kc_max: usize,
    /// weight blobs placed (one per K chunk and output part of each linear)
    pub blobs: usize,
    /// output parts placed, summed over the linears (a linear that is not split has one)
    pub parts: usize,
    /// delivered values that were not finite (only counted with OXI_NPU_STATS=1)
    pub nonfinite: u64,
    /// largest finite |value| delivered (only counted with OXI_NPU_STATS=1)
    pub max_abs: f32,
    /// token rows the accelerator computed (all accelerator calls; the rows of a split call are `n_npu`)
    pub npu_rows: u64,
    /// token rows the CPU computed in split calls (`cpu_rows_pct > 0`)
    pub cpu_rows: u64,
    /// wall time of the CPU phase of the split calls, seconds
    pub split_cpu_s: f64,
}

/// One accuracy line of the audit (see [`NpuCtx::audit_report`]). Errors are relative to the exact f32 product
/// of the audited input row with the weights (`ref`), in the same units as the outputs.
#[derive(Clone, Debug)]
pub struct AuditLine {
    pub name: String,
    /// audited output values
    pub samples: u64,
    /// RMS(y - ref) / RMS(ref) over the samples
    pub rel_rmse: f64,
    /// predicted relative RMSE from the fp16 rounding of every K-chunk partial sum (and of the output)
    pub pred_rel_rmse_chunks: f64,
    /// largest |x| of the audited input rows
    pub max_x: f64,
    /// largest |ref|
    pub max_ref: f64,
    /// largest |partial| / max(|ref|, 1e-6): how much the K-chunk partials exceed the result
    pub cancel: f64,
    /// largest |y - ref|
    pub max_abs_err: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct AuditAcc {
    samples: u64,
    se: f64,
    sr: f64,
    pred_err2: f64,
    max_x: f64,
    max_ref: f64,
    cancel: f64,
    max_abs_err: f64,
}

impl AuditAcc {
    fn merge(&mut self, o: &AuditAcc) {
        self.samples += o.samples;
        self.se += o.se;
        self.sr += o.sr;
        self.pred_err2 += o.pred_err2;
        self.max_x = self.max_x.max(o.max_x);
        self.max_ref = self.max_ref.max(o.max_ref);
        self.cancel = self.cancel.max(o.cancel);
        self.max_abs_err = self.max_abs_err.max(o.max_abs_err);
    }
}

/// The audit accumulators per linear name, and the names in order of first use.
#[derive(Default)]
struct AuditStore {
    order: Vec<String>,
    acc: BTreeMap<String, AuditAcc>,
}

struct WeightBuf {
    buf: Buf,
    used: usize,
}

/// The buffers and the placed weights of one backend session.
struct Session {
    /// activation buffers, ids 0 and 1
    act: [Buf; 2],
    out: Buf,
    weights: Vec<WeightBuf>,
    /// bytes of weights placed on this session
    weight_bytes: usize,
}

pub struct NpuCtx {
    backend: Mutex<Box<dyn Backend>>,
    cfg: NpuConfig,
    sess: Vec<Session>,
    stats: Mutex<NpuStats>,
    audit: Mutex<AuditStore>,
    pack_cpus: Vec<usize>,
    packing_failed: bool,
}

impl NpuCtx {
    pub fn new(mut backend: Box<dyn Backend>, cfg: NpuConfig) -> Result<NpuCtx, String> {
        if cfg.kc_max != 0 && cfg.kc_max < 32 {
            return Err(format!("npu: kc_max {} is below 32 (0 = no K chunking)", cfg.kc_max));
        }
        if !cfg.ycol {
            return Err("npu: ycol must be true (the output parts of a linear are column-major tile runs)".into());
        }
        if cfg.max_blob_bytes > WEIGHT_BYTES {
            return Err(format!("npu: max_blob_bytes {} is more than a weight buffer ({WEIGHT_BYTES})", cfg.max_blob_bytes));
        }
        let n = backend.sessions();
        if n == 0 {
            return Err("npu: the backend has no sessions".into());
        }
        let mut sess = Vec::with_capacity(n);
        for s in 0..n {
            let a0 = backend.alloc(s, ACT_BYTES)?;
            let a1 = backend.alloc(s, ACT_BYTES)?;
            let out = backend.alloc(s, OUT_BYTES)?;
            sess.push(Session { act: [a0, a1], out, weights: Vec::new(), weight_bytes: 0 });
        }
        let stats = NpuStats { sessions: n, weights: cfg.weights, kc_max: cfg.kc_max, ..Default::default() };
        Ok(NpuCtx {
            backend: Mutex::new(backend),
            cfg,
            sess,
            stats: Mutex::new(stats),
            audit: Mutex::new(AuditStore::default()),
            pack_cpus: Vec::new(),
            packing_failed: false,
        })
    }

    /// Use the image pool's CPUs for packing. Pool construction pins its caller to one CPU;
    /// workers must explicitly pin themselves rather than inherit that caller's narrow mask.
    /// The caller's mask is never changed by packing.
    pub fn with_pack_cpus(mut self, cpus: &[usize]) -> Self {
        self.pack_cpus = cpus.to_vec();
        self
    }

    /// K chunk of a linear with reduction size `k`: the largest multiple of 32 that divides `k` and is at most
    /// `kc_max`; `k` itself when `kc_max` is 0.
    fn kc_for(&self, k: usize) -> usize {
        if self.cfg.kc_max == 0 {
            return k;
        }
        let mut kc = (self.cfg.kc_max.min(k) / 32) * 32;
        while kc > 32 && k % kc != 0 {
            kc -= 32;
        }
        kc
    }

    /// The largest blob in bytes (`max_blob_bytes`, or the weight buffer size when it is 0).
    fn max_blob_bytes(&self) -> usize {
        if self.cfg.max_blob_bytes == 0 {
            WEIGHT_BYTES
        } else {
            self.cfg.max_blob_bytes
        }
    }

    /// Pack an `o x k` Q8_0 linear `name` (`q8` as in the GGUF; packed row n takes source row `perm[n]`) in the
    /// configured weight format and place it on the backend, whole, on the session with the fewest weight bytes
    /// that still has room under its share of the budget. `Ok(None)` when the shape is not a multiple of 32 or no
    /// session has room; the linear then stays on the CPU (counted in `cpu_linears`). An error when one strip
    /// does not fit `max_blob_bytes`.
    ///
    /// Blobs: one per (K chunk, output part); the parts cover the output tiles in order, each with the largest
    /// whole-tile count whose blob fits `max_blob_bytes`. Each blob is reserved in a weight buffer first, then
    /// packed in place by `pack_threads` workers (strips taken from a shared counter); no temporary copy is made.
    pub fn add_linear(&mut self, name: &str, q8: &[u8], o: usize, k: usize, perm: Option<&[u32]>) -> Result<Option<NpuLinear>, String> {
        if o == 0 || k == 0 || o % 32 != 0 || k % 32 != 0 {
            return self.stay_on_cpu();
        }
        if Some(q8.len()) != o.checked_mul(k / 32).and_then(|n| n.checked_mul(34)) {
            return Err(format!("npu: {name}: Q8_0 length {} for a {o}x{k} linear", q8.len()));
        }
        if let Some(p) = perm {
            if p.len() != o {
                return Err(format!("npu: {name}: perm length {} for {o} rows", p.len()));
            }
        }
        if let Some(p) = perm {
            if p.iter().any(|&row| row as usize >= o) {
                return Err(format!("npu: {name}: permutation row out of range"));
            }
        }
        let fmt = self.cfg.weights;
        self.place_linear(name, o, k, |dst, kc, c, ot0, threads, cpus| {
            pack_blob(dst, fmt, q8, o, k, kc, perm, c, ot0, threads, cpus)
        })
    }

    /// Place mixed-format logical rows as fp16 tiles. Unsupported shapes remain
    /// on the CPU. A read/conversion failure is an error, never a placed tensor.
    /// On error callers must discard this context (reserved buffers remain owned).
    pub fn add_linear_rows(&mut self, name: &str, rows: &dyn WeightRows) -> Result<Option<NpuLinear>, String> {
        if self.cfg.weights != WFmt::Fp16 {
            return Err("npu: logical row packing requires fp16 weights".into());
        }
        let (o, k) = rows.shape();
        if o == 0 || k == 0 || o % 32 != 0 || k % 32 != 0 {
            return self.stay_on_cpu();
        }
        self.place_linear(name, o, k, |dst, kc, c, ot0, threads, cpus| {
            pack_rows_blob(dst, rows, k, kc, c, ot0, threads, cpus)
        })
    }

    fn place_linear(&mut self, name: &str, o: usize, k: usize,
        mut pack: impl FnMut(&mut [u8], usize, usize, usize, usize, &[usize]) -> Result<usize, String>,
    ) -> Result<Option<NpuLinear>, String> {
        if self.packing_failed { return Err("npu: discard context after packing failure".into()); }
        let fmt = self.cfg.weights;
        k.checked_mul(o).and_then(|n| n.checked_mul(2))
            .ok_or_else(|| format!("npu: {name}: weight shape overflow"))?;
        let kc = self.kc_for(k);
        let nchunks = k / kc;
        let strip = fmt.strip_bytes(kc);
        let ntile_o = o / 32;
        let bytes = nchunks.checked_mul(ntile_o).and_then(|n| n.checked_mul(strip))
            .ok_or_else(|| format!("npu: {name}: packed shape overflow"))?;
        let max_blob = self.max_blob_bytes();
        if strip > max_blob {
            return Err(format!("npu: {name}: one {fmt:?} strip of {strip} bytes (kc {kc}) does not fit max_blob_bytes {max_blob}"));
        }
        let cap = self.cfg.budget_bytes / self.sess.len();
        let session = self
            .sess
            .iter()
            .enumerate()
            .filter(|(_, s)| bytes <= cap.saturating_sub(s.weight_bytes))
            .min_by_key(|(_, s)| s.weight_bytes)
            .map(|(i, _)| i);
        let Some(session) = session else {
            return self.stay_on_cpu();
        };
        let nt_max = (max_blob / strip).min(ntile_o);
        // output parts: (first output tile, tiles), in order
        let parts: Vec<(usize, usize)> = (0..ntile_o).step_by(nt_max).map(|ot0| (ot0, nt_max.min(ntile_o - ot0))).collect();
        let threads = self.pack_threads();
        self.packing_failed = true; // cleared only after all allocations and workers succeed
        let t_pack = Instant::now();
        let mut workers_used = 0;
        let mut blobs = Vec::with_capacity(nchunks * parts.len());
        {
            let mut be = self.backend.lock().expect("npu backend");
            let s = &mut self.sess[session];
            for c in 0..nchunks {
                for &(ot0, nt) in &parts {
                    let len = nt * strip;
                    // first fit over this session's weight buffers, else a new buffer
                    let fit = s.weights.iter().position(|w| align_up(w.used) + len <= w.buf.len);
                    let (wi, off) = match fit {
                        Some(i) => (i, align_up(s.weights[i].used)),
                        None => {
                            let buf = be.alloc(session, WEIGHT_BYTES)?;
                            s.weights.push(WeightBuf { buf, used: 0 });
                            (s.weights.len() - 1, 0)
                        }
                    };
                    let w = &mut s.weights[wi];
                    w.used = off + len;
                    // SAFETY: [off, off + len) lies inside the buffer (checked by the fit test) and is ours. The
                    // slice is the only reference to these bytes while it lives (this loop iteration only), and
                    // nothing else touches the backend buffers: the backend lock is held.
                    let dst = unsafe { std::slice::from_raw_parts_mut(w.buf.ptr.add(off), len) };
                    let packed_workers = match pack(dst, kc, c, ot0, threads, &self.pack_cpus) {
                        Ok(n) => n,
                        Err(e) => { self.packing_failed = true; return Err(e); }
                    };
                    workers_used = workers_used.max(packed_workers);
                    blobs.push(NpuBlob {
                        slot: w.buf.slot,
                        off: off as u32,
                        chunk: c as u32,
                        ot0: ot0 as u32,
                        nt: nt as u32,
                        // SAFETY: in bounds of the buffer (as above).
                        ptr: unsafe { w.buf.ptr.add(off) } as *const u8,
                    });
                }
            }
        }
        self.sess[session].weight_bytes += bytes;
        self.packing_failed = false;
        let mut st = self.stats.lock().expect("npu stats");
        st.weights_bytes += bytes;
        st.linears += 1;
        st.blobs += blobs.len();
        st.parts += parts.len();
        st.pack_s += t_pack.elapsed().as_secs_f64();
        st.pack_threads = st.pack_threads.max(workers_used);
        if st.linears == 1 && !self.pack_cpus.is_empty() {
            eprintln!("oxi: npu pack: workers {workers_used}, verified CPU masks {:?}", &self.pack_cpus[..workers_used]);
        }
        Ok(Some(NpuLinear { name: name.to_string(), o, k, kc, session, blobs }))
    }

    fn stay_on_cpu(&self) -> Result<Option<NpuLinear>, String> {
        self.stats.lock().expect("npu stats").cpu_linears += 1;
        Ok(None)
    }

    /// Resolve 0 from the explicit pool CPU list, before consulting the inherited caller mask.
    fn pack_threads(&self) -> usize {
        if !self.pack_cpus.is_empty() {
            return if self.cfg.pack_threads == 0 { self.pack_cpus.len() } else { self.cfg.pack_threads.min(self.pack_cpus.len()) };
        }
        match self.cfg.pack_threads {
            0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
            n => n,
        }
    }

    /// Activation view of `n` tokens x `k` channels in buffer `id` of `session`, with K chunks of `kc`.
    /// The layout's bytes are zeroed first (padding rows stay zero).
    pub fn act(&self, session: usize, id: usize, n: usize, k: usize, kc: usize) -> Result<NpuAct, String> {
        let s = self.sess.get(session).ok_or_else(|| format!("npu: no session {session}"))?;
        let buf = *s.act.get(id).ok_or_else(|| format!("npu: no activation buffer {id}"))?;
        if n == 0 || k == 0 || k % 32 != 0 || kc == 0 || kc % 32 != 0 || k % kc != 0 {
            return Err(format!("npu: bad activation shape n {n}, k {k}, kc {kc}"));
        }
        let lay = ActLayout::new(n, k, kc);
        if lay.bytes() > buf.len {
            return Err(format!(
                "npu: {n} tokens x {k} channels need {} MiB of activations, the buffer holds {} MiB",
                lay.bytes() >> 20,
                buf.len >> 20
            ));
        }
        // SAFETY: lay.bytes() <= buf.len, and no run reads this buffer while the caller owns the view.
        unsafe { ptr::write_bytes(buf.ptr, 0, lay.bytes()) };
        Ok(NpuAct { ptr: buf.ptr as *mut u16, lay, buf_off: 0, slot: buf.slot, session })
    }

    /// Y = X W^T for `lin`: one backend GEMM per blob on `lin.session` (each writes its output tiles of its K chunk),
    /// partial sums left for `NpuOut::row`. `x` must have been made for the same session.
    pub fn run(&self, lin: &NpuLinear, x: &NpuAct) -> Result<NpuOut, String> {
        if self.packing_failed { return Err("npu: discard context after packing failure".into()); }
        if x.session != lin.session {
            return Err(format!("npu: activation of session {} used by a linear of session {}", x.session, lin.session));
        }
        if x.lay.k != lin.k || x.lay.kc != lin.kc {
            return Err(format!("npu: activation k {} kc {} does not match the linear (k {}, kc {})", x.lay.k, x.lay.kc, lin.k, lin.kc));
        }
        let out = self.sess[lin.session].out;
        let nchunks = lin.k / lin.kc;
        let lay = OutLayout { t_pad: x.lay.t_pad, o: lin.o, nchunks, ycol: true };
        if lay.bytes() > out.len {
            return Err(format!("npu: output of {} MiB does not fit the {} MiB buffer", lay.bytes() >> 20, out.len >> 20));
        }
        let flags = self.cfg.weights.gemm_flag() | GF_YCOL;
        // one row tile of Y is t_pad/32 tiles of 2048 bytes; with column-major tiles, output tile ot0 starts after ot0 such columns
        let col_bytes = (x.lay.t_pad / 32) * tiles::TILE_BYTES;
        let mut calls = 0u64;
        let (mut wall, mut dsp) = (0.0f64, 0u64);
        let (mut pc, mut hmx, mut deq) = (0u64, 0u64, 0u64);
        {
            let mut be = self.backend.lock().expect("npu backend");
            for b in &lin.blobs {
                let c = b.chunk as usize;
                let a = GemmArgs {
                    x_slot: x.slot,
                    x_off: x.buf_off + (c * x.lay.chunk_bytes()) as u32,
                    w_slot: b.slot,
                    w_off: b.off,
                    y_slot: out.slot,
                    y_off: (c * lay.chunk_bytes() + b.ot0 as usize * col_bytes) as u32,
                    t: x.lay.t_pad as u32,
                    k: lin.kc as u32,
                    o: 32 * b.nt,
                    flags,
                    r: 0,
                    c: pick_c(be.vtcm_bytes() as usize, x.lay.t_pad, lin.kc, 32 * b.nt as usize, self.cfg.weights),
                };
                let s = be.gemm(lin.session, &a)?;
                calls += 1;
                wall += s.wall_s;
                dsp += s.dsp_us as u64;
                pc += s.pcycles;
                hmx += s.hmx_pcycles;
                deq += s.deq_pcycles;
            }
        }
        let mut st = self.stats.lock().expect("npu stats");
        st.calls += calls;
        st.gemm_s += wall;
        st.dsp_us += dsp;
        st.pcycles += pc;
        st.hmx_pcycles += hmx;
        st.deq_cycles += deq;
        Ok(NpuOut { ptr: out.ptr as *const u16, lay })
    }

    /// Accuracy audit of one delivered token: `x` is the exact f32 input row of the token (length k), `y(n)` the
    /// delivered output n of that token, for every n in `n_list`. For each n the reference is the f64 sum
    /// sum_k x[k] * w(n, k) with the weights read back from the blobs (`wfmt::weight_row`), and the K-chunk partial
    /// sums. Accumulates per linear name; does nothing unless `NpuConfig::audit` is on.
    pub fn audit_rows(&self, lin: &NpuLinear, x: &[f32], y: &dyn Fn(usize) -> f32, n_list: &[usize]) {
        if !self.cfg.audit || n_list.is_empty() {
            return;
        }
        assert_eq!(x.len(), lin.k, "audit: x length");
        let (fmt, kc) = (self.cfg.weights, lin.kc);
        let nchunks = lin.k / kc;
        let max_x = x.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
        let mut acc = AuditAcc { max_x, ..Default::default() };
        let mut wrow = vec![0f32; kc];
        for &n in n_list {
            assert!(n < lin.o, "audit: row {n} of {}", lin.o);
            let (mut reference, mut max_part, mut pred) = (0f64, 0f64, 0f64);
            for c in 0..nchunks {
                let t = n / 32;
                let b = lin
                    .blobs
                    .iter()
                    .find(|b| b.chunk as usize == c && (b.ot0 as usize..(b.ot0 + b.nt) as usize).contains(&t))
                    .expect("audit: blob of the row");
                // SAFETY: the blob holds output tiles ot0..ot0+nt of chunk c, and t is one of them.
                unsafe { wfmt::weight_row(fmt, b.ptr, kc, b.ot0 as usize, n, &mut wrow) };
                let mut part = 0f64;
                for (kl, &w) in wrow.iter().enumerate() {
                    part += x[c * kc + kl] as f64 * w as f64;
                }
                max_part = max_part.max(part.abs());
                // the accelerator rounds each partial sum to fp16: its error is uniform in +-step/2
                let step = fp16_step(part);
                pred += step * step / 12.0;
                reference += part;
            }
            let yv = y(n) as f64;
            let err = yv - reference;
            acc.samples += 1;
            acc.se += err * err;
            acc.sr += reference * reference;
            acc.pred_err2 += pred;
            acc.max_ref = acc.max_ref.max(reference.abs());
            acc.cancel = acc.cancel.max(max_part / reference.abs().max(1e-6));
            acc.max_abs_err = acc.max_abs_err.max(err.abs());
        }
        let mut st = self.audit.lock().expect("npu audit");
        if !st.acc.contains_key(&lin.name) {
            st.order.push(lin.name.clone());
        }
        st.acc.entry(lin.name.clone()).or_default().merge(&acc);
    }

    /// Whether the accuracy audit is on (`NpuConfig::audit`).
    pub fn audit_on(&self) -> bool {
        self.cfg.audit
    }

    /// The audit lines so far, one per audited linear, in order of first use.
    pub fn audit_report(&self) -> Vec<AuditLine> {
        let st = self.audit.lock().expect("npu audit");
        st.order
            .iter()
            .map(|name| {
                let a = st.acc[name];
                let rel = |num: f64, den: f64| {
                    if den > 0.0 {
                        (num / den).sqrt()
                    } else if num > 0.0 {
                        f64::INFINITY
                    } else {
                        0.0
                    }
                };
                AuditLine {
                    name: name.clone(),
                    samples: a.samples,
                    rel_rmse: rel(a.se, a.sr),
                    pred_rel_rmse_chunks: rel(a.pred_err2, a.sr),
                    max_x: a.max_x,
                    max_ref: a.max_ref,
                    cancel: a.cancel,
                    max_abs_err: a.max_abs_err,
                }
            })
            .collect()
    }

    pub fn stats(&self) -> NpuStats {
        let mut stats = *self.stats.lock().expect("npu stats");
        for session in &self.sess {
            for buffer in &session.act {
                stats.allocated_bytes = stats.allocated_bytes.saturating_add(buffer.len);
            }
            stats.allocated_bytes = stats.allocated_bytes.saturating_add(session.out.len);
            for weight in &session.weights {
                stats.weight_capacity_bytes = stats.weight_capacity_bytes.saturating_add(weight.buf.len);
                stats.allocated_bytes = stats.allocated_bytes.saturating_add(weight.buf.len);
            }
        }
        stats
    }

    /// Zero the per-forward counters (not the placement totals, nor `pack_s`).
    pub fn reset_step_stats(&self) {
        let mut st = self.stats.lock().expect("npu stats");
        st.calls = 0;
        st.gemm_s = 0.0;
        st.dsp_us = 0;
        st.pcycles = 0;
        st.hmx_pcycles = 0;
        st.deq_cycles = 0;
        st.unpack_s = 0.0;
        st.nonfinite = 0;
        st.max_abs = 0.0;
        st.npu_rows = 0;
        st.cpu_rows = 0;
        st.split_cpu_s = 0.0;
    }

    /// The token-row split percentage of the configuration (0 = off).
    pub fn cpu_rows_pct(&self) -> u32 {
        self.cfg.cpu_rows_pct
    }

    pub fn add_unpack(&self, secs: f64) {
        self.stats.lock().expect("npu stats").unpack_s += secs;
    }

    /// Row accounting of one accelerator call: `npu_rows` computed on the accelerator, `cpu_rows` on the CPU
    /// in the same call, and the wall time `cpu_s` of that CPU phase.
    pub fn note_rows(&self, npu_rows: u64, cpu_rows: u64, cpu_s: f64) {
        let mut st = self.stats.lock().expect("npu stats");
        st.npu_rows += npu_rows;
        st.cpu_rows += cpu_rows;
        st.split_cpu_s += cpu_s;
    }

    pub fn note_values(&self, nonfinite: u64, max_abs: f32) {
        let mut st = self.stats.lock().expect("npu stats");
        st.nonfinite += nonfinite;
        st.max_abs = st.max_abs.max(max_abs);
    }
}

/// Column tiles per block for one GEMM: the largest C in {4, 2, 1} that needs the fewest weight
/// passes (row blocks). It mirrors the kernel's VTCM budget (all VTCM minus a 4 KiB header; fp16
/// weights double-buffered, W8 as one staging block plus one fp16 block). Wide K (9216, 12288)
/// leaves room for only a few row tiles at C = 4, and every extra row block streams the whole
/// weight operand again (measured: 3.8 TOPS instead of 10 when the block choice is wrong).
/// 0 = let the kernel decide.
fn pick_c(vtcm: usize, t_pad: usize, kc: usize, o: usize, fmt: crate::wfmt::WFmt) -> u32 {
    let (nrt, nkt, not) = (t_pad / 32, kc / 32, o / 32);
    let tk = nkt * 2048;
    let budget = vtcm.saturating_sub(4096);
    let mut best = (usize::MAX, 0u32);
    for c in [4usize, 2, 1] {
        let c = c.min(not.max(1));
        let fixed = match fmt {
            crate::wfmt::WFmt::Fp16 => 2 * c * tk,
            crate::wfmt::WFmt::W8 => (c * nkt * 1152).div_ceil(2048) * 2048 + c * tk,
        };
        if fixed >= budget || nrt == 0 {
            continue;
        }
        let r = ((budget - fixed) / (tk + 2 * c * 2048)).min(nrt);
        if r == 0 {
            continue;
        }
        let passes = nrt.div_ceil(r);
        if passes < best.0 {
            best = (passes, c as u32);
        }
    }
    best.1
}

#[inline]
fn align_up(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

/// Spacing of fp16 values at `v` (the step of the rounding grid): 2^(e-10) for a normal half with exponent e,
/// 2^-24 in the subnormal range, 0 for 0.
fn fp16_step(v: f64) -> f64 {
    let a = v.abs() as f32;
    if a == 0.0 || !a.is_finite() {
        return 0.0;
    }
    if a < 2f32.powi(-14) {
        return 2f64.powi(-24);
    }
    let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    2f64.powi(e.clamp(-14, 15) - 10)
}

/// One f32 row per worker, reused for all strips. Output strips have disjoint
/// owners. Every worker is joined even after a source error or panic.
#[allow(clippy::too_many_arguments)]
fn pack_rows_blob(dst: &mut [u8], rows: &dyn WeightRows, k: usize, kc: usize,
    chunk: usize, ot0: usize, threads: usize, cpus: &[usize]) -> Result<usize, String> {
    let strip = WFmt::Fp16.strip_bytes(kc);
    let strips: Vec<Mutex<&mut [u8]>> = dst.chunks_mut(strip).map(Mutex::new).collect();
    let next = AtomicUsize::new(0);
    let worker = || -> Result<(), String> {
        let mut row = vec![0f32; k];
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= strips.len() { return Ok(()); }
            let mut dst = strips[i].lock().map_err(|_| "npu: row strip poisoned")?;
            for nl in 0..32 {
                rows.read_row((ot0 + i) * 32 + nl, &mut row)?;
                for kt in 0..kc / 32 {
                    let src = &row[chunk * kc + kt * 32..chunk * kc + (kt + 1) * 32];
                    if src.iter().any(|x| !x.is_finite() || x.abs() > 65504.0) {
                        return Err("npu: row cannot be represented as finite fp16".into());
                    }
                    let mut half = [0u16; 32];
                    tiles::f32s_to_halfs(src, &mut half);
                    let tile = &mut dst[kt * 2048..(kt + 1) * 2048];
                    for (kl, h) in half.iter().enumerate() {
                        let off = tiles::wt_elem_idx(kl, nl) * 2;
                        tile[off..off + 2].copy_from_slice(&h.to_le_bytes());
                    }
                }
            }
        }
    };
    let n = threads.clamp(1, strips.len().max(1));
    if n == 1 && cpus.is_empty() { worker()?; return Ok(1); }
    std::thread::scope(|scope| {
        let handles = (0..n).map(|i| {
            let worker = &worker;
            scope.spawn(move || {
                if !cpus.is_empty() {
                    let cpu = cpus[i % cpus.len()];
                    sched::pin_current_thread(cpu).map_err(|e| format!("npu pack: pin CPU {cpu}: {e}"))?;
                    if sched::allowed_cpus() != 1 { return Err(format!("npu pack: CPU {cpu} affinity was not applied")); }
                }
                worker()
            })
        }).collect();
        join_pack_workers(handles)?;
        Ok(n)
    })
}

/// Packs K chunk `chunk` of output tiles `ot0 ..` of an `o x k` linear into `dst`, one blob of
/// `dst.len() / strip` strips, with up to `threads` workers. Workers take strips from a shared counter; strip i
/// is `dst`'s i-th `strip`-byte piece (output tile ot0 + i), behind its own mutex, so each strip has exactly one writer.
#[allow(clippy::too_many_arguments)]
fn pack_blob(dst: &mut [u8], fmt: WFmt, q8: &[u8], o: usize, k: usize, kc: usize, perm: Option<&[u32]>, chunk: usize, ot0: usize, threads: usize, cpus: &[usize]) -> Result<usize, String> {
    let strip = fmt.strip_bytes(kc);
    let strips: Vec<Mutex<&mut [u8]>> = dst.chunks_mut(strip).map(Mutex::new).collect();
    let n_strips = strips.len();
    let next = AtomicUsize::new(0);
    let worker = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n_strips {
            break;
        }
        let mut s = strips[i].lock().expect("npu pack strip");
        wfmt::pack_strip(fmt, &mut **s, q8, o, k, kc, perm, chunk, ot0 + i);
    };
    let n = threads.clamp(1, n_strips.max(1));
    if n == 1 && cpus.is_empty() {
        worker();
        Ok(1)
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n).map(|i| {
                let worker = &worker;
                s.spawn(move || {
                    if !cpus.is_empty() {
                        let cpu = cpus[i % cpus.len()];
                        sched::pin_current_thread(cpu).map_err(|e| format!("npu pack: pin CPU {cpu}: {e}"))?;
                        if sched::allowed_cpus() != 1 {
                            return Err(format!("npu pack: CPU {cpu} affinity was not applied"));
                        }
                    }
                    worker();
                    Ok::<(), String>(())
                })
            }).collect();
            join_pack_workers(handles)?;
            Ok(n)
        })
    }
}

/// Drain every scoped worker before returning an error. Early return would leave a
/// later panic to `thread::scope`, which re-panics instead of returning our error.
fn join_pack_workers<'scope>(handles: Vec<std::thread::ScopedJoinHandle<'scope, Result<(), String>>>) -> Result<(), String> {
    let mut first_error = None;
    for handle in handles {
        let result = handle.join().unwrap_or_else(|_| Err("npu pack: worker panicked".to_string()));
        if let Err(error) = result {
            if first_error.is_none() {
                first_error = Some(error);
            }
        }
    }
    first_error.map_or(Ok(()), Err)
}

/// An activation in accelerator tiles (`ptr` is the buffer base, the view starts at `buf_off`).
#[derive(Clone, Copy, Debug)]
pub struct NpuAct {
    ptr: *mut u16,
    pub lay: ActLayout,
    pub buf_off: u32,
    pub slot: u32,
    /// the session whose buffer holds this activation
    pub session: usize,
}

// SAFETY: the pointer is into a backend buffer that lives as long as the backend (owned by NpuCtx).
unsafe impl Send for NpuAct {}
unsafe impl Sync for NpuAct {}

impl NpuAct {
    /// Write f32 `row` of token `t` into columns `k0 ..`. Concurrent calls for different tokens are fine.
    ///
    /// # Safety
    /// No run may read this activation while the write happens, and `k0`/`row.len()` are multiples of 32.
    pub unsafe fn put(&self, t: usize, k0: usize, row: &[f32]) {
        tiles::write_act_row(self.ptr.add(self.buf_off as usize / 2), &self.lay, t, k0, row)
    }

    /// Write one whole 32-token tile, byte-identical to 32 `put` calls.
    ///
    /// # Safety
    /// As `put`: no device read races, columns are multiples of32 and rows are
    /// exactly32*len elements. Padding tokens must be supplied as zeros.
    pub unsafe fn put_tile_rows(&self, rt:usize, k0:usize, len:usize, rows:&[f32]) {
        tiles::write_act_tile_rows(self.ptr.add(self.buf_off as usize/2), &self.lay, rt, k0, len, rows)
    }
}

/// The output of a run (valid until the next run on the same session).
#[derive(Clone, Copy, Debug)]
pub struct NpuOut {
    ptr: *const u16,
    pub lay: OutLayout,
}

// SAFETY: as NpuAct.
unsafe impl Send for NpuOut {}
unsafe impl Sync for NpuOut {}

impl NpuOut {
    /// f32 values of columns `n0 .. n0 + out.len()` of token `t` (the K chunks summed).
    ///
    /// # Safety
    /// The run that produced this output must have finished and no other run may start meanwhile.
    pub unsafe fn row(&self, t: usize, n0: usize, out: &mut [f32]) {
        tiles::read_out_row(self.ptr, &self.lay, t, n0, out)
    }

    /// f32 values of columns `n0 .. n0 + len` of the 32 tokens `32*rt .. 32*rt+32` (`out[r * len + i]` is token
    /// `32*rt + r`, column `n0 + i`), bit-identical to 32 calls of [`NpuOut::row`]. `out.len() == 32 * len`.
    /// Tokens beyond the real count are padding: the caller must not deliver them.
    ///
    /// # Safety
    /// As [`NpuOut::row`]; `rt < lay.t_pad / 32` and `n0 + len <= lay.o`.
    pub unsafe fn tile_rows(&self, rt: usize, n0: usize, len: usize, out: &mut [f32]) {
        tiles::read_out_tile_rows(self.ptr, &self.lay, rt, n0, len, out)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn pick_c_minimises_weight_passes() {
        use crate::wfmt::WFmt::{Fp16, W8};
        let v = 8 << 20;
        // K = 3072, 1536 tokens: two passes whatever C is -> keep the widest block
        assert_eq!(super::pick_c(v, 1536, 3072, 9216, Fp16), 4);
        // K = 3072, 1024 tokens (32 row tiles): C = 4 holds 31 row tiles (two passes), C = 2 all 32 (one pass)
        assert_eq!(super::pick_c(v, 1024, 3072, 9216, Fp16), 2);
        // 512 tokens: everything fits at C = 4
        assert_eq!(super::pick_c(v, 512, 3072, 9216, Fp16), 4);
        // K = 9216: C = 4 leaves 6 row tiles, C = 2 ten, C = 1 twelve; for 32 row tiles: 6 / 4 / 3 passes
        assert_eq!(super::pick_c(v, 1024, 9216, 3072, Fp16), 1);
        // K = 12288, 48 row tiles: C = 4 -> 2 rows (24 passes), C = 1 -> 8 rows (6 passes)
        assert_eq!(super::pick_c(v, 1536, 12288, 3072, Fp16), 1);
        // W8 staging is smaller than a second fp16 block: K = 3072, 1536 tokens stays at 4
        assert_eq!(super::pick_c(v, 1536, 3072, 27648, W8), 4);
        // tiny shapes and a VTCM that fits nothing
        assert_eq!(super::pick_c(v, 32, 32, 32, Fp16), 1);
        assert_eq!(super::pick_c(4096, 32, 32, 32, Fp16), 0);
    }

    use super::*;
    use crate::emu::Emu;

    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Random Q8_0 bytes of an `o x k` linear: a scale and 32 codes per block.
    fn random_q8(rng: &mut Rng, o: usize, k: usize) -> Vec<u8> {
        let mut q8 = Vec::with_capacity(o * (k / 32) * 34);
        for _ in 0..o * (k / 32) {
            let d = tiles::f32_to_half(((rng.below(2000) as f32) - 1000.0) * 1e-4);
            q8.extend_from_slice(&d.to_le_bytes());
            for _ in 0..32 {
                q8.push(rng.next_u64() as u8);
            }
        }
        q8
    }

    fn random_perm(rng: &mut Rng, o: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..o as u32).collect();
        for i in (1..o).rev() {
            p.swap(i, rng.below(i + 1));
        }
        p
    }

    fn emu_cfg(budget_bytes: usize, pack_threads: usize, weights: WFmt, max_blob_bytes: usize) -> NpuConfig {
        NpuConfig { budget_bytes, kc_max: 64, ycol: true, pack_threads, cpu_rows_pct: 0, weights, max_blob_bytes, audit: false }
    }

    fn emu_ctx(budget_bytes: usize, pack_threads: usize) -> NpuCtx {
        NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), emu_cfg(budget_bytes, pack_threads, WFmt::W8, 0)).expect("npu context")
    }

    /// The bytes of a placed blob, read through its pointer.
    fn blob_bytes(b: &NpuBlob, len: usize) -> Vec<u8> {
        // SAFETY: the blob lies in a backend buffer that lives as long as the context, and nothing writes it during the test.
        unsafe { std::slice::from_raw_parts(b.ptr, len).to_vec() }
    }

    #[test]
    fn add_linear_bytes_match_reference_packer() {
        let (o, k) = (96usize, 256usize);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let q8 = random_q8(&mut rng, o, k);
        let perm = random_perm(&mut rng, o);
        let mut want = vec![0u8; tiles::w8_bytes(o, k)];
        tiles::pack_w8_from_q8_0(&mut want, &q8, o, k, 64, Some(&perm[..]));
        let blob = tiles::w8_bytes(o, 64);
        for threads in [1usize, 4] {
            let mut ctx = emu_ctx(1 << 30, threads);
            assert_eq!(ctx.stats().allocated_bytes, 2 * ACT_BYTES + OUT_BYTES);
            assert_eq!(ctx.stats().weight_capacity_bytes, 0);
            let lin = ctx.add_linear("t", &q8, o, k, Some(&perm[..])).expect("add_linear").expect("placed on the accelerator");
            assert_eq!((lin.kc, lin.blobs.len()), (64, 4));
            for c in 0..4 {
                let b = &lin.blobs[c];
                assert_eq!((b.chunk as usize, b.ot0 as usize, b.nt as usize), (c, 0, 3));
                assert_eq!(blob_bytes(b, blob), want[c * blob..(c + 1) * blob], "pack_threads {threads}, chunk {c}");
            }
            assert_eq!(ctx.stats().cpu_linears, 0);
            assert_eq!(ctx.stats().allocated_bytes, 2 * ACT_BYTES + OUT_BYTES + WEIGHT_BYTES);
            assert_eq!(ctx.stats().weight_capacity_bytes, WEIGHT_BYTES);
            assert!(ctx.stats().weights_bytes < ctx.stats().weight_capacity_bytes);
            ctx.reset_step_stats();
            assert_eq!(ctx.stats().allocated_bytes, 2 * ACT_BYTES + OUT_BYTES + WEIGHT_BYTES);
        }
    }

    #[test]
    fn pack_workers_escape_pool_caller_mask_without_changing_it() {
        let status = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        let allowed = status.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:")).unwrap();
        let cpus = sched::parse_cpu_list(allowed.trim()).unwrap();
        if cpus.len() < 2 {
            eprintln!("pack affinity regression needs two allowed CPUs");
            return;
        }
        let pool = sched::Pool::new(&cpus[..2], std::time::Duration::ZERO).unwrap();
        let pinned_status = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        let pinned = pinned_status.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:")).unwrap().trim().to_owned();
        assert_eq!(pinned, cpus[0].to_string());
        assert_eq!(sched::allowed_cpus(), 1, "pool caller must start pinned");
        let mut ctx = NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), emu_cfg(1 << 30, 0, WFmt::Fp16, 0))
            .unwrap().with_pack_cpus(pool.cpus());
        assert_eq!(ctx.pack_threads(), 2, "auto packing uses pool CPUs, not the one-CPU caller mask");
        let mut rng = Rng(0x255a_aa55_1234_abcd);
        let q8 = random_q8(&mut rng, 96, 64);
        let lin = ctx.add_linear("affinity", &q8, 96, 64, None).unwrap().unwrap();
        assert_eq!(ctx.stats().pack_threads, 2);
        assert_eq!(sched::allowed_cpus(), 1, "packing must preserve the caller's CPU0 role");
        let after = std::fs::read_to_string("/proc/thread-self/status").unwrap();
        assert_eq!(after.lines().find_map(|l|l.strip_prefix("Cpus_allowed_list:")).unwrap().trim(), pinned);
        let sb = WFmt::Fp16.strip_bytes(64);
        let mut want = vec![0; sb * 3];
        for ot in 0..3 {
            wfmt::pack_strip(WFmt::Fp16, &mut want[ot * sb..(ot + 1) * sb], &q8, 96, 64, 64, None, 0, ot);
        }
        assert_eq!(blob_bytes(&lin.blobs[0], want.len()), want);
    }

    #[test]
    fn pack_joins_later_panics_and_workers_after_first_error() {
        let finished = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let early = scope.spawn(|| Err("pin failed first".to_string()));
            let panic = scope.spawn(|| -> Result<(), String> { panic!("packing assertion fixture") });
            let late = scope.spawn(|| { finished.store(true, Ordering::SeqCst); Ok(()) });
            let result = join_pack_workers(vec![early, panic, late]);
            assert_eq!(result, Err("pin failed first".into()));
            assert!(finished.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn add_linear_parts_match_reference_packer() {
        let (o, k) = (96usize, 256usize);
        let mut rng = Rng(0x0bad_cafe_1234_5678);
        let q8 = random_q8(&mut rng, o, k);
        let perm = random_perm(&mut rng, o);
        for fmt in [WFmt::W8, WFmt::Fp16] {
            let sb = fmt.strip_bytes(64);
            // two strips per blob: parts of 2 and 1 output tiles
            let mut ctx = NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), emu_cfg(1 << 30, 3, fmt, 2 * sb)).expect("npu context");
            let lin = ctx.add_linear("t", &q8, o, k, Some(&perm[..])).expect("add_linear").expect("placed");
            assert_eq!(lin.blobs.len(), 4 * 2, "4 chunks x 2 parts");
            // reference: the whole linear, strip by strip (chunk-major, then output tile)
            let mut want = vec![0u8; 4 * 3 * sb];
            for c in 0..4 {
                for ot in 0..3 {
                    let off = (c * 3 + ot) * sb;
                    wfmt::pack_strip(fmt, &mut want[off..off + sb], &q8, o, k, 64, Some(&perm[..]), c, ot);
                }
            }
            for b in &lin.blobs {
                let off = (b.chunk as usize * 3 + b.ot0 as usize) * sb;
                let len = b.nt as usize * sb;
                assert_eq!(blob_bytes(b, len), want[off..off + len], "{fmt:?} chunk {} ot0 {}", b.chunk, b.ot0);
            }
            assert_eq!(ctx.stats().parts, 2);
            assert_eq!(ctx.stats().blobs, 8);
        }
    }

    struct Q8Rows<'a> { raw: &'a [u8], o: usize, k: usize, perm: &'a [u32] }
    impl WeightRows for Q8Rows<'_> {
        fn shape(&self) -> (usize, usize) { (self.o, self.k) }
        fn read_row(&self, row: usize, dst: &mut [f32]) -> Result<(), String> {
            let src = self.perm[row] as usize;
            for (i, out) in dst.chunks_mut(32).enumerate() {
                let b = &self.raw[(src * (self.k / 32) + i) * 34..][..34];
                let d = tiles::half_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for (v, q) in out.iter_mut().zip(&b[2..]) { *v = d * (*q as i8) as f32; }
            }
            Ok(())
        }
    }

    #[test]
    fn logical_rows_match_q8_pack_across_chunks_parts_and_workers() {
        let (o, k) = (96, 256);
        let mut rng = Rng(0xaabb_ccdd_1122_3344);
        let q8 = random_q8(&mut rng, o, k);
        let perm = random_perm(&mut rng, o);
        let rows = Q8Rows { raw: &q8, o, k, perm: &perm };
        let sb = WFmt::Fp16.strip_bytes(64);
        for workers in [1, 3] {
            let cfg = emu_cfg(1 << 30, workers, WFmt::Fp16, 2 * sb);
            let mut ctx = NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), cfg).unwrap();
            let lin = ctx.add_linear_rows("mixed", &rows).unwrap().unwrap();
            for b in &lin.blobs {
                let mut want = vec![0; b.nt as usize * sb];
                for i in 0..b.nt as usize {
                    wfmt::pack_fp16_strip(&mut want[i * sb..(i + 1) * sb], &q8, o, k, 64,
                        Some(&perm), b.chunk as usize, b.ot0 as usize + i);
                }
                assert_eq!(blob_bytes(b, want.len()), want);
            }
            assert_eq!(ctx.stats().weights_bytes, o * k * 2);
        }
    }

    struct BadRows { shape: (usize, usize), value: f32, fail: bool }
    impl WeightRows for BadRows {
        fn shape(&self) -> (usize, usize) { self.shape }
        fn read_row(&self, _: usize, dst: &mut [f32]) -> Result<(), String> {
            if self.fail { return Err("source row failure".into()); }
            dst.fill(self.value); Ok(())
        }
    }

    #[test]
    fn logical_rows_fail_closed_on_source_nonfinite_and_shape_overflow() {
        for (value, fail) in [(0.0, true), (f32::NAN, false), (f32::INFINITY, false), (65536.0, false)] {
            let mut ctx = NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), emu_cfg(1 << 30, 3, WFmt::Fp16, 0)).unwrap();
            assert!(ctx.add_linear_rows("bad", &BadRows { shape: (96, 64), value, fail }).is_err());
            assert_eq!(ctx.stats().linears, 0);
            assert!(ctx.add_linear_rows("after-error", &BadRows { shape: (32, 64), value: 0.0, fail: false }).is_err());
        }
        let mut ctx = NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), emu_cfg(1 << 30, 1, WFmt::Fp16, 0)).unwrap();
        assert!(ctx.add_linear_rows("overflow", &BadRows { shape: (usize::MAX - 31, 64), value: 0.0, fail: false }).is_err());
        assert!(ctx.add_linear_rows("unaligned", &BadRows { shape: (33, 64), value: 0.0, fail: false }).unwrap().is_none());
    }

    #[test]
    fn add_linear_strip_larger_than_blob_is_an_error() {
        let (o, k) = (64usize, 256usize);
        let mut rng = Rng(0x7777_1111_2222_3333);
        let q8 = random_q8(&mut rng, o, k);
        let mut ctx = NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), emu_cfg(1 << 30, 1, WFmt::Fp16, 1024)).expect("npu context");
        assert!(ctx.add_linear("t", &q8, o, k, None).is_err());
    }

    #[test]
    fn add_linear_over_budget_counts_cpu_linear() {
        let (o, k) = (96usize, 256usize);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let q8 = random_q8(&mut rng, o, k);
        let mut ctx = emu_ctx(0, 2);
        assert!(ctx.add_linear("t", &q8, o, k, None).expect("add_linear").is_none());
        let s = ctx.stats();
        assert_eq!((s.linears, s.cpu_linears, s.weights_bytes), (0, 1, 0));
    }

    #[test]
    fn ycol_false_is_rejected() {
        let cfg = NpuConfig { ycol: false, ..emu_cfg(1 << 30, 1, WFmt::W8, 0) };
        assert!(NpuCtx::new(Box::new(Emu::new(8 << 20, 1)), cfg).is_err());
    }

    #[test]
    fn fp16_step_is_the_fp16_grid_spacing() {
        assert_eq!(fp16_step(1.0), 2f64.powi(-10));
        assert_eq!(fp16_step(1.9), 2f64.powi(-10));
        assert_eq!(fp16_step(2.0), 2f64.powi(-9));
        assert_eq!(fp16_step(-0.5), 2f64.powi(-11));
        assert_eq!(fp16_step(1e-6), 2f64.powi(-24));
        assert_eq!(fp16_step(0.0), 0.0);
    }
}
