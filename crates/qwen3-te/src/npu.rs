//! NPU linears for the text encoder through the shared oxn linear machinery (`oxn::linear::NpuCtx`):
//! weights placed once (resident for the whole prompt), activations and outputs moved as whole
//! 32-token tiles (`put_tile_rows` / `tile_rows`) by `threads` workers, attention and norms on the CPU.
//!
//! Weight formats: `Fp16` packs dequantized GGUF rows (`add_linear_rows`); `W8` hands the original
//! Q8_0 bytes to oxn (`add_linear`), which keeps the exact Q8_0 values at half the fp16 footprint.
//! A linear that does not fit the budget, or whose input leaves the fp16 range for one call, runs
//! on the CPU f32 path instead (counted, never silent).

use crate::cpu::{par_ranges, SendPtr};
use crate::encoder::{CpuLinears, CpuMode, Linears};
use crate::weights::{Kind, TeWeights};
use oxn::linear::{NpuConfig, NpuCtx, NpuLinear};
use oxn::wfmt::WFmt;
use oxn::Backend;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct NpuOpts {
    pub budget_bytes: usize,
    pub weights: WFmt,
    /// K chunk limit (0 = whole K per GEMM)
    pub kc_max: usize,
    /// packing workers (0 = all CPUs)
    pub pack_threads: usize,
    /// host workers for activation/output tile delivery
    pub io_threads: usize,
}

#[derive(Default, Debug, Clone)]
struct Stats {
    npu_calls: u64,
    cpu_calls: u64,
    range_fallbacks: u64,
    put_s: f64,
    run_s: f64,
    get_s: f64,
}

pub struct NpuLinears<'w, 'a> {
    inner: Mutex<NpuCtx>,
    placed: HashMap<(usize, Kind), NpuLinear>,
    cpu: CpuLinears<'w, 'a>,
    opts: NpuOpts,
    stats: Mutex<Stats>,
    pub place_s: f64,
    pub placed_bytes: u64,
}

/// Backend wrapper that counts every byte the accelerator allocates (activation, output and weight
/// buffers): the ACTUAL DMA allocation of the encoder, reported for admission.
pub struct CountingBackend {
    inner: Box<dyn Backend>,
    bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl CountingBackend {
    pub fn new(inner: Box<dyn Backend>, bytes: std::sync::Arc<std::sync::atomic::AtomicU64>) -> Self {
        CountingBackend { inner, bytes }
    }
}

impl Backend for CountingBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn sessions(&self) -> usize {
        self.inner.sessions()
    }
    fn vtcm_bytes(&self) -> u32 {
        self.inner.vtcm_bytes()
    }
    fn alloc(&mut self, session: usize, bytes: usize) -> Result<oxn::Buf, String> {
        let b = self.inner.alloc(session, bytes)?;
        self.bytes.fetch_add(b.len as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(b)
    }
    fn gemm(&mut self, session: usize, a: &oxn::GemmArgs) -> Result<oxn::GemmStats, String> {
        self.inner.gemm(session, a)
    }
}

/// Largest |x| an fp16 activation can hold.
const F16_MAX: f32 = 65504.0;

impl<'w, 'a> NpuLinears<'w, 'a> {
    /// Place every linear of `w` (in block order, Q K V O gate up down) that fits `opts.budget_bytes`.
    pub fn new(w: &'w TeWeights<'a>, backend: Box<dyn Backend>, opts: NpuOpts, cpu_threads: usize) -> Result<Self, String> {
        let t = Instant::now();
        let mut ctx = NpuCtx::new(
            backend,
            NpuConfig {
                budget_bytes: opts.budget_bytes,
                kc_max: opts.kc_max,
                ycol: true,
                pack_threads: opts.pack_threads,
                cpu_rows_pct: 0,
                weights: opts.weights,
                max_blob_bytes: 0,
                audit: false,
            },
        )?;
        let mut placed = HashMap::new();
        let mut placed_bytes = 0u64;
        for (l, lw) in w.layers.iter().enumerate() {
            for kd in Kind::ALL {
                let m = lw.mat(kd);
                let name = format!("blk.{l}.{}.weight", kd.gguf());
                let r = match (opts.weights, m.q8_0()) {
                    (WFmt::W8, Some(q8)) => ctx.add_linear(&name, q8, m.o(), m.k(), None)?,
                    (WFmt::W8, None) => None, // W8 takes Q8_0 sources only: stays on the CPU
                    (WFmt::Fp16, _) => ctx.add_linear_rows(&name, m)?,
                };
                if let Some(nl) = r {
                    placed_bytes += (m.o() * m.k()) as u64 * if opts.weights == WFmt::Fp16 { 2 } else { 1 };
                    placed.insert((l, kd), nl);
                }
            }
        }
        Ok(NpuLinears {
            inner: Mutex::new(ctx),
            placed,
            cpu: CpuLinears { w, threads: cpu_threads, mode: CpuMode::F32 },
            opts,
            stats: Mutex::new(Stats::default()),
            place_s: t.elapsed().as_secs_f64(),
            placed_bytes,
        })
    }

    pub fn placed_count(&self) -> usize {
        self.placed.len()
    }

    fn cpu_one(&self, layer: usize, kd: Kind, x: &[f32], n: usize, out: &mut Vec<f32>) -> Result<(), String> {
        self.stats.lock().unwrap().cpu_calls += 1;
        let mut tmp = vec![std::mem::take(out)];
        self.cpu.project(layer, &[kd], x, n, &mut tmp)?;
        *out = tmp.pop().unwrap();
        Ok(())
    }
}

impl Linears for NpuLinears<'_, '_> {
    fn name(&self) -> String {
        format!("npu-{}", if self.opts.weights == WFmt::Fp16 { "fp16" } else { "w8" })
    }

    fn project(&self, layer: usize, kinds: &[Kind], x: &[f32], n: usize, outs: &mut [Vec<f32>]) -> Result<(), String> {
        let in_range = x.iter().all(|v| v.is_finite() && v.abs() <= F16_MAX);
        let ctx = self.inner.lock().map_err(|_| "npu: context poisoned")?;
        let mut act: Option<(usize, usize, oxn::linear::NpuAct)> = None; // (session, kc, view)
        for (i, &kd) in kinds.iter().enumerate() {
            let Some(nl) = self.placed.get(&(layer, kd)) else {
                self.cpu_one(layer, kd, x, n, &mut outs[i])?;
                continue;
            };
            if !in_range {
                self.stats.lock().unwrap().range_fallbacks += 1;
                self.cpu_one(layer, kd, x, n, &mut outs[i])?;
                continue;
            }
            let k = nl.k;
            if x.len() != n * k {
                return Err(format!("npu: {}: input {} values for n {n} k {k}", nl.name, x.len()));
            }
            let mut st = Stats::default();
            // one activation per (session, kc): q/k/v and gate/up share it when placed together
            if act.as_ref().map_or(true, |a| a.0 != nl.session || a.1 != nl.kc) {
                let t = Instant::now();
                let a = ctx.act(nl.session, 0, n, k, nl.kc)?;
                let full = n / 32;
                par_ranges(full, self.opts.io_threads, |rts| {
                    for rt in rts {
                        // SAFETY: no run reads this buffer now; each worker writes its own token tiles.
                        unsafe { a.put_tile_rows(rt, 0, k, &x[rt * 32 * k..(rt + 1) * 32 * k]) };
                    }
                });
                for t in full * 32..n {
                    // SAFETY: as above, tail tokens one by one.
                    unsafe { a.put(t, 0, &x[t * k..(t + 1) * k]) };
                }
                st.put_s += t.elapsed().as_secs_f64();
                act = Some((nl.session, nl.kc, a));
            }
            let a = &act.as_ref().unwrap().2;
            let t = Instant::now();
            let y = ctx.run(nl, a)?;
            st.run_s += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let o = nl.o;
            let out = &mut outs[i];
            out.resize(n * o, 0.0);
            let full = n / 32;
            let op = SendPtr(out.as_mut_ptr());
            par_ranges(full, self.opts.io_threads, |rts| {
                let op = op;
                for rt in rts {
                    // SAFETY: the run has finished and no other run starts while we read (ctx is locked);
                    // each worker writes its own 32 output rows.
                    unsafe {
                        let dst = std::slice::from_raw_parts_mut(op.0.add(rt * 32 * o), 32 * o);
                        y.tile_rows(rt, 0, o, dst);
                    }
                }
            });
            for t in full * 32..n {
                // SAFETY: as above.
                unsafe { y.row(t, 0, &mut out[t * o..(t + 1) * o]) };
            }
            st.get_s += t.elapsed().as_secs_f64();
            if out.iter().any(|v| !v.is_finite()) {
                return Err(format!("npu: {}: non-finite output", nl.name));
            }
            let mut g = self.stats.lock().unwrap();
            g.npu_calls += 1;
            g.put_s += st.put_s;
            g.run_s += st.run_s;
            g.get_s += st.get_s;
        }
        Ok(())
    }

    fn summary(&self) -> serde_json::Value {
        let s = self.stats.lock().unwrap().clone();
        let ns = self.inner.lock().map(|c| c.stats()).ok();
        serde_json::json!({
            "placed_linears": self.placed.len(),
            "placed_bytes": self.placed_bytes,
            "place_s": self.place_s,
            "weights": if self.opts.weights == WFmt::Fp16 { "fp16" } else { "w8" },
            "npu_calls": s.npu_calls,
            "cpu_calls": s.cpu_calls,
            "range_fallbacks": s.range_fallbacks,
            "put_s": s.put_s,
            "run_s": s.run_s,
            "get_s": s.get_s,
            "gemm_s": ns.as_ref().map(|n| n.gemm_s),
            "dsp_us": ns.as_ref().map(|n| n.dsp_us),
            "sessions": ns.as_ref().map(|n| n.sessions),
        })
    }
}
