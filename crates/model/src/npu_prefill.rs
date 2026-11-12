//! Opt-in synchronous dense prefill. Immutable scheduling, original GGUF inputs,
//! and a fault latch; decode never uses this module. No environment auto-enable.
use crate::{
    npu_rows::GgufRows,
    plan::{Mixer, Plan, Tid},
    weights::WeightStore,
};
use oxn::{
    linear::{NpuConfig, NpuCtx, NpuLinear, NpuStats, WeightRows},
    wfmt::WFmt,
    Backend,
};
use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrentMode {
    Hmx,
    Native,
    F32Prefill,
    F32Both,
}
impl RecurrentMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Hmx => "hmx",
            Self::Native => "native",
            Self::F32Prefill => "f32-prefill",
            Self::F32Both => "f32-both",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "hmx" => Ok(Self::Hmx),
            "native" => Ok(Self::Native),
            "f32-prefill" => Ok(Self::F32Prefill),
            "f32-both" => Ok(Self::F32Both),
            _ => Err("npu recurrent mode hmx|native|f32-prefill|f32-both".into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub min_rows: usize,
    pub max_rows: usize,
    pub weight_budget: usize,
    pub pack_threads: usize,
    /// Diagnostic only: sampled original-f32 rows, never changes outputs.
    pub audit_original: bool,
    pub recurrent: RecurrentMode,
    pub recurrent_mask: u64,
    pub overlap_shared: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            min_rows: 32,
            max_rows: 1024,
            weight_budget: 3 << 30,
            pack_threads: 4,
            audit_original: false,
            recurrent: RecurrentMode::Native,
            recurrent_mask: u64::MAX,
            overlap_shared: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub prepared: usize,
    pub skipped: usize,
    pub calls: u64,
    pub rows: u64,
    pub host_scratch_bytes: usize,
    pub reserved_dma_bytes: usize,
    pub occupied_weight_bytes: usize,
    pub first_error: Option<String>,
    pub accelerator: Option<NpuStats>,
    pub original_audit: Vec<Audit>,
    pub recurrent_mode: String,
    pub recurrent_mask: u64,
    pub cpu_f32_matrices: usize,
    /// Logical fp16 GEMM operations (2 per MAC), successful accelerator calls only.
    pub gemm_flops: u64,
    /// Non-overlapping host wall phases, under the projection mutex. Includes failed attempts.
    pub input_s: f64,
    pub device_run_s: f64,
    pub output_s: f64,
    pub commit_s: f64,
    pub projection_s: f64,
    /// Waiting for the single backend's projection mutex; thread-seconds, may overlap.
    pub lock_wait_s: f64,
    pub overlap_shared: bool,
    pub shared_chain_units: u64,
    pub shared_chain_s: f64,
    pub expert_units_started_during_shared: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Audit {
    pub tensor: String,
    pub k: usize,
    pub o: usize,
    pub samples: u64,
    pub squared_error: f64,
    pub squared_reference: f64,
    pub max_error: f64,
    pub max_input: f32,
    pub max_output: f32,
    pub max_reference: f64,
}
struct AuditRows {
    rows: Vec<(usize, Vec<f32>)>,
    stat: Audit,
}
impl AuditRows {
    fn prepare(name: &str, source: &dyn WeightRows) -> Result<Self, String> {
        let (o, k) = source.shape();
        let count = o.min(16);
        let mut rows = Vec::new();
        for i in 0..count {
            let id = if count == 1 {
                0
            } else {
                i * (o - 1) / (count - 1)
            };
            let mut row = vec![0.; k];
            source.read_row(id, &mut row)?;
            rows.push((id, row));
        }
        Ok(Self {
            rows,
            stat: Audit {
                tensor: name.into(),
                k,
                o,
                ..Default::default()
            },
        })
    }
    unsafe fn observe(&mut self, n: usize, x: *const f32, y: &[f32]) {
        for t in [0, n / 2, n - 1].into_iter().collect::<BTreeSet<_>>() {
            let input = std::slice::from_raw_parts(x.add(t * self.stat.k), self.stat.k);
            self.stat.max_input = self
                .stat
                .max_input
                .max(input.iter().map(|v| v.abs()).fold(0., f32::max));
            for (c, w) in &self.rows {
                let reference: f64 = input
                    .iter()
                    .zip(w)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum();
                let output = y[t * self.stat.o + *c];
                let diff = output as f64 - reference;
                self.stat.samples += 1;
                self.stat.squared_error += diff * diff;
                self.stat.squared_reference += reference * reference;
                self.stat.max_error = self.stat.max_error.max(diff.abs());
                self.stat.max_output = self.stat.max_output.max(output.abs());
                self.stat.max_reference = self.stat.max_reference.max(reference.abs());
            }
        }
    }
}

struct Inner {
    ctx: Option<NpuCtx>,
    output: Vec<f32>,
    audit: HashMap<usize, AuditRows>,
    stats: Stats,
}
pub(crate) struct Prefill {
    cfg: Config,
    // Keys are the immutable CPU WeightRef bases, not pointers into original GGUF.
    // The same entry remains scheduled as one whole unit after a fault, with CPU
    // fallback; no thread may see a different unit count within a phase.
    linears: HashMap<usize, NpuLinear>,
    cpu_f32: Option<crate::cpu_recurrent::Rows>,
    shared_active: AtomicBool,
    experts_during_shared: AtomicU64,
    inner: Mutex<Inner>,
}

/// Bound reserved capacity, including slab slack and activation/output buffers,
/// before forwarding an allocation to the device. Occupied tensor bytes alone
/// do not bound the PD virtual-address footprint. No allocation is freed/reused
/// during preparation, so this ledger is monotonic until the backend is dropped.
struct BoundedBackend {
    inner: Box<dyn Backend>,
    reserved: Vec<usize>,
}
impl Backend for BoundedBackend {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn sessions(&self) -> usize {
        self.inner.sessions()
    }
    fn vtcm_bytes(&self) -> u32 {
        self.inner.vtcm_bytes()
    }
    fn alloc(&mut self, s: usize, bytes: usize) -> Result<oxn::Buf, String> {
        let used = *self
            .reserved
            .get(s)
            .ok_or("npu prefill: bad allocation session")?;
        let next = used
            .checked_add(bytes)
            .ok_or("npu prefill: reserved capacity overflow")?;
        if next > (7usize << 29) {
            return Err("npu prefill: local reserved-capacity limit3.5GiB/PD".into());
        }
        let b = self.inner.alloc(s, bytes)?;
        self.reserved[s] = next;
        Ok(b)
    }
    fn gemm(&mut self, s: usize, a: &oxn::GemmArgs) -> Result<oxn::GemmStats, String> {
        self.inner.gemm(s, a)
    }
}

fn candidates(plan: &Plan, cfg: Config) -> BTreeSet<Tid> {
    let mut ids = BTreeSet::new();
    for (index, l) in plan.layers.iter().enumerate() {
        match l.mixer {
            Mixer::Recurrent { qkv, z, out, .. }
                if cfg.recurrent == RecurrentMode::Hmx
                    || cfg.recurrent_mask & (1u64 << index) == 0 =>
            {
                ids.extend([qkv, z, out])
            }
            Mixer::Recurrent { .. } => {}
            Mixer::Attention { q, k, v, out, .. } => ids.extend([q, k, v, out]),
        }
        ids.extend([l.gate_shexp, l.up_shexp, l.down_shexp]);
    }
    ids
}

pub(crate) struct SharedOverlap<'a> {
    p: &'a Prefill,
    start: Instant,
}
impl Drop for SharedOverlap<'_> {
    fn drop(&mut self) {
        self.p.shared_active.store(false, Ordering::Release);
        let mut g = self.p.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.stats.shared_chain_units += 1;
        g.stats.shared_chain_s += self.start.elapsed().as_secs_f64();
    }
}
impl Prefill {
    pub(crate) fn shared_overlap(&self, keys: [usize; 3], n: usize) -> bool {
        self.cfg.overlap_shared && keys.into_iter().all(|k| self.selected(k, n))
    }
    pub(crate) fn begin_shared_overlap(&self) -> SharedOverlap<'_> {
        assert!(
            !self.shared_active.swap(true, Ordering::AcqRel),
            "one shared branch per prefill phase"
        );
        SharedOverlap {
            p: self,
            start: Instant::now(),
        }
    }
    pub(crate) fn note_expert_unit(&self) {
        if self.shared_active.load(Ordering::Acquire) {
            self.experts_during_shared.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub(crate) fn prepare(
        plan: &Plan,
        ws: &WeightStore,
        g: &gguf::Gguf,
        model_sha: &str,
        backend: Box<dyn Backend>,
        cfg: Config,
        pack_cpus: &[usize],
    ) -> Result<Self, String> {
        if cfg.min_rows < 4
            || cfg.max_rows < cfg.min_rows
            || cfg.max_rows > 1024
            || cfg.weight_budget > 3 << 30
            || cfg.weight_budget == 0
            || cfg.pack_threads > 12
        {
            return Err("npu prefill: invalid bounded configuration".into());
        }
        if model_sha.len() != 64
            || !model_sha.bytes().all(|x| x.is_ascii_hexdigit())
            || model_sha != ws.info.model_sha256
        {
            return Err("npu prefill: source/CPU model identity mismatch".into());
        }
        if plan.layers.len() > 64 {
            return Err("recurrent ablation supports <=64 layers".into());
        }
        let cpu_f32 = if matches!(
            cfg.recurrent,
            RecurrentMode::F32Prefill | RecurrentMode::F32Both
        ) {
            Some(crate::cpu_recurrent::Rows::new(
                g,
                plan,
                ws,
                cfg.recurrent_mask,
            )?)
        } else {
            None
        };
        let cpu_f32_matrices = cpu_f32.as_ref().map_or(0, |r| r.count());
        let ns = backend.sessions();
        if ns != 1 {
            return Err("npu prefill: initial proof supports exactly one PD".into());
        }
        let backend = Box::new(BoundedBackend {
            inner: backend,
            reserved: vec![0; ns],
        });
        let mut ctx = NpuCtx::new(
            backend,
            NpuConfig {
                budget_bytes: cfg.weight_budget,
                kc_max: 0,
                ycol: true,
                pack_threads: cfg.pack_threads,
                cpu_rows_pct: 0,
                weights: WFmt::Fp16,
                max_blob_bytes: 128 << 20,
                audit: false,
            },
        )?
        .with_pack_cpus(pack_cpus);
        let mut linears = HashMap::new();
        let mut audit = HashMap::new();
        let mut skipped = 0;
        for id in candidates(plan, cfg) {
            let t = g
                .header
                .tensors
                .get(id)
                .ok_or("npu prefill: tensor index outside source")?;
            let w = ws.get(id);
            if t.shape.len() != 2 || w.n_expert != 1 || w.is_f32() {
                skipped += 1;
                continue;
            }
            if t.shape[0] != w.k as u64 || t.shape[1] != w.rows as u64 {
                return Err("npu prefill: source shape mismatch".into());
            }
            let rows = GgufRows::new(t, g.tensor_data(t))?;
            match ctx.add_linear_rows(&t.name, &rows)? {
                Some(l) => {
                    if cfg.audit_original {
                        audit.insert(w.ptr, AuditRows::prepare(&t.name, &rows)?);
                    }
                    if linears.insert(w.ptr, l).is_some() {
                        return Err("npu prefill: aliased CPU tensors".into());
                    }
                }
                None => skipped += 1,
            }
            g.release_range(t, 0, t.bytes() as usize);
            g.drop_cache(t);
        }
        if linears.is_empty() {
            return Err("npu prefill: no eligible linears".into());
        }
        let s = ctx.stats();
        Ok(Self {
            cfg,
            linears,
            cpu_f32,
            shared_active: AtomicBool::new(false),
            experts_during_shared: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                ctx: Some(ctx),
                output: Vec::new(),
                audit,
                stats: Stats {
                    overlap_shared: cfg.overlap_shared,
                    recurrent_mode: cfg.recurrent.name().into(),
                    recurrent_mask: cfg.recurrent_mask,
                    cpu_f32_matrices,
                    prepared: s.linears,
                    skipped,
                    reserved_dma_bytes: s.allocated_bytes,
                    occupied_weight_bytes: s.weights_bytes,
                    ..Default::default()
                },
            }),
        })
    }

    pub(crate) fn selected(&self, key: usize, n: usize) -> bool {
        n >= self.cfg.min_rows && n <= self.cfg.max_rows && self.linears.contains_key(&key)
    }

    pub(crate) fn cpu_f32_selected(&self, key: usize, decode: bool) -> bool {
        (!decode || self.cfg.recurrent == RecurrentMode::F32Both)
            && self.cpu_f32.as_ref().is_some_and(|r| r.contains(key))
    }
    /// Same exclusive output ownership as the CPU dense unit / decode matvec.
    pub(crate) unsafe fn cpu_f32(
        &self,
        key: usize,
        decode: bool,
        n: usize,
        c0: usize,
        c1: usize,
        x: *const f32,
        out: *mut f32,
        os: usize,
        add: *const f32,
    ) -> bool {
        self.cpu_f32_selected(key, decode)
            && self
                .cpu_f32
                .as_ref()
                .unwrap()
                .project(key, n, c0, c1, x, out, os, add)
    }

    /// This is a full projection transaction: output is checked before committing
    /// any row (important when residual `add` aliases the destination). The next
    /// phase barrier is outside this call and cannot consume a partial result.
    ///
    /// # Safety
    /// Caller owns all n output rows at stride os and keeps n*K input values and
    /// optional n output-width residual rows immutable until return. No other
    /// projection writes these output elements. Selection must be stable per phase.
    pub(crate) unsafe fn run(
        &self,
        key: usize,
        n: usize,
        x: *const f32,
        out: *mut f32,
        os: usize,
        add: *const f32,
    ) -> bool {
        let Some(lin) = self.linears.get(&key) else {
            return false;
        };
        let wait_start = Instant::now();
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.stats.lock_wait_s += wait_start.elapsed().as_secs_f64();
        let projection_start = Instant::now();
        let mut phase = [0f64; 4];
        if inner.ctx.is_none() || inner.stats.first_error.is_some() {
            return false;
        }
        let result = (|| -> Result<(), String> {
            let Inner {
                ctx, output, audit, ..
            } = &mut *inner;
            let ctx = ctx.as_ref().ok_or("npu prefill: context inactive")?;
            if !self.selected(key, n) || os < lin.o {
                return Err("npu prefill: projection bounds".into());
            }
            let t = Instant::now();
            let act = ctx.act(lin.session, 0, n, lin.k, lin.kc)?;
            let input = std::slice::from_raw_parts(x, n * lin.k);
            if input.iter().any(|v| !v.is_finite() || v.abs() > 65504.0) {
                return Err("npu prefill: nonfinite/fp16-range input".into());
            }
            for rt in 0..n / 32 {
                act.put_tile_rows(rt, 0, lin.k, &input[rt * 32 * lin.k..(rt + 1) * 32 * lin.k]);
            }
            for t in n / 32 * 32..n {
                act.put(t, 0, &input[t * lin.k..(t + 1) * lin.k]);
            }
            phase[0] = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let y = ctx.run(lin, &act)?;
            phase[1] = t.elapsed().as_secs_f64();
            let t = Instant::now();
            output.resize(
                n.checked_mul(lin.o)
                    .ok_or("npu prefill: output size overflow")?,
                0.0,
            );
            for rt in 0..n / 32 {
                y.tile_rows(
                    rt,
                    0,
                    lin.o,
                    &mut output[rt * 32 * lin.o..(rt + 1) * 32 * lin.o],
                );
            }
            for t in n / 32 * 32..n {
                y.row(t, 0, &mut output[t * lin.o..(t + 1) * lin.o]);
            }
            if output.iter().any(|v| !v.is_finite()) {
                return Err("npu prefill: nonfinite output".into());
            }
            if let Some(a) = audit.get_mut(&key) {
                a.observe(n, x, output);
            }
            phase[2] = t.elapsed().as_secs_f64();
            let t = Instant::now();
            // All checks have passed; these stores cannot fail halfway through.
            for t in 0..n {
                if add.is_null() {
                    std::ptr::copy_nonoverlapping(
                        output.as_ptr().add(t * lin.o),
                        out.add(t * os),
                        lin.o,
                    );
                } else {
                    for c in 0..lin.o {
                        *out.add(t * os + c) = output[t * lin.o + c] + *add.add(t * os + c);
                    }
                }
            }
            phase[3] = t.elapsed().as_secs_f64();
            Ok(())
        })();
        inner.stats.input_s += phase[0];
        inner.stats.device_run_s += phase[1];
        inner.stats.output_s += phase[2];
        inner.stats.commit_s += phase[3];
        inner.stats.projection_s += projection_start.elapsed().as_secs_f64();
        inner.stats.host_scratch_bytes = inner.output.capacity() * std::mem::size_of::<f32>();
        match result {
            Ok(()) => {
                inner.stats.gemm_flops += 2 * n as u64 * lin.k as u64 * lin.o as u64;
                inner.stats.calls += 1;
                inner.stats.rows += n as u64;
                true
            }
            Err(e) => {
                eprintln!("oxe: npu-prefill disabled: {e}; overwrite projection on CPU");
                inner.stats.first_error = Some(e);
                // Backend errors are sticky. Keep ownership and conservative DMA
                // accounting until the exclusive engine owner explicitly releases
                // the context. No retry, recovery or unproven DMA=0 claim.
                false
            }
        }
    }

    pub(crate) fn stats(&self) -> Stats {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = g.stats.clone();
        s.expert_units_started_during_shared = self.experts_during_shared.load(Ordering::Relaxed);
        s.accelerator = g.ctx.as_ref().map(|c| c.stats());
        s.original_audit = g.audit.values().map(|a| a.stat.clone()).collect();
        s.original_audit.sort_by(|a, b| a.tensor.cmp(&b.tensor));
        s.host_scratch_bytes += g
            .audit
            .values()
            .flat_map(|a| a.rows.iter())
            .map(|(_, row)| row.capacity() * 4)
            .sum::<usize>();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxn::{linear::WeightRows, Buf, GemmArgs, GemmStats};
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    struct Rows;
    impl WeightRows for Rows {
        fn shape(&self) -> (usize, usize) {
            (32, 32)
        }
        fn read_row(&self, r: usize, d: &mut [f32]) -> Result<(), String> {
            for (c, x) in d.iter_mut().enumerate() {
                *x = ((r + c) % 7) as f32 * 0.125 - 0.375;
            }
            Ok(())
        }
    }
    #[test]
    fn original_audit_compares_raw_float_weights_and_detects_wrong_outputs() {
        let mut a = AuditRows::prepare("fixture", &Rows).unwrap();
        let x = vec![0.5f32; 2 * 32];
        let mut y = vec![0f32; 2 * 32];
        for t in 0..2 {
            for c in 0..32 {
                y[t * 32 + c] = (0..32)
                    .map(|k| 0.5 * (((c + k) % 7) as f32 * 0.125 - 0.375))
                    .sum();
            }
        }
        let before = y.clone();
        unsafe {
            a.observe(2, x.as_ptr(), &y);
        }
        assert_eq!(a.stat.samples, 32);
        assert_eq!(a.stat.squared_error, 0.);
        assert_eq!(y, before);
        y[0] += 1.;
        unsafe {
            a.observe(2, x.as_ptr(), &y);
        }
        assert_eq!(a.stat.max_error, 1.);
        assert_eq!(a.stat.squared_error, 1.);
    }
    struct NoCalls;
    impl Backend for NoCalls {
        fn name(&self) -> &'static str {
            "no-calls"
        }
        fn sessions(&self) -> usize {
            1
        }
        fn vtcm_bytes(&self) -> u32 {
            8 << 20
        }
        fn alloc(&mut self, _: usize, _: usize) -> Result<Buf, String> {
            panic!("local admission must not call backend")
        }
        fn gemm(&mut self, _: usize, _: &GemmArgs) -> Result<GemmStats, String> {
            unreachable!()
        }
    }
    #[test]
    fn reserved_capacity_is_refused_before_backend_or_mapping() {
        let mut b = BoundedBackend {
            inner: Box::new(NoCalls),
            reserved: vec![7usize << 29],
        };
        assert!(b.alloc(0, 4096).unwrap_err().contains("reserved-capacity"));
        assert!(b.alloc(0, usize::MAX).unwrap_err().contains("overflow"));
        assert!(b.alloc(1, 4096).unwrap_err().contains("session"));
        assert_eq!(b.reserved, vec![7usize << 29]);
    }
    struct FaultBackend {
        emu: oxn::emu::Emu,
        fail: Arc<AtomicBool>,
        calls: Arc<AtomicUsize>,
    }
    impl Backend for FaultBackend {
        fn name(&self) -> &'static str {
            "fault-emulator"
        }
        fn sessions(&self) -> usize {
            1
        }
        fn vtcm_bytes(&self) -> u32 {
            8 << 20
        }
        fn alloc(&mut self, s: usize, n: usize) -> Result<Buf, String> {
            self.emu.alloc(s, n)
        }
        fn gemm(&mut self, s: usize, a: &GemmArgs) -> Result<GemmStats, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let result = self.emu.gemm(s, a)?;
            // The backend wrote its private output before failing: committing it
            // to live engine state would be a partial-result bug.
            if self.fail.load(Ordering::Relaxed) {
                Err("injected post-write backend failure".into())
            } else {
                Ok(result)
            }
        }
    }
    #[test]
    fn projection_commits_atomically_then_latches_without_changing_schedule() {
        let fail = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = FaultBackend {
            emu: oxn::emu::Emu::new(8 << 20, 1),
            fail: fail.clone(),
            calls: calls.clone(),
        };
        let mut ctx = NpuCtx::new(
            Box::new(backend),
            NpuConfig {
                budget_bytes: 128 << 20,
                kc_max: 0,
                ycol: true,
                pack_threads: 1,
                cpu_rows_pct: 0,
                weights: WFmt::Fp16,
                max_blob_bytes: 0,
                audit: false,
            },
        )
        .unwrap();
        let linear = ctx.add_linear_rows("fixture", &Rows).unwrap().unwrap();
        let st = ctx.stats();
        let p = Prefill {
            cfg: Config::default(),
            linears: HashMap::from([(1, linear)]),
            cpu_f32: None,
            shared_active: AtomicBool::new(false),
            experts_during_shared: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                ctx: Some(ctx),
                output: Vec::new(),
                audit: HashMap::new(),
                stats: Stats {
                    prepared: 1,
                    reserved_dma_bytes: st.allocated_bytes,
                    ..Default::default()
                },
            }),
        };
        assert!(!p.selected(1, 1));
        assert!(p.selected(1, 32));
        assert!(!p.selected(1, 1025));
        let x = vec![1.0; 33 * 32];
        let mut output = vec![9.0; 33 * 40];
        let out = output.as_mut_ptr();
        // SAFETY: live whole projection, same destination/residual deliberately.
        assert!(unsafe { p.run(1, 33, x.as_ptr(), out, 40, out) });
        for t in 0..33 {
            for c in 0..32 {
                let mut w = vec![0.0; 32];
                Rows.read_row(c, &mut w).unwrap();
                assert_eq!(output[t * 40 + c], w.iter().sum::<f32>() + 9.0);
            }
            assert_eq!(&output[t * 40 + 32..(t + 1) * 40], &[9.0; 8]);
        }
        let expected = output[0] - 9.0;
        std::thread::scope(|scope| {
            let handles = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        let mut out = vec![0.0; 32 * 40];
                        // SAFETY: distinct local output per worker; shared input is immutable.
                        assert!(unsafe {
                            p.run(1, 32, x.as_ptr(), out.as_mut_ptr(), 40, std::ptr::null())
                        });
                        assert_eq!(out[0], expected);
                    })
                })
                .collect::<Vec<_>>();
            for h in handles {
                h.join().unwrap();
            }
        });
        output.fill(-123.0);
        fail.store(true, Ordering::Relaxed);
        assert!(!unsafe { p.run(1, 32, x.as_ptr(), out, 40, std::ptr::null()) });
        assert!(output.iter().all(|&v| v == -123.0));
        let before = calls.load(Ordering::Relaxed);
        assert!(!unsafe { p.run(1, 32, x.as_ptr(), out, 40, std::ptr::null()) });
        assert_eq!(
            calls.load(Ordering::Relaxed),
            before,
            "no retry after backend error"
        );
        assert!(
            p.selected(1, 32),
            "all workers must keep the same whole-projection unit after fault"
        );
        let st = p.stats();
        assert!(st.first_error.is_some());
        assert_eq!(st.calls, 3);
        assert_eq!(
            st.gemm_flops,
            (33 + 2 * 32) * 2 * 32 * 32,
            "failed or latched calls must not count as useful GEMM work"
        );
        assert_eq!(Config::default().recurrent, RecurrentMode::Native);
        assert!(
            st.reserved_dma_bytes > 0,
            "failed mapping ownership must not be reported as zero"
        );
    }
}
