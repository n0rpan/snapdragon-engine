//! Window planner for NPU-resident linear layers.
//!
//! Pure planning arithmetic over the GGUF tensor table. Byte counts come from the
//! table; every timing value is a parameter in [`PlanConfig`] and every per-step time
//! produced here is an ESTIMATE, not a measurement.

use std::collections::HashMap;
use std::fmt::Write as _;

use image_dit::types::Dims;
use serde_json::{json, Value};

pub const MIB: u64 = 1 << 20;
pub const GIB: u64 = 1 << 30;

const Q8_0_BLOCK_ELEMS: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

/// One row of the tensor table (`tests/fixtures/klein-4b-q8.tensors.txt`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRow {
    pub name: String,
    pub dtype: String,
    /// GGUF order: `[in, out]` for linears.
    pub dims: Vec<u64>,
    /// Data offset relative to the start of the GGUF tensor data section.
    pub offset: u64,
    pub bytes: u64,
}

/// Parses `name<TAB>type<TAB>[d0, d1]<TAB>offset<TAB>bytes` rows.
/// Blank lines and `#` comments are skipped. Malformed rows panic with the line number.
pub fn parse_tensor_table(text: &str) -> Vec<TensorRow> {
    let mut rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert!(
            f.len() == 5,
            "tensor table line {}: expected 5 tab-separated fields, got {}",
            lineno + 1,
            f.len()
        );
        let dims: Vec<u64> = f[2]
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .split(',')
            .map(|d| {
                d.trim()
                    .parse::<u64>()
                    .unwrap_or_else(|_| panic!("tensor table line {}: bad dim {:?}", lineno + 1, d))
            })
            .collect();
        rows.push(TensorRow {
            name: f[0].to_string(),
            dtype: f[1].to_string(),
            dims,
            offset: f[3]
                .parse()
                .unwrap_or_else(|_| panic!("tensor table line {}: bad offset", lineno + 1)),
            bytes: f[4]
                .parse()
                .unwrap_or_else(|_| panic!("tensor table line {}: bad bytes", lineno + 1)),
        });
    }
    rows
}

/// A linear in execution order.
///
/// `segment` changes at every CPU attention point: linears sharing a segment may share one
/// NPU publish (planning rule from the task). `stage` changes whenever a linear depends on
/// the output of the previous group (pessimistic rule: one publish per stage).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub offset: u64,
    pub bytes: u64,
    pub segment: usize,
    pub stage: usize,
}

/// Segment/stage bookkeeping shared by the klein table order and the generic order.
struct Order {
    out: Vec<Linear>,
    segment: usize,
    stage: usize,
}

type Resolve<'f> = dyn FnMut(&str, usize, usize) -> Result<(u64, u64), String> + 'f;

impl Order {
    fn next_stage(&mut self) {
        self.stage += 1;
    }

    fn attention_point(&mut self) {
        self.segment += 1;
    }

    fn lin(&mut self, name: &str, in_dim: usize, out_dim: usize, resolve: &mut Resolve) -> Result<(), String> {
        let (offset, bytes) = resolve(name, in_dim, out_dim)?;
        self.out.push(Linear {
            name: name.to_string(),
            in_dim,
            out_dim,
            offset,
            bytes,
            segment: self.segment,
            stage: self.stage,
        });
        Ok(())
    }
}

/// Q8_0 byte count of a `[out][in]` matrix.
pub fn q8_bytes(in_dim: usize, out_dim: usize) -> u64 {
    out_dim as u64 * (in_dim as u64 / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES
}

/// Execution order of the block linears: double block = qkv pair | attention | proj pair,
/// mlp.0 pair, mlp.2 pair; single block = linear1 | attention | linear2.
fn sequence(n_double: usize, n_single: usize, d: Dims, resolve: &mut Resolve) -> Result<Vec<Linear>, String> {
    let (h, m) = (d.hidden, d.mlp);
    let mut o = Order { out: Vec::new(), segment: 0, stage: 0 };
    for i in 0..n_double {
        let p = format!("double_blocks.{i}.");
        o.next_stage();
        o.lin(&format!("{p}img_attn.qkv.weight"), h, 3 * h, resolve)?;
        o.lin(&format!("{p}txt_attn.qkv.weight"), h, 3 * h, resolve)?;
        o.attention_point();
        o.next_stage();
        o.lin(&format!("{p}img_attn.proj.weight"), h, h, resolve)?;
        o.lin(&format!("{p}txt_attn.proj.weight"), h, h, resolve)?;
        o.next_stage();
        o.lin(&format!("{p}img_mlp.0.weight"), h, 2 * m, resolve)?;
        o.lin(&format!("{p}txt_mlp.0.weight"), h, 2 * m, resolve)?;
        o.next_stage();
        o.lin(&format!("{p}img_mlp.2.weight"), m, h, resolve)?;
        o.lin(&format!("{p}txt_mlp.2.weight"), m, h, resolve)?;
    }
    for j in 0..n_single {
        let p = format!("single_blocks.{j}.");
        o.next_stage();
        o.lin(&format!("{p}linear1.weight"), h, d.lin1_out(), resolve)?;
        o.attention_point();
        o.next_stage();
        o.lin(&format!("{p}linear2.weight"), d.lin2_in(), h, resolve)?;
    }
    Ok(o.out)
}

fn count_suffix(rows: &[TensorRow], prefix: &str, suffix: &str) -> usize {
    rows.iter()
        .filter(|r| r.name.starts_with(prefix) && r.name.ends_with(suffix))
        .count()
}

/// Builds klein's 80 block linears (Q8_0) in execution order, with segment and stage ids, checking
/// every row against the table. Embedders, modulation linears (BF16) and the final layer are not
/// included (CPU).
pub fn klein_linear_order(rows: &[TensorRow]) -> Result<Vec<Linear>, String> {
    let n_double = count_suffix(rows, "double_blocks.", ".img_attn.qkv.weight");
    let n_single = count_suffix(rows, "single_blocks.", ".linear1.weight");
    if n_double == 0 || n_single == 0 {
        return Err("no klein double/single blocks in table".to_string());
    }
    let d = Dims::KLEIN_4B;
    d.check();
    let by_name: HashMap<&str, &TensorRow> = rows.iter().map(|r| (r.name.as_str(), r)).collect();
    let mut resolve = |name: &str, i: usize, o: usize| -> Result<(u64, u64), String> {
        let row = by_name.get(name).copied().ok_or_else(|| format!("tensor {name} missing from table"))?;
        if row.dtype != "Q8_0" {
            return Err(format!("tensor {name} has type {}, expected Q8_0", row.dtype));
        }
        if row.dims != [i as u64, o as u64] {
            return Err(format!("tensor {name} dims {:?}, expected [{i}, {o}]", row.dims));
        }
        if i as u64 % Q8_0_BLOCK_ELEMS != 0 {
            return Err(format!("tensor {name}: in_dim {i} not a multiple of 32"));
        }
        let expect = q8_bytes(i, o);
        if row.bytes != expect {
            return Err(format!("tensor {name}: {} bytes, Q8_0 arithmetic gives {expect}", row.bytes));
        }
        Ok((row.offset, row.bytes))
    };
    sequence(n_double, n_single, d, &mut resolve)
}

/// The same execution order for any block shapes, bytes from Q8_0 arithmetic (no table). Used by
/// the tiny-model tests. `in` widths must be multiples of 32.
pub fn block_linear_order(n_double: usize, n_single: usize, hidden: usize, mlp: usize) -> Vec<Linear> {
    let d = Dims { hidden, heads: 1, head_dim: hidden, mlp };
    let mut resolve = |_: &str, i: usize, o: usize| -> Result<(u64, u64), String> { Ok((0, q8_bytes(i, o))) };
    sequence(n_double, n_single, d, &mut resolve).expect("arithmetic order cannot fail")
}

/// Activation reserve model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActReserve {
    /// Sum over every linear in the window of input+output bytes (conservative; no buffer reuse).
    Sum,
    /// Largest single linear's input+output bytes (one linear live at a time).
    PeakPerLinear,
}

/// Planner and estimate parameters. Timing fields are assumptions unless noted.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanConfig {
    /// Session budget for weights + activations + overhead. Default 1 GiB (task / handoff).
    pub budget_bytes: u64,
    /// Per-buffer chunk limit. Default 128 MiB (GGML_HEXAGON_MBUF dyn/static field).
    pub chunk_bytes: u64,
    /// Largest single tensor. Default 256 MiB (standing cap in npu-measurements.md).
    pub max_tensor_bytes: u64,
    pub n_tokens: usize,
    /// 4 for f32 activations, 2 for f16.
    pub act_elem_bytes: u64,
    pub act_reserve: ActReserve,
    /// Fixed overhead reserve. ASSUMPTION: 64 MiB, not measured.
    pub overhead_bytes: u64,
    /// Dispatch floor per publish. 10.8 ms measured (npu-measurements.md); 0.3-1 after the floor fix (task).
    pub floor_ms: f64,
    /// NPU rate for the compute term. ASSUMPTION: 10 TOPS (resident matmul slope, npu-measurements.md).
    pub tops: f64,
    /// CPU attention time per denoising step. Not modelled: caller supplies it (default 0).
    pub cpu_attn_ms: f64,
    /// Fixed cost per window load. ASSUMPTION: 11.3 ms, the measured alloc (6.5) + upload (4.8) upper bounds.
    pub remap_fixed_ms: f64,
    /// Weight upload rate used for window loads. ASSUMPTION: 22.8 GB/s, the measured NPU streaming
    /// read rate used as a stand-in. No upload rate was measured.
    pub load_gbps: f64,
    /// Activation transfer rate. 27.77 GB/s NPU effective read+write (npu-measurements.md, shared-bus table).
    pub act_gbps: f64,
}

impl PlanConfig {
    pub fn klein_default(n_tokens: usize) -> Self {
        PlanConfig {
            budget_bytes: GIB,
            chunk_bytes: 128 * MIB,
            max_tensor_bytes: 256 * MIB,
            n_tokens,
            act_elem_bytes: 4,
            act_reserve: ActReserve::Sum,
            overhead_bytes: 64 * MIB,
            floor_ms: 10.8,
            tops: 10.0,
            cpu_attn_ms: 0.0,
            remap_fixed_ms: 11.3,
            load_gbps: 22.8,
            act_gbps: 27.77,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.n_tokens == 0 {
            return Err("n_tokens must be > 0".into());
        }
        if self.budget_bytes <= self.overhead_bytes {
            return Err("budget must exceed the overhead reserve".into());
        }
        if self.act_elem_bytes == 0 {
            return Err("act_elem_bytes must be > 0".into());
        }
        if !(self.tops > 0.0 && self.load_gbps > 0.0 && self.act_gbps > 0.0) {
            return Err("rates must be positive".into());
        }
        Ok(())
    }
}

/// One planned window: consecutive linears that are loaded together.
#[derive(Clone, Debug, PartialEq)]
pub struct Window {
    pub index: usize,
    pub members: Vec<Linear>,
    pub weight_bytes: u64,
    pub act_bytes: u64,
    /// act_bytes + overhead_bytes: what the window charges against the budget.
    pub reserved_bytes: u64,
    /// Number of buffers of at most `chunk_bytes` (tensors are never split).
    pub buffers: usize,
    /// Publishes per step for this window, attention-separated rule (planning rule).
    pub publishes: usize,
    /// Publishes per step for this window, one per dependency stage (pessimistic bound).
    pub stage_publishes: usize,
    /// publishes * floor_ms (ESTIMATE).
    pub dispatch_ms: f64,
    pub warnings: Vec<String>,
}

impl Window {
    pub fn names(&self) -> Vec<&str> {
        self.members.iter().map(|l| l.name.as_str()).collect()
    }
}

fn io_bytes(l: &Linear, cfg: &PlanConfig) -> u64 {
    cfg.n_tokens as u64 * (l.in_dim + l.out_dim) as u64 * cfg.act_elem_bytes
}

fn act_bytes<'a, I: IntoIterator<Item = &'a Linear>>(members: I, cfg: &PlanConfig) -> u64 {
    match cfg.act_reserve {
        ActReserve::Sum => members.into_iter().map(|l| io_bytes(l, cfg)).sum(),
        ActReserve::PeakPerLinear => members.into_iter().map(|l| io_bytes(l, cfg)).max().unwrap_or(0),
    }
}

fn fits(linears: &[Linear], idx: &[usize], cfg: &PlanConfig) -> bool {
    let weights: u64 = idx.iter().map(|&i| linears[i].bytes).sum();
    let act = act_bytes(idx.iter().map(|&i| &linears[i]), cfg);
    weights + act + cfg.overhead_bytes <= cfg.budget_bytes
}

fn count_runs<I: IntoIterator<Item = usize>>(it: I) -> usize {
    let mut n = 0;
    let mut prev = None;
    for v in it {
        if prev != Some(v) {
            n += 1;
            prev = Some(v);
        }
    }
    n
}

fn chunk_count(members: &[Linear], chunk: u64) -> usize {
    let mut n = 0usize;
    let mut used = 0u64;
    for l in members {
        if n == 0 || used + l.bytes > chunk {
            n += 1;
            used = 0;
        }
        used += l.bytes;
    }
    n
}

fn build_window(index: usize, idx: &[usize], linears: &[Linear], cfg: &PlanConfig) -> Window {
    let members: Vec<Linear> = idx.iter().map(|&i| linears[i].clone()).collect();
    let weight_bytes: u64 = members.iter().map(|l| l.bytes).sum();
    let act = act_bytes(members.iter(), cfg);
    let publishes = count_runs(members.iter().map(|l| l.segment));
    let stage_publishes = count_runs(members.iter().map(|l| l.stage));
    let mut warnings = Vec::new();
    for l in &members {
        if l.bytes > cfg.max_tensor_bytes {
            warnings.push(format!(
                "{} is {} bytes, above the {} byte single-tensor cap",
                l.name, l.bytes, cfg.max_tensor_bytes
            ));
        } else if l.bytes > cfg.chunk_bytes {
            warnings.push(format!(
                "{} is {} bytes, above the {} byte chunk limit (occupies its own buffer)",
                l.name, l.bytes, cfg.chunk_bytes
            ));
        }
    }
    Window {
        index,
        weight_bytes,
        act_bytes: act,
        reserved_bytes: act + cfg.overhead_bytes,
        buffers: chunk_count(&members, cfg.chunk_bytes),
        publishes,
        stage_publishes,
        dispatch_ms: publishes as f64 * cfg.floor_ms,
        warnings,
        members,
    }
}

/// Greedy in-order packing. A window is closed when adding the next linear would exceed the budget.
/// Tensors are never split. Errors if one linear cannot fit a window by itself.
pub fn plan_windows(linears: &[Linear], cfg: &PlanConfig) -> Result<Vec<Window>, String> {
    cfg.validate()?;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    for i in 0..linears.len() {
        let mut cand = cur.clone();
        cand.push(i);
        if fits(linears, &cand, cfg) {
            cur = cand;
            continue;
        }
        if !cur.is_empty() {
            groups.push(std::mem::take(&mut cur));
        }
        if !fits(linears, &[i], cfg) {
            let l = &linears[i];
            return Err(format!(
                "{} cannot fit one window: weights {} B + activations {} B + overhead {} B > budget {} B",
                l.name,
                l.bytes,
                act_bytes(std::iter::once(l), cfg),
                cfg.overhead_bytes,
                cfg.budget_bytes
            ));
        }
        cur.push(i);
    }
    if !cur.is_empty() {
        groups.push(cur);
    }
    Ok(groups
        .iter()
        .enumerate()
        .map(|(k, g)| build_window(k, g, linears, cfg))
        .collect())
}

/// Device sessions (slots) that hold windows at once. `session_budget_bytes` is the per-session
/// budget; windows are planned against it (see `PlanConfig::budget_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyConfig {
    pub max_sessions: usize,
    pub session_budget_bytes: u64,
}

/// Slot of each window.
///
/// - `sessions >= windows`: window `w` lives in slot `w`; every window is loaded once per image.
/// - otherwise slots `0..sessions-1` hold windows `0..sessions-1` permanently (pinned), and the last
///   slot streams windows `sessions-1..windows` in order. Each streamed window is reloaded every
///   step. Planning rule, not an optimum: the reload count is `windows - sessions + 1` per step.
pub fn assign_slots(n_windows: usize, sessions: usize) -> Vec<usize> {
    assert!(sessions >= 1, "at least one session");
    (0..n_windows)
        .map(|w| if sessions >= n_windows || w + 1 < sessions { w } else { sessions - 1 })
        .collect()
}

/// Loads per denoising step after the first one (steady state) under `assign_slots`.
pub fn steady_reloads_per_step(n_windows: usize, sessions: usize) -> usize {
    assert!(sessions >= 1, "at least one session");
    if sessions >= n_windows {
        0
    } else {
        n_windows - sessions + 1
    }
}

/// Residency estimate for `sessions` slots (ESTIMATE; load times use the same stand-in parameters
/// as `estimate_step`).
#[derive(Clone, Debug, PartialEq)]
pub struct ResidencyEstimate {
    pub sessions: usize,
    pub windows: usize,
    /// Windows held permanently.
    pub pinned_windows: usize,
    /// Loads per step after the first one.
    pub reloads_per_step: usize,
    pub reloaded_bytes_per_step: u64,
    pub reload_ms_per_step: f64,
    /// One load of every window (first step of an image).
    pub cold_load_ms_per_image: f64,
    /// Sum of window charges in the pinned slots (all windows when every window is resident).
    pub pinned_charge_bytes: u64,
}

fn load_ms(bytes: u64, cfg: &PlanConfig) -> f64 {
    cfg.remap_fixed_ms + bytes as f64 / (cfg.load_gbps * 1e9) * 1e3
}

pub fn estimate_residency(windows: &[Window], cfg: &PlanConfig, sessions: usize) -> ResidencyEstimate {
    let n = windows.len();
    let slots = assign_slots(n, sessions);
    let mut reloads = 0usize;
    let mut reload_bytes = 0u64;
    let mut reload_ms = 0.0;
    let mut pinned_charge = 0u64;
    for (w, win) in windows.iter().enumerate() {
        let streamed = slots[w] == sessions.saturating_sub(1) && sessions < n && w + 1 >= sessions;
        if streamed {
            reloads += 1;
            reload_bytes += win.weight_bytes;
            reload_ms += load_ms(win.weight_bytes, cfg);
        } else {
            pinned_charge += win.weight_bytes + win.reserved_bytes;
        }
    }
    ResidencyEstimate {
        sessions,
        windows: n,
        pinned_windows: n - reloads,
        reloads_per_step: reloads,
        reloaded_bytes_per_step: reload_bytes,
        reload_ms_per_step: reload_ms,
        cold_load_ms_per_image: windows.iter().map(|w| load_ms(w.weight_bytes, cfg)).sum(),
        pinned_charge_bytes: pinned_charge,
    }
}

/// Text table of residency estimates for several session counts.
pub fn residency_report(windows: &[Window], cfg: &PlanConfig, sessions: &[usize]) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "residency (ESTIMATE): {} windows, session budget {} MiB, activation reserve {:?}, tokens {}",
        windows.len(),
        cfg.budget_bytes / MIB,
        cfg.act_reserve,
        cfg.n_tokens
    );
    let _ = writeln!(
        s,
        "{:>8} {:>7} {:>8} {:>12} {:>14} {:>12} {:>16}",
        "sessions", "pinned", "reloads", "reload_MiB", "reload_ms/step", "cold_ms", "pinned_chg_MiB"
    );
    for &k in sessions {
        let e = estimate_residency(windows, cfg, k);
        let _ = writeln!(
            s,
            "{:>8} {:>7} {:>8} {:>12.2} {:>14.1} {:>12.1} {:>16.2}",
            e.sessions,
            e.pinned_windows,
            e.reloads_per_step,
            e.reloaded_bytes_per_step as f64 / MIB as f64,
            e.reload_ms_per_step,
            e.cold_load_ms_per_image,
            e.pinned_charge_bytes as f64 / MIB as f64
        );
    }
    s
}

/// Publishes per step for the split-stream schedule: one publish per (dependency stage, stream),
/// where the stream is `img`, `txt` or `single`. Counted per window, like `Window::stage_publishes`.
pub fn stream_split_publishes(windows: &[Window]) -> usize {
    windows
        .iter()
        .map(|w| {
            count_runs(w.members.iter().map(|l| {
                let stream = if l.name.contains(".img_") {
                    1
                } else if l.name.contains(".txt_") {
                    2
                } else {
                    3
                };
                l.stage * 4 + stream
            }))
        })
        .sum()
}

/// Per-step ESTIMATE (planning arithmetic, labelled as such everywhere it is printed).
#[derive(Clone, Debug, PartialEq)]
pub struct StepEstimate {
    pub windows: usize,
    pub publishes: usize,
    pub stage_publishes: usize,
    /// Weight bytes loaded per step (each window loaded once per step).
    pub weight_bytes_loaded: u64,
    /// sum over linears of 2*n*in*out / (tops * 1e12).
    pub compute_ms: f64,
    /// publishes * floor_ms.
    pub dispatch_ms: f64,
    /// windows * (remap_fixed_ms + window weight bytes / load_gbps).
    pub remap_ms: f64,
    /// compute + dispatch + remap (spec formula).
    pub npu_ms: f64,
    /// Same with one publish per dependency stage.
    pub stage_npu_ms: f64,
    pub cpu_attn_ms: f64,
    /// npu_ms + cpu_attn_ms (no overlap).
    pub serial_ms: f64,
    /// max(npu_ms, cpu_attn_ms) (perfect overlap).
    pub overlapped_ms: f64,
    /// Activation bytes in+out per step / act_gbps. Not included in npu_ms.
    pub act_transfer_ms: f64,
    pub act_bytes_per_step: u64,
}

pub fn estimate_step(windows: &[Window], cfg: &PlanConfig) -> StepEstimate {
    let n = cfg.n_tokens as f64;
    let mut macs = 0f64;
    let mut io = 0u64;
    for w in windows {
        for l in &w.members {
            macs += n * l.in_dim as f64 * l.out_dim as f64;
            io += io_bytes(l, cfg);
        }
    }
    let compute_ms = 2.0 * macs / (cfg.tops * 1e12) * 1e3;
    let publishes: usize = windows.iter().map(|w| w.publishes).sum();
    let stage_publishes: usize = windows.iter().map(|w| w.stage_publishes).sum();
    let dispatch_ms = publishes as f64 * cfg.floor_ms;
    let stage_dispatch_ms = stage_publishes as f64 * cfg.floor_ms;
    let weight_bytes_loaded: u64 = windows.iter().map(|w| w.weight_bytes).sum();
    let remap_ms: f64 = windows
        .iter()
        .map(|w| cfg.remap_fixed_ms + w.weight_bytes as f64 / (cfg.load_gbps * 1e9) * 1e3)
        .sum();
    let npu_ms = compute_ms + dispatch_ms + remap_ms;
    let stage_npu_ms = compute_ms + stage_dispatch_ms + remap_ms;
    StepEstimate {
        windows: windows.len(),
        publishes,
        stage_publishes,
        weight_bytes_loaded,
        compute_ms,
        dispatch_ms,
        remap_ms,
        npu_ms,
        stage_npu_ms,
        cpu_attn_ms: cfg.cpu_attn_ms,
        serial_ms: npu_ms + cfg.cpu_attn_ms,
        overlapped_ms: npu_ms.max(cfg.cpu_attn_ms),
        act_transfer_ms: io as f64 / (cfg.act_gbps * 1e9) * 1e3,
        act_bytes_per_step: io,
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / MIB as f64
}

/// Plain-text plan. Every timing line is labelled ESTIMATE.
pub fn plan_report(windows: &[Window], cfg: &PlanConfig) -> String {
    let mut s = String::new();
    let e = estimate_step(windows, cfg);
    let _ = writeln!(s, "klein linear window plan (planning arithmetic; timings are ESTIMATES)");
    let _ = writeln!(
        s,
        "config: tokens={} budget={} B ({:.2} MiB) chunk={} MiB cap={} MiB act={}B/elem reserve={:?} overhead={} MiB floor={} ms tops={} cpu_attn={} ms remap_fixed={} ms load_gbps={} act_gbps={}",
        cfg.n_tokens,
        cfg.budget_bytes,
        mib(cfg.budget_bytes),
        mib(cfg.chunk_bytes) as u64,
        mib(cfg.max_tensor_bytes) as u64,
        cfg.act_elem_bytes,
        cfg.act_reserve,
        mib(cfg.overhead_bytes) as u64,
        cfg.floor_ms,
        cfg.tops,
        cfg.cpu_attn_ms,
        cfg.remap_fixed_ms,
        cfg.load_gbps,
        cfg.act_gbps
    );
    let _ = writeln!(
        s,
        "{:>3} {:>3} {:<34} {:<34} {:>10} {:>9} {:>9} {:>4} {:>4} {:>4} {:>9}",
        "win", "n", "first", "last", "weight_MiB", "act_MiB", "resv_MiB", "buf", "pub", "pubS", "disp_ms"
    );
    for w in windows {
        let first = &w.members[0];
        let last = &w.members[w.members.len() - 1];
        let _ = writeln!(
            s,
            "{:>3} {:>3} {:<34} {:<34} {:>10.2} {:>9.2} {:>9.2} {:>4} {:>4} {:>4} {:>9.2}",
            w.index,
            w.members.len(),
            first.name,
            last.name,
            mib(w.weight_bytes),
            mib(w.act_bytes),
            mib(w.reserved_bytes),
            w.buffers,
            w.publishes,
            w.stage_publishes,
            w.dispatch_ms
        );
    }
    for w in windows {
        for warn in &w.warnings {
            let _ = writeln!(s, "WARNING window {}: {}", w.index, warn);
        }
    }
    let _ = writeln!(s, "members (rel offset = data-section offset, bytes, segment, stage):");
    for w in windows {
        for l in &w.members {
            let _ = writeln!(
                s,
                "  w{:<2} {:<34} [{:>12}, {:>12}) {:>10} seg={:<2} stage={:<2}",
                w.index,
                l.name,
                l.offset,
                l.offset + l.bytes,
                l.bytes,
                l.segment,
                l.stage
            );
        }
    }
    let _ = writeln!(s, "ESTIMATE per denoising step:");
    let _ = writeln!(s, "  windows (loads per step)      {}", e.windows);
    let _ = writeln!(s, "  weight bytes loaded per step  {} ({:.2} MiB)", e.weight_bytes_loaded, mib(e.weight_bytes_loaded));
    let _ = writeln!(s, "  publishes (attention rule)    {}", e.publishes);
    let _ = writeln!(s, "  publishes (per stage, bound)  {}", e.stage_publishes);
    let _ = writeln!(s, "  compute_ms   = {:.1}", e.compute_ms);
    let _ = writeln!(s, "  dispatch_ms  = {:.1}", e.dispatch_ms);
    let _ = writeln!(s, "  remap_ms     = {:.1}", e.remap_ms);
    let _ = writeln!(s, "  npu_ms       = {:.1}  (compute + dispatch + remap)", e.npu_ms);
    let _ = writeln!(s, "  npu_ms, per-stage publishes = {:.1}", e.stage_npu_ms);
    let _ = writeln!(s, "  cpu_attn_ms  = {:.1}  (parameter)", e.cpu_attn_ms);
    let _ = writeln!(s, "  serial_ms    = {:.1}", e.serial_ms);
    let _ = writeln!(s, "  overlapped_ms= {:.1}", e.overlapped_ms);
    let _ = writeln!(
        s,
        "  activation bytes in+out per step = {} ({:.2} GiB), transfer_ms at act_gbps = {:.1} (not in npu_ms)",
        e.act_bytes_per_step,
        e.act_bytes_per_step as f64 / GIB as f64,
        e.act_transfer_ms
    );
    s
}

/// The same plan as JSON.
pub fn plan_json(windows: &[Window], cfg: &PlanConfig) -> Value {
    let e = estimate_step(windows, cfg);
    let wins: Vec<Value> = windows
        .iter()
        .map(|w| {
            json!({
                "index": w.index,
                "weight_bytes": w.weight_bytes,
                "act_bytes": w.act_bytes,
                "reserved_bytes": w.reserved_bytes,
                "buffers": w.buffers,
                "publishes": w.publishes,
                "stage_publishes": w.stage_publishes,
                "dispatch_ms_estimate": w.dispatch_ms,
                "warnings": w.warnings,
                "members": w.members.iter().map(|l| json!({
                    "name": l.name,
                    "in": l.in_dim,
                    "out": l.out_dim,
                    "offset_rel": l.offset,
                    "bytes": l.bytes,
                    "segment": l.segment,
                    "stage": l.stage,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    json!({
        "kind": "planning_estimate",
        "config": {
            "n_tokens": cfg.n_tokens,
            "budget_bytes": cfg.budget_bytes,
            "chunk_bytes": cfg.chunk_bytes,
            "max_tensor_bytes": cfg.max_tensor_bytes,
            "act_elem_bytes": cfg.act_elem_bytes,
            "act_reserve": format!("{:?}", cfg.act_reserve),
            "overhead_bytes": cfg.overhead_bytes,
            "floor_ms": cfg.floor_ms,
            "tops": cfg.tops,
            "cpu_attn_ms": cfg.cpu_attn_ms,
            "remap_fixed_ms": cfg.remap_fixed_ms,
            "load_gbps": cfg.load_gbps,
            "act_gbps": cfg.act_gbps,
        },
        "windows": wins,
        "estimate": {
            "windows": e.windows,
            "publishes": e.publishes,
            "stage_publishes": e.stage_publishes,
            "weight_bytes_loaded": e.weight_bytes_loaded,
            "compute_ms": e.compute_ms,
            "dispatch_ms": e.dispatch_ms,
            "remap_ms": e.remap_ms,
            "npu_ms": e.npu_ms,
            "stage_npu_ms": e.stage_npu_ms,
            "cpu_attn_ms": e.cpu_attn_ms,
            "serial_ms": e.serial_ms,
            "overlapped_ms": e.overlapped_ms,
            "act_transfer_ms": e.act_transfer_ms,
            "act_bytes_per_step": e.act_bytes_per_step,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../../../tests/fixtures/klein-4b-q8.tensors.txt");

    fn klein() -> (Vec<TensorRow>, Vec<Linear>) {
        let rows = parse_tensor_table(FIXTURE);
        let order = klein_linear_order(&rows).expect("klein order");
        (rows, order)
    }

    #[test]
    fn parses_fixture_rows() {
        let rows = parse_tensor_table(FIXTURE);
        let data_lines = FIXTURE
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .count();
        assert_eq!(rows.len(), data_lines);
        assert_eq!(rows[0].name, "double_blocks.0.img_attn.norm.key_norm.scale");
        assert_eq!(rows[0].dims, vec![128]);
        let qkv = rows.iter().find(|r| r.name == "double_blocks.0.img_attn.qkv.weight").unwrap();
        assert_eq!(qkv.dims, vec![3072, 9216]);
        assert_eq!(qkv.bytes, 30_081_024);
    }

    #[test]
    fn klein_order_covers_every_q8_linear_once() {
        let (rows, order) = klein();
        let q8: Vec<&str> = rows.iter().filter(|r| r.dtype == "Q8_0").map(|r| r.name.as_str()).collect();
        assert_eq!(q8.len(), 80, "table has 80 Q8_0 tensors, all block linears");
        assert_eq!(order.len(), 80);
        let mut names: Vec<&str> = order.iter().map(|l| l.name.as_str()).collect();
        names.sort();
        let mut want = q8.clone();
        want.sort();
        assert_eq!(names, want, "every Q8_0 linear exactly once");
        let mut uniq = names.clone();
        uniq.dedup();
        assert_eq!(uniq.len(), 80);
    }

    #[test]
    fn klein_structure_counts() {
        let (_, order) = klein();
        // 26 segments: 25 CPU attention points (5 double + 20 single).
        let segs = count_runs(order.iter().map(|l| l.segment));
        assert_eq!(segs, 26);
        // Dependency stages: 4 per double block + 2 per single block.
        let stages = count_runs(order.iter().map(|l| l.stage));
        assert_eq!(stages, 5 * 4 + 20 * 2);
        assert_eq!(order[0].name, "double_blocks.0.img_attn.qkv.weight");
        assert_eq!(order[1].name, "double_blocks.0.txt_attn.qkv.weight");
        assert_eq!(order[2].name, "double_blocks.0.img_attn.proj.weight");
    }

    #[test]
    fn windows_cover_order_within_budget_and_are_deterministic() {
        let (_, order) = klein();
        for n_tokens in [1536usize, 4608] {
            for act in [ActReserve::Sum, ActReserve::PeakPerLinear] {
                let mut cfg = PlanConfig::klein_default(n_tokens);
                cfg.act_reserve = act;
                let a = plan_windows(&order, &cfg).expect("plan");
                let b = plan_windows(&order, &cfg).expect("plan again");
                assert_eq!(a, b, "deterministic");
                let flat: Vec<&str> = a.iter().flat_map(|w| w.names()).collect();
                let want: Vec<&str> = order.iter().map(|l| l.name.as_str()).collect();
                assert_eq!(flat, want, "windows concatenate to the execution order");
                for w in &a {
                    assert!(w.weight_bytes + w.reserved_bytes <= cfg.budget_bytes, "window {} over budget", w.index);
                    assert!(w.members.len() >= 1);
                }
            }
        }
    }

    #[test]
    fn oversized_tensor_is_flagged_not_fatal() {
        let linears = vec![Linear {
            name: "big".into(),
            in_dim: 32,
            out_dim: 1,
            offset: 0,
            bytes: 300 * MIB,
            segment: 0,
            stage: 0,
        }];
        let mut cfg = PlanConfig::klein_default(1);
        cfg.budget_bytes = 2 * GIB;
        let w = plan_windows(&linears, &cfg).unwrap();
        assert_eq!(w.len(), 1);
        assert!(w[0].warnings.iter().any(|s| s.contains("single-tensor cap")));
    }

    #[test]
    fn tensor_that_cannot_fit_is_an_error() {
        let linears = vec![Linear {
            name: "huge".into(),
            in_dim: 32,
            out_dim: 1,
            offset: 0,
            bytes: 2 * GIB,
            segment: 0,
            stage: 0,
        }];
        let cfg = PlanConfig::klein_default(1);
        let err = plan_windows(&linears, &cfg).unwrap_err();
        assert!(err.contains("huge cannot fit one window"), "{err}");
    }

    /// Prints the reports used for PLAN-NPU.md. Run the test harness with --nocapture.
    #[test]
    fn print_plan_numbers() {
        let (_, order) = klein();
        for n in [1536usize, 4608] {
            for act in [ActReserve::Sum, ActReserve::PeakPerLinear] {
                for floor in [10.8f64, 0.5] {
                    let mut cfg = PlanConfig::klein_default(n);
                    cfg.act_reserve = act;
                    cfg.floor_ms = floor;
                    let w = plan_windows(&order, &cfg).unwrap();
                    println!("==== tokens={n} act={act:?} floor_ms={floor} ====");
                    print!("{}", plan_report(&w, &cfg));
                }
            }
            for act in [ActReserve::Sum, ActReserve::PeakPerLinear] {
                let mut cfg = PlanConfig::klein_default(n);
                cfg.act_reserve = act;
                let w = plan_windows(&order, &cfg).unwrap();
                let nw = w.len();
                println!("==== residency tokens={n} act={act:?} windows={nw} ====");
                let mut ks = vec![1usize, 2, 4];
                if !ks.contains(&nw) {
                    ks.push(nw);
                }
                print!("{}", residency_report(&w, &cfg, &ks));
                let merged: usize = w.iter().map(|x| x.stage_publishes).sum();
                let split = stream_split_publishes(&w);
                println!("publishes per step: merged(stage)={merged} stream-split={split} attention-segment={}",
                    w.iter().map(|x| x.publishes).sum::<usize>());
            }
        }
    }

    #[test]
    fn estimate_counts_are_consistent() {
        let (_, order) = klein();
        let cfg = PlanConfig::klein_default(1536);
        let w = plan_windows(&order, &cfg).unwrap();
        let e = estimate_step(&w, &cfg);
        assert_eq!(e.publishes, w.iter().map(|x| x.publishes).sum::<usize>());
        assert!(e.publishes >= 26);
        assert!(e.stage_publishes >= 60);
        let total: u64 = order.iter().map(|l| l.bytes).sum();
        assert_eq!(e.weight_bytes_loaded, total);
        assert!(e.npu_ms > e.compute_ms);
        let report = plan_report(&w, &cfg);
        assert!(report.contains("ESTIMATE"));
        let j = plan_json(&w, &cfg);
        assert_eq!(j["windows"].as_array().unwrap().len(), w.len());
    }
}
