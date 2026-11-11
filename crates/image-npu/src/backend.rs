//! Device boundary for resident windows. No device code lives here.
//!
//! A device has `sessions()` slots. A slot holds one window at a time. Windows are loaded into a
//! slot with [`LinearDevice::load_window`] (the device receives the Q8_0 bytes of every linear in
//! the window; a real device uploads them once, a mock dequantises them). Each publish is a
//! [`LinearDevice::submit`] of linears that all live in one slot, and [`LinearDevice::wait`]
//! returns its results. Several publishes may be in flight; a slot cannot be unloaded while a
//! publish on it is in flight.
//!
//! [`WindowedRunner`] maps windows to slots (see `plan::assign_slots`), loads on demand, drains
//! in-flight work before it unloads a slot, and counts loads, reloads and publishes. A reload is a
//! load of a window that was loaded before (out-of-order access or streaming). Results stay
//! correct either way.

use std::collections::{HashMap, HashSet};

use image_dit::fast_model::WeightSource;
use image_dit::weights::dequant_q8_0;

use crate::plan::{assign_slots, Window};

#[derive(Clone, Debug, PartialEq)]
pub struct LinearCall {
    pub tensor: String,
    pub n_tokens: usize,
    /// Row-major `[n_tokens][in]`.
    pub input: Vec<f32>,
}

/// One linear as a device needs it: the GGUF Q8_0 bytes, `[out][in/32]` blocks of 34 bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct PayloadLinear {
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub q8: Vec<u8>,
}

/// The weights of one window, in the window's member order.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct WindowPayload {
    pub linears: Vec<PayloadLinear>,
}

/// Copies the Q8_0 bytes of every member of `w` from `src`, checking shapes and byte counts
/// against the plan.
pub fn window_payload(src: &dyn WeightSource, w: &Window) -> Result<WindowPayload, String> {
    let mut linears = Vec::with_capacity(w.members.len());
    for l in &w.members {
        let (bytes, rows, k) = src.q8(&l.name)?;
        if rows != l.out_dim || k != l.in_dim || bytes.len() as u64 != l.bytes {
            return Err(format!("{}: source shape or bytes disagree with the plan", l.name));
        }
        linears.push(PayloadLinear { name: l.name.clone(), in_dim: l.in_dim, out_dim: l.out_dim, q8: bytes.to_vec() });
    }
    Ok(WindowPayload { linears })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ticket(pub u64);

/// What a device implements. One `submit` is one publish; all its calls must be in the same slot.
///
/// A real device needs, per slot: a session (with its buffer budget), a one-time upload of the
/// window's Q8_0 byte ranges from `WindowPayload`, a graph per publish that multiplies resident
/// Q8_0 weights by f32 (or f16) activations, and activations in and out for every call.
pub trait LinearDevice {
    /// Number of slots (sessions) this device can hold.
    fn sessions(&self) -> usize;
    /// Loads `w` into an empty `slot`. Errors if the slot is occupied or the window exceeds the
    /// slot budget.
    fn load_window(&mut self, slot: usize, w: &Window, payload: &WindowPayload) -> Result<(), String>;
    /// Empties `slot`. Errors if it is empty or a publish on it is in flight.
    fn unload_window(&mut self, slot: usize) -> Result<(), String>;
    /// Starts one publish. Returns one row-major `[n_tokens][out]` buffer per call from `wait`.
    fn submit(&mut self, calls: &[LinearCall]) -> Result<Ticket, String>;
    fn wait(&mut self, t: Ticket) -> Result<Vec<Vec<f32>>, String>;
    /// Submit and wait.
    fn run_linears(&mut self, calls: &[LinearCall]) -> Result<Vec<Vec<f32>>, String> {
        let t = self.submit(calls)?;
        self.wait(t)
    }
}

/// Dot product with 8 independent lanes, same order as `image_flow::full_ref::dot`.
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let main = n - n % 8;
    let mut acc = [0f32; 8];
    for (ca, cb) in a[..main].chunks_exact(8).zip(b[..main].chunks_exact(8)) {
        for l in 0..8 {
            acc[l] += ca[l] * cb[l];
        }
    }
    for (l, (x, y)) in a[main..].iter().zip(&b[main..]).enumerate() {
        acc[l] += x * y;
    }
    (acc[0] + acc[1] + acc[2] + acc[3]) + (acc[4] + acc[5] + acc[6] + acc[7])
}

struct MockLinear {
    in_dim: usize,
    out_dim: usize,
    /// Dequantised, row-major `[out][in]`.
    w: Vec<f32>,
}

struct MockSlot {
    window: usize,
    tensors: HashMap<String, MockLinear>,
    pending: usize,
}

/// Model-free device. Keeps dequantised f32 weights per slot and computes `y = W x` at `wait`.
/// Enforces the per-slot budget (charge = weight bytes + reserved bytes of the window) and the
/// load/unload discipline. Counts per slot.
pub struct MockDevice {
    budget_bytes: u64,
    slots: Vec<Option<MockSlot>>,
    tickets: HashMap<u64, (usize, Vec<LinearCall>)>,
    next_ticket: u64,
    pub loads: Vec<usize>,
    pub unloads: Vec<usize>,
    pub publishes: Vec<usize>,
    /// Largest charge seen per slot.
    pub max_charged: Vec<u64>,
}

impl MockDevice {
    pub fn new(n_slots: usize, budget_bytes: u64) -> Self {
        assert!(n_slots >= 1);
        MockDevice {
            budget_bytes,
            slots: (0..n_slots).map(|_| None).collect(),
            tickets: HashMap::new(),
            next_ticket: 0,
            loads: vec![0; n_slots],
            unloads: vec![0; n_slots],
            publishes: vec![0; n_slots],
            max_charged: vec![0; n_slots],
        }
    }

    /// Window index loaded in `slot`, if any.
    pub fn resident(&self, slot: usize) -> Option<usize> {
        self.slots.get(slot).and_then(|s| s.as_ref()).map(|s| s.window)
    }

    pub fn total_loads(&self) -> usize {
        self.loads.iter().sum()
    }

    pub fn total_publishes(&self) -> usize {
        self.publishes.iter().sum()
    }

    fn slot_of_tensor(&self, name: &str) -> Option<usize> {
        self.slots.iter().position(|s| s.as_ref().is_some_and(|s| s.tensors.contains_key(name)))
    }
}

impl LinearDevice for MockDevice {
    fn sessions(&self) -> usize {
        self.slots.len()
    }

    fn load_window(&mut self, slot: usize, w: &Window, payload: &WindowPayload) -> Result<(), String> {
        if slot >= self.slots.len() {
            return Err(format!("load_window: slot {slot} out of range"));
        }
        if self.slots[slot].is_some() {
            return Err(format!("load_window: slot {slot} is occupied; unload first"));
        }
        let charged = w.weight_bytes + w.reserved_bytes;
        if charged > self.budget_bytes {
            return Err(format!("load_window {}: charge {charged} B exceeds slot budget {} B", w.index, self.budget_bytes));
        }
        if payload.linears.len() != w.members.len() {
            return Err("load_window: payload does not match the window".into());
        }
        let mut tensors = HashMap::new();
        for (p, m) in payload.linears.iter().zip(&w.members) {
            if p.name != m.name || p.in_dim != m.in_dim || p.out_dim != m.out_dim {
                return Err(format!("load_window: payload entry {} does not match the window", p.name));
            }
            let want = m.out_dim * (m.in_dim / 32) * 34;
            if p.q8.len() != want {
                return Err(format!("load_window: {} has {} bytes, want {want}", p.name, p.q8.len()));
            }
            let w32 = dequant_q8_0(&p.q8, p.out_dim, p.in_dim);
            tensors.insert(p.name.clone(), MockLinear { in_dim: p.in_dim, out_dim: p.out_dim, w: w32 });
        }
        self.slots[slot] = Some(MockSlot { window: w.index, tensors, pending: 0 });
        self.loads[slot] += 1;
        self.max_charged[slot] = self.max_charged[slot].max(charged);
        Ok(())
    }

    fn unload_window(&mut self, slot: usize) -> Result<(), String> {
        match self.slots.get(slot).and_then(|s| s.as_ref()) {
            None => Err(format!("unload_window: slot {slot} is empty")),
            Some(s) if s.pending > 0 => Err(format!("unload_window: slot {slot} has a publish in flight")),
            Some(_) => {
                self.slots[slot] = None;
                self.unloads[slot] += 1;
                Ok(())
            }
        }
    }

    fn submit(&mut self, calls: &[LinearCall]) -> Result<Ticket, String> {
        if calls.is_empty() {
            return Err("submit: empty publish".into());
        }
        let slot = self
            .slot_of_tensor(&calls[0].tensor)
            .ok_or_else(|| format!("submit: {} is not loaded", calls[0].tensor))?;
        for c in calls {
            if self.slot_of_tensor(&c.tensor) != Some(slot) {
                return Err(format!("submit: {} is not in slot {slot}; split the publish", c.tensor));
            }
            let m = &self.slots[slot].as_ref().expect("slot found above").tensors[&c.tensor];
            if c.input.len() != c.n_tokens * m.in_dim {
                return Err(format!("submit: {} input length {} != {}", c.tensor, c.input.len(), c.n_tokens * m.in_dim));
            }
        }
        self.slots[slot].as_mut().expect("slot found above").pending += 1;
        self.publishes[slot] += 1;
        let id = self.next_ticket;
        self.next_ticket += 1;
        self.tickets.insert(id, (slot, calls.to_vec()));
        Ok(Ticket(id))
    }

    fn wait(&mut self, t: Ticket) -> Result<Vec<Vec<f32>>, String> {
        let (slot, calls) = self.tickets.remove(&t.0).ok_or_else(|| format!("wait: unknown ticket {}", t.0))?;
        let s = self.slots[slot].as_mut().expect("in-flight publish keeps its slot loaded");
        s.pending -= 1;
        let mut outs = Vec::with_capacity(calls.len());
        for c in &calls {
            let m = &s.tensors[&c.tensor];
            let mut y = vec![0f32; c.n_tokens * m.out_dim];
            for tk in 0..c.n_tokens {
                let x = &c.input[tk * m.in_dim..(tk + 1) * m.in_dim];
                for o in 0..m.out_dim {
                    y[tk * m.out_dim + o] = dot8(&m.w[o * m.in_dim..(o + 1) * m.in_dim], x);
                }
            }
            outs.push(y);
        }
        Ok(outs)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RunnerStats {
    pub loads: usize,
    /// Loads of a window that was loaded before (streaming or out-of-order access).
    pub reloads: usize,
    pub publishes: usize,
}

/// A submitted group of calls: one or more device publishes, in call order.
#[derive(Debug)]
pub struct Handle {
    parts: Vec<u64>,
}

struct Inflight {
    slot: usize,
    ticket: Option<Ticket>,
    result: Option<Vec<Vec<f32>>>,
}

/// Maps windows to slots and drives a [`LinearDevice`].
pub struct WindowedRunner<'a, D: LinearDevice> {
    plan: Vec<Window>,
    owner: HashMap<String, usize>,
    slot_of: Vec<usize>,
    src: &'a dyn WeightSource,
    dev: D,
    resident: Vec<Option<usize>>,
    ever_loaded: HashSet<usize>,
    inflight: HashMap<u64, Inflight>,
    next_id: u64,
    stats: RunnerStats,
}

impl<'a, D: LinearDevice> WindowedRunner<'a, D> {
    /// `src` provides the Q8_0 bytes of the plan's linears.
    pub fn new(plan: Vec<Window>, src: &'a dyn WeightSource, dev: D) -> Result<Self, String> {
        let mut owner = HashMap::new();
        for w in &plan {
            for l in &w.members {
                if owner.insert(l.name.clone(), w.index).is_some() {
                    return Err(format!("tensor {} appears in two windows", l.name));
                }
            }
        }
        let sessions = dev.sessions();
        if sessions == 0 {
            return Err("device has no sessions".into());
        }
        let slot_of = assign_slots(plan.len(), sessions);
        Ok(WindowedRunner {
            resident: vec![None; sessions],
            slot_of,
            plan,
            owner,
            src,
            dev,
            ever_loaded: HashSet::new(),
            inflight: HashMap::new(),
            next_id: 0,
            stats: RunnerStats::default(),
        })
    }

    pub fn stats(&self) -> RunnerStats {
        self.stats
    }

    pub fn device(&self) -> &D {
        &self.dev
    }

    pub fn plan(&self) -> &[Window] {
        &self.plan
    }

    pub fn slot_of_window(&self, w: usize) -> usize {
        self.slot_of[w]
    }

    fn owner_of(&self, tensor: &str) -> Result<usize, String> {
        self.owner.get(tensor).copied().ok_or_else(|| format!("tensor {tensor} is not in the plan"))
    }

    /// Waits for every in-flight publish on `slot` and stores its result.
    fn drain_slot(&mut self, slot: usize) -> Result<(), String> {
        let ids: Vec<u64> = self
            .inflight
            .iter()
            .filter(|(_, f)| f.slot == slot && f.result.is_none())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.complete(id)?;
        }
        Ok(())
    }

    fn complete(&mut self, id: u64) -> Result<(), String> {
        let t = match self.inflight.get(&id) {
            Some(f) if f.result.is_none() => f.ticket.expect("pending publish has a ticket"),
            _ => return Ok(()),
        };
        let res = self.dev.wait(t)?;
        let f = self.inflight.get_mut(&id).expect("checked above");
        f.result = Some(res);
        f.ticket = None;
        Ok(())
    }

    /// Makes window `w` resident in its slot and returns the slot.
    fn ensure_loaded(&mut self, w: usize) -> Result<usize, String> {
        let slot = self.slot_of[w];
        if self.resident[slot] == Some(w) {
            return Ok(slot);
        }
        if self.resident[slot].is_some() {
            self.drain_slot(slot)?;
            self.dev.unload_window(slot)?;
            self.resident[slot] = None;
        }
        if !self.ever_loaded.insert(w) {
            self.stats.reloads += 1;
        }
        let payload = window_payload(self.src, &self.plan[w])?;
        self.dev.load_window(slot, &self.plan[w], &payload)?;
        self.resident[slot] = Some(w);
        self.stats.loads += 1;
        Ok(slot)
    }

    /// Submits calls in order. Consecutive calls in the same window form one device publish;
    /// a window change starts a new one. Returns a handle for `wait`.
    pub fn submit(&mut self, calls: &[LinearCall]) -> Result<Handle, String> {
        if calls.is_empty() {
            return Err("submit: empty call group".into());
        }
        let mut parts = Vec::new();
        let mut start = 0;
        while start < calls.len() {
            let w = self.owner_of(&calls[start].tensor)?;
            let mut end = start + 1;
            while end < calls.len() && self.owner_of(&calls[end].tensor)? == w {
                end += 1;
            }
            let slot = self.ensure_loaded(w)?;
            let t = self.dev.submit(&calls[start..end])?;
            let id = self.next_id;
            self.next_id += 1;
            self.inflight.insert(id, Inflight { slot, ticket: Some(t), result: None });
            self.stats.publishes += 1;
            parts.push(id);
            start = end;
        }
        Ok(Handle { parts })
    }

    /// Results of a handle, in call order.
    pub fn wait(&mut self, h: Handle) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::new();
        for id in h.parts {
            self.complete(id)?;
            let f = self.inflight.remove(&id).expect("submitted publish");
            out.extend(f.result.expect("completed publish"));
        }
        Ok(out)
    }

    /// Submit and wait.
    pub fn run_publish(&mut self, calls: &[LinearCall]) -> Result<Vec<Vec<f32>>, String> {
        let h = self.submit(calls)?;
        self.wait(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{plan_windows, ActReserve, PlanConfig};
    use image_dit::fast_model::MemSource;
    use image_dit::weights::quant_q8_0;

    const IN: usize = 32;
    const OUT: usize = 4;
    const TOKENS: usize = 2;
    const N: usize = 6;

    fn source() -> MemSource {
        let mut src = MemSource::default();
        for i in 0..N {
            let v: Vec<f32> = (0..IN * OUT).map(|k| (((k * 7 + i * 3) % 11) as f32) * 0.1 - 0.5).collect();
            src.q8.insert(format!("t{i}"), (quant_q8_0(&v, OUT, IN), OUT, IN));
        }
        src
    }

    /// Six linears, one stage each, segments [0,1,1,2,2,3]. Budget 600 B, peak f32 activations
    /// (2 tokens x 36 x 4 = 288 B): a window holds two linears (272 + 288 B), so three windows.
    fn plan() -> (Vec<Window>, PlanConfig) {
        let segs = [0usize, 1, 1, 2, 2, 3];
        let linears: Vec<crate::plan::Linear> = (0..N)
            .map(|i| crate::plan::Linear {
                name: format!("t{i}"),
                in_dim: IN,
                out_dim: OUT,
                offset: 0,
                bytes: (OUT * (IN / 32) * 34) as u64,
                segment: segs[i],
                stage: i,
            })
            .collect();
        let cfg = PlanConfig {
            budget_bytes: 600,
            overhead_bytes: 0,
            n_tokens: TOKENS,
            act_reserve: ActReserve::PeakPerLinear,
            ..PlanConfig::klein_default(TOKENS)
        };
        (plan_windows(&linears, &cfg).unwrap(), cfg)
    }

    fn input(i: usize) -> Vec<f32> {
        (0..TOKENS * IN).map(|k| ((k + i) as f32) * 0.05 - 0.7).collect()
    }

    fn call(i: usize) -> LinearCall {
        LinearCall { tensor: format!("t{i}"), n_tokens: TOKENS, input: input(i) }
    }

    fn direct(i: usize) -> Vec<f32> {
        let src = source();
        let (bytes, _, _) = src.q8(&format!("t{i}")).unwrap();
        let w = dequant_q8_0(bytes, OUT, IN);
        let x = input(i);
        let mut y = vec![0f32; TOKENS * OUT];
        for t in 0..TOKENS {
            for o in 0..OUT {
                y[t * OUT + o] = dot8(&w[o * IN..(o + 1) * IN], &x[t * IN..(t + 1) * IN]);
            }
        }
        y
    }

    #[test]
    fn plan_has_three_windows_for_the_fixture_budget() {
        let (p, _) = plan();
        assert_eq!(p.len(), 3);
        assert_eq!(p.iter().map(|w| w.members.len()).sum::<usize>(), N);
    }

    #[test]
    fn single_slot_in_order_pass_loads_each_window_each_pass() {
        let (p, _) = plan();
        let src = source();
        let mut r = WindowedRunner::new(p.clone(), &src, MockDevice::new(1, 600)).unwrap();
        for pass in 0..2 {
            let mut outs = Vec::new();
            for i in 0..N {
                outs.push(r.run_publish(&[call(i)]).unwrap().remove(0));
            }
            for i in 0..N {
                assert_eq!(outs[i].len(), direct(i).len());
                for (a, b) in outs[i].iter().zip(direct(i)) {
                    assert!((a - b).abs() < 1e-5, "pass {pass} t{i}");
                }
            }
        }
        // One slot, per-tensor publishes: pass 1 loads w0, w1, w2 (3 loads); pass 2 reloads all
        // three (3 more loads, 3 reloads). 12 publishes over two passes.
        let st = r.stats();
        assert_eq!(st.loads, 6);
        assert_eq!(st.reloads, 3);
        assert_eq!(st.publishes, 12);
        assert_eq!(r.device().total_loads(), 6);
        assert!(r.device().max_charged[0] <= 600);
    }

    #[test]
    fn enough_slots_load_each_window_once_across_passes() {
        let (p, _) = plan();
        let w = p.len();
        let src = source();
        let mut r = WindowedRunner::new(p, &src, MockDevice::new(w, 600)).unwrap();
        for _ in 0..3 {
            for i in 0..N {
                r.run_publish(&[call(i)]).unwrap();
            }
        }
        assert_eq!(r.stats().loads, w);
        assert_eq!(r.stats().reloads, 0);
        assert_eq!(r.stats().publishes, 3 * N);
        assert_eq!(r.device().total_publishes(), 3 * N);
    }

    #[test]
    fn in_flight_publish_blocks_unload_and_waits_return_in_order() {
        let (p, _) = plan();
        let src = source();
        let mut dev = MockDevice::new(3, 600);
        let w0 = window_payload(&src, &p[0]).unwrap();
        dev.load_window(0, &p[0], &w0).unwrap();
        let t = dev.submit(&[call(0)]).unwrap();
        assert!(dev.unload_window(0).is_err(), "unload while in flight");
        let t2 = dev.submit(&[call(1)]).unwrap();
        let r2 = dev.wait(t2).unwrap();
        let r1 = dev.wait(t).unwrap();
        assert_eq!(r1[0], direct(0));
        assert_eq!(r2[0], direct(1));
        dev.unload_window(0).unwrap();
    }

    #[test]
    fn mock_rejects_occupied_slot_and_overbudget_window() {
        let (p, _) = plan();
        let src = source();
        let mut dev = MockDevice::new(1, 600);
        let pl = window_payload(&src, &p[0]).unwrap();
        dev.load_window(0, &p[0], &pl).unwrap();
        assert!(dev.load_window(0, &p[1], &window_payload(&src, &p[1]).unwrap()).is_err());
        let mut small = MockDevice::new(1, 100);
        assert!(small.load_window(0, &p[0], &pl).unwrap_err().contains("exceeds slot budget"));
    }
}
