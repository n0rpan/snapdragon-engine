//! flash-iobench: model-free bench of the Flash expert read path (F-IO, 2026-11-10).
//!
//! It runs the ENGINE'S OWN IO code (`qwen4exp/src/io.rs` and `uring.rs`, included by path: no model, no ggml)
//! against the real expert sidecar with O_DIRECT, and replays a real route trace (P2 CSV: turn,phase,step,layer,
//! slot,expert) through a copy of the engine's cache policy (LFU with aging + per-layer quotas, cache.rs) to get
//! the exact (token, layer, expert) miss sequence of a decode at a given cache size.
//!
//! Modes:
//!   layers  the decode pattern: per token, per layer, issue ALL the layer's misses at once (the engine's FETCH
//!           node), spin until all are READY (= the stall of that layer if compute hid nothing), then "think"
//!           (`--think-us`, the rest of the layer's compute). `--lookahead 1` also issues the NEXT layer's misses
//!           (oracle) at prefetch class at the same moment, plus `--la-waste K` useless reads per layer.
//!   closed  closed loop at `--depth D` whole-record reads from the same miss list: throughput and latency vs depth.
//!           `--len BYTES` overrides the read size, `--align-to BYTES` rounds the file offset down.
//!   misses  only the cache replay (misses per token), no IO.
//! IO configuration: `--io "cpus=8,9,10,11;split=4;..."` (IoTuning text form, same as the engine's OXF_IO),
//! `--qd`, `--io-cpu` (the single worker's core when `cpus=` is not given), `--fbuf 1` (register the buffer pool).
//! Output: one JSON line per run on stdout (and appended to `--json FILE`), human summary on stderr.

#[path = "../../qwen4exp/src/uring.rs"]
#[allow(dead_code)]
mod uring;
mod cache {
    pub const PENDING: u8 = 1;
    pub const READY: u8 = 2;
    pub const FAILED: u8 = 3;
}
#[path = "../../qwen4exp/src/io.rs"]
#[allow(dead_code)]
mod io;

use cache::{FAILED, PENDING, READY};
use io::{IoConfig, IoQueue, IoTuning, Promote, Req, ReqClass};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering::*};
use std::sync::Arc;
use std::time::{Duration, Instant};

const N_LAYER: usize = 48;
const N_EXPERT: usize = 512;
const LFU_AGE: u64 = 1000;
/// = qwen4exp::stream::FLASH_QUOTA_WEIGHTS (tools/flash-quota-weights.txt)
const QUOTA_WEIGHTS: [f64; 48] = [342.0, 270.0, 249.0, 138.0, 211.0, 245.0, 266.0, 263.0, 186.0, 172.0, 232.0, 193.0, 246.0, 242.0, 195.0, 121.0, 135.0, 198.0, 242.0, 200.0, 211.0, 236.0, 285.0, 183.0, 196.0, 176.0, 222.0, 196.0, 242.0, 233.0, 203.0, 160.0, 169.0, 227.0, 235.0, 202.0, 228.0, 227.0, 277.0, 192.0, 191.0, 242.0, 249.0, 202.0, 243.0, 226.0, 224.0, 218.0];

struct Args(HashMap<String, String>);
impl Args {
    fn parse() -> Args {
        let v: Vec<String> = std::env::args().skip(1).collect();
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].strip_prefix("--").unwrap_or_else(|| panic!("argument {}: want --key value", v[i]));
            let val = v.get(i + 1).unwrap_or_else(|| panic!("--{k}: missing value")).clone();
            m.insert(k.to_string(), val);
            i += 2;
        }
        Args(m)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn n<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.0.get(k).map(|v| v.parse().unwrap_or_else(|_| panic!("--{k} {v}: not a number"))).unwrap_or(d)
    }
}

struct Layout {
    slot: Vec<usize>,
    base: Vec<u64>,
    total: u64,
}

fn load_layout(manifest: &str) -> Layout {
    let m: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest).unwrap_or_else(|e| panic!("{manifest}: {e}"))).unwrap();
    let ls = m["layers"].as_array().expect("manifest layers");
    assert_eq!(ls.len(), N_LAYER);
    Layout {
        slot: ls.iter().map(|l| l["slot"].as_u64().unwrap() as usize).collect(),
        base: ls.iter().map(|l| l["base"].as_u64().unwrap()).collect(),
        total: m["total_bytes"].as_u64().unwrap(),
    }
}

/// cache.rs `Cache::quotas` with stream.rs's weights (weight x slot bytes, floor 64 slots)
fn quotas(lay: &Layout, budget: usize, builtin: bool) -> Vec<usize> {
    let floor = 64usize;
    let w: Vec<f64> = (0..N_LAYER).map(|l| if builtin { QUOTA_WEIGHTS[l] } else { 1.0 } * lay.slot[l] as f64).collect();
    let floor_bytes: usize = (0..N_LAYER).map(|l| floor * lay.slot[l]).sum();
    assert!(floor_bytes <= budget, "budget below the per-layer floor");
    let wsum: f64 = w.iter().sum();
    let spare = (budget - floor_bytes) as f64;
    let mut q: Vec<usize> = (0..N_LAYER).map(|l| (floor + (spare * w[l] / wsum / lay.slot[l] as f64).floor() as usize).min(N_EXPERT)).collect();
    while (0..N_LAYER).map(|l| q[l] * lay.slot[l]).sum::<usize>() > budget {
        let l = (0..N_LAYER).max_by_key(|&l| q[l]).unwrap();
        q[l] -= 1;
    }
    q
}

/// One layer's pool with cache.rs's LFU (counts for all experts, halved every LFU_AGE accesses, ties to the least
/// recently used, empty slot first, never a slot stamped by the current call).
struct Pool {
    map: Vec<i32>,
    owner: Vec<i32>,
    stamp: Vec<u64>,
    cnt: Vec<u32>,
    last: Vec<u64>,
    tick: u64,
}
impl Pool {
    fn new(n: usize) -> Pool {
        Pool { map: vec![-1; N_EXPERT], owner: vec![-1; n], stamp: vec![u64::MAX; n], cnt: vec![0; N_EXPERT], last: vec![0; n], tick: 0 }
    }
    /// true = miss
    fn lookup(&mut self, e: usize, stamp: u64) -> bool {
        self.tick += 1;
        self.cnt[e] = self.cnt[e].saturating_add(1);
        if self.tick % LFU_AGE == 0 {
            for c in self.cnt.iter_mut() {
                *c >>= 1;
            }
        }
        if self.map[e] >= 0 {
            let s = self.map[e] as usize;
            self.last[s] = self.tick;
            self.stamp[s] = stamp;
            return false;
        }
        let mut best: Option<(u32, u64, usize)> = None;
        for h in 0..self.owner.len() {
            if self.stamp[h] == stamp {
                continue;
            }
            if self.owner[h] < 0 {
                best = Some((0, 0, h));
                break;
            }
            let key = (self.cnt[self.owner[h] as usize], self.last[h], h);
            if best.is_none_or(|b| (key.0, key.1) < (b.0, b.1)) {
                best = Some(key);
            }
        }
        let v = best.expect("no evictable slot").2;
        self.last[v] = self.tick;
        if self.owner[v] >= 0 {
            self.map[self.owner[v] as usize] = -1;
        }
        self.owner[v] = e as i32;
        self.map[e] = v as i32;
        self.stamp[v] = stamp;
        true
    }
}

/// (decode tokens: per token the (layer, expert) misses in layer order, prefill distinct reads)
fn replay_trace(trace: &str, quota: &[usize]) -> (Vec<Vec<(u8, u16)>>, usize) {
    let text = std::fs::read_to_string(trace).unwrap_or_else(|e| panic!("{trace}: {e}"));
    let mut prefill: Vec<Vec<u16>> = vec![Vec::new(); N_LAYER]; // distinct, first-seen order
    let mut seen = vec![[false; N_EXPERT]; N_LAYER];
    // decode: step -> layer -> experts
    let mut dec: std::collections::BTreeMap<u32, Vec<Vec<u16>>> = Default::default();
    for line in text.lines().skip(1) {
        let f: Vec<u32> = line.split(',').map(|x| x.trim().parse().unwrap_or_else(|_| panic!("trace line {line}"))).collect();
        let (ph, step, l, e) = (f[1], f[2], f[3] as usize, f[5] as usize);
        if ph == 0 {
            if !seen[l][e] {
                seen[l][e] = true;
                prefill[l].push(e as u16);
            }
        } else {
            dec.entry(step).or_insert_with(|| vec![Vec::new(); N_LAYER])[l].push(e as u16);
        }
    }
    let mut pools: Vec<Pool> = quota.iter().map(|&n| Pool::new(n)).collect();
    let mut stamp = 0u64;
    let mut pre = 0;
    for l in 0..N_LAYER {
        for &e in &prefill[l] {
            stamp += 1;
            pre += pools[l].lookup(e as usize, stamp) as usize;
        }
    }
    let mut out = Vec::new();
    for (_, layers) in dec {
        let mut tok = Vec::new();
        for (l, ex) in layers.iter().enumerate() {
            stamp += 1;
            for &e in ex {
                if pools[l].lookup(e as usize, stamp) {
                    tok.push((l as u8, e));
                }
            }
        }
        out.push(tok);
    }
    (out, pre)
}

/// per-CPU (user+nice, system, irq, softirq) jiffies from /proc/stat
fn cpu_times() -> Vec<[u64; 4]> {
    let s = std::fs::read_to_string("/proc/stat").unwrap_or_default();
    s.lines()
        .filter(|l| l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(|b| b.is_ascii_digit()))
        .map(|l| {
            let f: Vec<u64> = l.split_whitespace().skip(1).map(|x| x.parse().unwrap_or(0)).collect();
            [f[0] + f[1], f[2], f[5], f[6]]
        })
        .collect()
}

/// involuntary context switches of all our threads: > a few per second means another process ran on our cores
fn invol_switches() -> u64 {
    let mut n = 0;
    if let Ok(d) = std::fs::read_dir("/proc/self/task") {
        for e in d.flatten() {
            // io_uring's own io-wq workers ("iou-wrk-*", they preempt the spinning worker once per punted read) do not count
            if std::fs::read_to_string(e.path().join("comm")).is_ok_and(|c| c.starts_with("iou-")) {
                continue;
            }
            if let Ok(s) = std::fs::read_to_string(e.path().join("status")) {
                n += s.lines().find_map(|l| l.strip_prefix("nonvoluntary_ctxt_switches:")).and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0);
            }
        }
    }
    n
}

fn spin_until(t: Instant) {
    while Instant::now() < t {
        std::hint::spin_loop();
    }
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn dist(mut v: Vec<f64>) -> serde_json::Value {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
    let r = |x: f64| (x * 1000.0).round() / 1000.0;
    serde_json::json!({"n": v.len(), "mean": r(mean), "p50": r(pct(&v, 0.5)), "p90": r(pct(&v, 0.9)), "p99": r(pct(&v, 0.99)), "max": r(pct(&v, 1.0))})
}

struct Bufs {
    arena: uring::Arena,
    slot: usize,
    n: usize,
    state: Vec<AtomicU8>,
    next: usize,
}
impl Bufs {
    fn new(n: usize, slot: usize) -> Bufs {
        Bufs { arena: uring::Arena::new(n * slot, true).expect("arena"), slot, n, state: (0..n).map(|_| AtomicU8::new(READY)).collect(), next: 0 }
    }
    /// a buffer no read is writing to, marked PENDING
    fn take(&mut self) -> usize {
        for _ in 0..self.n {
            let i = self.next;
            self.next = (self.next + 1) % self.n;
            if self.state[i].load(Acquire) != PENDING {
                self.state[i].store(PENDING, Release);
                return i;
            }
        }
        panic!("buffer pool exhausted (raise --bufs)");
    }
    fn ptr(&self, i: usize) -> *mut u8 {
        // SAFETY: inside the arena.
        unsafe { self.arena.ptr.add(i * self.slot) }
    }
}

fn main() {
    let a = Args::parse();
    let file = a.s("file", &format!("{}/.cache/local-ai/snapdragon-engine/flash/experts-r4.bin", std::env::var("HOME").unwrap()));
    let manifest = a.s("manifest", &format!("{}/.cache/local-ai/snapdragon-engine/flash/experts-r4.bin.json", std::env::var("HOME").unwrap()));
    let trace = a.s("trace", &std::env::var("OXF_EXPERT_TRACE").unwrap_or_else(|_| format!("{}/.cache/local-ai/snapdragon-engine/flash/flash-fixed353-expert-ids.csv", std::env::var("HOME").unwrap())));
    let mode = a.s("mode", "layers");
    let cache_gib: f64 = a.n("cache-gib", 20.0);
    let lay = load_layout(&manifest);
    let quota = quotas(&lay, (cache_gib * (1u64 << 30) as f64) as usize, a.s("quota", "builtin") == "builtin");
    let (mut toks, prefill_reads) = replay_trace(&trace, &quota);
    let n_all = toks.len();
    let skip: usize = a.n("skip-tokens", 0);
    let max_tok: usize = a.n("tokens", usize::MAX);
    toks = toks.into_iter().skip(skip).take(max_tok).collect();
    let misses: usize = toks.iter().map(|t| t.len()).sum();
    let bytes: u64 = toks.iter().flatten().map(|&(l, _)| lay.slot[l as usize] as u64).sum();
    let miss_layers: usize = toks.iter().map(|t| { let mut v: Vec<u8> = t.iter().map(|x| x.0).collect(); v.dedup(); v.len() }).sum();
    eprintln!(
        "trace: {n_all} decode tokens ({} used), cache {cache_gib} GiB: {:.2} misses/token, {:.2} layers with a miss/token, {:.1} MiB/token, prefill distinct reads {prefill_reads}",
        toks.len(),
        misses as f64 / toks.len().max(1) as f64,
        miss_layers as f64 / toks.len().max(1) as f64,
        bytes as f64 / toks.len().max(1) as f64 / 1048576.0
    );
    let mut out = serde_json::json!({
        "tool": "flash-iobench", "mode": mode, "file": file, "cache_gib": cache_gib, "tokens": toks.len(),
        "misses_per_token": misses as f64 / toks.len().max(1) as f64,
        "miss_layers_per_token": miss_layers as f64 / toks.len().max(1) as f64,
        "mib_per_token": bytes as f64 / toks.len().max(1) as f64 / 1048576.0,
        "label": a.s("label", ""),
    });
    if mode == "misses" {
        println!("{out}");
        return;
    }

    let ctl_cpu: usize = a.n("ctl-cpu", 7);
    sched::pin_current_thread(ctl_cpu).expect("pin ctl");
    let fd = uring::open_direct(std::path::Path::new(&file)).unwrap_or_else(|e| panic!("{file}: {e}"));
    assert_eq!(std::fs::metadata(&file).unwrap().len(), lay.total, "file size vs manifest");
    let max_slot = *lay.slot.iter().max().unwrap();
    let mut bufs = Bufs::new(a.n("bufs", 128), max_slot);
    let mut tuning = IoTuning::parse(&a.s("io", "")).unwrap_or_else(|e| panic!("--io: {e}"));
    if a.n("fbuf", 0u32) != 0 {
        tuning.fixed_buf = Some((bufs.arena.ptr as usize, bufs.arena.len));
    }
    let cfg = IoConfig { qd: a.n("qd", 32u32), prefetch_qd: a.n("prefetch-qd", 16u32), cpu: Some(a.n("io-cpu", 8usize)) };
    let desc = tuning.describe(&cfg);
    eprintln!("io: {desc}; ctl on cpu {ctl_cpu}");
    let q = Arc::new(IoQueue::new());
    let h = io::spawn_tuned(q.clone(), cfg, tuning);
    std::thread::sleep(Duration::from_millis(50));
    let len_override: u32 = a.n("len", 0u32);
    let align_to: u64 = a.n("align-to", 0u64);
    let req = |bufs: &mut Bufs, l: usize, e: usize, class: ReqClass| -> usize {
        let i = bufs.take();
        let mut off = lay.base[l] + (e * lay.slot[l]) as u64;
        if align_to > 0 {
            off &= !(align_to - 1);
        }
        let len = if len_override > 0 { len_override } else { lay.slot[l] as u32 };
        q.push(Req { class, fd, buf: bufs.ptr(i), len, off, state: &bufs.state[i] });
        i
    };
    let wait = |bufs: &Bufs, i: usize| {
        let t = Instant::now();
        loop {
            match bufs.state[i].load(Acquire) {
                READY => return,
                FAILED => panic!("read failed: {:?}", q.take_error()),
                _ => {}
            }
            if t.elapsed() > Duration::from_secs(20) {
                panic!("read not complete after 20 s");
            }
            std::hint::spin_loop();
        }
    };
    // warm-up: 64 reads, not measured (ring setup, device power state, first-touch)
    for k in 0..64usize {
        let i = req(&mut bufs, k % N_LAYER, (k * 37) % N_EXPERT, ReqClass::Demand);
        wait(&bufs, i);
    }
    let c = &q.counters;
    let snap = || (c.completed[0].load(Relaxed), c.lat_ns[0].load(Relaxed), c.bytes[0].load(Relaxed) + c.bytes[2].load(Relaxed), c.parks.load(Relaxed), c.busy_ns.load(Relaxed), c.parts.load(Relaxed), c.completed[2].load(Relaxed));
    let s0 = snap();
    let cpu0 = cpu_times();
    let inv0 = invol_switches();
    let t_run = Instant::now();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;

    if mode == "layers" {
        let think = Duration::from_micros(a.n("think-us", 0u64));
        let la: usize = a.n("lookahead", 0);
        let waste: usize = a.n("la-waste", 0);
        let mut layer_stall = Vec::new(); // ms, layers with >= 1 miss
        let mut tok_stall = Vec::new(); // ms per token
        let mut tok_wall = Vec::new();
        let mut stall_by_n: Vec<Vec<f64>> = vec![Vec::new(); 11];
        let mut spec: HashMap<(u8, u16), usize> = HashMap::new(); // speculative reads issued for the next layer
        let (mut promoted, mut awaited, mut spec_ready) = (0u64, 0u64, 0u64);
        let mut rng = 0x9e3779b97f4a7c15u64;
        // flat (token, layer) sequence so "the next layer" crosses token ends
        let mut seq: Vec<Vec<(u8, u16)>> = Vec::new();
        for t in &toks {
            for l in 0..N_LAYER {
                seq.push(t.iter().filter(|x| x.0 as usize == l).copied().collect());
            }
        }
        let mut cur: Vec<usize> = Vec::new();
        for (si, m) in seq.iter().enumerate() {
            let t_tok = if si % N_LAYER == 0 { Some(Instant::now()) } else { None };
            if let Some(t) = t_tok {
                tok_wall.push(t);
                tok_stall.push(0.0);
            }
            let t0 = Instant::now();
            cur.clear();
            for &(l, e) in m {
                if let Some(i) = spec.remove(&(l, e)) {
                    // the engine's "demand hit on a pending prefetched slot"
                    if bufs.state[i].load(Acquire) != READY {
                        match q.promote(&bufs.state[i]) {
                            Promote::Moved => promoted += 1,
                            Promote::InFlight => awaited += 1,
                        }
                    } else {
                        spec_ready += 1;
                    }
                    cur.push(i);
                } else {
                    cur.push(req(&mut bufs, l as usize, e as usize, ReqClass::Demand));
                }
            }
            if la > 0 {
                // wasted speculative reads of the last round are forgotten (their buffers free up when READY)
                spec.clear();
                if let Some(next) = seq.get(si + 1) {
                    let nl = (si + 1) % N_LAYER;
                    for &(l, e) in next {
                        spec.insert((l, e), req(&mut bufs, l as usize, e as usize, ReqClass::Prefetch));
                    }
                    for _ in 0..waste {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        req(&mut bufs, nl, (rng % N_EXPERT as u64) as usize, ReqClass::Prefetch);
                    }
                }
            }
            for &i in &cur {
                wait(&bufs, i);
            }
            if !m.is_empty() {
                let st = ms(t0.elapsed());
                layer_stall.push(st);
                stall_by_n[m.len().min(10)].push(st);
                *tok_stall.last_mut().unwrap() += st;
            }
            if !think.is_zero() {
                spin_until(Instant::now() + think);
            }
        }
        let wall = t_run.elapsed();
        let by_n: Vec<serde_json::Value> = (1..=10).filter(|&n| !stall_by_n[n].is_empty()).map(|n| serde_json::json!({"misses": n, "stall_ms": dist(stall_by_n[n].clone())})).collect();
        out["think_us"] = serde_json::json!(think.as_micros() as u64);
        out["lookahead"] = serde_json::json!(la);
        out["la_waste"] = serde_json::json!(waste);
        out["stall_ms_per_token"] = dist(tok_stall);
        out["layer_stall_ms"] = dist(layer_stall);
        out["layer_stall_by_misses"] = serde_json::json!(by_n);
        out["spec"] = serde_json::json!({"ready_at_demand": spec_ready, "promoted": promoted, "awaited_in_flight": awaited});
        out["wall_s"] = serde_json::json!(wall.as_secs_f64());
    } else if mode == "closed" {
        let depth: usize = a.n("depth", 1);
        let list: Vec<(u8, u16)> = toks.iter().flatten().copied().collect();
        let max_reads: usize = a.n("reads", list.len());
        let list = &list[..max_reads.min(list.len())];
        let mut live: Vec<(usize, Instant)> = Vec::new();
        let mut lat = Vec::new();
        let mut next = 0;
        while next < list.len() || !live.is_empty() {
            while live.len() < depth && next < list.len() {
                let (l, e) = list[next];
                next += 1;
                let i = req(&mut bufs, l as usize, e as usize, ReqClass::Demand);
                live.push((i, Instant::now()));
            }
            let mut k = 0;
            while k < live.len() {
                match bufs.state[live[k].0].load(Acquire) {
                    READY => {
                        lat.push(ms(live[k].1.elapsed()));
                        live.swap_remove(k);
                    }
                    FAILED => panic!("read failed: {:?}", q.take_error()),
                    _ => k += 1,
                }
            }
            std::hint::spin_loop();
        }
        out["depth"] = serde_json::json!(depth);
        out["len"] = serde_json::json!(len_override);
        out["align_to"] = serde_json::json!(align_to);
        out["latency_ms"] = dist(lat);
        out["wall_s"] = serde_json::json!(t_run.elapsed().as_secs_f64());
    } else {
        panic!("--mode layers | closed | misses");
    }
    let wall = t_run.elapsed().as_secs_f64();
    // let stragglers (wasted speculative reads) finish before the counters are read
    std::thread::sleep(Duration::from_millis(100));
    let s1 = snap();
    let cpu1 = cpu_times();
    let inv = invol_switches().saturating_sub(inv0);
    out["invol_ctx_switches"] = serde_json::json!(inv);
    out["invol_per_s"] = serde_json::json!(inv as f64 / wall);
    out["loadavg"] = serde_json::json!(std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim());
    if inv as f64 / wall > 20.0 {
        eprintln!("WARNING: {inv} involuntary context switches ({:.0}/s): another process ran on the bench cores; this arm is PERTURBED", inv as f64 / wall);
    }
    let reads = s1.0 - s0.0;
    out["io"] = serde_json::json!({
        "config": desc,
        "demand_reads": reads, "prefetch_reads": s1.6 - s0.6, "parts": s1.5 - s0.5,
        "mean_latency_ms": (s1.1 - s0.1) as f64 / 1e6 / reads.max(1) as f64,
        "mib_per_s_wall": (s1.2 - s0.2) as f64 / 1048576.0 / wall,
        "parks": s1.3 - s0.3, "worker_busy_s": (s1.4 - s0.4) as f64 / 1e9,
        "max_inflight_per_worker": c.max_inflight.load(Relaxed), "retried": c.retried.load(Relaxed), "failed": c.failed.load(Relaxed),
    });
    // where the kernel spent time: per-CPU [user, system, irq, softirq] ms over the run (USER_HZ 100)
    let cpu: Vec<serde_json::Value> = cpu0.iter().zip(&cpu1).enumerate().map(|(i, (x, y))| serde_json::json!([i, (y[0] - x[0]) * 10, (y[1] - x[1]) * 10, (y[2] - x[2]) * 10, (y[3] - x[3]) * 10])).collect();
    out["cpu_ms_user_sys_irq_softirq"] = serde_json::json!(cpu);
    q.stop();
    h.join().unwrap();
    println!("{out}");
    if let Some(p) = a.0.get("json") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap();
        writeln!(f, "{out}").unwrap();
    }
    if mode == "layers" {
        eprintln!(
            "stall ms/token {} | layer stall ms {} | reads {} mean latency {:.3} ms | {:.0} MiB/s wall | parks {}",
            out["stall_ms_per_token"], out["layer_stall_ms"], reads, out["io"]["mean_latency_ms"].as_f64().unwrap(), out["io"]["mib_per_s_wall"].as_f64().unwrap(), out["io"]["parks"]
        );
    } else {
        eprintln!("latency ms {} | {:.0} MiB/s | parks {}", out["latency_ms"], out["io"]["mib_per_s_wall"].as_f64().unwrap(), out["io"]["parks"]);
    }
}
