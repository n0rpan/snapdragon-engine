//! The IO workers: one io_uring each, pinned to their own cores outside the compute team, spinning while there is
//! work and parked on a condition variable when idle (woken by `push` / `stop`, 100 ms timeout).
//! Requests arrive through small mutex-protected queues per worker and priority class (demand before
//! N-gram before prefetch; prefetches capped so a demand read never queues behind many of
//! them). A completion flips the request's state word to READY with a release store. A short or failed read is
//! retried once; a second failure marks the slot FAILED and records the error (`IoQueue::take_error`) instead of
//! aborting the process: the engine fails the request, not the server.
//!
//! F-IO (2026-11-10, `IoTuning`, default = the old single thread, whole reads):
//! * WHY: the sidecar is on dm-crypt opened with `no_read_workqueue`, so every 256 KiB bio of a read is decrypted
//!   inline in the completion softirq of the CPU that SUBMITTED it. One IO thread = one core decrypting: a 2 MiB
//!   expert record costs ~1.8 ms of serial AES-XTS on the critical path (tools/flash-iobench measures it).
//! * `cpus=a,b,c`: several workers, one ring each, one core each. `split=N`: a read is striped into up to N parts
//!   (>= `min` bytes each, boundaries on `align`), one part per worker, so the parts decrypt in parallel; the
//!   state word flips READY when the last part lands (`Join`). Unsplit reads go round-robin over the workers.
//! * ONE READ PER SLOT, ALWAYS (not an option): a demand lookup that finds its slot still PENDING from a prefetch
//!   moves the still-queued parts to the demand class (`IoQueue::promote`) and otherwise just waits for the parts in
//!   flight. The old code read the record a second time into the same slot; the first completion flipped READY and
//!   the second read then rewrote the buffer under the kernels (through dm-crypt + O_DIRECT the buffer holds
//!   ciphertext until it is decrypted in place): wrong outputs after `Stream::warm` (prefill lane's evidence test,
//!   now the regression test `stream::tests::warm_then_batched_op_single_read`). Workers pop from the shared queues
//!   only what fits their ring, so queued parts stay promotable. (`promote=` is still accepted in the text form.)
//! * `sqpoll=cpu,..`, `defer=1`, `ffile=1`, fixed buffer: ring variants for the bench.
//! This file depends only on `crate::cache::{FAILED, READY}`, `crate::uring` and the `sched` crate, so
//! tools/flash-iobench includes it by path and benches exactly this code without the model.

use crate::cache::{FAILED, READY};
use crate::uring::{Ring, RingOpts};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

pub const MAX_WORKERS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReqClass {
    Demand = 0,
    Ngram = 1,
    Prefetch = 2,
}

pub struct Req {
    pub class: ReqClass,
    pub fd: i32,
    pub buf: *mut u8,
    pub len: u32,
    pub off: u64,
    pub state: *const AtomicU8,
}
unsafe impl Send for Req {}

/// One logical read (a `Req`), possibly striped into parts: READY / FAILED is stored when the last part lands.
struct Join {
    state: *const AtomicU8,
    left: AtomicU32,
    failed: AtomicBool,
    t0: Instant,
    /// FD-1 trace: push time and the time the first part entered a ring (0 = not yet), ns since `epoch`
    push_ns: u64,
    submit_ns: AtomicU64,
    len: u32,
    class: ReqClass,
}
unsafe impl Send for Join {}
unsafe impl Sync for Join {}

struct Part {
    /// queue class (a promoted prefetch part is Demand here; `Join::class` keeps the original for the counters)
    class: ReqClass,
    fd: i32,
    buf: *mut u8,
    len: u32,
    off: u64,
    retried: bool,
    join: Arc<Join>,
}
unsafe impl Send for Part {}

#[derive(Default)]
pub struct IoCounters {
    pub completed: [AtomicU64; 3],
    pub bytes: [AtomicU64; 3],
    /// sum of push->ready latency of whole requests, ns (before F-IO: SQE submit->complete)
    pub lat_ns: [AtomicU64; 3],
    pub lat_max_ns: [AtomicU64; 3],
    pub max_inflight: AtomicU64,
    pub busy_ns: AtomicU64,
    /// reads that failed twice (slot marked FAILED) / retries issued
    pub failed: AtomicU64,
    pub retried: AtomicU64,
    /// times an IO worker parked on the condition variable
    pub parks: AtomicU64,
    /// parts submitted (= requests when nothing is striped) / queued prefetch parts moved to demand
    pub parts: AtomicU64,
    pub promoted: AtomicU64,
}

/// FD-1 per-read trace record: class, push -> submit (first part in a ring) -> completion (last part landed), ns
/// since the process's first trace
#[derive(Clone, Copy, Debug)]
pub struct ReadStamp {
    pub class: u8,
    pub push_ns: u64,
    pub submit_ns: u64,
    pub done_ns: u64,
}

fn epoch() -> Instant {
    static E: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *E.get_or_init(Instant::now)
}
fn now_ns() -> u64 {
    epoch().elapsed().as_nanos() as u64
}

#[derive(Default)]
struct WorkerQ {
    pushed: AtomicU64,
    sleeping: AtomicBool,
    cv: Condvar,
}

/// What `IoQueue::promote` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Promote {
    /// queued parts were moved to the demand class
    Moved,
    /// nothing queued: every part is already in flight (or done); just wait
    InFlight,
}

pub struct IoQueue {
    /// [worker][class]
    q: Mutex<Vec<[VecDeque<Part>; 3]>>,
    w: [WorkerQ; MAX_WORKERS],
    trace_on: AtomicBool,
    stamps: Mutex<Vec<ReadStamp>>,
    workers: AtomicUsize,
    split: AtomicU32,
    split_min: AtomicU32,
    split_align: AtomicU32,
    split_prefetch: AtomicBool,
    rr: AtomicUsize,
    stop: AtomicBool,
    error: Mutex<Option<String>>,
    pub counters: IoCounters,
}

impl IoQueue {
    pub fn new() -> IoQueue {
        IoQueue {
            q: Mutex::new((0..MAX_WORKERS).map(|_| [VecDeque::new(), VecDeque::new(), VecDeque::new()]).collect()),
            w: Default::default(),
            trace_on: AtomicBool::new(false),
            stamps: Mutex::new(Vec::new()),
            workers: AtomicUsize::new(1),
            split: AtomicU32::new(1),
            split_min: AtomicU32::new(256 << 10),
            split_align: AtomicU32::new(4096),
            split_prefetch: AtomicBool::new(false),
            rr: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            error: Mutex::new(None),
            counters: IoCounters::default(),
        }
    }

    /// Set by `spawn_tuned` before the workers start.
    fn configure(&self, t: &IoTuning) {
        self.workers.store(t.workers(), SeqCst);
        self.split.store(t.split.max(1), SeqCst);
        self.split_min.store(t.split_min.max(4096), SeqCst);
        self.split_align.store(t.split_align.max(4096).next_multiple_of(4096), SeqCst);
        self.split_prefetch.store(t.split_prefetch, SeqCst);
    }

    /// The parts of a read of `len` bytes: (offset in the read, bytes) with boundaries on `align`.
    fn stripe(len: u32, parts: u32, align: u32) -> impl Iterator<Item = (u32, u32)> {
        let per = len.div_ceil(parts.max(1)).next_multiple_of(align).max(align);
        (0..len.div_ceil(per)).map(move |i| (i * per, per.min(len - i * per)))
    }

    pub fn push(&self, r: Req) {
        let n = self.workers.load(Relaxed).max(1);
        let mut parts = 1u32;
        if n > 1 && (r.class == ReqClass::Demand || self.split_prefetch.load(Relaxed)) {
            parts = (r.len / self.split_min.load(Relaxed)).clamp(1, self.split.load(Relaxed).min(n as u32));
        }
        let align = self.split_align.load(Relaxed);
        let np = Self::stripe(r.len, parts, align).count() as u32;
        let push_ns = if self.trace_on.load(Relaxed) { now_ns().max(1) } else { 0 };
        let join = Arc::new(Join { state: r.state, left: AtomicU32::new(np), failed: AtomicBool::new(false), t0: Instant::now(), push_ns, submit_ns: AtomicU64::new(0), len: r.len, class: r.class });
        let start = self.rr.fetch_add(1, Relaxed);
        {
            let mut g = self.q.lock().unwrap();
            for (i, (o, l)) in Self::stripe(r.len, parts, align).enumerate() {
                g[(start + i) % n][r.class as usize].push_back(Part {
                    class: r.class,
                    fd: r.fd,
                    // SAFETY: inside the caller's buffer of r.len bytes.
                    buf: unsafe { r.buf.add(o as usize) },
                    len: l,
                    off: r.off + o as u64,
                    retried: false,
                    join: join.clone(),
                });
            }
        }
        self.counters.parts.fetch_add(np as u64, Relaxed);
        for i in 0..(np as usize).min(n) {
            let wi = (start + i) % n;
            // SeqCst pair with the worker's (sleeping = true; read pushed): one of the two sees the other
            self.w[wi].pushed.fetch_add(1, SeqCst);
            self.wake(wi);
        }
    }

    /// A demand lookup found the slot behind `state` still PENDING from a prefetch: move its queued parts to the
    /// demand class. Parts already in a ring are left alone (they complete at device speed).
    pub fn promote(&self, state: *const AtomicU8) -> Promote {
        let n = self.workers.load(Relaxed).max(1);
        let mut touched = [false; MAX_WORKERS];
        let mut moved = 0u64;
        {
            let mut g = self.q.lock().unwrap();
            for wi in 0..n {
                for class in 1..3 {
                    let mut i = 0;
                    while i < g[wi][class].len() {
                        if g[wi][class][i].join.state == state {
                            let mut p = g[wi][class].remove(i).unwrap();
                            p.class = ReqClass::Demand;
                            g[wi][0].push_back(p);
                            touched[wi] = true;
                            moved += 1;
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }
        if moved == 0 {
            return Promote::InFlight;
        }
        self.counters.promoted.fetch_add(moved, Relaxed);
        for wi in 0..n {
            if touched[wi] {
                self.w[wi].pushed.fetch_add(1, SeqCst);
                self.wake(wi);
            }
        }
        Promote::Moved
    }

    pub fn stop(&self) {
        self.stop.store(true, SeqCst);
        for wi in 0..MAX_WORKERS {
            self.wake(wi);
        }
    }
    fn wake(&self, wi: usize) {
        if self.w[wi].sleeping.load(SeqCst) {
            let _g = self.q.lock().unwrap();
            self.w[wi].cv.notify_one();
        }
    }
    /// FD-1: start / stop recording one ReadStamp per completed read (push, submit, completion times).
    pub fn set_trace(&self, on: bool) {
        if on {
            self.stamps.lock().unwrap().clear();
        }
        self.trace_on.store(on, SeqCst);
    }
    /// The recorded stamps (and clears them).
    pub fn take_stamps(&self) -> Vec<ReadStamp> {
        std::mem::take(&mut *self.stamps.lock().unwrap())
    }

    /// The first read error since the last call (a slot was marked FAILED), if any.
    pub fn take_error(&self) -> Option<String> {
        self.error.lock().unwrap().take()
    }
}

impl Default for IoQueue {
    fn default() -> Self {
        IoQueue::new()
    }
}

pub struct IoConfig {
    /// ring depth per worker
    pub qd: u32,
    pub prefetch_qd: u32,
    pub cpu: Option<usize>,
}

/// F-IO options (module docs). `IoTuning::default()` = the pre-F-IO behaviour. Text form (env `OXF_IO`, bench
/// `--io`): `key=value` pairs separated by `;` or whitespace, e.g. `cpus=8,9,10,11;split=4;promote=1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoTuning {
    /// one worker per entry, pinned there (empty = one worker on `IoConfig::cpu`); at most MAX_WORKERS
    pub cpus: Vec<usize>,
    /// stripe a demand read into up to this many parts, one per worker (1 = whole reads)
    pub split: u32,
    /// smallest part, bytes
    pub split_min: u32,
    /// part boundary alignment, bytes (a multiple of 4096)
    pub split_align: u32,
    /// also stripe N-gram / prefetch reads
    pub split_prefetch: bool,
    /// accepted and ignored: promotion is always on (one read per slot); kept so `promote=1` strings still parse
    pub promote: bool,
    /// spins of an idle worker before it parks
    pub idle_spins: u32,
    /// SQPOLL rings; `sqpoll_cpus[i]` pins worker i's kernel submit thread
    pub sqpoll: bool,
    pub sqpoll_cpus: Vec<usize>,
    pub defer_taskrun: bool,
    pub fixed_file: bool,
    /// register this region (address, bytes) as a fixed buffer in every ring; set by the owner of the arena
    pub fixed_buf: Option<(usize, usize)>,
}

impl Default for IoTuning {
    fn default() -> Self {
        IoTuning {
            cpus: Vec::new(),
            split: 1,
            split_min: 256 << 10,
            split_align: 4096,
            split_prefetch: false,
            promote: true,
            idle_spins: IDLE_SPINS,
            sqpoll: false,
            sqpoll_cpus: Vec::new(),
            defer_taskrun: false,
            fixed_file: false,
            fixed_buf: None,
        }
    }
}

impl IoTuning {
    pub fn workers(&self) -> usize {
        self.cpus.len().clamp(1, MAX_WORKERS)
    }

    pub fn parse(s: &str) -> Result<IoTuning, String> {
        let mut t = IoTuning::default();
        let list = |v: &str| v.split(',').filter(|x| !x.is_empty()).map(|x| x.parse::<usize>().map_err(|e| format!("{x}: {e}"))).collect::<Result<Vec<_>, _>>();
        let num = |v: &str| v.parse::<u32>().map_err(|e| format!("{v}: {e}"));
        for kv in s.split(|c: char| c == ';' || c.is_whitespace()).filter(|x| !x.is_empty()) {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("io option {kv}: want key=value"))?;
            match k {
                "cpus" => t.cpus = list(v)?,
                "split" => t.split = num(v)?,
                "min" => t.split_min = num(v)?,
                "align" => t.split_align = num(v)?,
                "splitpf" => t.split_prefetch = num(v)? != 0,
                "promote" => {
                    num(v)?; // always on
                }
                "spins" => t.idle_spins = num(v)?,
                "sqpoll" => {
                    t.sqpoll = true;
                    t.sqpoll_cpus = list(v)?;
                }
                "defer" => t.defer_taskrun = num(v)? != 0,
                "ffile" => t.fixed_file = num(v)? != 0,
                _ => return Err(format!("io option {k}: unknown (cpus, split, min, align, splitpf, promote, spins, sqpoll, defer, ffile)")),
            }
        }
        if t.cpus.len() > MAX_WORKERS {
            return Err(format!("io cpus: at most {MAX_WORKERS} workers"));
        }
        if t.split == 0 || t.split_min < 4096 || t.split_align % 4096 != 0 || t.split_align == 0 {
            return Err("io split >= 1, min >= 4096, align a multiple of 4096".into());
        }
        Ok(t)
    }

    /// The Flash engine's IO default (F-IO wave 3: one IO thread that spins ~1 s before parking, so it never parks
    /// between the layers of a token; -18.5 ms/token predicted with the plain sidecar). Used when OXF_IO is unset;
    /// `OXF_IO=default` restores `IoTuning::default()` (parks after 20k idle spins).
    pub const FLASH_DEFAULT: &'static str = "spins=1000000";

    /// The Flash engine's tuning: OXF_IO when set (`default` = `IoTuning::default()`), else `FLASH_DEFAULT`.
    pub fn flash_from_env() -> Result<IoTuning, String> {
        match std::env::var("OXF_IO") {
            Ok(s) if s.trim() == "default" => Ok(IoTuning::default()),
            Ok(s) => IoTuning::parse(&s).map_err(|e| format!("OXF_IO: {e}")),
            Err(_) => IoTuning::parse(Self::FLASH_DEFAULT),
        }
    }

    /// `OXF_IO` (unset or empty = default).
    pub fn from_env() -> Result<IoTuning, String> {
        match std::env::var("OXF_IO") {
            Ok(s) => IoTuning::parse(&s).map_err(|e| format!("OXF_IO: {e}")),
            Err(_) => Ok(IoTuning::default()),
        }
    }

    /// One line for receipts.
    pub fn describe(&self, cfg: &IoConfig) -> String {
        format!(
            "workers={} cpus={:?} qd={} split={} min={} align={} splitpf={} promote={} spins={} sqpoll={}{:?} defer={} ffile={} fbuf={}",
            self.workers(),
            if self.cpus.is_empty() { cfg.cpu.into_iter().collect::<Vec<_>>() } else { self.cpus.clone() },
            cfg.qd,
            self.split,
            self.split_min,
            self.split_align,
            self.split_prefetch as u8,
            self.promote as u8,
            self.idle_spins,
            self.sqpoll as u8,
            self.sqpoll_cpus,
            self.defer_taskrun as u8,
            self.fixed_file as u8,
            self.fixed_buf.is_some() as u8
        )
    }
}

/// Spawn the IO thread (default tuning). It runs until `IoQueue::stop` and every request has completed.
pub fn spawn(q: Arc<IoQueue>, cfg: IoConfig) -> std::thread::JoinHandle<()> {
    spawn_tuned(q, cfg, IoTuning::default())
}

/// Spawn the IO workers; the returned handle is worker 0, which starts and joins the others.
pub fn spawn_tuned(q: Arc<IoQueue>, cfg: IoConfig, t: IoTuning) -> std::thread::JoinHandle<()> {
    q.configure(&t);
    std::thread::Builder::new()
        .name("oxf-io".into())
        .spawn(move || {
            let (cfg, t) = (Arc::new(cfg), Arc::new(t));
            let others: Vec<_> = (1..t.workers())
                .map(|wi| {
                    let (q, cfg, t) = (q.clone(), cfg.clone(), t.clone());
                    std::thread::Builder::new().name(format!("oxf-io{wi}")).spawn(move || worker(&q, wi, &cfg, &t)).unwrap()
                })
                .collect();
            worker(&q, 0, &cfg, &t);
            for h in others {
                h.join().ok();
            }
        })
        .unwrap()
}

fn worker(q: &IoQueue, wi: usize, cfg: &IoConfig, t: &IoTuning) {
    let cpu = if t.cpus.is_empty() { cfg.cpu } else { Some(t.cpus[wi]) };
    // usize::MAX = leave this worker unpinned (tests)
    if let Some(c) = cpu.filter(|&c| c != usize::MAX) {
        if let Err(e) = sched::pin_current_thread(c) {
            eprintln!("oxf-io{wi}: pin to cpu {c} failed: {e}");
        }
    }
    io_loop(q, wi, cfg, t)
}

struct Flight {
    part: Part,
}

fn io_loop(q: &IoQueue, wi: usize, cfg: &IoConfig, t: &IoTuning) {
    let entries = 64.max(cfg.qd.next_power_of_two());
    let opts = RingOpts { sqpoll: t.sqpoll, sqpoll_cpu: t.sqpoll_cpus.get(wi).map(|&c| c as u32), sqpoll_idle_ms: 2000, defer_taskrun: t.defer_taskrun };
    let mut ring = match Ring::with_opts(entries, opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("oxf-io{wi}: io_uring_setup with {opts:?} failed ({e}); using the plain ring");
            Ring::new(entries).expect("io_uring_setup")
        }
    };
    if let Some((p, l)) = t.fixed_buf {
        // SAFETY: the owner of the tuning keeps the region mapped while the IO threads run.
        if let Err(e) = unsafe { ring.register_buffer(p as *mut u8, l) } {
            eprintln!("oxf-io{wi}: fixed buffer of {l} bytes not registered ({e}); plain reads");
        }
    }
    let mut file_registered = false;
    let me = &q.w[wi];
    let mut seen = 0u64;
    let mut backlog = false;
    let mut cap_changed = false;
    let mut batch: VecDeque<Part> = VecDeque::new();
    let mut flights: Vec<Option<Flight>> = (0..ring_cap(cfg)).map(|_| None).collect();
    let mut free: Vec<usize> = (0..flights.len()).rev().collect();
    let mut pf_inflight = 0u32;
    let mut idle = 0u32;
    let mut retry_q: Vec<Part> = Vec::new();
    let c = &q.counters;
    loop {
        let busy0 = Instant::now();
        // take from the shared queues only what fits the ring now: demand, then N-gram, then prefetch (capped);
        // what stays queued stays promotable
        let p = me.pushed.load(Acquire);
        if batch.is_empty() && (p != seen || (backlog && cap_changed)) {
            cap_changed = false;
            let mut room = cfg.qd.saturating_sub(ring.inflight).min(free.len() as u32);
            if room == 0 {
                backlog = true;
            } else {
                seen = p;
                let mut g = q.q.lock().unwrap();
                let qs = &mut g[wi];
                for class in 0..3 {
                    while room > 0 && !(class == 2 && pf_inflight >= cfg.prefetch_qd) {
                        let Some(part) = qs[class].pop_front() else { break };
                        if part.class == ReqClass::Prefetch {
                            pf_inflight += 1;
                        }
                        batch.push_back(part);
                        room -= 1;
                    }
                }
                backlog = qs.iter().any(|d| !d.is_empty());
            }
        }
        let mut queued = false;
        while let Some(part) = batch.front() {
            let Some(id) = free.pop() else { break };
            if t.fixed_file && !file_registered {
                file_registered = true;
                if let Err(e) = ring.register_file(part.fd) {
                    eprintln!("oxf-io{wi}: fixed file not registered ({e})");
                }
            }
            // SAFETY: the buffer is a cache/page slot reserved (PENDING) for this read.
            if !unsafe { ring.push_read(part.fd, part.buf, part.len, part.off, id as u64) } {
                free.push(id);
                break;
            }
            let part = batch.pop_front().unwrap();
            if part.join.push_ns != 0 {
                let _ = part.join.submit_ns.compare_exchange(0, now_ns(), AcqRel, Relaxed);
            }
            flights[id] = Some(Flight { part });
            queued = true;
        }
        if queued || ring.inflight > 0 {
            ring.enter(false).expect("io_uring_enter");
            c.max_inflight.fetch_max(ring.inflight as u64, Relaxed);
        }
        let n = ring.reap(|cq| {
            let id = cq.user_data as usize;
            let mut f = flights[id].take().expect("known completion").part;
            free.push(id);
            if cq.res != f.len as i32 {
                if !f.retried {
                    // retry once, ahead of everything else this worker holds
                    c.retried.fetch_add(1, Relaxed);
                    f.retried = true;
                    retry_q.push(f);
                    return;
                }
                c.failed.fetch_add(1, Relaxed);
                let mut e = q.error.lock().unwrap();
                if e.is_none() {
                    *e = Some(format!("expert/N-gram read failed twice: res {} (wanted {} bytes) at offset {}", cq.res, f.len, f.off));
                }
                f.join.failed.store(true, Release);
            }
            if f.class == ReqClass::Prefetch {
                pf_inflight -= 1;
            }
            let j = &f.join;
            if j.left.fetch_sub(1, AcqRel) == 1 {
                // last part of the request
                if j.failed.load(Acquire) {
                    // SAFETY: as below.
                    unsafe { (*j.state).store(FAILED, Release) };
                    return;
                }
                let k = j.class as usize;
                let ns = j.t0.elapsed().as_nanos() as u64;
                if j.push_ns != 0 && q.trace_on.load(Relaxed) {
                    let done = now_ns();
                    q.stamps.lock().unwrap().push(ReadStamp { class: j.class as u8, push_ns: j.push_ns, submit_ns: j.submit_ns.load(Acquire).max(j.push_ns), done_ns: done });
                }
                c.completed[k].fetch_add(1, Relaxed);
                c.bytes[k].fetch_add(j.len as u64, Relaxed);
                c.lat_ns[k].fetch_add(ns, Relaxed);
                c.lat_max_ns[k].fetch_max(ns, Relaxed);
                // SAFETY: the state word lives in the cache / page table, which outlives the IO threads.
                unsafe { (*j.state).store(READY, Release) };
            }
        });
        for f in retry_q.drain(..) {
            batch.push_front(f);
        }
        if n > 0 {
            cap_changed = true;
        }
        if queued || n > 0 {
            c.busy_ns.fetch_add(busy0.elapsed().as_nanos() as u64, Relaxed);
        }
        if ring.inflight == 0 && batch.is_empty() && backlog {
            cap_changed = true; // nothing in flight can free capacity later: look at the queue again
        }
        let idle_now = ring.inflight == 0 && batch.is_empty() && !backlog && me.pushed.load(SeqCst) == seen;
        if q.stop.load(Acquire) && idle_now {
            return;
        }
        // park when idle: spin briefly (a demand read usually follows within microseconds while a token runs), then
        // wait on the condition variable (woken by push / stop; 100 ms timeout bounds a missed wake); never while reads
        // are in flight. A shared core no longer loses a whole core to the spin (ticket 011: 12 threads with IO
        // sharing core 8 ran at 15.3 vs 22.7 tok/s), and an idle engine does not wake 50k times a second (audit).
        if idle_now && !queued && n == 0 {
            idle += 1;
            if idle > t.idle_spins {
                let g = q.q.lock().unwrap();
                me.sleeping.store(true, SeqCst);
                if me.pushed.load(SeqCst) == seen && !q.stop.load(SeqCst) {
                    c.parks.fetch_add(1, Relaxed);
                    drop(me.cv.wait_timeout(g, std::time::Duration::from_millis(100)).unwrap());
                } else {
                    drop(g);
                }
                me.sleeping.store(false, SeqCst);
                idle = 0;
                continue;
            }
        } else {
            idle = 0;
        }
        std::hint::spin_loop();
    }
}

pub const IDLE_SPINS: u32 = 20_000;

fn ring_cap(cfg: &IoConfig) -> usize {
    cfg.qd as usize + 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{FAILED, PENDING, READY};
    use crate::uring::{open_direct, Arena};
    use std::io::Write;

    fn scratch(name: &str, bytes: usize) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("oxf-io-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f.bin");
        let data: Vec<u8> = (0..bytes).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]).collect();
        std::fs::File::create(&p).unwrap().write_all(&data).unwrap();
        p
    }

    fn wait_state(s: &AtomicU8, want: u8, ms: u64) -> bool {
        let t = Instant::now();
        while t.elapsed().as_millis() < ms as u128 {
            if s.load(Acquire) == want {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        false
    }

    /// audit item: a short read is retried once, then the slot is FAILED and the error recorded; no abort
    #[test]
    fn short_read_fails_the_slot_not_the_process() {
        let p = scratch("short", 64 << 10);
        let Ok(fd) = open_direct(&p) else { return };
        let q = Arc::new(IoQueue::new());
        let h = spawn(q.clone(), IoConfig { qd: 8, prefetch_qd: 4, cpu: None });
        let arena = Arena::new(1 << 20, false).unwrap();
        let st = AtomicU8::new(PENDING);
        // 1 MiB from a 64 KiB file: short read
        q.push(Req { class: ReqClass::Demand, fd, buf: arena.ptr, len: 1 << 20, off: 0, state: &st });
        assert!(wait_state(&st, FAILED, 5000), "slot not FAILED");
        assert_eq!(q.counters.retried.load(Relaxed), 1);
        assert_eq!(q.counters.failed.load(Relaxed), 1);
        assert!(q.take_error().unwrap().contains("failed twice"));
        // the thread still serves good reads afterwards
        let ok = AtomicU8::new(PENDING);
        q.push(Req { class: ReqClass::Demand, fd, buf: arena.ptr, len: 64 << 10, off: 0, state: &ok });
        assert!(wait_state(&ok, READY, 5000), "good read after a failure");
        q.stop();
        h.join().unwrap();
        unsafe { libc_close(fd) };
    }

    /// FD-1: with tracing on every completed read leaves one stamp, push <= submit <= done; off records nothing
    /// (F-IO: also when the read is striped over several workers: still ONE stamp per read)
    #[test]
    fn trace_stamps_each_completed_read() {
        let p = scratch("stamps", 4 << 20);
        let Ok(fd) = open_direct(&p) else { return };
        for workers in [1usize, 3] {
            let q = Arc::new(IoQueue::new());
            let t = IoTuning { cpus: vec![usize::MAX; workers], split: workers as u32, ..IoTuning::default() };
            let h = spawn_tuned(q.clone(), IoConfig { qd: 8, prefetch_qd: 4, cpu: None }, t);
            let arena = Arena::new(4 << 20, false).unwrap();
            let a = AtomicU8::new(PENDING);
            q.push(Req { class: ReqClass::Demand, fd, buf: arena.ptr, len: 1 << 20, off: 0, state: &a });
            assert!(wait_state(&a, READY, 5000));
            assert!(q.take_stamps().is_empty(), "stamps recorded while tracing is off");
            q.set_trace(true);
            let sts: Vec<AtomicU8> = (0..3).map(|_| AtomicU8::new(PENDING)).collect();
            for (i, st) in sts.iter().enumerate() {
                let buf = unsafe { arena.ptr.add(i << 20) };
                q.push(Req { class: ReqClass::Demand, fd, buf, len: 1 << 20, off: (i as u64) << 20, state: st });
            }
            for st in &sts {
                assert!(wait_state(st, READY, 5000));
            }
            let v = q.take_stamps();
            assert_eq!(v.len(), 3, "{workers} workers");
            for s in &v {
                assert!(s.push_ns <= s.submit_ns && s.submit_ns <= s.done_ns, "{s:?}");
                assert_eq!(s.class, ReqClass::Demand as u8);
            }
            q.stop();
            h.join().unwrap();
        }
        unsafe { libc_close(fd) };
    }

    /// audit item: an idle IO thread parks (few wake-ups) and still answers a request promptly
    #[test]
    fn idle_thread_parks_and_wakes_on_push() {
        let p = scratch("park", 64 << 10);
        let Ok(fd) = open_direct(&p) else { return };
        let q = Arc::new(IoQueue::new());
        let h = spawn(q.clone(), IoConfig { qd: 8, prefetch_qd: 4, cpu: None });
        std::thread::sleep(std::time::Duration::from_millis(350));
        let parks = q.counters.parks.load(Relaxed);
        assert!((1..=10).contains(&parks), "parks in 350 ms idle: {parks} (100 ms timeout => ~3)");
        let arena = Arena::new(64 << 10, false).unwrap();
        // a push must wake the parked thread at once (a missed wake would wait for the 100 ms timeout); the median of
        // three idle -> push trials keeps this robust on a loaded machine (one sample at nice 19 / load 9 took > 50 ms)
        let mut ms = Vec::new();
        for _ in 0..3 {
            std::thread::sleep(std::time::Duration::from_millis(150)); // parked again
            let st = AtomicU8::new(PENDING);
            let t = Instant::now();
            q.push(Req { class: ReqClass::Demand, fd, buf: arena.ptr, len: 64 << 10, off: 0, state: &st });
            assert!(wait_state(&st, READY, 2000));
            ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(ms[1] < 80.0, "median wake + read {:.1} ms (samples {ms:?})", ms[1]);
        q.stop();
        h.join().unwrap();
        unsafe { libc_close(fd) };
    }

    fn pattern(off: usize, len: usize) -> Vec<u8> {
        (off..off + len).map(|i| (i as u32).wrapping_mul(2654435761).to_le_bytes()[3]).collect()
    }

    #[test]
    fn tuning_text_form() {
        assert_eq!(IoTuning::parse("").unwrap(), IoTuning::default());
        let t = IoTuning::parse("cpus=8,9,10;split=3;min=131072 align=65536;promote=1;spins=5").unwrap();
        assert_eq!((t.cpus.clone(), t.split, t.split_min, t.split_align, t.idle_spins), (vec![8, 9, 10], 3, 131072, 65536, 5));
        assert_eq!(IoTuning::parse("promote=0").unwrap(), IoTuning::default(), "promotion cannot be switched off");
        assert_eq!(t.workers(), 3);
        assert!(IoTuning::parse("cpus=1,2,3,4,5").is_err());
        assert!(IoTuning::parse("align=1000").is_err());
        assert!(IoTuning::parse("nope=1").is_err());
        assert!(IoTuning::parse("split").is_err());
    }

    #[test]
    fn stripes_cover_the_read_exactly() {
        for &(len, parts, align) in &[(1974272u32, 4u32, 4096u32), (2179072, 3, 65536), (2662400, 4, 262144), (1511424, 2, 4096), (4096, 4, 4096), (8192, 3, 4096)] {
            let v: Vec<(u32, u32)> = IoQueue::stripe(len, parts, align).collect();
            assert!(v.len() as u32 <= parts && !v.is_empty());
            let mut at = 0;
            for &(o, l) in &v {
                assert_eq!(o, at);
                assert!(l > 0 && o % align == 0 && l % 4096 == 0, "{len} {parts} {align}: {v:?}");
                at += l;
            }
            assert_eq!(at, len, "{len} {parts} {align}: {v:?}");
        }
    }

    /// F-IO: a read striped over 3 workers lands the same bytes as the file holds (all four slot sizes of the
    /// sidecar, at an unaligned-to-stripe offset), and counts as ONE completed request
    #[test]
    fn striped_read_is_byte_identical() {
        let p = scratch("stripe", 8 << 20);
        let Ok(fd) = open_direct(&p) else { return };
        for tuning in ["cpus=0,0,0;split=3;min=65536", "cpus=0,0,0,0;split=4;align=262144", "cpus=0,0;split=1", "cpus=0,0,0;split=3;sqpoll=;ffile=1", "cpus=0,0;split=2;defer=1"] {
            let q = Arc::new(IoQueue::new());
            let mut t = IoTuning::parse(tuning).unwrap();
            t.cpus = vec![usize::MAX; t.cpus.len()]; // unpinned workers in the test
            let h = spawn_tuned_unpinned(q.clone(), IoConfig { qd: 8, prefetch_qd: 4, cpu: None }, t);
            let arena = Arena::new(4 << 20, false).unwrap();
            let mut done = 0;
            for &len in &[1974272usize, 2179072, 1511424, 2662400, 4096] {
                let off = 4096 * 37;
                let st = AtomicU8::new(PENDING);
                unsafe { std::ptr::write_bytes(arena.ptr, 0xEE, len + 4096) };
                q.push(Req { class: ReqClass::Demand, fd, buf: arena.ptr, len: len as u32, off: off as u64, state: &st });
                assert!(wait_state(&st, READY, 5000), "{tuning}: read of {len} not READY");
                let got = unsafe { std::slice::from_raw_parts(arena.ptr, len + 4096) };
                assert!(got[..len] == pattern(off, len)[..], "{tuning}: bytes differ for len {len}");
                assert!(got[len..].iter().all(|&b| b == 0xEE), "{tuning}: wrote past the read");
                done += 1;
                assert_eq!(q.counters.completed[0].load(Relaxed), done);
            }
            assert!(q.counters.parts.load(Relaxed) >= done);
            q.stop();
            h.join().unwrap();
        }
        unsafe { libc_close(fd) };
    }

    /// F-IO: one failing part fails the whole request exactly once (FAILED, one error), the other parts' bytes
    /// do not flip it READY
    #[test]
    fn striped_short_read_fails_the_slot() {
        let p = scratch("stripeshort", 1 << 20);
        let Ok(fd) = open_direct(&p) else { return };
        let q = Arc::new(IoQueue::new());
        let t = IoTuning { cpus: vec![usize::MAX; 4], split: 4, ..IoTuning::default() };
        let h = spawn_tuned_unpinned(q.clone(), IoConfig { qd: 8, prefetch_qd: 4, cpu: None }, t);
        let arena = Arena::new(2 << 20, false).unwrap();
        let st = AtomicU8::new(PENDING);
        // 2 MiB from a 1 MiB file: parts 0-1 succeed, parts 2-3 read 0 bytes
        q.push(Req { class: ReqClass::Demand, fd, buf: arena.ptr, len: 2 << 20, off: 0, state: &st });
        assert!(wait_state(&st, FAILED, 5000), "slot not FAILED");
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(st.load(Acquire), FAILED);
        assert_eq!(q.counters.completed[0].load(Relaxed), 0);
        assert_eq!(q.counters.retried.load(Relaxed), 2);
        assert_eq!(q.counters.failed.load(Relaxed), 2);
        assert!(q.take_error().unwrap().contains("failed twice"));
        q.stop();
        h.join().unwrap();
        unsafe { libc_close(fd) };
    }

    /// F-IO: a queued prefetch is promoted to demand and overtakes the prefetch backlog (default tuning too)
    #[test]
    fn queued_prefetch_is_promoted() {
        let p = scratch("promote", 4 << 20);
        let Ok(fd) = open_direct(&p) else { return };
        let q0 = IoQueue::new();
        let dummy = AtomicU8::new(PENDING);
        assert_eq!(q0.promote(&dummy), Promote::InFlight, "nothing queued for this slot");
        let q = Arc::new(IoQueue::new());
        let t = IoTuning { cpus: vec![usize::MAX; 2], split: 2, ..IoTuning::default() };
        const N: usize = 200;
        let arena = Arena::new(N << 20, false).unwrap();
        let st: Vec<AtomicU8> = (0..N).map(|_| AtomicU8::new(PENDING)).collect();
        // queue the backlog BEFORE the workers start (push only needs the configured queue), one prefetch in flight at a time
        q.configure(&t);
        for i in 0..N {
            q.push(Req { class: ReqClass::Prefetch, fd, buf: unsafe { arena.ptr.add(i << 20) }, len: 1 << 20, off: ((i % 4) << 20) as u64, state: &st[i] });
        }
        assert_eq!(q.promote(&st[N - 1]), Promote::Moved);
        assert_eq!(q.promote(&st[N - 1]), Promote::InFlight, "already in the demand class: nothing left to move");
        let h = spawn_tuned_unpinned(q.clone(), IoConfig { qd: 8, prefetch_qd: 1, cpu: None }, t);
        assert!(wait_state(&st[N - 1], READY, 10000));
        let before = q.counters.completed[2].load(Relaxed);
        assert!(before < (N / 2) as u64, "the promoted read finished after {before} of {N} queued prefetches");
        assert_eq!(q.counters.promoted.load(Relaxed), 1, "an unsplit prefetch is one part");
        let got = unsafe { std::slice::from_raw_parts(arena.ptr.add((N - 1) << 20), 1 << 20) };
        assert!(got == &pattern(((N - 1) % 4) << 20, 1 << 20)[..]);
        for s in &st {
            assert!(wait_state(s, READY, 20000));
        }
        assert_eq!(q.counters.completed[2].load(Relaxed), N as u64);
        q.stop();
        h.join().unwrap();
        unsafe { libc_close(fd) };
    }

    /// as `spawn_tuned`, with cpu ids of usize::MAX meaning "do not pin" (tests run on whatever cores are free)
    fn spawn_tuned_unpinned(q: Arc<IoQueue>, cfg: IoConfig, t: IoTuning) -> std::thread::JoinHandle<()> {
        spawn_tuned(q, cfg, t)
    }

    extern "C" {
        #[link_name = "close"]
        fn libc_close(fd: i32) -> i32;
    }
}
