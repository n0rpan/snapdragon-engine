//! Persistent pinned worker pool with a sense-reversing spin barrier.
//!
//! - `Pool::new(cpus)` creates `cpus.len() - 1` worker threads once, each pinned to one
//!   core with `sched_setaffinity`; the calling thread becomes worker 0 and is pinned to
//!   `cpus[0]`. No thread is created per op or per token.
//! - `Pool::run(job)` publishes `job` through a "go" epoch, runs it on every worker
//!   (including the caller) and ends with one full barrier (the join), which is counted.
//! - Idle workers spin on the epoch for `idle_spin` and then sleep on a futex, so an idle
//!   server costs nothing; inside a job there is no sleeping at all.
//! - `Barrier` is a centralised sense-reversing barrier on cache-line-padded atomics. The
//!   last arriver bumps an episode counter, so barriers/token is measured, not assumed.

use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::*};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Two cache lines: Oryon has 64-byte lines but adjacent-line prefetch pairs them.
#[repr(align(128))]
#[derive(Default)]
pub struct Padded<T>(pub T);

impl<T> std::ops::Deref for Padded<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

#[derive(Default)]
struct Arrival {
    arrived: AtomicU32,
    episodes: AtomicU64,
}

/// Centralised sense-reversing spin barrier for a fixed team size.
pub struct Barrier {
    n: u32,
    arrival: Padded<Arrival>,
    sense: Padded<AtomicU32>,
}

impl Barrier {
    pub fn new(n: usize) -> Barrier {
        assert!(n >= 1);
        Barrier { n: n as u32, arrival: Padded::default(), sense: Padded::default() }
    }

    pub fn team_size(&self) -> usize {
        self.n as usize
    }

    /// Wait until all `n` threads have arrived. `local_sense` is per-thread state that
    /// starts at 0 and must only be used with this barrier.
    #[inline]
    pub fn wait(&self, local_sense: &mut u32) {
        let s = *local_sense ^ 1;
        *local_sense = s;
        if self.n == 1 {
            self.arrival.episodes.fetch_add(1, Relaxed);
            return;
        }
        // AcqRel: release this thread's writes before the barrier; the last arriver
        // acquires everyone's and republishes them with the sense store below.
        if self.arrival.arrived.fetch_add(1, AcqRel) + 1 == self.n {
            self.arrival.arrived.store(0, Relaxed);
            self.arrival.episodes.fetch_add(1, Relaxed);
            self.sense.store(s, Release);
        } else {
            while self.sense.load(Acquire) != s {
                spin_loop();
            }
        }
    }

    /// Number of completed barrier episodes so far.
    pub fn episodes(&self) -> u64 {
        self.arrival.episodes.load(Relaxed)
    }
}

/// Per-thread handle passed to jobs.
pub struct Ctx<'a> {
    pub tid: usize,
    pub nthreads: usize,
    barrier: &'a Barrier,
    sense: &'a mut u32,
}

impl Ctx<'_> {
    #[inline]
    pub fn barrier(&mut self) {
        self.barrier.wait(self.sense);
    }
}

type JobPtr = *const (dyn Fn(&mut Ctx) + Sync);

struct Shared {
    n: usize,
    epoch: Padded<AtomicU32>,
    sleepers: Padded<AtomicU32>,
    shutdown: AtomicBool,
    idle_spin: Duration,
    job: UnsafeCell<Option<JobPtr>>,
    barrier: Barrier,
    /// Per-thread barrier sense (each slot touched only by its owner).
    senses: Vec<Padded<UnsafeCell<u32>>>,
}

// SAFETY: `job` is written only by the owner thread while every worker is parked between
// epochs (after the join barrier of the previous run), and read by workers only after
// they observe the new epoch (Acquire), which orders the write before the read. Each
// `senses[i]` is accessed only by thread i.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    handles: Vec<JoinHandle<()>>,
    cpus: Vec<usize>,
    tids: Arc<Vec<AtomicU32>>,
}

/// Pin the calling thread to one CPU.
pub fn pin_current_thread(cpu: usize) -> std::io::Result<()> {
    // SAFETY: cpu_set_t is plain data; CPU_SET writes within it; pid 0 = calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

pub fn gettid() -> u32 {
    // SAFETY: trivial syscall.
    unsafe { libc::syscall(libc::SYS_gettid) as u32 }
}

fn futex_wait(addr: &AtomicU32, expected: u32) {
    // SAFETY: FUTEX_WAIT on a valid aligned u32; returns immediately if the value changed.
    unsafe {
        libc::syscall(
            libc::SYS_futex,
            addr.as_ptr(),
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            std::ptr::null::<libc::timespec>(),
        );
    }
}

fn futex_wake_all(addr: &AtomicU32) {
    // SAFETY: FUTEX_WAKE on a valid aligned u32.
    unsafe {
        libc::syscall(libc::SYS_futex, addr.as_ptr(), libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG, i32::MAX);
    }
}

/// Parse "0-7", "0,1,4,5", "0-3,8-11" into a CPU list.
pub fn parse_cpu_list(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let (a, b): (usize, usize) = (a.parse().map_err(|_| format!("bad cpu {a}"))?, b.parse().map_err(|_| format!("bad cpu {b}"))?);
            if b < a {
                return Err(format!("bad range {part}"));
            }
            out.extend(a..=b);
        } else {
            out.push(part.parse().map_err(|_| format!("bad cpu {part}"))?);
        }
    }
    if out.is_empty() {
        return Err("empty cpu list".into());
    }
    let mut seen = out.clone();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != out.len() {
        return Err("duplicate cpu in list".into());
    }
    Ok(out)
}

impl Pool {
    /// `cpus[i]` is the core of worker i; `None` entries are not supported (always pinned).
    pub fn new(cpus: &[usize], idle_spin: Duration) -> std::io::Result<Pool> {
        let n = cpus.len();
        assert!(n >= 1);
        let shared = Arc::new(Shared {
            n,
            epoch: Padded::default(),
            sleepers: Padded::default(),
            shutdown: AtomicBool::new(false),
            idle_spin,
            job: UnsafeCell::new(None),
            barrier: Barrier::new(n),
            senses: (0..n).map(|_| Padded(UnsafeCell::new(0))).collect(),
        });
        let tids: Arc<Vec<AtomicU32>> = Arc::new((0..n).map(|_| AtomicU32::new(0)).collect());
        pin_current_thread(cpus[0])?;
        tids[0].store(gettid(), Relaxed);
        let mut handles = Vec::new();
        for (i, &cpu) in cpus.iter().enumerate().skip(1) {
            let sh = shared.clone();
            let tids = tids.clone();
            let h = std::thread::Builder::new().name(format!("oxe-w{i}")).spawn(move || {
                if let Err(e) = pin_current_thread(cpu) {
                    eprintln!("oxe-w{i}: pin to cpu {cpu} failed: {e}");
                }
                tids[i].store(gettid(), Release);
                worker_loop(&sh, i);
            })?;
            handles.push(h);
        }
        // wait until every worker has pinned itself
        while tids.iter().any(|t| t.load(Acquire) == 0) {
            std::thread::yield_now();
        }
        Ok(Pool { shared, handles, cpus: cpus.to_vec(), tids })
    }

    pub fn nthreads(&self) -> usize {
        self.shared.n
    }
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }
    /// Kernel thread ids of the workers (index = worker id).
    pub fn thread_ids(&self) -> Vec<u32> {
        self.tids.iter().map(|t| t.load(Relaxed)).collect()
    }
    /// Barrier episodes completed so far (including the join barrier of every run).
    pub fn barrier_episodes(&self) -> u64 {
        self.shared.barrier.episodes()
    }

    /// Run `job` on all workers; returns after the join barrier.
    pub fn run<F: Fn(&mut Ctx) + Sync>(&self, job: F) {
        let sh = &*self.shared;
        let ptr: *const (dyn Fn(&mut Ctx) + Sync + '_) = &job;
        // SAFETY: the job outlives this call and every worker finishes with it before the
        // join barrier below returns, so erasing the lifetime is sound.
        let ptr: JobPtr = unsafe { std::mem::transmute(ptr) };
        // SAFETY: workers are parked (previous run ended with the join barrier).
        unsafe { *sh.job.get() = Some(ptr) };
        let e = sh.epoch.fetch_add(1, SeqCst).wrapping_add(1);
        if sh.sleepers.load(SeqCst) > 0 {
            futex_wake_all(&sh.epoch);
        }
        let _ = e;
        // SAFETY: slot 0 belongs to the calling thread.
        run_job(sh, 0, unsafe { &mut *sh.senses[0].get() });
    }
}

fn run_job(sh: &Shared, tid: usize, sense: &mut u32) {
    // SAFETY: published before the epoch bump that this thread observed (or by itself).
    let job = unsafe { (*sh.job.get()).expect("job published") };
    let mut ctx = Ctx { tid, nthreads: sh.n, barrier: &sh.barrier, sense };
    // SAFETY: the job pointer is valid until the join barrier completes (see Pool::run).
    unsafe { (*job)(&mut ctx) };
    ctx.barrier(); // join
}

fn worker_loop(sh: &Shared, tid: usize) {
    let mut seen = sh.epoch.load(Acquire);
    // SAFETY: slot tid belongs to this thread only.
    let sense = unsafe { &mut *sh.senses[tid].get() };
    loop {
        // spin, then sleep
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            let e = sh.epoch.load(Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if sh.shutdown.load(Relaxed) {
                return;
            }
            spin_loop();
            spins = spins.wrapping_add(1);
            if spins % 1024 == 0 && start.elapsed() >= sh.idle_spin {
                sh.sleepers.fetch_add(1, SeqCst);
                if sh.epoch.load(SeqCst) == seen && !sh.shutdown.load(SeqCst) {
                    futex_wait(&sh.epoch, seen);
                }
                sh.sleepers.fetch_sub(1, SeqCst);
            }
        }
        if sh.shutdown.load(Acquire) {
            return;
        }
        run_job(sh, tid, sense);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, SeqCst);
        self.shared.epoch.fetch_add(1, SeqCst);
        futex_wake_all(&self.shared.epoch);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// What the engine needs from a thread team: run one job on every worker (with a join).
pub trait Team {
    fn nthreads(&self) -> usize;
    fn run_job(&self, job: &(dyn Fn(&mut Ctx) + Sync));
    fn barrier_episodes(&self) -> u64;
}

impl Team for Pool {
    fn nthreads(&self) -> usize {
        Pool::nthreads(self)
    }
    fn run_job(&self, job: &(dyn Fn(&mut Ctx) + Sync)) {
        self.run(|c| job(c))
    }
    fn barrier_episodes(&self) -> u64 {
        Pool::barrier_episodes(self)
    }
}

/// A pool whose worker 0 is a dedicated leader thread (pinned to `cpus[0]`), so jobs can be
/// submitted from ANY thread (e.g. a server's blocking-task pool) without pinning or
/// borrowing the caller. The caller hands over a job pointer and blocks (bounded spin, then
/// a condvar) until the leader has run it on the whole team.
pub struct DetachedPool {
    inner: Arc<Detached>,
    leader: Option<JoinHandle<()>>,
    n: usize,
    cpus: Vec<usize>,
}

struct Detached {
    /// 0 idle, 1 job posted, 2 done, 3 shutdown
    state: AtomicU32,
    job: UnsafeCell<Option<JobPtr>>,
    episodes: AtomicU64,
    lock: std::sync::Mutex<()>,
    cv: std::sync::Condvar,
    spin: Duration,
    /// serialises submitters
    submit: std::sync::Mutex<()>,
}

// SAFETY: `job` is written by the submitting thread only while state == 0 and read by the
// leader only after it observes state == 1 (Acquire), and the submitter waits for state == 2
// before returning, so the pointee outlives every use.
unsafe impl Sync for Detached {}
unsafe impl Send for Detached {}

impl DetachedPool {
    pub fn new(cpus: &[usize], idle_spin: Duration) -> std::io::Result<DetachedPool> {
        let d = Arc::new(Detached {
            state: AtomicU32::new(0),
            job: UnsafeCell::new(None),
            episodes: AtomicU64::new(0),
            lock: std::sync::Mutex::new(()),
            cv: std::sync::Condvar::new(),
            spin: idle_spin,
            submit: std::sync::Mutex::new(()),
        });
        let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<()>>();
        let dd = d.clone();
        let cs = cpus.to_vec();
        let leader = std::thread::Builder::new().name("oxe-leader".into()).spawn(move || {
            let pool = match Pool::new(&cs, dd.spin) {
                Ok(p) => {
                    let _ = tx.send(Ok(()));
                    p
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                    return;
                }
            };
            loop {
                // wait for a job: spin, then sleep on the condvar
                let start = Instant::now();
                let mut spins = 0u32;
                loop {
                    let s = dd.state.load(Acquire);
                    if s == 1 || s == 3 {
                        break;
                    }
                    spin_loop();
                    spins = spins.wrapping_add(1);
                    if spins % 1024 == 0 && start.elapsed() >= dd.spin {
                        let g = dd.lock.lock().unwrap();
                        let s = dd.state.load(Acquire);
                        if s != 1 && s != 3 {
                            let _g = dd.cv.wait_timeout(g, Duration::from_millis(50)).unwrap();
                        }
                    }
                }
                if dd.state.load(Acquire) == 3 {
                    return;
                }
                // SAFETY: published before state = 1 (see Detached).
                let job = unsafe { (*dd.job.get()).expect("job") };
                struct J(JobPtr);
                // SAFETY: the pointee is Sync (dyn Fn + Sync) and outlives the run.
                unsafe impl Sync for J {}
                impl J {
                    fn call(&self, c: &mut Ctx) {
                        // SAFETY: the submitter keeps the job alive until state == 2.
                        unsafe { (*self.0)(c) }
                    }
                }
                let j = J(job);
                pool.run(|c| j.call(c));
                dd.episodes.store(pool.barrier_episodes(), Relaxed);
                {
                    let _g = dd.lock.lock().unwrap();
                    dd.state.store(2, Release);
                }
                dd.cv.notify_all();
            }
        })?;
        rx.recv().map_err(|_| std::io::Error::other("leader thread died"))??;
        Ok(DetachedPool { inner: d, leader: Some(leader), n: cpus.len(), cpus: cpus.to_vec() })
    }
    pub fn cpus(&self) -> &[usize] {
        &self.cpus
    }
}

impl Team for DetachedPool {
    fn nthreads(&self) -> usize {
        self.n
    }
    fn run_job(&self, job: &(dyn Fn(&mut Ctx) + Sync)) {
        let d = &*self.inner;
        let ptr: *const (dyn Fn(&mut Ctx) + Sync + '_) = job;
        // SAFETY: we block below until the leader reports completion (state 2), so the job
        // outlives every use; erasing the lifetime is sound.
        let ptr: JobPtr = unsafe { std::mem::transmute(ptr) };
        let _turn = d.submit.lock().unwrap();
        // SAFETY: state == 0: the leader does not read `job` now.
        unsafe { *d.job.get() = Some(ptr) };
        {
            let _g = d.lock.lock().unwrap();
            d.state.store(1, Release);
        }
        d.cv.notify_all();
        let start = Instant::now();
        let mut spins = 0u32;
        while d.state.load(Acquire) != 2 {
            spin_loop();
            spins = spins.wrapping_add(1);
            if spins % 1024 == 0 && start.elapsed() >= d.spin {
                let g = d.lock.lock().unwrap();
                if d.state.load(Acquire) != 2 {
                    let _g = d.cv.wait_timeout(g, Duration::from_millis(50)).unwrap();
                }
            }
        }
        d.state.store(0, Release);
    }
    fn barrier_episodes(&self) -> u64 {
        self.inner.episodes.load(Relaxed)
    }
}

impl Drop for DetachedPool {
    fn drop(&mut self) {
        {
            let _g = self.inner.lock.lock().unwrap();
            self.inner.state.store(3, Release);
        }
        self.inner.cv.notify_all();
        if let Some(h) = self.leader.take() {
            let _ = h.join();
        }
    }
}

/// Longest-processing-time-first assignment of `costs` to `bins` bins.
/// Returns, per item, its bin; deterministic (ties broken by index).
pub fn lpt_assign(costs: &[u64], bins: usize, out: &mut [u8], loads: &mut [u64]) {
    debug_assert!(bins <= loads.len() && costs.len() <= out.len() && bins <= 255);
    let mut order: [u16; 64] = [0; 64];
    let n = costs.len();
    assert!(n <= 64);
    for (i, o) in order.iter_mut().enumerate().take(n) {
        *o = i as u16;
    }
    // insertion sort, descending cost (n <= 64, allocation-free)
    for i in 1..n {
        let mut j = i;
        while j > 0 && costs[order[j - 1] as usize] < costs[order[j] as usize] {
            order.swap(j - 1, j);
            j -= 1;
        }
    }
    loads[..bins].iter_mut().for_each(|l| *l = 0);
    for &i in &order[..n] {
        let i = i as usize;
        let mut best = 0;
        for b in 1..bins {
            if loads[b] < loads[best] {
                best = b;
            }
        }
        out[i] = best as u8;
        loads[best] += costs[i];
    }
}

#[cfg(test)]
mod tests;
