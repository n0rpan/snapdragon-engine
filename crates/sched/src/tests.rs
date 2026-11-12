use super::*;
use std::sync::atomic::AtomicUsize;

fn test_cpus(n: usize) -> Vec<usize> {
    let avail = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
    (0..n).map(|i| i % avail).collect::<Vec<_>>()
}

/// Barrier correctness under contention: every thread writes its slot for round r, waits,
/// checks all slots hold r, waits again. 8 threads, a short burst.
#[test]
fn barrier_orders_all_writes_8_threads() {
    let n = 8;
    let rounds = 20_000u64;
    let b = Arc::new(Barrier::new(n));
    let slots: Arc<Vec<Padded<AtomicU64>>> = Arc::new((0..n).map(|_| Padded(AtomicU64::new(u64::MAX))).collect());
    let errors = Arc::new(AtomicUsize::new(0));
    let hs: Vec<_> = (0..n)
        .map(|t| {
            let (b, slots, errors) = (b.clone(), slots.clone(), errors.clone());
            std::thread::spawn(move || {
                let mut sense = 0;
                for r in 0..rounds {
                    slots[t].store(r, Relaxed); // Relaxed on purpose: the barrier must order it
                    b.wait(&mut sense);
                    for s in slots.iter() {
                        if s.load(Relaxed) != r {
                            errors.fetch_add(1, Relaxed);
                        }
                    }
                    b.wait(&mut sense);
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    assert_eq!(errors.load(Relaxed), 0);
    assert_eq!(b.episodes(), 2 * rounds);
}

#[test]
fn barrier_single_thread_counts() {
    let b = Barrier::new(1);
    let mut s = 0;
    for _ in 0..5 {
        b.wait(&mut s);
    }
    assert_eq!(b.episodes(), 5);
}

/// The pool runs every job on every worker exactly once, barriers inside jobs order
/// writes, and the join barrier is counted once per run.
#[test]
fn pool_runs_jobs_and_counts_barriers() {
    let n = 8;
    let pool = Pool::new(&test_cpus(n), Duration::from_micros(200)).unwrap();
    let data: Vec<Padded<AtomicU64>> = (0..n).map(|_| Padded(AtomicU64::new(0))).collect();
    let bad = AtomicUsize::new(0);
    let runs = 200u64;
    let before = pool.barrier_episodes();
    for r in 1..=runs {
        pool.run(|ctx| {
            data[ctx.tid].store(r, Relaxed);
            ctx.barrier();
            let sum: u64 = data.iter().map(|d| d.load(Relaxed)).sum();
            if sum != r * n as u64 {
                bad.fetch_add(1, Relaxed);
            }
            ctx.barrier();
        });
    }
    assert_eq!(bad.load(Relaxed), 0);
    assert_eq!(pool.barrier_episodes() - before, runs * 3);
    // idle long enough for the workers to fall asleep on the futex, then wake them
    std::thread::sleep(Duration::from_millis(5));
    let count = AtomicUsize::new(0);
    pool.run(|_| {
        count.fetch_add(1, Relaxed);
    });
    assert_eq!(count.load(Relaxed), n);
}

/// Affinity: each worker's /proc status shows exactly its assigned CPU.
#[test]
fn workers_are_pinned() {
    let cpus = test_cpus(2.min(std::thread::available_parallelism().unwrap().get()));
    let cpus: Vec<usize> = if cpus.len() == 2 && cpus[0] == cpus[1] { vec![0] } else { cpus };
    let pool = Pool::new(&cpus, Duration::from_micros(50)).unwrap();
    for (i, tid) in pool.thread_ids().iter().enumerate() {
        let status = std::fs::read_to_string(format!("/proc/self/task/{tid}/status")).unwrap();
        let line = status.lines().find(|l| l.starts_with("Cpus_allowed_list:")).unwrap();
        assert_eq!(line.split_whitespace().nth(1).unwrap(), cpus[i].to_string(), "worker {i}");
    }
}

#[test]
fn cpu_list_parsing() {
    assert_eq!(parse_cpu_list("0-3").unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(parse_cpu_list("0,1,4,5").unwrap(), vec![0, 1, 4, 5]);
    assert_eq!(parse_cpu_list("0-1,8-9").unwrap(), vec![0, 1, 8, 9]);
    assert!(parse_cpu_list("3-1").is_err());
    assert!(parse_cpu_list("1,1").is_err());
    assert!(parse_cpu_list("").is_err());
}

#[test]
fn lpt_balances_moe_units() {
    // 8 routed experts x 2 halves (Q3_K-sized) + shared x 2 (Q6_K-sized), 8 threads
    let mut costs = vec![675_840u64; 16];
    costs.extend([1_290_240, 1_290_240]);
    let mut out = [0u8; 64];
    let mut loads = [0u64; 8];
    lpt_assign(&costs, 8, &mut out, &mut loads);
    let max = *loads.iter().max().unwrap();
    let min = *loads.iter().min().unwrap();
    assert_eq!(loads.iter().sum::<u64>(), costs.iter().sum::<u64>());
    // the two big units go to different threads first
    assert_ne!(out[16], out[17]);
    assert!(max <= 2 * 1_290_240, "max load {max}");
    assert!(min > 0);
}

#[test]
fn detached_pool_runs_jobs_from_any_thread() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let cpus = test_cpus(3);
    let pool = std::sync::Arc::new(DetachedPool::new(&cpus, Duration::from_micros(50)).unwrap());
    let hits = std::sync::Arc::new(AtomicUsize::new(0));
    let mut hs = Vec::new();
    for _ in 0..4 {
        let (p, h) = (pool.clone(), hits.clone());
        hs.push(std::thread::spawn(move || {
            for _ in 0..25 {
                let seen = AtomicUsize::new(0);
                p.run_job(&|c: &mut Ctx| {
                    seen.fetch_add(1, Ordering::SeqCst);
                    c.barrier();
                    assert_eq!(seen.load(Ordering::SeqCst), c.nthreads);
                });
                assert_eq!(seen.load(Ordering::SeqCst), 3);
                h.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }
    for h in hs {
        h.join().unwrap();
    }
    assert_eq!(hits.load(Ordering::SeqCst), 100);
    // 2 barriers per job (one inside + the join)
    assert_eq!(pool.barrier_episodes(), 200);
}

// ---------------------------------------------------------------- REVIEW-v0.1.1 S6 / F15

/// Two threads with `&Pool` call `run` at the same time. Before the busy flag both wrote the
/// job pointer and `senses[0]` (undefined behaviour, typically a hang). Now the second call
/// panics before touching anything and the first one completes normally.
#[test]
fn concurrent_run_is_refused() {
    let pool = Arc::new(Pool::new(&test_cpus(3), Duration::from_millis(1)).unwrap());
    let inside = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let first = {
        let (pool, inside, release) = (pool.clone(), inside.clone(), release.clone());
        std::thread::spawn(move || {
            pool.run(|c| {
                if c.tid == 0 {
                    inside.store(true, SeqCst);
                    let t0 = Instant::now();
                    while !release.load(SeqCst) && t0.elapsed() < Duration::from_secs(10) {
                        spin_loop();
                    }
                }
            })
        })
    };
    while !inside.load(SeqCst) {
        spin_loop();
    }
    let (tx, rx) = std::sync::mpsc::channel();
    {
        let pool = pool.clone();
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.run(|_| {})));
            let _ = tx.send(r.is_err());
        });
    }
    let refused = rx.recv_timeout(Duration::from_secs(5));
    release.store(true, SeqCst);
    assert_eq!(refused, Ok(true), "a second concurrent run must panic up front");
    first.join().unwrap();
    // the pool is still usable
    let n = AtomicUsize::new(0);
    pool.run(|_| {
        n.fetch_add(1, SeqCst);
    });
    assert_eq!(n.load(SeqCst), 3);
}

/// Child-process half of the abort tests: `OXE_SCHED_PANIC_CHILD=caller|worker` runs a job that
/// panics on that thread. The parent expects SIGABRT, not an unwind (F15: the caller's unwind
/// freed the closure under the workers) or a hang (a dead worker left the barrier one short).
#[test]
fn panic_child() {
    let Ok(who) = std::env::var("OXE_SCHED_PANIC_CHILD") else { return };
    // This child aborts on purpose: keep the SIGABRT but never write a core dump
    // (a dump per CI run shows up as a "process crashed" notification on dev machines).
    let no_core = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    unsafe { libc::setrlimit(libc::RLIMIT_CORE, &no_core) };
    let pool = Pool::new(&test_cpus(3), Duration::from_millis(1)).unwrap();
    let victim = if who == "caller" { 0 } else { 1 };
    pool.run(|c| {
        if c.tid == victim {
            panic!("synthetic job failure on tid {victim}");
        }
        std::thread::sleep(Duration::from_millis(20));
    });
    // only reached if the panic did not abort: exit differently from SIGABRT
    std::process::exit(3);
}

fn run_panic_child(who: &str) -> std::process::ExitStatus {
    use std::os::unix::process::ExitStatusExt;
    let exe = std::env::current_exe().unwrap();
    let mut child = std::process::Command::new(exe)
        .args(["--exact", "tests::panic_child", "--nocapture", "--test-threads", "1"])
        .env("OXE_SCHED_PANIC_CHILD", who)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    loop {
        if let Some(s) = child.try_wait().unwrap() {
            return s;
        }
        if t0.elapsed() > Duration::from_secs(20) {
            let _ = child.kill();
            let s = child.wait().unwrap();
            panic!("child ({who} panic) hung: {s:?} (signal {:?})", s.signal());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn caller_job_panic_aborts() {
    use std::os::unix::process::ExitStatusExt;
    let s = run_panic_child("caller");
    assert_eq!(s.signal(), Some(libc::SIGABRT), "caller-side job panic must abort, got {s:?}");
}

#[test]
fn worker_job_panic_aborts() {
    use std::os::unix::process::ExitStatusExt;
    let s = run_panic_child("worker");
    assert_eq!(s.signal(), Some(libc::SIGABRT), "worker-side job panic must abort, got {s:?}");
}

/// Dropping a pool gives the creating thread its CPU mask back, so threads it spawns later are
/// not all stuck on `cpus[0]` (the oxi VAE ran its 12 threads on one core after the fast DiT).
#[test]
fn drop_restores_caller_affinity() {
    std::thread::spawn(|| {
        let before = thread_affinity(0).unwrap();
        let n_before = unsafe { libc::CPU_COUNT(&before) } as usize;
        let cpus = test_cpus(2.min(n_before.max(1)));
        let pool = Pool::new(&cpus, Duration::from_micros(50)).unwrap();
        assert_eq!(allowed_cpus(), 1, "Pool::new pins the caller");
        // a thread spawned now inherits the single-CPU mask (the bug)
        let inherited = std::thread::spawn(allowed_cpus).join().unwrap();
        assert_eq!(inherited, 1);
        drop(pool);
        let after = thread_affinity(0).unwrap();
        assert!(unsafe { libc::CPU_EQUAL(&before, &after) }, "mask restored on drop");
        let spawned = std::thread::spawn(allowed_cpus).join().unwrap();
        assert_eq!(spawned, n_before);
    })
    .join()
    .unwrap();
}
