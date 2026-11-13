//! F-VAE1: threads spawned by the pool's creating thread inside `Pool::unpinned` get the full mask.
use super::*;

fn count(set: &libc::cpu_set_t) -> usize {
    // SAFETY: CPU_COUNT reads the set only.
    unsafe { libc::CPU_COUNT(set) as usize }
}

#[test]
fn unpinned_gives_spawned_threads_the_original_mask_and_repins() {
    // own thread: the test harness thread's mask must not leak into other tests
    std::thread::spawn(|| {
        let before = count(&thread_affinity(0).unwrap());
        if before < 2 {
            eprintln!("skip: a single allowed CPU");
            return;
        }
        let avail = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1);
        let cpus: Vec<usize> = (0..2).map(|i| i % avail).collect();
        let pool = Pool::new(&cpus, Duration::from_millis(1)).unwrap();
        assert_eq!(count(&thread_affinity(0).unwrap()), 1, "worker 0 is pinned");
        // the bug class: a thread spawned now inherits one CPU
        let pinned_child = std::thread::spawn(|| count(&thread_affinity(0).unwrap())).join().unwrap();
        assert_eq!(pinned_child, 1);
        let (inside, child) = pool.unpinned(|| {
            let inside = count(&thread_affinity(0).unwrap());
            let child = std::thread::scope(|s| s.spawn(|| count(&thread_affinity(0).unwrap())).join().unwrap());
            (inside, child)
        });
        assert_eq!(inside, before);
        assert_eq!(child, before, "a loader thread spawned inside unpinned() must get the full mask");
        assert_eq!(count(&thread_affinity(0).unwrap()), 1, "re-pinned to cpus[0] after unpinned()");
        // the pool still works
        let n = AtomicU32::new(0);
        pool.run(|_| {
            n.fetch_add(1, SeqCst);
        });
        assert_eq!(n.load(SeqCst), 2);
    })
    .join()
    .unwrap();
}
