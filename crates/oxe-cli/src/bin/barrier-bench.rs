//! barrier-bench: ns per empty barrier on the persistent pinned pool (SPEC B.8 / B.12 #2).
//! One JSON line per configuration. Short: 200k barriers per configuration by default.

use std::time::{Duration, Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut threads = vec![4usize, 6, 8, 12];
    let mut reps = 200_000usize;
    let mut cpu_list: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().expect("--threads 4,6,8").split(',').map(|s| s.parse().expect("number")).collect(),
            "--reps" => reps = args.next().expect("--reps N").parse().expect("number"),
            "--cpu-list" => cpu_list = Some(args.next().expect("--cpu-list LIST")),
            _ => {
                eprintln!("usage: barrier-bench [--threads 4,6,8,12] [--reps 200000] [--cpu-list LIST (overrides --threads)]");
                std::process::exit(2);
            }
        }
    }
    let configs: Vec<Vec<usize>> = match cpu_list {
        Some(s) => vec![sched::parse_cpu_list(&s).expect("cpu list")],
        None => threads.iter().map(|&n| (0..n).collect()).collect(),
    };
    for cpus in configs {
        let pool = sched::Pool::new(&cpus, Duration::from_millis(1)).expect("pool");
        pool.run(|ctx| {
            for _ in 0..1000 {
                ctx.barrier();
            }
        });
        let t = Instant::now();
        pool.run(|ctx| {
            for _ in 0..reps {
                ctx.barrier();
            }
        });
        let ns = t.elapsed().as_nanos() as f64 / (reps + 1) as f64;
        println!(
            "{{\"tool\":\"barrier-bench\",\"threads\":{},\"cpu_list\":{:?},\"reps\":{},\"ns_per_barrier\":{:.1}}}",
            cpus.len(),
            cpus,
            reps,
            ns
        );
    }
}
