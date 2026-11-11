//! OD-6a / OD-6b kernel micro-benchmark (model-free): what one more verify lane costs in the weight
//! GEMVs of a decode token, at production shapes and with the weights of a whole token streamed
//! from memory (cold, far larger than the caches), on 8 threads split the way the engine splits rows.
//!
//! Per case, for 1-4 lanes ("columns"):
//!   base = today's verify path: every lane runs the one-column GEMV on a ~24 KiB chunk of rows
//!          while it is in L1 (`Engine::mv_lanes`, head: 256-row chunks);
//!   vl   = the lane GEMV (`gemv_lanes_raw`, csrc/oxe_vl.inc): rows read and unpacked once.
//! Cases (Ornith-1.5 35B Compact unless marked; bytes are per token):
//!   head     Q6_K 248320 x 2048 (417 MB)
//!   dense    Q4_K, 8192-row matrices x 2048 (849 MB: qkv, gate, out, q, k, v of 40 layers)
//!   shexp    Q6_K 512-row matrices x 2048 (103 MB: shared expert gate / up / down)
//!   experts  Q3_K 512 x 2048 units picked from a 0.8 GB pool, 720 units per token (8 experts x
//!            gate, up, down-equivalent x 30 layers); lanes share ALL experts here, i.e. this is the
//!            cost of an expert both lanes routed to. An expert only one lane uses costs a full
//!            one-lane read whatever the kernel.
//!   qdense   Q8_0 8192-row matrices x 2048 (Qwen3.6: 1397 MB of Q8_0 dense per token)
//! Not covered (per lane, no shared weights to save): attention over the KV, the delta-net
//! recurrence, the f32 router, norms, activation quantisation.
//!
//! Run explicitly, on a quiet machine (all build locks held), pinned to the decode cores:
//! `taskset -c 0-7 cargo test --release -p kernels-sys --test vl_kbench -- --ignored --nocapture`
//! Optional: OXE_KBENCH_REPS (default 5), OXE_KBENCH_THREADS (default 8), OXE_KBENCH_CASES
//! (default "head,dense,shexp,experts,qdense").

use kernels_sys::{gemv_lanes_raw, gemv_raw, quantize_act, repack, QType};
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn f(&mut self) -> f32 {
        (self.next() % 20001) as f32 / 10000.0 - 1.0
    }
}

#[derive(Clone, Copy)]
struct P(*const u8);
unsafe impl Send for P {}
unsafe impl Sync for P {}
impl P {
    // a method call captures the whole wrapper (Sync), not the raw-pointer field
    fn p(&self) -> *const u8 {
        self.0
    }
}

/// `rows` packed rows of `k` weights of type `ty`: a small random matrix, repacked, then tiled.
fn weights(ty: QType, rows: usize, k: usize, r: &mut Lcg) -> Vec<u8> {
    let src_rows = 64usize;
    let bb = ty.block_bytes();
    let mut src = vec![0u8; src_rows * ty.row_bytes(k)];
    for b in src.iter_mut() {
        *b = r.next() as u8;
    }
    let d = 0x2000u16.to_le_bytes(); // a small finite f16
    for blk in src.chunks_mut(bb) {
        match ty {
            QType::Q4K | QType::Q5K => {
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&d);
            }
            QType::Q3K => blk[108..110].copy_from_slice(&d),
            QType::Q6K => blk[208..210].copy_from_slice(&d),
            QType::Q8_0 => blk[0..2].copy_from_slice(&d),
        }
    }
    let mut packed = vec![0u8; src.len()];
    repack(ty, &src, &mut packed, src_rows, k);
    let mut out = vec![0u8; rows * ty.row_bytes(k)];
    for c in out.chunks_mut(packed.len()) {
        c.copy_from_slice(&packed[..c.len()]);
    }
    out
}

fn acts(ty: QType, k: usize, n: usize, r: &mut Lcg) -> Vec<Vec<u8>> {
    (0..n)
        .map(|_| {
            let x: Vec<f32> = (0..k).map(|_| r.f()).collect();
            let mut a = vec![0u8; ty.act().bytes(k)];
            quantize_act(ty.act(), &x, &mut a);
            a
        })
        .collect()
}

/// Wall seconds per rep: `nt` threads run `f(tid, rep)` `reps` times; the slowest thread counts.
fn timed(nt: usize, reps: usize, f: &(dyn Fn(usize, usize) + Sync)) -> f64 {
    let mut worst = 0f64;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt)
            .map(|t| {
                s.spawn(move || {
                    let t0 = Instant::now();
                    for rep in 0..reps {
                        f(t, rep);
                    }
                    t0.elapsed().as_secs_f64()
                })
            })
            .collect();
        for h in hs {
            worst = worst.max(h.join().unwrap());
        }
    });
    worst / reps as f64
}

struct Case {
    name: &'static str,
    ty: QType,
    k: usize,
    /// rows of one matrix (split evenly over the threads) and matrices walked per token
    mat_rows: usize,
    mats: usize,
    /// matrices in the pool (>= mats; a bigger pool is sampled, as routed experts are)
    pool: usize,
    /// rows per base-path chunk
    chunk: usize,
}

/// Rows per lane-interleaved GEMV chunk of the engine (`lane_chunk`): about 24 KiB of weights.
fn lane_chunk(ty: QType, k: usize) -> usize {
    ((24 * 1024 / ty.row_bytes(k)) / 8 * 8).max(8)
}

fn run_case(c: &Case, nt: usize, reps: usize) -> Vec<(usize, f64, f64)> {
    let mut rng = Lcg(0x5eed + c.k as u64);
    let w = weights(c.ty, c.mat_rows * c.pool, c.k, &mut rng);
    let a = acts(c.ty, c.k, 4, &mut rng);
    let mrb = c.mat_rows * c.ty.row_bytes(c.k);
    let wp = P(w.as_ptr());
    let ap: Vec<P> = a.iter().map(|v| P(v.as_ptr())).collect();
    let per = (c.mat_rows / nt / 8 * 8).max(8);
    let mut res = Vec::new();
    let mut sink = 0f32;
    for nl in 1..=4usize {
        let mut t = [0f64; 2];
        for (mode, tm) in t.iter_mut().enumerate() {
            if nl == 1 && mode == 1 {
                continue;
            }
            let ap = &ap;
            let f = move |tid: usize, rep: usize| {
                let mut out = vec![0f32; 4 * per];
                let r0 = tid * per;
                for m in 0..c.mats {
                    // pool sampling: a different set of matrices every rep (cold), same for every thread
                    let mi = if c.pool == c.mats { m } else { (m * 2654435761usize + rep * 40503 + 17) % c.pool };
                    let base = unsafe { wp.p().add(mi * mrb) };
                    let outs: [*mut f32; 4] = std::array::from_fn(|l| unsafe { out.as_mut_ptr().add(l * per) });
                    let av: [*const u8; 4] = std::array::from_fn(|l| ap[l].p());
                    if mode == 1 {
                        // SAFETY: rows r0..r0+per of matrix mi; per floats per lane.
                        unsafe {
                            assert!(gemv_lanes_raw(c.ty, c.k, &outs[..nl], base.add(r0 * c.ty.row_bytes(c.k)), &av[..nl], per));
                        }
                    } else {
                        let mut r = 0;
                        while r < per {
                            let n = c.chunk.min(per - r);
                            for l in 0..nl {
                                // SAFETY: as above, rows r0+r..r0+r+n.
                                unsafe { gemv_raw(c.ty, c.k, outs[l].add(r), base.add((r0 + r) * c.ty.row_bytes(c.k)), av[l], n) };
                            }
                            r += n;
                        }
                    }
                }
                std::hint::black_box(&out);
            };
            f(0, 0); // page in
            *tm = (0..3).map(|_| timed(nt, reps, &f)).fold(f64::MAX, f64::min);
        }
        sink += t[0] as f32;
        res.push((nl, t[0], t[1]));
    }
    std::hint::black_box(sink);
    let bytes = (per * nt * c.mats * c.ty.row_bytes(c.k)) as f64;
    let t1 = res[0].1;
    println!(
        "{:<8} {:?} {} x {} x {} mats ({:.0} MB/token), {} threads: 1 lane {:.3} ms ({:.1} GB/s)",
        c.name,
        c.ty,
        c.mat_rows,
        c.k,
        c.mats,
        bytes / 1e6,
        nt,
        t1 * 1e3,
        bytes / t1 / 1e9
    );
    for &(nl, b, v) in &res[1..] {
        println!(
            "         {nl} lanes: base {:.3} ms = {:.2}x (+{:.2}x per lane) | vl {:.3} ms = {:.2}x (+{:.2}x per lane) | saves {:.3} ms",
            b * 1e3,
            b / t1,
            (b / t1 - 1.0) / (nl - 1) as f64,
            v * 1e3,
            v / t1,
            (v / t1 - 1.0) / (nl - 1) as f64,
            (b - v) * 1e3
        );
    }
    res
}

#[test]
#[ignore]
fn kbench_verify_lanes() {
    let reps: usize = std::env::var("OXE_KBENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let nt: usize = std::env::var("OXE_KBENCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let want = std::env::var("OXE_KBENCH_CASES").unwrap_or_else(|_| "head,dense,shexp,experts,qdense".into());
    let cases = [
        Case { name: "head", ty: QType::Q6K, k: 2048, mat_rows: 248320, mats: 1, pool: 1, chunk: 256 },
        Case { name: "dense", ty: QType::Q4K, k: 2048, mat_rows: 8192, mats: 90, pool: 90, chunk: lane_chunk(QType::Q4K, 2048) },
        Case { name: "shexp", ty: QType::Q6K, k: 2048, mat_rows: 512, mats: 120, pool: 120, chunk: lane_chunk(QType::Q6K, 2048) },
        Case { name: "experts", ty: QType::Q3K, k: 2048, mat_rows: 512, mats: 720, pool: 1800, chunk: lane_chunk(QType::Q3K, 2048) },
        Case { name: "qdense", ty: QType::Q8_0, k: 2048, mat_rows: 8192, mats: 78, pool: 78, chunk: lane_chunk(QType::Q8_0, 2048) },
    ];
    println!("vl_kbench: {nt} threads, {reps} reps x best of 3, weights cold (streamed), base = one-column GEMV per lane on cache-warm chunks");
    let mut tot = [[0f64; 2]; 5];
    for c in cases.iter().filter(|c| want.split(',').any(|w| w == c.name)) {
        let r = run_case(c, nt, reps);
        if c.name != "qdense" {
            for &(nl, b, v) in &r {
                tot[nl][0] += b;
                tot[nl][1] += if nl == 1 { b } else { v };
            }
        }
    }
    if tot[1][0] > 0.0 {
        println!("sum of the Ornith cases run (head + dense + shexp + experts shared by all lanes):");
        for nl in 2..=4 {
            println!(
                "         {nl} lanes: base {:.3} ms, vl {:.3} ms (1 lane {:.3} ms): extra lanes cost {:.3} -> {:.3} ms",
                tot[nl][0] * 1e3,
                tot[nl][1] * 1e3,
                tot[1][0] * 1e3,
                (tot[nl][0] - tot[1][0]) * 1e3,
                (tot[nl][1] - tot[1][0]) * 1e3
            );
        }
    }
}
