//! OD-4c kernel micro-benchmark (model-free): one attention layer's decode reads at production
//! shapes, f16 KV (`oxe_dot_f16_tile` scores over K rows + `oxe_dot_f16_tile` V^T . p, the decode
//! tiled path) against int8 KV (`oxe_dot_q8_tile` scores + the V . p kernel per 256-position chunk, the
//! fused O-13 path; OD-4d: all four V . p kernels are timed: `oxe_pv_q8_fast` (production, bitwise =
//! the reference), `oxe_pv_q8` (reference), `oxe_pv_q8_int` with 16- and 8-bit weights (lossy)). head_dim 256, 2 KV heads x 8 query columns (Ornith: 16 heads / 2 KV heads, one
//! lane), positions 6.5K / 16K / 32K, n_ctx 16384 and 32768 strides (score columns at stride n_ctx,
//! V^T rows at n_ctx + 32 as OXE_KV_VPAD's default), 1 and 8 threads.
//!
//! Run explicitly (DEV window, under build.lock):
//! `cargo test --release -p kernels-sys --test kv_kbench -- --ignored --nocapture`
//! Optional: OXE_KBENCH_REPS (default 40), OXE_KBENCH_THREADS (default "1,8").
//!
//! Threads split the work the way the engine does (scores by position, f16 V^T by row, int8 V by
//! 256-position chunk) and repeat their share `reps` times without a barrier per rep; time per
//! rep = the slowest thread's loop / reps. Bytes = K and V bytes of the positions (each read once).

use kernels_sys::ffi;
use std::time::Instant;

const HD: usize = 256;
const NKV: usize = 2;
const NCOL: usize = 8;
const QK: usize = 32;
const RB: usize = HD / QK * 34; // Q8_0 row bytes
const VPAD: usize = 32;
const PV_CHUNK: usize = 256;

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

fn f16s(v: &[f32]) -> Vec<u16> {
    let mut o = vec![0u16; v.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::oxe_fp32_to_fp16_row(v.as_ptr(), o.as_mut_ptr(), v.len() as i64) };
    o
}

fn q8_rows(n: usize, r: &mut Lcg) -> Vec<u8> {
    let mut o = vec![0u8; n * RB];
    for row in o.chunks_mut(34) {
        let d = f16s(&[0.01 + 0.001 * (r.next() % 10) as f32])[0];
        row[0..2].copy_from_slice(&d.to_le_bytes());
        for b in row[2..].iter_mut() {
            *b = (r.next() % 255) as u8;
        }
    }
    o
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

/// Wall seconds per rep: `nt` threads run `f(tid)` `reps` times; the slowest thread counts.
fn timed(nt: usize, reps: usize, f: &(dyn Fn(usize) + Sync)) -> f64 {
    for t in 0..nt {
        f(t); // warm (page faults, caches) outside the timer, single-threaded
    }
    let mut worst = 0f64;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt)
            .map(|t| {
                s.spawn(move || {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        f(t);
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

#[test]
#[ignore]
fn kv_kbench_f16_vs_q8() {
    let reps: usize = std::env::var("OXE_KBENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let threads: Vec<usize> = std::env::var("OXE_KBENCH_THREADS").unwrap_or("1,8".into()).split(',').map(|x| x.trim().parse().unwrap()).collect();
    let mut r = Lcg(0x5eed);
    println!("kv-kbench: head_dim {HD}, {NKV} KV heads x {NCOL} columns (one layer, one lane), reps {reps}");
    println!(
        "{:>6} {:>6} {:>3} | {:>9} {:>9} {:>9} {:>7} {:>8} | {:>9} {:>9} {:>9} {:>7} {:>8} | {:>6} | {:>9} {:>9} {:>9} | {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6}",
        "n_ctx", "pos", "thr", "f16 sc us", "f16 pv us", "f16 us", "GB/s", "ns/pos", "q8 sc us", "q8 pv us", "q8 us", "GB/s", "ns/pos", "q8/f16", "pv ref us", "pv i16 us", "pv i8 us", "pv/f16", "ref/f16", "i16/f16", "i8/f16", "q8i16/f", "q8i8/f"
    );
    for n_ctx in [16384usize, 32768] {
        let vs = n_ctx + VPAD;
        // f16 K [kvh][n_ctx][HD], V^T [kvh][HD][vs]; int8 K / V [kvh][n_ctx] rows of RB bytes
        let kf: Vec<u16> = f16s(&(0..NKV * n_ctx * HD).map(|_| r.f() * 0.5).collect::<Vec<_>>());
        let vf: Vec<u16> = f16s(&(0..NKV * HD * vs).map(|_| r.f() * 0.5).collect::<Vec<_>>());
        let k8 = q8_rows(NKV * n_ctx, &mut r);
        let v8 = q8_rows(NKV * n_ctx, &mut r);
        let qf: Vec<u16> = f16s(&(0..NKV * NCOL * HD).map(|_| r.f()).collect::<Vec<_>>());
        let q8 = q8_rows(NKV * NCOL, &mut r);
        for pos in [6656usize, 16384, 32768] {
            if pos > n_ctx {
                continue;
            }
            let n32 = pos.div_ceil(32) * 32;
            let nfull = vec![(pos.div_ceil(256) * 256).min(n_ctx) as i32; NCOL];
            // probabilities f16 [kvh][col][n_ctx] (+0 beyond pos)
            let mut pv = vec![0f32; NKV * NCOL * n_ctx];
            for kvh in 0..NKV {
                for c in 0..NCOL {
                    for p in 0..pos {
                        pv[(kvh * NCOL + c) * n_ctx + p] = (r.next() % 1000) as f32 / (1000.0 * pos as f32);
                    }
                }
            }
            let p16 = f16s(&pv);
            for &nt in &threads {
                // per-thread scratch / outputs (score columns at stride n_ctx, as the engine's acs)
                let scores: Vec<std::sync::Mutex<Vec<f32>>> = (0..nt).map(|_| std::sync::Mutex::new(vec![0f32; NKV * NCOL * n_ctx])).collect();
                let outs: Vec<std::sync::Mutex<Vec<f32>>> = (0..nt).map(|_| std::sync::Mutex::new(vec![0f32; 32 * NCOL + NCOL * HD + 8192])).collect();
                let (kf, vf, k8, v8, qf, q8, p16, nfull) = (P(kf.as_ptr() as _), P(vf.as_ptr() as _), P(k8.as_ptr()), P(v8.as_ptr()), P(qf.as_ptr() as _), P(q8.as_ptr()), P(p16.as_ptr() as _), P(nfull.as_ptr() as _));
                let rows = pos * NKV;
                // f16 scores: rows (kvh, p) split over threads, 64-position tiles
                let f16_sc = |t: usize| {
                    let (a0, b0) = (rows * t / nt, rows * (t + 1) / nt);
                    let mut sc = scores[t].lock().unwrap();
                    for kvh in 0..NKV {
                        let a = a0.max(kvh * pos).min((kvh + 1) * pos) - kvh * pos;
                        let b = b0.max(kvh * pos).min((kvh + 1) * pos) - kvh * pos;
                        let mut p = a;
                        while p < b {
                            let pe = (p + 64).min(b);
                            // SAFETY: rows within the buffers; outputs column-strided n_ctx.
                            unsafe {
                                ffi::oxe_dot_f16_tile(
                                    HD as i32,
                                    (kf.p() as *const u16).add((kvh * n_ctx + p) * HD),
                                    HD as i64,
                                    (pe - p) as i32,
                                    (qf.p() as *const u16).add(kvh * NCOL * HD),
                                    HD as i64,
                                    NCOL as i32,
                                    sc.as_mut_ptr().add(kvh * NCOL * n_ctx + p),
                                    1,
                                    n_ctx as i64,
                                    std::ptr::null(),
                                );
                            }
                            p = pe;
                        }
                    }
                };
                // f16 V^T . p: V^T rows (kvh, d) split over threads, 32-row tiles over n32 positions
                let f16_pv = |t: usize| {
                    let all = NKV * HD;
                    let (a0, b0) = (all * t / nt, all * (t + 1) / nt);
                    let mut o = outs[t].lock().unwrap();
                    for kvh in 0..NKV {
                        let a = a0.max(kvh * HD).min((kvh + 1) * HD) - kvh * HD;
                        let b = b0.max(kvh * HD).min((kvh + 1) * HD) - kvh * HD;
                        let mut d = a;
                        while d < b {
                            let de = (d + 32).min(b);
                            // SAFETY: as above; n32 <= n_ctx < vs.
                            unsafe {
                                ffi::oxe_dot_f16_tile(
                                    n32 as i32,
                                    (vf.p() as *const u16).add((kvh * HD + d) * vs),
                                    vs as i64,
                                    (de - d) as i32,
                                    (p16.p() as *const u16).add(kvh * NCOL * n_ctx),
                                    n_ctx as i64,
                                    NCOL as i32,
                                    o.as_mut_ptr(),
                                    NCOL as i64,
                                    1,
                                    nfull.p() as *const i32,
                                );
                            }
                            d = de;
                        }
                    }
                };
                // int8 scores: Q8_0 K rows x Q8_0 query columns, rows split over threads
                let q8_sc = |t: usize| {
                    let (a0, b0) = (rows * t / nt, rows * (t + 1) / nt);
                    let mut sc = scores[t].lock().unwrap();
                    for kvh in 0..NKV {
                        let a = a0.max(kvh * pos).min((kvh + 1) * pos) - kvh * pos;
                        let b = b0.max(kvh * pos).min((kvh + 1) * pos) - kvh * pos;
                        if a < b {
                            // SAFETY: rows within the buffers.
                            unsafe {
                                ffi::oxe_dot_q8_tile(
                                    HD as i32,
                                    k8.p().add((kvh * n_ctx + a) * RB),
                                    RB as i64,
                                    (b - a) as i32,
                                    q8.p().add(kvh * NCOL * RB),
                                    RB as i64,
                                    NCOL as i32,
                                    sc.as_mut_ptr().add(kvh * NCOL * n_ctx + a),
                                    1,
                                    n_ctx as i64,
                                );
                            }
                        }
                    }
                };
                // int8 V . p: (kvh, 256-position chunk) units split over threads; kernel 0 = fast (production),
                // 1 = reference, 2 = int16 weights, 3 = int8 weights
                let q8_pv_k = |t: usize, kern: usize| {
                    let nch = n32.div_ceil(PV_CHUNK);
                    let units = NKV * nch;
                    let mut o = outs[t].lock().unwrap();
                    let (o1, scr) = o.split_at_mut(32 * NCOL + NCOL * HD);
                    for u in units * t / nt..units * (t + 1) / nt {
                        let (kvh, ch) = (u / nch, u % nch);
                        let (p0, p1) = (ch * PV_CHUNK, ((ch + 1) * PV_CHUNK).min(n32));
                        // SAFETY: V rows [p0, p1) of head kvh; probabilities column-strided n_ctx.
                        unsafe {
                            let (v, pr) = (v8.p().add(kvh * n_ctx * RB), (p16.p() as *const u16).add(kvh * NCOL * n_ctx));
                            let (hd, rb, a, b, ps, nc) = (HD as i32, RB as i64, p0 as i32, p1 as i32, n_ctx as i64, NCOL as i32);
                            match kern {
                                0 => ffi::oxe_pv_q8_fast(hd, v, rb, a, b, pr, ps, nc, o1.as_mut_ptr(), scr.as_mut_ptr()),
                                1 => ffi::oxe_pv_q8(hd, v, rb, a, b, pr, ps, nc, o1.as_mut_ptr(), scr.as_mut_ptr()),
                                _ => ffi::oxe_pv_q8_int(hd, v, rb, a, b, pr, ps, nc, o1.as_mut_ptr(), scr.as_mut_ptr() as *mut core::ffi::c_void, if kern == 2 { 16 } else { 8 }),
                            }
                        }
                    }
                };
                let q8_pv = |t: usize| q8_pv_k(t, 0);
                let q8_pv_ref = |t: usize| q8_pv_k(t, 1);
                let q8_pv_i16 = |t: usize| q8_pv_k(t, 2);
                let q8_pv_i8 = |t: usize| q8_pv_k(t, 3);
                let tf_sc = timed(nt, reps, &f16_sc);
                let tf_pv = timed(nt, reps, &f16_pv);
                let tq_sc = timed(nt, reps, &q8_sc);
                let tq_pv = timed(nt, reps, &q8_pv);
                let tq_pv_ref = timed(nt, reps, &q8_pv_ref);
                let tq_pv_i16 = timed(nt, reps, &q8_pv_i16);
                let tq_pv_i8 = timed(nt, reps, &q8_pv_i8);
                let (tf, tq) = (tf_sc + tf_pv, tq_sc + tq_pv);
                let bf = (NKV * pos * HD * 2 * 2) as f64;
                let bq = (NKV * pos * RB * 2) as f64;
                println!(
                    "{:>6} {:>6} {:>3} | {:>9.1} {:>9.1} {:>9.1} {:>7.1} {:>8.2} | {:>9.1} {:>9.1} {:>9.1} {:>7.1} {:>8.2} | {:>6.3} | {:>9.1} {:>9.1} {:>9.1} | {:>6.3} {:>6.3} {:>6.3} {:>6.3} | {:>6.3} {:>6.3}",
                    n_ctx,
                    pos,
                    nt,
                    tf_sc * 1e6,
                    tf_pv * 1e6,
                    tf * 1e6,
                    bf / tf / 1e9,
                    tf * 1e9 / pos as f64,
                    tq_sc * 1e6,
                    tq_pv * 1e6,
                    tq * 1e6,
                    bq / tq / 1e9,
                    tq * 1e9 / pos as f64,
                    tq / tf,
                    tq_pv_ref * 1e6,
                    tq_pv_i16 * 1e6,
                    tq_pv_i8 * 1e6,
                    tq_pv / tf_pv,
                    tq_pv_ref / tf_pv,
                    tq_pv_i16 / tf_pv,
                    tq_pv_i8 / tf_pv,
                    (tq_sc + tq_pv_i16) / tf,
                    (tq_sc + tq_pv_i8) / tf
                );
            }
        }
    }
    println!("q8 pv = oxe_pv_q8_fast (production, bitwise = pv ref = oxe_pv_q8); pv i16 / i8 = oxe_pv_q8_int (lossy); x/f16 = time ratio to the f16 V^T . p; q8i16/f, q8i8/f = whole int8 read with that kernel / f16 read");
    println!("GB/s = K+V bytes of the positions (f16 2 B/elem; int8 {RB} B per {HD}-row, K and V) / time; ns/pos = per position per layer ({NKV} KV heads)");
}
