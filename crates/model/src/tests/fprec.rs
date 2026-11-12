//! O-70 precision probe (model-free): why is the fused attention further from the f32-attention
//! reference than the tiled path on the mini model? Captures the mini model's real attention
//! inputs (q f32 roped, stored f16 K / V) through the `attn_ref` hook and replays them through
//! the kernels and through a step-by-step emulation that switches one f16 step at a time to
//! exact arithmetic.
//!
//! Finding (2026-11-12): fused is NOT further from the reference. On the mini attention inputs
//! (near-uniform softmax, score spread ~0.03) both kernels are ~1e-4 relative, fused slightly
//! better (1.19e-4 at ch 256, 8.3e-5 at ch 128, tiled 1.42e-4); ~90% of that is the f16 V^T p
//! accumulation that both share (f32 accumulation: 1e-5 / 1.8e-5), the f16 weights and the
//! f16 query are negligible. At the logits the per-position error is heavy-tailed: ~10% of the
//! positions jump to ~1e-2..1e-1 RMS for ANY ~1e-4 attention change (a ~4e-7 change never does),
//! so `oxe fused-ref`'s 20-sample mean (tiled 1.3e-5, fused 1e-2) was two jumps landing on fused.

use super::*;
use kernels_sys::ffi;
use std::sync::Mutex;

struct Cap {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    n: usize,
    hd: usize,
    scale: f32,
}

static CAPS: Mutex<Vec<Cap>> = Mutex::new(Vec::new());
static CAPTURE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn ref64(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, scale: f32) -> Vec<f64> {
    let mut s = vec![0f64; n];
    let mut m = f64::NEG_INFINITY;
    for p in 0..n {
        let a: f64 = (0..hd).map(|j| q[j] as f64 * k[p * hd + j] as f64).sum();
        s[p] = a * scale as f64;
        m = m.max(s[p]);
    }
    let mut sum = 0f64;
    for x in s.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    let mut o = vec![0f64; hd];
    for p in 0..n {
        for j in 0..hd {
            o[j] += s[p] * v[p * hd + j] as f64;
        }
    }
    o.iter().map(|x| x / sum).collect()
}

fn hook(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, scale: f32, out: &mut [f32]) {
    let o = ref64(q, k, v, n, hd, scale);
    for j in 0..hd {
        out[j] = o[j] as f32;
    }
    if CAPTURE.load(std::sync::atomic::Ordering::Relaxed) {
        CAPS.lock().unwrap().push(Cap { q: q.to_vec(), k: k[..n * hd].to_vec(), v: v[..n * hd].to_vec(), n, hd, scale });
    }
}

/// The reference over the f16-rounded query: an attention perturbation of ~4e-7 relative.
fn hook_q16(q: &[f32], k: &[f32], v: &[f32], n: usize, hd: usize, scale: f32, out: &mut [f32]) {
    let q16: Vec<f32> = f16v(q).iter().map(|&h| h2f(h) as f32).collect();
    let o = ref64(&q16, k, v, n, hd, scale);
    for j in 0..hd {
        out[j] = o[j] as f32;
    }
}

fn f16v(x: &[f32]) -> Vec<u16> {
    let mut o = vec![0u16; x.len()];
    // SAFETY: equal lengths.
    unsafe { ffi::oxe_fp32_to_fp16_row(x.as_ptr(), o.as_mut_ptr(), x.len() as i64) };
    o
}
fn r16(x: f64) -> f64 {
    h2f(f16v(&[x as f32])[0]) // f64 -> f32 -> f16: double rounding is irrelevant at this level
}
fn h2f(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => if m == 0.0 { s * f64::INFINITY } else { f64::NAN },
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// f16 FMA accumulation of one ggml_vec_dot_f16 (32 lanes: 4 accumulators x 8, lane = i mod 32,
/// pairwise f16 reduce, f32 horizontal sum) of exact f64 products; `acc32`: f32 accumulators.
fn dot_emu(x: &[f64], y: &[f64], acc32: bool) -> f64 {
    let rnd = |v: f64| if acc32 { v as f32 as f64 } else { r16(v) };
    let mut a = [0f64; 32];
    for (i, (&xi, &yi)) in x.iter().zip(y).enumerate() {
        a[i % 32] = rnd(a[i % 32] + xi * yi);
    }
    let mut l = [0f64; 8];
    for t in 0..8 {
        let s02 = rnd(a[t] + a[16 + t]);
        let s13 = rnd(a[8 + t] + a[24 + t]);
        l[t] = rnd(s02 + s13);
    }
    l.iter().map(|&v| v as f32).fold(0f32, |s, v| s + v) as f64
}

struct Knobs {
    /// f16 weights (else exact f64)
    w16: bool,
    /// f16 accumulation of V^T p (else f32)
    acc16: bool,
    /// fused: unnormalised chunk weights + chunk merge; tiled: normalised probabilities
    fused: bool,
    ch: usize,
}

/// Emulation over f16 q16 / K / V (scores exact in f64 from the f16 inputs, then f32).
fn emu(c: &Cap, kb: &Knobs) -> Vec<f64> {
    let (n, hd) = (c.n, c.hd);
    let q16: Vec<f64> = f16v(&c.q).iter().map(|&h| h2f(h)).collect();
    let s: Vec<f64> = (0..n).map(|p| ((0..hd).map(|j| q16[j] * c.k[p * hd + j] as f64).sum::<f64>() as f32 * c.scale) as f64).collect();
    let w = |v: f64| if kb.w16 { r16(v) } else { v };
    let mut out = vec![0f64; hd];
    if !kb.fused {
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = s.iter().map(|x| (x - m).exp()).sum();
        let p: Vec<f64> = s.iter().map(|x| w((x - m).exp() / sum)).collect();
        for j in 0..hd {
            let vj: Vec<f64> = (0..n).map(|q| c.v[q * hd + j] as f64).collect();
            out[j] = dot_emu(&vj, &p, !kb.acc16);
        }
        return out;
    }
    let mut parts = Vec::new();
    for s0 in (0..n).step_by(kb.ch) {
        let e = (s0 + kb.ch).min(n);
        let m = s[s0..e].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let wv: Vec<f64> = s[s0..e].iter().map(|x| w((x - m).exp())).collect();
        let l: f64 = wv.iter().sum();
        let pd: Vec<f64> = (0..hd).map(|j| dot_emu(&(s0..e).map(|q| c.v[q * hd + j] as f64).collect::<Vec<_>>(), &wv, !kb.acc16)).collect();
        parts.push((m, l, pd));
    }
    let mx = parts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let lsum: f64 = parts.iter().map(|p| (p.0 - mx).exp() * p.1).sum();
    for (m, _, pd) in &parts {
        let wc = (m - mx).exp();
        for j in 0..hd {
            out[j] += wc * pd[j];
        }
    }
    out.iter().map(|x| x / lsum).collect()
}

/// The engine's tiled f16 steps through the kernels (scores, softmax row, V^T p).
fn tiled_kernel(c: &Cap, n_ctx: usize) -> Vec<f64> {
    let (n, hd) = (c.n, c.hd);
    let pos = n - 1;
    let n_kv = crate::engine::padded_n_kv(pos, n_ctx);
    let n32 = n.div_ceil(32) * 32;
    let k16 = f16v(&c.k);
    let q16 = f16v(&c.q);
    let mut kq = vec![0f32; n_ctx];
    let stride = n_ctx + 32;
    let mut vt = vec![0u16; hd * stride];
    let v16 = f16v(&c.v);
    for p in 0..n {
        for j in 0..hd {
            vt[j * stride + p] = v16[p * hd + j];
        }
    }
    let (mut wp, mut dp, mut p16) = (vec![0f32; n_ctx], vec![0f32; n_ctx], vec![0u16; n_ctx]);
    let mut out = vec![0f32; hd];
    let nf = [n_kv as i32];
    // SAFETY: buffers sized above.
    unsafe {
        ffi::oxe_dot_f16_tile(hd as i32, k16.as_ptr(), hd as i64, n as i32, q16.as_ptr(), hd as i64, 1, kq.as_mut_ptr(), 1, 1, std::ptr::null());
        ffi::oxe_attn_softmax_row_f16(kq.as_ptr(), pos as i32, n_kv as i32, c.scale, wp.as_mut_ptr(), dp.as_mut_ptr(), p16.as_mut_ptr(), n32 as i32);
        ffi::oxe_dot_f16_tile(n32 as i32, vt.as_ptr(), stride as i64, hd as i32, p16.as_ptr(), n32 as i64, 1, out.as_mut_ptr(), 1, 1, nf.as_ptr());
    }
    out.iter().map(|&x| x as f64).collect()
}

/// The fused kernels (chunk items + merge), one column.
fn fused_kernel(c: &Cap, n_ctx: usize, ch: usize) -> Vec<f64> {
    let (n, hd) = (c.n, c.hd);
    let n32 = n.div_ceil(32) * 32;
    let nch = n32.div_ceil(ch);
    let k16 = f16v(&c.k);
    let q16 = f16v(&c.q);
    let stride = n_ctx + 32;
    let mut vt = vec![0u16; hd * stride];
    let v16 = f16v(&c.v);
    for p in 0..n {
        for j in 0..hd {
            vt[j * stride + p] = v16[p * hd + j];
        }
    }
    let mut part = vec![0f32; nch * hd];
    let mut ml = vec![0f32; nch * 2];
    let mut s = vec![0f32; ch];
    let mut p16 = vec![0u16; ch];
    // SAFETY: buffers sized above; V^T finite up to the stride.
    unsafe {
        for it in 0..nch {
            let s0 = it * ch;
            let e = (s0 + ch).min(n32);
            let len = e - s0;
            let nv = [(n - s0).min(len) as i32];
            ffi::oxe_dot_f16_tile(hd as i32, k16.as_ptr().add(s0 * hd), hd as i64, nv[0], q16.as_ptr(), hd as i64, 1, s.as_mut_ptr(), 1, 1, std::ptr::null());
            ffi::oxe_attn_fchunk_f16(s.as_mut_ptr(), ch as i64, 1, nv.as_ptr(), len as i32, ((s0 + ch).min(n_ctx) - s0) as i32, c.scale, vt.as_ptr().add(s0), stride as i64, hd as i32, p16.as_mut_ptr(), part.as_mut_ptr().add(it * hd), ml.as_mut_ptr().add(it * 2));
        }
        let gate = vec![1f32; hd];
        let mut out = vec![0f32; hd];
        let mut w = vec![0f32; nch.div_ceil(4) * 4];
        ffi::oxe_attn_fmerge_f32(nch as i32, part.as_ptr(), hd as i64, ml.as_ptr(), ml.as_ptr().add(1), 2, 0, hd as i32, gate.as_ptr(), out.as_mut_ptr(), w.as_mut_ptr());
        out.iter().map(|&x| x as f64).collect()
    }
}

fn rel(x: &[f64], r: &[f64]) -> f64 {
    let e: f64 = x.iter().zip(r).map(|(a, b)| (a - b) * (a - b)).sum();
    let d: f64 = r.iter().map(|b| b * b).sum();
    (e / d.max(1e-300)).sqrt()
}

/// Probe: real mini-model attention inputs, every arm and every one-step-exact variant, rel L2 vs
/// the f64 reference (f32 q, as `oxe fused-ref`) and vs f64 over the f16 query.
#[test]
fn mini_fused_precision_probe() {
    let (dir, path) = mini_paths("fprec");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt = mini_opts(&dir);
    let g = gguf::Gguf::open(&path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    let plan = Plan::build(&g.header, &desc, 2, 512).unwrap();
    let ws = WeightStore::load(g, &plan, &opt).unwrap();
    let pool = sched::Pool::new(&test_cpus(2), Duration::from_micros(100)).unwrap();
    let cfg = EngineConfig { n_ctx: 512, logits: true, trace: false, profile: false };
    let prefix: Vec<u32> = (0..200u32).map(|i| (i * 13 + 7) % 64).collect();
    let cont: Vec<u32> = (0..250u32).map(|i| (i * 29 + 5) % 64).collect();
    let mut a = Engine::new(&plan, &ws, cfg.clone()).unwrap();
    a.prefill(&pool, &prefix, 64).unwrap();
    a.set_attn_legacy(false);
    a.set_attn_tiled_min_kv(0);
    let rms = |x: &[f32], y: &[f32]| (x.iter().zip(y).map(|(&p, &q)| ((p - q) as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    // arms: tiled, fused 256, fused 128, reference over the f16 query (a ~4e-7 attention change)
    let mut errs: [Vec<f64>; 4] = Default::default();
    let mut stats = Vec::new();
    for (i, &t) in cont.iter().enumerate() {
        let s = a.save_recurrent();
        CAPTURE.store(i % 10 == 0, std::sync::atomic::Ordering::Relaxed);
        a.set_attn_ref(Some(hook));
        a.set_attn_fused(false);
        a.step(&pool, t);
        let lr = a.logits().to_vec();
        CAPTURE.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut outs = Vec::new();
        for arm in 0..4 {
            a.restore_recurrent(&s);
            a.set_attn_ref(if arm == 3 { Some(hook_q16) } else { None });
            a.set_attn_fused(arm == 1 || arm == 2);
            a.set_attn_fused_chunk(if arm == 2 { 128 } else { 256 });
            a.step(&pool, t);
            outs.push(a.logits().to_vec());
        }
        a.set_attn_ref(None);
        a.set_attn_fused(false);
        // continue on the tiled step (the incumbent state), bit-identical to the tiled arm
        a.restore_recurrent(&s);
        a.step(&pool, t);
        assert!(a.logits().iter().zip(&outs[0]).all(|(x, y)| x.to_bits() == y.to_bits()));
        for k in 0..4 {
            errs[k].push(rms(&outs[k], &lr));
        }
    }
    for (k, nm) in ["tiled", "fused256", "fused128", "ref(q16)"].iter().enumerate() {
        let mut e = errs[k].clone();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = e.len();
        let big: Vec<usize> = (0..n).filter(|&i| errs[k][i] > 1e-3).collect();
        println!("LOGIT RMS vs f32-attn ref, {nm:>9} ({n} forks): mean {:.3e} median {:.3e} p90 {:.3e} max {:.3e}; > 1e-3 at {} forks {:?}", e.iter().sum::<f64>() / n as f64, e[n / 2], e[n * 9 / 10], e[n - 1], big.len(), big);
        stats.push((e[n / 2], e[n * 9 / 10], e[n - 1]));
    }
    // control: a ~4e-7 attention change never jumps; the f16 arms (~1e-4) jump at ~10% of the
    // positions (discontinuous downstream: MoE top-2 routing), so a mean over a few forks is
    // dominated by which arm happens to hit a jump. Compare the typical error instead.
    assert!(stats[3].2 < 1e-4, "control arm jumped: {:?}", stats[3]);
    for k in [1, 2] {
        assert!(stats[k].0 <= stats[0].0 && stats[k].1 <= stats[0].1, "fused arm {k} worse than tiled: {:?} vs {:?}", stats[k], stats[0]);
    }
    let caps = std::mem::take(&mut *CAPS.lock().unwrap());
    let names = ["tiled_kernel", "fused_kernel256", "fused_kernel128", "T emu", "T w-exact", "T acc32", "T exact", "F emu", "F w-exact", "F acc32", "F exact", "F emu ch128"];
    let mut sum32 = vec![0f64; names.len()];
    let mut sum16 = vec![0f64; names.len()];
    let (mut spread, mut vabs, mut oabs) = (0f64, 0f64, 0f64);
    for c in &caps {
        let r32 = ref64(&c.q, &c.k, &c.v, c.n, c.hd, c.scale);
        let q16: Vec<f32> = f16v(&c.q).iter().map(|&h| h2f(h) as f32).collect();
        let r16q = ref64(&q16, &c.k, &c.v, c.n, c.hd, c.scale);
        let kb = |w16, acc16, fused, ch| Knobs { w16, acc16, fused, ch };
        let outs = [
            tiled_kernel(c, 512),
            fused_kernel(c, 512, 256),
            fused_kernel(c, 512, 128),
            emu(c, &kb(true, true, false, 0)),
            emu(c, &kb(false, true, false, 0)),
            emu(c, &kb(true, false, false, 0)),
            emu(c, &kb(false, false, false, 0)),
            emu(c, &kb(true, true, true, 256)),
            emu(c, &kb(false, true, true, 256)),
            emu(c, &kb(true, false, true, 256)),
            emu(c, &kb(false, false, true, 256)),
            emu(c, &kb(true, true, true, 128)),
        ];
        for (i, o) in outs.iter().enumerate() {
            sum32[i] += rel(o, &r32);
            sum16[i] += rel(o, &r16q);
        }
        let s: Vec<f64> = (0..c.n).map(|p| (0..c.hd).map(|j| c.q[j] as f64 * c.k[p * c.hd + j] as f64).sum::<f64>() * c.scale as f64).collect();
        spread += s.iter().cloned().fold(f64::MIN, f64::max) - s.iter().cloned().fold(f64::MAX, f64::min);
        vabs += (c.v.iter().map(|&x| (x * x) as f64).sum::<f64>() / c.v.len() as f64).sqrt();
        oabs += (r32.iter().map(|x| x * x).sum::<f64>() / r32.len() as f64).sqrt();
    }
    let m = caps.len() as f64;
    println!("{} captures: mean score spread {:.3e}, V rms {:.3e}, out rms {:.3e}", caps.len(), spread / m, vabs / m, oabs / m);
    for (i, nm) in names.iter().enumerate() {
        println!("  {nm:>16}: rel L2 vs ref(f32 q) {:.3e}  vs ref(f16 q) {:.3e}", sum32[i] / m, sum16[i] / m);
    }
    let mean = |v: &[f64]| v[..].iter().sum::<f64>() / m;
    let (t, f256, f128) = (mean(&sum32[0..1]), mean(&sum32[1..2]), mean(&sum32[2..3]));
    assert!(f256 <= t && f128 <= t, "fused kernel error above tiled on the mini attention inputs: {f256:.3e} / {f128:.3e} vs {t:.3e}");
    // the error is the f16 V^T p accumulation (shared with ggml's FA-off path), not the f16
    // weights, the query rounding or the chunk merge
    for (emu_i, w_i, acc_i) in [(3, 4, 5), (7, 8, 9)] {
        assert!(sum32[acc_i] < sum32[emu_i] / 3.0 && sum32[w_i] > sum32[emu_i] * 0.9, "step attribution changed: {:?}", &sum32[..]);
    }
    std::fs::remove_dir_all(&dir).ok();
}
