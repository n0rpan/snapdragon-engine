//! LONG-CTX (model-free): context validation, buffer sizing, decode / verify / MTP / rollback at
//! n_kv far beyond 32K for every KV type, int8-vs-f16 attention error against f64, and the state
//! round trip (prefix store / parked session bytes) beyond 32K.
//!
//! Long contexts come from synthetic states: K / V rows of `n_pos` positions written in the
//! [`Engine::write_state`] format and read back with [`Engine::read_state`] (the prefix-store
//! path), so a 70K-position context costs one byte copy instead of a 70K-token prefill.

use super::*;
use crate::engine::{ctx_mem, validate_n_ctx, EngineOptions, KvType, MAX_LANES, MAX_N_CTX};

/// The trained context of Ornith / Qwen3.6 (GGUF context_length).
const CTX_TRAIN: u32 = 262_144;
/// K amplitude of the synthetic engine contexts: peaked softmax, score spread ~10-15. At x400 the
/// fused int8 K path (query quantised to Q8_0, as llama.cpp does for a q8_0 K cache) leaves the
/// 2 x f16 + 1e-3 band (1.5e-3; 9.7e-3 at x1000): Q8_0 score error grows with the score scale.
const K_AMP: f32 = 250.0;

fn long_parts(tag: &str, nt: usize, mtp: bool) -> (PathBuf, Plan, WeightStore, Option<crate::mtp::MtpWeights>, sched::Pool) {
    let dir = std::env::temp_dir().join(format!("oxe-longctx-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mini-long.gguf");
    mini_model_ctx(&path, mtp, CTX_TRAIN);
    std::env::set_var("OXE_MODEL_SHA256", if mtp { "1".repeat(64) } else { "0".repeat(64) });
    let opt = mini_opts(&dir);
    let g = gguf::Gguf::open(&path).unwrap();
    let desc = ModelDesc::from_header(&g.header).unwrap();
    assert_eq!(desc.n_ctx_train, CTX_TRAIN as usize);
    let plan = Plan::build(&g.header, &desc, nt, 512).unwrap();
    let ws = WeightStore::load(g, &plan, &opt).unwrap();
    let mw = mtp.then(|| crate::mtp::MtpWeights::load(&path, &plan, &opt).unwrap());
    let pool = sched::Pool::new(&test_cpus(nt), Duration::from_micros(50)).unwrap();
    (dir, plan, ws, mw, pool)
}

fn long_engine<'a>(plan: &'a Plan, ws: &'a WeightStore, n_ctx: usize, kv: KvType, lanes: usize, mtp: Option<&'a crate::mtp::MtpWeights>) -> Engine<'a> {
    let cfg = EngineConfig { n_ctx, logits: true, trace: false, profile: false };
    Engine::with_options(plan, ws, cfg, EngineOptions { verify_lanes: lanes, mtp, kv_type: Some(kv) }).unwrap()
}

fn h2f(h: u16) -> f32 {
    let s = if h >> 15 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * (2f32).powi(-24),
        31 => s * f32::INFINITY,
        _ => s * (1.0 + m / 1024.0) * (2f32).powi(e - 15),
    }
}

fn to_f16(x: &[f32]) -> Vec<u16> {
    let mut h = vec![0u16; x.len()];
    kernels_sys::ops::fp32_to_fp16(x, &mut h);
    h
}

fn q8(x: &[f32]) -> Vec<u8> {
    let mut b = vec![0u8; x.len() / 32 * 34];
    // SAFETY: x.len() floats (a multiple of 32) into x.len() / 32 blocks of 34 bytes.
    unsafe { kernels_sys::ffi::quantize_row_q8_0(x.as_ptr(), b.as_mut_ptr(), x.len() as i64) };
    b
}

fn deq8(b: &[u8], out: &mut [f32]) {
    for (blk, o) in b.chunks(34).zip(out.chunks_mut(32)) {
        let d = h2f(u16::from_le_bytes([blk[0], blk[1]]));
        for j in 0..32 {
            o[j] = (blk[2 + j] as i8) as f32 * d;
        }
    }
}

/// K / V row (f32) of attention layer `ai`, KV head `h`, position `p` of the synthetic context:
/// uniform [-1, 1) times `amp`, two outlier channels x6 in K (as post-norm, roped K rows have).
fn synth_row(ai: usize, h: usize, p: usize, hd: usize, is_k: bool) -> Vec<f32> {
    let seed = ((ai * 7 + h) as u64) << 40 | (p as u64) << 1 | is_k as u64;
    let mut r = lcg_f32(hd, seed);
    if is_k {
        r[3] *= 6.0;
        r[hd / 2 + 5] *= 6.0;
    }
    r
}

/// A [`Engine::write_state`] image of `n_pos` synthetic positions for engine `e` (its KV type and
/// attention layers), followed by `rec` (the recurrent bytes of one slot).
fn synth_state(e: &Engine, n_pos: usize, rec: &[u8]) -> Vec<u8> {
    synth_state_shift(e, n_pos, rec, 0)
}

/// [`synth_state`] with every V row taken from position p + `shift` (V misaligned with K by
/// `shift`, as a wrong-row read would see it; shifting K and V together would only permute the
/// (K, V) pairs, which attention cannot see): the sensitivity control of the comparisons.
fn synth_state_shift(e: &Engine, n_pos: usize, rec: &[u8], shift: usize) -> Vec<u8> {
    let d = &e.plan.desc;
    let (hd, kvh, kv) = (d.head_dim, d.n_head_kv, e.kv_type());
    let mut b = Vec::with_capacity(e.ctx_mem().kv_per_pos * n_pos + rec.len());
    for ai in 0..e.n_att() {
        for h in 0..kvh {
            for p in 0..n_pos {
                // x250 (the mini q is ~0.05 per dim): scores spread far enough that the softmax is peaked over a few positions
                // (a diffuse 70K average is ~0 and vanishes in the activation quantisation)
                let k: Vec<f32> = synth_row(ai, h, p, hd, true).iter().map(|x| x * K_AMP).collect();
                if kv.k8() {
                    b.extend(q8(&k));
                } else {
                    to_f16(&k).iter().for_each(|x| b.extend(x.to_le_bytes()));
                }
            }
        }
        if kv.v8() {
            for h in 0..kvh {
                for p in 0..n_pos {
                    b.extend(q8(&synth_row(ai, h, p + shift, hd, false)));
                }
            }
        } else {
            // V^T: row j = kv head * head_dim + dim, n_pos f16 values
            let mut vt = vec![0u16; kvh * hd * n_pos];
            for h in 0..kvh {
                for p in 0..n_pos {
                    let v = to_f16(&synth_row(ai, h, p + shift, hd, false));
                    for j in 0..hd {
                        vt[(h * hd + j) * n_pos + p] = v[j];
                    }
                }
            }
            vt.iter().for_each(|x| b.extend(x.to_le_bytes()));
        }
    }
    b.extend_from_slice(rec);
    b
}

/// Recurrent bytes of one slot after a few real tokens (so the synthetic context has a
/// non-trivial delta-net state too).
fn rec_bytes(e: &Engine, pool: &sched::Pool) -> Vec<u8> {
    e.reset();
    for t in [5u32, 9, 13, 2] {
        e.step(pool, t);
    }
    let mut b = Vec::new();
    e.write_state(&mut b).unwrap();
    let kv = e.ctx_mem().kv_per_pos * e.pos();
    e.reset();
    b[kv..].to_vec()
}

fn rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let sc = b.iter().fold(1f32, |m, x| m.max(x.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max) / sc
}

/// (1) n_ctx is validated against the GGUF context_length and every context-sized buffer is sized
/// from the validated capacity: the engine's own buffers equal [`ctx_mem`] for every KV type up to
/// the 256K trained context (MTP and verify-lane engines included); `state_bytes` is the exact
/// length `write_state` writes (int8 K engines counted no KV before LONG-CTX).
#[test]
fn long_ctx_validation_and_buffer_sizing() {
    assert!(validate_n_ctx(0, 4096).is_err());
    assert_eq!(validate_n_ctx(1, 4096), Ok(256));
    assert_eq!(validate_n_ctx(4096, 4096), Ok(4096));
    assert!(validate_n_ctx(4097, 4096).unwrap_err().contains("trained context"));
    assert_eq!(validate_n_ctx(262_144, 262_144), Ok(262_144));
    assert!(validate_n_ctx(262_145, 262_144).is_err());
    assert_eq!(validate_n_ctx(32_768, 262_144), Ok(32_768));
    assert_eq!(validate_n_ctx(40_001, 262_144), Ok(40_192));
    assert!(validate_n_ctx(MAX_N_CTX + 1, 0).is_err());
    // the 4096-context mini model refuses 8K; the 256K one takes it
    let (dir4, path4) = mini_paths("ctx4k");
    std::env::set_var("OXE_MODEL_SHA256", "0".repeat(64));
    let opt4 = mini_opts(&dir4);
    let (plan4, ws4, _pool4) = mini_engine_parts(&path4, &opt4, 1);
    let err = Engine::new(&plan4, &ws4, EngineConfig { n_ctx: 8192, logits: false, trace: false, profile: false }).err().expect("8K on a 4K model must fail");
    assert!(err.contains("trained context"), "{err}");
    assert!(Engine::new(&plan4, &ws4, EngineConfig { n_ctx: 4096, logits: false, trace: false, profile: false }).is_ok());
    std::fs::remove_dir_all(&dir4).ok();

    let (dir, plan, ws, _, pool) = long_parts("size", 2, false);
    let d = &plan.desc;
    for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
        for n in [512usize, 32_768, 40_001, 65_536, 131_072, 262_144] {
            for lanes in [0usize, 4] {
                let e = long_engine(&plan, &ws, n, kv, lanes, None);
                let got = e.ctx_mem();
                let want = ctx_mem(d, n, kv, e.n_att(), e.lanes(), e.verify_lanes());
                assert_eq!(got, want, "{kv:?} n_ctx {n} lanes {lanes}");
                assert_eq!(got.n_ctx, n.div_ceil(256) * 256);
                assert!(got.kv >= got.n_ctx * got.kv_per_pos);
            }
        }
    }
    // f16 vs Q8_0 per position: 2 x 256 B vs 2 x 136 B per (layer, KV head) at head_dim 128
    let f = ctx_mem(d, 65_536, KvType::F16, 1, 1, 1).kv_per_pos;
    let q = ctx_mem(d, 65_536, KvType::KV8, 1, 1, 1).kv_per_pos;
    assert_eq!((f, q), (2 * 2 * 128 * 2, 2 * 2 * 136));
    // state_bytes == the bytes write_state writes, every KV type (the prefix store's floor check)
    for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
        let e = long_engine(&plan, &ws, 1024, kv, 0, None);
        for t in 0..37u32 {
            e.step(&pool, t % 64);
        }
        let mut b = Vec::new();
        e.write_state(&mut b).unwrap();
        assert_eq!(e.state_bytes(), b.len(), "{kv:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
    // MTP engine: MTP layer KV + MAX_LANES attention lanes, 2 recurrent slots
    let (dir, plan, ws, mw, _pool) = long_parts("sizemtp", 1, true);
    for kv in [KvType::F16, KvType::KV8] {
        let e = long_engine(&plan, &ws, 131_072, kv, 2, mw.as_ref());
        assert_eq!((e.n_att(), e.lanes(), e.verify_lanes()), (2, MAX_LANES, 2));
        assert_eq!(e.ctx_mem(), ctx_mem(&plan.desc, 131_072, kv, 2, MAX_LANES, 2), "MTP {kv:?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// (1b) The KV memory table of the real models (GGUF header fixtures, no model file): f16 KV is
/// 20 KiB per position over the 10 attention layers, 22 KiB with the MTP layer; Q8_0 is 17/32 of
/// it. Printed for the design note (LONG-CTX-DESIGN.md).
#[test]
fn long_ctx_memory_table_of_ornith_and_qwen36() {
    for key in ["ornith", "qwen36"] {
        let h = header(&fixture(key));
        let d = ModelDesc::from_header(&h).unwrap();
        assert_eq!((d.n_attention(), d.n_head_kv, d.head_dim, d.n_ctx_train), (10, 2, 256, 262_144), "{key}");
        let lanes = MAX_LANES; // MTP engine
        for (kv, n_att, per) in [(KvType::F16, 10, 20_480), (KvType::F16, 11, 22_528), (KvType::KV8, 11, 11_968), (KvType::K8, 11, 17_248)] {
            assert_eq!(ctx_mem(&d, 4096, kv, n_att, lanes, 2).kv_per_pos, per, "{key} {kv:?} {n_att}");
        }
        let rec = ctx_mem(&d, 4096, KvType::F16, 11, lanes, 1).recurrent;
        assert_eq!(rec, 30 * ((3 * 8192) + 32 * 128 * 128) * 4, "{key}: recurrent slot");
        let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
        for n in [32_768usize, 65_536, 131_072, 262_144] {
            let f = ctx_mem(&d, n, KvType::F16, 11, lanes, 2);
            let q = ctx_mem(&d, n, KvType::KV8, 11, lanes, 2);
            eprintln!(
                "{key} n_ctx {n:>6}: KV f16 {:.3} GiB, kv8 {:.3} GiB; recurrent (2 slots) {:.3}; decode scratch f16 {:.3} / kv8 {:.3}; per thread decode {:.4}, prefill {:.4}; total (8 dec, 12 pf) f16 {:.3} kv8 {:.3} GiB",
                gib(f.kv), gib(q.kv), gib(f.recurrent), gib(f.decode_scratch), gib(q.decode_scratch),
                gib(f.decode_per_thread), gib(f.prefill_per_thread), gib(f.total(8, 12)), gib(q.total(8, 12))
            );
        }
    }
}

/// Exact attention of one query in f64 over stored K / V rows (`k(p)`, `v(p)` give the f32 values
/// a read of position p sees); `p16`: round the normalised probabilities to f16 first, as the
/// engine's tiled decode (and ggml's non-flash attention) does. Returns (output, prob of `needle`).
fn attn64(q: &[f32], n: usize, hd: usize, scale: f64, k: &dyn Fn(usize, &mut [f32]), v: &dyn Fn(usize, &mut [f32]), p16: bool, needle: usize) -> (Vec<f64>, f64) {
    let mut row = vec![0f32; hd];
    let mut s = vec![0f64; n];
    let mut m = f64::NEG_INFINITY;
    for p in 0..n {
        k(p, &mut row);
        s[p] = q.iter().zip(&row).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() * scale;
        m = m.max(s[p]);
    }
    let mut sum = 0f64;
    for x in s.iter_mut() {
        *x = (*x - m).exp();
        sum += *x;
    }
    for x in s.iter_mut() {
        *x /= sum;
        if p16 {
            *x = h2f(to_f16(&[*x as f32])[0]) as f64;
        }
    }
    let mut out = vec![0f64; hd];
    for p in 0..n {
        v(p, &mut row);
        for j in 0..hd {
            out[j] += s[p] * row[j] as f64;
        }
    }
    (out, s[needle])
}

fn rows_f32(src: &[f32], hd: usize) -> impl Fn(usize, &mut [f32]) + '_ {
    move |p, o| o.copy_from_slice(&src[p * hd..(p + 1) * hd])
}

/// Rows as an f16 cache holds them.
fn rows_f16(src: &[f32], hd: usize) -> impl Fn(usize, &mut [f32]) {
    let h = to_f16(src);
    move |p, o| {
        for j in 0..hd {
            o[j] = h2f(h[p * hd + j]);
        }
    }
}

/// Rows as a Q8_0 cache holds them (quantised once per row, dequantised exactly).
fn rows_q8(src: &[f32], hd: usize) -> impl Fn(usize, &mut [f32]) {
    let rb = hd / 32 * 34;
    let b: Vec<u8> = src.chunks(hd).flat_map(q8).collect();
    move |p, o| deq8(&b[p * rb..(p + 1) * rb], o)
}

fn rel_l2(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-300)).sqrt()
}

/// (2) Storage error of the KV types against exact f64 attention at Ornith's head_dim 256, at 32K,
/// 64K and 128K positions: f16 K/V, Q8_0 K (k8), Q8_0 V (v8), both (kv8); plus the engine's
/// f16-probability rounding (tiled decode) on top of f16 and kv8 KV. K rows carry two outlier
/// channels and the context one "needle" position aligned with q (long-range retrieval). Prints the
/// table (LONG-CTX-DESIGN.md); the asserts are loose bounds that catch a broken quantiser or a
/// context-dependent blow-up, not a quality verdict (O-62 KLD decides that on the real model).
#[test]
fn long_ctx_int8_vs_f16_attention_error_vs_f64() {
    let hd = 256;
    let scale = 1.0 / (hd as f64).sqrt();
    let q: Vec<f32> = synth_row(99, 0, 0, hd, true).iter().map(|x| x * 1.7).collect();
    for n in [32_768usize, 65_536, 131_072] {
        let needle = n / 3;
        // the true f32 rows (needle: K = 0.9 q: score ~17 against a diffuse rest whose
        // exp-sum is ~e^16-17, so the needle holds ~0.5-0.8 of the mass: retrieval + diffuse mix)
        let mut kt = vec![0f32; n * hd];
        let mut vt = vec![0f32; n * hd];
        for p in 0..n {
            let k = if p == needle { q.iter().map(|x| x * 0.9).collect() } else { synth_row(0, 0, p, hd, true).iter().map(|x| x * 1.7).collect::<Vec<f32>>() };
            kt[p * hd..(p + 1) * hd].copy_from_slice(&k);
            vt[p * hd..(p + 1) * hd].copy_from_slice(&synth_row(0, 0, p, hd, false));
        }
        let (k32, v32) = (rows_f32(&kt, hd), rows_f32(&vt, hd));
        let (k16, v16) = (rows_f16(&kt, hd), rows_f16(&vt, hd));
        let (k8, v8) = (rows_q8(&kt, hd), rows_q8(&vt, hd));
        let (want, pn) = attn64(&q, n, hd, scale, &k32, &v32, false, needle);
        let arms: [(&str, &dyn Fn(usize, &mut [f32]), &dyn Fn(usize, &mut [f32]), bool); 6] = [
            ("f16", &k16, &v16, false),
            ("k8", &k8, &v16, false),
            ("v8", &k16, &v8, false),
            ("kv8", &k8, &v8, false),
            ("f16+p16", &k16, &v16, true),
            ("kv8+p16", &k8, &v8, true),
        ];
        let mut line = format!("n {n:>6} needle p {pn:.4}:");
        let mut errs = Vec::new();
        for (name, k, v, p16) in arms {
            let (got, gn) = attn64(&q, n, hd, scale, k, v, p16, needle);
            let e = rel_l2(&got, &want);
            assert!(e.is_finite());
            assert!(((gn - pn) / pn).abs() < 0.05, "n {n} {name}: needle probability {gn} vs {pn}");
            line += &format!(" {name} {e:.2e}");
            errs.push((name, e));
        }
        eprintln!("long-ctx attention rel L2 vs f64: {line}");
        let err = |nm: &str| errs.iter().find(|x| x.0 == nm).unwrap().1;
        assert!(err("f16") < 1e-3, "n {n}: f16 KV {}", err("f16"));
        for nm in ["k8", "v8", "kv8"] {
            assert!(err(nm) < 3e-2 && err(nm) >= err("f16") * 0.5, "n {n}: {nm} {}", err(nm));
        }
        assert!(err("f16+p16") < 5e-2 && err("kv8+p16") < 6e-2, "n {n}: f16-probability rounding {} / {}", err("f16+p16"), err("kv8+p16"));
    }
}

/// (3) Decode, verify lanes and rollback at a 70,001-position context (n_ctx 131072, tiled
/// attention), every KV type: (a) decode equals the engine with the attention core replaced by a
/// straightforward f32 reference over the SAME stored rows (int8 arms within twice the f16 cache's
/// own gap; a wrong row / chunk / offset beyond 32K or 64K is far above it); (b) a 4-lane verify
/// equals one-token steps bit for bit, and a partial commit (rollback of the rejected lanes) then
/// continues bit-identically; (c) f16: the per-head legacy path equals the tiled one bit for bit;
/// (d) the recurrent checkpoint (save / restore, the server's snapshot) replays exactly.
#[test]
fn long_ctx_decode_verify_rollback_every_kv_type_at_70k() {
    let (dir, plan, ws, _, pool) = long_parts("dec70k", 2, false);
    let n_pos = 70_001;
    let n_ctx = 131_072;
    let toks = [7u32, 21, 3, 50, 11, 4];
    let mut base = 0f32;
    for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
        let a = long_engine(&plan, &ws, n_ctx, kv, 4, None);
        let rec = rec_bytes(&a, &pool);
        let blob = synth_state(&a, n_pos, &rec);
        assert_eq!(blob.len(), a.ctx_mem().kv_per_pos * n_pos + rec.len());
        // (a) vs the f32 reference attention
        let mut r = long_engine(&plan, &ws, n_ctx, kv, 4, None);
        r.set_attn_ref(Some(ref_attn_f32 as crate::engine::AttnRef));
        a.read_state(&mut &blob[..], n_pos).unwrap();
        r.read_state(&mut &blob[..], n_pos).unwrap();
        assert!(a.barriers_per_token() == r.barriers_per_token());
        let mut want = Vec::new();
        let mut worst = 0f32;
        for &t in &toks {
            let g = a.step(&pool, t);
            r.step(&pool, t);
            assert!(a.logits().iter().all(|x| x.is_finite()), "{kv:?}: non-finite logits");
            assert_eq!(g as usize, argmax(a.logits()));
            worst = worst.max(rel_diff(a.logits(), r.logits()));
            want.push(a.logits().to_vec());
        }
        if kv == KvType::F16 {
            base = worst;
            assert!(base < 0.05, "f16 tiled vs f32 reference at 70K: {base}");
            // sensitivity: V misaligned with K by one position (an off-by-one row bug)
            // must land far outside the tolerance used below
            let z = long_engine(&plan, &ws, n_ctx, kv, 0, None);
            z.read_state(&mut &synth_state_shift(&z, n_pos, &rec, 1)[..], n_pos).unwrap();
            let mut off = 0f32;
            for (i, &t) in toks.iter().enumerate() {
                z.step(&pool, t);
                off = off.max(rel_diff(z.logits(), &want[i]));
            }
            eprintln!("long-ctx sensitivity: off-by-one context differs by {off:.2e} (f16 vs reference {base:.2e})");
            assert!(off > 10.0 * (2.0 * base + 1e-3), "the comparison cannot see an off-by-one context ({off})");
        } else {
            assert!(worst <= 2.0 * base + 1e-3, "{kv:?}: {worst} vs the f32 reference (f16 cache: {base})");
        }
        assert!(kv_all(&a, 1) == kv_all(&r, 1), "{kv:?}: reference engine stored other KV bytes");
        eprintln!("long-ctx {kv:?} at {n_pos}: max relative logit diff vs f32 reference attention {worst:.2e}");
        // (b) verify lanes + rollback
        let b = long_engine(&plan, &ws, n_ctx, kv, 4, None);
        b.read_state(&mut &blob[..], n_pos).unwrap();
        let g = b.verify(&pool, &[toks[0], toks[1], toks[2], 63]);
        for ln in 0..3 {
            assert!(bits_eq(b.logits_lane(ln), &want[ln]), "{kv:?}: verify lane {ln} differs from the step");
            assert_eq!(g[ln] as usize, argmax(&want[ln]));
        }
        b.commit(2); // toks[2] and the junk lane rejected
        assert_eq!(b.pos(), n_pos + 2);
        for (i, &t) in toks.iter().enumerate().skip(2) {
            b.step(&pool, t);
            assert!(bits_eq(b.logits(), &want[i]), "{kv:?}: step {i} after the rollback differs");
        }
        assert!(kv_all(&a, 1) == kv_all(&b, 1), "{kv:?}: KV after rollback differs");
        // (d) checkpoint at n_pos + 2, wander off, restore, replay
        let c = long_engine(&plan, &ws, n_ctx, kv, 0, None);
        c.read_state(&mut &blob[..], n_pos).unwrap();
        c.step(&pool, toks[0]);
        c.step(&pool, toks[1]);
        let snap = c.save_recurrent();
        for t in [1u32, 2, 3, 4, 5] {
            c.step(&pool, t);
        }
        c.restore_recurrent(&snap);
        for (i, &t) in toks.iter().enumerate().skip(2) {
            c.step(&pool, t);
            assert!(bits_eq(c.logits(), &want[i]), "{kv:?}: step {i} after the checkpoint restore differs");
        }
        // (c) legacy per-head path (f16 only: int8 always takes the tiled path)
        if kv == KvType::F16 {
            let mut l = long_engine(&plan, &ws, n_ctx, kv, 0, None);
            l.set_attn_legacy(true);
            l.read_state(&mut &blob[..], n_pos).unwrap();
            for (i, &t) in toks.iter().enumerate() {
                l.step(&pool, t);
                assert!(bits_eq(l.logits(), &want[i]), "legacy attention differs from tiled at step {i}");
            }
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// (4) The state round trip beyond 32K, as the prefix store and parked sessions use it: an engine
/// at 40,960 + 5 positions writes its state; engines with another capacity (65536, 262144) read it,
/// hold the same KV / recurrent bytes, and decode bit-identically; for every KV type. A state of
/// another KV type is refused by layout.
#[test]
fn long_ctx_state_round_trip_beyond_32k() {
    let (dir, plan, ws, _, pool) = long_parts("rt", 2, false);
    let n_pos = 40_960;
    for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
        let a = long_engine(&plan, &ws, 65_536, kv, 0, None);
        let rec = rec_bytes(&a, &pool);
        a.read_state(&mut &synth_state(&a, n_pos, &rec)[..], n_pos).unwrap();
        for t in [3u32, 8, 13, 21, 34] {
            a.step(&pool, t);
        }
        let mut buf = Vec::new();
        a.write_state(&mut buf).unwrap();
        assert_eq!(buf.len(), a.state_bytes());
        let b = long_engine(&plan, &ws, 262_144, kv, 0, None);
        assert_eq!(a.state_layout(), b.state_layout());
        b.read_state(&mut &buf[..], a.pos()).unwrap();
        assert_eq!(b.pos(), n_pos + 5);
        assert!(kv_all(&a, 1) == kv_all(&b, 1), "{kv:?}: KV bytes differ after the round trip");
        assert!(same_state(&a.dump_state(), &b.dump_state()), "{kv:?}: state differs after the round trip");
        let (mut xa, mut xb) = (1u32, 1u32);
        for s in 0..6 {
            xa = a.step(&pool, xa);
            xb = b.step(&pool, xb);
            assert!(bits_eq(a.logits(), b.logits()), "{kv:?}: decode step {s} after the round trip differs");
        }
        // a smaller engine cannot take it
        let small = long_engine(&plan, &ws, 32_768, kv, 0, None);
        assert!(small.read_state(&mut &buf[..], n_pos + 5).is_err());
        if kv != KvType::F16 {
            assert_ne!(a.state_layout(), long_engine(&plan, &ws, 1024, KvType::F16, 0, None).state_layout());
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// (5) MTP speculation at a 70K context (depth 1 and 2, f16 and kv8 KV, MTP layer KV synthetic
/// too): greedy output and every accepted row are bit-identical to plain decode of the same
/// context, with rejected drafts (rollback) exercised.
#[test]
fn long_ctx_mtp_greedy_identical_at_70k() {
    let (dir, plan, ws, mw, pool) = long_parts("mtp70k", 2, true);
    let mw = mw.unwrap();
    let n_pos = 70_001;
    let n_gen = 24;
    let prompt = [3u32, 17, 42, 5];
    for kv in [KvType::F16, KvType::KV8] {
        // plain reference: the main layer's rows only (layout without the MTP layer)
        let r = long_engine(&plan, &ws, 131_072, kv, 0, None);
        let rec = rec_bytes(&r, &pool);
        r.read_state(&mut &synth_state(&r, n_pos, &rec)[..], n_pos).unwrap();
        let mut next = 0;
        for &t in &prompt {
            next = r.step(&pool, t);
        }
        let mut want_ids = vec![next];
        let mut want_rows = vec![r.logits().to_vec()];
        for _ in 1..n_gen {
            next = r.step(&pool, next);
            want_ids.push(next);
            want_rows.push(r.logits().to_vec());
        }
        drop(r);
        for depth in [1usize, 2] {
            let e = long_engine(&plan, &ws, 131_072, kv, depth + 1, Some(&mw));
            // same main-layer rows (layer 0) + a synthetic MTP layer (layer 1); same recurrent state
            let blob = synth_state(&e, n_pos, &rec);
            e.read_state(&mut &blob[..], n_pos).unwrap();
            let mut d = crate::mtp::MtpDriver::new(&e, depth).unwrap();
            let truth = want_ids.clone();
            let p0 = n_pos + prompt.len();
            d.draft_override = Some(Box::new(move |pos, dr| {
                let i = pos - p0;
                if pos % 4 == 0 || i >= truth.len() {
                    dr ^ 1
                } else {
                    truth[i]
                }
            }));
            let mut x = d.prefill(&e, &pool, &prompt);
            assert_eq!(x, want_ids[0], "{kv:?} depth {depth}: prefill argmax");
            let mut ids = vec![x];
            let mut out = Vec::new();
            while ids.len() < n_gen {
                let p = e.pos();
                d.step(&e, &pool, x, &mut out);
                for (i, &t) in out.iter().enumerate() {
                    let row = p + i + 1 - p0;
                    if row < want_rows.len() {
                        assert!(bits_eq(e.logits_lane(i), &want_rows[row]), "{kv:?} depth {depth}: row {row} differs");
                    }
                    ids.push(t);
                }
                x = *out.last().unwrap();
            }
            ids.truncate(n_gen);
            assert_eq!(ids, want_ids, "{kv:?} depth {depth}: ids differ");
            assert!(d.stats.accepted > 0 && d.stats.accepted < d.stats.drafted, "{kv:?} depth {depth}: {:?}", d.stats);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// CACHE-PERFECT P2 (model-free, tiny synthetic hybrid model): the state store's segmented state
/// (256-row KV blocks + KV rows of the partial block + the recurrent snapshot) loaded into a FRESH
/// engine gives exactly the uninterrupted run: same state bytes, bit-identical logits for every
/// following decode step. Every KV type, with and without the MTP layer (its KV rows are in the
/// blocks). The snapshot is taken at the end of the prompt (head) and at an earlier checkpoint
/// below a live position (turn: rows below it are unchanged, the recurrent state is the copy).
#[test]
fn state_blocks_round_trip_bitwise() {
    for mtp in [false, true] {
        let (dir, plan, ws, mw, pool) = long_parts(if mtp { "p2rt-mtp" } else { "p2rt" }, 2, mtp);
        let a: Vec<u32> = (0..700).map(|i| (i * 13 + 7) % 64).collect();
        for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
            let decode = |e: &Engine, n: usize| -> Vec<Vec<f32>> {
                let mut tok = 5u32;
                (0..n)
                    .map(|_| {
                        tok = e.step(&pool, tok);
                        e.logits().to_vec()
                    })
                    .collect()
            };
            let lanes = if mtp { 2 } else { 0 };
            let src = long_engine(&plan, &ws, 1024, kv, lanes, mw.as_ref());
            // turn checkpoint at 600 (taken while prefilling), head at 700
            src.prefill(&pool, &a[..600], 256).unwrap();
            let turn = src.save_recurrent();
            src.prefill(&pool, &a[600..], 256).unwrap();
            assert_eq!(src.pos(), 700);
            let mut blocks = Vec::new();
            for i in 0..2 {
                let mut b = Vec::new();
                src.write_kv_rows(i * 256, i * 256 + 256, &mut b).unwrap();
                assert_eq!(b.len(), 256 * src.kv_row_bytes(), "{kv:?}: block payload size");
                blocks.push(b);
            }
            let snap = |pos: usize, rec: &crate::engine::RecurrentSnapshot| {
                let mut s = Vec::new();
                src.write_kv_rows(512, pos, &mut s).unwrap();
                rec.write_to(&mut s).unwrap();
                s
            };
            let head = snap(700, &src.save_recurrent());
            let turn_s = snap(600, &turn);
            let mut want_state = Vec::new();
            src.write_state(&mut want_state).unwrap();
            let want = decode(&src, 8);
            // the turn reference: the uninterrupted run from 600
            src.restore_recurrent(&turn);
            let mut want_turn_state = Vec::new();
            src.write_state(&mut want_turn_state).unwrap();
            let want_turn = decode(&src, 8);
            for (pos, s, want_st, want_rows) in [(700usize, &head, &want_state, &want), (600, &turn_s, &want_turn_state, &want_turn)] {
                let dst = long_engine(&plan, &ws, 1024, kv, lanes, mw.as_ref());
                // a used engine (other content first): the load must not depend on what was there
                dst.prefill(&pool, &a[..300].iter().map(|x| (x + 1) % 64).collect::<Vec<_>>(), 256).unwrap();
                dst.reset();
                for (i, b) in blocks.iter().enumerate() {
                    dst.read_kv_rows(i * 256, i * 256 + 256, &mut &b[..]).unwrap();
                }
                let mut r = &s[..];
                dst.read_kv_rows(512, pos, &mut r).unwrap();
                let rec = crate::engine::RecurrentSnapshot::read_from(&mut r, dst.recurrent_lens()).unwrap();
                assert!(r.is_empty());
                dst.restore_recurrent(&rec);
                assert_eq!(dst.pos(), pos);
                let mut got_state = Vec::new();
                dst.write_state(&mut got_state).unwrap();
                assert!(got_state == *want_st, "{kv:?} mtp {mtp} pos {pos}: state bytes differ from the uninterrupted run");
                let got = decode(&dst, 8);
                for (i, (x, y)) in got.iter().zip(want_rows).enumerate() {
                    assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "{kv:?} mtp {mtp} pos {pos}: decode row {i} differs");
                }
            }
            // a recurrent snapshot of another layout is refused, rows beyond the position too
            let mut bad = Vec::new();
            src.save_recurrent().write_to(&mut bad).unwrap();
            assert!(crate::engine::RecurrentSnapshot::read_from(&mut &bad[..], (1, 1)).is_err());
            assert!(src.write_kv_rows(0, src.pos() + 1, &mut Vec::new()).is_err());
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// CACHE-PERFECT P2 numerics epoch guard (O-96): the state cache's identity holds
/// `engine::NUMERICS_EPOCH` instead of the binary's hash, so every numerics change
/// MUST bump it. This golden test hashes the state and logits of a fixed prefill + decode of the
/// tiny synthetic model on every KV type. If it fails, the numerics changed: bump NUMERICS_EPOCH
/// (engine.rs) and put the new hash and epoch in GOLDEN (one commit).
///
/// P2b (review M3): the golden is keyed by the kernel path (`engine::kernel_path`: target, CPU
/// features detected and built for, libm probe), the same string the state cache's identity
/// holds, so a platform is "known" exactly when its caches are comparable. An UNKNOWN kernel path
/// FAILS (it printed and passed before): record its hash here first. Only the GitHub-hosted runner
/// (build.yml, Cobalt 100, not recorded: its minutes are not ours to spend on this) opts out with
/// `OXE_NUMERICS_GOLDEN_UNLISTED=print`. Covers 600 tokens: deep-context paths (tiled attention
/// beyond ATTN_TILED_MIN_KV) are outside it.
#[test]
fn numerics_epoch_golden_state() {
    const EPOCH: u32 = crate::engine::NUMERICS_EPOCH;
    // (Check64 of the kernel path, epoch, golden); the comment names the machine
    const GOLDEN: &[(&str, u32, &str)] = &[
        // Snapdragon X Elite laptop (Oryon) AND the lab-arm64 helper (Apple M2, Asahi): the same kernel
        // path (dotprod+i8mm+fp16+bf16, built i8mm+dotprod, same libm probe) and the same golden
        // (ci-arm64 run 43208505724 enforced it: unknown keys fail since P2b)
        ("4dbfb70103822fda", EPOCH, "fd48bb85e25c7a76"),
    ];
    let (dir, plan, ws, _, pool) = long_parts("golden", 2, false);
    let a: Vec<u32> = (0..600).map(|i| (i * 13 + 7) % 64).collect();
    // FNV-1a 64 over every logit bit and state byte (a change detector, not a security hash)
    struct Fnv(u64);
    impl Fnv {
        fn update(&mut self, b: &[u8]) {
            for &x in b {
                self.0 = (self.0 ^ x as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    let mut h = Fnv(0xcbf2_9ce4_8422_2325);
    for kv in [KvType::F16, KvType::K8, KvType::V8, KvType::KV8] {
        let e = long_engine(&plan, &ws, 1024, kv, 0, None);
        e.prefill(&pool, &a, 256).unwrap();
        let mut tok = 5u32;
        for _ in 0..4 {
            tok = e.step(&pool, tok);
            for x in e.logits() {
                h.update(&x.to_bits().to_le_bytes());
            }
        }
        let mut st = Vec::new();
        e.write_state(&mut st).unwrap();
        h.update(&st);
    }
    let got = format!("{:016x}", h.0);
    let kp = crate::engine::kernel_path();
    let key = {
        let mut c = crate::check::Check64::default();
        c.update(kp.as_bytes());
        format!("{:016x}", c.finish())
    };
    std::fs::remove_dir_all(&dir).ok();
    // one line CI logs can be grepped for (`numerics-golden:`), on every platform
    eprintln!("numerics-golden: key {key} epoch {EPOCH} state {got} kernel_path [{kp}]");
    match GOLDEN.iter().find(|g| g.0 == key) {
        Some(&(_, epoch, want)) => {
            assert_eq!(epoch, EPOCH, "NUMERICS_EPOCH was bumped: record the new golden for kernel path {key} (got {got})");
            assert_eq!(got, want, "numerics changed on kernel path {key} [{kp}]: bump NUMERICS_EPOCH (engine.rs) and this golden together");
        }
        None if std::env::var("OXE_NUMERICS_GOLDEN_UNLISTED").as_deref() == Ok("print") => {
            eprintln!("numerics_epoch_golden_state: kernel path {key} is not allowlisted (OXE_NUMERICS_GOLDEN_UNLISTED=print): not checked");
        }
        None => panic!("kernel path {key} [{kp}] is not allowlisted: add (\"{key}\", {EPOCH}, \"{got}\") to GOLDEN in numerics_epoch_golden_state (one commit), or this platform's state cache identity is unguarded"),
    }
}
