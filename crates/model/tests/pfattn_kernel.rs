//! v0.1.3 PREFILL-LONG (LC-9) model-free tests of the fa2 prefill attention kernel
//! (`model::engine::pfattn::Tile`, the code the engine runs per unit) at production shapes:
//! head_dim 256, 8 query heads per KV head, 8 tokens per unit = 64 query columns.
//!
//! - random q / K / V at n_kv ~8K / 40K / 70K: within 3e-3 relative L2 per column of exact f64
//!   attention over the same f16 inputs (scaled with the peakiness, as the O-70 tests do) and no
//!   worse than a replica of today's tiled prefill steps (+ slack for f16 noise);
//! - exact-count fixtures (uniform attention / +128 logit spikes over one-hot V rows, where every
//!   product and sum is exact): 1e-6 against f64, which catches a masked position read, a dropped
//!   or doubled chunk, a wrong causal mask between the unit's tokens or a broken online rescale;
//! - unit-split invariance: a column's output is the same bits whether its unit holds 8, 2 or 1
//!   tokens (tail rule), so the engine's unit grid never changes a result.
//!
//! K rows beyond the unit's last position and the V^T row pad beyond n_ctx are f16 NaN: any read
//! there makes an output NaN. `cargo test --release -p model --test pfattn_kernel -- --nocapture`

use kernels_sys::ffi;
use model::engine::padded_n_kv;
use model::engine::pfattn::{Scratch, Tile, PFA_CH, PFA_MAX_COLS};

const HD: usize = 256;
const GROUP: usize = 8;
const TT: usize = 8;
const NCOL: usize = TT * GROUP;
const VPAD: usize = 32;
const F16_NAN: u16 = 0x7e00;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// uniform in [-1, 1]
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

fn h2f(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                s * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

/// One KV head's cache and one unit's query columns, laid out as the engine holds them.
struct Case {
    n_ctx: usize,
    /// position of the unit's first token (tokens pos0 .. pos0 + TT)
    pos0: usize,
    /// K rows [position][HD] (f16; NaN beyond the last token)
    k: Vec<u16>,
    /// V^T rows [dim][v_stride] (f16; finite on [0, n_ctx), NaN in the pad)
    vt: Vec<u16>,
    v_stride: usize,
    /// query columns [col][HD] (f16), col = token * GROUP + head
    q16: Vec<u16>,
}

enum Kind {
    Random(f32),
    /// q = 0: uniform attention; V one-hot (V[p][d] = 1 iff d == p % HD)
    Uniform,
    /// q = 2, K = 4 at the spike positions (score 128 after scaling), 0 elsewhere; V one-hot
    Spikes(Vec<usize>),
}

impl Case {
    fn new(n_ctx: usize, pos0: usize, kind: Kind, seed: u64) -> Case {
        let last = pos0 + TT - 1;
        assert!(n_ctx % 256 == 0 && last < n_ctx);
        let v_stride = n_ctx + VPAD;
        let mut r = Lcg(seed);
        let mut k = vec![F16_NAN; n_ctx * HD];
        let mut vt = vec![F16_NAN; HD * v_stride];
        let q: Vec<f32>;
        match &kind {
            Kind::Random(qs) => {
                let kf: Vec<f32> = (0..(last + 1) * HD).map(|_| r.f()).collect();
                k[..(last + 1) * HD].copy_from_slice(&f16s(&kf));
                for d in 0..HD {
                    let row: Vec<f32> = (0..n_ctx).map(|_| r.f()).collect();
                    vt[d * v_stride..d * v_stride + n_ctx].copy_from_slice(&f16s(&row));
                }
                q = (0..NCOL * HD).map(|_| r.f() * qs).collect();
            }
            Kind::Uniform | Kind::Spikes(_) => {
                let sp: &[usize] = if let Kind::Spikes(s) = &kind { s } else { &[] };
                for p in 0..=last {
                    let x = if sp.contains(&p) { 4.0f32 } else { 0.0 };
                    k[p * HD..(p + 1) * HD].copy_from_slice(&f16s(&vec![x; HD]));
                }
                for d in 0..HD {
                    for p in 0..n_ctx {
                        vt[d * v_stride + p] = if p % HD == d { 0x3c00 } else { 0 };
                    }
                }
                let qv = if sp.is_empty() { 0.0 } else { 2.0 };
                q = vec![qv; NCOL * HD];
            }
        }
        Case { n_ctx, pos0, k, vt, v_stride, q16: f16s(&q) }
    }

    fn last(&self) -> usize {
        self.pos0 + TT - 1
    }

    /// Columns [c0, c0 + ncol) as one unit (c0 a multiple of GROUP).
    fn tile(&self, c0: usize, ncol: usize) -> Tile {
        Tile {
            hd: HD,
            ncol,
            group: GROUP,
            pos0: self.pos0 + c0 / GROUP,
            scale: 1.0 / (HD as f32).sqrt(),
            k: self.k.as_ptr(),
            vt: self.vt.as_ptr(),
            v_stride: self.v_stride,
            n_ctx: self.n_ctx,
            q16: self.q16[c0 * HD..].as_ptr(),
        }
    }
}

/// The fa2 kernel: outputs [col][HD], units of `per` tokens.
fn fa2(c: &Case, per: usize) -> Vec<f32> {
    let mut out = vec![0f32; NCOL * HD];
    let mut scr = Scratch::new(per * GROUP, HD);
    for t0 in (0..TT).step_by(per) {
        let ncol = per.min(TT - t0) * GROUP;
        let tile = c.tile(t0 * GROUP, ncol);
        // SAFETY: the case holds every row the tile reads (see the field docs).
        unsafe { tile.run(&mut scr) };
        for col in 0..ncol {
            let g = t0 * GROUP + col;
            out[g * HD..(g + 1) * HD].copy_from_slice(scr.out(col));
        }
    }
    out
}

/// Replica of today's tiled prefill unit (`pf_attention_tiled`, f16 KV): scores of every column
/// over all rows, one FA-off softmax row per column over the ubatch's padded n_kv (f16
/// probabilities), V^T . p over round_up(rows, 32) with the tail rule.
fn old_tiled(c: &Case) -> Vec<f32> {
    let (n_ctx, last) = (c.n_ctx, c.last());
    let rows = last + 1;
    let n32 = rows.div_ceil(32) * 32;
    let acs = n_ctx;
    let nkv = padded_n_kv(last, n_ctx);
    let mut akq = vec![0f32; NCOL * acs];
    let mut ap16 = vec![0u16; NCOL * acs];
    let (mut wp, mut dp) = (vec![0f32; n_ctx], vec![0f32; n_ctx]);
    let mut aot = vec![0f32; HD * NCOL];
    let nf = [nkv as i32; PFA_MAX_COLS];
    let scale = 1.0 / (HD as f32).sqrt();
    // SAFETY: buffers sized for the case; K rows < rows finite, V^T finite up to n_ctx >= nkv.
    unsafe {
        ffi::oxe_dot_f16_tile(HD as i32, c.k.as_ptr(), HD as i64, rows as i32, c.q16.as_ptr(), HD as i64, NCOL as i32, akq.as_mut_ptr(), 1, acs as i64, std::ptr::null());
        for col in 0..NCOL {
            let pos = c.pos0 + col / GROUP;
            ffi::oxe_attn_softmax_row_f16(akq.as_ptr().add(col * acs), pos as i32, nkv as i32, scale, wp.as_mut_ptr(), dp.as_mut_ptr(), ap16.as_mut_ptr().add(col * acs), n32 as i32);
        }
        ffi::oxe_dot_f16_tile(n32 as i32, c.vt.as_ptr(), c.v_stride as i64, HD as i32, ap16.as_ptr(), acs as i64, NCOL as i32, aot.as_mut_ptr(), NCOL as i64, 1, nf.as_ptr());
    }
    let mut out = vec![0f32; NCOL * HD];
    for col in 0..NCOL {
        for d in 0..HD {
            out[col * HD + d] = aot[d * NCOL + col];
        }
    }
    out
}

/// Exact attention per column in f64 over the same f16 inputs (held as f32: exact for f16).
fn ref_f64(c: &Case) -> Vec<f64> {
    let rows = c.last() + 1;
    let k: Vec<f32> = c.k[..rows * HD].iter().map(|&h| h2f(h) as f32).collect();
    let mut vt = vec![0f32; HD * rows];
    for d in 0..HD {
        for p in 0..rows {
            vt[d * rows + p] = h2f(c.vt[d * c.v_stride + p]) as f32;
        }
    }
    let scale = 1.0 / (HD as f64).sqrt();
    let mut out = vec![0f64; NCOL * HD];
    let mut s = vec![0f64; rows];
    for col in 0..NCOL {
        let n = c.pos0 + col / GROUP + 1;
        let q: Vec<f64> = c.q16[col * HD..(col + 1) * HD].iter().map(|&h| h2f(h)).collect();
        let mut m = f64::NEG_INFINITY;
        for p in 0..n {
            s[p] = k[p * HD..(p + 1) * HD].iter().zip(&q).map(|(&a, b)| a as f64 * b).sum::<f64>() * scale;
            m = m.max(s[p]);
        }
        let mut sum = 0f64;
        for x in s[..n].iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        for d in 0..HD {
            let v = &vt[d * rows..d * rows + n];
            out[col * HD + d] = s[..n].iter().zip(v).map(|(a, &b)| a * b as f64).sum::<f64>() / sum;
        }
    }
    out
}

/// Max over columns of ||x - ref||_2 / ||ref||_2.
fn rel_err(x: &[f32], r: &[f64]) -> f64 {
    x.chunks(HD)
        .zip(r.chunks(HD))
        .map(|(xs, rs)| {
            let e2: f64 = xs.iter().zip(rs).map(|(&a, &b)| (a as f64 - b) * (a as f64 - b)).sum();
            let r2: f64 = rs.iter().map(|b| b * b).sum();
            (e2 / r2.max(1e-300)).sqrt()
        })
        .fold(0f64, f64::max)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn n_ctx_for(last: usize) -> usize {
    (last + 1).div_ceil(256) * 256 + 256
}

/// Random data at n_kv ~8K / 40K / 70K (unit straddling a chunk boundary at the deepest point):
/// fa2 within the O-70 bound of exact f64 and no worse than today's tiled prefill unit.
#[test]
fn fa2_random_matches_f64_and_tiled_at_8k_40k_70k() {
    for &pos0 in &[8_190usize, 40_000, 70_001 - 4] {
        for &qs in &[1.0f32, 4.0, 16.0] {
            let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Random(qs), 0x9fa2 ^ pos0 as u64 ^ ((qs as u64) << 32));
            let r = ref_f64(&c);
            let f = fa2(&c, TT);
            assert!(f.iter().all(|v| v.is_finite()), "pos0 {pos0} qs {qs}: non-finite fa2 output (read beyond the rows?)");
            let old = old_tiled(&c);
            let (ef, eo) = (rel_err(&f, &r), rel_err(&old, &r));
            let efo = rel_err(&f, &old.iter().map(|&x| x as f64).collect::<Vec<_>>());
            println!("pfattn qs {qs:>4} n_kv {:>6}: rel L2 vs f64 fa2 {ef:.2e} tiled {eo:.2e}; fa2 vs tiled {efo:.2e}", pos0 + TT);
            let tol = 3e-3 * (qs as f64 / 4.0).max(1.0);
            assert!(ef <= tol, "pos0 {pos0} qs {qs}: fa2 error {ef:.3e} vs f64 (bound {tol:.1e})");
            assert!(ef <= eo * 1.25 + 3e-4, "pos0 {pos0} qs {qs}: fa2 error {ef:.3e} worse than tiled {eo:.3e}");
        }
    }
}

/// Uniform attention over one-hot V: every output is count / (pos + 1) exactly, at contexts
/// around the chunk / 32-block boundaries and deep (the unit straddles them).
#[test]
fn fa2_uniform_exact_counts() {
    for &pos0 in &[0usize, 1, 25, 30, 250, 252, 255, 256, 509, 8_184, 8_190, 40_000, 70_001] {
        let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Uniform, 3);
        let r = ref_f64(&c);
        for per in [TT, 2, 1] {
            let f = fa2(&c, per);
            for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                assert!((x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12, "pos0 {pos0} per {per}: out[{i}] = {x} want {y}");
            }
        }
    }
}

/// +128 logit spikes (exp under- / overflows without the running max) at chunk and 32-block
/// boundaries, deep in the context, and at a position inside the unit (only the unit's later
/// tokens may see it; earlier columns must not): exact against f64.
#[test]
fn fa2_spikes_masking_and_large_logits() {
    for &pos0 in &[0usize, 250, 8_188, 40_000] {
        let mut sp = vec![0usize, 31, 32, 255, 256, 257, 8_191, 8_192, 8_447, 8_448, 39_999];
        sp.extend([pos0 + 3, pos0 + TT - 1]);
        sp.retain(|&p| p < pos0 + TT);
        for keep_only_inside in [false, true] {
            let s: Vec<usize> = if keep_only_inside { vec![pos0 + 3] } else { sp.clone() };
            let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Spikes(s.clone()), 5);
            let r = ref_f64(&c);
            for per in [TT, 1] {
                let f = fa2(&c, per);
                for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                    assert!((x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12, "pos0 {pos0} spikes {s:?} per {per}: out[{i}] = {x} want {y}");
                }
            }
        }
    }
}

/// A column's output does not depend on the unit it is computed in (8, 2 or 1 tokens: the unit's
/// reach changes, the tail rule keeps the last chunk's dot the same), bit for bit.
#[test]
fn fa2_unit_split_is_bitwise_invariant() {
    for &pos0 in &[3usize, 253, 9_000, 40_000 - 3] {
        let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Random(4.0), 77 ^ pos0 as u64);
        let a = fa2(&c, TT);
        assert!(a.iter().all(|v| v.is_finite()));
        for per in [4usize, 2, 1] {
            assert!(same_bits(&fa2(&c, per), &a), "pos0 {pos0}: units of {per} tokens differ from units of {TT}");
        }
    }
    assert_eq!(PFA_CH % 32, 0);
}

// ---------------------------------------------------------------------------------------------
// v0.1.3 APPEND-PREFILL: the fa2x kernel (`Tile::run_x`, csrc/oxe_pfa2x.cpp), same fixtures.
// ---------------------------------------------------------------------------------------------

use model::engine::pfattn::{XParams, FA2X_MAX_CH};

const FOLDS: [usize; 3] = [8, 16, 32];

/// The fa2x kernel: outputs [col][HD], units of `per` tokens.
fn fa2x(c: &Case, per: usize, xp: XParams) -> Vec<f32> {
    let mut out = vec![0f32; NCOL * HD];
    let mut scr = Scratch::new(per * GROUP, HD);
    for t0 in (0..TT).step_by(per) {
        let ncol = per.min(TT - t0) * GROUP;
        let tile = c.tile(t0 * GROUP, ncol);
        // SAFETY: the case holds every row the tile reads (see the field docs).
        unsafe { tile.run_x(&mut scr, xp) };
        for col in 0..ncol {
            let g = t0 * GROUP + col;
            out[g * HD..(g + 1) * HD].copy_from_slice(scr.out(col));
        }
    }
    out
}

/// Random data at n_kv ~8K / 40K / 70K: fa2x within the fa2 bound of exact f64 and, at the
/// default fold 8, no worse than fa2 AND today's tiled prefill (both measured on the same case,
/// 5% for the shared f16-weight rounding noise);
/// folds 16 / 32 (lossy speed options) only within 2x the f64 bound.
#[test]
fn fa2x_random_error_vs_f64_no_worse_than_fa2_and_tiled() {
    let mut worst = Vec::new();
    for &pos0 in &[8_190usize, 40_000, 70_001 - 4] {
        for &qs in &[1.0f32, 4.0, 16.0] {
            let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Random(qs), 0x9fa2 ^ pos0 as u64 ^ ((qs as u64) << 32));
            let r = ref_f64(&c);
            let (ef, eo) = (rel_err(&fa2(&c, TT), &r), rel_err(&old_tiled(&c), &r));
            for fold in FOLDS {
                for ch in [128usize, 256] {
                    let x = fa2x(&c, TT, XParams::new(fold, ch).unwrap());
                    assert!(x.iter().all(|v| v.is_finite()), "pos0 {pos0} qs {qs} fold {fold}: non-finite fa2x output");
                    let ex = rel_err(&x, &r);
                    println!("pfattn-x qs {qs:>4} n_kv {:>6} fold {fold:>2} ch {ch:>3}: rel L2 vs f64 fa2x {ex:.3e} fa2 {ef:.3e} tiled {eo:.3e}", pos0 + TT);
                    worst.push(format!("n_kv {} qs {qs} fold {fold} ch {ch}: fa2x {ex:.3e} fa2 {ef:.3e} tiled {eo:.3e}", pos0 + TT));
                    // folds 16 / 32 are lossy speed options (CI 556f57b: up to 1.4x / 2.2x fa2's error):
                    // only the f64 bound (x2); fold 8 (the default) must be no worse than fa2 AND tiled
                    let tol = 3e-3 * (qs as f64 / 4.0).max(1.0) * if fold == 8 { 1.0 } else { 2.0 };
                    assert!(ex <= tol, "pos0 {pos0} qs {qs} fold {fold} ch {ch}: fa2x error {ex:.3e} vs f64 (bound {tol:.1e})");
                    if fold == 8 {
                        assert!(ex <= ef * 1.05 + 1e-6 && ex <= eo * 1.05 + 1e-6, "fold 8 must be no worse than today (5% noise): {}", worst.last().unwrap());
                    }
                }
            }
        }
    }
}

/// Uniform attention over one-hot V and +128 logit spikes (exact fixtures): every fold and chunk,
/// units of 8 / 2 / 1 tokens, against f64 at 1e-6.
#[test]
fn fa2x_exact_fixtures_all_folds_and_chunks() {
    let xps: Vec<XParams> = [(8usize, 32usize), (8, 128), (16, 256), (32, FA2X_MAX_CH), (32, 96)].iter().map(|&(f, c)| XParams::new(f, c).unwrap()).collect();
    for &pos0 in &[0usize, 1, 25, 30, 120, 125, 250, 252, 255, 256, 509, 8_184, 8_190, 40_000, 70_001] {
        let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Uniform, 3);
        let r = ref_f64(&c);
        for &xp in &xps {
            for per in [TT, 2, 1] {
                let f = fa2x(&c, per, xp);
                for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                    assert!((x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12, "uniform pos0 {pos0} {xp:?} per {per}: out[{i}] = {x} want {y}");
                }
            }
        }
    }
    for &pos0 in &[0usize, 120, 250, 8_188, 40_000] {
        let mut sp = vec![0usize, 31, 32, 95, 96, 127, 128, 255, 256, 257, 8_191, 8_192, 8_447, 8_448, 39_999];
        sp.extend([pos0 + 3, pos0 + TT - 1]);
        sp.retain(|&p| p < pos0 + TT);
        for keep_only_inside in [false, true] {
            let s: Vec<usize> = if keep_only_inside { vec![pos0 + 3] } else { sp.clone() };
            let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Spikes(s.clone()), 5);
            let r = ref_f64(&c);
            for &xp in &xps {
                for per in [TT, 1] {
                    let f = fa2x(&c, per, xp);
                    for (i, (&x, &y)) in f.iter().zip(&r).enumerate() {
                        assert!((x as f64 - y).abs() <= 1e-6 * y.abs() + 1e-12, "spikes pos0 {pos0} {s:?} {xp:?} per {per}: out[{i}] = {x} want {y}");
                    }
                }
            }
        }
    }
}

/// A column's fa2x output is the same bits whether its unit holds 8, 4, 2 or 1 tokens (the
/// unit's reach changes the score rows computed, never a weight or the V^T . P length).
#[test]
fn fa2x_unit_split_is_bitwise_invariant() {
    for &pos0 in &[3usize, 125, 253, 9_000, 40_000 - 3] {
        let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Random(4.0), 77 ^ pos0 as u64);
        for fold in FOLDS {
            let xp = XParams::new(fold, 128).unwrap();
            let a = fa2x(&c, TT, xp);
            assert!(a.iter().all(|v| v.is_finite()));
            for per in [4usize, 2, 1] {
                assert!(same_bits(&fa2x(&c, per, xp), &a), "pos0 {pos0} fold {fold}: units of {per} tokens differ from units of {TT}");
            }
        }
    }
}

/// GQA reuse: the unit runs the 8 query heads of the KV head (x 8 tokens) against ONE pass over
/// K / V^T. Every column equals the same query run alone (a 1-column unit, group 1: its own pass)
/// bit for bit, so no column reads another column's query, mask or running max.
#[test]
fn fa2x_gqa_shared_pass_equals_single_column_runs() {
    for &pos0 in &[5usize, 254, 9_001] {
        let c = Case::new(n_ctx_for(pos0 + TT - 1), pos0, Kind::Random(8.0), 0x6aa ^ pos0 as u64);
        let xp = XParams::default();
        let all = fa2x(&c, TT, xp);
        let mut scr = Scratch::new(1, HD);
        for col in 0..NCOL {
            let mut t = c.tile(0, NCOL);
            t.ncol = 1;
            t.group = 1;
            t.pos0 = c.pos0 + col / GROUP;
            t.q16 = c.q16[col * HD..].as_ptr();
            // SAFETY: one column of the case (rows up to its own position).
            unsafe { t.run_x(&mut scr, xp) };
            assert!(same_bits(scr.out(0), &all[col * HD..(col + 1) * HD]), "pos0 {pos0}: column {col} (head {} of token {}) differs alone", col % GROUP, col / GROUP);
        }
    }
}
