//! FFI-vs-C-reference tests on small synthetic shapes (single-threaded).
use super::*;

pub(crate) struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn f16_bits(v: f32) -> [u8; 2] {
    let mut h = [0u16];
    f32_to_f16(&[v], &mut h);
    h[0].to_le_bytes()
}

/// Random but valid row-major blocks: random quant bytes, small positive f16 scales.
fn random_rows(ty: QType, nrows: usize, k: usize, rng: &mut Rng) -> Vec<u8> {
    let bb = ty.block_bytes();
    let mut v: Vec<u8> = (0..nrows * ty.row_bytes(k)).map(|_| rng.next_u64() as u8).collect();
    for blk in v.chunks_mut(bb) {
        let d = f16_bits(0.001 + 0.02 * rng.f32());
        match ty {
            QType::Q4K | QType::Q5K => {
                blk[0..2].copy_from_slice(&d);
                let m = f16_bits(0.001 + 0.01 * rng.f32());
                blk[2..4].copy_from_slice(&m);
            }
            QType::Q3K => blk[108..110].copy_from_slice(&d),
            QType::Q6K => blk[208..210].copy_from_slice(&d),
            QType::Q8_0 => blk[0..2].copy_from_slice(&d),
        }
    }
    v
}

fn random_act(at: ActType, k: usize, rng: &mut Rng) -> Vec<u8> {
    let x: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let mut q = vec![0u8; at.bytes(k)];
    quantize_act(at, &x, &mut q);
    q
}

fn close(a: f32, b: f32, scale: f32, rel: f32) -> bool {
    (a - b).abs() <= rel * scale.max(1e-6)
}

#[test]
fn block_sizes_match_c() {
    unsafe {
        assert_eq!(ffi::oxe_sizeof_block_q8_K(), std::mem::size_of::<BlockQ8K>());
        assert_eq!(ffi::oxe_sizeof_block_q3_K(), QType::Q3K.block_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q4_K(), QType::Q4K.block_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q6_K(), QType::Q6K.block_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q3_Kx8(), QType::Q3K.block_x_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q4_Kx8(), QType::Q4K.block_x_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q6_Kx8(), QType::Q6K.block_x_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q5_K(), QType::Q5K.block_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q5_Kx8(), QType::Q5K.block_x_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q8_0(), QType::Q8_0.block_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q8_0x4(), QType::Q8_0.block_x_bytes());
        assert_eq!(ffi::oxe_sizeof_block_q8_K(), Q8K_BLOCK_BYTES);
    }
    assert_eq!(std::mem::size_of::<BlockQ8K>(), 292);
}

fn check_type(ty: QType, nrows: usize, k: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let rows = random_rows(ty, nrows, k, &mut rng);
    let mut packed = vec![0u8; rows.len()];
    repack(ty, &rows, &mut packed, nrows, k);
    let act = random_act(ty.act(), k, &mut rng);

    let mut neon = vec![0f32; nrows];
    let mut generic = vec![0f32; nrows];
    gemv(ty, k, &mut neon, &packed, &act);
    gemv_generic(ty, k, &mut generic, &packed, &act);

    let scale = generic.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut exact = 0;
    for r in 0..nrows {
        // the row-major reference also checks the repack (layout) itself
        let reference = vec_dot_generic(ty, k, &rows[r * ty.row_bytes(k)..], &act);
        assert!(close(neon[r], generic[r], scale, 1e-5), "{} row {r}: neon {} generic {}", ty.name(), neon[r], generic[r]);
        assert!(close(neon[r], reference, scale, 1e-4), "{} row {r}: neon {} vec_dot {}", ty.name(), neon[r], reference);
        exact += (neon[r].to_bits() == generic[r].to_bits()) as usize;
    }
    eprintln!("{}: {nrows}x{k}: {exact}/{nrows} rows bit-identical NEON vs generic, max |y| {scale:.3}", ty.name());

    // row-range call (what each worker does): rows [8, nrows-8) via the group offset
    let (r0, r1) = (8, nrows - 8);
    let mut part = vec![0f32; r1 - r0];
    let off = ty.group_offset(r0, k);
    gemv(ty, k, &mut part, &packed[off..], &act);
    assert_eq!(&part[..], &neon[r0..r1], "{} row-range call differs", ty.name());
}

#[test]
fn q4k_gemv_matches_references() {
    check_type(QType::Q4K, 32, 2048, 1);
    check_type(QType::Q4K, 24, 512, 2);
}

#[test]
fn q3k_gemv_matches_references() {
    check_type(QType::Q3K, 32, 2048, 3);
    check_type(QType::Q3K, 24, 512, 4);
}

#[test]
fn q6k_gemv_matches_references() {
    check_type(QType::Q6K, 32, 2048, 5);
    check_type(QType::Q6K, 24, 512, 6);
}

#[test]
fn q5k_gemv_matches_references() {
    check_type(QType::Q5K, 32, 2048, 8);
    check_type(QType::Q5K, 24, 512, 9);
}

#[test]
fn q8_0_gemv_matches_references() {
    check_type(QType::Q8_0, 32, 2048, 10);
    check_type(QType::Q8_0, 24, 512, 11);
}

#[test]
fn ops_basic_sanity() {
    let x: Vec<f32> = (0..256).map(|i| (i as f32 - 100.0) * 0.01).collect();
    let mut y = vec![0f32; 256];
    ops::rms_norm(&x, &mut y, 1e-6);
    let ms: f32 = y.iter().map(|v| v * v).sum::<f32>() / 256.0;
    assert!((ms - 1.0).abs() < 1e-4);
    let mut p = vec![0f32; 256];
    let mut wp = vec![0f32; 256];
    ops::soft_max(&x, &mut p, &mut wp);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    let mut ids = [0i32; 8];
    let mut scr = vec![0i32; 256];
    ops::argsort_top_k(&x, 8, &mut ids, &mut scr);
    assert_eq!(ids, [255, 254, 253, 252, 251, 250, 249, 248]);
    // ties: falls back to std::sort, still a valid descending top-k
    let t = vec![1.0f32; 256];
    ops::argsort_top_k(&t, 8, &mut ids, &mut scr);
    assert!(ids.iter().all(|&i| (0..256).contains(&i)));
    assert!((ops::sigmoid1(0.0) - 0.5).abs() < 1e-7);
}

#[test]
fn quantize_q8k_roundtrip_is_sane() {
    let mut rng = Rng::new(7);
    let x: Vec<f32> = (0..512).map(|_| rng.f32() * 4.0 - 2.0).collect();
    let mut q = vec![BlockQ8K::default(); 2];
    quantize_q8k(&x, &mut q);
    for (b, blk) in q.iter().enumerate() {
        for i in 0..QK_K {
            let back = blk.qs[i] as f32 * blk.d;
            assert!((back - x[b * QK_K + i]).abs() <= blk.d.abs() * 0.51 + 1e-6);
        }
        for g in 0..16 {
            let s: i32 = blk.qs[g * 16..g * 16 + 16].iter().map(|&v| v as i32).sum();
            assert_eq!(s, blk.bsums[g] as i32);
        }
    }
}

// ---------------------------------------------------------------- M3.2 prefill kernels

/// `nrows_act` activation rows laid out as llama.cpp's forward_mul_mat wdata: the first
/// nrows_act - nrows_act % 4 rows as 4-row interleaved groups, the rest as from_float rows.
fn mat_act(at: ActType, x: &[f32], nrows_act: usize, k: usize) -> Vec<u8> {
    let rb = at.bytes(k);
    let mut out = vec![0u8; nrows_act * rb];
    let full = nrows_act - nrows_act % 4;
    for g in (0..full).step_by(4) {
        unsafe { quantize_mat4_raw(at, x[g * k..].as_ptr(), out[g * rb..].as_mut_ptr(), k) };
    }
    for r in full..nrows_act {
        unsafe { quantize_row_raw(at, x[r * k..].as_ptr(), out[r * rb..].as_mut_ptr(), k) };
    }
    out
}

fn check_gemm(ty: QType, ncols: usize, k: usize, nact: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let rows = random_rows(ty, ncols, k, &mut rng);
    let mut packed = vec![0u8; rows.len()];
    repack(ty, &rows, &mut packed, ncols, k);
    let at = ty.act();
    let x: Vec<f32> = (0..nact * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let act = mat_act(at, &x, nact, k);
    let mut neon = vec![0f32; nact * ncols];
    let mut generic = vec![0f32; nact * ncols];
    unsafe {
        gemm_raw(ty, k, neon.as_mut_ptr(), ncols, packed.as_ptr(), act.as_ptr(), nact - nact % 4, ncols);
        gemm_generic_raw(ty, k, generic.as_mut_ptr(), ncols, packed.as_ptr(), act.as_ptr(), nact - nact % 4, ncols);
    }
    // per-row reference: the same rows quantised with from_float, gemv
    let mut gv = vec![0f32; nact * ncols];
    for r in 0..nact {
        let mut qa = vec![0u8; at.bytes(k)];
        quantize_act(at, &x[r * k..(r + 1) * k], &mut qa);
        gemv(ty, k, &mut gv[r * ncols..(r + 1) * ncols], &packed, &qa);
    }
    let scale = generic.iter().fold(0f32, |m, v| m.max(v.abs()));
    for i in 0..(nact - nact % 4) * ncols {
        assert!(close(neon[i], generic[i], scale, 1e-5), "{ty:?} gemm vs generic at {i}: {} {}", neon[i], generic[i]);
        assert!(close(neon[i], gv[i], scale, 1e-4), "{ty:?} gemm vs gemv at {i}: {} {}", neon[i], gv[i]);
    }
    // MUL_MAT_ID gemm: from_float rows interleaved into tiles must equal gemv bit for bit
    // for every type the models use (llama.cpp's GGML_CPU_MOE_GEMM "exact" list).
    let rb = at.bytes(k);
    let mut rowq = vec![0u8; nact * rb];
    for r in 0..nact {
        quantize_act(at, &x[r * k..(r + 1) * k], &mut rowq[r * rb..(r + 1) * rb]);
    }
    let mut tile = vec![0u8; 4 * rb];
    let mut out = vec![0f32; 4 * ncols];
    for g in (0..nact - nact % 4).step_by(4) {
        let rp: [*const u8; 4] = std::array::from_fn(|j| rowq[(g + j) * rb..].as_ptr());
        unsafe {
            interleave_rows4_raw(at, &rp, tile.as_mut_ptr(), k);
            gemm_mmid_raw(ty, k, out.as_mut_ptr(), ncols, packed.as_ptr(), tile.as_ptr(), 4, ncols);
        }
        for j in 0..4 {
            for c in 0..ncols {
                assert_eq!(out[j * ncols + c].to_bits(), gv[(g + j) * ncols + c].to_bits(), "{ty:?} mmid gemm != gemv row {} col {c}", g + j);
            }
        }
    }
    // remainder rows padded into a tile with copies of the last row (prefill.rs mmid_unit)
    for rem in 1..4usize.min(nact) {
        let rp: [*const u8; 4] = std::array::from_fn(|j| rowq[j.min(rem - 1) * rb..].as_ptr());
        unsafe {
            interleave_rows4_raw(at, &rp, tile.as_mut_ptr(), k);
            gemm_mmid_raw(ty, k, out.as_mut_ptr(), ncols, packed.as_ptr(), tile.as_ptr(), 4, ncols);
        }
        for j in 0..rem {
            for c in 0..ncols {
                assert_eq!(out[j * ncols + c].to_bits(), gv[j * ncols + c].to_bits(), "{ty:?} padded tile row {j} col {c}");
            }
        }
    }
}

#[test]
fn gemm_kernels_vs_generic_and_gemv() {
    assert_eq!(unsafe { ffi::oxe_sizeof_block_q8_Kx4() }, 4 * Q8K_BLOCK_BYTES);
    for (i, ty) in QType::ALL.into_iter().enumerate() {
        check_gemm(ty, 32, 512, 9, 100 + i as u64);
        check_gemm(ty, 16, 256, 4, 200 + i as u64);
    }
}

#[test]
fn quantize_mat4_matches_rows() {
    // the interleaved 4-row quantisers produce the same per-row values as from_float
    let mut rng = Rng::new(7);
    for at in [ActType::Q8K, ActType::Q8_0] {
        let k = 512;
        let x: Vec<f32> = (0..4 * k).map(|_| rng.f32() * 4.0 - 2.0).collect();
        let rb = at.bytes(k);
        let mut tile = vec![0u8; 4 * rb];
        unsafe { quantize_mat4_raw(at, x.as_ptr(), tile.as_mut_ptr(), k) };
        let mut rows = vec![0u8; 4 * rb];
        for r in 0..4 {
            quantize_act(at, &x[r * k..(r + 1) * k], &mut rows[r * rb..(r + 1) * rb]);
        }
        let rp: [*const u8; 4] = std::array::from_fn(|j| rows[j * rb..].as_ptr());
        let mut tile2 = vec![0u8; 4 * rb];
        unsafe { interleave_rows4_raw(at, &rp, tile2.as_mut_ptr(), k) };
        assert!(tile == tile2, "{at:?}: quantize_mat_4x8 differs from interleaved from_float rows");
    }
}

// ------------------------------------------------------------------ M3-ATTN kernels

fn h16(v: f32) -> u16 {
    let mut h = [0u16];
    ops::fp32_to_fp16(&[v], &mut h);
    h[0]
}

fn padded_n_kv(pos: usize, n_ctx: usize) -> usize {
    n_ctx.min(256usize.max((pos + 1).div_ceil(256) * 256))
}

/// The tiled K^T q / softmax-row / tiled V^T p path equals `oxe_attn_head_f16` bit for bit
/// (scores, f16 probabilities and the output) for every position class: below / at / across
/// 4-, 32- and 256-boundaries, at the end of the context.
#[test]
fn attn_tiled_matches_attn_head() {
    let (hd, n_ctx, nh) = (256usize, 2048usize, 8usize);
    let mut rng = Rng::new(77);
    let mut sym = |s: f32| (rng.f32() * 2.0 - 1.0) * s;
    let k: Vec<u16> = (0..n_ctx * hd).map(|_| h16(sym(2.0))).collect();
    let vt: Vec<u16> = (0..hd * n_ctx).map(|_| h16(sym(1.5))).collect();
    let q: Vec<f32> = (0..nh * hd).map(|_| sym(1.0)).collect();
    let scale = 1.0 / (hd as f32).sqrt();
    let mut q16 = vec![0u16; nh * hd];
    ops::fp32_to_fp16(&q, &mut q16);
    for &pos in &[0usize, 1, 2, 3, 4, 30, 31, 32, 33, 254, 255, 256, 257, 300, 511, 1000, 1023, 1500, 2046, 2047] {
        let n_kv = padded_n_kv(pos, n_ctx);
        let n32 = (pos + 1).div_ceil(32) * 32;
        // reference: one head at a time over the padded n_kv
        let mut ref_out = vec![0f32; nh * hd];
        let mut ref_p16 = vec![0u16; nh * n_ctx];
        for h in 0..nh {
            let (mut tq, mut kq, mut mask, mut wp) = (vec![0u16; hd], vec![0f32; n_ctx], vec![0f32; n_ctx], vec![0f32; n_ctx]);
            unsafe {
                ffi::oxe_attn_head_f16(
                    q[h * hd..].as_ptr(), k.as_ptr(), hd as i64, k[pos * hd..].as_ptr(), vt.as_ptr(), n_ctx as i64,
                    n_kv as i32, pos as i32, hd as i32, scale, tq.as_mut_ptr(), kq.as_mut_ptr(), mask.as_mut_ptr(),
                    wp.as_mut_ptr(), ref_p16[h * n_ctx..].as_mut_ptr(), ref_out[h * hd..].as_mut_ptr(),
                );
            }
        }
        // tiled: scores of rows 0..=pos x 8 heads, one softmax row per head, then V^T rows x 8 heads
        let mut kq = vec![0f32; nh * n_ctx];
        let mut p16 = vec![0xFFFFu16; nh * n_ctx];
        let mut out = vec![0f32; nh * hd];
        let (mut wp, mut dp) = (vec![0f32; n_ctx], vec![0f32; n_ctx]);
        let nf = vec![n_kv as i32; nh];
        unsafe {
            ffi::oxe_dot_f16_tile(hd as i32, k.as_ptr(), hd as i64, (pos + 1) as i32, q16.as_ptr(), hd as i64, nh as i32, kq.as_mut_ptr(), 1, n_ctx as i64, std::ptr::null());
            for h in 0..nh {
                ffi::oxe_attn_softmax_row_f16(kq[h * n_ctx..].as_ptr(), pos as i32, n_kv as i32, scale, wp.as_mut_ptr(), dp.as_mut_ptr(), p16[h * n_ctx..].as_mut_ptr(), n_kv as i32);
            }
            ffi::oxe_dot_f16_tile(n32 as i32, vt.as_ptr(), n_ctx as i64, hd as i32, p16.as_ptr(), n_ctx as i64, nh as i32, out.as_mut_ptr(), 1, hd as i64, nf.as_ptr());
        }
        for h in 0..nh {
            assert_eq!(&p16[h * n_ctx..h * n_ctx + n_kv], &ref_p16[h * n_ctx..h * n_ctx + n_kv], "p16 pos {pos} head {h}");
        }
        let a: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        let b: Vec<u32> = ref_out.iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "attention output at pos {pos}");
    }
}

/// Element-wise: every tile shape (1-4 rows x 1-70 columns, multi-block n) equals ggml_vec_dot_f16.
#[test]
fn dot_f16_tile_matches_vec_dot() {
    let mut rng = Rng::new(5);
    for &(n, nx, ny) in &[(32usize, 1usize, 1usize), (256, 4, 8), (288, 3, 5), (544, 7, 70), (1024, 9, 3), (40, 2, 3)] {
        let x: Vec<u16> = (0..n * nx).map(|_| h16(rng.f32() * 4.0 - 2.0)).collect();
        let y: Vec<u16> = (0..n * ny).map(|_| h16(rng.f32() * 4.0 - 2.0)).collect();
        let mut out = vec![0f32; nx * ny];
        unsafe { ffi::oxe_dot_f16_tile(n as i32, x.as_ptr(), n as i64, nx as i32, y.as_ptr(), n as i64, ny as i32, out.as_mut_ptr(), ny as i64, 1, std::ptr::null()) };
        for r in 0..nx {
            for c in 0..ny {
                let e = unsafe { ffi::oxe_vec_dot_f16(n as i32, x[r * n..].as_ptr(), y[c * n..].as_ptr()) };
                assert_eq!(out[r * ny + c].to_bits(), e.to_bits(), "n {n} r {r} c {c}");
            }
        }
    }
}

/// Tail rule: stopping at n < n_full (y = +0 beyond n) equals the full-length dot, including
/// accumulator lanes that hold -0 at step n (tiny negative products rounding to -0) followed by
/// positive, zero or negative x in the tail.
#[test]
fn dot_f16_tile_tail_rule() {
    let (n, n_full) = (64usize, 256usize);
    let tiny = 0x0001u16; // smallest f16 subnormal, 2^-24
    for (case, tail_x) in [h16(0.5), 0u16, h16(-0.75), 0x8000u16].into_iter().enumerate() {
        let mut x = vec![h16(-0.3); n_full];
        for v in x[n..].iter_mut() {
            *v = tail_x;
        }
        // column 0: every product rounds to -0 (all lanes -0); column 1: mixed lanes;
        // column 2: ordinary values
        let mut y = vec![0u16; 3 * n_full];
        for i in 0..n {
            y[i] = tiny;
            y[n_full + i] = if i % 3 == 0 { tiny } else { h16(0.25) };
            y[2 * n_full + i] = h16(0.125 * (i as f32 % 7.0) - 0.3);
        }
        x[5] = h16(0.0);
        let mut out = vec![0f32; 3];
        let nf = [n_full as i32; 3];
        unsafe { ffi::oxe_dot_f16_tile(n as i32, x.as_ptr(), 0, 1, y.as_ptr(), n_full as i64, 3, out.as_mut_ptr(), 0, 1, nf.as_ptr()) };
        for c in 0..3 {
            let e = unsafe { ffi::oxe_vec_dot_f16(n_full as i32, x.as_ptr(), y[c * n_full..].as_ptr()) };
            assert_eq!(out[c].to_bits(), e.to_bits(), "case {case} column {c}: {} vs {}", out[c], e);
        }
        // moe-prefill: the position-blocked kernel follows the same tail rule
        let mut out_pb = vec![0f32; 3];
        let mut st = vec![0u128; 3 * 4];
        unsafe { ffi::oxe_dot_f16_tile_pb(n as i32, x.as_ptr(), 0, 1, y.as_ptr(), n_full as i64, 3, out_pb.as_mut_ptr(), 0, 1, nf.as_ptr(), 32, st.as_mut_ptr() as *mut u8) };
        for c in 0..3 {
            assert_eq!(out_pb[c].to_bits(), out[c].to_bits(), "pb case {case} column {c}");
        }
        if case == 0 {
            // the check matters: without the tail the all -0 column would come out as -0.0
            let mut o2 = [0f32; 1];
            unsafe { ffi::oxe_dot_f16_tile(n as i32, x.as_ptr(), 0, 1, y.as_ptr(), 0, 1, o2.as_mut_ptr(), 0, 0, std::ptr::null()) };
            assert_eq!(o2[0].to_bits(), 0x8000_0000);
            assert_eq!(out[0].to_bits(), 0);
        }
    }
}

// ---------------------------------------------------------------- moe-prefill tile kernels

/// Activation rows for the tile tests: random rows plus the extremes (all +max, all -max, zero,
/// alternating) so the bsum high/low byte split sees +4064, -4064 and 0.
fn tile_x(nact: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x: Vec<f32> = (0..nact * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
    for r in 0..nact {
        let row = &mut x[r * k..(r + 1) * k];
        match r % 7 {
            1 => row.fill(1.0),
            2 => row.fill(-1.0),
            3 => row.fill(0.0),
            4 => row.iter_mut().enumerate().for_each(|(i, v)| *v = if (i / 32) % 2 == 0 { -1.0 } else { 1.0 }),
            _ => {}
        }
    }
    x
}

/// The multi-token tile GEMM must equal the vendored reference GEMM bit for bit, for token
/// counts and row counts that cross the kernel's token block / row chunk and leave an odd
/// 4-token group.
#[test]
fn pf_tile_dense_bitwise() {
    for ty in [QType::Q4K, QType::Q6K] {
        assert!(gemm_tile_has(ty), "{ty:?}: no tile kernel in this build");
        let mut seed = 7000u64;
        for &(ncols, k, nact) in &[(8usize, 256usize, 4usize), (8, 256, 8), (16, 512, 12), (72, 512, 68), (136, 2048, 132), (64, 2048, 256), (200, 768, 60)] {
            seed += 1;
            let mut rng = Rng::new(seed);
            let mut rows = random_rows(ty, ncols, k, &mut rng);
            // extreme sub-block scales / mins and quants in the first rows
            let rbw = ty.row_bytes(k);
            // (keep the f16 d fields: Q4_K bytes 0..4, Q6_K bytes 208..210)
            let (lo, hi) = if ty == QType::Q6K { (0, 208) } else { (4, 144) };
            rows[lo..hi].fill(0xff);
            if ncols > 8 {
                rows[rbw + lo..rbw + hi].fill(0x00);
            }
            let mut packed = vec![0u8; rows.len()];
            repack(ty, &rows, &mut packed, ncols, k);
            let x = tile_x(nact, k, &mut rng);
            let act = mat_act(ty.act(), &x, nact, k);
            let bs = ncols + 3; // a stride larger than the row
            let mut want = vec![f32::NAN; nact * bs];
            let mut got = vec![f32::NAN; nact * bs];
            unsafe {
                gemm_raw(ty, k, want.as_mut_ptr(), bs, packed.as_ptr(), act.as_ptr(), nact, ncols);
                gemm_tile_raw(ty, k, got.as_mut_ptr(), bs, packed.as_ptr(), act.as_ptr(), nact, ncols);
            }
            let mut nonzero = 0;
            for t in 0..nact {
                for c in 0..ncols {
                    let (a, b) = (want[t * bs + c], got[t * bs + c]);
                    assert!(a.is_finite(), "{ty:?} reference not finite at token {t} col {c}");
                    assert_eq!(a.to_bits(), b.to_bits(), "{ty:?} tile != gemm: shape ({ncols}, {k}, {nact}) token {t} col {c}: {a} {b}");
                    nonzero += (a != 0.0) as usize;
                }
                for c in ncols..bs {
                    assert!(got[t * bs + c].is_nan(), "{ty:?} tile wrote past the row at token {t}");
                }
            }
            assert!(nonzero > nact * ncols / 2);
        }
    }
}

/// Model-free kernel timing (DEV windows only; not authoritative):
/// `taskset -c 4 cargo test --release -p kernels-sys -- --ignored kbench --nocapture`.
/// Prints GMAC/s of one thread for the reference kernels and the tiles on prefill shapes.
#[test]
#[ignore]
fn kbench() {
    use std::time::Instant;
    let best = |f: &mut dyn FnMut()| -> f64 {
        f();
        let mut b = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            f();
            b = b.min(t.elapsed().as_secs_f64());
        }
        b
    };
    println!("{:<34} {:>6} {:>5} {:>5} {:>9} {:>9}", "kernel", "k", "rows", "toks", "ms", "GMAC/s");
    // dense shapes: one 64-row unit (prefill.rs CU) x a 256-token ubatch, and a 512-row slab
    for ty in QType::ALL {
        for &(ncols, k, nact) in &[(64usize, 2048usize, 256usize), (512, 2048, 256), (64, 4096, 256), (512, 2048, 64)] {
            let mut rng = Rng::new(1);
            let rows = random_rows(ty, ncols, k, &mut rng);
            let mut packed = vec![0u8; rows.len()];
            repack(ty, &rows, &mut packed, ncols, k);
            let x: Vec<f32> = (0..nact * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let act = mat_act(ty.act(), &x, nact, k);
            let mut out = vec![0f32; nact * ncols];
            let mac = (ncols * k * nact) as f64;
            let reps = (2.0e9 / mac).ceil() as usize; // ~2 GMAC per timed sample
            let t = best(&mut || unsafe {
                for _ in 0..reps {
                    gemm_raw(ty, k, out.as_mut_ptr(), ncols, packed.as_ptr(), act.as_ptr(), nact, ncols)
                }
            }) / reps as f64;
            println!("{:<34} {:>6} {:>5} {:>5} {:>9.3} {:>9.1}", format!("gemm {}", ty.name()), k, ncols, nact, t * 1e3, mac / t / 1e9);
            if gemm_tile_has(ty) {
                let mut out2 = vec![0f32; nact * ncols];
                let t2 = best(&mut || unsafe {
                    for _ in 0..reps {
                        gemm_tile_raw(ty, k, out2.as_mut_ptr(), ncols, packed.as_ptr(), act.as_ptr(), nact, ncols)
                    }
                }) / reps as f64;
                assert!(out.iter().zip(&out2).all(|(a, b)| a.to_bits() == b.to_bits()));
                println!("{:<34} {:>6} {:>5} {:>5} {:>9.3} {:>9.1}  x{:.2}", format!("tile {}", ty.name()), k, ncols, nact, t2 * 1e3, mac / t2 / 1e9, t / t2);
            }
        }
    }
    // MUL_MAT_ID shapes: one expert (gate/up: k 2048 x 512 rows; down: k 512 x 2048 rows), `cnt`
    // routed rows in 4-row tiles incl. the interleave (prefill.rs mmid_unit), and the GEMV
    for ty in QType::ALL {
        for &(ncols, k) in &[(512usize, 2048usize), (2048, 512)] {
            let mut rng = Rng::new(2);
            let rows = random_rows(ty, ncols, k, &mut rng);
            let mut packed = vec![0u8; rows.len()];
            repack(ty, &rows, &mut packed, ncols, k);
            let at = ty.act();
            let rb = at.bytes(k);
            let cnt = 8usize;
            let mut rowq = vec![0u8; cnt * rb];
            for r in 0..cnt {
                let x: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect();
                quantize_act(at, &x, &mut rowq[r * rb..(r + 1) * rb]);
            }
            let mut tile = vec![0u8; 4 * rb];
            let mut out = vec![0f32; cnt * ncols];
            let mac = (ncols * k * cnt) as f64;
            let reps = (5.0e8 / mac).ceil() as usize;
            let t = best(&mut || unsafe {
                for _ in 0..reps {
                    for g in (0..cnt).step_by(4) {
                        let rp: [*const u8; 4] = std::array::from_fn(|j| rowq[(g + j) * rb..].as_ptr());
                        interleave_rows4_raw(at, &rp, tile.as_mut_ptr(), k);
                        let mut cc = 0;
                        while cc < ncols {
                            let nc = 256.min(ncols - cc);
                            gemm_mmid_raw(ty, k, out.as_mut_ptr().add(g * ncols + cc), ncols, packed.as_ptr().add(ty.group_offset(cc, k)), tile.as_ptr(), 4, nc);
                            cc += nc;
                        }
                    }
                }
            }) / reps as f64;
            println!("{:<34} {:>6} {:>5} {:>5} {:>9.3} {:>9.1}", format!("mmid 4-row tiles {}", ty.name()), k, ncols, cnt, t * 1e3, mac / t / 1e9);
            let tv = best(&mut || unsafe {
                for _ in 0..reps {
                    for r in 0..cnt {
                        gemv_raw(ty, k, out.as_mut_ptr().add(r * ncols), packed.as_ptr(), rowq.as_ptr().add(r * rb), ncols);
                    }
                }
            }) / reps as f64;
            println!("{:<34} {:>6} {:>5} {:>5} {:>9.3} {:>9.1}", format!("gemv {}", ty.name()), k, ncols, cnt, tv * 1e3, mac / tv / 1e9);
        }
    }
}

/// Model-free timing of the prefill attention tile on long-prefix shapes (DEV windows only):
/// `taskset -c 4 cargo test --release -p kernels-sys -- --ignored kbench_attn --nocapture`.
/// One (KV head, 4 tokens) unit = 32 query columns: scores K[P x 256] . q[32 x 256] and outputs
/// V^T[256 x P] . p[32 x P], with the scratch column stride of the engine (n_ctx) and a padded
/// stride (tests L1 set aliasing of the 32 column streams).
#[test]
#[ignore]
fn kbench_attn() {
    use std::time::Instant;
    let hd = 256usize;
    // KB_NCOL=64: 8 tokens per unit (PF_ATTN_TT8)
    let ncol = std::env::var("KB_NCOL").ok().and_then(|v| v.parse().ok()).unwrap_or(32usize);
    println!("{:<44} {:>6} {:>7} {:>9} {:>9}", "kernel", "P", "n_ctx", "ms", "GFMA/s");
    for &(p, n_ctx) in &[(2048usize, 16384usize), (8192, 16384), (8192, 8448), (14336, 16384), (14336, 32768)] {
        let mut rng = Rng::new(3);
        let mut h = |n: usize| -> Vec<u16> {
            let x: Vec<f32> = (0..n).map(|_| rng.f32() * 0.5 - 0.25).collect();
            let mut y = vec![0u16; n];
            f32_to_f16(&x, &mut y);
            y
        };
        let kc = h(p * hd); // K rows [pos][256]
        // KB_VPAD=32: V^T rows one cache line off the n_ctx stride (OXE_KV_VPAD)
        let vs = n_ctx + std::env::var("KB_VPAD").ok().and_then(|v| v.parse().ok()).unwrap_or(0usize);
        let vt = h(hd * vs); // V^T rows [dim][vs]
        let q = h(ncol * hd);
        for &pad in &[0usize, 104] {
            let ys = n_ctx + pad;
            let pr = h(ncol * ys);
            let mut kq = vec![0f32; ncol * ys];
            let mut out = vec![0f32; hd * ncol];
            let mut best = [f64::MAX; 2];
            for _ in 0..4 {
                let t = Instant::now();
                unsafe { ffi::oxe_dot_f16_tile(hd as c_int, kc.as_ptr(), hd as i64, p as c_int, q.as_ptr(), hd as i64, ncol as c_int, kq.as_mut_ptr(), 1, ys as i64, std::ptr::null()) };
                best[0] = best[0].min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                unsafe { ffi::oxe_dot_f16_tile(p as c_int, vt.as_ptr(), vs as i64, hd as c_int, pr.as_ptr(), ys as i64, ncol as c_int, out.as_mut_ptr(), ncol as i64, 1, std::ptr::null()) };
                best[1] = best[1].min(t.elapsed().as_secs_f64());
            }
            let mut st = vec![0u128; hd * ncol * 4];
            let mut out2 = vec![0f32; hd * ncol];
            for &pb in &[256usize, 512, 1024, 2048] {
                let mut bt = f64::MAX;
                for _ in 0..4 {
                    let t = Instant::now();
                    unsafe { ffi::oxe_dot_f16_tile_pb(p as c_int, vt.as_ptr(), vs as i64, hd as c_int, pr.as_ptr(), ys as i64, ncol as c_int, out2.as_mut_ptr(), ncol as i64, 1, std::ptr::null(), pb as c_int, st.as_mut_ptr() as *mut u8) };
                    bt = bt.min(t.elapsed().as_secs_f64());
                }
                assert!(out.iter().zip(&out2).all(|(a, b)| a.to_bits() == b.to_bits()));
                println!("{:<44} {:>6} {:>7} {:>9.3} {:>9.2}", format!("outputs V^T.p pb {pb} (stride n_ctx+{pad})"), p, n_ctx, bt * 1e3, (p * hd * ncol) as f64 / bt / 1e9);
            }
            let fma = (p * hd * ncol) as f64;
            println!("{:<44} {:>6} {:>7} {:>9.3} {:>9.2}", format!("scores K.q (col stride n_ctx+{pad})"), p, n_ctx, best[0] * 1e3, fma / best[0] / 1e9);
            println!("{:<44} {:>6} {:>7} {:>9.3} {:>9.2}", format!("outputs V^T.p (col stride n_ctx+{pad})"), p, n_ctx, best[1] * 1e3, fma / best[1] / 1e9);
        }
    }
}

/// The position-blocked f16 dot tile must equal oxe_dot_f16_tile bit for bit (random data, odd
/// row / column counts, several block sizes, with and without the tail rule).
#[test]
fn pf_dot_f16_tile_pb_bitwise() {
    let mut rng = Rng::new(77);
    for &(n, nx, ny, pb) in &[(32usize, 1usize, 1usize, 32usize), (96, 5, 7, 32), (1024, 9, 32, 256), (2048, 13, 33, 512), (1504, 6, 64, 1024), (512, 3, 2, 4096)] {
        let n_full = n + 64;
        let (xs, ys) = (n_full + 8, n_full + 40);
        let mut gen = |len: usize| -> Vec<u16> {
            let x: Vec<f32> = (0..len).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let mut y = vec![0u16; len];
            f32_to_f16(&x, &mut y);
            y
        };
        let x = gen(nx * xs);
        let mut y = gen(ny * ys);
        for c in 0..ny {
            y[c * ys + n..c * ys + n_full].fill(0); // +0 beyond n (masked probabilities)
            if c % 3 == 1 {
                y[c * ys..c * ys + n].fill(0x0001); // tiny: products round to +-0
            }
        }
        for tail in [false, true] {
            let nf: Vec<i32> = (0..ny).map(|c| if c % 2 == 0 { n_full as i32 } else { n as i32 }).collect();
            let nfp = if tail { nf.as_ptr() } else { std::ptr::null() };
            let mut want = vec![f32::NAN; nx * ny];
            let mut got = vec![f32::NAN; nx * ny];
            let mut st = vec![0u128; nx * ny * 4];
            unsafe {
                ffi::oxe_dot_f16_tile(n as i32, x.as_ptr(), xs as i64, nx as i32, y.as_ptr(), ys as i64, ny as i32, want.as_mut_ptr(), ny as i64, 1, nfp);
                ffi::oxe_dot_f16_tile_pb(n as i32, x.as_ptr(), xs as i64, nx as i32, y.as_ptr(), ys as i64, ny as i32, got.as_mut_ptr(), ny as i64, 1, nfp, pb as i32, st.as_mut_ptr() as *mut u8);
            }
            for i in 0..nx * ny {
                assert_eq!(want[i].to_bits(), got[i].to_bits(), "shape ({n}, {nx}, {ny}, pb {pb}) tail {tail} element {i}: {} {}", want[i], got[i]);
            }
        }
    }
}

/// Routed-expert tiles: every (row, column) must equal the GEMV of that row bit for bit (which
/// is what gemm_mmid produces, see check_gemm), for 1..19 rows (pair tiles of 1-4 pairs, odd
/// counts) and column counts that cross the 64-row decode chunk.
#[test]
fn pf_tile_mmid_bitwise() {
    assert_eq!(unsafe { ffi::oxe_pf_xblk_bytes() }, PF_XBLK_BYTES);
    for ty in [QType::Q3K, QType::Q4K, QType::Q5K, QType::Q6K] {
        assert!(mmid_tile_has(ty));
        let mut seed = 9000u64;
        for &(ncols, k) in &[(8usize, 256usize), (72, 512), (136, 2048), (64, 768)] {
            seed += 1;
            let mut rng = Rng::new(seed);
            let mut rows = random_rows(ty, ncols, k, &mut rng);
            let rbw = ty.row_bytes(k);
            let bb = ty.block_bytes();
            // extreme scales / quants in row 0's first block, zeros in row 1's (keep the f16 d fields)
            let (lo, hi) = match ty {
                QType::Q3K => (0, 108),
                QType::Q6K => (0, 208),
                _ => (4, bb),
            };
            rows[lo..hi].fill(0xff);
            rows[rbw + lo..rbw + hi].fill(0x00);
            let mut packed = vec![0u8; rows.len()];
            repack(ty, &rows, &mut packed, ncols, k);
            let at = ty.act();
            let rb = at.bytes(k);
            let nmax = 19usize;
            let x = tile_x(nmax, k, &mut rng);
            let mut rowq = vec![0u8; nmax * rb];
            let mut gv = vec![0f32; nmax * ncols];
            for r in 0..nmax {
                quantize_act(at, &x[r * k..(r + 1) * k], &mut rowq[r * rb..(r + 1) * rb]);
                gemv(ty, k, &mut gv[r * ncols..(r + 1) * ncols], &packed, &rowq[r * rb..(r + 1) * rb]);
            }
            for cnt in 1..=nmax {
                // rows in a scrambled order, as an expert's routed list
                let list: Vec<usize> = (0..cnt).map(|j| (j * 7 + 3) % nmax).collect();
                let rp: Vec<*const u8> = list.iter().map(|&r| rowq[r * rb..].as_ptr()).collect();
                let mut xq = vec![0u32; cnt.div_ceil(2) * (k / QK_K) * PF_XBLK_BYTES / 4];
                let mut out = vec![f32::NAN; cnt * (ncols + 1)];
                let orow: Vec<*mut f32> = (0..cnt).map(|j| out[j * (ncols + 1)..].as_mut_ptr()).collect();
                unsafe {
                    mmid_tile_pack_raw(&rp, k, xq.as_mut_ptr() as *mut u8);
                    assert!(mmid_tile_raw(ty, k, &orow, packed.as_ptr(), xq.as_ptr() as *const u8, ncols));
                }
                for (j, &r) in list.iter().enumerate() {
                    for c in 0..ncols {
                        let (a, b) = (gv[r * ncols + c], out[j * (ncols + 1) + c]);
                        assert!(a.is_finite());
                        assert_eq!(a.to_bits(), b.to_bits(), "{ty:?} tile != gemv: ({ncols}, {k}) cnt {cnt} row {j} col {c}: {a} {b}");
                    }
                    assert!(out[j * (ncols + 1) + ncols].is_nan(), "{ty:?} wrote past row {j}");
                }
            }
        }
    }
}

/// Model-free timing of the routed-expert tiles against the 4-row gemm_mmid tiles (+ interleave)
/// for `cnt` routed rows of one expert (DEV windows only):
/// `taskset -c 4 cargo test --release -p kernels-sys -- --ignored kbench_mmid --nocapture`.
#[test]
#[ignore]
fn kbench_mmid() {
    use std::time::Instant;
    println!("{:<10} {:>6} {:>5} {:>4} {:>12} {:>12} {:>7}   (GMAC/s of the REAL rows; pack / interleave included)", "type", "k", "rows", "cnt", "mmid 4-row", "pair tiles", "x");
    for ty in [QType::Q3K, QType::Q4K, QType::Q5K, QType::Q6K] {
        for &(ncols, k) in &[(512usize, 2048usize), (2048, 512)] {
            let mut rng = Rng::new(4);
            let rows = random_rows(ty, ncols, k, &mut rng);
            let mut packed = vec![0u8; rows.len()];
            repack(ty, &rows, &mut packed, ncols, k);
            let at = ty.act();
            let rb = at.bytes(k);
            let nmax = 64usize;
            let mut rowq = vec![0u8; nmax * rb];
            for r in 0..nmax {
                let x: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect();
                quantize_act(at, &x, &mut rowq[r * rb..(r + 1) * rb]);
            }
            for &cnt in &[1usize, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64] {
                let mut tile = vec![0u8; 4 * rb];
                let mut out = vec![0f32; cnt.div_ceil(4) * 4 * ncols];
                let mut out2 = vec![0f32; cnt * ncols];
                let mut xq = vec![0u32; cnt.div_ceil(2) * (k / QK_K) * PF_XBLK_BYTES / 4];
                let mac = (ncols * k * cnt) as f64;
                let reps = (3.0e8 / mac).ceil() as usize;
                let time = |f: &mut dyn FnMut()| -> f64 {
                    f();
                    let mut b = f64::MAX;
                    for _ in 0..5 {
                        let t = Instant::now();
                        for _ in 0..reps {
                            f();
                        }
                        b = b.min(t.elapsed().as_secs_f64() / reps as f64);
                    }
                    b
                };
                // today: 4-row tiles, the leftover padded with copies of the last row (prefill.rs mmid_unit)
                let t_ref = time(&mut || unsafe {
                    for g in (0..cnt).step_by(4) {
                        let rp: [*const u8; 4] = std::array::from_fn(|j| rowq[(g + j).min(cnt - 1) * rb..].as_ptr());
                        interleave_rows4_raw(at, &rp, tile.as_mut_ptr(), k);
                        let mut cc = 0;
                        while cc < ncols {
                            let nc = 256.min(ncols - cc);
                            gemm_mmid_raw(ty, k, out.as_mut_ptr().add(g * ncols + cc), ncols, packed.as_ptr().add(ty.group_offset(cc, k)), tile.as_ptr(), 4, nc);
                            cc += nc;
                        }
                    }
                });
                let rp: Vec<*const u8> = (0..cnt).map(|r| rowq[r * rb..].as_ptr()).collect();
                let orow: Vec<*mut f32> = (0..cnt).map(|j| out2[j * ncols..].as_mut_ptr()).collect();
                let t_new = time(&mut || unsafe {
                    mmid_tile_pack_raw(&rp, k, xq.as_mut_ptr() as *mut u8);
                    mmid_tile_raw(ty, k, &orow, packed.as_ptr(), xq.as_ptr() as *const u8, ncols);
                });
                assert!(out[..cnt * ncols].iter().zip(&out2).all(|(a, b)| a.to_bits() == b.to_bits()));
                println!("{:<10} {:>6} {:>5} {:>4} {:>12.1} {:>12.1} {:>7.2}", ty.name(), k, ncols, cnt, mac / t_ref / 1e9, mac / t_new / 1e9, t_ref / t_new);
            }
        }
    }
}

/// The fused 4-row gated-delta-net step must equal the reference step bit for bit: state, delta
/// and output over a sequence of steps (the state feeds back), several head sizes.
#[test]
fn pf_gdn_step_x4_bitwise() {
    for &sv in &[16usize, 32, 48, 128] {
        let mut rng = Rng::new(500 + sv as u64);
        let mut s1: Vec<f32> = (0..sv * sv).map(|_| rng.f32() * 0.2 - 0.1).collect();
        let mut s2 = s1.clone();
        for step in 0..40 {
            let q: Vec<f32> = (0..sv).map(|_| (rng.f32() * 2.0 - 1.0) / (sv as f32).sqrt()).collect();
            let k: Vec<f32> = (0..sv).map(|_| (rng.f32() * 2.0 - 1.0) / (sv as f32).sqrt()).collect();
            let v: Vec<f32> = (0..sv).map(|_| rng.f32() * 4.0 - 2.0).collect();
            let g = -rng.f32() * 0.7;
            let beta = rng.f32();
            let (mut d1, mut d2) = (vec![0f32; sv], vec![0f32; sv]);
            let (mut a1, mut a2) = (vec![0f32; sv], vec![0f32; sv]);
            unsafe {
                ffi::oxe_gated_delta_net_step_f32(s1.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(), g, beta, sv as i64, d1.as_mut_ptr(), a1.as_mut_ptr());
                ffi::oxe_gated_delta_net_step_f32_x4(s2.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(), g, beta, sv as i64, d2.as_mut_ptr(), a2.as_mut_ptr());
            }
            let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same(&a1, &a2), "S_v {sv} step {step}: output differs");
            assert!(same(&d1, &d2), "S_v {sv} step {step}: delta differs");
            assert!(same(&s1, &s2), "S_v {sv} step {step}: state differs");
            assert!(a1.iter().all(|x| x.is_finite()) && a1.iter().any(|&x| x != 0.0));
        }
    }
}

/// Model-free timing of the delta-net step (128 x 128 state in L1, DEV windows only):
/// `taskset -c 4 cargo test --release -p kernels-sys -- --ignored kbench_gdn --nocapture`.
#[test]
#[ignore]
fn kbench_gdn() {
    use std::time::Instant;
    let sv = 128usize;
    let mut rng = Rng::new(9);
    let q: Vec<f32> = (0..sv).map(|_| (rng.f32() * 2.0 - 1.0) / 11.3).collect();
    let k: Vec<f32> = (0..sv).map(|_| (rng.f32() * 2.0 - 1.0) / 11.3).collect();
    let v: Vec<f32> = (0..sv).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let (mut d, mut a) = (vec![0f32; sv], vec![0f32; sv]);
    for x4 in [false, true] {
        let mut s: Vec<f32> = (0..sv * sv).map(|_| rng.f32() * 0.2 - 0.1).collect();
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..2000 {
                unsafe {
                    if x4 {
                        ffi::oxe_gated_delta_net_step_f32_x4(s.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(), -0.05, 0.5, sv as i64, d.as_mut_ptr(), a.as_mut_ptr());
                    } else {
                        ffi::oxe_gated_delta_net_step_f32(s.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(), -0.05, 0.5, sv as i64, d.as_mut_ptr(), a.as_mut_ptr());
                    }
                }
            }
            best = best.min(t.elapsed().as_secs_f64() / 2000.0);
        }
        println!("gdn step {:<10} {:>8.2} us per (token, head)", if x4 { "x4 fused" } else { "reference" }, best * 1e6);
    }
}

/// Model-free timing of the two address-only fixes of the aliasing sweep at production shapes
/// (DEV windows only): `taskset -c 4 cargo test --release -p kernels-sys -- --ignored kbench_alias --nocapture`.
/// (a) the prefill V^T cache write of a thread's 22 tokens (512 rows of f16): token-outer vs
/// row-outer, row stride n_ctx vs n_ctx + 32; (b) the MoE weighted sum of prefill.rs step [7]
/// (8 expert rows + shared + residual -> x, n_embd 2048, 256 tokens) with the expert rows at a
/// stride of n_embd vs n_embd + 16 floats.
#[test]
#[ignore]
fn kbench_alias() {
    use std::time::Instant;
    let best = |f: &mut dyn FnMut()| -> f64 {
        f();
        let mut b = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            f();
            b = b.min(t.elapsed().as_secs_f64());
        }
        b
    };
    // (a) V write: 256 tokens of an ubatch, this thread owns 22 consecutive ones per layer; 10 layers
    let (kvw, ntok, layers) = (512usize, 22usize, 10usize);
    let vr: Vec<f32> = (0..ntok * kvw).map(|i| (i % 97) as f32 * 0.01 - 0.4).collect();
    for &n_ctx in &[16384usize, 32768] {
        for &pad in &[0usize, 32] {
            let vs = n_ctx + pad;
            let mut vc = vec![0u16; layers * kvw * vs];
            for row_outer in [false, true] {
                let t = best(&mut || {
                    for start in (4096..4096 + 2048).step_by(256) {
                        for l in 0..layers {
                            let base = l * kvw * vs;
                            let mut h = [0u16; 1];
                            if row_outer {
                                for r in 0..kvw {
                                    for i in 0..ntok {
                                        f32_to_f16(&[vr[i * kvw + r]], &mut h);
                                        vc[base + r * vs + start + i] = h[0];
                                    }
                                }
                            } else {
                                for i in 0..ntok {
                                    for r in 0..kvw {
                                        f32_to_f16(&[vr[i * kvw + r]], &mut h);
                                        vc[base + r * vs + start + i] = h[0];
                                    }
                                }
                            }
                        }
                    }
                });
                let per_tok = t / (8 * ntok) as f64;
                println!("V write   n_ctx {:>5} pad {:>2} {:<11} {:>8.2} us per token (10 layers)", n_ctx, pad, if row_outer { "row-outer" } else { "token-outer" }, per_tok * 1e6);
            }
        }
    }
    // (b) weighted sum
    let (de, ku, n) = (2048usize, 8usize, 256usize);
    let w: Vec<f32> = (0..n * ku).map(|i| 0.05 + (i % 7) as f32 * 0.02).collect();
    for &pad in &[0usize, 16] {
        let ds = de + pad;
        let dn: Vec<f32> = (0..n * ku * ds).map(|i| (i % 113) as f32 * 0.01 - 0.5).collect();
        let sdn: Vec<f32> = (0..n * de).map(|i| (i % 89) as f32 * 0.01).collect();
        let x1: Vec<f32> = (0..n * de).map(|i| (i % 61) as f32 * 0.02).collect();
        let mut x = vec![0f32; n * de];
        let t = best(&mut || {
            for t in 0..n {
                let d = &dn[t * ku * ds..(t + 1) * ku * ds];
                let wt = &w[t * ku..(t + 1) * ku];
                let (s, r, o) = (&sdn[t * de..(t + 1) * de], &x1[t * de..(t + 1) * de], &mut x[t * de..(t + 1) * de]);
                for c in 0..de {
                    let mut m = d[c] * wt[0];
                    for j in 1..ku {
                        m += d[j * ds + c] * wt[j];
                    }
                    o[c] = (m + s[c] * 0.5) + r[c];
                }
            }
        });
        println!("MoE sum   expert row stride n_embd + {:>2}: {:>8.2} us per token per layer (x 40 layers = {:.2} thread-ms per token)", pad, t / n as f64 * 1e6, t / n as f64 * 40.0 * 1e3);
    }
}

/// Routed experts as the prefill meets them: MANY experts, each used once per pass (weights come
/// from DRAM, not from L2 as in `kbench_mmid`, which reuses one expert). 96 experts of 512 x 2048
/// (gate-sized; ~43-84 MB of weights), `cnt` routed rows each, one pass over all experts per sample.
/// `taskset -c 4 cargo test --release -p kernels-sys -- --ignored kbench_mmid_cold --nocapture`
#[test]
#[ignore]
fn kbench_mmid_cold() {
    use std::time::Instant;
    let (ncols, k, ne) = (512usize, 2048usize, 96usize);
    println!("{:<6} {:>4} {:>14} {:>14} {:>7} {:>16}   GMAC/s of real rows, weights cold (96 experts per pass)", "type", "cnt", "4-row tiles", "pair tiles", "x", "hot pair tiles");
    for ty in [QType::Q3K, QType::Q4K, QType::Q5K, QType::Q6K] {
        let mut rng = Rng::new(11);
        let one = random_rows(ty, ncols, k, &mut rng);
        let ebytes = one.len();
        let mut packed = vec![0u8; ne * ebytes];
        for e in 0..ne {
            // distinct memory per expert (same values: the timing does not depend on them)
            repack(ty, &one, &mut packed[e * ebytes..(e + 1) * ebytes], ncols, k);
        }
        let at = ty.act();
        let rb = at.bytes(k);
        let nmax = 32usize;
        let mut rowq = vec![0u8; nmax * rb];
        for r in 0..nmax {
            let x: Vec<f32> = (0..k).map(|_| rng.f32() * 2.0 - 1.0).collect();
            quantize_act(at, &x, &mut rowq[r * rb..(r + 1) * rb]);
        }
        for &cnt in &[2usize, 4, 8, 16, 32] {
            let mut tile = vec![0u8; 4 * rb];
            let mut out = vec![0f32; cnt.div_ceil(4) * 4 * ncols];
            let mut out2 = vec![0f32; cnt * ncols];
            let mut xq = vec![0u32; cnt.div_ceil(2) * (k / QK_K) * PF_XBLK_BYTES / 4];
            let mac = (ncols * k * cnt * ne) as f64;
            let best = |f: &mut dyn FnMut()| -> f64 {
                f();
                let mut b = f64::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    f();
                    b = b.min(t.elapsed().as_secs_f64());
                }
                b
            };
            let t_ref = best(&mut || unsafe {
                for e in 0..ne {
                    let w = packed.as_ptr().add(e * ebytes);
                    for g in (0..cnt).step_by(4) {
                        let rp: [*const u8; 4] = std::array::from_fn(|j| rowq[(g + j).min(cnt - 1) * rb..].as_ptr());
                        interleave_rows4_raw(at, &rp, tile.as_mut_ptr(), k);
                        let mut cc = 0;
                        while cc < ncols {
                            let nc = 256.min(ncols - cc);
                            gemm_mmid_raw(ty, k, out.as_mut_ptr().add(g * ncols + cc), ncols, w.add(ty.group_offset(cc, k)), tile.as_ptr(), 4, nc);
                            cc += nc;
                        }
                    }
                }
            });
            let rp: Vec<*const u8> = (0..cnt).map(|r| rowq[r * rb..].as_ptr()).collect();
            let orow: Vec<*mut f32> = (0..cnt).map(|j| out2[j * ncols..].as_mut_ptr()).collect();
            let t_new = best(&mut || unsafe {
                for e in 0..ne {
                    mmid_tile_pack_raw(&rp, k, xq.as_mut_ptr() as *mut u8);
                    mmid_tile_raw(ty, k, &orow, packed.as_ptr().add(e * ebytes), xq.as_ptr() as *const u8, ncols);
                }
            });
            let t_hot = best(&mut || unsafe {
                for _ in 0..ne {
                    mmid_tile_pack_raw(&rp, k, xq.as_mut_ptr() as *mut u8);
                    mmid_tile_raw(ty, k, &orow, packed.as_ptr(), xq.as_ptr() as *const u8, ncols);
                }
            });
            println!("{:<6} {:>4} {:>14.1} {:>14.1} {:>7.2} {:>16.1}", ty.name(), cnt, mac / t_ref / 1e9, mac / t_new / 1e9, t_ref / t_new, mac / t_hot / 1e9);
        }
    }
}

/// OD-6b: the verify-lane GEMVs (csrc/oxe_vl.inc) are bitwise the GEMV per lane, for every weight
/// type, 2-4 lanes, every assignment of lanes to pair slots, with extreme weight codes.
#[test]
fn vl_gemv_bitwise() {
    for ty in QType::ALL {
        assert!(gemv_lanes_has(ty), "{ty:?}: no lane GEMV in this build");
        let mut seed = 77_000u64;
        for &(ncols, k) in &[(8usize, 256usize), (72, 512), (136, 2048), (64, 768), (16, 4096)] {
            seed += 1;
            let mut rng = Rng::new(seed);
            let mut rows = random_rows(ty, ncols, k, &mut rng);
            let rbw = ty.row_bytes(k);
            let bb = ty.block_bytes();
            // extreme scales / quants in row 0's first block, zeros in row 1's (keep the f16 d fields)
            let (lo, hi) = match ty {
                QType::Q3K => (0, 108),
                QType::Q6K => (0, 208),
                QType::Q8_0 => (2, bb),
                _ => (4, bb),
            };
            rows[lo..hi].fill(0xff);
            rows[rbw + lo..rbw + hi].fill(0x00);
            let mut packed = vec![0u8; rows.len()];
            repack(ty, &rows, &mut packed, ncols, k);
            let at = ty.act();
            let rb = at.bytes(k);
            let nmax = 6usize;
            let mut x = tile_x(nmax, k, &mut rng);
            // one lane of alternating +-max values (codes +-127 everywhere: the largest integer sums)
            for (i, v) in x[..k].iter_mut().enumerate() {
                *v = if i % 2 == 0 { 3.0 } else { -3.0 };
            }
            let mut rowq = vec![0u8; nmax * rb];
            let mut gv = vec![0f32; nmax * ncols];
            for r in 0..nmax {
                quantize_act(at, &x[r * k..(r + 1) * k], &mut rowq[r * rb..(r + 1) * rb]);
                gemv(ty, k, &mut gv[r * ncols..(r + 1) * ncols], &packed, &rowq[r * rb..(r + 1) * rb]);
            }
            for nl in 1..=VL_MAX_LANES {
                for rot in 0..nmax {
                    let list: Vec<usize> = (0..nl).map(|j| (j * 5 + rot) % nmax).collect();
                    let acts: Vec<*const u8> = list.iter().map(|&r| rowq[r * rb..].as_ptr()).collect();
                    let mut out = vec![f32::NAN; nl * (ncols + 1)];
                    let outs: Vec<*mut f32> = (0..nl).map(|j| out[j * (ncols + 1)..].as_mut_ptr()).collect();
                    // SAFETY: packed holds ncols rows of k weights; every lane has rb activation bytes and ncols + 1 floats.
                    assert!(unsafe { gemv_lanes_raw(ty, k, &outs, packed.as_ptr(), &acts, ncols) });
                    for (j, &r) in list.iter().enumerate() {
                        for c in 0..ncols {
                            let (a, b) = (gv[r * ncols + c], out[j * (ncols + 1) + c]);
                            assert!(a.is_finite());
                            assert_eq!(a.to_bits(), b.to_bits(), "{ty:?} lane gemv != gemv: ({ncols}, {k}) nl {nl} rot {rot} lane {j} row {c}: {a} {b}");
                        }
                        assert!(out[j * (ncols + 1) + ncols].is_nan(), "{ty:?} wrote past lane {j}");
                    }
                }
            }
            // out of range lane counts: nothing written
            let mut o = vec![f32::NAN; 5 * ncols];
            let outs: Vec<*mut f32> = (0..5).map(|j| o[j * ncols..].as_mut_ptr()).collect();
            let acts: Vec<*const u8> = (0..5).map(|j| rowq[j * rb..].as_ptr()).collect();
            assert!(!unsafe { gemv_lanes_raw(ty, k, &outs, packed.as_ptr(), &acts, ncols) });
            assert!(o.iter().all(|v| v.is_nan()));
        }
    }
}

/// OD-6b: the out-of-place delta-net steps (`_from`: old state read from another buffer) give the
/// bits of "copy the state, then step in place", for the reference and the fused 4-row step.
#[test]
fn gdn_step_from_bitwise() {
    for &sv in &[16usize, 20, 32, 48, 128] {
        let mut rng = Rng::new(900 + sv as u64);
        let mut s_ref: Vec<f32> = (0..sv * sv).map(|_| rng.f32() * 0.2 - 0.1).collect();
        for step in 0..24 {
            let q: Vec<f32> = (0..sv).map(|_| (rng.f32() * 2.0 - 1.0) / (sv as f32).sqrt()).collect();
            let k: Vec<f32> = (0..sv).map(|_| (rng.f32() * 2.0 - 1.0) / (sv as f32).sqrt()).collect();
            let v: Vec<f32> = (0..sv).map(|_| rng.f32() * 4.0 - 2.0).collect();
            let g = -rng.f32() * 0.7;
            let beta = rng.f32();
            let old = s_ref.clone();
            let (mut d0, mut a0) = (vec![0f32; sv], vec![0f32; sv]);
            unsafe { ffi::oxe_gated_delta_net_step_f32(s_ref.as_mut_ptr(), q.as_ptr(), k.as_ptr(), v.as_ptr(), g, beta, sv as i64, d0.as_mut_ptr(), a0.as_mut_ptr()) };
            let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            for x4 in [false, true] {
                // out of place into a poisoned buffer, and "in place" through the same entry
                for inplace in [false, true] {
                    let mut dst = if inplace { old.clone() } else { vec![f32::NAN; sv * sv] };
                    let src = if inplace { dst.as_ptr() } else { old.as_ptr() };
                    let (mut d, mut a) = (vec![0f32; sv], vec![0f32; sv]);
                    let f = if x4 { ffi::oxe_gated_delta_net_step_f32_x4_from } else { ffi::oxe_gated_delta_net_step_f32_from };
                    unsafe { f(dst.as_mut_ptr(), src, q.as_ptr(), k.as_ptr(), v.as_ptr(), g, beta, sv as i64, d.as_mut_ptr(), a.as_mut_ptr()) };
                    assert!(same(&dst, &s_ref), "S_v {sv} step {step} x4 {x4} inplace {inplace}: state differs");
                    assert!(same(&a, &a0), "S_v {sv} step {step} x4 {x4} inplace {inplace}: output differs");
                    assert!(same(&d, &d0), "S_v {sv} step {step} x4 {x4} inplace {inplace}: delta differs");
                }
            }
        }
    }
}
