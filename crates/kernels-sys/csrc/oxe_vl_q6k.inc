// oxe_vl_q6k.inc: lane GEMV for Q6_K 8x8 (see oxe_vl.inc for the contract and the lane layout).
// Reference: ggml_gemv_q6_K_8x8_q8_K (dotprod branch). Per (lane, row), per 256-block the reference
// forms the exact int32 I = sum over the 16 sub-blocks i of scale_i[row] * (sum of 16 codes * q8),
// where the codes are the raw 6-bit values minus 32 (the reference applies the -32 through the q8
// bsums; here the codes are formed signed, which is the same integer), and then does, per block:
//     acc = acc + f32(I) * (d_w * d_x)      (multiply, then add: NOT fused; see the note at the update)
// starting from acc = +0. There is no min term for Q6_K.
// Lane-vector layout: for row pair q = (2q, 2q+1) and lane pair (a, b), the vector is
// [r.a, r.b, r'.a, r'.b]. The integer per-row scale vector is [s_r, s_r, s_r', s_r'].

namespace {

template <int NP>
static void oxe_vl_q6k_run(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    const int        nb      = n / QK_K;
    const uint8x16_t m4b     = vdupq_n_u8(0x0f);
    const uint8x16_t mask_lo = vdupq_n_u8(0x03);
    const uint8x16_t mask_hi = vdupq_n_u8(0x30);
    const int8x16_t  m32s    = vdupq_n_s8(32);
    const block_q8_K * ya[NP];
    const block_q8_K * yb[NP];
    float *            oa[NP];
    float *            ob[NP];
    for (int p = 0; p < NP; p++) {
        const int la = 2 * p, lb = 2 * p + 1;
        ya[p] = (const block_q8_K *) vy[la];
        yb[p] = (const block_q8_K *) vy[lb < nl ? lb : la];
        oa[p] = s[la];
        ob[p] = lb < nl ? s[lb] : nullptr;
    }
    for (int x = 0; x < nc / 8; x++) {
        const block_q6_Kx8 * w = (const block_q6_Kx8 *) vx + (size_t) x * nb;
        float32x4_t accf[NP][4];
        for (int p = 0; p < NP; p++) {
            for (int q = 0; q < 4; q++) {
                accf[p][q] = vdupq_n_f32(0);
            }
        }
        for (int b = 0; b < nb; b++) {
            const float32x4_t wd0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d));      // d of rows 0..3
            const float32x4_t wd1 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d + 4));  // rows 4..7
            const float32x4_t wdq[4] = { oxe_vl_rr_lo(wd0), oxe_vl_rr_hi(wd0), oxe_vl_rr_lo(wd1), oxe_vl_rr_hi(wd1) };
            float32x4_t sbs[NP][4];   // d_w * d_x
            int32x4_t   I[NP][4];     // the reference's int32 block sum, per (lane, row)
            for (int p = 0; p < NP; p++) {
                const float32x2_t dab = vset_lane_f32(yb[p][b].d, vdup_n_f32(ya[p][b].d), 1);
                const float32x4_t xd  = vcombine_f32(dab, dab);   // [d_a, d_b, d_a, d_b]
                for (int q = 0; q < 4; q++) {
                    sbs[p][q] = vmulq_f32(wdq[q], xd);
                    I[p][q]   = vdupq_n_s32(0);
                }
            }
            for (int half = 0; half < 2; half++) {
                const uint8_t * ql_base = w[b].ql + half * 512;
                const uint8_t * qh_base = w[b].qh + half * 256;
                for (int sb = 0; sb < QK_K / 64; sb++) {
                    const int st0    = half * 16 + sb * 2;   // first 8-weight step of the low part
                    const int qh_off = (sb * QK_K / 2) & 255;
                    // weight codes (signed, -32 applied) of the 4 vectors per column pair q:
                    // W[q][0]: step st0, W[q][1]: step st0+1 (low nibbles, +0 / +64 of ql)
                    // W[q][2]: step st0+8, W[q][3]: step st0+9 (high nibbles of the same ql bytes)
                    int8x16_t W[4][4];
                    for (int q = 0; q < 4; q++) {
                        const uint8x16_t l0  = vld1q_u8(ql_base + sb * QK_K / 2 + 16 * q);
                        const uint8x16_t l1  = vld1q_u8(ql_base + sb * QK_K / 2 + 64 + 16 * q);
                        uint8x16_t       hq0 = vld1q_u8(qh_base + qh_off + 16 * q);
                        uint8x16_t       hq1 = vld1q_u8(qh_base + qh_off + 64 + 16 * q);
                        if (sb > 1) {
                            hq0 = vshrq_n_u8(hq0, 2);
                            hq1 = vshrq_n_u8(hq1, 2);
                        }
                        // q6 = (low4 | high2 << 4) - 32, as the reference forms it (without the -32)
                        W[q][0] = vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(l0, m4b), vandq_u8(hq0, mask_lo), 4)), m32s);
                        W[q][1] = vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(l1, m4b), vandq_u8(hq1, mask_lo), 4)), m32s);
                        W[q][2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vandq_u8(hq0, mask_hi))), m32s);
                        W[q][3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vandq_u8(hq1, mask_hi))), m32s);
                    }
                    // activation codes of the 4 steps per lane pair: X[p][k] = weights 8*step .. +7
                    int8x16_t X[NP][4];
                    for (int p = 0; p < NP; p++) {
                        const int8_t * qa = ya[p][b].qs;
                        const int8_t * qb = yb[p][b].qs;
                        X[p][0] = oxe_vl_x(qa + 8 * st0, qb + 8 * st0);
                        X[p][1] = oxe_vl_x(qa + 8 * (st0 + 1), qb + 8 * (st0 + 1));
                        X[p][2] = oxe_vl_x(qa + 8 * (st0 + 8), qb + 8 * (st0 + 8));
                        X[p][3] = oxe_vl_x(qa + 8 * (st0 + 9), qb + 8 * (st0 + 9));
                    }
                    for (int part = 0; part < 2; part++) {
                        // part 0: low 16 weights, sub-block half*8+sb; part 1: high 16, sub-block half*8+sb+4
                        const int i = half * 8 + sb + 4 * part;
                        const int16x8_t sc16 = vmovl_s8(vld1_s8(w[b].scales + i * 8));   // 8 rows
                        const int16x8_t z0 = vzip1q_s16(sc16, sc16), z1 = vzip2q_s16(sc16, sc16);
                        // row pair q: [s(2q), s(2q), s(2q+1), s(2q+1)]
                        const int32x4_t sc[4] = { vmovl_s16(vget_low_s16(z0)), vmovl_s16(vget_high_s16(z0)),
                                                  vmovl_s16(vget_low_s16(z1)), vmovl_s16(vget_high_s16(z1)) };
                        for (int q = 0; q < 4; q++) {
                            for (int p = 0; p < NP; p++) {
                                int32x4_t d = vmmlaq_s32(vdupq_n_s32(0), W[q][2 * part], X[p][2 * part]);
                                d           = vmmlaq_s32(d, W[q][2 * part + 1], X[p][2 * part + 1]);
                                I[p][q]     = vmlaq_s32(I[p][q], sc[q], d);
                            }
                        }
                    }
                }
            }
            for (int p = 0; p < NP; p++) {
                for (int q = 0; q < 4; q++) {
                    // The reference multiplies in 64-bit halves and adds the combined vector, which GCC
                    // does not contract: an UNFUSED multiply then add. vmlaq_f32 is GCC's unfused
                    // multiply-add (the float_mla builtin; also what oxe_pf_tiles.inc uses for the
                    // Q6_K contract), so the result does not depend on -ffp-contract seeing through
                    // vget / vcombine. A plain vmulq_f32 + vaddq_f32 here WOULD be fused (1 ulp off).
                    accf[p][q] = vmlaq_f32(accf[p][q], vcvtq_f32_s32(I[p][q]), sbs[p][q]);
                }
            }
        }
        for (int p = 0; p < NP; p++) {
            oxe_vl_store8(accf[p], oa[p] + x * 8, ob[p] ? ob[p] + x * 8 : nullptr);
        }
    }
}

}  // namespace

static void oxe_vl_gemv_q6_K(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    if (nl <= 2) {
        oxe_vl_q6k_run<1>(n, s, vx, vy, nl, nc);
    } else {
        oxe_vl_q6k_run<2>(n, s, vx, vy, nl, nc);
    }
}
