// oxe_vl_q5k.inc: lane GEMV for Q5_K 8x8 (see oxe_vl.inc for the contract and the lane layout).
// Reference: ggml_gemv_q5_K_8x8_q8_K, the only NEON branch of that function, guarded by
// __aarch64__ && __ARM_NEON && __ARM_FEATURE_DOTPROD (repack_subset.cpp, the #if at line 1283; this
// build defines DOTPROD, and the reference has no i8mm-specific branch). The 5-bit code of a weight is
// its 4 bits from qs plus bit 0 (lo sub-block) or bit 1 (hi sub-block) of qh, shifted by 2 per step.
// Per (lane, row), per 256-block, per 64-weight step sb = 0..3, with I_lo / I_hi the int32 dots of
// the two 32-weight sub-blocks, the reference does, in this order:
//     acc = vfmaq_f32(acc, d_w * d_x, f32(scale_lo * I_lo))
//     acc = vfmaq_f32(acc, d_w * d_x, f32(scale_hi * I_hi))
//     acc = vmlsq_f32(acc, dmin_w * d_x, f32(min_lo * bsum_lo + min_hi * bsum_hi))
// The min term is applied per sub-block step (not once per block as in Q4_K). acc starts at +0 for
// every output column group.
//
// Lane layout: W16 = [row r: 8 codes | row r+1: 8 codes] (from qs[16q + 64j ..] and qh[16q + 64j ..])
// and X = [lane a: 8 codes | lane b: 8 codes], so vmmlaq_s32(acc, W16, X) gives [r.a, r.b, r'.a, r'.b]
// and every float vector below is in that layout. Scales and mins are spread per row as
// [s_r, s_r, s_r', s_r'], the lane products of d_w and d_x as [d_a, d_b, d_a, d_b] per row.

namespace {

template <int NP>
static void oxe_vl_q5k_run(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    const int        nb   = n / QK_K;
    const uint8x16_t m4b  = vdupq_n_u8(0x0f);
    const uint8x16_t mone = vdupq_n_u8(1);
    const uint8x16_t mtwo = vdupq_n_u8(2);
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
        const block_q5_Kx8 * w = (const block_q5_Kx8 *) vx + (size_t) x * nb;
        float32x4_t accf[NP][4];
        for (int p = 0; p < NP; p++) {
            for (int q = 0; q < 4; q++) {
                accf[p][q] = vdupq_n_f32(0);
            }
        }
        for (int b = 0; b < nb; b++) {
            const float32x4_t wd0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d));         // d of rows 0..3
            const float32x4_t wd1 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d + 4));     // rows 4..7
            const float32x4_t wm0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].dmin));
            const float32x4_t wm1 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].dmin + 4));
            const float32x4_t wdq[4] = { oxe_vl_rr_lo(wd0), oxe_vl_rr_hi(wd0), oxe_vl_rr_lo(wd1), oxe_vl_rr_hi(wd1) };
            const float32x4_t wmq[4] = { oxe_vl_rr_lo(wm0), oxe_vl_rr_hi(wm0), oxe_vl_rr_lo(wm1), oxe_vl_rr_hi(wm1) };
            float32x4_t sbs[NP][4];   // d_w * d_x
            float32x4_t sbm[NP][4];   // dmin_w * d_x
            int32_t     bsz[NP][8];   // [sub-block j]: int16 pair (bsum_a[j], bsum_b[j])
            for (int p = 0; p < NP; p++) {
                const float32x2_t dab = vset_lane_f32(yb[p][b].d, vdup_n_f32(ya[p][b].d), 1);
                const float32x4_t xd  = vcombine_f32(dab, dab);   // [d_a, d_b, d_a, d_b]
                for (int q = 0; q < 4; q++) {
                    sbs[p][q] = vmulq_f32(wdq[q], xd);
                    sbm[p][q] = vmulq_f32(wmq[q], xd);
                }
                // the reference's pairwise add: the 8 sums of 32 values of the block
                const int16x8_t ba = vpaddq_s16(vld1q_s16(ya[p][b].bsums), vld1q_s16(ya[p][b].bsums + 8));
                const int16x8_t bb = vpaddq_s16(vld1q_s16(yb[p][b].bsums), vld1q_s16(yb[p][b].bsums + 8));
                vst1q_s32(bsz[p], vreinterpretq_s32_s16(vzip1q_s16(ba, bb)));
                vst1q_s32(bsz[p] + 4, vreinterpretq_s32_s16(vzip2q_s16(ba, bb)));
            }
            // high bits of the 5-bit codes, loaded once per block and shifted by 2 after each step
            uint8x16_t qh[4][4];
            for (int q = 0; q < 4; q++) {
                for (int j = 0; j < 4; j++) {
                    qh[q][j] = vld1q_u8(w[b].qh + 16 * q + 64 * j);
                }
            }
            for (int sb = 0; sb < QK_K / 64; sb++) {
                // 6-bit scales and mins of the two 32-weight sub-blocks of this step, rows 0..7
                int16x8_t mins[2];
                int16x8_t scales[2];
                for (int i = 0; i < 2; i++) {
                    int8_t aux[8];
                    decode_q_Kx8_6bit_scales(&w[b].scales[sb * 24 + i * 12], &mins[i], aux);
                    scales[i] = vmovl_s8(vld1_s8(aux));
                }
                // row pair q: [v(2q), v(2q), v(2q+1), v(2q+1)]
                const int16x8_t sl0 = vzip1q_s16(scales[0], scales[0]), sl1 = vzip2q_s16(scales[0], scales[0]);
                const int16x8_t sh0 = vzip1q_s16(scales[1], scales[1]), sh1 = vzip2q_s16(scales[1], scales[1]);
                const int16x8_t ml0 = vzip1q_s16(mins[0], mins[0]), ml1 = vzip2q_s16(mins[0], mins[0]);
                const int16x8_t mh0 = vzip1q_s16(mins[1], mins[1]), mh1 = vzip2q_s16(mins[1], mins[1]);
                const int32x4_t sc_lo[4] = { vmovl_s16(vget_low_s16(sl0)), vmovl_high_s16(sl0), vmovl_s16(vget_low_s16(sl1)), vmovl_high_s16(sl1) };
                const int32x4_t sc_hi[4] = { vmovl_s16(vget_low_s16(sh0)), vmovl_high_s16(sh0), vmovl_s16(vget_low_s16(sh1)), vmovl_high_s16(sh1) };
                const int16x4_t mn_lo[4] = { vget_low_s16(ml0), vget_high_s16(ml0), vget_low_s16(ml1), vget_high_s16(ml1) };
                const int16x4_t mn_hi[4] = { vget_low_s16(mh0), vget_high_s16(mh0), vget_low_s16(mh1), vget_high_s16(mh1) };

                // the 64 activation codes of this step per lane pair: X[k] = weights 8k .. 8k+7
                int8x16_t X[NP][8];
                for (int p = 0; p < NP; p++) {
                    const int8_t * qa = ya[p][b].qs + sb * 64;
                    const int8_t * qb = yb[p][b].qs + sb * 64;
                    for (int k = 0; k < 8; k++) {
                        X[p][k] = oxe_vl_x(qa + 8 * k, qb + 8 * k);
                    }
                }
                const uint8_t * qs = w[b].qs + sb * QK_K;
                for (int q = 0; q < 4; q++) {
                    // rows 2q, 2q+1: low 4 bits of the codes of weights 0..31 of the step, high 4 bits of 32..63
                    const uint8x16_t v[4] = { vld1q_u8(qs + 16 * q), vld1q_u8(qs + 16 * q + 64),
                                              vld1q_u8(qs + 16 * q + 128), vld1q_u8(qs + 16 * q + 192) };
                    int8x16_t Wl[4], Wh[4];
                    for (int j = 0; j < 4; j++) {
                        const uint8x16_t hbit_lo = vandq_u8(qh[q][j], mone);
                        const uint8x16_t hbit_hi = vshlq_n_u8(vandq_u8(qh[q][j], mtwo), 3);
                        qh[q][j]                 = vshrq_n_u8(qh[q][j], 2);
                        Wl[j] = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(v[j], m4b), hbit_lo, 4));
                        Wh[j] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(v[j], 4), hbit_hi));
                    }
                    for (int p = 0; p < NP; p++) {
                        int32x4_t lo = vmmlaq_s32(vdupq_n_s32(0), Wl[0], X[p][0]);
                        int32x4_t hi = vmmlaq_s32(vdupq_n_s32(0), Wh[0], X[p][4]);
                        lo = vmmlaq_s32(lo, Wl[1], X[p][1]);
                        hi = vmmlaq_s32(hi, Wh[1], X[p][5]);
                        lo = vmmlaq_s32(lo, Wl[2], X[p][2]);
                        hi = vmmlaq_s32(hi, Wh[2], X[p][6]);
                        lo = vmmlaq_s32(lo, Wl[3], X[p][3]);
                        hi = vmmlaq_s32(hi, Wh[3], X[p][7]);
                        const float32x4_t f_lo = vcvtq_f32_s32(vmulq_s32(sc_lo[q], lo));
                        accf[p][q]             = vfmaq_f32(accf[p][q], sbs[p][q], f_lo);
                        const float32x4_t f_hi = vcvtq_f32_s32(vmulq_s32(sc_hi[q], hi));
                        accf[p][q]             = vfmaq_f32(accf[p][q], sbs[p][q], f_hi);
                        // [bsum_a, bsum_b, bsum_a, bsum_b] x [min r, min r, min r', min r']
                        const int16x4_t bs_lo = vreinterpret_s16_s32(vdup_n_s32(bsz[p][2 * sb]));
                        const int16x4_t bs_hi = vreinterpret_s16_s32(vdup_n_s32(bsz[p][2 * sb + 1]));
                        int32x4_t       bias  = vmull_s16(bs_lo, mn_lo[q]);
                        bias                  = vmlal_s16(bias, bs_hi, mn_hi[q]);
                        accf[p][q]            = vmlsq_f32(accf[p][q], sbm[p][q], vcvtq_f32_s32(bias));
                    }
                }
            }
        }
        for (int p = 0; p < NP; p++) {
            oxe_vl_store8(accf[p], oa[p] + x * 8, ob[p] ? ob[p] + x * 8 : nullptr);
        }
    }
}

}  // namespace

static void oxe_vl_gemv_q5_K(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    if (nl <= 2) {
        oxe_vl_q5k_run<1>(n, s, vx, vy, nl, nc);
    } else {
        oxe_vl_q5k_run<2>(n, s, vx, vy, nl, nc);
    }
}
