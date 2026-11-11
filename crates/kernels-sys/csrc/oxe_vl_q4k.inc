// oxe_vl_q4k.inc: lane GEMV for Q4_K 8x8 (see oxe_vl.inc for the contract and the lane layout).
// Reference: ggml_gemv_q4_K_8x8_q8_K (dotprod branch). Per (lane, row), per 256-block, per 64-weight
// step sb = 0..3 the reference does, with I_lo / I_hi the int32 dots of the two 32-weight sub-blocks:
//     acc = vfmaq_f32(acc, d_w * d_x, f32(scale_lo * I_lo))
//     acc = vfmaq_f32(acc, d_w * d_x, f32(scale_hi * I_hi))
// and once per block, with B = sum over the 8 sub-blocks of min * bsum:
//     acc = vmlsq_f32(acc, f32(B), dmin_w * d_x)

namespace {

template <int NP>
static void oxe_vl_q4k_run(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    const int        nb  = n / QK_K;
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
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
        const block_q4_Kx8 * w = (const block_q4_Kx8 *) vx + (size_t) x * nb;
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
            int32x4_t   bias[NP][4];
            int32_t     bsz[NP][8];   // [sub-block pair j]: int16 pair (bsum_a[j], bsum_b[j])
            for (int p = 0; p < NP; p++) {
                const float32x2_t dab = vset_lane_f32(yb[p][b].d, vdup_n_f32(ya[p][b].d), 1);
                const float32x4_t xd  = vcombine_f32(dab, dab);   // [d_a, d_b, d_a, d_b]
                for (int q = 0; q < 4; q++) {
                    sbs[p][q]  = vmulq_f32(wdq[q], xd);
                    sbm[p][q]  = vmulq_f32(wmq[q], xd);
                    bias[p][q] = vdupq_n_s32(0);
                }
                // the reference's pairwise add: the 8 sums of 32 values of the block
                const int16x8_t ba = vpaddq_s16(vld1q_s16(ya[p][b].bsums), vld1q_s16(ya[p][b].bsums + 8));
                const int16x8_t bb = vpaddq_s16(vld1q_s16(yb[p][b].bsums), vld1q_s16(yb[p][b].bsums + 8));
                vst1q_s32(bsz[p], vreinterpretq_s32_s16(vzip1q_s16(ba, bb)));
                vst1q_s32(bsz[p] + 4, vreinterpretq_s32_s16(vzip2q_s16(ba, bb)));
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
                const uint8_t * q4 = w[b].qs + sb * QK_K;
                for (int q = 0; q < 4; q++) {
                    // rows 2q, 2q+1: low nibbles = weights 0..31 of the step, high nibbles = 32..63
                    const uint8x16_t v0 = vld1q_u8(q4 + 16 * q);
                    const uint8x16_t v1 = vld1q_u8(q4 + 16 * q + 64);
                    const uint8x16_t v2 = vld1q_u8(q4 + 16 * q + 128);
                    const uint8x16_t v3 = vld1q_u8(q4 + 16 * q + 192);
                    const int8x16_t  l0 = vreinterpretq_s8_u8(vandq_u8(v0, m4b)), h0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
                    const int8x16_t  l1 = vreinterpretq_s8_u8(vandq_u8(v1, m4b)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
                    const int8x16_t  l2 = vreinterpretq_s8_u8(vandq_u8(v2, m4b)), h2 = vreinterpretq_s8_u8(vshrq_n_u8(v2, 4));
                    const int8x16_t  l3 = vreinterpretq_s8_u8(vandq_u8(v3, m4b)), h3 = vreinterpretq_s8_u8(vshrq_n_u8(v3, 4));
                    for (int p = 0; p < NP; p++) {
                        int32x4_t lo = vmmlaq_s32(vdupq_n_s32(0), l0, X[p][0]);
                        int32x4_t hi = vmmlaq_s32(vdupq_n_s32(0), h0, X[p][4]);
                        lo = vmmlaq_s32(lo, l1, X[p][1]);
                        hi = vmmlaq_s32(hi, h1, X[p][5]);
                        lo = vmmlaq_s32(lo, l2, X[p][2]);
                        hi = vmmlaq_s32(hi, h2, X[p][6]);
                        lo = vmmlaq_s32(lo, l3, X[p][3]);
                        hi = vmmlaq_s32(hi, h3, X[p][7]);
                        const float32x4_t f_lo = vcvtq_f32_s32(vmulq_s32(sc_lo[q], lo));
                        accf[p][q]             = vfmaq_f32(accf[p][q], sbs[p][q], f_lo);
                        const float32x4_t f_hi = vcvtq_f32_s32(vmulq_s32(sc_hi[q], hi));
                        accf[p][q]             = vfmaq_f32(accf[p][q], sbs[p][q], f_hi);
                        // [bsum_a, bsum_b, bsum_a, bsum_b] x [min r, min r, min r', min r']
                        bias[p][q] = vmlal_s16(bias[p][q], vreinterpret_s16_s32(vdup_n_s32(bsz[p][2 * sb])), mn_lo[q]);
                        bias[p][q] = vmlal_s16(bias[p][q], vreinterpret_s16_s32(vdup_n_s32(bsz[p][2 * sb + 1])), mn_hi[q]);
                    }
                }
            }
            for (int p = 0; p < NP; p++) {
                for (int q = 0; q < 4; q++) {
                    accf[p][q] = vmlsq_f32(accf[p][q], vcvtq_f32_s32(bias[p][q]), sbm[p][q]);
                }
            }
        }
        for (int p = 0; p < NP; p++) {
            oxe_vl_store8(accf[p], oa[p] + x * 8, ob[p] ? ob[p] + x * 8 : nullptr);
        }
    }
}

}  // namespace

static void oxe_vl_gemv_q4_K(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    if (nl <= 2) {
        oxe_vl_q4k_run<1>(n, s, vx, vy, nl, nc);
    } else {
        oxe_vl_q4k_run<2>(n, s, vx, vy, nl, nc);
    }
}
