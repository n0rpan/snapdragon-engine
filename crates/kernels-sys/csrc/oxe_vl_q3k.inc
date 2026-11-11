// oxe_vl_q3k.inc: lane GEMV for Q3_K 8x8 (see oxe_vl.inc for the contract and the lane layout).
// Reference: ggml_gemv_q3_K_8x8_q8_K (dotprod branch, ggml_q3x8_dot_row_dotprod). Per (lane, row),
// per 256-block, the reference forms the exact integer
//     isum = sum over the 16 sub-blocks s of scale(row, s) * dot16(row, s, lane)
// (scale(row, s) = int8 6-bit scale minus 32, dot16 = 16 weight codes times 16 activation codes)
// and does, once per block, with da = d_x of the lane and dw = the row's f16 super-block scale:
//     accf = vfmaq_f32(accf, vcvtq_f32_s32(isum), vmulq_f32(da, dw))      accf starts at +0
// This kernel forms the same integers with SMMLA (two 8-code steps per sub-block, four row pairs,
// two lanes per instruction) and performs the same float update per (lane, row) with the same
// operand values in the same order.

namespace {

template <int NP>
static void oxe_vl_q3k_run(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    const int nb = n / QK_K;
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
        const block_q3_Kx8 * w = (const block_q3_Kx8 *) vx + (size_t) x * nb;
        float32x4_t accf[NP][4];
        for (int p = 0; p < NP; p++) {
            for (int q = 0; q < 4; q++) {
                accf[p][q] = vdupq_n_f32(0);
            }
        }
        for (int b = 0; b < nb; b++) {
            // weight side, shared by all lanes
            const float32x4_t wd0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d));       // rows 0..3
            const float32x4_t wd1 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d + 4));   // rows 4..7
            // row pair q: [d(2q), d(2q), d(2q+1), d(2q+1)] in the lane layout [r.a, r.b, r'.a, r'.b]
            const float32x4_t wdq[4] = { oxe_vl_rr_lo(wd0), oxe_vl_rr_hi(wd0), oxe_vl_rr_lo(wd1), oxe_vl_rr_hi(wd1) };

            int8x16_t S[8];
            ggml_q3x8_decode_scales(w[b].scales, S);
            // sub-block s = 4t + gi uses S[s / 2], low half for even s, high half for odd s (rows 0..7)
            int32_t scb[16][4][4];   // [s][row pair q] = [sc(2q), sc(2q), sc(2q+1), sc(2q+1)]
            for (int s_ = 0; s_ < 16; s_++) {
                const int8x8_t  h8 = (s_ & 1) ? vget_high_s8(S[s_ >> 1]) : vget_low_s8(S[s_ >> 1]);
                const int16x8_t v  = vmovl_s8(h8);
                const int16x8_t z1 = vzip1q_s16(v, v), z2 = vzip2q_s16(v, v);
                vst1q_s32(scb[s_][0], vmovl_s16(vget_low_s16(z1)));
                vst1q_s32(scb[s_][1], vmovl_s16(vget_high_s16(z1)));
                vst1q_s32(scb[s_][2], vmovl_s16(vget_low_s16(z2)));
                vst1q_s32(scb[s_][3], vmovl_s16(vget_high_s16(z2)));
            }

            float32x4_t sbd[NP][4];   // da * dw
            int32x4_t   isum[NP][4];
            for (int p = 0; p < NP; p++) {
                const float32x2_t dab = vset_lane_f32(yb[p][b].d, vdup_n_f32(ya[p][b].d), 1);
                const float32x4_t xd  = vcombine_f32(dab, dab);   // [d_a, d_b, d_a, d_b]
                for (int q = 0; q < 4; q++) {
                    sbd[p][q]  = vmulq_f32(xd, wdq[q]);
                    isum[p][q] = vdupq_n_s32(0);
                }
            }

            for (int t = 0; t < 4; t++) {
                // activation steps 8t .. 8t+7 (codes 64t .. 64t+63) per lane pair
                int8x16_t X[NP][8];
                for (int p = 0; p < NP; p++) {
                    const int8_t * qa = ya[p][b].qs + 64 * t;
                    const int8_t * qb = yb[p][b].qs + 64 * t;
                    for (int i = 0; i < 8; i++) {
                        X[p][i] = oxe_vl_x(qa + 8 * i, qb + 8 * i);
                    }
                }
                for (int cp = 0; cp < 4; cp++) {
                    // triple cp of this t: rows 2cp (bytes 0..7 of each R) and 2cp+1 (bytes 8..15)
                    int8x16_t R[8];
                    GGML_Q3X8_DECODE_TRIPLE(w[b].qs + (4 * t + cp) * 48, R);
                    for (int gi = 0; gi < 4; gi++) {
                        const int32x4_t sc = vld1q_s32(scb[4 * t + gi][cp]);
                        for (int p = 0; p < NP; p++) {
                            // 16 codes of sub-block 4t+gi: steps 2gi and 2gi+1 of this t
                            int32x4_t d16 = vmmlaq_s32(vdupq_n_s32(0), R[2 * gi], X[p][2 * gi]);
                            d16           = vmmlaq_s32(d16, R[2 * gi + 1], X[p][2 * gi + 1]);
                            isum[p][cp]   = vmlaq_s32(isum[p][cp], sc, d16);
                        }
                    }
                }
            }

            for (int p = 0; p < NP; p++) {
                for (int q = 0; q < 4; q++) {
                    accf[p][q] = vfmaq_f32(accf[p][q], vcvtq_f32_s32(isum[p][q]), sbd[p][q]);
                }
            }
        }
        for (int p = 0; p < NP; p++) {
            oxe_vl_store8(accf[p], oa[p] + x * 8, ob[p] ? ob[p] + x * 8 : nullptr);
        }
    }
}

}  // namespace

static void oxe_vl_gemv_q3_K(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    if (nl <= 2) {
        oxe_vl_q3k_run<1>(n, s, vx, vy, nl, nc);
    } else {
        oxe_vl_q3k_run<2>(n, s, vx, vy, nl, nc);
    }
}
