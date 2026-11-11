// oxe_vl_q80.inc: lane GEMV for Q8_0 4x8 (see oxe_vl.inc for the contract and the lane layout).
// Reference: ggml_gemv_q8_0_4x8_q8_0, NEON + DOTPROD branch (the one compiled with
// -march=armv8.6-a+dotprod+i8mm+fp16+bf16+nosve+nosme: __ARM_FEATURE_SVE undefined, so the SVE
// branch is out; the i8mm flag does not change this function). Per (lane, row r), per 32-weight
// block b, with isum = the exact int32 dot of the 32 weight codes of row r with the 32 activation
// codes of the lane, and d_w = f32(block_q8_0x4.d[r]), d_x = f32(block_q8_0.d) of the lane:
//     acc = vfmaq_f32(acc, f32(isum), d_x * d_w)      (acc starts at +0, blocks in order)
// The reference's vdotq/vpaddq integer tree is only an exact integer grouping of isum.
//
// Weight layout (block_q8_0x4): qs[32 k + 8 j + i] = code i of 8-group k of row j (k = 0..3).
// Per 8-group k, rows 2q and 2q+1 are the 16 contiguous bytes qs[32 k + 16 q ..], which is the
// W operand of vmmlaq_s32: W = [row 2q: 8 codes | row 2q+1: 8 codes]. The activation 8-group k of
// lane pair (a, b) is X = [a: qs[8 k ..] | b: qs[8 k ..]]. SMMLA then gives the 8-code dots
// [r.a, r.b, r'.a, r'.b] and summing the 4 groups gives the 32-code isum per (lane, row).
// Float side per (lane, row pair): prod = [d_w(r) d_x(a), d_w(r) d_x(b), d_w(r') d_x(a), d_w(r') d_x(b)]
// computed as vmulq_f32(xd, wd) with xd = [d_a, d_b, d_a, d_b] (same operand order as the
// reference's vmulq_f32(vcvt(ad), vcvt(bd))), then one vfmaq_f32 per block.

namespace {

template <int NP>
static void oxe_vl_q80_run(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    const int nb = n / QK8_0;
    const block_q8_0 * ya[NP];
    const block_q8_0 * yb[NP];
    float *            oa[NP];
    float *            ob[NP];
    for (int p = 0; p < NP; p++) {
        const int la = 2 * p, lb = 2 * p + 1;
        ya[p] = (const block_q8_0 *) vy[la];
        yb[p] = (const block_q8_0 *) vy[lb < nl ? lb : la];
        oa[p] = s[la];
        ob[p] = lb < nl ? s[lb] : nullptr;
    }
    for (int x = 0; x < nc / 4; x++) {
        const block_q8_0x4 * w = (const block_q8_0x4 *) vx + (size_t) x * nb;
        float32x4_t accf[NP][2];
        for (int p = 0; p < NP; p++) {
            accf[p][0] = vdupq_n_f32(0);
            accf[p][1] = vdupq_n_f32(0);
        }
        for (int b = 0; b < nb; b++) {
            // weight side, shared by all lane pairs: d of rows 0..3, codes of rows (2q, 2q+1) per group k
            const float32x4_t wd0 = vcvt_f32_f16(vld1_f16((const __fp16 *) w[b].d));
            const float32x4_t wdq[2] = { oxe_vl_rr_lo(wd0), oxe_vl_rr_hi(wd0) };
            int8x16_t         W[2][4];
            for (int q = 0; q < 2; q++) {
                for (int k = 0; k < 4; k++) {
                    W[q][k] = vld1q_s8(w[b].qs + 32 * k + 16 * q);
                }
            }
            for (int p = 0; p < NP; p++) {
                // activation side of lane pair p
                const float32x4_t dA = vcvt_f32_f16(vld1_dup_f16((const __fp16 *) &ya[p][b].d));
                const float32x4_t dB = vcvt_f32_f16(vld1_dup_f16((const __fp16 *) &yb[p][b].d));
                const float32x4_t xd = vzip1q_f32(dA, dB);   // [d_a, d_b, d_a, d_b]
                int8x16_t X[4];
                for (int k = 0; k < 4; k++) {
                    X[k] = oxe_vl_x(ya[p][b].qs + 8 * k, yb[p][b].qs + 8 * k);
                }
                for (int q = 0; q < 2; q++) {
                    int32x4_t isum = vdupq_n_s32(0);
                    for (int k = 0; k < 4; k++) {
                        isum = vmmlaq_s32(isum, W[q][k], X[k]);
                    }
                    const float32x4_t prod = vmulq_f32(xd, wdq[q]);
                    accf[p][q]             = vfmaq_f32(accf[p][q], vcvtq_f32_s32(isum), prod);
                }
            }
        }
        // f[q] = [r(2q).a, r(2q).b, r(2q+1).a, r(2q+1).b]
        for (int p = 0; p < NP; p++) {
            vst1q_f32(oa[p] + 4 * x, vuzp1q_f32(accf[p][0], accf[p][1]));
            if (ob[p]) {
                vst1q_f32(ob[p] + 4 * x, vuzp2q_f32(accf[p][0], accf[p][1]));
            }
        }
    }
}

}  // namespace

static void oxe_vl_gemv_q8_0(int n, float * const * s, const void * vx, const void * const * vy, int nl, int nc) {
    if (nl <= 2) {
        oxe_vl_q80_run<1>(n, s, vx, vy, nl, nc);
    } else {
        oxe_vl_q80_run<2>(n, s, vx, vy, nl, nc);
    }
}
