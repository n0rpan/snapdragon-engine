// pf_attn (WORKSTREAM PREFILL, FP-1): the fixed-chunk attention part op of many-row ubatches (prefill) in register
// tiles, after oxe's tiled attention (kernels-sys oxe_dot_f16_tile, M3-ATTN.md). Mode 1 (`--attn online`, the
// default) only. BITWISE equal to attn_part_tiled / the untiled unit in flash_graph.cpp: per (query, cell) the same
// operations in the same order, only independent outputs are interleaved.
//
// Unit = (tile of 8 query rows, kv head, 256-cell block), as attn_part_tiled (same unit -> thread map). Per unit:
//   1. per row: the active (unmasked) cells of the block, common to the row's rk heads; Q -> f16 (ggml from_float)
//   2. scores: ggml_vec_dot_f16(K cell, Q) for 4 cells x 4 heads at a time: four float16x8 accumulators per dot,
//      accumulator j takes elements [i + 8j, i + 8j + 8) of every 32-step, vfmaq_f16, ggml's reduce (16 independent
//      FMA chains per pass instead of one call per pair); then `s * scale + mask` as the reference writes it
//   3. per query the online softmax scalars in cell order, the reference's expressions: M, S, and per cell
//      (ms = rescale when the max grows, vs = weight); expf from libm as before
//   4. P.V: per 32-dim chunk the block's V rows are converted to f32 once (32 KB, L1); 3 heads of a row at a time keep
//      their 3 x 32 accumulators in registers over the row's active cells: acc = fma(v, vs, acc) per (cell, dim) in
//      cell order, `acc *= ms` before the cell's FMA where the reference does it (ms != 1; rare, handled as breaks)
// The reference keeps the accumulators in memory (96 queries x 256 dims = 98 KB per unit, re-walked per cell) and
// calls vec_dot through a pointer per pair.
//
// Compiled with flash_graph.cpp's flags (FMA contraction on, as the reference: its `acc += vf * vs` is an fmla, its
// `S * ms + vs` and `s * sc + mask` are fmadd); those two are explicit fma calls here, because GCC did not contract
// the first one in this file's loop (found by the off-vs-on check: S one ulp off in a few records).
#include "ggml-cpu.h"
#include "ggml.h"

#include "pf_attn.h"
#include "pf_phase.h"

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t BLK = 256;   // FG_ATTN_BLOCK
constexpr int64_t ROWS = 8;    // AT_ROWS
constexpr int CH = 32;         // P.V dims per register chunk
constexpr int MAXH = 16;       // heads per kv head handled here

std::atomic<int> g_on{0};
std::atomic<long> g_units{0};

struct Scratch {
    std::vector<float> mk;         // [cell] mask row of the block, f32
    std::vector<uint16_t> act;     // [row][cell] active cells of the row, in order
    std::vector<ggml_fp16_t> qh;   // [head][DK]
    std::vector<float> sco;        // [head][k] scores of the row's active cells
    std::vector<float> vs;         // [row][head][k]
    std::vector<uint16_t> evk;     // [row][head][e] k of the cells with ms != 1
    std::vector<float> evm;        // [row][head][e] their ms
    std::vector<float> vbuf;       // [cell][CH] V chunk in f32
    std::vector<float> rec;        // fused op: [block][row][head][2 + DV] block records of one (tile, kv head)
};
Scratch & scratch() {
    static thread_local Scratch s;
    return s;
}

// GGML_F16x8_REDUCE (GGML_F16_ARR = 4), then ggml_vec_dot_f16's double sum and float store
inline float f16_reduce(float16x8_t a0, float16x8_t a1, float16x8_t a2, float16x8_t a3) {
    a0 = vaddq_f16(a0, a2);
    a1 = vaddq_f16(a1, a3);
    a0 = vaddq_f16(a0, a1);
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(a0));
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(a0));
    const double sumf = (double) vaddvq_f32(vaddq_f32(t0, t1));
    return (float) sumf;
}

// out[r * 4 + c] = ggml_vec_dot_f16(n, x[r], y[c]) for 4 x 4 (n % 32 == 0); callers repeat a row / column to pad
inline void dot_f16_4x4(int n, const __fp16 * const x[4], const __fp16 * const y[4], float * out) {
    float16x8_t st[4][16];
    for (int j = 0; j < 4; j++) {
        float16x8_t a[16];
        for (int i = 0; i < 16; i++) a[i] = vdupq_n_f16(0.0f);
        for (int i = 8 * j; i < n; i += 32) {
            const float16x8_t x0 = vld1q_f16(x[0] + i), x1 = vld1q_f16(x[1] + i), x2 = vld1q_f16(x[2] + i), x3 = vld1q_f16(x[3] + i);
            const float16x8_t y0 = vld1q_f16(y[0] + i), y1 = vld1q_f16(y[1] + i), y2 = vld1q_f16(y[2] + i), y3 = vld1q_f16(y[3] + i);
            a[0] = vfmaq_f16(a[0], x0, y0);
            a[1] = vfmaq_f16(a[1], x0, y1);
            a[2] = vfmaq_f16(a[2], x0, y2);
            a[3] = vfmaq_f16(a[3], x0, y3);
            a[4] = vfmaq_f16(a[4], x1, y0);
            a[5] = vfmaq_f16(a[5], x1, y1);
            a[6] = vfmaq_f16(a[6], x1, y2);
            a[7] = vfmaq_f16(a[7], x1, y3);
            a[8] = vfmaq_f16(a[8], x2, y0);
            a[9] = vfmaq_f16(a[9], x2, y1);
            a[10] = vfmaq_f16(a[10], x2, y2);
            a[11] = vfmaq_f16(a[11], x2, y3);
            a[12] = vfmaq_f16(a[12], x3, y0);
            a[13] = vfmaq_f16(a[13], x3, y1);
            a[14] = vfmaq_f16(a[14], x3, y2);
            a[15] = vfmaq_f16(a[15], x3, y3);
        }
        for (int i = 0; i < 16; i++) st[j][i] = a[i];
    }
    for (int i = 0; i < 16; i++) out[i] = f16_reduce(st[0][i], st[1][i], st[2][i], st[3][i]);
}

// cells [k0, k1) of the row's active list into the accumulators of NQ heads: a[q] = fma(v, vs[q][k], a[q]) per dim
template <int NQ>
inline void pv_segment(float32x4_t (*a)[CH / 4], const float * vbuf, const uint16_t * act, int k0, int k1, const float * const * vs) {
    for (int k = k0; k < k1; ++k) {
        const float * vr = vbuf + (size_t) act[k] * CH;
        float32x4_t s[NQ];
        for (int q = 0; q < NQ; ++q) s[q] = vld1q_dup_f32(vs[q] + k);
        for (int i = 0; i < CH / 4; ++i) {
            const float32x4_t v = vld1q_f32(vr + 4 * i);
            for (int q = 0; q < NQ; ++q) a[q][i] = vfmaq_f32(a[q][i], v, s[q]);
        }
    }
}

// one row's NQ heads (h0 ..) over one dim chunk: zero, the row's active cells in order with the rescale breaks, store
template <int NQ>
inline void pv_group(const float * vbuf, const uint16_t * act, int nact, const float * const * vs, const uint16_t * const * evk,
                     const float * const * evm, const int * nev, float * const * out) {
    float32x4_t a[NQ][CH / 4];
    for (int q = 0; q < NQ; ++q)
        for (int i = 0; i < CH / 4; ++i) a[q][i] = vdupq_n_f32(0.0f);
    int e[NQ];
    for (int q = 0; q < NQ; ++q) e[q] = 0;
    int k = 0;
    while (k < nact) {
        // the reference multiplies acc by ms BEFORE the cell's FMA when ms != 1
        int nb = nact;
        for (int q = 0; q < NQ; ++q) {
            if (e[q] < nev[q] && evk[q][e[q]] == k) {
                const float32x4_t m = vdupq_n_f32(evm[q][e[q]]);
                for (int i = 0; i < CH / 4; ++i) a[q][i] = vmulq_f32(a[q][i], m);
                ++e[q];
            }
            if (e[q] < nev[q]) nb = std::min(nb, (int) evk[q][e[q]]);
        }
        pv_segment<NQ>(a, vbuf, act, k, nb, vs);
        k = nb;
    }
    for (int q = 0; q < NQ; ++q)
        for (int i = 0; i < CH / 4; ++i) vst1q_f32(out[q] + 4 * i, a[q][i]);
}

}  // namespace

// shapes / types the tile code handles (q, k, v, mask as the part op gets them)
static bool attn_supported(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask) {
    const int64_t DK = k->ne[0], DV = v->ne[0], n_head = q->ne[2], n_kvh = k->ne[2];
    if (n_kvh < 1 || n_head % n_kvh != 0) return false;
    const int64_t rk = n_head / n_kvh;
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16) return false;
    return DK % 32 == 0 && DV % CH == 0 && rk <= MAXH && rk >= 1 && v->ne[2] == n_kvh;
}

// The tile code for both node kinds. fused == false: dst is the part tensor [2 + DV, blocks, heads, rows], unit =
// (row tile, kv head, block) as attn_part_tiled. fused == true (W4-I, FP-8: no partial tensor, which is 6.5 GB at
// 8,192 rows): dst is the attention output [DV, heads, rows]; unit = (row tile, kv head); the unit's block records
// go to thread scratch and are merged per query with attn_merge_op's expressions (global max over the blocks, then
// in block order S += S_b * w and out += acc_b * w with w = expf(M_b - M), then out *= 1 / S).
static bool attn_core(ggml_tensor * dst, int ith, int nth, bool fused) {
    const float mode = ((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2];
    if (mode == 2.0f || mode == 3.0f) return false;   // precise / two-pass keep their own code
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t DK = k->ne[0], DV = v->ne[0];
    const int64_t n_rows = q->ne[1], n_head = q->ne[2];
    const int64_t n_kv = k->ne[1];
    const int64_t nb = fused ? (n_kv + BLK - 1) / BLK : dst->ne[1];
    const int64_t n_kvh = k->ne[2];
    if (!attn_supported(q, k, v, mask)) return false;
    const int64_t rk = n_head / n_kvh;
    const int64_t REC = 2 + DV;
    const float sc = ((const float *) dst->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1];
    const auto * tf16 = ggml_get_type_traits_cpu(GGML_TYPE_F16);
    FgPfPhaseTimer pf_timer_(FG_PF_PHASE_ATTN, ith == 0); // FP-9
    Scratch & S_ = scratch();
    S_.mk.resize(BLK);
    S_.act.resize(ROWS * BLK);
    S_.qh.resize((size_t) MAXH * DK);
    S_.sco.resize((size_t) MAXH * BLK);
    S_.vs.resize((size_t) ROWS * MAXH * BLK);
    S_.evk.resize((size_t) ROWS * MAXH * BLK);
    S_.evm.resize((size_t) ROWS * MAXH * BLK);
    S_.vbuf.resize((size_t) BLK * CH);
    if (fused) S_.rec.resize((size_t) nb * ROWS * MAXH * REC);
    const int64_t n_rt = (n_rows + ROWS - 1) / ROWS;
    const int64_t units = n_rt * n_kvh * nb;
    if (ith == 0) g_units.fetch_add(units, std::memory_order_relaxed);
    // one (row tile t, kv head g, block b): the block records (M, S, acc[DV]) of the tile's queries
    auto unit_block = [&](const int64_t t, const int64_t g, const int64_t b) {
        const int64_t r0 = t * ROWS, r1 = std::min(n_rows, r0 + ROWS);
        const int64_t c0 = b * BLK, c1 = std::min(n_kv, (b + 1) * BLK), ncell = c1 - c0;
        auto outp = [&](int64_t row, int64_t h) {
            if (fused) return S_.rec.data() + (size_t) ((b * ROWS + (row - r0)) * MAXH + h) * REC;
            return (float *) ((char *) dst->data + b * dst->nb[1] + (g * rk + h) * dst->nb[2] + row * dst->nb[3]);
        };
        const char * kbase = (const char *) k->data + g * k->nb[2];
        const char * vbase = (const char *) v->data + g * v->nb[2];
        int nact[ROWS];
        int nev[ROWS][MAXH];
        bool any = false;
        // 1-3: per row, active cells, scores, online softmax scalars
        for (int64_t r = r0; r < r1; ++r) {
            const int ri = (int) (r - r0);
            const ggml_fp16_t * mp = (const ggml_fp16_t *) ((const char *) mask->data + r * mask->nb[1]);
            float * mk = S_.mk.data();
            uint16_t * act = S_.act.data() + (size_t) ri * BLK;
            int na = 0;
            for (int64_t j = 0; j < ncell; ++j) {
                mk[j] = ggml_fp16_to_fp32(mp[c0 + j]);
                if (mk[j] != -INFINITY) act[na++] = (uint16_t) j;
            }
            nact[ri] = na;
            if (na == 0) {
                // fully masked block for this row: M = -inf, S = 0, acc = 0 (as the reference)
                for (int64_t h = 0; h < rk; ++h) {
                    float * o = outp(r, h);
                    o[0] = -INFINITY;
                    o[1] = 0.0f;
                    std::fill(o + 2, o + 2 + DV, 0.0f);
                }
                continue;
            }
            any = true;
            const __fp16 * qp[MAXH];
            for (int64_t h = 0; h < rk; ++h) {
                const float * pq = (const float *) ((const char *) q->data + r * q->nb[1] + (g * rk + h) * q->nb[2]);
                tf16->from_float(pq, S_.qh.data() + h * DK, DK);
                qp[h] = reinterpret_cast<const __fp16 *>(S_.qh.data() + h * DK);
            }
            float * sco = S_.sco.data();
            for (int kk = 0; kk < na; kk += 4) {
                const __fp16 * xr[4];
                for (int i = 0; i < 4; ++i) {
                    const int kc = std::min(kk + i, na - 1);
                    xr[i] = reinterpret_cast<const __fp16 *>(kbase + (size_t) (c0 + act[kc]) * k->nb[1]);
                }
                for (int64_t h = 0; h < rk; h += 4) {
                    const __fp16 * yc[4];
                    for (int i = 0; i < 4; ++i) yc[i] = qp[std::min<int64_t>(h + i, rk - 1)];
                    float o16[16];
                    dot_f16_4x4((int) DK, xr, yc, o16);
                    for (int i = 0; i < 4 && kk + i < na; ++i) {
                        const float m = mk[act[kk + i]];
                        for (int c = 0; c < 4 && h + c < rk; ++c) {
                            const float s = o16[i * 4 + c];
                            sco[(h + c) * BLK + kk + i] = __builtin_fmaf(s, sc, m);   // reference: fmadd
                        }
                    }
                }
            }
            for (int64_t h = 0; h < rk; ++h) {
                const float * sq = sco + h * BLK;
                float * vsq = S_.vs.data() + ((size_t) ri * MAXH + h) * BLK;
                uint16_t * ek = S_.evk.data() + ((size_t) ri * MAXH + h) * BLK;
                float * em = S_.evm.data() + ((size_t) ri * MAXH + h) * BLK;
                int ne = 0;
                float M = -INFINITY, S = 0.0f;
                for (int kk = 0; kk < na; ++kk) {
                    const float s = sq[kk];
                    const float Mold = M;
                    float ms = 1.0f, vs = 1.0f;
                    if (s > M) {
                        M = s;
                        ms = expf(Mold - M);
                    } else {
                        vs = expf(s - M);
                    }
                    if (ms != 1.0f) {
                        ek[ne] = (uint16_t) kk;
                        em[ne] = ms;
                        ++ne;
                    }
                    vsq[kk] = vs;
                    S = __builtin_fmaf(S, ms, vs);   // the reference's `S = S * ms + vs` is one fmadd (object code); explicit here
                }
                nev[ri][h] = ne;
                float * o = outp(r, h);
                o[0] = M;
                o[1] = S;
            }
        }
        if (!any) return;
        // 4: P.V per dim chunk
        for (int64_t i0 = 0; i0 < DV; i0 += CH) {
            float * vb = S_.vbuf.data();
            for (int64_t j = 0; j < ncell; ++j) {
                const float16_t * vh = reinterpret_cast<const float16_t *>(vbase + (size_t) (c0 + j) * v->nb[1]) + i0;
                for (int i = 0; i < CH / 4; ++i) vst1q_f32(vb + j * CH + 4 * i, vcvt_f32_f16(vld1_f16(vh + 4 * i)));
            }
            for (int64_t r = r0; r < r1; ++r) {
                const int ri = (int) (r - r0);
                if (nact[ri] == 0) continue;
                const uint16_t * act = S_.act.data() + (size_t) ri * BLK;
                for (int64_t h = 0; h < rk; h += 3) {
                    const int nq = (int) std::min<int64_t>(3, rk - h);
                    const float * vs[3];
                    const uint16_t * ek[3];
                    const float * em[3];
                    float * out[3];
                    for (int qq = 0; qq < nq; ++qq) {
                        vs[qq] = S_.vs.data() + ((size_t) ri * MAXH + h + qq) * BLK;
                        ek[qq] = S_.evk.data() + ((size_t) ri * MAXH + h + qq) * BLK;
                        em[qq] = S_.evm.data() + ((size_t) ri * MAXH + h + qq) * BLK;
                        out[qq] = outp(r, h + qq) + 2 + i0;
                    }
                    switch (nq) {
                        case 3: pv_group<3>(vb, act, nact[ri], vs, ek, em, &nev[ri][h], out); break;
                        case 2: pv_group<2>(vb, act, nact[ri], vs, ek, em, &nev[ri][h], out); break;
                        default: pv_group<1>(vb, act, nact[ri], vs, ek, em, &nev[ri][h], out); break;
                    }
                }
            }
        }
    };
    if (!fused) {
        for (int64_t u = ith; u < units; u += nth) {
            unit_block(u / (nb * n_kvh), (u / nb) % n_kvh, u % nb);
        }
        return true;
    }
    for (int64_t u = ith; u < n_rt * n_kvh; u += nth) {
        const int64_t g = u % n_kvh, t = u / n_kvh;
        const int64_t r0 = t * ROWS, r1 = std::min(n_rows, r0 + ROWS);
        for (int64_t b = 0; b < nb; ++b) unit_block(t, g, b);
        // merge, per query: attn_merge_op's operations in its order
        for (int64_t r = r0; r < r1; ++r) {
            for (int64_t h = 0; h < rk; ++h) {
                float * out = (float *) ((char *) dst->data + (g * rk + h) * dst->nb[1] + r * dst->nb[2]);
                auto rec = [&](int64_t b) { return S_.rec.data() + (size_t) ((b * ROWS + (r - r0)) * MAXH + h) * REC; };
                float M = -INFINITY;
                for (int64_t b = 0; b < nb; ++b) M = std::max(M, rec(b)[0]);
                float32x4_t acc[CH / 4];
                float S = 0.0f;
                // (w is recomputed per dim chunk below: the same expf of the same floats, so the same value)
                // dim chunks outer so that the output accumulators stay in registers; per element the blocks are
                // still added in block order, as in the reference; S and w do not depend on the chunk
                for (int64_t i0 = 0; i0 < DV; i0 += CH) {
                    for (int i = 0; i < CH / 4; ++i) acc[i] = vdupq_n_f32(0.0f);
                    for (int64_t b = 0; b < nb; ++b) {
                        const float * pb = rec(b);
                        if (pb[1] == 0.0f) continue; // fully masked block
                        const float w = expf(pb[0] - M);
                        if (i0 == 0) S = __builtin_fmaf(pb[1], w, S);   // reference: S += pb[1] * w (fmadd)
                        const float32x4_t wv = vdupq_n_f32(w);
                        for (int i = 0; i < CH / 4; ++i) acc[i] = vfmaq_f32(acc[i], vld1q_f32(pb + 2 + i0 + 4 * i), wv);
                    }
                    for (int i = 0; i < CH / 4; ++i) vst1q_f32(out + i0 + 4 * i, acc[i]);
                }
                const float inv = S == 0.0f ? 0.0f : 1.0f / S;
                const float32x4_t iv = vdupq_n_f32(inv);
                for (int64_t i = 0; i < DV; i += 4) vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), iv));
            }
        }
    }
    return true;
}

bool fg_pf_attn_part(ggml_tensor * dst, int ith, int nth) {
    if (!g_on.load(std::memory_order_relaxed)) return false;
    return attn_core(dst, ith, nth, false);
}

bool fg_pf_attn_fused_ok(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask) {
    // as the tiled dispatch in attn_part_op: many rows, 64-dim chunks; plus this file's own shape rules
    return g_on.load(std::memory_order_relaxed) == 2 && q->ne[1] >= 2 * ROWS && v->ne[0] % 64 == 0 && attn_supported(q, k, v, mask);
}

// dst [DV, heads, rows] = merge(part(q, k, v, mask)) without the part tensor; src = q, k, v, mask; op_params as the
// part node (scale, mode 1). Built only when fg_pf_attn_fused_ok.
void fg_pf_attn_fused_op(ggml_tensor * dst, int ith, int nth, void *) {
    const bool ok = attn_core(dst, ith, nth, true);
    GGML_ASSERT(ok && "fg_pf_attn_fused_op: unsupported node (fg_pf_attn_fused_ok was not checked)");
}

extern "C" {

void fg_pf_set_attn(int on) { g_on.store(on, std::memory_order_relaxed); }
int fg_pf_get_attn(void) { return g_on.load(std::memory_order_relaxed); }
long fg_pf_attn_units(int reset) {
    const long u = g_units.load();
    if (reset) g_units.store(0);
    return u;
}

// Model-free check + timing of the part op on a synthetic fixture (HEAD_D 256, 24 heads, 2 kv heads): the SAME graph
// node computed with the switch off (flash_graph.cpp's tiled path) and on (this file), compared over the whole
// output, bitwise. pattern 0: causal mask with holes; 1: pool-sparse (4-cell pools kept with p = 0.4) under the causal
// bound; 2: dense causal with exact-zero V rows / elements and large scores (many max updates, tiny weights).
// Rows are the LAST n_rows positions of an n_kv context. out2 = ms off, ms on (best of reps). Returns 0 = bitwise equal.
int fg_pf_selftest_attn(int n_kv, int n_rows, int nth, int pattern, int reps, double * out2) {
    // out2: [0] ms part op off, [1] ms part op with tiles, [2] ms part + merge (tiles), [3] ms fused node
    const int64_t DK = 256, NH = 24, NKV = 2;
    const int64_t nb = (n_kv + BLK - 1) / BLK;
    const size_t part_bytes = (size_t) (2 + DK) * nb * NH * n_rows * 4;
    const size_t out_bytes = (size_t) DK * NH * n_rows * 4;
    const size_t mem = ((size_t) 64 << 20) + (size_t) n_kv * DK * NKV * 2 * 2 + (size_t) n_kv * n_rows * 2 + (size_t) DK * n_rows * NH * 4 + 2 * part_bytes + 2 * out_bytes;
    std::vector<uint8_t> meta(mem);
    ggml_init_params ip = {meta.size(), meta.data(), false};
    ggml_context * ctx = ggml_init(ip);
    uint64_t x = 0x9e3779b97f4a7c15ull ^ ((uint64_t) pattern << 32) ^ (uint64_t) n_kv;
    auto rnu = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    auto rnd = [&]() { return (float) ((rnu() >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f; };
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DK, n_rows, NH);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, DK, n_kv, NKV);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, DK, n_kv, NKV);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_rows);
    const float qs = pattern == 2 ? 40.0f : 3.0f;
    for (int64_t i = 0; i < ggml_nelements(q); ++i) ((float *) q->data)[i] = rnd() * qs;
    for (int64_t i = 0; i < ggml_nelements(k); ++i) ((ggml_fp16_t *) k->data)[i] = ggml_fp32_to_fp16(rnd());
    for (int64_t i = 0; i < ggml_nelements(v); ++i) {
        float val = rnd();
        if (pattern == 2 && ((i / DK) % 5 == 0 || rnu() % 16 == 0)) val = (rnu() & 1) ? 0.0f : -0.0f;
        ((ggml_fp16_t *) v->data)[i] = ggml_fp32_to_fp16(val);
    }
    std::vector<uint8_t> pool((size_t) (n_kv / 4 + 1));
    for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t pos = n_kv - n_rows + r;   // the row's own cell
        if (pattern == 1)
            for (auto & p : pool) p = rnu() % 5 < 2;
        for (int64_t c2 = 0; c2 < n_kv; ++c2) {
            bool ok = c2 <= pos;
            if (pattern == 0) ok = ok && (c2 % 97) != 5;
            if (pattern == 1) ok = ok && (pool[c2 / 4] || c2 + 3 >= pos);
            ((ggml_fp16_t *) mask->data)[r * n_kv + c2] = ggml_fp32_to_fp16(ok ? 0.0f : -INFINITY);
        }
    }
    const float scale = 1.0f / sqrtf((float) DK);
    ggml_tensor * part[2];
    for (int w = 0; w < 2; ++w) {
        ggml_tensor * pa[4] = {q, k, v, mask};
        part[w] = ggml_custom_4d(ctx, GGML_TYPE_F32, 2 + DK, nb, NH, n_rows, pa, 4, fg_attn_part_op_ptr(), GGML_N_TASKS_MAX, nullptr);
        ((float *) part[w]->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1] = scale;
        ((float *) part[w]->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] = 1.0f;
    }
    const int prev = g_on.load();
    double best[2] = {1e30, 1e30};
    long units = 0;
    for (int rep = 0; rep < std::max(1, reps); ++rep) {
        for (int w = 0; w < 2; ++w) {
            g_on.store(w);
            const long u0 = g_units.load();
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, part[w]);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_graph_compute_with_ctx(ctx, gf, nth);
            best[w] = std::min(best[w], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            if (w) units = g_units.load() - u0;
        }
    }
    // W4-I: the fused node (no part tensor) against merge(part) of the reference path
    ggml_tensor * ma[1] = {part[0]};
    ggml_tensor * merged = ggml_custom_4d(ctx, GGML_TYPE_F32, DK, NH, n_rows, 1, ma, 1, fg_attn_merge_op_ptr(), GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * fa[4] = {q, k, v, mask};
    ggml_tensor * fusedt = ggml_custom_4d(ctx, GGML_TYPE_F32, DK, NH, n_rows, 1, fa, 4, fg_pf_attn_fused_op, GGML_N_TASKS_MAX, nullptr);
    ((float *) fusedt->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 1] = scale;
    ((float *) fusedt->op_params)[GGML_MAX_OP_PARAMS / sizeof(float) - 2] = 1.0f;
    double best_m[2] = {1e30, 1e30};
    bool fused_same = true;
    if (attn_supported(q, k, v, mask)) {
        for (int rep = 0; rep < std::max(1, reps); ++rep) {
            // reference: part (tiles on, already shown equal to off) + merge, timed together
            g_on.store(1);
            ggml_cgraph * g1 = ggml_new_graph(ctx);
            ggml_build_forward_expand(g1, merged);
            auto t0 = std::chrono::steady_clock::now();
            ggml_graph_compute_with_ctx(ctx, g1, nth);
            best_m[0] = std::min(best_m[0], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            ggml_cgraph * g2 = ggml_new_graph(ctx);
            ggml_build_forward_expand(g2, fusedt);
            t0 = std::chrono::steady_clock::now();
            ggml_graph_compute_with_ctx(ctx, g2, nth);
            best_m[1] = std::min(best_m[1], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        fused_same = std::memcmp(merged->data, fusedt->data, out_bytes) == 0;
        if (!fused_same) {
            size_t first = 0;
            while (first < out_bytes / 4 && ((uint32_t *) merged->data)[first] == ((uint32_t *) fusedt->data)[first]) ++first;
            std::fprintf(stderr, "fg_pf_selftest_attn n_kv %d rows %d nth %d pattern %d: FUSED node differs from merge(part), first diff at float %zu (dim %zu): %a / %a\n",
                         n_kv, n_rows, nth, pattern, first, first % (size_t) DK, ((float *) merged->data)[first], ((float *) fusedt->data)[first]);
        }
    }
    g_on.store(prev);
    const bool same = std::memcmp(part[0]->data, part[1]->data, part_bytes) == 0;
    const bool took = n_rows >= 16 ? units > 0 : true;   // below 16 rows flash_graph.cpp uses the untiled path
    if (out2) {
        out2[0] = best[0];
        out2[1] = best[1];
        out2[2] = best_m[0];
        out2[3] = best_m[1];
    }
    if (!same || !took) {
        size_t first = 0;
        while (first < part_bytes / 4 && ((uint32_t *) part[0]->data)[first] == ((uint32_t *) part[1]->data)[first]) ++first;
        std::fprintf(stderr, "fg_pf_selftest_attn n_kv %d rows %d nth %d pattern %d: bitwise %d, tile path taken %d, first diff at float %zu (element %zu of its [2 + DV] record)\n",
                     n_kv, n_rows, nth, pattern, (int) same, (int) took, first, first % (size_t) (2 + DK));
        if (!same) {
            const float * A = (const float *) part[0]->data, * B = (const float *) part[1]->data;
            size_t nd = 0, nrec = 0, nS = 0, nM = 0;
            for (size_t rec = 0; rec < part_bytes / 4 / (size_t) (2 + DK); ++rec) {
                size_t d = 0;
                for (size_t e = 0; e < (size_t) (2 + DK); ++e) d += std::memcmp(A + rec * (2 + DK) + e, B + rec * (2 + DK) + e, 4) != 0;
                nd += d;
                nrec += d != 0;
                nM += std::memcmp(A + rec * (2 + DK), B + rec * (2 + DK), 4) != 0;
                nS += std::memcmp(A + rec * (2 + DK) + 1, B + rec * (2 + DK) + 1, 4) != 0;
            }
            const size_t r0 = first / (size_t) (2 + DK) * (size_t) (2 + DK);
            std::fprintf(stderr, "  differing floats %zu in %zu records (M differs in %zu, S in %zu); first record: M %a / %a  S %a / %a  acc0 %a / %a\n", nd, nrec, nM,
                         nS, A[r0], B[r0], A[r0 + 1], B[r0 + 1], A[r0 + 2], B[r0 + 2]);
        }
    }
    ggml_free(ctx);
    return same && took && fused_same ? 0 : 1;
}

}  // extern "C"
