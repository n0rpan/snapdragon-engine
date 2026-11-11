// tools/vl_check: model-free bitwise check and one-thread timing of the verify-lane GEMVs
// (crates/kernels-sys/csrc/oxe_vl*.inc) against the reference ggml GEMV called once per lane.
// Build + run: tools/vl_check/run.sh [type]   (type: 3, 4, 5, 6, 80; none = every kernel present)
// Exit 0 only if every compared float is bitwise equal.
#include "oxe_shim.cpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void quantize_row_q8_0(const float * x, void * y, int64_t k);

namespace {

struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float unit() { return (float) ((next() >> 11) & 0xffffff) / (float) 0x1000000; }   // [0, 1)
};

void put_f16(void * p, float v) {
    __fp16 h = (__fp16) v;
    memcpy(p, &h, 2);
}

// random source rows (nrows x k / qk blocks of `bb` bytes) with finite f16 scale fields at the given offsets
std::vector<uint8_t> random_rows(Rng & r, size_t nrows, size_t nblk, size_t bb, int off_d, int off_dmin) {
    std::vector<uint8_t> v(nrows * nblk * bb);
    for (auto & c : v) {
        c = (uint8_t) r.next();
    }
    for (size_t i = 0; i < nrows * nblk; i++) {
        put_f16(&v[i * bb + off_d], (r.unit() - 0.5f) * 0.05f);
        if (off_dmin >= 0) {
            put_f16(&v[i * bb + off_dmin], (r.unit() - 0.5f) * 0.05f);
        }
    }
    return v;
}

typedef int (*repack_fn)(const void *, void *, int64_t, int64_t);
typedef void (*gemv_fn)(int, float *, size_t, const void *, const void *, int, int);

struct Type {
    int          id;
    const char * name;
    size_t       bb;        // source block bytes
    int          qk;        // values per block
    int          nr;        // interleaved rows
    int          off_d, off_dmin;
    int          ext_lo, ext_hi;   // byte range of a block to fill with 0xff / 0x00 (extreme codes), keeps d
    repack_fn    repack;
    gemv_fn      gemv;
    bool         q80;
};

const Type TYPES[] = {
    { 4, "Q4_K", sizeof(block_q4_K), QK_K, 8, 0, 2, 4, (int) sizeof(block_q4_K), oxe_repack_q4_K_8x8, ggml_gemv_q4_K_8x8_q8_K, false },
    { 6, "Q6_K", sizeof(block_q6_K), QK_K, 8, 208, -1, 0, 208, oxe_repack_q6_K_8x8, ggml_gemv_q6_K_8x8_q8_K, false },
    { 3, "Q3_K", sizeof(block_q3_K), QK_K, 8, 108, -1, 0, 108, oxe_repack_q3_K_8x8, ggml_gemv_q3_K_8x8_q8_K, false },
    { 5, "Q5_K", sizeof(block_q5_K), QK_K, 8, 0, 2, 4, (int) sizeof(block_q5_K), oxe_repack_q5_K_8x8, ggml_gemv_q5_K_8x8_q8_K, false },
    { 80, "Q8_0", sizeof(block_q8_0), 32, 4, 0, -1, 2, (int) sizeof(block_q8_0), oxe_repack_q8_0_4x8, ggml_gemv_q8_0_4x8_q8_0, true },
};

size_t act_bytes(const Type & t, int k) { return t.q80 ? (size_t) k / 32 * sizeof(block_q8_0) : (size_t) k / QK_K * sizeof(block_q8_K); }

void quant_act(const Type & t, const float * x, void * y, int k) {
    if (t.q80) {
        quantize_row_q8_0(x, y, k);
    } else {
        quantize_row_q8_K_ref(x, (block_q8_K *) y, k);
    }
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

bool check_type(const Type & t, bool timing) {
    const void * probe[2] = { nullptr, nullptr };
    float *      po[2]    = { nullptr, nullptr };
    if (!oxe_vl_gemv(t.id, t.qk, po, nullptr, probe, 2, 0) && !oxe_vl_gemv(t.id, 0, po, nullptr, probe, 1, 0)) {
        // nc = 0 always returns 0; ask with a real call below instead
    }
    Rng  rng{ 0x9e3779b97f4a7c15ull + (uint64_t) t.id };
    bool have = true, ok = true;
    long cmp  = 0;
    const int shapes[][2] = { { 8, 256 }, { 72, 512 }, { 136, 2048 }, { 64, 768 }, { 16, 4096 } };
    for (auto & sh : shapes) {
        const int    nc = sh[0], k = sh[1];
        const size_t nblk = (size_t) k / t.qk;
        auto         rows = random_rows(rng, nc, nblk, t.bb, t.off_d, t.off_dmin);
        // extreme codes in row 0's first block, zeros in row 1's (d fields kept)
        memset(&rows[t.ext_lo], 0xff, t.ext_hi - t.ext_lo);
        memset(&rows[nblk * t.bb + t.ext_lo], 0x00, t.ext_hi - t.ext_lo);
        std::vector<uint8_t> packed(rows.size() + 64);
        if (t.repack(rows.data(), packed.data(), nc, k) != 0) {
            printf("%s: repack failed\n", t.name);
            return false;
        }
        const int            NL = 4;
        const size_t         ab = act_bytes(t, k);
        std::vector<float>   x((size_t) NL * k);
        std::vector<uint8_t> act(NL * ab);
        for (int l = 0; l < NL; l++) {
            for (int i = 0; i < k; i++) {
                float v = (rng.unit() - 0.5f) * 4.0f;
                if ((rng.next() & 63) == 0) {
                    v *= 40.0f;   // outliers: codes pile up near 0 with a few at +-127
                }
                if (l == 3 && i < 300) {
                    v = (i & 1) ? 1.0f : -1.0f;   // a block of +-127 codes (largest sums)
                }
                x[(size_t) l * k + i] = v;
            }
            quant_act(t, &x[(size_t) l * k], &act[l * ab], k);
        }
        std::vector<float> ref((size_t) NL * nc);
        for (int l = 0; l < NL; l++) {
            t.gemv(k, &ref[(size_t) l * nc], 0, packed.data(), &act[l * ab], 1, nc);
        }
        for (int nl = 1; nl <= NL; nl++) {
            // lanes in a rotated order so that every lane meets every pair slot
            for (int rot = 0; rot < NL; rot++) {
                std::vector<float> out((size_t) nl * (nc + 1), -12345.0f);
                float *            s[4];
                const void *       y[4];
                int                id[4];
                for (int l = 0; l < nl; l++) {
                    id[l] = (l + rot) % NL;
                    s[l]  = &out[(size_t) l * (nc + 1)];
                    y[l]  = &act[id[l] * ab];
                }
                if (!oxe_vl_gemv(t.id, k, s, packed.data(), y, nl, nc)) {
                    have = false;
                    break;
                }
                for (int l = 0; l < nl; l++) {
                    for (int c = 0; c < nc; c++) {
                        uint32_t a, b;
                        memcpy(&a, &ref[(size_t) id[l] * nc + c], 4);
                        memcpy(&b, &s[l][c], 4);
                        cmp++;
                        if (a != b) {
                            if (ok) {
                                printf("%s MISMATCH: nc %d k %d nl %d rot %d lane %d row %d: ref %.9g (%08x) got %.9g (%08x)\n", t.name, nc, k, nl, rot, l, c,
                                       ref[(size_t) id[l] * nc + c], a, s[l][c], b);
                            }
                            ok = false;
                        }
                    }
                    if (s[l][nc] != -12345.0f) {
                        printf("%s wrote past lane %d (nc %d k %d nl %d)\n", t.name, l, nc, k, nl);
                        ok = false;
                    }
                }
            }
            if (!have) {
                break;
            }
        }
        if (!have) {
            break;
        }
    }
    if (!have) {
        printf("%s: no lane kernel in this build (skipped)\n", t.name);
        return true;
    }
    printf("%s: %s (%ld floats compared bitwise against the GEMV per lane)\n", t.name, ok ? "BITWISE OK" : "FAILED", cmp);
    if (!ok || !timing) {
        return ok;
    }
    // one thread, hot weights (fit L2): two production shapes
    const int tshapes[][2] = { { 512, 2048 }, { 2048, 512 } };
    for (auto & sh : tshapes) {
        const int    nc = sh[0], k = sh[1];
        const size_t nblk = (size_t) k / t.qk;
        auto         rows = random_rows(rng, nc, nblk, t.bb, t.off_d, t.off_dmin);
        std::vector<uint8_t> packed(rows.size() + 64);
        t.repack(rows.data(), packed.data(), nc, k);
        const size_t         ab = act_bytes(t, k);
        std::vector<float>   x((size_t) 4 * k);
        std::vector<uint8_t> act(4 * ab);
        for (auto & v : x) {
            v = (rng.unit() - 0.5f) * 4.0f;
        }
        for (int l = 0; l < 4; l++) {
            quant_act(t, &x[(size_t) l * k], &act[l * ab], k);
        }
        std::vector<float> out((size_t) 4 * nc);
        float *            s[4];
        const void *       y[4];
        for (int l = 0; l < 4; l++) {
            s[l] = &out[(size_t) l * nc];
            y[l] = &act[l * ab];
        }
        const double mac = (double) nc * k;
        double       t1  = 0;
        for (int nl = 1; nl <= 4; nl++) {
            double best_ref = 1e9, best_vl = 1e9;
            for (int rep = 0; rep < 7; rep++) {
                const int it = 40;
                double    t0 = now();
                for (int i = 0; i < it; i++) {
                    for (int l = 0; l < nl; l++) {
                        t.gemv(k, s[l], 0, packed.data(), y[l], 1, nc);
                    }
                }
                double d = (now() - t0) / it;
                best_ref = d < best_ref ? d : best_ref;
                {
                    t0 = now();
                    for (int i = 0; i < it; i++) {
                        oxe_vl_gemv(t.id, k, s, packed.data(), y, nl, nc);
                    }
                    d       = (now() - t0) / it;
                    best_vl = d < best_vl ? d : best_vl;
                }
            }
            if (nl == 1) {
                t1 = best_ref;
                printf("  %s %4d x %4d  1 lane : gemv %7.1f us (%5.1f GMAC/s) | lane kernel with 1 lane %7.1f us = %.2fx\n", t.name, nc, k, best_ref * 1e6, mac / best_ref / 1e9, best_vl * 1e6, best_vl / best_ref);
            } else {
                printf("  %s %4d x %4d  %d lanes: gemv x%d %7.1f us = %.2fx | lane kernel %7.1f us = %.2fx of one gemv (%5.1f GMAC/s), extra lane %.2fx\n", t.name, nc, k,
                       nl, nl, best_ref * 1e6, best_ref / t1, best_vl * 1e6, best_vl / t1, mac * nl / best_vl / 1e9, (best_vl / t1 - 1.0) / (nl - 1));
            }
        }
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const int  only   = argc > 1 ? atoi(argv[1]) : 0;
    const bool timing = !(argc > 2 && argv[2][0] == 'n');
    bool       ok     = true;
    for (const Type & t : TYPES) {
        if (only == 0 || only == t.id) {
            ok = check_type(t, timing) && ok;
        }
    }
    printf("%s\n", ok ? "vl_check: ALL OK" : "vl_check: FAILED");
    return ok ? 0 : 1;
}
