/* SPDX-License-Identifier: MIT
 * oxn pre-op validation, shared by the host bench and the DSP skel (NPU-1b).
 * Pure C, no SDK material. The skel is authoritative; the host runs the same check before
 * every invoke so a host-side bug never reaches the device.
 */
#ifndef OXN_CHECK_H
#define OXN_CHECK_H

#include <stdint.h>
#include "oxn_proto.h"

/* GEMM working set: header + 1 row tile set + 2 weight blocks + 2 output tiles at R=C=1.
 * Must stay >= the oxn_gemm.c R=C=1 need (checked against the kernel in the simulator). */
static inline uint32_t oxn_gemm_vtcm_min(uint32_t K) {
    const uint32_t nkt = K / 32;
    return 4096u + 2048u * nkt /* act */ + 2u * 2048u * nkt /* weights x2 */ + 2u * 2048u /* out x2 */;
}

struct oxn_slot_view { uint32_t bytes; };  /* 0 = not mapped */

static inline uint32_t oxn_sat_kib(uint64_t v) { v = (v + 1023) >> 10; return v > 0xffffffffu ? 0xffffffffu : (uint32_t) v; }

/* a[] as in OXN_GEMM_*; vtcm_bytes = acquired VTCM. Returns OXN_OK or an OXN_E_GEMM_* code. */
static inline int oxn_gemm_check(const uint32_t * a, const struct oxn_slot_view * slots, uint32_t vtcm_bytes,
                                 uint32_t * detail) {
    uint32_t d = 0;
    const uint32_t T = a[OXN_GEMM_T], K = a[OXN_GEMM_K], O = a[OXN_GEMM_O];
    const uint32_t dims[3] = { T, K, O };
    for (int i = 0; i < 3; i++)
        if (!dims[i] || dims[i] % 32 || dims[i] > 65536) { *detail = dims[i]; return OXN_E_GEMM_DIMS; }

    if (a[OXN_GEMM_FLAGS] & ~OXN_GF_KNOWN) { *detail = a[OXN_GEMM_FLAGS]; return OXN_E_GEMM_FLAGS; }
    if (a[OXN_GEMM_R] > 255u) { *detail = a[OXN_GEMM_R]; return OXN_E_GEMM_FLAGS; }
    if (a[OXN_GEMM_C] > 255u) { *detail = a[OXN_GEMM_C]; return OXN_E_GEMM_FLAGS; }

    const uint32_t sl[3]  = { a[OXN_GEMM_X_SLOT], a[OXN_GEMM_W_SLOT], a[OXN_GEMM_Y_SLOT] };
    const uint32_t off[3] = { a[OXN_GEMM_X_OFF],  a[OXN_GEMM_W_OFF],  a[OXN_GEMM_Y_OFF] };
    const uint64_t wlen   = (a[OXN_GEMM_FLAGS] & OXN_GF_W8) ? (uint64_t) (O / 32) * (K / 32) * 1152u : 2ull * O * K;
    const uint64_t len[3] = { 2ull * T * K, wlen, 2ull * T * O };
    for (int i = 0; i < 3; i++)
        if (sl[i] >= OXN_MAX_SLOTS || !slots[sl[i]].bytes) { *detail = sl[i]; return OXN_E_GEMM_SLOT_X + i; }
    for (int i = 0; i < 3; i++) {
        const uint64_t end = (uint64_t) off[i] + len[i];
        if (end > slots[sl[i]].bytes) { *detail = oxn_sat_kib(end - slots[sl[i]].bytes); return OXN_E_GEMM_BOUNDS_X + i; }
    }
    for (int i = 0; i < 3; i++)
        if (off[i] & 127u) { *detail = off[i]; return OXN_E_GEMM_ALIGN; }
    for (int i = 0; i < 2; i++)
        if (sl[i] == sl[2] && (uint64_t) off[i] < (uint64_t) off[2] + len[2] && (uint64_t) off[2] < (uint64_t) off[i] + len[i]) {
            *detail = (uint32_t) i; return OXN_E_GEMM_OVERLAP;
        }
    uint32_t cap = a[OXN_GEMM_VTCM_CAP];
    if (!cap || cap > vtcm_bytes) cap = vtcm_bytes;
    if (cap < oxn_gemm_vtcm_min(K)) { *detail = oxn_gemm_vtcm_min(K); return OXN_E_GEMM_VTCM_CAP; }
    *detail = d;
    return OXN_OK;
}

static inline const char * oxn_strerror(int e) {
    switch (e) {
        case OXN_OK:                return "ok";
        case OXN_E_PROTO_ARGS:      return "PROTO_ARGS: remote_arg count/size mismatch (host/skel ABI)";
        case OXN_E_PROTO_VER:       return "PROTO_VER: host and skel protocol versions differ";
        case OXN_E_BAD_OP:          return "BAD_OP: unknown op";
        case OXN_E_ALREADY_STARTED: return "ALREADY_STARTED: START twice";
        case OXN_E_NOT_STARTED:     return "NOT_STARTED: op before START";
        case OXN_E_RES_ACQUIRE:     return "RES_ACQUIRE: HAP_compute_res_acquire (VTCM+HMX) failed";
        case OXN_E_VTCM_PTR:        return "VTCM_PTR: no VTCM pointer from compute_res";
        case OXN_E_RES_CACHED:      return "RES_CACHED: HAP_compute_res_acquire_cached failed (resources not held)";
        case OXN_E_HMX_PROBE:       return "HMX_PROBE: HMX lock probe at START failed";
        case OXN_E_DMA_ALLOC:       return "DMA_ALLOC: DMA ring allocation failed";
        case OXN_E_POWER:           return "POWER: HAP_power_set failed";
        case OXN_E_RES_NOT_HELD:    return "RES_NOT_HELD: GEMM while VTCM/HMX not held";
        case OXN_E_RES_PEROP:       return "RES_PEROP: per-op acquire_cached failed (HMX/VTCM busy in another session?)";
        case OXN_E_MODE_ARGS:       return "MODE_ARGS: bad power or resource mode";
        case OXN_E_MMAP_ARGS:       return "MMAP_ARGS: bad slot, slot busy or bad size";
        case OXN_E_MMAP_FAIL:       return "MMAP_FAIL: HAP_mmap of the fd failed";
        case OXN_E_MUNMAP_ARGS:     return "MUNMAP_ARGS: slot not mapped";
        case OXN_E_MUNMAP_FAIL:     return "MUNMAP_FAIL: unmap API failed; slot remains owned";
        case OXN_E_GEMM_DIMS:       return "GEMM_DIMS: T/K/O zero, not a multiple of 32 or > 65536";
        case OXN_E_GEMM_SLOT_X:     return "GEMM_SLOT_X: X slot bad or not mapped";
        case OXN_E_GEMM_SLOT_W:     return "GEMM_SLOT_W: W slot bad or not mapped";
        case OXN_E_GEMM_SLOT_Y:     return "GEMM_SLOT_Y: Y slot bad or not mapped";
        case OXN_E_GEMM_BOUNDS_X:   return "GEMM_BOUNDS_X: X runs past its slot (detail KiB)";
        case OXN_E_GEMM_BOUNDS_W:   return "GEMM_BOUNDS_W: W runs past its slot (detail KiB)";
        case OXN_E_GEMM_BOUNDS_Y:   return "GEMM_BOUNDS_Y: Y runs past its slot (detail KiB)";
        case OXN_E_GEMM_ALIGN:      return "GEMM_ALIGN: operand offset not 128-byte aligned";
        case OXN_E_GEMM_OVERLAP:    return "GEMM_OVERLAP: Y overlaps X (0) or W (1)";
        case OXN_E_GEMM_VTCM_CAP:   return "GEMM_VTCM_CAP: VTCM cap below the R=C=1 minimum (detail bytes)";
        case OXN_E_GEMM_FLAGS:      return "GEMM_FLAGS: unknown flag bit or R/C override out of range";
        case OXN_E_HMX_LOCK:        return "HMX_LOCK: HAP_compute_res_hmx_lock failed in GEMM";
        case OXN_E_GEMM_KERNEL:     return "GEMM_KERNEL: kernel error (detail -1 dim, -2 vtcm, -3 dma ring, -4 align)";
        case OXN_E_DMATEST_ARGS:    return "DMATEST_ARGS: bad slot, bounds, size, alignment or overlap (detail: arg index)";
        case OXN_E_DMATEST_FAIL:    return "DMATEST_FAIL: DMA test kernel error (detail rc)";
        case OXN_E_K1_ARGS:         return "K1_ARGS: synthetic matvec bounds rejected";
        case OXN_E_K1_KERNEL:       return "K1_KERNEL: synthetic matvec worker failed";
        default:                    return (uint32_t) e >= 0x80000400u && (uint32_t) e < 0x80000600u
                                           ? "remote framework error (raw code retained)" : "unknown";
    }
}

#endif
