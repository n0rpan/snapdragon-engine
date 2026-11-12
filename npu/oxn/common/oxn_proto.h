/* SPDX-License-Identifier: MIT
 * oxn wire protocol (host <-> cDSP), hand-written. No IDL, no qaic, no SDK material.
 *
 * FastRPC remote_handle64 contract (qualcomm/fastrpc, BSD-3, inc/remote.h):
 *   method 0 open  : sc = MAKEX(0,0,2,0,0,1)
 *   method 1 close : sc = MAKEX(0,1,0,0,1,0)
 *   method 2 CMD   : sc = MAKEX(0,2,1,1,0,0) = 0x02010100
 *                    pra[0] = struct oxn_cmd (in), pra[1] = struct oxn_rsp (out)
 * One synchronous invoke per command. The DSP returns as soon as the command is
 * done: there is no dspqueue and no poll tail.
 *
 * All structs are fixed-size little-endian scalars, identical on the 64-bit host
 * and the 32-bit DSP.
 */
#ifndef OXN_PROTO_H
#define OXN_PROTO_H

#include <stdint.h>

#define OXN_SKEL_NAME   "liboxn_skel_v73.so"
#define OXN_MODULE_URI  "file:///liboxn_skel_v73.so?oxn_skel_handle_invoke&_modver=1.0"
#define OXN_SC_CMD      0x02010100u
#define OXN_METHOD_CMD  2u
#define OXN_HANDLE      0x6f786e31ull   /* 'oxn1' opaque cookie */
#define OXN_MAGIC       0x6f786e21u     /* 'oxn!' */
#define OXN_PROTO_VER   3u     /* 2: oxn status codes + v[OXN_V_DETAIL]; 3: GEMM flags/R/C, 32 slots */

enum oxn_op {
    OXN_OP_ECHO   = 1,  /* a[0] = seq                        -> v[0] = seq, v[1] = magic */
    OXN_OP_START  = 2,  /* a[0] = power mode, a[1] = vtcm request bytes (0 = all), a[2] = enum oxn_res_mode
                           -> v[0] = vtcm bytes, v[1] = hmx ok, v[2] = n hw threads, v[3] = proto ver */
    OXN_OP_STOP   = 3,  /* release VTCM/HMX, power votes, mappings */
    OXN_OP_MMAP   = 4,  /* a[0] = fd, a[1] = bytes, a[2] = slot    -> v[0] = dsp va */
    OXN_OP_MUNMAP = 5,  /* a[0] = slot */
    OXN_OP_GEMM   = 6,  /* see OXN_GEMM_* argument indices          -> OXN_GV_* */
    OXN_OP_INFO   = 7,  /* -> v[0] = vtcm bytes, v[1] = started, v[2] = mapped slots, v[3] = proto ver */
    OXN_OP_DMATEST = 8, /* DDR -> VTCM -> DDR copy through UDMA, no HMX (isolates DMA/mapping from HMX).
                           a[0] = src slot, a[1] = src off, a[2] = dst slot, a[3] = dst off,
                           a[4] = bytes, a[5] = chunk bytes (both multiples of 128, chunk <= VTCM)
                           -> v[0] = u32 word sum of everything that passed through VTCM, v[1] = chunks,
                              v[2],v[3] = pcycles DDR->VTCM (lo,hi), v[4],v[5] = pcycles VTCM->DDR (lo,hi),
                              v[6] = qtimer us for the op */
    OXN_OP_POWER   = 9, /* a[0] = power mode: re-vote inside a started session              -> v[0] = mode */
    OXN_OP_RESMODE = 10,/* a[0] = OXN_RES_HOLD | OXN_RES_PEROP: how VTCM+HMX are held       -> v[0] = mode */
    OXN_OP_PWRINFO = 11,/* reserved for a private, untracked power query (npu/oxn/local/); the tracked skel
                           returns v[7] = 0 ("not available") */
    OXN_OP_HMXCLK  = 12,/* a[0] = 1: explicit HMX clock vote at the highest corner; 0: back to the plain HMX
                           power-on vote -> v[0] = return code of the vote (data, not an op failure) */
    OXN_OP_K1     = 13, /* synthetic gathered quantized matvec; oxn_k1_check.h defines the arguments */
};
/* Resource modes. HOLD: acquire_cached once at START, kept for the whole session (one session
 * owns the NPU). PEROP: every GEMM/DMATEST does acquire_cached -> work -> release_cached, so
 * several sessions (PDs) can take turns on the one HMX unit; VTCM content does not survive
 * between ops (none of our ops needs it to). START a[2] selects the initial mode. */
enum oxn_res_mode { OXN_RES_HOLD = 0, OXN_RES_PEROP = 1 };

/* power modes for OXN_OP_START a[0] */
enum oxn_power { OXN_PWR_NONE = 0, OXN_PWR_HIGH = 1, OXN_PWR_BURST = 2 };

/* GEMM: Y[T,O] = X[T,K] * W[O,K]^T, fp16 in, fp16 out, all three in HMX tile layout
 * (see oxn_tiles.h). T, K, O multiples of 32. */
enum {
    OXN_GEMM_X_SLOT = 0, OXN_GEMM_X_OFF, OXN_GEMM_W_SLOT, OXN_GEMM_W_OFF, OXN_GEMM_Y_SLOT, OXN_GEMM_Y_OFF,
    OXN_GEMM_T, OXN_GEMM_K, OXN_GEMM_O, OXN_GEMM_VTCM_CAP, /* 0 = use all acquired VTCM */
    OXN_GEMM_FLAGS,      /* OXN_GF_* bits, 0 = the iter-1 kernel behaviour */
    OXN_GEMM_R,          /* row tiles per block, 0 = auto (largest that fits, then balanced) */
    OXN_GEMM_C,          /* column tiles per block, 0 = auto (4) */
    OXN_GEMM_NARGS
};
/* GEMM flags (a[OXN_GEMM_FLAGS]) */
#define OXN_GF_YCOL      0x01u  /* Y tiles column-major: tile (rt, ot) at index ot*(T/32) + rt. The write-back
                                   of a block is then C contiguous 1-D DMA transfers (no stride limit, async). */
#define OXN_GF_NO_WFETCH 0x02u  /* PROBE: fetch only the first weight block and reuse it (outputs are wrong) */
#define OXN_GF_NO_YWRITE 0x04u  /* PROBE: no output write-back (Y is not written) */
#define OXN_GF_NO_HMX    0x08u  /* PROBE: skip the HMX compute (DMA streaming only; Y gets stale VTCM data) */
#define OXN_GF_W8        0x10u  /* weights are W8 records (common/oxn_w8.h), expanded to fp16 on the DSP */
#define OXN_GF_KNOWN     0x1fu
#define OXN_GF_PROBES    (OXN_GF_NO_WFETCH | OXN_GF_NO_YWRITE | OXN_GF_NO_HMX)
enum {
    OXN_GV_PCYCLES = 0,  /* total UPCYCLE delta inside the DSP op (lo/hi in v[0], v[1]) */
    OXN_GV_PCYCLES_HI,
    OXN_GV_WAIT_PCYCLES, /* cycles spent waiting for DMA (lo) */
    OXN_GV_WAIT_PCYCLES_HI,
    OXN_GV_QTIMER_US,    /* HAP qtimer microseconds for the op */
    OXN_GV_STEPS,        /* number of (row block, col block) steps */
    OXN_GV_ROWBLK_TILES, /* R chosen */
    OXN_GV_COLBLK_TILES, /* C chosen */
};

#define OXN_MAX_SLOTS 32
#define OXN_NARGS     14
#define OXN_NVALS     12
#define OXN_V_DETAIL  (OXN_NVALS - 1)  /* on any oxn error: the raw SDK rc or the failing value */

/* oxn status codes (NPU-1b). Every pre-op validation failure and every resource failure
 * has its own code, so a device log is self-explaining without a debugger. They are
 * returned both as rsp.status and as the invoke rc; rsp.v[OXN_V_DETAIL] carries the
 * detail. Band 0x4f580000 ('OX'): positive, disjoint from AEE (0x80000400+) and from
 * Linux errnos. AEE codes can still appear for framework-level failures (FastRPC itself).
 * Keep oxn_strerror() in oxn_check.h in sync. */
#define OXN_E_BASE        0x4f580000
enum oxn_err {
    OXN_OK                = 0,
    /* protocol */
    OXN_E_PROTO_ARGS      = OXN_E_BASE + 0x01, /* remote_arg count/size mismatch (ABI)      detail: nLen of arg 0 */
    OXN_E_PROTO_VER       = OXN_E_BASE + 0x02, /* cmd.ver != OXN_PROTO_VER                  detail: cmd.ver */
    OXN_E_BAD_OP          = OXN_E_BASE + 0x03, /* unknown op                                detail: op */
    /* session / resources (START) */
    OXN_E_ALREADY_STARTED = OXN_E_BASE + 0x10, /* START twice */
    OXN_E_NOT_STARTED     = OXN_E_BASE + 0x11, /* MMAP/GEMM before START */
    OXN_E_RES_ACQUIRE     = OXN_E_BASE + 0x12, /* HAP_compute_res_acquire returned 0        detail: requested VTCM bytes */
    OXN_E_VTCM_PTR        = OXN_E_BASE + 0x13, /* no VTCM pointer from the attr             detail: SDK rc */
    OXN_E_RES_CACHED      = OXN_E_BASE + 0x14, /* HAP_compute_res_acquire_cached failed     detail: SDK rc */
    OXN_E_HMX_PROBE       = OXN_E_BASE + 0x15, /* HMX lock probe at START failed            detail: SDK rc */
    OXN_E_DMA_ALLOC       = OXN_E_BASE + 0x16, /* DMA ring allocation failed */
    OXN_E_POWER           = OXN_E_BASE + 0x17, /* HAP_power_set failed                      detail: SDK rc */
    OXN_E_RES_NOT_HELD    = OXN_E_BASE + 0x18, /* GEMM while VTCM/HMX not held (internal state) */
    OXN_E_RES_PEROP       = OXN_E_BASE + 0x19, /* per-op acquire_cached failed (another session holds HMX?)  detail: SDK rc */
    OXN_E_MODE_ARGS       = OXN_E_BASE + 0x1a, /* bad power or resource mode                 detail: the value */
    /* mapping */
    OXN_E_MMAP_ARGS       = OXN_E_BASE + 0x20, /* bad slot / slot in use / size 0 or >1 GiB detail: slot */
    OXN_E_MMAP_FAIL       = OXN_E_BASE + 0x21, /* HAP_mmap failed                           detail: fd */
    OXN_E_MUNMAP_ARGS     = OXN_E_BASE + 0x22, /* slot not mapped                           detail: slot */
    OXN_E_MUNMAP_FAIL     = OXN_E_BASE + 0x23, /* unmap API failed; slot remains owned      detail: raw rc */
    /* GEMM pre-op validation (oxn_gemm_check, run on host and on DSP) */
    OXN_E_GEMM_DIMS       = OXN_E_BASE + 0x30, /* T/K/O zero, not %32 or >65536             detail: first bad dim */
    OXN_E_GEMM_SLOT_X     = OXN_E_BASE + 0x31, /* X slot out of range or not mapped         detail: slot */
    OXN_E_GEMM_SLOT_W     = OXN_E_BASE + 0x32,
    OXN_E_GEMM_SLOT_Y     = OXN_E_BASE + 0x33,
    OXN_E_GEMM_BOUNDS_X   = OXN_E_BASE + 0x34, /* off + bytes beyond the slot               detail: bytes needed past the end (KiB, sat.) */
    OXN_E_GEMM_BOUNDS_W   = OXN_E_BASE + 0x35,
    OXN_E_GEMM_BOUNDS_Y   = OXN_E_BASE + 0x36,
    OXN_E_GEMM_ALIGN      = OXN_E_BASE + 0x37, /* an operand offset not 128-byte aligned    detail: offset */
    OXN_E_GEMM_OVERLAP    = OXN_E_BASE + 0x38, /* Y overlaps X or W                         detail: 0 = X, 1 = W */
    OXN_E_GEMM_VTCM_CAP   = OXN_E_BASE + 0x39, /* VTCM (cap) below the R=C=1 minimum        detail: minimum bytes */
    OXN_E_GEMM_FLAGS      = OXN_E_BASE + 0x3a, /* unknown flag bit, or R/C override > 255      detail: the bad value */
    /* GEMM execution */
    OXN_E_HMX_LOCK        = OXN_E_BASE + 0x40, /* HAP_compute_res_hmx_lock failed in GEMM   detail: SDK rc */
    OXN_E_GEMM_KERNEL     = OXN_E_BASE + 0x41, /* oxn_gemm_hmx returned an error            detail: OXN_GEMM_E* (negative) */
    /* DMATEST */
    OXN_E_DMATEST_ARGS    = OXN_E_BASE + 0x50, /* bad slot/bounds/size/alignment/overlap    detail: index of the first bad a[] */
    OXN_E_DMATEST_FAIL    = OXN_E_BASE + 0x51, /* oxn_dmatest returned an error             detail: its rc (negative) */
    OXN_E_K1_ARGS         = OXN_E_BASE + 0x60, /* synthetic matvec request rejected */
    OXN_E_K1_KERNEL       = OXN_E_BASE + 0x61, /* synthetic matvec worker failed */
};

struct oxn_cmd {
    uint32_t op;
    uint32_t ver;
    uint32_t a[OXN_NARGS];
};

struct oxn_rsp {
    int32_t  status;   /* 0 ok, else an OXN_E_* code (or an AEE code from the framework) */
    uint32_t op;
    uint32_t v[OXN_NVALS];
    uint32_t magic;
    uint32_t pad;
};

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(struct oxn_cmd) == 64, "oxn_cmd layout");
_Static_assert(sizeof(struct oxn_rsp) == 64, "oxn_rsp layout");
#endif

#endif
