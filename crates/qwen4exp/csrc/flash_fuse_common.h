// S-c fused decode phases: shared helpers (included by flash_graph.cpp before the flash_fuse_*.inc files).
// A fused phase is ONE ggml custom node (ggml_custom_4d, GGML_N_TASKS_MAX): ggml calls it on every graph thread with
// (ith, nth); the phase splits each step over all threads and synchronises dependent steps with FuseBarrier.
// Numerics rules (results/m3/sc-ggml-numerics.md): matvec rows = production from_float + vec_dot per (row, column);
// elementwise = the op's per-element formula (libm expf / log1pf where ggml uses libm, ggml's exported vec helpers
// for silu / swiglu with chunk bounds on multiples of 4); reductions (rms_norm, soft_max, l2_norm, sum_rows) one
// thread per row in ggml's order. Every phase must be bitwise equal to the ggml subgraph it replaces (model-free test).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdint>

extern "C" void ggml_vec_silu_f32(const int n, float * y, const float * x);
extern "C" void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);

// spin barrier for the threads of one custom-op call (all nth threads call the op; generation counter)
// FD-1 tracing (fg_ctx_set_trace): total spin time inside fused-phase barriers, summed over threads, in ns
inline std::atomic<bool> & fuse_trace_on() { static std::atomic<bool> v{false}; return v; }
inline std::atomic<uint64_t> & fuse_spin_ns() { static std::atomic<uint64_t> v{0}; return v; }
inline std::atomic<uint64_t> & fuse_waits() { static std::atomic<uint64_t> v{0}; return v; }

struct FuseBarrier {
    // count and gen on separate 128-byte lines (aliasing sweep: arrivals' RMW on count must not bounce the line the
    // waiters spin on)
    alignas(128) std::atomic<int> count{0};
    alignas(128) std::atomic<int> gen{0};
    void wait(int nth) {
        if (nth <= 1) return;
        const bool tr = fuse_trace_on().load(std::memory_order_relaxed);
        const auto t0 = tr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        wait_inner(nth);
        if (tr) {
            fuse_spin_ns().fetch_add((uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                                     std::memory_order_relaxed);
            fuse_waits().fetch_add(1, std::memory_order_relaxed);
        }
    }
    void wait_inner(int nth) {
        const int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == nth) {
            count.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) {
#if defined(__aarch64__)
                __asm__ __volatile__("yield");
#endif
            }
        }
    }
};

// [lo, hi) of n items for thread ith of nth, bounds rounded to multiples of `align` (the last thread takes the tail)
static inline void fuse_split(int64_t n, int ith, int nth, int64_t align, int64_t & lo, int64_t & hi) {
    const int64_t units = (n + align - 1) / align;
    const int64_t u0 = units * ith / nth, u1 = units * (ith + 1) / nth;
    lo = std::min(n, u0 * align);
    hi = std::min(n, u1 * align);
}
