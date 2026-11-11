// pf_phase: cheap per-phase clocks of the prefill custom ops (FP-9). See pf_phase.h.
#include "pf_phase.h"

#include <atomic>

static std::atomic<int64_t> g_ns[FG_PF_PHASE_N];
static std::atomic<long> g_calls[FG_PF_PHASE_N];

void fg_pf_phase_add(int phase, int64_t ns) {
    if (phase < 0 || phase >= FG_PF_PHASE_N) {
        return;
    }
    g_ns[phase].fetch_add(ns, std::memory_order_relaxed);
    g_calls[phase].fetch_add(1, std::memory_order_relaxed);
}

extern "C" void fg_pf_phase_ms(double * out, long * calls, int reset) {
    for (int i = 0; i < FG_PF_PHASE_N; ++i) {
        if (out) {
            out[i] = (double) g_ns[i].load(std::memory_order_relaxed) / 1e6;
        }
        if (calls) {
            calls[i] = g_calls[i].load(std::memory_order_relaxed);
        }
        if (reset) {
            g_ns[i].store(0, std::memory_order_relaxed);
            g_calls[i].store(0, std::memory_order_relaxed);
        }
    }
}
