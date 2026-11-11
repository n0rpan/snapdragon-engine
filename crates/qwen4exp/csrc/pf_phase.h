// pf_phase: cheap per-phase clocks of the prefill custom ops (FP-9). Thread 0 of each op adds its own wall time.
#pragma once

#include <chrono>
#include <cstdint>

enum { FG_PF_PHASE_ATTN = 0, FG_PF_PHASE_DENSE_PACK = 1, FG_PF_PHASE_DENSE_MM = 2, FG_PF_PHASE_N = 3 };

void fg_pf_phase_add(int phase, int64_t ns);

// RAII: measures from construction to destruction and adds to `phase` only when `active` (callers pass ith == 0)
struct FgPfPhaseTimer {
    int phase;
    bool active;
    std::chrono::steady_clock::time_point t0;
    FgPfPhaseTimer(int phase_, bool active_)
        : phase(phase_), active(active_),
          t0(active_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point()) {}
    ~FgPfPhaseTimer() {
        if (active) {
            fg_pf_phase_add(phase, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - t0).count());
        }
    }
};

extern "C" {
// out[FG_PF_PHASE_N] = accumulated milliseconds per phase since the last reset; calls[FG_PF_PHASE_N] = op invocations (may be NULL)
void fg_pf_phase_ms(double * out, long * calls, int reset);
}
