//! Prefill ubatch planning: how a prompt of `n_new` new tokens is cut into passes.
//!
//! The dense kernels pick their tile shape from the parity of the ubatch size (even: 2x2,
//! odd: 1x1), so only even pieces may be merged. Full base-sized chunks are merged into
//! larger passes of up to `max_pass` tokens; `max_pass_tokens` sizes a pass against a
//! memory budget and `mem_available_bytes` reads the current headroom.

/// Estimated bytes per prefill token besides the attention partial tensor. Derived from ticket G4:
/// peak 23.84 GiB at 2,048 tokens with an 18 GiB cache and 4.62 GiB of dense weights and state,
/// i.e. about 0.4 MB per token besides the partial tensor; rounded up to 512 KiB.
pub const PER_TOKEN_BYTES: u64 = 512 * 1024;

/// Bytes of the attention partial tensor per (row, 256-cell block, head): [2 + head_dim] f32
/// with head_dim = 256, for 24 heads: (2 + 256) * 24 * 4.
pub const PARTIAL_BYTES_PER_ROW_BLOCK: u64 = (2 + 256) * 24 * 4;

/// Tokens in each prefill ubatch, in order; the sum is `n_new`.
///
/// `base` is the configured ubatch (treated as 1 when 0). When `max_pass <= base` or `base` is
/// odd, this is exactly `chunks(base)`. Otherwise full base chunks are merged `max_pass / base`
/// at a time, leftover full chunks form one pass, and the tail (`n_new % base`) is appended to
/// the last pass only when it is even, at least 256 tokens, and the merged pass still fits in
/// `max_pass`; otherwise the tail is its own final entry.
pub fn plan_ubatches(n_new: usize, base: usize, max_pass: usize) -> Vec<usize> {
    if n_new == 0 {
        return Vec::new();
    }
    let base = if base == 0 { 1 } else { base };
    let mut out: Vec<usize> = Vec::new();

    if max_pass <= base || base % 2 == 1 {
        let full = n_new / base;
        let tail = n_new % base;
        for _ in 0..full {
            out.push(base);
        }
        if tail > 0 {
            out.push(tail);
        }
        return out;
    }

    let full = n_new / base;
    let tail = n_new % base;
    // per >= 1 because max_pass > base.
    let per = max_pass / base;
    let mut left = full;
    while left >= per {
        out.push(per * base);
        left -= per;
    }
    if left > 0 {
        out.push(left * base);
    }

    if tail > 0 {
        let merge = match out.last() {
            Some(&last) => tail % 2 == 0 && tail >= 256 && last + tail <= max_pass,
            None => false,
        };
        if merge {
            let idx = out.len() - 1;
            out[idx] += tail;
        } else {
            out.push(tail);
        }
    }
    out
}

/// Estimated extra bytes of one prefill pass of `n` tokens starting after `n_past` tokens.
fn pass_bytes(n: usize, n_past: usize, fused_attn: bool) -> u64 {
    let mut bytes = n as u64 * PER_TOKEN_BYTES;
    if !fused_attn {
        let blocks = ((n_past + n + 255) / 256) as u64;
        bytes += n as u64 * blocks * PARTIAL_BYTES_PER_ROW_BLOCK;
    }
    bytes
}

/// Largest pass size (a multiple of `base`, at least `base`, at most `cap`) whose estimated extra
/// memory fits `budget_bytes`. If even `base` does not fit, returns `base`; the caller's existing
/// admission handles that case.
pub fn max_pass_tokens(base: usize, cap: usize, n_past: usize, budget_bytes: u64, fused_attn: bool) -> usize {
    let base = if base == 0 { 1 } else { base };
    if cap < base {
        return base;
    }
    let mut best = base;
    let mut n = base;
    loop {
        if pass_bytes(n, n_past, fused_attn) <= budget_bytes {
            best = n;
        }
        // Invariant: n <= cap. The next candidate n + base is within cap iff cap - n >= base.
        if cap - n < base {
            break;
        }
        n += base;
    }
    best
}

/// MemAvailable from /proc/meminfo in bytes (None if it cannot be read).
pub fn mem_available_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return kb.checked_mul(1024);
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------------------------
// F99 / W5-C / F118 / O-27 admission: `--flash-cache-gib auto` (tiers 20 -> 18 -> 16 -> 14 -> refuse), the user
// preference `--flash-cache-pref auto|20|18|16|14`, and the long-prompt pass guard (same harness + floor terms).
//
// need(cache) = cache + dense/base + model growth + harness growth allowance + floor, against MemAvailable read
// before anything is loaded. Constants are receipts:
// - dense/base 4.825 GiB: VmRSS after load at 16 GiB 20.825 GiB - 16 (G3' F99 16 GiB arm).
// - model growth 3.50 GiB: growth above the loaded size during the protected OpenCode task, 3.27 / 3.41 GiB RSS at
//   16 / 14 GiB (charge up to 3.47), rounded up as F118 proposes (062's 14 GiB peak beat the old 3.465 by 24 MiB);
//   measured at --ctx-size 16384, ubatch 4096. NOT measured at 18 or 20 GiB with the task running (F99's 18 GiB arm
//   tripped the floor); 064 measures 20.
// - harness growth allowance 0.80 GiB (F118): the largest positive growth envelope of OpenCode + Claude + Codex +
//   desktop during the three protected-task runs is 0.565 GiB; the processes already running are outside
//   MemAvailable, so only their growth counts (was 1.40 GiB full RSS, an upper bound that double-counted them).
// - floor: the 1 GiB host floor of the memory guard.
// ---------------------------------------------------------------------------------------------------------------

const GIB_F: f64 = (1u64 << 30) as f64;
/// Dense weights + base buffers above the expert cache after load [M G3' F99 16 GiB].
pub const ADMIT_DENSE_BASE_GIB: f64 = 4.825;
/// Model growth above the loaded size during the protected task at n_ctx 16384 [M 3.27-3.47, F118 rounded to 3.50].
pub const ADMIT_ANON_PEAK_GIB: f64 = 3.50;
/// n_ctx at which [`ADMIT_DENSE_BASE_GIB`] was measured (KV is allocated for n_ctx at load).
pub const ADMIT_MEASURED_CTX: usize = 16384;
/// KV bytes per context token: 12 attention layers x (K 512 + V 512 + indexer K 256) f16.
pub const KV_BYTES_PER_TOKEN: u64 = 12 * (512 + 512 + 256) * 2;
/// Harness GROWTH allowance [F118: max measured envelope 0.565 GiB, proposed 0.80].
pub const ADMIT_HARNESS_GIB: f64 = 0.80;
/// Host floor of the memory guard.
pub const ADMIT_FLOOR_GIB: f64 = 1.0;
/// Cache tiers, largest first: `auto` takes the largest tier that fits, a preference takes its tier when it fits and
/// otherwise the largest that fits; below the last tier the server refuses. Only 14 and 16 GiB have measured peaks
/// with the protected task; 18 / 20 use the same (cache-independent) terms.
pub const AUTO_CACHE_GIB: [f64; 4] = [20.0, 18.0, 16.0, 14.0];

/// Parse a `--flash-cache-pref` value: `auto` = None, else one of [`AUTO_CACHE_GIB`] (`Err` otherwise).
pub fn parse_cache_pref(v: &str) -> Result<Option<f64>, String> {
    if v == "auto" {
        return Ok(None);
    }
    match v.parse::<f64>() {
        Ok(g) if AUTO_CACHE_GIB.contains(&g) => Ok(Some(g)),
        _ => Err(format!("--flash-cache-pref {v:?}: auto or one of 20, 18, 16, 14")),
    }
}

/// Inputs of the load-time decision (all GiB except n_ctx).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmitInputs {
    pub mem_available_gib: f64,
    pub dense_base_gib: f64,
    pub anon_peak_gib: f64,
    pub harness_gib: f64,
    pub floor_gib: f64,
    pub n_ctx: usize,
}

impl AdmitInputs {
    /// The receipts above, with `mem_available_gib` and `n_ctx` from the caller and the harness allowance overridable.
    pub fn measured(mem_available_gib: f64, n_ctx: usize, harness_gib: Option<f64>) -> Self {
        AdmitInputs {
            mem_available_gib,
            dense_base_gib: ADMIT_DENSE_BASE_GIB,
            anon_peak_gib: ADMIT_ANON_PEAK_GIB,
            harness_gib: harness_gib.unwrap_or(ADMIT_HARNESS_GIB),
            floor_gib: ADMIT_FLOOR_GIB,
            n_ctx,
        }
    }

    /// KV above the measured context (0 at or below 16384 tokens).
    pub fn kv_extra_gib(&self) -> f64 {
        (self.n_ctx.saturating_sub(ADMIT_MEASURED_CTX) as u64 * KV_BYTES_PER_TOKEN) as f64 / GIB_F
    }

    /// GiB of MemAvailable a load with `cache_gib` needs.
    pub fn need_gib(&self, cache_gib: f64) -> f64 {
        cache_gib + self.dense_base_gib + self.kv_extra_gib() + self.anon_peak_gib + self.harness_gib + self.floor_gib
    }
}

/// Why a tier was chosen (the controller shows it next to the tier).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CacheReason {
    /// no preference: the largest tier that fits
    Auto,
    /// the preferred tier fits
    Preferred,
    /// an explicit 20 / 18 preference admitted by the swap-tolerant rule (MemTotal-based; idle pages may go to swap)
    PreferredSwapTolerant,
    /// the preferred tier does not fit: the largest tier that does
    PreferredDoesNotFit,
    /// not even the smallest tier fits
    Refuse,
}

impl CacheReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            CacheReason::Auto => "auto",
            CacheReason::Preferred => "preferred",
            CacheReason::PreferredDoesNotFit => "preferred-does-not-fit",
            CacheReason::PreferredSwapTolerant => "preferred-swap-tolerant",
            CacheReason::Refuse => "refuse",
        }
    }
}

/// Outcome of `--flash-cache-gib auto` / `--flash-cache-pref`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheDecision {
    /// chosen cache size; None = refuse (not even the smallest tier fits)
    pub cache_gib: Option<f64>,
    /// MemAvailable minus the need of the chosen tier (of the smallest tier when refused: then negative)
    pub margin_gib: f64,
    /// need of every tier, in [`AUTO_CACHE_GIB`] order
    pub needs_gib: [f64; 4],
    /// the user's preference (None = auto)
    pub pref_gib: Option<f64>,
    pub reason: CacheReason,
    /// MemAvailable the decision used
    pub mem_available_gib: f64,
}

/// `--flash-cache-gib auto` without a preference: the largest tier that fits, else refuse.
pub fn cache_auto(inp: &AdmitInputs) -> CacheDecision {
    cache_select(inp, None)
}

/// O-27 swap-tolerant rule (opt-in): only an explicit preference of 20 or 18 GiB uses it; `auto` and 16 / 14 stay on
/// the MemAvailable rule. Budget = MemTotal - (cache + dense/base + KV extra + model growth) - an active-desktop
/// resident floor (`--flash-desktop-floor-gib`, default 3.0): idle desktop / agent pages may be pushed to swap
/// (16 GiB swapfile + zswap); the engine's own scope stays swap.max = 0 (controller), so the model never swaps.
pub const SWAP_TOLERANT_TIERS_GIB: [f64; 2] = [20.0, 18.0];
pub const DESKTOP_FLOOR_GIB: f64 = 3.0;

/// MemTotal - (cache + dense/base + KV extra + model growth) - desktop floor, GiB (negative = refuse this tier).
pub fn swap_tolerant_budget_gib(inp: &AdmitInputs, mem_total_gib: f64, cache_gib: f64, desktop_floor_gib: f64) -> f64 {
    mem_total_gib - (cache_gib + inp.dense_base_gib + inp.kv_extra_gib() + inp.anon_peak_gib) - desktop_floor_gib
}

/// S2 (REVIEW-v0.1.1): the MemTotal rule assumes idle desktop pages can go to swap; without swap the OOM killer
/// picks desktop victims instead. Margin on top of what must leave RAM (lab `flash_cache_pref.SWAP_MARGIN_GIB`).
pub const SWAP_MARGIN_GIB: f64 = 1.0;

/// GiB that must leave RAM (to swap) for a `cache_gib` load: the working set (cache + dense/base + KV extra + model
/// growth + the 1 GiB host floor; no harness allowance) minus MemAvailable, >= 0. Mirrors the lab's
/// `flash_cache_pref.swap_needed_bytes`.
pub fn swap_needed_gib(inp: &AdmitInputs, cache_gib: f64) -> f64 {
    let working_set = cache_gib + inp.dense_base_gib + inp.kv_extra_gib() + inp.anon_peak_gib + inp.floor_gib;
    (working_set - inp.mem_available_gib).max(0.0)
}

/// SwapFree covers what must leave RAM plus [`SWAP_MARGIN_GIB`]; no or unknown swap (<= 0, NaN) never does.
/// Mirrors the lab's `flash_cache_pref.swap_covers`.
pub fn swap_covers(inp: &AdmitInputs, swap_free_gib: f64, cache_gib: f64) -> bool {
    swap_free_gib.is_finite() && swap_free_gib > 0.0 && swap_free_gib >= swap_needed_gib(inp, cache_gib) + SWAP_MARGIN_GIB
}

/// The decision with the opt-in swap-tolerant rule: an explicit preference of 20 (or 18) takes the largest
/// swap-tolerant tier <= the preference whose budget is >= 0 AND whose RAM overflow SwapFree covers (S2,
/// [`swap_covers`]); if none, the MemAvailable rule of [`cache_select`] (largest tier that fits, NOTE, refuse below
/// 14). Any other preference (auto, 16, 14) is exactly [`cache_select`]. Returns the decision and, for the
/// swap-tolerant tiers considered, (tier, budget).
pub fn cache_select_swap_tolerant(
    inp: &AdmitInputs,
    pref_gib: Option<f64>,
    mem_total_gib: f64,
    desktop_floor_gib: f64,
    swap_free_gib: f64,
) -> (CacheDecision, Vec<(f64, f64)>) {
    let mut seen = Vec::new();
    if let Some(p) = pref_gib.filter(|p| SWAP_TOLERANT_TIERS_GIB.contains(p)) {
        for t in SWAP_TOLERANT_TIERS_GIB.iter().copied().filter(|&t| t <= p) {
            let b = swap_tolerant_budget_gib(inp, mem_total_gib, t, desktop_floor_gib);
            seen.push((t, b));
            if mem_total_gib.is_finite() && b >= 0.0 && swap_covers(inp, swap_free_gib, t) {
                let mut d = cache_select(inp, Some(t));
                d.cache_gib = Some(t);
                d.reason = CacheReason::PreferredSwapTolerant;
                d.pref_gib = pref_gib;
                d.margin_gib = b;
                return (d, seen);
            }
        }
    }
    (cache_select(inp, pref_gib), seen)
}

/// MemTotal from /proc/meminfo in bytes.
pub fn mem_total_bytes() -> Option<u64> {
    meminfo_bytes("MemTotal:")
}

/// SwapFree from /proc/meminfo in bytes (S2).
pub fn swap_free_bytes() -> Option<u64> {
    meminfo_bytes("SwapFree:")
}

fn meminfo_bytes(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let rest = text.lines().find_map(|l| l.strip_prefix(key))?;
    rest.trim().split_whitespace().next()?.parse::<u64>().ok()?.checked_mul(1024)
}

/// `/proc/pressure/memory` -> (some avg10, full avg10) in percent.
pub fn parse_psi(text: &str) -> Option<(f64, f64)> {
    let field = |kind: &str| -> Option<f64> {
        let line = text.lines().find(|l| l.starts_with(kind))?;
        line.split_whitespace().find_map(|kv| kv.strip_prefix("avg10="))?.parse().ok()
    };
    Some((field("some")?, field("full")?))
}

/// Memory PSI now (some avg10, full avg10), or None when unreadable.
pub fn memory_psi() -> Option<(f64, f64)> {
    parse_psi(&std::fs::read_to_string("/proc/pressure/memory").ok()?)
}

/// PSI pass cap (O-27): under memory pressure a prefill pass gets smaller (less scratch, less to page): `full avg10`
/// above `thr_full` -> 256-token passes, `some avg10` above `thr_some` -> 1,024; else None. A threshold <= 0 is off.
pub fn psi_pass_cap(some10: f64, full10: f64, thr_some: f64, thr_full: f64) -> Option<usize> {
    if thr_full > 0.0 && full10 > thr_full {
        Some(256)
    } else if thr_some > 0.0 && some10 > thr_some {
        Some(1024)
    } else {
        None
    }
}

/// Split even passes (>= 512, halves stay even and >= 256) until each is <= `cap`; odd passes are never split
/// (F-W4I-2). Lossless class like [`admit_passes`].
pub fn cap_passes(plan: &[usize], cap: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(plan.len());
    for &p in plan {
        let mut piece = p;
        while piece > cap && piece % 4 == 0 && piece >= 512 {
            piece /= 2;
        }
        out.extend(std::iter::repeat(piece).take(p / piece));
    }
    out
}

/// The decision with a preference: the preferred tier when its need fits MemAvailable; otherwise the largest tier
/// that fits (a visible note in the log); refuse only when not even the smallest tier fits.
pub fn cache_select(inp: &AdmitInputs, pref_gib: Option<f64>) -> CacheDecision {
    let needs_gib = AUTO_CACHE_GIB.map(|c| inp.need_gib(c));
    let avail = inp.mem_available_gib;
    let fits = |i: usize| avail.is_finite() && needs_gib[i] <= avail;
    let largest = (0..AUTO_CACHE_GIB.len()).find(|&i| fits(i));
    let pref_i = pref_gib.and_then(|p| AUTO_CACHE_GIB.iter().position(|&c| c == p));
    let (pick, reason) = match (pref_i, largest) {
        (_, None) => (None, CacheReason::Refuse),
        (Some(i), Some(_)) if fits(i) => (Some(i), CacheReason::Preferred),
        (Some(_), Some(l)) => (Some(l), CacheReason::PreferredDoesNotFit),
        (None, Some(l)) => (Some(l), CacheReason::Auto),
    };
    let last = AUTO_CACHE_GIB.len() - 1;
    CacheDecision {
        cache_gib: pick.map(|i| AUTO_CACHE_GIB[i]),
        margin_gib: avail - needs_gib[pick.unwrap_or(last)],
        needs_gib,
        pref_gib,
        reason,
        mem_available_gib: avail,
    }
}

impl CacheDecision {
    /// The per-load log line (also printed by `--flash-admit-dry-run`): every tier's need and margin, then the choice.
    pub fn log_line(&self, inp: &AdmitInputs) -> String {
        let tiers: Vec<String> = AUTO_CACHE_GIB
            .iter()
            .zip(self.needs_gib.iter())
            .map(|(c, n)| format!("{c:.0}: need {n:.2} margin {:+.2}", inp.mem_available_gib - n))
            .collect();
        let pref = match self.pref_gib {
            Some(p) => format!("pref {p:.0}"),
            None => "pref auto".into(),
        };
        let choice = match (self.cache_gib, self.reason) {
            (None, _) => format!("refuse (margin {:+.2} GiB at {:.0} GiB)", self.margin_gib, AUTO_CACHE_GIB[AUTO_CACHE_GIB.len() - 1]),
            (Some(c), CacheReason::PreferredDoesNotFit) => format!(
                "NOTE: preferred {:.0} GiB does not fit (margin {:+.2} GiB): cache {c:.0} GiB instead (margin {:+.2} GiB)",
                self.pref_gib.unwrap_or(0.0),
                inp.mem_available_gib - inp.need_gib(self.pref_gib.unwrap_or(0.0)),
                self.margin_gib
            ),
            (Some(c), _) => format!("cache {c:.0} GiB (margin {:+.2} GiB)", self.margin_gib),
        };
        format!(
            "flash cache auto: MemAvailable {:.2} GiB; {pref}; need = cache + dense/base {:.3} + kv extra {:.3} + model growth {:.3} + harness growth {:.2} (F118) + floor {:.2}; tiers [{}] -> {}",
            inp.mem_available_gib,
            inp.dense_base_gib,
            inp.kv_extra_gib(),
            inp.anon_peak_gib,
            inp.harness_gib,
            inp.floor_gib,
            tiers.join("; "),
            choice
        )
    }

    /// One JSON object for the controller's status (`flash_cache` field): chosen tier, reason, preference, margins.
    /// F127-3: the memory-guard mode this decision implies for the controller, known before the model loads:
    /// `swap-tolerant` (an explicit 20 / 18 preference admitted by the MemTotal rule), `memavailable` (every other
    /// admitted tier: the 1 GiB host floor) or `refuse`.
    pub fn swap_guard_mode(&self) -> &'static str {
        match (self.cache_gib, self.reason) {
            (None, _) => "refuse",
            (Some(_), CacheReason::PreferredSwapTolerant) => "swap-tolerant",
            (Some(_), _) => "memavailable",
        }
    }

    pub fn status_json(&self) -> String {
        let num = |x: f64| if x.is_finite() { format!("{x:.4}") } else { "null".into() };
        let opt = |x: Option<f64>| x.map_or("null".to_string(), |v| format!("{v:.0}"));
        let needs: Vec<String> = self.needs_gib.iter().map(|&n| num(n)).collect();
        let tiers: Vec<String> = AUTO_CACHE_GIB.iter().map(|&c| format!("{c:.0}")).collect();
        format!(
            "{{\"cache_gib\":{},\"pref_gib\":{},\"reason\":\"{}\",\"swap_guard\":\"{}\",\"refuse\":{},\"margin_gib\":{},\"mem_available_gib\":{},\"tiers_gib\":[{}],\"needs_gib\":[{}]}}",
            opt(self.cache_gib),
            opt(self.pref_gib),
            self.reason.as_str(),
            self.swap_guard_mode(),
            self.cache_gib.is_none(),
            num(self.margin_gib),
            num(self.mem_available_gib),
            tiers.join(","),
            needs.join(",")
        )
    }
}

/// Long-prompt guard: split each planned pass whose NEW memory would not fit into even halves (>= 256 tokens).
///
/// `plan` is the pass plan (from [`plan_ubatches`]); `n_past0` the tokens already in the context; `budget_bytes`
/// = MemAvailable - harness - floor at the call; `high_water` = the largest pass estimate this process has already
/// run (its compute buffers stay allocated, so only the part above it is new memory). Only even passes of >= 512
/// tokens are split, into two even halves, repeatedly: QC-FP8 showed 2 x 256 and 1 x 512 give the same state, and
/// odd passes keep their kernel class (F-W4I-2), so the split stays lossless class. Returns the new plan.
pub fn admit_passes(plan: &[usize], n_past0: usize, budget_bytes: u64, high_water: u64, fused_attn: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(plan.len());
    let mut n_past = n_past0;
    let mut hw = high_water;
    for &p in plan {
        let mut piece = p;
        // judged at the deepest piece (the largest partial tensor of the split)
        while piece % 4 == 0 && piece >= 512 && pass_bytes(piece, n_past + p - piece, fused_attn).saturating_sub(hw) > budget_bytes {
            piece /= 2;
        }
        // `piece` divides `p` (p / piece is a power of two)
        for _ in 0..p / piece {
            out.push(piece);
            hw = hw.max(pass_bytes(piece, n_past, fused_attn));
            n_past += piece;
        }
    }
    out
}

/// Estimated bytes of one pass (public for the server's high-water bookkeeping).
pub fn pass_bytes_est(n: usize, n_past: usize, fused_attn: bool) -> u64 {
    pass_bytes(n, n_past, fused_attn)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// What `chunks(base)` gives for `n` tokens.
    fn chunks_plan(n: usize, base: usize) -> Vec<usize> {
        let mut out = Vec::new();
        if n == 0 {
            return out;
        }
        let mut rest = n;
        while rest > 0 {
            let take = if rest < base { rest } else { base };
            out.push(take);
            rest -= take;
        }
        out
    }

    #[test]
    fn zero_tokens_gives_empty_plan() {
        assert!(plan_ubatches(0, 2048, 8192).is_empty());
        assert!(plan_ubatches(0, 0, 0).is_empty());
    }

    #[test]
    fn sum_always_equals_n_new() {
        for &base in [256usize, 2048, 4096].iter() {
            for &max_pass in [0usize, 2048, 4096, 8192].iter() {
                for n in (0usize..=9000).step_by(37) {
                    let plan = plan_ubatches(n, base, max_pass);
                    let sum: usize = plan.iter().sum();
                    assert_eq!(sum, n, "base {} max_pass {} n {}", base, max_pass, n);
                }
            }
        }
    }

    #[test]
    fn no_merge_matches_chunks() {
        // max_pass <= base: exactly chunks(base).
        for &base in [256usize, 2048, 4096].iter() {
            for &max_pass in [0usize, base].iter() {
                for n in (0usize..=9000).step_by(37) {
                    assert_eq!(plan_ubatches(n, base, max_pass), chunks_plan(n, base),
                        "base {} max_pass {} n {}", base, max_pass, n);
                }
            }
        }
        // Odd base: exactly chunks(base) even with a large max_pass.
        for &base in [1usize, 255, 2047].iter() {
            for n in (0usize..=9000).step_by(37) {
                assert_eq!(plan_ubatches(n, base, 8192), chunks_plan(n, base),
                    "odd base {} n {}", base, n);
            }
        }
    }

    #[test]
    fn zero_base_is_treated_as_one() {
        assert_eq!(plan_ubatches(5, 0, 8192), vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn entries_have_expected_shape() {
        for &base in [256usize, 2048, 4096].iter() {
            for &max_pass in [0usize, 2048, 4096, 8192].iter() {
                let cap = if max_pass > base { max_pass } else { base };
                for n in (0usize..=9000).step_by(37) {
                    let plan = plan_ubatches(n, base, max_pass);
                    let last = plan.len().saturating_sub(1);
                    for (i, &e) in plan.iter().enumerate() {
                        assert!(e > 0, "zero entry, base {} max_pass {} n {}", base, max_pass, n);
                        assert!(e <= cap, "entry {} over cap {}, base {} max_pass {} n {}",
                            e, cap, base, max_pass, n);
                        // Every entry except a final tail is a multiple of base, or a multiple of
                        // base plus an even tail of at least 256 tokens.
                        if i < last {
                            let r = e % base;
                            assert!(r == 0 || (r % 2 == 0 && r >= 256),
                                "bad interior entry {} base {} max_pass {} n {}", e, base, max_pass, n);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn merged_tail_case() {
        // 2 full chunks (4096) + tail 1404: even, >= 256, 5500 <= 8192 -> merged.
        assert_eq!(plan_ubatches(5500, 2048, 8192), vec![5500]);
    }

    #[test]
    fn odd_tail_is_not_merged() {
        assert_eq!(plan_ubatches(5501, 2048, 8192), vec![4096, 1405]);
    }

    #[test]
    fn short_tail_is_not_merged() {
        assert_eq!(plan_ubatches(4096 + 100, 2048, 8192), vec![4096, 100]);
    }

    #[test]
    fn full_passes_then_remainder() {
        assert_eq!(plan_ubatches(2048 * 5, 2048, 8192), vec![8192, 2048]);
    }

    #[test]
    fn tail_does_not_fit_in_last_pass() {
        // 8192 + 1000 > 8192, so the tail stays separate.
        assert_eq!(plan_ubatches(2048 * 4 + 1000, 2048, 8192), vec![8192, 1000]);
    }

    #[test]
    fn prompt_shorter_than_base() {
        assert_eq!(plan_ubatches(1000, 2048, 8192), vec![1000]);
    }

    #[test]
    fn odd_base_does_not_merge() {
        assert_eq!(plan_ubatches(3000, 2047, 8192), vec![2047, 953]);
    }

    #[test]
    fn max_pass_tokens_fused_huge_budget_gives_largest_multiple() {
        assert_eq!(max_pass_tokens(2048, 8192, 0, u64::MAX, true), 8192);
        assert_eq!(max_pass_tokens(2048, 5000, 0, u64::MAX, true), 4096);
        assert_eq!(max_pass_tokens(256, 1000, 0, u64::MAX, true), 768);
    }

    #[test]
    fn max_pass_tokens_cap_below_base_returns_base() {
        assert_eq!(max_pass_tokens(2048, 1000, 0, u64::MAX, true), 2048);
        assert_eq!(max_pass_tokens(2048, 1000, 0, 0, false), 2048);
    }

    #[test]
    fn max_pass_tokens_none_fits_returns_base() {
        assert_eq!(max_pass_tokens(2048, 8192, 0, 0, true), 2048);
    }

    #[test]
    fn max_pass_tokens_is_monotone_in_budget() {
        let mut prev = 0usize;
        let mut budget: u64 = 0;
        while budget <= 6 * GIB {
            let got = max_pass_tokens(2048, 8192, 0, budget, false);
            assert!(got >= prev, "budget {} gave {} after {}", budget, got, prev);
            assert!(got % 2048 == 0 && got >= 2048 && got <= 8192);
            prev = got;
            budget += GIB / 16;
        }
        let mut prev = 0usize;
        let mut budget: u64 = 0;
        while budget <= 6 * GIB {
            let got = max_pass_tokens(2048, 8192, 4096, budget, true);
            assert!(got >= prev);
            prev = got;
            budget += GIB / 16;
        }
    }

    #[test]
    fn max_pass_tokens_unfused_3gib() {
        // Hand arithmetic (n_past = 0, base 2048, cap 8192, unfused: blocks = ceil(n / 256) for n <= 4096 region):
        // n = 2048: per-token 2048 * 512 KiB = 1 GiB = 1,073,741,824 B.
        //           blocks = (0 + 2048 + 255) / 256 = 8; partial = 2048 * 8 * 24768 = 405,798,912 B.
        //           total = 1,479,540,736 B <= 3 GiB (3,221,225,472 B): fits.
        // n = 4096: per-token 4096 * 512 KiB = 2 GiB = 2,147,483,648 B.
        //           blocks = (0 + 4096 + 255) / 256 = 16; partial = 4096 * 16 * 24768 = 1,623,195,648 B.
        //           total = 3,770,679,296 B (about 3.51 GiB) > 3 GiB: does not fit.
        // So the largest fitting candidate is 2048, not 4096.
        assert_eq!(max_pass_tokens(2048, 8192, 0, 3 * GIB, false), 2048);
    }

    #[test]
    fn max_pass_tokens_unfused_exact_boundary() {
        // Exact 4096 estimate from the hand arithmetic above: 3,770,679,296 B.
        assert_eq!(max_pass_tokens(2048, 8192, 0, 3_770_679_296, false), 4096);
        assert_eq!(max_pass_tokens(2048, 8192, 0, 3_770_679_295, false), 2048);
    }

    #[test]
    fn max_pass_tokens_fused_3gib() {
        // Fused: per-token only. 6144 * 512 KiB = 3 GiB = 3,221,225,472 B fits exactly;
        // 8192 * 512 KiB = 4 GiB does not.
        assert_eq!(max_pass_tokens(2048, 8192, 0, 3 * GIB, true), 6144);
    }

    #[test]
    fn mem_available_is_readable_on_linux() {
        let x = mem_available_bytes().expect("MemAvailable should be readable on Linux");
        assert!(x > 0);
    }

    // ---- W5-C admission (model-free) ----
    fn inp(avail: f64, n_ctx: usize) -> AdmitInputs {
        AdmitInputs::measured(avail, n_ctx, None)
    }

    #[test]
    fn admit_need_matches_receipts() {
        let i = inp(31.0, 16384);
        // cache + 4.825 + 0 + 3.50 + 0.80 + 1.0 (F118 table: 24.125 / 26.125 / 28.125)
        for (c, n) in [(20.0, 30.125), (18.0, 28.125), (16.0, 26.125), (14.0, 24.125)] {
            assert!((i.need_gib(c) - n).abs() < 1e-9, "{c}: {}", i.need_gib(c));
        }
        assert!((ADMIT_HARNESS_GIB - 0.80).abs() < 1e-12);
    }

    #[test]
    fn cache_auto_tier_20() {
        let i = inp(inp(0.0, 16384).need_gib(20.0), 16384); // exactly the need: fits
        let d = cache_auto(&i);
        assert_eq!((d.cache_gib, d.reason), (Some(20.0), CacheReason::Auto));
        assert!(d.margin_gib.abs() < 1e-9);
        assert!(d.log_line(&i).contains("cache 20 GiB"));
    }

    #[test]
    fn cache_auto_tier_18() {
        let d = cache_auto(&inp(30.12, 16384));
        assert_eq!(d.cache_gib, Some(18.0));
        assert!((d.margin_gib - 1.995).abs() < 1e-9);
        assert_eq!(cache_auto(&inp(inp(0.0, 16384).need_gib(18.0), 16384)).cache_gib, Some(18.0));
    }

    #[test]
    fn cache_auto_tier_16() {
        assert_eq!(cache_auto(&inp(28.12, 16384)).cache_gib, Some(16.0));
        // F118: F99's 16 GiB preload 26.335 admits 16 now (was 14 under the full-RSS 1.40 term)
        let d = cache_auto(&inp(26.335270, 16384));
        assert_eq!(d.cache_gib, Some(16.0));
        assert!((d.margin_gib - 0.21027).abs() < 1e-6);
    }

    #[test]
    fn cache_auto_tier_14() {
        // F122's real desktop (25.6 free) and F118's quiet QC-ADM samples (25.45-25.50) -> 14
        for a in [25.6, 25.453991, 26.063148, inp(0.0, 16384).need_gib(14.0)] {
            assert_eq!(cache_auto(&inp(a, 16384)).cache_gib, Some(14.0), "{a}");
        }
        let i = inp(25.6, 16384);
        let d = cache_auto(&i);
        assert!((d.needs_gib[3] - 24.125).abs() < 1e-9 && (d.margin_gib - 1.475).abs() < 1e-9);
        assert!(d.log_line(&i).contains("cache 14 GiB (margin +1.4"), "{}", d.log_line(&i));
    }

    #[test]
    fn cache_auto_refuses_below_14() {
        let i = inp(24.12, 16384);
        let d = cache_auto(&i);
        assert_eq!((d.cache_gib, d.reason), (None, CacheReason::Refuse));
        assert!(d.margin_gib < 0.0 && (d.margin_gib + 0.005).abs() < 1e-9);
        assert!(d.log_line(&i).contains("-> refuse"));
        assert!(d.status_json().contains("\"refuse\":true") && d.status_json().contains("\"cache_gib\":null"));
        // unreadable MemAvailable (NaN) refuses
        assert_eq!(cache_auto(&inp(f64::NAN, 16384)).cache_gib, None);
    }

    #[test]
    fn cache_pref_honoured_when_it_fits() {
        // a smaller preference is honoured even when a larger tier fits
        for p in AUTO_CACHE_GIB {
            let d = cache_select(&inp(30.2, 16384), Some(p));
            assert_eq!((d.cache_gib, d.reason), (Some(p), CacheReason::Preferred), "{p}");
        }
        let i = inp(26.2, 16384);
        let d = cache_select(&i, Some(16.0));
        assert_eq!((d.cache_gib, d.reason), (Some(16.0), CacheReason::Preferred));
        assert!(d.status_json().contains("\"reason\":\"preferred\"") && d.status_json().contains("\"pref_gib\":16"));
    }

    #[test]
    fn cache_pref_20_falls_back_to_largest_that_fits() {
        // pref 20 on the measured desktop: 20 / 18 / 16 do not fit, 14 does -> 14 with a visible note, never refuse
        let i = inp(25.6, 16384);
        let d = cache_select(&i, Some(20.0));
        assert_eq!((d.cache_gib, d.reason), (Some(14.0), CacheReason::PreferredDoesNotFit));
        let line = d.log_line(&i);
        assert!(line.contains("NOTE: preferred 20 GiB does not fit (margin -4.5") && line.contains("cache 14 GiB instead"), "{line}");
        assert!(d.status_json().contains("\"reason\":\"preferred-does-not-fit\""));
        // pref 20 with 18 fitting -> 18; pref 18 with 16 fitting -> 16
        assert_eq!(cache_select(&inp(29.0, 16384), Some(20.0)).cache_gib, Some(18.0));
        assert_eq!(cache_select(&inp(27.0, 16384), Some(18.0)).cache_gib, Some(16.0));
        // pref 16 when only 14 fits -> 14 (not refuse)
        assert_eq!(cache_select(&inp(24.2, 16384), Some(16.0)).cache_gib, Some(14.0));
        // a preference never makes it refuse while 14 fits, and refuses like auto below 14
        assert_eq!(cache_select(&inp(inp(0.0, 16384).need_gib(14.0), 16384), Some(20.0)).cache_gib, Some(14.0));
        assert_eq!(cache_select(&inp(24.0, 16384), Some(14.0)).reason, CacheReason::Refuse);
    }

    #[test]
    fn cache_pref_parse() {
        assert_eq!(parse_cache_pref("auto"), Ok(None));
        for (v, g) in [("20", 20.0), ("18", 18.0), ("16", 16.0), ("14", 14.0)] {
            assert_eq!(parse_cache_pref(v), Ok(Some(g)));
        }
        for bad in ["21", "12", "abc", "", "-16"] {
            assert!(parse_cache_pref(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn cache_auto_counts_kv_above_16k_and_harness_override() {
        let i = inp(28.5, 32768);
        let kv = (16384u64 * KV_BYTES_PER_TOKEN) as f64 / GIB_F;
        assert!((i.kv_extra_gib() - kv).abs() < 1e-12 && kv > 0.4 && kv < 0.5);
        assert_eq!(cache_auto(&inp(28.5, 16384)).cache_gib, Some(18.0));
        assert_eq!(cache_auto(&i).cache_gib, Some(16.0)); // 28.125 + 0.47 > 28.5
        // the old full-RSS term as an override
        let o = AdmitInputs::measured(28.5, 16384, Some(1.40));
        assert_eq!(cache_auto(&o).cache_gib, Some(16.0));
    }

    #[test]
    fn guard_budget_matches_each_tier() {
        // The guard's budget subtracts the same harness growth + floor as the load decision. At each tier, loading at
        // exactly that tier's need leaves model growth (3.50) + harness + floor free after load, so the guard's budget
        // is the 3.50 GiB model growth: the first 4,096 pass (3.51 GiB est.) is split once, never below 2,048; deep in
        // 16K context it is split further. Same at every tier (the cache cancels out).
        for c in AUTO_CACHE_GIB {
            let need = inp(31.0, 16384).need_gib(c);
            let after_load = need - c - ADMIT_DENSE_BASE_GIB;
            let budget = ((after_load - ADMIT_HARNESS_GIB - ADMIT_FLOOR_GIB) * GIB_F) as u64;
            assert!(budget.abs_diff((ADMIT_ANON_PEAK_GIB * GIB_F) as u64) < 4096, "{c}");
            let first = admit_passes(&[4096], 0, budget, 0, false);
            assert!(first.iter().all(|&p| p >= 2048) && first.iter().sum::<usize>() == 4096, "{c}: {first:?}");
            let deep = admit_passes(&[4096], 12288, budget, 0, false);
            assert!(deep.len() >= 2 && deep.iter().sum::<usize>() == 4096, "{c}: {deep:?}");
        }
        // F122's desktop at 14: 25.6 free -> 6.775 after load -> 4.975 budget: the first 4,096 pass runs unsplit
        let budget = (((25.6 - 14.0 - ADMIT_DENSE_BASE_GIB) - ADMIT_HARNESS_GIB - ADMIT_FLOOR_GIB) * GIB_F) as u64;
        assert_eq!(admit_passes(&[4096], 0, budget, 0, false), vec![4096]);
    }

    #[test]
    fn admit_passes_keeps_plan_when_it_fits() {
        let plan = plan_ubatches(5500, 4096, 0);
        assert_eq!(admit_passes(&plan, 0, 64 * GIB, 0, false), plan);
        // already-allocated high water counts: same pass again needs no new memory
        let hw = pass_bytes_est(4096, 0, false);
        assert_eq!(admit_passes(&[4096], 0, 0, hw, false), vec![4096]);
    }

    #[test]
    fn admit_passes_halves_even_passes_and_keeps_tokens_and_tail() {
        // 5,500 tokens at ubatch 4096: [4096, 1404]; a tight budget halves the full pass, the tail (1404 = 4 x 351) too
        let plan = plan_ubatches(5500, 4096, 0);
        let out = admit_passes(&plan, 0, GIB, 0, false);
        assert_eq!(out.iter().sum::<usize>(), 5500);
        assert!(out.iter().all(|&p| p % 2 == 0 && p >= 256), "{out:?}");
        assert!(out.len() > plan.len());
        for &p in &out[..out.len() - 1] {
            assert!(4096 % p == 0, "{out:?}");
        }
        // an odd tail is never split (kernel class follows ubatch parity, F-W4I-2)
        let out = admit_passes(&[4096, 1403], 0, 0, 0, false);
        assert_eq!(*out.last().unwrap(), 1403);
        assert_eq!(out.iter().sum::<usize>(), 4096 + 1403);
        assert!(out[..out.len() - 1].iter().all(|&p| p == 256));
        // a zero budget stops at 256-token pieces, never below
        assert_eq!(admit_passes(&[1024], 0, 0, 0, false), vec![256; 4]);
        assert_eq!(admit_passes(&[256], 0, 0, 0, false), vec![256]);
        assert_eq!(admit_passes(&[300], 0, 0, 0, false), vec![300]);
    }

    #[test]
    fn admit_passes_depth_matters_without_fused_attention() {
        // deep in the context the partial tensor dominates (6.5 GB at 4,096 rows x 64 blocks): split there, not at the start
        let b = 4 * GIB;
        assert_eq!(admit_passes(&[4096], 0, b, 0, false), vec![4096]);
        let deep = admit_passes(&[4096], 12288, b, 0, false);
        assert!(deep.len() >= 2 && deep.iter().sum::<usize>() == 4096, "{deep:?}");
        // the fused node (attn-tile 2) has no partial tensor: no split
        assert_eq!(admit_passes(&[4096], 12288, b, 0, true), vec![4096]);
    }

    // ---- O-27 swap-tolerant preference + PSI guard (model-free) ----
    const MT: f64 = 30.72; // this machine's MemTotal (32,215,352 kB)

    #[test]
    fn swap_tolerant_budget_terms() {
        let i = inp(25.5, 16384);
        // 30.72 - (20 + 4.825 + 3.50) - 3.0 = -0.605; 18: +1.395
        assert!((swap_tolerant_budget_gib(&i, MT, 20.0, 3.0) + 0.605).abs() < 1e-9);
        assert!((swap_tolerant_budget_gib(&i, MT, 18.0, 3.0) - 1.395).abs() < 1e-9);
        assert!((swap_tolerant_budget_gib(&i, MT, 20.0, 2.0) - 0.395).abs() < 1e-9);
    }

    #[test]
    fn pref_20_swap_tolerant_admits_20_when_budget_nonnegative() {
        // MemAvailable 25.5 (the 1st rule would give 14) but MemTotal admits 20 with a 2 GiB desktop floor
        let (d, seen) = cache_select_swap_tolerant(&inp(25.5, 16384), Some(20.0), MT, 2.0, 16.0);
        assert_eq!((d.cache_gib, d.reason), (Some(20.0), CacheReason::PreferredSwapTolerant));
        assert!((d.margin_gib - 0.395).abs() < 1e-9 && seen.len() == 1);
        assert!(d.status_json().contains("\"reason\":\"preferred-swap-tolerant\""));
    }

    #[test]
    fn pref_20_falls_to_18_then_memavailable_rule() {
        // 3 GiB floor: 20 budget -0.605 -> 18 (+1.395)
        let (d, seen) = cache_select_swap_tolerant(&inp(25.5, 16384), Some(20.0), MT, 3.0, 16.0);
        assert_eq!((d.cache_gib, d.reason), (Some(18.0), CacheReason::PreferredSwapTolerant));
        assert_eq!(seen.len(), 2);
        // 5 GiB floor: 20 and 18 negative -> MemAvailable rule (25.5 -> 14, NOTE)
        let (d, _) = cache_select_swap_tolerant(&inp(25.5, 16384), Some(20.0), MT, 5.0, 16.0);
        assert_eq!((d.cache_gib, d.reason), (Some(14.0), CacheReason::PreferredDoesNotFit));
        // and refuse only when the MemAvailable rule refuses too
        let (d, _) = cache_select_swap_tolerant(&inp(24.0, 16384), Some(20.0), MT, 5.0, 16.0);
        assert_eq!(d.reason, CacheReason::Refuse);
        // unreadable MemTotal never admits the swap-tolerant tiers
        let (d, _) = cache_select_swap_tolerant(&inp(25.5, 16384), Some(20.0), f64::NAN, 0.0, 16.0);
        assert_eq!(d.cache_gib, Some(14.0));
    }

    /// REVIEW-v0.1.1 S2: total 32, available 12, no swap -> never PreferredSwapTolerant (the OOM killer would pick
    /// desktop victims); with enough swap the MemTotal rule still applies. Same numbers as the lab's swap_covers.
    #[test]
    fn swap_tolerant_rule_needs_free_swap() {
        let (d, _) = cache_select_swap_tolerant(&inp(12.0, 16384), Some(20.0), 32.0, 0.0, 0.0);
        assert_ne!(d.reason, CacheReason::PreferredSwapTolerant);
        let (d, _) = cache_select_swap_tolerant(&inp(12.0, 16384), Some(20.0), 32.0, 0.0, f64::NAN);
        assert_ne!(d.reason, CacheReason::PreferredSwapTolerant);
        // 20 needs 20 + 4.825 + 3.50 + 1.0 = 29.325 GiB resident: 17.325 must leave RAM, + 1 GiB margin
        let i = inp(12.0, 16384);
        assert!((swap_needed_gib(&i, 20.0) - 17.325).abs() < 1e-9);
        assert!(!swap_covers(&i, 18.3, 20.0) && swap_covers(&i, 18.325, 20.0));
        // short of 20 but enough for 18 (15.325 + 1) -> 18
        let (d, _) = cache_select_swap_tolerant(&i, Some(20.0), 32.0, 0.0, 16.5);
        assert_eq!((d.cache_gib, d.reason), (Some(18.0), CacheReason::PreferredSwapTolerant));
        let (d, _) = cache_select_swap_tolerant(&i, Some(20.0), 32.0, 0.0, 18.5);
        assert_eq!((d.cache_gib, d.reason), (Some(20.0), CacheReason::PreferredSwapTolerant));
        // nothing has to leave RAM: the margin alone is enough
        assert!(swap_covers(&inp(40.0, 16384), 1.0, 20.0) && !swap_covers(&inp(40.0, 16384), 0.5, 20.0));
    }

    #[test]
    fn swap_tolerant_rule_is_opt_in_only() {
        // auto, 16 and 14 never use MemTotal: identical to cache_select
        for p in [None, Some(16.0), Some(14.0)] {
            let i = inp(25.5, 16384);
            let (d, seen) = cache_select_swap_tolerant(&i, p, MT, 0.0, 16.0);
            assert_eq!(d, cache_select(&i, p), "{p:?}");
            assert!(seen.is_empty());
        }
        // pref 18 considers only 18 (never 20)
        let (d, seen) = cache_select_swap_tolerant(&inp(25.5, 16384), Some(18.0), MT, 0.0, 16.0);
        assert_eq!((d.cache_gib, seen.len()), (Some(18.0), 1));
    }

    #[test]
    fn psi_parse_and_cap() {
        let t = "some avg10=12.50 avg60=3.10 avg300=0.80 total=123\nfull avg10=0.40 avg60=0.10 avg300=0.00 total=45\n";
        assert_eq!(parse_psi(t), Some((12.5, 0.4)));
        assert_eq!(parse_psi("garbage"), None);
        assert_eq!(psi_pass_cap(12.5, 0.4, 10.0, 2.0), Some(1024));
        assert_eq!(psi_pass_cap(12.5, 2.5, 10.0, 2.0), Some(256));
        assert_eq!(psi_pass_cap(5.0, 0.0, 10.0, 2.0), None);
        assert_eq!(psi_pass_cap(99.0, 99.0, 0.0, 0.0), None); // off
        assert!(memory_psi().is_some(), "PSI should be readable on this kernel");
    }

    #[test]
    fn cap_passes_keeps_tokens_parity_and_floor() {
        assert_eq!(cap_passes(&[4096, 1404], 1024), [vec![1024; 4], vec![702, 702]].concat());
        assert_eq!(cap_passes(&[4096], 256), vec![256; 16]);
        assert_eq!(cap_passes(&[4096, 1403], 256).last(), Some(&1403));
        assert_eq!(cap_passes(&[300, 17], 256), vec![300, 17]);
        assert_eq!(cap_passes(&[2048], 4096), vec![2048]);
    }
}
