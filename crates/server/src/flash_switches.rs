//! The Flash engine's `--flash-*` 0|1 switches as the server parses them (kept outside the
//! `flash`-gated engine module so the flag parsing is the same in every build).
//!
//! - `--flash-attn-tile`, `--flash-dense-gemm`, `--flash-expert-tile` (G4 prefill compute stack):
//!   default 1 (on). Applied on the fast path only (`FlashEngine::open`); the exact path never uses them.
//! - `--flash-warmup`: default 0. The hot-set warm-up READS (load at process start, state resume) run only
//!   when it is 1; `--flash-hotset` alone writes the hot-set file but does not warm from it.

/// Parsed values of the switches above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashSwitches {
    pub attn_tile: bool,
    pub dense_gemm: bool,
    pub expert_tile: bool,
    pub warmup: bool,
}

impl Default for FlashSwitches {
    fn default() -> Self {
        FlashSwitches { attn_tile: true, dense_gemm: true, expert_tile: true, warmup: false }
    }
}

impl FlashSwitches {
    /// Apply one `--flash-*` switch with its value. `Ok(false)` if `flag` is not one of these switches;
    /// `Err` if the value is not `0` or `1`.
    pub fn parse_flag(&mut self, flag: &str, value: &str) -> Result<bool, String> {
        let slot = match flag {
            "--flash-attn-tile" => &mut self.attn_tile,
            "--flash-dense-gemm" => &mut self.dense_gemm,
            "--flash-expert-tile" => &mut self.expert_tile,
            "--flash-warmup" => &mut self.warmup,
            _ => return Ok(false),
        };
        *slot = match value {
            "1" => true,
            "0" => false,
            v => return Err(format!("{flag} {v:?}: expected 0 or 1")),
        };
        Ok(true)
    }
}
