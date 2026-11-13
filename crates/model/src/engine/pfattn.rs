//! v0.1.3 PREFILL-LONG (LC-9): flash-attention-2 style CPU prefill attention over the f16 KV
//! cache, with scratch O(chunk) per thread instead of O(n_ctx).
//!
//! The tiled prefill attention (`Engine::pf_attention_tiled`) writes the f32 scores of a unit's
//! query columns over ALL n_kv positions, softmaxes each column into f16 probabilities and then
//! runs V^T . p: per prefill thread `att_tt * group * n_ctx * 6` bytes of score / probability
//! tiles plus four n_ctx rows (~0.9 GiB over 12 threads at 128K, ~1.9 GiB at 256K), and every
//! score row goes through memory twice.
//!
//! Here one unit is (KV head, the unit's query columns = up to 8 tokens x the 8 query heads of the
//! group = 64 columns). The unit walks the positions in fixed chunks of [`PFA_CH`] (256) from 0:
//! per chunk the scores of all columns (`oxe_dot_f16_tile`, each K row read once for the unit), a
//! chunk-local softmax with f16 weights exp(x - m_chunk) and the chunk's V^T . p
//! (`oxe_attn_fchunk_f16`, the O-70 fused decode kernel: scores, weights and the V^T segment stay
//! in L1), then an online merge into f32 accumulators (running max M, running sum L, outputs
//! rescaled by exp(M_old - M_new)). One normalisation at the end. Numerics class: the O-70 fused
//! decode (chunk-local f16 accumulation, f32 combination), NOT the ggml FA-off order of the tiled
//! path, so it is opt-in (`OXE_PF_ATTN=fa2`) and the numerics epoch does not move until it becomes
//! a default path. The merge is online (running max) rather than O-70's two-pass merge, so the
//! two are the same class but not the same bits.
//!
//! Every unit is computed by one thread from position 0 upwards in fixed chunks: the result is
//! independent of the thread count and of which thread runs it. Chunk boundaries are absolute
//! positions and the last chunk's V^T dot uses O-70's tail rule (length up to the chunk end
//! within n_ctx), so a column's output does not depend on how far the unit's last token reaches.
//!
//! Selection (`Engine`): `OXE_PF_ATTN=fa2|tiled|auto` (unset = auto: fa2 only when the engine's
//! n_ctx > [`PFA_AUTO_MIN_CTX`], i.e. only a 256K-class engine, where the O(n_ctx) scratch is what
//! does not fit). Even with fa2 active, a unit whose last token's padded n_kv is at most
//! [`PFA_MIN_KV`] keeps today's tiled kernel bit for bit (short prompts never change).
//!
//! v0.1.3 APPEND-PREFILL (`OXE_PF_ATTN=fa2x`, opt-in): the same units, cutoffs and scratch policy,
//! but each unit runs `oxe_pfa2x_unit` (csrc/oxe_pfa2x.cpp): an 8x16 register-blocked f16 FMLA
//! micro-kernel for both S = K q and O += V^T P (K / V^T rows broadcast by lane, Q^T / P^T rows of
//! 16 columns reused by 8 rows: ~0.13 loads per FMLA instead of ~0.75 in `oxe_dot_f16_tile`, no
//! horizontal reductions), the online softmax vertical over 4-column f32 vectors, f16 chains of
//! [`FA2X_FOLD`] terms folded into f32 (shorter f16 chains than ggml's / fa2's), chunks of
//! [`FA2X_CH`] absolute positions. Same invariances as fa2 (thread count, unit split, tail rule);
//! a different numerics class from both fa2 and tiled (its own `pf_attn=fa2x` state identity).
//! `OXE_PF_FA2X_FOLD` (8 | 16 | 32) and `OXE_PF_FA2X_CH` (multiple of 32, <= 512) tune it.
//!
//! v0.1.3 PREFILL-POLICY (steering 11-12 evening #2): what `auto` resolves to is configurable
//! ([`AutoPolicy`]: `OXE_PF_ATTN_AUTO_MIN_KV` / `--pf-attn-auto-min-kv N`, `OXE_PF_ATTN_AUTO_KERNEL`
//! / `--pf-attn-auto-kernel fa2|fa2x`). With a min-kv N, an engine whose n_ctx exceeds the cutoff
//! `clamp(N, PFA_MIN_KV, PFA_AUTO_MIN_CTX)` runs the auto kernel for units whose padded n_kv exceeds
//! that cutoff and the tiled kernel (bit for bit) below it. Unset ([`PFA_AUTO_MIN_KV`] = None, the
//! PLACEHOLDER default until AP-1 measures the crossover) keeps the v0.1.3-lane-A rule exactly:
//! fa2 beyond PFA_MIN_KV only when n_ctx > PFA_AUTO_MIN_CTX. Every mode resolves to one
//! [`PfPath`] (kernel + cutoff) per engine; the state identity keys on the resolved path.

use kernels_sys::ffi;

/// Positions per chunk (a multiple of 32; the O-70 fused default).
pub const PFA_CH: usize = 256;
/// Units whose last token's padded n_kv is at most this keep the tiled kernel (bit-identical to
/// today); with fa2 active the context-sized prefill scratch is sized for this many positions.
pub const PFA_MIN_KV: usize = 8192;
/// `OXE_PF_ATTN` unset / `auto`: fa2 only for engines with n_ctx above this (256K class).
pub const PFA_AUTO_MIN_CTX: usize = 131_072;
/// PREFILL-POLICY: `auto`'s KV cutoff when `OXE_PF_ATTN_AUTO_MIN_KV` is unset. PLACEHOLDER: None
/// = legacy (fa2 only for n_ctx > PFA_AUTO_MIN_CTX, i.e. tiled everywhere people work). When AP-1
/// has measured the tiled vs fa2 / fa2x crossover at 16-64K, this becomes `Some(<crossover>)`.
pub const PFA_AUTO_MIN_KV: Option<usize> = None;
/// PREFILL-POLICY: the kernel `auto` runs beyond its cutoff when `OXE_PF_ATTN_AUTO_KERNEL` is unset
/// ([`PfAttn::Fa2`] or [`PfAttn::Fa2x`]; AP-1 decides: fa2x < 1.2x fa2 drops fa2x).
pub const PFA_AUTO_KERNEL: PfAttn = PfAttn::Fa2;
/// Most query columns of one unit (8 tokens x 8 query heads; the tiled `nf` array bound).
pub const PFA_MAX_COLS: usize = 64;
/// fa2x default f16 chain length before the f32 fold (8: shorter than ggml's 8 + tree of 32).
pub const FA2X_FOLD: usize = 8;
/// fa2x default chunk positions.
pub const FA2X_CH: usize = 128;
/// fa2x largest chunk (the scratch is sized for it, so `OXE_PF_FA2X_CH` never changes ctx_mem).
pub const FA2X_MAX_CH: usize = 512;

/// Prefill attention kernel choice (`OXE_PF_ATTN`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PfAttn {
    /// fa2 only when n_ctx > [`PFA_AUTO_MIN_CTX`] (the default)
    #[default]
    Auto,
    /// today's tiled kernel at every context (O(n_ctx) scratch)
    Tiled,
    /// fa2 for units beyond [`PFA_MIN_KV`] at every context
    Fa2,
    /// fa2's units and cutoffs, run by the register-blocked fa2x kernel (APPEND-PREFILL)
    Fa2x,
}

impl PfAttn {
    /// `fa2`, `tiled`, `auto` (case-insensitive, trimmed); anything else: None.
    pub fn parse(s: &str) -> Option<PfAttn> {
        match s.trim().to_ascii_lowercase().as_str() {
            "fa2" => Some(PfAttn::Fa2),
            "fa2x" => Some(PfAttn::Fa2x),
            "tiled" => Some(PfAttn::Tiled),
            "auto" | "" => Some(PfAttn::Auto),
            _ => None,
        }
    }

    /// `OXE_PF_ATTN` (unset or unparsable: [`PfAttn::Auto`]; an unparsable value is reported once
    /// on stderr so a typo does not silently look like a measurement of fa2).
    pub fn from_env() -> PfAttn {
        match std::env::var("OXE_PF_ATTN") {
            Err(_) => PfAttn::Auto,
            Ok(v) => PfAttn::parse(&v).unwrap_or_else(|| {
                static WARN: std::sync::Once = std::sync::Once::new();
                WARN.call_once(|| eprintln!("oxe: OXE_PF_ATTN={v:?} not understood (fa2 | fa2x | tiled | auto): using auto"));
                PfAttn::Auto
            }),
        }
    }

    /// Whether an engine of capacity `n_ctx` runs fa2 or fa2x under the default auto policy (f16 KV;
    /// the engine adds its other conditions).
    pub fn fa2_at(self, n_ctx: usize) -> bool {
        self.path(n_ctx, AutoPolicy::default()).fa2()
    }

    /// PREFILL-POLICY: the path an engine of capacity `n_ctx` takes under this mode and `auto`'s
    /// policy (f16 KV, no trace / f32 reference; the engine forces tiled otherwise).
    pub fn path(self, n_ctx: usize, auto: AutoPolicy) -> PfPath {
        let at = |kernel: PfAttn, cut: usize| PfPath { kernel: if kernel == PfAttn::Fa2x { PfKernel::Fa2x } else { PfKernel::Fa2 }, cut };
        match self {
            PfAttn::Tiled => PfPath::TILED,
            PfAttn::Fa2 | PfAttn::Fa2x => at(self, PFA_MIN_KV),
            PfAttn::Auto => {
                let cut = auto.cut();
                let on = match auto.min_kv {
                    None => n_ctx > PFA_AUTO_MIN_CTX,
                    Some(_) => n_ctx > cut,
                };
                if on {
                    at(auto.kernel, cut)
                } else {
                    PfPath::TILED
                }
            }
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            PfAttn::Auto => "auto",
            PfAttn::Tiled => "tiled",
            PfAttn::Fa2 => "fa2",
            PfAttn::Fa2x => "fa2x",
        }
    }
}

/// PREFILL-POLICY: what `auto` resolves to ([`PfAttn::path`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AutoPolicy {
    /// None: legacy (fa2-class beyond PFA_MIN_KV only when n_ctx > PFA_AUTO_MIN_CTX). Some(n): the
    /// auto kernel for units whose padded n_kv exceeds `clamp(n, PFA_MIN_KV, PFA_AUTO_MIN_CTX)`
    pub min_kv: Option<usize>,
    /// [`PfAttn::Fa2`] or [`PfAttn::Fa2x`]
    pub kernel: PfAttn,
}

impl Default for AutoPolicy {
    fn default() -> AutoPolicy {
        AutoPolicy { min_kv: PFA_AUTO_MIN_KV, kernel: PFA_AUTO_KERNEL }
    }
}

impl AutoPolicy {
    /// The v0.1.3 lane-A rule (today's behaviour), whatever the defaults say.
    pub const LEGACY: AutoPolicy = AutoPolicy { min_kv: None, kernel: PfAttn::Fa2 };

    /// `kernel` must be fa2 or fa2x.
    pub fn new(min_kv: Option<usize>, kernel: PfAttn) -> Option<AutoPolicy> {
        matches!(kernel, PfAttn::Fa2 | PfAttn::Fa2x).then_some(AutoPolicy { min_kv, kernel })
    }

    /// The KV cutoff (positions): PFA_MIN_KV for the legacy rule, else min_kv clamped to
    /// [PFA_MIN_KV, PFA_AUTO_MIN_CTX] (short prompts keep tiled bits; the tiled scratch, sized for
    /// the cutoff, never exceeds the 128K-class size).
    pub fn cut(self) -> usize {
        self.min_kv.map_or(PFA_MIN_KV, |n| n.clamp(PFA_MIN_KV, PFA_AUTO_MIN_CTX))
    }

    /// `OXE_PF_ATTN_AUTO_MIN_KV` value: "legacy" = None (the lane-A rule, whatever the default),
    /// else a position count ("" / "default" are handled by [`AutoPolicy::from_lookup`]).
    pub fn parse_min_kv(s: &str) -> Option<Option<usize>> {
        match s.trim().to_ascii_lowercase().as_str() {
            "legacy" => Some(None),
            t => t.parse::<usize>().ok().map(Some),
        }
    }

    /// `OXE_PF_ATTN_AUTO_KERNEL` value: fa2 | fa2x.
    pub fn parse_kernel(s: &str) -> Option<PfAttn> {
        PfAttn::parse(s).filter(|k| matches!(k, PfAttn::Fa2 | PfAttn::Fa2x))
    }

    /// `OXE_PF_ATTN_AUTO_MIN_KV` / `OXE_PF_ATTN_AUTO_KERNEL` over the defaults ([`PFA_AUTO_MIN_KV`],
    /// [`PFA_AUTO_KERNEL`]); an unparsable value warns once and keeps the default.
    pub fn from_env() -> AutoPolicy {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> AutoPolicy {
        let mut p = AutoPolicy::default();
        if let Some(v) = get("OXE_PF_ATTN_AUTO_MIN_KV").filter(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "" | "default")) {
            match AutoPolicy::parse_min_kv(&v) {
                Some(m) => p.min_kv = m,
                None => {
                    static W: std::sync::Once = std::sync::Once::new();
                    W.call_once(|| eprintln!("oxe: OXE_PF_ATTN_AUTO_MIN_KV={v:?} not understood (N positions | legacy): using the default"));
                }
            }
        }
        if let Some(v) = get("OXE_PF_ATTN_AUTO_KERNEL") {
            match AutoPolicy::parse_kernel(&v) {
                Some(k) => p.kernel = k,
                None => {
                    static W: std::sync::Once = std::sync::Once::new();
                    W.call_once(|| eprintln!("oxe: OXE_PF_ATTN_AUTO_KERNEL={v:?} not understood (fa2 | fa2x): using the default"));
                }
            }
        }
        p
    }
}

/// PREFILL-POLICY: the kernel units beyond the cutoff run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PfKernel {
    Tiled,
    Fa2,
    Fa2x,
}

/// PREFILL-POLICY: the resolved prefill attention path of an engine: units whose last token's
/// padded n_kv exceeds `cut` run `kernel`, the others the tiled kernel (bit for bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PfPath {
    pub kernel: PfKernel,
    /// usize::MAX for [`PfKernel::Tiled`]
    pub cut: usize,
}

impl PfPath {
    pub const TILED: PfPath = PfPath { kernel: PfKernel::Tiled, cut: usize::MAX };

    /// fa2 or fa2x units exist (fa2's scratch policy).
    pub fn fa2(self) -> bool {
        self.kernel != PfKernel::Tiled
    }

    /// The cutoff when fa2-class units exist.
    pub fn cut_opt(self) -> Option<usize> {
        self.fa2().then_some(self.cut)
    }

    pub fn name(self) -> &'static str {
        match self.kernel {
            PfKernel::Tiled => "tiled",
            PfKernel::Fa2 => "fa2",
            PfKernel::Fa2x => "fa2x",
        }
    }
}

/// fa2x kernel parameters: f16 chain length before the f32 fold and chunk positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XParams {
    pub fold: usize,
    pub ch: usize,
}

impl Default for XParams {
    fn default() -> XParams {
        XParams { fold: FA2X_FOLD, ch: FA2X_CH }
    }
}

impl XParams {
    /// Valid: fold 8 / 16 / 32, ch a multiple of 32 in [32, FA2X_MAX_CH].
    pub fn new(fold: usize, ch: usize) -> Option<XParams> {
        (matches!(fold, 8 | 16 | 32) && ch >= 32 && ch % 32 == 0 && ch <= FA2X_MAX_CH).then_some(XParams { fold, ch })
    }

    /// `OXE_PF_FA2X_FOLD` / `OXE_PF_FA2X_CH` (read once; an invalid value warns and keeps the default).
    pub fn from_env() -> XParams {
        static P: std::sync::OnceLock<XParams> = std::sync::OnceLock::new();
        *P.get_or_init(|| {
            let get = |k: &str, d: usize| std::env::var(k).ok().map(|v| v.trim().parse::<usize>().ok().unwrap_or(usize::MAX)).unwrap_or(d);
            let (f, c) = (get("OXE_PF_FA2X_FOLD", FA2X_FOLD), get("OXE_PF_FA2X_CH", FA2X_CH));
            XParams::new(f, c).unwrap_or_else(|| {
                eprintln!("oxe: OXE_PF_FA2X_FOLD / OXE_PF_FA2X_CH not understood (fold 8|16|32, ch multiple of 32 <= {FA2X_MAX_CH}): using {FA2X_FOLD} / {FA2X_CH}");
                XParams::default()
            })
        })
    }
}

/// `OXE_PF_ATTN_THREADS` (THREADS-BY-PHASE, F34/F106): threads of the prefill team that take
/// attention-core work units (0 or unset = all, today's behaviour). Units are scheduled
/// dynamically and each is computed by one thread, so any value gives the same bits.
pub fn pf_attn_threads_from_env() -> usize {
    std::env::var("OXE_PF_ATTN_THREADS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0)
}

/// Per-thread scratch of the fa2 kernel for up to `ncol` columns of `hd` dims: everything is
/// O(chunk), nothing grows with the context.
pub struct Scratch {
    ncol: usize,
    hd: usize,
    /// scores [col][PFA_CH] (f32, overwritten by the chunk kernel)
    s: Vec<f32>,
    /// chunk weights [col][PFA_CH] (f16)
    p: Vec<u16>,
    /// chunk partial outputs [col][hd] and (m, l) pairs [m: col][l: col]
    part: Vec<f32>,
    ml: Vec<f32>,
    /// running outputs [col][hd], max and sum per column
    acc: Vec<f32>,
    m: Vec<f32>,
    l: Vec<f32>,
    nv: Vec<i32>,
    /// fa2x kernel scratch (`oxe_pfa2x_scratch_bytes` for ncol x hd at FA2X_MAX_CH); its output
    /// goes to `acc`
    xs: Vec<u8>,
}

impl Scratch {
    pub fn new(ncol: usize, hd: usize) -> Scratch {
        Scratch {
            ncol,
            hd,
            s: vec![0.0; ncol * PFA_CH],
            p: vec![0; ncol * PFA_CH],
            part: vec![0.0; ncol * hd],
            ml: vec![0.0; 2 * ncol],
            acc: vec![0.0; ncol * hd],
            m: vec![0.0; ncol],
            l: vec![0.0; ncol],
            nv: vec![0; ncol],
            xs: vec![0; x_bytes(ncol, hd)],
        }
    }

    pub fn empty() -> Scratch {
        Scratch::new(0, 0)
    }

    /// Bytes of [`Scratch::new`] (`ctx_mem` accounting; equal to [`Scratch::bytes_allocated`]).
    pub fn bytes(ncol: usize, hd: usize) -> usize {
        ncol * PFA_CH * (4 + 2) + ncol * hd * (4 + 4) + ncol * (2 + 1 + 1) * 4 + ncol * 4 + x_bytes(ncol, hd)
    }

    pub fn bytes_allocated(&self) -> usize {
        self.s.len() * 4 + self.p.len() * 2 + (self.part.len() + self.ml.len() + self.acc.len() + self.m.len() + self.l.len()) * 4 + self.nv.len() * 4 + self.xs.len()
    }

    /// Normalised (ungated) output of column `c` after [`Tile::run`].
    pub fn out(&self, c: usize) -> &[f32] {
        &self.acc[c * self.hd..(c + 1) * self.hd]
    }
}

/// fa2x kernel scratch bytes for `ncol` columns of `hd` dims (0 for an empty scratch).
fn x_bytes(ncol: usize, hd: usize) -> usize {
    if ncol == 0 || hd == 0 {
        return 0;
    }
    // SAFETY: pure size computation.
    unsafe { ffi::oxe_pfa2x_scratch_bytes(hd as i32, ncol as i32, FA2X_MAX_CH as i32) as usize }
}

/// One fa2 unit: `ncol` query columns of ONE KV head; column c belongs to the token at position
/// `pos0 + c / group` (columns of a token are contiguous) and attends to positions 0..=that.
#[derive(Clone, Copy)]
pub struct Tile {
    pub hd: usize,
    pub ncol: usize,
    /// columns per token (query heads of the KV head in this unit)
    pub group: usize,
    /// position of the unit's first token
    pub pos0: usize,
    pub scale: f32,
    /// K rows of this KV head: position p at `k + p * hd`; read on [0, last + 1)
    pub k: *const u16,
    /// V^T rows of this KV head: dim d at `vt + d * v_stride`; read on [0, round_up(last + 1, 32)),
    /// and up to the chunk end within `n_ctx` by the tail rule (must be readable and finite there)
    pub vt: *const u16,
    pub v_stride: usize,
    /// context capacity (a multiple of 32): the tail-rule bound
    pub n_ctx: usize,
    /// f16 query columns: column c at `q16 + c * hd`
    pub q16: *const u16,
}

impl Tile {
    #[inline]
    pub fn last(&self) -> usize {
        self.pos0 + (self.ncol - 1) / self.group
    }

    /// Computes the unit into `scr` ([`Scratch::out`]: softmax(q K^T * scale) V per column).
    ///
    /// # Safety
    /// Pointers valid as documented on the fields; `scr` built for at least `ncol` columns of `hd`;
    /// `ncol <= PFA_MAX_COLS`, `n_ctx % 32 == 0`, `last() < n_ctx`.
    pub unsafe fn run(&self, scr: &mut Scratch) {
        let (hd, ncol, group) = (self.hd, self.ncol, self.group);
        assert!(ncol >= 1 && ncol <= scr.ncol && hd == scr.hd && ncol <= PFA_MAX_COLS && group >= 1);
        assert!(self.n_ctx % 32 == 0 && self.last() < self.n_ctx);
        let rows = self.last() + 1;
        let n32 = rows.div_ceil(32) * 32;
        let acc = &mut scr.acc[..ncol * hd];
        acc.fill(0.0);
        scr.m[..ncol].fill(f32::NEG_INFINITY);
        scr.l[..ncol].fill(0.0);
        let mut s0 = 0;
        while s0 < n32 {
            let e = (s0 + PFA_CH).min(n32);
            let len = e - s0;
            let kr = rows.min(e) - s0;
            // scores of the chunk's K rows (rows >= rows are masked for every column: not computed)
            ffi::oxe_dot_f16_tile(
                hd as i32,
                self.k.add(s0 * hd),
                hd as i64,
                kr as i32,
                self.q16,
                hd as i64,
                ncol as i32,
                scr.s.as_mut_ptr(),
                1,
                PFA_CH as i64,
                std::ptr::null(),
            );
            for (c, v) in scr.nv[..ncol].iter_mut().enumerate() {
                *v = (self.pos0 + c / group + 1).saturating_sub(s0).min(len) as i32;
            }
            // tail rule: the dot of the full chunk (within n_ctx), whatever the unit's reach
            let len_full = (s0 + PFA_CH).min(self.n_ctx) - s0;
            ffi::oxe_attn_fchunk_f16(
                scr.s.as_mut_ptr(),
                PFA_CH as i64,
                ncol as i32,
                scr.nv.as_ptr(),
                len as i32,
                len_full as i32,
                self.scale,
                self.vt.add(s0),
                self.v_stride as i64,
                hd as i32,
                scr.p.as_mut_ptr(),
                scr.part.as_mut_ptr(),
                scr.ml.as_mut_ptr(),
            );
            // online merge (f32): O = O * exp(M - M') + part * exp(m - M'), L likewise
            for c in 0..ncol {
                let (mc, lc) = (scr.ml[c], scr.ml[ncol + c]);
                if !(lc > 0.0) {
                    continue; // no valid position of this column in the chunk
                }
                let mo = scr.m[c];
                let o = &mut acc[c * hd..(c + 1) * hd];
                let pc = &scr.part[c * hd..(c + 1) * hd];
                if mc <= mo {
                    let b = (mc - mo).exp();
                    scr.l[c] += lc * b;
                    for (x, &y) in o.iter_mut().zip(pc) {
                        *x += y * b;
                    }
                } else {
                    // new max: rescale what is there (nothing on the first valid chunk)
                    let a = if mo == f32::NEG_INFINITY { 0.0 } else { (mo - mc).exp() };
                    scr.l[c] = scr.l[c] * a + lc;
                    for (x, &y) in o.iter_mut().zip(pc) {
                        *x = *x * a + y;
                    }
                    scr.m[c] = mc;
                }
            }
            s0 = e;
        }
        for c in 0..ncol {
            let inv = 1.0 / scr.l[c];
            for x in &mut acc[c * hd..(c + 1) * hd] {
                *x *= inv;
            }
        }
    }
}

impl Tile {
    /// fa2x: computes the unit into `scr` ([`Scratch::out`]) with the register-blocked kernel.
    ///
    /// # Safety
    /// As [`Tile::run`]; additionally `hd % 32 == 0` and ncol <= 256.
    pub unsafe fn run_x(&self, scr: &mut Scratch, xp: XParams) {
        let (hd, ncol) = (self.hd, self.ncol);
        assert!(ncol >= 1 && ncol <= scr.ncol && hd == scr.hd && self.group >= 1);
        assert!(self.n_ctx % 32 == 0 && self.last() < self.n_ctx);
        let rc = ffi::oxe_pfa2x_unit(
            hd as i32,
            ncol as i32,
            self.group as i32,
            self.pos0 as i32,
            self.scale,
            self.k,
            self.vt,
            self.v_stride as i64,
            self.n_ctx as i32,
            self.q16,
            xp.ch as i32,
            xp.fold as i32,
            scr.xs.as_mut_ptr(),
            scr.xs.len() as i64,
            scr.acc.as_mut_ptr(),
        );
        assert_eq!(rc, 0, "oxe_pfa2x_unit refused hd {hd} ncol {ncol} {xp:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pf_attn_mode_parse_and_auto_threshold() {
        assert_eq!(PfAttn::parse("fa2"), Some(PfAttn::Fa2));
        assert_eq!(PfAttn::parse(" FA2 "), Some(PfAttn::Fa2));
        assert_eq!(PfAttn::parse("tiled"), Some(PfAttn::Tiled));
        assert_eq!(PfAttn::parse("auto"), Some(PfAttn::Auto));
        assert_eq!(PfAttn::parse("flash"), None);
        assert_eq!(PfAttn::parse("FA2X"), Some(PfAttn::Fa2x));
        assert!(PfAttn::Fa2x.fa2_at(4096) && PfAttn::Fa2x.name() == "fa2x");
        assert_eq!(XParams::default(), XParams::new(FA2X_FOLD, FA2X_CH).unwrap());
        assert!(XParams::new(12, 128).is_none() && XParams::new(8, 100).is_none() && XParams::new(8, 1024).is_none());
        // auto: only a 256K-class engine
        for n in [4096usize, 32_768, 131_072] {
            assert!(!PfAttn::Auto.fa2_at(n), "{n}");
            assert!(PfAttn::Fa2.fa2_at(n));
            assert!(!PfAttn::Tiled.fa2_at(n));
        }
        assert!(PfAttn::Auto.fa2_at(131_328) && PfAttn::Auto.fa2_at(262_144));
        assert!(!PfAttn::Tiled.fa2_at(262_144));
    }

    /// PREFILL-POLICY: the resolved path at the boundaries, per mode and auto policy.
    #[test]
    fn pf_attn_auto_policy_resolves_at_the_boundaries() {
        let leg = AutoPolicy::LEGACY;
        // legacy: tiled up to and including 131072, fa2 beyond PFA_MIN_KV above it
        assert_eq!(PfAttn::Auto.path(131_072, leg), PfPath::TILED);
        assert_eq!(PfAttn::Auto.path(131_328, leg), PfPath { kernel: PfKernel::Fa2, cut: PFA_MIN_KV });
        // explicit modes ignore the auto policy
        let p32 = AutoPolicy::new(Some(32_768), PfAttn::Fa2x).unwrap();
        for a in [leg, p32] {
            assert_eq!(PfAttn::Tiled.path(262_144, a), PfPath::TILED);
            assert_eq!(PfAttn::Fa2.path(16_384, a), PfPath { kernel: PfKernel::Fa2, cut: PFA_MIN_KV });
            assert_eq!(PfAttn::Fa2x.path(16_384, a), PfPath { kernel: PfKernel::Fa2x, cut: PFA_MIN_KV });
        }
        // min-kv 32768: an engine whose n_ctx cannot exceed the cutoff stays tiled; above it, the
        // auto kernel beyond 32768
        assert_eq!(PfAttn::Auto.path(32_768, p32), PfPath::TILED);
        assert_eq!(PfAttn::Auto.path(32_769, p32), PfPath { kernel: PfKernel::Fa2x, cut: 32_768 });
        assert_eq!(PfAttn::Auto.path(65_536, p32).name(), "fa2x");
        let p32f = AutoPolicy::new(Some(32_768), PfAttn::Fa2).unwrap();
        assert_eq!(PfAttn::Auto.path(65_536, p32f), PfPath { kernel: PfKernel::Fa2, cut: 32_768 });
        // clamps: below PFA_MIN_KV short prompts keep tiled bits; above PFA_AUTO_MIN_CTX the
        // tiled scratch would outgrow the 128K class
        let p0 = AutoPolicy::new(Some(0), PfAttn::Fa2).unwrap();
        assert_eq!(p0.cut(), PFA_MIN_KV);
        assert_eq!(PfAttn::Auto.path(PFA_MIN_KV, p0), PfPath::TILED);
        assert_eq!(PfAttn::Auto.path(PFA_MIN_KV + 256, p0).cut, PFA_MIN_KV);
        let pbig = AutoPolicy::new(Some(1 << 20), PfAttn::Fa2).unwrap();
        assert_eq!(pbig.cut(), PFA_AUTO_MIN_CTX);
        assert_eq!(PfAttn::Auto.path(131_072, pbig), PfPath::TILED);
        assert_eq!(PfAttn::Auto.path(262_144, pbig).cut, PFA_AUTO_MIN_CTX);
        assert!(AutoPolicy::new(Some(1), PfAttn::Tiled).is_none() && AutoPolicy::new(None, PfAttn::Auto).is_none());
        assert_eq!(PfPath::TILED.cut_opt(), None);
    }

    /// PREFILL-POLICY: the default is the placeholder (behaviour unchanged) until AP-1; env parsing.
    #[test]
    fn pf_attn_auto_policy_default_and_env() {
        // the one-line change after AP-1 updates this assertion together with PFA_AUTO_MIN_KV
        assert_eq!(AutoPolicy::default(), AutoPolicy::LEGACY, "PFA_AUTO_MIN_KV / PFA_AUTO_KERNEL changed: update the release notes");
        for n in [4096usize, 32_768, 65_536, 131_072] {
            assert!(!PfAttn::Auto.fa2_at(n), "{n}: auto must stay tiled at <= 131072 by default");
        }
        let env = |kv: &'static [(&'static str, &'static str)]| AutoPolicy::from_lookup(|k| kv.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string()));
        assert_eq!(env(&[]), AutoPolicy::default());
        assert_eq!(env(&[("OXE_PF_ATTN_AUTO_MIN_KV", "24576")]).min_kv, Some(24_576));
        assert_eq!(env(&[("OXE_PF_ATTN_AUTO_MIN_KV", "legacy")]).min_kv, None);
        assert_eq!(env(&[("OXE_PF_ATTN_AUTO_MIN_KV", "default")]).min_kv, PFA_AUTO_MIN_KV);
        assert_eq!(env(&[("OXE_PF_ATTN_AUTO_MIN_KV", "lots")]), AutoPolicy::default());
        assert_eq!(env(&[("OXE_PF_ATTN_AUTO_KERNEL", "FA2X")]).kernel, PfAttn::Fa2x);
        assert_eq!(env(&[("OXE_PF_ATTN_AUTO_KERNEL", "tiled")]).kernel, PFA_AUTO_KERNEL);
        assert_eq!(AutoPolicy::parse_min_kv(" 16384 "), Some(Some(16_384)));
        assert_eq!(AutoPolicy::parse_min_kv("-1"), None);
    }

    #[test]
    fn scratch_bytes_match_allocation_and_do_not_grow_with_context() {
        for (ncol, hd) in [(64usize, 256usize), (32, 128), (8, 128), (1, 64)] {
            let s = Scratch::new(ncol, hd);
            assert_eq!(s.bytes_allocated(), Scratch::bytes(ncol, hd), "{ncol} x {hd}");
        }
        // Ornith's unit (64 columns x 256 dims): ~224 KiB fa2 + ~304 KiB fa2x, whatever the context
        assert!(Scratch::bytes(64, 256) < 576 << 10);
    }
}
