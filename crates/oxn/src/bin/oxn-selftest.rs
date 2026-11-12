//! Model-free self-test of an oxn backend.
//!
//! Runs GEMMs Y[T,O] = X[T,K] * W[O,K]^T on synthetic data and checks sampled
//! outputs against an f64 reference computed on the host from the same quantised
//! inputs: X is rounded to half, W is decoded from the Q8_0 codes on the fly.
//!
//! Usage: oxn-selftest [--backend emu|device] [--sessions N] [--set quick|klein|all]
//!                     [--rounds N] [--samples N] [--seed N] [--break-ref]
//!
//! `--break-ref` multiplies every reference value by 1.01 (self-check of the checker);
//! the run must then report FAIL and exit 1.
//!
//! Exit codes: 0 all rounds pass; 1 numeric failure or skipped case; 2 usage error
//! (also `--backend device` in a build without the `device` feature); 3 backend error.

use std::cmp::Ordering;
use std::io::Write;
use std::str::FromStr;

use oxn::tiles::{self, ActLayout, OutLayout, TILE, TILE_ELMS};
use oxn::{Backend, Buf, GemmArgs, GF_W8, GF_YCOL};

const X_BYTES: usize = 64 << 20;
const W_BYTES: usize = 128 << 20;
const Y_BYTES: usize = 128 << 20;
const REL_TOL: f64 = 1e-3;
/// VTCM size requested from the CPU emulator.
const EMU_VTCM_BYTES: u32 = 8 << 20;

const USAGE: &str = "usage: oxn-selftest [--backend emu|device] [--sessions N] [--set quick|klein|all]
                   [--rounds N] [--samples N] [--seed N] [--break-ref]
  defaults: --backend emu --sessions 1 --set quick --rounds 3 --samples 2048 --seed 1
  --break-ref  self-check of the checker: multiply every reference value by 1.01
               (the run must report FAIL and exit 1)
  exit codes: 0 pass, 1 numeric failure or skipped case, 2 usage, 3 backend error";

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    t: usize,
    k: usize,
    o: usize,
    w8: bool,
}

const QUICK: &[Case] = &[
    Case { name: "q-1tile", t: 32, k: 32, o: 32, w8: true },
    Case { name: "q-multi", t: 64, k: 128, o: 96, w8: true },
    Case { name: "q-tails", t: 160, k: 256, o: 288, w8: true },
    // T=40 is padded to 64 rows by ActLayout; rows 40..63 stay zero.
    Case { name: "q-pad", t: 40, k: 64, o: 64, w8: true },
    Case { name: "q-fp16", t: 64, k: 128, o: 96, w8: false },
];

const KLEIN: &[Case] = &[
    Case { name: "k-txt-qkv", t: 512, k: 3072, o: 9216, w8: true },
    Case { name: "k-img-qkv", t: 1024, k: 3072, o: 9216, w8: true },
    Case { name: "k-img-mlp0", t: 1024, k: 3072, o: 18432, w8: true },
    Case { name: "k-lin1", t: 1536, k: 3072, o: 27648, w8: true },
    Case { name: "k-lin2-chunk", t: 1536, k: 3072, o: 3072, w8: true },
    Case { name: "k-proj", t: 1024, k: 3072, o: 3072, w8: true },
    Case { name: "k-qkv-fp16", t: 1536, k: 3072, o: 9216, w8: false },
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Set {
    Quick,
    Klein,
    All,
}

struct Opts {
    device: bool,
    sessions: usize,
    set: Set,
    rounds: usize,
    samples: usize,
    seed: u64,
    break_ref: bool,
}

// ---------------------------------------------------------------------------
// Small helpers: RNG, JSON, args
// ---------------------------------------------------------------------------

fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of an independent stream derived from the user seed and a list of indices.
fn stream_seed(seed: u64, parts: &[u64]) -> u64 {
    parts.iter().fold(splitmix(seed), |h, &p| splitmix(h ^ p))
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        let s = splitmix(seed);
        Rng(if s == 0 { 0x1234_5678_9abc_def1 } else { s })
    }
    /// xorshift64*
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// [0, 1)
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }
    /// [lo, hi)
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn jstr(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// JSON number, or null for NaN/inf.
fn jf(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_string()
    }
}

fn emit(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

fn next_val(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn num<T: FromStr>(s: &str, flag: &str) -> Result<T, String> {
    s.parse::<T>().map_err(|_| format!("{flag}: bad value '{s}'"))
}

fn parse_args(args: Vec<String>) -> Result<Opts, String> {
    let mut o = Opts {
        device: false,
        sessions: 1,
        set: Set::Quick,
        rounds: 3,
        samples: 2048,
        seed: 1,
        break_ref: false,
    };
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--backend" => match next_val(&mut it, &a)?.as_str() {
                "emu" => o.device = false,
                "device" => o.device = true,
                other => return Err(format!("--backend: unknown backend '{other}'")),
            },
            "--sessions" => {
                o.sessions = num(&next_val(&mut it, &a)?, &a)?;
                if o.sessions == 0 {
                    return Err("--sessions must be >= 1".into());
                }
            }
            "--set" => {
                o.set = match next_val(&mut it, &a)?.as_str() {
                    "quick" => Set::Quick,
                    "klein" => Set::Klein,
                    "all" => Set::All,
                    other => return Err(format!("--set: unknown set '{other}'")),
                }
            }
            "--rounds" => {
                o.rounds = num(&next_val(&mut it, &a)?, &a)?;
                if o.rounds == 0 {
                    return Err("--rounds must be >= 1".into());
                }
            }
            "--samples" => {
                o.samples = num(&next_val(&mut it, &a)?, &a)?;
                if o.samples == 0 {
                    return Err("--samples must be >= 1".into());
                }
            }
            "--seed" => o.seed = num(&next_val(&mut it, &a)?, &a)?,
            "--break-ref" => o.break_ref = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(o)
}

// ---------------------------------------------------------------------------
// Cases, buffers, data
// ---------------------------------------------------------------------------

/// Selected cases with their global index (used to derive the data seed, so a case
/// has the same data whichever set it is run from).
fn selected(set: Set) -> Vec<(usize, Case)> {
    let all: Vec<Case> = QUICK.iter().chain(KLEIN.iter()).copied().collect();
    let range = match set {
        Set::Quick => 0..QUICK.len(),
        Set::Klein => QUICK.len()..all.len(),
        Set::All => 0..all.len(),
    };
    range.map(|i| (i, all[i])).collect()
}

fn out_layout(case: &Case) -> OutLayout {
    OutLayout {
        t_pad: case.t.div_ceil(TILE) * TILE,
        o: case.o,
        nchunks: 1,
        ycol: true,
    }
}

/// Bytes the weights need in the W buffer.
fn w_bytes(case: &Case) -> usize {
    if case.w8 {
        tiles::w8_bytes(case.o, case.k)
    } else {
        case.o * case.k * 2
    }
}

/// True when the case's X, W and Y layouts fit the buffers.
fn fits(case: &Case, x: &Slot, w: &Slot, y: &Slot) -> bool {
    let x_need = ActLayout::new(case.t, case.k, case.k).bytes();
    x_need <= x.len && w_bytes(case) <= w.len && out_layout(case).bytes() <= y.len
}

/// One backend buffer of one session.
#[derive(Clone, Copy)]
struct Slot {
    ptr: *mut u8,
    len: usize,
    id: u32,
}

impl Slot {
    fn from_buf(b: &Buf) -> Slot {
        Slot { ptr: b.ptr, len: b.len, id: b.slot }
    }
}

/// Guard pattern size on each side of the used Y range.
const GUARD: usize = 2048;
const GUARD_BYTE: u8 = 0xA5;

/// Operand byte offsets (x_off, w_off, y_off) of a case inside its X/W/Y buffers.
/// The three cases with fp16 weights at zero keep (0,0,0); the others sit at
/// non-zero offsets, as the real model does.
fn offsets(case: &Case) -> (usize, usize, usize) {
    match case.name {
        "q-1tile" | "q-fp16" | "k-qkv-fp16" => (0, 0, 0),
        _ => (3 * 2048, 5 * 1152, 7 * 2048),
    }
}

/// The part of a buffer starting at `off`, or None when `off` is past its end.
fn view(s: &Slot, off: usize) -> Option<Slot> {
    if off > s.len {
        return None;
    }
    Some(Slot { ptr: s.ptr.wrapping_add(off), len: s.len - off, id: s.id })
}

/// Guard regions (start, len) in the full Y buffer around the used range
/// [y_off, y_off + used): GUARD bytes before it and after it, where they fit in the buffer.
fn guard_regions(y_len: usize, y_off: usize, used: usize) -> Vec<(usize, usize)> {
    let mut g = Vec::new();
    if y_off >= GUARD {
        g.push((y_off - GUARD, GUARD));
    }
    let after = y_off + used;
    if after + GUARD <= y_len {
        g.push((after, GUARD));
    }
    g
}

/// Q8_0 source weights (34-byte blocks: f16 scale, 32 i8 codes) of an O x K matrix.
fn gen_q8(case: &Case, rng: &mut Rng) -> Vec<u8> {
    let nb = case.k / TILE;
    let base = 1.0f32 / (127.0 * (case.k as f32).sqrt());
    let mut q8 = Vec::with_capacity(case.o * nb * 34);
    for _ in 0..case.o * nb {
        let d = tiles::f32_to_half(rng.range(0.5, 1.5) * base);
        q8.extend_from_slice(&d.to_le_bytes());
        for _ in 0..TILE {
            let q = (rng.next_u64() % 255) as i32 - 127;
            q8.push(q as i8 as u8);
        }
    }
    q8
}

/// f32 weight product d * q for W[n][k] (decoded from the Q8_0 bytes).
fn wprod(q8: &[u8], nb: usize, n: usize, k: usize) -> f32 {
    let blk = (n * nb + k / TILE) * 34;
    let d = tiles::half_to_f32(u16::from_le_bytes([q8[blk], q8[blk + 1]]));
    d * (q8[blk + 2 + k % TILE] as i8) as f32
}

/// Reference weight: the f32 product for W8, the half-rounded product for FP16.
fn wref(q8: &[u8], nb: usize, n: usize, k: usize, w8: bool) -> f32 {
    let v = wprod(q8, nb, n, k);
    if w8 {
        v
    } else {
        tiles::half_to_f32(tiles::f32_to_half(v))
    }
}

/// Fill the X buffer (zeroed first) with uniform [-1,1] rows and return the
/// half-rounded values used by the reference.
fn gen_x(case: &Case, rng: &mut Rng, x: &Slot) -> Result<(ActLayout, Vec<f32>), String> {
    let lay = ActLayout::new(case.t, case.k, case.k);
    if lay.bytes() > x.len {
        return Err(format!("{}: X layout does not fit", case.name));
    }
    // SAFETY: x.ptr is the backend's X buffer for this session, x.len >= lay.bytes()
    // (checked above). No other reference to it is alive, and the backend does not run
    // while the host has control in this function.
    unsafe { std::slice::from_raw_parts_mut(x.ptr, lay.bytes()).fill(0) };
    let mut xr = vec![0f32; case.t * case.k];
    let mut row = vec![0f32; case.k];
    for m in 0..case.t {
        for (k, v) in row.iter_mut().enumerate() {
            *v = rng.range(-1.0, 1.0);
            xr[m * case.k + k] = tiles::half_to_f32(tiles::f32_to_half(*v));
        }
        // SAFETY: m < t <= t_pad, the destination is the zeroed layout of lay.bytes()
        // bytes, and the buffer is 2-byte aligned (checked at alloc). Single writer.
        unsafe { tiles::write_act_row(x.ptr as *mut u16, &lay, m, 0, &row) };
    }
    Ok((lay, xr))
}

/// Write the weights of the case into the W buffer: W8 records via the tiles packer,
/// FP16 tiles element by element.
fn write_w(case: &Case, q8: &[u8], w: &Slot) -> Result<(), String> {
    let nbytes = w_bytes(case);
    if nbytes > w.len {
        return Err(format!("{}: W layout does not fit", case.name));
    }
    if case.w8 {
        // SAFETY: w.ptr is the backend's W buffer for this session, nbytes <= w.len
        // (checked above); the slice is the only live reference to it.
        let dst = unsafe { std::slice::from_raw_parts_mut(w.ptr, nbytes) };
        tiles::pack_w8_from_q8_0(dst, q8, case.o, case.k, case.k, None);
    } else {
        let nb = case.k / TILE;
        let nkt = case.k / TILE;
        for ot in 0..case.o / TILE {
            for kt in 0..nkt {
                for kl in 0..TILE {
                    for nl in 0..TILE {
                        let n = ot * TILE + nl;
                        let k = kt * TILE + kl;
                        let h = tiles::f32_to_half(wprod(q8, nb, n, k));
                        let idx = (ot * nkt + kt) * TILE_ELMS + tiles::wt_elem_idx(kl, nl);
                        // SAFETY: idx < (o/32)*(k/32)*1024 = o*k <= w.len/2, the buffer is
                        // 2-byte aligned (checked at alloc), and the host is the only writer.
                        unsafe { (w.ptr as *mut u16).add(idx).write(h) };
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Check
// ---------------------------------------------------------------------------

struct Chk {
    rel: f64,
    nonfinite: u64,
}

/// Compare `samples` random real outputs against the f64 reference.
#[allow(clippy::too_many_arguments)]
fn check(
    case: &Case,
    lay_y: &OutLayout,
    y: *const u16,
    xr: &[f32],
    q8: &[u8],
    samples: usize,
    rng: &mut Rng,
    break_ref: bool,
) -> Chk {
    let nb = case.k / TILE;
    let mut num = 0f64;
    let mut den = 0f64;
    let mut nonfinite = 0u64;
    let mut one = [0f32; 1];
    for _ in 0..samples {
        let m = rng.below(case.t);
        let n = rng.below(case.o);
        // SAFETY: y is the backend's Y buffer for this session and covers lay_y.bytes()
        // (checked by fits); m < t <= t_pad and n < o; the gemm that wrote it has returned.
        unsafe { tiles::read_out_row(y, lay_y, m, n, &mut one) };
        let mut r = 0f64;
        for kk in 0..case.k {
            r += xr[m * case.k + kk] as f64 * wref(q8, nb, n, kk, case.w8) as f64;
        }
        if break_ref {
            r *= 1.01;
        }
        let yv = one[0];
        if !yv.is_finite() {
            nonfinite += 1;
            continue;
        }
        let e = yv as f64 - r;
        num += e * e;
        den += r * r;
    }
    let rel = if den > 0.0 {
        (num / den).sqrt()
    } else if num == 0.0 {
        0.0
    } else {
        f64::INFINITY
    };
    Chk { rel, nonfinite }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Agg {
    wall_ms: Vec<f64>,
    pc: u64,
    hmx: u64,
    deq: u64,
    worst_rel: f64,
    nonfinite: u64,
    all_ok: bool,
    /// Reason of the first failing round, if any.
    reason: &'static str,
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = v.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn open_backend(opts: &Opts) -> Result<Box<dyn Backend>, String> {
    if opts.device {
        #[cfg(feature = "device")]
        {
            return oxn::device::Device::open(opts.sessions).map(|d| Box::new(d) as Box<dyn Backend>);
        }
        #[cfg(not(feature = "device"))]
        {
            return Err("built without the device feature".into());
        }
    }
    Ok(Box::new(oxn::emu::Emu::new(EMU_VTCM_BYTES, opts.sessions)))
}

fn backend_error(text: &str) -> i32 {
    emit(&format!("{{\"ev\":\"error\",\"what\":{}}}", jstr(text)));
    eprintln!("oxn-selftest: backend error: {text}");
    3
}

/// Runs the whole self-test on an open backend. Ok(exit code) for test outcomes,
/// Err(text) for a backend error (the run stops at once).
fn drive(opts: &Opts, be: &mut dyn Backend) -> Result<i32, String> {
    let sessions = opts.sessions;
    if be.sessions() < sessions {
        return Err(format!("backend has {} sessions, {sessions} requested", be.sessions()));
    }
    emit(&format!(
        "{{\"ev\":\"start\",\"backend\":{},\"sessions\":{sessions},\"vtcm_bytes\":{}}}",
        jstr(be.name()),
        be.vtcm_bytes()
    ));

    let mut slots: Vec<[Slot; 3]> = Vec::with_capacity(sessions);
    for s in 0..sessions {
        let x = Slot::from_buf(&be.alloc(s, X_BYTES)?);
        let w = Slot::from_buf(&be.alloc(s, W_BYTES)?);
        let y = Slot::from_buf(&be.alloc(s, Y_BYTES)?);
        for b in [&x, &w, &y] {
            if b.ptr as usize % 2 != 0 {
                return Err(format!("session {s}: buffer not 2-byte aligned"));
            }
        }
        slots.push([x, w, y]);
    }

    let cases = selected(opts.set);
    let mut total_rounds = 0u64;
    let mut failed = 0u64;

    for (gi, case) in cases.iter().copied() {
        let mut case_numeric_fail = false;
        for (s, [x, w, y]) in slots.iter().enumerate() {
            let (x_off, w_off, y_off) = offsets(&case);
            // Views start at the operand offsets; fits() checks them against the views.
            let (xv, wv, yv) = match (view(x, x_off), view(w, w_off), view(y, y_off)) {
                (Some(a), Some(b), Some(c)) if fits(&case, &a, &b, &c) => (a, b, c),
                _ => {
                    eprintln!("{:<12} session {s}: skipped (does not fit)", case.name);
                    emit(&format!(
                        "{{\"ev\":\"skip\",\"case\":{},\"session\":{s},\"reason\":\"does not fit\"}}",
                        jstr(case.name)
                    ));
                    failed += 1;
                    continue;
                }
            };
            let mut rng = Rng::new(stream_seed(opts.seed, &[gi as u64, s as u64]));
            let (lay_x, xr) = gen_x(&case, &mut rng, &xv)?;
            let q8 = gen_q8(&case, &mut rng);
            write_w(&case, &q8, &wv)?;
            let lay_y = out_layout(&case);
            let y_used = lay_y.bytes();
            let guards = guard_regions(y.len, y_off, y_used);
            let flops = 2.0 * case.t as f64 * case.k as f64 * case.o as f64;
            let mut agg = Agg { all_ok: true, ..Default::default() };

            for round in 0..opts.rounds {
                // SAFETY: yv covers y_used bytes of the backend's Y buffer (checked by fits);
                // the guard regions lie inside y.len (guard_regions). No other reference is
                // alive and the previous gemm has returned.
                unsafe { std::slice::from_raw_parts_mut(yv.ptr, y_used).fill(0xff) };
                for &(st, ln) in &guards {
                    // SAFETY: (st, ln) is inside the full Y buffer (guard_regions).
                    unsafe { std::slice::from_raw_parts_mut(y.ptr.wrapping_add(st), ln).fill(GUARD_BYTE) };
                }
                let args = GemmArgs {
                    x_slot: x.id,
                    x_off: x_off as u32,
                    w_slot: w.id,
                    w_off: w_off as u32,
                    y_slot: y.id,
                    y_off: y_off as u32,
                    // The backend contract needs T a multiple of 32, so the padded row count
                    // is passed (q-pad: 40 -> 64). Rows >= T are not checked.
                    t: lay_x.t_pad as u32,
                    k: case.k as u32,
                    o: case.o as u32,
                    flags: GF_YCOL | if case.w8 { GF_W8 } else { 0 },
                    r: 0,
                    c: 0,
                };
                let st = be.gemm(s, &args)?;
                // Guard check: every guard byte must still be GUARD_BYTE.
                let guard_ok = guards.iter().all(|&(gs, ln)| {
                    // SAFETY: (gs, ln) is inside the full Y buffer; read-only view after the gemm returned.
                    unsafe { std::slice::from_raw_parts(y.ptr.wrapping_add(gs), ln) }
                        .iter()
                        .all(|&b| b == GUARD_BYTE)
                });
                let mut srng = Rng::new(stream_seed(opts.seed, &[gi as u64, s as u64, round as u64, 0x5a]));
                let chk = check(
                    &case,
                    &lay_y,
                    yv.ptr as *const u16,
                    &xr,
                    &q8,
                    opts.samples,
                    &mut srng,
                    opts.break_ref,
                );
                let reason = if !guard_ok {
                    "y guard overwritten"
                } else if chk.nonfinite > 0 {
                    "nonfinite output"
                } else if !(chk.rel <= REL_TOL) {
                    "rel_rmse above 1e-3"
                } else {
                    ""
                };
                let ok = reason.is_empty();
                let wall_ms = st.wall_s * 1e3;
                let dsp_ms = st.dsp_us as f64 / 1e3;
                let tops_wall = flops / st.wall_s / 1e12;
                let tops_dsp = if st.dsp_us > 0 {
                    flops / (st.dsp_us as f64 * 1e-6) / 1e12
                } else {
                    f64::NAN
                };
                emit(&format!(
                    "{{\"ev\":\"gemm\",\"case\":{},\"session\":{s},\"round\":{round},\"T\":{},\"K\":{},\"O\":{},\"w8\":{},\"wall_ms\":{},\"dsp_ms\":{},\"tops_wall\":{},\"tops_dsp\":{},\"pcycles\":{},\"hmx_pcycles\":{},\"deq_pcycles\":{},\"wait_pcycles\":{},\"steps\":{},\"R\":{},\"C\":{},\"rel_rmse\":{},\"nonfinite\":{},\"x_off\":{x_off},\"w_off\":{w_off},\"y_off\":{y_off},\"guard_ok\":{guard_ok},\"ok\":{ok},\"reason\":{}}}",
                    jstr(case.name),
                    case.t,
                    case.k,
                    case.o,
                    case.w8,
                    jf(wall_ms),
                    jf(dsp_ms),
                    jf(tops_wall),
                    jf(tops_dsp),
                    st.pcycles,
                    st.hmx_pcycles,
                    st.deq_pcycles,
                    st.wait_pcycles,
                    st.steps,
                    st.r,
                    st.c,
                    jf(chk.rel),
                    chk.nonfinite,
                    jstr(reason),
                ));
                total_rounds += 1;
                agg.wall_ms.push(wall_ms);
                agg.pc += st.pcycles;
                agg.hmx += st.hmx_pcycles;
                agg.deq += st.deq_pcycles;
                if chk.rel > agg.worst_rel || chk.rel.is_nan() {
                    agg.worst_rel = chk.rel;
                }
                agg.nonfinite += chk.nonfinite;
                if !ok {
                    agg.all_ok = false;
                    if agg.reason.is_empty() {
                        agg.reason = reason;
                    }
                    failed += 1;
                }
            }

            let med = median(&mut agg.wall_ms);
            let pct = if agg.pc > 0 {
                format!(
                    "  hmx {:5.1}%  deq {:5.1}%",
                    100.0 * agg.hmx as f64 / agg.pc as f64,
                    100.0 * agg.deq as f64 / agg.pc as f64
                )
            } else {
                String::new()
            };
            eprintln!(
                "{:<12} s{s} median {:9.3} ms  TOPS(wall) {:8.4}{pct}  rel_rmse {:.3e}  nonfinite {}  {}",
                case.name,
                med,
                flops / (med / 1e3) / 1e12,
                agg.worst_rel,
                agg.nonfinite,
                if agg.all_ok { "PASS" } else { "FAIL" }
            );
            if !agg.all_ok {
                eprintln!("    {}: first failure reason: {}", case.name, agg.reason);
                case_numeric_fail = true;
            }
        }
        if case_numeric_fail {
            eprintln!("oxn-selftest: numeric failure in {}, stopping after this case", case.name);
            break;
        }
    }

    let pass = failed == 0;
    emit(&format!(
        "{{\"ev\":\"result\",\"cases\":{},\"rounds\":{total_rounds},\"failed\":{failed},\"pass\":{pass}}}",
        cases.len()
    ));
    eprintln!("oxn-selftest: {} rounds, {failed} failed: {}", total_rounds, if pass { "PASS" } else { "FAIL" });
    Ok(if pass { 0 } else { 1 })
}

fn run(opts: &Opts) -> i32 {
    let mut be = match open_backend(opts) {
        Ok(b) => b,
        Err(e) => return backend_error(&e),
    };
    let res = drive(opts, be.as_mut());
    // Drop the backend (tears down the device session) before reporting.
    drop(be);
    match res {
        Ok(code) => code,
        Err(e) => backend_error(&e),
    }
}

fn main() {
    let opts = match parse_args(std::env::args().skip(1).collect()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("oxn-selftest: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if opts.device && !cfg!(feature = "device") {
        eprintln!("oxn-selftest: built without the device feature");
        std::process::exit(2);
    }
    let code = run(&opts);
    let _ = std::io::stdout().flush();
    std::process::exit(code);
}
