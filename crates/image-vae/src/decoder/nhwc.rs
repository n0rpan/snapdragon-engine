//! Channel-last decoder path (the default since branch `image-vae-conv`).
//!
//! Every feature map is `[h][w][c]` (index `(y*w + x)*c + ch`) from the un-patchify to the last
//! convolution, so no stage copies between layouts:
//!
//! * f32 convolution: implicit GEMM straight on the channel-last input (no padded copy, borders
//!   through a small per-tile patch), tiles of 4 pixels x 16 output channels held in registers,
//!   swept one group of taps at a time along an output row so that the weights of the sweep stay
//!   in L1. Output is written channel-last in place. Per output value the terms are added in
//!   (tap, input channel) order with fused multiply-adds on top of the bias: the same arithmetic
//!   as `Conv::forward_f32` of the channel-major path.
//! * nearest x2 upsampling followed by a 3x3 convolution is computed as four 2x2 convolutions on
//!   the small map (one per output phase, weights of coinciding taps summed once): 16 instead of
//!   36 tap products per source pixel, and the upsampled tensor is never built. The summed
//!   weights change the rounding (not the mathematics) of these convolutions.
//! * GroupNorm: one statistics pass (per-row f64 partial sums, combined in row order, so the
//!   result does not depend on the thread count) and one NEON pass for affine + SiLU, in place
//!   where the input is not needed again.
//! * the residual add runs inside the convolution's row band while the band is in cache.
//! * int8 flag: GroupNorm + SiLU + quantisation are one pass that writes int8 codes directly in
//!   the SMMLA tile layout (three copies shifted by one pixel, so every 3x3 tap of every 8-pixel
//!   group is a contiguous operand); the convolution calls the SMMLA tile per tap with no
//!   per-pixel repacking.
//!
//! The channel-major path of `decoder.rs` is kept as `VaeDecoder::decode_timed_channel_major`
//! (A/B reference).

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use super::{exp_fast, int8_conv, winograd, Attn, Conv, DecodeTimes, Norm, ResBlock, VaeDecoder, GC, GN_EPS, I8_CHUNK, PATCH};

/// Channel-last feature map; `d` may be longer than `c*h*w` (pooled buffer).
pub(super) struct F {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub d: Vec<f32>,
}

impl F {
    fn n(&self) -> usize {
        self.c * self.h * self.w
    }
    fn data(&self) -> &[f32] {
        &self.d[..self.n()]
    }
}

/// Reuses the large buffers of one decode (a fresh 134 MB vector per stage costs page faults).
/// Buffers are handed out with stale contents: every producer writes every element it owns.
pub(super) struct Pool<T>(Mutex<Vec<Vec<T>>>);

impl<T: Clone + Default> Pool<T> {
    pub fn new() -> Pool<T> {
        Pool(Mutex::new(Vec::new()))
    }
    fn take(&self, n: usize) -> Vec<T> {
        let mut free = self.0.lock().unwrap();
        let mut best: Option<usize> = None;
        for (i, v) in free.iter().enumerate() {
            if v.len() >= n && v.len() <= 4 * n.max(1) && best.is_none_or(|b| v.len() < free[b].len()) {
                best = Some(i);
            }
        }
        match best {
            Some(i) => free.swap_remove(i),
            None => {
                drop(free);
                vec![T::default(); n]
            }
        }
    }
    fn give(&self, v: Vec<T>) {
        self.0.lock().unwrap().push(v);
    }
}

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
// SAFETY: used only for writes to disjoint regions from the tasks of one `par_for`.
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// The worker threads of one decode: `threads - 1` parked threads plus the caller.
/// (Spawning scoped threads per stage costs about half a millisecond per stage at 12 threads,
/// and a decode has some 150 parallel stages.)
pub(super) struct Workers {
    sh: Arc<Shared>,
    handles: Vec<std::thread::JoinHandle<()>>,
    run_lock: Mutex<()>,
}

struct Shared {
    st: Mutex<State>,
    work: Condvar,
    done: Condvar,
}

struct State {
    gen: u64,
    job: Option<Job>,
    running: usize,
    quit: bool,
    panicked: bool,
}

#[derive(Clone, Copy)]
struct Job {
    f: *const (dyn Fn(usize) + Sync),
    n: usize,
    next: *const AtomicUsize,
}
// SAFETY: the pointers are only dereferenced while `Workers::run` (which owns the targets) waits.
unsafe impl Send for Job {}

impl Job {
    /// # Safety
    /// Only while the `run` call that published this job has not returned.
    unsafe fn work(&self) {
        let (f, next) = (&*self.f, &*self.next);
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= self.n {
                break;
            }
            f(i);
        }
    }
}

struct Finish<'a>(&'a Shared);
impl Drop for Finish<'_> {
    fn drop(&mut self) {
        let mut st = self.0.st.lock().unwrap_or_else(|e| e.into_inner());
        if std::thread::panicking() {
            st.panicked = true;
        }
        st.running -= 1;
        if st.running == 0 {
            self.0.done.notify_one();
        }
    }
}

struct PendingJob<'a>(&'a Shared);

impl Drop for PendingJob<'_> {
    fn drop(&mut self) {
        let mut st = self.0.st.lock().unwrap_or_else(|e| e.into_inner());
        while st.running > 0 {
            st = self.0.done.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = None;
    }
}

impl Workers {
    pub fn new(threads: usize) -> Workers {
        let sh = Arc::new(Shared { st: Mutex::new(State { gen: 0, job: None, running: 0, quit: false, panicked: false }), work: Condvar::new(), done: Condvar::new() });
        let handles = (1..threads.max(1))
            .map(|_| {
                let sh = sh.clone();
                std::thread::spawn(move || {
                    let mut seen = 0u64;
                    loop {
                        let job = {
                            let mut st = sh.st.lock().unwrap_or_else(|e| e.into_inner());
                            while st.gen == seen && !st.quit {
                                st = sh.work.wait(st).unwrap_or_else(|e| e.into_inner());
                            }
                            if st.quit {
                                return;
                            }
                            seen = st.gen;
                            st.job.expect("job published with the generation")
                        };
                        let _fin = Finish(&sh);
                        // SAFETY: `run` waits for `running == 0` (decremented by `_fin`) before it returns.
                        unsafe { job.work() };
                    }
                })
            })
            .collect();
        Workers { sh, handles, run_lock: Mutex::new(()) }
    }

    /// Runs `f(i)` for `i in 0..n`, tasks taken from a shared counter by the workers and the
    /// caller. Every task writes only its own output, so results do not depend on the schedule.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if self.handles.is_empty() || n <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        let _serial = self.run_lock.lock().unwrap_or_else(|e| e.into_inner());
        let next = AtomicUsize::new(0);
        // SAFETY (lifetime erasure): the job is withdrawn below before `f` and `next` go out of scope.
        let f_static: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        let job = Job { f: f_static, n, next: &next };
        {
            let mut st = self.sh.st.lock().unwrap_or_else(|e| e.into_inner());
            assert!(!st.panicked, "a decode worker panicked");
            st.job = Some(job);
            st.gen = st.gen.wrapping_add(1);
            st.running = self.handles.len();
            self.sh.work.notify_all();
        }
        let pending = PendingJob(&self.sh);
        // SAFETY: this call has not returned.
        unsafe { job.work() };
        drop(pending);
        let st = self.sh.st.lock().unwrap_or_else(|e| e.into_inner());
        assert!(!st.panicked, "a decode worker panicked");
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.sh.st.lock().unwrap_or_else(|e| e.into_inner()).quit = true;
        self.sh.work.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Time per kind of work, summed over the decode.
#[derive(Clone, Copy)]
enum Kind {
    ConvF32 = 0,
    ConvI8 = 1,
    Norm = 2,
    Quant = 3,
    Attn = 4,
}
const KIND_NAMES: [&str; 6] = ["conv f32", "conv int8", "groupnorm+silu", "norm+quantise (int8)", "attention core", "other"];

pub(super) struct Ctx {
    pub pool: Pool<f32>,
    pub pool8: Pool<u8>,
    pub par: Workers,
    pub groups: usize,
    pub int8: bool,
    /// Winograd form for the plain 3x3 f32 convolutions (off while `int8`).
    pub wino: bool,
    prof: RefCell<[f64; 6]>,
}

impl Ctx {
    pub fn new(threads: usize, groups: usize, int8: bool, wino: bool) -> Ctx {
        Ctx { wino: wino && !int8, pool: Pool::new(), pool8: Pool::new(), par: Workers::new(threads), groups, int8, prof: RefCell::new([0.0; 6]) }
    }
    fn add(&self, k: Kind, t0: Instant) {
        self.prof.borrow_mut()[k as usize] += t0.elapsed().as_secs_f64();
    }
}

// ---------------------------------------------------------------------------------------------
// f32 convolution
// ---------------------------------------------------------------------------------------------

/// Pixels per f32 tile.
const TP: usize = 4;
/// Weight bytes of one sweep along a row (a group of taps of one output-channel block): L1 share.
const SWEEP_W_BYTES: usize = 48 * 1024;
/// Source rows per task of the f32 convolution.
const BAND: usize = 4;

#[derive(Clone, Copy)]
struct Seg {
    x: *const f32,
    w: *const f32,
    /// Input channels of this segment.
    n: usize,
    /// Distance between consecutive pixels at `x`, in floats.
    xs: usize,
}

/// One tile: `TP` output pixels x `GC` output channels, accumulated over the segments in order;
/// a segment is (part of) one tap: `n` consecutive channels of `TP` consecutive input pixels
/// (pixel `p` at `x + p*xs`), weights `[n][GC]` at `w`. The accumulators start from `bias` (`GC` floats) when it is non-null,
/// otherwise from the current contents of `acc`; pixel `p` lives at `acc + p*stride`.
///
/// # Safety
/// `acc + p*stride` must be valid for `GC` floats for `p < TP` (read unless `bias` is given, and
/// write); every segment must be valid for `n` floats at `x + p*xs` and `n*GC` floats at `w`.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn tile(acc: *mut f32, stride: usize, segs: &[Seg], bias: *const f32) {
    use core::arch::aarch64::*;
    let (o0, o1, o2, o3) = (acc, acc.add(stride), acc.add(2 * stride), acc.add(3 * stride));
    let (mut a00, mut a01, mut a02, mut a03);
    let (mut a10, mut a11, mut a12, mut a13);
    let (mut a20, mut a21, mut a22, mut a23);
    let (mut a30, mut a31, mut a32, mut a33);
    if bias.is_null() {
        (a00, a01, a02, a03) = (vld1q_f32(o0), vld1q_f32(o0.add(4)), vld1q_f32(o0.add(8)), vld1q_f32(o0.add(12)));
        (a10, a11, a12, a13) = (vld1q_f32(o1), vld1q_f32(o1.add(4)), vld1q_f32(o1.add(8)), vld1q_f32(o1.add(12)));
        (a20, a21, a22, a23) = (vld1q_f32(o2), vld1q_f32(o2.add(4)), vld1q_f32(o2.add(8)), vld1q_f32(o2.add(12)));
        (a30, a31, a32, a33) = (vld1q_f32(o3), vld1q_f32(o3.add(4)), vld1q_f32(o3.add(8)), vld1q_f32(o3.add(12)));
    } else {
        (a00, a01, a02, a03) = (vld1q_f32(bias), vld1q_f32(bias.add(4)), vld1q_f32(bias.add(8)), vld1q_f32(bias.add(12)));
        (a10, a11, a12, a13) = (a00, a01, a02, a03);
        (a20, a21, a22, a23) = (a00, a01, a02, a03);
        (a30, a31, a32, a33) = (a00, a01, a02, a03);
    }
    for sg in segs {
        let (cin, c4, xs) = (sg.n, sg.n & !3, sg.xs);
        let x0 = sg.x;
        let (x1, x2, x3) = (x0.add(xs), x0.add(2 * xs), x0.add(3 * xs));
        let mut w = sg.w;
        let mut ci = 0;
        macro_rules! lane {
            ($l:literal, $v0:ident, $v1:ident, $v2:ident, $v3:ident) => {
                let (w0, w1, w2, w3) = (vld1q_f32(w), vld1q_f32(w.add(4)), vld1q_f32(w.add(8)), vld1q_f32(w.add(12)));
                w = w.add(GC);
                a00 = vfmaq_laneq_f32::<$l>(a00, w0, $v0);
                a01 = vfmaq_laneq_f32::<$l>(a01, w1, $v0);
                a02 = vfmaq_laneq_f32::<$l>(a02, w2, $v0);
                a03 = vfmaq_laneq_f32::<$l>(a03, w3, $v0);
                a10 = vfmaq_laneq_f32::<$l>(a10, w0, $v1);
                a11 = vfmaq_laneq_f32::<$l>(a11, w1, $v1);
                a12 = vfmaq_laneq_f32::<$l>(a12, w2, $v1);
                a13 = vfmaq_laneq_f32::<$l>(a13, w3, $v1);
                a20 = vfmaq_laneq_f32::<$l>(a20, w0, $v2);
                a21 = vfmaq_laneq_f32::<$l>(a21, w1, $v2);
                a22 = vfmaq_laneq_f32::<$l>(a22, w2, $v2);
                a23 = vfmaq_laneq_f32::<$l>(a23, w3, $v2);
                a30 = vfmaq_laneq_f32::<$l>(a30, w0, $v3);
                a31 = vfmaq_laneq_f32::<$l>(a31, w1, $v3);
                a32 = vfmaq_laneq_f32::<$l>(a32, w2, $v3);
                a33 = vfmaq_laneq_f32::<$l>(a33, w3, $v3);
            };
        }
        while ci < c4 {
            let (v0, v1, v2, v3) = (vld1q_f32(x0.add(ci)), vld1q_f32(x1.add(ci)), vld1q_f32(x2.add(ci)), vld1q_f32(x3.add(ci)));
            lane!(0, v0, v1, v2, v3);
            lane!(1, v0, v1, v2, v3);
            lane!(2, v0, v1, v2, v3);
            lane!(3, v0, v1, v2, v3);
            ci += 4;
        }
        while ci < cin {
            let (v0, v1, v2, v3) = (vld1q_dup_f32(x0.add(ci)), vld1q_dup_f32(x1.add(ci)), vld1q_dup_f32(x2.add(ci)), vld1q_dup_f32(x3.add(ci)));
            lane!(0, v0, v1, v2, v3);
            ci += 1;
        }
    }
    vst1q_f32(o0, a00);
    vst1q_f32(o0.add(4), a01);
    vst1q_f32(o0.add(8), a02);
    vst1q_f32(o0.add(12), a03);
    vst1q_f32(o1, a10);
    vst1q_f32(o1.add(4), a11);
    vst1q_f32(o1.add(8), a12);
    vst1q_f32(o1.add(12), a13);
    vst1q_f32(o2, a20);
    vst1q_f32(o2.add(4), a21);
    vst1q_f32(o2.add(8), a22);
    vst1q_f32(o2.add(12), a23);
    vst1q_f32(o3, a30);
    vst1q_f32(o3.add(4), a31);
    vst1q_f32(o3.add(8), a32);
    vst1q_f32(o3.add(12), a33);
}

/// Portable form of [`tile`] (same order of operations).
#[cfg(not(target_arch = "aarch64"))]
unsafe fn tile(acc: *mut f32, stride: usize, segs: &[Seg], bias: *const f32) {
    let mut a = [0f32; TP * GC];
    for p in 0..TP {
        for j in 0..GC {
            a[p * GC + j] = if bias.is_null() { *acc.add(p * stride + j) } else { *bias.add(j) };
        }
    }
    for sg in segs {
        for ci in 0..sg.n {
            for p in 0..TP {
                let s = *sg.x.add(p * sg.xs + ci);
                for j in 0..GC {
                    a[p * GC + j] = (*sg.w.add(ci * GC + j)).mul_add(s, a[p * GC + j]);
                }
            }
        }
    }
    for p in 0..TP {
        for j in 0..GC {
            *acc.add(p * stride + j) = a[p * GC + j];
        }
    }
}

/// The taps of one output phase: source offset `(dy, dx)` per tap; output pixel of source pixel
/// `(i, j)` is `(s*i + a, s*j + b)`.
struct Phase {
    a: usize,
    b: usize,
    taps: Vec<(isize, isize)>,
}

fn phases(k: usize, up: bool) -> Vec<Phase> {
    if up {
        assert_eq!(k, 3, "fused upsample needs a 3x3 convolution");
        let mut v = Vec::new();
        for a in 0..2usize {
            for b in 0..2usize {
                let mut taps = Vec::new();
                for ry in 0..2isize {
                    for rx in 0..2isize {
                        taps.push((ry - 1 + a as isize, rx - 1 + b as isize));
                    }
                }
                v.push(Phase { a, b, taps });
            }
        }
        v
    } else {
        let pad = (k / 2) as isize;
        let mut taps = Vec::new();
        for ky in 0..k as isize {
            for kx in 0..k as isize {
                taps.push((ky - pad, kx - pad));
            }
        }
        vec![Phase { a: 0, b: 0, taps }]
    }
}

/// Kernel rows (or columns) of a 3-tap kernel that land on source offset `r` (0 or 1) of output
/// phase `a` after nearest x2 upsampling.
fn up_taps(a: usize, r: usize) -> &'static [usize] {
    match (a, r) {
        (0, 0) => &[0],
        (0, _) => &[1, 2],
        (1, 0) => &[0, 1],
        _ => &[2],
    }
}

impl Conv {
    /// The 3x3 kernel folded for "nearest x2 upsample, then convolve": per output phase
    /// `(a, b)` a 2x2 kernel on the source map, `tap[ry][rx] = sum of the 3x3 taps that read the
    /// same source pixel` (f32 sums in ascending (ky, kx) order). Row-major
    /// `[phase][cout][tap][cin]`, phase = `2a + b`, tap = `2ry + rx`.
    fn folded_up(&self) -> Vec<f32> {
        assert_eq!(self.k, 3);
        let (cin, cout) = (self.cin, self.cout);
        let mut f = vec![0f32; 4 * cout * 4 * cin];
        for a in 0..2 {
            for b in 0..2 {
                for ry in 0..2 {
                    for rx in 0..2 {
                        let (ph, tap) = (a * 2 + b, ry * 2 + rx);
                        for co in 0..cout {
                            for ci in 0..cin {
                                let mut s = 0f32;
                                for &ky in up_taps(a, ry) {
                                    for &kx in up_taps(b, rx) {
                                        s += self.w[(co * cin + ci) * 9 + ky * 3 + kx];
                                    }
                                }
                                f[((ph * cout + co) * 4 + tap) * cin + ci] = s;
                            }
                        }
                    }
                }
            }
        }
        f
    }

    /// Folded upsample weights in GEMM order `[phase][cout block][tap][cin][GC]`.
    fn packed_up(&self) -> &[f32] {
        self.wup.get_or_init(|| {
            let (cin, cout) = (self.cin, self.cout);
            let f = self.folded_up();
            let nb = cout.div_ceil(GC);
            let mut p = vec![0f32; 4 * nb * 4 * cin * GC];
            for ph in 0..4 {
                for co in 0..cout {
                    for tap in 0..4 {
                        for ci in 0..cin {
                            p[(((ph * nb + co / GC) * 4 + tap) * cin + ci) * GC + co % GC] = f[((ph * cout + co) * 4 + tap) * cin + ci];
                        }
                    }
                }
            }
            p
        })
    }

    /// Folded upsample weights as int8 rows over `[tap][cin]`, one matrix per phase.
    fn packed_i8_up(&self) -> &[int8_gemm::Packed] {
        self.wq_up.get_or_init(|| {
            let (cin, cout) = (self.cin, self.cout);
            let f = self.folded_up();
            let kk = 4 * cin;
            (0..4).map(|ph| int8_gemm::Packed::from_f32(&f[ph * cout * kk..(ph + 1) * cout * kk], cout, kk, I8_CHUNK)).collect()
        })
    }
}

/// Input channels per sweep segment (a longer tap is cut into chunks of this many channels).
const KC: usize = 512;

/// One f32 GEMM convolution problem (see [`conv_f32`]); `band` computes a range of source rows
/// for a range of output-channel blocks.
struct Gemm<'a> {
    xd: &'a [f32],
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    /// Floats between consecutive input pixels (`cin` for a plain map) and between consecutive
    /// output pixels (`cout` for a plain map); larger when the pixels are records of several vectors.
    xpitch: usize,
    opitch: usize,
    /// `[phase][cout block][tap][cin][GC]`.
    wp: &'a [f32],
    /// `cout` rounded up to a multiple of `GC`.
    bias: &'a [f32],
    phases: &'a [Phase],
    /// Output scale (2 for the fused upsample).
    s: usize,
    out: SendPtr<f32>,
}

impl Gemm<'_> {
    /// Computes output rows `s*i0 .. s*i1`, output-channel blocks `cb0 .. cb1`.
    ///
    /// Loop order: block of 16 output channels, phase, row, sweep. A sweep is a few (tap, channel
    /// chunk) segments whose weights fit in L1; it walks the tiles of the row, with the tile's
    /// accumulators loaded from / stored to the output (the first sweep starts from the bias).
    ///
    /// # Safety
    /// No other thread may touch those rows and channels of `out` during the call.
    unsafe fn band(&self, i0: usize, i1: usize, cb0: usize, cb1: usize) {
        let (h, w, cin, cout, s) = (self.h, self.w, self.cin, self.cout, self.s);
        let (xpitch, opitch) = (self.xpitch, self.opitch);
        let ow = s * w;
        let nb = cout.div_ceil(GC);
        let ntaps = self.phases[0].taps.len();
        let nchunks = cin.div_ceil(KC);
        let per_sweep = (SWEEP_W_BYTES / (cin.min(KC) * GC * 4)).clamp(1, MAX_SEGS);
        let mut tmp = vec![0f32; per_sweep * TP * cin.min(KC)];
        let tmp_base = tmp.as_mut_ptr();
        let mut segs = [Seg { x: std::ptr::null(), w: std::ptr::null(), n: 0, xs: 0 }; MAX_SEGS];
        // (tap, first channel, channels) of the row's segments
        let mut list: Vec<(usize, usize, usize)> = Vec::with_capacity(ntaps * nchunks);
        for cb in cb0..cb1 {
            let ncb = GC.min(cout - cb * GC);
            for (pi, p) in self.phases.iter().enumerate() {
                let wph = &self.wp[(pi * nb + cb) * ntaps * cin * GC..(pi * nb + cb + 1) * ntaps * cin * GC];
                for i in i0..i1 {
                    let oy = s * i + p.a;
                    list.clear();
                    for (t, &(dy, _)) in p.taps.iter().enumerate() {
                        let sy = i as isize + dy;
                        if sy >= 0 && (sy as usize) < h {
                            for c0 in (0..cin).step_by(KC) {
                                list.push((t, c0, KC.min(cin - c0)));
                            }
                        }
                    }
                    let mut first = true;
                    for sweep in list.chunks(per_sweep) {
                        for j0 in (0..w).step_by(TP) {
                            let npx = TP.min(w - j0);
                            for (kq, &(t, c0, n)) in sweep.iter().enumerate() {
                                let (dy, dx) = p.taps[t];
                                let sy = (i as isize + dy) as usize;
                                let sx = j0 as isize + dx;
                                let (xp, xs) = if sx >= 0 && sx as usize + TP <= w {
                                    (self.xd[(sy * w + sx as usize) * xpitch + c0..].as_ptr(), xpitch)
                                } else {
                                    // border or partial tile: a patch of TP pixels, zero where outside
                                    // Each segment owns one disjoint scratch region until tile().
                                    let ptr = tmp_base.add(kq * TP * cin.min(KC));
                                    let dst = std::slice::from_raw_parts_mut(ptr, TP * n);
                                    for q in 0..TP {
                                        let xx = sx + q as isize;
                                        let d = &mut dst[q * n..(q + 1) * n];
                                        if xx >= 0 && (xx as usize) < w {
                                            d.copy_from_slice(&self.xd[(sy * w + xx as usize) * xpitch + c0..][..n]);
                                        } else {
                                            d.fill(0.0);
                                        }
                                    }
                                    (ptr.cast_const(), n)
                                };
                                segs[kq] = Seg { x: xp, w: wph[(t * cin + c0) * GC..].as_ptr(), n, xs };
                            }
                            let bp = if first { self.bias[cb * GC..].as_ptr() } else { std::ptr::null() };
                            let o = self.out.0.add((oy * ow + s * j0 + p.b) * opitch + cb * GC);
                            let direct = ncb == GC && npx == TP;
                            let mut acc = [0f32; TP * GC];
                            let acc_base = acc.as_mut_ptr();
                            let (ap, astride) = if direct { (o, s * opitch) } else { (acc_base, GC) };
                            if !direct && !first {
                                for q in 0..npx {
                                    std::ptr::copy_nonoverlapping(o.add(q * s * opitch), acc_base.add(q * GC), ncb);
                                }
                            }
                            tile(ap, astride, &segs[..sweep.len()], bp);
                            if !direct {
                                for q in 0..npx {
                                    std::ptr::copy_nonoverlapping(acc_base.add(q * GC), o.add(q * s * opitch), ncb);
                                }
                            }
                        }
                        first = false;
                    }
                }
            }
        }
    }
}
/// Segments per sweep at most.
const MAX_SEGS: usize = 9;

/// f32 convolution on a channel-last map: same-size (`up = false`), or nearest x2 upsampling
/// followed by the 3x3 convolution (`up = true`, output `2h x 2w`). `residual` (plain only) is
/// added to the result. One task = a band of source rows, all output channels.
pub(super) fn conv_f32(conv: &Conv, x: &F, up: bool, residual: Option<&F>, cx: &Ctx) -> F {
    if cx.wino && !up && wino_ok(conv, x.h, x.w) {
        return conv_wino(conv, x, residual, cx);
    }
    let t0 = Instant::now();
    assert_eq!(x.c, conv.cin, "conv input channels");
    assert!(!(up && residual.is_some()));
    let cout = conv.cout;
    let (h, w) = (x.h, x.w);
    let s = if up { 2 } else { 1 };
    let (oh, ow) = (s * h, s * w);
    let ph = phases(conv.k, up);
    let nb = cout.div_ceil(GC);
    let mut bias = vec![0f32; nb * GC];
    bias[..cout].copy_from_slice(&conv.b);
    if let Some(r) = residual {
        assert_eq!((r.c, r.h, r.w), (cout, oh, ow), "residual shape");
    }
    let mut out = cx.pool.take(oh * ow * cout);
    let g = Gemm { xd: x.data(), h, w, cin: conv.cin, cout, xpitch: conv.cin, opitch: cout, wp: if up { conv.packed_up() } else { conv.packed() }, bias: &bias, phases: &ph, s, out: SendPtr(out.as_mut_ptr()) };
    let band = if up { BAND / 2 } else { BAND };
    cx.par.run(h.div_ceil(band), &|bi: usize| {
        let g = &g;
        let (i0, i1) = (bi * band, ((bi + 1) * band).min(h));
        // SAFETY: this task owns output rows s*i0 .. s*i1.
        unsafe {
            g.band(i0, i1, 0, nb);
            if let Some(r) = residual {
                let rd = &r.data()[i0 * w * cout..i1 * w * cout];
                let o = std::slice::from_raw_parts_mut(g.out.0.add(i0 * w * cout), rd.len());
                for (d, v) in o.iter_mut().zip(rd) {
                    *d += *v;
                }
            }
        }
    });
    cx.add(Kind::ConvF32, t0);
    F { c: cout, h: oh, w: ow, d: out }
}

// ---------------------------------------------------------------------------------------------
// Winograd F(2x2, 3x3)
// ---------------------------------------------------------------------------------------------

/// Products per GEMM task of the Winograd convolution (sets the tiles per task).
const WINO_TASK_MACS: usize = 4 << 20;
/// Tiles per task of the Winograd input and output transforms.
const WINO_TT: usize = 32;

/// Shapes the Winograd form takes: 3x3, even height and width, channel counts that are multiples
/// of 4 and large enough for the transforms to be noise next to the products.
fn wino_ok(conv: &Conv, h: usize, w: usize) -> bool {
    conv.k == 3 && h % 2 == 0 && w % 2 == 0 && conv.cin % 4 == 0 && conv.cout % 4 == 0 && conv.cin >= 32 && cfg!(target_arch = "aarch64")
}

impl Conv {
    /// `U = G g G^T` per (output, input) channel pair, computed in f64 and rounded to f32, in GEMM
    /// order per transform position: `[pos][cout block][cin][GC]`, `pos = 4i + j`.
    fn packed_wino(&self) -> &[f32] {
        self.wwin.get_or_init(|| {
            const G: [[f64; 3]; 4] = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]];
            let (cin, cout) = (self.cin, self.cout);
            let nb = cout.div_ceil(GC);
            let mut p = vec![0f32; 16 * nb * cin * GC];
            for co in 0..cout {
                for ci in 0..cin {
                    let g = &self.w[(co * cin + ci) * 9..][..9];
                    for i in 0..4 {
                        for j in 0..4 {
                            let mut u = 0f64;
                            for a in 0..3 {
                                for b in 0..3 {
                                    u += G[i][a] * g[a * 3 + b] as f64 * G[j][b];
                                }
                            }
                            p[(((i * 4 + j) * nb + co / GC) * cin + ci) * GC + co % GC] = u as f32;
                        }
                    }
                }
            }
            p
        })
    }
}

/// Input transform of one tile, `V = B^T d B` per channel: `px[4r + c]` points at the `cin`
/// floats of input pixel `(r, c)` of the 4x4 tile; position `p` is written to `v + p*pstride`
/// (`pstride = cin`: the 16 positions of a tile are one contiguous record, so the 16 write
/// streams of a tile do not alias in L1).
///
/// # Safety
/// All 16 pointers valid for `cin` floats (`cin % 4 == 0`); `v + p*pstride` valid for `cin` floats.
#[cfg(target_arch = "aarch64")]
unsafe fn wino_in(px: &[*const f32; 16], cin: usize, v: *mut f32, pstride: usize) {
    use core::arch::aarch64::*;
    for ci in (0..cin).step_by(4) {
        let mut t = [[vdupq_n_f32(0.0); 4]; 4];
        for c in 0..4 {
            let (d0, d1, d2, d3) = (vld1q_f32(px[c].add(ci)), vld1q_f32(px[4 + c].add(ci)), vld1q_f32(px[8 + c].add(ci)), vld1q_f32(px[12 + c].add(ci)));
            t[0][c] = vsubq_f32(d0, d2);
            t[1][c] = vaddq_f32(d1, d2);
            t[2][c] = vsubq_f32(d2, d1);
            t[3][c] = vsubq_f32(d1, d3);
        }
        for (i, ti) in t.iter().enumerate() {
            let o = v.add(i * 4 * pstride + ci);
            vst1q_f32(o, vsubq_f32(ti[0], ti[2]));
            vst1q_f32(o.add(pstride), vaddq_f32(ti[1], ti[2]));
            vst1q_f32(o.add(2 * pstride), vsubq_f32(ti[2], ti[1]));
            vst1q_f32(o.add(3 * pstride), vsubq_f32(ti[1], ti[3]));
        }
    }
}

/// Output transform of one tile, `Y = A^T M A` per channel, plus bias (and residual): position
/// `p` is read from `m + p*pstride`; `o[2r + c]` points at output pixel `(r, c)` of the 2x2 tile.
///
/// # Safety
/// `m + p*pstride`, `bias`, every `o[..]` (and `res[..]`) valid for `cout` floats (`cout % 4 == 0`).
#[cfg(target_arch = "aarch64")]
unsafe fn wino_out(m: *const f32, pstride: usize, cout: usize, bias: *const f32, o: &[*mut f32; 4], res: Option<&[*const f32; 4]>) {
    use core::arch::aarch64::*;
    for co in (0..cout).step_by(4) {
        let mut s = [[vdupq_n_f32(0.0); 4]; 2];
        for j in 0..4 {
            let (m0, m1, m2, m3) = (vld1q_f32(m.add(j * pstride + co)), vld1q_f32(m.add((4 + j) * pstride + co)), vld1q_f32(m.add((8 + j) * pstride + co)), vld1q_f32(m.add((12 + j) * pstride + co)));
            s[0][j] = vaddq_f32(vaddq_f32(m0, m1), m2);
            s[1][j] = vsubq_f32(vsubq_f32(m1, m2), m3);
        }
        let b = vld1q_f32(bias.add(co));
        for (r, sr) in s.iter().enumerate() {
            let mut y0 = vaddq_f32(vaddq_f32(vaddq_f32(sr[0], sr[1]), sr[2]), b);
            let mut y1 = vaddq_f32(vsubq_f32(vsubq_f32(sr[1], sr[2]), sr[3]), b);
            if let Some(res) = res {
                y0 = vaddq_f32(y0, vld1q_f32(res[2 * r].add(co)));
                y1 = vaddq_f32(y1, vld1q_f32(res[2 * r + 1].add(co)));
            }
            vst1q_f32(o[2 * r].add(co), y0);
            vst1q_f32(o[2 * r + 1].add(co), y1);
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
unsafe fn wino_in(_: &[*const f32; 16], _: usize, _: *mut f32, _: usize) {
    unreachable!("wino_ok is false off aarch64")
}
#[cfg(not(target_arch = "aarch64"))]
unsafe fn wino_out(_: *const f32, _: usize, _: usize, _: *const f32, _: &[*mut f32; 4], _: Option<&[*const f32; 4]>) {
    unreachable!("wino_ok is false off aarch64")
}

/// Floats added to the pitch of the position planes of a Winograd block, so that the 16 planes
/// (written and read together, one stream each, by the transforms) do not share L1 sets.
const WINO_PLANE_PAD: usize = 80;
/// Transformed weights up to this many bytes: every task keeps its own `V`/`M` block in cache
/// and runs all three steps itself; above, the steps run map-wide over shared super-bands.
const WINO_LOCAL_U_BYTES: usize = 3 << 20;

std::thread_local! {
    /// Per-thread `V` and `M` blocks of the Winograd convolution (local mode).
    static WINO_SCRATCH: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// One Winograd convolution problem; a block is a run of `tb` consecutive tiles (row-major tile
/// index) whose `V` and `M` are 16 position planes of `tb * channels` floats at pitch `vp`/`mp`.
struct Wino<'a> {
    xd: &'a [f32],
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
    /// `[pos][cout block][cin][GC]`.
    u: &'a [f32],
    bias: &'a [f32],
    zero_px: &'a [f32],
    zero_bias: &'a [f32],
    one: &'a [Phase],
    residual: Option<&'a F>,
    out: SendPtr<f32>,
    /// Plane pitches of a block, in floats.
    vp: usize,
    mp: usize,
}

impl Wino<'_> {
    /// Input transform of tiles `t0 .. t0+n` into the planes at `v` (tile `t0` at plane offset 0).
    ///
    /// # Safety
    /// `v` valid for 16 planes; no other thread writes them.
    unsafe fn transform_in(&self, t0: usize, n: usize, v: *mut f32, slot0: usize) {
        let (h, w, cin, tw) = (self.h, self.w, self.cin, self.w / 2);
        for k in 0..n {
            let (ty, tx) = ((t0 + k) / tw, (t0 + k) % tw);
            let mut px = [self.zero_px.as_ptr(); 16];
            for r in 0..4 {
                let yy = (2 * ty + r) as isize - 1;
                if yy < 0 || yy as usize >= h {
                    continue;
                }
                for c in 0..4 {
                    let xx = (2 * tx + c) as isize - 1;
                    if xx >= 0 && (xx as usize) < w {
                        px[r * 4 + c] = self.xd[(yy as usize * w + xx as usize) * cin..].as_ptr();
                    }
                }
            }
            wino_in(&px, cin, v.add((slot0 + k) * cin), self.vp);
        }
    }

    /// `M[pos] = U[pos] . V[pos]` for the `n` tiles of a block.
    ///
    /// # Safety
    /// `v` complete for the block and not written meanwhile; `m` plane `pos` owned by the caller.
    unsafe fn gemm(&self, pos: usize, n: usize, v: *const f32, m: *mut f32) {
        let (cin, cout) = (self.cin, self.cout);
        let nb = cout.div_ceil(GC);
        let g = Gemm {
            xd: std::slice::from_raw_parts(v.add(pos * self.vp), n * cin),
            h: 1,
            w: n,
            cin,
            cout,
            xpitch: cin,
            opitch: cout,
            wp: &self.u[pos * nb * cin * GC..(pos + 1) * nb * cin * GC],
            bias: self.zero_bias,
            phases: self.one,
            s: 1,
            out: SendPtr(m.add(pos * self.mp)),
        };
        g.band(0, 1, 0, nb);
    }

    /// Output transform (+ bias, + residual) of tiles `t0 .. t0+n` from the planes at `m`.
    ///
    /// # Safety
    /// `m` complete for these tiles; their output pixels are owned by the caller.
    unsafe fn transform_out(&self, t0: usize, n: usize, m: *const f32, slot0: usize) {
        let (w, cout, tw) = (self.w, self.cout, self.w / 2);
        for k in 0..n {
            let (ty, tx) = ((t0 + k) / tw, (t0 + k) % tw);
            let pix = |r: usize, c: usize| ((2 * ty + r) * w + 2 * tx + c) * cout;
            let o = [self.out.0.add(pix(0, 0)), self.out.0.add(pix(0, 1)), self.out.0.add(pix(1, 0)), self.out.0.add(pix(1, 1))];
            let res = self.residual.map(|r| {
                let rd = r.data().as_ptr();
                [rd.add(pix(0, 0)), rd.add(pix(0, 1)), rd.add(pix(1, 0)), rd.add(pix(1, 1))]
            });
            wino_out(m.add((slot0 + k) * cout), self.mp, cout, self.bias.as_ptr(), &o, res.as_ref());
        }
    }
}

/// 3x3 same-size convolution in Winograd F(2x2, 3x3) form: per 2x2 output tile 16 products per
/// channel pair instead of 36. Three steps: input transform (`V`), one 1x1 GEMM per transform
/// position on the convolution tile kernel (`M = U . V`), output transform (+ bias, + residual).
///
/// `V` and `M` are four times the size of the maps, so where they live decides the speed:
/// * small weights (`U` up to 3 MB, e.g. 128 -> 128): one task = a block of tiles, all three
///   steps on thread-local `V`/`M` of about 1 MB each, `U` resident in L2;
/// * large weights: map-wide steps over super-bands of tiles (`V` about 8x `U`, 16..64 MB),
///   GEMM tasks ordered by position so that a thread keeps one position's weights in L2.
///
/// Tile values do not depend on the partition or the thread count.
fn conv_wino(conv: &Conv, x: &F, residual: Option<&F>, cx: &Ctx) -> F {
    let t0 = Instant::now();
    assert_eq!(x.c, conv.cin, "conv input channels");
    let (cin, cout) = (conv.cin, conv.cout);
    let (h, w) = (x.h, x.w);
    let nt = (h / 2) * (w / 2);
    if let Some(r) = residual {
        assert_eq!((r.c, r.h, r.w), (cout, h, w), "residual shape");
    }
    let nb = cout.div_ceil(GC);
    let u_bytes = 16 * cin * cout * 4;
    let local = u_bytes <= WINO_LOCAL_U_BYTES;
    // tiles per block: local = V and M about 1 MB each; shared = a GEMM task of about 4 M products
    let tb = if local { ((1 << 20) / (16 * 4 * cin.max(cout))).clamp(8, 1024) & !3 } else { ((WINO_TASK_MACS / (cin * cout)).div_ceil(64) * 64).clamp(64, 1024) };
    let mut out = cx.pool.take(h * w * cout);
    let zero_px = vec![0f32; cin];
    let zero_bias = vec![0f32; nb * GC];
    let one = [Phase { a: 0, b: 0, taps: vec![(0, 0)] }];
    let wn = Wino {
        xd: x.data(),
        h,
        w,
        cin,
        cout,
        u: conv.packed_wino(),
        bias: &conv.b,
        zero_px: &zero_px,
        zero_bias: &zero_bias,
        one: &one,
        residual,
        out: SendPtr(out.as_mut_ptr()),
        vp: tb * cin + WINO_PLANE_PAD,
        mp: tb * cout + WINO_PLANE_PAD,
    };
    let wn = &wn;
    if local {
        cx.par.run(nt.div_ceil(tb), &|bi: usize| {
            let (tl0, n) = (bi * tb, tb.min(nt - bi * tb));
            WINO_SCRATCH.with(|sc| {
                let mut sc = sc.borrow_mut();
                let (v, m) = &mut *sc;
                if v.len() < 16 * wn.vp {
                    v.resize(16 * wn.vp, 0.0);
                }
                if m.len() < 16 * wn.mp {
                    m.resize(16 * wn.mp, 0.0);
                }
                // SAFETY: V and M are this thread's; tiles tl0 .. tl0+n (and their output pixels) are this task's.
                unsafe {
                    wn.transform_in(tl0, n, v.as_mut_ptr(), 0);
                    for pos in 0..16 {
                        wn.gemm(pos, n, v.as_ptr(), m.as_mut_ptr());
                    }
                    wn.transform_out(tl0, n, m.as_ptr(), 0);
                }
            });
        });
    } else {
        let vsb = (8 * u_bytes).clamp(16 << 20, 64 << 20);
        let sb_blocks = (vsb / (16 * cin * 4)).div_ceil(tb).min(nt.div_ceil(tb));
        let st = sb_blocks * tb;
        let mut v = cx.pool.take(sb_blocks * 16 * wn.vp);
        let mut m = cx.pool.take(sb_blocks * 16 * wn.mp);
        let (vptr, mptr) = (SendPtr(v.as_mut_ptr()), SendPtr(m.as_mut_ptr()));
        for t_base in (0..nt).step_by(st) {
            let n = st.min(nt - t_base);
            let nblk = n.div_ceil(tb);
            let ntt = n.div_ceil(WINO_TT);
            // SAFETY (all three): a task owns the tiles it names: their slots in every plane of V
            // (step 1), plane `pos` of its block of M (step 2), their output pixels (step 3).
            // V is complete before step 2 reads it, M before step 3 reads it.
            cx.par.run(ntt, &|ti: usize| {
                let vptr = vptr;
                let (tl0, nn) = (ti * WINO_TT, WINO_TT.min(n - ti * WINO_TT));
                debug_assert!(tb % WINO_TT == 0);
                unsafe { wn.transform_in(t_base + tl0, nn, vptr.0.add((tl0 / tb) * 16 * wn.vp), tl0 % tb) };
            });
            cx.par.run(16 * nblk, &|ti: usize| {
                let (vptr, mptr) = (vptr, mptr);
                let (pos, b) = (ti / nblk, ti % nblk);
                unsafe { wn.gemm(pos, tb.min(n - b * tb), vptr.0.add(b * 16 * wn.vp), mptr.0.add(b * 16 * wn.mp)) };
            });
            cx.par.run(ntt, &|ti: usize| {
                let mptr = mptr;
                let (tl0, nn) = (ti * WINO_TT, WINO_TT.min(n - ti * WINO_TT));
                unsafe { wn.transform_out(t_base + tl0, nn, mptr.0.add((tl0 / tb) * 16 * wn.mp), tl0 % tb) };
            });
        }
        cx.pool.give(v);
        cx.pool.give(m);
    }
    cx.add(Kind::ConvF32, t0);
    F { c: cout, h, w, d: out }
}

// ---------------------------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------------------------

/// Bytes of one score block (query block x all keys): L2 share of a core.
const ATT_SCORE_BYTES: usize = 2 << 20;

/// Single-head attention `out = softmax(Q K^T / sqrt(c)) V` on position-major `[l][c]` operands,
/// as two GEMMs per block of queries on the convolution tile: scores `[nq][l]` (K as the weight
/// matrix), softmax in place (max-subtracted, `exp_fast`), then `[nq][c]` (V as the weight
/// matrix). One task = one query block with its own score buffer; the block size does not depend
/// on the thread count, so neither does the result.
fn attention_gemm(q: &[f32], k: &[f32], v: &[f32], l: usize, c: usize, cx: &Ctx) -> Vec<f32> {
    let scale = 1.0 / (c as f32).sqrt();
    let one = [Phase { a: 0, b: 0, taps: vec![(0, 0)] }];
    // K as GEMM weights: output channel = key, input = channel: [key block][c][GC]
    let nbk = l.div_ceil(GC);
    let mut kp = vec![0f32; nbk * c * GC];
    for j in 0..l {
        for ci in 0..c {
            kp[((j / GC) * c + ci) * GC + j % GC] = k[j * c + ci];
        }
    }
    // V as GEMM weights: output channel = channel, input = key: [channel block][l][GC]
    let nbc = c.div_ceil(GC);
    let mut vp = vec![0f32; nbc * l * GC];
    for j in 0..l {
        for ch in 0..c {
            vp[((ch / GC) * l + j) * GC + ch % GC] = v[j * c + ch];
        }
    }
    let zero_k = vec![0f32; nbk * GC];
    let zero_c = vec![0f32; nbc * GC];
    let nq = (ATT_SCORE_BYTES / (4 * l)).clamp(TP, 128) & !(TP - 1);
    let mut out = vec![0f32; l * c];
    let op = SendPtr(out.as_mut_ptr());
    cx.par.run(l.div_ceil(nq), &|bi: usize| {
        let op = op;
        let q0 = bi * nq;
        let n = nq.min(l - q0);
        let mut sc = vec![0f32; n * l];
        // SAFETY: `sc` is private to this task; output rows q0 .. q0+n belong to this task.
        unsafe {
            let g1 = Gemm { xd: &q[q0 * c..(q0 + n) * c], h: 1, w: n, cin: c, cout: l, xpitch: c, opitch: l, wp: &kp, bias: &zero_k, phases: &one, s: 1, out: SendPtr(sc.as_mut_ptr()) };
            g1.band(0, 1, 0, nbk);
            for row in sc.chunks_exact_mut(l) {
                softmax_row(row, scale);
            }
            let g2 = Gemm { xd: &sc, h: 1, w: n, cin: l, cout: c, xpitch: l, opitch: c, wp: &vp, bias: &zero_c, phases: &one, s: 1, out: SendPtr(op.0.add(q0 * c)) };
            g2.band(0, 1, 0, nbc);
        }
    });
    out
}

/// `row = softmax(row * scale)`: scale, subtract the maximum, `exp_fast`, divide by the sum
/// (the sum is taken in four lanes, then lane 0 + 1 + 2 + 3, then the tail in order).
fn softmax_row(row: &mut [f32], scale: f32) {
    let mut m = f32::NEG_INFINITY;
    for x in row.iter_mut() {
        *x *= scale;
        m = m.max(*x);
    }
    let n4 = if cfg!(target_arch = "aarch64") { row.len() & !3 } else { 0 };
    #[cfg(not(target_arch = "aarch64"))]
    let mut sum = 0f32;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: indices below n4 <= len.
    let mut sum = unsafe {
        use core::arch::aarch64::*;
        let (mv, mut acc) = (vdupq_n_f32(m), vdupq_n_f32(0.0));
        for i in (0..n4).step_by(4) {
            let e = exp_fast_v(vsubq_f32(vld1q_f32(row.as_ptr().add(i)), mv));
            vst1q_f32(row.as_mut_ptr().add(i), e);
            acc = vaddq_f32(acc, e);
        }
        ((vgetq_lane_f32::<0>(acc) + vgetq_lane_f32::<1>(acc)) + vgetq_lane_f32::<2>(acc)) + vgetq_lane_f32::<3>(acc)
    };
    for x in row[n4..].iter_mut() {
        *x = exp_fast(*x - m);
        sum += *x;
    }
    let inv = 1.0 / sum;
    for x in row.iter_mut() {
        *x *= inv;
    }
}

// ---------------------------------------------------------------------------------------------
// GroupNorm (+ SiLU)
// ---------------------------------------------------------------------------------------------

/// Per-channel form of one GroupNorm: `y = (v - mean[c]) * a[c] + b[c]`.
struct GnCoef {
    mean: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

/// GroupNorm statistics of a channel-last map. Per row and channel f64 sums of `v` and `v*v`
/// (one task per row), then combined per group in (row, channel) order: the result does not
/// depend on the thread count. `var = E[v^2] - mean^2` in f64.
fn gn_coef(x: &F, norm: &Norm, cx: &Ctx) -> GnCoef {
    let (c, h, w) = (x.c, x.h, x.w);
    let groups = cx.groups;
    assert!(groups > 0 && c % groups == 0, "group_norm: {c} channels, {groups} groups");
    let cg = c / groups;
    let xd = x.data();
    let mut part = vec![0f64; h * 2 * c];
    let pp = SendPtr(part.as_mut_ptr());
    cx.par.run(h, &|y: usize| {
        let pp = pp;
        // SAFETY: row y's 2*c partial sums belong to this task.
        let (s, q) = unsafe { std::slice::from_raw_parts_mut(pp.0.add(y * 2 * c), 2 * c) }.split_at_mut(c);
        for px in xd[y * w * c..(y + 1) * w * c].chunks_exact(c) {
            for ci in 0..c {
                let v = px[ci] as f64;
                s[ci] += v;
                q[ci] += v * v;
            }
        }
    });
    let n = (cg * h * w) as f64;
    let mut co = GnCoef { mean: vec![0.0; c], a: vec![0.0; c], b: norm.b.clone() };
    for g in 0..groups {
        let (mut s, mut q) = (0f64, 0f64);
        for y in 0..h {
            for ch in g * cg..(g + 1) * cg {
                s += part[y * 2 * c + ch];
                q += part[y * 2 * c + c + ch];
            }
        }
        let mean = s / n;
        let var = (q / n - mean * mean).max(0.0);
        let inv = 1.0 / (var + GN_EPS).sqrt();
        for ch in g * cg..(g + 1) * cg {
            co.mean[ch] = mean as f32;
            co.a[ch] = (inv * norm.w[ch] as f64) as f32;
        }
    }
    co
}

/// Affine (+ SiLU) of `n` floats laid out as pixels of `c` channels. `src == dst` is allowed.
/// Arithmetic per element is exactly `y = (v - mean)*a + b; y / (1 + exp_fast(-y))` with separate
/// multiplies and adds, the same as the channel-major path.
///
/// # Safety
/// `src` and `dst` must be valid for `n` floats (`n` a multiple of `c`), and either equal or disjoint.
unsafe fn gn_apply(src: *const f32, dst: *mut f32, n: usize, c: usize, co: &GnCoef, silu: bool) {
    let c4 = c & !3;
    let mut base = 0;
    while base < n {
        let (s, d) = (src.add(base), dst.add(base));
        #[cfg(target_arch = "aarch64")]
        {
            use core::arch::aarch64::*;
            let mut ci = 0;
            while ci < c4 {
                let v = vld1q_f32(s.add(ci));
                let y = vaddq_f32(vmulq_f32(vsubq_f32(v, vld1q_f32(co.mean.as_ptr().add(ci))), vld1q_f32(co.a.as_ptr().add(ci))), vld1q_f32(co.b.as_ptr().add(ci)));
                let o = if silu { vdivq_f32(y, vaddq_f32(vdupq_n_f32(1.0), exp_fast_v(vnegq_f32(y)))) } else { y };
                vst1q_f32(d.add(ci), o);
                ci += 4;
            }
        }
        let tail0 = if cfg!(target_arch = "aarch64") { c4 } else { 0 };
        for ci in tail0..c {
            let y = (*s.add(ci) - co.mean[ci]) * co.a[ci] + co.b[ci];
            *d.add(ci) = if silu { y / (1.0 + exp_fast(-y)) } else { y };
        }
        base += c;
    }
}

/// Four lanes of `super::exp_fast` (same operations in the same order, nothing fused).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp_fast_v(x: core::arch::aarch64::float32x4_t) -> core::arch::aarch64::float32x4_t {
    use core::arch::aarch64::*;
    let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(87.0));
    let n = vrndaq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
    let r = vaddq_f32(vsubq_f32(x, vmulq_f32(n, vdupq_n_f32(0.693_359_4))), vmulq_f32(n, vdupq_n_f32(2.121_944_4e-4)));
    let mut p = vaddq_f32(vmulq_f32(vdupq_n_f32(1.987_569_1e-4), r), vdupq_n_f32(1.398_2e-3));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(8.333_452e-3));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(4.166_579_6e-2));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(1.666_666_5e-1));
    p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(0.5));
    let y = vaddq_f32(vaddq_f32(vmulq_f32(vmulq_f32(p, r), r), r), vdupq_n_f32(1.0));
    let e = vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)));
    vmulq_f32(y, vreinterpretq_f32_s32(e))
}

/// GroupNorm (+ SiLU) into a new map.
fn group_norm(x: &F, norm: &Norm, silu: bool, cx: &Ctx) -> F {
    let t0 = Instant::now();
    let co = gn_coef(x, norm, cx);
    let (c, h, w) = (x.c, x.h, x.w);
    let mut out = cx.pool.take(c * h * w);
    let op = SendPtr(out.as_mut_ptr());
    let xd = x.data();
    cx.par.run(h, &|y: usize| {
        let op = op;
        // SAFETY: row y of the output belongs to this task; src and dst are different buffers.
        unsafe { gn_apply(xd.as_ptr().add(y * w * c), op.0.add(y * w * c), w * c, c, &co, silu) }
    });
    cx.add(Kind::Norm, t0);
    F { c, h, w, d: out }
}

/// GroupNorm (+ SiLU) in place.
fn group_norm_inplace(x: &mut F, norm: &Norm, silu: bool, cx: &Ctx) {
    let t0 = Instant::now();
    let co = gn_coef(x, norm, cx);
    let (c, h, w) = (x.c, x.h, x.w);
    let op = SendPtr(x.d.as_mut_ptr());
    cx.par.run(h, &|y: usize| {
        let op = op;
        // SAFETY: row y belongs to this task; in place (src == dst).
        unsafe { gn_apply(op.0.add(y * w * c), op.0.add(y * w * c), w * c, c, &co, silu) }
    });
    cx.add(Kind::Norm, t0);
}

// ---------------------------------------------------------------------------------------------
// int8 path
// ---------------------------------------------------------------------------------------------

/// int8 activations of a channel-last map in the SMMLA tile layout, three shifted copies.
///
/// Per row `y`, per shift `s` in 0..3, per group `g` of 8 pixel slots: `c/8` k-steps of 64 bytes
/// (slot-major, 8 channels each) and `c/64` chunks of 8 scales. Slot `j` of group `g` of shift
/// `s` holds pixel `8g + j + s - 1` (zero codes and scale outside the row), so the operand of a
/// tap with column offset `dx` for output pixels `8g .. 8g+8` is group `g` of shift `dx + 1`.
struct QF {
    c: usize,
    h: usize,
    w: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
}

impl QF {
    /// Buffers come from the pools with stale contents: `quantize` writes every slot, the
    /// out-of-row edge slots included.
    fn new(c: usize, h: usize, w: usize, cx: &Ctx) -> QF {
        assert!(c % I8_CHUNK == 0 && w % 8 == 0);
        QF { c, h, w, codes: cx.pool8.take(h * 3 * w * c), scales: cx.pool.take(h * 3 * (w / 8) * (c / I8_CHUNK) * 8) }
    }
    fn release(self, cx: &Ctx) {
        cx.pool8.give(self.codes);
        cx.pool.give(self.scales);
    }
    fn row_code_bytes(&self) -> usize {
        3 * self.w * self.c
    }
    fn row_scales(&self) -> usize {
        3 * (self.w / 8) * (self.c / I8_CHUNK) * 8
    }
}

/// Quantises pixels `x0 .. x0+npx` of one row (`npx*c` floats at `src`) into the three shifted copies of the row.
/// Per pixel and chunk of 64 channels: `scale = max|v| / 127`, `code = round_half_away(v * (1/scale))`
/// clamped to +-127 (scale 0 and zero codes for an all-zero or non-finite chunk): the same rule
/// as the channel-major int8 path.
///
/// # Safety
/// `src` valid for `npx*c` floats; `codes`/`scales` point at the row's region, owned by the caller.
unsafe fn quantize_px(src: *const f32, x0: usize, npx: usize, w: usize, c: usize, codes: *mut u8, scales: *mut f32) {
    let nch = c / I8_CHUNK;
    let shift_bytes = w * c;
    let shift_scales = (w / 8) * nch * 8;
    for x in x0..x0 + npx {
        let px = src.add((x - x0) * c);
        for ch in 0..nch {
            let v = px.add(ch * I8_CHUNK);
            let mut q = [0u64; I8_CHUNK / 8];
            #[cfg(target_arch = "aarch64")]
            let amax = {
                use core::arch::aarch64::*;
                let mut m = vdupq_n_f32(0.0);
                for i in (0..I8_CHUNK).step_by(4) {
                    m = vmaxq_f32(m, vabsq_f32(vld1q_f32(v.add(i))));
                }
                vmaxvq_f32(m)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let amax = (0..I8_CHUNK).fold(0f32, |m, i| m.max((*v.add(i)).abs()));
            let s = if amax > 0.0 && amax.is_finite() { amax / 127.0 } else { 0.0 };
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            #[cfg(target_arch = "aarch64")]
            {
                use core::arch::aarch64::*;
                let (vi, lo, hi) = (vdupq_n_f32(inv), vdupq_n_f32(-127.0), vdupq_n_f32(127.0));
                for (st, qv) in q.iter_mut().enumerate() {
                    let a = vminq_f32(vmaxq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(v.add(st * 8)), vi)), lo), hi);
                    let b = vminq_f32(vmaxq_f32(vrndaq_f32(vmulq_f32(vld1q_f32(v.add(st * 8 + 4)), vi)), lo), hi);
                    let n16 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(a)), vqmovn_s32(vcvtq_s32_f32(b)));
                    *qv = vget_lane_u64::<0>(vreinterpret_u64_s8(vqmovn_s16(n16)));
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for (st, qv) in q.iter_mut().enumerate() {
                let mut b = [0u8; 8];
                for i in 0..8 {
                    b[i] = (*v.add(st * 8 + i) * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                }
                *qv = u64::from_le_bytes(b);
            }
            for sft in 0..3usize {
                let pos = x as isize + 1 - sft as isize;
                if pos < 0 || pos as usize >= w {
                    continue;
                }
                let (g, j) = (pos as usize / 8, pos as usize % 8);
                *scales.add(sft * shift_scales + (g * nch + ch) * 8 + j) = s;
                let cb = codes.add(sft * shift_bytes + g * 8 * c + ch * I8_CHUNK * 8 + j * 8);
                for (st, qv) in q.iter().enumerate() {
                    (cb.add(st * 64) as *mut u64).write_unaligned(*qv);
                }
            }
        }
    }
}

/// Zeroes the two slots of a row that lie outside the map (pixel -1 in shift 0, pixel `w` in shift 2).
///
/// # Safety
/// As [`quantize_px`].
unsafe fn zero_edges(w: usize, c: usize, codes: *mut u8, scales: *mut f32) {
    let nch = c / I8_CHUNK;
    for (sft, g, j) in [(0usize, 0usize, 0usize), (2, w / 8 - 1, 7)] {
        for ch in 0..nch {
            *scales.add(sft * (w / 8) * nch * 8 + (g * nch + ch) * 8 + j) = 0.0;
        }
        let cb = codes.add(sft * w * c + g * 8 * c + j * 8);
        for st in 0..c / 8 {
            (cb.add(st * 64) as *mut u64).write_unaligned(0);
        }
    }
}

/// Pixels per slab of the fused norm + quantise pass (the f32 slab stays in L1/L2).
const Q_SLAB: usize = 32;

/// Quantises a map; with `norm`, GroupNorm + SiLU are applied first, slab by slab through a
/// small f32 buffer (the f32 result is never stored as a whole map).
fn quantize(x: &F, norm: Option<&Norm>, cx: &Ctx) -> QF {
    let t0 = Instant::now();
    let (c, h, w) = (x.c, x.h, x.w);
    let co = norm.map(|n| gn_coef(x, n, cx));
    let mut q = QF::new(c, h, w, cx);
    let (rb, rs) = (q.row_code_bytes(), q.row_scales());
    let (cp, sp) = (SendPtr(q.codes.as_mut_ptr()), SendPtr(q.scales.as_mut_ptr()));
    let xd = x.data();
    cx.par.run(h, &|y: usize| {
        let (cp, sp) = (cp, sp);
        let src = xd[y * w * c..(y + 1) * w * c].as_ptr();
        // SAFETY: row y's codes and scales belong to this task.
        unsafe {
            let (codes, scales) = (cp.0.add(y * rb), sp.0.add(y * rs));
            zero_edges(w, c, codes, scales);
            match &co {
                Some(co) => {
                    let mut tmp = vec![0f32; Q_SLAB * c];
                    for x0 in (0..w).step_by(Q_SLAB) {
                        let npx = Q_SLAB.min(w - x0);
                        gn_apply(src.add(x0 * c), tmp.as_mut_ptr(), npx * c, c, co, true);
                        quantize_px(tmp.as_ptr(), x0, npx, w, c, codes, scales);
                    }
                }
                None => quantize_px(src, 0, w, w, c, codes, scales),
            }
        }
    });
    cx.add(Kind::Quant, t0);
    q
}

/// Shapes the int8 path takes: 3x3, input channels a multiple of 64, output channels a multiple
/// of 8, width a multiple of 8.
fn i8_ok(conv: &Conv, w: usize) -> bool {
    conv.k == 3 && conv.cin >= I8_CHUNK && conv.cin % I8_CHUNK == 0 && conv.cout % 8 == 0 && w % 8 == 0 && cfg!(target_arch = "aarch64")
}

/// Weight bytes per cache block of the int8 convolution (row groups x taps): L2 share.
const I8_W_BLOCK_BYTES: usize = 256 * 1024;
/// Pixel groups per batch of the int8 convolution.
const I8_XB: usize = 4;

/// LOSSY int8 convolution (see `Conv::forward_int8` for the number format): for every group of 8
/// output pixels and 8 output channels the SMMLA tile is called once per tap, accumulating in
/// the output; then bias and the optional residual are added. `up` as in [`conv_f32`].
fn conv_i8(conv: &Conv, q: &QF, up: bool, residual: Option<&F>, cx: &Ctx) -> F {
    use int8_gemm::{raw, Packed, G};
    let t0 = Instant::now();
    assert_eq!(q.c, conv.cin);
    let (cin, cout) = (conv.cin, conv.cout);
    let (h, w) = (q.h, q.w);
    let s = if up { 2 } else { 1 };
    let (oh, ow) = (s * h, s * w);
    let ph = phases(conv.k, up);
    let ntaps = ph[0].taps.len();
    let wq: &[Packed] = if up { conv.packed_i8_up() } else { std::slice::from_ref(conv.packed_i8()) };
    let nch = cin / I8_CHUNK;
    let row_groups = cout / G;
    let gb = ntaps * cin * G; // bytes of one weight row group
    let rblock = (I8_W_BLOCK_BYTES / gb).max(1);
    let (rb, rs) = (q.row_code_bytes(), q.row_scales());
    let (shift_bytes, shift_scales) = (w * cin, (w / 8) * nch * 8);
    let mut out = cx.pool.take(oh * ow * cout);
    let optr = SendPtr(out.as_mut_ptr());
    let (ph_ref, bias) = (&ph, &conv.b);
    cx.par.run(h, &|i: usize| {
        let optr = optr;
        for (pi, p) in ph_ref.iter().enumerate() {
            let wm = &wq[pi];
            let (wd, ws) = (wm.data().as_ptr() as *const i8, wm.scales().as_ptr());
            let oy = s * i + p.a;
            for g0 in (0..w / 8).step_by(I8_XB) {
                let g1 = (g0 + I8_XB).min(w / 8);
                let mut r0 = 0;
                while r0 < row_groups {
                    let r1 = (r0 + rblock).min(row_groups);
                    for g in g0..g1 {
                        for r in r0..r1 {
                            let mut acc = 0;
                            for (t, &(dy, dx)) in p.taps.iter().enumerate() {
                                let sy = i as isize + dy;
                                if sy < 0 || sy as usize >= h {
                                    continue;
                                }
                                let sy = sy as usize;
                                let sft = (dx + 1) as usize;
                                // SAFETY: operands are one 8-pixel group of one tap (cin*8 bytes, nch*8
                                // scales) inside row sy of `q` and one row group of the weights; the 8x8
                                // output tile (pixel stride s*cout) lies in output row oy, owned by this task.
                                unsafe {
                                    raw::i8g_tile8x8(
                                        q.codes.as_ptr().add(sy * rb + sft * shift_bytes + g * 8 * cin) as *const i8,
                                        wd.add(r * gb + t * cin * G),
                                        I8_CHUNK / 8,
                                        nch,
                                        q.scales.as_ptr().add(sy * rs + sft * shift_scales + g * nch * 8),
                                        ws.add((r * ntaps + t) * nch * G),
                                        optr.0.add((oy * ow + s * 8 * g + p.b) * cout + r * G),
                                        (s * cout) as isize,
                                        acc,
                                        0,
                                    );
                                }
                                acc = 1;
                            }
                        }
                    }
                    r0 = r1;
                }
            }
        }
        // bias (+ residual) on the finished output rows of this source row
        for oy in s * i..s * (i + 1) {
            // SAFETY: output rows s*i .. s*(i+1) belong to this task and are complete.
            let o = unsafe { std::slice::from_raw_parts_mut(optr.0.add(oy * ow * cout), ow * cout) };
            for px in o.chunks_exact_mut(cout) {
                for (d, b) in px.iter_mut().zip(bias) {
                    *d += *b;
                }
            }
            if let Some(r) = residual {
                for (d, v) in o.iter_mut().zip(&r.data()[oy * ow * cout..(oy + 1) * ow * cout]) {
                    *d += *v;
                }
            }
        }
    });
    cx.add(Kind::ConvI8, t0);
    F { c: cout, h: oh, w: ow, d: out }
}

// ---------------------------------------------------------------------------------------------
// blocks
// ---------------------------------------------------------------------------------------------

/// `conv(silu(groupnorm(x)))` (+ residual). `owned` = the input may be overwritten (norm in place).
fn norm_conv(x: &mut F, owned: bool, norm: &Norm, conv: &Conv, residual: Option<&F>, cx: &Ctx) -> F {
    if cx.int8 && i8_ok(conv, x.w) {
        let q = quantize(x, Some(norm), cx);
        let o = conv_i8(conv, &q, false, residual, cx);
        q.release(cx);
        return o;
    }
    if owned {
        group_norm_inplace(x, norm, true, cx);
        conv_f32(conv, x, false, residual, cx)
    } else {
        let hn = group_norm(x, norm, true, cx);
        let o = conv_f32(conv, &hn, false, residual, cx);
        cx.pool.give(hn.d);
        o
    }
}

fn resblock(b: &ResBlock, mut x: F, cx: &Ctx) -> F {
    let mut h = norm_conv(&mut x, false, &b.norm1, &b.conv1, None, cx);
    let out = match &b.shortcut {
        Some(sc) => {
            let scx = conv_f32(sc, &x, false, None, cx);
            let o = norm_conv(&mut h, true, &b.norm2, &b.conv2, Some(&scx), cx);
            cx.pool.give(scx.d);
            o
        }
        None => norm_conv(&mut h, true, &b.norm2, &b.conv2, Some(&x), cx),
    };
    cx.pool.give(h.d);
    cx.pool.give(x.d);
    out
}

fn attn_block(a: &Attn, x: F, cx: &Ctx) -> F {
    let (c, h, w) = (x.c, x.h, x.w);
    let l = h * w;
    let xn = group_norm(&x, &a.norm, false, cx);
    // the 1x1 projections of a channel-last map are already position-major [L][C]
    let q = conv_f32(&a.q, &xn, false, None, cx);
    let k = conv_f32(&a.k, &xn, false, None, cx);
    let v = conv_f32(&a.v, &xn, false, None, cx);
    let t0 = Instant::now();
    let o = attention_gemm(q.data(), k.data(), v.data(), l, c, cx);
    cx.add(Kind::Attn, t0);
    let ot = F { c, h, w, d: o };
    let out = conv_f32(&a.proj, &ot, false, Some(&x), cx);
    for f in [xn, q, k, v, x] {
        cx.pool.give(f.d);
    }
    out
}

/// Nearest x2 upsampling + 3x3 convolution, fused.
fn upsample_conv(conv: &Conv, x: F, cx: &Ctx) -> F {
    let o = if cx.int8 && i8_ok(conv, x.w) {
        let q = quantize(&x, None, cx);
        let o = conv_i8(conv, &q, true, None, cx);
        q.release(cx);
        o
    } else {
        conv_f32(conv, &x, true, None, cx)
    };
    cx.pool.give(x.d);
    o
}

impl VaeDecoder {
    /// The channel-last decode (inputs already validated by `decode_timed`).
    pub(super) fn decode_nhwc(&self, z: &[f32], channels: usize, h: usize, w: usize, threads: usize) -> (Vec<f32>, DecodeTimes) {
        let mut tm = DecodeTimes::default();
        let t_all = Instant::now();
        let cx = Ctx::new(threads, self.groups, int8_conv(), winograd());
        let base_c = channels / (PATCH * PATCH);
        let (hh, ww) = (PATCH * h, PATCH * w);
        let t0 = Instant::now();
        let mut d = vec![0f32; base_c * hh * ww];
        for cc in 0..base_c {
            for b in 0..PATCH {
                for a in 0..PATCH {
                    let in_c = cc * PATCH * PATCH + b * PATCH + a;
                    for y in 0..h {
                        let src = &z[(in_c * h + y) * w..][..w];
                        for (xx, v) in src.iter().enumerate() {
                            d[((PATCH * y + b) * ww + PATCH * xx + a) * base_c + cc] = *v;
                        }
                    }
                }
            }
        }
        let x = F { c: base_c, h: hh, w: ww, d };
        let x = conv_f32(&self.post_quant, &x, false, None, &cx);
        tm.push("unpatchify+post_quant", t0);
        let t0 = Instant::now();
        let mut x = conv_f32(&self.conv_in, &x, false, None, &cx);
        tm.push("conv_in", t0);
        let t0 = Instant::now();
        x = resblock(&self.mid1, x, &cx);
        tm.push("mid_resnet_1", t0);
        let t0 = Instant::now();
        x = attn_block(&self.mid_attn, x, &cx);
        tm.push("mid_attn", t0);
        let t0 = Instant::now();
        x = resblock(&self.mid2, x, &cx);
        tm.push("mid_resnet_2", t0);
        for (li, level) in self.up.iter().enumerate() {
            let t0 = Instant::now();
            for b in &level.blocks {
                x = resblock(b, x, &cx);
            }
            tm.push(&format!("up_level{li}_resnets ({}x{})", x.h, x.w), t0);
            if let Some(up) = &level.upsample {
                let t0 = Instant::now();
                x = upsample_conv(up, x, &cx);
                tm.push(&format!("up_level{li}_upsample_conv"), t0);
            }
        }
        let t0 = Instant::now();
        group_norm_inplace(&mut x, &self.norm_out, true, &cx);
        let y = conv_f32(&self.conv_out, &x, false, None, &cx);
        // channel-last [H][W][3] -> the public [3][H][W]
        let (oc, oh, ow) = (y.c, y.h, y.w);
        let mut out = vec![0f32; oc * oh * ow];
        for (i, px) in y.data().chunks_exact(oc).enumerate() {
            for (ch, v) in px.iter().enumerate() {
                out[ch * oh * ow + i] = *v;
            }
        }
        tm.push("norm_out+conv_out", t0);
        tm.total = t_all.elapsed().as_secs_f64();
        let prof = cx.prof.borrow();
        let known: f64 = prof.iter().sum();
        for (name, secs) in KIND_NAMES.iter().zip(prof.iter()) {
            let secs = if *name == "other" { tm.total - known } else { *secs };
            tm.kinds.push((name.to_string(), secs));
        }
        (out, tm)
    }
}

#[cfg(test)]
pub(super) mod testing {
    //! Layout converters and entry points for the unit tests and the kernel bench of `decoder.rs`.
    use super::*;
    use crate::decoder::Feat;

    #[test]
    fn caller_unwind_waits_for_workers_and_leaves_the_pool_reusable() {
        let pool = Workers::new(2);
        let caller = std::thread::current().id();
        let ready = std::sync::Barrier::new(2);
        let finished = std::sync::atomic::AtomicBool::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(2, &|_| {
                ready.wait();
                if std::thread::current().id() == caller {
                    panic!("synthetic caller failure");
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
                finished.store(true, Ordering::Relaxed);
            });
        }));
        assert!(result.is_err());
        assert!(finished.load(Ordering::Relaxed));
        let count = AtomicUsize::new(0);
        pool.run(2, &|_| { count.fetch_add(1, Ordering::Relaxed); });
        assert_eq!(count.load(Ordering::Relaxed), 2);
    }

    pub fn to_nhwc(x: &Feat) -> F {
        let (c, hw) = (x.c, x.h * x.w);
        let mut d = vec![0f32; c * hw];
        for ch in 0..c {
            for i in 0..hw {
                d[i * c + ch] = x.d[ch * hw + i];
            }
        }
        F { c, h: x.h, w: x.w, d }
    }

    pub fn to_planes(x: &F) -> Feat {
        let (c, hw) = (x.c, x.h * x.w);
        let mut d = vec![0f32; c * hw];
        for ch in 0..c {
            for i in 0..hw {
                d[ch * hw + i] = x.d[i * c + ch];
            }
        }
        Feat { c, h: x.h, w: x.w, d }
    }

    pub fn conv(conv: &Conv, x: &F, up: bool, residual: Option<&F>, threads: usize) -> F {
        let cx = Ctx::new(threads, 1, false, false);
        let s = if up { 2 } else { 1 };
        cx.pool.give(vec![f32::NAN; s * s * x.h * x.w * conv.cout]);
        conv_f32(conv, x, up, residual, &cx)
    }

    pub fn conv_winograd(conv: &Conv, x: &F, residual: Option<&F>, threads: usize) -> F {
        let cx = Ctx::new(threads, 1, false, true);
        cx.pool.give(vec![f32::NAN; x.h * x.w * conv.cout]);
        conv_wino(conv, x, residual, &cx)
    }

    pub fn wino_ok(c: &Conv, h: usize, w: usize) -> bool {
        super::wino_ok(c, h, w)
    }

    pub fn attention(q: &[f32], k: &[f32], v: &[f32], l: usize, c: usize, threads: usize) -> Vec<f32> {
        attention_gemm(q, k, v, l, c, &Ctx::new(threads, 1, false, false))
    }

    pub fn conv_int8(conv: &Conv, x: &F, up: bool, residual: Option<&F>, threads: usize) -> F {
        assert!(i8_ok(conv, x.w));
        let cx = Ctx::new(threads, 1, true, false);
        // stale pool buffers: every slot the convolution reads must have been written by `quantize`
        cx.pool8.give(vec![0xa5u8; 3 * x.n()]);
        cx.pool.give(vec![7.0f32; 3 * (x.w / 8) * x.h * (x.c / I8_CHUNK) * 8]);
        let q = quantize(x, None, &cx);
        conv_i8(conv, &q, up, residual, &cx)
    }

    pub fn int8_ok(c: &Conv, w: usize) -> bool {
        i8_ok(c, w)
    }

    pub fn norm_silu(x: &F, norm: &Norm, groups: usize, silu: bool, inplace: bool, threads: usize) -> F {
        let cx = Ctx::new(threads, groups, false, false);
        if inplace {
            let mut y = F { c: x.c, h: x.h, w: x.w, d: x.data().to_vec() };
            group_norm_inplace(&mut y, norm, silu, &cx);
            y
        } else {
            group_norm(x, norm, silu, &cx)
        }
    }
}
