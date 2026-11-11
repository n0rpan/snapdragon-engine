//! Expert cache: one anonymous THP arena split into per-layer slot pools (fixed per-layer
//! quotas, slot = sidecar record stride), CLOCK (second chance) replacement inside each pool,
//! and a per-slot state word (EMPTY / PENDING / READY) that the IO thread flips with a release
//! store when a read lands. Compute threads only ever load the state (acquire) and spin.
//!
//! Single-writer discipline: the map / owner / ref / stamp arrays are touched only by the
//! control thread (team thread 0 between barriers, or the main thread outside a job).

use crate::io::{IoQueue, Promote, Req, ReqClass};
use crate::model::{N_EXPERT, N_LAYER};
use crate::sidecar::Layout;
use crate::uring::Arena;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering::*};

pub const EMPTY: u8 = 0;
pub const PENDING: u8 = 1;
pub const READY: u8 = 2;
/// the read failed (short read twice, see io.rs): the bytes are invalid; the next lookup re-reads the slot
pub const FAILED: u8 = 3;

struct Pool {
    slot: usize,
    n: usize,
    base: *mut u8,
    map: Vec<i32>,   // expert -> slot
    owner: Vec<i32>, // slot -> expert
    refb: Vec<bool>,
    stamp: Vec<u64>,
    hand: usize,
    state: Vec<AtomicU8>,
    // Policy::Lfu: demand-access counts of every expert of the layer (resident or not), halved every
    // LFU_AGE accesses of the layer; last access tick per slot (tie-break: least recently used first)
    cnt: Vec<u32>,
    last: Vec<u64>,
    tick: u64,
}

/// Eviction policy of the per-layer pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// second-chance CLOCK (M0, the default)
    Clock,
    /// "perfect" LFU with aging: victim = the evictable resident expert with the fewest demand accesses
    /// (counts kept for all experts, halved every LFU_AGE accesses of the layer), ties to the least recently
    /// used. Offline replay (tools/eviction_replay.py, fixed353): -9% misses vs CLOCK at 20 GiB.
    Lfu,
}

const LFU_AGE: u64 = 1000;

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub demand_lookups: u64,
    pub demand_hits_ready: u64,
    /// found in flight (issued by an earlier prefetch)
    pub demand_hits_pending: u64,
    /// found because a prefetch brought it in (subset of hits_ready + hits_pending)
    pub demand_hits_prefetched: u64,
    pub demand_misses: u64,
    pub prefetch_lookups: u64,
    pub prefetch_issued: u64,
    pub read_bytes: u64,
    pub evictions: u64,
    /// always 0 since F-IO (kept for the JSON): the old path read a pending prefetched slot a second time
    pub demand_reissued: u64,
    /// demand hits on a pending prefetched slot whose queued parts were moved to the demand class / whose parts
    /// were all in flight already (no second read in either case)
    pub demand_promoted: u64,
    pub demand_awaited: u64,
    /// lookahead (`Cache::speculate`): reads issued; demand hits on a speculated expert before / after its read
    /// landed; speculated experts evicted without a demand hit (wasted reads)
    pub spec_issued: u64,
    pub spec_hit_ready: u64,
    pub spec_hit_pending: u64,
    pub spec_evicted_unused: u64,
}

pub struct Cache {
    /// per-layer slot counts (read-only after construction)
    pub quota: Vec<usize>,
    _arena: Arena,
    pools: UnsafeCell<Vec<Pool>>,
    /// slot was filled by a prefetch and not yet used by a demand lookup
    prefetched: UnsafeCell<Vec<Vec<bool>>>,
    /// slot was filled by `speculate` (lookahead) and not yet used by a demand lookup: statistics only
    spec: UnsafeCell<Vec<Vec<bool>>>,
    pub stats: UnsafeCell<Stats>,
    pub sidecar_fd: i32,
    pub layout: Layout,
    /// do not read misses (mode "io none": a miss reuses the slot's old bytes)
    pub no_io: bool,
    /// eviction policy; set before the first lookup
    pub policy: Policy,
}

unsafe impl Sync for Cache {}
unsafe impl Send for Cache {}

/// Where an expert's bytes are (or will be) and the word to wait on.
#[derive(Clone, Copy)]
pub struct Entry {
    pub ptr: *const u8,
    pub state: *const AtomicU8,
}
unsafe impl Send for Entry {}
unsafe impl Sync for Entry {}

impl Entry {
    pub fn ready(&self) -> bool {
        // SAFETY: points into the cache's state vector, which outlives every Entry.
        unsafe { (*self.state).load(Acquire) == READY }
    }
    pub fn failed(&self) -> bool {
        // SAFETY: as `ready`.
        unsafe { (*self.state).load(Acquire) == FAILED }
    }
}

impl Cache {
    /// `quota[l]` = slots of layer l.
    pub fn new(layout: Layout, quota: &[usize], sidecar_fd: i32, no_io: bool, prefault: bool) -> std::io::Result<Cache> {
        assert_eq!(quota.len(), N_LAYER);
        let total: usize = (0..N_LAYER).map(|l| quota[l] * layout.geom[l].slot).sum();
        let arena = Arena::new(total, prefault)?;
        let mut pools = Vec::new();
        let mut off = 0;
        for l in 0..N_LAYER {
            let slot = layout.geom[l].slot;
            let n = quota[l];
            assert!(n >= 16, "layer {l}: quota {n} < 16 slots");
            pools.push(Pool {
                slot,
                n,
                // SAFETY: inside the arena.
                base: unsafe { arena.ptr.add(off) },
                map: vec![-1; N_EXPERT],
                owner: vec![-1; n],
                refb: vec![false; n],
                stamp: vec![u64::MAX; n],
                hand: 0,
                state: (0..n).map(|_| AtomicU8::new(EMPTY)).collect(),
                cnt: vec![0; N_EXPERT],
                last: vec![0; n],
                tick: 0,
            });
            off += n * slot;
        }
        Ok(Cache {
            quota: quota.to_vec(),
            _arena: arena,
            pools: UnsafeCell::new(pools),
            prefetched: UnsafeCell::new(quota.iter().map(|&n| vec![false; n]).collect()),
            spec: UnsafeCell::new(quota.iter().map(|&n| vec![false; n]).collect()),
            stats: UnsafeCell::new(Stats::default()),
            sidecar_fd,
            layout,
            no_io,
            policy: Policy::Clock,
        })
    }

    pub fn arena_bytes(&self) -> usize {
        self._arena.len
    }

    /// Slots per layer for a byte budget: `weights[l]` relative shares (e.g. all 1.0 = equal
    /// bytes, or prefill distinct-expert counts), with a floor of `floor` slots, scaled so the
    /// total fits `budget` bytes.
    pub fn quotas(layout: &Layout, budget: usize, weights: &[f64], floor: usize) -> Vec<usize> {
        let slot = |l: usize| layout.geom[l].slot;
        let floor_bytes: usize = (0..N_LAYER).map(|l| floor * slot(l)).sum();
        assert!(floor_bytes <= budget, "budget below the per-layer floor");
        let wsum: f64 = weights.iter().sum();
        let spare = (budget - floor_bytes) as f64;
        let mut q: Vec<usize> =
            (0..N_LAYER).map(|l| (floor + (spare * weights[l] / wsum / slot(l) as f64).floor() as usize).min(N_EXPERT)).collect();
        // never exceed the budget
        while (0..N_LAYER).map(|l| q[l] * slot(l)).sum::<usize>() > budget {
            let l = (0..N_LAYER).max_by_key(|&l| q[l]).unwrap();
            q[l] -= 1;
        }
        q
    }

    /// # Safety
    /// Control thread only (see module docs).
    #[allow(clippy::mut_from_ref)]
    unsafe fn pools(&self) -> &mut Vec<Pool> {
        &mut *self.pools.get()
    }

    pub fn slots(&self, l: usize) -> usize {
        // SAFETY: read-only field.
        unsafe { (&(*self.pools.get()))[l].n }
    }

    /// Look up (layer l, expert e). On a miss, pick a victim with CLOCK (never a PENDING slot or
    /// one stamped with `stamp`, i.e. needed by the current token), mark it PENDING and queue
    /// the read. Returns the entry and whether it was a miss.
    ///
    /// # Safety
    /// Control thread only.
    pub unsafe fn lookup(&self, io: &IoQueue, l: usize, e: usize, stamp: u64, class: ReqClass) -> (Entry, bool) {
        self.lookup_protect(io, l, e, stamp, stamp, class)
    }

    /// As [`Cache::lookup`], but slots stamped with any stamp in `protect_from..=stamp` are never
    /// evicted (a batched op keeps the wave being computed while it queues the next one).
    ///
    /// # Safety
    /// Control thread only.
    pub unsafe fn lookup_protect(&self, io: &IoQueue, l: usize, e: usize, stamp: u64, protect_from: u64, class: ReqClass) -> (Entry, bool) {
        let p = &mut self.pools()[l];
        let st = &mut *self.stats.get();
        let pf = &mut (&mut *self.prefetched.get())[l];
        let demand = class != ReqClass::Prefetch;
        if demand {
            st.demand_lookups += 1;
        } else {
            st.prefetch_lookups += 1;
        }
        let lfu = self.policy == Policy::Lfu;
        if lfu && demand {
            p.tick += 1;
            p.cnt[e] = p.cnt[e].saturating_add(1);
            if p.tick % LFU_AGE == 0 {
                for c in p.cnt.iter_mut() {
                    *c >>= 1;
                }
            }
        }
        let s = p.map[e];
        if s >= 0 {
            let s = s as usize;
            p.refb[s] = true;
            if demand {
                p.last[s] = p.tick;
            }
            if p.state[s].load(Acquire) == FAILED && !self.no_io {
                // an earlier read of this slot failed: read it again (the waiters of that read already gave up)
                // W4-2: stamp it like any hit, or a later miss of this chunk may evict it once the re-read is READY
                p.stamp[s] = stamp;
                // lookahead: a speculated slot whose read failed is used now, by a demand read of its own: it counts
                // as evicted unused (the speculative read bought nothing) and is an ordinary slot from here on
                if demand {
                    let sp = &mut (&mut *self.spec.get())[l];
                    if sp[s] {
                        sp[s] = false;
                        st.spec_evicted_unused += 1;
                    }
                }
                p.state[s].store(PENDING, Release);
                st.read_bytes += p.slot as u64;
                io.push(Req { class, fd: self.sidecar_fd, buf: p.base.add(s * p.slot) as *mut u8, len: p.slot as u32, off: self.layout.offset(l, e), state: &p.state[s] });
                return (Entry { ptr: p.base.add(s * p.slot), state: &p.state[s] }, true);
            }
            p.stamp[s] = stamp;
            if demand {
                let pending = p.state[s].load(Acquire) != READY;
                if pending {
                    st.demand_hits_pending += 1;
                } else {
                    st.demand_hits_ready += 1;
                }
                let sp = &mut (&mut *self.spec.get())[l];
                if sp[s] {
                    sp[s] = false;
                    if pending {
                        st.spec_hit_pending += 1;
                    } else {
                        st.spec_hit_ready += 1;
                    }
                }
                if pf[s] {
                    st.demand_hits_prefetched += 1;
                    pf[s] = false;
                    // ONE READ PER SLOT: the queued prefetch parts move to the demand class, parts in flight are
                    // awaited. Never a second read into the slot: the first completion would flip READY while the other
                    // read still rewrites the buffer under the kernels (dm-crypt decrypts in place in the O_DIRECT
                    // buffer); that was the old "demand_reissued" path (wrong outputs after Stream::warm, 6 of 6).
                    if pending && !self.no_io {
                        match io.promote(&p.state[s]) {
                            Promote::Moved => st.demand_promoted += 1,
                            Promote::InFlight => st.demand_awaited += 1,
                        }
                    }
                }
            }
            return (Entry { ptr: p.base.add(s * p.slot), state: &p.state[s] }, false);
        }
        let evictable = |p: &Pool, h: usize| {
            !((p.stamp[h] != u64::MAX && p.stamp[h] >= protect_from && p.stamp[h] <= stamp) || p.state[h].load(Acquire) == PENDING)
        };
        // miss: LFU (empty slot first, else fewest counts, then least recently used) or CLOCK
        let lfu_victim = if lfu {
            let mut best: Option<(u32, u64, usize)> = None;
            for h in 0..p.n {
                if !evictable(p, h) {
                    continue;
                }
                if p.owner[h] < 0 {
                    best = Some((0, 0, h));
                    break;
                }
                let key = (p.cnt[p.owner[h] as usize], p.last[h], h);
                if best.is_none_or(|b| (key.0, key.1) < (b.0, b.1)) {
                    best = Some(key);
                }
            }
            Some(best.unwrap_or_else(|| panic!("layer {l}: no evictable slot")).2)
        } else {
            None
        };
        let mut steps = 0;
        let v = if let Some(v) = lfu_victim { v } else { loop {
            let h = p.hand;
            p.hand = (p.hand + 1) % p.n;
            steps += 1;
            assert!(steps <= 3 * p.n + 3, "layer {l}: no evictable slot");
            if !evictable(p, h) {
                continue;
            }
            if p.refb[h] {
                p.refb[h] = false;
                continue;
            }
            break h;
        } };
        p.last[v] = p.tick;
        if p.owner[v] >= 0 {
            p.map[p.owner[v] as usize] = -1;
            st.evictions += 1;
        }
        {
            let sp = &mut (&mut *self.spec.get())[l];
            if sp[v] {
                sp[v] = false;
                st.spec_evicted_unused += 1;
            }
        }
        p.owner[v] = e as i32;
        p.map[e] = v as i32;
        p.refb[v] = true;
        p.stamp[v] = stamp;
        pf[v] = !demand;
        if demand {
            st.demand_misses += 1;
        } else {
            st.prefetch_issued += 1;
        }
        let ptr = p.base.add(v * p.slot);
        if self.no_io && p.state[v].load(Relaxed) != EMPTY {
            // keep the old (valid) bytes, no read
            p.state[v].store(READY, Release);
        } else {
            p.state[v].store(PENDING, Release);
            st.read_bytes += p.slot as u64;
            io.push(Req {
                class,
                fd: self.sidecar_fd,
                buf: ptr as *mut u8,
                len: p.slot as u32,
                off: self.layout.offset(l, e),
                state: &p.state[v],
            });
        }
        (Entry { ptr, state: &p.state[v] }, true)
    }

    /// PREFILL IO: is (layer l, expert e) in the pool (READY or read in flight)? No side effects.
    ///
    /// # Safety
    /// Control thread only.
    pub unsafe fn resident(&self, l: usize, e: usize) -> bool {
        let p = &self.pools()[l];
        let s = p.map[e];
        s >= 0 && p.state[s as usize].load(Acquire) != FAILED
    }

    /// PREFILL IO read-ahead: bring (l, e) in at prefetch priority WITHOUT evicting an expert of `keep` (bitset over
    /// the experts). The victim is an empty slot, else the resident expert outside `keep` with the fewest LFU counts
    /// (then the least recently used); never a slot whose read is in flight. Returns 0 = already there, 1 = read
    /// issued, 2 = no slot may be taken (the caller stops). Counted under the prefetch statistics, but the read is an
    /// ordinary single read of the slot (see the comment at the push): a later demand lookup just waits for it.
    ///
    /// # Safety
    /// Control thread only, and no op is using layer l's slots (called between ops).
    pub unsafe fn prefetch_guarded(&self, io: &IoQueue, l: usize, e: usize, keep: &[u64; N_EXPERT / 64]) -> u8 {
        let p = &mut self.pools()[l];
        let st = &mut *self.stats.get();
        let pf = &mut (&mut *self.prefetched.get())[l];
        st.prefetch_lookups += 1;
        if p.map[e] >= 0 {
            let s = p.map[e] as usize;
            // F76b: a slot whose read failed is not "there": read it again now (one read: FAILED means no read of
            // this slot is in flight), like the demand lookup's FAILED path
            if p.state[s].load(Acquire) == FAILED && !self.no_io {
                p.stamp[s] = 0;
                pf[s] = true;
                p.state[s].store(PENDING, Release);
                st.prefetch_issued += 1;
                st.read_bytes += p.slot as u64;
                io.push(Req { class: ReqClass::Prefetch, fd: self.sidecar_fd, buf: p.base.add(s * p.slot) as *mut u8, len: p.slot as u32, off: self.layout.offset(l, e), state: &p.state[s] });
                return 1;
            }
            return 0;
        }
        let mut best: Option<(u32, u64, usize)> = None;
        for h in 0..p.n {
            if p.state[h].load(Acquire) == PENDING {
                continue;
            }
            if p.owner[h] < 0 {
                best = Some((0, 0, h));
                break;
            }
            let o = p.owner[h] as usize;
            if keep[o / 64] >> (o % 64) & 1 == 1 {
                continue;
            }
            let key = (p.cnt[o], p.last[h], h);
            if best.is_none_or(|b| (key.0, key.1) < (b.0, b.1)) {
                best = Some(key);
            }
        }
        let Some((_, _, v)) = best else { return 2 };
        if p.owner[v] >= 0 {
            p.map[p.owner[v] as usize] = -1;
            st.evictions += 1;
        }
        {
            let sp = &mut (&mut *self.spec.get())[l];
            if sp[v] {
                sp[v] = false;
                st.spec_evicted_unused += 1;
            }
        }
        p.owner[v] = e as i32;
        p.map[e] = v as i32;
        p.refb[v] = true;
        p.stamp[v] = 0; // older than every wave stamp: not protected, ranked last in the hot list until a demand uses it
        p.last[v] = p.tick;
        // W4-C (F-IO decision): speculative reads go in the PREFETCH class and are marked as prefetched, so a demand
        // lookup that finds one pending PROMOTES its queued parts (IoQueue::promote) or waits for the parts in flight.
        // Still one read per slot: since F-IO's root fix (8bd5dad) no path re-reads a pending slot (the old re-issue
        // corrupted QC-5's read-ahead arm through dm-crypt + O_DIRECT; regression tests
        // stream::tests::batched_op_is_order_and_io_independent and warm_then_batched_op_single_read).
        pf[v] = true;
        st.prefetch_issued += 1;
        let ptr = p.base.add(v * p.slot);
        if self.no_io && p.state[v].load(Relaxed) != EMPTY {
            p.state[v].store(READY, Release);
        } else {
            p.state[v].store(PENDING, Release);
            st.read_bytes += p.slot as u64;
            io.push(Req { class: ReqClass::Prefetch, fd: self.sidecar_fd, buf: ptr as *mut u8, len: p.slot as u32, off: self.layout.offset(l, e), state: &p.state[v] });
        }
        1
    }

    /// LOOKAHEAD back-pressure: is (l, e)'s read still in flight? No side effects.
    ///
    /// # Safety
    /// Control thread only.
    pub unsafe fn pending(&self, l: usize, e: usize) -> bool {
        let p = &self.pools()[l];
        let s = p.map[e];
        s >= 0 && p.state[s as usize].load(Acquire) == PENDING
    }

    /// LOOKAHEAD: `prefetch_guarded` for a predicted expert of a layer no op is using (decode: layer l+1 during layer
    /// l's FETCH), marked so that its later demand hit or its unused eviction is counted (`Stats::spec_*`). Same
    /// return codes: 0 = already there, 1 = read issued, 2 = no slot may be taken. The read goes out in the PREFETCH
    /// class (flash-int2 1950385 / F-IO finding 2: queued behind every demand read, counted apart in the IO receipts)
    /// and the slot is marked prefetched; one read per slot: a demand lookup that finds it pending promotes the queued
    /// read to the demand class or waits for the one in flight, never re-reads. The slot is stamped `stamp` = the stamp the layer's NEXT demand call
    /// will use, so that call's own misses cannot evict the speculated expert before its lookup comes (the lookups of
    /// one call run in slot order; an unstamped speculated slot has LFU count 0 and would be the first victim: 6 of 24
    /// in the model-free test). After that call it is an ordinary slot.
    ///
    /// # Safety
    /// As [`Cache::prefetch_guarded`].
    pub unsafe fn speculate(&self, io: &IoQueue, l: usize, e: usize, keep: &[u64; N_EXPERT / 64], stamp: u64) -> u8 {
        let r = self.prefetch_guarded(io, l, e, keep);
        if r == 1 {
            let s = self.pools()[l].map[e] as usize;
            self.pools()[l].stamp[s] = stamp;
            (&mut *self.spec.get())[l][s] = true;
            (*self.stats.get()).spec_issued += 1;
        }
        r
    }

    /// The resident experts (READY slots) as (layer, expert), rank-major: every layer's most recently
    /// used expert first, then every layer's second, ... (a truncated list keeps each layer's hottest).
    ///
    /// # Safety
    /// Control thread only.
    pub unsafe fn hot_list(&self) -> Vec<(u16, u16)> {
        let pools = self.pools();
        let per: Vec<Vec<u16>> = pools
            .iter()
            .map(|p| {
                let mut v: Vec<(u64, u16)> = (0..p.n)
                    .filter(|&s| p.owner[s] >= 0 && p.state[s].load(Acquire) == READY)
                    .map(|s| (if p.stamp[s] == u64::MAX { 0 } else { p.stamp[s] }, p.owner[s] as u16))
                    .collect();
                v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                v.into_iter().map(|x| x.1).collect()
            })
            .collect();
        let mut out = Vec::new();
        for r in 0..per.iter().map(|v| v.len()).max().unwrap_or(0) {
            for (l, v) in per.iter().enumerate() {
                if let Some(&e) = v.get(r) {
                    out.push((l as u16, e));
                }
            }
        }
        out
    }

    /// # Safety
    /// Control thread only; no lookups in flight on other threads.
    pub unsafe fn take_stats(&self) -> Stats {
        std::mem::take(&mut *self.stats.get())
    }
    pub fn peek_stats(&self) -> Stats {
        // SAFETY: copy of plain counters; may be torn only if read concurrently with a lookup.
        unsafe { *self.stats.get() }
    }
}

#[allow(dead_code)]
fn _n(_: usize) -> usize {
    N_LAYER
}

#[cfg(test)]
mod pf_tests {
    use super::*;
    use crate::ffi::W;
    use crate::model::ExpertGeom;

    fn tiny_cache() -> Cache {
        let g = ExpertGeom { gate_w: W::Iq2S, down_w: W::Iq4Nl, gate_bytes: 1024, down_bytes: 1024, record: 3072, slot: 4096 };
        let geom = vec![g; N_LAYER];
        let base: Vec<u64> = (0..N_LAYER).map(|l| (l * 4096 * N_EXPERT) as u64).collect();
        let layout = Layout { geom, base, total: (N_LAYER * 4096 * N_EXPERT) as u64, r4: true };
        Cache::new(layout, &[16; N_LAYER], -1, true, false).unwrap()
    }

    /// F76b: the read-ahead re-reads a FAILED slot (one read, in place), and leaves PENDING / READY slots alone.
    #[test]
    fn prefetch_guarded_rereads_a_failed_slot() {
        let g = ExpertGeom { gate_w: W::Iq2S, down_w: W::Iq4Nl, gate_bytes: 1024, down_bytes: 1024, record: 3072, slot: 4096 };
        let base: Vec<u64> = (0..N_LAYER).map(|l| (l * 4096 * N_EXPERT) as u64).collect();
        let layout = Layout { geom: vec![g; N_LAYER], base, total: (N_LAYER * 4096 * N_EXPERT) as u64, r4: true };
        // IO on, but no IO thread: pushed reads stay queued, slot states are set by hand
        let c = Cache::new(layout, &[16; N_LAYER], -1, false, false).unwrap();
        let io = IoQueue::new();
        let keep = [0u64; N_EXPERT / 64];
        unsafe {
            assert_eq!(c.prefetch_guarded(&io, 2, 77, &keep), 1, "first read");
            let s = c.pools()[2].map[77] as usize;
            assert_eq!(c.pools()[2].state[s].load(Acquire), PENDING);
            assert_eq!(c.prefetch_guarded(&io, 2, 77, &keep), 0, "pending: no second read");
            c.pools()[2].state[s].store(FAILED, Release);
            let issued = (*c.stats.get()).prefetch_issued;
            assert_eq!(c.prefetch_guarded(&io, 2, 77, &keep), 1, "FAILED: read again");
            assert_eq!(c.pools()[2].map[77] as usize, s, "same slot");
            assert_eq!(c.pools()[2].state[s].load(Acquire), PENDING);
            assert_eq!((*c.stats.get()).prefetch_issued, issued + 1);
            c.pools()[2].state[s].store(READY, Release);
            assert_eq!(c.prefetch_guarded(&io, 2, 77, &keep), 0, "READY: there");
            assert_eq!((*c.stats.get()).evictions, 0);
        }
    }

    /// LOOKAHEAD: a speculated expert is counted once: as a hit when a demand lookup finds it (pending or ready), or as
    /// unused when it is evicted first; speculation never counts as a demand access (LFU counts untouched).
    #[test]
    fn speculate_counts_hits_and_unused_evictions() {
        let mut c = tiny_cache();
        c.policy = Policy::Lfu;
        let io = IoQueue::new();
        unsafe {
            let l = 7;
            let keep_of = |es: &[usize]| {
                let mut k = [0u64; N_EXPERT / 64];
                for &e in es {
                    k[e / 64] |= 1 << (e % 64);
                }
                k
            };
            // 14 demand entries, each used twice (LFU count 2), stamp 1
            for _ in 0..2 {
                for e in 100..114 {
                    c.lookup(&io, l, e, 1, ReqClass::Demand);
                }
            }
            // as the IO thread would: every issued read lands (empty slots stay EMPTY, so their first read is PENDING)
            let ready = |c: &Cache| {
                for h in 0..16 {
                    if c.pools()[l].owner[h] >= 0 {
                        c.pools()[l].state[h].store(READY, Release);
                    }
                }
            };
            ready(&c);
            let tick = c.pools()[l].tick;
            // two speculated experts take the two empty slots
            let k = keep_of(&[200, 201]);
            assert_eq!(c.speculate(&io, l, 200, &k, 0), 1);
            assert_eq!(c.speculate(&io, l, 201, &k, 0), 1);
            assert_eq!(c.speculate(&io, l, 201, &k, 0), 0, "already there");
            // speculative reads go out in the PREFETCH class: the slot is marked prefetched (a demand hit promotes it)
            let pf = &(&*c.prefetched.get())[l];
            assert!(pf[c.pools()[l].map[200] as usize] && pf[c.pools()[l].map[201] as usize], "speculation = prefetch class");
            // back-pressure query: both reads are in flight; a resident READY expert and an absent one are not
            assert!(c.pending(l, 200) && c.pending(l, 201) && !c.pending(l, 100) && !c.pending(l, 300));
            assert_eq!(c.pools()[l].tick, tick, "speculation is not a demand access");
            assert_eq!(c.pools()[l].cnt[200], 0);
            let st = *c.stats.get();
            assert_eq!((st.spec_issued, st.evictions, st.demand_misses), (2, 0, 14));
            // 200 is demanded while its read is still pending: a hit, not a miss, no second read
            let (ent, miss) = c.lookup(&io, l, 200, 2, ReqClass::Demand);
            assert!(!miss && !ent.ready());
            assert_eq!((*c.stats.get()).demand_hits_prefetched, 1, "the demand hit found the prefetch-class read");
            ready(&c);
            assert!(!c.pending(l, 200) && !c.pending(l, 201), "landed");
            // a demand miss evicts the coldest slot = the unused speculated 201 (count 0)
            assert!(c.lookup(&io, l, 300, 2, ReqClass::Demand).1);
            assert!(!c.resident(l, 201) && c.resident(l, 200));
            let st = *c.stats.get();
            assert_eq!((st.spec_hit_pending, st.spec_hit_ready, st.spec_evicted_unused), (1, 0, 1));
            ready(&c);
            // a speculated expert hit after its read landed; a later speculation never evicts the kept set
            let k = keep_of(&[400, 401]);
            assert_eq!(c.speculate(&io, l, 400, &k, 3), 1);
            assert_eq!(c.speculate(&io, l, 401, &k, 3), 1);
            assert!(c.resident(l, 400) && c.resident(l, 401), "the second speculation must not evict the first");
            ready(&c);
            // the call stamped 3 misses on 500 first: the speculated 400 / 401 (count 0, the coldest) are protected
            assert!(c.lookup(&io, l, 500, 3, ReqClass::Demand).1);
            assert!(c.resident(l, 400) && c.resident(l, 401), "a speculated expert must survive its own call's misses");
            ready(&c);
            assert!(!c.lookup(&io, l, 400, 3, ReqClass::Demand).1);
            let st = *c.stats.get();
            assert_eq!((st.spec_issued, st.spec_hit_pending, st.spec_hit_ready), (4, 1, 1));
            // every slot's flag is consistent: only 401 is still speculated-unused
            let sp = &(&*c.spec.get())[l];
            assert_eq!(sp.iter().filter(|&&x| x).count(), 1);
            assert!(sp[c.pools()[l].map[401] as usize]);
        }
    }

    /// PREFILL IO: the read-ahead takes empty slots first, never evicts a kept expert, stops when nothing may be taken;
    /// `resident` sees demand and prefetch entries.
    #[test]
    fn prefetch_guarded_respects_keep() {
        let c = tiny_cache();
        let io = IoQueue::new();
        unsafe {
            let l = 5;
            // 10 demand entries (experts 100..110), stamps 1
            for e in 100..110 {
                assert!(c.lookup(&io, l, e, 1, ReqClass::Demand).1);
            }
            assert!(c.resident(l, 105) && !c.resident(l, 3) && !c.resident(l + 1, 105));
            let mut keep = [0u64; N_EXPERT / 64];
            for e in 100..108usize {
                keep[e / 64] |= 1 << (e % 64);
            }
            // 6 empty slots: experts 0..6 go there, nothing evicted
            for e in 0..6 {
                assert_eq!(c.prefetch_guarded(&io, l, e, &keep), 1);
            }
            assert_eq!((*c.stats.get()).evictions, 0);
            assert_eq!(c.prefetch_guarded(&io, l, 3, &keep), 0, "already there");
            // the first-fill reads are in flight (no IO thread in this test): mark everything READY as the IO thread would
            for h in 0..16 {
                c.pools()[l].state[h].store(READY, Release);
            }
            // pool full. As in the engine, the targets are kept too (keep = the set wanted): only 108, 109 and the
            // unkept prefetched 0..6 may go; kept 100..108 stay
            for e in 200..230usize {
                keep[e / 64] |= 1 << (e % 64);
            }
            let mut issued = 0;
            for e in 200..230 {
                match c.prefetch_guarded(&io, l, e, &keep) {
                    1 => issued += 1,
                    2 => break,
                    x => panic!("unexpected {x}"),
                }
            }
            assert_eq!(issued, 8);
            for e in 100..108 {
                assert!(c.resident(l, e), "kept expert {e} evicted");
            }
            assert!(!c.resident(l, 108) && !c.resident(l, 109));
            // everything kept: nothing may be taken
            let all = [u64::MAX; N_EXPERT / 64];
            assert_eq!(c.prefetch_guarded(&io, l, 400, &all), 2);
            assert_eq!((*c.stats.get()).prefetch_issued, 14);
            assert_eq!((*c.stats.get()).evictions, 8);
        }
    }
}
