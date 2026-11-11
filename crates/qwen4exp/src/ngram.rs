//! N-gram (per-layer-embedding) rows read with O_DIRECT page reads plus a small page cache.
//!
//! Geometry from the P4 header audit: `per_layer_token_embd.weight` is IQ4_NL [160, 320001536],
//! 90 B per stored row; 16 heads (n-gram orders 2 and 3 x 8 heads) of ~20 M rows each
//! (`ple.head_offsets` / `ple.head_vocab_sizes`). One decode token needs 16 rows = 16 random
//! 4 KiB pages (a row straddles two pages 90/4096 of the time).
//!
//! M0 approximation: the row index hash is NOT the model's (not ported yet): it is a
//! multiplicative hash of the last 2/3 token ids with the `ple.layer_multipliers`, reduced
//! modulo each head's vocab size. It has the right address distribution (uniform over each
//! head's 1.8 GB region), so the page traffic and cache behaviour are representative for a
//! given token stream.

use crate::cache::{EMPTY, PENDING, READY};
use crate::ffi::dequant_iq4_nl;
use crate::io::{IoQueue, Req, ReqClass};
use crate::model::{PLE_DIM, PLE_HEADS, PLE_ROW_BYTES};
use crate::uring::Arena;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering::*};

const PAGE: u64 = 4096;
const MULT: [u64; 3] = [23703573157769, 20109073645365, 8052911324071];

pub struct NgramTable {
    pub fd: i32,
    /// absolute file offset of row 0
    pub base: u64,
    pub head_off: Vec<u64>,
    pub head_vocab: Vec<u64>,
    arena: Arena,
    cap: usize,
    map: HashMap<u64, u32>,
    owner: Vec<u64>,
    refb: Vec<bool>,
    hand: usize,
    state: Vec<AtomicU8>,
    /// per head: (first page slot, second page slot or u32::MAX, byte offset in first page)
    rows: Vec<(u32, u32, u32)>,
    pub lookups: u64,
    pub hits: u64,
    pub reads: u64,
}

unsafe impl Send for NgramTable {}
unsafe impl Sync for NgramTable {}

impl NgramTable {
    pub fn new(fd: i32, base: u64, head_off: Vec<u64>, head_vocab: Vec<u64>, cache_bytes: usize) -> std::io::Result<NgramTable> {
        assert_eq!(head_off.len(), PLE_HEADS);
        assert_eq!(head_vocab.len(), PLE_HEADS);
        let cap = (cache_bytes / PAGE as usize).max(64);
        Ok(NgramTable {
            fd,
            base,
            head_off,
            head_vocab,
            arena: Arena::new(cap * PAGE as usize, true)?,
            cap,
            map: HashMap::with_capacity(cap * 2),
            owner: vec![u64::MAX; cap],
            refb: vec![false; cap],
            hand: 0,
            state: (0..cap).map(|_| AtomicU8::new(EMPTY)).collect(),
            rows: vec![(0, u32::MAX, 0); PLE_HEADS],
            lookups: 0,
            hits: 0,
            reads: 0,
        })
    }

    /// Row index of head h for the n-gram ending at `toks.last()` (order 2 for heads 0-7, 3 for 8-15).
    pub fn row_index(&self, h: usize, toks: &[u32]) -> u64 {
        let order = if h < PLE_HEADS / 2 { 2 } else { 3 };
        let mut x = 0u64;
        for i in 0..order {
            let t = if toks.len() > i { toks[toks.len() - 1 - i] as u64 } else { 0 };
            x = x.wrapping_add(t.wrapping_mul(MULT[i]));
        }
        x ^= (h as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 29;
        self.head_off[h] + x % self.head_vocab[h]
    }

    fn page_slot(&mut self, io: &IoQueue, page: u64, protect: &[u32]) -> u32 {
        self.lookups += 1;
        if let Some(&s) = self.map.get(&page) {
            self.hits += 1;
            self.refb[s as usize] = true;
            return s;
        }
        let v = loop {
            let h = self.hand;
            self.hand = (self.hand + 1) % self.cap;
            if self.state[h].load(Acquire) == PENDING || protect.contains(&(h as u32)) {
                continue;
            }
            if self.refb[h] {
                self.refb[h] = false;
                continue;
            }
            break h;
        };
        if self.owner[v] != u64::MAX {
            self.map.remove(&self.owner[v]);
        }
        self.owner[v] = page;
        self.map.insert(page, v as u32);
        self.refb[v] = true;
        self.state[v].store(PENDING, Release);
        self.reads += 1;
        io.push(Req {
            class: ReqClass::Ngram,
            fd: self.fd,
            // SAFETY: inside the page arena.
            buf: unsafe { self.arena.ptr.add(v * PAGE as usize) },
            len: PAGE as u32,
            off: page * PAGE,
            state: &self.state[v],
        });
        v as u32
    }

    /// Issue the page reads for one token (control thread, at token start).
    pub fn issue(&mut self, io: &IoQueue, toks: &[u32]) {
        let mut used: Vec<u32> = Vec::with_capacity(2 * PLE_HEADS);
        for h in 0..PLE_HEADS {
            let off = self.base + self.row_index(h, toks) * PLE_ROW_BYTES as u64;
            let p0 = off / PAGE;
            let p1 = (off + PLE_ROW_BYTES as u64 - 1) / PAGE;
            let s0 = self.page_slot(io, p0, &used);
            used.push(s0);
            let s1 = if p1 != p0 {
                let s = self.page_slot(io, p1, &used);
                used.push(s);
                s
            } else {
                u32::MAX
            };
            self.rows[h] = (s0, s1, (off % PAGE) as u32);
        }
    }

    /// Wait for the token's pages and dequantise its 16 rows into `out` (16 x 160 = 2560).
    /// Returns the ns spent waiting. Safe to call from any thread after `issue` (read-only).
    pub fn gather(&self, out: &mut [f32]) -> u64 {
        let t0 = std::time::Instant::now();
        let mut waited = 0;
        for &(s0, s1, _) in &self.rows {
            for s in [s0, s1] {
                if s != u32::MAX {
                    while self.state[s as usize].load(Acquire) != READY {
                        std::hint::spin_loop();
                    }
                }
            }
        }
        waited += t0.elapsed().as_nanos() as u64;
        let mut row = [0u8; PLE_ROW_BYTES];
        for (h, &(s0, s1, o)) in self.rows.iter().enumerate() {
            let o = o as usize;
            // SAFETY: page slots inside the arena, READY (read complete).
            unsafe {
                let p0 = std::slice::from_raw_parts(self.arena.ptr.add(s0 as usize * PAGE as usize), PAGE as usize);
                if s1 == u32::MAX {
                    row.copy_from_slice(&p0[o..o + PLE_ROW_BYTES]);
                } else {
                    let a = PAGE as usize - o;
                    row[..a].copy_from_slice(&p0[o..]);
                    let p1 = std::slice::from_raw_parts(self.arena.ptr.add(s1 as usize * PAGE as usize), PAGE as usize);
                    row[a..].copy_from_slice(&p1[..PLE_ROW_BYTES - a]);
                }
            }
            dequant_iq4_nl(&row, &mut out[h * PLE_DIM..(h + 1) * PLE_DIM]);
        }
        waited
    }
}
