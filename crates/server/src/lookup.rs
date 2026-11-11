//! WP-D2 phase c: prompt-lookup (suffix-match) drafting for exact greedy speculation.
//!
//! The drafter proposes the tokens that followed the most recent earlier occurrence of the
//! sequence's current suffix (longest match of `kmax` down to `kmin` tokens) in the request's own
//! tokens (prompt + generated so far). The engine verifies `[t, d1..dn]` in one batch-invariant
//! run and keeps the longest matching prefix, so the output is identical to plain greedy decoding
//! whatever the drafts are; only speed changes. Design and the offline simulation:
//! tools/lookup_sim.py (M2 recording: session wall-time gain 9.2-11.3% [E]).
//!
//! Policy (simulator defaults that won): draft only on a match; after a fully rejected draft skip
//! drafting for `backoff` steps, doubling on consecutive misses (cap 16), reset on an accept.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LookupConfig {
    pub kmin: usize,
    pub kmax: usize,
    /// draft tokens per verify (lanes - 1)
    pub n_max: usize,
    /// initial skip after a fully rejected draft (0 = never back off)
    pub backoff: usize,
}

impl Default for LookupConfig {
    fn default() -> Self {
        LookupConfig { kmin: 4, kmax: 8, n_max: 4, backoff: 0 }
    }
}

/// Incremental k-gram index over one request's token sequence.
pub struct LookupDrafter {
    cfg: LookupConfig,
    toks: Vec<u32>,
    /// per k (kmin..=kmax): hash of the k tokens ending at i -> latest start positions (most recent last)
    index: Vec<HashMap<u64, Vec<u32>>>,
    skip: usize,
    streak: usize,
    pub drafted: u64,
    pub accepted: u64,
}

fn hash(t: &[u32]) -> u64 {
    // FNV-1a over the token ids (collisions are checked against the tokens on lookup)
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in t {
        h ^= x as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl LookupDrafter {
    pub fn new(cfg: LookupConfig) -> LookupDrafter {
        let n = cfg.kmax.saturating_sub(cfg.kmin) + 1;
        LookupDrafter { cfg, toks: Vec::new(), index: (0..n).map(|_| HashMap::new()).collect(), skip: 0, streak: 0, drafted: 0, accepted: 0 }
    }

    /// Append committed tokens (the prompt, then every token as it is appended to the state).
    pub fn push(&mut self, ts: &[u32]) {
        for &t in ts {
            self.toks.push(t);
            let n = self.toks.len();
            for (i, k) in (self.cfg.kmin..=self.cfg.kmax).enumerate() {
                if n >= k {
                    let s = n - k;
                    self.index[i].entry(hash(&self.toks[s..n])).or_default().push(s as u32);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.toks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    /// Drafts to follow the current sequence (empty = no match or backing off).
    pub fn draft(&mut self) -> Vec<u32> {
        if self.skip > 0 {
            self.skip -= 1;
            return Vec::new();
        }
        let n = self.toks.len();
        for k in (self.cfg.kmin..=self.cfg.kmax).rev() {
            let i = k - self.cfg.kmin;
            if n < k + 1 {
                continue;
            }
            let suf = &self.toks[n - k..];
            let Some(starts) = self.index[i].get(&hash(suf)) else { continue };
            // most recent earlier occurrence (not the suffix itself) with a following token
            for &s in starts.iter().rev() {
                let s = s as usize;
                if s + k >= n || &self.toks[s..s + k] != suf {
                    continue;
                }
                let st = s + k;
                let end = (st + self.cfg.n_max).min(n);
                return self.toks[st..end].to_vec();
            }
        }
        Vec::new()
    }

    /// Outcome of a verify of `drafted` drafts with `accepted` matching (for the back-off).
    pub fn feedback(&mut self, drafted: usize, accepted: usize) {
        self.drafted += drafted as u64;
        self.accepted += accepted as u64;
        if drafted == 0 || self.cfg.backoff == 0 {
            return;
        }
        if accepted == 0 {
            self.streak += 1;
            self.skip = (self.cfg.backoff << (self.streak - 1).min(4)).min(16);
        } else {
            self.streak = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_the_continuation_of_the_latest_match() {
        let mut d = LookupDrafter::new(LookupConfig { kmin: 2, kmax: 4, n_max: 3, backoff: 2 });
        d.push(&[1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 8, 1, 2]);
        // suffix [1,2]: latest earlier occurrence at 6 -> continuation 3,7,8
        assert_eq!(d.draft(), vec![3, 7, 8]);
        d.push(&[3]);
        // suffix [1,2,3] (k=3) at 6 -> 7,8,1 ; longest k wins over k=2
        assert_eq!(d.draft(), vec![7, 8, 1]);
        // no match
        let mut e = LookupDrafter::new(LookupConfig::default());
        e.push(&(0..50).collect::<Vec<_>>());
        assert!(e.draft().is_empty());
    }

    #[test]
    fn continuation_stops_at_the_sequence_end_and_backoff_works() {
        let mut d = LookupDrafter::new(LookupConfig { kmin: 2, kmax: 2, n_max: 4, backoff: 2 });
        d.push(&[5, 6, 7, 5, 6]);
        assert_eq!(d.draft(), vec![7, 5, 6]);
        d.feedback(3, 0);
        assert!(d.draft().is_empty() && d.draft().is_empty());
        assert_eq!(d.draft(), vec![7, 5, 6]);
        d.feedback(3, 0);
        for _ in 0..4 {
            assert!(d.draft().is_empty());
        }
        assert!(!d.draft().is_empty());
        d.feedback(3, 2);
        assert!(!d.draft().is_empty());
    }
}
