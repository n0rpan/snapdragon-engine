//! WP-T2: MTP speculative decoding driver for the Flash engine (design: MTP-RUST.md). Pure control logic
//! over two traits, so it is unit-tested without a model; the graph side (all-row outputs + pre-head `h`,
//! per-token recurrent snapshots, the block-48 head graph with a restricted-vocabulary output) implements them.
//!
//! One cycle, greedy:
//!   1. draft: from the last committed token `t` and its pre-head residual `h`, the head proposes d_1..d_k
//!      (k <= depth), stopping early when the head's probability of its own top-1 falls below `floor`;
//!   2. verify: the target evaluates [t, d_1..d_k] in one ubatch and returns k+1 logit rows (+ h rows); every
//!      row is computed exactly as that token's one-token decode would be (fixed-chunk attention, row-wise
//!      expert ops, per-token recurrent steps), so the verify is batch-invariant by construction;
//!   3. accept: d_{i+1} is kept while argmax(row_i) == d_{i+1}; the first mismatching row's argmax (or the
//!      row after the last draft) is the bonus token. Committed = accepted drafts + bonus;
//!   4. commit(m): the target keeps the first m = accepted + 1 evaluated positions (KV by position, recurrent
//!      state from snapshot slot k - m, token/PLE history truncated) and the head catches up on them.
//!
//! The committed stream equals plain greedy decoding of the same path whenever verify rows are bit-identical
//! to one-token decodes (the bug detector); the gate is G-num vs production (MASTER-PLAN 2.1).

/// The target model (the 48-layer trunk) as the driver needs it.
pub trait Target {
    /// Evaluate `toks` at the current position; return one logit row per token (N_VOCAB each) and the pre-head
    /// residual `h` of every row (the head's input). Positions stay provisional until `commit`.
    fn verify(&mut self, toks: &[i32]) -> Result<Rows, String>;
    /// Keep the first `m` (1..=n) positions of the last verify; roll back the rest.
    fn commit(&mut self, m: usize) -> Result<(), String>;
}

/// The MTP head (block 48 + restricted output).
pub trait Draft {
    /// Advance the head over the committed positions of the last verify: `toks[i]` is the token evaluated at
    /// position p + i and `h[i]` the trunk residual of that position. The head pairs the token of position q
    /// with the residual of position q - 1 (llama's MTP driver: `(h_p, x_{p+1})` at head position p + 1), so
    /// the implementation keeps the last residual as the pending one for the next token.
    fn catch_up(&mut self, toks: &[i32], h: &[Vec<f32>]) -> Result<(), String>;
    /// From the last committed token `t` (not yet evaluated by the target; paired with the pending residual)
    /// propose up to `k` tokens, chaining on the head's own residual.
    /// Returns (token, head probability of that token) per step; may return fewer (the driver applies `floor`).
    fn propose(&mut self, t: i32, k: usize, floor: f32) -> Result<Vec<(i32, f32)>, String>;
    /// Drop the head's provisional positions beyond the last `catch_up`.
    fn rewind(&mut self) -> Result<(), String>;
}

pub struct Rows {
    pub logits: Vec<Vec<f32>>,
    pub h: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct MtpConfig {
    /// max drafts per cycle (verify width = depth + 1)
    pub max_depth: usize,
    /// stop drafting when the head's top-1 probability is below this
    pub floor: f32,
    /// adaptive depth: EWMA weight of the per-position survival estimate
    pub ewma: f32,
    /// relative cost of a verify of width w vs a one-token decode, w = 1..=max_depth+1 (from timing; the expert
    /// union grows with w). The depth that maximises expected tokens per unit cost is used.
    pub verify_cost: Vec<f32>,
    /// relative cost of one head step vs a one-token decode
    pub draft_cost: f32,
}

impl Default for MtpConfig {
    fn default() -> Self {
        // [E] costs: verify w tokens ~ 1 + 0.25 (w - 1) of a decode (shared dense pass, union of experts),
        // head step ~ 0.06 (one block + 32K-row output). Replace with Codex's timings.
        MtpConfig { max_depth: 3, floor: 0.3, ewma: 0.05, verify_cost: vec![1.0, 1.25, 1.5, 1.75], draft_cost: 0.06 }
    }
}

#[derive(Default, Clone, Debug)]
pub struct MtpStats {
    pub cycles: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub committed: u64,
    /// accepted[i]: cycles where draft position i+1 survived; proposed[i]: cycles that proposed position i+1
    pub pos_proposed: Vec<u64>,
    pub pos_accepted: Vec<u64>,
    pub floor_stops: u64,
}

impl MtpStats {
    pub fn tokens_per_verify(&self) -> f64 {
        self.committed as f64 / self.cycles.max(1) as f64
    }
}

pub struct MtpDriver {
    pub cfg: MtpConfig,
    pub stats: MtpStats,
    /// survival estimate per draft position (P(position i survives | proposed))
    surv: Vec<f32>,
}

pub fn argmax(l: &[f32]) -> i32 {
    let mut b = 0;
    for (i, &x) in l.iter().enumerate() {
        if x > l[b] {
            b = i;
        }
    }
    b as i32
}

/// Greedy acceptance: (number of drafts accepted, the bonus token).
pub fn accept(drafts: &[i32], rows: &[Vec<f32>]) -> (usize, i32) {
    assert_eq!(rows.len(), drafts.len() + 1);
    for (i, &d) in drafts.iter().enumerate() {
        let a = argmax(&rows[i]);
        if a != d {
            return (i, a);
        }
    }
    (drafts.len(), argmax(&rows[drafts.len()]))
}

impl MtpDriver {
    pub fn new(cfg: MtpConfig) -> MtpDriver {
        let n = cfg.max_depth;
        MtpDriver { surv: vec![0.8; n], stats: MtpStats { pos_proposed: vec![0; n], pos_accepted: vec![0; n], ..Default::default() }, cfg }
    }

    /// Depth that maximises expected committed tokens per unit cost under the survival estimates.
    pub fn depth(&self) -> usize {
        let (mut best, mut best_rate) = (0usize, 1.0 / self.cfg.verify_cost[0]);
        let mut p = 1.0f32; // P(first i drafts all survive)
        let mut exp = 1.0f32; // expected committed = 1 + sum_i P(i survive)
        for k in 1..=self.cfg.max_depth {
            p *= self.surv[k - 1];
            exp += p;
            let rate = exp / (self.cfg.verify_cost[k] + self.cfg.draft_cost * k as f32);
            if rate > best_rate {
                best = k;
                best_rate = rate;
            }
        }
        best
    }

    /// One cycle from the last committed token `t` (already caught up in the head). Returns the newly committed
    /// tokens (accepted drafts then the bonus token); the last one is the next cycle's `t`.
    /// `e` is both the target and the head (they share one graph context in the engine).
    pub fn cycle<E: Target + Draft>(&mut self, e: &mut E, t: i32) -> Result<Vec<i32>, String> {
        let k = self.depth();
        let props = if k > 0 { e.propose(t, k, self.cfg.floor)? } else { Vec::new() };
        let mut drafts: Vec<i32> = Vec::new();
        for &(tok, p) in &props {
            if p < self.cfg.floor {
                self.stats.floor_stops += 1;
                break;
            }
            drafts.push(tok);
        }
        e.rewind()?;
        let mut toks = vec![t];
        toks.extend_from_slice(&drafts);
        let rows = e.verify(&toks)?;
        if rows.logits.len() != toks.len() || rows.h.len() != toks.len() {
            return Err(format!("verify returned {} rows for {} tokens", rows.logits.len(), toks.len()));
        }
        let (n_acc, bonus) = accept(&drafts, &rows.logits);
        e.commit(n_acc + 1)?;
        // the head catches up on the committed positions: inputs t, d_1..d_nacc with their trunk residuals
        e.catch_up(&toks[..n_acc + 1], &rows.h[..n_acc + 1])?;
        // stats + survival estimates
        let s = &mut self.stats;
        s.cycles += 1;
        s.drafted += drafts.len() as u64;
        s.accepted += n_acc as u64;
        s.committed += n_acc as u64 + 1;
        for i in 0..drafts.len() {
            s.pos_proposed[i] += 1;
            let ok = i < n_acc;
            s.pos_accepted[i] += ok as u64;
            self.surv[i] += self.cfg.ewma * ((ok as u8 as f32) - self.surv[i]);
        }
        let mut out = drafts[..n_acc].to_vec();
        out.push(bonus);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: usize = 16;

    fn onehot(t: i32) -> Vec<f32> {
        let mut v = vec![0.0; V];
        v[t as usize] = 1.0;
        v
    }

    /// Target whose greedy continuation is a fixed sequence; checks that commits never skip or reorder.
    struct SeqTarget {
        seq: Vec<i32>,
        pos: usize,
        pending: usize,
        verifies: usize,
    }
    impl Target for SeqTarget {
        fn verify(&mut self, toks: &[i32]) -> Result<Rows, String> {
            self.verifies += 1;
            // row i predicts the token after position pos + i, given toks[i] is the true token there
            let mut logits = Vec::new();
            for i in 0..toks.len() {
                logits.push(onehot(self.seq[self.pos + i + 1]));
            }
            self.pending = toks.len();
            Ok(Rows { h: vec![vec![0.0]; toks.len()], logits })
        }
        fn commit(&mut self, m: usize) -> Result<(), String> {
            assert!(m >= 1 && m <= self.pending);
            self.pos += m;
            Ok(())
        }
    }

    /// Head that is right with a pattern: position i of each proposal is correct iff good[i].
    struct PatHead {
        seq: Vec<i32>,
        pos: usize,
        good: Vec<bool>,
        prob: f32,
    }
    impl Draft for PatHead {
        fn catch_up(&mut self, toks: &[i32], _h: &[Vec<f32>]) -> Result<(), String> {
            self.pos += toks.len();
            Ok(())
        }
        fn propose(&mut self, _t: i32, k: usize, _floor: f32) -> Result<Vec<(i32, f32)>, String> {
            Ok((0..k).map(|i| {
                let want = self.seq[self.pos + i + 1];
                (if self.good.get(i).copied().unwrap_or(false) { want } else { (want + 1) % V as i32 }, self.prob)
            })
            .collect())
        }
        fn rewind(&mut self) -> Result<(), String> {
            Ok(())
        }
    }

    struct Both {
        tgt: SeqTarget,
        head: PatHead,
    }
    impl Target for Both {
        fn verify(&mut self, toks: &[i32]) -> Result<Rows, String> {
            self.tgt.verify(toks)
        }
        fn commit(&mut self, m: usize) -> Result<(), String> {
            self.tgt.commit(m)
        }
    }
    impl Draft for Both {
        fn catch_up(&mut self, toks: &[i32], h: &[Vec<f32>]) -> Result<(), String> {
            self.head.catch_up(toks, h)
        }
        fn propose(&mut self, t: i32, k: usize, floor: f32) -> Result<Vec<(i32, f32)>, String> {
            self.head.propose(t, k, floor)
        }
        fn rewind(&mut self) -> Result<(), String> {
            self.head.rewind()
        }
    }

    fn run(good: Vec<bool>, prob: f32, cfg: MtpConfig) -> (Vec<i32>, Vec<i32>, MtpStats, usize) {
        let seq: Vec<i32> = (0..400).map(|i| ((i * 7 + 3) % V) as i32).collect();
        let mut e = Both { tgt: SeqTarget { seq: seq.clone(), pos: 0, pending: 0, verifies: 0 }, head: PatHead { seq: seq.clone(), pos: 0, good, prob } };
        let mut d = MtpDriver::new(cfg);
        let mut got = vec![seq[0]];
        while got.len() < 300 {
            let t = *got.last().unwrap();
            let c = d.cycle(&mut e, t).unwrap();
            got.extend(c);
        }
        got.truncate(300);
        (got, seq[..300].to_vec(), d.stats.clone(), e.tgt.verifies)
    }

    #[test]
    fn output_equals_greedy_whatever_the_head_does() {
        for good in [vec![true, true, true], vec![true, false, true], vec![false, false, false], vec![true, true, false]] {
            let (got, want, st, _) = run(good.clone(), 0.9, MtpConfig::default());
            assert_eq!(got, want, "{good:?}");
            assert_eq!(st.committed, st.accepted + st.cycles);
        }
    }

    #[test]
    fn perfect_head_reaches_full_depth() {
        let (_, _, st, v) = run(vec![true; 3], 0.9, MtpConfig::default());
        assert!(st.tokens_per_verify() > 3.5, "{}", st.tokens_per_verify());
        assert!(v < 300 / 3);
    }

    #[test]
    fn useless_head_backs_off_to_plain_decoding() {
        let (_, _, st, _) = run(vec![false; 3], 0.9, MtpConfig::default());
        // survival estimates decay, depth goes to 0: most cycles are plain one-token decodes
        assert!(st.drafted < 120, "drafted {}", st.drafted);
    }

    #[test]
    fn floor_stops_drafting() {
        let (got, want, st, _) = run(vec![true; 3], 0.1, MtpConfig::default());
        assert_eq!(got, want);
        assert_eq!(st.drafted, 0);
        assert!(st.floor_stops > 0);
    }

    // ---- a stateful mini model (FLASH-MTP): every row depends on ALL tokens fed so far, so a draft that is not
    // rolled back exactly, or a commit of the wrong width, changes every later token ----
    fn mix(s: u64, t: i32) -> u64 {
        let mut x = (s ^ (t as u64).wrapping_mul(0x9E3779B97F4A7C15)).wrapping_add(0xD1B54A32D192ED03);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^ (x >> 32)
    }
    fn mini_row(s: u64) -> Vec<f32> {
        // dense logits (not one-hot) with a unique maximum, derived from the state bits
        (0..V).map(|i| ((mix(s, i as i32) >> 40) as f32) + if i as u64 == (s >> 9) % V as u64 { 3.0e7 } else { 0.0 }).collect()
    }
    fn h_of(s: u64) -> Vec<f32> {
        vec![f32::from_bits(s as u32), f32::from_bits((s >> 32) as u32)]
    }
    fn s_of(h: &[f32]) -> u64 {
        h[0].to_bits() as u64 | (h[1].to_bits() as u64) << 32
    }
    /// The target as the Flash engine implements it: verify leaves the state untouched and keeps one snapshot
    /// per row; commit(m) applies snapshot m - 1. `decode` is the plain one-token path.
    struct MiniTarget {
        state: u64,
        snaps: Vec<u64>,
        rollbacks: usize,
    }
    impl MiniTarget {
        fn decode(&mut self, t: i32) -> Vec<f32> {
            assert!(self.snaps.is_empty(), "decode with an uncommitted verify");
            self.state = mix(self.state, t);
            mini_row(self.state)
        }
    }
    impl Target for MiniTarget {
        fn verify(&mut self, toks: &[i32]) -> Result<Rows, String> {
            assert!(self.snaps.is_empty(), "verify with an uncommitted verify");
            let mut s = self.state;
            let (mut logits, mut h) = (Vec::new(), Vec::new());
            for &t in toks {
                s = mix(s, t);
                self.snaps.push(s);
                logits.push(mini_row(s));
                h.push(h_of(s));
            }
            Ok(Rows { logits, h })
        }
        fn commit(&mut self, m: usize) -> Result<(), String> {
            if m < 1 || m > self.snaps.len() {
                return Err(format!("commit {m} of {}", self.snaps.len()));
            }
            self.rollbacks += (m < self.snaps.len()) as usize;
            self.state = self.snaps[m - 1];
            self.snaps.clear();
            Ok(())
        }
    }
    /// A head that predicts from the pending residual (the committed state) like the real one, and is wrong
    /// on a fixed schedule: proposal number n is wrong iff n % wrong_every == wrong_every - 1.
    struct MiniHead {
        pending: u64,
        n: usize,
        wrong_every: usize,
    }
    impl Draft for MiniHead {
        fn catch_up(&mut self, toks: &[i32], h: &[Vec<f32>]) -> Result<(), String> {
            assert_eq!(toks.len(), h.len());
            self.pending = s_of(&h[h.len() - 1]);
            Ok(())
        }
        fn propose(&mut self, t: i32, k: usize, _floor: f32) -> Result<Vec<(i32, f32)>, String> {
            let mut s = mix(self.pending, t);
            let mut out = Vec::new();
            for _ in 0..k {
                let right = argmax(&mini_row(s));
                let d = if self.n % self.wrong_every == self.wrong_every - 1 { (right + 1) % V as i32 } else { right };
                self.n += 1;
                out.push((d, 0.9));
                s = mix(s, d);
            }
            Ok(out)
        }
        fn rewind(&mut self) -> Result<(), String> {
            Ok(())
        }
    }
    struct Mini {
        tgt: MiniTarget,
        head: MiniHead,
    }
    impl Target for Mini {
        fn verify(&mut self, toks: &[i32]) -> Result<Rows, String> {
            self.tgt.verify(toks)
        }
        fn commit(&mut self, m: usize) -> Result<(), String> {
            self.tgt.commit(m)
        }
    }
    impl Draft for Mini {
        fn catch_up(&mut self, toks: &[i32], h: &[Vec<f32>]) -> Result<(), String> {
            self.head.catch_up(toks, h)
        }
        fn propose(&mut self, t: i32, k: usize, floor: f32) -> Result<Vec<(i32, f32)>, String> {
            self.head.propose(t, k, floor)
        }
        fn rewind(&mut self) -> Result<(), String> {
            self.head.rewind()
        }
    }

    /// Depth 1 (and 2, 3) on the stateful mini model: the token stream AND the final state are identical to
    /// plain one-token decoding, with real rollbacks on the way.
    #[test]
    fn mtp_equals_plain_decode_on_a_stateful_mini_model() {
        const N: usize = 500;
        for seed in [1u64, 0xFEED, 0x1234_5678_9ABC] {
            let t0 = (seed % V as u64) as i32;
            // plain
            let mut p = MiniTarget { state: seed, snaps: Vec::new(), rollbacks: 0 };
            let mut plain = vec![t0];
            let mut states = vec![seed];
            while plain.len() < N {
                let row = p.decode(*plain.last().unwrap());
                states.push(p.state);
                plain.push(argmax(&row));
            }
            for depth in 1..=3usize {
                for wrong_every in [2usize, 3, 7, 1_000_000] {
                    let mut e = Mini { tgt: MiniTarget { state: seed, snaps: Vec::new(), rollbacks: 0 }, head: MiniHead { pending: seed, n: 0, wrong_every } };
                    let cfg = MtpConfig { max_depth: depth, verify_cost: (0..=depth).map(|i| 1.0 + 0.25 * i as f32).collect(), ..MtpConfig::default() };
                    let mut d = MtpDriver::new(cfg);
                    let mut got = vec![t0];
                    while got.len() < N {
                        let t = *got.last().unwrap();
                        let c = d.cycle(&mut e, t).unwrap();
                        assert!(!c.is_empty());
                        got.extend(c);
                        // the state after every cycle is the plain state at that position (evaluated = got.len() - 1)
                        if got.len() <= N {
                            assert_eq!(e.tgt.state, states[got.len() - 1], "state after {} tokens (depth {depth})", got.len() - 1);
                        }
                    }
                    got.truncate(N);
                    assert_eq!(got, plain, "seed {seed:x} depth {depth} wrong_every {wrong_every}");
                    let st = &d.stats;
                    assert_eq!(st.committed, st.accepted + st.cycles);
                    if wrong_every < 100 {
                        assert!(e.tgt.rollbacks > 0, "no rollback exercised (depth {depth}, wrong_every {wrong_every})");
                        assert!(st.accepted > 0, "no draft accepted (depth {depth}, wrong_every {wrong_every})");
                    } else {
                        assert_eq!(e.tgt.rollbacks, 0);
                        assert!(st.tokens_per_verify() > 1.0 + 0.9 * depth as f64, "{}", st.tokens_per_verify());
                    }
                }
            }
        }
    }

    #[test]
    fn accept_rule() {
        let rows = vec![onehot(3), onehot(5), onehot(7)];
        assert_eq!(accept(&[3, 5], &rows), (2, 7));
        assert_eq!(accept(&[3, 6], &rows), (1, 5));
        assert_eq!(accept(&[4, 5], &rows), (0, 3));
        assert_eq!(accept(&[], &rows[..1]), (0, 3));
    }
}
