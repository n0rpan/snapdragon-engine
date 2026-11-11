//! Expert lookahead (F-LOOKAHEAD, FLASH-LOOKAHEAD-DESIGN.md section 6): at layer L's FETCH of a one-token decode,
//! score the experts of layer L+1 from the experts routed at L, L-1, L-2 of the SAME token and speculate the best
//! non-resident ones into layer L+1's pool (`Cache::speculate`). Lossless: only WHEN bytes are read changes.
//!
//! Integer model (tools/la_dense.py writes the table and the reference vectors):
//!   score[e] = bias[T][e] + sum over d in 1..=3 with T-d >= 0, over the 10 experts s routed at layer T-d, of
//!              tab[T][d-1][s][e]                                  (bias i16, tab i8, score i32)
//!   candidates = experts with score >= thr[T]; the best `max` non-resident ones are speculated
//!   (score descending, then expert id ascending).
//!
//! File LAT2, little-endian: b"LAT2", u32 version = 2, u32 n_layer = 48, u32 n_expert = 512, u32 n_dist = 3,
//! i32 scale, i32 thr[48], i16 bias[48][512], i8 tab[48][3][512][512] ([target layer][d-1][source][target]).

use crate::model::{N_EXPERT, N_LAYER, TOP_K};

pub const N_DIST: usize = 3;
const HDR: usize = 24;
pub const LAT_BYTES: usize = HDR + 4 * N_LAYER + 2 * N_LAYER * N_EXPERT + N_LAYER * N_DIST * N_EXPERT * N_EXPERT;
/// most candidates one layer may speculate (`--la-max` is clamped to it)
pub const MAX_SPEC: usize = 8;

#[derive(Clone, Copy, Debug)]
pub struct LaStats {
    /// layers scored (one per decode token and layer 0..46)
    pub scored: u64,
    /// experts at or above the layer threshold, resident or not
    pub above_thr: u64,
    /// `Cache::speculate` calls that issued a read
    pub issued: u64,
    /// `Cache::speculate` found no slot it may take (the layer's speculation stopped there)
    pub no_slot: u64,
    /// layers whose speculation was skipped because more than `max_pending` speculative reads were still in flight,
    /// and the candidates (non-resident, above the threshold, within `max`) not issued for that reason
    pub skipped_pressure_layers: u64,
    pub skipped_pressure_reads: u64,
    /// most speculative reads seen in flight at a hook
    pub max_in_flight: u64,
    pub score_ns: u64,
    /// reads issued per TARGET layer
    pub issued_per_layer: [u64; N_LAYER],
}

impl Default for LaStats {
    fn default() -> LaStats {
        LaStats { scored: 0, above_thr: 0, issued: 0, no_slot: 0, skipped_pressure_layers: 0, skipped_pressure_reads: 0, max_in_flight: 0, score_ns: 0, issued_per_layer: [0; N_LAYER] }
    }
}

pub struct Lookahead {
    pub scale: i32,
    thr: Vec<i32>,
    bias: Vec<i16>,
    tab: Vec<i8>,
    /// speculative reads per layer at most
    pub max: usize,
    /// back-pressure (F-IO review): a layer speculates nothing while more than this many speculative reads are still
    /// in flight (the device saturates at about 3 records per layer; `--la-pending`, default 2)
    pub max_pending: usize,
    /// speculative reads issued and not yet seen landed: (layer, expert); pruned at every hook
    pub in_flight: Vec<(u16, u16)>,
    /// routed experts of the current token per layer; `have` bit l = layer l seen in this token
    ids: Vec<[u16; TOP_K]>,
    have: u64,
    acc: Vec<i16>,
    pub stats: LaStats,
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

impl Lookahead {
    pub fn from_bytes(b: &[u8], max: usize) -> Result<Lookahead, String> {
        if b.len() < HDR || &b[..4] != b"LAT2" {
            return Err("lookahead table: not a LAT2 file".into());
        }
        let (ver, nl, ne, nd) = (rd_u32(b, 4), rd_u32(b, 8), rd_u32(b, 12), rd_u32(b, 16));
        if ver != 2 || nl as usize != N_LAYER || ne as usize != N_EXPERT || nd as usize != N_DIST {
            return Err(format!("lookahead table: version {ver}, {nl} layers, {ne} experts, {nd} distances (want 2, {N_LAYER}, {N_EXPERT}, {N_DIST})"));
        }
        if b.len() != LAT_BYTES {
            return Err(format!("lookahead table: {} bytes, want {LAT_BYTES}", b.len()));
        }
        let scale = rd_u32(b, 20) as i32;
        if scale <= 0 {
            return Err(format!("lookahead table: scale {scale}"));
        }
        let mut o = HDR;
        let thr: Vec<i32> = (0..N_LAYER).map(|i| rd_u32(b, o + 4 * i) as i32).collect();
        o += 4 * N_LAYER;
        let bias: Vec<i16> = (0..N_LAYER * N_EXPERT).map(|i| i16::from_le_bytes([b[o + 2 * i], b[o + 2 * i + 1]])).collect();
        o += 2 * N_LAYER * N_EXPERT;
        let tab: Vec<i8> = b[o..].iter().map(|&x| x as i8).collect();
        Ok(Lookahead {
            scale,
            thr,
            bias,
            tab,
            max: max.clamp(1, MAX_SPEC),
            max_pending: 2,
            in_flight: Vec::with_capacity(64),
            ids: vec![[0; TOP_K]; N_LAYER],
            have: 0,
            acc: vec![0; N_EXPERT],
            stats: LaStats::default(),
        })
    }

    pub fn load(path: &std::path::Path, max: usize) -> Result<Lookahead, String> {
        let b = std::fs::read(path).map_err(|e| format!("lookahead table {}: {e}", path.display()))?;
        Lookahead::from_bytes(&b, max)
    }

    pub fn thr(&self, t: usize) -> i32 {
        self.thr[t]
    }

    /// Scores of target layer `t` from `src[d-1]` = the experts routed at layer t-d (None = no such layer / not seen).
    pub fn score(&mut self, t: usize, src: [Option<&[u16; TOP_K]>; N_DIST], out: &mut [i32; N_EXPERT]) {
        // 30 rows of i8 into i16: |sum| <= 30 * 128 = 3840, no overflow
        let acc = &mut self.acc[..N_EXPERT];
        acc.fill(0);
        for (d, s) in src.iter().enumerate() {
            let Some(s) = s else { continue };
            let base = (t * N_DIST + d) * N_EXPERT * N_EXPERT;
            for &e in s.iter() {
                let row = &self.tab[base + e as usize * N_EXPERT..base + (e as usize + 1) * N_EXPERT];
                for (a, &r) in acc.iter_mut().zip(row) {
                    *a += r as i16;
                }
            }
        }
        let bias = &self.bias[t * N_EXPERT..(t + 1) * N_EXPERT];
        for e in 0..N_EXPERT {
            out[e] = bias[e] as i32 + acc[e] as i32;
        }
    }

    /// The best `max` experts with score >= thr[t] that `skip` does not exclude: score descending, then id ascending.
    /// Returns (how many are written to `out`, how many experts were at or above the threshold before `skip`).
    pub fn pick(&self, t: usize, scores: &[i32; N_EXPERT], max: usize, skip: impl Fn(usize) -> bool, out: &mut [(u16, i32); MAX_SPEC]) -> (usize, usize) {
        let thr = self.thr[t];
        let max = max.min(MAX_SPEC);
        let (mut n, mut above) = (0usize, 0usize);
        if thr == i32::MAX {
            return (0, 0);
        }
        for (e, &s) in scores.iter().enumerate() {
            if s < thr {
                continue;
            }
            above += 1;
            if skip(e) {
                continue;
            }
            // insertion into the sorted prefix; ids ascend, so an equal score stays behind the earlier id
            let mut i = n;
            while i > 0 && out[i - 1].1 < s {
                i -= 1;
            }
            if i >= max {
                continue;
            }
            let end = (n + 1).min(max);
            let mut k = end - 1;
            while k > i {
                out[k] = out[k - 1];
                k -= 1;
            }
            out[i] = (e as u16, s);
            n = end;
        }
        (n, above)
    }

    /// Record layer `l`'s routed experts of the current one-token decode (layer 0 starts a new token) and return the
    /// sources for target layer l+1, or None for the last layer.
    pub fn note(&mut self, l: usize, ids: &[i32]) -> Option<usize> {
        if l == 0 {
            self.have = 0;
        }
        for (j, &e) in ids.iter().take(TOP_K).enumerate() {
            self.ids[l][j] = e as u16;
        }
        self.have |= 1 << l;
        (l + 1 < N_LAYER).then_some(l + 1)
    }

    /// Forget the noted layers (a multi-token call ran: its layers are not this decode token's).
    pub fn note_reset(&mut self) {
        self.have = 0;
    }

    /// Scores of target layer `t` from the layers noted for this token.
    pub fn score_noted(&mut self, t: usize, out: &mut [i32; N_EXPERT]) {
        let mut rows = [[0u16; TOP_K]; N_DIST];
        let mut on = [false; N_DIST];
        for d in 1..=N_DIST {
            if t >= d && self.have >> (t - d) & 1 == 1 {
                rows[d - 1] = self.ids[t - d];
                on[d - 1] = true;
            }
        }
        let src = [on[0].then_some(&rows[0]), on[1].then_some(&rows[1]), on[2].then_some(&rows[2])];
        self.score(t, src, out);
    }
}

/// Check a table against the reference vectors written by tools/la_dense.py (LAV1): b"LAV1", u32 n, then per vector
/// u32 T, u16 ids[3][10] (0xFFFF = no such layer), u32 n_out (<= 8), n_out x (u16 expert, i32 score) = the experts at
/// or above thr[T], best first. Returns (vectors, mismatches).
pub fn check_vectors(la: &mut Lookahead, v: &[u8]) -> Result<(usize, usize), String> {
    if v.len() < 8 || &v[..4] != b"LAV1" {
        return Err("vectors: not a LAV1 file".into());
    }
    let n = rd_u32(v, 4) as usize;
    let mut o = 8;
    let mut bad = 0;
    let mut scores = [0i32; N_EXPERT];
    for i in 0..n {
        if o + 4 + 2 * N_DIST * TOP_K + 4 > v.len() {
            return Err(format!("vectors: truncated at vector {i}"));
        }
        let t = rd_u32(v, o) as usize;
        o += 4;
        if t >= N_LAYER {
            return Err(format!("vectors: layer {t} in vector {i}"));
        }
        let mut rows = [[0u16; TOP_K]; N_DIST];
        let mut on = [false; N_DIST];
        for d in 0..N_DIST {
            for j in 0..TOP_K {
                rows[d][j] = u16::from_le_bytes([v[o], v[o + 1]]);
                o += 2;
            }
            on[d] = rows[d][0] != 0xFFFF;
            if on[d] && rows[d].iter().any(|&e| e as usize >= N_EXPERT) {
                return Err(format!("vectors: expert id out of range in vector {i}"));
            }
        }
        let n_out = rd_u32(v, o) as usize;
        o += 4;
        if n_out > MAX_SPEC || o + 6 * n_out > v.len() {
            return Err(format!("vectors: bad n_out {n_out} in vector {i}"));
        }
        let src = [on[0].then_some(&rows[0]), on[1].then_some(&rows[1]), on[2].then_some(&rows[2])];
        la.score(t, src, &mut scores);
        let mut got = [(0u16, 0i32); MAX_SPEC];
        let (k, _) = la.pick(t, &scores, MAX_SPEC, |_| false, &mut got);
        let mut same = k == n_out;
        for g in got.iter().take(n_out.min(k)) {
            let e = u16::from_le_bytes([v[o], v[o + 1]]);
            let s = i32::from_le_bytes(v[o + 2..o + 6].try_into().unwrap());
            o += 6;
            same &= *g == (e, s);
        }
        o += 6 * n_out.saturating_sub(k);
        bad += !same as usize;
    }
    if o != v.len() {
        return Err(format!("vectors: {} trailing bytes", v.len() - o));
    }
    Ok((n, bad))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(seed: u64) -> Vec<u8> {
        let mut b = Vec::with_capacity(LAT_BYTES);
        b.extend_from_slice(b"LAT2");
        for v in [2u32, N_LAYER as u32, N_EXPERT as u32, N_DIST as u32, 16] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let mut x = seed;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for t in 0..N_LAYER {
            // a few layers never speculate
            let thr: i32 = if t % 11 == 5 { i32::MAX } else { 40 + (rnd() % 60) as i32 };
            b.extend_from_slice(&thr.to_le_bytes());
        }
        for _ in 0..N_LAYER * N_EXPERT {
            b.extend_from_slice(&(((rnd() % 400) as i16) - 300).to_le_bytes());
        }
        for _ in 0..N_LAYER * N_DIST * N_EXPERT * N_EXPERT / 8 {
            b.extend_from_slice(&rnd().to_le_bytes());
        }
        assert_eq!(b.len(), LAT_BYTES);
        b
    }

    #[test]
    fn score_and_pick_match_a_naive_reference() {
        let bytes = table(0x1234_5678_9abc_def1);
        let mut la = Lookahead::from_bytes(&bytes, 2).unwrap();
        assert!(Lookahead::from_bytes(&bytes[..LAT_BYTES - 1], 2).is_err());
        let mut x = 99u64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let tab_off = HDR + 4 * N_LAYER + 2 * N_LAYER * N_EXPERT;
        let mut some_pick = 0;
        for case in 0..200 {
            let t = (rnd() % N_LAYER as u64) as usize;
            let mut rows = [[0u16; TOP_K]; N_DIST];
            for r in rows.iter_mut() {
                for e in r.iter_mut() {
                    *e = (rnd() % N_EXPERT as u64) as u16;
                }
            }
            let on = [t >= 1, t >= 2 && case % 7 != 0, t >= 3];
            let src = [on[0].then_some(&rows[0]), on[1].then_some(&rows[1]), on[2].then_some(&rows[2])];
            let mut got = [0i32; N_EXPERT];
            la.score(t, src, &mut got);
            // naive: straight from the file bytes
            let mut want = [0i32; N_EXPERT];
            for e in 0..N_EXPERT {
                let o = HDR + 4 * N_LAYER + 2 * (t * N_EXPERT + e);
                let mut s = i16::from_le_bytes([bytes[o], bytes[o + 1]]) as i32;
                for d in 0..N_DIST {
                    if on[d] {
                        for &src_e in &rows[d] {
                            s += bytes[tab_off + ((t * N_DIST + d) * N_EXPERT + src_e as usize) * N_EXPERT + e] as i8 as i32;
                        }
                    }
                }
                want[e] = s;
            }
            assert_eq!(got, want, "case {case} layer {t}");
            let skip = |e: usize| e % 3 == 0;
            let mut out = [(0u16, 0i32); MAX_SPEC];
            for max in [1usize, 2, 8] {
                let (n, above) = la.pick(t, &got, max, skip, &mut out);
                let mut all: Vec<(u16, i32)> = (0..N_EXPERT).filter(|&e| got[e] >= la.thr(t) && la.thr(t) != i32::MAX).map(|e| (e as u16, got[e])).collect();
                assert_eq!(above, all.len());
                all.retain(|&(e, _)| !skip(e as usize));
                all.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                all.truncate(max);
                assert_eq!(&out[..n], &all[..], "case {case} layer {t} max {max}");
                some_pick += n;
            }
        }
        assert!(some_pick > 100, "the random table should select something ({some_pick})");
    }

    #[test]
    fn note_tracks_one_token() {
        let bytes = table(7);
        let mut la = Lookahead::from_bytes(&bytes, 2).unwrap();
        let ids = |l: usize| -> Vec<i32> { (0..TOP_K as i32).map(|j| (l as i32 * 13 + j * 7) % 512).collect() };
        let mut a = [0i32; N_EXPERT];
        let mut b = [0i32; N_EXPERT];
        for l in 0..N_LAYER {
            let t = la.note(l, &ids(l));
            assert_eq!(t, (l + 1 < N_LAYER).then_some(l + 1));
            let Some(t) = t else { continue };
            la.score_noted(t, &mut a);
            let rows: Vec<[u16; TOP_K]> = (1..=N_DIST).map(|d| if t >= d { std::array::from_fn(|j| ids(t - d)[j] as u16) } else { [0; TOP_K] }).collect();
            let src = [(t >= 1).then_some(&rows[0]), (t >= 2).then_some(&rows[1]), (t >= 3).then_some(&rows[2])];
            la.score(t, src, &mut b);
            assert_eq!(a, b, "layer {l}");
        }
        // a new token forgets the old layers: target 1 sees only layer 0
        assert_eq!(la.note(0, &ids(5)), Some(1));
        la.score_noted(1, &mut a);
        let r0: [u16; TOP_K] = std::array::from_fn(|j| ids(5)[j] as u16);
        la.score(1, [Some(&r0), None, None], &mut b);
        assert_eq!(a, b);
    }
}
