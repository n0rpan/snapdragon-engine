//! WP-T2: the Flash engine as the MTP driver's `Target` + `Draft` (`mtp.rs`), over one `GraphCtx`
//! (`mtp_graph.rs`). Pairing as llama's MTP driver: the head evaluates token x_q at head position q together
//! with the trunk residual of position q - 1 and predicts x_{q+1}; the first prompt token pairs with zeros.
//!
//! One cycle with the last committed token `t` at position P (not yet evaluated by the trunk):
//!   propose: head step (t, pending_h) at P -> d_1; then (d_i, the head's own residual) at P + i -> d_{i+1};
//!   verify:  trunk rows for [t, d_1..d_k] at P..P+k (provisional), logits + h per row;
//!   commit(m), catch_up: the head's cells of P+1..P+m-1 are rewritten with the true residuals (the cell of P
//!     was written by the first draft step from exactly these inputs and is kept); pending_h = h of P+m-1.
//! The head's cells beyond the committed position are stale but never visible (rank masks), like the trunk's.

use crate::graph::GraphCtx;
use crate::model::N_VOCAB;
use crate::mtp::{Draft, Rows, Target};
use crate::mtp_graph::H_DIM;
use std::time::Instant;

#[derive(Default, Clone, Debug)]
pub struct MtpTimes {
    pub verify_s: f64,
    pub commit_s: f64,
    pub draft_s: f64,
    pub catch_up_s: f64,
    pub draft_steps: u64,
    pub catch_up_rows: u64,
}

pub struct MtpEngine<'a> {
    pub g: &'a mut GraphCtx,
    pub threads: usize,
    pub n_ctx: usize,
    /// false: no head attached (propose returns nothing; every cycle is a one-row verify)
    pub head: bool,
    pub times: MtpTimes,
    pending_h: Vec<f32>,
    /// the head cell of the current position was written by this cycle's first draft step
    drafted: bool,
    lbuf: Vec<f32>,
    hbuf: Vec<f32>,
    chain: Vec<f32>,
}

impl<'a> MtpEngine<'a> {
    /// `g`: a cleared context (position 0); with `head` the model must have the head attached and
    /// `g.mtp_init()` done.
    pub fn new(g: &'a mut GraphCtx, n_ctx: usize, threads: usize, head: bool) -> MtpEngine<'a> {
        MtpEngine { g, threads, n_ctx, head, times: MtpTimes::default(), pending_h: vec![0.0; H_DIM], drafted: false, lbuf: Vec::new(), hbuf: Vec::new(), chain: vec![0.0; H_DIM] }
    }

    /// Prefill `prompt` in ubatches; the head (if any) follows every ubatch. `logits`: the last row's.
    pub fn prefill(&mut self, prompt: &[i32], ubatch: usize, threads_batch: usize, logits: &mut [f32]) -> Result<(), String> {
        for ch in prompt.chunks(ubatch.max(1)) {
            let pos0 = self.g.pos();
            if !self.head {
                self.g.decode(ch, threads_batch, logits, None)?;
                continue;
            }
            let n = ch.len();
            let mut h = vec![0f32; n * H_DIM];
            self.g.decode_h(ch, threads_batch, logits, &mut h)?;
            // head row i = (token i, residual of the position before it)
            let mut hh = vec![0f32; n * H_DIM];
            hh[..H_DIM].copy_from_slice(&self.pending_h);
            hh[H_DIM..].copy_from_slice(&h[..(n - 1) * H_DIM]);
            self.g.mtp_step(ch, &hh, pos0, threads_batch, false, &mut [])?;
            self.pending_h.copy_from_slice(&h[(n - 1) * H_DIM..]);
        }
        self.drafted = false;
        Ok(())
    }
}

impl Target for MtpEngine<'_> {
    fn verify(&mut self, toks: &[i32]) -> Result<Rows, String> {
        let n = toks.len();
        self.lbuf.resize(n * N_VOCAB, 0.0);
        self.hbuf.resize(n * H_DIM, 0.0);
        let t = Instant::now();
        self.g.verify(toks, self.threads, &mut self.lbuf, &mut self.hbuf)?;
        self.times.verify_s += t.elapsed().as_secs_f64();
        Ok(Rows {
            logits: self.lbuf.chunks_exact(N_VOCAB).take(n).map(|r| r.to_vec()).collect(),
            h: self.hbuf.chunks_exact(H_DIM).take(n).map(|r| r.to_vec()).collect(),
        })
    }
    fn commit(&mut self, m: usize) -> Result<(), String> {
        let t = Instant::now();
        let r = self.g.commit(m);
        self.times.commit_s += t.elapsed().as_secs_f64();
        r
    }
}

impl Draft for MtpEngine<'_> {
    fn propose(&mut self, t: i32, k: usize, floor: f32) -> Result<Vec<(i32, f32)>, String> {
        let mut out = Vec::new();
        if !self.head || k == 0 {
            return Ok(out);
        }
        let t0 = Instant::now();
        let pos0 = self.g.pos();
        let mut tok = t;
        let mut h_in = self.pending_h.clone();
        for i in 0..k {
            if pos0 + i >= self.n_ctx {
                break;
            }
            let mut h_out = std::mem::take(&mut self.chain);
            let r = self.g.mtp_step(&[tok], &h_in, pos0 + i, self.threads, true, &mut h_out);
            self.chain = h_out;
            let (d, p) = r?.ok_or("mtp_step returned no draft")?;
            self.times.draft_steps += 1;
            if i == 0 {
                self.drafted = true;
            }
            out.push((d, p));
            if p < floor {
                break; // the driver stops here too; no point in chaining further
            }
            tok = d;
            h_in.copy_from_slice(&self.chain);
        }
        self.times.draft_s += t0.elapsed().as_secs_f64();
        Ok(out)
    }

    fn rewind(&mut self) -> Result<(), String> {
        Ok(()) // head cells are indexed by position: the catch-up rewrites what it needs
    }

    fn catch_up(&mut self, toks: &[i32], h: &[Vec<f32>]) -> Result<(), String> {
        let m = toks.len();
        if !self.head || m == 0 {
            self.drafted = false;
            return Ok(());
        }
        let t0 = Instant::now();
        // the committed positions are p0..p0+m-1 (the trunk is already at p0 + m)
        let p0 = self.g.pos() - m;
        let skip = self.drafted as usize; // position p0's cell: written by the first draft step from (t, pending_h)
        if m > skip {
            let n = m - skip;
            let mut hh = vec![0f32; n * H_DIM];
            for i in skip..m {
                let src: &[f32] = if i == 0 { &self.pending_h } else { &h[i - 1] };
                hh[(i - skip) * H_DIM..(i - skip + 1) * H_DIM].copy_from_slice(src);
            }
            self.g.mtp_step(&toks[skip..], &hh, p0 + skip, self.threads, false, &mut [])?;
            self.times.catch_up_rows += n as u64;
        }
        self.pending_h.copy_from_slice(&h[m - 1]);
        self.drafted = false;
        self.times.catch_up_s += t0.elapsed().as_secs_f64();
        Ok(())
    }
}
