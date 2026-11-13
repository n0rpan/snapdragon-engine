//! M3.1: MTP (nextn) speculative decoding for qwen35moe.
//!
//! The head (llama.cpp `llama_model_qwen35moe::graph_mtp`, PR #22673) is one extra decoder
//! block `blk.{n_layer}` (full attention + MoE, its own KV) fed by
//! `eh_proj . [enorm(embd(tok)); hnorm(h)]`, then `shared_head_norm` and the shared output
//! head. `h` is the target's normalised final hidden state (`h_nextn` = output_norm(x)) of
//! the previous position, or for chained drafts the MTP's own `h_nextn`.
//!
//! Semantics follow `common/speculative.cpp` `common_speculative_impl_draft_mtp` (single
//! head, greedy, p_min 0):
//! - MTP KV position p always holds the pair (x_p, h_{p-1}) from the target (catch-up over
//!   every target-committed token; h_{-1} = 0).
//! - Draft: (x_last, h_{P-1}) at position P -> d1; then (d_i, h_mtp) at P+i -> d_{i+1}.
//! - Verify [x_last, d1..dk] through the target in one batch-invariant run, accept the
//!   longest prefix with d_i == argmax(lane i-1), commit a+1 tokens, bonus = argmax(lane a).
//!
//! Exactness: the verify run computes every lane with the one-token arithmetic (see
//! `engine.rs`), so the output is bit-identical to plain greedy decoding; drafts only
//! decide how many tokens one run yields.

use crate::desc::ModelDesc;
use crate::engine::{Engine, MAX_LANES};
use crate::plan::{split, split_rows, LayerTensors, Mixer, Phase, PhaseKind, Plan, Tid, Work};
use crate::weights::{LoadOptions, WeightRef, WeightStore};
use gguf::{GgmlType, Gguf};
use sched::Team;
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

/// MTP tensor ids are offset by this so they never collide with the target's ids (the MTP
/// block may come from another GGUF, e.g. a sidecar head, whose tensor indices overlap the
/// target's). `refs` is indexed by `tid - MTP_TID_BASE`.
pub const MTP_TID_BASE: Tid = 1 << 24;

/// Restricted-vocabulary draft head (WP-D3b): the target's output rows of `ids` (ascending,
/// padded to the kernels' row group by repeating the last id), repacked; the MTP's draft
/// argmax runs over these rows only. Drafts only: verify (and so the output) is unchanged.
pub struct DraftHead {
    pub w: WeightRef,
    pub ids: Vec<u32>,
    _buf: Box<[u8]>,
}

/// The MTP block's tensors (ids offset by [`MTP_TID_BASE`]; refs indexed by the local id).
pub struct MtpWeights {
    pub layer: LayerTensors,
    pub eh_proj: Tid,
    pub enorm: Tid,
    pub hnorm: Tid,
    pub head_norm: Tid,
    pub refs: Vec<Option<WeightRef>>,
    pub store: WeightStore,
    /// BF16 tensors converted to f32 (Qwen3.6's router / shared gate); referenced by `refs`.
    _f32: Vec<Box<[f32]>>,
    /// restricted-vocabulary draft head (None: drafts use the full output head)
    pub draft: Option<DraftHead>,
}

// SAFETY: read-only after load (the f32 boxes are never mutated).
unsafe impl Send for MtpWeights {}
unsafe impl Sync for MtpWeights {}

fn bf16_to_f32(b: &[u8]) -> Box<[f32]> {
    b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
}

impl MtpWeights {
    /// Load `blk.{n_layer}.*` of `model` (its own cache file `<sha>-<engine>-mtp.bin`).
    pub fn load(model: &Path, plan: &Plan, opt: &LoadOptions) -> Result<MtpWeights, String> {
        let g = Gguf::open(model).map_err(|e| e.to_string())?;
        let h = &g.header;
        let il = plan.desc.n_layer;
        let find = |s: &str| h.tensors.iter().position(|t| t.name == format!("blk.{il}.{s}")).ok_or(format!("MTP head: missing blk.{il}.{s}"));
        let layer = LayerTensors {
            attn_norm: find("attn_norm.weight")?,
            post_norm: find("post_attention_norm.weight")?,
            mixer: Mixer::Attention {
                q: find("attn_q.weight")?,
                k: find("attn_k.weight")?,
                v: find("attn_v.weight")?,
                q_norm: find("attn_q_norm.weight")?,
                k_norm: find("attn_k_norm.weight")?,
                out: find("attn_output.weight")?,
            },
            router: find("ffn_gate_inp.weight")?,
            shexp_gate: find("ffn_gate_inp_shexp.weight")?,
            gate_exps: find("ffn_gate_exps.weight")?,
            up_exps: find("ffn_up_exps.weight")?,
            down_exps: find("ffn_down_exps.weight")?,
            gate_shexp: find("ffn_gate_shexp.weight")?,
            up_shexp: find("ffn_up_shexp.weight")?,
            down_shexp: find("ffn_down_shexp.weight")?,
        };
        let (eh_proj, enorm, hnorm, head_norm) =
            (find("nextn.eh_proj.weight")?, find("nextn.enorm.weight")?, find("nextn.hnorm.weight")?, find("nextn.shared_head_norm.weight")?);
        if find("nextn.embed_tokens.weight").is_ok() || find("nextn.shared_head_head.weight").is_ok() {
            return Err("MTP head with its own embedding/head is not supported (both GGUFs share the target's)".into());
        }
        let d = &plan.desc;
        let t = |id: Tid| &h.tensors[id];
        if t(eh_proj).shape != vec![2 * d.n_embd as u64, d.n_embd as u64] {
            return Err(format!("nextn.eh_proj shape {:?}", t(eh_proj).shape));
        }
        let mut ids = layer.all();
        ids.extend([eh_proj, enorm, hnorm, head_norm]);
        ids.sort_unstable();
        ids.dedup();
        // BF16 small tensors: converted to f32 here (llama.cpp rounds the activation to bf16
        // for these matmuls; the drafts may differ in rare ties, the verified output cannot).
        let mut f32s = Vec::new();
        let mut conv = Vec::new();
        for &id in &ids {
            if h.tensors[id].ty == GgmlType(30) {
                let ti = &h.tensors[id];
                let v = bf16_to_f32(g.tensor_data(ti));
                let r = WeightRef {
                    ptr: v.as_ptr() as usize,
                    layout: crate::weights::Layout::F32,
                    k: ti.shape[0] as usize,
                    rows: ti.shape.get(1).copied().unwrap_or(1) as usize,
                    n_expert: 1,
                    bytes: v.len() * 4,
                };
                conv.push((id, r));
                f32s.push(v);
            }
        }
        let rest: Vec<Tid> = ids.iter().copied().filter(|id| !conv.iter().any(|c| c.0 == *id)).collect();
        let n_tensors = h.tensors.len();
        let store = WeightStore::load_ids(g, rest, None, opt, "-mtp")?;
        let mut refs = vec![None; n_tensors];
        for (i, r) in store.refs.iter().enumerate() {
            if r.is_some() {
                refs[i] = *r;
            }
        }
        for (id, r) in conv {
            refs[id] = Some(r);
        }
        // move the ids out of the target's range (see MTP_TID_BASE)
        let b = MTP_TID_BASE;
        let mixer = match layer.mixer {
            Mixer::Attention { q, k, v, q_norm, k_norm, out } => Mixer::Attention { q: q + b, k: k + b, v: v + b, q_norm: q_norm + b, k_norm: k_norm + b, out: out + b },
            Mixer::Recurrent { .. } => unreachable!("the MTP block is an attention layer"),
        };
        let layer = LayerTensors {
            attn_norm: layer.attn_norm + b,
            post_norm: layer.post_norm + b,
            mixer,
            router: layer.router + b,
            shexp_gate: layer.shexp_gate + b,
            gate_exps: layer.gate_exps + b,
            up_exps: layer.up_exps + b,
            down_exps: layer.down_exps + b,
            gate_shexp: layer.gate_shexp + b,
            up_shexp: layer.up_shexp + b,
            down_shexp: layer.down_shexp + b,
        };
        Ok(MtpWeights { layer, eh_proj: eh_proj + b, enorm: enorm + b, hnorm: hnorm + b, head_norm: head_norm + b, refs, store, _f32: f32s, draft: None })
    }

    /// The local ref of an offset MTP tensor id.
    pub fn get(&self, id: Tid) -> Option<&WeightRef> {
        id.checked_sub(MTP_TID_BASE).and_then(|i| self.refs.get(i)).and_then(|r| r.as_ref())
    }

    /// Restrict the draft head to `ids` (WP-D3b): rows of the target's `output.weight`, read from
    /// the target GGUF `target` (row-major quantised rows), gathered and repacked.
    pub fn set_draft_vocab(&mut self, target: &Path, ids: &[u32]) -> Result<(), String> {
        let g = Gguf::open(target).map_err(|e| e.to_string())?;
        let ti = g.header.tensors.iter().find(|t| t.name == "output.weight").ok_or("target has no output.weight")?;
        let q = crate::desc::qtype(ti.ty).ok_or(format!("output.weight type {} not supported for a draft head", ti.ty.name()))?;
        let k = ti.shape[0] as usize;
        let n_vocab = ti.shape[1] as usize;
        let mut v: Vec<u32> = ids.iter().copied().filter(|&i| (i as usize) < n_vocab).collect();
        v.sort_unstable();
        v.dedup();
        if v.is_empty() {
            return Err("empty draft vocabulary".into());
        }
        let nr = q.nr();
        while v.len() % nr != 0 {
            v.push(*v.last().unwrap());
        }
        let rb = q.row_bytes(k);
        let src = g.tensor_data(ti);
        let mut rows = vec![0u8; v.len() * rb];
        for (j, &id) in v.iter().enumerate() {
            rows[j * rb..(j + 1) * rb].copy_from_slice(&src[id as usize * rb..(id as usize + 1) * rb]);
        }
        let mut buf = vec![0u8; v.len() * rb].into_boxed_slice();
        kernels_sys::repack(q, &rows, &mut buf, v.len(), k);
        let w = WeightRef { ptr: buf.as_ptr() as usize, layout: crate::weights::Layout::Packed(q), k, rows: v.len(), n_expert: 1, bytes: buf.len() };
        self.draft = Some(DraftHead { w, ids: v, _buf: buf });
        Ok(())
    }

    /// Apply the draft vocabulary for an MTP head read from `src`, for target `target`: the
    /// ranking file `ranking` (one token id per line, most frequent first) cut to `k` ids if
    /// given, else the vocabulary stored in the sidecar `src` if any. Returns the draft rows.
    pub fn apply_draft_vocab(&mut self, src: &Path, target: &Path, ranking: Option<&Path>, k: Option<usize>) -> Result<Option<usize>, String> {
        let ids = match ranking {
            Some(r) => {
                let text = std::fs::read_to_string(r).map_err(|e| format!("{}: {e}", r.display()))?;
                let mut v: Vec<u32> = text.lines().filter_map(|l| l.split_whitespace().next()?.parse().ok()).collect();
                if let Some(k) = k {
                    v.truncate(k);
                }
                Some(v)
            }
            None => Self::sidecar_draft_vocab(src)?,
        };
        match ids {
            Some(ids) => {
                self.set_draft_vocab(target, &ids)?;
                Ok(self.draft.as_ref().map(|d| d.w.rows))
            }
            None => Ok(None),
        }
    }

    /// The draft vocabulary stored in an MTP-head sidecar (tensor `oxe.draft_vocab.ids`, I32), if any.
    pub fn sidecar_draft_vocab(src: &Path) -> Result<Option<Vec<u32>>, String> {
        let g = Gguf::open(src).map_err(|e| e.to_string())?;
        let Some(ti) = g.header.tensors.iter().find(|t| t.name == "oxe.draft_vocab.ids") else { return Ok(None) };
        if ti.ty != GgmlType(26) {
            return Err(format!("oxe.draft_vocab.ids: type {}, expected I32", ti.ty.name()));
        }
        Ok(Some(g.tensor_data(ti).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32).collect()))
    }

    /// Bytes of the MTP head arena (cache file) plus the converted tensors.
    pub fn bytes(&self) -> u64 {
        self.store.repacked_bytes + self._f32.iter().map(|v| 4 * v.len() as u64).sum::<u64>()
    }
}

/// The MTP block's phases (layer index n_layer: attention + MoE + head) and the per-thread
/// rows of eh_proj (the entry runs before them, then a barrier).
/// `head_rows`: rows of a restricted draft head (None: the full vocabulary).
pub fn phases(d: &ModelDesc, n: usize, head_rows: Option<usize>) -> (Vec<Phase>, Vec<Range<usize>>) {
    let il = d.n_layer;
    let qh = split(d.n_head, n);
    let kv_rows = d.n_head_kv * d.head_dim;
    let (ks, vs) = (split_rows(kv_rows, n), split_rows(kv_rows, n));
    let mut p = Vec::new();
    let ph = |kind, work, barrier_after| Phase { kind, layer: il, barrier_after, work };
    p.push(ph(
        PhaseKind::AttnQkv,
        (0..n).map(|t| Work::AttnQkv { qheads: qh[t].clone(), k_rows: ks[t].clone(), v_rows: vs[n - 1 - t].clone() }).collect(),
        true,
    ));
    p.push(ph(PhaseKind::AttnCore, qh.iter().map(|q| Work::AttnCore { qheads: q.clone() }).collect(), true));
    p.push(ph(PhaseKind::AttnOut, split_rows(d.n_embd, n).into_iter().map(|rows| Work::AttnOut { rows }).collect(), true));
    let sh = split_rows(d.ff_shexp, n);
    p.push(ph(PhaseKind::MoeRouter, split(d.n_expert, n).into_iter().zip(sh).map(|(rows, sh_rows)| Work::MoeRouter { rows, sh_rows }).collect(), true));
    let units = d.n_expert_used * (d.ff_exp / kernels_sys::QK_K);
    p.push(ph(PhaseKind::MoeGateUp, split(units, n).into_iter().map(|units| Work::MoeGateUp { units }).collect(), true));
    p.push(ph(PhaseKind::MoeDown, split_rows(d.n_embd, n).into_iter().map(|rows| Work::MoeDown { rows }).collect(), true));
    p.push(ph(PhaseKind::Head, split_rows(head_rows.unwrap_or(d.n_vocab), n).into_iter().map(|rows| Work::Head { rows }).collect(), false));
    (p, split_rows(d.n_embd, n))
}

/// OXE_MTP_TRACE=1: one stderr line per speculative step (drafts, target argmaxes, accepted).
fn mtp_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("OXE_MTP_TRACE").is_some())
}

/// Counters of one speculative decode (M3.1 instrumentation).
#[derive(Clone, Debug, Default)]
pub struct MtpStats {
    /// verify runs (each yields 1 + accepted tokens)
    pub steps: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// accepted[i] = verify steps whose draft i+1 was accepted
    pub accepted_at: [u64; MAX_LANES],
    /// tokens produced by verify steps (accepted drafts + bonus)
    pub tokens_out: u64,
    /// MTP runs (catch-up + draft chain) / their lanes
    pub mtp_runs: u64,
    pub mtp_lanes: u64,
    pub draft_s: f64,
    pub verify_s: f64,
    pub catchup_s: f64,
    /// batched prompt prefill through the driver (`prefill_nextn`), and its MTP seeding
    pub prefill_s: f64,
    pub prefill_tokens: u64,
    pub prefill_seed_s: f64,
    /// plain (non-speculative) target steps through the driver (prefill tail, context end)
    pub plain_steps: u64,
}

impl MtpStats {
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
    pub fn tokens_per_step(&self) -> f64 {
        if self.steps == 0 {
            0.0
        } else {
            self.tokens_out as f64 / self.steps as f64
        }
    }
}

/// Greedy MTP speculative decoding over an [`Engine`] built with an MTP head.
pub struct MtpDriver {
    pub depth: usize,
    ne: usize,
    /// MTP KV holds correct (target) pairs for positions < q
    q: usize,
    /// pending catch-up pairs for positions q, q+1, ...
    cu_tok: Vec<u32>,
    cu_h: Vec<f32>,
    /// target h_nextn of position pos-1 (zeros at pos 0)
    hl: Vec<f32>,
    /// h rows of the last verify's committed lanes (for `rewind`), its pos0 and length
    last_h: Vec<f32>,
    last: Option<(usize, usize)>,
    pub stats: MtpStats,
    /// Test hook: replaces draft i (at position P+1+i) after the MTP proposed it.
    #[doc(hidden)]
    pub draft_override: Option<Box<dyn FnMut(usize, u32) -> u32 + Send>>,
    /// prefill hidden rows (reused)
    scratch: Vec<f32>,
    /// LC-6: most tokens per `prefill_nextn` call inside [`MtpDriver::prefill_batched`] (rounded
    /// down to a multiple of the ubatch, at least one ubatch; 0 = [`PREFILL_CHUNK_TOKENS`]). Bounds
    /// the hidden-row scratch and the pending catch-up rows to chunk x n_embd floats each.
    #[doc(hidden)]
    pub prefill_chunk: usize,
}

/// LC-6: default tokens per chunk of [`MtpDriver::prefill_batched`]: 8192 x n_embd f32 = 64 MiB of
/// hidden rows for Ornith's 2048 (plus the same for the pending MTP catch-up), not 1 GiB per
/// 128K-token call. Exact: chunk ends fall on ubatch boundaries, so every llama.cpp ubatch, its
/// padded n_kv and the grouped passes are what one call computes; the MTP catch-up runs the same
/// full lane groups from the same positions (each flush leaves fewer than `mtp_lanes` pairs).
pub const PREFILL_CHUNK_TOKENS: usize = 8192;

impl MtpDriver {
    pub fn new(eng: &Engine, depth: usize) -> Result<MtpDriver, String> {
        if eng.mtp_lanes() == 0 {
            return Err("engine has no MTP head".into());
        }
        if depth == 0 || depth + 1 > eng.verify_lanes() {
            return Err(format!("MTP depth {depth}: the engine verifies {} tokens per run", eng.verify_lanes()));
        }
        let ne = eng.plan.desc.n_embd;
        let mut d = MtpDriver { depth, ne, q: 0, cu_tok: vec![], cu_h: vec![], hl: vec![0.0; ne], last_h: vec![], last: None, stats: MtpStats::default(), draft_override: None, scratch: Vec::new(), prefill_chunk: 0 };
        d.sync(eng, None);
        Ok(d)
    }

    /// Re-synchronise with the engine after it was reset or restored outside the driver.
    /// `h_prev` = the target's h_nextn of position pos-1 (None: zeros, e.g. after a reset).
    pub fn sync(&mut self, eng: &Engine, h_prev: Option<&[f32]>) {
        let p = eng.pos();
        if p < self.q {
            self.q = p;
        }
        let keep = p.saturating_sub(self.q).min(self.cu_tok.len());
        self.cu_tok.truncate(keep);
        self.cu_h.truncate(keep * self.ne);
        if self.q + self.cu_tok.len() < p {
            // positions the driver did not see: the MTP KV stays stale there (drafts only)
            self.q = p;
            self.cu_tok.clear();
            self.cu_h.clear();
        }
        match h_prev {
            Some(h) => self.hl.copy_from_slice(&h[..self.ne]),
            None => self.hl.fill(0.0),
        }
        self.last = None;
    }

    /// Driver state for the persistent prefix store: (q, pending catch-up tokens, their h
    /// rows, h_prev). The MTP KV of positions < q is in the engine's state.
    pub fn export_state(&self) -> (usize, Vec<u32>, Vec<f32>, Vec<f32>) {
        (self.q, self.cu_tok.clone(), self.cu_h.clone(), self.hl.clone())
    }

    /// Restore [`MtpDriver::export_state`] after the engine state was read back.
    pub fn import_state(&mut self, eng: &Engine, q: usize, cu_tok: Vec<u32>, cu_h: Vec<f32>, hl: &[f32]) -> Result<(), String> {
        if cu_h.len() != cu_tok.len() * self.ne || hl.len() != self.ne || q + cu_tok.len() != eng.pos() {
            return Err("MTP state does not match the engine".into());
        }
        self.q = q;
        self.cu_tok = cu_tok;
        self.cu_h = cu_h;
        self.hl.copy_from_slice(hl);
        self.last = None;
        Ok(())
    }

    /// The target's h_nextn at position pos-1 (to store with a snapshot).
    pub fn h_prev(&self) -> &[f32] {
        &self.hl
    }

    /// Record target tokens committed at positions p0.. with the engine's hidden lanes 0...
    fn note(&mut self, eng: &Engine, p0: usize, toks: &[u32]) {
        self.note_with(p0, toks, |i| eng.hidden(i));
    }

    /// Record target tokens committed at positions p0..; `h(i)` = the target's h_nextn after
    /// token i (pairs (t_i, h(i-1)), the first with the previous h).
    fn note_with<'h>(&mut self, p0: usize, toks: &[u32], h: impl Fn(usize) -> &'h [f32]) {
        for (i, &t) in toks.iter().enumerate() {
            let p = p0 + i;
            if p >= self.q {
                assert_eq!(p, self.q + self.cu_tok.len(), "MTP catch-up out of order");
                self.cu_tok.push(t);
                if i == 0 {
                    self.cu_h.extend_from_slice(&self.hl);
                } else {
                    self.cu_h.extend_from_slice(&h(i - 1)[..self.ne]);
                }
            }
        }
        self.hl.copy_from_slice(&h(toks.len() - 1)[..self.ne]);
    }

    /// Run catch-up MTP lanes until at most `keep` pairs are pending.
    fn flush<P: Team + ?Sized>(&mut self, eng: &Engine, pool: &P, keep: usize) {
        let l = eng.mtp_lanes();
        while self.cu_tok.len() > keep {
            let t = Instant::now();
            let n = l.min(self.cu_tok.len());
            let hs: Vec<&[f32]> = (0..n).map(|i| &self.cu_h[i * self.ne..(i + 1) * self.ne]).collect();
            eng.mtp_seed(pool, &self.cu_tok[..n], self.q, &hs);
            self.cu_tok.drain(..n);
            self.cu_h.drain(..n * self.ne);
            self.q += n;
            self.stats.mtp_runs += 1;
            self.stats.mtp_lanes += n as u64;
            self.stats.catchup_s += t.elapsed().as_secs_f64();
        }
    }

    /// Append prompt tokens (verify runs of up to `verify_lanes` tokens: exact, and the
    /// weights are read once per run). Returns the argmax after the last token.
    pub fn prefill<P: Team + ?Sized>(&mut self, eng: &Engine, pool: &P, toks: &[u32]) -> u32 {
        let mut next = u32::MAX;
        for c in toks.chunks(eng.verify_lanes()) {
            let p0 = eng.pos();
            let g = eng.verify(pool, c);
            eng.commit(c.len());
            next = g[c.len() - 1];
            self.note(eng, p0, c);
            self.flush(eng, pool, eng.mtp_lanes() - 1);
        }
        self.last = None;
        next
    }

    /// Append a prompt with the batched prefill (`Engine::prefill_nextn` on `team`: the same
    /// KV, recurrent state and last-token logits as `Engine::prefill`), then seed the MTP KV
    /// from every position's h_nextn (catch-up runs on `pool`, the decode team). Returns the
    /// argmax after the last token; its logits are `eng.logits()`.
    pub fn prefill_batched<T: Team + ?Sized, P: Team + ?Sized>(&mut self, eng: &Engine, team: &T, pool: &P, toks: &[u32], ubatch: usize) -> u32 {
        if toks.len() == 1 {
            let mut out = Vec::new();
            self.step_plain(eng, pool, toks[0], &mut out);
            return out[0];
        }
        let ub = ubatch.max(1);
        let want = if self.prefill_chunk == 0 { PREFILL_CHUNK_TOKENS } else { self.prefill_chunk };
        let chunk = (want / ub).max(1) * ub;
        let (mut pf_s, mut seed_s) = (0f64, 0f64);
        let mut g = 0;
        let mut hid = std::mem::take(&mut self.scratch);
        for c in toks.chunks(chunk) {
            let p0 = eng.pos();
            let t0 = Instant::now();
            g = eng.prefill_nextn(team, c, ubatch, &mut hid).expect("non-empty prompt");
            let t1 = Instant::now();
            let ne = self.ne;
            self.note_with(p0, c, |i| &hid[i * ne..(i + 1) * ne]);
            self.flush(eng, pool, eng.mtp_lanes() - 1);
            pf_s += (t1 - t0).as_secs_f64();
            seed_s += t1.elapsed().as_secs_f64();
        }
        self.scratch = hid;
        self.stats.prefill_s += pf_s;
        self.stats.prefill_seed_s += seed_s;
        self.stats.prefill_tokens += toks.len() as u64;
        if std::env::var_os("OXE_MTP_LOG").is_some() {
            eprintln!("oxe mtp: prefill {} tokens: prefill_nextn {:.1} ms, MTP seeding {:.1} ms", toks.len(), pf_s * 1e3, seed_s * 1e3);
        }
        self.last = None;
        g
    }

    /// One plain (non-speculative) target step that keeps the MTP catch-up in sync (sampled
    /// requests). `out` = [argmax after `x`].
    pub fn step_plain<P: Team + ?Sized>(&mut self, eng: &Engine, pool: &P, x: u32, out: &mut Vec<u32>) {
        out.clear();
        let p = eng.pos();
        let g = eng.step(pool, x);
        self.note(eng, p, &[x]);
        self.stats.plain_steps += 1;
        self.last = Some((p, 1));
        self.last_h.clear();
        self.last_h.extend_from_slice(eng.hidden(0));
        self.flush(eng, pool, eng.mtp_lanes() - 1);
        out.push(g);
    }

    /// One speculative step: `x` (the next token, not yet in the target) is appended
    /// together with every accepted draft. Returns the greedy tokens that follow, in order:
    /// out[..len-1] were committed to the target with `x` (they equal the accepted drafts),
    /// out[len-1] is the bonus token (not yet in the target: the next call's `x`).
    /// `eng.logits_lane(i)` is the target's logit row after the i-th committed token.
    pub fn step<P: Team + ?Sized>(&mut self, eng: &Engine, pool: &P, x: u32, out: &mut Vec<u32>) {
        out.clear();
        let p = eng.pos();
        let k = self.depth.min(eng.n_ctx.saturating_sub(p + 1));
        self.flush(eng, pool, eng.mtp_lanes() - 1);
        if k == 0 {
            let g = eng.step(pool, x);
            self.note(eng, p, &[x]);
            self.stats.plain_steps += 1;
            self.last = Some((p, 1));
            self.last_h.clear();
            self.last_h.extend_from_slice(eng.hidden(0));
            out.push(g);
            return;
        }
        // draft: seed the pending catch-up pairs (KV only), then (x, h_{P-1}) at P with the
        // head, then the chain. The seeded KV rows are the bits a full MTP run writes.
        let t0 = Instant::now();
        self.flush(eng, pool, 0);
        let n0 = 1;
        let mut drafts = [0u32; MAX_LANES];
        drafts[0] = eng.mtp_run(pool, &[x], p, &[&self.hl], true);
        self.stats.mtp_runs += 1;
        self.stats.mtp_lanes += 1;
        self.q = p + 1;
        let mut h = eng.hidden(n0 - 1).to_vec();
        for i in 1..k {
            drafts[i] = eng.mtp_run(pool, &[drafts[i - 1]], p + i, &[&h], true);
            h.copy_from_slice(eng.hidden(0));
            self.stats.mtp_runs += 1;
            self.stats.mtp_lanes += 1;
        }
        if let Some(f) = &mut self.draft_override {
            for (i, dr) in drafts.iter_mut().enumerate().take(k) {
                *dr = f(p + 1 + i, *dr);
            }
        }
        let t1 = Instant::now();
        // verify [x, d1..dk]
        let mut vt = [0u32; MAX_LANES];
        vt[0] = x;
        vt[1..=k].copy_from_slice(&drafts[..k]);
        let g = eng.verify(pool, &vt[..=k]);
        let mut a = 0;
        while a < k && drafts[a] == g[a] {
            a += 1;
        }
        eng.commit(a + 1);
        let t2 = Instant::now();
        if mtp_trace() {
            eprintln!("oxe mtp step: pos {p} x {x} drafts {:?} target {:?} accepted {a}", &drafts[..k], &g[..=k]);
        }
        // the pair (x, h_{P-1}) is already in the MTP KV (q = P+1): record d_1..d_a
        self.last_h.clear();
        for i in 0..=a {
            self.last_h.extend_from_slice(eng.hidden(i));
        }
        self.note(eng, p, &vt[..=a]);
        self.last = Some((p, a + 1));
        let s = &mut self.stats;
        s.steps += 1;
        s.drafted += k as u64;
        s.accepted += a as u64;
        for i in 0..a {
            s.accepted_at[i] += 1;
        }
        s.tokens_out += a as u64 + 1;
        s.draft_s += (t1 - t0).as_secs_f64();
        s.verify_s += (t2 - t1).as_secs_f64();
        out.extend_from_slice(&g[..=a]);
    }

    /// Keep only the first `m` (>= 1) tokens the last `step` committed (the caller stopped
    /// early, e.g. on EOG inside an accepted run). Valid until the next engine run.
    pub fn rewind(&mut self, eng: &Engine, m: usize) {
        let (p0, n) = self.last.expect("rewind without a step");
        assert!(m >= 1 && m <= n);
        if m == n {
            return;
        }
        eng.commit(m);
        let keep = (p0 + m).saturating_sub(self.q).min(self.cu_tok.len());
        self.cu_tok.truncate(keep);
        self.cu_h.truncate(keep * self.ne);
        self.hl.copy_from_slice(&self.last_h[(m - 1) * self.ne..m * self.ne]);
        self.last = Some((p0, m));
    }
}
