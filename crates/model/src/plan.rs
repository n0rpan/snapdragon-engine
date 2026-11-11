//! The static per-token plan (SPEC B.2/B.3, adapted for bit-exactness in M1): phases,
//! per-thread work slices, barriers, and the byte accounting derived from it. Built once at
//! load; immutable afterwards.

use crate::desc::{role, LayerKind, ModelDesc, Role};
use gguf::{Header, TensorInfo};
use std::ops::Range;

pub type Tid = usize;

#[derive(Clone, Debug)]
pub enum Mixer {
    Recurrent { qkv: Tid, z: Tid, beta: Tid, alpha: Tid, conv1d: Tid, a: Tid, dt_bias: Tid, norm: Tid, out: Tid },
    Attention { q: Tid, k: Tid, v: Tid, q_norm: Tid, k_norm: Tid, out: Tid },
}

#[derive(Clone, Debug)]
pub struct LayerTensors {
    pub attn_norm: Tid,
    pub post_norm: Tid,
    pub mixer: Mixer,
    pub router: Tid,
    pub shexp_gate: Tid,
    pub gate_exps: Tid,
    pub up_exps: Tid,
    pub down_exps: Tid,
    pub gate_shexp: Tid,
    pub up_shexp: Tid,
    pub down_shexp: Tid,
}

impl LayerTensors {
    pub fn all(&self) -> Vec<Tid> {
        let mut v = vec![self.attn_norm, self.post_norm, self.router, self.shexp_gate, self.gate_exps, self.up_exps, self.down_exps];
        v.extend([self.gate_shexp, self.up_shexp, self.down_shexp]);
        match self.mixer {
            Mixer::Recurrent { qkv, z, beta, alpha, conv1d, a, dt_bias, norm, out } => v.extend([qkv, z, beta, alpha, conv1d, a, dt_bias, norm, out]),
            Mixer::Attention { q, k, v: vv, q_norm, k_norm, out } => v.extend([q, k, vv, q_norm, k_norm, out]),
        }
        v
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhaseKind {
    RecMix,
    RecOut,
    AttnQkv,
    AttnCore,
    AttnOut,
    MoeRouter,
    MoeGateUp,
    MoeDown,
    Head,
}

impl PhaseKind {
    pub const ALL: [PhaseKind; 9] = [
        PhaseKind::RecMix,
        PhaseKind::RecOut,
        PhaseKind::AttnQkv,
        PhaseKind::AttnCore,
        PhaseKind::AttnOut,
        PhaseKind::MoeRouter,
        PhaseKind::MoeGateUp,
        PhaseKind::MoeDown,
        PhaseKind::Head,
    ];
    pub fn name(self) -> &'static str {
        match self {
            PhaseKind::RecMix => "deltanet_mix",
            PhaseKind::RecOut => "deltanet_out",
            PhaseKind::AttnQkv => "attn_qkv",
            PhaseKind::AttnCore => "attn_core",
            PhaseKind::AttnOut => "attn_out",
            PhaseKind::MoeRouter => "moe_router",
            PhaseKind::MoeGateUp => "moe_gate_up",
            PhaseKind::MoeDown => "moe_down",
            PhaseKind::Head => "head",
        }
    }
    pub fn index(self) -> usize {
        self as usize
    }
}

/// One thread's share of a phase. Row ranges are in tensor rows (multiples of 8), head
/// ranges in heads. Every phase ends with a barrier except the head (the pool join).
#[derive(Clone, Debug, PartialEq)]
pub enum Work {
    /// Delta-net mixing for key heads `kheads` and the value heads that use them
    /// (llama.cpp maps value head v to key head v % n_kheads).
    RecMix { kheads: Range<usize> },
    RecOut { rows: Range<usize> },
    /// q (+gate) rows of `qheads`; raw K rows `k_rows`; V rows `v_rows` (written to the cache).
    AttnQkv { qheads: Range<usize>, k_rows: Range<usize>, v_rows: Range<usize> },
    AttnCore { qheads: Range<usize> },
    AttnOut { rows: Range<usize> },
    /// Router logits rows, and the shared expert's gate/up rows (its h slice).
    MoeRouter { rows: Range<usize>, sh_rows: Range<usize> },
    /// Routed units (slot j, 256-row block b) flattened as j * blocks + b.
    MoeGateUp { units: Range<usize> },
    /// Output rows of every expert's down projection, combined in llama.cpp order.
    MoeDown { rows: Range<usize> },
    Head { rows: Range<usize> },
}

#[derive(Clone, Debug)]
pub struct Phase {
    pub kind: PhaseKind,
    pub layer: usize,
    pub barrier_after: bool,
    pub work: Vec<Work>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub desc: ModelDesc,
    pub nthreads: usize,
    /// Cached positions attended to per token (the new token is written at n_kv - 1).
    pub n_kv: usize,
    pub layers: Vec<LayerTensors>,
    pub embd: Tid,
    pub out_norm: Tid,
    pub output: Tid,
    pub phases: Vec<Phase>,
    /// Rows of the output head per GEMV call (bounds the per-thread scratch).
    pub head_chunk: usize,
}

/// Bytes moved per token, by category (unique DRAM traffic, not counting redundancy).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Budget {
    /// Tensors read in full every token (dense projections, norms, router, shared experts, head).
    pub weights_dense: u64,
    /// Routed experts at top-k / n_expert.
    pub weights_expert: u64,
    pub embedding_row: u64,
    /// Recurrent state read + write.
    pub state_rw: u64,
    pub kv_read: u64,
    pub kv_write: u64,
}

impl Budget {
    /// What model_inspect.py calls active weight bytes per token.
    pub fn weights(&self) -> u64 {
        self.weights_dense + self.weights_expert
    }
    pub fn total(&self) -> u64 {
        self.weights() + self.embedding_row + self.state_rw + self.kv_read + self.kv_write
    }
}

/// Split `n` items into `parts` contiguous ranges, sizes differing by at most one.
pub fn split(n: usize, parts: usize) -> Vec<Range<usize>> {
    (0..parts).map(|i| (n * i / parts)..(n * (i + 1) / parts)).collect()
}

/// Split `rows` (a multiple of 8) into per-thread ranges of whole 8-row groups.
pub fn split_rows(rows: usize, parts: usize) -> Vec<Range<usize>> {
    assert!(rows % 8 == 0, "rows {rows} not a multiple of 8");
    split(rows / 8, parts).into_iter().map(|r| r.start * 8..r.end * 8).collect()
}

fn find(h: &Header, name: &str) -> Result<Tid, String> {
    h.tensors.iter().position(|t| t.name == name).ok_or(format!("missing tensor {name}"))
}

impl Plan {
    pub fn build(h: &Header, desc: &ModelDesc, nthreads: usize, n_kv: usize) -> Result<Plan, String> {
        assert!(nthreads >= 1 && n_kv >= 1);
        let mut layers = Vec::new();
        for (il, kind) in desc.layers.iter().enumerate() {
            let t = |s: &str| find(h, &format!("blk.{il}.{s}"));
            let mixer = match kind {
                LayerKind::Recurrent => Mixer::Recurrent {
                    qkv: t("attn_qkv.weight")?,
                    z: t("attn_gate.weight")?,
                    beta: t("ssm_beta.weight")?,
                    alpha: t("ssm_alpha.weight")?,
                    conv1d: t("ssm_conv1d.weight")?,
                    a: t("ssm_a")?,
                    dt_bias: t("ssm_dt.bias")?,
                    norm: t("ssm_norm.weight")?,
                    out: t("ssm_out.weight")?,
                },
                LayerKind::Attention => Mixer::Attention {
                    q: t("attn_q.weight")?,
                    k: t("attn_k.weight")?,
                    v: t("attn_v.weight")?,
                    q_norm: t("attn_q_norm.weight")?,
                    k_norm: t("attn_k_norm.weight")?,
                    out: t("attn_output.weight")?,
                },
            };
            layers.push(LayerTensors {
                attn_norm: t("attn_norm.weight")?,
                post_norm: t("post_attention_norm.weight")?,
                mixer,
                router: t("ffn_gate_inp.weight")?,
                shexp_gate: t("ffn_gate_inp_shexp.weight")?,
                gate_exps: t("ffn_gate_exps.weight")?,
                up_exps: t("ffn_up_exps.weight")?,
                down_exps: t("ffn_down_exps.weight")?,
                gate_shexp: t("ffn_gate_shexp.weight")?,
                up_shexp: t("ffn_up_shexp.weight")?,
                down_shexp: t("ffn_down_shexp.weight")?,
            });
        }
        let n = nthreads;
        let d = desc;
        let mut phases = Vec::new();
        let kh_split = split(d.ssm_n_kheads, n);
        let qh_split = split(d.n_head, n);
        let units = d.n_expert_used * (d.ff_exp / kernels_sys::QK_K);
        let unit_split = split(units, n);
        for (il, kind) in d.layers.iter().enumerate() {
            match kind {
                LayerKind::Recurrent => {
                    let work = kh_split.iter().map(|kh| Work::RecMix { kheads: kh.clone() }).collect();
                    phases.push(Phase { kind: PhaseKind::RecMix, layer: il, barrier_after: true, work });
                    let work = split_rows(d.n_embd, n).into_iter().map(|rows| Work::RecOut { rows }).collect();
                    phases.push(Phase { kind: PhaseKind::RecOut, layer: il, barrier_after: true, work });
                }
                LayerKind::Attention => {
                    let kv_rows = d.n_head_kv * d.head_dim;
                    let (ks, vs) = (split_rows(kv_rows, n), split_rows(kv_rows, n));
                    let work = (0..n)
                        .map(|t| Work::AttnQkv { qheads: qh_split[t].clone(), k_rows: ks[t].clone(), v_rows: vs[n - 1 - t].clone() })
                        .collect();
                    phases.push(Phase { kind: PhaseKind::AttnQkv, layer: il, barrier_after: true, work });
                    let work = qh_split.iter().map(|q| Work::AttnCore { qheads: q.clone() }).collect();
                    phases.push(Phase { kind: PhaseKind::AttnCore, layer: il, barrier_after: true, work });
                    let work = split_rows(d.n_embd, n).into_iter().map(|rows| Work::AttnOut { rows }).collect();
                    phases.push(Phase { kind: PhaseKind::AttnOut, layer: il, barrier_after: true, work });
                }
            }
            let sh = split_rows(d.ff_shexp, n);
            let work = split(d.n_expert, n).into_iter().zip(sh).map(|(rows, sh_rows)| Work::MoeRouter { rows, sh_rows }).collect();
            phases.push(Phase { kind: PhaseKind::MoeRouter, layer: il, barrier_after: true, work });
            let work = unit_split.iter().map(|u| Work::MoeGateUp { units: u.clone() }).collect();
            phases.push(Phase { kind: PhaseKind::MoeGateUp, layer: il, barrier_after: true, work });
            let work = split_rows(d.n_embd, n).into_iter().map(|rows| Work::MoeDown { rows }).collect();
            phases.push(Phase { kind: PhaseKind::MoeDown, layer: il, barrier_after: true, work });
        }
        let work = split_rows(d.n_vocab, n).into_iter().map(|rows| Work::Head { rows }).collect();
        // no barrier_after: the pool's join barrier ends the token
        phases.push(Phase { kind: PhaseKind::Head, layer: d.n_layer, barrier_after: false, work });
        Ok(Plan {
            desc: desc.clone(),
            nthreads,
            n_kv,
            layers,
            embd: find(h, "token_embd.weight")?,
            out_norm: find(h, "output_norm.weight")?,
            output: find(h, "output.weight")?,
            phases,
            head_chunk: 256,
        })
    }

    /// Barriers per token: one per phase with `barrier_after`, plus the pool join.
    pub fn barriers_per_token(&self) -> usize {
        self.phases.iter().filter(|p| p.barrier_after).count() + 1
    }

    /// Every tensor the plan reads during decode.
    pub fn tensors(&self) -> Vec<Tid> {
        let mut v: Vec<Tid> = self.layers.iter().flat_map(|l| l.all()).collect();
        v.extend([self.embd, self.out_norm, self.output]);
        v
    }

    /// Unique bytes per token from the tensor table (header) and the state/KV layout.
    pub fn budget(&self, tensors: &[TensorInfo]) -> Budget {
        let d = &self.desc;
        let mut b = Budget::default();
        for id in self.tensors() {
            let t = &tensors[id];
            match role(t, d) {
                Role::Dense => b.weights_dense += t.bytes(),
                Role::Expert => b.weights_expert += t.bytes() * d.n_expert_used as u64 / d.n_expert as u64,
                Role::Embedding => b.embedding_row += t.bytes() / t.shape[1],
                Role::Mtp => {}
            }
        }
        b.state_rw = 2 * (d.recurrent_state_bytes_per_layer() * d.n_recurrent()) as u64;
        b.kv_read = (d.kv_bytes_per_pos_per_layer() * d.n_attention() * self.n_kv) as u64;
        b.kv_write = (d.kv_bytes_per_pos_per_layer() * d.n_attention()) as u64;
        b
    }

    /// Check that every per-thread row/head partition covers its tensor exactly once
    /// (redundant beta/alpha groups excepted: those must cover, overlap allowed).
    pub fn check_coverage(&self) -> Result<(), String> {
        let d = &self.desc;
        fn exact(name: &str, ranges: &[Range<usize>], n: usize) -> Result<(), String> {
            let mut at = 0;
            for r in ranges {
                if r.start != at || r.end < r.start {
                    return Err(format!("{name}: gap or overlap at {at}"));
                }
                at = r.end;
            }
            if at != n {
                return Err(format!("{name}: covers {at} of {n}"));
            }
            Ok(())
        }
        for p in &self.phases {
            let name = format!("{} layer {}", p.kind.name(), p.layer);
            if p.work.len() != self.nthreads {
                return Err(format!("{name}: {} slices for {} threads", p.work.len(), self.nthreads));
            }
            let ranges = |f: &dyn Fn(&Work) -> Range<usize>| p.work.iter().map(f).collect::<Vec<_>>();
            match p.kind {
                PhaseKind::RecMix => {
                    exact(&name, &ranges(&|w| match w { Work::RecMix { kheads } => kheads.clone(), _ => 0..0 }), d.ssm_n_kheads)?
                }
                PhaseKind::RecOut => exact(&name, &ranges(&|w| match w { Work::RecOut { rows } => rows.clone(), _ => 0..0 }), d.n_embd)?,
                PhaseKind::AttnQkv => {
                    exact(&name, &ranges(&|w| match w { Work::AttnQkv { qheads, .. } => qheads.clone(), _ => 0..0 }), d.n_head)?;
                    let kv = d.n_head_kv * d.head_dim;
                    exact(&name, &ranges(&|w| match w { Work::AttnQkv { k_rows, .. } => k_rows.clone(), _ => 0..0 }), kv)?;
                    let mut v = ranges(&|w| match w { Work::AttnQkv { v_rows, .. } => v_rows.clone(), _ => 0..0 });
                    v.sort_by_key(|r| (r.start, r.end));
                    exact(&name, &v, kv)?;
                }
                PhaseKind::AttnCore => exact(&name, &ranges(&|w| match w { Work::AttnCore { qheads } => qheads.clone(), _ => 0..0 }), d.n_head)?,
                PhaseKind::AttnOut => exact(&name, &ranges(&|w| match w { Work::AttnOut { rows } => rows.clone(), _ => 0..0 }), d.n_embd)?,
                PhaseKind::MoeRouter => {
                    exact(&name, &ranges(&|w| match w { Work::MoeRouter { rows, .. } => rows.clone(), _ => 0..0 }), d.n_expert)?;
                    exact(&name, &ranges(&|w| match w { Work::MoeRouter { sh_rows, .. } => sh_rows.clone(), _ => 0..0 }), d.ff_shexp)?;
                }
                PhaseKind::MoeGateUp => {
                    let units = d.n_expert_used * (d.ff_exp / kernels_sys::QK_K);
                    exact(&name, &ranges(&|w| match w { Work::MoeGateUp { units } => units.clone(), _ => 0..0 }), units)?
                }
                PhaseKind::MoeDown => exact(&name, &ranges(&|w| match w { Work::MoeDown { rows } => rows.clone(), _ => 0..0 }), d.n_embd)?,
                PhaseKind::Head => exact(&name, &ranges(&|w| match w { Work::Head { rows } => rows.clone(), _ => 0..0 }), d.n_vocab)?,
            }
        }
        Ok(())
    }

    /// Human-readable phase table (for `m0-timing --print-plan`).
    pub fn describe(&self) -> String {
        let mut s = String::new();
        let rec = self.phases.iter().filter(|p| p.layer == 0).count();
        s += &format!(
            "plan: {} threads, {} layers ({} delta-net, {} attention), {} phases, {} barriers/token, n_kv {}\n",
            self.nthreads,
            self.desc.n_layer,
            self.desc.n_recurrent(),
            self.desc.n_attention(),
            self.phases.len(),
            self.barriers_per_token(),
            self.n_kv
        );
        let _ = rec;
        for p in self.phases.iter().filter(|p| p.layer <= 3 || p.kind == PhaseKind::Head) {
            s += &format!("  L{:<2} {:<13} {} {:?}\n", p.layer, p.kind.name(), if p.barrier_after { "B" } else { "-" }, p.work[0]);
        }
        s
    }
}
