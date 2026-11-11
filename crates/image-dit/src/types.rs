//! Shapes and plain-f32 data of one single-stream block (FLUX.2 klein and relatives).
//!
//! Block math (sd.cpp `Flux::SingleStreamBlock::forward`, klein settings: no biases, shared
//! modulation, SiLU-gated MLP, QK RMS norm, interleaved RoPE):
//!
//! ```text
//! x_mod  = LayerNorm(x) * (1 + scale) + shift            LN over hidden, no affine, eps 1e-6
//! y      = W1 x_mod                                      W1: [3*hidden + 2*mlp] x hidden
//! q,k,v  = y[0..h], y[h..2h], y[2h..3h]                  each [heads][head_dim]
//! q      = rope(rms_norm(q) * q_scale), k likewise       rms over head_dim, eps 1e-6
//! attn   = softmax(q k^T / sqrt(head_dim)) v             per head, over all tokens
//! m      = silu(y[3h + j]) * y[3h + mlp + j]             j in 0..mlp   (ggml_swiglu: first half is the gate)
//! z      = W2 concat(attn, m)                            W2: hidden x [hidden + mlp]
//! out    = x + gate * z
//! ```

/// Block shapes. For klein 4B: hidden 3072, heads 24, head_dim 128, mlp 9216.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub mlp: usize,
}

impl Dims {
    pub const KLEIN_4B: Dims = Dims { hidden: 3072, heads: 24, head_dim: 128, mlp: 9216 };

    pub fn check(&self) {
        assert_eq!(self.heads * self.head_dim, self.hidden);
        assert!(self.head_dim % 2 == 0);
    }
    /// Output width of linear1.
    pub fn lin1_out(&self) -> usize {
        3 * self.hidden + 2 * self.mlp
    }
    /// Input width of linear2.
    pub fn lin2_in(&self) -> usize {
        self.hidden + self.mlp
    }
}

/// The block's weights as f32 (the reference path and the source for packing in tests).
/// `w1[r * hidden + i]`, `w2[r * lin2_in + i]` (row = output channel).
pub struct BlockWeightsF32 {
    pub dims: Dims,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    /// QK-norm scales, `[head_dim]` each (shared by all heads).
    pub q_scale: Vec<f32>,
    pub k_scale: Vec<f32>,
}

/// One forward's inputs. Token order = sd.cpp's (text tokens first, then image tokens).
pub struct BlockInput {
    pub n_tokens: usize,
    /// `[n_tokens][hidden]`
    pub x: Vec<f32>,
    /// RoPE table `[n_tokens][head_dim/2][2][2]` = per (token, pair) `[[cos, -sin], [sin, cos]]`.
    pub pe: Vec<f32>,
    /// Modulation, `[hidden]` each.
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
    pub gate: Vec<f32>,
}

/// Outputs with the intermediates needed to attribute error to a stage.
#[derive(Default)]
pub struct BlockOutput {
    /// `[n_tokens][hidden]`: `x + gate * z`
    pub out: Vec<f32>,
    /// `[n_tokens][hidden]`: attention output (before linear2)
    pub attn: Vec<f32>,
    /// `[n_tokens][mlp]`: gated MLP activations (before linear2)
    pub mlp: Vec<f32>,
    /// `[n_tokens][hidden]`: `z` (linear2 output, before gate and residual)
    pub z: Vec<f32>,
}

/// Relative RMSE of `got` against `want`: `sqrt(sum (g-w)^2 / sum w^2)`, in f64.
pub fn rel_rmse(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut e, mut s) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let d = *g as f64 - *w as f64;
        e += d * d;
        s += *w as f64 * *w as f64;
    }
    if s == 0.0 {
        return if e == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (e / s).sqrt()
}
