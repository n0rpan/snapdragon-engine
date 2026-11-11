//! How redundant is the text context? klein's conditioner pads every prompt to 512 tokens, so at
//! 512 px a third of the DiT's tokens are text and most of those are padding. The pad positions are
//! not identical by construction (each pad token passes through the text encoder at its own
//! position), but they may be nearly identical. If they are, runs of them can be merged into one
//! representative token with a multiplicity (as a key it counts `ln(count)` more in every
//! attention; as a query and in the linears it is computed once): fewer tokens for the same image
//! up to the merge tolerance. This module only MEASURES that on a dumped context; nothing uses it
//! for generation yet.

/// Greedy runs of consecutive rows: a run grows while each new row is within `tol` (relative L2
/// distance to the run's FIRST row, relative to that row's norm). Returns the run lengths.
pub fn greedy_runs(x: &[f32], n: usize, dim: usize, tol: f64) -> Vec<usize> {
    assert_eq!(x.len(), n * dim);
    let mut runs = Vec::new();
    let mut start = 0usize;
    let norm = |i: usize| -> f64 { x[i * dim..(i + 1) * dim].iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() };
    let dist = |i: usize, j: usize| -> f64 {
        x[i * dim..(i + 1) * dim].iter().zip(&x[j * dim..(j + 1) * dim]).map(|(a, b)| ((*a - *b) as f64).powi(2)).sum::<f64>().sqrt()
    };
    for i in 1..=n {
        let split = i == n || {
            let nf = norm(start);
            if nf == 0.0 {
                norm(i) != 0.0
            } else {
                dist(i, start) / nf > tol
            }
        };
        if split {
            runs.push(i - start);
            start = i;
        }
    }
    runs
}

/// Summary for one tolerance.
#[derive(Clone, Debug)]
pub struct MergeEstimate {
    pub tol: f64,
    /// Tokens left after merging runs.
    pub n_eff: usize,
    pub longest_run: usize,
    /// Linear MACs of the DiT relative to no merging (tokens ratio).
    pub linear_ratio: f64,
    /// Attention MACs relative to no merging (tokens ratio squared).
    pub attention_ratio: f64,
}

/// Merge estimates of a text context `[n_txt][dim]` next to `n_img` image tokens, for each tolerance.
pub fn estimate(txt: &[f32], n_txt: usize, dim: usize, n_img: usize, tols: &[f64]) -> Vec<MergeEstimate> {
    tols.iter()
        .map(|&tol| {
            let runs = greedy_runs(txt, n_txt, dim, tol);
            let n_eff = runs.len();
            let r = (n_eff + n_img) as f64 / (n_txt + n_img) as f64;
            MergeEstimate { tol, n_eff, longest_run: runs.iter().copied().max().unwrap_or(0), linear_ratio: r, attention_ratio: r * r }
        })
        .collect()
}

/// Largest absolute difference between any text token's RoPE row and the first one's
/// (`pe` is `[n_tokens][pe_row]`, text tokens first). 0 means all text tokens share one position.
pub fn txt_pe_spread(pe: &[f32], n_txt: usize, pe_row: usize) -> f32 {
    let first = &pe[..pe_row];
    let mut m = 0f32;
    for t in 1..n_txt {
        for (a, b) in pe[t * pe_row..(t + 1) * pe_row].iter().zip(first) {
            m = m.max((a - b).abs());
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_split_where_rows_change() {
        // rows: a a a b b c (dim 2)
        let x = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 2.0, 3.0, 3.0];
        assert_eq!(greedy_runs(&x, 6, 2, 1e-6), vec![3, 2, 1]);
        // a tolerance that swallows everything
        assert_eq!(greedy_runs(&x, 6, 2, 100.0), vec![6]);
        // nearly equal rows merge only above their distance
        let y = [1.0f32, 0.0, 1.0, 0.01, 1.0, 0.03];
        assert_eq!(greedy_runs(&y, 3, 2, 0.02), vec![2, 1]);
        assert_eq!(greedy_runs(&y, 3, 2, 0.05), vec![3]);
        // zero rows only merge with zero rows
        let z = [0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(greedy_runs(&z, 3, 2, 0.5), vec![2, 1]);
        assert_eq!(greedy_runs(&[], 0, 2, 0.1), Vec::<usize>::new());
    }

    #[test]
    fn estimate_ratios() {
        let x = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0];
        let e = estimate(&x, 4, 2, 4, &[1e-6]);
        assert_eq!(e[0].n_eff, 2);
        assert_eq!(e[0].longest_run, 3);
        assert!((e[0].linear_ratio - 0.75).abs() < 1e-12);
        assert!((e[0].attention_ratio - 0.5625).abs() < 1e-12);
    }

    #[test]
    fn pe_spread() {
        let pe = [1.0f32, 0.0, 1.0, 0.0, 0.5, 0.0, 9.0, 9.0];
        assert_eq!(txt_pe_spread(&pe, 2, 2), 0.0);
        assert_eq!(txt_pe_spread(&pe, 3, 2), 0.5);
    }
}
