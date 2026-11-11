//! Small helpers shared by the flash-m* drivers: argument map, oracle token records, memory probes.

use std::collections::HashMap;
use std::path::Path;

/// End-of-generation tokens of the Flash vocabulary.
pub const EOG: [i32; 2] = [248046, 248044];

pub fn die(m: &str) -> ! {
    eprintln!("{m}");
    std::process::exit(2)
}

pub struct Args(pub HashMap<String, String>);
impl Args {
    pub fn parse(v: &[String]) -> Args {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            let k = v[i].strip_prefix("--").unwrap_or_else(|| die(&format!("unexpected {}", v[i])));
            if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                m.insert(k.to_string(), v[i + 1].clone());
                i += 2;
            } else {
                m.insert(k.to_string(), "1".into());
                i += 1;
            }
        }
        Args(m)
    }
    pub fn s(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    pub fn n<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        match self.0.get(k) {
            Some(v) => v.parse().unwrap_or_else(|_| die(&format!("--{k} {v}: bad value"))),
            None => d,
        }
    }
    pub fn has(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
    /// boolean flag with a default: `--k` / `--k 1` / `--k 0`
    pub fn flag(&self, k: &str, d: bool) -> bool {
        match self.0.get(k).map(|s| s.as_str()) {
            None => d,
            Some("0") | Some("false") | Some("off") => false,
            Some(_) => true,
        }
    }
}

/// The oracle dump's token records: one per production ubatch (u0 warm-up, u1 prompt, u2.. decode).
pub fn read_tokens(p: &Path) -> Vec<Vec<i32>> {
    let b = std::fs::read(p).unwrap_or_else(|e| die(&format!("{}: {e}", p.display())));
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        assert_eq!(&b[i..i + 4], b"FGD1");
        let mut n = 1i64;
        for k in 0..4 {
            n *= i64::from_le_bytes(b[i + 8 + 8 * k..i + 16 + 8 * k].try_into().unwrap());
        }
        let n = n as usize;
        let v = (0..n).map(|j| i32::from_le_bytes(b[i + 40 + 4 * j..i + 44 + 4 * j].try_into().unwrap())).collect();
        out.push(v);
        i += 40 + 4 * n;
    }
    out
}

/// Generated token i matches production: equal to the oracle's, or the end-of-generation token right
/// after the oracle's last generated token (the oracle records only tokens fed back).
pub fn want_ok(want: &[i32], i: usize, next: i32) -> bool {
    match want.get(i) {
        Some(&w) => w == next,
        None => i == want.len() && EOG.contains(&next),
    }
}

pub fn status_kib(key: &str) -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    s.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok()
}

pub fn cgroup_file(name: &str) -> Option<String> {
    let cg = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = cg.lines().find_map(|l| l.strip_prefix("0::"))?;
    std::fs::read_to_string(format!("/sys/fs/cgroup{}/{name}", path.trim())).ok().map(|s| s.trim().to_string())
}

/// KL(p_ref || p) of two logit vectors (softmax in f64), plus whether the argmaxes agree.
pub fn kld(ref_logits: &[f32], logits: &[f32]) -> (f64, bool) {
    fn lse(v: &[f32]) -> (f64, usize) {
        let mut mx = f32::NEG_INFINITY;
        let mut am = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > mx {
                mx = x;
                am = i;
            }
        }
        let s: f64 = v.iter().map(|&x| ((x - mx) as f64).exp()).sum();
        (mx as f64 + s.ln(), am)
    }
    let (la, aa) = lse(ref_logits);
    let (lb, ab) = lse(logits);
    let mut k = 0.0f64;
    for (&a, &b) in ref_logits.iter().zip(logits) {
        let lpa = a as f64 - la;
        let lpb = b as f64 - lb;
        let pa = lpa.exp();
        if pa > 0.0 {
            k += pa * (lpa - lpb);
        }
    }
    (k.max(0.0), aa == ab)
}

/// AnonHugePages of this process in KiB (/proc/self/smaps_rollup), for the FD-5 huge-page coverage check.
pub fn anon_huge_kib() -> Option<u64> {
    let t = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    t.lines().find(|l| l.starts_with("AnonHugePages:"))?.split_whitespace().nth(1)?.parse().ok()
}
