//! Replayed routing: the Codex P2 expert-ID trace (`turn,phase,step,layer,slot,expert` CSV,
//! phase 0 = prefill batch, phase 1 = one decode token per step).

use crate::model::{N_EXPERT, N_LAYER, TOP_K};
use std::path::Path;

pub struct Trace {
    /// per layer: the distinct experts the prefill batch used, in first-use order
    pub prefill: Vec<Vec<u16>>,
    /// per decode token, per layer: the 10 routed experts in slot order
    pub decode: Vec<[[u16; TOP_K]; N_LAYER]>,
    pub records: usize,
}

impl Trace {
    pub fn load(path: &Path) -> Result<Trace, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Trace::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Trace, String> {
        let mut lines = text.lines();
        let head = lines.next().ok_or("empty trace")?;
        if head.trim() != "turn,phase,step,layer,slot,expert" {
            return Err(format!("unexpected header {head:?}"));
        }
        let mut prefill: Vec<Vec<u16>> = vec![Vec::new(); N_LAYER];
        let mut seen = vec![[false; N_EXPERT]; N_LAYER];
        let mut dec: Vec<(u32, [[u16; TOP_K]; N_LAYER], [[bool; TOP_K]; N_LAYER])> = Vec::new();
        let mut records = 0;
        for (i, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let v: Vec<u32> = line
                .split(',')
                .map(|x| x.trim().parse::<u32>().map_err(|e| format!("line {}: {e}", i + 2)))
                .collect::<Result<_, _>>()?;
            if v.len() != 6 {
                return Err(format!("line {}: {} fields", i + 2, v.len()));
            }
            let (phase, step, layer, slot, e) = (v[1], v[2], v[3] as usize, v[4] as usize, v[5] as usize);
            if layer >= N_LAYER || slot >= TOP_K || e >= N_EXPERT {
                return Err(format!("line {}: out of range", i + 2));
            }
            records += 1;
            match phase {
                0 => {
                    if !seen[layer][e] {
                        seen[layer][e] = true;
                        prefill[layer].push(e as u16);
                    }
                }
                1 => {
                    let idx = match dec.iter().position(|d| d.0 == step) {
                        Some(i) => i,
                        None => {
                            dec.push((step, [[0; TOP_K]; N_LAYER], [[false; TOP_K]; N_LAYER]));
                            dec.len() - 1
                        }
                    };
                    let d = &mut dec[idx];
                    if d.2[layer][slot] {
                        return Err(format!("line {}: duplicate (step, layer, slot)", i + 2));
                    }
                    d.1[layer][slot] = e as u16;
                    d.2[layer][slot] = true;
                }
                p => return Err(format!("line {}: phase {p}", i + 2)),
            }
        }
        dec.sort_by_key(|d| d.0);
        for d in &dec {
            if d.2.iter().any(|l| l.iter().any(|&b| !b)) {
                return Err(format!("decode step {} incomplete", d.0));
            }
        }
        Ok(Trace { prefill, decode: dec.into_iter().map(|d| d.1).collect(), records })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_prefill_and_decode() {
        let mut s = String::from("turn,phase,step,layer,slot,expert\n");
        for l in 0..N_LAYER {
            for k in 0..TOP_K {
                s += &format!("0,0,0,{l},{k},{}\n", k * 3);
                s += &format!("0,0,1,{l},{k},{}\n", k * 3 + 1);
            }
        }
        for step in [7, 5] {
            for l in 0..N_LAYER {
                for k in 0..TOP_K {
                    s += &format!("0,1,{step},{l},{k},{}\n", (l + k + step) % N_EXPERT);
                }
            }
        }
        let t = Trace::parse(&s).unwrap();
        assert_eq!(t.prefill[3].len(), 20);
        assert_eq!(t.decode.len(), 2);
        assert_eq!(t.decode[0][2][1], (2 + 1 + 5) as u16); // sorted by step
        assert_eq!(t.records, 2 * N_LAYER * TOP_K * 2);
    }
    #[test]
    fn real_trace_if_present() {
        let p = std::env::var_os("OXF_EXPERT_TRACE").map(std::path::PathBuf::from).unwrap_or_else(|| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join(".cache/local-ai/snapdragon-engine/flash/flash-fixed353-expert-ids.csv")
        });
        if !p.exists() {
            return;
        }
        let t = Trace::load(&p).unwrap();
        assert_eq!(t.records, 242770);
        assert_eq!(t.decode.len(), 353);
        for tok in &t.decode {
            for l in tok {
                let mut v = l.to_vec();
                v.sort();
                v.dedup();
                assert_eq!(v.len(), TOP_K);
            }
        }
    }
}
