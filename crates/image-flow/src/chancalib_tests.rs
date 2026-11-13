use super::*;

fn rows(n: usize, h: usize, big: &[(usize, f32)], seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut x = vec![0f32; n * h];
    for t in 0..n {
        for k in 0..h {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
            let g = big.iter().find(|(c, _)| *c == k).map_or(1.0, |(_, g)| *g);
            x[t * h + k] = u * g;
        }
    }
    x
}

fn accum(x: &[f32], n: usize, h: usize) -> ChanAccum {
    let mut a = ChanAccum::new(h);
    a.add(x, n).unwrap();
    a
}

#[test]
fn no_outliers_keeps_the_natural_order() {
    let (n, h) = (96, 512);
    let cal = rows(n, h, &[], 1);
    let hold = rows(n, h, &[], 2);
    let p = plan_operand("single.0.lin1_in", &accum(&cal, n, h), &hold, n, 256, 0.10).unwrap();
    assert_eq!(p.arrange, "natural");
    assert!(p.meets_target && p.perm.is_none() && p.scale.is_none());
}

#[test]
fn spread_outliers_are_fixed_on_held_out_rows() {
    // one 60x channel in each 256-chunk: natural chunk 256 is far worse than chunk 32
    let (n, h) = (128, 1024);
    let big = [(5, 60.0), (300, 60.0), (700, 60.0), (900, 60.0)];
    let cal = rows(n, h, &big, 11);
    let hold = rows(n, h, &big, 12);
    let p = plan_operand("single.3.lin1_in", &accum(&cal, n, h), &hold, n, 256, 0.25).unwrap();
    let e = &p.errors;
    let (target, nat) = (e["target_natural_chunk32"].as_f64().unwrap(), e["natural"].as_f64().unwrap());
    assert!(nat > 2.0 * target, "natural {nat} vs target {target}");
    assert_ne!(p.arrange, "natural");
    // whichever fix is chosen, its held-out error is below natural
    let chosen = match p.arrange {
        "sorted" => e["sorted"].as_f64().unwrap(),
        _ => e["equalized"].as_array().unwrap().iter().find(|v| v["alpha"].as_f64().unwrap() as f32 == p.alpha.unwrap()).unwrap()["error"].as_f64().unwrap(),
    };
    assert!(chosen < nat, "chosen {chosen} natural {nat}");
}

#[test]
fn sorting_groups_outliers_into_one_chunk() {
    let mag: Vec<f32> = (0..64).map(|k| if k == 3 || k == 40 { 100.0 } else { 1.0 + k as f32 * 1e-3 }).collect();
    let p = sorted_perm(&mag);
    assert_eq!(&p[62..], &[3, 40]);
    let mut q = p.clone();
    q.sort();
    assert_eq!(q, (0..64).collect::<Vec<u32>>());
    // ties are broken by index: deterministic
    assert_eq!(sorted_perm(&[1.0, 1.0, 0.5]), vec![2, 0, 1]);
}

#[test]
fn eq_scale_has_unit_geometric_mean_and_handles_zero() {
    let mag = [4.0f32, 1.0, 0.25, 0.0];
    let s = eq_scale(&mag, 1.0);
    assert!((s[0] * s[1] * s[2] - 1.0).abs() < 1e-5);
    assert_eq!(s[3], 1.0);
    let s0 = eq_scale(&mag, 0.0);
    assert!(s0.iter().all(|v| (*v - 1.0).abs() < 1e-6));
}

#[test]
fn quant_error_with_identity_matches_actstats() {
    let (n, h) = (40, 256);
    let x = rows(n, h, &[(7, 30.0)], 5);
    let mag = crate::actstats::channel_rms(&x, n, h);
    let id: Vec<u32> = (0..h as u32).collect();
    let ones = vec![1f32; h];
    for chunk in [32, 64, 256] {
        let a = crate::actstats::quant_error(&x, n, h, &mag, chunk, crate::actstats::Arrange::Natural);
        let b = quant_error_with(&x, n, h, &id, &ones, &mag, chunk);
        assert_eq!(a.0.to_bits(), b.0.to_bits());
        assert_eq!(a.1.to_bits(), b.1.to_bits());
    }
}

#[test]
fn record_round_trips_and_validates_transforms() {
    let (n, h) = (64, 512);
    let big = [(1, 50.0), (260, 50.0)];
    let cal = rows(n, h, &big, 21);
    let hold = rows(n, h, &big, 22);
    let p = plan_operand("single.1.lin1_in", &accum(&cal, n, h), &hold, n, 256, 0.25).unwrap();
    let rec = record(256, 0.25, std::slice::from_ref(&p), &["ab".repeat(32)]);
    let text = serde_json::to_string(&rec).unwrap();
    let back: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(back["schema"], SCHEMA);
    let (hh, perm, scale) = operand_transform(&back["operands"][0]).unwrap();
    assert_eq!(hh, h);
    assert_eq!(perm.is_some(), p.perm.is_some());
    assert_eq!(scale.is_some(), p.scale.is_some());
    // a broken permutation is refused
    let mut bad = back["operands"][0].clone();
    bad["perm"] = json!((0..h).map(|_| 0).collect::<Vec<u32>>());
    assert!(operand_transform(&bad).is_err());
    let mut bad = back["operands"][0].clone();
    bad["perm"] = Value::Null;
    bad["scale"] = json!(vec![0.0f32; h]);
    assert!(operand_transform(&bad).is_err());
}

#[test]
fn bad_inputs_are_errors() {
    let mut a = ChanAccum::new(64);
    assert!(a.add(&[1.0; 63], 1).is_err());
    assert!(a.add(&[f32::NAN; 64], 1).is_err());
    a.add(&[1.0; 64], 1).unwrap();
    assert!(plan_operand("x", &a, &[1.0; 64], 1, 48, 0.1).is_err());
    assert!(plan_operand("x", &a, &[], 0, 32, 0.1).is_err());
}
