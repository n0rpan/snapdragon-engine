use super::*;
use serde_json::Value as J;
use std::path::PathBuf;

fn fixture(name: &str) -> J {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// Model path: env override, else the laptop's model directory; None (test skipped) if absent.
fn model_path(env: &str, file: &str) -> Option<PathBuf> {
    let p = std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap()).join(".local/share/ai-stack/models/llm").join(file));
    p.is_file().then_some(p)
}

fn json_eq(ours: &Value, theirs: &J) -> bool {
    match (ours, theirs) {
        (Value::Str(a), J::String(b)) => a == b,
        (Value::Bool(a), J::Bool(b)) => a == b,
        (Value::Array(a), J::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_eq(x, y)),
        (v, J::Number(n)) => match (v.as_f64(), n.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        },
        _ => false,
    }
}

/// Header-only comparison against model_inspect.py's tensor table (reads ~11 MB of header).
fn compare_with_fixture(env: &str, key: &str) {
    let fx = fixture(key);
    let Some(path) = model_path(env, fx["file"].as_str().unwrap()) else {
        eprintln!("skip: {key} model file not present");
        return;
    };
    let g = Gguf::open(&path).unwrap();
    let h = &g.header;
    assert_eq!(g.file_bytes, fx["file_bytes"].as_u64().unwrap());
    assert_eq!(h.version as u64, fx["gguf_version"].as_u64().unwrap());
    assert_eq!(h.data_offset, fx["data_offset"].as_u64().unwrap());
    assert_eq!(h.header_bytes_read, fx["header_bytes_read"].as_u64().unwrap());
    assert_eq!(h.metadata.len() as u64, fx["n_metadata"].as_u64().unwrap());
    assert_eq!(h.architecture(), fx["architecture"].as_str());
    for (k, v) in fx["metadata"].as_object().unwrap() {
        let ours = h.get(k).unwrap_or_else(|| panic!("missing metadata {k}"));
        assert!(json_eq(ours, v), "metadata {k}: ours {ours:?} model_inspect {v}");
    }
    let tensors = fx["tensors"].as_array().unwrap();
    assert_eq!(h.tensors.len(), tensors.len());
    let mut total = 0;
    for (t, r) in h.tensors.iter().zip(tensors) {
        let r = r.as_array().unwrap();
        assert_eq!(t.name, r[0].as_str().unwrap());
        assert_eq!(t.ty.name(), r[1].as_str().unwrap(), "{}", t.name);
        let shape: Vec<u64> = r[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(t.shape, shape, "{}", t.name);
        assert_eq!(t.offset, r[3].as_u64().unwrap(), "{}", t.name);
        assert_eq!(t.bytes(), r[4].as_u64().unwrap(), "{}", t.name);
        total += t.bytes();
    }
    assert_eq!(total, fx["tensor_bytes"].as_u64().unwrap());
    eprintln!("{key}: {} tensors, {} metadata keys match model_inspect.py", h.tensors.len(), h.metadata.len());
}

#[test]
fn ornith_header_matches_model_inspect() {
    compare_with_fixture("OXE_MODEL_ORNITH", "ornith");
}

#[test]
fn qwen36_header_matches_model_inspect() {
    compare_with_fixture("OXE_MODEL_QWEN36", "qwen36");
}

// ---------------------------------------------------------------- synthetic round trip

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

#[test]
fn parses_synthetic_gguf() {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(2u64.to_le_bytes()); // tensors
    b.extend(5u64.to_le_bytes()); // kv
    put_str(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    put_str(&mut b, "qwen35moe");
    put_str(&mut b, "general.alignment");
    b.extend(4u32.to_le_bytes());
    b.extend(64u32.to_le_bytes());
    put_str(&mut b, "x.eps");
    b.extend(6u32.to_le_bytes());
    b.extend(1e-6f32.to_le_bytes());
    put_str(&mut b, "x.sections");
    b.extend(9u32.to_le_bytes());
    b.extend(5u32.to_le_bytes());
    b.extend(3u64.to_le_bytes());
    for v in [11i32, 11, 10] {
        b.extend(v.to_le_bytes());
    }
    put_str(&mut b, "x.long");
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(5000u64.to_le_bytes());
    for i in 0..5000 {
        put_str(&mut b, &format!("t{i}"));
    }
    // tensor infos
    put_str(&mut b, "a.weight");
    b.extend(2u32.to_le_bytes());
    b.extend(256u64.to_le_bytes());
    b.extend(8u64.to_le_bytes());
    b.extend(12u32.to_le_bytes()); // Q4_K
    b.extend(0u64.to_le_bytes());
    put_str(&mut b, "b.norm");
    b.extend(1u32.to_le_bytes());
    b.extend(256u64.to_le_bytes());
    b.extend(0u32.to_le_bytes()); // F32
    b.extend(1152u64.to_le_bytes());
    let header_end = b.len() as u64;

    let h = parse_header(&b).unwrap();
    assert_eq!(h.version, 3);
    assert_eq!(h.architecture(), Some("qwen35moe"));
    assert_eq!(h.get("x.eps"), Some(&Value::F32(1e-6)));
    assert_eq!(h.get("x.sections"), Some(&Value::Array(vec![Value::I32(11), Value::I32(11), Value::I32(10)])));
    assert_eq!(h.get("x.long"), Some(&Value::LongArray { item_type: 8, len: 5000 }));
    assert_eq!(h.header_bytes_read, header_end);
    assert_eq!(h.data_offset, header_end.div_ceil(64) * 64);
    assert_eq!(h.tensors[0].ty, GgmlType::Q4_K);
    assert_eq!(h.tensors[0].bytes(), 8 * 144);
    assert_eq!(h.tensors[1].bytes(), 1024);
    assert_eq!(h.tensors[1].offset, 1152);

    // truncation is an error, not a panic
    assert!(parse_header(&b[..b.len() - 3]).is_err());
    let mut bad = b.clone();
    bad[0] = b'X';
    assert!(parse_header(&bad).is_err());
}

#[test]
fn type_table_round_trips_names() {
    for name in ["F32", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"] {
        assert_eq!(GgmlType::from_name(name).unwrap().name(), name);
    }
}

// ---------------------------------------------------------------- REVIEW-v0.1.1 S1: crafted headers

/// A GGUF image with one kv (`general.alignment` if given) and one tensor `t` of `shape`,
/// type `ty`, data offset `offset`, followed by `data` bytes of tensor data.
fn crafted(n_tensors: u64, align: Option<u64>, shape: &[u64], ty: u32, offset: u64, data: usize) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(n_tensors.to_le_bytes());
    b.extend((align.is_some() as u64).to_le_bytes());
    if let Some(a) = align {
        put_str(&mut b, "general.alignment");
        b.extend(10u32.to_le_bytes()); // u64
        b.extend(a.to_le_bytes());
    }
    put_str(&mut b, "t");
    b.extend((shape.len() as u32).to_le_bytes());
    for d in shape {
        b.extend(d.to_le_bytes());
    }
    b.extend(ty.to_le_bytes());
    b.extend(offset.to_le_bytes());
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend(std::iter::repeat(0u8).take(data));
    b
}

fn tmp_file(tag: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gguf-s1-{tag}-{}.gguf", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn crafted_valid_header_still_opens() {
    let b = crafted(1, None, &[64, 2], 0, 0, 512); // F32 64x2 = 512 bytes
    let p = tmp_file("ok", &b);
    let g = Gguf::open(&p);
    std::fs::remove_file(&p).ok();
    let g = g.unwrap();
    assert_eq!(g.header.tensors[0].bytes(), 512);
    // one byte short: data beyond end of file
    let p = tmp_file("short", &b[..b.len() - 1]);
    let r = Gguf::open(&p);
    std::fs::remove_file(&p).ok();
    assert!(r.is_err());
}

#[test]
fn shape_product_overflow_is_an_error() {
    // 2^33 * 2^33 = 2^66 wrapped to 0 elements (0 bytes) and passed every check
    let b = crafted(1, None, &[1 << 33, 1 << 33], 0, 0, 0);
    assert!(parse_header(&b).is_err());
    // F32 bytes overflow without an element overflow: 2^62 elements * 4 bytes
    let b = crafted(1, None, &[1 << 31, 1 << 31], 0, 0, 0);
    assert!(parse_header(&b).is_err());
    // a dimension that is negative as ggml's int64
    let b = crafted(1, None, &[1u64 << 63, 0], 0, 0, 0);
    assert!(parse_header(&b).is_err());
}

#[test]
fn offset_near_u64_max_is_out_of_bounds() {
    // data_offset + offset + bytes wrapped to a small number and passed the end-of-file check
    let b = crafted(1, None, &[64], 0, u64::MAX - 8, 256);
    let p = tmp_file("off", &b);
    let r = Gguf::open(&p);
    std::fs::remove_file(&p).ok();
    assert!(r.is_err(), "offset u64::MAX - 8 must be refused");
    let h = parse_header(&b).unwrap();
    assert!(check_bounds(&h, b.len() as u64).is_err());
}

#[test]
fn huge_counts_do_not_allocate_or_panic() {
    // n_tensors = 2^60: Vec::with_capacity overflowed (panic / abort)
    let b = crafted(1 << 60, None, &[64], 0, 0, 256);
    assert!(parse_header(&b).is_err());
    // n_kv = 2^60
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend((1u64 << 60).to_le_bytes());
    b.extend([0u8; 64]);
    assert!(parse_header(&b).is_err());
    // a long scalar array whose byte size overflows usize
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    put_str(&mut b, "x");
    b.extend(9u32.to_le_bytes());
    b.extend(10u32.to_le_bytes()); // u64 items
    b.extend((u64::MAX / 4).to_le_bytes());
    assert!(parse_header(&b).is_err());
    // 2^31 dimensions
    let mut b = crafted(1, None, &[64], 0, 0, 0);
    let nd_at = 4 + 4 + 8 + 8 + 8 + 1;
    b[nd_at..nd_at + 4].copy_from_slice(&(1u32 << 31).to_le_bytes());
    assert!(parse_header(&b).is_err());
}

#[test]
fn zero_or_odd_alignment_is_an_error() {
    // general.alignment = 0 divided by zero
    assert!(parse_header(&crafted(1, Some(0), &[64], 0, 0, 256)).is_err());
    assert!(parse_header(&crafted(1, Some(48), &[64], 0, 0, 256)).is_err());
    assert!(parse_header(&crafted(1, Some(64), &[64], 0, 0, 256)).is_ok());
}

/// Hand-built headers (fixtures, other crates) that overflow get u64::MAX sizes, which no
/// bounds check accepts.
#[test]
fn accessors_saturate_on_overflow() {
    let t = TensorInfo { name: "x".into(), shape: vec![1 << 40, 1 << 40], ty: GgmlType(0), offset: 0 };
    assert_eq!(t.checked_elements(), None);
    assert_eq!(t.elements(), u64::MAX);
    assert_eq!(t.bytes(), u64::MAX);
    let h = Header { tensors: vec![t], ..Default::default() };
    assert!(check_bounds(&h, u64::MAX).is_err());
}

/// Fuzz-style: random byte flips and truncations of a valid image never panic, and whatever
/// parses passes or fails the bounds check without overflow.
#[test]
fn mutated_headers_never_panic() {
    let base = crafted(1, Some(32), &[64, 2], 0, 0, 512);
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let mut parsed = 0;
    for i in 0..20_000 {
        let mut b = base.clone();
        for _ in 0..1 + rnd() % 4 {
            let at = (rnd() as usize) % b.len();
            b[at] = match rnd() % 3 {
                0 => rnd() as u8,
                1 => 0xff,
                _ => 0,
            };
        }
        if i % 7 == 0 {
            let n = (rnd() as usize) % b.len();
            b.truncate(n);
        }
        if let Ok(h) = parse_header(&b) {
            parsed += 1;
            let _ = check_bounds(&h, b.len() as u64);
            for t in &h.tensors {
                let _ = (t.elements(), t.bytes());
            }
        }
    }
    assert!(parsed > 0);
}
