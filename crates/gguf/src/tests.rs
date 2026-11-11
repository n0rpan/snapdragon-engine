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
