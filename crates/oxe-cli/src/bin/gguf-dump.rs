//! gguf-dump: print a GGUF's header summary and tensor table as JSON (header-only read),
//! in the same row format as tests/fixtures (name, type, shape, offset, bytes).

fn main() {
    sched::cli::exit_zero_on_epipe(); // F-CI9: `| head` ends with status 0, not a core dump
    let args: Vec<String> = std::env::args().collect();
    let meta = args.iter().any(|a| a == "--meta");
    let path = args.iter().skip(1).find(|a| !a.starts_with("--")).cloned().unwrap_or_else(|| {
        eprintln!("usage: gguf-dump [--meta] MODEL.gguf");
        std::process::exit(2)
    });
    let g = gguf::Gguf::open(&path).unwrap_or_else(|e| {
        eprintln!("gguf-dump: {e}");
        std::process::exit(2)
    });
    let h = &g.header;
    if meta {
        for (k, v) in &h.metadata {
            if k.starts_with("tokenizer.ggml.") && !k.ends_with("_id") && !k.ends_with(".model") && !k.ends_with(".pre") {
                continue;
            }
            let s = format!("{v:?}");
            println!("{k} = {}", &s[..s.len().min(160)]);
        }
        return;
    }
    let tensors: Vec<serde_json::Value> = h
        .tensors
        .iter()
        .map(|t| serde_json::json!([t.name, t.ty.name(), t.shape, t.offset, t.bytes()]))
        .collect();
    let out = serde_json::json!({
        "file_bytes": g.file_bytes,
        "gguf_version": h.version,
        "data_offset": h.data_offset,
        "header_bytes_read": h.header_bytes_read,
        "n_metadata": h.metadata.len(),
        "architecture": h.architecture(),
        "tensor_bytes": h.tensors.iter().map(|t| t.bytes()).sum::<u64>(),
        "tensors": tensors,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
