//! `qwen3-te`: the klein text encoder on its own (timing and raw output; the comparison against an
//! sd.cpp hand-over is `oxi te-check`).
//!
//! ```text
//! qwen3-te tokens --gguf TE.gguf --prompt-file P
//! qwen3-te encode --gguf TE.gguf --prompt-file P [--out CTX.f32] [--ref CTX.f32] [--threads 12]
//!      [--te-backend cpu-f32|cpu-q8|npu] [--te-npu emu|device] [--te-npu-weights fp16|w8]
//!      [--te-npu-sessions 3] [--te-npu-budget-mib 6144] [--te-npu-kc 0] [--te-io-threads 8] [--json FILE]
//! ```
//! The prompt is read from a file (never argv). Exit code 3: the prompt needs the reference
//! encoder (emphasis/escape syntax, BREAK, special-token strings). Summary JSON on stdout.

use qwen3_te::{backend_from_opts, encode_prompt, metrics, read_f32, write_f32, TeError};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

fn main() {
    match real_main() {
        Ok(()) => {}
        Err(TeError::NeedsReference(s)) => {
            eprintln!("qwen3-te: needs the reference encoder: {s}");
            std::process::exit(3);
        }
        Err(TeError::Other(e)) => {
            eprintln!("qwen3-te: {e}");
            std::process::exit(1);
        }
    }
}

fn parse(args: &[String]) -> Result<HashMap<String, String>, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let k = args[i].strip_prefix("--").ok_or_else(|| format!("unexpected argument '{}'", args[i]))?;
        let v = args.get(i + 1).ok_or_else(|| format!("--{k} needs a value"))?;
        m.insert(k.to_string(), v.clone());
        i += 2;
    }
    Ok(m)
}

fn real_main() -> Result<(), TeError> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = argv.split_first().ok_or_else(|| TeError::Other("usage: qwen3-te tokens|encode --gguf F --prompt-file P ...".into()))?;
    let a = parse(rest)?;
    let req = |k: &str| a.get(k).cloned().ok_or_else(|| TeError::Other(format!("--{k} is required")));
    let gguf = req("gguf")?;
    let prompt = std::fs::read_to_string(req("prompt-file")?).map_err(|e| TeError::Other(format!("prompt file: {e}")))?;
    match cmd.as_str() {
        "tokens" => {
            let tok = tokenizer::Tokenizer::from_gguf(&gguf).map_err(|e| TeError::Other(e.0))?;
            let spec = qwen3_te::config::KleinSpec::klein();
            let e = qwen3_te::prompt::klein_tokens(&tok, &spec, &prompt).map_err(|e| match e {
                qwen3_te::prompt::PromptError::NeedsReference(s) => TeError::NeedsReference(s),
                o => TeError::Other(o.to_string()),
            })?;
            println!("{}", json!({"n": e.tokens.len(), "n_real": e.n_real, "ids": &e.tokens[..e.n_real]}));
            Ok(())
        }
        "encode" => {
            let threads: usize = a.get("threads").map_or(Ok(12), |v| v.parse()).map_err(|_| TeError::Other("--threads".into()))?;
            let backend = backend_from_opts(&|k| a.get(k).cloned())?;
            let out = encode_prompt(Path::new(&gguf), &prompt, backend, threads)?;
            let mut s = out.summary.clone();
            if let Some(p) = a.get("out") {
                write_f32(Path::new(p), &out.ctx)?;
            }
            if let Some(p) = a.get("ref") {
                let r = read_f32(Path::new(p))?;
                let spec = qwen3_te::config::KleinSpec::klein();
                let rep = metrics::report(&out.ctx, &r, out.n, out.n_real, out.ctx_dim / spec.out_layers.len(), &spec.out_layers)?;
                s["compare"] = rep;
            }
            let text = serde_json::to_string_pretty(&s).unwrap();
            if let Some(p) = a.get("json") {
                std::fs::write(p, &text).map_err(|e| TeError::Other(format!("{p}: {e}")))?;
            }
            println!("{text}");
            Ok(())
        }
        o => Err(TeError::Other(format!("unknown command '{o}' (tokens|encode)"))),
    }
}
