use super::*;
use std::path::PathBuf;

/// Model GGUF for the vocab: $OXE_TOKENIZER_GGUF, else the production Ornith model.
pub(crate) fn model_path() -> Option<PathBuf> {
    let p = std::env::var_os("OXE_TOKENIZER_GGUF").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".local/share/ai-stack/models/llm/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf")
    });
    p.exists().then_some(p)
}

fn llama_src() -> PathBuf {
    std::env::var_os("LLAMA_SRC").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/ai-stack/src/llama.cpp")
    })
}

fn load() -> Option<Tokenizer> {
    let Some(p) = model_path() else {
        eprintln!("skip: no model gguf");
        return None;
    };
    Some(Tokenizer::from_gguf(p).expect("load tokenizer"))
}

#[test]
fn split_qwen35_basic() {
    let c = |s: &str| s.chars().map(|c| c as u32).collect::<Vec<_>>();
    let words = |s: &str, m: bool| {
        let cp = c(s);
        let mut st = 0;
        unicode::split_qwen(&cp, m)
            .into_iter()
            .map(|l| {
                let w: String = cp[st..st + l].iter().map(|&x| char::from_u32(x).unwrap()).collect();
                st += l;
                w
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(words("Hello world", true), ["Hello", " world"]);
    assert_eq!(words("I'm 123  x\n\n", true), ["I", "'m", " ", "1", "2", "3", " ", " x", "\n\n"]);
    // combining mark joins the letter run only with marks=true
    assert_eq!(words("e\u{301}x", true), ["e\u{301}x"]);
    assert_eq!(words("e\u{301}x", false), ["e", "\u{301}x"]);
}

#[test]
fn byte_map_roundtrip() {
    for b in 0..=255u8 {
        let c = unicode::byte_to_cpt_table()[b as usize];
        assert_eq!(unicode::cpt_to_byte(c), Some(b));
    }
    assert_eq!(unicode::byte_to_cpt_table()[b' ' as usize], 'Ġ');
    assert_eq!(unicode::byte_to_cpt_table()[b'\n' as usize], 'Ċ');
}

#[test]
fn utf8_prefix() {
    assert_eq!(utf8_valid_prefix(b"abc"), 3);
    assert_eq!(utf8_valid_prefix(&[b'a', 0xF0, 0x9F]), 1);
    assert_eq!(utf8_valid_prefix(&[0xF0, 0x9F, 0xA6, 0x99]), 4);
    let mut d = StreamDecoder::default();
    let llama = "🦙".as_bytes();
    assert_eq!(d.push(&llama[..2]), "");
    assert_eq!(d.push(&llama[2..]), "🦙");
}

/// llama.cpp's own vocab test vectors (models/ggml-vocab-qwen35.gguf{,.inp,.out}).
#[test]
fn llama_vocab_test_vectors_qwen35() {
    let dir = llama_src().join("models");
    let g = dir.join("ggml-vocab-qwen35.gguf");
    if !g.exists() {
        eprintln!("skip: {g:?} missing");
        return;
    }
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let inp = std::fs::read_to_string(dir.join("ggml-vocab-qwen35.gguf.inp")).unwrap();
    let out = std::fs::read_to_string(dir.join("ggml-vocab-qwen35.gguf.out")).unwrap();
    let mut texts: Vec<&str> = inp.split("\n__ggml_vocab_test__\n").collect();
    if texts.last() == Some(&"") {
        texts.pop(); // like read_tests(): a trailing separator adds no case
    }
    let ids: Vec<&str> = out.lines().collect();
    assert_eq!(texts.len(), ids.len());
    let mut bad = 0;
    for (t, o) in texts.iter().zip(ids) {
        let want: Vec<u32> = o.split_whitespace().map(|x| x.parse().unwrap()).collect();
        let got = tok.encode(t, false, false);
        if got != want {
            bad += 1;
            eprintln!("MISMATCH {t:?}\n  want {want:?}\n  got  {got:?}");
        }
        // detokenize round trip
        assert_eq!(tok.decode(&got, false), *t);
    }
    assert_eq!(bad, 0);
}

/// Differential fixture produced by llama.cpp (crates/server/oracle, `common_tokenize`) on the
/// Ornith GGUF: add_special+parse_special and plain.
#[test]
fn llama_oracle_fixture_ornith() {
    let Some(tok) = load() else { return };
    let fx = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen35_llama_tokenize.jsonl")).unwrap();
    let mut n = 0;
    let mut bad = 0;
    for line in fx.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let text = v["text"].as_str().unwrap();
        let ids = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect::<Vec<_>>();
        for (k, sp) in [("special", true), ("plain", false)] {
            let want = ids(k);
            let got = tok.encode(text, sp, sp);
            n += 1;
            if got != want {
                bad += 1;
                eprintln!("MISMATCH ({k}) {:?}\n  want {want:?}\n  got  {got:?}", &text[..text.len().min(80)]);
            } else {
                assert_eq!(tok.decode(&got, sp), text, "decode roundtrip");
            }
        }
    }
    eprintln!("{n} cases, {bad} mismatches");
    assert_eq!(bad, 0);
}

#[test]
fn special_tokens_ornith() {
    let Some(tok) = load() else { return };
    let im_end = tok.token_id("<|im_end|>").unwrap();
    assert!(tok.is_eog(im_end));
    assert_eq!(tok.eos, Some(248046));
    assert!(!tok.add_bos);
    assert_eq!(tok.encode("<|im_end|>", false, true), vec![im_end]);
    assert_ne!(tok.encode("<|im_end|>", false, false), vec![im_end]);
    // control tokens render only with special=true
    assert_eq!(tok.decode(&[im_end], false), "");
    assert_eq!(tok.decode(&[im_end], true), "<|im_end|>");
    // user-defined (<tool_call>, <think>) always render
    let tc = tok.token_id("<tool_call>").unwrap();
    assert_eq!(tok.decode(&[tc], false), "<tool_call>");
    assert!(tok.chat_template.is_some());
}
