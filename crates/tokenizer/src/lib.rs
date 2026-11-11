//! Byte-level BPE tokenizer read from GGUF `tokenizer.ggml.*`, a port of llama.cpp's
//! `llama_vocab` BPE path (`src/llama-vocab.cpp`, `src/unicode.cpp`) for the
//! `gpt2` tokenizer model with the `qwen2` / `qwen35` pre-tokenizers.
//!
//! Encode and decode are meant to be identical to `llama_tokenize` /
//! `llama_detokenize` (and `common_token_to_piece`), including special-token
//! partitioning (`parse_special`), `add_special` (BOS/EOS per GGUF flags), byte
//! fallback and the token attribute rules.

pub mod gguf_meta;
pub mod unicode;
#[allow(clippy::all)]
mod unicode_data;

use gguf_meta::Meta;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

pub type Token = u32;

/// llama_token_attr bits.
pub mod attr {
    pub const UNDEFINED: u32 = 0;
    pub const UNKNOWN: u32 = 1 << 0;
    pub const UNUSED: u32 = 1 << 1;
    pub const NORMAL: u32 = 1 << 2;
    pub const CONTROL: u32 = 1 << 3;
    pub const USER_DEFINED: u32 = 1 << 4;
    pub const BYTE: u32 = 1 << 5;
    pub const NORMALIZED: u32 = 1 << 6;
    pub const LSTRIP: u32 = 1 << 7;
    pub const RSTRIP: u32 = 1 << 8;
    pub const SINGLE_WORD: u32 = 1 << 9;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreType {
    Qwen2,
    Qwen35,
}

#[derive(Debug)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
fn e(s: impl Into<String>) -> Error {
    Error(s.into())
}

pub struct Tokenizer {
    pre: PreType,
    tokens: Vec<String>,
    attrs: Vec<u32>,
    token_to_id: HashMap<String, Token>,
    /// merge rank keyed by (left id, right id); merges whose halves are not tokens
    /// fall back to `ranks_str`.
    ranks: HashMap<(Token, Token), u32>,
    ranks_str: HashMap<(String, String), u32>,
    /// special tokens for partitioning, longest text first.
    special_cache: Vec<Token>,
    pub bos: Option<Token>,
    pub eos: Option<Token>,
    pub eot: Option<Token>,
    pub pad: Option<Token>,
    pub add_bos: bool,
    pub add_eos: bool,
    eog: Vec<Token>,
    /// the raw `tokenizer.chat_template`, if any
    pub chat_template: Option<String>,
    /// sha256-free identity: hash of tokens+merges+template, for cache keys
    pub fingerprint: u64,
}

const EOT_NAMES: &[&str] = &[
    "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<EOT>", "_<EOT>",
    "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>",
];
const EOG_NAMES: &[&str] = &[
    "<|eot_id|>", "<|eot|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
    "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
    "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[", "<|ifm|im_end|>",
];

fn fnv(h: &mut u64, b: &[u8]) {
    for &x in b {
        *h ^= x as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
    *h ^= 0xff;
    *h = h.wrapping_mul(0x100000001b3);
}

impl Tokenizer {
    /// Load from a GGUF file (reads only the metadata section).
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Tokenizer, Error> {
        let md = gguf_meta::read_metadata(path).map_err(|x| e(x.to_string()))?;
        Self::from_metadata(&md)
    }

    pub fn from_metadata(md: &HashMap<String, Meta>) -> Result<Tokenizer, Error> {
        let model = md.get("tokenizer.ggml.model").and_then(Meta::as_str).unwrap_or("");
        if model != "gpt2" {
            return Err(e(format!("unsupported tokenizer model '{model}' (only gpt2/BPE)")));
        }
        let pre = match md.get("tokenizer.ggml.pre").and_then(Meta::as_str).unwrap_or("") {
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => PreType::Qwen2,
            "qwen35" => PreType::Qwen35,
            p => return Err(e(format!("unsupported pre-tokenizer '{p}' (only qwen2/qwen35)"))),
        };
        let toks = md.get("tokenizer.ggml.tokens").and_then(Meta::as_array).ok_or_else(|| e("missing tokens"))?;
        let mut fp = 0xcbf29ce484222325u64;
        let mut tokens = Vec::with_capacity(toks.len());
        for (i, t) in toks.iter().enumerate() {
            let mut s = t.as_str().ok_or_else(|| e("token not a string"))?.to_string();
            if s.is_empty() {
                s = format!("[EMPTY_{i}]");
            }
            fnv(&mut fp, s.as_bytes());
            tokens.push(s);
        }
        let n = tokens.len();
        let types: Vec<i64> = match md.get("tokenizer.ggml.token_type").and_then(Meta::as_array) {
            Some(a) => a.iter().map(|v| v.as_i64().unwrap_or(0)).collect(),
            None => vec![1; n],
        };
        let mut attrs: Vec<u32> = types
            .iter()
            .map(|t| match t {
                2 => attr::UNKNOWN,
                5 => attr::UNUSED,
                1 => attr::NORMAL,
                3 => attr::CONTROL,
                4 => attr::USER_DEFINED,
                6 => attr::BYTE,
                _ => attr::UNDEFINED,
            })
            .collect();
        attrs.resize(n, attr::NORMAL);

        let mut token_to_id = HashMap::with_capacity(n);
        for (i, s) in tokens.iter().enumerate() {
            // later duplicates win, like `token_to_id[word] = i`
            token_to_id.insert(s.clone(), i as Token);
        }

        let mut ranks = HashMap::new();
        let mut ranks_str = HashMap::new();
        if let Some(merges) = md.get("tokenizer.ggml.merges").and_then(Meta::as_array) {
            ranks.reserve(merges.len());
            for (i, m) in merges.iter().enumerate() {
                let w = m.as_str().unwrap_or("");
                fnv(&mut fp, w.as_bytes());
                // strchr(word + 1, ' ')
                let (a, b) = match w.as_bytes().iter().skip(1).position(|&c| c == b' ') {
                    Some(p) => (&w[..p + 1], &w[p + 2..]),
                    None => ("", ""),
                };
                match (token_to_id.get(a), token_to_id.get(b)) {
                    (Some(&x), Some(&y)) => {
                        ranks.entry((x, y)).or_insert(i as u32);
                    }
                    _ => {
                        ranks_str.entry((a.to_string(), b.to_string())).or_insert(i as u32);
                    }
                }
            }
        } else {
            return Err(e("cannot find tokenizer merges in model file"));
        }

        let id = |k: &str| md.get(k).and_then(Meta::as_i64).filter(|&v| v >= 0 && (v as usize) < n).map(|v| v as Token);
        let bos = id("tokenizer.ggml.bos_token_id");
        let eos = id("tokenizer.ggml.eos_token_id");
        let pad = id("tokenizer.ggml.padding_token_id");
        let mut eot = id("tokenizer.ggml.eot_token_id");
        let add_bos = md.get("tokenizer.ggml.add_bos_token").and_then(Meta::as_bool).unwrap_or(false);
        let add_eos = md.get("tokenizer.ggml.add_eos_token").and_then(Meta::as_bool).unwrap_or(false);

        // auto-detect EOT by text (first match in hash order in llama.cpp; we pick the
        // lowest id among matches, which is what single-match vocabs give anyway)
        if eot.is_none() {
            eot = EOT_NAMES.iter().filter_map(|t| token_to_id.get(*t).copied()).min();
            if let Some(t) = eot {
                attrs[t as usize] |= attr::CONTROL;
            }
        }
        // "unused" control tokens
        for (s, &t) in token_to_id.iter() {
            let a = &mut attrs[t as usize];
            if *a & attr::CONTROL != 0 && *a & attr::UNUSED == 0 && s.contains("unused") {
                *a |= attr::UNUSED;
            }
        }
        let mut eog: Vec<Token> = Vec::new();
        for name in EOG_NAMES {
            if let Some(&t) = token_to_id.get(*name) {
                eog.push(t);
                attrs[t as usize] |= attr::CONTROL;
            }
        }
        for name in ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"] {
            if let Some(&t) = token_to_id.get(name) {
                attrs[t as usize] = attr::USER_DEFINED;
            }
        }
        for t in [eos, eot].into_iter().flatten() {
            if !eog.contains(&t) {
                eog.push(t);
            }
        }
        eog.sort_unstable();
        eog.dedup();

        let mut special_cache: Vec<Token> = (0..n as Token)
            .filter(|&t| attrs[t as usize] & (attr::CONTROL | attr::USER_DEFINED | attr::UNKNOWN) != 0)
            .collect();
        special_cache.sort_by(|&a, &b| tokens[b as usize].len().cmp(&tokens[a as usize].len()).then(a.cmp(&b)));

        let chat_template = md.get("tokenizer.chat_template").and_then(Meta::as_str).map(str::to_string);
        if let Some(t) = &chat_template {
            fnv(&mut fp, t.as_bytes());
        }

        Ok(Tokenizer {
            pre,
            tokens,
            attrs,
            token_to_id,
            ranks,
            ranks_str,
            special_cache,
            bos,
            eos,
            eot,
            pad,
            add_bos,
            add_eos,
            eog,
            chat_template,
            fingerprint: fp,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }
    pub fn pre_type(&self) -> PreType {
        self.pre
    }
    pub fn token_text(&self, t: Token) -> Option<&str> {
        self.tokens.get(t as usize).map(String::as_str)
    }
    pub fn token_attr(&self, t: Token) -> u32 {
        self.attrs.get(t as usize).copied().unwrap_or(attr::UNDEFINED)
    }
    pub fn token_id(&self, text: &str) -> Option<Token> {
        self.token_to_id.get(text).copied()
    }
    pub fn is_eog(&self, t: Token) -> bool {
        self.eog.binary_search(&t).is_ok()
    }
    pub fn eog_tokens(&self) -> &[Token] {
        &self.eog
    }

    /// `llama_tokenize(text, add_special, parse_special)`.
    pub fn encode(&self, text: &str, add_special: bool, parse_special: bool) -> Vec<Token> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        if !text.is_empty() {
            for frag in self.partition(text, parse_special) {
                match frag {
                    Frag::Text(s) => self.bpe(s, &mut out),
                    Frag::Tok(t) => out.push(t),
                }
            }
        }
        if add_special && self.add_eos {
            if let Some(t) = self.eos {
                out.push(t);
            }
        }
        out
    }

    /// Port of `tokenizer_st_partition`.
    fn partition<'a>(&self, text: &'a str, parse_special: bool) -> Vec<Frag<'a>> {
        let mut buf: Vec<Frag<'a>> = vec![Frag::Text(text)];
        for &sid in &self.special_cache {
            let a = self.attrs[sid as usize];
            if !parse_special && a & (attr::CONTROL | attr::UNKNOWN) != 0 {
                continue;
            }
            let needle = self.tokens[sid as usize].as_str();
            let mut next: Vec<Frag<'a>> = Vec::with_capacity(buf.len());
            for f in buf {
                let Frag::Text(mut s) = f else {
                    next.push(f);
                    continue;
                };
                while let Some(m) = s.find(needle) {
                    let mut left = &s[..m];
                    if a & attr::LSTRIP != 0 {
                        left = left.trim_end_matches(|c: char| c.is_ascii_whitespace() || c == '\x0b');
                    }
                    if !left.is_empty() {
                        next.push(Frag::Text(left));
                    }
                    next.push(Frag::Tok(sid));
                    s = &s[m + needle.len()..];
                    if a & attr::RSTRIP != 0 {
                        s = s.trim_start_matches(|c: char| c.is_ascii_whitespace() || c == '\x0b');
                    }
                }
                if !s.is_empty() {
                    next.push(Frag::Text(s));
                }
            }
            buf = next;
        }
        buf
    }

    fn rank(&self, a: &str, b: &str) -> Option<u32> {
        match (self.token_to_id.get(a), self.token_to_id.get(b)) {
            (Some(&x), Some(&y)) => self.ranks.get(&(x, y)).copied(),
            _ if !self.ranks_str.is_empty() => self.ranks_str.get(&(a.to_string(), b.to_string())).copied(),
            _ => None,
        }
    }

    /// Port of `llm_tokenizer_bpe_session::tokenize` for one raw-text fragment.
    fn bpe(&self, text: &str, out: &mut Vec<Token>) {
        let cpts: Vec<u32> = text.chars().map(|c| c as u32).collect();
        let lens = unicode::split_qwen(&cpts, self.pre == PreType::Qwen35);
        let bmap = unicode::byte_to_cpt_table();
        let mut start = 0usize;
        let mut word = String::new();
        for len in lens {
            word.clear();
            for &c in &cpts[start..start + len] {
                let mut b = [0u8; 4];
                for &x in char::from_u32(c).unwrap().encode_utf8(&mut b).as_bytes() {
                    word.push(bmap[x as usize]);
                }
            }
            start += len;
            self.bpe_word(&word, out);
        }
    }

    fn bpe_word(&self, word: &str, out: &mut Vec<Token>) {
        // fast path: whole word is a token and it has a single symbol, or the merges
        // would rebuild it; we still run the merge loop to stay exact (ignore_merges=false).
        #[derive(Clone, Copy)]
        struct Sym {
            prev: i32,
            next: i32,
            start: usize,
            n: usize,
        }
        let mut syms: Vec<Sym> = Vec::with_capacity(word.len());
        let mut off = 0usize;
        for (i, ch) in word.char_indices() {
            let l = ch.len_utf8();
            syms.push(Sym { prev: syms.len() as i32 - 1, next: -1, start: i, n: l });
            off = i + l;
        }
        let ns = syms.len();
        for (i, s) in syms.iter_mut().enumerate() {
            s.next = if i + 1 == ns { -1 } else { i as i32 + 1 };
        }
        debug_assert_eq!(off, word.len());

        // (rank, left, right, text_len) ; min-heap on (rank, left)
        let mut heap: BinaryHeap<Reverse<(u32, i32, i32, usize)>> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<(u32, i32, i32, usize)>>, syms: &Vec<Sym>, l: i32, r: i32| {
            if l < 0 || r < 0 {
                return;
            }
            let (a, b) = (syms[l as usize], syms[r as usize]);
            let lt = &word[a.start..a.start + a.n];
            let rt = &word[b.start..b.start + b.n];
            if let Some(rank) = self.rank(lt, rt) {
                heap.push(Reverse((rank, l, r, a.n + b.n)));
            }
        };
        for i in 1..ns {
            push(&mut heap, &syms, i as i32 - 1, i as i32);
        }
        while let Some(Reverse((_, l, r, size))) = heap.pop() {
            let (ls, rs) = (syms[l as usize], syms[r as usize]);
            if ls.n == 0 || rs.n == 0 {
                continue;
            }
            // outdated check: left+right text must equal the bigram text. Symbols are
            // contiguous slices, so equality reduces to adjacency + same total length
            // + left start unchanged (left start never changes) and r adjacent.
            if ls.next != r || ls.n + rs.n != size {
                continue;
            }
            syms[l as usize].n += rs.n;
            syms[r as usize].n = 0;
            syms[l as usize].next = rs.next;
            if rs.next >= 0 {
                syms[rs.next as usize].prev = l;
            }
            let lp = syms[l as usize].prev;
            let ln = syms[l as usize].next;
            push(&mut heap, &syms, lp, l);
            push(&mut heap, &syms, l, ln);
        }

        let mut i: i32 = if ns > 0 { 0 } else { -1 };
        while i >= 0 {
            let s = syms[i as usize];
            i = s.next;
            if s.n == 0 {
                continue;
            }
            let t = &word[s.start..s.start + s.n];
            match self.token_to_id.get(t) {
                Some(&id) => out.push(id),
                None => {
                    // byte fallback: each byte as a single-char token (byte_encode=true)
                    for &b in t.as_bytes() {
                        let one = [b];
                        if let Ok(s1) = std::str::from_utf8(&one) {
                            if let Some(&id) = self.token_to_id.get(s1) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// `llama_token_to_piece(token, lstrip=0, special)` as raw bytes.
    pub fn piece(&self, t: Token, special: bool) -> Vec<u8> {
        let a = self.token_attr(t);
        if !special && a & (attr::UNKNOWN | attr::CONTROL) != 0 {
            return Vec::new();
        }
        let Some(text) = self.tokens.get(t as usize) else {
            return Vec::new();
        };
        if a & (attr::UNKNOWN | attr::CONTROL | attr::USER_DEFINED) != 0 {
            return text.as_bytes().to_vec();
        }
        if a & attr::NORMAL != 0 {
            let mut out = Vec::with_capacity(text.len());
            for c in text.chars() {
                match unicode::cpt_to_byte(c) {
                    Some(b) => out.push(b),
                    None => {
                        let mut tmp = [0u8; 4];
                        out.extend_from_slice(b"[UNK_BYTE_0x");
                        for &x in c.encode_utf8(&mut tmp).as_bytes() {
                            out.extend_from_slice(format!("{x:02x}").as_bytes());
                        }
                        out.extend_from_slice(text.as_bytes());
                        out.push(b']');
                    }
                }
            }
            return out;
        }
        if a & attr::BYTE != 0 {
            // not produced by gpt2 vocabs; mirror llama.cpp's token_to_byte for <0xXX>
            if let Some(h) = text.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                if let Ok(b) = u8::from_str_radix(h, 16) {
                    return vec![b];
                }
            }
        }
        Vec::new()
    }

    /// `llama_detokenize(tokens, remove_special=false, unparse_special=special)` as bytes
    /// (qwen: no space prefix, clean_spaces=false).
    pub fn decode_bytes(&self, toks: &[Token], special: bool) -> Vec<u8> {
        let mut out = Vec::new();
        for &t in toks {
            out.extend(self.piece(t, special));
        }
        out
    }

    /// Lossy UTF-8 decode (what `/detokenize` returns).
    pub fn decode(&self, toks: &[Token], special: bool) -> String {
        String::from_utf8_lossy(&self.decode_bytes(toks, special)).into_owned()
    }
}

enum Frag<'a> {
    Text(&'a str),
    Tok(Token),
}

/// Incremental detokenizer for streaming: holds back incomplete UTF-8 sequences
/// (llama-server's `validate_utf8` behaviour) so each emitted chunk is valid UTF-8.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = utf8_valid_prefix(&self.pending);
        let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        s
    }
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Length of the longest prefix that does not end inside an incomplete (but so far
/// valid-looking) multi-byte sequence. Invalid bytes are let through (lossy).
pub fn utf8_valid_prefix(b: &[u8]) -> usize {
    let n = b.len();
    // look at up to the last 3 bytes for a lead byte whose sequence is incomplete
    for back in 1..=3.min(n) {
        let c = b[n - back];
        if c & 0xC0 == 0x80 {
            continue; // continuation byte, keep looking for the lead
        }
        let need = if c & 0xE0 == 0xC0 {
            2
        } else if c & 0xF0 == 0xE0 {
            3
        } else if c & 0xF8 == 0xF0 {
            4
        } else {
            1
        };
        return if need > back { n - back } else { n };
    }
    n
}

#[cfg(test)]
mod tests;
