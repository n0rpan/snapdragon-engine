//! Model output -> OpenAI message, a hand port of llama.cpp's Qwen3-Coder PEG parser
//! (`common/parsers/qwen3-coder.cpp` + the `common_chat_peg_mapper`), including the
//! partial (streaming) behaviour, and `common_chat_msg_diff::compute_diffs`.
//!
//! The parser runs on `generation_prompt + output` and is re-run on the whole text at every
//! step, exactly like `common_chat_session::feed`.

use crate::request::{ReasoningFormat, Tool, ToolChoice};
use crate::template::Format;
use serde_json::{Map, Value as Json};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub id: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatMsg {
    pub content: String,
    pub reasoning_content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatMsg {
    pub fn is_empty(&self) -> bool {
        self.content.is_empty() && self.reasoning_content.is_empty() && self.tool_calls.is_empty()
    }

    /// `common_chat_msg::to_json_oaicompat` for an assistant message.
    pub fn to_json(&self) -> Json {
        let mut m = Map::new();
        m.insert("role".into(), "assistant".into());
        m.insert("content".into(), Json::String(self.content.clone()));
        if !self.reasoning_content.is_empty() {
            m.insert("reasoning_content".into(), Json::String(self.reasoning_content.clone()));
        }
        if !self.tool_calls.is_empty() {
            let v = self
                .tool_calls
                .iter()
                .map(|tc| {
                    let mut o = Map::new();
                    o.insert("type".into(), "function".into());
                    let mut f = Map::new();
                    f.insert("name".into(), Json::String(tc.name.clone()));
                    f.insert("arguments".into(), Json::String(tc.arguments.clone()));
                    o.insert("function".into(), Json::Object(f));
                    if !tc.id.is_empty() {
                        o.insert("id".into(), Json::String(tc.id.clone()));
                    }
                    Json::Object(o)
                })
                .collect();
            m.insert("tool_calls".into(), Json::Array(v));
        }
        Json::Object(m)
    }

    /// `set_tool_call_ids`: keep ids per index across the stream.
    pub fn set_tool_call_ids(&mut self, cache: &mut Vec<String>, mut gen: impl FnMut() -> String) {
        for (i, tc) in self.tool_calls.iter_mut().enumerate() {
            if cache.len() <= i {
                let id = if tc.id.is_empty() { gen() } else { tc.id.clone() };
                cache.push(id);
            }
            tc.id = cache[i].clone();
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MsgDiff {
    pub reasoning_content_delta: String,
    pub content_delta: String,
    pub tool_call_index: Option<usize>,
    pub tool_call_delta: ToolCall,
}

impl MsgDiff {
    /// `server_chat_msg_diff_to_json_oaicompat`
    pub fn to_delta_json(&self) -> Json {
        let mut d = Map::new();
        if !self.reasoning_content_delta.is_empty() {
            d.insert("reasoning_content".into(), Json::String(self.reasoning_content_delta.clone()));
        }
        if !self.content_delta.is_empty() {
            d.insert("content".into(), Json::String(self.content_delta.clone()));
        }
        if let Some(i) = self.tool_call_index {
            let mut tc = Map::new();
            tc.insert("index".into(), Json::from(i));
            if !self.tool_call_delta.id.is_empty() {
                tc.insert("id".into(), Json::String(self.tool_call_delta.id.clone()));
                tc.insert("type".into(), "function".into());
            }
            if !self.tool_call_delta.name.is_empty() || !self.tool_call_delta.arguments.is_empty() {
                let mut f = Map::new();
                if !self.tool_call_delta.name.is_empty() {
                    f.insert("name".into(), Json::String(self.tool_call_delta.name.clone()));
                }
                if !self.tool_call_delta.arguments.is_empty() {
                    f.insert("arguments".into(), Json::String(self.tool_call_delta.arguments.clone()));
                }
                tc.insert("function".into(), Json::Object(f));
            }
            d.insert("tool_calls".into(), Json::Array(vec![Json::Object(tc)]));
        }
        Json::Object(d)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffError(pub String);

fn string_diff(last: &str, cur: &str) -> Result<String, DiffError> {
    if last.is_empty() {
        return Ok(cur.to_string());
    }
    if !cur.starts_with(last) {
        if last.starts_with(cur) {
            return Ok(String::new());
        }
        return Err(DiffError(format!("Invalid diff: '{last}' not found at start of '{cur}'")));
    }
    Ok(cur[last.len()..].to_string())
}

/// `common_chat_msg_diff::compute_diffs`
pub fn compute_diffs(prv: &ChatMsg, new: &ChatMsg) -> Result<Vec<MsgDiff>, DiffError> {
    let mut out = Vec::new();
    if prv.reasoning_content != new.reasoning_content {
        out.push(MsgDiff { reasoning_content_delta: string_diff(&prv.reasoning_content, &new.reasoning_content)?, ..Default::default() });
    }
    if prv.content != new.content {
        out.push(MsgDiff { content_delta: string_diff(&prv.content, &new.content)?, ..Default::default() });
    }
    if new.tool_calls.len() < prv.tool_calls.len() {
        return Err(DiffError("Invalid diff: now finding less tool calls!".into()));
    }
    if !prv.tool_calls.is_empty() {
        let idx = prv.tool_calls.len() - 1;
        let (p, n) = (&prv.tool_calls[idx], &new.tool_calls[idx]);
        if !p.name.is_empty() && !n.name.starts_with(&p.name) {
            return Err(DiffError("Invalid diff: tool call mismatch!".into()));
        }
        let args = string_diff(&p.arguments, &n.arguments)?;
        if !args.is_empty() || p.id != n.id || p.name != n.name {
            let mut d = MsgDiff { tool_call_index: Some(idx), ..Default::default() };
            if p.id != n.id || p.name != n.name {
                d.tool_call_delta.id = n.id.clone();
                d.tool_call_delta.name = n.name.clone();
            }
            d.tool_call_delta.arguments = args;
            out.push(d);
        }
    }
    for idx in prv.tool_calls.len()..new.tool_calls.len() {
        out.push(MsgDiff { tool_call_index: Some(idx), tool_call_delta: new.tool_calls[idx].clone(), ..Default::default() });
    }
    Ok(out)
}

// ----------------------------------------------------------------------------------------
// schema value types (common/json-schema.cpp value_types)

const T_NULL: u8 = 1;
const T_BOOL: u8 = 2;
const T_INT: u8 = 4;
const T_NUM: u8 = 8;
const T_STR: u8 = 16;
const T_ARR: u8 = 32;
const T_OBJ: u8 = 64;
const T_ALL: u8 = 127;

fn json_type(v: &Json) -> u8 {
    match v {
        Json::Null => T_NULL,
        Json::Bool(_) => T_BOOL,
        Json::Number(n) if n.is_i64() || n.is_u64() => T_INT,
        Json::Number(_) => T_NUM,
        Json::String(_) => T_STR,
        Json::Array(_) => T_ARR,
        Json::Object(_) => T_OBJ,
    }
}

fn type_name(t: &str) -> u8 {
    match t {
        "string" => T_STR,
        "integer" => T_INT,
        "number" => T_NUM | T_INT,
        "boolean" => T_BOOL,
        "null" => T_NULL,
        "array" => T_ARR,
        "object" => T_OBJ,
        _ => T_ALL,
    }
}

fn resolve_ref<'a>(root: &'a Json, r: &str) -> Option<&'a Json> {
    let p = r.strip_prefix('#')?;
    if p.is_empty() {
        return Some(root);
    }
    root.pointer(p)
}

fn value_types(s: &Json, root: &Json, depth: usize) -> u8 {
    if depth > 32 {
        return 0;
    }
    let Some(o) = s.as_object() else {
        return T_ALL; // `true` schema etc.
    };
    if let Some(r) = o.get("$ref").and_then(Json::as_str) {
        return resolve_ref(root, r).map(|t| value_types(t, root, depth + 1)).unwrap_or(0);
    }
    for k in ["oneOf", "anyOf"] {
        if let Some(a) = o.get(k).and_then(Json::as_array) {
            return a.iter().fold(0, |acc, c| acc | value_types(c, root, depth + 1));
        }
    }
    match o.get("type") {
        Some(Json::Array(a)) => {
            if a.is_empty() {
                return T_ALL;
            }
            return a.iter().fold(0, |acc, t| acc | t.as_str().map(type_name).unwrap_or(0));
        }
        Some(Json::String(_)) | None => {}
        Some(_) => return T_ALL,
    }
    if let Some(c) = o.get("const") {
        return json_type(c);
    }
    if let Some(Json::Array(e)) = o.get("enum") {
        return e.iter().fold(0, |acc, v| acc | json_type(v));
    }
    let tn = o.get("type").and_then(Json::as_str).unwrap_or("");
    if tn.is_empty() {
        if o.contains_key("properties") {
            return T_OBJ;
        }
        if let Some(a) = o.get("allOf").and_then(Json::as_array) {
            return a.iter().fold(T_ALL, |acc, c| acc & value_types(c, root, depth + 1));
        }
        if o.contains_key("items") || o.contains_key("prefixItems") {
            return T_ARR;
        }
        if o.contains_key("pattern") || o.contains_key("minLength") || o.contains_key("maxLength") || o.contains_key("format") {
            return T_STR;
        }
        return T_ALL;
    }
    if tn == "object" && !o.contains_key("properties") {
        if let Some(a) = o.get("allOf").and_then(Json::as_array) {
            return a.iter().fold(T_ALL, |acc, c| acc & value_types(c, root, depth + 1));
        }
    }
    type_name(tn)
}

struct Param {
    name: String,
    required: bool,
    types: u8,
}

fn tool_params(t: &Tool) -> Vec<Param> {
    // common_chat_tool_parameters: missing / null / {} -> no parameters
    let p = &t.parameters;
    let props = p.get("properties").and_then(Json::as_object);
    let req: Vec<&str> = p.get("required").and_then(Json::as_array).map(|a| a.iter().filter_map(Json::as_str).collect()).unwrap_or_default();
    let mut out = Vec::new();
    if let Some(props) = props {
        for (k, s) in props {
            out.push(Param { name: k.clone(), required: req.contains(&k.as_str()), types: value_types(s, p, 0) });
        }
    }
    out
}

// ----------------------------------------------------------------------------------------
// the parser

#[derive(Clone, Debug)]
pub struct ParserConfig {
    pub format: Format,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub reasoning_format: ReasoningFormat,
    /// template contains `<think>`
    pub supports_thinking: bool,
    /// Lenient tool calls (the server; the llama.cpp parity tests run without): parameters in any
    /// order (llama.cpp: required first); on the FINAL parse, a call whose required parameters are
    /// complete but whose closing tags are cut off (stream ended) is closed, and the text of a
    /// call that cannot be parsed is returned as content instead of being dropped.
    pub lenient: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError(pub String);

const GEN_PREFIX: &str = "<|im_start|>assistant\n";

#[derive(PartialEq, Debug)]
enum St {
    Ok,
    More,
    Fail,
}

fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r')
}

fn skip_space(s: &str, mut p: usize) -> usize {
    let b = s.as_bytes();
    while p < b.len() && is_space(b[p]) {
        p += 1;
    }
    p
}

/// `until_one_of`: returns (end of text, Some(delim index) if a delimiter matched fully).
/// With partial input a trailing prefix of a delimiter is held back.
fn until_one_of(s: &str, from: usize, delims: &[String], partial: bool) -> (usize, Option<usize>) {
    let b = s.as_bytes();
    let mut i = from;
    while i < b.len() {
        for (k, d) in delims.iter().enumerate() {
            if b[i..].starts_with(d.as_bytes()) {
                return (i, Some(k));
            }
        }
        if partial && delims.iter().any(|d| d.as_bytes().starts_with(&b[i..])) {
            return (i, None);
        }
        i += 1;
        while i < b.len() && !s.is_char_boundary(i) {
            i += 1;
        }
    }
    (b.len(), None)
}

/// literal at p: Ok(new p) / More (input ends inside a prefix) / Fail
fn lit(s: &str, p: usize, l: &str, partial: bool) -> (St, usize) {
    let rest = &s.as_bytes()[p..];
    if rest.starts_with(l.as_bytes()) {
        (St::Ok, p + l.len())
    } else if partial && l.as_bytes().starts_with(rest) {
        (St::More, s.len())
    } else {
        (St::Fail, p)
    }
}

/// JSON value scanner. Returns (state, end). `More` means the input ended inside a value
/// that could still be valid; end is then s.len().
fn scan_json(s: &[u8], p: usize, allowed: u8, partial: bool) -> (St, usize) {
    fn ws(s: &[u8], mut p: usize) -> usize {
        while p < s.len() && is_space(s[p]) {
            p += 1;
        }
        p
    }
    fn eof(partial: bool, n: usize) -> (St, usize) {
        if partial {
            (St::More, n)
        } else {
            (St::Fail, n)
        }
    }
    fn value(s: &[u8], p: usize, allowed: u8, partial: bool, depth: usize) -> (St, usize) {
        if depth > 128 {
            return (St::Fail, p);
        }
        if p >= s.len() {
            return eof(partial, s.len());
        }
        match s[p] {
            b'{' if allowed & T_OBJ != 0 => {
                let mut q = ws(s, p + 1);
                if q >= s.len() {
                    return eof(partial, s.len());
                }
                if s[q] == b'}' {
                    return (St::Ok, q + 1);
                }
                loop {
                    let (st, e) = string(s, q, partial);
                    if st != St::Ok {
                        return (st, e);
                    }
                    q = ws(s, e);
                    if q >= s.len() {
                        return eof(partial, s.len());
                    }
                    if s[q] != b':' {
                        return (St::Fail, q);
                    }
                    q = ws(s, q + 1);
                    let (st, e) = value(s, q, T_ALL, partial, depth + 1);
                    if st != St::Ok {
                        return (st, e);
                    }
                    q = ws(s, e);
                    if q >= s.len() {
                        return eof(partial, s.len());
                    }
                    match s[q] {
                        b',' => q = ws(s, q + 1),
                        b'}' => return (St::Ok, q + 1),
                        _ => return (St::Fail, q),
                    }
                }
            }
            b'[' if allowed & T_ARR != 0 => {
                let mut q = ws(s, p + 1);
                if q >= s.len() {
                    return eof(partial, s.len());
                }
                if s[q] == b']' {
                    return (St::Ok, q + 1);
                }
                loop {
                    let (st, e) = value(s, q, T_ALL, partial, depth + 1);
                    if st != St::Ok {
                        return (st, e);
                    }
                    q = ws(s, e);
                    if q >= s.len() {
                        return eof(partial, s.len());
                    }
                    match s[q] {
                        b',' => q = ws(s, q + 1),
                        b']' => return (St::Ok, q + 1),
                        _ => return (St::Fail, q),
                    }
                }
            }
            b'"' if allowed & T_STR != 0 => string(s, p, partial),
            b't' if allowed & T_BOOL != 0 => word(s, p, b"true", partial),
            b'f' if allowed & T_BOOL != 0 => word(s, p, b"false", partial),
            b'n' if allowed & T_NULL != 0 => word(s, p, b"null", partial),
            b'-' | b'0'..=b'9' if allowed & (T_NUM | T_INT) != 0 => number(s, p, partial),
            _ => (St::Fail, p),
        }
    }
    fn word(s: &[u8], p: usize, w: &[u8], partial: bool) -> (St, usize) {
        let r = &s[p..];
        if r.starts_with(w) {
            (St::Ok, p + w.len())
        } else if w.starts_with(r) {
            eof(partial, s.len())
        } else {
            (St::Fail, p)
        }
    }
    fn string(s: &[u8], p: usize, partial: bool) -> (St, usize) {
        if p >= s.len() {
            return eof(partial, s.len());
        }
        if s[p] != b'"' {
            return (St::Fail, p);
        }
        let mut q = p + 1;
        while q < s.len() {
            match s[q] {
                b'"' => return (St::Ok, q + 1),
                b'\\' => {
                    if q + 1 >= s.len() {
                        return eof(partial, s.len());
                    }
                    match s[q + 1] {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => q += 2,
                        b'u' => {
                            for k in 0..4 {
                                if q + 2 + k >= s.len() {
                                    return eof(partial, s.len());
                                }
                                if !s[q + 2 + k].is_ascii_hexdigit() {
                                    return (St::Fail, q);
                                }
                            }
                            q += 6;
                        }
                        _ => return (St::Fail, q),
                    }
                }
                c if c < 0x20 => return (St::Fail, q),
                _ => q += 1,
            }
        }
        eof(partial, s.len())
    }
    fn number(s: &[u8], p: usize, partial: bool) -> (St, usize) {
        let mut q = p;
        if s[q] == b'-' {
            q += 1;
        }
        if q >= s.len() {
            return eof(partial, s.len());
        }
        if s[q] == b'0' {
            q += 1;
        } else if s[q].is_ascii_digit() {
            while q < s.len() && s[q].is_ascii_digit() {
                q += 1;
            }
        } else {
            return (St::Fail, q);
        }
        if q < s.len() && s[q] == b'.' {
            q += 1;
            if q >= s.len() {
                return eof(partial, s.len());
            }
            if !s[q].is_ascii_digit() {
                return (St::Fail, q);
            }
            while q < s.len() && s[q].is_ascii_digit() {
                q += 1;
            }
        }
        if q < s.len() && (s[q] == b'e' || s[q] == b'E') {
            q += 1;
            if q < s.len() && (s[q] == b'+' || s[q] == b'-') {
                q += 1;
            }
            if q >= s.len() {
                return eof(partial, s.len());
            }
            if !s[q].is_ascii_digit() {
                return (St::Fail, q);
            }
            while q < s.len() && s[q].is_ascii_digit() {
                q += 1;
            }
        }
        (St::Ok, q)
    }
    value(s, p, allowed, partial, 0)
}

/// `escape_json_string_inner` (nlohmann dump of a string without the quotes)
fn escape_inner(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    crate::template::json_escape_into(&mut o, s);
    o[1..o.len() - 1].to_string()
}

/// Where a partial parse stopped, when appending text can only extend one string of the
/// message until the next delimiter (used by `Session`'s incremental path). Byte offsets in
/// `generation_prompt + text`. Each variant is recorded only where its start offset can no
/// longer move when text is appended (see `parse_full`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Tail {
    /// unclosed think block: `reasoning_content == s[q..end]`, `end` = where the scan for
    /// the reasoning ends stopped (no delimiter starts before it)
    Reasoning { q: usize, end: usize },
    /// `content == s[p..end]`, no tool call; `tools`: content ends at a tool-call start (else
    /// content runs to the end of the text)
    Content { p: usize, end: usize, tools: bool },
    /// the last tool call's last argument is an open string-only value:
    /// `arguments == prefix + escape_inner(s[q..end])`
    StrArg { q: usize, end: usize },
}

/// Parameter value close tag (Qwen3-Coder format).
const PARAM_CLOSE: &str = "\n</parameter>\n";

pub(crate) fn reasoning_ends(cfg: &ParserConfig) -> Vec<String> {
    if cfg.format == Format::Qwen3Coder {
        vec!["\n</think>".into(), "</think>".into(), "<tool_call>".into()]
    } else {
        vec!["</think>".into()]
    }
}

pub(crate) fn content_ends(cfg: &ParserConfig) -> Vec<String> {
    let mut starts = vec!["<tool_call>".to_string()];
    if !cfg.supports_thinking {
        for t in &cfg.tools {
            starts.push(format!("<function={}>", t.name));
        }
    }
    let mut content_ends = Vec::new();
    for st in &starts {
        content_ends.push(format!("\n\n{st}"));
        content_ends.push(format!("\n{st}"));
        content_ends.push(st.clone());
    }
    content_ends
}

/// Incremental step: `s` = the previous parse's string with text appended, `tail` = that
/// parse's [`Tail`] and `prev` its message. Returns the message `parse_full(s, partial=true)`
/// would return, or None when the step leaves the tail's state (then run the full parse).
///
/// Why this equals the full parse: the full parse reaches the same tail position by the same
/// decisions, because everything before the tail start was decided on complete input (a
/// matched literal or a delimiter hit stays the same when text is appended; the start offset
/// is recorded only when a non-space byte follows it, so `skip_space` cannot move it). Inside
/// the tail, `until_one_of` rejected every position before `end` for good: a position i was
/// neither a full delimiter match nor (partial) a prefix of one, so the bytes from i to the
/// old end of text already differ from every delimiter, and appending cannot create a match
/// at i. Resuming the scan at `end` therefore gives the full scan's result. If it hits a
/// delimiter, None (the full parse handles the transition). If not, the string grows to the
/// new `end` and nothing after it is produced (the rest of the text is entirely a prefix of
/// a delimiter, which cannot contain a complete tool-call opener). For string arguments,
/// `escape_inner` maps each char independently and `end` is a char boundary, so the escape
/// of the longer slice is the old escape plus the escape of the new part.
pub(crate) fn parse_step(cfg: &ParserConfig, s: &str, tail: &Tail, prev: &ChatMsg) -> Option<(ChatMsg, Tail, usize)> {
    match *tail {
        Tail::Reasoning { q, end } => {
            let (e, hit) = until_one_of(s, end, &reasoning_ends(cfg), true);
            if hit.is_some() {
                return None;
            }
            let mut m = prev.clone();
            m.reasoning_content.push_str(&s[end..e]);
            debug_assert_eq!(m.reasoning_content, &s[q..e]);
            Some((m, Tail::Reasoning { q, end: e }, 0))
        }
        Tail::Content { p, end, tools: false } => {
            let mut m = prev.clone();
            m.content.push_str(&s[end..]);
            debug_assert_eq!(m.content, &s[p..]);
            Some((m, Tail::Content { p, end: s.len(), tools: false }, 0))
        }
        Tail::Content { p, end, tools: true } => {
            let (e, hit) = until_one_of(s, end, &content_ends(cfg), true);
            if hit.is_some() {
                return None;
            }
            let mut m = prev.clone();
            m.content.push_str(&s[end..e]);
            debug_assert_eq!(m.content, &s[p..e]);
            Some((m, Tail::Content { p, end: e, tools: true }, 0))
        }
        Tail::StrArg { q, end } => {
            let (e, hit) = until_one_of(s, end, &[PARAM_CLOSE.to_string()], true);
            if hit.is_some() {
                return None;
            }
            let mut m = prev.clone();
            let tc = m.tool_calls.last_mut()?;
            tc.arguments += &escape_inner(&s[end..e]);
            let _ = q;
            let complete = m.tool_calls.len() - 1;
            Some((m, Tail::StrArg { q, end: e }, complete))
        }
    }
}

/// Parse the output of a Qwen3-Coder-format template. `text` is the model output only;
/// `generation_prompt` is prepended like llama.cpp does.
pub fn parse(cfg: &ParserConfig, generation_prompt: &str, text: &str, partial: bool) -> Result<ChatMsg, ParseError> {
    let full = format!("{generation_prompt}{text}");
    parse_full(cfg, &full, partial).map(|r| r.0)
}

/// [`parse`] on `s = generation_prompt + text`, also returning the [`Tail`] of a partial parse
/// (None when the parse did not stop in a tail state, or on a final parse).
/// The third value is the number of complete tool calls (closed `</tool_call>`, or every call of
/// a final parse): a partial parse's last call can still vanish or change when text is appended
/// (e.g. the model writes a parameter the grammar rejects), complete ones cannot.
pub(crate) fn parse_full(cfg: &ParserConfig, s: &str, partial: bool) -> Result<(ChatMsg, Option<Tail>, usize), ParseError> {
    let mut msg = ChatMsg::default();

    let (st, mut p) = lit(s, 0, GEN_PREFIX, partial);
    match st {
        St::Ok => {}
        St::More => return Ok((msg, None, 0)),
        St::Fail => {
            return Err(ParseError("The model produced output that does not match the expected format".into()));
        }
    }

    let qwen3_coder = cfg.format == Format::Qwen3Coder;
    let extract = cfg.reasoning_format != ReasoningFormat::None;
    // reasoning: optional("<think>" space reasoning(until) close)
    if cfg.supports_thinking && extract {
        let ends = reasoning_ends(cfg);
        let (st, q) = lit(s, p, "<think>", partial);
        match st {
            St::Ok => {
                let q = skip_space(s, q);
                let (end, hit) = until_one_of(s, q, &ends, partial);
                let body = &s[q..end];
                match hit {
                    Some(k) => {
                        msg.reasoning_content = body.to_string();
                        // close: "\n</think>" | "</think>" | peek("<tool_call>")
                        p = if ends[k] == "<tool_call>" { end } else { end + ends[k].len() };
                    }
                    None => {
                        // unclosed think block: everything so far is reasoning (lenient parse,
                        // also on the final complete parse)
                        msg.reasoning_content = body.to_string();
                        clear_ws_reasoning(&mut msg);
                        // q is fixed once a non-space byte follows it (then the body is either
                        // empty or starts with that byte, so clear_ws_reasoning is a no-op)
                        let tail = (partial && q < s.len()).then_some(Tail::Reasoning { q, end });
                        return Ok((msg, tail, 0));
                    }
                }
            }
            St::More => return Ok((msg, None, 0)),
            St::Fail => {}
        }
        clear_ws_reasoning(&mut msg);
    }

    p = skip_space(s, p);
    let tools_active = qwen3_coder && !cfg.tools.is_empty() && cfg.tool_choice != ToolChoice::None;
    if !tools_active {
        msg.content = s[p..].to_string();
        let tail = (partial && p < s.len()).then_some(Tail::Content { p, end: s.len(), tools: false });
        return Ok((msg, tail, 0));
    }

    let is_qwen3_coder_model = !cfg.supports_thinking;
    let content_ends = content_ends(cfg);
    let (end, hit) = until_one_of(s, p, &content_ends, partial);
    msg.content = s[p..end].to_string();
    let content_tail = (partial && hit.is_none() && p < s.len()).then_some(Tail::Content { p, end, tools: true });
    p = skip_space(s, end);

    let max_calls = if cfg.parallel_tool_calls { usize::MAX } else { 1 };
    let mut ncalls = 0;
    let mut tail = None;
    while ncalls < max_calls {
        let first = ncalls == 0;
        let mut str_tail = None;
        let (st, q, call) = parse_call(cfg, s, p, partial, first && is_qwen3_coder_model, &mut str_tail);
        match st {
            St::Ok => {
                msg.tool_calls.push(call.unwrap());
                p = skip_space(s, q);
                ncalls += 1;
            }
            St::More => {
                if let Some(c) = call {
                    msg.tool_calls.push(c);
                    tail = str_tail.map(|(q, end)| Tail::StrArg { q, end });
                }
                break;
            }
            St::Fail => break,
        }
    }
    if msg.tool_calls.is_empty() {
        tail = content_tail;
    }
    if !partial && cfg.lenient && p < s.len() {
        // a call that cannot be parsed (or text after the last call): return it as text
        // instead of dropping it
        msg.content.push_str(&s[p..]);
    }
    let complete = if partial { ncalls } else { msg.tool_calls.len() };
    Ok((msg, tail, complete))
}

fn clear_ws_reasoning(msg: &mut ChatMsg) {
    if msg.reasoning_content.bytes().all(|c| matches!(c, b' ' | b'\n' | b'\r' | b'\t')) {
        msg.reasoning_content.clear();
    }
}

fn is_space_char(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// Integer-only parameter (`integer`, not `number`) given a number with a fraction or exponent:
/// `Ok(Some(int))` when the value is exactly integral (`1.0`, `2e3`, within +-2^53), `Err` when
/// it is not, `Ok(None)` when nothing applies (another type, a plain integer, not a number).
fn normalize_int(v: &str, types: u8) -> Result<Option<String>, ()> {
    if types & T_INT == 0 || types & T_NUM != 0 {
        return Ok(None);
    }
    let b = v.as_bytes();
    if b.is_empty() || !(b[0] == b'-' || b[0].is_ascii_digit()) || !b.iter().any(|c| matches!(c, b'.' | b'e' | b'E')) {
        return Ok(None);
    }
    match v.parse::<f64>() {
        Ok(f) if f.is_finite() && f.fract() == 0.0 && f.abs() <= 9_007_199_254_740_992.0 => Ok(Some(format!("{}", f as i64))),
        _ => Err(()),
    }
}

/// Lenient typed coercion (F105, after vLLM's qwen3_coder `convert_param_value`) for the raw
/// text of a parameter that cannot be a string (`types` has no T_STR), used only when its
/// JSON value failed. Returns the JSON to emit, or None when the text is ambiguous (the call
/// then fails as before). Accepted, after trimming ASCII whitespace (the format's own newlines
/// are already consumed by the tags; JSON allows whitespace around a value):
/// - a JSON value of an allowed type (`  5 `, `5\r`);
/// - the same value quoted once as a JSON string: `"5"`, `"1.5"`, `"true"`, `"null"`, and
///   nested JSON for an object/array parameter (`"[1, 2]"`, `"{\"a\": 1}"`);
/// - Python literals `True` / `False` (any case) for a boolean, `None` for a null;
/// - for an integer-only parameter a number must be exactly integral (`1.0` -> `1`; `1.5` is
///   ambiguous and fails).
/// Not accepted: empty values, leading zeros, hex, NaN/Infinity, `yes`/`1` for a boolean,
/// Python containers (`['a']`), text after the value.
fn coerce_value(raw: &str, types: u8) -> Option<String> {
    let t = raw.trim_matches(is_space_char);
    let (t, quoted) = if t.starts_with('"') { (serde_json::from_str::<String>(t).ok()?, true) } else { (t.to_string(), false) };
    let t = t.trim_matches(is_space_char);
    if t.is_empty() || (quoted && t.starts_with('"')) {
        return None; // one level of quoting only
    }
    let b = t.as_bytes();
    if let (St::Ok, e) = scan_json(b, 0, types, false) {
        if e == b.len() {
            return match normalize_int(t, types) {
                Ok(Some(n)) => Some(n),
                Ok(None) => Some(t.to_string()),
                Err(()) => None,
            };
        }
    }
    if types & T_BOOL != 0 {
        if t.eq_ignore_ascii_case("true") {
            return Some("true".into());
        }
        if t.eq_ignore_ascii_case("false") {
            return Some("false".into());
        }
    }
    if types & T_NULL != 0 && t == "None" {
        return Some("null".into());
    }
    None
}

/// One `<tool_call>\n<function=NAME>\n(args)</function>\n</tool_call>`. Returns the state,
/// the end position, and the (possibly partial) call.
/// `str_tail` is set to (q, end) when the call stops inside an open string-only argument
/// value (`arguments` ends with `escape_inner(s[q..end])`), see [`Tail::StrArg`].
fn parse_call(cfg: &ParserConfig, s: &str, p0: usize, partial: bool, opt_open: bool, str_tail: &mut Option<(usize, usize)>) -> (St, usize, Option<ToolCall>) {
    let mut p = p0;
    let (st, q) = lit(s, p, "<tool_call>\n", partial);
    match st {
        St::Ok => p = q,
        St::More => return (St::More, s.len(), None),
        St::Fail if opt_open => {}
        St::Fail => return (St::Fail, p0, None),
    }
    // choice over tools: tool_open is atomic "<function=" NAME ">\n"
    let mut chosen: Option<&Tool> = None;
    let mut any_more = false;
    for t in &cfg.tools {
        let open = format!("<function={}>\n", t.name);
        match lit(s, p, &open, partial) {
            (St::Ok, q) => {
                chosen = Some(t);
                p = q;
                break;
            }
            (St::More, _) => any_more = true,
            (St::Fail, _) => {}
        }
    }
    let Some(tool) = chosen else {
        return if any_more { (St::More, s.len(), None) } else { (St::Fail, p0, None) };
    };
    let mut call = ToolCall { name: tool.name.trim_end().to_string(), arguments: "{".into(), id: String::new() };
    let params = tool_params(tool);
    let n_req = params.iter().filter(|x| x.required).count();
    let mut used = vec![false; params.len()];
    let mut n_args = 0usize;
    let mut req_done = 0usize;
    loop {
        // try an argument (required ones first, until all are seen; then optional ones)
        let mut matched = None;
        let mut more = false;
        for (i, prm) in params.iter().enumerate() {
            // llama.cpp's grammar: required ones first, then optional ones. With
            // `lenient` an unused parameter is also accepted out of that order (the
            // model wrote a valid call in another order); inputs the grammar accepts parse the
            // same, since at one position at most one `<parameter=NAME>` literal matches.
            let in_phase = (if req_done < n_req { prm.required && !used[i] } else { !prm.required }) || (cfg.lenient && !used[i]);
            if !in_phase {
                continue;
            }
            match lit(s, p, &format!("<parameter={}>\n", prm.name), partial) {
                (St::Ok, q) => {
                    matched = Some((i, q));
                    break;
                }
                (St::More, _) => more = true,
                (St::Fail, _) => {}
            }
        }
        let Some((i, q)) = matched else {
            if more {
                return (St::More, s.len(), Some(call));
            }
            break;
        };
        if req_done < n_req && !params[i].required && !cfg.lenient {
            break;
        }
        let prm = &params[i];
        let key = format!("{}{}:", if n_args > 0 { "," } else { "" }, Json::String(prm.name.trim().to_string()));
        let close = PARAM_CLOSE;
        let types = prm.types;
        let string_only = types == T_STR;
        let has_string = types & T_STR != 0;
        call.arguments += &key;
        let mut val_end;
        // Lenient typed coercion (F105): a parameter that cannot be a string gets a second
        // chance when its JSON value fails (see `coerce_value`); string-capable ones never do.
        let coerce = cfg.lenient && !has_string;
        if !has_string || !string_only {
            // JSON alternative (atomic only when a string fallback exists)
            let allowed = if has_string { types & !T_STR } else { types };
            let (jst, je) = scan_json(s.as_bytes(), q, allowed, partial);
            let mut fallback = false;
            match jst {
                St::Ok => match lit(s, je, close, partial) {
                    (St::Ok, ce) => {
                        let v = &s[q..je];
                        match normalize_int(v, types) {
                            Ok(Some(n)) if coerce => call.arguments += &n,
                            _ => call.arguments += v,
                        }
                        val_end = ce;
                    }
                    (St::More, _) => {
                        if has_string {
                            return (St::More, s.len(), Some(call)); // atomic: withheld
                        }
                        call.arguments += &s[q..je];
                        return (St::More, s.len(), Some(call));
                    }
                    (St::Fail, _) => {
                        fallback = coerce;
                        if !has_string && !coerce {
                            return (St::Fail, p0, None);
                        }
                        val_end = usize::MAX; // fall back to string (or coercion)
                    }
                },
                St::More => {
                    if has_string {
                        return (St::More, s.len(), Some(call)); // withheld
                    }
                    call.arguments += &s[q..];
                    return (St::More, s.len(), Some(call));
                }
                St::Fail => {
                    fallback = coerce;
                    if !has_string && !coerce {
                        return (St::Fail, p0, None);
                    }
                    val_end = usize::MAX;
                }
            }
            if fallback {
                // the whole value up to the close tag, atomic: withheld until closed
                let (end, hit) = until_one_of(s, q, &[close.to_string()], partial);
                if hit.is_none() {
                    return if partial { (St::More, s.len(), Some(call)) } else { (St::Fail, p0, None) };
                }
                let Some(v) = coerce_value(&s[q..end], types) else {
                    return (St::Fail, p0, None); // ambiguous: a failure, as before
                };
                call.arguments += &v;
                val_end = end + close.len();
            }
        } else {
            val_end = usize::MAX;
        }
        if val_end == usize::MAX {
            // string value: until("\n</parameter>\n") then close
            let (end, hit) = until_one_of(s, q, &[close.to_string()], partial);
            call.arguments.push('"');
            call.arguments += &escape_inner(&s[q..end]);
            match hit {
                Some(_) => {
                    call.arguments.push('"');
                    val_end = end + close.len();
                }
                None => {
                    if partial {
                        // q is fixed (after the matched "<parameter=..>\n" literal); only a
                        // string-only value is resumable (a JSON alternative is not tried)
                        if string_only {
                            *str_tail = Some((q, end));
                        }
                        return (St::More, s.len(), Some(call));
                    }
                    return (St::Fail, p0, None);
                }
            }
        }
        used[i] = true;
        if prm.required {
            req_done += 1;
        }
        n_args += 1;
        p = val_end;
    }
    if req_done < n_req {
        // permute needs all required args
        return if partial && p >= s.len() { (St::More, s.len(), Some(call)) } else { (St::Fail, p0, None) };
    }
    if !partial && cfg.lenient {
        // repair: the stream ended inside the closing tags (unambiguous: every required
        // argument is complete and only a prefix of the closing tags follows)
        const CLOSING: &str = "</function>\n</tool_call>";
        let r = &s[skip_space(s, p)..];
        if r.len() < CLOSING.len() && CLOSING.starts_with(r) {
            call.arguments.push('}');
            return (St::Ok, s.len(), Some(call));
        }
    }
    match lit(s, p, "</function>\n", partial) {
        (St::Ok, q) => {
            call.arguments.push('}');
            p = q;
        }
        (St::More, _) => return (St::More, s.len(), Some(call)),
        (St::Fail, _) => return (St::Fail, p0, None),
    }
    match lit(s, p, "</tool_call>", partial) {
        (St::Ok, q) => (St::Ok, q, Some(call)),
        (St::More, _) => (St::More, s.len(), Some(call)),
        (St::Fail, _) => (St::Fail, p0, None),
    }
}
