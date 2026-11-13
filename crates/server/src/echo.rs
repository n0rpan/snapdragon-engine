//! CACHE-PERFECT ECHO (P1): the model sees exactly what it wrote.
//!
//! After every chat completion that ended on an end-of-generation token, the server keeps the
//! exact token ids of that turn: the generation-prompt opener that was in the prompt (e.g.
//! `<|im_start|>assistant\n<think>\n`) and the generated ids (without the final EOG token). The
//! record is keyed by the SHA-256 of the prompt tokens before the opener (the chain position), so
//! a record is only ever used at the same place of the same conversation.
//!
//! When a later request's rendered prompt holds that assistant turn, [`EchoStore::splice`] puts
//! the stored ids in place of the re-rendered span (from the assistant `<|im_start|>` up to, not
//! including, its `<|im_end|>`). The rendered prompt then continues exactly where the generation
//! stopped, so the prefix cache keeps every token whatever the template or the client did to the
//! turn (an opener the history form leaves out, JSON arguments in another key order, a BPE split
//! the tokenizer would not produce, reasoning the client did not send back).
//!
//! Safety: a record is spliced only when the request's message is the same turn as parsed from
//! the generation: equal content (both trimmed, as the template trims), equal tool calls (name
//! and arguments as JSON values: key order does not matter, values do), and equal reasoning when
//! the client sends reasoning. Otherwise the render is used unchanged and the turn is counted
//! under a miss class. A TTL (default 7 days, O-96) and a token budget bound the records; P2 (M3)
//! keeps them on disk next to the state cache (`attach`), so they survive a restart or a switch.
//!
//! C2 coverage (Codex P1 review 3), stated exactly: the round trip is checked only when a LATER
//! request re-renders a RECORDED turn, i.e. a turn that ended on EOG behind a generation prompt
//! and whose message the client sent back unchanged (a splice hit compares the render with the
//! generated ids: `echo_fixed`, `roundtrip_miss`). It is NOT checked at completion time, and turns
//! cut by a stop word, the length limit, a cancel or an error are never recorded, so they are never
//! checked. The timings say so (`roundtrip_scope`); this is not an all-completion C2 proof. The
//! P4 gate runs the real Ornith and Qwen templates and tokenizers on top of these synthetic tests.
//!
//! Privacy: records hold token ids and message text of the conversation, in process memory
//! only; nothing of them is logged (counts and enum labels only).

use crate::prefix_store::Sha256;
use chat::ChatMsg;
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use std::time::Duration;
use tokenizer::Token;

/// Miss classes of an assistant turn that was not spliced.
pub const MISS_NO_RECORD: &str = "no-record";
pub const MISS_CONTENT: &str = "content-mismatch";
pub const MISS_TOOL: &str = "tool-mismatch";
pub const MISS_REASONING: &str = "reasoning-mismatch";
pub const MISS_LAYOUT: &str = "layout";

#[derive(Clone, Debug)]
pub struct EchoConfig {
    pub enabled: bool,
    /// opener + generated tokens kept in total; the oldest records go first
    pub max_tokens: usize,
    pub ttl: Duration,
    /// v0.1.3: a session's echo file is rewritten with only its live records once it is at least
    /// this large and at least twice their size (`--echo-compact-bytes`, 0 = never)
    pub compact_bytes: u64,
}

impl Default for EchoConfig {
    fn default() -> Self {
        EchoConfig { enabled: true, max_tokens: 4 << 20, ttl: Duration::from_secs(7 * 24 * 3600), compact_bytes: 4 << 20 }
    }
}

impl EchoConfig {
    /// `OXE_ECHO=0|off` disables, `OXE_ECHO_MAX_TOKENS`, `OXE_ECHO_TTL_SECS` (set by the
    /// `--echo`, `--echo-max-tokens`, `--echo-ttl-secs` flags).
    pub fn from_env() -> EchoConfig {
        let mut c = EchoConfig::default();
        if let Ok(v) = std::env::var("OXE_ECHO") {
            c.enabled = !matches!(v.as_str(), "0" | "off" | "false" | "no");
        }
        if let Some(n) = std::env::var("OXE_ECHO_MAX_TOKENS").ok().and_then(|v| v.parse().ok()) {
            c.max_tokens = n;
        }
        if let Some(s) = std::env::var("OXE_ECHO_TTL_SECS").ok().and_then(|v| v.parse::<u64>().ok()) {
            c.ttl = Duration::from_secs(s);
        }
        if let Some(b) = std::env::var("OXE_ECHO_COMPACT_BYTES").ok().and_then(|v| v.parse::<u64>().ok()) {
            c.compact_bytes = b;
        }
        c
    }
}

/// File stem of an eviction group's echo file (P2b, one naming scheme with the manifests' `hint`
/// field and the controller's GC): a session group (`s:<hint>`) is `hint_hash(hint)`, exactly the
/// manifest's hint; a group without a session hint (`k:<key>`) hashes the group name.
fn group_stem(group: &str) -> String {
    match group.strip_prefix("s:") {
        Some(h) => crate::state_store::hint_hash(h),
        None => crate::state_store::hint_hash(group),
    }
}

/// one session's echo file larger than this is not read back (records are a few KiB each)
pub const MAX_ECHO_FILE_BYTES: u64 = 256 << 20;

struct Record {
    id: u64,
    prefix_len: usize,
    opener: Vec<Token>,
    body: Vec<Token>,
    content: String,
    reasoning: String,
    tools: Vec<(String, Json)>,
    /// session hint (`X-Oxe-Session`); never part of the key
    session: Option<String>,
    /// eviction group: the session hint, else the record's own key
    group: String,
    /// wall-clock ms when made
    at_ms: u64,
}

impl Record {
    fn n_tokens(&self) -> usize {
        self.opener.len() + self.body.len()
    }
}

/// What one splice did, for the timings and the ledger (numbers and enum labels only).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpliceReport {
    /// assistant turns in the request's history
    pub blocks: usize,
    /// turns replaced by their echo record
    pub hits: usize,
    /// of those, turns whose render differed from the generated ids (a C2 round-trip miss
    /// that echo repaired)
    pub fixed: usize,
    /// first divergent token offset (inside the turn) of the first repaired turn; checked for
    /// spliced turns only (see the module docs: C2 coverage)
    pub roundtrip_miss: Option<usize>,
    /// per-class counts of the turns that were not spliced
    pub misses: Vec<(&'static str, usize)>,
    /// tokens taken from records
    pub echo_tokens: usize,
    /// spliced spans (start, end in the render, length fed): maps render offsets to prompt offsets
    pub edits: Vec<(usize, usize, usize)>,
}

impl SpliceReport {
    fn miss(&mut self, class: &'static str) {
        match self.misses.iter_mut().find(|(c, _)| *c == class) {
            Some((_, n)) => *n += 1,
            None => self.misses.push((class, 1)),
        }
    }
    /// The class of the latest unspliced turn that matters most: a mismatch before no-record.
    pub fn miss_class(&self) -> Option<&'static str> {
        for c in [MISS_LAYOUT, MISS_CONTENT, MISS_TOOL, MISS_REASONING, MISS_NO_RECORD] {
            if self.misses.iter().any(|(k, _)| *k == c) {
                return Some(c);
            }
        }
        None
    }
    /// The prompt offset of render offset `p` (a position outside every spliced span, or its start).
    pub fn map_pos(&self, p: usize) -> usize {
        let mut q = p as isize;
        for &(s, e, n) in &self.edits {
            if p >= e && e > s {
                q += n as isize - (e - s) as isize;
            }
        }
        q.max(0) as usize
    }
    pub fn to_json(&self) -> Json {
        let misses: serde_json::Map<String, Json> = self.misses.iter().map(|(c, n)| (c.to_string(), json!(n))).collect();
        json!({
            "echo_blocks": self.blocks, "echo_n": self.hits, "echo_fixed": self.fixed,
            "echo_tokens": self.echo_tokens, "roundtrip_miss": self.roundtrip_miss,
            "echo_miss_class": self.miss_class(), "echo_misses": misses,
            "roundtrip_scope": "spliced-eog-turns",
        })
    }
}

fn hash_tokens(h: &mut Sha256, t: &[Token]) {
    let mut b = Vec::with_capacity(t.len() * 4);
    for x in t {
        b.extend_from_slice(&x.to_le_bytes());
    }
    h.update(&b);
}

/// Hasher of the chain-position key: the store's identity (M4: model, numerics epoch, state
/// ABI ...; empty in tests), then the token ids (u32 little endian).
fn key_hasher(identity: &str) -> Sha256 {
    let mut h = Sha256::default();
    if !identity.is_empty() {
        h.update(&(identity.len() as u64).to_le_bytes());
        h.update(identity.as_bytes());
    }
    h
}

/// SHA-256 (hex) of `identity` and the token ids: the chain-position key.
pub fn prefix_key(identity: &str, t: &[Token]) -> String {
    let mut h = key_hasher(identity);
    hash_tokens(&mut h, t);
    h.hex()
}

/// Wall-clock milliseconds (records outlive the process in P2).
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn first_diff(a: &[Token], b: &[Token]) -> usize {
    a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()))
}

/// Tool arguments as a JSON value for the equality check. A text whose numbers may not survive a
/// parse exactly (an integer beyond 64 bits, a decimal with more than 15 significant digits: both
/// can compare equal to a different number) stays text (v0.1.3, review-carried): then only the
/// same text matches, a re-serialised one is a miss, never a false hit.
fn parse_args(s: &str) -> Json {
    match serde_json::from_str::<Json>(s) {
        Ok(v) if numbers_exact(s) => v,
        _ => Json::String(s.to_string()),
    }
}

/// Is every number literal of the JSON text `s` exact after a parse (a u64 or i64, or a decimal of
/// at most 15 significant digits)? Strings are skipped.
fn numbers_exact(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'-' | b'0'..=b'9' => {
                let st = i;
                i += 1;
                while i < b.len() && matches!(b[i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-') {
                    i += 1;
                }
                let lit = &s[st..i];
                // a decimal of at most 15 significant digits maps to one f64 no other such decimal
                // maps to (DBL_DIG); more digits (or an integer past 64 bits) can collide
                let ok = lit.parse::<u64>().is_ok() || lit.parse::<i64>().is_ok() || sig_digits(lit) <= 15;
                if !ok {
                    return false;
                }
            }
            _ => i += 1,
        }
    }
    true
}

/// Significant digits of a JSON number literal's mantissa (leading and trailing zeros dropped).
fn sig_digits(lit: &str) -> usize {
    let m = lit.split(['e', 'E']).next().unwrap_or("");
    let d: String = m.chars().filter(|c| c.is_ascii_digit()).collect();
    d.trim_start_matches('0').trim_end_matches('0').len()
}

/// Reasoning field names clients use besides `reasoning_content` (v0.1.3, review-carried).
const REASONING_ALIASES: &[&str] = &["reasoning", "thinking", "reasoning_text"];

/// Text of a message's content as the template sees it (string, or the text parts joined).
fn content_text(m: &Json) -> String {
    match m.get("content") {
        Some(Json::String(s)) => s.clone(),
        Some(Json::Array(parts)) => parts.iter().filter_map(|p| p.get("text").and_then(Json::as_str)).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// Is the request's (normalized) assistant message the turn this record holds? `sent` is the
/// `reasoning_content` as the client sent it (None: field absent or null).
fn same_turn(r: &Record, m: &Json, sent: Option<&str>) -> Result<(), &'static str> {
    if content_text(m).trim() != r.content {
        return Err(MISS_CONTENT);
    }
    let calls: Vec<(String, Json)> = m
        .get("tool_calls")
        .and_then(Json::as_array)
        .map(|a| {
            a.iter()
                .map(|tc| {
                    let f = tc.get("function").unwrap_or(tc);
                    let name = f.get("name").and_then(Json::as_str).unwrap_or("").to_string();
                    let args = match f.get("arguments") {
                        Some(Json::String(s)) => parse_args(s),
                        Some(a) => a.clone(),
                        None => Json::Null,
                    };
                    (name, args)
                })
                .collect()
        })
        .unwrap_or_default();
    if calls != r.tools {
        return Err(MISS_TOOL);
    }
    // reasoning (Codex P1 review 1): a field the client did not send (absent or null) may be
    // restored from the record (O-96: reasoning stays until compaction); a string the client sent,
    // the empty one included, is its edit (an explicit deletion is "") and must equal the record
    let rc = match sent {
        Some(x) => Some(x),
        None => m.get("reasoning_content").and_then(Json::as_str),
    };
    if let Some(rc) = rc {
        if rc.trim() != r.reasoning {
            return Err(MISS_REASONING);
        }
    }
    // the same reasoning under another name: a string must equal the record, anything else (an
    // object, a list) cannot be compared and is a miss; absent or null may be restored
    for k in REASONING_ALIASES {
        match m.get(*k) {
            None | Some(Json::Null) => {}
            Some(Json::String(x)) if x.trim() == r.reasoning => {}
            Some(_) => return Err(MISS_REASONING),
        }
    }
    Ok(())
}

/// One eviction group (a session): M4 evicts whole sessions, least recently used first, and a
/// splice hit refreshes its session, so dropping turn 1 of a live session (which costs that whole
/// session) no longer happens while another, idle session holds records.
#[derive(Default)]
struct Group {
    used: u64,
    n_tokens: usize,
}

/// One record as a disk line: JSON (identity hash, key, ids, parsed turn, time) + tab + Check64.
fn record_line(identity: &str, key: &str, r: &Record) -> String {
    let j = json!({
        "v": 1, "id": crate::prefix_store::sha256_hex(identity.as_bytes()), "key": key, "prefix_len": r.prefix_len,
        "opener": r.opener, "body": r.body, "content": r.content, "reasoning": r.reasoning,
        "tools": r.tools.iter().map(|(n, a)| json!([n, a])).collect::<Vec<_>>(),
        "session": r.session, "group": r.group, "at_ms": r.at_ms,
    })
    .to_string();
    let sum = crate::prefix_store::check64(j.as_bytes());
    format!("{j}\t{sum:016x}\n")
}

fn parse_line(line: &str, idh: &str) -> Option<(Record, String)> {
    let (j, sum) = line.rsplit_once('\t')?;
    if u64::from_str_radix(sum, 16).ok()? != crate::prefix_store::check64(j.as_bytes()) {
        return None;
    }
    let v: Json = serde_json::from_str(j).ok()?;
    if v.get("v")?.as_u64()? != 1 || v.get("id")?.as_str()? != idh {
        return None;
    }
    let toks = |k: &str| -> Option<Vec<Token>> { v.get(k)?.as_array()?.iter().map(|x| x.as_u64().and_then(|t| Token::try_from(t).ok())).collect() };
    let tools = v.get("tools")?.as_array()?.iter().map(|t| Some((t.get(0)?.as_str()?.to_string(), t.get(1)?.clone()))).collect::<Option<Vec<_>>>()?;
    let rec = Record {
        id: 0,
        prefix_len: v.get("prefix_len")?.as_u64()? as usize,
        opener: toks("opener")?,
        body: toks("body")?,
        content: v.get("content")?.as_str()?.to_string(),
        reasoning: v.get("reasoning")?.as_str()?.to_string(),
        tools,
        session: v.get("session").and_then(Json::as_str).map(str::to_string),
        group: v.get("group")?.as_str()?.to_string(),
        at_ms: v.get("at_ms")?.as_u64()?,
    };
    Some((rec, v.get("key")?.as_str()?.to_string()))
}

pub struct EchoStore {
    pub cfg: EchoConfig,
    /// identity in every key (M4): records of another model, state ABI or numerics epoch never match
    identity: String,
    by_key: HashMap<String, Vec<Record>>,
    groups: HashMap<String, Group>,
    clock: u64,
    n_tokens: usize,
    next_id: u64,
    /// M3: records on disk (`<state dir>/<model>/echo/<group hash>.jsonl`), None = RAM only
    dir: Option<std::path::PathBuf>,
    /// v0.1.3 compaction: the file size of a group at which its file is checked again
    next_check: HashMap<String, u64>,
    /// files rewritten by compaction (diagnostics, tests)
    pub compactions: u64,
}

impl EchoStore {
    pub fn new(cfg: EchoConfig) -> EchoStore {
        EchoStore { cfg, identity: String::new(), by_key: HashMap::new(), groups: HashMap::new(), clock: 0, n_tokens: 0, next_id: 0, dir: None, next_check: HashMap::new(), compactions: 0 }
    }

    /// Set the identity mixed into every key (drops records made under another identity).
    pub fn set_identity(&mut self, identity: &str) {
        if self.identity != identity {
            self.clear();
            self.identity = identity.to_string();
        }
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn n_records(&self) -> usize {
        self.by_key.values().map(Vec::len).sum()
    }

    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    /// Eviction groups held (one per session with records).
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    /// Records of one session hint (diagnostics).
    pub fn n_session_records(&self, session: &str) -> usize {
        self.by_key.values().flatten().filter(|r| r.session.as_deref() == Some(session)).count()
    }

    /// Keep one finished turn. `prompt` is the prompt that was fed (after splicing), `asst` the
    /// token ids of the assistant opener (`<|im_start|>assistant`), `generated` the generated ids
    /// with the EOG token last. Turns that did not end on EOG (stop word, length, cancel) are not
    /// kept: their re-render ends differently from what was generated. Returns true if kept.
    /// `im_end` is the turn end id: the opener must be the generation prompt alone. A prompt
    /// without one (`add_generation_prompt: false`), or one whose last opener text sits inside a
    /// user message, has a turn end after its last opener; such a record would hold other
    /// messages and is never made.
    pub fn record(&mut self, prompt: &[Token], asst: &[Token], im_end: Token, msg: &ChatMsg, generated: &[Token], ended_on_eog: bool, session: Option<&str>) -> bool {
        if !self.cfg.enabled || !ended_on_eog || generated.is_empty() || asst.is_empty() {
            return false;
        }
        // a turn whose own text holds the turn delimiters would make the span search ambiguous
        let delim = |s: &str| s.contains("<|im_start|>") || s.contains("<|im_end|>");
        if delim(&msg.content) || delim(&msg.reasoning_content) || msg.tool_calls.iter().any(|t| delim(&t.arguments) || delim(&t.name)) {
            return false;
        }
        let Some(start) = prompt.windows(asst.len()).rposition(|w| w == asst) else { return false };
        if prompt[start..].contains(&im_end) {
            return false;
        }
        let key = prefix_key(&self.identity, &prompt[..start]);
        let group = match session {
            Some(s) => format!("s:{s}"),
            None => format!("k:{key}"),
        };
        let rec = Record {
            id: 0,
            prefix_len: start,
            opener: prompt[start..].to_vec(),
            body: generated[..generated.len() - 1].to_vec(),
            content: msg.content.trim().to_string(),
            reasoning: msg.reasoning_content.trim().to_string(),
            tools: msg.tool_calls.iter().map(|t| (t.name.clone(), parse_args(&t.arguments))).collect(),
            session: session.map(str::to_string),
            group,
            at_ms: now_ms(),
        };
        let line = self.dir.is_some().then(|| record_line(&self.identity, &key, &rec));
        let group = rec.group.clone();
        let kept = self.insert(key, rec);
        if let (true, Some(line)) = (kept, line) {
            self.append(&group, &line);
            self.maybe_compact(&group);
        }
        kept
    }

    /// v0.1.3 (lane B item 4): the echo files are append-only, so a regenerate loop, replaced
    /// records and expired ones grow them. Once a group's file is at least `compact_bytes` and at
    /// least twice its live records, it is rewritten with exactly the live records, oldest first
    /// (the order `attach` reads them back in), atomically (`.pending`, fsync, rename, directory
    /// fsync). Records in RAM are unchanged, so a splice after a restart feeds exactly the ids it
    /// fed before. Returns whether the file was rewritten.
    pub fn maybe_compact(&mut self, group: &str) -> bool {
        let lim = self.cfg.compact_bytes;
        let Some(p) = self.group_file(group) else { return false };
        if lim == 0 {
            return false;
        }
        let Ok(md) = std::fs::symlink_metadata(&p) else { return false };
        if !md.is_file() || md.len() < self.next_check.get(group).copied().unwrap_or(0).max(lim) {
            return false;
        }
        let mut recs: Vec<(u64, String)> = Vec::new();
        for (k, v) in &self.by_key {
            for r in v.iter().filter(|r| r.group == group) {
                recs.push((r.id, record_line(&self.identity, k, r)));
            }
        }
        recs.sort_by_key(|r| r.0);
        let body: String = recs.into_iter().map(|r| r.1).collect();
        let live = body.len() as u64;
        // check again only once the file has grown to twice the live records
        self.next_check.insert(group.to_string(), live.saturating_mul(2));
        if live.saturating_mul(2) > md.len() {
            return false;
        }
        let mut pending = p.clone().into_os_string();
        pending.push(".pending");
        let pending = std::path::PathBuf::from(pending);
        let r = (|| -> std::io::Result<()> {
            use std::io::Write;
            use std::os::unix::fs::OpenOptionsExt;
            let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(&pending)?;
            f.write_all(body.as_bytes())?;
            f.sync_all()?;
            std::fs::rename(&pending, &p)?;
            if let Some(d) = p.parent().and_then(|d| std::fs::File::open(d).ok()) {
                let _ = d.sync_all();
            }
            Ok(())
        })();
        match r {
            Ok(()) => {
                self.compactions += 1;
                true
            }
            Err(e) => {
                let _ = std::fs::remove_file(&pending);
                eprintln!("oxe-server: echo records: compaction failed: {e}");
                false
            }
        }
    }

    /// M3: keep records on disk under `dir` (0700, files 0600) and read back the ones of this
    /// identity that are within the TTL. A thinking-on session that continues after an engine
    /// restart or a model switch then still gets its own reasoning back (no empty think blocks).
    /// Lines are `<json>\t<check64 hex>`; a torn or altered line is skipped. Returns how many
    /// records were read.
    pub fn attach(&mut self, dir: &std::path::Path) -> std::io::Result<usize> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::create_dir_all(dir)?;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
        self.dir = Some(dir.to_path_buf());
        let mut n = 0;
        let idh = crate::prefix_store::sha256_hex(self.identity.as_bytes());
        for de in std::fs::read_dir(dir)?.flatten() {
            let p = de.path();
            if p.extension().and_then(|e| e.to_str()) != Some("jsonl") || de.file_type().map_or(true, |t| !t.is_file()) {
                continue;
            }
            // bounded (Codex P2/P3 review): an echo file over the cap is skipped unread, never parsed whole
            if de.metadata().map_or(true, |m| m.len() > MAX_ECHO_FILE_BYTES) {
                eprintln!("oxe-server: echo records: {} over {} MiB, skipped", p.display(), MAX_ECHO_FILE_BYTES >> 20);
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            if !text.is_empty() && !text.ends_with('\n') {
                // a line torn by a kill: end it, or the next record would be appended to it and lost
                self.append_path(&p, "\n");
            }
            let (mut live, mut foreign) = (0, false);
            for line in text.lines() {
                let Some((rec, key)) = parse_line(line, &idh) else {
                    // another identity's (or a torn) line: the file is not ours to delete
                    foreign |= !line.contains(&idh);
                    continue;
                };
                if self.insert(key, rec) {
                    live += 1;
                    n += 1;
                }
            }
            if live == 0 && !foreign {
                let _ = std::fs::remove_file(&p);
            }
        }
        Ok(n)
    }

    fn group_file(&self, group: &str) -> Option<std::path::PathBuf> {
        self.dir.as_ref().map(|d| d.join(format!("{}.jsonl", group_stem(group))))
    }

    fn append(&self, group: &str, line: &str) {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let Some(p) = self.group_file(group) else { return };
        self.append_path(&p, line)
    }

    fn append_path(&self, p: &std::path::Path, line: &str) {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let r = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).custom_flags(libc::O_NOFOLLOW).open(p).and_then(|mut f| {
            f.write_all(line.as_bytes())?;
            f.sync_data()
        });
        if let Err(e) = r {
            eprintln!("oxe-server: echo record not written: {e}");
        }
    }

    /// Add a record (made now or read back from disk); the budget and the TTL apply.
    fn insert(&mut self, key: String, mut rec: Record) -> bool {
        if rec.n_tokens() > self.cfg.max_tokens || self.expired(&rec, now_ms()) {
            return false;
        }
        rec.id = self.next_id;
        self.next_id += 1;
        // a retry that produced the same turn replaces the older record (Codex P1 review 2: no
        // bookkeeping is left behind for it)
        if let Some(v) = self.by_key.get_mut(&key) {
            let same = |r: &Record| r.content == rec.content && r.tools == rec.tools && r.reasoning == rec.reasoning;
            let gone: Vec<(String, usize)> = v.iter().filter(|r| same(r)).map(|r| (r.group.clone(), r.n_tokens())).collect();
            v.retain(|r| !same(r));
            for (g, n) in gone {
                self.untrack(&g, n);
            }
        }
        self.clock += 1;
        let g = self.groups.entry(rec.group.clone()).or_default();
        g.used = self.clock;
        g.n_tokens += rec.n_tokens();
        self.n_tokens += rec.n_tokens();
        let protect = rec.group.clone();
        self.by_key.entry(key).or_default().push(rec);
        self.evict_protecting(Some(&protect));
        true
    }

    fn untrack(&mut self, group: &str, n: usize) {
        self.n_tokens -= n;
        if let Some(g) = self.groups.get_mut(group) {
            g.n_tokens -= n;
            if g.n_tokens == 0 {
                self.groups.remove(group);
            }
        }
    }

    fn expired(&self, r: &Record, now: u64) -> bool {
        let ttl = self.cfg.ttl.as_millis() as u64;
        ttl == 0 || now.saturating_sub(r.at_ms) > ttl
    }

    fn evict(&mut self) {
        self.evict_protecting(None)
    }

    /// TTL first, then whole sessions, least recently used first (M4); `protect` (the session that
    /// was just written) goes last, and then only its oldest records.
    fn evict_protecting(&mut self, protect: Option<&str>) {
        let now = now_ms();
        let mut gone: Vec<(String, usize)> = Vec::new();
        let ttl_ms = self.cfg.ttl.as_millis() as u64;
        for v in self.by_key.values_mut() {
            v.retain(|r| {
                let dead = ttl_ms == 0 || now.saturating_sub(r.at_ms) > ttl_ms;
                if dead {
                    gone.push((r.group.clone(), r.n_tokens()));
                }
                !dead
            });
        }
        for (g, n) in gone {
            self.untrack(&g, n);
        }
        while self.n_tokens > self.cfg.max_tokens {
            let victim = self.groups.iter().filter(|(k, _)| Some(k.as_str()) != protect).min_by_key(|(_, g)| g.used).map(|(k, _)| k.clone());
            match victim {
                Some(g) => self.drop_group(&g),
                None => {
                    // only the protected session is left: its oldest record goes
                    let oldest = self.by_key.iter().flat_map(|(k, v)| v.iter().map(move |r| (r.id, k.clone()))).min();
                    let Some((id, k)) = oldest else { break };
                    self.remove(&k, id);
                }
            }
        }
        self.by_key.retain(|_, v| !v.is_empty());
    }

    /// Drop every record of one session (eviction group), on disk too.
    fn drop_group(&mut self, group: &str) {
        if let Some(p) = self.group_file(group) {
            let _ = std::fs::remove_file(p);
        }
        self.next_check.remove(group);
        for v in self.by_key.values_mut() {
            v.retain(|r| r.group != group);
        }
        self.by_key.retain(|_, v| !v.is_empty());
        if let Some(g) = self.groups.remove(group) {
            self.n_tokens -= g.n_tokens;
        }
    }

    fn remove(&mut self, k: &str, id: u64) {
        let Some(v) = self.by_key.get_mut(k) else { return };
        let Some(i) = v.iter().position(|r| r.id == id) else { return };
        let r = v.remove(i);
        if v.is_empty() {
            self.by_key.remove(k);
        }
        self.untrack(&r.group, r.n_tokens());
    }

    /// Drop every record of the session hint whose [`crate::state_store::hint_hash`] is `h`, in
    /// RAM and its file (session clear). Returns the records dropped.
    pub fn drop_hint_hash(&mut self, h: &str) -> usize {
        let groups: Vec<String> = self.groups.keys().filter(|g| group_stem(g) == h).cloned().collect();
        let n0 = self.n_records();
        for g in groups {
            self.drop_group(&g);
        }
        if let Some(d) = &self.dir {
            if h.len() == 32 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
                let _ = std::fs::remove_file(d.join(format!("{h}.jsonl")));
            }
        }
        n0 - self.n_records()
    }

    /// Stop (`None`) or resume (`Some(dir)`) writing records to disk; RAM records stay. Budget 0
    /// means no conversation state on disk (Codex P2/P3 review 2). Returns the previous directory.
    pub fn set_disk(&mut self, dir: Option<std::path::PathBuf>) -> Option<std::path::PathBuf> {
        std::mem::replace(&mut self.dir, dir)
    }

    pub fn disk_dir(&self) -> Option<&std::path::Path> {
        self.dir.as_deref()
    }

    /// Drop every record (clear cache), on disk too.
    pub fn clear(&mut self) {
        if let Some(rd) = self.dir.as_ref().and_then(|d| std::fs::read_dir(d).ok()) {
            for de in rd.flatten() {
                if de.path().extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    let _ = std::fs::remove_file(de.path());
                }
            }
        }
        self.by_key.clear();
        self.groups.clear();
        self.n_tokens = 0;
    }

    /// Replace each assistant turn of `rendered` that has a matching record by the record's ids.
    /// `msgs` are the normalized messages the prompt was rendered from, `sent` the client's
    /// `reasoning_content` per message as sent (see `ChatInputs::reasoning_sent`), `spans` the
    /// token range of each history assistant turn in `rendered`: from its `<|im_start|>` to its
    /// `<|im_end|>` (exclusive), taken from the renderer's own delimiter offsets (M1: delimiter
    /// text inside a message is not counted). None = a layout the caller could not map. Returns
    /// the prompt to feed and a report. With no records or an unknown layout the render comes
    /// back unchanged (never a guess).
    pub fn splice(&mut self, rendered: &[Token], msgs: &Json, sent: &[Option<String>], spans: Option<&[(usize, usize)]>) -> (Vec<Token>, SpliceReport) {
        let mut rep = SpliceReport::default();
        let turns: Vec<(usize, &Json)> = msgs.as_array().map(|a| a.iter().enumerate().filter(|(_, m)| m.get("role").and_then(Json::as_str) == Some("assistant")).collect()).unwrap_or_default();
        rep.blocks = turns.len();
        if turns.is_empty() {
            return (rendered.to_vec(), rep);
        }
        if !self.cfg.enabled {
            for _ in &turns {
                rep.miss(MISS_NO_RECORD);
            }
            return (rendered.to_vec(), rep);
        }
        self.evict();
        let spans = match spans {
            Some(sp) if sp.len() == turns.len() && sp.iter().all(|&(s, e)| s < e && e <= rendered.len()) && sp.windows(2).all(|w| w[0].1 <= w[1].0) => sp,
            _ => {
                for _ in &turns {
                    rep.miss(MISS_LAYOUT);
                }
                return (rendered.to_vec(), rep);
            }
        };
        // no records (a restart: P1 keeps them in RAM only): skip the prompt hash
        if self.by_key.is_empty() {
            for _ in &turns {
                rep.miss(MISS_NO_RECORD);
            }
            return (rendered.to_vec(), rep);
        }
        let mut out: Vec<Token> = Vec::with_capacity(rendered.len() + 64);
        let mut h = key_hasher(&self.identity);
        let mut cursor = 0usize;
        for (i, &(mi, m)) in turns.iter().enumerate() {
            let (s, e) = spans[i];
            let sent_r = sent.get(mi).and_then(|x| x.as_deref());
            hash_tokens(&mut h, &rendered[cursor..s]);
            out.extend_from_slice(&rendered[cursor..s]);
            cursor = s;
            let key = h.clone().hex();
            let found = match self.by_key.get(&key) {
                None => Err(MISS_NO_RECORD),
                Some(v) => {
                    let mut class = MISS_NO_RECORD;
                    let mut hit = None;
                    // newest first: after a regenerate the live KV holds the latest of two turns
                    // that parse to the same message (same answer, other reasoning)
                    for r in v.iter().rev().filter(|r| r.prefix_len == out.len()) {
                        match same_turn(r, m, sent_r) {
                            Ok(()) => {
                                hit = Some(([r.opener.as_slice(), r.body.as_slice()].concat(), r.group.clone()));
                                break;
                            }
                            Err(c) if class == MISS_NO_RECORD => class = c,
                            Err(_) => {}
                        }
                    }
                    hit.ok_or(class)
                }
            };
            match found {
                Ok((ids, group)) => {
                    // M4: a hit refreshes its session
                    self.clock += 1;
                    if let Some(g) = self.groups.get_mut(&group) {
                        g.used = self.clock;
                    }
                    let span = &rendered[s..e];
                    rep.hits += 1;
                    if span != ids.as_slice() {
                        rep.fixed += 1;
                        rep.roundtrip_miss.get_or_insert(first_diff(span, &ids));
                    }
                    rep.echo_tokens += ids.len();
                    rep.edits.push((s, e, ids.len()));
                    hash_tokens(&mut h, &ids);
                    out.extend_from_slice(&ids);
                    cursor = e;
                }
                Err(c) => rep.miss(c),
            }
        }
        out.extend_from_slice(&rendered[cursor..]);
        (out, rep)
    }
}
