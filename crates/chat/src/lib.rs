//! Chat layer of the oxe server: template rendering ([`template`]), OpenAI request
//! normalization ([`request`]) and model-output parsing / streaming diffs ([`parser`]),
//! each a port of the llama.cpp code that llama-server uses for the same job.

pub mod parser;
pub mod request;
pub mod template;

pub use parser::{compute_diffs, ChatMsg, MsgDiff, ParserConfig, ToolCall};
pub use request::{apply, chat_inputs, Applied, ChatInputs, ReasoningFormat, ServerChatOptions, Tool, ToolChoice};
pub use template::{ChatTemplate, Format};

/// Incremental parse state for one response (`common_chat_session` + `task_result_state`).
pub struct Session {
    pub cfg: ParserConfig,
    pub generation_prompt: String,
    /// generation_prompt + the text fed so far (what the parser runs on)
    full: String,
    msg: ChatMsg,
    /// the last partial parse's raw result and tail (incremental path, see `parser::parse_step`)
    last: Option<(ChatMsg, parser::Tail)>,
    /// false: re-parse the whole text every step (reference behaviour, for tests)
    pub incremental: bool,
    /// steps served by the incremental path / by a full parse
    pub n_incremental: usize,
    pub n_full: usize,
    ids: Vec<String>,
    gen_id: Box<dyn FnMut() -> String + Send>,
}

impl Session {
    pub fn new(cfg: ParserConfig, generation_prompt: String, gen_id: Box<dyn FnMut() -> String + Send>) -> Session {
        let full = generation_prompt.clone();
        Session { cfg, generation_prompt, full, msg: ChatMsg::default(), last: None, incremental: true, n_incremental: 0, n_full: 0, ids: Vec::new(), gen_id }
    }

    pub fn msg(&self) -> &ChatMsg {
        &self.msg
    }

    /// Add generated text; returns the diffs since the previous state.
    pub fn feed(&mut self, chunk: &str, is_final: bool) -> Result<Vec<MsgDiff>, String> {
        self.full.push_str(chunk);
        let partial = !is_final;
        // incremental path: extend the open string of the last parse when the new text stays
        // inside it (identical to a full parse, see parser::parse_step)
        let step = match (&self.last, partial && self.incremental) {
            (Some((prev, tail)), true) => parser::parse_step(&self.cfg, &self.full, tail, prev),
            _ => None,
        };
        let (mut new, complete) = match step {
            Some((m, t, c)) => {
                self.n_incremental += 1;
                self.last = Some((m.clone(), t));
                (m, c)
            }
            None => {
                self.n_full += 1;
                self.last = None;
                let (m, t, c) = parser::parse_full(&self.cfg, &self.full, partial).map_err(|e| e.0)?;
                self.last = t.map(|t| (m.clone(), t));
                (m, c)
            }
        };
        // Stream a tool call only once it is complete. A partial parse's last call can vanish or
        // change when more text arrives (the model writes a parameter the grammar rejects, a
        // truncated stream): llama-server then fails the stream with "Invalid diff: now finding
        // less tool calls!" after the client already received part of the call. Complete calls
        // never change, so the streamed message only grows; the final parse adds the rest.
        new.tool_calls.truncate(complete);
        if new.is_empty() {
            return Ok(Vec::new());
        }
        let g = &mut self.gen_id;
        new.set_tool_call_ids(&mut self.ids, || g());
        match compute_diffs(&self.msg, &new) {
            Ok(d) => {
                self.msg = new;
                Ok(d)
            }
            // not expected any more (the visible message only grows); never fail the stream:
            // hold this step back, the next step or the final message carries the change
            Err(e) => {
                eprintln!("chat: diff held back: {}", e.0);
                if !partial {
                    self.msg = new;
                }
                Ok(Vec::new())
            }
        }
    }
}

/// llama-server `random_string()`: 32 chars of [0-9A-Za-z].
pub fn random_string() -> String {
    const A: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut seed = [0u8; 32];
    if std::fs::File::open("/dev/urandom").and_then(|mut f| std::io::Read::read_exact(&mut f, &mut seed)).is_err() {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (t >> (i % 16 * 8)) as u8 ^ i as u8;
        }
    }
    seed.iter().map(|b| A[(*b as usize) % A.len()] as char).collect()
}

#[cfg(test)]
mod tests;
