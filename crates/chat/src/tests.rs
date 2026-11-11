//! Differential tests against fixtures produced by llama.cpp itself (tools/make_fixtures.py
//! drives crates/server/oracle, which calls llama-server's `oaicompat_chat_params_parse`,
//! `common_chat_session::feed/finish` and `compute_diffs`).

use super::*;
use serde_json::{json, Value as Json};
use std::path::PathBuf;

#[path = "tests_coerce.rs"]
mod coerce;

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local/share/ai-stack/models/llm")
}

struct Ctx {
    tok: tokenizer::Tokenizer,
    tmpl: ChatTemplate,
}

fn load(gguf: &str) -> Option<Ctx> {
    let p = models_dir().join(gguf);
    if !p.exists() {
        eprintln!("skip: {p:?} missing");
        return None;
    }
    let tok = tokenizer::Tokenizer::from_gguf(&p).unwrap();
    let bos = tok.bos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
    let eos = tok.eos.and_then(|t| tok.token_text(t)).unwrap_or("").to_string();
    let tmpl = ChatTemplate::new(tok.chat_template.as_deref().unwrap(), &bos, &eos).unwrap();
    Some(Ctx { tok, tmpl })
}

fn opts(kwargs: &Json) -> ServerChatOptions {
    let mut o = ServerChatOptions::default();
    if let Json::Object(m) = kwargs {
        o.chat_template_kwargs = m.clone();
        if let Some(Json::Bool(b)) = m.get("enable_thinking") {
            o.enable_thinking = *b;
        }
    }
    o
}

fn parser_cfg(c: &Ctx, inp: &ChatInputs) -> ParserConfig {
    ParserConfig {
        format: c.tmpl.format,
        tools: inp.tools.clone(),
        tool_choice: inp.tool_choice,
        parallel_tool_calls: inp.parallel_tool_calls,
        reasoning_format: inp.reasoning_format,
        supports_thinking: c.tmpl.supports_thinking,
        lenient: false,
    }
}

/// llama.cpp fixtures where a tool call is streamed and then fails to parse (see run_fixtures).
const DIVERGE_FROM_LLAMA: &[&str] = &["parse:optional_first", "parse:missing_required", "parse:truncated_mid_value", "parse:int_with_spaces", "parse:int_invalid"];

/// The client's view of a stream: content and reasoning concatenated, tool calls assembled by
/// index (name, arguments; ids ignored), from a list of per-step delta arrays.
fn assemble(steps: &[Json]) -> Json {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut calls: Vec<(String, String)> = Vec::new();
    for st in steps {
        for d in st.as_array().into_iter().flatten() {
            if let Some(c) = d.get("content").and_then(Json::as_str) {
                content += c;
            }
            if let Some(c) = d.get("reasoning_content").and_then(Json::as_str) {
                reasoning += c;
            }
            for tc in d.get("tool_calls").and_then(Json::as_array).into_iter().flatten() {
                let i = tc["index"].as_u64().unwrap() as usize;
                while calls.len() <= i {
                    calls.push(Default::default());
                }
                if let Some(n) = tc.pointer("/function/name").and_then(Json::as_str) {
                    calls[i].0 += n;
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(Json::as_str) {
                    calls[i].1 += a;
                }
            }
        }
    }
    json!({"content": content, "reasoning": reasoning, "tool_calls": calls})
}

/// A step's deltas without tool-call deltas (those are held back until a call is complete).
fn non_tool(st: &Json) -> Json {
    Json::Array(st.as_array().into_iter().flatten().filter(|d| d.get("tool_calls").is_none()).cloned().collect())
}

fn run_fixtures(gguf: &str, fixture: &str) {
    let Some(c) = load(gguf) else { return };
    let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
    let data = std::fs::read_to_string(path).unwrap();
    let mut fails = Vec::new();
    let mut diverged = 0;
    let mut n = 0;
    for line in data.lines() {
        let fx: Json = serde_json::from_str(line).unwrap();
        let name = fx["name"].as_str().unwrap().to_string();
        let exp = &fx["expect"];
        n += 1;
        let inp = match chat_inputs(&fx["body"], &opts(&fx["kwargs"]), &c.tmpl) {
            Ok(i) => i,
            Err(e) => {
                fails.push(format!("{name}: inputs error {e}"));
                continue;
            }
        };
        let applied = match apply(&c.tmpl, &inp) {
            Ok(a) => a,
            Err(e) => {
                fails.push(format!("{name}: render error {e:#}"));
                continue;
            }
        };
        if exp.get("error").is_some() {
            fails.push(format!("{name}: oracle error {}", exp["error"]));
            continue;
        }
        if applied.prompt != exp["prompt"].as_str().unwrap() {
            fails.push(format!("{name}: PROMPT\n--- want\n{}\n--- got\n{}", exp["prompt"].as_str().unwrap(), applied.prompt));
            continue;
        }
        if applied.generation_prompt != exp["generation_prompt"].as_str().unwrap() {
            fails.push(format!("{name}: generation_prompt {:?} vs {:?}", applied.generation_prompt, exp["generation_prompt"]));
        }
        let Some(output) = fx.get("output").and_then(Json::as_str) else { continue };
        // stream it token by token like the oracle did
        let mut n_id = 0;
        let mut s = Session::new(
            parser_cfg(&c, &inp),
            applied.generation_prompt.clone(),
            Box::new(move || {
                let id = format!("call{n_id}");
                n_id += 1;
                id
            }),
        );
        let toks = c.tok.encode(output, false, true);
        let steps = exp["steps"].as_array().unwrap();
        if steps.len() != toks.len() {
            fails.push(format!("{name}: token count {} vs {}", toks.len(), steps.len()));
            continue;
        }
        let mut bad = None;
        let mut got_steps = Vec::new();
        let mut dec = tokenizer::StreamDecoder::default();
        for (i, t) in toks.iter().enumerate() {
            // the server only ever feeds complete UTF-8 (incomplete sequences are held back)
            let piece = dec.push(&c.tok.piece(*t, true));
            let got: Vec<Json> = match s.feed(&piece, false) {
                Ok(d) => d.iter().map(|d| d.to_delta_json()).collect(),
                Err(e) => vec![json!({"error": e})],
            };
            // content and reasoning deltas: step by step as llama-server; tool-call deltas: only
            // once the call is complete (see Session::feed), compared after assembly below
            if non_tool(&Json::Array(got.clone())) != non_tool(&steps[i]) {
                bad = Some(format!("{name}: step {i} (piece {piece:?})\n  want {}\n  got  {}", steps[i], Json::Array(got)));
                break;
            }
            got_steps.push(Json::Array(got));
        }
        if let Some(b) = bad {
            fails.push(b);
            continue;
        }
        let fd: Vec<Json> = match s.feed("", true) {
            Ok(d) => d.iter().map(|d| d.to_delta_json()).collect(),
            Err(e) => vec![json!({"error": e})],
        };
        got_steps.push(Json::Array(fd.clone()));
        let mut want_steps = steps.clone();
        want_steps.push(exp["final_diffs"].clone());
        if assemble(&got_steps) != assemble(&want_steps) {
            // Intentional divergence: llama-server streams a call before it is complete, and when
            // the call then fails to parse it keeps the stale partial call (invalid JSON
            // arguments such as "{") as the final message, or, if text was also produced, fails
            // the stream with "Invalid diff: now finding less tool calls!". We stream only
            // complete calls; the stream must then equal our own final message.
            let base = name.split('/').next().unwrap_or("");
            if DIVERGE_FROM_LLAMA.contains(&base) {
                let mine = json!({"content": s.msg().content, "reasoning": s.msg().reasoning_content,
                    "tool_calls": s.msg().tool_calls.iter().map(|c| (c.name.clone(), c.arguments.clone())).collect::<Vec<_>>()});
                if assemble(&got_steps) != mine {
                    fails.push(format!("{name}: stream differs from its own final message"));
                }
                diverged += 1;
                continue;
            }
            fails.push(format!("{name}: assembled stream\n  want {}\n  got  {}", assemble(&want_steps), assemble(&got_steps)));
            continue;
        }
        if s.msg().to_json() != exp["final"] {
            fails.push(format!("{name}: final msg\n  want {}\n  got  {}", exp["final"], s.msg().to_json()));
        }
    }
    for f in &fails {
        eprintln!("FAIL {f}\n");
    }
    eprintln!("{fixture}: {n} fixtures, {} failures, {diverged} intentional divergences (incomplete calls not streamed)", fails.len());
    assert!(diverged <= 2 * DIVERGE_FROM_LLAMA.len());
    assert!(fails.is_empty());
}

#[test]
fn llama_fixtures_ornith() {
    run_fixtures("Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf", "ornith.jsonl");
}

#[test]
fn llama_fixtures_qwen36() {
    run_fixtures("Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf", "qwen36.jsonl");
}

// ------------------------------------------------------------------ Qwen3.8-Flash-Next template (model-free)

const FLASH: &str = include_str!("fixtures/qwen38-flash-next.jinja");

/// One tool-loop request: system, user, an assistant tool call (arguments as the JSON string
/// OpenAI clients send), the tool result. `thinking` is set through chat_template_kwargs.
fn flash_tool_body(thinking: bool) -> Json {
    json!({
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "What is in src/main.rs?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"src/main.rs\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"}
        ],
        "tools": [
            {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
        ],
        "chat_template_kwargs": {"enable_thinking": thinking}
    })
}

/// The server's path: `chat_inputs` with the default server options, then `apply`.
fn flash_render(body: &Json) -> (ChatTemplate, ChatInputs, Applied) {
    let tmpl = ChatTemplate::new(FLASH, "", "<|im_end|>").unwrap();
    let inp = chat_inputs(body, &ServerChatOptions::default(), &tmpl).unwrap();
    let applied = apply(&tmpl, &inp).unwrap();
    (tmpl, inp, applied)
}

#[test]
fn flash_template_format() {
    let t = ChatTemplate::new(FLASH, "", "<|im_end|>").unwrap();
    assert_eq!(t.format, Format::Qwen3Coder);
    assert!(t.supports_thinking);
}

#[test]
fn flash_template_renders_tools_and_tool_loop() {
    // thinking off (chat_template_kwargs): tools block, tool call and tool response rendered
    let (_, _, a) = flash_render(&flash_tool_body(false));
    let p = &a.prompt;
    assert!(p.starts_with("<|im_start|>system\n# Tools"), "{p}");
    assert!(p.contains(r#""name": "read""#), "tool JSON in the system block: {p}");
    assert!(p.contains("<tool_call>\n<function=read>\n<parameter=path>\n"), "{p}");
    assert!(p.contains("<|im_start|>user\n<tool_response>\n"), "{p}");
    assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{p}");
    assert_eq!(a.generation_prompt, "<|im_start|>assistant\n<think>\n\n</think>\n\n");

    // thinking enabled (the default): the generation prompt opens a think block
    let mut body = flash_tool_body(true);
    body.as_object_mut().unwrap().remove("chat_template_kwargs");
    let (_, _, a) = flash_render(&body);
    assert!(a.prompt.ends_with("<|im_start|>assistant\n<think>\n"), "{}", a.prompt);
}

#[test]
fn flash_tool_call_parses() {
    let (tmpl, inp, a) = flash_render(&flash_tool_body(false));
    let cfg = ParserConfig {
        format: tmpl.format,
        tools: inp.tools.clone(),
        tool_choice: inp.tool_choice,
        parallel_tool_calls: inp.parallel_tool_calls,
        reasoning_format: inp.reasoning_format,
        supports_thinking: tmpl.supports_thinking,
        lenient: true, // as the server parses
    };
    // a tool call as the model writes it after the (empty) think block
    let out = "<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>";
    let mut s = Session::new(cfg.clone(), a.generation_prompt.clone(), Box::new(|| "call0".to_string()));
    s.feed(out, true).unwrap();
    let m = s.msg();
    assert_eq!(m.tool_calls.len(), 1, "{m:?}");
    assert_eq!(m.tool_calls[0].name, "read");
    assert_eq!(serde_json::from_str::<Json>(&m.tool_calls[0].arguments).unwrap(), json!({"path": "src/main.rs"}));

    // plain text: content, no tool calls
    let mut s = Session::new(cfg, a.generation_prompt.clone(), Box::new(|| "call0".to_string()));
    s.feed("Hello there.", true).unwrap();
    assert_eq!(s.msg().content, "Hello there.");
    assert!(s.msg().tool_calls.is_empty());
}

// ---------------------------------------------------------------------------------------
// incremental Session (parser::parse_step) vs the full re-parse at every step

/// Feed `output` in `chunks` (char counts, cycled) to an incremental and a reference
/// (full re-parse) session; every step's diffs (or error), the final diffs and the final
/// message must be identical.
fn diff_sessions(cfg: &ParserConfig, gen_prompt: &str, output: &str, chunks: &[usize]) -> (usize, usize) {
    let mk = |inc: bool| {
        let mut n = 0;
        let mut s = Session::new(
            cfg.clone(),
            gen_prompt.to_string(),
            Box::new(move || {
                n += 1;
                format!("id{n}")
            }),
        );
        s.incremental = inc;
        s
    };
    let (mut a, mut b) = (mk(true), mk(false));
    let chars: Vec<char> = output.chars().collect();
    let (mut i, mut k) = (0, 0);
    let dj = |r: Result<Vec<MsgDiff>, String>| match r {
        Ok(d) => Json::Array(d.iter().map(|d| d.to_delta_json()).collect()),
        Err(e) => json!({ "error": e }),
    };
    while i < chars.len() {
        let n = chunks[k % chunks.len()].max(1).min(chars.len() - i);
        k += 1;
        let piece: String = chars[i..i + n].iter().collect();
        i += n;
        let (da, db) = (dj(a.feed(&piece, false)), dj(b.feed(&piece, false)));
        assert_eq!(da, db, "step at char {i} ({piece:?}) of {output:?}");
    }
    assert_eq!(dj(a.feed("", true)), dj(b.feed("", true)), "final diffs of {output:?}");
    assert_eq!(a.msg(), b.msg());
    (a.n_incremental, a.n_full)
}

fn chunkings() -> Vec<Vec<usize>> {
    let mut v = vec![vec![1], vec![2], vec![3], vec![5], vec![13]];
    // pseudo-random 1..9
    let mut x = 0x2545_f491u32;
    let mut r = Vec::new();
    for _ in 0..61 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        r.push((x % 9 + 1) as usize);
    }
    v.push(r);
    v
}

fn synth_tools() -> Vec<Tool> {
    vec![
        Tool {
            name: "write".into(),
            description: String::new(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
        },
        Tool {
            name: "bash".into(),
            description: String::new(),
            parameters: json!({"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"},"cwd":{"type":["string","null"]}},"required":["command"]}),
        },
    ]
}

const SYNTH_OUTPUTS: &[&str] = &[
    "Let me think.\nIt is not </thin> yet, <tool_ maybe.\n</think>\n\nI'll write the file.\n\n<tool_call>\n<function=write>\n<parameter=path>\nsrc/a.py\n</parameter>\n<parameter=content>\ndef f(x):\n    return \"q\\\\\" + x  # \u{1}\t</parameter> not yet\n\n</parameter\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la | grep \"é\"\n</parameter>\n<parameter=timeout>\n120\n</parameter>\n<parameter=cwd>\nnull\n</parameter>\n</function>\n</tool_call>",
    "\n\n  \nHéllo ✓ 日本語 plain answer with <tool and \n<tool_cal but no call.\nDone.",
    "Short reasoning</think>Answer then <tool_call>\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n</tool_call> trailing",
    "<think>\nnested-looking <think> text\n</think>\n<tool_call>\n<function=write>\n<parameter=path>\nx\n</parameter>\n<parameter=content>\n",
    "<tool_call>\n<function=nosuch>\n<parameter=a>\nb\n</parameter>\n</function>\n</tool_call>",
    "<function=write>\n<parameter=path>\np\n</parameter>\n<parameter=content>\nbody \"quoted\" \\ back\n</parameter>\n</function>\n",
    "",
    "   ",
];

#[test]
fn incremental_session_equals_full_reparse_synthetic() {
    let (mut n, mut inc, mut full) = (0, 0, 0);
    for supports_thinking in [true, false] {
        for (tools, choice) in [(synth_tools(), ToolChoice::Auto), (Vec::new(), ToolChoice::Auto), (synth_tools(), ToolChoice::None)] {
            for parallel in [true, false] {
                for rf in [ReasoningFormat::Auto, ReasoningFormat::None] {
                    for format in [Format::Qwen3Coder, Format::ContentOnly] {
                        let cfg = ParserConfig {
                            format,
                            tools: tools.clone(),
                            tool_choice: choice,
                            parallel_tool_calls: parallel,
                            reasoning_format: rf,
                            supports_thinking,
                            lenient: false,
                        };
                        for gp in ["<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n"] {
                            for out in SYNTH_OUTPUTS {
                                for ch in chunkings() {
                                    let (i, f) = diff_sessions(&cfg, gp, out, &ch);
                                    inc += i;
                                    full += f;
                                    n += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("incremental vs full: {n} synthetic streams identical; steps incremental {inc}, full {full}");
    assert!(inc > full, "the incremental path must carry most steps");
}

/// The same check on the llama.cpp fixture outputs and request configs (needs the GGUFs for
/// the chat template; skipped without them).
fn incremental_fixtures(gguf: &str, fixture: &str) {
    let Some(c) = load(gguf) else { return };
    let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
    let data = std::fs::read_to_string(path).unwrap();
    let (mut n, mut inc, mut full) = (0, 0, 0);
    for line in data.lines() {
        let fx: Json = serde_json::from_str(line).unwrap();
        let Some(output) = fx.get("output").and_then(Json::as_str) else { continue };
        let Ok(inp) = chat_inputs(&fx["body"], &opts(&fx["kwargs"]), &c.tmpl) else { continue };
        let Ok(applied) = apply(&c.tmpl, &inp) else { continue };
        for ch in chunkings() {
            let (i, f) = diff_sessions(&parser_cfg(&c, &inp), &applied.generation_prompt, output, &ch);
            inc += i;
            full += f;
            n += 1;
        }
    }
    eprintln!("{fixture}: {n} incremental streams identical to the full re-parse; steps incremental {inc}, full {full}");
    assert!(n > 0);
}

#[test]
fn incremental_fixtures_ornith() {
    incremental_fixtures("Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf", "ornith.jsonl");
}

#[test]
fn incremental_fixtures_qwen36() {
    incremental_fixtures("Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf", "qwen36.jsonl");
}

// ---------------------------------------------------------------------------------------
// streaming tool calls never fail the stream (Codex replay 14:19: "Invalid diff: now finding
// less tool calls!"), lenient repair, truncation fuzz

fn subagent_tool() -> Tool {
    Tool {
        name: "subagent".into(),
        description: String::new(),
        parameters: json!({"type":"object","properties":{"agent":{"type":"string"},"description":{"type":"string"},"prompt":{"type":"string"},
            "model":{"type":"string"},"background":{"type":"boolean"}},"required":["agent","description","prompt"]}),
    }
}

fn cfg_with(tools: Vec<Tool>, lenient: bool) -> ParserConfig {
    ParserConfig {
        format: Format::Qwen3Coder,
        tools,
        tool_choice: ToolChoice::Auto,
        parallel_tool_calls: true,
        reasoning_format: ReasoningFormat::Auto,
        supports_thinking: true,
        lenient,
    }
}

/// Stream `output` char by char; never an error; returns (assembled stream incl. final, final msg).
fn stream_all(cfg: &ParserConfig, gp: &str, output: &str) -> (Json, ChatMsg) {
    let mut n = 0;
    let mut s = Session::new(cfg.clone(), gp.to_string(), Box::new(move || {
        n += 1;
        format!("id{n}")
    }));
    let mut steps = Vec::new();
    let mut buf = [0u8; 4];
    for c in output.chars() {
        let d = s.feed(c.encode_utf8(&mut buf), false).unwrap_or_else(|e| panic!("stream error {e:?} on {output:?}"));
        steps.push(Json::Array(d.iter().map(|d| d.to_delta_json()).collect()));
    }
    let d = s.feed("", true).unwrap_or_else(|e| panic!("final error {e:?} on {output:?}"));
    steps.push(Json::Array(d.iter().map(|d| d.to_delta_json()).collect()));
    (assemble(&steps), s.msg().clone())
}

fn msg_view(m: &ChatMsg) -> Json {
    let calls: Vec<(String, String)> = m.tool_calls.iter().map(|c| (c.name.clone(), c.arguments.clone())).collect();
    json!({"content": m.content, "reasoning": m.reasoning_content, "tool_calls": calls})
}

/// The replay case: a call whose parameters arrive in an order the strict grammar rejects
/// (optional `model` before required `prompt`). The partial parse shows the call while the
/// required ones stream, then loses it when `<parameter=model>` completes: llama-server (and
/// 649ac806) fail the stream there with HTTP 500 inside a 200 response.
#[test]
fn shrinking_tool_call_never_fails_the_stream() {
    let gp = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let out = "The valid model ID is `x`. Let me retry the subagent with it.\n\n<tool_call>\n<function=subagent>\n<parameter=agent>\nexplore\n</parameter>\n<parameter=description>\nReview CLI error handling\n</parameter>\n<parameter=model>\nlocal\n</parameter>\n<parameter=prompt>\nCheck the CLI.\n</parameter>\n</function>\n</tool_call>";
    // the trigger exists: the strict partial parse finds the call, then fewer calls
    let strict = cfg_with(vec![subagent_tool()], false);
    let mut seen_call = false;
    let mut shrank = false;
    for (i, _) in out.char_indices().skip(1) {
        let (m, _, _) = parser::parse_full(&strict, &format!("{gp}{}", &out[..i]), true).unwrap();
        if !m.tool_calls.is_empty() {
            seen_call = true;
        } else if seen_call {
            shrank = true;
        }
    }
    assert!(seen_call && shrank, "the shrinking-call case is not reproduced");
    // what 649ac806 / 1b1f0ad (and llama-server) did with it: diff the full parses -> error
    let at = |i: usize| parser::parse_full(&strict, &format!("{gp}{}", &out[..i]), true).unwrap().0;
    let with_call = out.find("<parameter=model>").unwrap();
    let without = with_call + "<parameter=model>\n".len() + 1;
    let (a, b) = (at(with_call), at(without));
    assert_eq!((a.tool_calls.len(), b.tool_calls.len()), (1, 0));
    assert_eq!(compute_diffs(&a, &b).unwrap_err().0, "Invalid diff: now finding less tool calls!");
    // strict (llama.cpp grammar): no stream error; the call is dropped, the stream equals the final
    let (asm, fin) = stream_all(&strict, gp, out);
    assert!(fin.tool_calls.is_empty());
    assert_eq!(asm, msg_view(&fin));
    // lenient (the server): the call is parsed with its arguments in the model's order
    let (asm, fin) = stream_all(&cfg_with(vec![subagent_tool()], true), gp, out);
    assert_eq!(fin.tool_calls.len(), 1);
    assert_eq!(fin.tool_calls[0].arguments, r#"{"agent":"explore","description":"Review CLI error handling","model":"local","prompt":"Check the CLI."}"#);
    assert_eq!(fin.content, "The valid model ID is `x`. Let me retry the subagent with it.");
    assert_eq!(asm, msg_view(&fin));
}

/// Lenient repair on the final parse: closing tags cut off -> closed; an unparsable call -> text.
#[test]
fn lenient_repair_and_text_fallback() {
    let gp = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let cfg = cfg_with(vec![subagent_tool()], true);
    let full = "Go.\n\n<tool_call>\n<function=subagent>\n<parameter=agent>\na\n</parameter>\n<parameter=description>\nb\n</parameter>\n<parameter=prompt>\nc\n</parameter>\n</function>\n</tool_call>";
    // every cut inside the closing tags is repaired to the same call
    let close = full.find("</function>").unwrap();
    for cut in close..=full.len() {
        let (asm, fin) = stream_all(&cfg, gp, &full[..cut]);
        assert_eq!(fin.tool_calls.len(), 1, "cut {cut}: {:?}", &full[cut.saturating_sub(12)..cut]);
        assert_eq!(fin.tool_calls[0].arguments, r#"{"agent":"a","description":"b","prompt":"c"}"#);
        assert_eq!(asm, msg_view(&fin));
    }
    // cut inside a value: ambiguous -> the call's text is returned as content, nothing is lost
    let cut = full.find("\nc\n").unwrap() + 2;
    let (asm, fin) = stream_all(&cfg, gp, &full[..cut]);
    assert!(fin.tool_calls.is_empty());
    assert!(fin.content.starts_with("Go.") && fin.content.contains("<tool_call>") && fin.content.ends_with("c"), "{:?}", fin.content);
    assert_eq!(asm, msg_view(&fin));
    // an unknown function: text
    let bad = "Hm.\n<tool_call>\n<function=nosuch>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
    let (asm, fin) = stream_all(&cfg, gp, bad);
    assert!(fin.tool_calls.is_empty() && fin.content.contains("<function=nosuch>"));
    assert_eq!(asm, msg_view(&fin));
}

/// Truncation fuzz: stream `output`; at every cut, (1) a fresh session fed the prefix and
/// finished never errors, (2) what the client has received so far is a prefix of that
/// truncated stream's final message (content and reasoning are string prefixes, the streamed
/// tool calls equal the final's first calls). Returns the number of cuts checked.
fn fuzz_truncations(cfg: &ParserConfig, gp: &str, output: &str) -> usize {
    let cuts: Vec<usize> = output.char_indices().map(|(i, _)| i).chain([output.len()]).collect();
    let stride = (cuts.len() / 1500).max(1);
    let mut n = 0;
    let mut s = Session::new(cfg.clone(), gp.to_string(), Box::new(|| "id".into()));
    let mut steps = Vec::new();
    let mut at = 0;
    for (ci, &cut) in cuts.iter().enumerate() {
        let d = s.feed(&output[at..cut], false).unwrap_or_else(|e| panic!("stream error {e:?} at {cut} of {output:?}"));
        at = cut;
        steps.push(Json::Array(d.iter().map(|d| d.to_delta_json()).collect()));
        if ci % stride != 0 && cut != output.len() {
            continue;
        }
        let mut f = Session::new(cfg.clone(), gp.to_string(), Box::new(|| "id".into()));
        f.feed(&output[..cut], false).unwrap_or_else(|e| panic!("prefix error {e:?} at {cut}"));
        f.feed("", true).unwrap_or_else(|e| panic!("final error {e:?} at {cut}"));
        let fin = msg_view(f.msg());
        let got = assemble(&steps);
        let pre = |k: &str| fin[k].as_str().unwrap().starts_with(got[k].as_str().unwrap());
        assert!(pre("content") && pre("reasoning"), "cut {cut}: streamed {got} is not a prefix of final {fin}");
        let (gc, fc) = (got["tool_calls"].as_array().unwrap(), fin["tool_calls"].as_array().unwrap());
        assert!(gc.len() <= fc.len() && gc.iter().zip(fc).all(|(a, b)| a == b), "cut {cut}: streamed calls {gc:?} vs final {fc:?}");
        n += 1;
    }
    n
}

#[test]
fn fuzz_truncations_synthetic() {
    let mut n = 0;
    for lenient in [false, true] {
        for tools in [synth_tools(), vec![subagent_tool()], Vec::new()] {
            let cfg = cfg_with(tools, lenient);
            for gp in ["<|im_start|>assistant\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n<think>\n"] {
                for out in SYNTH_OUTPUTS {
                    n += fuzz_truncations(&cfg, gp, out);
                }
            }
        }
    }
    eprintln!("fuzz: {n} truncations, no error, streams consistent");
}

fn fuzz_fixtures(gguf: &str, fixture: &str) {
    let Some(c) = load(gguf) else { return };
    let path = format!("{}/tests/fixtures/{fixture}", env!("CARGO_MANIFEST_DIR"));
    let mut n = 0;
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let fx: Json = serde_json::from_str(line).unwrap();
        let Some(output) = fx.get("output").and_then(Json::as_str) else { continue };
        let Ok(inp) = chat_inputs(&fx["body"], &opts(&fx["kwargs"]), &c.tmpl) else { continue };
        let Ok(applied) = apply(&c.tmpl, &inp) else { continue };
        for lenient in [false, true] {
            let mut cfg = parser_cfg(&c, &inp);
            cfg.lenient = lenient;
            n += fuzz_truncations(&cfg, &applied.generation_prompt, output);
        }
    }
    eprintln!("{fixture}: fuzz {n} truncations");
}

#[test]
fn fuzz_truncations_fixtures() {
    fuzz_fixtures("Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf", "ornith.jsonl");
    fuzz_fixtures("Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf", "qwen36.jsonl");
}

/// The same fuzz over a recorded OpenCode session (raw, private: never in git): set
/// OXE_FUZZ_RECORDING=<recording.jsonl> (session_replay format). Each turn's parsed response is
/// rendered back to Qwen3-Coder text (content, then each tool call) and fuzzed with the turn's
/// own tools, under both templates.
#[test]
fn fuzz_truncations_recording() {
    let Ok(rec) = std::env::var("OXE_FUZZ_RECORDING") else {
        eprintln!("skip: OXE_FUZZ_RECORDING not set");
        return;
    };
    for gguf in ["Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf", "Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf"] {
        let Some(c) = load(gguf) else { continue };
        let (mut turns, mut n) = (0, 0);
        for line in std::fs::read_to_string(&rec).unwrap().lines() {
            let t: Json = serde_json::from_str(line).unwrap();
            if t["kind"] != "turn" || t["path"] != "/v1/chat/completions" {
                continue;
            }
            let r = &t["response"];
            let mut out = r["content"].as_str().unwrap_or("").to_string();
            for tc in r["tool_calls"].as_array().into_iter().flatten() {
                let args: Json = serde_json::from_str(tc["arguments"].as_str().unwrap_or("{}")).unwrap_or(json!({}));
                out += &format!("\n\n<tool_call>\n<function={}>\n", tc["name"].as_str().unwrap_or(""));
                for (k, v) in args.as_object().into_iter().flatten() {
                    let v = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
                    out += &format!("<parameter={k}>\n{v}\n</parameter>\n");
                }
                out += "</function>\n</tool_call>";
            }
            let Ok(inp) = chat_inputs(&t["body"], &opts(&t["body"]["chat_template_kwargs"]), &c.tmpl) else { continue };
            let Ok(applied) = apply(&c.tmpl, &inp) else { continue };
            let mut cfg = parser_cfg(&c, &inp);
            cfg.lenient = true;
            n += fuzz_truncations(&cfg, &applied.generation_prompt, &out);
            turns += 1;
        }
        eprintln!("{gguf}: recording fuzz {turns} turns, {n} truncations");
        assert!(turns > 0);
    }
}

/// Request tolerance: a tool_choice object names one function; a late system message is
/// passed on as a user message (both were HTTP 400).
#[test]
fn request_tool_choice_object_and_late_system() {
    let Some(c) = load("Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf") else { return };
    let tools = json!([{"type":"function","function":{"name":"a","parameters":{"type":"object","properties":{}}}},
                       {"type":"function","function":{"name":"b","parameters":{"type":"object","properties":{}}}}]);
    let body = json!({"messages":[{"role":"system","content":"S"},{"role":"user","content":"U"},{"role":"system","content":"late"}],
        "tools": tools, "tool_choice": {"type":"function","function":{"name":"b"}}});
    let inp = chat_inputs(&body, &opts(&json!({})), &c.tmpl).unwrap();
    assert_eq!(inp.tool_choice, ToolChoice::Required);
    assert_eq!(inp.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), vec!["b"]);
    assert_eq!(inp.messages[2]["role"], "user");
    assert!(apply(&c.tmpl, &inp).unwrap().prompt.contains("late"));
    let bad = json!({"messages":[{"role":"user","content":"U"}], "tools": tools, "tool_choice": {"type":"function","function":{"name":"zz"}}});
    assert!(chat_inputs(&bad, &opts(&json!({})), &c.tmpl).is_err());
}

// G0' (batch 2, 2026-11-11 01:45): OpenCode's final edit call carried `replaceAll` = Python
// `False`; the boolean-only parameter failed the JSON scan, so the lenient final parse returned
// the whole call as plain content and no edit ran. Exact model text from both failing runs.
fn opencode_edit_tool() -> Tool {
    Tool {
        name: "edit".into(),
        description: String::new(),
        parameters: json!({"type":"object","properties":{"path":{"type":"string"},"oldString":{"type":"string"},
            "newString":{"type":"string"},"replaceAll":{"type":"boolean"}},"required":["path","oldString","newString"]}),
    }
}

fn g0prime_text(b: &str) -> String {
    format!("Confirmed the failure on `'\u{ff11}\u{ff12}'`. Now I'll fix `parse_port.py` to only accept ASCII digits by checking each character is in `\"0123456789\"`.<tool_call>\n<function=edit>\n<parameter=replaceAll>\n{b}\n</parameter>\n<parameter=path>\n/work/project/parse_port.py\n</parameter>\n<parameter=oldString>\n    value=value.strip()\n    if not value.isdigit():\n        raise ValueError(\"digits required\")\n</parameter>\n<parameter=newString>\n    value=value.strip()\n    if len(value)==0 or any(c not in \"0123456789\" for c in value):\n        raise ValueError(\"digits required\")\n</parameter>\n</function>\n</tool_call>")
}

#[test]
fn g0prime_python_bool_parameter_is_a_tool_call() {
    let gp = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let cfg = cfg_with(vec![opencode_edit_tool()], true);
    for (b, want) in [("False", "false"), ("True", "true"), ("false", "false")] {
        let m = parser::parse(&cfg, gp, &g0prime_text(b), false).unwrap();
        assert_eq!(m.tool_calls.len(), 1, "{b}: {m:?}");
        assert!(!m.content.contains("<tool_call>"), "{b}");
        let args: Json = serde_json::from_str(&m.tool_calls[0].arguments).unwrap();
        assert_eq!(args["replaceAll"].to_string(), want);
        assert_eq!(args["path"], "/work/project/parse_port.py");
        // streaming char by char: never an error, same final call
        let (_, sm) = stream_all(&cfg, gp, &g0prime_text(b));
        assert_eq!(sm.tool_calls.len(), 1, "stream {b}");
        assert_eq!(sm.tool_calls[0].arguments, m.tool_calls[0].arguments, "stream {b}");
    }
    // strict (llama.cpp parity) mode is unchanged: still rejected
    let strict = cfg_with(vec![opencode_edit_tool()], false);
    let m = parser::parse(&strict, gp, &g0prime_text("False"), true).unwrap();
    assert!(m.tool_calls.is_empty() || m.tool_calls[0].arguments.ends_with('{') || !m.tool_calls[0].arguments.contains("replaceAll\":false"));
}
