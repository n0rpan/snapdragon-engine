//! OpenAI chat body -> template context, the way llama-server's `oaicompat_chat_params_parse`
//! and `common_chat_templates_apply_jinja` do it (messages are round-tripped through
//! `common_chat_msg` and its `to_json_oaicompat`, tools through `common_chat_tool`, then the
//! llama.cpp workarounds are applied).

use crate::template::{ChatTemplate, Format};
use serde_json::{json, Map, Value as Json};

#[derive(Debug)]
pub struct BadRequest(pub String);
impl std::fmt::Display for BadRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BadRequest {}
fn bad(s: impl Into<String>) -> BadRequest {
    BadRequest(s.into())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// JSON schema of the parameters (`{}` -> `{"type":"object","properties":{}}` only for the parser)
    pub parameters: Json,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningFormat {
    None,
    Auto,
    Deepseek,
    DeepseekLegacy,
}

/// Server-level defaults (CLI flags of llama-server).
#[derive(Clone, Debug)]
pub struct ServerChatOptions {
    /// `--chat-template-kwargs` (values are JSON)
    pub chat_template_kwargs: Map<String, Json>,
    pub reasoning_format: ReasoningFormat,
    /// `--reasoning` default (true unless disabled); combined with the template's support
    pub enable_thinking: bool,
}

impl Default for ServerChatOptions {
    fn default() -> Self {
        ServerChatOptions { chat_template_kwargs: Map::new(), reasoning_format: ReasoningFormat::Deepseek, enable_thinking: true }
    }
}

/// Everything the template, the parser and the sampler need from a chat request.
#[derive(Clone, Debug)]
pub struct ChatInputs {
    /// normalized messages as given to the template
    pub messages: Json,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    pub parallel_tool_calls: bool,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub reasoning_format: ReasoningFormat,
    /// extra context: datetime/date_string + chat_template_kwargs
    pub extra: Map<String, Json>,
}

/// Result of applying the template.
#[derive(Clone, Debug)]
pub struct Applied {
    pub prompt: String,
    /// the part of the prompt added by add_generation_prompt (prepended to the output for parsing)
    pub generation_prompt: String,
    pub format: Format,
    pub thinking_start: Option<String>,
    pub thinking_end: Vec<String>,
}

fn tools_to_json(tools: &[Tool]) -> Json {
    Json::Array(
        tools
            .iter()
            .map(|t| {
                json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}})
            })
            .collect(),
    )
}

/// `common_chat_tools_parse_oaicompat`
pub fn parse_tools(tools: &Json) -> Result<Vec<Tool>, BadRequest> {
    let mut out = Vec::new();
    match tools {
        Json::Null => {}
        Json::Array(a) => {
            for t in a {
                if t.get("type").and_then(Json::as_str) != Some("function") {
                    return Err(bad(format!("Failed to parse tools: Unsupported tool type: {t}")));
                }
                let f = t.get("function").ok_or_else(|| bad(format!("Failed to parse tools: Missing tool function: {t}")))?;
                let name = f.get("name").and_then(Json::as_str).ok_or_else(|| bad("Failed to parse tools: missing name"))?;
                out.push(Tool {
                    name: name.to_string(),
                    description: f.get("description").and_then(Json::as_str).unwrap_or("").to_string(),
                    parameters: f.get("parameters").cloned().unwrap_or_else(|| json!({})),
                });
            }
        }
        _ => return Err(bad(format!("Failed to parse tools: Expected 'tools' to be an array, got {tools}"))),
    }
    Ok(out)
}

/// `common_chat_msgs_parse_oaicompat` + `common_chat_msg::to_json_oaicompat` + workarounds
/// (developer->system, func_args_not_string).
pub fn normalize_messages(messages: &Json) -> Result<Json, BadRequest> {
    let arr = messages.as_array().ok_or_else(|| bad("Expected 'messages' to be an array"))?;
    let mut out = Vec::with_capacity(arr.len());
    for m in arr {
        let m = m.as_object().ok_or_else(|| bad(format!("Expected 'message' to be an object, got {m}")))?;
        let role = m.get("role").and_then(Json::as_str).ok_or_else(|| bad("Missing 'role' in message"))?;
        let has_content = m.contains_key("content");
        let has_tool_calls = m.contains_key("tool_calls");
        if !has_content && !has_tool_calls {
            return Err(bad("Expected 'content' or 'tool_calls'"));
        }
        let mut j = Map::new();
        j.insert("role".into(), Json::String(if role == "developer" { "system".into() } else { role.into() }));
        let content = match m.get("content") {
            Some(Json::String(s)) => Json::String(s.clone()),
            Some(Json::Array(parts)) => {
                let mut ps = Vec::new();
                for p in parts {
                    let ty = p.get("type").and_then(Json::as_str).ok_or_else(|| bad(format!("Missing content part type: {p}")))?;
                    match ty {
                        "text" | "media_marker" => {
                            let text = p.get("text").and_then(Json::as_str).unwrap_or("");
                            ps.push(json!({"type": ty, "text": text}));
                        }
                        "tool_reference" => {
                            ps.push(json!({"type": ty, "name": p.get("name").cloned().unwrap_or(Json::Null)}));
                        }
                        _ => return Err(bad(format!("Unsupported content part type: \"{ty}\""))),
                    }
                }
                if ps.is_empty() {
                    Json::String(String::new())
                } else {
                    Json::Array(ps)
                }
            }
            Some(Json::Null) | None => Json::String(String::new()),
            Some(c) => return Err(bad(format!("Invalid 'content' type: expected string or array, got {c}"))),
        };
        j.insert("content".into(), content);
        if let Some(r) = m.get("reasoning_content").and_then(Json::as_str) {
            if !r.is_empty() {
                j.insert("reasoning_content".into(), Json::String(r.into()));
            }
        }
        if let Some(n) = m.get("name").and_then(Json::as_str) {
            if !n.is_empty() {
                j.insert("name".into(), Json::String(n.into()));
            }
        }
        if let Some(n) = m.get("tool_call_id").and_then(Json::as_str) {
            if !n.is_empty() {
                j.insert("tool_call_id".into(), Json::String(n.into()));
            }
        }
        if let Some(tcs) = m.get("tool_calls").and_then(Json::as_array) {
            if !tcs.is_empty() {
                let mut v = Vec::new();
                for tc in tcs {
                    if tc.get("type").and_then(Json::as_str) != Some("function") {
                        return Err(bad(format!("Unsupported tool call type: {tc}")));
                    }
                    let f = tc.get("function").ok_or_else(|| bad("Missing tool call function"))?;
                    let name = f.get("name").and_then(Json::as_str).ok_or_else(|| bad("Missing tool call name"))?;
                    // func_args_not_string: the template sees parsed arguments
                    let args = match f.get("arguments") {
                        Some(Json::String(s)) => serde_json::from_str::<Json>(s)
                            .map_err(|e| bad(format!("Failed to parse tool call arguments as JSON: {e}")))?,
                        Some(a) => a.clone(),
                        None => return Err(bad("Missing tool call arguments")),
                    };
                    let mut o = Map::new();
                    o.insert("type".into(), "function".into());
                    o.insert("function".into(), json!({"name": name, "arguments": args}));
                    if let Some(id) = tc.get("id").and_then(Json::as_str) {
                        if !id.is_empty() {
                            o.insert("id".into(), Json::String(id.into()));
                        }
                    }
                    v.push(Json::Object(o));
                }
                j.insert("tool_calls".into(), Json::Array(v));
            }
        }
        out.push(Json::Object(j));
    }
    Ok(Json::Array(out))
}

fn reasoning_format_from_name(s: &str) -> Result<ReasoningFormat, BadRequest> {
    Ok(match s {
        "none" => ReasoningFormat::None,
        "auto" => ReasoningFormat::Auto,
        "deepseek" => ReasoningFormat::Deepseek,
        "deepseek-legacy" => ReasoningFormat::DeepseekLegacy,
        _ => return Err(bad(format!("Unknown reasoning format: {s}"))),
    })
}

/// `oaicompat_chat_params_parse` (the template-related part).
pub fn chat_inputs(body: &Json, opt: &ServerChatOptions, tmpl: &ChatTemplate) -> Result<ChatInputs, BadRequest> {
    let messages = body.get("messages").ok_or_else(|| bad("'messages' is required"))?;
    for m in messages.as_array().ok_or_else(|| bad("Expected 'messages' to be an array"))? {
        let role = m.get("role").and_then(Json::as_str).unwrap_or("");
        if role != "assistant" && m.get("content").is_none() {
            return Err(bad("All non-assistant messages must contain 'content'"));
        }
        if role == "assistant" && m.get("content").is_none() && m.get("tool_calls").is_none() {
            return Err(bad("Assistant message must contain either 'content' or 'tool_calls'!"));
        }
    }
    let mut tools = parse_tools(body.get("tools").unwrap_or(&Json::Null))?;
    let tool_choice = match body.get("tool_choice") {
        None | Some(Json::Null) => ToolChoice::Auto,
        Some(Json::String(s)) => match s.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => return Err(bad(format!("Invalid tool_choice: {s}"))),
        },
        // OpenAI's named-function form {"type":"function","function":{"name":N}} (llama-server
        // rejects it): call N, i.e. tool_choice "required" with the tool list cut to N
        Some(o) => match o.pointer("/function/name").and_then(Json::as_str) {
            Some(name) if o.get("type").and_then(Json::as_str).unwrap_or("function") == "function" => {
                if !tools.iter().any(|t| t.name == name) {
                    return Err(bad(format!("Invalid tool_choice: function {name} is not in 'tools'")));
                }
                tools.retain(|t| t.name == name);
                ToolChoice::Required
            }
            _ => match o.get("type").and_then(Json::as_str) {
                Some("auto") => ToolChoice::Auto,
                Some("none") => ToolChoice::None,
                Some("required") | Some("any") => ToolChoice::Required,
                _ => return Err(bad(format!("Invalid tool_choice: {o}"))),
            },
        },
    };
    let mut msgs = normalize_messages(messages)?;
    // a system message after the first message (agents insert reminders mid-conversation): the
    // Qwen templates reject it, so it is passed on as a user message at the same position
    if let Json::Array(a) = &mut msgs {
        for m in a.iter_mut().skip(1) {
            if m.get("role").and_then(Json::as_str) == Some("system") {
                if let Some(o) = m.as_object_mut() {
                    o.insert("role".into(), Json::String("user".into()));
                }
            }
        }
    }
    // prefill_assistant: a trailing assistant message is continued; not supported yet
    if let Some(last) = msgs.as_array().and_then(|a| a.last()) {
        if last.get("role").and_then(Json::as_str) == Some("assistant") {
            return Err(bad("continuing a trailing assistant message (prefill) is not supported by this server yet"));
        }
    }
    let mut extra = Map::new();
    extra.insert("datetime".into(), Json::String(crate::template::strftime_now("%b %d %Y")));
    extra.insert("date_string".into(), Json::String(crate::template::strftime_now("%d %b %Y")));
    let mut kwargs = opt.chat_template_kwargs.clone();
    if let Some(Json::Object(k)) = body.get("chat_template_kwargs") {
        for (a, b) in k {
            kwargs.insert(a.clone(), b.clone());
        }
    }
    let mut enable_thinking = opt.enable_thinking && tmpl.supports_thinking;
    match kwargs.get("enable_thinking") {
        Some(Json::Bool(b)) => enable_thinking = *b,
        Some(Json::String(_)) => return Err(bad("invalid type for \"enable_thinking\" (expected boolean, got string)")),
        _ => {}
    }
    if let Some(re) = body.get("reasoning_effort").and_then(Json::as_str) {
        if re == "none" {
            enable_thinking = false;
            kwargs.remove("reasoning_effort");
        } else if !re.is_empty() {
            kwargs.insert("reasoning_effort".into(), Json::String(re.into()));
        }
    }
    for (k, v) in kwargs {
        extra.insert(k, v);
    }
    let reasoning_format = match body.get("reasoning_format").and_then(Json::as_str) {
        Some(s) => reasoning_format_from_name(s)?,
        None => opt.reasoning_format,
    };
    // llama.cpp: parallel_tool_calls defaults to the template caps (Qwen templates: true)
    let parallel_tool_calls = body.get("parallel_tool_calls").and_then(Json::as_bool).unwrap_or(true);
    let add_generation_prompt = body.get("add_generation_prompt").and_then(Json::as_bool).unwrap_or(true);
    if let Json::Array(a) = &mut msgs {
        let _ = a;
    }
    Ok(ChatInputs {
        messages: msgs,
        tools,
        tool_choice,
        parallel_tool_calls,
        add_generation_prompt,
        enable_thinking,
        reasoning_format,
        extra,
    })
}

fn context(inp: &ChatInputs, add_generation_prompt: bool) -> Json {
    let mut ctx = Map::new();
    ctx.insert("messages".into(), inp.messages.clone());
    ctx.insert("enable_thinking".into(), Json::Bool(inp.enable_thinking));
    if !inp.tools.is_empty() {
        ctx.insert("tools".into(), tools_to_json(&inp.tools));
    }
    for (k, v) in &inp.extra {
        ctx.insert(k.clone(), v.clone());
    }
    if add_generation_prompt {
        ctx.insert("add_generation_prompt".into(), Json::Bool(true));
    }
    Json::Object(ctx)
}

/// `common_chat_templates_apply` for the supported formats.
pub fn apply(tmpl: &ChatTemplate, inp: &ChatInputs) -> Result<Applied, minijinja::Error> {
    let prompt = tmpl.render_ctx(&context(inp, inp.add_generation_prompt))?;
    let no_gen = tmpl.render_ctx(&context(inp, false))?;
    let gen = if inp.add_generation_prompt { prompt.clone() } else { tmpl.render_ctx(&context(inp, true))? };
    let p = no_gen.bytes().zip(gen.bytes()).take_while(|(a, b)| a == b).count();
    let mut p = p;
    while !gen.is_char_boundary(p) {
        p -= 1;
    }
    let generation_prompt = gen[p..].to_string();
    let (thinking_start, thinking_end) = if tmpl.format == Format::Qwen3Coder && tmpl.supports_thinking {
        (Some("<think>".to_string()), vec!["\n</think>".to_string(), "</think>".to_string(), "<tool_call>".to_string()])
    } else if tmpl.supports_thinking {
        (Some("<think>".to_string()), vec!["</think>".to_string()])
    } else {
        (None, vec![])
    };
    Ok(Applied { prompt, generation_prompt, format: tmpl.format, thinking_start, thinking_end })
}
