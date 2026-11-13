//! v0.1.3 lane G (AB-1-642, CP1-4-714): the Qwen3.8-Flash-Next chat template and prefix stability across turns,
//! model-free (the committed template fixture of the chat crate). The Flash server passes NO preserve_thinking and
//! has no ECHO (decided: neither is enabled for Flash before its hardware gate); these tests pin what the template
//! then renders, so the prefix cache's behaviour per turn is known:
//! - the template's own default (preserve_thinking undefined) keeps earlier reasoning when the client returns it:
//!   turn N+1's prompt starts with turn N's prompt + the generated text (full prefix hit);
//! - a client that drops reasoning (OpenCode today) breaks the prefix exactly at the previous answer's think block:
//!   everything before turn N's generation prompt is still a hit, the re-prefill starts at the newest assistant turn;
//! - preserve_thinking=false (what the oxe path may pass) breaks it one block earlier, at that turn's header;
//! - thinking off: the empty think block is rendered back identically (full prefix hit without any reasoning).
use chat::{apply, chat_inputs, ChatTemplate, ServerChatOptions};
use serde_json::{json, Value as Json};

const FLASH: &str = include_str!("../../chat/src/fixtures/qwen38-flash-next.jinja");

fn render(messages: Json, kwargs: Json) -> String {
    let tmpl = ChatTemplate::new(FLASH, "", "<|im_end|>").unwrap();
    let body = json!({ "messages": messages, "chat_template_kwargs": kwargs });
    let inp = chat_inputs(&body, &ServerChatOptions::default(), &tmpl).unwrap();
    apply(&tmpl, &inp).unwrap().prompt
}

fn common(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

const SYS: &str = "You are a coding agent.";
const U1: &str = "List the files.";
const U2: &str = "Now open the first one.";
const R1: &str = "The user wants a listing.";
const A1: &str = "Here they are.";

fn turn1() -> Json {
    json!([{"role": "system", "content": SYS}, {"role": "user", "content": U1}])
}

fn turn2(assistant: Json) -> Json {
    json!([{"role": "system", "content": SYS}, {"role": "user", "content": U1}, assistant, {"role": "user", "content": U2}])
}

#[test]
fn thinking_on_reasoning_returned_is_a_full_prefix_hit_by_default() {
    let p1 = render(turn1(), json!({}));
    assert!(p1.ends_with("<|im_start|>assistant\n<think>\n"), "{p1}");
    // what the model generated after the generation prompt
    let gen = format!("{R1}\n</think>\n\n{A1}");
    let p2 = render(turn2(json!({"role": "assistant", "content": A1, "reasoning_content": R1})), json!({}));
    assert!(p2.starts_with(&format!("{p1}{gen}<|im_end|>\n")), "turn 2 must extend turn 1 + its generation:\n{p2}");
    // preserve_thinking=true renders the same
    assert_eq!(p2, render(turn2(json!({"role": "assistant", "content": A1, "reasoning_content": R1})), json!({"preserve_thinking": true})));
}

#[test]
fn thinking_on_reasoning_dropped_breaks_at_the_newest_think_block() {
    let p1 = render(turn1(), json!({}));
    let gen = format!("{R1}\n</think>\n\n{A1}");
    let p2 = render(turn2(json!({"role": "assistant", "content": A1})), json!({}));
    // the previous answer comes back with an EMPTY think block: the prefix holds through turn 1's generation prompt
    assert!(p2.contains("<|im_start|>assistant\n<think>\n\n</think>\n\nHere they are.<|im_end|>"), "{p2}");
    assert_eq!(common(&p2, &format!("{p1}{gen}")), p1.len(), "the hit must end exactly at turn 1's generation prompt");
}

#[test]
fn preserve_thinking_false_breaks_one_block_earlier() {
    let p1 = render(turn1(), json!({"preserve_thinking": false}));
    let gen = format!("{R1}\n</think>\n\n{A1}");
    let p2 = render(turn2(json!({"role": "assistant", "content": A1, "reasoning_content": R1})), json!({"preserve_thinking": false}));
    // an earlier turn (before the last user query) loses its think block entirely
    assert!(p2.contains("<|im_start|>assistant\nHere they are.<|im_end|>"), "{p2}");
    assert_eq!(common(&p2, &format!("{p1}{gen}")), p1.len() - "<think>\n".len(), "the hit ends at turn 1's assistant header");
}

#[test]
fn tool_loop_keeps_reasoning_whatever_preserve_thinking_says() {
    // inside the current tool loop (after the last real user query) the template always renders the reasoning
    let msgs = json!([
        {"role": "system", "content": SYS},
        {"role": "user", "content": U1},
        {"role": "assistant", "content": "", "reasoning_content": R1,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.rs"}
    ]);
    for kw in [json!({}), json!({"preserve_thinking": false}), json!({"preserve_thinking": true})] {
        let p = render(msgs.clone(), kw.clone());
        assert!(p.contains(&format!("<|im_start|>assistant\n<think>\n{R1}\n</think>\n\n")), "{kw}: {p}");
    }
}

#[test]
fn thinking_off_is_a_full_prefix_hit_without_reasoning() {
    let kw = json!({"enable_thinking": false});
    let p1 = render(turn1(), kw.clone());
    assert!(p1.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{p1}");
    let p2 = render(turn2(json!({"role": "assistant", "content": A1})), kw);
    assert!(p2.starts_with(&format!("{p1}{A1}<|im_end|>\n")), "{p2}");
}
