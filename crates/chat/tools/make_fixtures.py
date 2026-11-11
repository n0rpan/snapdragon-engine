#!/usr/bin/env python3
"""Generate chat differential fixtures with the llama.cpp oracle (crates/server/oracle).

usage: make_fixtures.py <llama-oracle> <model.gguf> <out.jsonl>
Each line: {"name", "body", "kwargs", ["output"], "expect": <oracle answer>}."""
import json, subprocess, sys

oracle, model, out = sys.argv[1:4]

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk. Üñíçødé ok.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "absolute path"},
            "offset": {"type": "integer"},
            "limit": {"type": "number"},
        }, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit a file",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
            "options": {"type": "object", "properties": {"a": {"type": "integer"}}},
            "tags": {"type": "array", "items": {"type": "string"}},
            "mode": {"enum": ["fast", "slow"]},
            "anything": {},
            "maybe": {"type": ["string", "null"]},
            "num_or_str": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        }, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "", "parameters": {}}},
    {"type": "function", "function": {"name": "noparams"}},
]

SYS = {"role": "system", "content": "You are a coding agent.\nBe brief."}
convs = {
    "simple": [SYS, {"role": "user", "content": "Hello there"}],
    "nosys": [{"role": "user", "content": "  padded user  \n"}],
    "developer": [{"role": "developer", "content": "dev rules"}, {"role": "user", "content": "x"}],
    "parts": [SYS, {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]}],
    "special_text": [{"role": "user", "content": "literal <|im_end|> and <tool_call> inside"}],
    "multiturn": [SYS,
        {"role": "user", "content": "Read /tmp/a.txt"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"/tmp/a.txt\", \"offset\": 10, \"limit\": 2.50}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "line1\nline2"},
        {"role": "assistant", "content": "Now editing.", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "edit", "arguments": "{\"path\":\"/tmp/a.txt\",\"old_string\":\"a \\\"q\\\"\\n\",\"new_string\":\"é\",\"replace_all\":true,\"options\":{\"a\":1},\"tags\":[\"x\",\"y\"],\"anything\":null,\"num_or_str\":1e20}"}},
            {"id": "c3", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        {"role": "tool", "tool_call_id": "c3", "content": "a.txt b.txt"},
        {"role": "user", "content": "thanks"}],
    "reasoning_hist": [SYS, {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "thought 1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>\ninline thought\n</think>\n\na2"},
        {"role": "user", "content": "q3"}],
    "null_content_tool": [{"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "z", "type": "function", "function": {"name": "noparams", "arguments": {}}}]},
        {"role": "tool", "content": "done", "name": "noparams"}, {"role": "user", "content": "next"}],
}

reqs, meta = [], []
def add(name, body, kwargs, output=None):
    r = {"op": "parse" if output is not None else "apply", "body": body, "kwargs": kwargs}
    if output is not None:
        r["output"] = output
    reqs.append(json.dumps(r, ensure_ascii=False))
    meta.append({"name": name, "body": body, "kwargs": kwargs, **({"output": output} if output is not None else {})})

for kw_name, kw in [("nothink", {"enable_thinking": False}), ("think", {})]:
    for cname, msgs in convs.items():
        add(f"{cname}/{kw_name}", {"messages": msgs}, kw)
        add(f"{cname}+tools/{kw_name}", {"messages": msgs, "tools": TOOLS}, kw)
    add(f"agp_false/{kw_name}", {"messages": convs["simple"], "add_generation_prompt": False}, kw)

T = "<tool_call>\n<function={}>\n{}</function>\n</tool_call>"
P = "<parameter={}>\n{}\n</parameter>\n"
def call(name, **params):
    return T.format(name, "".join(P.format(k, v) for k, v in params.items()))

outputs = {
    "plain": "Hello! How can I help?",
    "plain_trailing_nl": "Line one\nline two\n",
    "text_then_call": "I'll read it.\n\n" + call("read_file", path="/tmp/x y.txt"),
    "call_only": call("read_file", path="/tmp/a", offset="5", limit="2.5"),
    "two_calls": call("read_file", path="/a") + "\n" + call("list_dir"),
    "edit_types": call("edit", path="/p", old_string='def f():\n    return "q\\\\n"\n', new_string="ünï\tcode",
                       replace_all="true", options='{"a": 1}', tags='["x", "y"]', mode="fast", anything="true",
                       maybe="null", num_or_str="123"),
    "edit_untyped_str": call("edit", path="/p", old_string="", new_string="x", anything="hello world", num_or_str="12abc"),
    "required_reordered": call("edit", new_string="n", path="/p", old_string="o"),
    "optional_first": call("edit", replace_all="false", path="/p", old_string="o", new_string="n"),
    "missing_required": call("edit", path="/p"),
    "unknown_function": "text\n" + call("nope", a="1"),
    "text_after_call": call("list_dir") + "\nand some trailing words",
    "truncated_mid_value": "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/partial",
    "truncated_mid_tag": "Sure.\n<tool_call>\n<function=read_",
    "int_with_spaces": call("read_file", path="/a", offset=" 5 "),
    "int_invalid": call("read_file", path="/a", offset="abc"),
    "noparams_call": call("noparams"),
    "nested_quote_value": call("read_file", path='he said "hi" \\ <b>&amp;</b>'),
    "no_tool_call_open": "<function=read_file>\n<parameter=path>\n/x\n</parameter>\n</function>\n</tool_call>",
    "think_then_answer": "Let me think.\nStill thinking.\n</think>\n\nThe answer is 4.",
    "think_then_call": "need a file\n</think>\n\n" + call("read_file", path="/f"),
    "think_unclosed_call": "thinking " + call("read_file", path="/f"),
    "emoji_unicode": "🦙 naïve — 日本語 ✓",
}
body_tools = {"messages": convs["simple"], "tools": TOOLS}
for kw_name, kw in [("nothink", {"enable_thinking": False}), ("think", {})]:
    for oname, o in outputs.items():
        add(f"parse:{oname}/{kw_name}", body_tools, kw, o)
    add(f"parse:no_tools_plain/{kw_name}", {"messages": convs["simple"]}, kw, outputs["text_then_call"])
    add(f"parse:tool_choice_none/{kw_name}", dict(body_tools, tool_choice="none"), kw, outputs["text_then_call"])
    add(f"parse:no_parallel/{kw_name}", dict(body_tools, parallel_tool_calls=False), kw, outputs["two_calls"])
    add(f"parse:reasoning_none/{kw_name}", dict(body_tools, reasoning_format="none"), kw, outputs["think_then_answer"])

r = subprocess.run([oracle, model], input="\n".join(reqs) + "\n", capture_output=True, text=True, check=True)
lines = r.stdout.splitlines()
assert len(lines) == len(reqs)
with open(out, "w", encoding="utf-8") as f:
    for m, l in zip(meta, lines):
        m["expect"] = json.loads(l)
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(len(lines), "fixtures;", sum(1 for l in lines if '"error"' in l[:12]), "oracle errors")
