#!/usr/bin/env python3
# Usage: start oxe-server on PORT (MTP on, then --spec-type none) and run: python3 tools/mtp_multiturn_check.py PORT LABEL > out-LABEL.jsonl; compare the "sha" fields per (variant, turn).
"""OpenCode-like multi-turn transcript against an oxe-server (scripted history, so every mode
sees the same prompts). Prints one JSON line per (variant, turn) with the reply's sha1, length,
finish reason, usage and timings. Usage: multiturn.py PORT LABEL"""
import hashlib, json, sys, urllib.request

port, label = sys.argv[1], sys.argv[2]
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file from the workspace",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string", "description": "absolute path"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace text in a file",
        "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}}, "required": ["filePath", "oldString", "newString"]}}},
]
SYSTEM = ("You are opencode, an interactive CLI coding agent. Use the tools to inspect and change files. "
          "Be concise. Working directory: /home/user/project.\n" + "\n".join(f"Rule {i}: keep changes minimal and test them." for i in range(40)))
SRC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
TURNS = [
    [{"role": "user", "content": "The add() function in calc.py is wrong. Look at the file first."}],
    [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "/home/user/project/calc.py"})}}]},
     {"role": "tool", "tool_call_id": "call_1", "content": SRC}],
    [{"role": "assistant", "content": None, "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "edit", "arguments": json.dumps({"filePath": "/home/user/project/calc.py", "oldString": "return a - b", "newString": "return a + b"})}}]},
     {"role": "tool", "tool_call_id": "call_2", "content": "Edit applied successfully."}],
    [{"role": "assistant", "content": "Fixed: add() now returns a + b."},
     {"role": "user", "content": "Now write a pytest test file for both functions and tell me how to run it."}],
]
VARIANTS = {
    "greedy": {"temperature": 0},
    "sampled-seed42": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42},
    "greedy-selfhist": {"temperature": 0},  # the model's own replies go into the history
    "opencode-default": {},  # what the captured OpenCode requests send: no sampling params
}

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

for vname, params in VARIANTS.items():
    msgs = [{"role": "system", "content": SYSTEM}]
    prev = None
    for ti, turn in enumerate(TURNS):
        turn = json.loads(json.dumps(turn))
        if vname == "greedy-selfhist" and prev is not None and turn[0]["role"] == "assistant":
            gen = {"role": "assistant", "content": prev.get("content")}
            if prev.get("tool_calls"):
                gen["tool_calls"] = prev["tool_calls"]
                for m2 in turn[1:]:
                    if m2["role"] == "tool":
                        m2["tool_call_id"] = prev["tool_calls"][0].get("id", m2["tool_call_id"])
            turn[0] = gen
        msgs += turn
        body = {"model": "x", "messages": msgs, "tools": TOOLS, "max_tokens": 160, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}, **params}
        r = post(body)
        m = r["choices"][0]["message"]
        prev = m
        tcs = [{k: v for k, v in tc.items() if k != "id"} for tc in (m.get("tool_calls") or [])]  # ids are random
        reply = json.dumps({"content": m.get("content"), "tool_calls": tcs}, sort_keys=True)
        t = r.get("timings", {})
        print(json.dumps({"label": label, "variant": vname, "turn": ti, "sha": hashlib.sha1(reply.encode()).hexdigest()[:12],
                          "len": len(reply), "reply": reply, "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage"),
                          "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
                          "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}), flush=True)
