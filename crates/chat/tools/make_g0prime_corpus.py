#!/usr/bin/env python3
# usage: make_g0prime_corpus.py OUT.jsonl <request-NNN.http>... (the last request of each captured
# G0' protected task; its history holds every earlier turn)
# Build crates/chat/tests/fixtures/g0prime_corpus.jsonl from G0' request captures:
# tool parameter schemas (descriptions stripped) + every assistant turn rendered back to
# Qwen3-Coder text. No system prompt, no ids, no headers.
import json, sys
def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if not (k in ("description", "$schema") and isinstance(v, str))}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x
def body(path):
    raw = open(path, 'rb').read()
    sep = b'\r\n\r\n' if b'\r\n\r\n' in raw else b'\n\n'
    return json.loads(raw.split(sep, 1)[1])
out = []
tools = None
seen = set()
for path in sys.argv[2:]:
    b = body(path)
    t = [{"name": x["function"]["name"], "parameters": strip(x["function"]["parameters"])} for x in b["tools"]]
    if tools is None:
        tools = t
    assert tools == t, path
    turn = 0
    for m in b["messages"]:
        if m["role"] != "assistant":
            continue
        turn += 1
        text = m.get("content") or ""
        for tc in m.get("tool_calls") or []:
            args = json.loads(tc["function"]["arguments"])
            text += "\n\n<tool_call>\n<function=%s>\n" % tc["function"]["name"]
            for k, v in args.items():
                v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                text += "<parameter=%s>\n%s\n</parameter>\n" % (k, v)
            text += "</function>\n</tool_call>"
        if text in seen:
            continue
        seen.add(text)
        out.append({"kind": "output", "source": "%s turn %d" % (path.split("local-ai/")[-1], turn), "output": text})
with open(sys.argv[1], "w") as f:
    f.write(json.dumps({"kind": "tools", "source": "OpenCode 2.2.6 tool schemas from the G0' request captures (descriptions stripped)", "tools": tools}, ensure_ascii=False) + "\n")
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(len(out), "outputs")
