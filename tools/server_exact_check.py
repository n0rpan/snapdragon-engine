#!/usr/bin/env python3
"""Capture a server's greedy outputs for a fixed request set, or compare two captures (G-exact for
server-loop changes: same tokens, same text, same streamed deltas, same tool calls).

  python3 tools/server_exact_check.py capture --url http://127.0.0.1:8093 --prompt-ids 1,2,3 --out a.json
  python3 tools/server_exact_check.py compare a.json b.json

Requests (all temperature 0, cache_prompt false):
  completion       /completion, prompt = ids, n_predict 200, return_tokens
  completion_sse   the same, streamed: concatenated content + per-chunk list
  chat_sse         /v1/chat/completions streamed, prose + code question
  chat_tools_sse   /v1/chat/completions streamed with tools (a write-file task, so the reply is a long
                   string tool argument), max_tokens 400
  chat_tools       the same, non-streamed
Normalised away: ids (chatcmpl-*, tool-call ids), created, timings, system_fingerprint.
"""
import argparse, json, sys, urllib.request

TOOLS = [
    {"type": "function", "function": {"name": "write", "description": "Write a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                    "required": ["command"]}}},
]


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        if body.get("stream"):
            chunks = []
            for line in r:
                line = line.strip()
                if line.startswith(b"data:"):
                    p = line[5:].strip()
                    if p == b"[DONE]":
                        continue
                    chunks.append(json.loads(p))
            return chunks
        return json.loads(r.read())


DROP = {"id", "created", "timings", "system_fingerprint", "model", "usage"}


def norm(x):
    if isinstance(x, dict):
        # "id" covers both the response id and the random tool-call ids
        return {k: norm(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def capture(a):
    ids = [int(v) for v in a.prompt_ids.split(",") if v]
    u = a.url
    res = {}
    base = {"temperature": 0, "cache_prompt": False}
    res["completion"] = norm(post(u + "/completion", {**base, "prompt": ids, "n_predict": 200, "return_tokens": True}))
    ch = post(u + "/completion", {**base, "prompt": ids, "n_predict": 200, "stream": True, "return_tokens": True})
    res["completion_sse"] = {"text": "".join(c.get("content", "") for c in ch), "chunks": norm(ch)}
    msgs = [{"role": "user", "content": "Explain in a short paragraph what a hash map is, then give a 10-line Python example."}]
    ch = post(u + "/v1/chat/completions", {**base, "messages": msgs, "max_tokens": 300, "stream": True})
    res["chat_sse"] = norm(ch)
    tmsgs = [{"role": "system", "content": "You are a coding agent. Use the tools."},
             {"role": "user", "content": "Create the file src/lru.py with a complete LRU cache class (get, put, capacity, docstrings, type hints). Use the write tool."}]
    ch = post(u + "/v1/chat/completions", {**base, "messages": tmsgs, "tools": TOOLS, "max_tokens": 400, "stream": True})
    res["chat_tools_sse"] = norm(ch)
    res["chat_tools"] = norm(post(u + "/v1/chat/completions", {**base, "messages": tmsgs, "tools": TOOLS, "max_tokens": 400}))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    n_tc = sum(1 for c in res["chat_tools_sse"] for ch in c.get("choices", []) if ch.get("delta", {}).get("tool_calls"))
    print(f"captured {len(res)} requests -> {a.out}; tool-call delta chunks in chat_tools_sse: {n_tc}")


def assembled(v):
    """The client's view of a streamed chat response (content, reasoning, tool calls by index)."""
    if not (isinstance(v, list) and v and isinstance(v[0], dict) and "choices" in v[0]):
        return v
    content, reasoning, calls, finish = "", "", {}, None
    for c in v:
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                e = calls.setdefault(tc["index"], ["", ""])
                e[0] += tc.get("function", {}).get("name") or ""
                e[1] += tc.get("function", {}).get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    return {"content": content, "reasoning": reasoning, "tool_calls": [calls[k] for k in sorted(calls)], "finish": finish}


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    if a.assembled:
        # tool-call deltas may be chunked differently (complete calls only since 78ba300): compare
        # what the client assembles, not the chunk boundaries
        x = {k: assembled(v) for k, v in x.items()}
        y = {k: assembled(v) for k, v in y.items()}
    bad = 0
    for k in sorted(set(x) | set(y)):
        if x.get(k) == y.get(k):
            print(f"{k}: identical")
        else:
            bad += 1
            print(f"{k}: DIFFERENT")
    print("PASS" if bad == 0 else f"FAIL ({bad} differ)")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("capture")
    c.add_argument("--url", default="http://127.0.0.1:8093")
    c.add_argument("--prompt-ids", required=True)
    c.add_argument("--out", required=True)
    d = sp.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--assembled", action="store_true", help="compare assembled streams, not chunk lists")
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else compare(a)


if __name__ == "__main__":
    main()
