#!/usr/bin/env python3
"""Verification (a): identical greedy output, oxe-server vs llama-server.

usage: compare_greedy.py --ref http://127.0.0.1:8093 --oxe http://127.0.0.1:8094 [--n 256]

For each fixed prompt: render the chat prompt with the REFERENCE server's /apply-template
(and check oxe renders the same bytes), then run /completion with temperature 0 and
return_tokens on both and compare the token ids. Then run the same request through
/v1/chat/completions (non-stream) and compare message content / tool_calls.
Both servers must already be running (start/stop them under the inference lease).
"""
import argparse, json, sys, urllib.request

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
PROMPTS = [
    {"messages": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."}]},
    {"messages": [{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": "Explain in three short paragraphs why the sky is blue."}]},
    {"messages": [{"role": "user", "content": "What is in /etc/hostname? Use the tool."}], "tools": TOOLS},
]

def post(base, path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--oxe", required=True)
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    ok = True
    for i, p in enumerate(PROMPTS):
        pr = post(a.ref, "/apply-template", p)["prompt"]
        po = post(a.oxe, "/apply-template", p)["prompt"]
        if pr != po:
            ok = False
            print(f"[{i}] PROMPT MISMATCH"); continue
        body = {"prompt": pr, "n_predict": a.n, "temperature": 0, "cache_prompt": False, "return_tokens": True}
        r, o = post(a.ref, "/completion", body), post(a.oxe, "/completion", body)
        tr, to = r["tokens"], o["tokens"]
        same = tr == to
        first = next((k for k, (x, y) in enumerate(zip(tr, to)) if x != y), min(len(tr), len(to)))
        print(f"[{i}] tokens ref={len(tr)} oxe={len(to)} identical={same}" + ("" if same else f" first diff at {first}"))
        ok &= same
        chat = dict(p, temperature=0, max_tokens=a.n)
        cr, co = post(a.ref, "/v1/chat/completions", chat), post(a.oxe, "/v1/chat/completions", chat)
        mr, mo = cr["choices"][0]["message"], co["choices"][0]["message"]
        strip = lambda m: {k: v for k, v in m.items() if k != "tool_calls"} | {"calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls", [])]}
        cs = strip(mr) == strip(mo) and cr["choices"][0]["finish_reason"] == co["choices"][0]["finish_reason"]
        print(f"[{i}] chat message identical={cs} finish={cr['choices'][0]['finish_reason']}/{co['choices'][0]['finish_reason']}")
        ok &= cs
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
