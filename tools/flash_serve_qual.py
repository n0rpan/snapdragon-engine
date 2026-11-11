#!/usr/bin/env python3
"""Client-side qualification for an OpenAI/llama-server-compatible endpoint (oxe-server).

Subcommands: capture, compare, teacher, toolcall, timing, cancel, selftest.  Python 3 stdlib only;
run as `python3 -I tools/flash_serve_qual.py <cmd>`.
"""
import argparse
import contextlib
import hashlib
import http.client
import io
import json
import os
import re
import socket
import statistics
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TIMEOUT = 1800
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "prompts")
PROMPTS = [("p1-code", "p1-code.txt"), ("p2-prose", "p2-prose.txt"), ("p3-chat", "p3-chat.txt")]
THINK_OFF = {"enable_thinking": False}
TEACHER_CAP = 128
CHATML = re.compile(r"<\|im_start\|>(system|user)\n(.*?)<\|im_end\|>", re.S)
# Direct connections only: an inherited http_proxy must not capture LAN/loopback traffic.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOOL = {"type": "function", "function": {
    "name": "read", "description": "Read a file from the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
TOOL_SYSTEM = "You are a coding agent. To inspect a file you MUST call the read tool; never guess file contents."
TOOL_USER = "Read the file src/main.rs and tell me the name of the function it defines."
TOOL_RESULT = "fn compute_checksum(data: &[u8]) -> u32 { 0 }"


def read_key(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.chmod(path, 0o600)  # the mode only applies on creation; enforce it for existing files


def post(base, path, body, key=None, timeout=TIMEOUT):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat_body(messages, max_tokens, model, stream=False):
    """The chat request shape shared by timing and cancel (same fields as capture)."""
    body = {"messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": stream,
            "cache_prompt": True, "chat_template_kwargs": THINK_OFF}
    if model:
        body["model"] = model
    return body


def parse_chatml(text):
    """System/user messages only; the trailing open assistant turn is not matched.
    A file without ChatML markers (p1/p2 are plain text) is one user message."""
    if "<|im_start|>" not in text:
        return [{"role": "user", "content": text.strip()}]
    return [{"role": r, "content": c} for r, c in CHATML.findall(text)]


def load_prompts():
    out = []
    for name, fn in PROMPTS:
        with open(os.path.join(PROMPT_DIR, fn), encoding="utf-8") as f:
            out.append({"name": name, "messages": parse_chatml(f.read())})
    return out


def tokenize(base, text, parse_special, key):
    r = post(base, "/tokenize", {"content": text, "add_special": False, "parse_special": parse_special}, key)
    return [int(t) for t in r["tokens"]]


def next_token(base, ids, key):
    """One greedy token id from /completion; needs return_tokens (ids are in "tokens")."""
    r = post(base, "/completion", {"prompt": ids, "n_predict": 1, "temperature": 0, "cache_prompt": True,
                                   "stream": False, "return_tokens": True}, key)
    toks = r.get("tokens") or []
    if not toks:
        raise RuntimeError(f"no token id in /completion response (keys: {sorted(r)})")
    return int(toks[0])


def cmd_capture(a):
    key = read_key(a.api_key_file)
    records = []
    for p in load_prompts():
        body = {"messages": p["messages"], "temperature": 0, "max_tokens": a.max_tokens, "stream": False,
                "cache_prompt": True, "chat_template_kwargs": THINK_OFF}
        if a.model:
            body["model"] = a.model
        t0 = time.monotonic()
        r = post(a.url, "/v1/chat/completions", body, key)
        wall = time.monotonic() - t0
        ch = r["choices"][0]
        content = ch["message"].get("content") or ""
        records.append({"name": p["name"], "messages": p["messages"], "content": content,
                        "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                        "timings": r.get("timings"), "wall_s": round(wall, 3)})
        print(f"{p['name']}: finish={ch.get('finish_reason')} chars={len(content)} wall={wall:.1f}s")
    write_json(a.out, {"model": a.model, "max_tokens": a.max_tokens, "prompts": records})
    return 0


def cmd_compare(a):
    ref = {p["name"]: p for p in read_json(a.ref)["prompts"]}
    test = {p["name"]: p for p in read_json(a.test)["prompts"]}
    n_ok = 0
    for name, rp in ref.items():
        ra = rp["content"]
        if name not in test:
            print(f"{name}: missing in TEST")
            continue
        ta = test[name]["content"]
        same = ra == ta
        n_ok += same
        k = len(os.path.commonprefix([ra, ta]))
        print(f"{name}: identical {'yes' if same else 'no'} common_prefix={k} chars "
              f"(ref {len(ra)}, test {len(ta)})")
        if not same:
            print(f"  ref : {ra[k:k + 60]!r}")
            print(f"  test: {ta[k:k + 60]!r}")
    print(f"IDENTICAL {n_ok}/{len(ref)}")
    return 0


def cmd_teacher(a):
    key = read_key(a.api_key_file)
    total = agree_total = 0
    flips, per_prompt = [], []
    for p in read_json(a.ref)["prompts"]:
        prompt = post(a.url, "/apply-template", {"messages": p["messages"],
                                                 "chat_template_kwargs": THINK_OFF}, key)["prompt"]
        prompt_ids = tokenize(a.url, prompt, True, key)
        ref_ids = tokenize(a.url, p["content"], False, key)[:TEACHER_CAP]
        ag = 0
        for i, want in enumerate(ref_ids):
            got = next_token(a.url, prompt_ids + ref_ids[:i], key)
            if got == want:
                ag += 1
            else:
                flips.append({"prompt": p["name"], "index": i, "expected": want, "got": got})
        print(f"{p['name']}: agree {ag}/{len(ref_ids)}")
        per_prompt.append({"name": p["name"], "agree": ag, "n": len(ref_ids)})
        total += len(ref_ids)
        agree_total += ag
    pct = 100.0 * agree_total / total if total else 0.0
    print(f"TOP1 {pct:.2f}% (flips {len(flips)} of {total})")
    write_json(a.out, {"min_agree": a.min_agree, "agree": agree_total, "total": total,
                       "top1_pct": round(pct, 4), "per_prompt": per_prompt, "flips": flips})
    return 0 if pct >= a.min_agree else 1


def cmd_toolcall(a):
    key = read_key(a.api_key_file)
    base_msgs = [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": TOOL_USER}]

    def request(messages):
        body = {"messages": messages, "tools": [TOOL], "temperature": 0, "max_tokens": 256, "stream": False,
                "cache_prompt": True, "chat_template_kwargs": THINK_OFF}
        if a.model:
            body["model"] = a.model
        t0 = time.monotonic()
        r = post(a.url, "/v1/chat/completions", body, key)
        return r, round(time.monotonic() - t0, 3)

    fails1, fails2 = [], []
    r1, w1 = request(base_msgs)
    ch1 = r1["choices"][0]
    msg1 = ch1["message"]
    calls = msg1.get("tool_calls") or []
    if ch1.get("finish_reason") != "tool_calls":
        fails1.append(f"request 1 finish_reason is {ch1.get('finish_reason')!r}, not 'tool_calls'")
    if not calls:
        fails1.append("request 1 returned no tool_calls")
    else:
        fn = calls[0].get("function") or {}
        if fn.get("name") != "read":
            fails1.append(f"first tool call is {fn.get('name')!r}, not 'read'")
        raw = fn.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            args = None
        if not (isinstance(args, dict) and isinstance(args.get("path"), str)):
            fails1.append(f"arguments are not JSON with a string 'path': {raw!r}")

    r2 = w2 = None
    if not fails1:
        asst = {k: msg1[k] for k in ("role", "content", "tool_calls") if k in msg1}
        msgs2 = base_msgs + [asst, {"role": "tool", "tool_call_id": calls[0].get("id", ""),
                                    "content": TOOL_RESULT}]
        r2, w2 = request(msgs2)
        ch2 = r2["choices"][0]
        text = ch2["message"].get("content") or ""
        if ch2.get("finish_reason") != "stop":
            fails2.append(f"request 2 finish_reason is {ch2.get('finish_reason')!r}, not 'stop'")
        if "compute_checksum" not in text:
            fails2.append("request 2 content does not contain 'compute_checksum'")
        print(f"request 2 cache_n={(r2.get('timings') or {}).get('cache_n')} wall={w2:.2f}s")
    else:
        fails2.append("skipped, part 1 failed")

    print(f"part 1 wall={w1:.2f}s: {'PASS' if not fails1 else 'FAIL'}")
    print(f"part 2: {'PASS' if not fails2 else 'FAIL'}")
    write_json(a.out, {"response1": r1, "wall1_s": w1, "response2": r2, "wall2_s": w2,
                       "part1_failures": fails1, "part2_failures": fails2})
    if fails1 or fails2:
        print("TOOLCALL FAIL: " + "; ".join(fails1 + fails2))
        return 1
    print("TOOLCALL PASS")
    return 0


def cmd_timing(a):
    if a.runs < 2:
        sys.stderr.write("timing: --runs must be >= 2 (the first run is a warm-up and is dropped)\n")
        return 2
    key = read_key(a.api_key_file)
    prompt = next(p for p in load_prompts() if p["name"] == a.prompt)
    body = chat_body(prompt["messages"], a.max_tokens, a.model)
    runs = []
    for i in range(a.runs):
        t0 = time.monotonic()
        r = post(a.url, "/v1/chat/completions", body, key)
        wall = time.monotonic() - t0
        text = r["choices"][0]["message"].get("content") or ""
        t = r.get("timings") or {}
        if r.get("timings"):
            source, n, ms = "timings", t.get("predicted_n"), t.get("predicted_ms")
        else:
            source = "wall"
            n = (r.get("usage") or {}).get("completion_tokens")
            ms = wall * 1000.0
        rec = {"run": i, "source": source, "predicted_n": n, "predicted_ms": ms,
               "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"), "cache_n": t.get("cache_n"),
               "wall_s": round(wall, 3), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "ms_per_token": None}
        if not n or ms is None:
            rec["error"] = "predicted_n is 0 or missing: no tokens generated"
        else:
            rec["ms_per_token"] = ms / n
        runs.append(rec)
        shown = "ERROR" if rec["ms_per_token"] is None else f"{rec['ms_per_token']:.2f}"
        print(f"run {i}: {source} predicted_n={n} ms/token={shown} wall={wall:.2f}s")
    warm = runs[1:]  # the first run is a warm-up
    errors = [r for r in warm if "error" in r]
    vals = [r["ms_per_token"] for r in warm if "error" not in r]
    all_same = len({r["text_sha256"] for r in runs}) == 1
    med = statistics.median(vals) if vals else None
    lo = min(vals) if vals else None
    hi = max(vals) if vals else None
    if not vals or errors:
        verdict = "FAIL"
    elif a.baseline_ms is None:
        verdict = "INFO"
    else:
        verdict = "PASS" if med <= a.baseline_ms * (1 + a.max_slower_pct / 100) else "FAIL"
    base = "none" if a.baseline_ms is None else f"{a.baseline_ms:.2f}"
    if med is None:
        print(f"TIMING {verdict} no valid warm runs baseline={base}")
    else:
        print(f"TIMING {verdict} median={med:.2f} ms/token (n={len(vals)}, range {lo:.2f}-{hi:.2f}) baseline={base}")
    write_json(a.out, {"url": a.url, "model": a.model, "prompt": a.prompt, "max_tokens": a.max_tokens,
                       "runs": runs, "warm_ms_per_token": vals, "median_ms_per_token": med,
                       "min_ms_per_token": lo, "max_ms_per_token": hi, "all_texts_identical": all_same,
                       "baseline_ms": a.baseline_ms, "max_slower_pct": a.max_slower_pct, "verdict": verdict})
    return 1 if verdict == "FAIL" else 0


def cmd_cancel(a):
    key = read_key(a.api_key_file)
    u = urllib.parse.urlsplit(a.url)
    path = u.path.rstrip("/") + "/v1/chat/completions"
    long_body = chat_body([{"role": "user", "content": CANCEL_LONG_MSG}], a.long_tokens, a.model, stream=True)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    # (a) streamed long request: read for --cancel-after-s, then drop the connection mid-generation
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=TIMEOUT)  # direct, no proxy
    conn.request("POST", path, body=json.dumps(long_body).encode(), headers=headers)
    sock = conn.sock  # kept here: http.client detaches conn.sock once the response is read
    resp = conn.getresponse()
    stream_status, got = resp.status, 0
    deadline = time.monotonic() + a.cancel_after_s
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            break
        sock.settimeout(left)
        try:
            chunk = resp.read1(4096)
        except socket.timeout:
            break
        if not chunk:
            break  # the server finished before the cancel point
        got += len(chunk)
    resp.close()
    conn.close()
    print(f"streamed request: status={stream_status} bytes_before_cancel={got}")
    # (b) short non-streamed request through the normal helper, timed
    short_body = chat_body([{"role": "user", "content": "Reply with OK."}], 8, a.model)
    t0 = time.monotonic()
    try:
        post(a.url, "/v1/chat/completions", short_body, key, timeout=a.max_wait_s)
        short_status = "ok"
    except urllib.error.HTTPError as e:
        short_status = f"HTTP {e.code}"
    except Exception as e:  # timeout, refused connection, bad JSON
        short_status = f"error: {e}"
    t_short = time.monotonic() - t0
    # (c) same long body, non-streamed, tiny max_tokens: records what the server reused after the cancel
    retry_cache_n = retry_err = None
    try:
        r = post(a.url, "/v1/chat/completions", dict(long_body, stream=False, max_tokens=16), key)
        retry_cache_n = (r.get("timings") or {}).get("cache_n")
    except Exception as e:
        retry_err = str(e)
    verdict = "PASS" if short_status == "ok" and t_short <= a.max_wait_s else "FAIL"
    write_json(a.out, {"url": a.url, "model": a.model, "long_tokens": a.long_tokens,
                       "cancel_after_s": a.cancel_after_s, "stream_status": stream_status,
                       "bytes_before_cancel": got, "t_short_s": round(t_short, 3), "short_status": short_status,
                       "retry_cache_n": retry_cache_n, "retry_error": retry_err,
                       "max_wait_s": a.max_wait_s, "verdict": verdict})
    print(f"CANCEL {verdict} short request served in {t_short:.2f} s (limit {a.max_wait_s:g})")
    return 0 if verdict == "PASS" else 1


# ---- selftest: in-process fake server -------------------------------------------------

CANNED = "Fake server reply: the quick brown fox jumps over 13 lazy dogs."
FLIP_AT = 7
TIMINGS = {"cache_n": 0, "prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 1, "predicted_ms": 1.0}
STREAM_CHUNKS = 40  # SSE chunks for a streamed chat request
STREAM_DELAY_S = 0.05
CANCEL_LONG_MSG = "List 500 distinct facts about the ocean, numbered, one per line."


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if body.get("stream") and self.path.split("?")[0] == "/v1/chat/completions":
            return self.stream_chat()
        status, out = self.server.route(self.path.split("?")[0], body)
        data = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream_chat(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for i in range(STREAM_CHUNKS):
                chunk = {"choices": [{"index": 0, "delta": {"content": CANNED[i % len(CANNED)]},
                                      "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(STREAM_DELAY_S)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client disconnected (cancel): stop quietly


class FakeServer(ThreadingHTTPServer):
    def __init__(self, mode):
        super().__init__(("127.0.0.1", 0), Fake)
        self.mode = mode  # normal | flip (wrong token at FLIP_AT) | notool (no tool call)

    def route(self, path, b):
        if path == "/tokenize":
            return 200, {"tokens": list(b.get("content", "").encode("utf-8"))}
        if path == "/apply-template":
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in b["messages"])
            return 200, {"prompt": text + "<|im_start|>assistant\n"}
        if path == "/completion":
            gen = bytes(b["prompt"]).decode("utf-8", "replace").split("<|im_start|>assistant\n")[-1]
            k = len(gen)
            ch = CANNED[k] if k < len(CANNED) else "\n"
            if self.mode == "flip" and k == FLIP_AT:
                ch = "X"
            return 200, {"content": ch, "tokens": [ord(ch)], "stop": True, "timings": TIMINGS}
        if path == "/v1/chat/completions":
            roles = [m.get("role") for m in b["messages"]]
            usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            if b.get("tools") and "tool" not in roles and self.mode != "notool":
                msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
                       "function": {"name": "read", "arguments": json.dumps({"path": "src/main.rs"})}}]}
                fr = "tool_calls"
            elif b.get("tools"):
                msg, fr = {"role": "assistant", "content": "The function is compute_checksum."}, "stop"
            else:
                msg, fr = {"role": "assistant", "content": CANNED}, "stop"
            ct = usage["completion_tokens"]
            timings = {"cache_n": 0, "prompt_n": 1, "prompt_ms": 1.0, "predicted_n": ct, "predicted_ms": ct * 100.0}
            return 200, {"choices": [{"index": 0, "finish_reason": fr, "message": msg}],
                         "usage": usage, "timings": timings}
        return 404, {"error": {"message": "File Not Found", "code": 404}}


@contextlib.contextmanager
def served(mode):
    srv = FakeServer(mode)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def run(argv):
    """Run a subcommand, echoing its stdout; returns (exit code, output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    out = buf.getvalue()
    sys.stdout.write(out)
    return rc, out


def cmd_selftest(_a):
    checks = []

    def check(cond, what):
        checks.append(bool(cond))
        print(("ok   " if cond else "FAIL ") + what)

    with tempfile.TemporaryDirectory() as td, served("normal") as url:
        ref = os.path.join(td, "ref.json")
        rc, _ = run(["capture", "--url", url, "--out", ref, "--max-tokens", "64"])
        check(rc == 0, "capture exit 0")
        prompts = read_json(ref)["prompts"]
        check([p["name"] for p in prompts] == [n for n, _ in PROMPTS], "capture prompt names")
        check(all(p["content"] == CANNED and p["finish_reason"] == "stop" for p in prompts),
              "capture content and finish_reason")
        check(all([m["role"] for m in p["messages"]] == (["system", "user"] if p["name"] == "p3-chat" else ["user"])
                  for p in prompts), "capture parsed roles (p3 has system+user, p1/p2 user only)")

        rc, out = run(["compare", ref, ref])
        check("IDENTICAL 3/3" in out, "compare identical self")
        edited = read_json(ref)
        edited["prompts"][1]["content"] = CANNED[:10] + "Z" + CANNED[11:]
        test = os.path.join(td, "test.json")
        write_json(test, edited)
        rc, out = run(["compare", ref, test])
        check(rc == 0 and "IDENTICAL 2/3" in out and "common_prefix=10" in out, "compare detects one diff")

        rc, out = run(["teacher", "--url", url, "--ref", ref, "--out", os.path.join(td, "t.json")])
        check(rc == 0 and "TOP1 100.00%" in out, "teacher 100% on normal server")

        with served("flip") as furl:
            tout = os.path.join(td, "tf.json")
            rc, out = run(["teacher", "--url", furl, "--ref", ref, "--out", tout])
            flips = read_json(tout)["flips"]
            check(rc == 1, "teacher exit 1 when below --min-agree")
            check(len(flips) == len(PROMPTS) and all(f["index"] == FLIP_AT for f in flips),
                  "teacher records one flip per prompt at the flipped index")
            check(flips and flips[0]["expected"] == ord(CANNED[FLIP_AT]) and flips[0]["got"] == ord("X"),
                  "flip record holds expected and got ids")
            rc, _ = run(["teacher", "--url", furl, "--ref", ref, "--out", tout, "--min-agree", "90"])
            check(rc == 0, "teacher exit 0 with --min-agree 90")

        rc, out = run(["toolcall", "--url", url, "--out", os.path.join(td, "tc.json")])
        check(rc == 0 and "TOOLCALL PASS" in out, "toolcall PASS on normal server")
        with served("notool") as nurl:
            rc, out = run(["toolcall", "--url", nurl, "--out", os.path.join(td, "tn.json")])
            check(rc == 1 and "TOOLCALL FAIL:" in out, "toolcall FAIL when no tool call is made")

        tm = os.path.join(td, "timing.json")
        rc, out = run(["timing", "--url", url, "--runs", "3", "--baseline-ms", "100", "--out", tm])
        tj = read_json(tm)
        check(rc == 0 and "TIMING PASS" in out, "timing PASS at baseline 100 ms/token")
        check(len(tj["runs"]) == 3 and len(tj["warm_ms_per_token"]) == 2, "timing drops the warm-up run")
        check(tj["median_ms_per_token"] is not None and abs(tj["median_ms_per_token"] - 100.0) < 1e-6,
              "timing median is 100.0 ms/token")
        check(tj["verdict"] == "PASS" and tj["all_texts_identical"] is True,
              "timing JSON verdict PASS, texts identical")
        rc, out = run(["timing", "--url", url, "--runs", "3", "--baseline-ms", "50", "--out", tm])
        check(rc == 1 and "TIMING FAIL" in out and read_json(tm)["verdict"] == "FAIL",
              "timing FAIL at baseline 50 ms/token")
        cj = os.path.join(td, "cancel.json")
        rc, out = run(["cancel", "--url", url, "--cancel-after-s", "0.2", "--max-wait-s", "5", "--out", cj])
        cs = read_json(cj)
        check(rc == 0 and "CANCEL PASS" in out and cs["verdict"] == "PASS", "cancel PASS: short request served")
        check(cs["bytes_before_cancel"] > 0 and cs["short_status"] == "ok",
              "cancel: streamed bytes received before cancel, short ok")

    if all(checks):
        print("SELFTEST PASS")
        return 0
    print(f"SELFTEST FAIL ({checks.count(False)} of {len(checks)} checks failed)")
    return 1


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def server_args(sp):
        sp.add_argument("--url", required=True, help="server base URL, e.g. http://host:8080")
        sp.add_argument("--model", help="model name sent in the request body")
        sp.add_argument("--api-key-file", help="file holding a bearer token")

    c = sub.add_parser("capture", help="record chat answers for the three fixed prompts")
    server_args(c)
    c.add_argument("--out", required=True)
    c.add_argument("--max-tokens", type=int, default=128)
    c.set_defaults(fn=cmd_capture)

    c = sub.add_parser("compare", help="compare two capture files")
    c.add_argument("ref")
    c.add_argument("test")
    c.set_defaults(fn=cmd_compare)

    c = sub.add_parser("teacher", help="teacher-forced top-1 agreement against a capture")
    c.add_argument("--url", required=True)
    c.add_argument("--api-key-file", help="file holding a bearer token")
    c.add_argument("--ref", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--min-agree", type=float, default=99.5, help="pass threshold in percent")
    c.set_defaults(fn=cmd_teacher)

    c = sub.add_parser("toolcall", help="one tool-call round trip")
    server_args(c)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_toolcall)

    c = sub.add_parser("timing", help="warm decode ms/token over repeated identical chat requests")
    server_args(c)
    c.add_argument("--runs", type=int, default=4, help="total runs; the first is a warm-up and is dropped")
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--prompt", default="p2-prose", choices=[n for n, _ in PROMPTS])
    c.add_argument("--baseline-ms", type=float, help="reference ms/token; enables PASS/FAIL")
    c.add_argument("--max-slower-pct", type=float, default=3.0)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_timing)

    c = sub.add_parser("cancel", help="check that a client disconnect frees the server")
    server_args(c)
    c.add_argument("--long-tokens", type=int, default=400)
    c.add_argument("--cancel-after-s", type=float, default=3.0)
    c.add_argument("--max-wait-s", type=float, default=10.0)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_cancel)

    c = sub.add_parser("selftest", help="run every subcommand against an in-process fake server")
    c.set_defaults(fn=cmd_selftest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
