#!/usr/bin/env python3
"""WP-D1 step 1 (runq/004): server decode cost per token vs `oxe bench`, same token ids.

Run against a server started with OXE_TIMING_DIAG=1 and stderr redirected to --log.
Each arm sends the same prompt ids, temperature 0 (unless the arm says otherwise),
ignore_eos, n_predict = warmup + tokens, cache_prompt false, and reads the server's
`OXE_DIAG {"phase":"request",...}` line for that request.

  python3 tools/d1_server_split.py --url http://127.0.0.1:8093 --log server.err \
      --prompt-ids 1,2,3 --tokens 256 --warmup 16 --reps 3 --arms completion,completion_stream,chat_stream \
      --label ornith-plain --out d1-ornith-plain.json

Output JSON: per arm, per rep the diag split; per arm the mean ms/token of decode_ms and of
each component (engine, copy, sample, probs, text, emit, other = loop - sum).
"""
import argparse, json, statistics, time, urllib.request


def post(url, body, stream):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        if stream:
            n = 0
            for line in r:
                if line.startswith(b"data:"):
                    n += 1
            return {"frames": n, "wall_s": time.perf_counter() - t}
        j = json.loads(r.read())
        return {"frames": 0, "wall_s": time.perf_counter() - t, "timings": j.get("timings")}


def last_diag(log, start_off):
    with open(log, "rb") as f:
        f.seek(start_off)
        lines = f.read().decode(errors="replace").splitlines()
    ds = [json.loads(l[len("OXE_DIAG "):]) for l in lines if l.startswith("OXE_DIAG ")]
    ds = [d for d in ds if d.get("phase") == "request"]
    return ds[-1] if ds else None


def size(log):
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            return f.tell()
    except FileNotFoundError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8093")
    ap.add_argument("--log", required=True, help="the server's stderr file (OXE_TIMING_DIAG=1)")
    ap.add_argument("--prompt-ids", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--arms", default="completion,completion_stream,chat_stream,sampled_stream")
    ap.add_argument("--chat-text", default="Write a Python function that parses an ISO-8601 date string and returns a datetime. Include a docstring and three unit tests.")
    ap.add_argument("--label", default="")
    ap.add_argument("--discard", type=int, default=0, help="requests sent and ignored before each arm (cold first use)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ids = [int(x) for x in a.prompt_ids.split(",") if x]
    n_pred = a.warmup + a.tokens
    res = {"label": a.label, "prompt_tokens": len(ids), "n_predict": n_pred, "arms": {}}
    for arm in a.arms.split(","):
        stream = arm.endswith("_stream")
        if arm.startswith("chat"):
            url = a.url + "/v1/chat/completions"
            body = {"messages": [{"role": "user", "content": a.chat_text}], "max_tokens": n_pred, "temperature": 0,
                    "ignore_eos": True, "cache_prompt": False, "stream": stream}
        else:
            url = a.url + "/completion"
            body = {"prompt": ids, "n_predict": n_pred, "temperature": 0, "ignore_eos": True, "cache_prompt": False,
                    "stream": stream}
            if arm.startswith("sampled"):
                body.update({"temperature": 0.8, "seed": 1})
        for _ in range(a.discard):
            post(url, body, stream)
        reps = []
        for _ in range(a.reps):
            off = size(a.log)
            r = post(url, body, stream)
            time.sleep(0.3)  # let the server flush its stderr line
            d = last_diag(a.log, off)
            if d is None:
                raise SystemExit(f"no OXE_DIAG request line in {a.log} after arm {arm}: is OXE_TIMING_DIAG=1 set?")
            r["diag"] = d
            reps.append(r)
        def per_tok(k):
            return statistics.mean(x["diag"][k] / max(1, x["diag"]["generated_n"]) for x in reps)
        comp = ["engine_ms", "copy_ms", "sample_ms", "probs_ms", "text_ms", "emit_ms"]
        summ = {k: per_tok(k) for k in ["decode_ms", "loop_ms"] + comp}
        summ["other_ms"] = summ["loop_ms"] - sum(summ[k] for k in comp)
        # compare with `oxe bench` mean ms/token: loop time per engine call (the first token
        # comes from the prompt's logits, so calls = generated - 1)
        summ["loop_ms_per_call"] = statistics.mean(x["diag"]["loop_ms"] / max(1, x["diag"]["engine_calls"]) for x in reps)
        # bench-matched (OXE_TIMING_DIAG_WARMUP, default 16 engine calls discarded, like `oxe bench --warmup 16`)
        w = [x["diag"] for x in reps if x["diag"].get("engine_calls_warm")]
        if w:
            summ["warm_loop_ms_per_call"] = statistics.mean(d["loop_ms_warm"] / d["engine_calls_warm"] for d in w)
            summ["warm_engine_ms_per_call"] = statistics.mean(d["engine_ms_warm"] / d["engine_calls_warm"] for d in w)
            summ["warm_loop_ms_per_token"] = statistics.mean(d["loop_ms_warm"] / max(1, d["tokens_warm"]) for d in w)
        summ["engine_ms_per_call"] = statistics.mean(x["diag"]["engine_ms"] / max(1, x["diag"]["engine_calls"]) for x in reps)
        summ["checkpoint_ms"] = statistics.mean(x["diag"]["checkpoint_ms"] for x in reps)
        res["arms"][arm] = {"per_token_ms": summ, "reps": reps}
        if "warm_loop_ms_per_call" in summ:
            print(f"{a.label} {arm}: AFTER WARM-UP loop/call {summ['warm_loop_ms_per_call']:.3f} ms, engine/call {summ['warm_engine_ms_per_call']:.3f}, "
                  f"loop/token {summ['warm_loop_ms_per_token']:.3f}", flush=True)
        print(f"{a.label} {arm}: loop/call {summ['loop_ms_per_call']:.3f} ms, engine/call {summ['engine_ms_per_call']:.3f}; per token (engine {summ['engine_ms']:.3f}, copy {summ['copy_ms']:.3f}, "
              f"sample {summ['sample_ms']:.3f}, text {summ['text_ms']:.3f}, emit {summ['emit_ms']:.3f}, other {summ['other_ms']:.3f})", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
