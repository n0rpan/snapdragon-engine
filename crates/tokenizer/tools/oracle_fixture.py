import json, subprocess, sys
oracle, model, corpus, out = sys.argv[1:5]
texts = [json.loads(l)["text"] for l in open(corpus, encoding="utf-8")]
reqs = []
for t in texts:
    reqs.append(json.dumps(dict(op="tokenize", text=t, add_special=True, parse_special=True)))
    reqs.append(json.dumps(dict(op="tokenize", text=t, add_special=False, parse_special=False)))
r = subprocess.run([oracle, model], input="\n".join(reqs) + "\n", capture_output=True, text=True, check=True)
lines = r.stdout.splitlines()
assert len(lines) == 2 * len(texts), (len(lines), len(texts))
with open(out, "w", encoding="utf-8") as f:
    for i, t in enumerate(texts):
        a, b = json.loads(lines[2 * i]), json.loads(lines[2 * i + 1])
        f.write(json.dumps(dict(text=t, special=a["ids"], plain=b["ids"]), ensure_ascii=False) + "\n")
