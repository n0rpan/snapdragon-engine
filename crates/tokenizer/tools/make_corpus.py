#!/usr/bin/env python3
"""Build the tokenizer differential corpus (JSONL of {"text": ...}) on stdout.

Sources: hand-written edge cases, llama.cpp's own vocab test inputs, and fixed slices of
text/code files given as arguments (deterministic: no randomness)."""
import json, sys

cases = [
    "", " ", "  ", "\t", "\n", "\n\n", "\r\n", " \r\n\r\n ", "\t\n \t", "a b", "x　y", " line",
    "Hello world", " Hello World!", "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
    "I'm don't we'll they've she'd it's YOU'RE I'LL 'S 'T", "''''", "' ' '",
    "3", "33", "333", "3333", "1234567890", "١٢٣ ٤٥", "½ ⅓ ²",
    "Äpfel naïve café résumé ñ", "é ä ộ", "́́abc", " ́x", "ष् क्षि",
    "нещо на Български", "Ελληνικά", "עברית ושלום", "مرحبا بالعالم", "日本語のテキスト、カタカナ。", "한국어 텍스트",
    "🦙🦙 llama.cpp 👨‍👩‍👧‍👦 🏳️‍🌈 ❤️", "\U0001F600\U0001F601", "ℝ ∑ ∫ → ⇒ ≤", "☃☃☃", "﻿BOM", "​‌‍",
    "def f(x):\n    return x**2  # comment\n", "    indented\n\tTabbed\n", "if (a && b || !c) { x->y[0] = *p; }",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n", "<div class=\"a\">&amp;</div>", "{\"key\": [1, 2.5e-3, null, true]}",
    "path/to/file.rs:12:5", "https://example.com/a?b=c&d=e#f", "SELECT * FROM t WHERE x='y';",
    "   leading", "trailing   ", "multiple     spaces", "a\n\n\n\nb", "a \n b \n\n c",
    "!!!???...,,,", " !!! ", "--- *** ___", "$$\\frac{a}{b}$$", "C++ C# F# .NET",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", "<think>\n\n</think>\n\n",
    "<tool_call>\n<function=read>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>",
    "<|endoftext|>", "<|im_end|><|im_end|>", "x<|im_start|>y", "<|im_start", "<|vision_start|><|image_pad|><|vision_end|>",
    "<tool_response>\nok\n</tool_response>", "</think>text", "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "ab" * 300, "a" * 1000, " " * 50 + "x", "\n" * 20, "0" * 40, "🦙" * 30,
]

out = [dict(text=t) for t in cases]

def slices(path, n, size):
    data = open(path, encoding="utf-8", errors="replace").read()
    if not data:
        return
    step = max(1, len(data) // n)
    for i in range(n):
        s = data[i * step: i * step + size]
        if s:
            out.append(dict(text=s))

args = sys.argv[1:]
if args and args[0].endswith(".inp"):
    raw = open(args[0], encoding="utf-8").read()
    for t in raw.split("\n__ggml_vocab_test__\n"):
        out.append(dict(text=t))
    args = args[1:]
for p in args:
    slices(p, 25, 400)

for o in out:
    print(json.dumps(o, ensure_ascii=False))
