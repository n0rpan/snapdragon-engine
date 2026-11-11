#!/usr/bin/env python3
"""Check a fixed353 answer written by `flash-m3 run --answer F` (or tools/run_flash_m3.sh).

usage: python3 -I tools/flash_answer_check.py RUN.json [ANSWER.txt]
  - SHA-256 of the answer text vs production's fixed353 answer (460169e2...0a4093)
  - the fixed353 code oracle (local-ai-optimization screen.py `oracle`: parse_port / parse_ports tests),
    the same check Codex's runs use; the answer is checked in a temporary directory, then deleted.
Prints one JSON line; exit 0 if the oracle passes.
"""
import os
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

ANSWER_SHA = '460169e27739a86fbb94340a3d7ffcbace214719d46fcda2cc8bb3d0fc0a4093'
SCREEN = Path(os.environ.get('FLASH_SCREEN', 'screen.py'))  # the lab's code-scheduling screen (flash-next-research)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run = Path(sys.argv[1])
    j = json.loads(run.read_text())
    ans = Path(sys.argv[2]) if len(sys.argv) > 2 else run.with_name(run.name.replace('run-', 'answer-').replace('.json', '.txt'))
    text = ans.read_bytes().decode('utf-8', errors='replace')
    sha = hashlib.sha256(text.encode()).hexdigest()
    sys.dont_write_bytecode = True
    sp = importlib.util.spec_from_file_location('fixed_flash', SCREEN)
    m = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = m
    sp.loader.exec_module(m)
    d = Path(tempfile.mkdtemp(prefix='flash-answer-'))
    try:
        res = m.oracle(text, int(j['generated']), d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    row = {'run': str(run), 'generated': j['generated'], 'answer_sha256': sha, 'sha_identical': sha == ANSWER_SHA,
           'oracle': res}
    print(json.dumps(row))
    return 0 if isinstance(res, dict) and res.get('passed') else 1


if __name__ == '__main__':
    sys.exit(main())
