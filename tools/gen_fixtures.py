#!/usr/bin/env python3
"""Write the parser/byte-accounting test fixtures from tools/model_inspect.py.

Usage:
  python3 tools/gen_fixtures.py [MODEL_INSPECT_PY] [OUT_DIR]

For each target model it records, from model_inspect.py (the reference):
  - `--json` summary fields used by the tests (decode bytes/token, recurrent state, KV),
  - the full tensor table from its built-in parser (name, type, shape, offset, bytes),
  - the qwen35moe scalar metadata.
Only the file basename is stored, never the absolute path. Header-only: no tensor data is read.

Where model_inspect.py is (N4, audit): it is not in either repository's main. The committed
fixtures were made with the lab repository's commit fea4c56 (branch claude/model-onboarding);
extract exactly that version to regenerate them:
  git -C <lab checkout> show fea4c56:tools/model_inspect.py > /tmp/model_inspect.py
  python3 tools/gen_fixtures.py /tmp/model_inspect.py
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()
if len(sys.argv) < 2 and not os.environ.get('MODEL_INSPECT'):
    sys.exit('gen_fixtures: pass the path of model_inspect.py (or set MODEL_INSPECT); see the docstring for where it is')
INSPECT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ['MODEL_INSPECT'])  # lab fea4c56:tools/model_inspect.py
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'tests/fixtures'
MODELS = {
    'ornith': HOME / '.local/share/ai-stack/models/llm/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf',
    'qwen36': HOME / '.local/share/ai-stack/models/llm/Qwen3.6-35B-A3B-MTP-UD-Q4_K_M.gguf',
}


def main():
    spec = importlib.util.spec_from_file_location('model_inspect', INSPECT)
    mi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mi)
    try:
        commit = subprocess.run(['git', '-C', str(INSPECT.parent), 'log', '-1', '--format=%H', '--', INSPECT.name],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    OUT.mkdir(parents=True, exist_ok=True)
    for key, path in MODELS.items():
        # the --json output, exactly as the CLI prints it
        cli = json.loads(subprocess.run([sys.executable, str(INSPECT), '--json', str(path)],
                                        capture_output=True, text=True, check=True).stdout)
        parsed = mi.parse_builtin(path)
        tensors = []
        for t in parsed['tensors']:
            name, block, size = mi.type_info(t['type_id'])
            n = 1
            for d in t['shape']:
                n *= d
            tensors.append([t['name'], name, t['shape'], t['offset'], n // block * size])
        arch = cli['architecture']
        meta = {k: v for k, v in parsed['metadata'].items()
                if (k.startswith(arch + '.') or k in ('general.architecture', 'general.alignment', 'general.file_type'))
                and not isinstance(v, dict)}
        doc = {
            'source': {'tool': 'model_inspect.py', 'commit': commit, 'parser': cli['parser']},
            'file': path.name,
            'file_bytes': cli['file_bytes'],
            'gguf_version': parsed['version'],
            'data_offset': parsed['data_offset'],
            'header_bytes_read': parsed['header_bytes_read'],
            'n_metadata': len(parsed['metadata']),
            'architecture': arch,
            'layers': cli['layers'],
            'experts': {k: cli['experts'][k] for k in ('count', 'used', 'shared')},
            'kv_bytes_per_token_f16': cli['kv']['bytes_per_token_f16'],
            'recurrent_state_bytes': cli['recurrent_state_bytes'],
            'decode': cli['decode'],
            'tensor_bytes': cli['tensor_bytes'],
            'metadata': meta,
            'tensors': tensors,
        }
        out = OUT / f'{key}.json'
        with open(out, 'w') as f:
            # one tensor per line keeps diffs readable
            body = json.dumps({k: v for k, v in doc.items() if k != 'tensors'}, indent=1)[:-2]
            f.write(body + ',\n "tensors": [\n')
            f.write(',\n'.join('  ' + json.dumps(t) for t in tensors))
            f.write('\n ]\n}\n')
        print(f'{out}: {len(tensors)} tensors, active {cli["decode"]["active_bytes_per_token"]} B/token')


if __name__ == '__main__':
    main()
