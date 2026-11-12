#!/usr/bin/env python3
"""Extract the tokenizer + chat-template metadata of a GGUF HEADER into a small committed fixture.

Reads only the GGUF header (metadata key/values, via tools/gguf_header.py); never the tensor data.
The fixture feeds tools/ci/synth_qwen36_model.py, which builds a tiny synthetic model that carries the
real Qwen3.6 tokenizer and chat template (no weights anywhere).

NOTICE: the tokenizer and chat template are metadata of Qwen3.6-35B-A3B (Qwen, Apache-2.0; GGUF
packaging by Unsloth). The fixture contains no model weights. Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0

usage:
  python3 -I tools/ci/qwen36_tokenizer_fixture.py extract --gguf FILE --out tests/fixtures/qwen36-tokenizer.json.gz
  python3 -I tools/ci/qwen36_tokenizer_fixture.py check tests/fixtures/qwen36-tokenizer.json.gz
"""
import argparse, gzip, hashlib, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gguf_header import parse  # noqa: E402  (tools/gguf_header.py: header-only parser)

SCHEMA = 'qwen36-tokenizer-fixture/1'
NOTICE = ('Tokenizer and chat-template metadata of Qwen3.6-35B-A3B (Qwen; Apache-2.0; GGUF packaging by Unsloth). '
          'Metadata only: no model weights. Licence: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0).')
REQUIRED = ['tokenizer.ggml.model', 'tokenizer.ggml.pre', 'tokenizer.ggml.tokens', 'tokenizer.ggml.token_type',
            'tokenizer.ggml.merges', 'tokenizer.chat_template']


def extract(gguf_path):
    h = parse(gguf_path)
    kv = h['kv']
    missing = [k for k in REQUIRED if k not in kv]
    if missing:
        raise SystemExit(f'missing metadata keys: {missing}')
    # provenance: hash of the header bytes only (everything before the tensor data)
    with open(gguf_path, 'rb') as f:
        head_sha = hashlib.sha256(f.read(h['data_start'])).hexdigest()
    tok = {
        'model': kv['tokenizer.ggml.model'],
        'pre': kv['tokenizer.ggml.pre'],
        'tokens': kv['tokenizer.ggml.tokens'],
        'token_type': kv['tokenizer.ggml.token_type'],
        'merges': kv['tokenizer.ggml.merges'],
        'bos_token_id': kv.get('tokenizer.ggml.bos_token_id'),
        'eos_token_id': kv.get('tokenizer.ggml.eos_token_id'),
        'padding_token_id': kv.get('tokenizer.ggml.padding_token_id'),
        'add_bos_token': bool(kv.get('tokenizer.ggml.add_bos_token', False)),
    }
    return {
        'schema': SCHEMA,
        'notice': NOTICE,
        'source': {'model': kv.get('general.name', ''), 'header_sha256': head_sha, 'header_bytes': h['data_start']},
        'tokenizer': tok,
        'chat_template': kv['tokenizer.chat_template'],
    }


def write_fixture(doc, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    data = json.dumps(doc, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with gzip.GzipFile(out, 'wb', mtime=0, compresslevel=9) as g:
        g.write(data)
    return len(data)


def load(path):
    with gzip.open(path, 'rb') as g:
        doc = json.loads(g.read().decode('utf-8'))
    assert doc['schema'] == SCHEMA, 'fixture schema'
    assert doc['notice'] == NOTICE, 'fixture notice changed'
    t = doc['tokenizer']
    assert len(t['tokens']) == len(t['token_type']) > 0, 'token/type count'
    assert isinstance(doc['chat_template'], str) and '<|im_start|>' in doc['chat_template'], 'chat template'
    return doc


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--gguf', required=True)
    e.add_argument('--out', required=True)
    c = sub.add_parser('check')
    c.add_argument('fixture')
    a = ap.parse_args()
    if a.cmd == 'extract':
        doc = extract(a.gguf)
        n = write_fixture(doc, a.out)
        t = doc['tokenizer']
        print(f"wrote {a.out}: tokens={len(t['tokens'])} merges={len(t['merges'])} "
              f"template={len(doc['chat_template'])} chars, gz={os.path.getsize(a.out)} bytes (raw json {n})")
    else:
        doc = load(a.fixture)
        t = doc['tokenizer']
        print(f"ok: tokens={len(t['tokens'])} merges={len(t['merges'])} pre={t['pre']}")


if __name__ == '__main__':
    main()
