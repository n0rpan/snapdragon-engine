#!/usr/bin/env python3
"""Write a tiny SYNTHETIC qwen35moe GGUF that carries the real Qwen3.6 tokenizer and chat template.

No weights are read or shipped: tensor data is deterministic filler (hash-stream bytes with the K-quant
scales pinned to 2^-7), the same layout as crates/model/src/tests.rs `mini_model_ext` (4 layers: 3
delta-net + 1 attention, 8 experts top-2, Q3_K/Q4_K/Q5_K experts, Q6_K/Q8_0 shared experts, Q6_K head).
The vocabulary and the token_embd / output rows come from tests/fixtures/qwen36-tokenizer.json.gz
(248320 tokens), so the token ids and the chat template are the real Qwen3.6 ones.

usage: python3 -I tools/ci/synth_qwen36_model.py --fixture tests/fixtures/qwen36-tokenizer.json.gz --out DIR/synth-qwen36.gguf
"""
import argparse, hashlib, os, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen36_tokenizer_fixture import load  # noqa: E402

# ggml types used here (ids as in gguf.h); block bytes as in tools/gguf_header.py
F32, F16, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 = 0, 1, 11, 12, 13, 14, 8
BLOCK = {F32: (1, 4), Q3_K: (256, 110), Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210), Q8_0: (32, 34)}
# where the f16 scale(s) sit in each block (mirrors fix_scales in crates/model/src/tests.rs)
SCALE_AT = {Q4_K: (0, 2), Q5_K: (0, 2), Q3_K: (108,), Q6_K: (208,), Q8_0: (0,)}
ONE_F16 = struct.pack('<e', 0.0078125)

# GGUF metadata value types
U32, I32, F32T, BOOL, STR, ARR, U64 = 4, 5, 6, 7, 8, 9, 10


def filler(n, seed):
    """n deterministic pseudo-random bytes (sha256 counter stream)."""
    out = bytearray()
    ctr = 0
    key = seed.to_bytes(8, 'little')
    while len(out) < n:
        out += hashlib.sha256(key + ctr.to_bytes(8, 'little')).digest()
        ctr += 1
    return bytearray(out[:n])


def qtensor_bytes(ty, n_elem, seed):
    per, bsz = BLOCK[ty]
    assert n_elem % per == 0
    b = filler(n_elem // per * bsz, seed)
    if ty in SCALE_AT:
        for blk in range(0, len(b), bsz):
            for at in SCALE_AT[ty]:
                b[blk + at:blk + at + 2] = ONE_F16
    return bytes(b)


def head_bytes(n_vocab, eos):
    """The output head (Q6_K, one block per vocabulary row), set so that greedy decoding ends turns.

    Every row is constant: value = d * scale * (q - 32) with d * scale = 1/64. Ordinary rows hold
    q = 0 (value -0.5), the end-of-generation row holds q = 63 (value +0.484). With S = sum of the
    normalised hidden state, the EOG logit is 0.484 S and every other logit is -0.5 S, so the argmax is
    the EOG token whenever S > 0 and token 0 (the first of the equal rows) otherwise. Turns therefore
    end on the real EOG token about half the time: the engine's ECHO records only ended turns, and a
    random head never stops. Synthetic test behaviour only; the values carry no language.
    """
    d = struct.pack('<e', 1.0 / 64)
    ordinary = bytes(128 + 64) + bytes([1] * 16) + d          # ql, qh, 16 int8 scales, f16 d
    eog = b'\xff' * (128 + 64) + bytes([1] * 16) + d          # q = 63 everywhere
    out = bytearray(ordinary * n_vocab)
    out[eos * 210:(eos + 1) * 210] = eog
    return bytes(out)


def f32_bytes(n_elem, seed):
    # small finite values, as the Rust mini model writes its F32 tensors
    return b''.join(struct.pack('<f', ((i * 7 + seed) % 19) * 0.01 - 0.09) for i in range(n_elem))


class Writer:
    def __init__(self):
        self.kv = bytearray()
        self.n_kv = 0
        self.tensors = []  # (name, shape, ty, bytes)

    def _key(self, k, vtype):
        kb = k.encode()
        self.kv += struct.pack('<Q', len(kb)) + kb + struct.pack('<I', vtype)
        self.n_kv += 1

    def u32(self, k, v):
        self._key(k, U32)
        self.kv += struct.pack('<I', v)

    def f32(self, k, v):
        self._key(k, F32T)
        self.kv += struct.pack('<f', v)

    def boolean(self, k, v):
        self._key(k, BOOL)
        self.kv += struct.pack('<?', v)

    def string(self, k, v):
        self._key(k, STR)
        self.kv += self._str(v)

    def arr_i32(self, k, vals):
        self._key(k, ARR)
        self.kv += struct.pack('<IQ', I32, len(vals)) + b''.join(struct.pack('<i', x) for x in vals)

    def arr_str(self, k, vals):
        self._key(k, ARR)
        self.kv += struct.pack('<IQ', STR, len(vals)) + b''.join(self._str(v) for v in vals)

    def arr_i32_fast(self, k, vals):
        self._key(k, ARR)
        self.kv += struct.pack('<IQ', I32, len(vals)) + struct.pack('<%di' % len(vals), *vals)

    @staticmethod
    def _str(v):
        vb = v.encode('utf-8')
        return struct.pack('<Q', len(vb)) + vb

    def tensor(self, name, shape, ty, data):
        self.tensors.append((name, shape, ty, data))

    def write(self, path):
        b = bytearray(b'GGUF' + struct.pack('<IQQ', 3, len(self.tensors), self.n_kv))
        b += self.kv
        offs, off = [], 0
        for name, shape, ty, data in self.tensors:
            nb = name.encode()
            b += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(shape))
            b += b''.join(struct.pack('<Q', d) for d in shape) + struct.pack('<IQ', ty, off)
            offs.append(off)
            off = (off + len(data) + 31) // 32 * 32
        b += b'\x00' * ((-len(b)) % 32)
        base = len(b)
        for (name, shape, ty, data), o in zip(self.tensors, offs):
            b += b'\x00' * (base + o - len(b))
            b += data
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(b)
        os.replace(tmp, path)


def build(fx, out):
    tok = fx['tokenizer']
    n_vocab = len(tok['tokens'])
    w = Writer()
    a = 'qwen35moe'
    w.string('general.architecture', a)
    w.string('general.name', 'synthetic-qwen36-shaped (CI, no weights)')
    w.u32('general.alignment', 32)
    for k, v in [('block_count', 4), ('nextn_predict_layers', 0), ('full_attention_interval', 4),
                 ('embedding_length', 256), ('attention.head_count', 4), ('attention.head_count_kv', 2),
                 ('attention.key_length', 128), ('attention.value_length', 128), ('expert_count', 8),
                 ('expert_used_count', 2), ('expert_feed_forward_length', 256),
                 ('expert_shared_feed_forward_length', 256), ('ssm.conv_kernel', 4), ('ssm.state_size', 128),
                 ('ssm.group_count', 1), ('ssm.time_step_rank', 2), ('ssm.inner_size', 256)]:
        w.u32(f'{a}.{k}', v)
    w.f32(f'{a}.attention.layer_norm_rms_epsilon', 1e-6)
    w.u32(f'{a}.rope.dimension_count', 64)
    w.u32(f'{a}.context_length', 262144)
    w.f32(f'{a}.rope.freq_base', 1e7)
    w.arr_i32(f'{a}.rope.dimension_sections', [11, 11, 10, 0])
    # the real tokenizer and chat template (fixture)
    w.string('tokenizer.ggml.model', tok['model'])
    w.string('tokenizer.ggml.pre', tok['pre'])
    w.arr_str('tokenizer.ggml.tokens', tok['tokens'])
    w.arr_i32_fast('tokenizer.ggml.token_type', tok['token_type'])
    w.arr_str('tokenizer.ggml.merges', tok['merges'])
    for k in ('bos_token_id', 'eos_token_id', 'padding_token_id'):
        if tok.get(k) is not None:
            w.u32(f'tokenizer.ggml.{k}', tok[k])
    w.boolean('tokenizer.ggml.add_bos_token', tok['add_bos_token'])
    w.string('tokenizer.chat_template', fx['chat_template'])

    e = 256
    # embedding: one row per vocabulary token (Q4_K: one 256-wide block per row)
    w.tensor('token_embd.weight', [e, n_vocab], Q4_K, qtensor_bytes(Q4_K, e * n_vocab, 1))
    w.tensor('output_norm.weight', [e], F32, f32_bytes(e, 2))
    w.tensor('output.weight', [e, n_vocab], Q6_K, head_bytes(n_vocab, tok['eos_token_id']))
    q = lambda ty, shape, s: (shape, ty, qtensor_bytes(ty, _prod(shape), s))
    f = lambda shape, s: (shape, F32, f32_bytes(_prod(shape), s))

    def put(name, spec):
        shape, ty, data = spec
        w.tensor(name, shape, ty, data)

    for il in range(4):
        p = lambda s: f'blk.{il}.{s}'
        s = 100 * (il + 1)
        put(p('attn_norm.weight'), f([e], s))
        put(p('post_attention_norm.weight'), f([e], s + 1))
        if il == 3:
            put(p('attn_q.weight'), q(Q4_K, [e, 1024], s + 2))
            put(p('attn_k.weight'), q(Q4_K, [e, 256], s + 3))
            put(p('attn_v.weight'), q(Q4_K, [e, 256], s + 4))
            put(p('attn_q_norm.weight'), f([128], s + 5))
            put(p('attn_k_norm.weight'), f([128], s + 6))
            put(p('attn_output.weight'), q(Q4_K, [512, e], s + 7))
        else:
            put(p('attn_qkv.weight'), q(Q4_K, [e, 512], s + 2))
            put(p('attn_gate.weight'), q(Q4_K, [e, 256], s + 3))
            put(p('ssm_beta.weight'), f([e, 2], s + 4))
            put(p('ssm_alpha.weight'), f([e, 2], s + 5))
            put(p('ssm_conv1d.weight'), f([4, 512], s + 6))
            put(p('ssm_a'), f([2], s + 7))
            put(p('ssm_dt.bias'), f([2], s + 8))
            put(p('ssm_norm.weight'), f([128], s + 9))
            put(p('ssm_out.weight'), q(Q4_K, [256, e], s + 10))
        xt = [Q3_K, Q4_K, Q5_K, Q4_K][il]
        put(p('ffn_gate_inp.weight'), f([e, 8], s + 11))
        put(p('ffn_gate_inp_shexp.weight'), f([e], s + 12))
        put(p('ffn_gate_exps.weight'), q(xt, [e, 256, 8], s + 13))
        put(p('ffn_up_exps.weight'), q(xt, [e, 256, 8], s + 14))
        put(p('ffn_down_exps.weight'), q(xt, [256, e, 8], s + 15))
        st = Q8_0 if il == 1 else Q6_K
        put(p('ffn_gate_shexp.weight'), q(st, [e, 256], s + 16))
        put(p('ffn_up_shexp.weight'), q(st, [e, 256], s + 17))
        put(p('ffn_down_shexp.weight'), q(Q6_K, [256, e], s + 18))
    w.write(out)
    return n_vocab


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    fx = load(a.fixture)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n = build(fx, a.out)
    print(f'wrote {a.out}: {os.path.getsize(a.out)} bytes, vocab {n}, synthetic weights (no real weights)')


if __name__ == '__main__':
    main()
