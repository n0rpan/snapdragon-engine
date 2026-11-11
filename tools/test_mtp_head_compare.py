"""Tests for tools/mtp_head_compare.py. Run: python3 tools/test_mtp_head_compare.py
or: python3 -m unittest tools/test_mtp_head_compare.py (use the gguf/numpy venv python).
Synthetic GGUF files only (written with gguf.GGUFWriter); never touches the real models.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mtp_head_compare as M  # noqa: E402


def write_gguf(path, tensors):
    """tensors: list of (name, float32 array in numpy order, QT or None for F32)."""
    w = GGUFWriter(path, arch="test")
    for name, x, qt in tensors:
        x = np.asarray(x, dtype=np.float32)
        if qt is None or qt == QT.F32:
            w.add_tensor(name, x)
        elif qt == QT.F16:
            w.add_tensor(name, x.astype(np.float16))
        else:
            w.add_tensor(name, quantize(x, qt), raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tensors_of(path):
    return {t.name: t for t in GGUFReader(path, "r").tensors}


def np_dequant(x, qt):
    return np.asarray(dequantize(quantize(x, qt), qt), dtype=np.float32).reshape(x.shape)


def direct_stats(a, b):
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    cos = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(b))
    return cos, rel


class TestSelection(unittest.TestCase):
    NAMES = ["token_embd.weight", "output.weight", "output_norm.weight", "rope_freqs.weight",
             "blk.0.attn_q.weight", "blk.3.ffn_up_exps.weight", "blk.5.attn_q.weight",
             "blk.38.attn_q.weight", "blk.40.nextn.eh_proj.weight", "blk.40.ffn_gate_exps.weight",
             "blk.39.ffn_down_exps.weight"]

    def test_mtp_only_block_40(self):
        self.assertEqual(M.select_names(self.NAMES, "mtp"),
                         ["blk.40.ffn_gate_exps.weight", "blk.40.nextn.eh_proj.weight"])

    def test_trunk_sample(self):
        got = set(M.select_names(self.NAMES, "trunk-sample"))
        want = {"token_embd.weight", "output.weight", "output_norm.weight", "blk.0.attn_q.weight",
                "blk.3.ffn_up_exps.weight", "blk.38.attn_q.weight", "blk.39.ffn_down_exps.weight"}
        self.assertEqual(got, want)  # blk.5 and rope_freqs not in the sample

    def test_all(self):
        self.assertEqual(sorted(M.select_names(self.NAMES, "all")), sorted(self.NAMES))

    def test_outer_selection(self):
        self.assertEqual(M.outer_selection((16, 4, 32), 4).tolist(), [0, 5, 10, 15])
        self.assertEqual(M.outer_selection((16, 4, 32), 0).tolist(), list(range(16)))
        self.assertEqual(M.outer_selection((8, 32), 4).tolist(), list(range(8)))  # 2-D ignores sample
        self.assertEqual(M.outer_selection((32,), 4).tolist(), [0])

    def test_supports_quantize(self):
        self.assertTrue(M.supports_quantize(QT.Q8_0))
        self.assertTrue(M.supports_quantize(QT.Q4_0))
        self.assertFalse(M.supports_quantize(QT.Q4_K))
        self.assertFalse(M.supports_quantize(QT.Q6_K))


class TestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="mtpcmp-")
        rng = np.random.default_rng(1234)
        cls.W = rng.standard_normal((8, 16, 64)).astype(np.float32)  # 3-D: 8 experts
        cls.W2 = rng.standard_normal((4, 256)).astype(np.float32)   # 2-D
        cls.P = (cls.W + 0.5 * rng.standard_normal(cls.W.shape)).astype(np.float32)  # perturbed
        cls.N1 = rng.standard_normal((64,)).astype(np.float32)      # 1-D norm vector
        specs = {
            "f32_a": [("w", cls.W2, None)],
            "f32_b": [("w", cls.W2, None)],
            "q8_same": [("w", cls.W, QT.Q8_0)],
            "q8_pert": [("w", cls.P, QT.Q8_0)],
            "q8_base": [("w", cls.W, QT.Q8_0)],
            "q4_a": [("w", cls.W2, QT.Q4_0)],
            "f32_w": [("w", cls.W2, None)],
            "f32_w_big": [("w", cls.W, None)],
        }
        cls.paths = {}
        for k, t in specs.items():
            p = os.path.join(cls.tmp, k + ".gguf")
            write_gguf(p, t)
            cls.paths[k] = p

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def pair(self, a, b):
        ta = tensors_of(self.paths[a])["w"]
        tb = tensors_of(self.paths[b])["w"]
        return ta, tb

    def run_compare(self, a, b, expert_sample=0, chunk_mb=M.DEFAULT_CHUNK_MB):
        ta, tb = self.pair(a, b)
        sel = M.outer_selection(M.nshape(ta), expert_sample)
        return M.compare_tensor(ta, tb, sel, chunk_mb)

    def test_round_trip_shape(self):
        t = tensors_of(self.paths["q8_same"])["w"]
        self.assertEqual(M.nshape(t), (8, 16, 64))
        self.assertEqual(t.tensor_type, QT.Q8_0)

    def test_identical_f32_pair(self):
        r = self.run_compare("f32_a", "f32_b")
        self.assertAlmostEqual(r["cos"], 1.0, places=12)
        self.assertEqual(r["rel"], 0.0)
        self.assertIsNone(r["noise_ref"])  # A is lossless: no exact reference
        self.assertEqual(set(r["noise_proxies"]), {"Q4_0", "Q8_0"})
        self.assertEqual(r["verdict"], "n/a")

    def test_identical_q8_pair_is_unchanged(self):
        r = self.run_compare("q8_same", "q8_base")
        self.assertAlmostEqual(r["cos"], 1.0, places=12)
        self.assertEqual(r["rel"], 0.0)
        self.assertLess(r["noise_ref"], 1e-6)  # B already on A's Q8_0 grid
        self.assertEqual(r["verdict"], "unchanged")

    def test_perturbed_q8_pair_is_trained(self):
        r = self.run_compare("q8_pert", "q8_same")
        self.assertLess(r["cos"], 1.0)
        self.assertGreater(r["rel"], 0.1)
        self.assertEqual(r["verdict"], "trained")

    def test_quant_noise_only_is_unchanged(self):
        # A = Q4_0(W), B = W (float). rel(A,B) and noise_ref = rel(B, Q4_0(B)) are the same quantity.
        r = self.run_compare("q4_a", "f32_w")
        self.assertEqual(r["noise_basis"], "requant_to_A_type:Q4_0")
        self.assertAlmostEqual(r["rel"], r["noise_ref"], places=6)
        self.assertAlmostEqual(r["ratio"], 1.0, places=5)
        self.assertEqual(r["verdict"], "unchanged")
        self.assertTrue(0.05 < r["noise_ref"] < 0.13)  # Q4_0 on Gaussian data, sanity band

    def test_matches_direct_numpy(self):
        r = self.run_compare("f32_a", "f32_b")
        cos, rel = direct_stats(self.W2, self.W2)
        self.assertAlmostEqual(r["cos"], cos, places=12)
        self.assertAlmostEqual(r["rel"], rel, places=12)
        r2 = self.run_compare("q8_pert", "q8_same")
        a = dequantize(quantize(self.P, QT.Q8_0), QT.Q8_0).reshape(self.P.shape)
        b = dequantize(quantize(self.W, QT.Q8_0), QT.Q8_0).reshape(self.W.shape)
        cos, rel = direct_stats(a, b)
        self.assertAlmostEqual(r2["cos"], cos, places=6)
        self.assertAlmostEqual(r2["rel"], rel, places=6)
        self.assertAlmostEqual(r2["norm_b"], float(np.linalg.norm(b)), places=3)

    def test_per_expert_stats(self):
        r = self.run_compare("q8_pert", "q8_same")
        e = r["experts"]
        self.assertEqual(e["n_total"], 8)
        self.assertEqual(e["n_selected"], 8)
        self.assertIsNone(e["selected"])
        a = dequantize(quantize(self.P, QT.Q8_0), QT.Q8_0).reshape(self.P.shape)
        b = dequantize(quantize(self.W, QT.Q8_0), QT.Q8_0).reshape(self.W.shape)
        cos_each = [direct_stats(a[i], b[i])[0] for i in range(8)]
        rel_each = [direct_stats(a[i], b[i])[1] for i in range(8)]
        self.assertAlmostEqual(e["cos_min"], min(cos_each), places=6)
        self.assertAlmostEqual(e["cos_max"], max(cos_each), places=6)
        self.assertAlmostEqual(e["cos_median"], float(np.median(cos_each)), places=6)
        self.assertAlmostEqual(e["rel_median"], float(np.median(rel_each)), places=6)

    def test_expert_sample(self):
        r = self.run_compare("q8_pert", "q8_same", expert_sample=4)
        e = r["experts"]
        self.assertEqual(e["selected"], [0, 2, 5, 7])  # linspace(0,7,4) rounded
        self.assertEqual(e["n_selected"], 4)
        a = dequantize(quantize(self.P, QT.Q8_0), QT.Q8_0).reshape(self.P.shape)
        b = dequantize(quantize(self.W, QT.Q8_0), QT.Q8_0).reshape(self.W.shape)
        sub_a, sub_b = a[[0, 2, 5, 7]], b[[0, 2, 5, 7]]
        cos, rel = direct_stats(sub_a, sub_b)
        self.assertAlmostEqual(r["cos"], cos, places=6)  # global stats cover the sample only
        self.assertAlmostEqual(r["rel"], rel, places=6)

    def test_chunked_equals_unchunked(self):
        for kwargs in ({"expert_sample": 0}, {"expert_sample": 3}):
            big = self.run_compare("q8_pert", "q8_same", chunk_mb=4096, **kwargs)
            tiny = self.run_compare("q8_pert", "q8_same", chunk_mb=1e-5, **kwargs)  # one expert per chunk
            for key in ("cos", "rel", "norm_a", "norm_b", "noise_ref"):
                self.assertAlmostEqual(big[key], tiny[key], delta=1e-12 * max(1.0, abs(big[key])), msg=key)
            for key in ("cos_min", "cos_median", "cos_max", "rel_median", "rel_max"):
                self.assertAlmostEqual(big["experts"][key], tiny["experts"][key], places=12, msg=key)

    def test_chunked_equals_unchunked_2d_f32(self):
        big = self.run_compare("f32_a", "f32_b", chunk_mb=4096)
        tiny = self.run_compare("f32_a", "f32_b", chunk_mb=1e-5)
        self.assertEqual(big["rel"], tiny["rel"])
        self.assertEqual(big["cos"], tiny["cos"])

    def test_cols_not_block_multiple_skips_proxies(self):
        # Like ssm_conv1d (8192 x 4): last dim 4 is not a multiple of the Q4_0/Q8_0 block size 32.
        small = np.random.default_rng(7).standard_normal((50, 4)).astype(np.float32)
        p_a = os.path.join(self.tmp, "conv_a.gguf")
        p_b = os.path.join(self.tmp, "conv_b.gguf")
        write_gguf(p_a, [("conv", small, None)])
        write_gguf(p_b, [("conv", small * 1.1, None)])
        ta = tensors_of(p_a)["conv"]
        tb = tensors_of(p_b)["conv"]
        r = M.compare_tensor(ta, tb, M.outer_selection(M.nshape(ta), 0))
        self.assertIsNone(r["noise_proxies"])
        self.assertIsNone(r["noise_ref"])
        self.assertEqual(r["verdict"], "n/a")
        self.assertAlmostEqual(r["rel"], 0.1 / 1.1, places=5)  # ||A-B|| / ||B|| with B = 1.1 A
        self.assertTrue(np.isfinite(r["cos"]))

    def test_one_d_tensor(self):
        p_a = os.path.join(self.tmp, "n_a.gguf")
        p_b = os.path.join(self.tmp, "n_b.gguf")
        write_gguf(p_a, [("norm", self.N1, None)])
        write_gguf(p_b, [("norm", self.N1 * 1.0, None)])
        ta = tensors_of(p_a)["norm"]
        tb = tensors_of(p_b)["norm"]
        r = M.compare_tensor(ta, tb, M.outer_selection(M.nshape(ta), 0))
        self.assertAlmostEqual(r["cos"], 1.0, places=12)
        self.assertEqual(r["rel"], 0.0)
        self.assertIsNone(r["experts"])

    def test_main_end_to_end(self):
        a = os.path.join(self.tmp, "ma.gguf")
        b = os.path.join(self.tmp, "mb.gguf")
        write_gguf(a, [
            ("blk.40.ffn_gate_exps.weight", self.W, QT.Q8_0),
            ("blk.40.nextn.eh_proj.weight", self.W2, QT.Q8_0),
            ("blk.0.attn_q.weight", self.P, QT.Q8_0),
            ("token_embd.weight", self.W2, QT.Q8_0),
            ("only_a.weight", self.W2, None),
            ("blk.40.shape_mismatch", self.W2, None),
        ])
        write_gguf(b, [
            ("blk.40.ffn_gate_exps.weight", self.W, QT.Q8_0),
            ("blk.40.nextn.eh_proj.weight", self.W2, QT.Q8_0),
            ("blk.0.attn_q.weight", self.W, QT.Q8_0),
            ("token_embd.weight", self.W2, QT.Q8_0),
            ("blk.40.shape_mismatch", self.W2.T.copy(), None),
        ])
        out = os.path.join(self.tmp, "out.json")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(["--a", a, "--b", b, "--set", "all", "--expert-sample", "3", "--json", out])
        self.assertEqual(rc, 0)
        with open(out) as f:
            data = json.load(f)
        names = {r["name"] for r in data["tensors"]}
        self.assertEqual(names, {"blk.40.ffn_gate_exps.weight", "blk.40.nextn.eh_proj.weight",
                                 "blk.0.attn_q.weight", "token_embd.weight"})
        self.assertEqual(data["skipped"]["only_in_a"], ["only_a.weight"])
        self.assertEqual(data["skipped"]["shape_mismatch"], ["blk.40.shape_mismatch"])
        by = {r["name"]: r for r in data["tensors"]}
        self.assertEqual(by["blk.40.ffn_gate_exps.weight"]["verdict"], "unchanged")
        self.assertEqual(by["blk.0.attn_q.weight"]["verdict"], "trained")
        self.assertEqual(by["blk.40.ffn_gate_exps.weight"]["experts"]["n_selected"], 3)
        self.assertIn("mtp", data["summary"])
        self.assertIn("trunk", data["summary"])
        self.assertIn("by_layer", data["summary"]["trunk"])
        self.assertIn("embd_head", data["summary"])
        self.assertIn("wrote", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
