"""
Cross-check the NumPy golden model (tools/q80.py) against the C runtime.

Requires models/stories260K_q80.bin + tok512.bin and the host build
(make -C sw/runtime host). Run inside the dev container:
    pytest -q tests/test_golden_model.py
"""
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from q80 import Q80Model, QTensor, gemv_q80, quantize_q80  # noqa: E402

M = os.path.join(ROOT, "models")
CKPT = os.path.join(M, "stories260K_q80.bin")
TOK = os.path.join(M, "tok512.bin")
EXE = os.path.join(ROOT, "sw", "runtime", "build", "host", "run_int8")
need = pytest.mark.skipif(not (os.path.exists(CKPT) and os.path.exists(EXE)),
                          reason="model or host build missing")


def load_vocab(path):
    with open(path, "rb") as f:
        data = f.read()
    (n_max,) = struct.unpack_from("<i", data, 0)
    off, vocab = 4, []
    while off < len(data):
        _, ln = struct.unpack_from("<fi", data, off)
        off += 8
        vocab.append(data[off: off + ln])
        off += ln
    return vocab


def test_quantize_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256).astype(np.float32)
    q = quantize_q80(x, 32)
    assert q.q.dtype == np.int8 and np.abs(q.q).max() == 127
    deq = q.q.reshape(-1, 32) * q.s.reshape(-1, 1)
    assert np.max(np.abs(deq.ravel() - x)) <= q.s.max() / 2 + 1e-7


def test_gemv_psum_exact():
    rng = np.random.default_rng(1)
    w = QTensor(rng.integers(-128, 128, (16, 64), dtype=np.int8), np.ones((16, 2), np.float32))
    x = QTensor(rng.integers(-128, 128, (1, 64), dtype=np.int8), np.ones((1, 2), np.float32))
    psum, out = gemv_q80(w, x, 32)
    ref = np.einsum("rgk,gk->rg", w.q.reshape(16, 2, 32).astype(np.int64),
                    x.q.reshape(2, 32).astype(np.int64))
    assert np.array_equal(psum, ref)
    assert np.allclose(out, psum.sum(1))


@need
def test_greedy_matches_c_runtime():
    steps = 48
    vocab = load_vocab(TOK)
    m = Q80Model(CKPT)
    tok, out = 1, b""
    for pos in range(steps - 1):
        nxt = int(np.argmax(m.forward(tok, pos)))
        if nxt == 1:
            break
        piece = vocab[nxt]
        if tok == 1 and piece.startswith(b" "):
            piece = piece[1:]  # same BOS rule as decode() in run_int8.c
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            piece = bytes([int(piece[3:5], 16)])  # raw byte token, e.g. <0x0A>
        out += piece
        tok = nxt
    c = subprocess.run([EXE, CKPT, "-z", TOK, "-t", "0", "-n", str(steps), "-i", ""],
                       capture_output=True, check=True).stdout
    assert c.strip().startswith(out.strip()), (out, c)
