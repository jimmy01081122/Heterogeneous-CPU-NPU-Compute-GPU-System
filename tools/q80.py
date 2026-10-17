"""
q80.py - reader + NumPy reference model for llama2.c Q8_0 (version 2) checkpoints.

Serves as the golden model for accelerator verification:
  * `load_q80()`       parses the exact on-disk layout used by run_int8.c
  * `quantize_q80()`   activation quantization identical to run_int8.c quantize()
  * `gemv_q80()`       returns both the exact INT32 per-group partial sums (what an
                       NPU computes) and the dequantized float output
  * `Q80Model.forward()` full token step; `capture=` hook exposes intermediates

Integer results are bit-exact vs the C runtime; float results match to ~1e-6
relative (summation order differs).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

MAGIC = 0x616B3432
HEADER_BYTES = 256


@dataclass
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool
    group_size: int

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.dim * self.n_kv_heads // self.n_heads


@dataclass
class QTensor:
    q: np.ndarray  # int8, shape (rows, cols)
    s: np.ndarray  # float32, shape (rows, cols // GS)


@dataclass
class Weights:
    rms_att: np.ndarray
    rms_ffn: np.ndarray
    rms_final: np.ndarray
    tokens: QTensor
    wq: List[QTensor] = field(default_factory=list)
    wk: List[QTensor] = field(default_factory=list)
    wv: List[QTensor] = field(default_factory=list)
    wo: List[QTensor] = field(default_factory=list)
    w1: List[QTensor] = field(default_factory=list)
    w2: List[QTensor] = field(default_factory=list)
    w3: List[QTensor] = field(default_factory=list)
    wcls: Optional[QTensor] = None


def load_q80(path: str) -> tuple[Config, Weights]:
    raw = np.fromfile(path, dtype=np.uint8)
    magic, version = struct.unpack_from("<Ii", raw, 0)
    if magic != MAGIC or version != 2:
        raise ValueError(f"{path}: not a Q8_0 v2 checkpoint (magic={magic:#x}, version={version})")
    dim, hidden, nl, nh, nkv, vocab, seq = struct.unpack_from("<7i", raw, 8)
    shared = bool(raw[36])
    (gs,) = struct.unpack_from("<i", raw, 37)
    cfg = Config(dim, hidden, nl, nh, nkv, vocab, seq, shared, gs)

    off = HEADER_BYTES

    def f32(n: int) -> np.ndarray:
        nonlocal off
        a = raw[off: off + 4 * n].view(np.float32).copy()
        off += 4 * n
        return a

    def qt(rows: int, cols: int) -> QTensor:
        nonlocal off
        n = rows * cols
        q = raw[off: off + n].view(np.int8).reshape(rows, cols).copy()
        off += n
        s = raw[off: off + 4 * (n // gs)].view(np.float32).reshape(rows, cols // gs).copy()
        off += 4 * (n // gs)
        return QTensor(q, s)

    rms_att = f32(nl * dim).reshape(nl, dim)
    rms_ffn = f32(nl * dim).reshape(nl, dim)
    rms_final = f32(dim)
    w = Weights(rms_att, rms_ffn, rms_final, tokens=qt(vocab, dim))
    hs = dim // nh
    # (out_features, in_features) per layer, matching init_quantized_tensors order
    for name, rows, cols in (("wq", nh * hs, dim), ("wk", nkv * hs, dim), ("wv", nkv * hs, dim),
                             ("wo", dim, nh * hs), ("w1", hidden, dim), ("w2", dim, hidden),
                             ("w3", hidden, dim)):
        getattr(w, name).extend(qt(rows, cols) for _ in range(nl))
    w.wcls = w.tokens if shared else qt(vocab, dim)
    if off != raw.size:
        raise ValueError(f"{path}: parsed {off} bytes but file has {raw.size}")
    return cfg, w


def quantize_q80(x: np.ndarray, gs: int) -> QTensor:
    """Bit-exact port of run_int8.c quantize(): symmetric, per-group, round-half-away."""
    x = x.astype(np.float32).reshape(-1, gs)
    wmax = np.max(np.abs(x), axis=1)
    scale = (wmax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (x / scale[:, None]).astype(np.float64)  # C: round((double)(x/scale))
    q = np.where(np.isfinite(v), np.sign(v) * np.floor(np.abs(v) + 0.5), 0).astype(np.int8)
    return QTensor(q.reshape(1, -1), scale.reshape(1, -1))


def gemv_q80(w: QTensor, x: QTensor, gs: int) -> tuple[np.ndarray, np.ndarray]:
    """W(d,n) @ x(n) -> (int32 partial sums (d, n//gs), float32 output (d,))."""
    d, n = w.q.shape
    wq = w.q.reshape(d, n // gs, gs).astype(np.int32)
    xq = x.q.reshape(1, n // gs, gs).astype(np.int32)
    psum = np.sum(wq * xq, axis=2, dtype=np.int32)
    out = np.sum(psum.astype(np.float32) * w.s * x.s, axis=1, dtype=np.float32)
    return psum, out


def dequant(t: QTensor, gs: int) -> np.ndarray:
    r, c = t.q.shape
    return (t.q.reshape(r, c // gs, gs).astype(np.float32) * t.s[:, :, None]).reshape(r, c)


def rmsnorm(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    ss = np.float32(1.0) / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(1e-5))
    return (w * (ss * x)).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return (e / np.sum(e)).astype(np.float32)


CaptureFn = Callable[[str, int, Dict[str, np.ndarray]], None]


class Q80Model:
    def __init__(self, path: str):
        self.cfg, self.w = load_q80(path)
        c = self.cfg
        self.emb = dequant(self.w.tokens, c.group_size)
        self.kcache = np.zeros((c.n_layers, c.seq_len, c.kv_dim), np.float32)
        self.vcache = np.zeros_like(self.kcache)

    def _mm(self, w: QTensor, x: np.ndarray, tag: str, layer: int, cap: Optional[CaptureFn]):
        xq = quantize_q80(x, self.cfg.group_size)
        psum, out = gemv_q80(w, xq, self.cfg.group_size)
        if cap:
            cap(tag, layer, {"w": w.q, "ws": w.s, "xq": xq.q, "xs": xq.s, "psum": psum, "out": out})
        return out

    def forward(self, token: int, pos: int, capture: Optional[CaptureFn] = None) -> np.ndarray:
        c, w = self.cfg, self.w
        hs, kv_mul = c.head_size, c.n_heads // c.n_kv_heads
        x = self.emb[token].copy()
        hd = (np.arange(0, c.dim, 2) % hs).astype(np.float32)
        ang = np.float32(pos) * (np.float32(1.0) / np.power(np.float32(10000.0), hd / np.float32(hs)))
        fcr, fci = np.cos(ang), np.sin(ang)
        for l in range(c.n_layers):
            xb = rmsnorm(x, w.rms_att[l])
            q = self._mm(w.wq[l], xb, "wq", l, capture)
            k = self._mm(w.wk[l], xb, "wk", l, capture)
            v = self._mm(w.wv[l], xb, "wv", l, capture)
            for vec, n in ((q, c.dim), (k, c.kv_dim)):
                v0, v1 = vec[0:n:2].copy(), vec[1:n:2].copy()
                vec[0:n:2] = v0 * fcr[: n // 2] - v1 * fci[: n // 2]
                vec[1:n:2] = v0 * fci[: n // 2] + v1 * fcr[: n // 2]
            self.kcache[l, pos], self.vcache[l, pos] = k, v
            att_out = np.zeros(c.dim, np.float32)
            for h in range(c.n_heads):
                kvh = h // kv_mul
                qh = q[h * hs:(h + 1) * hs]
                K = self.kcache[l, : pos + 1, kvh * hs:(kvh + 1) * hs]
                V = self.vcache[l, : pos + 1, kvh * hs:(kvh + 1) * hs]
                a = softmax((K @ qh) / np.sqrt(np.float32(hs)))
                att_out[h * hs:(h + 1) * hs] = a @ V
            x = x + self._mm(w.wo[l], att_out, "wo", l, capture)
            xb = rmsnorm(x, w.rms_ffn[l])
            h1 = self._mm(w.w1[l], xb, "w1", l, capture)
            h3 = self._mm(w.w3[l], xb, "w3", l, capture)
            hb = (h1 * (np.float32(1.0) / (np.float32(1.0) + np.exp(-h1)))) * h3
            x = x + self._mm(w.w2[l], hb.astype(np.float32), "w2", l, capture)
        x = rmsnorm(x, w.rms_final)
        return self._mm(w.wcls, x, "wcls", -1, capture)
