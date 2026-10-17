#!/usr/bin/env python3
"""
Static workload model for llama2.c checkpoints: MACs and weight bytes per token,
broken down by operator, plus roofline-style bounds for PYNQ-Z2.

    python tools/workload.py models/stories15M_q80.bin
    python tools/workload.py models/stories15M_q80.bin --prof prof.csv   # compare to measurement

The numbers drive the NPU sizing argument (docs/feasibility.md §2): decode-phase
GEMV is weight-streaming bound, so the useful MAC count is set by DDR bandwidth.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q80 import load_q80  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Q8_0 v2 checkpoint")
    ap.add_argument("--ddr-gbps", type=float, default=1.5, help="usable DDR bandwidth GB/s (PYNQ-Z2 ~1.2-1.6)")
    ap.add_argument("--pl-mhz", type=float, default=150.0)
    ap.add_argument("--pos", type=int, default=128, help="sequence position for attention cost")
    ap.add_argument("--prof", help="CSV from HETERO_PROF_CSV to compare against")
    a = ap.parse_args()

    c, _ = load_q80(a.checkpoint)
    gs, L, d, h, kv, V = c.group_size, c.n_layers, c.dim, c.hidden_dim, c.kv_dim, c.vocab_size
    wbytes = lambda n: n + 4 * n // gs  # int8 + fp32 scale per group  # noqa: E731
    ops = {
        "mm_qkv": L * d * (d + 2 * kv),
        "mm_o": L * d * d,
        "mm_ffn13": L * 2 * d * h,
        "mm_ffn2": L * h * d,
        "mm_cls": V * d,
    }
    attn_macs = L * c.n_heads * (a.pos + 1) * c.head_size * 2
    total_macs = sum(ops.values())
    total_bytes = sum(wbytes(n) for n in ops.values())

    print(f"config: dim={d} hidden={h} layers={L} heads={c.n_heads}/{c.n_kv_heads} vocab={V} GS={gs}")
    print(f"\n{'op':10s} {'MACs':>12s} {'weight B':>12s} {'share':>7s}")
    for k, n in ops.items():
        print(f"{k:10s} {n:12,d} {wbytes(n):12,d} {100 * n / total_macs:6.1f}%")
    print(f"{'attention':10s} {attn_macs:12,d} {'(KV cache)':>12s}   @pos={a.pos}")
    print(f"{'TOTAL':10s} {total_macs:12,d} {total_bytes:12,d}")

    t_bw = total_bytes / (a.ddr_gbps * 1e9)
    bpc = a.ddr_gbps * 1e9 / (a.pl_mhz * 1e6)
    print(f"\nDDR-bound decode ceiling : {1 / t_bw:7.1f} tok/s  ({t_bw * 1e3:.2f} ms/token @ {a.ddr_gbps} GB/s)")
    print(f"bytes/cycle @ {a.pl_mhz:.0f} MHz   : {bpc:7.1f}  -> ~{bpc * gs / (gs + 4):.0f} INT8 MAC/cycle saturate DDR")
    on_chip = total_bytes <= 140 * 36 * 1024 // 8
    print(f"fits in BRAM (630 KB)    : {'YES - compute-bound regime' if on_chip else 'no - weight-streaming bound'}")

    if a.prof:
        import pandas as pd
        df = pd.read_csv(a.prof)
        df = df[df.calls > 0]
        tok = int(df.tokens.iloc[0])
        df["us_per_tok"] = df.total_ns / 1e3 / tok
        df["share"] = 100 * df.total_ns / df.total_ns.sum()
        print(f"\nmeasured ({tok} tokens): {1e6 / df.us_per_tok.sum():.1f} tok/s")
        for _, r in df.sort_values("total_ns", ascending=False).iterrows():
            extra = ""
            if r.op in ops:
                gmacs = ops[r.op] / (r.us_per_tok * 1e3)
                extra = f"  {gmacs:6.2f} GMAC/s  {wbytes(ops[r.op]) / (r.us_per_tok * 1e3):6.2f} GB/s"
            print(f"  {r.op:10s} {r.us_per_tok:10.1f} us/tok {r.share:6.1f}%{extra}")


if __name__ == "__main__":
    main()
