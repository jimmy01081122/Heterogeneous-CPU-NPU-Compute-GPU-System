#!/usr/bin/env python3
"""
Export a llama2.c PyTorch checkpoint (.pt) to the Q8_0 (version 2) format used
by sw/runtime/src/run_int8.c.

Difference vs upstream `export.py --version 2`: the group size is chosen so it
divides *both* dim and hidden_dim. Upstream only checks dim, which silently
drops the tail of every W2 row for stories260K (hidden_dim=172), because the
runtime's quantize()/matmul() iterate over n/GS whole groups only.

Usage:
    python tools/export_q8.py models/stories15M.pt  models/stories15M_q80.bin
    python tools/export_q8.py models/stories260K.pt models/stories260K_q80.bin
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sw", "runtime", "upstream"))

from export import load_checkpoint, version2_export  # noqa: E402


def pick_group_size(dim: int, hidden_dim: int, max_gs: int) -> int:
    gs = max_gs
    while gs > 1 and (dim % gs or hidden_dim % gs):
        gs //= 2
    return gs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="input .pt checkpoint")
    ap.add_argument("output", help="output .bin (Q8_0, version 2)")
    ap.add_argument("--max-group-size", type=int, default=64)
    args = ap.parse_args()

    model = load_checkpoint(args.checkpoint)
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    gs = pick_group_size(p.dim, hidden_dim, args.max_group_size)
    print(f"dim={p.dim} hidden_dim={hidden_dim} -> group_size={gs}")
    version2_export(model, args.output, group_size=gs)


if __name__ == "__main__":
    main()
