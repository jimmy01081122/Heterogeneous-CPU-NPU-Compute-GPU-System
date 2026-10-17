#!/usr/bin/env python3
"""
Generate NPU GEMV test vectors from *real* model weights and activations.

Runs the NumPy golden model (tools/q80.py) for a short prompt and stores, for
selected matmuls, the int8 weights/activations, their group scales, the exact
INT32 per-group partial sums and the float output.

    python tools/gen_npu_vectors.py models/stories260K_q80.bin -o tests/vectors/260K
    python tools/gen_npu_vectors.py models/stories15M_q80.bin  -o tests/vectors/15M \
           --tokens 1 403 2501 --ops wq w1 w2 wcls

Output: one .npz per (op, layer, pos) + index.json summarising shapes.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q80 import Q80Model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 403, 2501],
                    help="token ids fed at pos 0..N-1 (default: BOS 'Once' 'upon')")
    ap.add_argument("--ops", nargs="+", default=["wq", "wk", "wv", "wo", "w1", "w2", "w3", "wcls"])
    ap.add_argument("--layers", type=int, nargs="+", default=[0], help="-1 = classifier")
    args = ap.parse_args()

    m = Q80Model(args.checkpoint)
    vocab = m.cfg.vocab_size
    os.makedirs(args.out, exist_ok=True)
    index = {"checkpoint": os.path.basename(args.checkpoint), "group_size": m.cfg.group_size, "cases": []}
    layers = set(args.layers) | {-1}
    pos = 0

    def cap(tag, layer, d):
        if tag not in args.ops or layer not in layers:
            return
        name = f"{tag}_L{layer}_p{pos}" if layer >= 0 else f"{tag}_p{pos}"
        np.savez_compressed(os.path.join(args.out, name + ".npz"), **d)
        index["cases"].append({"name": name, "op": tag, "layer": layer, "pos": pos,
                               "rows": int(d["w"].shape[0]), "cols": int(d["w"].shape[1]),
                               "psum_absmax": int(np.abs(d["psum"]).max())})

    for pos, tok in enumerate(args.tokens):
        logits = m.forward(tok % vocab, pos, capture=cap)
    index["next_token_argmax"] = int(np.argmax(logits))
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"{len(index['cases'])} cases -> {args.out}  "
          f"(max |psum| = {max(c['psum_absmax'] for c in index['cases'])}, "
          f"argmax next token = {index['next_token_argmax']})")


if __name__ == "__main__":
    main()
