#!/usr/bin/env bash
# Download pretrained TinyStories checkpoints (karpathy/tinyllamas) into models/.
# Usage: tools/download_models.sh [260K|15M|all]   (default: all)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${ROOT}/models"
HF="https://huggingface.co/karpathy/tinyllamas/resolve/main"
mkdir -p "${DST}"

fetch() {  # fetch <url> <dst>
    if [[ -s "$2" ]]; then echo "[skip] $(basename "$2")"; return; fi
    echo "[get ] $(basename "$2")"
    curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

want="${1:-all}"
if [[ "${want}" == "260K" || "${want}" == "all" ]]; then
    fetch "${HF}/stories260K/stories260K.bin" "${DST}/stories260K.bin"
    fetch "${HF}/stories260K/stories260K.pt"  "${DST}/stories260K.pt"
    fetch "${HF}/stories260K/tok512.bin"      "${DST}/tok512.bin"
    fetch "${HF}/stories260K/tok512.model"    "${DST}/tok512.model"
fi
if [[ "${want}" == "15M" || "${want}" == "all" ]]; then
    fetch "${HF}/stories15M.bin" "${DST}/stories15M.bin"
    fetch "${HF}/stories15M.pt"  "${DST}/stories15M.pt"
fi
# default Llama-2 32000-vocab tokenizer used by stories15M
cp -n "${ROOT}/sw/runtime/upstream/tokenizer.bin" "${DST}/tokenizer.bin" 2>/dev/null || true

( cd "${DST}" && sha256sum ./*.bin ./*.pt ./*.model 2>/dev/null > SHA256SUMS )
echo "done -> ${DST}"
