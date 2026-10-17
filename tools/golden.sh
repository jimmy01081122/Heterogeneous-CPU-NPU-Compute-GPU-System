#!/usr/bin/env bash
# Golden-output regression for the CPU runtimes.
#
#   tools/golden.sh record   # (re)generate tests/golden/*.txt from the host build
#   tools/golden.sh check    # host + arm(qemu) must reproduce the recorded outputs
#
# Greedy decoding (-t 0) makes the token stream deterministic, so any change in
# arithmetic (e.g. an accelerator path) that alters the argmax shows up as a diff.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
M="${ROOT}/models"
BIN="${ROOT}/sw/runtime/build"
GOLD="${ROOT}/tests/golden"
STEPS="${STEPS:-128}"
PROMPT="${PROMPT:-Once upon a time}"
mkdir -p "${GOLD}"

# name | binary | checkpoint | tokenizer
CASES=(
  "260K_fp32|run_fp32|stories260K.bin|tok512.bin"
  "260K_int8|run_int8|stories260K_q80.bin|tok512.bin"
  "15M_fp32|run_fp32|stories15M.bin|tokenizer.bin"
  "15M_int8|run_int8|stories15M_q80.bin|tokenizer.bin"
)

run_case() {  # run_case <target> <bin> <ckpt> <tok>
    local exe="${BIN}/$1/$2"
    local cmd=("${exe}" "${M}/$3" -z "${M}/$4" -t 0 -n "${STEPS}" -i "${PROMPT}")
    if [[ "$1" == "arm" ]]; then cmd=(qemu-arm "${cmd[@]}"); fi
    "${cmd[@]}" 2>/dev/null
}

mode="${1:-check}"
fail=0
for c in "${CASES[@]}"; do
    IFS='|' read -r name bin ckpt tok <<< "${c}"
    [[ -f "${M}/${ckpt}" ]] || { echo "[skip] ${name}: ${ckpt} missing"; continue; }
    if [[ "${mode}" == "record" ]]; then
        run_case host "${bin}" "${ckpt}" "${tok}" > "${GOLD}/${name}.txt"
        echo "[rec ] ${name}: $(head -c 70 "${GOLD}/${name}.txt" | tr '\n' ' ')..."
    else
        for tgt in host arm; do
            if diff -q <(run_case "${tgt}" "${bin}" "${ckpt}" "${tok}") "${GOLD}/${name}.txt" > /dev/null; then
                echo "[PASS] ${name} (${tgt})"
            else
                echo "[FAIL] ${name} (${tgt})"; fail=1
            fi
        done
    fi
done
exit "${fail}"
