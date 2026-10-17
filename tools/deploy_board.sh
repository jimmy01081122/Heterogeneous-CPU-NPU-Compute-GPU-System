#!/usr/bin/env bash
# Deploy ARM runtimes + models to the PYNQ-Z2 and run the CPU baseline
# (experiments A: CPU FP32, B: CPU INT8). Run inside the dev container.
#
#   tools/deploy_board.sh push        # rsync binaries & models
#   tools/deploy_board.sh baseline    # run A/B on board, pull CSVs to results/raw/<date>/
#
# Env: PYNQ_HOST (default 192.168.2.99), PYNQ_USER (default xilinx)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${PYNQ_USER:-xilinx}@${PYNQ_HOST:-192.168.2.99}"
RDIR="${PYNQ_DIR:-/home/xilinx/hetero}"
STEPS="${STEPS:-256}"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/known_hosts_pynq)

push() {
    "${SSH[@]}" "${HOST}" "mkdir -p ${RDIR}/bin ${RDIR}/models ${RDIR}/out"
    rsync -az -e "${SSH[*]}" "${ROOT}/sw/runtime/build/arm/" "${HOST}:${RDIR}/bin/"
    rsync -az -e "${SSH[*]}" --include='stories*.bin' --include='tok*.bin' --exclude='*' \
        "${ROOT}/models/" "${HOST}:${RDIR}/models/"
}

baseline() {
    local out="${ROOT}/results/raw/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${out}"
    "${SSH[@]}" "${HOST}" bash -s <<EOF | tee "${out}/console.txt"
set -e
cd ${RDIR}
uname -a; cat /proc/cpuinfo | grep -m1 -i 'model name\|Processor'; free -m | head -2
run() { # tag bin ckpt tok
  echo "=== \$1"
  HETERO_PROF_CSV=out/\$1.csv ./bin/\$2 models/\$3 -z models/\$4 -t 0 -n ${STEPS} -i "Once upon a time" > out/\$1.txt
}
run A_260K_fp32 run_fp32_prof stories260K.bin     tok512.bin
run B_260K_int8 run_int8_prof stories260K_q80.bin tok512.bin
run A_15M_fp32  run_fp32_prof stories15M.bin      tokenizer.bin
run B_15M_int8  run_int8_prof stories15M_q80.bin  tokenizer.bin
EOF
    rsync -az -e "${SSH[*]}" "${HOST}:${RDIR}/out/" "${out}/"
    echo "results -> ${out}"
}

case "${1:-}" in
    push) push ;;
    baseline) push; baseline ;;
    *) echo "usage: $0 push|baseline"; exit 1 ;;
esac
