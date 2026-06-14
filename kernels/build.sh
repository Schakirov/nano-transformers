#!/usr/bin/env bash
# Compile and run the two CUDA kernels.  sm_89 = NVIDIA L4 (Ada Lovelace).
set -e
cd "$(dirname "$0")"
ARCH="${ARCH:-sm_89}"
echo "== building (arch=$ARCH) =="
nvcc -O3 -arch=$ARCH matmul.cu  -o matmul
nvcc -O3 -arch=$ARCH softmax.cu -o softmax
mkdir -p ../web/asm
# also tee the benchmark output to web/asm/bench.txt so the website can show it
{ echo "== matmul =="; ./matmul; echo; echo "== softmax =="; ./softmax; } \
    | tee ../web/asm/bench.txt
