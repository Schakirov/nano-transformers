#!/usr/bin/env bash
# ===========================================================================
# dump.sh  --  expose the GPU assembly hiding under the C++ kernels.
# ===========================================================================
# There are TWO levels below CUDA C++:
#
#   PTX  -- a virtual, portable assembly (NVIDIA's "IR").  nvcc emits it; the
#           driver JIT-compiles it for your exact GPU.  Human-ish, typed,
#           shows .shared memory, fma.rn.f32, bar.sync, shfl.sync.down.b32.
#
#   SASS -- the REAL machine code that the SMs execute.  Produced by ptxas and
#           extracted with cuobjdump.  This is the floor: FFMA, LDS/STS,
#           BAR.SYNC, SHFL.DOWN, scheduling, register allocation.
#
# We write both to web/asm/ so the website can show them with annotations.
# ===========================================================================
set -e
cd "$(dirname "$0")"
ARCH="${ARCH:-sm_89}"
OUT="../web/asm"
mkdir -p "$OUT"

# cuobjdump (turns a cubin into SASS) ships with the full CUDA toolkit; on this
# box it lives in a Triton wheel.  Find whatever is available.
CUOBJDUMP="$(command -v cuobjdump || true)"
if [ -z "$CUOBJDUMP" ]; then
    CUOBJDUMP="$(find / -name cuobjdump -type f 2>/dev/null | head -1)"
fi

for f in matmul softmax; do
    echo "== $f: PTX (virtual ISA) -> $OUT/$f.ptx =="
    nvcc -arch=$ARCH -ptx $f.cu -o "$OUT/$f.ptx"

    echo "== $f: SASS (real machine code) -> $OUT/$f.sass =="
    nvcc -arch=$ARCH -cubin $f.cu -o "/tmp/$f.cubin"
    if [ -n "$CUOBJDUMP" ]; then
        "$CUOBJDUMP" -sass "/tmp/$f.cubin" > "$OUT/$f.sass"
    else
        echo "(cuobjdump not found -- skipping SASS for $f)" > "$OUT/$f.sass"
    fi
done

echo
echo "PTX line counts:";  wc -l "$OUT"/*.ptx
echo "SASS line counts:"; wc -l "$OUT"/*.sass
echo
echo "Highlights you can grep for:"
echo "  tiled-matmul shared mem :  grep -i 'shared'  $OUT/matmul.ptx"
echo "  the fused multiply-add  :  grep -i 'fma'     $OUT/matmul.ptx"
echo "  the warp shuffle        :  grep -i 'shfl'    $OUT/softmax.ptx"
echo "  real-iron FMA (SASS)    :  grep -i 'FFMA'    $OUT/matmul.sass"
