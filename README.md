# nano-transformers

**A tiny transformer you can read in an hour — then fall through every layer
of abstraction beneath it, from the Python you write down to the actual SASS
machine code your GPU executes.**

Nothing here is imported from a giant framework. The autograd engine, the
transformer, the optimizer, and the GPU kernels are all written from scratch
and heavily commented. There is an interactive website that explains the whole
thing and lets you watch the weights learn in real time.

```
model.py        the transformer            ── Tensor math you recognise
   │
nanograd.py     ~200-line autograd         ── this IS backpropagation
   │
numpy / BLAS    dense linear algebra
   ╎
kernels/*.cu    CUDA: tiled matmul,        ── the same math, on the GPU
                warp-shuffle softmax
   ╎
*.ptx           NVIDIA's virtual ISA       ── fma.rn.f32, .shared, shfl.sync
   ╎
*.sass          real machine code          ── FFMA, LDS, BAR.SYNC, SHFL.DOWN
```

## Quick start

```bash
# 1. the website (training dashboard + GPU/PTX/SASS explorer) on :8081
python3 serve.py
#    open http://localhost:8081 , pick a task, press ▶ Train

# 2. or train from the command line and watch the console
python3 train.py sort         # symbolic : sort 6 digits         (~10s to 90%+)
python3 train.py add          # arithmetic: two-digit addition   (~30s to 90%+)
python3 train.py shapes       # vision    : classify 8x8 images  (~30s to 100%)
python3 train.py all          # all three, back to back

# 3. the GPU descent (needs an NVIDIA GPU + CUDA toolkit)
cd kernels && ./build.sh      # compile + verify + benchmark the kernels
              ./dump.sh        # emit PTX and SASS into web/asm/
```

Only dependency: **NumPy**. The kernels additionally need **nvcc** (and, for
SASS, `cuobjdump`). Everything was developed and verified on an NVIDIA L4.

## The files (read them in this order)

| file | what it is | lines |
|------|------------|-------|
| `nanograd.py` | reverse-mode autograd over NumPy — `Tensor`, `.backward()` | ~210 |
| `model.py`    | the decoder-only transformer, built from `Tensor`s | ~190 |
| `tasks.py`    | three modalities as next-token prediction | ~170 |
| `train.py`    | Adam + the training loop + live JSON logging | ~150 |
| `serve.py`    | the tiny web server behind :8081 | ~90 |
| `kernels/matmul.cu`  | naive vs **tiled** SGEMM, verified + benchmarked | ~160 |
| `kernels/softmax.cu` | row softmax via **warp shuffles** (`__shfl_down_sync`) | ~110 |
| `web/`        | the interactive site (vanilla JS, no CDNs) | — |

## The one idea that unifies everything

Every task is encoded as *"given this prefix of tokens, predict the next one"*,
and we only score the answer region:

```
sort    4 5 7 2 2 4  >  2 2 4 4 5 7
add     8 8 + 6 8 =  6 5 1            (answer written least-significant first)
shapes  [64 quantised pixel tokens] =  box
```

The **identical** 152k-parameter model (3 layers, 4 heads, dim 64) learns all
three. It never knows which modality it is looking at — it only ever predicts
the next integer token.

## What the website shows

- **Descend** — the layered map above, every layer a real file.
- **Architecture** — the forward pass and the one gradient rule that matters.
- **Train & watch** — launch training from the browser; live loss/accuracy
  curves, sample predictions, **weight heatmaps and histograms updating as the
  model learns**, and a causal attention map.
- **Into the GPU** — each CUDA kernel side-by-side with its **PTX and SASS**,
  annotated: shared-memory tiling, `__syncthreads` barriers, fused multiply-add,
  and the warp shuffle that reduces 32 values with no shared memory at all.
- **Read the code** — the entire repository, served to you, syntax-lit.

## Why it's built this way

It is deliberately *small and overlookable*: you can hold the whole thing in
your head. float32 everywhere so the numbers match what a GPU would compute.
The autograd, the model, and the kernels are three windows onto the same math
at three depths — the point is that the journey from `A @ B` in Python to a
warp moving floats through shared memory feels **continuous**, with no magic in
between.
