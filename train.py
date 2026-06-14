"""
train.py  --  train the tiny Transformer and stream everything to the website.
==============================================================================

Run one task:                      python3 train.py sort
Run with more steps:               python3 train.py add --steps 1500
Run all three back to back:        python3 train.py all

Every few steps this writes web/runs/<task>.json, containing the loss/accuracy
curves, live sample predictions, weight heatmaps + histograms, and one
attention map.  Open http://localhost:8081 and watch the weights learn.

The optimiser is Adam, hand-written in ~12 lines so nothing is hidden.
"""

import sys, os, json, time, argparse
import numpy as np
from model import Transformer
from tasks import TASKS

RUNDIR = os.path.join(os.path.dirname(__file__), "web", "runs")
os.makedirs(RUNDIR, exist_ok=True)


class Adam:
    """The standard Adam optimiser.  m,v are per-parameter running moments."""
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.95, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m[:] = self.b1 * m + (1 - self.b1) * g          # 1st moment (mean)
            v[:] = self.b2 * v + (1 - self.b2) * g * g       # 2nd moment (variance)
            mhat = m / (1 - self.b1 ** self.t)               # bias correction
            vhat = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad[:] = 0.0


# --- helpers that summarise a weight matrix for the website ----------------
def heatmap(w, k=40):
    a = w.data
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a = a[:k, :k]                                            # top-left corner slice
    return a.tolist(), float(a.min()), float(a.max())


def histogram(w, bins=30):
    counts, edges = np.histogram(w.data.reshape(-1), bins=bins)
    return {"counts": counts.tolist(), "edges": [float(e) for e in edges]}


def snapshot_weights(model):
    named = [("wte (token embed)", model.wte),
             ("blk0 QKV proj", model.blocks[0]["attn_w"]),
             ("blk0 MLP fc", model.blocks[0]["fc_w"]),
             ("head (unembed)", model.head_w)]
    out = []
    for name, w in named:
        heat, vmin, vmax = heatmap(w)
        out.append({"name": name, "shape": list(w.shape),
                    "heat": heat, "vmin": vmin, "vmax": vmax,
                    "hist": histogram(w), "norm": float(np.sqrt((w.data ** 2).mean()))})
    return out


def evaluate(model, task, bs=128):
    """Teacher-forced accuracy + a few decoded sample predictions."""
    x, y, mask = task.batch(bs)
    logits = model.forward(x).data
    pred = logits.argmax(-1)                                 # (B,T) predicted next tokens
    mask_b = mask.astype(bool)
    # a row is "correct" if every scored position is predicted right
    row_ok = np.array([(pred[i][mask_b[i]] == y[i][mask_b[i]]).all() for i in range(bs)])
    acc = float(row_ok.mean())

    samples = []
    for i in range(min(6, bs)):
        shown = x[i].copy()
        # splice the model's predicted token into each scored answer slot, so the
        # decoded sample shows what the model *would* have written, not the truth
        for p in np.where(mask_b[i])[0]:
            if p + 1 < len(shown):
                shown[p + 1] = pred[i][p]
        samples.append({"text": task.decode(shown), "correct": bool(row_ok[i])})
    extra = {}
    if task.name == "shapes":
        extra["image"] = task.image_of(x[0])
        extra["image_label"] = task.decode(x[0])
    return acc, samples, model.last_attn, extra


def write_run(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)                                    # atomic for the poller


def train(task_name, steps, batch=32, log_every=10):
    task = TASKS[task_name]
    model = Transformer(task.vocab, task.block_size, n_layer=3, n_head=4, n_embd=64)
    opt = Adam(model.parameters())
    nparam = sum(p.data.size for p in model.parameters())
    print(f"[{task_name}] vocab={task.vocab} block={task.block_size} params={nparam}")

    path = os.path.join(RUNDIR, f"{task_name}.json")
    loss_hist, acc_hist = [], []
    t0 = time.time()

    for step in range(steps + 1):
        x, y, mask = task.batch(batch)
        loss = model.loss(x, y, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % log_every == 0:
            acc, samples, attn, extra = evaluate(model, task)
            loss_hist.append([step, float(loss.data)])
            acc_hist.append([step, acc])
            # one attention map (first example, head 0) for the website
            amap = attn[0, 0].tolist() if attn is not None else []
            payload = {
                "task": task_name, "step": step, "max_steps": steps,
                "loss": float(loss.data), "acc": acc, "elapsed": time.time() - t0,
                "config": {"n_layer": 3, "n_head": 4, "n_embd": 64,
                           "vocab": task.vocab, "block_size": task.block_size,
                           "params": int(nparam)},
                "loss_history": loss_hist, "acc_history": acc_hist,
                "samples": samples, "attn": amap, "weights": snapshot_weights(model),
                "done": step == steps, **extra,
            }
            write_run(path, payload)
            print(f"  step {step:4d}/{steps}  loss {loss.data:.4f}  acc {acc:5.1%}"
                  f"  ({time.time()-t0:4.1f}s)")
    print(f"[{task_name}] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASKS) + ["all"])
    ap.add_argument("--steps", type=int, default=None,
                    help="override; otherwise a sensible per-task default is used")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    DEFAULT_STEPS = {"sort": 600, "add": 1500, "shapes": 400}
    names = list(TASKS) if a.task == "all" else [a.task]
    for nm in names:
        np.random.seed(1234)
        train(nm, a.steps or DEFAULT_STEPS[nm], a.batch)
