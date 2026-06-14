"""
nanograd.py  --  a tiny reverse-mode autograd engine over NumPy arrays.
=======================================================================

This is the layer *directly below* the transformer.  The whole point of the
repo is that you can fall through every abstraction:

    model.py        the transformer, written in terms of Tensors
        |
    nanograd.py     <-- YOU ARE HERE: how gradients actually flow
        |
    numpy           dense linear algebra on the CPU
        :
    kernels/*.cu    the same math, but on the GPU, with PTX & SASS exposed

A `Tensor` wraps a NumPy array (`.data`) and its gradient (`.grad`).  Every
operation records a tiny closure (`_backward`) that knows how to push the
gradient from the output back to its inputs.  `loss.backward()` walks those
closures in reverse-topological order -- that *is* backpropagation.

Everything is float32 so the numbers feel like what a GPU would compute.
Read this file top to bottom; it is ~200 lines and there is no magic.
"""

import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`.

    NumPy silently *broadcasts* a (D,) bias across a (B,T,D) tensor during the
    forward pass.  In the backward pass we must do the opposite: sum the
    gradient over every axis that was stretched, so the bias gets one gradient
    per element of its own shape.  This one helper makes broadcasting safe.
    """
    while grad.ndim > len(shape):           # extra leading axes -> sum them away
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):           # axes that were size-1 -> sum, keep dim
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None       # closure filled in by each op
        self._prev = set(_children)         # the Tensors this one was built from
        self._op = _op                      # label, purely for debugging/printing

    # ----- helpers ---------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    # ----- elementwise ops -------------------------------------------------
    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        """Batched matrix multiply -- the operation the GPU kernels accelerate.

        C[...,m,n] = A[...,m,k] @ B[...,k,n].  The gradient of a matmul is two
        more matmuls:  dA = dC @ B^T  and  dB = A^T @ dC.  Memorise that; it is
        the single most important backward rule in all of deep learning.
        """
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            a, b, g = self.data, other.data, out.grad
            dA = g @ np.swapaxes(b, -1, -2)
            dB = np.swapaxes(a, -1, -2) @ g
            self.grad += _unbroadcast(dA, a.shape)   # _unbroadcast handles the
            other.grad += _unbroadcast(dB, b.shape)  # batched/broadcast case
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)          # restore the reduced axis
            self.grad += np.broadcast_to(g, self.data.shape)
        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")
        out._backward = lambda: setattr(self, "grad", self.grad + out.grad * out.data)
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")
        out._backward = lambda: setattr(self, "grad", self.grad + out.grad / self.data)
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), (self,), "sqrt")
        out._backward = lambda: setattr(self, "grad", self.grad + out.grad * 0.5 / out.data)
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        out._backward = lambda: setattr(self, "grad", self.grad + out.grad * (1 - t * t))
        return out

    def gelu(self):
        """Gaussian Error Linear Unit (tanh approximation) -- the MLP nonlinearity.

        gelu(x) = 0.5 x (1 + tanh[ sqrt(2/pi) (x + 0.044715 x^3) ]).
        We give it an explicit derivative instead of composing primitives, both
        for speed and so you can read the exact gradient in one place.
        """
        x = self.data
        c = np.float32(0.7978845608)        # sqrt(2/pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out = Tensor(0.5 * x * (1 + t), (self,), "gelu")

        def _backward():
            dinner = c * (1 + 3 * 0.044715 * x ** 2)
            dgelu = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dinner
            self.grad += out.grad * dgelu
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        """Numerically-stable softmax with its exact Jacobian-vector backward.

        Used inside attention.  The backward rule  dx = s * (g - sum(g*s))  is
        the compact form of multiplying by the softmax Jacobian s(diag - s^T).
        """
        shift = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shift)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, (self,), "softmax")

        def _backward():
            dot = (out.grad * s).sum(axis=axis, keepdims=True)
            self.grad += s * (out.grad - dot)
        out._backward = _backward
        return out

    # ----- shape ops -------------------------------------------------------
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), "reshape")
        out._backward = lambda: setattr(self, "grad", self.grad + out.grad.reshape(self.data.shape))
        return out

    def transpose(self, ax1, ax2):
        out = Tensor(np.swapaxes(self.data, ax1, ax2), (self,), "transpose")
        out._backward = lambda: setattr(self, "grad", self.grad + np.swapaxes(out.grad, ax1, ax2))
        return out

    def embed(self, idx):
        """Row-gather: pick rows `idx` out of an embedding table `self`.

        Forward is a lookup; backward *scatter-adds* each output gradient back
        onto the row it came from (np.add.at handles repeated indices).
        """
        out = Tensor(self.data[idx], (self,), "embed")

        def _backward():
            np.add.at(self.grad, idx, out.grad)
        out._backward = _backward
        return out

    # ----- sugar -----------------------------------------------------------
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self._wrap(other)
        return self * other.reciprocal()

    def reciprocal(self):
        out = Tensor(1.0 / self.data, (self,), "recip")
        out._backward = lambda: setattr(self, "grad", self.grad - out.grad * out.data * out.data)
        return out

    # ----- the engine ------------------------------------------------------
    def backward(self):
        """Topologically sort the graph, then run each node's _backward once."""
        topo, seen = [], set()

        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)     # d(loss)/d(loss) = 1
        for v in reversed(topo):
            v._backward()


# ---------------------------------------------------------------------------
# Two functions that are easier (and more numerically stable) to write by hand
# than to compose out of the primitives above.
# ---------------------------------------------------------------------------

def layernorm(x, gamma, beta, eps=1e-5):
    """Normalise the last axis to mean 0 / var 1, then scale & shift.

    Written purely with the primitives above, so its gradient is produced by
    the autograd engine -- a good check that the engine actually composes.
    """
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    xn = xc / (var + eps).sqrt()
    return xn * gamma + beta


def cross_entropy(logits, targets):
    """Mean softmax cross-entropy over a batch of token positions.

    `logits` is (N, V), `targets` is an int array (N,).  We compute the loss
    with a stable log-sum-exp, and hand-write the famous backward:

        d(loss)/d(logit) = (softmax(logit) - one_hot(target)) / N

    -- the cleanest gradient in the whole network.
    """
    z = logits.data
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)               # softmax probabilities
    n = targets.shape[0]
    loss_val = -np.log(p[np.arange(n), targets] + 1e-9).mean()
    out = Tensor(loss_val, (logits,), "cross_entropy")

    def _backward():
        dz = p.copy()
        dz[np.arange(n), targets] -= 1.0
        logits.grad += (dz / n) * out.grad
    out._backward = _backward
    return out, p
