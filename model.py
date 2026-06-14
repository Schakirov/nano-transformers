"""
model.py  --  a tiny decoder-only Transformer, written on top of nanograd.
==========================================================================

This is GPT in miniature.  It is *decoder-only* and *autoregressive*: it reads
a sequence of integer tokens and predicts the next token at every position.
That single capability is enough to express every task in this repo --
sorting symbols, adding numbers, and classifying images -- because each task
is encoded as "given this prefix of tokens, what token comes next?".

Architecture (all the standard pieces, none of the production clutter):

    tokens ──► token embedding  +  positional embedding
           ──► [ Block ] x n_layer
           ──► final LayerNorm
           ──► unembed (logits over the vocabulary)

    Block = x + Attention(LayerNorm(x))      # pre-norm residual
            x + MLP(LayerNorm(x))            # pre-norm residual

Read it next to nanograd.py: every `@`, `+`, `.softmax()` here is a node in
the graph that file knows how to differentiate.
"""

import numpy as np
from nanograd import Tensor, layernorm, cross_entropy


def param(*shape, scale=0.02):
    """A trainable tensor, initialised small and random (GPT-2 style)."""
    return Tensor(np.random.randn(*shape).astype(np.float32) * scale)


class Transformer:
    def __init__(self, vocab, block_size, n_layer=3, n_head=4, n_embd=64):
        self.vocab, self.block_size = vocab, block_size
        self.n_layer, self.n_head, self.n_embd = n_layer, n_head, n_embd
        self.head_dim = n_embd // n_head

        # --- embeddings -----------------------------------------------------
        self.wte = param(vocab, n_embd)             # token  -> vector
        self.wpe = param(block_size, n_embd)        # position -> vector

        # --- per-layer weights (kept in plain lists; no nn.Module needed) ---
        self.blocks = []
        for _ in range(n_layer):
            self.blocks.append(dict(
                ln1_g=Tensor(np.ones(n_embd, np.float32)),  ln1_b=Tensor(np.zeros(n_embd, np.float32)),
                # one fused matrix produces Query, Key and Value at once (3*n_embd wide)
                attn_w=param(n_embd, 3 * n_embd),  attn_b=Tensor(np.zeros(3 * n_embd, np.float32)),
                proj_w=param(n_embd, n_embd),      proj_b=Tensor(np.zeros(n_embd, np.float32)),
                ln2_g=Tensor(np.ones(n_embd, np.float32)),  ln2_b=Tensor(np.zeros(n_embd, np.float32)),
                fc_w=param(n_embd, 4 * n_embd),    fc_b=Tensor(np.zeros(4 * n_embd, np.float32)),
                fcp_w=param(4 * n_embd, n_embd),   fcp_b=Tensor(np.zeros(n_embd, np.float32)),
            ))

        # --- output head ----------------------------------------------------
        self.lnf_g = Tensor(np.ones(n_embd, np.float32))
        self.lnf_b = Tensor(np.zeros(n_embd, np.float32))
        self.head_w = param(n_embd, vocab)
        # causal mask: position t may only attend to positions <= t (lower triangular)
        self.mask = np.tril(np.ones((block_size, block_size), np.float32))
        self.last_attn = None                        # stashed for the website to draw

    def parameters(self):
        ps = [self.wte, self.wpe, self.lnf_g, self.lnf_b, self.head_w]
        for b in self.blocks:
            ps += list(b.values())
        return ps

    # -----------------------------------------------------------------------
    def attention(self, x, blk):
        """Causal multi-head self-attention for one block.

        x is (B, T, C).  We project to Q,K,V, split into heads, compute scaled
        dot-product attention per head, re-assemble, and project back.
        """
        B, T, C = x.shape
        H, hd = self.n_head, self.head_dim

        qkv = x @ blk["attn_w"] + blk["attn_b"]             # (B,T,3C)
        # split the fused projection back into Query / Key / Value (graph-aware)
        q = _slice(qkv, 0, C);  k = _slice(qkv, C, 2 * C);  v = _slice(qkv, 2 * C, 3 * C)

        # (B,T,C) -> (B,H,T,hd): isolate each head onto its own batch axis
        def heads(t):
            return t.reshape(B, T, H, hd).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)              # (B,H,T,hd)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(hd))   # (B,H,T,T) scores
        # causal mask: set future positions to -inf before softmax
        att = att + Tensor((self.mask[:T, :T] - 1.0) * 1e9)
        att = att.softmax(axis=-1)
        self.last_attn = att.data                            # (B,H,T,T) for viz
        y = att @ v                                          # (B,H,T,hd) weighted values
        y = y.transpose(1, 2).reshape(B, T, C)               # re-merge heads
        return y @ blk["proj_w"] + blk["proj_b"]

    def block(self, x, blk):
        x = x + self.attention(layernorm(x, blk["ln1_g"], blk["ln1_b"]), blk)
        h = layernorm(x, blk["ln2_g"], blk["ln2_b"])
        h = (h @ blk["fc_w"] + blk["fc_b"]).gelu()           # expand 4x, nonlinearity
        h = h @ blk["fcp_w"] + blk["fcp_b"]                  # project back
        return x + h

    def forward(self, idx):
        """idx: int array (B,T) of token ids -> logits Tensor (B,T,vocab)."""
        B, T = idx.shape
        tok = self.wte.embed(idx)                            # (B,T,C)
        pos = self.wpe.embed(np.arange(T))                   # (T,C), broadcast over B
        x = tok + pos
        for blk in self.blocks:
            x = self.block(x, blk)
        x = layernorm(x, self.lnf_g, self.lnf_b)
        return x @ self.head_w                               # (B,T,vocab)

    def loss(self, idx, targets, loss_mask=None):
        """Cross-entropy of next-token prediction.

        targets: (B,T) ints, with -1 marking positions we don't score.
        loss_mask lets a task supervise only the "answer" tokens (e.g. only the
        sorted output, or only the image's class label).
        """
        logits = self.forward(idx)
        B, T, V = logits.shape
        flat = logits.reshape(B * T, V)
        tgt = targets.reshape(B * T)
        keep = tgt >= 0
        if loss_mask is not None:
            keep &= loss_mask.reshape(B * T).astype(bool)
        flat = _index_rows(flat, np.where(keep)[0])
        loss, _probs = cross_entropy(flat, tgt[keep])
        return loss


# --- two small graph-aware helpers used above ------------------------------
def _slice(t, a, b):
    """Differentiable slice of the last axis: t[..., a:b]."""
    out = Tensor(t.data[..., a:b], (t,), "slice")

    def _backward():
        g = np.zeros_like(t.data)
        g[..., a:b] = out.grad
        t.grad += g
    out._backward = _backward
    return out


def _index_rows(t, rows):
    """Differentiable gather of rows `rows` from a 2-D tensor."""
    out = Tensor(t.data[rows], (t,), "rows")

    def _backward():
        g = np.zeros_like(t.data)
        np.add.at(g, rows, out.grad)
        t.grad += g
    out._backward = _backward
    return out
