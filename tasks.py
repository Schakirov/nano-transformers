"""
tasks.py  --  three toy tasks from three modalities, one tiny Transformer.
==========================================================================

The whole repo leans on a single trick: *everything is a sequence of tokens,
and the model just predicts the next one.*  That lets the identical
architecture (model.py) learn problems that look nothing alike:

  * sort   (symbolic)  : reorder a handful of digits into ascending order
  * add    (arithmetic): add two numbers, digit by digit
  * shapes (vision)    : look at an 8x8 image and name the shape

Each task is a `Task` object that knows three things:
  - vocab        : how many distinct token ids exist
  - block_size   : the fixed sequence length
  - batch(n)     : produce (x, y, mask) arrays for training
  - decode(row)  : turn a token row back into something a human can read

`x` is the input tokens, `y` is the next-token target (-1 = "don't score"),
and `mask` marks the answer region so we only train on what matters.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Task 1 -- SORT.  Symbolic sequence-to-sequence.
#   sequence:  d d d d  ">"  s s s s        (input digits, marker, sorted digits)
#   we score only the sorted half.
# ---------------------------------------------------------------------------
class SortTask:
    name = "sort"
    n = 6                       # how many digits to sort
    D = 10                      # digit alphabet 0..9

    def __init__(self):
        self.MARK = self.D                      # ">" separator token id
        self.vocab = self.D + 1
        self.block_size = 2 * self.n + 1        # digits + marker + sorted

    def batch(self, bs):
        x = np.zeros((bs, self.block_size), np.int64)
        y = np.full((bs, self.block_size), -1, np.int64)
        mask = np.zeros((bs, self.block_size), np.float32)
        for i in range(bs):
            d = np.random.randint(0, self.D, self.n)
            s = np.sort(d)
            seq = np.concatenate([d, [self.MARK], s])
            x[i] = seq
            # target at position t is the token at t+1 (standard next-token shift)
            y[i, :-1] = seq[1:]
            mask[i, self.n:-1] = 1.0             # score only the sorted output
        return x, y, mask

    def decode(self, row):
        m = list(row).index(self.MARK)
        inp = "".join(str(t) for t in row[:m])
        out = "".join(str(t) for t in row[m + 1:])
        return f"{inp} > {out}"


# ---------------------------------------------------------------------------
# Task 2 -- ADD.  Two-digit addition, answer written LEAST significant first
# so each output digit depends only on digits already produced (easy to learn).
#   sequence:  a a "+" b b "=" c c c
# ---------------------------------------------------------------------------
class AddTask:
    name = "add"
    ndig = 2

    def __init__(self):
        self.PLUS, self.EQ = 10, 11
        self.vocab = 12                          # digits 0..9, '+', '='
        self.alen = self.ndig + 1                # answer can be one digit longer
        self.block_size = self.ndig * 2 + 2 + self.alen

    def _digits(self, v, k):
        return [(v // (10 ** i)) % 10 for i in range(k)]     # least-significant first

    def batch(self, bs):
        x = np.zeros((bs, self.block_size), np.int64)
        y = np.full((bs, self.block_size), -1, np.int64)
        mask = np.zeros((bs, self.block_size), np.float32)
        for i in range(bs):
            a = np.random.randint(0, 10 ** self.ndig)
            b = np.random.randint(0, 10 ** self.ndig)
            c = a + b
            seq = (self._digits(a, self.ndig) + [self.PLUS] +
                   self._digits(b, self.ndig) + [self.EQ] +
                   self._digits(c, self.alen))
            seq = np.array(seq, np.int64)
            x[i] = seq
            y[i, :-1] = seq[1:]
            mask[i, -self.alen - 1:-1] = 1.0     # score only the answer digits
        return x, y, mask

    def decode(self, row):
        a = row[:self.ndig][::-1]; b = row[self.ndig + 1:2 * self.ndig + 1][::-1]
        c = row[2 * self.ndig + 2:][::-1]
        s = lambda d: "".join(str(t) for t in d)
        return f"{s(a)}+{s(b)}={s(c)}"


# ---------------------------------------------------------------------------
# Task 3 -- SHAPES.  Vision as token prediction.
#   An 8x8 image is quantised to 4 grey levels, flattened into 64 pixel tokens,
#   then a "=" marker, then the class token.  We score only the class token.
#   Five procedurally generated classes -> no dataset download required.
# ---------------------------------------------------------------------------
class ShapesTask:
    name = "shapes"
    S = 8                       # image side
    LEVELS = 4                  # pixel quantisation levels
    CLASSES = ["hline", "vline", "diag", "box", "cross"]

    def __init__(self):
        self.npix = self.S * self.S
        self.MARK = self.LEVELS                              # "=" marker
        self.CLS0 = self.LEVELS + 1                          # first class token id
        self.vocab = self.LEVELS + 1 + len(self.CLASSES)
        self.block_size = self.npix + 1 + 1                  # pixels + marker + label

    def _render(self, cls):
        S = self.S
        img = np.random.rand(S, S) * 0.15                    # faint background noise
        if cls == 0:                                         # horizontal line
            r = np.random.randint(1, S - 1); img[r, :] = 1.0
        elif cls == 1:                                       # vertical line
            c = np.random.randint(1, S - 1); img[:, c] = 1.0
        elif cls == 2:                                       # diagonal
            for k in range(S): img[k, k] = 1.0
        elif cls == 3:                                       # box outline
            img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 1.0
        elif cls == 4:                                       # cross
            img[S // 2, :] = 1.0; img[:, S // 2] = 1.0
        q = np.clip((img * (self.LEVELS - 1) + 0.5), 0, self.LEVELS - 1).astype(np.int64)
        return q                                             # (S,S) ints in 0..LEVELS-1

    def batch(self, bs):
        x = np.zeros((bs, self.block_size), np.int64)
        y = np.full((bs, self.block_size), -1, np.int64)
        mask = np.zeros((bs, self.block_size), np.float32)
        for i in range(bs):
            cls = np.random.randint(0, len(self.CLASSES))
            pix = self._render(cls).reshape(-1)
            seq = np.concatenate([pix, [self.MARK], [self.CLS0 + cls]])
            x[i] = seq
            y[i, -2] = self.CLS0 + cls           # predict the class after the marker
            mask[i, -2] = 1.0
        return x, y, mask

    def decode(self, row):
        cls = int(row[-1]) - self.CLS0
        name = self.CLASSES[cls] if 0 <= cls < len(self.CLASSES) else "?"
        return f"image -> {name}"

    def image_of(self, row):
        """Return the 8x8 grid (for the website to draw the actual picture)."""
        return np.array(row[:self.npix]).reshape(self.S, self.S).tolist()


TASKS = {t.name: t for t in [SortTask(), AddTask(), ShapesTask()]}
