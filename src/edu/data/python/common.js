/**
 * Shared Python helpers for exported train.py scripts (docs/edu/SPEC.md §6.11.1). Each string is a block of
 * top-level Python definitions, ported line by line from data/rng.js and data/split.js so the script regenerates the
 * browser's data and split exactly (scripts/edu-python-parity.test.mjs).
 *
 * PY_PRNG and PY_SPLIT need only `math`; PY_LOADERS needs torch (names are resolved when make_loaders runs, so the
 * definitions can be executed without torch, as the parity test does).
 */

export const PY_COMMON_IMPORTS = ['import math', 'import numpy as np'];
export const PY_TORCH_IMPORTS = ['import torch', 'from torch.utils.data import DataLoader, TensorDataset'];

export const PY_PRNG = `M32 = 0xFFFFFFFF


def _imul(a, b):
    return (a * b) & M32


def mulberry32(seed):
    """Seeded generator of floats in [0, 1): the same sequence as the browser trainer."""
    a = seed & M32

    def next_():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = _imul(a ^ (a >> 15), a | 1)
        t = (t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0

    return next_


def _fnv1a(text):
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch)
        h = _imul(h, 0x01000193)
    return h


def derive_seed(seed, label):
    """Independent stream per purpose: "data", "split", "init", "batches", "dropout"."""
    x = (_imul(seed & M32, 0x9E3779B1) ^ _fnv1a(label)) & M32
    x ^= x >> 16
    x = _imul(x, 0x85EBCA6B)
    x ^= x >> 13
    x = _imul(x, 0xC2B2AE35)
    x ^= x >> 16
    return x


def uniform(next_, a, b):
    return next_() * (b - a) + a


def normal(next_, mean=0.0, std=1.0):
    """Marsaglia polar method; one value per call, the second value is discarded."""
    while True:
        v1 = 2 * next_() - 1
        v2 = 2 * next_() - 1
        s = v1 * v1 + v2 * v2
        if not (s >= 1 or s == 0):
            break
    return mean + std * v1 * math.sqrt(-2 * math.log(s) / s)
`;

export const PY_SPLIT = `def split_indices(n, val, test, seed):
    """Shuffle 0..n-1 with the "split" stream; validation = the first n_val, test = the next n_test, train = the rest.
    Sizes use floor(n * f + 0.5), exactly as in the browser (not round(), which rounds half to even)."""
    next_ = mulberry32(derive_seed(seed, "split"))
    order = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(next_() * (i + 1))
        order[i], order[j] = order[j], order[i]
    n_val = math.floor(n * val + 0.5)
    n_test = math.floor(n * test + 0.5)
    return order[n_val + n_test:], order[:n_val], order[n_val:n_val + n_test]
`;

export const PY_LOADERS = `def make_loaders(X, y, splits, batch_size, train_seed):
    """Wrap arrays in DataLoaders. The training loader reshuffles every epoch with a seeded generator; the
    validation and test loaders keep a fixed order. Returns (train_loader, val_loader, test_loader or None)."""
    X = torch.as_tensor(X)
    y = torch.as_tensor(y)

    def subset(indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        return TensorDataset(X[index], y[index])

    train_idx, val_idx, test_idx = splits
    train_loader = DataLoader(subset(train_idx), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(train_seed))
    val_loader = DataLoader(subset(val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(subset(test_idx), batch_size=batch_size, shuffle=False) if len(test_idx) else None
    return train_loader, val_loader, test_loader
`;

/** A JS number as a Python literal that parses to the same double ("0.1", "200", "-0.5"). */
export function pyNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`Cannot write ${value} as a Python number`);
    return Object.is(number, -0) ? '0' : String(number);
}

/** A JS string as a Python string literal. */
export const pyString = (value) => JSON.stringify(String(value));

/** Join source blocks with the two blank lines PEP 8 puts between top-level definitions. */
export const joinBlocks = (...blocks) => blocks.filter(Boolean).map((block) => block.trimEnd()).join('\n\n\n') + '\n';

/** Merge import lists in order without duplicates. */
export const mergeImports = (...lists) => [...new Set(lists.flat())];

/** Split fractions as Python literals, with the browser's defaults. */
export function pySplit(split = {}) {
    return { val: pyNumber(split.val ?? 0.2), test: pyNumber(split.test ?? 0) };
}
