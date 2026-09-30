"""Context-mixing coder for boolean bitmaps."""

import numpy as np
from numba import njit

from .coder import (
    MODEL_ONE,
    _dec_bit,
    _dec_init,
    _enc_bit,
    _enc_flush,
    _enc_init,
    _hash,
    _make_dt,
    _predict,
    _to_p12,
    _update,
)

NMODELS = 5
TABLE_BITS = 22


@njit(cache=True)
def _bitmap_ctx(m, t, i, j, T, Y, X, idx, mask):
    # neighbours in the current slice (causal) and the previous slice; 2 = outside
    def g(tt, ii, jj):
        if ii < 0 or ii >= Y or jj < 0 or jj >= X or tt < 0:
            return 2
        return m[tt, ii, jj]

    L = g(t, i, j - 1)
    LL = g(t, i, j - 2)
    L3 = g(t, i, j - 3)
    U = g(t, i - 1, j)
    UL = g(t, i - 1, j - 1)
    UR = g(t, i - 1, j + 1)
    ULL = g(t, i - 1, j - 2)
    URR = g(t, i - 1, j + 2)
    UU = g(t, i - 2, j)
    UUL = g(t, i - 2, j - 1)
    UUR = g(t, i - 2, j + 1)
    U3 = g(t, i - 3, j)
    P = g(t - 1, i, j)
    PL = g(t - 1, i, j - 1)
    PR = g(t - 1, i, j + 1)
    PU = g(t - 1, i - 1, j)
    PD = g(t - 1, i + 1, j)

    c1 = ((((L * 3 + U) * 3 + UL) * 3 + UR) * 3 + LL) * 3 + UU
    c2 = ((((c1 * 3 + ULL) * 3 + URR) * 3 + UUL) * 3 + UUR) * 3 + L3
    c3 = (((P * 3 + PL) * 3 + PR) * 3 + PU) * 3 + PD
    idx[0] = _hash(c1, 1, 0, 0, 0, mask)
    idx[1] = _hash(c2, 2, U3, 0, 0, mask)
    idx[2] = _hash(c1, 3, c3, 0, 0, mask)
    idx[3] = _hash(c2, 4, c3, U3, 0, mask)
    # run-length like context: distance to the last transition in the row
    d = 0
    while j - 1 - d >= 0 and d < 32 and m[t, i, j - 1 - d] == L:
        d += 1
    idx[4] = _hash(L, 5, U, d, P, mask)
    return L * 3 + U


@njit(cache=True)
def _code_bitmap(m, T, Y, X, out, inp, state, encode, lr, lim):
    probs = np.full((NMODELS, 1 << TABLE_BITS), MODEL_ONE // 2, np.int32)
    counts = np.zeros((NMODELS, 1 << TABLE_BITS), np.uint8)
    weights = np.full((16, NMODELS + 1), 0.3, np.float64)
    idx = np.zeros(NMODELS, np.int64)
    st = np.zeros(NMODELS + 1, np.float64)
    dt = _make_dt(lim)
    tmask = (1 << TABLE_BITS) - 1
    for t in range(T):
        for i in range(Y):
            for j in range(X):
                wsel = _bitmap_ctx(m, t, i, j, T, Y, X, idx, tmask)
                pmix = _predict(probs, counts, idx, weights, wsel, st, NMODELS)
                p12 = _to_p12(pmix)
                if encode:
                    bit = m[t, i, j]
                    _enc_bit(state, out, p12, bit)
                else:
                    bit = _dec_bit(state, inp, p12)
                    m[t, i, j] = bit
                _update(
                    probs, counts, idx, weights, wsel, st, NMODELS, pmix, bit, lr, dt
                )


@njit(cache=True)
def encode_bitmap(m, out, lr, lim):
    """Encode the uint8 0/1 array `m` of shape (T, Y, X); returns the length."""
    T, Y, X = m.shape
    state = np.zeros(5, np.int64)
    _enc_init(state)
    dummy = np.zeros(1, np.uint8)
    _code_bitmap(m, T, Y, X, out, dummy, state, True, lr, lim)
    _enc_flush(state, out)
    return state[4]


@njit(cache=True)
def decode_bitmap(inp, T, Y, X, lr, lim):
    """Decode a uint8 0/1 array of shape (T, Y, X) from the bytes `inp`."""
    state = np.zeros(5, np.int64)
    _dec_init(inp, state)
    m = np.zeros((T, Y, X), np.uint8)
    dummy = np.zeros(1, np.uint8)
    _code_bitmap(m, T, Y, X, dummy, inp, state, False, lr, lim)
    return m
