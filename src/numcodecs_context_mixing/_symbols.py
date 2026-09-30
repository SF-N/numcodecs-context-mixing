"""Context-mixing coder for integer arrays with a small alphabet.

Every symbol is coded directly (MSB first through a binary tree) with contexts
built from the neighbouring symbols in the current slice and the previous
slice, their local activity, and a median-edge prediction.
"""

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

NMODELS = 7
TABLE_BITS = 22
MAX_BITS = 16


@njit(cache=True)
def _neighbours(q, m, t, i, j, T, Y, X, nb):
    # neighbour symbols, or -1 outside the array / at masked positions
    def g(tt, ii, jj):
        if ii < 0 or ii >= Y or jj < 0 or jj >= X or tt < 0:
            return -1
        if m[tt, ii, jj] == 1:
            return -1
        return q[tt, ii, jj]

    nb[0] = g(t, i, j - 1)  # L
    nb[1] = g(t, i - 1, j)  # U
    nb[2] = g(t, i - 1, j - 1)  # UL
    nb[3] = g(t, i - 1, j + 1)  # UR
    nb[4] = g(t, i, j - 2)  # LL
    nb[5] = g(t, i - 2, j)  # UU
    nb[6] = g(t - 1, i, j)  # P (previous slice)
    nb[7] = g(t - 1, i, j + 1)  # PR
    nb[8] = g(t - 1, i + 1, j)  # PD
    nb[9] = g(t, i - 1, j + 2)  # URR


@njit(cache=True)
def _pred(nb):
    """Median-edge-detector style prediction from L, U, UL (with fallbacks)."""
    L, U, UL = nb[0], nb[1], nb[2]
    if L >= 0 and U >= 0 and UL >= 0:
        mx = max(L, U)
        mn = min(L, U)
        if UL >= mx:
            return mn
        if UL <= mn:
            return mx
        return L + U - UL
    if L >= 0 and U >= 0:
        return (L + U + 1) // 2
    if L >= 0:
        return L
    if U >= 0:
        return U
    if nb[3] >= 0:
        return nb[3]
    if nb[6] >= 0:
        return nb[6]
    return -1


@njit(cache=True)
def _code_symbols(q, m, T, Y, X, nbits, out, inp, state, encode, lr, lim):
    probs = np.full((NMODELS, 1 << TABLE_BITS), MODEL_ONE // 2, np.int32)
    counts = np.zeros((NMODELS, 1 << TABLE_BITS), np.uint8)
    nnodes = 1 << nbits
    weights = np.full((nnodes * 4, NMODELS + 1), 0.25, np.float64)
    idx = np.zeros(NMODELS, np.int64)
    st = np.zeros(NMODELS + 1, np.float64)
    nb = np.zeros(10, np.int64)
    dt = _make_dt(lim)
    tmask = (1 << TABLE_BITS) - 1
    for t in range(T):
        for i in range(Y):
            for j in range(X):
                if m[t, i, j] == 1:
                    continue
                _neighbours(q, m, t, i, j, T, Y, X, nb)
                L, U, UL, UR, LL, UU = nb[0], nb[1], nb[2], nb[3], nb[4], nb[5]
                P, PR, URR = nb[6], nb[7], nb[9]
                pred = _pred(nb)
                # activity / texture context
                act = 0
                if L >= 0 and U >= 0:
                    act = abs(L - U)
                if UL >= 0 and U >= 0:
                    act += abs(UL - U)
                if UR >= 0 and U >= 0:
                    act += abs(UR - U)
                if L >= 0 and LL >= 0:
                    act += abs(L - LL)
                if act > 12:
                    act = 12
                dP = 0
                if P >= 0 and L >= 0:
                    dP = P - L
                    if dP > 6:
                        dP = 6
                    elif dP < -6:
                        dP = -6
                # weight-set selector: tree node + coarse activity
                asel = 0 if act == 0 else (1 if act <= 2 else (2 if act <= 5 else 3))
                node = 1
                sym = q[t, i, j] if encode else 0
                for b in range(nbits - 1, -1, -1):
                    idx[0] = _hash(L + 1, U + 1, node, 11, 0, tmask)
                    idx[1] = _hash(L + 1, U + 1, UL + 1, UR + 1, node * 8 + 12, tmask)
                    idx[2] = _hash(pred + 1, act, node, 13, 0, tmask)
                    idx[3] = _hash(L + 1, P + 1, node, 14, 0, tmask)
                    idx[4] = _hash(U + 1, P + 1, PR + 1, node, 15, tmask)
                    idx[5] = _hash(L + 1, LL + 1, U + 1, UU + 1, node * 8 + 16, tmask)
                    idx[6] = _hash(pred + 1, dP + 8, URR + 1, node, 17, tmask)
                    wsel = node * 4 + asel
                    pmix = _predict(probs, counts, idx, weights, wsel, st, NMODELS)
                    p12 = _to_p12(pmix)
                    if encode:
                        bit = (sym >> b) & 1
                        _enc_bit(state, out, p12, bit)
                    else:
                        bit = _dec_bit(state, inp, p12)
                    _update(
                        probs,
                        counts,
                        idx,
                        weights,
                        wsel,
                        st,
                        NMODELS,
                        pmix,
                        bit,
                        lr,
                        dt,
                    )
                    node = node * 2 + bit
                if not encode:
                    q[t, i, j] = node - nnodes


@njit(cache=True)
def encode_symbols(q, m, nbits, out, lr, lim):
    """Encode the non-negative int64 array `q` of shape (T, Y, X), skipping the
    positions where the uint8 mask `m` is 1; returns the length."""
    T, Y, X = q.shape
    state = np.zeros(5, np.int64)
    _enc_init(state)
    dummy = np.zeros(1, np.uint8)
    _code_symbols(q, m, T, Y, X, nbits, out, dummy, state, True, lr, lim)
    _enc_flush(state, out)
    return state[4]


@njit(cache=True)
def decode_symbols(inp, m, T, Y, X, nbits, lr, lim):
    """Decode a non-negative int64 array of shape (T, Y, X) from the bytes `inp`;
    positions where the uint8 mask `m` is 1 were skipped and decode to 0."""
    state = np.zeros(5, np.int64)
    _dec_init(inp, state)
    q = np.zeros((T, Y, X), np.int64)
    dummy = np.zeros(1, np.uint8)
    _code_symbols(q, m, T, Y, X, nbits, dummy, inp, state, False, lr, lim)
    return q
