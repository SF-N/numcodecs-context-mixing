"""Context-mixing coder for integer arrays via prediction residuals.

Each value is predicted either by a median edge detector from its causal 2D
neighbours or, if a previous slice exists and has been more accurate recently,
from the co-located value of the previous slice corrected by the neighbours'
slice-to-slice differences.  The residual is coded MSB first through a binary
tree with contexts from the local activity, the disagreement of the two
predictors and the neighbouring residuals in the current and previous slice.
"""

import numpy as np
from numba import njit

from ._coder import (
    MODEL_ONE,
    _bucket,
    _dec_bit,
    _dec_init,
    _enc_bit,
    _enc_flush,
    _enc_init,
    _hash,
    _make_dt,
    _med,
    _predict,
    _sbucket,
    _to_p12,
    _update,
)

NMODELS = 8
TABLE_BITS = 22
MAX_BITS = 60


@njit(cache=True)
def _code_residuals(q, T, Y, X, nbits, out, inp, state, encode, lr, lim):
    tsize = 1 << TABLE_BITS
    tmask = tsize - 1
    probs = np.full((NMODELS, tsize), MODEL_ONE // 2, np.int32)
    counts = np.zeros((NMODELS, tsize), np.uint8)
    nsets = 4096 * 8
    weights = np.full((nsets, NMODELS + 1), 0.22, np.float64)
    idx = np.zeros(NMODELS, np.int64)
    st = np.zeros(NMODELS + 1, np.float64)
    dt = _make_dt(lim)
    rbits = nbits + 1
    roff = 1 << nbits
    # residual planes for context (current and previous slice)
    res = np.zeros((Y, X), np.int64)
    res_prev = np.zeros((Y, X), np.int64)
    # running error estimates of the two predictors
    err_med = 1.0
    err_tim = 1.0
    for t in range(T):
        for i in range(Y):
            for j in range(X):
                # causal neighbours (current slice), -1 outside
                L = q[t, i, j - 1] if j > 0 else -1
                U = q[t, i - 1, j] if i > 0 else -1
                UL = q[t, i - 1, j - 1] if (i > 0 and j > 0) else -1
                UR = q[t, i - 1, j + 1] if (i > 0 and j + 1 < X) else -1
                LL = q[t, i, j - 2] if j > 1 else -1
                UU = q[t, i - 2, j] if i > 1 else -1
                # previous slice
                P = q[t - 1, i, j] if t > 0 else -1

                # --- spatial prediction
                if L >= 0 and U >= 0 and UL >= 0:
                    pmed = _med(L, U, UL)
                elif L >= 0 and U >= 0:
                    pmed = (L + U + 1) // 2
                elif L >= 0:
                    pmed = L
                elif U >= 0:
                    pmed = U
                elif UR >= 0:
                    pmed = UR
                elif UL >= 0:
                    pmed = UL
                elif P >= 0:
                    pmed = P
                else:
                    pmed = roff // 2
                # --- previous-slice prediction
                have_tim = P >= 0
                ptim = pmed
                if have_tim:
                    dL = (L - q[t - 1, i, j - 1]) if L >= 0 else -100000
                    dU = (U - q[t - 1, i - 1, j]) if U >= 0 else -100000
                    dUL = (UL - q[t - 1, i - 1, j - 1]) if UL >= 0 else -100000
                    if dL > -100000 and dU > -100000 and dUL > -100000:
                        ptim = P + _med(dL, dU, dUL)
                    elif dL > -100000 and dU > -100000:
                        ptim = P + (dL + dU) // 2
                    elif dL > -100000:
                        ptim = P + dL
                    elif dU > -100000:
                        ptim = P + dU
                    else:
                        ptim = P
                    if ptim < 0:
                        ptim = 0
                    elif ptim >= roff:
                        ptim = roff - 1
                if pmed < 0:
                    pmed = 0
                elif pmed >= roff:
                    pmed = roff - 1
                use_tim = have_tim and (err_tim < err_med)
                pred = ptim if use_tim else pmed
                # --- contexts
                act = 0
                if L >= 0 and U >= 0:
                    act += abs(L - U)
                if UL >= 0 and U >= 0:
                    act += abs(UL - U)
                if UR >= 0 and U >= 0:
                    act += abs(UR - U)
                if L >= 0 and LL >= 0:
                    act += abs(L - LL)
                if U >= 0 and UU >= 0:
                    act += abs(U - UU)
                ab = _bucket(act)
                dis = _sbucket(ptim - pmed) if have_tim else 12
                rL = _sbucket(res[i, j - 1]) if j > 0 else 11
                rU = _sbucket(res[i - 1, j]) if i > 0 else 11
                rUL = _sbucket(res[i - 1, j - 1]) if (i > 0 and j > 0) else 11
                rUR = _sbucket(res[i - 1, j + 1]) if (i > 0 and j + 1 < X) else 11
                rP = _sbucket(res_prev[i, j]) if t > 0 else 11
                pb = pred >> max(0, nbits - 5)
                gdir = 0
                if L >= 0 and U >= 0:
                    gdir = 1 if L > U else (2 if L < U else 0)
                    if UL >= 0:
                        gdir = gdir * 3 + (1 if U > UL else (2 if U < UL else 0))
                sel_tim = 1 if use_tim else 0

                sym = (q[t, i, j] - pred + roff) if encode else 0
                node = 1
                for b in range(rbits - 1, -1, -1):
                    idx[0] = _hash(node, ab, sel_tim, 1, 0, tmask)
                    idx[1] = _hash(node, ab, dis + 20, sel_tim, 2, tmask)
                    idx[2] = _hash(node, rL + 20, rU + 20, 3, 0, tmask)
                    idx[3] = _hash(node, rL + 20, rUL + 20, rUR + 20, 4, tmask)
                    idx[4] = _hash(node, pb, ab, 5, 0, tmask)
                    idx[5] = _hash(node, gdir, ab, rL + 20, 6, tmask)
                    idx[6] = _hash(node, rP + 20, dis + 20, sel_tim, 7, tmask)
                    idx[7] = _hash(node, rL + 20, ab, dis + 20, 8, tmask)
                    nsel = node if node < 4096 else (4095 - (rbits - 1 - b))
                    wsel = (nsel * 8 + (ab if ab < 8 else 7)) % nsets
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
                    sym = node - (1 << rbits)
                    q[t, i, j] = sym - roff + pred
                res[i, j] = sym - roff
                # update predictor error trackers
                qv = q[t, i, j]
                err_med = 0.98 * err_med + 0.02 * abs(qv - pmed)
                if have_tim:
                    err_tim = 0.98 * err_tim + 0.02 * abs(qv - ptim)
        # roll residual plane
        for i in range(Y):
            for j in range(X):
                res_prev[i, j] = res[i, j]


@njit(cache=True)
def encode_residuals(q, nbits, out, lr, lim):
    """Encode the non-negative int64 array `q` of shape (T, Y, X); returns the length."""
    T, Y, X = q.shape
    state = np.zeros(5, np.int64)
    _enc_init(state)
    dummy = np.zeros(1, np.uint8)
    _code_residuals(q, T, Y, X, nbits, out, dummy, state, True, lr, lim)
    _enc_flush(state, out)
    return state[4]


@njit(cache=True)
def decode_residuals(inp, T, Y, X, nbits, lr, lim):
    """Decode a non-negative int64 array of shape (T, Y, X) from the bytes `inp`."""
    state = np.zeros(5, np.int64)
    _dec_init(inp, state)
    q = np.zeros((T, Y, X), np.int64)
    dummy = np.zeros(1, np.uint8)
    _code_residuals(q, T, Y, X, nbits, dummy, inp, state, False, lr, lim)
    return q
