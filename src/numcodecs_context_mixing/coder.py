"""
Low-level building blocks for context-mixing codecs: a binary range coder
(LZMA-style, 12-bit probabilities), adaptive bit models, a logistic mixer,
context hashing and a few bucketing helpers.

All functions are numba-compiled and operate on plain NumPy arrays so that the
encoder and decoder execute exactly the same arithmetic. They are shared by
the coders in this package and can be used to build further context-mixing
codecs (e.g. [`numcodecs-interp-ctx`](https://github.com/SF-N/numcodecs-interp-ctx)).
"""

import math

import numpy as np
from numba import njit

PROB_BITS = 12
PROB_ONE = 1 << PROB_BITS
TOP = 1 << 24
MASK32 = 0xFFFFFFFF

# model probability precision (16 bits) and adaptation schedule
MODEL_ONE = 1 << 16
ADAPT_LIMIT = 255


@njit(cache=True)
def _shift_low(low, cache, cache_size, out, pos):
    if low < 0xFF000000 or low >= (1 << 32):
        carry = low >> 32
        temp = cache
        while True:
            out[pos] = (temp + carry) & 0xFF
            pos += 1
            temp = 0xFF
            cache_size -= 1
            if cache_size == 0:
                break
        cache = (low >> 24) & 0xFF
    cache_size += 1
    low = (low & 0x00FFFFFF) << 8
    return low, cache, cache_size, pos


@njit(cache=True)
def _enc_init(state):
    state[0] = 0
    state[1] = MASK32
    state[2] = 0
    state[3] = 1
    state[4] = 0


@njit(cache=True)
def _enc_bit(state, out, p1, bit):
    """Encode `bit` with probability p1 = P(bit=1) in 12 bits (1..4095)."""
    low, rng, cache, cache_size, pos = state[0], state[1], state[2], state[3], state[4]
    p0 = PROB_ONE - p1
    bound = (rng >> PROB_BITS) * p0
    if bit == 0:
        rng = bound
    else:
        low += bound
        rng -= bound
    while rng < TOP:
        rng = (rng << 8) & MASK32
        low, cache, cache_size, pos = _shift_low(low, cache, cache_size, out, pos)
    state[0], state[1], state[2], state[3], state[4] = low, rng, cache, cache_size, pos


@njit(cache=True)
def _enc_flush(state, out):
    low, cache, cache_size, pos = state[0], state[2], state[3], state[4]
    for _ in range(5):
        low, cache, cache_size, pos = _shift_low(low, cache, cache_size, out, pos)
    state[0], state[2], state[3], state[4] = low, cache, cache_size, pos


@njit(cache=True)
def _dec_init(inp, state):
    code = 0
    pos = 0
    for _ in range(5):
        code = ((code << 8) | inp[pos]) & MASK32
        pos += 1
    state[0] = code
    state[1] = MASK32
    state[4] = pos


@njit(cache=True)
def _dec_bit(state, inp, p1):
    code, rng, pos = state[0], state[1], state[4]
    p0 = PROB_ONE - p1
    bound = (rng >> PROB_BITS) * p0
    if code < bound:
        rng = bound
        bit = 0
    else:
        code -= bound
        rng -= bound
        bit = 1
    while rng < TOP:
        rng = (rng << 8) & MASK32
        code = ((code << 8) | inp[pos]) & MASK32
        pos += 1
    state[0], state[1], state[4] = code, rng, pos
    return bit


@njit(cache=True)
def _stretch(p):
    return math.log(p / (1.0 - p))


@njit(cache=True)
def _squash(x):
    if x > 30.0:
        x = 30.0
    elif x < -30.0:
        x = -30.0
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True)
def _predict(probs, counts, idx, weights, wsel, st, nmodels):
    """Compute mixed P(bit=1) from model slots `idx[i]` and weight set `wsel`."""
    dot = 0.0
    for i in range(nmodels):
        p = (probs[i, idx[i]] + 0.5) / MODEL_ONE
        s = _stretch(p)
        st[i] = s
        dot += weights[wsel, i] * s
    # bias input
    st[nmodels] = 0.3
    dot += weights[wsel, nmodels] * 0.3
    return _squash(dot)


@njit(cache=True)
def _update(probs, counts, idx, weights, wsel, st, nmodels, pmix, bit, lr, dt):
    err = (bit - pmix) * lr
    for i in range(nmodels + 1):
        weights[wsel, i] += err * st[i]
    for i in range(nmodels):
        j = idx[i]
        n = counts[i, j]
        target = MODEL_ONE - 1 if bit == 1 else 0
        p = probs[i, j]
        p = p + int((target - p) * dt[n])
        if p < 1:
            p = 1
        elif p > MODEL_ONE - 1:
            p = MODEL_ONE - 1
        probs[i, j] = p
        if n < ADAPT_LIMIT:
            counts[i, j] = n + 1


@njit(cache=True)
def _to_p12(pmix):
    p = int(pmix * PROB_ONE)
    if p < 1:
        p = 1
    elif p > PROB_ONE - 1:
        p = PROB_ONE - 1
    return p


@njit(cache=True)
def _make_dt(limit_rate):
    """Per-count adaptation rates 1/(n+1.5), floored at `limit_rate`."""
    dt = np.empty(ADAPT_LIMIT + 1, np.float64)
    for n in range(ADAPT_LIMIT + 1):
        r = 1.0 / (n + 1.5)
        if r < limit_rate:
            r = limit_rate
        dt[n] = r
    return dt


@njit(cache=True)
def _hash(a, b, c, d, e, mask):
    h = a * 0x9E3779B1 + b
    h = (h ^ (h >> 15)) * 0x85EBCA77 + c
    h = (h ^ (h >> 13)) * 0xC2B2AE3D + d
    h = (h ^ (h >> 16)) * 0x27D4EB2F + e
    h = h ^ (h >> 15)
    return h & mask


@njit(cache=True)
def _bucket(v):
    """log2-ish bucket of a non-negative integer."""
    if v <= 0:
        return 0
    if v == 1:
        return 1
    if v == 2:
        return 2
    if v <= 4:
        return 3
    if v <= 8:
        return 4
    if v <= 16:
        return 5
    if v <= 32:
        return 6
    if v <= 64:
        return 7
    if v <= 128:
        return 8
    return 9


@njit(cache=True)
def _sbucket(v):
    """Signed log2-ish bucket."""
    if v < 0:
        return -_bucket(-v)
    return _bucket(v)


@njit(cache=True)
def _med(L, U, UL):
    """Median edge detector prediction (LOCO-I / JPEG-LS)."""
    mx = max(L, U)
    mn = min(L, U)
    if UL >= mx:
        return mn
    if UL <= mn:
        return mx
    return L + U - UL
