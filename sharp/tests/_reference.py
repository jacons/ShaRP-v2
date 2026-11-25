"""Brute-force Shapley reference from the ShaRP paper (Alg. 1-2). Independent of `sharp`."""
from itertools import combinations
from math import factorial

import numpy as np


def _drop_first_equal(X, v):
    for i, row in enumerate(X):
        if np.array_equal(row, v):
            return np.delete(X, i, axis=0)
    return X


def exact_shapley(f, X_rank, X_donors, v, qoi, top_k=None):
    assert qoi in ("score", "rank", "top_k")
    assert qoi != "top_k" or top_k is not None
    X_rank = np.asarray(X_rank, dtype=float)
    X_donors = np.asarray(X_donors, dtype=float)
    v = np.asarray(v, dtype=float)
    X_rank = _drop_first_equal(X_rank, v)
    donors = _drop_first_equal(X_donors, v)
    ref_scores = np.asarray(f(X_rank), dtype=float)
    n = len(v)

    def q(h):
        score = float(f(h[None, :])[0])
        if qoi == "score":
            return score
        rank = 1 + int(np.sum(ref_scores > score))
        return float(rank) if qoi == "rank" else float(rank <= top_k)

    def value(K):
        total = 0.0
        for u in donors:
            h = u.copy()
            h[list(K)] = v[list(K)]
            total += q(h)
        return total / len(donors)

    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for size in range(n):
            w = factorial(size) * factorial(n - size - 1) / factorial(n)
            for K in combinations(others, size):
                phi[i] += w * (value(K + (i,)) - value(K))
    return -phi if qoi == "rank" else phi
