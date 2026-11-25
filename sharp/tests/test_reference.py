import numpy as np
import pytest
from numpy.testing import assert_allclose

from sharp.tests._reference import exact_shapley
from sharp.tests.make_golden import DATASETS


def test_reference_hand_example():
    X = np.array([[2., 0.], [1., 1.], [0., 0.]]); f = lambda X: X[:, 0] + X[:, 1]
    assert_allclose(exact_shapley(f, X, X, X[0], "rank"), [0.75, -0.25])
    assert_allclose(exact_shapley(f, X, X, X[0], "top_k", top_k=1), [0.75, -0.25])
    assert_allclose(exact_shapley(f, X, X, X[0], "score"), [1.5, -0.5])


def _drop_first(X, v):
    for i, r in enumerate(X):
        if np.array_equal(r, v):
            return np.delete(X, i, axis=0)
    return X


@pytest.mark.parametrize("qoi,k", [("score", None), ("rank", None), ("top_k", 5)])
def test_reference_efficiency(qoi, k):
    X, f = DATASETS["int4"]
    X = np.asarray(X, dtype=float)
    v = X[0]
    Xr = _drop_first(X, v)
    donors = _drop_first(X, v)
    fr = f(Xr)

    def q(x):
        s = float(f(x[None, :])[0])
        if qoi == "score":
            return s
        rank = 1 + np.sum(fr > s)
        return float(rank) if qoi == "rank" else float(rank <= k)

    phi = exact_shapley(f, X, X, v, qoi, top_k=k)
    mean_u = np.mean([q(u) for u in donors])
    expected = mean_u - q(v) if qoi == "rank" else q(v) - mean_u
    assert_allclose(phi.sum(), expected, atol=1e-12, rtol=0)
