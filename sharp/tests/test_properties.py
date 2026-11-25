import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sharp.base import ShaRP
from sharp.tests._reference import exact_shapley
from sharp.tests.make_golden import DATASETS


def _fit(f, X, qoi="rank", **kw):
    xai = ShaRP(qoi=qoi, target_function=f, semantics="paper", **kw)
    xai.fit(X)
    return xai


def _run(method, n_jobs):
    X, f = DATASETS["gauss3"]
    xai = _fit(f, X, sample_size=7, random_state=0, n_jobs=n_jobs)
    if method == "individual":
        return xai.individual(0, X)
    if method == "all":
        return xai.all(X)
    return xai.pairwise_set([0, 1, 2], [3, 4, 5], X=X)


@pytest.mark.parametrize("method", ["individual", "all", "pairwise_set"])
def test_njobs_invariance(method):
    assert_array_equal(_run(method, 2), _run(method, 1))


def _drop_first(X, v):
    for i, r in enumerate(X):
        if np.array_equal(r, v):
            return np.delete(X, i, axis=0)
    return X


@pytest.mark.parametrize("qoi,k", [("rank", None), ("top_k", 5), ("rank_score", None)])
def test_efficiency(qoi, k):
    X, f = DATASETS["int4"]
    kw = {"top_k": k} if k else {}
    xai = _fit(f, X, qoi=qoi, sample_size=None, **kw)
    for i in (0, 7):
        v = X[i]
        donors = _drop_first(X, v)
        fr = f(donors)

        def q(x):
            s = float(f(x[None, :])[0])
            if qoi == "rank_score":
                return s
            rank = 1 + np.sum(fr > s)
            return float(rank) if qoi == "rank" else float(rank <= k)

        mean_u = np.mean([q(u) for u in donors])
        expected = mean_u - q(v) if qoi == "rank" else q(v) - mean_u
        assert_allclose(xai.individual(i, X).sum(), expected, atol=1e-12, rtol=0)


def test_symmetry():
    rs = np.random.RandomState(1)
    a, b = rs.normal(size=(20, 1)), rs.normal(size=(20, 1))
    X = np.c_[a, a, b]
    phi = _fit(lambda Z: Z.sum(axis=1), X, sample_size=None).individual(0, X)
    assert_allclose(phi[0], phi[1], atol=1e-12)


@pytest.mark.parametrize("qoi,kw", [("rank", {}), ("top_k", {"top_k": 5}), ("rank_score", {})])
def test_null_player(qoi, kw):
    X = np.random.RandomState(2).normal(size=(20, 3))
    f = lambda Z: Z[:, 0] + 2 * Z[:, 1]
    phi = _fit(f, X, qoi=qoi, sample_size=None, **kw).individual(0, X)
    assert_allclose(phi[2], 0, atol=1e-12)
