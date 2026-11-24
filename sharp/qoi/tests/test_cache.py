import inspect

import numpy as np

import sharp.qoi.base as b
from sharp import ShaRP
from sharp.qoi.base import TargetFunctionWrapper
from sharp.tests.make_golden import DATASETS


def test_cache_no_collision_on_close_floats():
    w = TargetFunctionWrapper(lambda X: X.astype(float).sum(axis=1))
    a = np.array([[1.000000001, 0.0]])
    b_ = np.array([[1.000000002, 0.0]])
    assert w(a)[0] != w(b_)[0]


def test_cache_calls_model_once_per_batch_on_uncached_rows():
    calls = []

    def f(X):
        calls.append(len(X))
        return X.astype(float).sum(axis=1)

    w = TargetFunctionWrapper(f)
    X = np.arange(12.0).reshape(6, 2).astype(object)
    w(X)
    w(X[:3])
    w(np.vstack([X[:2], [[99.0, 1.0]]]).astype(object))
    assert calls == [6, 1]


def test_cache_matches_no_cache():
    X, f = DATASETS["gauss3"]
    out = []
    for c in (True, False):
        s = ShaRP(qoi="rank", target_function=f, random_state=0, cache=c)
        s.fit(X)
        out.append(s.all(X))
    np.testing.assert_array_equal(*out)


def test_no_manager():
    assert "Manager" not in inspect.getsource(b)
