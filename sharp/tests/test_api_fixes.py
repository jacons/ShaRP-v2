import numpy as np
import pytest

from sharp.base import ShaRP
from sharp.tests.make_golden import DATASETS

X, f = DATASETS["gauss3"]


def _fitted(seed=0):
    xai = ShaRP(qoi="rank", target_function=f, random_state=seed)
    xai.fit(X)
    return xai


def test_feature_default_X():
    xai = _fitted()
    value = xai.feature(0)
    assert np.isfinite(value)


def test_individual_numpy_integer():
    a = _fitted().individual(3, X)
    b = _fitted().individual(np.int64(3), X)
    np.testing.assert_allclose(b, a)


def test_individual_returns_ndarray():
    xai = _fitted()
    out = xai.individual(0, X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_pairwise_with_X_and_indices():
    expected = _fitted().pairwise(X[2], X[3])
    actual = _fitted().pairwise(2, 3, X=X)
    np.testing.assert_allclose(actual, expected)


def test_individual_accepts_list():
    a = _fitted().individual(X[0], X)
    b = _fitted().individual(list(X[0]), X)
    np.testing.assert_allclose(b, a)


def test_fit_feature_names_argument():
    xai = ShaRP(qoi="rank", target_function=f, random_state=0)
    xai.fit(X, feature_names=["a", "b", "c"])
    assert xai.feature_names_.tolist() == ["a", "b", "c"]


def test_fit_dataframe_feature_names():
    pd = pytest.importorskip("pandas")
    xai = ShaRP(qoi="rank", target_function=f, random_state=0)
    xai.fit(pd.DataFrame(X, columns=["a", "b", "c"]))
    assert xai.feature_names_.tolist() == ["a", "b", "c"]
    assert np.isfinite(xai.feature("b"))
