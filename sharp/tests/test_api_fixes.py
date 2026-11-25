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
    assert xai.feature(0) == _fitted().feature(0, X)


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


# --- final-review fixes -----------------------------------------------------

from sharp.qoi.base import TargetFunctionWrapper  # noqa: E402

X4, f4 = DATASETS["int4"]
QOIS = [("rank", {}), ("top_k", {"top_k": 5}), ("rank_score", {})]


@pytest.mark.parametrize("qoi,kw", QOIS)
@pytest.mark.parametrize("extra", [{}, {"replace": True, "sample_size": 3}])
def test_A1_duplicate_only_donor(qoi, kw, extra):
    xai = ShaRP(qoi=qoi, target_function=f4, random_state=0, **kw, **extra)
    xai.fit(X4)
    np.testing.assert_array_equal(xai.pairwise(X4[10], X4[12]), 0)


@pytest.mark.parametrize("qoi,kw", QOIS)
def test_A1_single_row_fit(qoi, kw):
    xai = ShaRP(qoi=qoi, target_function=f4, random_state=0, **kw).fit(X4[:1])
    np.testing.assert_array_equal(xai.individual(0), 0)


def test_A2_all_uses_passed_X_under_paper():
    other = X[20:]
    mk = lambda: ShaRP(qoi="rank", target_function=f, random_state=0).fit(X[:20])
    out = mk().all(other)
    for i in (0, 3):
        np.testing.assert_allclose(
            out[i], mk().individual(i, other), atol=1e-12, rtol=0
        )


def test_A4_one_row_2d_sample():
    for sem in ("paper", "legacy"):
        mk = lambda: ShaRP(
            qoi="rank", target_function=f, random_state=0, semantics=sem
        ).fit(X)
        np.testing.assert_allclose(
            mk().individual(X[0:1], X), mk().individual(X[0], X)
        )
    ShaRP(qoi="rank", target_function=f, random_state=0).fit(X).pairwise(
        [2], [3, 4], X=X
    )


def test_B1_wrapper_rejects_wrong_output_size():
    w = TargetFunctionWrapper(lambda Z: np.c_[Z[:, 0], Z[:, 0]])
    with pytest.raises(ValueError):
        w(X[:4])


def test_B3_legacy_custom_measure_without_exclude():
    def measure(row, col_idx, set_cols_idx, X, qoi, sample_size, coalition_size,
                replace, rng):
        return 1.0

    xai = ShaRP(
        qoi="rank", target_function=f, measure=measure, semantics="legacy",
        random_state=0,
    ).fit(X)
    np.testing.assert_array_equal(xai.individual(0, X), 1.0)


def test_B4_fit_returns_self():
    xai = ShaRP(qoi="rank", target_function=f, random_state=0)
    assert xai.fit(X) is xai
    assert xai.fit(X).individual(0).shape == (3,)


def test_B5_unknown_measure():
    with pytest.raises(ValueError, match="shapley"):
        ShaRP(qoi="rank", target_function=f, measure="nope").fit(X)
