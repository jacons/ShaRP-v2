import numpy as np
import pytest

from sharp.base import ShaRP
from sharp.tests._reference import exact_shapley
from sharp.tests.make_golden import DATASETS


def _fit(f, X, qoi="rank_score", **kw):
    xai = ShaRP(qoi=qoi, target_function=f, random_state=0, **kw)
    xai.fit(X)
    return xai


def test_invalid_semantics():
    X, f = DATASETS["gauss3"]
    with pytest.raises(ValueError):
        ShaRP(qoi="rank", target_function=f, semantics="x").fit(X)


def test_paper_indices_follow_user_order():
    X, f = DATASETS["gauss3"]
    out = _fit(f, X, sample_size=None).individual(0)
    np.testing.assert_allclose(out, exact_shapley(f, X, X, X[0], "score"), atol=1e-12)


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("i", [0, 7, 13])
def test_paper_rank_score_matches_reference(name, i):
    X, f = DATASETS[name]
    out = _fit(f, X, sample_size=None).individual(i, X)
    np.testing.assert_allclose(out, exact_shapley(f, X, X, X[i], "score"), atol=1e-12)


def test_paper_all_rank_score_matches_reference():
    X, f = DATASETS["int4"]
    out = _fit(f, X, sample_size=None).all(X)
    for i in range(len(X)):
        np.testing.assert_allclose(
            out[i], exact_shapley(f, X, X, X[i], "score"), atol=1e-12
        )


def test_paper_tiny_pool():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    f = lambda Z: Z[:, 0] + 2 * Z[:, 1]
    out = _fit(f, X, sample_size=5, replace=False).individual(0, X)
    np.testing.assert_allclose(out, exact_shapley(f, X, X, X[0], "score"), atol=1e-12)


def test_legacy_index_uses_sorted_X():
    X, f = DATASETS["gauss3"]
    a = _fit(f, X, sample_size=None, semantics="legacy").individual(0)
    b = _fit(f, X, sample_size=None, semantics="legacy").individual(
        int(np.argmin(f(X))), X
    )
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_pairwise_numpy_integer_indices():
    X, f = DATASETS["gauss3"]
    a = _fit(f, X).pairwise(2, 3, X=X)
    b = _fit(f, X).pairwise(np.int64(2), np.int64(3), X=X)
    np.testing.assert_allclose(b, a)


def test_pairwise_caller_sample_and_coalition_size():
    X, f = DATASETS["gauss3"]
    out = _fit(f, X).pairwise(X[2], X[5:10], sample_size=3, coalition_size=1)
    assert out.shape == (3,)


def test_all_default_X():
    X, f = DATASETS["gauss3"]
    xai = _fit(f, X, sample_size=None)
    np.testing.assert_allclose(xai.all(), xai.all(X))


def test_positional_signature_unchanged():
    X, f = DATASETS["gauss3"]
    xai = ShaRP("rank", f, "shapley", None, None, False, 0)
    assert xai.random_state == 0 and xai.semantics == "paper"


@pytest.mark.parametrize("name", list(DATASETS))
@pytest.mark.parametrize("qoi,k", [("rank", None), ("top_k", 1), ("top_k", 5), ("top_k", 10)])
@pytest.mark.parametrize("i", [0, 7, 13])
def test_paper_rank_topk_match_reference(name, qoi, k, i):
    X, f = DATASETS[name]
    kw = {"top_k": k} if k else {}
    out = _fit(f, X, qoi, sample_size=None, **kw).individual(i, X)
    np.testing.assert_allclose(
        out, exact_shapley(f, X, X, X[i], qoi, k), atol=1e-12, rtol=0
    )


def test_paper_hand_example():
    X = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    f = lambda Z: Z[:, 0] + Z[:, 1]
    for qoi, kw, want in [
        ("rank", {}, [0.75, -0.25]),
        ("top_k", {"top_k": 1}, [0.75, -0.25]),
        ("rank_score", {}, [1.5, -0.5]),
    ]:
        out = _fit(f, X, qoi, sample_size=None, **kw).individual(0, X)
        np.testing.assert_allclose(out, want, atol=1e-12, rtol=0)


def test_paper_external_item():
    X, f = DATASETS["gauss3"]
    v = np.array([0.1, 0.2, 0.3])
    out = _fit(f, X, "rank", sample_size=None).individual(v, X)
    np.testing.assert_allclose(
        out, exact_shapley(f, X, X, v, "rank"), atol=1e-12, rtol=0
    )
    ranks = 1 + (f(X)[None, :] > f(X)[:, None]).sum(1)
    rank_v = 1 + (f(X) > f(v[None, :])[0]).sum()
    np.testing.assert_allclose(out.sum(), ranks.mean() - rank_v, atol=1e-12)


def test_paper_int_target():
    X, f0 = DATASETS["int4"]
    f = lambda Z: np.asarray(f0(Z)).astype(int)
    for i in (0, 7):
        out = _fit(f, X, "rank", sample_size=None).individual(i, X)
        np.testing.assert_allclose(
            out, exact_shapley(f, X, X, X[i], "rank"), atol=1e-12, rtol=0
        )


def test_paper_pairwise_matches_reference():
    X, f = DATASETS["gauss3"]
    out = _fit(f, X, "rank", sample_size=None).pairwise(X[2], X[3])
    np.testing.assert_allclose(
        out, exact_shapley(f, X, X[3:4], X[2], "rank"), atol=1e-12, rtol=0
    )
