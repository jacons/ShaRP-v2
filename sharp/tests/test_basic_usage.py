import pytest
from itertools import product
import numpy as np
from sklearn.utils import check_random_state
from sharp import ShaRP
from sharp.qoi import get_qoi
from sharp._measures import MEASURES

# Set up some envrionment variables
RNG_SEED = 123
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)

rank_qois_str = ["rank", "rank_score", "top_k"]
rank_qois_obj = [get_qoi(qoi) for qoi in rank_qois_str]

measures = list(MEASURES.keys())  # only "shapley"


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)


@pytest.mark.parametrize("qoi, measure", product(rank_qois_str, measures))
def test_explain_all_rank_str_qoi(qoi, measure):
    """
    Test simple initialization and computation of ``measure'' values using qoi definions
    passed as type ``str'' (for ranking only).
    """
    xai = ShaRP(
        qoi=qoi,
        target_function=score_function,
        measure=measure,
        random_state=RNG_SEED,
    )
    xai.fit(X)
    contributions = xai.all(X)

    np.testing.assert_allclose(contributions.mean(), 0, atol=1e-07)


@pytest.mark.parametrize("qoi, measure", product(rank_qois_obj, measures))
def test_explain_all_rank_obj_qoi(qoi, measure):
    """
    Test simple initialization and computation of ``measure'' values using qoi definions
    passed as type ``str'' (for ranking only).
    """
    qoi = qoi(target_function=score_function, X=X)
    xai = ShaRP(
        qoi=qoi,
        measure=measure,
        random_state=RNG_SEED,
    )
    xai.fit(X)
    contributions = xai.all(X)

    np.testing.assert_allclose(contributions.mean(), 0, atol=1e-07)
