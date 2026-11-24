import pytest
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import rankdata
from sharp.qoi import TopKQoI

# Set up some envrionment variables
RNG_SEED = 123
N_SAMPLES = 50
rng = check_random_state(RNG_SEED)


def score_function(X):
    return 0.5 * X[:, 0] + 0.5 * X[:, 1]


X = np.concatenate(
    [rng.normal(size=(N_SAMPLES, 1)), rng.binomial(1, 0.5, size=(N_SAMPLES, 1))], axis=1
)
y = score_function(X)
rank = rankdata(-y, method="ordinal").astype(int)


@pytest.mark.parametrize("top_k", range(0, X.shape[0], 5))
def test_top_k(top_k):
    qoi = TopKQoI(target_function=score_function, top_k=top_k, X=X)
    np.testing.assert_allclose(qoi.estimate(X), (rank <= top_k).astype(int))

