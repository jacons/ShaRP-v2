# ShaRP

``ShaRP`` is an open source library with the implementation of the ShaRP
algorithm (Shapley for Rankings and Preferences), a framework that can be used
to explain the contributions of features to different aspects of a ranked
outcome (rank, top-k membership, score, pairwise preference), based on Shapley
values.

## Installation

Python >= 3.10 is required. Runtime dependencies: numpy (>= 1.24),
scikit-learn (>= 1.2.0) and tqdm (>= 4.46.0).

    git clone https://github.com/DataResponsibly/sharp.git
    cd sharp
    pip install .

## Minimal example

```python
import numpy as np
from sharp import ShaRP

rng = np.random.RandomState(0)
X = rng.rand(50, 3)
weights = np.array([0.6, 0.3, 0.1])


def f(rows):
    return np.asarray(rows, dtype=float) @ weights  # higher score = better rank


sharp = ShaRP(qoi="rank", target_function=f, random_state=0)
sharp.fit(X)

print(sharp.individual(0))    # contribution of each feature to item 0's rank
print(sharp.all(X).shape)     # (50, 3): one row of contributions per item
```

Other QoIs are ``"top_k"`` (with ``top_k=10`` as extra keyword) and
``"rank_score"``. ``pairwise``, ``pairwise_set`` and ``feature`` explain
comparisons between items and a single feature averaged over items.

## Semantics

``ShaRP(..., semantics="paper")`` (default) follows the paper; ``"legacy"``
reproduces the behaviour of earlier versions.

|                         | ``legacy``                                        | ``paper``                                                                                                |
|-------------------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Rank of hybrid item u   | #{d in D : f(d) >= f(u)}                          | 1 + #{d in D \ {v} : f(d) > f(u)} (competition ranking, as in the paper's notebook)                     |
| Donor pool              | all of X                                          | X without **one** occurrence of v                                                                        |
| Top-k                   | legacy rank <= k                                  | paper rank <= k                                                                                          |
| Integer indices         | refer to the score-sorted reference data          | refer to the rows in the order passed by the user                                                        |
| Randomness with n_jobs  | shared generator; ``n_jobs>1`` can change results | one seed per (item, feature) task; results do not depend on ``n_jobs``                                   |

The explained item v is the ``sample`` in ``individual``/``pairwise`` and the
row in ``all``/``feature``.

**Two distinct sets.** The paper has a dataset D, over which the rank is
computed, and a reference set D', from which donors are drawn. In the code they
are the data given to ``fit`` and the ``X`` passed to the method (``sample2`` in
``pairwise``). Under ``paper`` the rule is applied to each set independently: if
the set contains a row exactly equal to v, its first occurrence is removed;
otherwise nothing is removed. For example, in ``pairwise(X[a], X[b])`` v = X[a]
is removed from the ranking set (it is in the ``fit`` data) while the donor pool
``{X[b]}`` is left untouched; in ``individual(row, X_other)`` v is removed from
the ``fit`` data and from ``X_other``, if present in them.

Under ``paper``, ``sample_size=None`` means the whole pool after the removal;
a ``sample_size`` larger than the pool, without ``replace``, is capped at the
pool size. ``coalition_size`` smaller than ``n_features - 1`` is an
approximation: weights are not renormalized, so efficiency does not hold.

## Citing ShaRP

If you use ``sharp`` in a scientific publication, we would appreciate citations to the following paper:

    @article{pliatsika2024sharp,
      title={ShaRP: Explaining Rankings with Shapley Values},
      author={Pliatsika, Venetia and Fonseca, Joao and Wang, Tilun and Stoyanovich, Julia},
      journal={arXiv preprint arXiv:2401.16744},
      year={2024}
    }
