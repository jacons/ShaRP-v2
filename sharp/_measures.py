"""
Quantitative Input Influence measures.
"""

import numpy as np
from math import comb
from itertools import combinations


def _marginal(
    row, col_idx, set_cols_idx, X, qoi, sample_size, replace, rng, exclude=None, **kwargs
):
    """
    Marginal QII of one feature of one row, for a given coalition of other features.

    Parameters
    ----------
    row : np.ndarray of shape (n_features,)
        The item being explained.
    col_idx : int
        Index of the feature being explained.
    set_cols_idx : sequence of int
        Indices of the features in the coalition.
    X : np.ndarray of shape (n_donors, n_features)
        Donor pool the hybrid values are drawn from.
    qoi : BaseQoI
        Quantity of interest; ``qoi.calculate`` scores the two hybrid sets.
    sample_size : int
        Number of donors drawn.
    replace : bool
        Whether donors are drawn with replacement.
    rng : np.random.RandomState
        Random number generator.
    exclude : np.ndarray, default=None
        The explained item, forwarded to ``qoi.calculate`` (see ``BaseRankQoI.rank``).

    Returns
    -------
    float
        Marginal influence of ``col_idx`` given the coalition.
    """
    # Draw new samples uniformly at random
    X_sampled = X[rng.choice(np.arange(X.shape[0]), size=sample_size, replace=replace)]

    # Marginal approach
    X_modded1 = np.repeat(row.reshape(1, -1), repeats=sample_size, axis=0)

    # For X_modded1, keep:
    # - ``X_sampled`` values for the columns in ``set_cols_idx``
    X_modded1[:, set_cols_idx] = X_sampled[:, set_cols_idx]

    # For X_modded2, keep:
    # - ``X_sampled`` values for the columns in ``set_cols_idx`` and ``col_idx``
    X_modded2 = X_modded1.copy()
    X_modded2[:, col_idx] = X_sampled[:, col_idx]

    # Return score
    kw = {} if exclude is None else {"exclude": exclude}
    return qoi.calculate(X_modded1, X_modded2, **kw)


def _shapley(
    row,
    col_idx,
    X,
    qoi,
    sample_size,
    coalition_size,
    replace,
    rng,
    exclude=None,
    **kwargs,
):
    """
    Shapley value of one feature of one row, averaging marginal QII over coalitions
    of the other features of size ``0..coalition_size``.

    Parameters
    ----------
    row, col_idx, X, qoi, sample_size, replace, rng, exclude
        As in :func:`_marginal`.
    coalition_size : int
        Largest coalition size considered. Weights are not renormalized, so if
        smaller than ``n_features - 1`` the result is an approximation for which
        efficiency does not hold.

    Returns
    -------
    float
        Shapley value of ``col_idx``.
    """

    # Get indices for all columns except the one being explained
    rest_cols_idx = np.arange(X.shape[1])
    rest_cols_idx = rest_cols_idx[rest_cols_idx != col_idx]

    # Set up variable to track the total score for the specific attribute
    total_score = 0

    # Calculate the marginal score of every combination for ``col_idx`` vs rest
    iterable = [
        set_cols_idx
        for set_size in range(0, coalition_size + 1)
        for set_cols_idx in combinations(rest_cols_idx, set_size)
    ]

    for set_cols_idx in iterable:
        score = _marginal(
            row=row,
            col_idx=col_idx,
            set_cols_idx=set_cols_idx,
            X=X,
            qoi=qoi,
            sample_size=sample_size,
            replace=replace,
            rng=rng,
            **({} if exclude is None else {"exclude": exclude}),
        )
        total_score += score / (comb(X.shape[1] - 1, len(set_cols_idx)) * X.shape[1])

    return total_score


MEASURES = {"shapley": _shapley}
