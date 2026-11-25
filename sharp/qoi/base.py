from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator


class TargetFunctionWrapper:
    """Exact in-process cache keyed by row values; one model call per batch.

    Rows containing NaN never hit the cache (NaN != NaN as a dict key), so they are
    recomputed each time; values stay correct.
    """

    def __init__(self, target_function):
        self.target_function = target_function
        self._cache = {}
        self._dtype = None

    def __call__(self, rows):
        rows = np.asarray(rows)
        keys = [tuple(r.tolist()) for r in rows]
        new = {}  # first index of each uncached key, in order
        for i, k in enumerate(keys):
            if k not in self._cache:
                new.setdefault(k, i)
        if new:
            sub = rows[list(new.values())]
            out = np.asarray(self.target_function(sub))
            if out.size != len(sub):
                raise ValueError(
                    f"target_function returned {out.size} values for {len(sub)} "
                    "rows; expected one score per row."
                )
            self._dtype = out.dtype
            self._cache.update(zip(new, out.reshape(-1)))
        return np.array([self._cache[k] for k in keys], dtype=self._dtype)


class BaseQoI(BaseEstimator, metaclass=ABCMeta):
    """
    Base class to implement Quantities of Interest (QoI).

    It should not be used directly. Any QoI must define at least 2 functions:

      :func:`qoi.estimate`:
          A function that predicts the model's result for one or multiple items.
      :func:`qoi.calculate`:
          A function that applies a difference metric to the predictions of two items, or
          two sets of items. If sets, then the mean difference is returned.

    Parameters
    ----------
    target_function : function
        Method used to calculate a score. The output of this function
        should be a 1-dimensional array with one score for each of the passed
        observations.

    X : array-like of shape (n_samples, n_features), default=None
        Reference dataset (required by rank-based QoIs).

    cache : bool, default=True
        Whether to cache ``target_function`` outputs, keyed by row values.

    semantics : {"paper", "legacy"}, default="paper"
        See :class:`sharp.ShaRP`.
    """

    _qoi_type = "score"

    def __init__(self, target_function=None, X=None, cache=True, semantics="paper"):
        self.target_function = (
            TargetFunctionWrapper(target_function) if cache else target_function
        )
        self.cache = cache
        self.X = X
        self.semantics = semantics

    def estimate(self, rows, exclude=None):
        """
        Prepares and runs ``self.target_function`` for a set of samples.

        Parameters
        ----------
        rows : array-like of shape (n_samples, n_features)
            Samples over which ``target_function`` will be applied.

        exclude : array-like of shape (n_features,), default=None
            The explained item; only used by rank-based QoIs (see
            :meth:`BaseRankQoI.rank`).

        Returns
        -------
        target_output : np.ndarray of shape (n_samples,)
            Score or rank estimations.
        """
        return self._estimate(rows, **({} if exclude is None else {"exclude": exclude}))

    def calculate(self, rows1, rows2, exclude=None):
        """
        Calculates the influence score based on the ``target_function`` outputs for
        ``rows1`` and ``rows2``.

        Parameters
        ----------
        rows1 : array-like of shape (n_samples, n_features)
            First set of samples to compare.

        rows2 : array-like of shape (n_samples, n_features)
            Second set of samples to compare.

        exclude : array-like of shape (n_features,), default=None
            The explained item; only used by rank-based QoIs.

        Returns
        -------
        influence_score : float
            Influence score for ``rows1``, compared to ``rows2``.
        """
        kw = {} if exclude is None else {"exclude": exclude}
        return self._calculate(rows1, rows2, **kw)

    @abstractmethod
    def _estimate(self, rows, exclude=None):
        pass

    @abstractmethod
    def _calculate(self, rows1, rows2, exclude=None):
        pass


class BaseRankQoI(BaseQoI, metaclass=ABCMeta):
    """
    Base class to implement Quantities of Interest (QoI) for ranking tasks. It should not
    be used directly. Any QoI must define at least 2 functions:

      :func:`qoi.estimate`:
          A function that predicts the model's result for one or multiple items.
      :func:`qoi.calculate`:
          A function that applies a difference metric to the predictions of two items, or
          two sets of items. If sets, then the mean difference is returned.

    Parameters
    ----------
    target_function : function
        Method used to calculate a score. The output of this function
        should be a 1-dimensional array with one score for each of the passed
        observations.

    X : array-like of shape (n_samples, n_features)
        The input samples.
    """

    _qoi_type = "rank"
    _X_sorted = False

    def sort_base(self):
        scores = self.target_function(self.X)
        idx_sorted = np.argsort(scores)
        if self.semantics == "legacy":
            self.X = self.X[idx_sorted]
        self._scores_sorted = scores[idx_sorted]
        self._X_sorted = True
        self._s_ex = {}
        return self

    def rank(self, X, exclude=None):
        """
        Computes the ranking of samples ``X`` within the reference dataset ``self.X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to calculate the rankings for.

        exclude : array-like of shape (n_features,), default=None
            The explained item v. Only used when ``semantics="paper"``: if ``self.X``
            contains a row equal to v, v is removed from the reference dataset before
            ranking. Ignored under ``"legacy"``.

        Returns
        -------
        rankings : np.ndarray of shape (n_samples,)
            Rankings for all samples in ``X``. Rank 1 is the highest score.
            ``"legacy"``: the number of reference items with score >= the sample's
            (ties share the worst rank; v is counted). ``"paper"``: competition rank,
            1 + the number of reference items (without v) with score strictly greater
            than the sample's.
        """
        if self.X is None:
            raise ValueError(
                "``self.X`` must be defined (in the object initialization)."
            )

        if not self._X_sorted:
            self.sort_base()

        scores_base = self._scores_sorted
        scores = self.target_function(X)
        if self.semantics == "legacy":
            return scores_base.shape[0] - np.searchsorted(
                scores_base, scores, side="left"
            )
        # paper: competition ranks against D minus the explained item
        ranks = 1 + scores_base.shape[0] - np.searchsorted(
            scores_base, scores, side="right"
        )
        if exclude is not None:
            s_ex = self._excluded_score(np.asarray(exclude))
            if s_ex is not None:
                ranks = ranks - (s_ex > scores)
        return ranks

    def _excluded_score(self, exclude):
        """f(exclude) if ``exclude`` is a row of ``self.X`` else None (memoized)."""
        key = tuple(exclude.tolist())
        if key not in self._s_ex:
            in_X = (np.asarray(self.X) == exclude).all(axis=1).any()
            self._s_ex[key] = (
                self.target_function(exclude[None, :])[0] if in_X else None
            )
        return self._s_ex[key]
