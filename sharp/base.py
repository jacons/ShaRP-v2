"""
Base object used to set up explainability objects.
"""

import numbers
import numpy as np
from itertools import product
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from .utils._parallelize import parallel_loop
from .utils import (
    check_feature_names,
    check_inputs,
    check_measure,
    check_qoi,
)


def _drop_row(X, row):
    """Remove the first row of ``X`` exactly equal to ``row``; ``X`` if none."""
    match = (X == row).all(axis=1)
    return np.delete(X, match.argmax(), axis=0) if match.any() else X


class ShaRP(BaseEstimator):
    """
    The ShaRP (Shapley for Rankings and Preferences) class provides a novel framework
    for explaining the contributions of features to various aspects of ranked
    outcomes. Built on Shapley values, it quantifies feature importance for rankings,
    which is fundamentally different from feature importance in classification or
    regression. This framework is essential for understanding, auditing,
    and improving algorithmic ranking systems in critical domains such as
    hiring, education, and lending.

    ShaRP extends the Quantitative Input Influence (QII) framework to compute feature
    contributions to multiple ranking-specific Quantities of Interest (QoIs).
    These QoIs include:
    - Rank: Impact of features on an item's rank.
    - Top-k: Influence of features on whether an item appears in the top-k positions.
    - Rank score: Contribution of features to an item's score.
    - Pairwise preference: obtained by calling :meth:`pairwise` with two items.

    ShaRP uses Shapley values, a cooperative game theory concept, to distribute
    the "value" of a ranked outcome among the features. For each QoI, the class:
    - Constructs feature coalitions by masking subsets of features.
    - Evaluates the impact of these coalitions on the QoI using a payoff function.
    - Aggregates the marginal contributions of features across all possible coalitions
    to compute their Shapley values.

    This algorithm is an implementation of Shapley for Rankings and Preferences (ShaRP),
    as presented in [1]_.

    Parameters
    ----------
    qoi : str or QoI object, optional
        The quantity of interest to compute feature contributions for. Options:
        - "rank" : Contribution to an item's rank.
        - "top_k" : Contribution to whether an item appears in the top-k
          (``top_k`` can be passed as an extra keyword argument, default 10).
        - "rank_score" : Contribution to an item's score.
        - a QoI object (see :mod:`sharp.qoi`), used as is.
        If None, "rank" is used in ``fit()``.

    target_function : function
        Function mapping an array of shape (n_samples, n_features) to a 1D array of
        scores. Required when ``qoi`` is a string.

    measure : str, default="shapley"
        The method used to compute feature contributions. Options include
        "shapley", None (same as "shapley") or a callable.

    sample_size : int, optional
        The number of perturbations to apply per data point when calculating
        feature importance. Default is `None`, which uses all available samples
        (under ``semantics="paper"``, all donors left after removing the explained
        item).

    coalition_size : int, optional
        The maximum size of feature coalitions to consider. Default is `None`,
        which uses all features except one. A smaller value is an approximation:
        the Shapley weights are not renormalized over the coalitions that are kept,
        so the efficiency property (contributions summing to the QoI difference)
        does not hold.

    replace : bool, default=False
        Whether to sample feature values with replacement during perturbation.

    semantics : {"paper", "legacy"}, default="paper"
        ``"paper"`` follows the ShaRP paper: the explained item is removed from the
        donor pool (one occurrence, if present) and from the ranking reference, the
        hybrid's rank is a competition rank (1 + number of reference items with a
        strictly higher score), and ``sample_size=None`` uses the whole pool.
        Integer indices (``individual(3)``) refer to the row order of the data as
        passed by the user. ``"legacy"`` reproduces the behaviour of earlier
        versions: no exclusion, rank = number of reference items with score >= the
        hybrid's, and, because the reference data is sorted by score, integer indices
        refer to the score-sorted order.

        Under ``"paper"`` every (item, feature) task draws from its own seed derived
        from ``random_state``, so results do not depend on ``n_jobs``. Under
        ``"legacy"`` all tasks share one generator and ``n_jobs>1`` keeps the old
        behaviour, so results can differ from ``n_jobs=1``.

    random_state : int, RandomState instance, or None, optional
        Seed or random number generator for reproducibility. Default is `None`.

    n_jobs : int, default=1
        Number of jobs to run in parallel for computations. Use `-1` to use all
        available processors.

    verbose : int, default=0
        Verbosity level. Use 0 for no output and higher numbers for more verbose output.

    kwargs : dict, optional
        Additional parameters such as:
        - ``X`` : array-like, reference input data.
        - ``y`` : array-like, target outcomes for the reference data.

    Notes
    -----
    See the original paper: [1]_ for more details.

    References
    ----------
    .. [1] V. Pliatsika, J. Fonseca, T. Wang, J. Stoyanovich, "ShaRP: Explaining
       Rankings with Shapley Values", arXiv:2401.16744.
    """

    def __init__(
        self,
        qoi=None,
        target_function=None,
        measure="shapley",
        sample_size=None,
        coalition_size=None,
        replace=False,
        random_state=None,
        cache=True,
        n_jobs=1,
        verbose=0,
        semantics="paper",
        **kwargs
    ):
        self.qoi = qoi
        self.target_function = target_function
        self.measure = measure
        self.sample_size = sample_size
        self.coalition_size = coalition_size
        self.replace = replace
        self.semantics = semantics
        self.random_state = random_state
        self.cache = cache
        self.n_jobs = n_jobs
        self.verbose = verbose

        if "X" in kwargs.keys():
            self._X = kwargs["X"]
            kwargs.pop("X")

        if "y" in kwargs.keys():
            self._y = kwargs["y"]
            kwargs.pop("y")

        self.qoi_kwargs = kwargs if kwargs is not None else {}

    def fit(self, X, y=None, feature_names=None):
        """
        Fit a ShaRP model to the given data.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Reference dataset used to compute explanations.
        y: array-like, shape (n_samples,), default=None
            Target variable.
        feature_names: array-like, shape (n_features,), default=None
            Names of features in X.
        """
        if self.semantics not in ("paper", "legacy"):
            raise ValueError(
                f"semantics must be 'paper' or 'legacy', got {self.semantics!r}."
            )

        X_, y_ = check_inputs(X, y)

        self._X = X_
        self._y = y_

        self._rng = check_random_state(self.random_state)

        qoi = self.qoi if self.qoi is not None else "rank"

        self.qoi_ = check_qoi(
            qoi,
            target_function=self.target_function,
            X=X_,
            cache=self.cache,
            semantics=self.semantics,
            **self.qoi_kwargs,
        )

        self.feature_names_ = check_feature_names(X, feature_names=feature_names)

        self.measure_ = check_measure(self.measure)

    def _donors(self, pool, row, sample_size, n_default):
        """Donor pool, row to exclude from rank references, and sample size."""
        if self.semantics == "legacy":
            return pool, None, n_default if sample_size is None else sample_size
        pool = _drop_row(pool, row)
        if sample_size is None:
            sample_size = len(pool)
        elif not self.replace:
            sample_size = min(sample_size, len(pool))
        return pool, row, sample_size

    def _task_rngs(self, n_tasks, seed=None):
        """Task -> rng: the shared one (legacy) or a seeded one (paper).

        Only int seeds are held, so closures stay cheap to ship to workers.
        """
        if self.semantics == "legacy":
            return lambda t: self._rng
        rng = self._rng if seed is None else np.random.RandomState(seed)
        seeds = rng.randint(np.iinfo(np.int32).max, size=n_tasks)
        return lambda t: np.random.RandomState(seeds[t])

    def individual(self, sample, X=None, y=None, **kwargs):
        """
        Provides an explanation for individual sample point based on reference dataset

        Parameters
        ----------
        sample : array-like, shape (n_features,) or int
            Sample to calculate explanation for.
            Can be passed directly or as an index in a reference dataset.
        X : array-like, shape (n_samples, n_features), default=None
            Reference dataset used to compute explanations.
        y : array-like, shape (n_samples,), default=None
            Target variable.
        set_cols_idx : 1D array-like, default=None
            Features in the coalition used to construct composite points to estimate
            feature importance.
        coalition_size : int, default=n_features-1
            Maximum number of features used during the construction of composite points.
        sample_size : int, default=n_samples
            Maximum number of samples used during the construction of composite points.

        Returns
        -------
        1D array-like, shape (n_features,)
            Influences of each feature on individual sample.
        """
        if X is None:
            X = self.qoi_.X

        X_, y_ = check_inputs(X, y)

        if "set_cols_idx" in kwargs.keys():
            set_cols_idx = kwargs["set_cols_idx"]
        else:
            set_cols_idx = None

        if "coalition_size" in kwargs.keys():
            coalition_size = kwargs["coalition_size"]
        elif self.coalition_size is not None:
            coalition_size = self.coalition_size
        else:
            coalition_size = X_.shape[-1] - 1

        if isinstance(sample, numbers.Integral):
            sample = X_[sample]
        else:
            sample = np.asarray(sample)

        sample_size = kwargs.get("sample_size", self.sample_size)
        pool, exclude, sample_size = self._donors(X_, sample, sample_size, X_.shape[0])

        n_jobs = kwargs["n_jobs"] if "n_jobs" in kwargs.keys() else self.n_jobs
        verbosity = kwargs["verbose"] if "verbose" in kwargs.keys() else self.verbose
        n_features = len(self.feature_names_)
        rngs = self._task_rngs(n_features, kwargs.get("_seed"))
        influences = parallel_loop(
            lambda col_idx: self.measure_(
                row=sample,
                col_idx=col_idx,
                set_cols_idx=set_cols_idx,
                X=pool,
                qoi=self.qoi_,
                sample_size=sample_size,
                coalition_size=coalition_size,
                replace=self.replace,
                rng=rngs(col_idx),
                exclude=exclude,
            ),
            range(n_features),
            n_jobs=n_jobs,
            progress_bar=verbosity,
        )

        return np.asarray(influences)

    def feature(self, feature, X=None, y=None, **kwargs):
        """
        Provides an explanation for all sample points for a specified feature
        based on reference dataset

        Parameters
        ----------
        feature : str or int
            Name or index of the targeted feature
        X : array-like, shape (n_samples, n_features), default=None
            Reference dataset used to compute explanations.
        y : array-like, shape (n_samples,), default=None
            Target variable.
        set_cols_idx : 1D array-like, default=None
            Features in the coalition used to construct composite points to estimate
            feature importance.
        coalition_size : int, default=n_features-1
            Maximum number of features used during the construction of composite points.
        sample_size : int, default=n_samples
            Maximum number of samples used during the construction of composite points.

        Returns
        -------
        float
            Average contributions of a specific feature along all sample points.
        """
        if X is None:
            X = self.qoi_.X

        X_, y_ = check_inputs(X, y)

        col_idx = (
            list(self.feature_names_).index(feature) if type(feature) is str else feature
        )

        if "set_cols_idx" in kwargs.keys():
            set_cols_idx = kwargs["set_cols_idx"]
        else:
            set_cols_idx = None

        if "coalition_size" in kwargs.keys():
            coalition_size = kwargs["coalition_size"]
        elif self.coalition_size is not None:
            coalition_size = self.coalition_size
        else:
            coalition_size = X_.shape[1] - 1

        requested_size = kwargs.get("sample_size", self.sample_size)

        influences = []
        rngs = self._task_rngs(X_.shape[0])
        for sample_idx in range(X_.shape[0]):
            sample = X_[sample_idx]
            pool, exclude, sample_size = self._donors(
                X_, sample, requested_size, X_.shape[0]
            )
            cell_influence = self.measure_(
                row=sample,
                col_idx=col_idx,
                set_cols_idx=set_cols_idx,
                X=pool,
                qoi=self.qoi_,
                sample_size=sample_size,
                coalition_size=coalition_size,
                replace=self.replace,
                rng=rngs(sample_idx),
                exclude=exclude,
            )
            influences.append(cell_influence)

        return np.mean(influences)

    def all(self, X=None, y=None, **kwargs):
        """
        Provides an explanation for all sample points based on reference dataset

        Parameters
        ----------
        feature : str or int
            Name or index of the targeted feature
        X : array-like, shape (n_samples, n_features), default=None
            Reference dataset used to compute explanations.
        y : array-like, shape (n_samples,), default=None
            Target variable.
        set_cols_idx : 1D array-like, default=None
            Features in the coalition used to construct composite points to estimate
            feature importance.
        coalition_size : int, default=n_features-1
            Maximum number of features used during the construction of composite points.
        sample_size : int, default=n_samples
            Maximum number of samples used during the construction of composite points.

        Returns
        -------
        array-like, shape (n_samples, n_features)
            Contribution of each feature to a point's qoi
        """
        if X is None:
            X = self.qoi_.X

        X_, y_ = check_inputs(X, y)

        if "set_cols_idx" in kwargs.keys():
            set_cols_idx = kwargs["set_cols_idx"]
        else:
            set_cols_idx = None

        if "coalition_size" in kwargs.keys():
            coalition_size = kwargs["coalition_size"]
        elif self.coalition_size is not None:
            coalition_size = self.coalition_size
        else:
            coalition_size = X_.shape[-1] - 1

        requested_size = kwargs.get("sample_size", self.sample_size)

        idx_samples = range(X_.shape[0])
        idx_cols = range(X_.shape[1])

        tasks = list(product(idx_samples, idx_cols))
        rngs = self._task_rngs(len(tasks))

        def contribution(task):
            sample_col_idx = tasks[task]
            row = X_[sample_col_idx[0]]
            pool, exclude, sample_size = self._donors(
                self.qoi_.X, row, requested_size, X_.shape[0]
            )
            return (
                sample_col_idx,
                self.measure_(
                    row=row,
                    col_idx=sample_col_idx[1],
                    set_cols_idx=set_cols_idx,
                    X=pool,
                    qoi=self.qoi_,
                    sample_size=sample_size,
                    coalition_size=coalition_size,
                    replace=self.replace,
                    rng=rngs(task),
                    exclude=exclude,
                ),
            )

        influences = parallel_loop(
            contribution,
            range(len(tasks)),
            n_jobs=self.n_jobs,
            progress_bar=self.verbose,
        )
        inf_ = np.full(X_.shape, np.nan)
        for (row_idx, col_idx), contr_ in influences:
            inf_[row_idx, col_idx] = contr_
        return inf_

    def pairwise(self, sample1, sample2, **kwargs):
        """
        Compare two samples, or one sample against a set of samples.
        If `sample1` or `sample2` are of type `int` or `list`, `X` also needs
        to be passed.

        Parameters
        ----------
        sample1 : array-like or int or list
            Sample or indices of samples that are used to calculate contributions.
        sample2 : array-like or int or list
            Sample or indices of samples against which contributions are calculated.
        X : array-like, shape (n_samples, n_features), default=None
            Reference dataset used to compute explanations.
        y : array-like, shape (n_samples,), default=None
            Target variable.
        set_cols_idx : 1D array-like, default=None
            Features in the coalition used to construct composite points to estimate
            feature importance.
        coalition_size : int, default=n_features-1
            Maximum number of features used during the construction of composite points.
        sample_size : int, default=n_samples
            Maximum number of samples used during the construction of composite points.

        Returns
        -------
        array-like
            Contributions of each feature to each `sample1` point's qoi
        """
        if "X" in kwargs.keys():
            X = kwargs.pop("X")

            if isinstance(sample1, (numbers.Integral, list)):
                sample1 = X[sample1]

            if isinstance(sample2, (numbers.Integral, list)):
                sample2 = X[sample2]

        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        sample2 = sample2.reshape(1, -1) if sample2.ndim == 1 else sample2

        if "sample_size" in kwargs.keys():
            sample_size = kwargs.pop("sample_size")
        elif self.sample_size is not None:
            sample_size = (
                sample2.shape[0]
                if self.sample_size > sample2.shape[0]
                else self.sample_size
            )
        else:
            sample_size = sample2.shape[0]

        if "coalition_size" in kwargs.keys():
            coalition_size = kwargs.pop("coalition_size")
        elif self.coalition_size is not None:
            coalition_size = self.coalition_size
        else:
            coalition_size = sample1.shape[-1] - 1

        return self.individual(
            sample1,
            X=sample2,
            sample_size=sample_size,
            coalition_size=coalition_size,
            **kwargs,
        )

    def pairwise_set(self, samples1, samples2, **kwargs):
        """
        Pairwise comparison of two samples sets.

        .. note:: if elements of `samples1` or `samples2` are of type `int` or `list`,
            `X` also needs to be passed.

        Parameters
        ----------
        samples1 : array-like
            Set of samples or indices that are used to calculate contributions.
        samples2 : array-like
            Set of samples or indices against which contributions are calculated.
        X : array-like, shape (n_samples, n_features), default=None
            Reference dataset used to compute explanations.
        y : array-like, shape (n_samples,), default=None
            Target variable.
        set_cols_idx : 1D array-like, default=None
            Features in the coalition used to construct composite points to estimate
            feature importance.
        coalition_size : int, default=n_features-1
            Maximum number of features used during the construction of composite points.
        sample_size : int, default=n_samples
            Maximum number of samples used during the construction of composite points.

        Returns
        -------
        array-like
            Contributions for each sample from `samples1` against respective sample in
             `samples2`
        """
        pairs = list(zip(samples1, samples2))
        if self.semantics == "legacy":
            seed_kw = [{}] * len(pairs)
        else:
            seeds = self._rng.randint(np.iinfo(np.int32).max, size=len(pairs))
            seed_kw = [{"_seed": int(sd)} for sd in seeds]
        contributions = parallel_loop(
            lambda t: self.pairwise(*pairs[t], verbose=False, **kwargs, **seed_kw[t]),
            range(len(pairs)),
            n_jobs=self.n_jobs,
            progress_bar=self.verbose,
        )

        return np.array(contributions)
