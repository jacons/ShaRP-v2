import copy
from .base import BaseQoI, BaseRankQoI


class RankQoI(BaseRankQoI):
    """
    Rank specific QoI. Uses rank as the quantity being measured. The influence score
    is based on the comparison between the rank of a sample and synthetic data (based on
    the original sample). ``target_function`` should output scores.

    Notes
    -----
    This QoI was formerly defined as ``RankingRank``.
    """

    def _estimate(self, rows, exclude=None):
        return self.rank(rows, exclude)

    def _calculate(self, rows1, rows2, exclude=None):
        return (self.estimate(rows2, exclude) - self.estimate(rows1, exclude)).mean()


class RankScoreQoI(BaseRankQoI):
    """
    Score QoI. Uses the score given by ``target_function`` as the quantity being
    measured; the influence is the mean score difference between the two sets of
    samples. ``X`` is still required, for API consistency with the other QoIs.

    Notes
    -----
    This QoI was formerly defined as ``RankingScore``.
    """

    def _estimate(self, rows, exclude=None):
        return self.target_function(rows)

    def _calculate(self, rows1, rows2, exclude=None):
        return (self.estimate(rows1, exclude) - self.estimate(rows2, exclude)).mean()


class TopKQoI(BaseRankQoI):
    """
    Rank-specific QoI. Estimates the likelihood of reaching the top-K as the
    quantity of interest.

    Parameters
    ----------
    top_k : int, default=10
        The number of items to consider as part of the top-ranked group.
    """

    def __init__(
        self, target_function=None, top_k=10, X=None, cache=True, semantics="paper"
    ):
        super().__init__(
            target_function=target_function, X=X, cache=cache, semantics=semantics
        )
        self.top_k = top_k

    def _estimate(self, rows, exclude=None):
        ranks = self.rank(rows, exclude)
        return (ranks <= self.top_k).astype(int)

    def _calculate(self, rows1, rows2, exclude=None):
        return (self.estimate(rows1, exclude) - self.estimate(rows2, exclude)).mean()


_QOI_OBJECTS = {
    "rank": RankQoI,
    "rank_score": RankScoreQoI,
    "top_k": TopKQoI,
}


def get_qoi_names():
    """Get the names of all available quantities of interest.

    These names can be passed to :func:`~sharp.qoi.get_qoi` to
    retrieve the QoI object.

    Returns
    -------
    list of str
        Names of all available quantities of interest.

    Examples
    --------
    >>> from sharp.qoi import get_qoi_names
    >>> all_qois = get_qoi_names()
    >>> type(all_qois)
    <class 'list'>
    >>> all_qois
    ['rank', 'rank_score', 'top_k']
    """
    return sorted(_QOI_OBJECTS.keys())


def get_qoi(qoi):
    """Get a quantity of interest from string.

    :func:`~sharp.qoi.get_qoi_names` can be used to retrieve the names
    of all available quantities of interest.

    Parameters
    ----------
    qoi : str or QoI object
        Quantity of interest as string (one of :func:`get_qoi_names`). Anything
        else is returned as is.

    Returns
    -------
    quantity : callable
        The quantity of interest.

    Notes
    -----
    When passed a string, this function returns a copy of the QoI class
    (not an instance), so it can be instantiated with the QoI's parameters.
    """
    if isinstance(qoi, str):
        try:
            quantity = copy.deepcopy(_QOI_OBJECTS[qoi])
        except KeyError:
            raise ValueError(
                "%r is not a valid QoI name. Valid names are: %s."
                % (qoi, ", ".join(get_qoi_names()))
            )
    else:
        quantity = qoi
    return quantity
