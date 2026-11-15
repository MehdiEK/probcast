import numpy as np
from numpy.typing import ArrayLike, NDArray

from probcast._validation import check_interval_inputs


def _winkler_scores(
    y_true: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Return vectorized Winkler scores for validated interval inputs."""
    return (upper - lower) + (2.0 / alpha) * (
        np.maximum(lower - y_true, 0.0) + np.maximum(y_true - upper, 0.0)
    )


def mean_winkler_score(
	y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> float:
    """Compute the mean Winkler score for prediction intervals.

    For each observation ``y`` and interval ``[l, u]``, the score is

    ``(u - l) + (2 / alpha) * max(l - y, 0) + (2 / alpha) * max(y - u, 0)``.

    The score is averaged over all elements. Inputs may have any matching
    shape with at least one dimension; N-dimensional arrays are supported.
    Non-finite values are rejected.

    :param y_true: ArrayLike
        Observed values.
    :param lower: ArrayLike
        Lower bounds of the prediction intervals.
    :param upper: ArrayLike
        Upper bounds of the prediction intervals.
    :param alpha: float
        Nominal miscoverage in the open interval (0, 1).

    :return: Mean Winkler score as a float. Lower scores are better.

    :raises TypeError: If an array cannot be converted to float or alpha is
        not a real scalar.
    :raises ValueError: If the arrays have different shapes, are empty or
        non-finite, have invalid bounds, or alpha is outside (0, 1).

    Examples
    --------
    >>> mean_winkler_score([0, 2], [-1, 1], [1, 3], 0.1)
    2.0
    """
    y, lo, hi, alpha_value = check_interval_inputs(y_true, lower, upper, alpha)
    return float(_winkler_scores(y, lo, hi, alpha_value).mean())
