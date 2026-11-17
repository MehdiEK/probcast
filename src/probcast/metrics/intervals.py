import numpy as np
from numpy.typing import ArrayLike, NDArray

from probcast._validation import (
    check_interval_bounds,
    check_interval_inputs,
    check_interval_inputs_with_alpha,
)


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
    y, lo, hi, alpha_value = check_interval_inputs_with_alpha(
        y_true, lower, upper, alpha
    )
    return float(_winkler_scores(y, lo, hi, alpha_value).mean())


def interval_coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Compute the empirical coverage of prediction intervals.

    For each observation ``y`` and interval ``[l, u]``, coverage is

    ``1`` when ``l <= y <= u`` and ``0`` otherwise.

    Coverage is averaged over all elements. Inputs may have any matching
    shape with at least one dimension; N-dimensional arrays are supported.
    Array-like inputs are converted to float arrays, and non-finite values are
    rejected.

    :param y_true: ArrayLike
        Observed values.
    :param lower: ArrayLike
        Lower bounds of the prediction intervals.
    :param upper: ArrayLike
        Upper bounds of the prediction intervals.

    :return: Empirical interval coverage as a float in [0, 1].

    :raises TypeError: If an input cannot be converted to a float array.
    :raises ValueError: If the arrays have different shapes, are empty or
        non-finite, or contain invalid bounds.

    Examples
    --------
    >>> interval_coverage([0, 2], [-1, 1], [1, 3])
    1.0
    """
    y, lo, hi = check_interval_inputs(y_true, lower, upper)
    return float(np.mean((lo <= y) & (y <= hi)))


def average_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Compute the mean width of prediction intervals.

    For each interval ``[l, u]``, its width is

    ``u - l``.

    Width is averaged over all elements. Inputs may have any matching shape
    with at least one dimension; N-dimensional arrays are supported.
    Array-like bounds are converted to float arrays; non-finite values and
    invalid intervals are rejected.

    :param lower: ArrayLike
        Lower bounds of the prediction intervals.
    :param upper: ArrayLike
        Upper bounds of the prediction intervals.

    :return: Mean interval width as a non-negative float.

    :raises TypeError: If a bound cannot be converted to a float array.
    :raises ValueError: If the bounds have different shapes, are empty or
        non-finite, or contain invalid intervals.
    :raises OverflowError: If an interval width cannot be represented as a
        finite float.

    Examples
    --------
    >>> average_width([-1, 1], [1, 3])
    2.0
    """
    lo, hi = check_interval_bounds(lower, upper)
    with np.errstate(over="ignore"):
        widths = hi - lo
    if not np.all(np.isfinite(widths)):
        raise OverflowError("interval width is not representable as a finite float")

    max_width = float(np.max(widths))
    if max_width == 0.0:
        return 0.0
    return float(max_width * np.mean(widths / max_width))
