from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from probcast._validation import check_quantile_inputs


@dataclass(frozen=True)
class CoverageResult:
    """Empirical quantile coverage and Wilson confidence intervals."""

    quantiles: NDArray[np.float64]
    empirical: NDArray[np.float64]
    deviation: NDArray[np.float64]
    ci_lower: NDArray[np.float64]
    ci_upper: NDArray[np.float64]
    n: int


def pinball_loss(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantiles: ArrayLike,
    *,
    reduction: str = "per_quantile",  # "mean" | "none" | "per_quantile"
) -> float | NDArray[np.float64]:
    """
    Pinball (quantile) loss aggregated across levels or globally.

    :param y_true: ArrayLike
        1D-Array of observed values of length n
    :param y_pred: ArrayLike
        1D or 2D-Array of forecasted quantiles (n, k)
    :param quantiles: ArrayLike
        1D-array of targets quantiles between 0 and 1
    :param reduction: ArrayLike
        Aggregation method among "mean", "none" or "per_quantile

    :return NDArray of float
    """
    y, q_pred, taus = check_quantile_inputs(y_true, y_pred, quantiles)

    diff = y[:, None] - q_pred
    loss = np.maximum(taus * diff, (taus - 1.0) * diff)

    if reduction == "none":
        return loss
    if reduction == "per_quantile":
        return loss.mean(axis=0)
    if reduction == "mean":
        return float(loss.mean())
    raise ValueError(f"Unknown reduction: {reduction!r}")


def quantile_coverage(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantiles: ArrayLike,
    *,
    strict: bool = False,
    nan_policy: str = "raise",
    confidence: float = 0.95,
) -> CoverageResult:
    """
    Compute empirical coverage and Wilson intervals for forecast quantiles.

    :param y_true : array-like of shape (n,)
        Observed values.
    :param y_pred : array-like of shape (n, k)
        Predicted quantile values.
    :param quantiles : array-like of shape (k,)
        Strictly increasing quantile levels in (0, 1).
    :param strict : bool, default=False
        If True, count observations strictly below the predicted quantile.
        Otherwise, ties count as covered.
    :param nan_policy : {"raise", "omit"}, default="raise"
        Raise on NaN values, or omit every row containing a NaN.
    :param confidence : float, default=0.95
        Confidence level for the two-sided Wilson score intervals.

    :return CoverageResult:
        Quantile levels, empirical coverage, deviation from each target,
        Wilson interval bounds, and the number of observations used.

    :raises ValueError:
        If inputs have invalid shapes or quantile levels, the confidence or
        NaN policy is invalid, NaNs are present with ``nan_policy="raise"``,
        or no observations remain after omission.

    Examples
    --------
    >>> quantile_coverage([1, 2], [[1], [3]], [0.5]).empirical
    array([1.])
    """
    y, q_pred, taus = check_quantile_inputs(y_true, y_pred, quantiles)

    if nan_policy not in ("raise", "omit"):
        raise ValueError("nan_policy must be either 'raise' or 'omit'")

    # check confidence is a scalar in (0, 1)
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("confidence must be a scalar in (0, 1)") from None
    if not np.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    # follow the nan_policy: raise on NaN or omit rows with any NaN
    rows_with_nan = np.isnan(y) | np.isnan(q_pred).any(axis=1)
    if nan_policy == "raise" and rows_with_nan.any():
        raise ValueError("y_true and y_pred must not contain NaN values")
    if nan_policy == "omit":
        y = y[~rows_with_nan]
        q_pred = q_pred[~rows_with_nan]

    # check there are some obs left after applying the nan_policy
    n = y.shape[0]
    if n == 0:
        raise ValueError("no observations remain after applying nan_policy")

    # compute empirical coverage and deviation from target quantiles
    covered = y[:, None] < q_pred if strict else y[:, None] <= q_pred
    empirical = np.count_nonzero(covered, axis=0).astype(float) / n
    deviation = empirical - taus

    # compute Wilson score intervals for the empirical coverage
    z = float(norm.ppf(0.5 + confidence_value / 2.0))
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    center = empirical + z_squared / (2.0 * n)
    half_width = z * np.sqrt(
        empirical * (1.0 - empirical) / n + z_squared / (4.0 * n**2)
    )

    return CoverageResult(
        quantiles=taus.copy(),
        empirical=empirical,
        deviation=deviation,
        ci_lower=np.clip((center - half_width) / denominator, 0.0, 1.0),
        ci_upper=np.clip((center + half_width) / denominator, 0.0, 1.0),
        n=n,
    )
