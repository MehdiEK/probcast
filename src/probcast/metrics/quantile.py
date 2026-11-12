from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from probcast._validation import check_quantile_inputs


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