from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def check_quantile_inputs(
    y_true: ArrayLike, y_pred: ArrayLike, quantiles: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Check the inputs for quantile loss functions.

    :param y_true: ArrayLike
        1D-Array of observed values of length n
    :param y_pred: ArrayLike
        1D or 2D-Array of forecasted quantiles (n, k)
    :param quantiles: ArrayLike
        1D-array of targets quantiles between 0 and 1

    :return: Tuple of (y, q_pred, taus) as np.ndarray
    """
    y = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(y_pred, dtype=float)
    taus = np.asarray(quantiles, dtype=float)

    if y.ndim != 1:
        raise ValueError("y_true must be 1-D")
    if taus.ndim != 1:
        raise ValueError("quantiles must be 1-D")
    if q_pred.shape != (y.shape[0], taus.shape[0]):
        raise ValueError(f"y_pred must have shape (n, k); got {q_pred.shape}")
    if not (np.all(taus > 0) and np.all(taus < 1) and np.all(np.diff(taus) > 0)):
        raise ValueError("quantiles must be strictly increasing and in (0, 1)")
    return y, q_pred, taus


def check_interval_inputs(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Validate observations, interval bounds, and nominal miscoverage."""
    arrays = []
    for name, values in (("y_true", y_true), ("lower", lower), ("upper", upper)):
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise TypeError(
                f"{name} must be convertible to a float array; got {values!r}"
            ) from None
        arrays.append(array)

    y, lo, hi = arrays
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            "y_true, lower, and upper must have identical shapes; "
            f"got {y.shape}, {lo.shape}, and {hi.shape}"
        )
    if y.ndim == 0:
        raise ValueError("y_true, lower, and upper must have at least 1 dimension")
    if y.size == 0:
        raise ValueError("y_true, lower, and upper must not be empty")

    for name, array in (("y_true", y), ("lower", lo), ("upper", hi)):
        non_finite = array[~np.isfinite(array)]
        if non_finite.size:
            raise ValueError(
                f"{name} must contain only finite values; found {non_finite}"
            )

    if np.any(lo > hi):
        invalid = np.argwhere(lo > hi)
        first_index = tuple(invalid[0])
        raise ValueError(
            "lower must be less than or equal to upper element-wise; "
            f"got lower={lo[first_index]} and upper={hi[first_index]} "
            f"at index {first_index}"
        )

    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError(f"alpha must be a real scalar; got {alpha!r}")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError(f"alpha must be finite and in (0, 1); got {alpha_value}")

    return y, lo, hi, alpha_value
