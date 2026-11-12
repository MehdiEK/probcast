import numpy as np


def check_quantile_inputs(y_true, y_pred, quantiles):
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