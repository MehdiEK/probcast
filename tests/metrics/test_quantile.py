import numpy as np
import pytest

from probcast.metrics.quantile import pinball_loss


def test_pinball_loss_returns_elementwise_losses():
    """Return one loss for each observation and quantile level."""
    y_true = [1, 3]
    y_pred = [[0, 2], [4, 1]]
    quantiles = [0.25, 0.75]

    # Positive errors use tau; negative errors use 1 - tau.
    expected = np.array([[0.25, 0.25], [0.75, 1.5]])
    result = pinball_loss(y_true, y_pred, quantiles, reduction="none")

    np.testing.assert_allclose(result, expected)


def test_pinball_loss_reduces_across_observations_and_globally():
    """Return per-quantile and scalar means for the requested reduction."""
    y_true = [1, 3]
    y_pred = [[0, 2], [4, 1]]
    quantiles = [0.25, 0.75]

    np.testing.assert_allclose(
        pinball_loss(y_true, y_pred, quantiles),
        np.array([0.5, 0.875]),
    )
    assert pinball_loss(y_true, y_pred, quantiles, reduction="mean") == pytest.approx(
        0.6875
    )


def test_pinball_loss_rejects_unknown_reduction():
    """Raise a clear error for unsupported reduction names."""
    with pytest.raises(ValueError, match="Unknown reduction"):
        pinball_loss([1], [[0]], [0.5], reduction="sum")
