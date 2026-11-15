from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import norm

from probcast.metrics.quantile import CoverageResult, pinball_loss, quantile_coverage


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


def test_quantile_coverage_returns_empirical_coverage_and_deviation():
    """Compute each quantile's share of observations at or below its forecast."""
    result = quantile_coverage(
        [1, 2, 3, 4],
        [[1, 0], [2, 2], [3, 4], [4, 5]],
        [0.25, 0.75],
    )

    assert isinstance(result, CoverageResult)
    np.testing.assert_array_equal(result.quantiles, [0.25, 0.75])
    np.testing.assert_allclose(result.empirical, [1.0, 0.75])
    np.testing.assert_allclose(result.deviation, [0.75, 0.0])
    assert result.n == 4
    assert result.ci_lower.shape == (2,)
    assert result.ci_upper.shape == (2,)
    with pytest.raises(FrozenInstanceError):
        result.n = 5


def test_quantile_coverage_strict_excludes_ties():
    """Use strict inequality only when requested."""
    y_true = [1, 2]
    y_pred = [[1], [3]]

    assert quantile_coverage(y_true, y_pred, [0.5]).empirical[0] == 1.0
    assert quantile_coverage(y_true, y_pred, [0.5], strict=True).empirical[0] == 0.5


def test_quantile_coverage_nan_policy_raise_and_omit():
    """Reject NaNs by default and omit complete rows when requested."""
    y_true = [1.0, np.nan, 3.0]
    y_pred = [[1.0, 0.0], [2.0, 3.0], [3.0, np.nan]]
    quantiles = [0.25, 0.75]

    with pytest.raises(ValueError, match="must not contain NaN"):
        quantile_coverage(y_true, y_pred, quantiles)

    result = quantile_coverage(y_true, y_pred, quantiles, nan_policy="omit")
    assert result.n == 1
    np.testing.assert_allclose(result.empirical, [1.0, 0.0])


def test_quantile_coverage_calibrated_normal_quantiles_are_in_wilson_intervals():
    """Contain target probabilities in Wilson intervals for calibrated data."""
    rng = np.random.default_rng(2026)
    quantiles = np.array([0.1, 0.5, 0.9])
    y_true = rng.standard_normal(20_000)
    y_pred = np.broadcast_to(norm.ppf(quantiles), (y_true.size, quantiles.size))

    result = quantile_coverage(y_true, y_pred, quantiles)

    assert result.n == y_true.size
    assert np.all(result.ci_lower <= quantiles)
    assert np.all(quantiles <= result.ci_upper)


def test_quantile_coverage_shifted_quantiles_have_positive_deviations():
    """Forecast quantiles shifted upward should be over-covered."""
    rng = np.random.default_rng(2026)
    quantiles = np.array([0.2, 0.5, 0.8])
    y_true = rng.standard_normal(10_000)
    shifted_quantiles = norm.ppf(quantiles) + 1.0
    y_pred = np.broadcast_to(shifted_quantiles, (y_true.size, quantiles.size))

    result = quantile_coverage(y_true, y_pred, quantiles)

    assert np.all(result.deviation > 0.1)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "quantiles", "message"),
    [
        ([1, 2], [[0]], [0.5], "y_pred must have shape"),
        ([1], [[0, 1]], [0.75, 0.25], "strictly increasing"),
        ([1], [[0]], [0.5], "confidence must be in"),
        ([1], [[0]], [0.5], "nan_policy must be"),
    ],
)
def test_quantile_coverage_rejects_invalid_inputs(y_true, y_pred, quantiles, message):
    """Raise clear validation errors for malformed inputs and options."""
    kwargs = {}
    if message == "confidence must be in":
        kwargs["confidence"] = 1.0
    elif message == "nan_policy must be":
        kwargs["nan_policy"] = "ignore"

    with pytest.raises(ValueError, match=message):
        quantile_coverage(y_true, y_pred, quantiles, **kwargs)


def test_quantile_coverage_raises_when_omission_removes_every_row():
    """Require at least one usable row after NaN omission."""
    with pytest.raises(ValueError, match="no observations remain"):
        quantile_coverage([np.nan], [[0]], [0.5], nan_policy="omit")
