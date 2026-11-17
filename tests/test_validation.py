import numpy as np
import pytest

from probcast._validation import (
    check_interval_bounds,
    check_interval_inputs,
    check_interval_inputs_with_alpha,
    check_quantile_inputs,
)


def test_check_quantile_inputs_converts_inputs_to_float_arrays():
    """Convert valid list inputs to float NumPy arrays without changing values."""
    y, predictions, quantiles = check_quantile_inputs(
        [1, 3], [[0, 2], [4, 1]], [0.25, 0.75]
    )

    np.testing.assert_array_equal(y, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(predictions, np.array([[0.0, 2.0], [4.0, 1.0]]))
    np.testing.assert_array_equal(quantiles, np.array([0.25, 0.75]))
    assert y.dtype == predictions.dtype == quantiles.dtype == np.dtype(float)


def test_check_quantile_inputs_rejects_non_1d_observations():
    """Require observed values to be a one-dimensional array."""
    with pytest.raises(ValueError, match="y_true must be 1-D"):
        check_quantile_inputs([[1, 2]], [[0], [1]], [0.5])


@pytest.mark.parametrize("predictions", [[[0], [1], [2]], [0, 1]])
def test_check_quantile_inputs_rejects_prediction_shape_mismatch(predictions):
    """Require predictions to have one row per observation and column per quantile."""
    with pytest.raises(ValueError, match="y_pred must have shape"):
        check_quantile_inputs([1, 2], predictions, [0.5])


@pytest.mark.parametrize(
    "quantiles",
    [
        [0.0],
        [1.0],
        [-0.1],
        [1.1],
        [0.75, 0.25],
        [0.5, 0.5],
        [np.nan],
    ],
)
def test_check_quantile_inputs_rejects_invalid_quantile_levels(quantiles):
    """Require levels to be finite, strictly increasing, and between zero and one."""
    with pytest.raises(ValueError, match="quantiles must be strictly increasing"):
        check_quantile_inputs([1], np.zeros((1, len(quantiles))), quantiles)


def test_check_quantile_inputs_rejects_non_1d_quantiles():
    """Require quantile levels to be one-dimensional, as documented."""
    with pytest.raises(ValueError, match="quantiles must be 1-D"):
        check_quantile_inputs([1, 2], [[0], [1]], [[0.5]])


def test_check_interval_bounds_converts_inputs():
    """Convert matching interval bounds to float arrays."""
    lower, upper = check_interval_bounds((0, 1), [2, 3])

    np.testing.assert_array_equal(lower, [0.0, 1.0])
    np.testing.assert_array_equal(upper, [2.0, 3.0])
    assert lower.dtype == upper.dtype == np.dtype(float)


def test_check_interval_bounds_supports_matching_nd_arrays():
    """Accept matching bounds with more than one dimension."""
    arrays = np.ones((2, 2))

    lower, upper = check_interval_bounds(arrays, arrays)

    assert lower.shape == upper.shape == (2, 2)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        ([0, 1], [2], "identical shapes"),
        ([], [], "must not be empty"),
        (1, 1, "at least 1 dimension"),
        ([np.inf], [1], "lower must contain only finite values"),
        ([0], [-np.inf], "upper must contain only finite values"),
        ([2], [1], "lower must be less than or equal to upper"),
    ],
)
def test_check_interval_bounds_rejects_invalid_inputs(lower, upper, message):
    """Reject shape, emptiness, finiteness, and interval-order violations."""
    with pytest.raises(ValueError, match=message):
        check_interval_bounds(lower, upper)


def test_check_interval_bounds_names_non_numeric_array_input():
    """Report which bound cannot be converted to a float array."""
    with pytest.raises(TypeError, match="lower must be convertible to a float array"):
        check_interval_bounds(["not numeric"], [1])


def test_check_interval_inputs_converts_actuals_and_bounds():
    """Convert valid actuals and bounds to float arrays."""
    y, lower, upper = check_interval_inputs([1, 2], (0, 1), [2, 3])

    np.testing.assert_array_equal(y, [1.0, 2.0])
    np.testing.assert_array_equal(lower, [0.0, 1.0])
    np.testing.assert_array_equal(upper, [2.0, 3.0])
    assert y.dtype == lower.dtype == upper.dtype == np.dtype(float)


def test_check_interval_inputs_supports_matching_nd_arrays():
    """Accept matching actuals and bounds with more than one dimension."""
    arrays = np.ones((2, 2))

    y, lower, upper = check_interval_inputs(arrays, arrays, arrays)

    assert y.shape == lower.shape == upper.shape == (2, 2)


@pytest.mark.parametrize(
    ("y_true", "lower", "upper", "message"),
    [
        ([1, 2], [0], [2, 3], "identical shapes"),
        ([], [], [], "must not be empty"),
        (1, 1, 1, "at least 1 dimension"),
        ([np.nan], [0], [1], "y_true must contain only finite values"),
    ],
)
def test_check_interval_inputs_rejects_invalid_inputs(y_true, lower, upper, message):
    """Reject actual and interval shape, emptiness, and finiteness violations."""
    with pytest.raises(ValueError, match=message):
        check_interval_inputs(y_true, lower, upper)


def test_check_interval_inputs_names_non_numeric_actual_input():
    """Report when actuals cannot be converted to a float array."""
    with pytest.raises(TypeError, match="y_true must be convertible to a float array"):
        check_interval_inputs(["not numeric"], [0], [1])


def test_check_interval_inputs_with_alpha_converts_inputs_and_alpha():
    """Convert valid interval inputs to float arrays and alpha to float."""
    y, lower, upper, alpha = check_interval_inputs_with_alpha(
        [1, 2], (0, 1), [2, 3], 0.1
    )

    np.testing.assert_array_equal(y, [1.0, 2.0])
    np.testing.assert_array_equal(lower, [0.0, 1.0])
    np.testing.assert_array_equal(upper, [2.0, 3.0])
    assert y.dtype == lower.dtype == upper.dtype == np.dtype(float)
    assert alpha == 0.1


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_check_interval_inputs_with_alpha_rejects_invalid_alpha_values(alpha):
    """Require finite nominal miscoverage strictly between zero and one."""
    with pytest.raises(ValueError, match="alpha must be finite and in \\(0, 1\\)"):
        check_interval_inputs_with_alpha([0], [0], [1], alpha)


@pytest.mark.parametrize("alpha", ["0.1", [0.1], True, None])
def test_check_interval_inputs_with_alpha_rejects_non_real_scalar_alpha(alpha):
    """Require alpha to be a real scalar rather than a coercible value."""
    with pytest.raises(TypeError, match="alpha must be a real scalar"):
        check_interval_inputs_with_alpha([0], [0], [1], alpha)


def test_check_interval_inputs_with_alpha_rejects_invalid_array_inputs():
    """Propagate input validation errors before validating alpha."""
    with pytest.raises(ValueError, match="lower must be less than or equal to upper"):
        check_interval_inputs_with_alpha([0], [2], [1], 0.1)
