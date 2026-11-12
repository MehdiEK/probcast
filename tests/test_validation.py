import numpy as np
import pytest

from probcast._validation import check_quantile_inputs


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