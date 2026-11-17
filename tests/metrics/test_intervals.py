import numpy as np
import pytest

from probcast.metrics.intervals import (
    average_width,
    interval_coverage,
    mean_winkler_score,
)
from probcast.metrics.quantile import pinball_loss


def test_mean_winkler_score_inside_interval_equals_mean_width():
    """With no misses, return the average interval width."""
    # Both observations are inside; each interval has width 2.
    assert mean_winkler_score([0, 2], [-1, 1], [1, 3], 0.1) == pytest.approx(2.0)


@pytest.mark.parametrize(("alpha", "expected"), [(0.1, 22.0), (0.2, 12.0)])
def test_mean_winkler_score_penalizes_misses_below_and_above(alpha, expected):
    """Apply the alpha-scaled penalty to misses on either side."""
    # Each width is 2 and each miss distance is 1: 2 + (2 / alpha) * 1.
    assert mean_winkler_score([-1, 4], [0, 1], [2, 3], alpha) == pytest.approx(expected)


def test_mean_winkler_score_matches_mixed_hand_computation():
    """Average inside, below, and above-interval scores."""
    # Scores are 2 + 10*1 = 12, 2, and 3 + 10*2 = 23; mean = 37/3.
    result = mean_winkler_score([0, 2, 6], [1, 1, 1], [3, 3, 4], 0.2)

    assert result == pytest.approx(37.0 / 3.0)


def test_mean_winkler_score_degenerate_intervals_have_penalty_only():
    """Give zero score to a covered point and only penalty to a miss."""
    # At y=1 the point interval has score 0; y=3 incurs (2/0.5)*(3-2)=4.
    assert mean_winkler_score([1, 3], [1, 2], [1, 2], 0.5) == pytest.approx(2.0)


def test_mean_winkler_score_matches_pinball_interval_identity():
    """Match the equivalent pair of lower and upper quantile losses."""
    y_true = np.array([-1.0, 0.5, 2.0, 5.0])
    lower = np.array([0.0, 0.0, 1.0, 2.0])
    upper = np.array([1.0, 2.0, 3.0, 4.0])
    alpha = 0.2

    lower_loss = pinball_loss(y_true, lower[:, None], [alpha / 2.0], reduction="none")
    upper_loss = pinball_loss(
        y_true, upper[:, None], [1.0 - alpha / 2.0], reduction="none"
    )
    expected = (2.0 / alpha) * (lower_loss + upper_loss).mean()

    assert mean_winkler_score(y_true, lower, upper, alpha) == pytest.approx(expected)


def test_mean_winkler_score_is_invariant_under_common_shift():
    """Adding a constant to all values preserves the score."""
    score = mean_winkler_score([0, 3], [1, 1], [2, 4], 0.2)
    shifted = mean_winkler_score([10, 13], [11, 11], [12, 14], 0.2)

    assert shifted == pytest.approx(score)


def test_mean_winkler_score_scales_with_positive_constant():
    """Multiplying observations and bounds scales the score equally."""
    score = mean_winkler_score([0, 3], [1, 1], [2, 4], 0.2)
    scaled = mean_winkler_score([0, 15], [5, 5], [10, 20], 0.2)

    assert scaled == pytest.approx(5.0 * score)


def test_mean_winkler_score_is_at_least_mean_width():
    """Equal the mean width without misses and exceed it when there are misses."""
    inside = mean_winkler_score([1, 2], [0, 1], [3, 4], 0.1)
    missed = mean_winkler_score([-1, 2], [0, 1], [3, 4], 0.1)
    mean_width = np.mean(np.array([3, 3]))

    assert inside == pytest.approx(mean_width)
    assert missed > mean_width


@pytest.mark.parametrize(
    "inputs",
    [
        ([0, 2], [-1, 1], [1, 3]),
        ((0, 2), (-1, 1), (1, 3)),
        (np.array([0, 2]), np.array([-1, 1]), np.array([1, 3])),
    ],
)
def test_mean_winkler_score_accepts_common_array_likes(inputs):
    """Accept lists, tuples, and NumPy arrays with identical results."""
    assert mean_winkler_score(*inputs, 0.1) == pytest.approx(2.0)


def test_mean_winkler_score_accepts_pandas_series():
    """Accept pandas Series when pandas is installed by the test environment."""
    pandas = pytest.importorskip("pandas")

    result = mean_winkler_score(
        pandas.Series([0, 2]), pandas.Series([-1, 1]), pandas.Series([1, 3]), 0.1
    )

    assert result == pytest.approx(2.0)


def test_mean_winkler_score_handles_single_observation():
    """Return the individual score when only one observation is supplied."""
    assert mean_winkler_score([3], [1], [2], 0.2) == pytest.approx(11.0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_mean_winkler_score_rejects_invalid_alpha_values(alpha):
    """Reject non-finite alpha values and values outside the open interval."""
    with pytest.raises(ValueError, match="alpha must be finite and in \\(0, 1\\)"):
        mean_winkler_score([0], [0], [1], alpha)


@pytest.mark.parametrize("alpha", ["0.1", [0.1], True])
def test_mean_winkler_score_rejects_non_real_alpha(alpha):
    """Reject alpha values that are not real scalars."""
    with pytest.raises(TypeError, match="alpha must be a real scalar"):
        mean_winkler_score([0], [0], [1], alpha)


def test_mean_winkler_score_rejects_inverted_bounds():
    """Identify the invalid lower and upper bounds in an error message."""
    with pytest.raises(ValueError, match="lower must be less than or equal to upper"):
        mean_winkler_score([0, 1], [0, 2], [1, 1], 0.1)


def test_mean_winkler_score_rejects_shape_mismatch():
    """Require all three data arrays to have identical shapes."""
    with pytest.raises(ValueError, match="identical shapes"):
        mean_winkler_score([0, 1], [0], [1, 2], 0.1)


def test_mean_winkler_score_rejects_empty_arrays():
    """Require at least one observation."""
    with pytest.raises(ValueError, match="must not be empty"):
        mean_winkler_score([], [], [], 0.1)


@pytest.mark.parametrize(
    ("y_true", "lower", "upper", "message"),
    [
        ([np.nan], [0], [1], "y_true must contain only finite values"),
        ([0], [np.inf], [1], "lower must contain only finite values"),
        ([0], [0], [-np.inf], "upper must contain only finite values"),
    ],
)
def test_mean_winkler_score_rejects_non_finite_inputs(y_true, lower, upper, message):
    """Reject NaN and infinity in each of the input arrays."""
    with pytest.raises(ValueError, match=message):
        mean_winkler_score(y_true, lower, upper, 0.1)


def test_mean_winkler_score_handles_very_small_alpha():
    """Keep the result finite for a small but valid miscoverage."""
    result = mean_winkler_score([1], [0], [0], 1e-6)

    assert np.isfinite(result)
    assert result == pytest.approx(2_000_000.0)


def test_mean_winkler_score_integer_and_float_inputs_match():
    """Convert integer inputs consistently with equivalent floats."""
    integer_result = mean_winkler_score([0, 3], [1, 1], [2, 4], 0.2)
    float_result = mean_winkler_score([0.0, 3.0], [1.0, 1.0], [2.0, 4.0], 0.2)

    assert integer_result == pytest.approx(float_result)


def test_mean_winkler_score_supports_nd_arrays():
    """Average over all elements in matching N-dimensional arrays."""
    result = mean_winkler_score([[0, 2]], [[-1, 1]], [[1, 3]], 0.1)

    assert result == pytest.approx(2.0)


class TestIntervalCoverage:
    @pytest.mark.parametrize(
        ("y_true", "lower", "upper", "expected"),
        [
            ([0.5, 0.6], [0, 0], [1, 1], 1.0),  # 2 covered / 2 total
            ([-1, 2], [0, 0], [1, 1], 0.0),  # 0 covered / 2 total
            ([0.5, -1, 3, 0.5], [0, 0, 0, 0], [1, 1, 1, 1], 0.5),  # 2 / 4
            ([0, 1], [0, 0], [1, 1], 1.0),  # Both closed-interval boundaries count.
            ([1, 2, 3, 4, 5], [0, 3, 2, 4, 6], [2, 4, 4, 4, 8], 0.6),  # 3 / 5
            ([2, 3, 4], [2, 2, 2], [2, 2, 2], 1.0 / 3.0),  # 1 match / 3 total
        ],
    )
    def test_known_values(self, y_true, lower, upper, expected):
        """Compute closed-interval coverage for representative cases."""
        assert interval_coverage(y_true, lower, upper) == pytest.approx(expected)

    def test_mixed_example_arithmetic(self):
        """Count three covered observations out of five."""
        # 1 in [0,2], 3 in [2,4], and 4 in [4,4] are covered: 3/5 = 0.6.
        assert interval_coverage(
            [1, 2, 3, 4, 5], [0, 3, 2, 4, 6], [2, 4, 4, 4, 8]
        ) == pytest.approx(0.6)

    @pytest.mark.parametrize(
        ("y_true", "expected"),
        [([1], 1.0), ([2], 0.0)],
    )
    def test_single_observation(self, y_true, expected):
        """Handle a single covered or uncovered observation."""
        assert interval_coverage(y_true, [0], [1]) == expected

    def test_coverage_bounds_and_count(self):
        """Coverage is a fraction whose product by n is an integer."""
        coverage = interval_coverage([0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 5])

        assert 0.0 <= coverage <= 1.0
        assert coverage * 4 == pytest.approx(round(coverage * 4))

    def test_common_shift_and_positive_scaling_invariance(self):
        """Preserve coverage under shifts and positive scale changes."""
        original = interval_coverage([0, 3], [1, 1], [2, 4])
        shifted = interval_coverage([10, 13], [11, 11], [12, 14])
        scaled = interval_coverage([0, 15], [5, 5], [10, 20])

        assert shifted == original
        assert scaled == original

    def test_widening_intervals_never_decreases_coverage(self):
        """Count at least as many observations after widening intervals."""
        original = interval_coverage([-1, 0, 2, 4], [0, 1, 1, 2], [1, 2, 3, 3])
        widened = interval_coverage([-1, 0, 2, 4], [-1, 0, 0, 1], [2, 3, 4, 4])

        assert widened >= original

    @pytest.mark.parametrize(
        "inputs",
        [
            ([0, 2], [-1, 1], [1, 3]),
            ((0, 2), (-1, 1), (1, 3)),
            (np.array([0, 2]), np.array([-1, 1]), np.array([1, 3])),
        ],
    )
    def test_common_array_likes(self, inputs):
        """Accept lists, tuples, and NumPy arrays identically."""
        assert interval_coverage(*inputs) == 1.0

    def test_pandas_series(self):
        """Accept pandas Series when installed in the test environment."""
        pandas = pytest.importorskip("pandas")

        assert (
            interval_coverage(
                pandas.Series([0, 2]), pandas.Series([-1, 1]), pandas.Series([1, 3])
            )
            == 1.0
        )

    def test_integer_and_float_inputs_match(self):
        """Convert integer inputs consistently with equivalent floats."""
        assert interval_coverage([0, 3], [1, 1], [2, 4]) == interval_coverage(
            [0.0, 3.0], [1.0, 1.0], [2.0, 4.0]
        )

    def test_supports_nd_arrays(self):
        """Average coverage over all elements in N-dimensional inputs."""
        assert interval_coverage([[0, 2]], [[-1, 1]], [[1, 3]]) == 1.0

    def test_handles_large_values(self):
        """Keep valid large magnitudes finite."""
        assert interval_coverage([1e300], [0.0], [2e300]) == 1.0

    @pytest.mark.parametrize(
        ("y_true", "lower", "upper", "error", "message"),
        [
            ([0, 1], [0, 2], [1, 1], ValueError, "lower must be less than or equal"),
            ([0, 1], [0], [1, 2], ValueError, "identical shapes"),
            ([], [], [], ValueError, "must not be empty"),
            ([np.nan], [0], [1], ValueError, "y_true must contain only finite"),
            ([0], [np.inf], [1], ValueError, "lower must contain only finite"),
            ([0], [0], [-np.inf], ValueError, "upper must contain only finite"),
        ],
    )
    def test_validation_errors_propagate(self, y_true, lower, upper, error, message):
        """Propagate array validation errors with their original messages."""
        with pytest.raises(error, match=message):
            interval_coverage(y_true, lower, upper)


class TestAverageWidth:
    @pytest.mark.parametrize(
        ("lower", "upper", "expected"),
        [
            ([0, 0, 0], [2, 2, 2], 2.0),  # (2 + 2 + 2) / 3 = 2
            ([0, 1], [0, 1], 0.0),  # (0 + 0) / 2 = 0
            ([0, 3, 2, 4, 6], [2, 4, 4, 4, 8], 1.4),  # 7 / 5
        ],
    )
    def test_known_values(self, lower, upper, expected):
        """Compute the mean distance between interval bounds."""
        assert average_width(lower, upper) == pytest.approx(expected)

    def test_mixed_example_arithmetic(self):
        """Average widths 2, 1, 2, 0, and 2 to get 7/5 = 1.4."""
        assert average_width([0, 3, 2, 4, 6], [2, 4, 4, 4, 8]) == pytest.approx(1.4)

    def test_width_is_non_negative(self):
        """Return a non-negative width for valid interval bounds."""
        assert average_width([0, 1], [2, 1]) >= 0.0

    def test_common_shift_and_positive_scaling(self):
        """Preserve widths under shifts and scale them by positive constants."""
        original = average_width([1, 1], [2, 4])
        shifted = average_width([11, 11], [12, 14])
        scaled = average_width([5, 5], [10, 20])

        assert shifted == pytest.approx(original)
        assert scaled == pytest.approx(5.0 * original)

    def test_widening_intervals_never_decreases_width(self):
        """Increase or preserve mean width after widening intervals."""
        original = average_width([0, 1], [1, 2])
        widened = average_width([-1, 0], [2, 3])

        assert widened >= original

    def test_winkler_score_agrees_with_width_and_coverage(self):
        """Winkler score equals width only when all observations are covered."""
        all_covered = ([1, 2], [0, 1], [3, 4])
        with_miss = ([-1, 2], [0, 1], [3, 4])

        score = mean_winkler_score(*all_covered, alpha=0.1)
        assert score == pytest.approx(average_width(all_covered[1], all_covered[2]))
        assert interval_coverage(*all_covered) == 1.0

        score = mean_winkler_score(*with_miss, alpha=0.1)
        assert score >= average_width(with_miss[1], with_miss[2])
        assert interval_coverage(*with_miss) < 1.0

    @pytest.mark.parametrize(
        "inputs",
        [
            ([-1, 1], [1, 3]),
            ((-1, 1), (1, 3)),
            (np.array([-1, 1]), np.array([1, 3])),
        ],
    )
    def test_common_array_likes(self, inputs):
        """Accept lists, tuples, and NumPy arrays identically."""
        assert average_width(*inputs) == 2.0

    def test_pandas_series(self):
        """Accept pandas Series when installed in the test environment."""
        pandas = pytest.importorskip("pandas")

        assert average_width(pandas.Series([-1, 1]), pandas.Series([1, 3])) == 2.0

    def test_integer_and_float_inputs_match(self):
        """Convert integer inputs consistently with equivalent floats."""
        assert average_width([1, 1], [2, 4]) == average_width([1.0, 1.0], [2.0, 4.0])

    def test_supports_nd_arrays(self):
        """Average widths across all elements in N-dimensional inputs."""
        assert average_width([[-1, 1]], [[1, 3]]) == 2.0

    def test_handles_large_values_without_overflow(self):
        """Compute finite widths at large but representable magnitudes."""
        assert average_width([0.0, 0.0], [1e300, 1e300]) == 1e300

    def test_unrepresentable_width_raises_overflow_error(self):
        """Report when subtracting finite bounds would overflow float64."""
        with pytest.raises(OverflowError, match="not representable as a finite float"):
            average_width([-1.7e308], [1.7e308])

    @pytest.mark.parametrize(
        ("lower", "upper", "error", "message"),
        [
            ([0, 2], [1, 1], ValueError, "lower must be less than or equal"),
            ([0], [1, 2], ValueError, "identical shapes"),
            ([], [], ValueError, "must not be empty"),
            ([np.inf], [1], ValueError, "lower must contain only finite"),
            ([0], [-np.inf], ValueError, "upper must contain only finite"),
        ],
    )
    def test_validation_errors_propagate(self, lower, upper, error, message):
        """Propagate array validation errors with their original messages."""
        with pytest.raises(error, match=message):
            average_width(lower, upper)
