"""Tests for bowley()."""

import math

import numpy as np
import pytest

from reflimpy import bowley


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2, 3, 4, 5], 0.0),
        ([1, 2, 3, 10, 20], 0.75),
        (list(range(1, 101)), 0.0),
    ],
)
def test_bowley(values: list[float], expected: float) -> None:
    """Bowley skewness matches known results."""
    assert bowley(values) == pytest.approx(expected)


def test_bowley_returns_nan_for_constant_values() -> None:
    """A zero quantile range produces NaN."""
    result = bowley([3, 3, 3, 3])

    assert math.isnan(result)


def test_bowley_rejects_empty_input() -> None:
    """An empty sequence cannot be evaluated."""
    with pytest.raises(ValueError, match="x must not be empty"):
        bowley([])


def test_bowley_rejects_multidimensional_input() -> None:
    """Only one-dimensional data are accepted."""
    values = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="x must be one-dimensional"):
        bowley(values)


@pytest.mark.parametrize(
    "values",
    [
        [1, math.nan, 3],
        [1, math.inf, 3],
        [1, -math.inf, 3],
    ],
)
def test_bowley_rejects_non_finite_values(
    values: list[float],
) -> None:
    """NaN and infinite inputs are rejected."""
    with pytest.raises(
        ValueError,
        match="x must contain only finite values",
    ):
        bowley(values)


@pytest.mark.parametrize("alpha", [-0.1, 0.5, 1.0])
def test_bowley_rejects_invalid_alpha(alpha: float) -> None:
    """Alpha must describe a valid lower quantile."""
    with pytest.raises(ValueError, match="alpha must satisfy"):
        bowley([1, 2, 3], alpha=alpha)
