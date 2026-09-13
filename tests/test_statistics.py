"""Tests for bowley()."""

import math

import numpy as np
import pytest

from reflimpy import bowley, lognorm


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


def test_lognorm_identifies_normal_data() -> None:
    """Symmetric data are classified as non-lognormal."""
    result = lognorm([1, 2, 3, 4, 5])

    assert result == {
        "lognormal": False,
        "bowley_skewness": {
            "normal": 0.0,
            "lognormal": 0.17,
            "delta": -0.17,
        },
    }


def test_lognorm_identifies_lognormal_data() -> None:
    """Exponentially transformed symmetric data are lognormal."""
    values = np.exp([1, 2, 3, 4, 5])
    result = lognorm(values)

    assert result == {
        "lognormal": True,
        "bowley_skewness": {
            "normal": 0.462,
            "lognormal": 0.0,
            "delta": 0.462,
        },
    }


def test_lognorm_removes_missing_values() -> None:
    """NaN values are omitted as in the R implementation."""
    result = lognorm([1, 2, math.nan, 3, 4, 5])
    expected = lognorm([1, 2, 3, 4, 5])

    assert result == expected


def test_lognorm_can_return_unrounded_skewness() -> None:
    """Passing digits=None preserves full floating-point precision."""
    result = lognorm([1, 2, 3, 4, 5], digits=None)
    skewness = result["bowley_skewness"]

    assert isinstance(skewness, dict)
    assert skewness["normal"] == pytest.approx(0.0)
    assert skewness["lognormal"] == pytest.approx(
        0.16992500144231257
    )
    assert skewness["delta"] == pytest.approx(
        -0.16992500144231257
    )


def test_lognorm_uses_cutoff_for_classification() -> None:
    """A larger cutoff can change the distribution classification."""
    values = np.exp([1, 2, 3, 4, 5])

    assert lognorm(values, cutoff=0.05)["lognormal"] is True
    assert lognorm(values, cutoff=0.5)["lognormal"] is False


def test_lognorm_rejects_multidimensional_input() -> None:
    """Only one-dimensional data are accepted."""
    values = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="x must be one-dimensional"):
        lognorm(values)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [1],
        [math.nan, 1],
    ],
)
def test_lognorm_requires_two_numeric_values(
    values: list[float],
) -> None:
    """At least two non-missing values are required."""
    with pytest.raises(ValueError, match="at least two"):
        lognorm(values)


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 2],
        [-1, 1, 2],
    ],
)
def test_lognorm_rejects_non_positive_values(
    values: list[float],
) -> None:
    """A logarithmic transformation requires positive values."""
    with pytest.raises(ValueError, match="only positive values"):
        lognorm(values)


@pytest.mark.parametrize(
    "values",
    [
        [1, math.inf],
        [1, -math.inf],
    ],
)
def test_lognorm_rejects_non_finite_values(
    values: list[float],
) -> None:
    """Infinite values are rejected."""
    with pytest.raises(ValueError, match="only finite values"):
        lognorm(values)


def test_lognorm_rejects_non_numeric_values() -> None:
    """Non-numeric input cannot be converted for analysis."""
    with pytest.raises(TypeError, match="numeric values"):
        lognorm(["one", "two"])  # type: ignore[list-item]


@pytest.mark.parametrize("cutoff", [-0.1, math.nan, math.inf])
def test_lognorm_rejects_invalid_cutoff(cutoff: float) -> None:
    """The cutoff must be finite and non-negative."""
    with pytest.raises(ValueError, match="cutoff must be finite"):
        lognorm([1, 2, 3], cutoff=cutoff)


@pytest.mark.parametrize("digits", [-1, 1.5, True])
def test_lognorm_rejects_invalid_digits(digits: object) -> None:
    """Digits must be a non-negative integer or None."""
    with pytest.raises(ValueError, match="digits must be"):
        lognorm([1, 2, 3], digits=digits)  # type: ignore[arg-type]
