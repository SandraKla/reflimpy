"""Tests for adjust_digits()."""

import math

import pytest

from reflimpy import adjust_digits


@pytest.mark.parametrize(
    ("value", "expected_round", "expected_digits"),
    [
        (0, 0.0, 0),
        (0.001234, 0.00123, 5),
        (-12.34, -12.3, 1),
        (5.4321, 5.43, 2),
        (1234.56, 1235.0, 0),
    ],
)
def test_adjust_digits(
    value: float,
    expected_round: float,
    expected_digits: int,
) -> None:
    """The value and number of digits match the expected results."""
    result = adjust_digits(value)

    assert result["x_round"] == pytest.approx(expected_round)
    assert result["digits"] == expected_digits


@pytest.mark.parametrize("value", [True, "12.34", [12.34]])
def test_adjust_digits_rejects_non_real_values(value: object) -> None:
    """Non-real and boolean inputs are rejected."""
    with pytest.raises(TypeError, match="x must be a single real number"):
        adjust_digits(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_adjust_digits_rejects_non_finite_values(value: float) -> None:
    """NaN and infinite inputs are rejected."""
    with pytest.raises(ValueError, match="x must be finite"):
        adjust_digits(value)
