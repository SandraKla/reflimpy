"""Test for zlog()."""

import math

import numpy as np
import pytest

from reflimpy import zlog

def test_middle_value_returns_zero() -> None:
    assert zlog(4, lower=2, upper=8) == pytest.approx(0.0)

def test_reference_limits() -> None:
    result = zlog([2, 4, 8], lower=2, upper=8)

    expected = np.array([-1.959964, 0.0, 1.959964])

    np.testing.assert_allclose(result, expected)

def test_invalid_measurements_return_nan() -> None:
    result = zlog(
        [4, 0, -1, np.nan, np.inf],
        lower=2,
        upper=8,
    )

    assert result[0] == pytest.approx(0.0)
    assert np.all(np.isnan(result[1:]))

@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (0, 8),
        (-1, 8),
        (2, 0),
        (8, 2),
        (2, 2),
        (math.nan, 8),
        (2, math.inf),
    ],
)
def test_invalid_limits_return_nan(
    lower: float,
    upper: float,
) -> None:
    result = zlog([2, 4, 8], lower=lower, upper=upper)

    assert np.all(np.isnan(result))


def test_non_numeric_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="numeric"):
        zlog(["a", "b"], lower=2, upper=8)


def test_two_dimensional_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        zlog([[2, 4], [8, 16]], lower=2, upper=8)