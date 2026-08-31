"""Tests for the Box-Cox transformations."""

import math

import numpy as np
import pytest

from reflimpy import (
    box_cox_transform,
    inverse_box_cox_transform,
)


def test_box_cox_lambda_zero() -> None:
    result = box_cox_transform(
        [1, 2, 4],
        lambda_=0,
    )

    expected = np.log([1, 2, 4])

    np.testing.assert_allclose(result, expected)


def test_box_cox_lambda_one() -> None:
    result = box_cox_transform(
        [1, 2, 4],
        lambda_=1,
    )

    expected = np.array([0, 1, 3])

    np.testing.assert_allclose(result, expected)


def test_box_cox_lambda_two() -> None:
    result = box_cox_transform(
        [1, 2, 4],
        lambda_=2,
    )

    expected = np.array([0, 1.5, 7.5])

    np.testing.assert_allclose(result, expected)


def test_inverse_box_cox_lambda_zero() -> None:
    transformed = np.log([1, 2, 4])

    result = inverse_box_cox_transform(
        transformed,
        lambda_=0,
    )

    np.testing.assert_allclose(result, [1, 2, 4])


@pytest.mark.parametrize(
    "lambda_",
    [-1, 0, 0.5, 1, 2],
)
def test_inverse_restores_original_values(
    lambda_: float,
) -> None:
    original = np.array([1, 2, 4], dtype=float)

    transformed = box_cox_transform(
        original,
        lambda_=lambda_,
    )
    restored = inverse_box_cox_transform(
        transformed,
        lambda_=lambda_,
    )

    np.testing.assert_allclose(restored, original)


def test_scalar_input_returns_float() -> None:
    result = box_cox_transform(4, lambda_=0)

    assert isinstance(result, float)
    assert result == pytest.approx(math.log(4))


def test_non_numeric_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="numeric"):
        box_cox_transform(["a", "b"], lambda_=1)


def test_two_dimensional_input_raises_value_error() -> None:
    with pytest.raises(
        ValueError,
        match="scalar or one-dimensional",
    ):
        box_cox_transform(
            [[1, 2], [3, 4]],
            lambda_=1,
        )


@pytest.mark.parametrize(
    "lambda_",
    [math.nan, math.inf, -math.inf],
)
def test_non_finite_lambda_raises_value_error(
    lambda_: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        box_cox_transform(
            [1, 2, 4],
            lambda_=lambda_,
        )


def test_invalid_inverse_domain_returns_nan() -> None:
    result = inverse_box_cox_transform(
        -1,
        lambda_=2,
    )

    assert math.isnan(result)
