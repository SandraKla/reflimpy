"""Mathematical transformations used by reflimpy."""

from collections.abc import Sequence
from numbers import Real

import numpy as np


def _as_numeric_array(
    x: Real | Sequence[float],
) -> np.ndarray:
    """Convert a scalar or one-dimensional sequence to a numeric array."""
    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("x must contain numeric values") from error

    if values.ndim > 1:
        raise ValueError("x must be a scalar or one-dimensional")

    return values


def _validate_lambda(lambda_: Real) -> float:
    """Validate and convert the Box-Cox parameter."""
    if isinstance(lambda_, bool) or not isinstance(lambda_, Real):
        raise TypeError("lambda_ must be a real number")

    parameter = float(lambda_)

    if not np.isfinite(parameter):
        raise ValueError("lambda_ must be finite")

    return parameter


def _restore_result_type(
    result: np.ndarray,
) -> float | np.ndarray:
    """Return a float for scalar input and an array otherwise."""
    if result.ndim == 0:
        return float(result.item())

    return result


def box_cox_transform(
    x: Real | Sequence[float],
    lambda_: Real = 1,
) -> float | np.ndarray:
    """Apply the Box-Cox transformation."""
    values = _as_numeric_array(x)
    parameter = _validate_lambda(lambda_)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if parameter == 0:
            result = np.log(values)
        else:
            result = (
                np.power(values, parameter) - 1
            ) / parameter

    return _restore_result_type(result)


def inverse_box_cox_transform(
    x: Real | Sequence[float],
    lambda_: Real = 1,
) -> float | np.ndarray:
    """Apply the inverse Box-Cox transformation."""
    values = _as_numeric_array(x)
    parameter = _validate_lambda(lambda_)

    with np.errstate(invalid="ignore", over="ignore"):
        if parameter == 0:
            result = np.exp(values)
        else:
            result = np.power(
                parameter * values + 1,
                1 / parameter,
            )

    return _restore_result_type(result)
