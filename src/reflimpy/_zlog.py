"""Standardized logarithmic values for laboratory measurements."""

from collections.abc import Sequence
from numbers import Real

import numpy as np


def zlog(
    x: Real | Sequence[float],
    lower: Real,
    upper: Real,
) -> float | np.ndarray:
    """Calculate standardized logarithmic values.
    Invalid measurements, such as non-positive or non-finite values,
    are returned as NaN
    """

    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("x must contain numeric values") from error

    if values.ndim > 1:
        raise ValueError("x must be a scalar or one-dimensional")

    valid_limits = (
        isinstance(lower, Real)
        and not isinstance(lower, bool)
        and isinstance(upper, Real)
        and not isinstance(upper, bool)
        and np.isfinite(lower)
        and np.isfinite(upper)
        and lower > 0
        and upper > lower
    )

    result = np.full(values.size, np.nan, dtype=float)

    if valid_limits:
        flat_values = values.reshape(-1)
        valid_values = np.isfinite(flat_values) & (flat_values > 0)

        log_lower = np.log(float(lower))
        log_upper = np.log(float(upper))
        mean_log = (log_lower + log_upper) / 2
        standard_deviation = (log_upper - log_lower) / 3.919928

        result[valid_values] = (
            np.log(flat_values[valid_values]) - mean_log
        ) / standard_deviation

    result = result.reshape(values.shape)

    if values.ndim == 0:
        return float(result.item())

    return result