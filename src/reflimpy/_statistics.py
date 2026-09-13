"""Statistical utilities used by reference-limit estimation."""

from collections.abc import Sequence
from numbers import Real

import numpy as np


def bowley(x: Sequence[float], alpha: float = 0.25) -> float:
    """Calculate Bowley's quantile skewness."""
    values = np.asarray(x, dtype=float)

    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")

    if values.size == 0:
        raise ValueError("x must not be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")

    if not 0 <= alpha < 0.5:
        raise ValueError("alpha must satisfy 0 <= alpha < 0.5")

    lower, median, upper = np.quantile(
        values,
        [alpha, 0.5, 1 - alpha],
        method="linear",
    )

    quantile_range = upper - lower

    if quantile_range == 0:
        return float("nan")

    skewness = (
        lower - 2 * median + upper
    ) / quantile_range

    return float(abs(skewness))


def lognorm(
    x: Sequence[float],
    cutoff: float = 0.05,
    alpha: float = 0.25,
    digits: int | None = 3,
) -> dict[str, bool | dict[str, float]]:
    """Determine whether data are better described as lognormal."""
    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("x must contain numeric values") from error

    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")

    values = values[~np.isnan(values)]

    if values.size < 2:
        raise ValueError("x must contain at least two numeric values")

    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")

    if np.min(values) <= 0:
        raise ValueError("x must contain only positive values")

    if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
        raise TypeError("cutoff must be a real number")

    cutoff_value = float(cutoff)

    if not np.isfinite(cutoff_value) or cutoff_value < 0:
        raise ValueError("cutoff must be finite and non-negative")

    if digits is not None and (
        isinstance(digits, bool)
        or not isinstance(digits, int)
        or digits < 0
    ):
        raise ValueError(
            "digits must be a non-negative integer or None"
        )

    normal_skewness = bowley(values, alpha)
    lognormal_skewness = bowley(np.log(values), alpha)
    delta = normal_skewness - lognormal_skewness
    is_lognormal = bool(delta >= cutoff_value)

    if digits is not None:
        normal_skewness = round(normal_skewness, digits)
        lognormal_skewness = round(lognormal_skewness, digits)
        delta = round(normal_skewness - lognormal_skewness, digits)

    return {
        "lognormal": is_lognormal,
        "bowley_skewness": {
            "normal": normal_skewness,
            "lognormal": lognormal_skewness,
            "delta": delta,
        },
    }
