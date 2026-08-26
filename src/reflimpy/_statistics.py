"""Statistical utilities used by reference-limit estimation."""

from collections.abc import Sequence

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
