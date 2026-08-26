"""Rounding utilities for reference-limit results."""

import math
from numbers import Real


def adjust_digits(x: Real) -> dict[str, float | int]:
    """Round a number to a plausible number of decimal places."""
    if isinstance(x, bool) or not isinstance(x, Real):
        raise TypeError("x must be a single real number")

    value = float(x)

    if not math.isfinite(value):
        raise ValueError("x must be finite")

    if value == 0:
        return {"x_round": 0.0, "digits": 0}

    digits = max(0, 2 - math.floor(math.log10(abs(value))))

    return {
        "x_round": round(value, digits),
        "digits": digits,
    }
