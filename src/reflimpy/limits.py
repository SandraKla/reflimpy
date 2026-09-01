"""Confidence intervals for reference limits."""

from math import exp,isfinite, log, sqrt
from numbers import Real

from ._rounding import adjust_digits


def conf_int95(
    n: int,
    lower_limit: Real,
    upper_limit: Real,
    lognormal: bool = True,
    apply_rounding: bool = True,
) -> dict[str, float | int]:
    """Calculate approximate 95% confidence intervals for reference limits."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be greater than zero")

    if(
        isinstance(lower_limit, bool)
        or not isinstance(lower_limit, Real)
        or isinstance(upper_limit, bool)
        or not isinstance(upper_limit, Real)
    ):
        raise TypeError("limits must be real numbers")

    lower = float(lower_limit)
    upper = float(upper_limit)

    if not isfinite(lower) or not isfinite(upper):
        raise ValueError("limits must be finite")

    if upper <= lower:
        raise ValueError("upper_limit must be higher than lower_limit")

    if lognormal:
        if lower <= 0:
            raise ValueError("limits must be positive when lognormal=True")

        lower = log(lower)
        upper = log(upper)

    sigma = (upper - lower) / 3.92

    diff_outer = sigma * 5.81 / (sqrt(n) + 0.66)
    diff_inner = sigma * 7.26 / (sqrt(n) - 5.58)

    result: dict[str, float | int] = {
        "lower_lim_low": lower - diff_outer,
        "lower_lim_upp": lower + diff_inner,
        "upper_lim_low": upper - diff_inner,
        "upper_lim_upp": upper + diff_outer,
        "n": n,
    }

    if lognormal:
        for key in (
                "lower_lim_low",
                "lower_lim_upp",
                "upper_lim_low",
                "upper_lim_upp",
        ):
            result[key] = exp(float(result[key]))

    if apply_rounding:
        digits = int(
            adjust_digits(result["lower_lim_low"])["digits"]
        )

        for key in (
                "lower_lim_low",
                "lower_lim_upp",
                "upper_lim_low",
                "upper_lim_upp",
        ):
            result[key] = round(float(result[key]), digits)

    return result
