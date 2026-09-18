"""Reference-limit estimation using a truncated Q-Q plot."""

import math
import warnings
from collections.abc import Sequence
from numbers import Real
from statistics import NormalDist

import numpy as np

from ._rounding import adjust_digits
from ._statistics import lognorm


def truncated_qqplot(
    x_trunc: Sequence[float],
    lognormal: bool | None = None,
    perc_trunc: float = 2.5,
    n_min: int = 200,
    apply_rounding: bool = True,
) -> dict[str, object]:
    """Estimate reference limits from truncated positive data."""
    try:
        values = np.asarray(x_trunc, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "x_trunc must contain numeric values"
        ) from error

    if values.ndim != 1:
        raise ValueError("x_trunc must be one-dimensional")

    values = values[~np.isnan(values)]

    if values.size < 40:
        raise ValueError("x_trunc must contain at least 40 values")

    if not np.all(np.isfinite(values)):
        raise ValueError("x_trunc must contain only finite values")

    if np.min(values) <= 0:
        raise ValueError("x_trunc must contain only positive values")

    if isinstance(perc_trunc, bool) or not isinstance(
        perc_trunc,
        Real,
    ):
        raise TypeError("perc_trunc must be a real number")

    perc_trunc_value = float(perc_trunc)

    if not np.isfinite(perc_trunc_value) or not (
        0 < perc_trunc_value < 50
    ):
        raise ValueError("perc_trunc must be between 0 and 50")

    if (
        isinstance(n_min, bool)
        or not isinstance(n_min, int)
    ):
        raise TypeError("n_min must be an integer")

    if n_min < 40:
        raise ValueError("n_min must be at least 40")

    if lognormal is not None and not isinstance(lognormal, bool):
        raise TypeError("lognormal must be True, False, or None")

    if not isinstance(apply_rounding, bool):
        raise TypeError("apply_rounding must be a boolean")

    if values.size < n_min:
        warnings.warn(
            f"At least {n_min} values are required",
            UserWarning,
            stacklevel=2,
        )
        return {"result": None, "lognormal": None}

    original_values = values.copy()

    if lognormal is None:
        lognormal = bool(lognorm(values)["lognormal"])

    if lognormal:
        values = np.log(values)

    number_of_quantiles = min(100, values.size)
    theoretical_probabilities = np.linspace(
        perc_trunc_value / 100,
        1 - perc_trunc_value / 100,
        number_of_quantiles,
    )
    sample_probabilities = np.linspace(
        0,
        1,
        number_of_quantiles,
    )

    normal_distribution = NormalDist()
    theoretical_quantiles = np.asarray(
        [
            normal_distribution.inv_cdf(probability)
            for probability in theoretical_probabilities
        ]
    )
    sample_quantiles = np.quantile(
        values,
        sample_probabilities,
        method="linear",
    )

    central_start = max(
        1,
        math.floor(0.05 * number_of_quantiles),
    ) - 1
    central_stop = math.ceil(0.95 * number_of_quantiles)

    slope, intercept = np.polyfit(
        theoretical_quantiles[central_start:central_stop],
        sample_quantiles[central_start:central_stop],
        1,
    )

    lower_limit = float(intercept - 1.96 * slope)
    upper_limit = float(intercept + 1.96 * slope)

    if lognormal:
        result = {
            "mean_log": round(float(intercept), 3),
            "sd_log": round(float(slope), 3),
            "lower_lim": float(np.exp(lower_limit)),
            "upper_lim": float(np.exp(upper_limit)),
        }
    else:
        result = {
            "mean": float(intercept),
            "sd": float(slope),
            "lower_lim": max(0.0, lower_limit),
            "upper_lim": upper_limit,
        }

    if apply_rounding:
        digits = int(
            adjust_digits(float(np.median(original_values)))[
                "digits"
            ]
        )
        result["lower_lim"] = round(result["lower_lim"], digits)
        result["upper_lim"] = round(result["upper_lim"], digits)

    return {"result": result, "lognormal": lognormal}
