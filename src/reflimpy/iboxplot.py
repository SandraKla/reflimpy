"""Iterative boxplot truncation."""

from collections.abc import Sequence
from numbers import Real
from statistics import NormalDist

import numpy as np

from ._rounding import adjust_digits
from ._statistics import lognorm


def iboxplot(
    x: Sequence[float],
    lognormal: bool | None = None,
    perc_trunc: float = 2.5,
    apply_rounding: bool = True,
) -> dict[str, object]:
    """Remove extreme values using iterative boxplot truncation."""
    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError("x must contain numeric values") from error

    if values.ndim != 1:
        raise ValueError("x must be one-dimensional")

    values = values[~np.isnan(values)]

    if values.size == 0:
        raise ValueError("x must not be empty")

    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")

    if np.min(values) <= 0:
        raise ValueError("x must contain only positive values")

    number_of_values = values.size

    if number_of_values < 40:
        raise ValueError(
            "at least 40 values are required for "
            "reference-limit estimation"
        )

    if lognormal is not None and not isinstance(lognormal, bool):
        raise TypeError("lognormal must be a boolean or None")

    if isinstance(perc_trunc, bool) or not isinstance(
        perc_trunc,
        Real,
    ):
        raise TypeError("perc_trunc must be a real number")

    perc_trunc = float(perc_trunc)

    if (
        not np.isfinite(perc_trunc)
        or perc_trunc <= 0
        or perc_trunc > 25
    ):
        raise ValueError(
            "perc_trunc must be greater than 0 "
            "and not greater than 25"
        )

    if not isinstance(apply_rounding, bool):
        raise TypeError("apply_rounding must be a boolean")

    digits = int(
        adjust_digits(float(np.median(values)))["digits"]
    )

    if lognormal is None:
        lognormal_result = lognorm(values, digits=None)
        use_lognormal = bool(lognormal_result["lognormal"])
    else:
        use_lognormal = lognormal

    working_values = values.copy()

    if use_lognormal:
        working_values = np.log(working_values)

    quantile_fraction = perc_trunc / 100
    normal_distribution = NormalDist()
    progress: list[dict[str, float | int | str]] = []

    def record_progress(
        cycle: int,
        current_values: np.ndarray,
    ) -> None:
        """Store the number, minimum, and maximum of one cycle."""
        displayed_values = current_values

        if use_lognormal:
            displayed_values = np.exp(displayed_values)

        progress.append(
            {
                "cycle": f"cycle{cycle}",
                "n": int(displayed_values.size),
                "min": round(float(np.min(displayed_values)), digits),
                "max": round(float(np.max(displayed_values)), digits),
            }
        )

    def truncate_values(
        current_values: np.ndarray,
        quantile_factor: float,
    ) -> np.ndarray:
        """Keep values inside the calculated truncation points."""
        lower_quartile, median, upper_quartile = np.quantile(
            current_values,
            [0.25, 0.5, 0.75],
            method="linear",
        )

        lower_distance = median - lower_quartile
        upper_distance = upper_quartile - median
        distance = min(lower_distance, upper_distance)
        lower_point = median - quantile_factor * distance
        upper_point = median + quantile_factor * distance

        return current_values[
            (current_values >= lower_point)
            & (current_values <= upper_point)
        ]

    record_progress(0, working_values)

    number_of_steps = 5

    for cycle in range(1, number_of_steps + 1):
        target_quantile = (
            1 - quantile_fraction * cycle / number_of_steps
        )
        alpha = (
            2 * quantile_fraction * (cycle - 1) / number_of_steps
        )
        quantile_factor = (
            normal_distribution.inv_cdf(target_quantile)
            / normal_distribution.inv_cdf(0.75 - 0.25 * alpha)
        )

        working_values = truncate_values(
            working_values,
            quantile_factor,
        )
        record_progress(cycle, working_values)

    quantile_factor = (
        normal_distribution.inv_cdf(1 - quantile_fraction)
        / normal_distribution.inv_cdf(
            0.75 - 0.5 * quantile_fraction
        )
    )

    while True:
        previous_size = working_values.size
        working_values = truncate_values(
            working_values,
            quantile_factor,
        )
        cycle += 1
        record_progress(cycle, working_values)

        if working_values.size == previous_size:
            break

    if use_lognormal:
        working_values = np.exp(working_values)

    lower_point = float(np.min(working_values))
    upper_point = float(np.max(working_values))

    if apply_rounding:
        lower_point = round(lower_point, digits)
        upper_point = round(upper_point, digits)

    percent_normal = (
        working_values.size
        * 100
        / (1 - 2 * perc_trunc / 100)
        / number_of_values
    )
    percent_normal = min(round(float(percent_normal), 1), 100.0)

    return {
        "truncated": working_values,
        "truncation_points": {
            "lower": lower_point,
            "upper": upper_point,
        },
        "lognormal": use_lognormal,
        "percent_normal": percent_normal,
        "progress": progress,
    }
