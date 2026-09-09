"""Interpretation of estimated and target reference limits."""

from collections.abc import Sequence
from math import isfinite
from numbers import Real

from .limits import permissible_uncertainty


def _validate_limits(
    values: Sequence[Real],
    name: str,
) -> tuple[float, float]:
    """Validate and convert a pair of reference limits."""
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")

    lower, upper = values

    if (
        isinstance(lower, bool)
        or not isinstance(lower, Real)
        or isinstance(upper, bool)
        or not isinstance(upper, Real)
    ):
        raise TypeError(f"{name} must contain real numbers")

    lower = float(lower)
    upper = float(upper)

    if not isfinite(lower) or not isfinite(upper):
        raise ValueError(f"{name} must contain finite values")

    if lower <= 0 or upper <= 0:
        raise ValueError(f"{name} must contain positive values")

    if lower >= upper:
        raise ValueError(
            f"the upper {name} value must be greater "
            f"than the lower value"
        )

    return lower, upper


def interpretation(
    limits: Sequence[Real],
    targets: Sequence[Real],
) -> dict[str, dict[str, float] | dict[str, str]]:
    """Compare estimated reference limits with target limits."""
    lower_limit, upper_limit = _validate_limits(
        limits,
        "limits",
    )
    lower_target, upper_target = _validate_limits(
        targets,
        "targets",
    )

    tolerance_limits = permissible_uncertainty(
        lower_limit,
        upper_limit,
    )
    tolerance_targets = permissible_uncertainty(
        lower_target,
        upper_target,
    )

    colors = {
        "lower_limit": "grey",
        "upper_limit": "grey",
    }

    deviations = {
        "lower_limit": "within tolerance",
        "upper_limit": "within tolerance",
    }

    # Compare the estimated lower limit with the target tolerance.
    if (
        tolerance_targets["lower_lim_low"]
        <= lower_limit
        <= tolerance_targets["lower_lim_upp"]
    ):
        colors["lower_limit"] = "green"

    elif (
        tolerance_targets["lower_lim_low"]
        <= tolerance_limits["lower_lim_low"]
        <= tolerance_targets["lower_lim_upp"]
        or tolerance_targets["lower_lim_low"]
        <= tolerance_limits["lower_lim_upp"]
        <= tolerance_targets["lower_lim_upp"]
    ):
        colors["lower_limit"] = "yellow"

        if lower_limit < lower_target:
            deviations["lower_limit"] = "slightly decreased"
        else:
            deviations["lower_limit"] = "slightly increased"

    else:
        colors["lower_limit"] = "red"

        if lower_limit < lower_target:
            deviations["lower_limit"] = "markedly decreased"
        else:
            deviations["lower_limit"] = "markedly increased"

    # Compare the estimated upper limit with the target tolerance.
    if (
        tolerance_targets["upper_lim_low"]
        <= upper_limit
        <= tolerance_targets["upper_lim_upp"]
    ):
        colors["upper_limit"] = "green"

    elif (
        tolerance_targets["upper_lim_low"]
        <= tolerance_limits["upper_lim_low"]
        <= tolerance_targets["upper_lim_upp"]
        or tolerance_targets["upper_lim_low"]
        <= tolerance_limits["upper_lim_upp"]
        <= tolerance_targets["upper_lim_upp"]
    ):
        colors["upper_limit"] = "yellow"

        if upper_limit < upper_target:
            deviations["upper_limit"] = "slightly decreased"
        else:
            deviations["upper_limit"] = "slightly increased"

    else:
        colors["upper_limit"] = "red"

        if upper_limit < upper_target:
            deviations["upper_limit"] = "markedly decreased"
        else:
            deviations["upper_limit"] = "markedly increased"

    return {
        "tolerance_limits": tolerance_limits,
        "tolerance_targets": tolerance_targets,
        "colors": colors,
        "deviations": deviations,
    }