"""Tests for interpretation()."""

import math

import pytest

from reflimpy import interpretation


def test_interpretation_returns_green_for_values_within_tolerance() -> None:
    """Limits inside the target tolerance are classified as green."""
    result = interpretation(
        limits=[2.6, 96.0],
        targets=[2.5, 97.5],
    )

    assert result["colors"] == {
        "lower_limit": "green",
        "upper_limit": "green",
    }

    assert result["deviations"] == {
        "lower_limit": "within tolerance",
        "upper_limit": "within tolerance",
    }


def test_interpretation_returns_yellow_for_slight_increase() -> None:
    """Overlapping uncertainty ranges indicate a slight increase."""
    result = interpretation(
        limits=[3.3, 110.0],
        targets=[2.5, 97.5],
    )

    assert result["colors"] == {
        "lower_limit": "yellow",
        "upper_limit": "yellow",
    }

    assert result["deviations"] == {
        "lower_limit": "slightly increased",
        "upper_limit": "slightly increased",
    }


def test_interpretation_returns_red_for_marked_decrease() -> None:
    """Non-overlapping uncertainty ranges indicate a marked decrease."""
    result = interpretation(
        limits=[1.0, 70.0],
        targets=[2.5, 97.5],
    )

    assert result["colors"] == {
        "lower_limit": "red",
        "upper_limit": "red",
    }

    assert result["deviations"] == {
        "lower_limit": "markedly decreased",
        "upper_limit": "markedly decreased",
    }


def test_interpretation_returns_tolerance_ranges() -> None:
    """The result contains tolerance ranges for limits and targets."""
    result = interpretation(
        limits=[2.6, 96.0],
        targets=[2.5, 97.5],
    )

    assert result["tolerance_targets"] == {
        "lower_lim_low": 1.79,
        "lower_lim_upp": 3.21,
        "upper_lim_low": 86.23,
        "upper_lim_upp": 108.77,
    }

    assert set(result["tolerance_limits"]) == {
        "lower_lim_low",
        "lower_lim_upp",
        "upper_lim_low",
        "upper_lim_upp",
    }


@pytest.mark.parametrize(
    "limits",
    [
        [],
        [2.5],
        [2.5, 50.0, 97.5],
    ],
)
def test_interpretation_rejects_invalid_limits_length(
    limits: list[float],
) -> None:
    """Limits must contain exactly two values."""
    with pytest.raises(
        ValueError,
        match="limits must contain exactly two values",
    ):
        interpretation(
            limits=limits,
            targets=[2.5, 97.5],
        )


@pytest.mark.parametrize(
    "targets",
    [
        [],
        [2.5],
        [2.5, 50.0, 97.5],
    ],
)
def test_interpretation_rejects_invalid_targets_length(
    targets: list[float],
) -> None:
    """Targets must contain exactly two values."""
    with pytest.raises(
        ValueError,
        match="targets must contain exactly two values",
    ):
        interpretation(
            limits=[2.6, 96.0],
            targets=targets,
        )


@pytest.mark.parametrize(
    "limits",
    [
        [0.0, 100.0],
        [-1.0, 100.0],
        [100.0, 100.0],
        [110.0, 100.0],
    ],
)
def test_interpretation_rejects_invalid_limit_values(
    limits: list[float],
) -> None:
    """Limits must be positive and in ascending order."""
    with pytest.raises(ValueError):
        interpretation(
            limits=limits,
            targets=[2.5, 97.5],
        )


@pytest.mark.parametrize(
    "targets",
    [
        [0.0, 100.0],
        [-1.0, 100.0],
        [100.0, 100.0],
        [110.0, 100.0],
    ],
)
def test_interpretation_rejects_invalid_target_values(
    targets: list[float],
) -> None:
    """Targets must be positive and in ascending order."""
    with pytest.raises(ValueError):
        interpretation(
            limits=[2.6, 96.0],
            targets=targets,
        )


@pytest.mark.parametrize(
    "limits",
    [
        [math.nan, 100.0],
        [1.0, math.inf],
        [1.0, -math.inf],
    ],
)
def test_interpretation_rejects_non_finite_values(
    limits: list[float],
) -> None:
    """NaN and infinite values are rejected."""
    with pytest.raises(
        ValueError,
        match="limits must contain finite values",
    ):
        interpretation(
            limits=limits,
            targets=[2.5, 97.5],
        )


@pytest.mark.parametrize(
    "limits",
    [
        [True, 100.0],
        [1.0, "100"],
    ],
)
def test_interpretation_rejects_non_real_values(
    limits: list[object],
) -> None:
    """Boolean and non-numeric values are rejected."""
    with pytest.raises(
        TypeError,
        match="limits must contain real numbers",
    ):
        interpretation(
            limits=limits,  # type: ignore[arg-type]
            targets=[2.5, 97.5],
        )