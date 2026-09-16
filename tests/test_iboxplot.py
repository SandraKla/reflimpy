"""Tests for iboxplot()."""

import math

import numpy as np
import pytest

from reflimpy import iboxplot


@pytest.fixture
def sample_with_extreme_values() -> np.ndarray:
    """Return positive central data with four extreme values."""
    central_values = np.linspace(80, 120, 80)
    return np.concatenate((central_values, [5, 10, 200, 300]))


def test_iboxplot_removes_extreme_values(
    sample_with_extreme_values: np.ndarray,
) -> None:
    """Values beyond the iterative limits are removed."""
    result = iboxplot(
        sample_with_extreme_values,
        lognormal=False,
    )

    expected = np.linspace(80, 120, 80)
    np.testing.assert_allclose(result["truncated"], expected)
    assert result["truncation_points"] == {
        "lower": 80.0,
        "upper": 120.0,
    }
    assert result["percent_normal"] == 100.0
    assert result["lognormal"] is False


def test_iboxplot_detects_lognormal_data() -> None:
    """The distribution type is detected when not specified."""
    central_values = np.exp(np.linspace(3, 5, 80))
    values = np.concatenate((central_values, [1, 1000]))

    result = iboxplot(values)

    assert result["lognormal"] is True
    assert result["truncation_points"] == {
        "lower": 20.1,
        "upper": 148.4,
    }
    np.testing.assert_allclose(result["truncated"], central_values)


def test_iboxplot_can_disable_rounding() -> None:
    """Unrounded truncation points retain their original precision."""
    central_values = np.linspace(80.04, 119.96, 80)
    values = np.concatenate((central_values, [5, 300]))

    rounded = iboxplot(values, lognormal=False)
    unrounded = iboxplot(
        values,
        lognormal=False,
        apply_rounding=False,
    )

    assert rounded["truncation_points"] == {
        "lower": 80.0,
        "upper": 120.0,
    }
    assert unrounded["truncation_points"] == pytest.approx(
        {
            "lower": 80.04,
            "upper": 119.96,
        }
    )


def test_iboxplot_records_non_increasing_cycle_sizes(
    sample_with_extreme_values: np.ndarray,
) -> None:
    """Progress records each cycle until the sample size is stable."""
    result = iboxplot(
        sample_with_extreme_values,
        lognormal=False,
    )
    progress = result["progress"]

    assert isinstance(progress, list)
    sizes = [cycle["n"] for cycle in progress]

    assert progress[0] == {
        "cycle": "cycle0",
        "n": 84,
        "min": 5.0,
        "max": 300.0,
    }
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] == sizes[-2] == 80


def test_iboxplot_removes_missing_values(
    sample_with_extreme_values: np.ndarray,
) -> None:
    """NaN values are omitted before truncation."""
    values = np.append(sample_with_extreme_values, math.nan)

    result = iboxplot(values, lognormal=False)

    assert len(result["truncated"]) == 80
    assert result["progress"][0]["n"] == 84


def test_iboxplot_requires_at_least_40_values() -> None:
    """A small sample cannot be used for reference-limit estimation."""
    with pytest.raises(ValueError, match="at least 40 values"):
        iboxplot(np.linspace(80, 120, 39))


def test_iboxplot_rejects_multidimensional_input() -> None:
    """Only one-dimensional data are accepted."""
    values = np.ones((40, 2))

    with pytest.raises(ValueError, match="x must be one-dimensional"):
        iboxplot(values)


def test_iboxplot_rejects_non_numeric_input() -> None:
    """Text values cannot be used for truncation."""
    with pytest.raises(TypeError, match="numeric values"):
        iboxplot(["value"] * 40)  # type: ignore[list-item]


@pytest.mark.parametrize("invalid_value", [math.inf, -math.inf])
def test_iboxplot_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    """Infinite observations are rejected."""
    values = np.linspace(80, 120, 40)
    values[0] = invalid_value

    with pytest.raises(ValueError, match="only finite values"):
        iboxplot(values)


@pytest.mark.parametrize("invalid_value", [0.0, -1.0])
def test_iboxplot_rejects_non_positive_values(
    invalid_value: float,
) -> None:
    """Zero and negative observations are rejected."""
    values = np.linspace(80, 120, 40)
    values[0] = invalid_value

    with pytest.raises(ValueError, match="only positive values"):
        iboxplot(values)


@pytest.mark.parametrize("lognormal", [0, 1, "yes"])
def test_iboxplot_rejects_invalid_lognormal(
    lognormal: object,
) -> None:
    """The lognormal option must be boolean or None."""
    with pytest.raises(TypeError, match="boolean or None"):
        iboxplot(
            np.linspace(80, 120, 40),
            lognormal=lognormal,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "perc_trunc",
    [0.0, -1.0, 25.1, math.nan, math.inf],
)
def test_iboxplot_rejects_invalid_truncation_percentage(
    perc_trunc: float,
) -> None:
    """The truncation percentage must lie in the valid interval."""
    with pytest.raises(ValueError, match="perc_trunc must be greater"):
        iboxplot(
            np.linspace(80, 120, 40),
            perc_trunc=perc_trunc,
        )


def test_iboxplot_rejects_non_numeric_truncation_percentage() -> None:
    """The truncation percentage must be numeric."""
    with pytest.raises(TypeError, match="perc_trunc must be"):
        iboxplot(
            np.linspace(80, 120, 40),
            perc_trunc="2.5",  # type: ignore[arg-type]
        )


def test_iboxplot_rejects_invalid_rounding_option() -> None:
    """The rounding option must be boolean."""
    with pytest.raises(TypeError, match="apply_rounding must be"):
        iboxplot(
            np.linspace(80, 120, 40),
            apply_rounding=1,  # type: ignore[arg-type]
        )
