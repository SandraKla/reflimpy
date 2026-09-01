"""Tests for conf_int95()."""

import math

import pytest

from reflimpy import conf_int95


def test_conf_int95_without_lognormal_or_rounding() -> None:
    """Confidence intervals match known untransformed results."""
    result = conf_int95(
        n=100,
        lower_limit=2.5,
        upper_limit=97.5,
        lognormal=False,
        apply_rounding=False,
    )

    assert result["lower_lim_low"] == pytest.approx(
        -10.708590190297507
    )
    assert result["lower_lim_upp"] == pytest.approx(
        42.30630713823991
    )
    assert result["upper_lim_low"] == pytest.approx(
        57.69369286176009
    )
    assert result["upper_lim_upp"] == pytest.approx(
        110.70859019029751
    )
    assert result["n"] == 100


def test_conf_int95_with_lognormal_transformation() -> None:
    """Lognormal results are transformed back to the original scale."""
    result = conf_int95(
        n=100,
        lower_limit=2.5,
        upper_limit=97.5,
        lognormal=True,
        apply_rounding=False,
    )

    assert result["lower_lim_low"] == pytest.approx(
        1.5021797375427288
    )
    assert result["lower_lim_upp"] == pytest.approx(
        11.604273899057208
    )
    assert result["upper_lim_low"] == pytest.approx(
        21.005191890532977
    )
    assert result["upper_lim_upp"] == pytest.approx(
        162.26420441453106
    )
    assert result["n"] == 100


def test_conf_int95_applies_rounding() -> None:
    """All confidence limits use the selected number of digits."""
    result = conf_int95(
        n=100,
        lower_limit=2.5,
        upper_limit=97.5,
    )

    assert result == {
        "lower_lim_low": 1.5,
        "lower_lim_upp": 11.6,
        "upper_lim_low": 21.01,
        "upper_lim_upp": 162.26,
        "n": 100,
    }


def test_conf_int95_returns_expected_keys() -> None:
    """The result contains four interval limits and the sample size."""
    result = conf_int95(
        n=100,
        lower_limit=2.5,
        upper_limit=97.5,
    )

    assert list(result) == [
        "lower_lim_low",
        "lower_lim_upp",
        "upper_lim_low",
        "upper_lim_upp",
        "n",
    ]


@pytest.mark.parametrize("n", [0, -1])
def test_conf_int95_rejects_invalid_sample_size(n: int) -> None:
    """The sample size must be greater than zero."""
    with pytest.raises(
        ValueError,
        match="n must be greater than zero",
    ):
        conf_int95(
            n=n,
            lower_limit=2.5,
            upper_limit=97.5,
        )


@pytest.mark.parametrize("n", [True, 10.5, "100"])
def test_conf_int95_rejects_non_integer_sample_size(
    n: object,
) -> None:
    """The sample size must be an integer."""
    with pytest.raises(
        TypeError,
        match="n must be an integer",
    ):
        conf_int95(
            n=n,  # type: ignore[arg-type]
            lower_limit=2.5,
            upper_limit=97.5,
        )


@pytest.mark.parametrize(
    ("lower_limit", "upper_limit"),
    [
        (10.0, 10.0),
        (20.0, 10.0),
    ],
)
def test_conf_int95_rejects_invalid_limit_order(
    lower_limit: float,
    upper_limit: float,
) -> None:
    """The upper limit must be higher than the lower limit."""
    with pytest.raises(
        ValueError,
        match="upper_limit must be higher than lower_limit",
    ):
        conf_int95(
            n=100,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )


@pytest.mark.parametrize(
    ("lower_limit", "upper_limit"),
    [
        (math.nan, 10.0),
        (1.0, math.inf),
        (1.0, -math.inf),
    ],
)
def test_conf_int95_rejects_non_finite_limits(
    lower_limit: float,
    upper_limit: float,
) -> None:
    """NaN and infinite limits are rejected."""
    with pytest.raises(
        ValueError,
        match="limits must be finite",
    ):
        conf_int95(
            n=100,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
        )


def test_conf_int95_requires_positive_lognormal_limits() -> None:
    """A logarithmic transformation requires positive limits."""
    with pytest.raises(
        ValueError,
        match="limits must be positive",
    ):
        conf_int95(
            n=100,
            lower_limit=0,
            upper_limit=10,
            lognormal=True,
        )