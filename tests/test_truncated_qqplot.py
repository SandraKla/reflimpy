import numpy as np
import pytest

from reflimpy import truncated_qqplot


def test_truncated_qqplot_matches_r_for_normal_data():
    values = np.linspace(80, 120, 200)

    output = truncated_qqplot(
        values,
        lognormal=False,
        apply_rounding=False,
    )
    result = output["result"]

    assert output["lognormal"] is False
    assert result["mean"] == pytest.approx(100.0338968390)
    assert result["sd"] == pytest.approx(14.0594810681)
    assert result["lower_lim"] == pytest.approx(72.4773139456)
    assert result["upper_lim"] == pytest.approx(127.5904797325)


def test_truncated_qqplot_matches_r_for_lognormal_data():
    values = np.exp(
        np.linspace(np.log(20), np.log(150), 200)
    )

    output = truncated_qqplot(
        values,
        lognormal=True,
        apply_rounding=False,
    )
    result = output["result"]

    assert output["lognormal"] is True
    assert result["mean_log"] == pytest.approx(4.005)
    assert result["sd_log"] == pytest.approx(0.708)
    assert result["lower_lim"] == pytest.approx(13.6917738312)
    assert result["upper_lim"] == pytest.approx(219.8591925558)


def test_truncated_qqplot_applies_rounding():
    values = np.linspace(80, 120, 200)

    output = truncated_qqplot(
        values,
        lognormal=False,
    )

    assert output["result"]["lower_lim"] == 72.0
    assert output["result"]["upper_lim"] == 128.0


def test_truncated_qqplot_detects_distribution():
    values = np.exp(
        np.linspace(np.log(20), np.log(150), 200)
    )

    output = truncated_qqplot(values)

    assert output["lognormal"] is True


def test_truncated_qqplot_removes_missing_values():
    values = np.append(np.linspace(80, 120, 200), np.nan)

    output = truncated_qqplot(
        values,
        lognormal=False,
    )

    assert output["result"] is not None


def test_truncated_qqplot_warns_for_small_sample():
    values = np.linspace(80, 120, 100)

    with pytest.warns(UserWarning, match="At least 200"):
        output = truncated_qqplot(values)

    assert output == {"result": None, "lognormal": None}


@pytest.mark.parametrize(
    "values, expected_message",
    [
        (
            np.linspace(1, 10, 39),
            "at least 40 values",
        ),
        (
            [1, 2, 3, 0] * 50,
            "only positive values",
        ),
        (
            [1, 2, 3, -1] * 50,
            "only positive values",
        ),
    ],
)
def test_truncated_qqplot_rejects_invalid_values(
    values,
    expected_message,
):
    with pytest.raises(ValueError, match=expected_message):
        truncated_qqplot(values)


@pytest.mark.parametrize("perc_trunc", [0, 50, np.inf])
def test_truncated_qqplot_rejects_invalid_truncation(
    perc_trunc,
):
    values = np.linspace(80, 120, 200)

    with pytest.raises(ValueError, match="between 0 and 50"):
        truncated_qqplot(values, perc_trunc=perc_trunc)
