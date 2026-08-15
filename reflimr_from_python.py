"""Run a reflimR reference-limit analysis from Python."""

from pathlib import Path

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import data, importr
from rpy2.robjects.vectors import FloatVector, Vector


def named_r_vector_to_series(vector: Vector) -> pd.Series:
    """Convert a named numeric R vector to a pandas Series."""
    return pd.Series(
        data=list(vector),
        index=list(vector.names),
        dtype=float,
    )


def main() -> None:
    """Load example data, run reflimR, and save the results."""
    reflim_r = importr("reflimR")
    livertests_r = data(reflim_r).fetch("livertests")["livertests"]

    # Convert the R data frame to a pandas DataFrame.
    with (ro.default_converter + pandas2ri.converter).context():
        livertests = ro.conversion.get_conversion().rpy2py(livertests_r)

    print("Example data loaded from reflimR:")
    print(livertests.head())

    # Select the bilirubin measurements and remove invalid values.
    bilirubin = pd.to_numeric(livertests["BIL"], errors="coerce").dropna()

    # Convert the values to an R vector and call reflim().
    result_r = reflim_r.reflim(
        FloatVector(bilirubin.tolist()),
        plot_it=False,
        main="Bilirubin reference limits",
        xlab="micromol/L",
    )

    limits = named_r_vector_to_series(result_r.rx2("limits"))
    confidence_intervals = named_r_vector_to_series(
        result_r.rx2("confidence.int")
    )

    print("\nEstimated reference limits:")
    print(limits)
    print("\n95% confidence intervals:")
    print(confidence_intervals)

    results = (
        pd.concat(
            {
                "limits": limits,
                "confidence_interval": confidence_intervals,
            },
            names=["section", "metric"],
        )
        .rename("value")
        .reset_index()
    )
    results.insert(0, "analyte", "BIL")

    # Save the Python result table as CSV.
    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "bilirubin_reference_limits.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
