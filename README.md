# Using reflimR from Python

This repository contains a small example of calling the R package
[`reflimR`](https://cran.r-project.org/package=reflimR) from Python with
[`rpy2`](https://rpy2.github.io/).

The example:

1. loads the `livertests` dataset provided by `reflimR`;
2. converts the R data frame to a pandas DataFrame;
3. selects and cleans bilirubin (`BIL`) measurements;
4. calls `reflimR::reflim()`;
5. converts the estimated limits back to pandas objects;
6. saves the results as `output/bilirubin_reference_limits.csv`.

## Requirements

- Python 3.12 (tested)
- R 4.6.1 (tested)
- R package `reflimR`

Install the R package from an R console:

```r
install.packages("reflimR")
```

Create a Python virtual environment and install the Python dependencies on
Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -X utf8 reflimr_from_python.py
```

The `-X utf8` option avoids a Windows encoding issue when `rpy2` starts R.

## Expected output

For the bilirubin example, the estimated reference limits are:

```text
lower limit: 2.85
upper limit: 17.17
```

The complete result table is written to
`output/bilirubin_reference_limits.csv`.
