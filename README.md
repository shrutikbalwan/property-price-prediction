# Property Price Prediction

Explore regression models for aggregate house values, inspect uncertainty intervals, and stress-test performance across geographic areas.

[Quick demo](DEMO.md) ? [Study findings](PROPERTY_GEOGRAPHIC_FINDINGS.md) ? [Model card](MODEL_CARD.md) ? [Reproduce](GITHUB_PREPARATION.md) ? [CI](https://github.com/shrutikbalwan/property-price-prediction/actions)

![Actual synthetic-demo input form and result](docs/screenshots/demo-result.jpg)

*Actual app screenshot using invented formulas. Demo outputs are not trained-model predictions.*

## Quick start

Python **3.13** is the tested version. From this repository folder on Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\.venv\Scripts\python.exe -m ml_projects.app --demo --port 8501
```

Open **http://127.0.0.1:8501** and look for **SYNTHETIC DEMO**. Stop with Ctrl+C.
On macOS/Linux, use `python3.13` and `.venv/bin/python`.
Demo mode needs no account, API key, original workbook or trained model.
Follow the [illustrated walkthrough](docs/WALKTHROUGH.md).

## Features

- Regression comparison with training-only cross-validation.
- Split-conformal intervals with documented coverage limits.
- Geographic evaluation with a 10 km exclusion buffer and price-band analysis.
- Single-record prediction, CSV batches and downloadable results.
- Input validation, same-origin checks and a 5,000-row / 2 MB request limit.
- An explicit fictional demo that never loads or modifies trained artifacts.

| Mode | Requirements | Output |
|---|---|---|
| `--demo` | Runtime dependencies only | Invented formula values marked as synthetic |
| Default trained mode | Authorized data and regenerated model artifacts | Saved model predictions and historical evidence |

The demo is an interface simulator, not a model trained on synthetic data.
Missing trained artifacts cause an explicit error; there is no silent fallback.

## Historical model results

| Metric | Value |
|---|---:|
| RMSE | 47,623.11 |
| MAE | 31,794.08 |
| R? | 0.8269 |
| Marginal interval coverage | 90.41% |

The original test set was inspected during earlier development. These are
**historical benchmark results**, not fresh external validation. They describe
the separate trained pipeline, not the fictional demo.

These are historical aggregate house values, not current property appraisals. Geographic stress tests showed weaker performance in unfamiliar locations and poor interval coverage for expensive homes. Marginal coverage does not guarantee coverage for an individual property.

See [the study findings](PROPERTY_GEOGRAPHIC_FINDINGS.md) and [MODEL_CARD.md](MODEL_CARD.md)
for methodology, intended use and limitations.

## Screenshots

| Single record | CSV batch workflow |
|---|---|
| ![Labelled fictional result](docs/screenshots/demo-result.jpg) | ![Synthetic batch interface](docs/screenshots/demo-batch.jpg) |

All screenshots use fictional demo data. [Walkthrough and presentation script](docs/WALKTHROUGH.md).

## Architecture

```text
Browser: index.html + app.js + demo.js
                  |
         loopback HTTP JSON / CSV
                  |
        ml_projects/app.py
           /             \
  explicit --demo     default trained mode
      demo.py           advanced.py
  invented formulas   saved sklearn pipeline
```

Training and evaluation studies are separate CLI workflows. The browser cannot
trigger training. Submitted inputs are processed locally and not saved by the
server. This educational app is not a production multi-user service.

## Reproduce the trained project

Follow [GITHUB_PREPARATION.md](GITHUB_PREPARATION.md) for dependency installation,
authorized workbook placement and the full training order. The study entry point
is `python -m ml_projects.geographic_study`, after generating the baseline and advanced
artifacts as documented there.

Original data, model binaries, row-level outputs, executed notebooks and generated
HTML reports are excluded from Git. A fresh clone provides the source to
regenerate them. Interactive notebooks additionally need
`requirements-notebook.txt`; regenerate notebooks before opening them.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests/test_public_smoke.py tests/test_demo.py -q
```

The **12 dataset-free tests** run in GitHub Actions on Windows and Ubuntu with
Python 3.13. [The demo implementation CI run passed](https://github.com/shrutikbalwan/property-price-prediction/actions/runs/45939419397);
check the latest Actions run for subsequent commits.

The full local suite passed **30 tests** with authorized data and generated
artifacts.
After reproduction, run `python -m pytest -q` for the full suite. CI smoke checks
do not validate real-world model accuracy.

## Data and licensing

See [DATASET.md](DATASET.md) for provenance and the unconfirmed redistribution
permission. Original workbooks are not uploaded. No code license has been selected;
this repository does not currently grant an explicit open-source license.
