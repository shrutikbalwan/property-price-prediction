# Property geographic evaluation: findings

**Decision: retain the current point-prediction model, with clearer limitations.**
Boosting retains a modest advantage in the geographic stress tests. These results
do not support a claim that the app's existing 90% prediction interval will cover
90% of expensive homes or properties in new regions. No model weights, deployed
interval radius or calibration artifacts were changed.

## Evaluation design

Only the original 16,512 training rows were used. The 4,128 original historical-test
rows were excluded. Both model families were freshly fitted on exactly the same
fitting/calibration rows in each fold. Model settings were frozen before the study;
there was no hyperparameter search or tuning against the evaluation results.

- **Random control:** five shuffled row folds, random calibration rows.
- **Geographic blocks:** five folds of non-overlapping one-degree floor-based
  latitude/longitude blocks. Entire blocks are reserved for calibration as well.
- **Geographic buffer:** the same geographic evaluation folds, removing development
  rows within 10 km of any evaluation coordinate before fitting/calibration splits.

All preprocessing is fitted on fitting rows only. Each model/fold receives its own
split-conformal residual calibration and interval. Geographic folds can contain
several non-contiguous blocks; they are not five contiguous administrative regions.
Geographic calibration reserves 20% of groups, so its row fraction varies. Training
sample sizes therefore differ between modes; the contrast includes location shift
and sample-composition effects, not geography alone.

## Pooled out-of-fold comparison

RMSE is in the source dataset's house-value units; lower is better.

| Evaluation | Linear RMSE | Advanced RMSE | Advanced coverage | Advanced mean interval width |
|---|---:|---:|---:|---:|
| Random control | 68,713 | 48,115 | 89.7% | 144,579 |
| Geographic blocks | 71,494 | 68,100 | 89.9% | 216,177 |
| Geographic blocks + 10 km buffer | 71,307 | 69,571 | 89.7% | 215,623 |

Boosting wins four of five buffered folds and has lower pooled RMSE in all three
modes. Its advantage shrinks from about 30.0% on random splits to about 2.4% on the
buffered geographic evaluation. This is an observed paired comparison, not a
statistical significance claim or an external validation result.

The geographic intervals are substantially wider than the random-control intervals.
The roughly 90% aggregate coverage does **not** validate the existing app interval:
these are separately refitted and recalibrated fold models, not the saved model.

## Coverage by actual price

Buffered geographic evaluation, advanced model:

| Actual value band | Rows | RMSE | Interval coverage |
|---|---:|---:|---:|
| Under 150,000 | 6,056 | 46,785 | 97.3% |
| 150,000 to under 300,000 | 7,348 | 45,425 | 96.9% |
| 300,000 to under 450,000 | 2,060 | 107,335 | 63.8% |
| 450,000 and above | 1,048 | 162,779 | 46.0% |

Errors systematically underpredict the high-value bands: mean signed error is
about -76,608 in the 300k-450k band and -133,451 in the highest band. Source target
capping also limits interpretation of the most expensive observations. Price bands
use the actual target for evaluation; they cannot be used directly to choose a
prediction-time interval, since the actual target is unknown at that point.

## Geographic weaknesses

Examples with at least 100 evaluation records:

| Block (latitude / longitude floor) | Rows | Coverage | RMSE |
|---|---:|---:|---:|
| 34 / -120 | 298 | 71.5% | 85,493 |
| 37 / -123 | 2,172 | 76.4% | 91,480 |
| 34 / -119 | 2,629 | 82.5% | 90,877 |
| 35 / -121 | 111 | 84.7% | 64,747 |

For example, block 34/-120 means latitude [34,35) and longitude [-120,-119).
The offline interactive maps display block-average coordinate positions, counts,
errors and coverage. They are not a projected basemap. Small groups are flagged;
pooled coverage must not be substituted for their conditional coverage.

## What changed

- Added the reproducible geographic study and six integrity tests.
- Added offline interactive maps and a detailed HTML report.
- Added an explicit geographic/high-value coverage limitation to the property app
  explanation and model card.
- Kept all fitted application models and calibration artifacts unchanged. Widening
  intervals based on evaluation labels would turn evaluation into tuning.

## Verification

18 property tests passed, including six geographic-study tests. Checks cover exact
partition isolation, no historical-test contamination, group separation, the 10 km
buffer, complete out-of-fold coverage, source-target equality, calibration quantiles,
interval bounds, metric recomputation, subgroup counts and unchanged model hashes.
Repeating the full study reproduced the same pooled metrics. The offline map was
visually reviewed in Chrome and its comparison dropdown was exercised. The chart
uses a fixed 1200px width to preserve labels; narrow windows scroll horizontally.

Evidence: verification/geographic_checks.json and verification/advanced_pytest.txt.
All protocol settings, row partitions, calibration residuals and predictions are in
outputs/geographic_study/. No external data or GitHub publication was involved.

## Reproduce and read

From this project folder:

```powershell
python -m pip install -r requirements-study.txt
python -m ml_projects.geographic_study
python -m ml_projects.advanced_verify
```

Open PROPERTY_GEOGRAPHIC_REPORT.html, or open
outputs/geographic_study/geographic_maps.html directly for the standalone map.
Plotly is bundled into the generated map; viewing it needs no internet connection.
The study was run with the laptop's installed dependencies; a clean installation
has not yet been verified.

This remains an internal stress test of previously studied data. Row-level conformal
coverage relies on exchangeability, which spatial dependence and location shift can
violate. Holding out calibration groups does not create a group-conformal guarantee.
A future experiment could test feature-dependent interval widths and geographic
features, using a new protocol and preferably new regional/time-separated data.

Method reference: [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
