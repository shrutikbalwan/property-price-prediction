# Advanced model card

## Intended use

Local educational exploration, reproducible evaluation and batch inference.

## Training and selection

Candidate families: linear model and histogram gradient boosting. Preprocessing
is fitted inside each three-fold search. Selection uses neg_root_mean_squared_error.
The selected model is hist_gradient_boosting. Configuration is in config.json.
Partition row IDs are in outputs/advanced/partitions.json. Calibration and
threshold selection use separate rows from model fitting. The original assignment
models and their reports remain in outputs/ and analysis.ipynb.

## Evaluation history

The original test set was inspected during earlier development. Results here are a historical benchmark, not new independent external validation.
No new external data was acquired. Bootstrap intervals resample rows and do not
account for selection uncertainty or geographic dependence. Subgroup comparisons
are descriptive; small samples and single-class groups are flagged.

## Limitations

Historic aggregate house values are not current property appraisals. Geographic CV uses coarse one-degree groups without a spatial buffer. Conformal coverage assumes exchangeability; intervals are marginal and can extend below zero. Target capping in the source data limits tail interpretation.

## Operational scope

The app binds to loopback only, stores no submitted records, limits batch size,
and validates numeric inputs. Range warnings are heuristics, not a fitted drift
detector. Joblib models must only be loaded from trusted sources. This local demo
does not include authentication, production hosting or clinical approval.

## Evidence

- outputs/advanced/summary.json: metrics, source/model hashes and configuration
- outputs/advanced/calibration_validation.json: calibration comparison (classification)
- advanced_report.html: readable model evidence
- verification/advanced_checks.json: final automated checks

## Geographic stress evaluation

Paired five-fold refits on original training rows show buffered geographic RMSE of
69,571 versus 71,307 for the linear baseline. Boosting wins four of five buffered
folds. Recalibrated fold intervals have 89.7% pooled coverage but only 46.0% coverage
for actual values at least 450,000. These are not evaluations of the saved interval.
See PROPERTY_GEOGRAPHIC_FINDINGS.md; do not interpret nominal marginal coverage as
conditional coverage for an individual location or price band. Application model
weights and the deployed interval radius remain unchanged.
