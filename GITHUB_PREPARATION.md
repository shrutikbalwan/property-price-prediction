# GitHub preparation and reproduction

This folder is a standalone repository. See the repository Actions tab for hosted CI results.

For immediate use without data or training, follow [the synthetic demo guide](DEMO.md).

## What a fresh clone contains

Source code, web UI, tests, configuration and written findings are included.
Local virtual environments, source datasets, model binaries, row-level outputs,
executed notebooks and generated HTML reports are excluded. These local files
remain on disk. Links to generated reports in the README work after reproduction.

Dataset redistribution permission is unconfirmed; see DATASET.md. Obtain the
authorized assignment workbook and place it in datasets/ using the filename in
ml_projects/data.py. No public dataset URL or license has been verified.
No open-source license has been selected for the project code.

## Install and test without the dataset

Use Python 3.13 from this directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-study.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_public_smoke.py -q
```

The GitHub workflow runs this synthetic test suite on Windows and Ubuntu.
It does not certify the trained models, reproduce the studies or access private
datasets. Initial smoke CI passed on GitHub; later changes need their own green run.

## Reproduce the complete local project

After installing dependencies and supplying the authorized workbook, run these
commands with the virtual environment's Python (shown as python below):

```powershell
python -m ml_projects.train --project property
python -m ml_projects.deliver --project property
python -m ml_projects.advanced --train
python -m ml_projects.advanced_deliver
python -m ml_projects.geographic_study
python -m pytest -q
python -m ml_projects.advanced_verify
python -m ml_projects.app --port 8501
```

Open http://127.0.0.1:8501. Training and nested studies can take considerable
time. Install requirements-notebook.txt separately for interactive notebook kernels.
The verification command compares earlier recorded metrics when those records
exist; a fresh clone does not contain those local verification records.

## Validation evidence

The user supplied successful Windows Python 3.13 clean-install terminal output:
pip check passed, 18 existing tests passed and advanced acceptance checks passed.
This evidence predates the additional public smoke tests.


## Before publication

Review the Git candidate list and secret-scan results. Do not force-add ignored
datasets or generated artifacts. Confirm dataset/source licensing before distributing
them; a project code license cannot grant rights to the supplied data.
Publish this folder as its own repository, not the parent workspace.

Workflow references: [checkout](https://github.com/actions/checkout) and
[setup-python](https://github.com/actions/setup-python).
