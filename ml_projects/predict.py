"""Batch inference from a CSV of unlabeled records.

Example: python -m ml_projects.predict --project churn --input customers.csv --output predictions.csv
Only load trusted model files. Joblib models are executable Python serialization.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import TARGETS
from .train import output_dir


def predict_frame(project, frame):
    if project not in TARGETS:
        raise ValueError('Unknown project: '+str(project))
    if not isinstance(frame,pd.DataFrame):
        raise TypeError('Prediction input must be a pandas DataFrame')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate input column names are not allowed')
    out = output_dir(project)
    manifest = json.loads((out/'run.json').read_text(encoding='utf-8'))
    required = manifest['input_columns']
    if TARGETS[project] in frame:
        raise ValueError('Prediction inputs must not contain the target column')
    missing = sorted(set(required)-set(frame.columns))
    if missing:
        raise ValueError('Missing input columns: '+', '.join(missing))
    if frame.empty:
        raise ValueError('Prediction input contains no rows')
    model = joblib.load(out/'model.joblib')
    X = frame[required].copy().replace(r'^\s*$',np.nan,regex=True)
    numeric = {
        'property':[c for c in required if c!='ocean_proximity'],
        'churn':['SeniorCitizen','tenure','MonthlyCharges','TotalCharges'],
        'disease':['active','age','alco','ap_hi','ap_lo','cholesterol','gender',
                   'gluc','height','smoke','weight'],
    }[project]
    for column in numeric:
        converted=pd.to_numeric(X[column],errors='coerce')
        invalid=X[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f'Invalid numeric value in {column}')
        if np.isinf(converted.to_numpy(dtype=float,na_value=np.nan)).any():
            raise ValueError(f'Infinite value in {column}')
        X[column]=converted
    result = pd.DataFrame(index=frame.index)
    for identifier in ['customerID','id']:
        if identifier in frame:
            result[identifier] = frame[identifier]
    result['prediction'] = model.predict(X)
    if project != 'property' and hasattr(model,'predict_proba'):
        result['probability'] = model.predict_proba(X)[:,1]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='property',choices=['property'])
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Output must differ from input')
    result=predict_frame(args.project,pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(f'Wrote {len(result)} predictions to {args.output}')
