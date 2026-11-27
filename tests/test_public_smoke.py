"""Public CI checks: synthetic inputs only; no workbooks or saved models."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from ml_projects.advanced import conformal_radius, predict
from ml_projects.app import csv_records

def test_csv_quoted_and_bom():
    assert csv_records('\ufeffname,value\n"a,b",2') == [{'name': 'a,b', 'value': '2'}]

@pytest.mark.parametrize('text', ['a,a\n1,2', 'a,b\n1', 'a,b\n1,2,3'])
def test_csv_rejects_ambiguous_rows(text):
    with pytest.raises(ValueError):
        csv_records(text)

def test_csv_batch_limit():
    assert len(csv_records('x\n' + '1\n' * 5000)) == 5000
    with pytest.raises(ValueError, match='5000'):
        csv_records('x\n' + '1\n' * 5001)

def test_conformal_finite_sample_rank():
    assert conformal_radius([4, 1, 3, 2], .6) == 3
    with pytest.raises(ValueError):
        conformal_radius([1], .99)

def test_synthetic_inference_and_validation():
    X = pd.DataFrame({'x': [0., 1., 2., 3.]})
    model = LinearRegression().fit(X, [1., 3., 5., 7.])
    artifact = {'project': 'property', 'model': model, 'radius': 2.,
                'schema': [{'name': 'x', 'numeric': True, 'min': 0., 'max': 3.}]}
    result = predict(artifact, [{'x': 2.}])
    assert result['results'][0] == pytest.approx({'prediction': 5., 'lower': 3., 'upper': 7.})
    assert predict(artifact, [{'x': 4.}])['warnings']
    with pytest.raises(ValueError, match='Missing columns'):
        predict(artifact, [{}])
    with pytest.raises(ValueError, match='Invalid numeric'):
        predict(artifact, [{'x': np.inf}])
