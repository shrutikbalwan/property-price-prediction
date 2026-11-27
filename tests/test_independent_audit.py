"""Recompute every selected-model metric from source rows and loaded models."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, r2_score)
from ml_projects.data import ROOT, read_dataset, split_target
from ml_projects.train import DIRECTORIES


@pytest.mark.parametrize('project',list(DIRECTORIES))
def test_all_predictions_metrics_and_cv_selection(project):
    out=ROOT/DIRECTORIES[project]/'outputs'
    X,y=split_target(read_dataset(project),project)
    saved=pd.read_csv(out/'test_predictions.csv')
    ids=saved.source_row.to_numpy()
    np.testing.assert_array_equal(saved.actual,y.loc[ids])
    model=joblib.load(out/'model.joblib')
    predictions=model.predict(X.loc[ids])
    np.testing.assert_allclose(predictions,saved.predicted)
    selection=json.loads((out/'selection.json').read_text())
    candidates=selection['candidates']
    chosen=max((r for r in candidates if 'baseline' not in r['model']),key=lambda r:r['cv_score'])
    assert chosen['model']==selection['winner']
    for candidate in candidates:
        cv=pd.read_csv(out/(candidate['model']+'_cv.csv'))
        assert cv.mean_test_score.max()==pytest.approx(candidate['cv_score'])
        assert all(f'split{i}_test_score' in cv for i in range(5))
        assert np.isfinite(cv.mean_test_score).all()
    recorded=pd.read_csv(out/'test_metrics.csv').set_index('model').loc[selection['winner']]
    if project=='property':
        mse=mean_squared_error(y.loc[ids],predictions)
        independent={'MSE':mse,'RMSE':np.sqrt(mse),'MAE':mean_absolute_error(y.loc[ids],predictions),
                     'R2':r2_score(y.loc[ids],predictions)}
    else:
        probabilities=model.predict_proba(X.loc[ids])[:,1]
        np.testing.assert_allclose(probabilities,saved.probability)
        independent={'accuracy':accuracy_score(y.loc[ids],predictions),
            'precision':precision_score(y.loc[ids],predictions),
            'recall':recall_score(y.loc[ids],predictions),
            'F1':f1_score(y.loc[ids],predictions),'ROC_AUC':roc_auc_score(y.loc[ids],probabilities),
            'average_precision':average_precision_score(y.loc[ids],probabilities)}
    for metric,value in independent.items():
        assert recorded[metric]==pytest.approx(value)
