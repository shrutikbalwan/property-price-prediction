"""Train and evaluate the assignment models with isolated test partitions.

Run: python -m ml_projects.train --project property (or churn/disease/all).
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .data import ROOT, FILES, TARGETS, PrepareFeatures, read_dataset, split_target

FOLD_SEED = 42
DIRECTORIES = {'property': '.'}


def output_dir(project):
    result = ROOT / DIRECTORIES[project] / 'outputs'
    result.mkdir(parents=True, exist_ok=True)
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding='utf-8')


def pipeline(project, X, estimator, simple=False, expanded=False, clean_bp=True):
    prepare = PrepareFeatures(project, expanded=expanded, clean_bp=clean_bp)
    prepared = prepare.fit_transform(X)
    numeric = list(prepared.select_dtypes(include='number').columns)
    categorical = [c for c in prepared if c not in numeric]
    if simple:
        numeric, categorical = ['median_income'], []
    numeric_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                             ('scale', StandardScaler())])
    categorical_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])
    transformer = ColumnTransformer([('numeric', numeric_pipe, numeric),
                                      ('categorical', categorical_pipe, categorical)])
    return Pipeline([('prepare', prepare), ('preprocess', transformer), ('model', estimator)])


def candidates(project, X):
    if project == 'property':
        return [
            ('mean_baseline', pipeline(project, X, DummyRegressor()), {}),
            ('simple_linear', pipeline(project, X, LinearRegression(), simple=True), {}),
            ('multiple_linear', pipeline(project, X, LinearRegression()), {})]
    result = [
        ('majority_baseline', pipeline(project, X, DummyClassifier(strategy='most_frequent')), {}),
        ('logistic', pipeline(project, X, LogisticRegression(max_iter=3000, random_state=42)),
         {'model__C':[0.1, 1.0, 10.0], 'model__class_weight':[None, 'balanced']}),
        ('decision_tree', pipeline(project, X, DecisionTreeClassifier(random_state=42)),
         {'model__max_depth':[4, 8, 12], 'model__min_samples_leaf':[10, 50]}),
        ('random_forest', pipeline(project, X, RandomForestClassifier(n_estimators=120,
                                                    n_jobs=1, random_state=42)),
         {'model__max_depth':[8, None], 'model__min_samples_leaf':[5, 20]})]
    if project == 'disease':
        result.append(('linear_svm', pipeline(project, X, LinearSVC(dual=False,
                    max_iter=5000, random_state=42)), {'model__C':[0.01, 0.1, 1.0]}))
    return result


def scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(project, y, predicted, ranking=None):
    if project == 'property':
        mse = mean_squared_error(y, predicted)
        return {'MAE': mean_absolute_error(y, predicted), 'MSE': mse,
                'RMSE': float(np.sqrt(mse)), 'R2': r2_score(y, predicted)}
    return {'accuracy':accuracy_score(y,predicted),
            'precision':precision_score(y,predicted,zero_division=0),
            'recall':recall_score(y,predicted,zero_division=0),
            'F1':f1_score(y,predicted,zero_division=0),
            'ROC_AUC':roc_auc_score(y,ranking),
            'average_precision':average_precision_score(y,ranking)}


def train(project):
    out = output_dir(project)
    frame = read_dataset(project)
    X, y = split_target(frame, project)
    train_ids, test_ids = train_test_split(X.index, test_size=0.2, random_state=42,
                                         stratify=y if project != 'property' else None)
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    write_json(out/'split.json', {'train_rows':train_ids.tolist(), 'test_rows':test_ids.tolist(),
                                 'seed':42, 'test_fraction':0.2})
    X_train.join(y_train).describe(include='all').to_csv(out/'training_summary.csv')
    X_train.isna().sum().rename('missing').to_csv(out/'training_missing.csv')
    X_train.select_dtypes(include='number').join(y_train).corr().to_csv(out/'training_correlations.csv')
    cv = (KFold(5, shuffle=True, random_state=42) if project == 'property'
          else StratifiedKFold(5, shuffle=True, random_state=42))
    scoring = 'neg_root_mean_squared_error' if project == 'property' else 'f1'
    fitted, rows = {}, []
    for name, estimator, parameters in candidates(project, X_train):
        start = time.monotonic()
        print(f'{project}: fitting {name}', flush=True)
        search = GridSearchCV(estimator, parameters, scoring=scoring, cv=cv,
                              n_jobs=1, error_score='raise', return_train_score=True)
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        pd.DataFrame(search.cv_results_).to_csv(out/f'{name}_cv.csv', index=False)
        rows.append({'model':name, 'cv_score':float(search.best_score_),
                     'cv_std':float(search.cv_results_['std_test_score'][search.best_index_]),
                     'parameters':search.best_params_, 'seconds':time.monotonic()-start})
        print(f'{project}: {name} validation={search.best_score_:.5f}', flush=True)
    # Select BEFORE any test-set evaluation. Baseline is a comparator, not a candidate.
    winner = max((r for r in rows if 'baseline' not in r['model']), key=lambda r:r['cv_score'])['model']
    write_json(out/'selection.json', {'winner':winner, 'criterion':scoring,
                                     'candidates':rows, 'selection_partition':'training CV only'})
    # Supplemental quality/feature checks never select from test scores.
    if project == 'disease':
        sensitivity = []
        for expanded, clean_bp in [(False,False),(True,True)]:
            estimator = pipeline(project,X_train,LogisticRegression(max_iter=3000),
                                 expanded=expanded,clean_bp=clean_bp)
            search = GridSearchCV(estimator, {'model__C':[0.1,1.0]}, scoring='f1',cv=cv,
                                  n_jobs=1,error_score='raise')
            search.fit(X_train,y_train)
            sensitivity.append({'expanded_features':expanded,'clean_bp':clean_bp,
                                'cv_f1':float(search.best_score_)})
        write_json(out/'sensitivity.json', sensitivity)
    evaluations = []
    for name, model in fitted.items():
        prediction = model.predict(X_test)
        ranking = None if project == 'property' else scores(model,X_test)
        evaluations.append({'model':name, **evaluate(project,y_test,prediction,ranking)})
        if project != 'property':
            pd.DataFrame(confusion_matrix(y_test,prediction,labels=[0,1]),
                         index=['actual_0','actual_1'],columns=['predicted_0','predicted_1']).to_csv(out/f'{name}_confusion.csv')
        if name == winner:
            predictions = pd.DataFrame({'source_row':test_ids,'actual':y_test.to_numpy(),
                                         'predicted':prediction})
            if ranking is not None:
                predictions['probability' if hasattr(model,'predict_proba') else 'decision_score'] = ranking
            predictions.to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(evaluations).to_csv(out/'test_metrics.csv',index=False)
    best = fitted[winner]
    joblib.dump(best, out/'model.joblib',compress=3)
    # Reload and verify numerical equality before reporting the serialized model.
    np.testing.assert_allclose(best.predict(X_test), joblib.load(out/'model.joblib').predict(X_test))
    sample = X_test.sample(min(1500,len(X_test)),random_state=42)
    importance = permutation_importance(best,sample,y_test.loc[sample.index],
                                        scoring=scoring,n_repeats=3,random_state=42,n_jobs=1)
    pd.DataFrame({'feature':X.columns,'importance':importance.importances_mean,
                  'std':importance.importances_std}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    if hasattr(best.named_steps['model'],'coef_'):
        pd.DataFrame({'feature':best.named_steps['preprocess'].get_feature_names_out(),
                      'coefficient':np.ravel(best.named_steps['model'].coef_)}).to_csv(out/'coefficients.csv',index=False)
    # Inference demonstration uses held-out records, explicitly not new observations.
    example = X_test.head(5)
    example.to_csv(out/'example_inputs.csv',index=False)
    example_predictions = pd.DataFrame({'prediction':best.predict(example)})
    if project != 'property' and hasattr(best,'predict_proba'):
        example_predictions['probability']=best.predict_proba(example)[:,1]
    example_predictions.to_csv(out/'example_predictions.csv',index=False)
    write_json(out/'run.json', {'project':project,'winner':winner,'rows':len(frame),
        'train_rows':len(train_ids),'test_rows':len(test_ids),'input_columns':X.columns.tolist(),
        'source_sha256':hashlib.sha256((ROOT/'datasets'/FILES[project]).read_bytes()).hexdigest(),
        'python':platform.python_version(),'sklearn':sklearn.__version__,
        'numpy':np.__version__,'pandas':pd.__version__,
        'example_note':'Held-out rows for inference demonstration; not newly collected observations.'})
    print(f'{project}: complete; selected {winner}',flush=True)
    return evaluations


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',choices=[*FILES,'all'],default='all')
    args=parser.parse_args()
    for project in FILES if args.project == 'all' else [args.project]:
        train(project)
