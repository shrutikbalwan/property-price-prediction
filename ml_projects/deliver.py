"""Generate EDA figures, readable reports and standard Jupyter notebooks."""
import argparse
import contextlib
import hashlib
import io
import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts
from .data import ROOT, FILES, TARGETS, read_dataset, split_target, PrepareFeatures
from .predict import predict_frame
from .train import DIRECTORIES, output_dir, write_json

TITLES={'property':'Property Price Prediction','churn':'Customer Churn Prediction','disease':'Early Disease Detection'}
DICTIONARIES={'property':'1DrAqBM1YGmxFK7uWA_mTKVMxEC5kqVRh',
              'churn':'1u67hVM82TDqi0KmHeesPVtCMFYky8hvp','disease':'1TWiUJQbpVsfqx9iBgxk9frHKSa0VADKo'}


def audit(project, frame):
    result={'rows':len(frame),'columns':list(frame),
            'missing':frame.isna().sum().astype(int).to_dict(),
            'exact_duplicates':int(frame.duplicated().sum())}
    if project=='property':
        result['target_maximum']=float(frame.median_house_value.max())
        result['rows_at_maximum']=int((frame.median_house_value==frame.median_house_value.max()).sum())
        result['units']='Income and price units not specified in supplied dictionary.'
    if project=='churn':
        result['class_counts']=frame.Churn.value_counts().to_dict()
        result['duplicate_ids']=int(frame.customerID.duplicated().sum())
    if project=='disease':
        result['class_counts']=frame.disease.value_counts().to_dict()
        result['duplicate_ids']=int(frame.id.duplicated().sum())
        result['feature_duplicates_excluding_id_date']=int(frame.drop(columns=['id','date']).duplicated().sum())
        prepared=PrepareFeatures('disease').fit_transform(frame)
        result['bp_rows_marked_missing']=int(prepared.ap_hi.isna().sum())
        result['bp_rule']='Nonpositive, systolic >300, diastolic >200, or diastolic >= systolic; both readings set missing; rows retained. Broad quality assumptions, not clinical cutoffs.'
        # Excel source uses 1900 date system. These serials require 1899-12-30 origin.
        serial=pd.to_numeric(frame.date,errors='coerce')
        parsed=pd.Series(pd.NaT,index=frame.index,dtype='datetime64[ns]')
        numeric=serial.notna()
        parsed.loc[numeric]=pd.to_datetime(serial[numeric],unit='D',origin='1899-12-30')
        parsed.loc[~numeric]=pd.to_datetime(frame.loc[~numeric,'date'],format='%d-%m-%Y',errors='coerce')
        result['date_numeric_count']=int(numeric.sum())
        result['date_unparsed']=int(parsed.isna().sum())
        result['date_range']=[str(parsed.min()),str(parsed.max())]
        result['units']='Age confirmed in days; divided by 365.25. Height unit incorrectly documented as DAYS; BMI omitted. Gender codes kept categorical without inferred labels.'
    return result


def figures(project, data, out):
    folder=out/'figures'; folder.mkdir(exist_ok=True)
    split=json.loads((out/'split.json').read_text())
    training=data.loc[split['train_rows']].copy()
    target=TARGETS[project]
    y=training[target] if project=='property' else (training[target].map({'Yes':1,'No':0}) if project=='churn' else training[target])
    if project=='property':
        charts.histogram(folder/'target.svg','Training target distribution',y,target,bins=20)
    else:
        counts=y.value_counts().sort_index()
        labels=['No churn','Churn'] if project=='churn' else ['Disease absent','Disease present']
        charts.bars(folder/'target.svg','Training target distribution',labels,counts.values,'Number of training records')
    if project=='property':
        charts.scatter(folder/'income_price.svg','Income and district median value',training.median_income,y,'Median income (source units)','Median house value (source units)')
        charts.scatter(folder/'geography.svg','Training districts: geographic coverage',training.longitude,training.latitude,'Longitude','Latitude')
        grouped=training.groupby('ocean_proximity')[target].mean()
        charts.bars(folder/'ocean.svg','Mean district value by ocean proximity',grouped.index,grouped.values,'Median house value (source units)')
    elif project=='churn':
        for column in ['Contract','InternetService','PaymentMethod']:
            grouped=training.assign(rate=y).groupby(column).rate.mean()
            charts.bars(folder/f'{column}.svg',f'Training churn rate by {column}',grouped.index,grouped.values,'Churn proportion')
        charts.histogram(folder/'tenure.svg','Training customer tenure',training.tenure,'Tenure (months)')
        charts.scatter(folder/'charges.svg','Monthly charges and tenure',training.tenure,training.MonthlyCharges,'Tenure (months)','Monthly charges (USD)')
    else:
        clean=PrepareFeatures('disease').fit_transform(training)
        for column,label in [('age','Age (years)'),('ap_hi','Systolic pressure (source units)'),('weight','Weight (kg)')]:
            charts.histogram(folder/f'{column}.svg',f'Training {column} distribution',clean[column],label)
        for column in ['cholesterol','gluc','active']:
            grouped=training.assign(rate=y).groupby(column).rate.mean()
            charts.bars(folder/f'{column}.svg',f'Disease prevalence by {column}',grouped.index,grouped.values,'Disease proportion; association only')
    importance=pd.read_csv(out/'feature_importance.csv')
    used=PrepareFeatures(project).fit_transform(training.drop(columns=[target])).columns
    importance=importance[importance.feature.isin(used)].head(10)
    charts.bars(folder/'importance.svg','Permutation importance on held-out sample',importance.feature,importance.importance,'Decrease in score after shuffling; 3 repeats')
    predictions=pd.read_csv(out/'test_predictions.csv')
    if project=='property':
        charts.scatter(folder/'actual_predicted.svg','Test predictions versus actual values',predictions.actual,predictions.predicted,'Actual median value','Predicted median value',diagonal=True)
        charts.scatter(folder/'residuals.svg','Test residuals',predictions.predicted,predictions.actual-predictions.predicted,'Predicted median value','Residual (actual minus prediction)')
    else:
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
        matrix=confusion_matrix(predictions.actual,predictions.predicted,labels=[0,1])
        charts.bars(folder/'confusion.svg','Test confusion counts',['True negative','False positive','False negative','True positive'],matrix.ravel(),'Number of records')
        ranking=predictions['probability'] if 'probability' in predictions else predictions.decision_score
        fpr,tpr,_=roc_curve(predictions.actual,ranking)
        precision,recall,_=precision_recall_curve(predictions.actual,ranking)
        charts.scatter(folder/'roc.svg','Test ROC curve',fpr,tpr,'False positive rate','True positive rate',connect=True)
        charts.scatter(folder/'precision_recall.svg','Test precision-recall curve',recall,precision,'Recall','Precision',connect=True)


def new_customers(out):
    common={'gender':'Female','SeniorCitizen':0,'Partner':'No','Dependents':'No',
        'PhoneService':'Yes','MultipleLines':'No','InternetService':'DSL','OnlineSecurity':'No',
        'OnlineBackup':'No','DeviceProtection':'No','TechSupport':'No','StreamingTV':'No',
        'StreamingMovies':'No','PaperlessBilling':'Yes','PaymentMethod':'Electronic check'}
    records=[]
    for key,tenure,contract,monthly in [('DEMO-001',2,'Month-to-month',55.),('DEMO-002',48,'Two year',55.),('DEMO-003',12,'One year',70.)]:
        records.append({**common,'customerID':key,'tenure':tenure,'Contract':contract,
                        'MonthlyCharges':monthly,'TotalCharges':tenure*monthly})
    inputs=pd.DataFrame(records)
    inputs.to_csv(out/'new_customers.csv',index=False)
    predict_frame('churn',inputs).to_csv(out/'new_customer_predictions.csv',index=False)


def report(project, data, out, quality):
    metrics=pd.read_csv(out/'test_metrics.csv')
    selection=json.loads((out/'selection.json').read_text())
    winner=selection['winner']; row=metrics.set_index('model').loc[winner]
    importance=pd.read_csv(out/'feature_importance.csv')
    top=', '.join(importance.loc[importance.importance>0].head(4).feature)
    common=(f'The selected model is {winner}. Selection used five-fold training-only cross-validation, '
            'with the final 20% test partition held aside. Learned imputation, scaling and encoding '
            'were refitted within each fold. Random seed: 42. The test comparison below is descriptive; '
            'it did not select the winner. Baselines provide a reference. No accuracy target was promised.')
    if project=='property':
        interpretation=(f'Test RMSE is {row.RMSE:,.2f}, MAE {row.MAE:,.2f}, and RÂ² {row.R2:.3f}. '
            f'The strongest permutation signals include {top}. Predictors describe district aggregates, '
            'not individual sale prices. Simple regression uses median_income; multiple regression '
            'adds the remaining predictors. Ordinary least squares has no regularization strength '
            'to tune; both required specifications are compared under the same folds. '
            'Coefficient values are for standardized numeric inputs and reference-coded categories. '
            'Correlated totals can make coefficients unstable; interpret them jointly, not causally. '
            'The upper concentration in recorded values suggests possible capping, which can impair '
            'predictions in the upper tail. Price units are not established by the dictionary. '
            'Random holdout measures performance for similar sampled districts; geographic extrapolation remains untested.')
    elif project=='churn':
        interpretation=(f'Test precision is {row.precision:.3f}, recall {row.recall:.3f}, F1 {row.F1:.3f}, '
            f'and ROC-AUC {row.ROC_AUC:.3f}. Important permutation signals include {top}. '
            'The selection metric is F1, so the highest-accuracy model need not win. '
            'Higher recall captures more churners but creates additional false-positive outreach. '
            'The supplied report does not provide intervention costs or a churn horizon, so no profit '
            'claim or cost-optimal threshold is made. Use the default model threshold and review the '
            'precision-recall curve before any future threshold tuning on validation data. '
            'Retention teams could pilot outreach to higher-scoring customers and measure outcomes '
            'in a controlled experiment. Contract and service associations do not establish that '
            'changing contracts or services causes retention. The new_customers.csv records are '
            'synthetic examples created for inference, with no observed outcomes; their predictions '
            'demonstrate functionality, not measured accuracy.')
    else:
        interpretation=(f'Test precision is {row.precision:.3f}, recall {row.recall:.3f}, F1 {row.F1:.3f}, '
            f'and ROC-AUC {row.ROC_AUC:.3f}. Important permutation signals include {top}. '
            'The model predicts recorded disease presence, not future disease onset. '
            'Both false negatives and false positives are reported in the confusion figure. '
            'No clinical operating threshold is justified by this assignment. Abnormal blood-pressure '
            'records are preserved and invalid readings imputed using training medians. Height and '
            'weight outliers are documented but retained without an unsupported medical exclusion rule. '
            'BMI is omitted because the height unit is misdocumented. Country and occupation are excluded '
            'from the main model and included only in a training-CV sensitivity comparison. Date is '
            'excluded from prediction. Future work should establish measurement units and provenance, '
            'perform temporal/external validation, and assess subgroup performance and calibration. '
            'This educational model has not been validated for clinical use.')
    styles='body{font:16px/1.6 Arial;color:#263449;max-width:1100px;margin:40px auto;padding:0 24px}h1,h2{color:#16475a}table{border-collapse:collapse;font-size:14px;width:100%}td,th{padding:9px;border-bottom:1px solid #dce5eb;text-align:left}pre{white-space:pre-wrap;background:#f4f7fa;padding:16px}img{width:100%;max-width:900px}section{margin:28px 0}a{color:#126483}'
    body=f'<h1>{TITLES[project]}</h1><p>{escape(common)}</p><h2>Results and interpretation</h2><p>{escape(interpretation)}</p>'
    body+='<h2>Held-out model comparison</h2>'+metrics.to_html(index=False,float_format=lambda v:f'{v:.4f}')
    body+='<h2>Data quality and decisions</h2><pre>'+escape(json.dumps(quality,indent=2,default=str))+'</pre>'
    body+='<h2>Training and tuning</h2><pre>'+escape(json.dumps(selection,indent=2))+'</pre>'
    if project=='disease':
        body+='<h2>Training-only sensitivity checks</h2>'+pd.read_json(out/'sensitivity.json').to_html(index=False)
    body+='<h2>Exploration and diagnostics</h2>'
    for figure in sorted((out/'figures').glob('*.svg')):
        body+=f'<section><img src="outputs/figures/{figure.name}" alt="{escape(figure.stem)}"></section>'
    body+=f'<h2>Sources and reproducibility</h2><p>Assignment: ML Project.docx (1).pdf. Dataset: datasets/{FILES[project]}. '
    body+=f'<a href="https://drive.google.com/file/d/{DICTIONARIES[project]}/view">Supplied data dictionary</a>. '
    body+='<a href="https://scikit-learn.org/stable/common_pitfalls.html">Leakage guidance</a>; <a href="https://scikit-learn.org/stable/modules/model_evaluation.html">Evaluation guidance</a>.</p>'
    body+='<p>See README.md for commands. Raw workbooks are unchanged. Figures are portable SVG. Source hashes and package versions are in outputs/run.json. Example inputs use original units, including disease age in days.</p>'
    (out.parent/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+TITLES[project]+'</title><style>'+styles+'</style><body>'+body+'</body></html>',encoding='utf-8')
    return common,interpretation


def notebook(project,out,common,interpretation):
    cells=[]
    def md(value): cells.append({'cell_type':'markdown','metadata':{},'source':value})
    def code(value): cells.append({'cell_type':'code','metadata':{},'source':value,'execution_count':None,'outputs':[]})
    md('# '+TITLES[project]+'\n\n'+common+'\n\nThis notebook uses the local, included `ml_projects` Python package. Run from the project root or this notebook directory. Set `RETRAIN=True` to rebuild all models and figures; otherwise it verifies and explores the existing completed training run. No results are hardcoded into analysis cells.')
    code("from pathlib import Path\nimport sys, json\nroot = Path.cwd()\nif (root / 'property_price_prediction' / 'analysis.ipynb').exists():\n    root = root / 'property_price_prediction'\nassert (root / 'ml_projects').exists(), 'Open this standalone project folder'\nsys.path.insert(0, str(root))\nimport pandas as pd\nfrom ml_projects.data import read_dataset, split_target, PrepareFeatures\nfrom ml_projects.train import train, output_dir\nfrom ml_projects.deliver import audit, figures, new_customers\nfrom ml_projects.predict import predict_frame\nproject = "+repr(project)+"\nout = output_dir(project)\nRETRAIN = False")
    md('## 1. Load and audit the original data\nInspect missingness, target labels and source-specific measurement issues before interpreting results. The full-dataset audit checks quality; model-driven exploration uses training rows only.')
    code("data = read_dataset(project)\nquality = audit(project, data)\nprint(json.dumps(quality, indent=2, default=str))\nprint(data.head().to_string(index=False))")
    md('## 2. Preprocessing and model development\nThe included train() implementation splits before fitting learned transformations. Numeric values use median imputation and scaling. Categorical features use most-frequent imputation and one-hot encoding. Each grid-search fold fits its own pipeline. Read ml_projects/train.py for the complete grids and ml_projects/data.py for deterministic cleaning rules.')
    code("if RETRAIN or not (out / 'run.json').exists():\n    train(project)\nsplit = json.loads((out / 'split.json').read_text())\nassert not set(split['train_rows']) & set(split['test_rows'])\nassert len(split['train_rows']) + len(split['test_rows']) == len(data)\nprint('Training:', len(split['train_rows']), 'Test:', len(split['test_rows']))\nprint((out / 'selection.json').read_text())")
    md('## 3. Training exploration\nDistributions, group differences and associations are descriptive. They do not establish causal effects.')
    code("training = data.loc[split['train_rows']]\nprint(training.describe(include='all').to_string())\nprint(training.isna().sum().to_string())\nfigures(project, data, out)")
    md('## 4. Held-out evaluation\nModel choice was fixed using cross-validation before these test metrics were calculated. F1 selects classification models; negative RMSE selects regression models. Metrics are recomputed from saved test predictions in the test suite.')
    code("print(pd.read_csv(out / 'test_metrics.csv').to_string(index=False))\nprint(pd.read_csv(out / 'feature_importance.csv').to_string(index=False))")
    md('## 5. Interpretation and limitations\n'+interpretation)
    md('## 6. Batch prediction\nThese first examples are held-out input records, not independent new observations. Churn additionally includes separately constructed synthetic customers in outputs/new_customers.csv. Remove the target from inference inputs.')
    code("inputs = pd.read_csv(out / 'example_inputs.csv')\nprint(predict_frame(project, inputs).to_string(index=False))")
    if project=='churn':
        code("new_customers(out)\nnew_inputs = pd.read_csv(out / 'new_customers.csv')\nprint(predict_frame(project, new_inputs).to_string(index=False))")
    md('## 7. Figures\nThe following figures are linked locally and embedded in the accompanying HTML report.')
    for figure in sorted((out/'figures').glob('*.svg')):
        md(f'![{figure.stem}](outputs/figures/{figure.name})')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
         'language_info':{'name':'python','version':'3.13'},
         'execution_method':'Sequential Python exec in a fresh namespace; not a Jupyter kernel. Outputs captured from actual execution.'},'nbformat':4,'nbformat_minor':5}
    for i,c in enumerate(cells): c['id']=f'{project}-{i:03}'
    # Actual sequential execution of ordinary Python cells, with no fabricated outputs.
    namespace={'__name__':'__notebook__'}; counter=0
    for cell in cells:
        if cell['cell_type']=='code':
            counter+=1; stream=io.StringIO()
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(cell['source'],f'{project}-cell-{counter}','exec'),namespace)
            cell['execution_count']=counter
            cell['outputs']=[{'output_type':'stream','name':'stdout','text':stream.getvalue()}]
    write_json(out.parent/'analysis.ipynb',nb)
    write_json(out/'notebook_execution.json',{'code_cells_executed':counter,'errors':0,
        'method':nb['metadata']['execution_method'],'retrained_in_notebook':False})


def build(project):
    out=output_dir(project)
    data=read_dataset(project)
    quality=audit(project,data)
    write_json(out/'data_audit.json',quality)
    figures(project,data,out)
    if project=='churn': new_customers(out)
    common,interpretation=report(project,data,out,quality)
    notebook(project,out,common,interpretation)
    (out.parent/'WORKFLOW.md').write_text(f'# {TITLES[project]}\n\nOpen `analysis.ipynb` for the executed workflow or `report.html` for the report.\n\nFrom the repository root:\n\n```powershell\npython -m ml_projects.train --project {project}\npython -m ml_projects.deliver --project {project}\npython -m ml_projects.predict --project {project} --input {DIRECTORIES[project]}/outputs/example_inputs.csv --output {DIRECTORIES[project]}/outputs/predictions_demo.csv\n```\n\nSee the root README for setup, verification and limitations.\n',encoding='utf-8')
    print(project,'report and executed notebook built',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',choices=[*FILES,'all'],default='all')
    args=parser.parse_args()
    for project in FILES if args.project=='all' else [args.project]: build(project)
