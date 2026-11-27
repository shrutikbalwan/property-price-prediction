"""Advanced evaluation, calibrated inference and split-conformal intervals.
Run from this project directory: python -m ml_projects.advanced --train
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, GroupKFold, train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from .data import ROOT, FILES, read_dataset, split_target
from .train import pipeline, evaluate, write_json

PROJECT=next(iter(FILES))
OUT=ROOT/'outputs'/'advanced'
HISTORY='The original test set was inspected during earlier development. Results here are a historical benchmark, not new independent external validation.'

def config():
    return json.loads((ROOT/'config.json').read_text())

def probability_metrics(y,p,threshold):
    result=evaluate(PROJECT,y,(p>=threshold).astype(int),p)
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    result.update(Brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),specificity=float(tn/(tn+fp)) if tn+fp else None,
                  confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])
    return result

def conformal_radius(residuals,coverage):
    values=np.sort(np.asarray(residuals,dtype=float))
    if not 0<coverage<1 or not len(values): raise ValueError('Invalid conformal inputs')
    rank=math.ceil((len(values)+1)*coverage)
    if rank>len(values): raise ValueError('Calibration sample too small for requested coverage')
    return float(values[rank-1])

def train_advanced():
    cfg=config(); seed=cfg['seed']; OUT.mkdir(parents=True,exist_ok=True)
    X,y=split_target(read_dataset(PROJECT),PROJECT)
    excluded={'property':[], 'churn':['customerID'], 'disease':['id','date','country','occupation']}[PROJECT]
    X=X.drop(columns=excluded,errors='ignore')
    split=json.loads((ROOT/'outputs'/'split.json').read_text())
    original=np.array(split['train_rows']); test=np.array(split['test_rows'])
    core,held=train_test_split(original,test_size=.2 if PROJECT=='property' else .4,random_state=seed,
                              stratify=y.loc[original] if PROJECT!='property' else None)
    if PROJECT=='property': cal=held; validation=np.array([],dtype=int)
    else: cal,validation=train_test_split(held,test_size=.5,random_state=seed,stratify=y.loc[held])
    partitions={'fit':core.tolist(),'calibration':cal.tolist(),'threshold_validation':validation.tolist(),'historical_test':test.tolist()}
    write_json(OUT/'partitions.json',partitions)
    trainX=X.loc[core]; trainy=y.loc[core]
    if PROJECT=='property':
        estimators=[('ridge',Ridge(),{'model__alpha':[1.,10.]}),('hist_gradient_boosting',HistGradientBoostingRegressor(random_state=seed,max_iter=cfg['boosting_iterations']),{'model__max_leaf_nodes':[15,31],'model__l2_regularization':[1.]})]
        cv=KFold(cfg['cv_folds'],shuffle=True,random_state=seed); scoring='neg_root_mean_squared_error'
    else:
        estimators=[('logistic',LogisticRegression(max_iter=3000,random_state=seed),{'model__C':[.1,1.]}),('hist_gradient_boosting',HistGradientBoostingClassifier(random_state=seed,max_iter=cfg['boosting_iterations']),{'model__max_leaf_nodes':[15,31],'model__l2_regularization':[1.]})]
        cv=StratifiedKFold(cfg['cv_folds'],shuffle=True,random_state=seed); scoring='neg_log_loss'
    candidates=[]; fitted={}
    for name,estimator,params in estimators:
        print(f'{PROJECT}: advanced CV {name}',flush=True)
        search=GridSearchCV(pipeline(PROJECT,trainX,estimator),params,cv=cv,scoring=scoring,n_jobs=1,error_score='raise')
        search.fit(trainX,trainy)
        pd.DataFrame(search.cv_results_).to_csv(OUT/f'{name}_cv.csv',index=False)
        candidates.append({'name':name,'cv_score':float(search.best_score_),'parameters':search.best_params_})
        fitted[name]=search.best_estimator_
    selected=max(candidates,key=lambda r:r['cv_score']); base=fitted[selected['name']]
    model=base; radius=None; threshold=None; geographical=None
    if PROJECT=='property':
        radius=conformal_radius(np.abs(y.loc[cal]-base.predict(X.loc[cal])),cfg['interval_coverage'])
        groups=(trainX.latitude.round().astype(str)+'/'+trainX.longitude.round().astype(str)).to_numpy()
        geo=cross_val_score(base,trainX,trainy,groups=groups,cv=GroupKFold(cfg['cv_folds']),scoring=scoring,n_jobs=1,error_score='raise')
        geographical={'block':'1-degree rounded latitude/longitude; no spatial buffer','fold_RMSE':(-geo).tolist(),'mean_RMSE':float(-geo.mean()),'purpose':'Training-only geographic stress check; not model selection or future-location guarantee'}
    else:
        model=CalibratedClassifierCV(FrozenEstimator(base),method='sigmoid',n_jobs=1)
        model.fit(X.loc[cal],y.loc[cal])
        p=model.predict_proba(X.loc[validation])[:,1]; v=y.loc[validation]
        grid=[]
        for t in np.linspace(.05,.95,91):
            m=probability_metrics(v,p,float(t)); grid.append({'threshold':float(t),'F1':m['F1'],'recall':m['recall'],'specificity':m['specificity']})
        if PROJECT=='disease':
            eligible=[r for r in grid if r['recall']>=cfg['target_validation_recall']]
            chosen=max(eligible,key=lambda r:(r['specificity'],r['F1'])) if eligible else max(grid,key=lambda r:r['recall'])
        else: chosen=max(grid,key=lambda r:r['F1'])
        threshold=chosen['threshold']; pd.DataFrame(grid).to_csv(OUT/'validation_thresholds.csv',index=False)
        write_json(OUT/'calibration_validation.json',{'before':probability_metrics(v,base.predict_proba(X.loc[validation])[:,1],.5),'after':probability_metrics(v,p,.5),'chosen_threshold':chosen})
    # Selection and calibration are finished before historical test evaluation.
    testX=X.loc[test]; truth=y.loc[test].to_numpy(); pred=model.predict(testX)
    detail=pd.DataFrame({'source_row':test,'actual':truth})
    if PROJECT=='property':
        metrics=evaluate(PROJECT,truth,pred); detail['prediction']=pred
        detail['lower']=pred-radius; detail['upper']=pred+radius
        metrics.update(interval_coverage=float(np.mean((truth>=pred-radius)&(truth<=pred+radius))),interval_width=float(2*radius))
    else:
        proba=model.predict_proba(testX)[:,1]; pred=(proba>=threshold).astype(int)
        detail['prediction']=pred; detail['probability']=proba
        metrics=probability_metrics(truth,proba,threshold)
        observed,forecast=calibration_curve(truth,proba,n_bins=10,strategy='quantile')
        pd.DataFrame({'mean_probability':forecast,'observed_frequency':observed}).to_csv(OUT/'calibration_curve.csv',index=False)
    detail.to_csv(OUT/'test_predictions.csv',index=False)
    rng=np.random.default_rng(seed); estimates=[]
    for _ in range(cfg['bootstrap_repeats']):
        ids=rng.integers(0,len(test),len(test))
        estimates.append(float(np.sqrt(np.mean((truth[ids]-pred[ids])**2))) if PROJECT=='property' else float(f1_score(truth[ids],pred[ids],zero_division=0)))
    uncertainty={'metric':'RMSE' if PROJECT=='property' else 'F1','percentile_95':np.quantile(estimates,[.025,.975]).tolist(),'method':'Row bootstrap on historical test predictions; does not capture model-selection or geographic uncertainty','repeats':cfg['bootstrap_repeats']}
    groups={}
    if PROJECT=='property':
        groups={'ocean_proximity':testX.ocean_proximity.astype(str).to_numpy(),'actual_price_band':pd.cut(truth,[-np.inf,150000,300000,np.inf],labels=['below 150k','150k to 300k','above 300k']).astype(str)}
    elif PROJECT=='churn': groups={c:testX[c].astype(str).to_numpy() for c in ['Contract','gender','SeniorCitizen']}
    else: groups={'gender_code':testX.gender.astype(str).to_numpy(),'age_band':pd.cut(testX.age/365.25,[0,40,55,70,np.inf],labels=['under 40','40 to 55','55 to 70','70 plus']).astype(str).to_numpy()}
    subgroups=[]
    for attribute,values in groups.items():
        for value in np.unique(values):
            mask=values==value; n=int(mask.sum()); row={'attribute':attribute,'group':str(value),'rows':n,'small_sample':n<100}
            if PROJECT=='property': row.update(MAE=float(np.mean(np.abs(truth[mask]-pred[mask]))),RMSE=float(np.sqrt(np.mean((truth[mask]-pred[mask])**2))))
            elif len(np.unique(truth[mask]))==2:
                m=probability_metrics(truth[mask],proba[mask],threshold); row.update({k:m[k] for k in ['F1','recall','specificity','Brier']})
            else: row['note']='Only one outcome class; subgroup metrics suppressed'
            subgroups.append(row)
    write_json(OUT/'subgroups.json',subgroups)
    sample=testX.sample(min(800,len(testX)),random_state=seed)
    importance=permutation_importance(model,sample,y.loc[sample.index],scoring=scoring,n_repeats=3,n_jobs=1,random_state=seed)
    features=pd.DataFrame({'feature':X.columns,'importance':importance.importances_mean,'std':importance.importances_std}).sort_values('importance',ascending=False)
    features.to_csv(OUT/'feature_importance.csv',index=False)
    schema=[]
    for col in X:
        numeric=pd.api.types.is_numeric_dtype(trainX[col])
        values=trainX[col].dropna()
        spec={'name':col,'numeric':numeric,'missing_allowed':True}
        if numeric: spec.update(min=float(values.min()),max=float(values.max()),median=float(values.median()))
        else: spec['categories']=sorted(values.astype(str).unique().tolist())
        if col=='age' and PROJECT=='disease': spec['unit']='days (years × 365.25)'
        if (PROJECT=='disease' and col in ['gender','cholesterol','gluc','active','alco','smoke']) or (PROJECT=='churn' and col=='SeniorCitizen'):
            spec['numeric_options']=sorted(float(v) for v in values.unique())
        if col=='height' and PROJECT=='disease': spec['unit']='original dataset units; unit documentation unresolved'
        schema.append(spec)
    artifact={'project':PROJECT,'model':model,'threshold':threshold,'radius':radius,'schema':schema,'coverage':cfg['interval_coverage'] if PROJECT=='property' else None}
    joblib.dump(artifact,OUT/'model.joblib',compress=3)
    reloaded=joblib.load(OUT/'model.joblib')['model']; np.testing.assert_allclose(model.predict(testX.head(20)),reloaded.predict(testX.head(20)))
    testX.head(5).to_csv(OUT/'example_inputs.csv',index=False)
    summary={'project':PROJECT,'created_at_utc':datetime.now(timezone.utc).isoformat(),'winner':selected['name'],'selection_metric':scoring,'candidates':candidates,'metrics':metrics,'threshold':threshold,'interval_coverage_target':artifact['coverage'],'interval_radius':radius,'geographic_validation':geographical,'uncertainty':uncertainty,'evaluation_history':HISTORY,'partition_sizes':{k:len(v) for k,v in partitions.items()},'source_sha256':hashlib.sha256((ROOT/'datasets'/FILES[PROJECT]).read_bytes()).hexdigest(),'model_sha256':hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest(),'configuration':cfg,'feature_importance':features.to_dict('records'),'subgroups':subgroups,'schema':schema}
    write_json(OUT/'summary.json',summary)
    print(json.dumps({'project':PROJECT,'winner':selected['name'],'metrics':metrics},indent=2),flush=True)
    return summary

def load_artifact():
    return joblib.load(OUT/'model.joblib')

def predict(artifact,records,threshold=None):
    if not isinstance(records,list) or not records or len(records)>5000 or not all(isinstance(r,dict) for r in records): raise ValueError('Provide 1 to 5000 records as objects')
    frame=pd.DataFrame(records).replace(r'^\s*$',np.nan,regex=True)
    names=[s['name'] for s in artifact['schema']]
    missing=[c for c in names if c not in frame]
    if missing: raise ValueError('Missing columns: '+', '.join(missing))
    frame=frame[names].copy(); warnings=[]
    for spec in artifact['schema']:
        col=spec['name']; original=frame[col]
        if spec['numeric']:
            values=pd.to_numeric(original,errors='coerce')
            if (original.notna() & values.isna()).any() or np.isinf(values.to_numpy(dtype=float,na_value=np.nan)).any(): raise ValueError('Invalid numeric value in '+col)
            if ((values<spec['min']) | (values>spec['max'])).any(): warnings.append(col+': values outside the fitting-data range; predictions may be unreliable')
            if 'numeric_options' in spec and (values.notna() & ~values.isin(spec['numeric_options'])).any(): raise ValueError('Invalid category code in '+col)
            frame[col]=values
        else:
            frame[col]=original.map(lambda v:str(v) if pd.notna(v) else np.nan)
            if (frame[col].notna() & ~frame[col].isin(spec['categories'])).any(): warnings.append(col+': unfamiliar category encoded using the fitted fallback')
    if artifact['project']=='disease':
        invalid=((frame.ap_hi<=0)|(frame.ap_lo<=0)|(frame.ap_hi>300)|(frame.ap_lo>200)|(frame.ap_lo>=frame.ap_hi))
        if invalid.any(): warnings.append('Invalid blood-pressure combinations will be treated as missing and imputed by the fitted pipeline')
    with threadpool_limits(limits=1):
        if artifact['project']=='property':
            values=artifact['model'].predict(frame); radius=artifact['radius']
            results=[{'prediction':float(v),'lower':float(v-radius),'upper':float(v+radius)} for v in values]
        else:
            t=artifact['threshold'] if threshold is None else float(threshold)
            if not math.isfinite(t) or not 0<=t<=1: raise ValueError('Threshold must be between 0 and 1')
            probabilities=artifact['model'].predict_proba(frame)[:,1]
            results=[{'probability':float(p),'prediction':int(p>=t),'threshold':t} for p in probabilities]
    return {'results':results,'warnings':warnings}

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--train',action='store_true'); parser.add_argument('--input',type=Path); parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.train:
        with threadpool_limits(limits=1): train_advanced()
    elif args.input and args.output:
        if args.input.resolve()==args.output.resolve(): parser.error('Output must differ from input')
        frame=pd.read_csv(args.input); result=predict(load_artifact(),json.loads(frame.to_json(orient='records')))
        args.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(result['results']).to_csv(args.output,index=False)
        print(json.dumps({'rows':len(result['results']),'warnings':result['warnings']}))
    else: parser.error('Use --train or --input FILE --output FILE')
