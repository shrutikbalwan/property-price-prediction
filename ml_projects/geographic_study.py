"""Paired property geographic stress study; no original test rows used.
Run: python -m ml_projects.geographic_study
"""
import hashlib,json,time
from datetime import datetime,timezone
from pathlib import Path
from html import escape
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,GroupKFold,GroupShuffleSplit,train_test_split
from sklearn.neighbors import BallTree
from threadpoolctl import threadpool_limits
from .data import ROOT,FILES,read_dataset,split_target
from .train import pipeline,write_json
from .advanced import conformal_radius

OUT=ROOT/'outputs'/'geographic_study'
EARTH_KM=6371.0088
PROTOCOL={'seed':20261127,'folds':5,'block_degrees':1.,'buffer_km':10.,'calibration_fraction':.2,'interval_target':.9,'modes':['random_control','geographic_blocks','geographic_buffer_10km'],'models':{'linear_baseline':'Multiple linear regression, matching original baseline family','advanced_boosting':{'max_iter':150,'max_leaf_nodes':31,'l2_regularization':1.,'random_state':42}},'small_group_rows':100,'price_bins':[-1.,150000.,300000.,450000.,float(1e12)],'price_labels':['under 150k','150k-300k','300k-450k','450k+'],'decision_rule':'Point advantage requires lower pooled RMSE in all modes and wins in at least four of five buffered folds. Interval transfer is flagged if buffered empirical coverage is below 85% for the 90% target. This is a descriptive engineering criterion, not a hypothesis test.','scope':'Original training rows only. Previously inspected historical test rows excluded. Hyperparameters frozen from previous development; this is a stress study, not new independent external validation.'}

def blocks(X):
    lat=np.floor(X.latitude/PROTOCOL['block_degrees']).astype(int)
    lon=np.floor(X.longitude/PROTOCOL['block_degrees']).astype(int)
    return lat.astype(str)+'/'+lon.astype(str)

def nearest_km(query,reference):
    if not len(reference):raise ValueError('Reference locations must not be empty')
    tree=BallTree(np.radians(reference[['latitude','longitude']].to_numpy(dtype=float)),metric='haversine')
    distance,_=tree.query(np.radians(query[['latitude','longitude']].to_numpy(dtype=float)),k=1)
    return distance[:,0]*EARTH_KM

def buffered_ids(X,development,evaluation,buffer_km):
    distances=nearest_km(X.loc[development],X.loc[evaluation])
    keep=distances>=buffer_km
    return np.asarray(development)[keep],np.asarray(development)[~keep]

def measure(frame):
    error=frame.prediction.to_numpy()-frame.actual.to_numpy()
    covered=(frame.actual>=frame.lower)&(frame.actual<=frame.upper)
    return {'rows':len(frame),'RMSE':float(np.sqrt(np.mean(error**2))),'MAE':float(np.mean(np.abs(error))),'mean_signed_error':float(np.mean(error)),'coverage':float(covered.mean()),'mean_interval_width':float((frame.upper-frame.lower).mean())}

def subgroup_tables(predictions):
    locations=[];prices=[]
    for (mode,model,block),data in predictions.groupby(['mode','model','block']):
        locations.append({'mode':mode,'model':model,'block':block,'latitude':float(data.latitude.mean()),'longitude':float(data.longitude.mean()),'small_sample':len(data)<PROTOCOL['small_group_rows'],**measure(data)})
    data=predictions.copy();data['price_band']=pd.cut(data.actual,PROTOCOL['price_bins'],labels=PROTOCOL['price_labels'],right=False)
    for (mode,model,band),group in data.groupby(['mode','model','price_band'],observed=True):
        prices.append({'mode':mode,'model':model,'price_band':str(band),'small_sample':len(group)<PROTOCOL['small_group_rows'],**measure(group)})
    return pd.DataFrame(locations),pd.DataFrame(prices)

def decision(pooled,folds):
    wide=pooled.pivot(index='mode',columns='model',values='RMSE')
    buffer=folds[folds['mode']=='geographic_buffer_10km'].pivot(index='fold',columns='model',values='RMSE')
    coverage=float(pooled[(pooled['mode']=='geographic_buffer_10km')&(pooled.model=='advanced_boosting')].coverage.iloc[0])
    wins=int((buffer.advanced_boosting<buffer.linear_baseline).sum())
    return {'point_advantage_confirmed':bool((wide.advanced_boosting<wide.linear_baseline).all() and wins>=4),'buffered_fold_wins':wins,'buffered_advanced_coverage':coverage,'interval_transfer_flag':bool(coverage<.85),'target_coverage':.9,'interpretation':'The coverage flag assesses pooled coverage only, not location or price-band coverage. Each fold is refitted and recalibrated; results do not validate the deployed fixed interval. No automatic model replacement or interval widening. Geographic distribution shift can invalidate marginal conformal coverage assumptions.'}

def run():
    OUT.mkdir(parents=True,exist_ok=True);write_json(OUT/'protocol.json',PROTOCOL)
    active=[ROOT/'outputs/model.joblib',ROOT/'outputs/advanced/model.joblib',ROOT/'outputs/advanced/summary.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in active}
    X,y=split_target(read_dataset('property'),'property'); groups=blocks(X)
    original=json.loads((ROOT/'outputs/split.json').read_text());ids=np.array(original['train_rows']);test=set(original['test_rows'])
    assert not set(ids)&test
    random_splits=list(KFold(5,shuffle=True,random_state=PROTOCOL['seed']).split(ids))
    geographic=list(GroupKFold(5).split(ids,groups=groups.loc[ids]))
    partitions=[];predictions=[];calibration=[];fold_metrics=[];start=time.time()
    for mode in PROTOCOL['modes']:
        splits=random_splits if mode=='random_control' else geographic
        for fold,(train_pos,eval_pos) in enumerate(splits,1):
            development,evaluation=ids[train_pos],ids[eval_pos];excluded=np.array([],dtype=int)
            if mode=='geographic_buffer_10km':development,excluded=buffered_ids(X,development,evaluation,PROTOCOL['buffer_km'])
            if mode=='random_control':fit,cal=train_test_split(development,test_size=.2,random_state=PROTOCOL['seed']+fold)
            else:
                fit_pos,cal_pos=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=PROTOCOL['seed']+fold).split(development,groups=groups.loc[development]))
                fit,cal=development[fit_pos],development[cal_pos]
            assert min(len(fit),len(cal))>=20
            nearest=nearest_km(X.loc[evaluation],X.loc[fit])
            part={'mode':mode,'fold':fold,'fit':fit.tolist(),'calibration':cal.tolist(),'evaluation':evaluation.tolist(),'buffer_excluded':excluded.tolist(),'fit_groups':sorted(groups.loc[fit].unique()),'calibration_groups':sorted(groups.loc[cal].unique()),'evaluation_groups':sorted(groups.loc[evaluation].unique()),'minimum_eval_to_fit_km':float(nearest.min())}
            partitions.append(part)
            for name,estimator in [('linear_baseline',LinearRegression()),('advanced_boosting',HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=31,l2_regularization=1.,random_state=42))]:
                print(f'{mode}, fold {fold}/5: {name} ({len(fit)} fit / {len(cal)} calibration / {len(evaluation)} evaluation)',flush=True)
                model=pipeline('property',X.loc[fit],estimator);model.fit(X.loc[fit],y.loc[fit])
                residual=np.abs(y.loc[cal].to_numpy()-model.predict(X.loc[cal]));radius=conformal_radius(residual,.9)
                for row,value in zip(cal,residual):calibration.append({'mode':mode,'fold':fold,'model':name,'source_row':int(row),'absolute_residual':float(value)})
                p=model.predict(X.loc[evaluation]); frame=pd.DataFrame({'mode':mode,'fold':fold,'model':name,'source_row':evaluation,'actual':y.loc[evaluation].to_numpy(),'prediction':p,'lower':p-radius,'upper':p+radius,'latitude':X.loc[evaluation,'latitude'].to_numpy(),'longitude':X.loc[evaluation,'longitude'].to_numpy(),'block':groups.loc[evaluation].to_numpy(),'nearest_fit_km':nearest})
                predictions.append(frame)
                fold_metrics.append({'mode':mode,'fold':fold,'model':name,'fit_rows':len(fit),'calibration_rows':len(cal),'buffer_excluded_rows':len(excluded),'interval_radius':radius,**measure(frame)})
    predictions=pd.concat(predictions,ignore_index=True);folds=pd.DataFrame(fold_metrics)
    pooled=pd.DataFrame([{'mode':mode,'model':model,**measure(frame)} for (mode,model),frame in predictions.groupby(['mode','model'])])
    locations,prices=subgroup_tables(predictions)
    for filename,frame in [('out_of_fold_predictions.csv',predictions),('calibration_residuals.csv',pd.DataFrame(calibration)),('fold_metrics.csv',folds),('pooled_metrics.csv',pooled),('location_metrics.csv',locations),('price_band_metrics.csv',prices)]:frame.to_csv(OUT/filename,index=False)
    write_json(OUT/'partitions.json',partitions)
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items());assert unchanged
    summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,'original_training_rows':len(ids),'excluded_historical_test_rows':len(test),'pooled_metrics':pooled.to_dict('records'),'decision':decision(pooled,folds),'active_models_unchanged':unchanged,'active_model_hashes':hashes,'source_sha256':hashlib.sha256((ROOT/'datasets'/FILES['property']).read_bytes()).hexdigest(),'study_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.time()-start}
    write_json(OUT/'summary.json',summary);write_report(summary,locations,prices,folds)
    print(json.dumps({'pooled_metrics':summary['pooled_metrics'],'decision':summary['decision']},indent=2),flush=True)
    return summary

def write_report(summary,locations,prices,folds):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    figure=make_subplots(rows=1,cols=2,subplot_titles=['Mean absolute error by block','Interval coverage by block'],horizontal_spacing=.15)
    options=[];pairs=list(locations.groupby(['mode','model']))
    for index,((mode,model),frame) in enumerate(pairs):
        detail=frame[['block','rows','RMSE','coverage','small_sample']].to_numpy()
        for column,value,colors,scale in [(1,'MAE','YlOrRd',None),(2,'coverage','RdYlGn',(0,1))]:
            marker={'size':np.clip(np.sqrt(frame.rows)*1.4,6,28),'color':frame[value],'colorscale':colors,'showscale':True,'colorbar':{'x':.43 if column==1 else 1.02,'title':value,'len':.75}}
            if scale:marker.update(cmin=scale[0],cmax=scale[1])
            figure.add_trace(go.Scatter(x=frame.longitude,y=frame.latitude,mode='markers',visible=index==0,marker=marker,customdata=detail,hovertemplate='Block %{customdata[0]}<br>Rows %{customdata[1]}<br>RMSE %{customdata[2]:,.0f}<br>Coverage %{customdata[3]:.1%}<br>Small sample %{customdata[4]}<extra></extra>',name=value),row=1,col=column)
        visible=[False]*(len(pairs)*2);visible[2*index]=visible[2*index+1]=True
        options.append({'label':mode+' / '+model,'method':'update','args':[{'visible':visible},{'title.text':'Geographic evidence: '+mode+' / '+model}]})
    figure.update_layout(title={'text':'Geographic evidence: '+pairs[0][0][0]+' / '+pairs[0][0][1],'y':.97,'x':.035,'font':{'size':19}},height=820,width=1200,autosize=False,template='plotly_white',showlegend=False,margin={'t':180,'b':120},updatemenus=[{'buttons':options,'x':0,'y':1.12,'xanchor':'left'}])
    for column in [1,2]:figure.update_xaxes(title_text='Longitude (degrees)',range=[-125,-114],row=1,col=column);figure.update_yaxes(title_text='Latitude (degrees)',range=[32,43],row=1,col=column)
    figure.add_annotation(text='Block-average coordinate plots, not a projected basemap. Marker size reflects row count; hover for small-sample flags.',x=.5,y=-.12,xref='paper',yref='paper',showarrow=False)
    figure.write_html(OUT/'geographic_maps.html',include_plotlyjs=True,full_html=True,config={'responsive':False,'displaylogo':False})
    table=lambda frame:pd.DataFrame(frame).to_html(index=False,escape=True,float_format=lambda v:f'{v:,.4f}')
    weak=locations[(locations['mode']=='geographic_buffer_10km')&(locations.model=='advanced_boosting')&(~locations.small_sample)].sort_values('RMSE',ascending=False).head(12)
    coverage_weak=locations[(locations['mode']=='geographic_buffer_10km')&(locations.model=='advanced_boosting')&(~locations.small_sample)].sort_values('coverage').head(12)
    body='<h1>Property geographic evaluation</h1><p>Paired refits of the original linear model family and the advanced histogram-gradient-boosting family. No historical-test rows are used; the application models are unchanged.</p><h2>Pooled out-of-fold results</h2>'+table(summary['pooled_metrics'])+'<h2>Decision</h2><pre>'+escape(json.dumps(summary['decision'],indent=2))+'</pre><h2>Error and coverage maps</h2><p>Select a split strategy and model. Hover over blocks for counts and metrics. Maps are descriptive coordinate plots; no external tiles or network access are required.</p><iframe title="Geographic error and interval coverage maps" src="outputs/geographic_study/geographic_maps.html" style="width:100%;height:820px;border:1px solid #ccd9df"></iframe><h2>Largest errors in buffered evaluation</h2>'+table(weak)+'<h2>Lowest interval coverage in buffered evaluation</h2>'+table(coverage_weak)+'<p>Pooled coverage can hide major conditional failures. These intervals are refitted and recalibrated separately for every fold; their widths differ from the currently deployed interval. They do not validate the app interval for new regions.</p><h2>Coverage and error by actual price band</h2>'+table(prices)+'<h2>Fold-level metrics and sample sizes</h2>'+table(folds)+'<h2>Protocol and limitations</h2><p>Random control uses shuffled row folds and random calibration. Geographic modes assign one-degree floor-based latitude/longitude blocks to non-overlapping evaluation folds; folds may contain several non-contiguous blocks. Calibration holds out whole blocks from fitting. Buffered evaluation removes all development rows within 10 km of any evaluation coordinate before splitting fitting/calibration. Distances are great-circle distances on a spherical Earth using supplied, rounded coordinates.</p><p>Both models use identical rows per condition. Different modes can have different fitting and calibration sample sizes, so the random/geographic contrast includes both distribution and sample-composition changes. Model settings are frozen from earlier development; there is no tuning on these folds. The original test partition is excluded, but the dataset was previously studied. These are internal stress results, not a new external benchmark.</p><p>Intervals use finite-sample split-conformal residual quantiles on separate calibration rows. Standard marginal coverage relies on exchangeability; held-out locations and correlated rows can violate it. Grouping calibration rows does not by itself create group-conformal guarantees. Coverage by location or price band is descriptive, not guaranteed. Price bands use actual prices for evaluation, not a deployable predictor. Target capping can affect high-price errors. Small groups (&lt;100 rows) are flagged. Symmetric intervals can have negative lower bounds.</p><p>Reproduce with <code>python -m ml_projects.geographic_study</code>. Protocol, exact partitions, calibration residuals and predictions are saved in outputs/geographic_study/. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html">GroupKFold reference</a>.</p>'
    style='body{font:16px/1.6 Segoe UI,sans-serif;max-width:1400px;margin:40px auto;padding:0 24px;color:#163044;background:#f5f8fa}h1{font-size:36px}h2{margin-top:35px}table{display:block;overflow:auto;border-collapse:collapse;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce5ea;text-align:left}th{background:#e7eff3}pre{padding:20px;background:#e7eff3;overflow:auto}a{color:#146b62}'
    (ROOT/'PROPERTY_GEOGRAPHIC_REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Property geographic evaluation</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
