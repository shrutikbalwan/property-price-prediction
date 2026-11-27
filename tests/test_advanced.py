"""Independent advanced-model and HTTP contract checks."""
import hashlib,json,threading,urllib.request,urllib.error
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error,f1_score,brier_score_loss,roc_auc_score
from ml_projects.advanced import ROOT,OUT,PROJECT,load_artifact,predict,conformal_radius
from ml_projects.data import read_dataset,split_target
from ml_projects.app import create_server,csv_records

@pytest.fixture(scope='module')
def artifact(): return load_artifact()

@pytest.fixture(scope='module')
def records(): return json.loads(pd.read_csv(OUT/'example_inputs.csv').to_json(orient='records'))

def test_partition_integrity_and_provenance():
    parts=json.loads((OUT/'partitions.json').read_text()); sets=[set(v) for v in parts.values()]
    for i,a in enumerate(sets):
        assert len(a)==len(list(parts.values())[i])
        for b in sets[i+1:]: assert not a&b
    original=json.loads((ROOT/'outputs/split.json').read_text())
    assert sets[0]|sets[1]|sets[2]==set(original['train_rows'])
    assert sets[3]==set(original['test_rows'])
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['model_sha256']==hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest()
    assert max(summary['candidates'],key=lambda r:r['cv_score'])['name']==summary['winner']
    assert 'historical benchmark' in summary['evaluation_history']

def test_recompute_metrics_and_intervals(artifact):
    X,y=split_target(read_dataset(PROJECT),PROJECT)
    saved=pd.read_csv(OUT/'test_predictions.csv'); ids=saved.source_row.to_numpy(); X=X.loc[ids]
    X=X[[s['name'] for s in artifact['schema']]]
    np.testing.assert_array_equal(saved.actual,y.loc[ids])
    summary=json.loads((OUT/'summary.json').read_text()); metrics=summary['metrics']
    if PROJECT=='property':
        prediction=artifact['model'].predict(X)
        assert np.sqrt(mean_squared_error(y.loc[ids],prediction))==pytest.approx(metrics['RMSE'])
        parts=json.loads((OUT/'partitions.json').read_text()); fullX,fully=split_target(read_dataset(PROJECT),PROJECT)
        cal=parts['calibration']; residuals=np.abs(fully.loc[cal]-artifact['model'].predict(fullX.loc[cal]))
        assert conformal_radius(residuals,.9)==pytest.approx(artifact['radius'])
        np.testing.assert_allclose(saved.lower,prediction-artifact['radius'])
        np.testing.assert_allclose(saved.upper,prediction+artifact['radius'])
    else:
        p=artifact['model'].predict_proba(X)[:,1]; prediction=(p>=artifact['threshold']).astype(int)
        np.testing.assert_allclose(saved.probability,p)
        assert f1_score(y.loc[ids],prediction)==pytest.approx(metrics['F1'])
        assert brier_score_loss(y.loc[ids],p)==pytest.approx(metrics['Brier'])
        assert roc_auc_score(y.loc[ids],p)==pytest.approx(metrics['ROC_AUC'])
    np.testing.assert_allclose(saved.prediction,prediction)

def test_batch_order_and_single_equivalence(artifact,records):
    original=json.dumps(records,sort_keys=True); batch=predict(artifact,records)['results']
    for row,result in zip(records,batch): assert predict(artifact,[row])['results'][0]==pytest.approx(result)
    assert predict(artifact,list(reversed(records)))['results']==list(reversed(batch))
    assert json.dumps(records,sort_keys=True)==original

def test_schema_errors(artifact,records):
    with pytest.raises(ValueError,match='1 to 5000'): predict(artifact,[])
    with pytest.raises(ValueError,match='Missing columns'): predict(artifact,[{}])
    key=next(s['name'] for s in artifact['schema'] if s['numeric'])
    for invalid in ['not a number',float('inf')]:
        row={**records[0],key:invalid}
        with pytest.raises(ValueError,match='Invalid numeric'): predict(artifact,[row])

def test_missing_and_out_of_range(artifact,records):
    spec=next(s for s in artifact['schema'] if s['numeric'] and 'numeric_options' not in s); key=spec['name']
    assert predict(artifact,[{**records[0],key:None}])['results']
    assert predict(artifact,[{**records[0],key:spec['max']+10000}])['warnings']

def test_threshold_contract(artifact,records):
    if PROJECT=='property':
        with pytest.raises(ValueError): conformal_radius([], .9)
        with pytest.raises(ValueError): conformal_radius([1], .99)
        assert conformal_radius(range(1,11),.8)==9
    else:
        assert all(r['prediction']==1 for r in predict(artifact,records,0)['results'])
        assert all(r['prediction']==0 for r in predict(artifact,records,1)['results'])
        for invalid in [-.1,1.1,float('nan')]:
            with pytest.raises(ValueError): predict(artifact,records,invalid)

def test_csv_validation():
    with pytest.raises(ValueError,match='unique'): csv_records('a,a\n1,2')
    with pytest.raises(ValueError,match='headers'): csv_records('a,b\n1')
    with pytest.raises(ValueError,match='headers'): csv_records('a,b\n1,2,3')
    assert csv_records('a,b\n"x,y",2')==[{'a':'x,y','b':'2'}]
    assert csv_records('\ufeffa,b\n1,2')==[{'a':'1','b':'2'}]

@pytest.fixture(scope='module')
def server():
    instance=create_server(0); worker=threading.Thread(target=instance.serve_forever,daemon=True); worker.start()
    yield f'http://127.0.0.1:{instance.server_port}'
    instance.shutdown(); instance.server_close(); worker.join()

def request(server,path,payload=None,headers=None):
    data=json.dumps(payload).encode() if payload is not None else None
    req=urllib.request.Request(server+path,data=data,headers={'Content-Type':'application/json',**(headers or {})})
    try:
        with urllib.request.urlopen(req,timeout=30) as response: return response.status,response.read()
    except urllib.error.HTTPError as exc: return exc.code,exc.read()

def test_http_success_and_static(server,records,artifact):
    assert request(server,'/api/health')[0]==200
    assert request(server,'/')[0]==200
    assert request(server,'/app.js')[0]==200
    code,body=request(server,'/api/predict',{'records':records}); assert code==200
    assert json.loads(body)['results']==predict(artifact,records)['results']
    code,body=request(server,'/api/predict',{'csv':(OUT/'example_inputs.csv').read_text()}); assert code==200
    assert len(json.loads(body)['results'])==5

def test_http_error_boundaries(server):
    assert request(server,'/api/predict',{'records':[]})[0]==422
    assert request(server,'/api/predict',[])[0]==422
    assert request(server,'/../config.json')[0]==404
    assert request(server,'/api/health',headers={'Origin':'https://untrusted.example'})[0]==403
    assert request(server,'/api/health',headers={'Host':'untrusted.example'})[0]==403


def test_category_codes_and_bp_warning(artifact,records):
    expected={'property':0,'churn':1,'disease':6}[PROJECT]
    assert sum('numeric_options' in s for s in artifact['schema'])==expected
    for spec in artifact['schema']:
        if 'numeric_options' in spec:
            with pytest.raises(ValueError,match='Invalid category code'):
                predict(artifact,[{**records[0],spec['name']:12345}])
    if PROJECT=='disease':
        result=predict(artifact,[{**records[0],'ap_hi':70,'ap_lo':120}])
        assert any('blood-pressure' in warning for warning in result['warnings'])
