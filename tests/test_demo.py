"""Exercise the actual demo HTTP server without source data or trained artifacts."""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
import pytest
from ml_projects import app
from ml_projects.demo import demo_bundle, demo_predict

@pytest.mark.parametrize('project', ['property', 'churn', 'disease'])
def test_deterministic_demo_batch_and_boundaries(project):
    schema, sample, summary, csv = demo_bundle(project)
    records=app.csv_records(csv.decode())
    result=demo_predict(project,schema,sample,records)
    assert result == demo_predict(project,schema,sample,records)
    assert result['mode']=='synthetic_demo'
    for row, expected in zip(records,result['results']):
        assert demo_predict(project,schema,sample,[row])['results']==[expected]
        assert expected['mode']=='synthetic_demo'
    assert demo_predict(project,schema,sample,list(reversed(records)))['results']==list(reversed(result['results']))
    if project=='property':
        assert result['results'][0]['prediction']==145000
        assert 'lower' not in result['results'][0]
    else:
        assert all(r['prediction']==1 for r in demo_predict(project,schema,sample,records,0)['results'])
        assert all(0<=r['probability']<=1 for r in result['results'])
    key=schema[0]['name']
    assert len(demo_predict(project,schema,sample,[{**sample,key:None}])['warnings'])==2
    for rows in [[],[{}],[sample]*5001,[{**sample,key:float('nan')}],
                 [{**sample,key:True}],[{**sample,key:schema[0]['max']+1}]]:
        with pytest.raises(ValueError):
            demo_predict(project,schema,sample,rows)
    for threshold in [-1,2,float('inf')]:
        with pytest.raises(ValueError):
            demo_predict(project,schema,sample,[sample],threshold)

def request(base,path,payload=None,headers=None):
    req=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json',**(headers or {})})
    try:
        with urllib.request.urlopen(req,timeout=10) as response:
            return response.status,response.read()
    except urllib.error.HTTPError as exc:
        return exc.code,exc.read()

def test_demo_http_without_artifacts(monkeypatch):
    def forbidden():
        raise AssertionError('Demo must never load a trained artifact')
    monkeypatch.setattr(app,'load_artifact',forbidden)
    monkeypatch.setattr(app,'OUT',Path('nonexistent-demo-test-artifacts'))
    server=app.create_server(0,demo=True)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        status,raw=request(base,'/api/schema')
        assert status==200
        schema=json.loads(raw)
        assert schema['mode']=='synthetic_demo'
        assert 'metrics' not in schema['summary']
        assert json.loads(request(base,'/api/health')[1])['mode']=='synthetic_demo'
        assert b'/demo.js' in request(base,'/')[1]
        assert request(base,'/demo.js')[0]==200
        status,raw=request(base,'/api/predict',{'records':[schema['sample']]})
        assert status==200
        single=json.loads(raw)
        assert single['results'][0]['mode']=='synthetic_demo'
        csv=request(base,'/example.csv')[1].decode()
        status,raw=request(base,'/api/predict',{'csv':csv})
        batch=json.loads(raw)
        assert status==200 and len(batch['results'])==3
        assert batch['results'][0]==single['results'][0]
        assert request(base,'/api/predict',{'records':[]})[0]==422
        assert request(base,'/api/health',headers={'Origin':'https://untrusted.example'})[0]==403
        assert request(base,'/../config.json')[0]==404
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=10)

def test_trained_mode_never_silently_uses_demo(monkeypatch):
    def missing():
        raise FileNotFoundError('missing test artifact')
    monkeypatch.setattr(app,'load_artifact',missing)
    with pytest.raises(FileNotFoundError,match='--demo'):
        app.create_server(0)
