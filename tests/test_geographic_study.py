"""Independent checks for geographic isolation, intervals, maps and recorded results."""
import hashlib,json
import numpy as np
import pandas as pd
import pytest
from ml_projects.geographic_study import ROOT,OUT,PROTOCOL,blocks,nearest_km,buffered_ids,measure,decision
from ml_projects.advanced import conformal_radius
from ml_projects.data import read_dataset,split_target

@pytest.fixture(scope='module')
def data():return split_target(read_dataset('property'),'property')

@pytest.fixture(scope='module')
def predictions():return pd.read_csv(OUT/'out_of_fold_predictions.csv')

def test_blocks_and_great_circle_distance():
    frame=pd.DataFrame({'latitude':[34.,34.99,35.01],'longitude':[-118.01,-118.99,-117.99]})
    assert blocks(frame).tolist()==['34/-119','34/-119','35/-118']
    origin=pd.DataFrame({'latitude':[0.],'longitude':[0.]})
    query=pd.DataFrame({'latitude':[0.,1.],'longitude':[0.,0.]})
    distances=nearest_km(query,origin)
    assert distances[0]==pytest.approx(0)
    assert distances[1]==pytest.approx(111.19508,rel=1e-6)
    kept,dropped=buffered_ids(query,np.array([1]),np.array([0]),100)
    assert kept.tolist()==[1] and not len(dropped)
    with pytest.raises(ValueError):nearest_km(origin,origin.iloc[:0])

def test_partition_isolation_and_buffer(data):
    X,_=data;parts=json.loads((OUT/'partitions.json').read_text());original=json.loads((ROOT/'outputs/split.json').read_text())
    train=set(original['train_rows']);historical=set(original['test_rows']);by_mode={m:[] for m in PROTOCOL['modes']}
    geo_evaluations={}
    for part in parts:
        sets=[set(part[k]) for k in ['fit','calibration','evaluation','buffer_excluded']]
        assert set.union(*sets)==train
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        assert all(not a&historical for a in sets)
        by_mode[part['mode']].extend(part['evaluation'])
        if part['mode']!='random_control':
            groups=[set(blocks(X.loc[part[k]])) for k in ['fit','calibration','evaluation']]
            assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
            if part['mode']=='geographic_blocks':geo_evaluations[part['fold']]=sets[2]
            else:
                assert sets[2]==geo_evaluations[part['fold']]
                development=part['fit']+part['calibration']
                assert nearest_km(X.loc[development],X.loc[part['evaluation']]).min()>=10-1e-9
                if part['buffer_excluded']:
                    assert (nearest_km(X.loc[part['buffer_excluded']],X.loc[part['evaluation']])<10).all()
    for ids in by_mode.values():assert set(ids)==train and len(ids)==len(train)

def test_predictions_targets_and_intervals(data,predictions):
    X,y=data;folds=pd.read_csv(OUT/'fold_metrics.csv');cal=pd.read_csv(OUT/'calibration_residuals.csv')
    for (mode,model,fold),frame in predictions.groupby(['mode','model','fold']):
        np.testing.assert_array_equal(frame.actual,y.loc[frame.source_row].to_numpy())
        row=folds[(folds['mode']==mode)&(folds.model==model)&(folds.fold==fold)].iloc[0]
        residual=cal[(cal['mode']==mode)&(cal.model==model)&(cal.fold==fold)].absolute_residual.to_numpy()
        radius=conformal_radius(residual,.9)
        assert row.interval_radius==pytest.approx(radius)
        np.testing.assert_allclose(frame.lower,frame.prediction-radius)
        np.testing.assert_allclose(frame.upper,frame.prediction+radius)
        for key,value in measure(frame).items():assert row[key]==pytest.approx(value)
        assert np.isfinite(frame[['prediction','lower','upper']]).all().all()
    for (mode,model),frame in predictions.groupby(['mode','model']):
        assert not frame.source_row.duplicated().any()
        assert len(frame)==16512

def test_pooled_and_subgroup_metrics(predictions):
    pooled=pd.read_csv(OUT/'pooled_metrics.csv')
    for _,row in pooled.iterrows():
        frame=predictions[(predictions['mode']==row['mode'])&(predictions.model==row.model)]
        for key,value in measure(frame).items():assert row[key]==pytest.approx(value)
    locations=pd.read_csv(OUT/'location_metrics.csv')
    for _,row in locations.iterrows():
        frame=predictions[(predictions['mode']==row['mode'])&(predictions.model==row.model)&(predictions.block==row.block)]
        for key,value in measure(frame).items():assert row[key]==pytest.approx(value)
        assert row.small_sample==(len(frame)<100)
    prices=pd.read_csv(OUT/'price_band_metrics.csv')
    assert prices.groupby(['mode','model']).rows.sum().eq(16512).all()

def test_decision_and_active_models():
    summary=json.loads((OUT/'summary.json').read_text())
    calculated=decision(pd.read_csv(OUT/'pooled_metrics.csv'),pd.read_csv(OUT/'fold_metrics.csv'))
    assert calculated['point_advantage_confirmed']==summary['decision']['point_advantage_confirmed']
    assert calculated['interval_transfer_flag']==summary['decision']['interval_transfer_flag']
    assert calculated['buffered_advanced_coverage']==pytest.approx(summary['decision']['buffered_advanced_coverage'])
    for name,digest in summary['active_model_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest

def test_report_and_offline_map():
    html=(ROOT/'PROPERTY_GEOGRAPHIC_REPORT.html').read_text(encoding='utf-8')
    assert 'outputs/geographic_study/geographic_maps.html' in html
    chart=(OUT/'geographic_maps.html').read_text(encoding='utf-8')
    assert 'Plotly.newPlot' in chart and 'geographic_buffer_10km' in chart
    assert '<script src=' not in chart
    assert len(chart)>100000
