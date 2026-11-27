import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from ml_projects.data import ROOT, FILES
from ml_projects.train import DIRECTORIES
from ml_projects.predict import predict_frame


@pytest.mark.parametrize('project',list(FILES))
def test_notebooks_reports_sources_and_figures(project):
    folder=ROOT/DIRECTORIES[project]
    out=folder/'outputs'
    manifest=json.loads((out/'run.json').read_text())
    assert manifest['source_sha256']==hashlib.sha256((ROOT/'datasets'/FILES[project]).read_bytes()).hexdigest()
    notebook=json.loads((folder/'analysis.ipynb').read_text(encoding='utf-8'))
    assert notebook['nbformat']==4
    codes=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(codes)>=6
    counts=[c['execution_count'] for c in codes]
    assert all(isinstance(count,int) and count>0 for count in counts)
    assert counts==list(range(counts[0],counts[0]+len(codes)))
    for cell in codes:
        source=cell['source']
        compile(''.join(source) if isinstance(source,list) else source,'notebook','exec')
        assert all(o['output_type']!='error' for o in cell['outputs'])
    log=json.loads((out/'notebook_execution.json').read_text())
    assert log['errors']==0 and log['code_cells_executed']==len(codes)
    figures=list((out/'figures').glob('*.svg'))
    assert len(figures)>=6
    report=(folder/'report.html').read_text(encoding='utf-8')
    for p in figures:
        root=ET.fromstring(p.read_text(encoding='utf-8'))
        assert root.tag.endswith('svg')
        assert f'outputs/figures/{p.name}' in report
        assert 'nan' not in p.read_text(encoding='utf-8').lower()
    assert 'Results and interpretation' in report
