"""Verify this standalone project from its own directory."""
import contextlib, io, json, hashlib, subprocess, sys
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from .data import ROOT, FILES
from .predict import predict_frame

def main():
    project=next(iter(FILES))
    out=ROOT/'outputs'
    manifest=json.loads((out/'run.json').read_text())
    assert hashlib.sha256((ROOT/'datasets'/FILES[project]).read_bytes()).hexdigest()==manifest['source_sha256']
    result=subprocess.run([sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider'],cwd=ROOT)
    result.check_returncode()
    notebook=json.loads((ROOT/'analysis.ipynb').read_text(encoding='utf-8'))
    namespace={'__name__':'__verification__'}
    count=0
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            source=cell['source']
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(''.join(source) if isinstance(source,list) else source,'analysis.ipynb','exec'),namespace)
            count+=1
    output=ROOT/'verification'; output.mkdir(exist_ok=True)
    subprocess.run([sys.executable,'-m','ml_projects.predict','--input',str(out/'example_inputs.csv'),'--output',str(output/'cli_predictions.csv')],cwd=ROOT,check=True)
    actual=pd.read_csv(output/'cli_predictions.csv'); expected=pd.read_csv(out/'example_predictions.csv')
    for column in expected:
        np.testing.assert_allclose(actual[column],expected[column],rtol=1e-10,atol=1e-10)
    record={'project':project,'project_root':str(ROOT),'verified_at_utc':datetime.now(timezone.utc).isoformat(),'tests_exit_code':0,'notebook_python_cells':count,'errors':0,'prediction_matches':True,'method':'Standalone Python execution and pytest; not an interactive frontend run'}
    (output/'standalone.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__': main()
