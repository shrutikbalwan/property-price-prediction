"""Run and record reproducibility, artifact and application checks."""
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from .advanced import ROOT,OUT,PROJECT
from .advanced_deliver import build

def main():
    folder=ROOT/'verification';folder.mkdir(exist_ok=True)
    first=folder/'advanced_first_run.json'
    current=json.loads((OUT/'summary.json').read_text())
    reproduced=None
    if first.exists():
        old=json.loads(first.read_text())
        assert old['metrics']==current['metrics'], 'Metrics differ from the recorded earlier training run'
        assert old['winner']==current['winner'] and old['candidates']==current['candidates']
        reproduced=True
    build()
    result=subprocess.run([sys.executable,'-m','pytest','tests','-q','-p','no:cacheprovider'],cwd=ROOT,capture_output=True,text=True)
    (folder/'advanced_pytest.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    print(result.stdout,flush=True)
    result.check_returncode()
    run=subprocess.run([sys.executable,'-m','ml_projects.advanced','--input',str(OUT/'example_inputs.csv'),'--output',str(folder/'advanced_predictions.csv')],cwd=ROOT,capture_output=True,text=True)
    run.check_returncode()
    notebook=json.loads((ROOT/'advanced_analysis.ipynb').read_text())
    record={'project':PROJECT,'verified_at_utc':datetime.now(timezone.utc).isoformat(),'tests_exit_code':result.returncode,'cli_exit_code':run.returncode,'repeated_training_metrics_identical':reproduced,'advanced_notebook_python_cells':len([c for c in notebook['cells'] if c['cell_type']=='code']),'native_advanced_notebook_ide_run':False,'model_sha256':hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest(),'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(ROOT/'ml_projects').glob('*.py'),*(ROOT/'web').glob('*'),ROOT/'config.json']},'scope':'Local installed Python environment. HTTP tests use real loopback requests. No clean dependency installation or GitHub publication claimed.'}
    (folder/'advanced_checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Advanced acceptance checks passed.',flush=True)

if __name__=='__main__':main()
