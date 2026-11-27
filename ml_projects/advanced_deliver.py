"""Build a static advanced report and an actually executed companion notebook."""
import contextlib,io,json
from html import escape
from .advanced import ROOT,OUT,PROJECT
from .train import write_json

def build():
 s=json.loads((OUT/'summary.json').read_text()); title={'property':'Property value explorer','churn':'Customer retention studio','disease':'Disease model explorer'}[PROJECT]
 def table(rows):
  columns=list(dict.fromkeys(k for row in rows for k in row))
  def value(v): return f'{v:,.4f}' if isinstance(v,float) else str(v)
  return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(value(row.get(c,'')))+'</td>' for c in columns)+'</tr>' for row in rows)+'</tbody></table></div>'
 body=f'<p class="eyebrow">ADVANCED MODEL REPORT</p><h1>{title}</h1><p>{escape(s["evaluation_history"])}</p>'
 body+='<h2>Selected model: '+escape(s['winner'])+'</h2>'+table([{'metric':k,'value':v} for k,v in s['metrics'].items()])
 body+='<h2>Model selection</h2><p>Selection used fitting-only cross-validation; preprocessing was fitted inside each fold.</p>'+table(s['candidates'])
 body+='<h2>Partition boundaries</h2>'+table([s['partition_sizes']])
 body+='<h2>Uncertainty</h2><pre>'+escape(json.dumps(s['uncertainty'],indent=2))+'</pre>'
 if s['geographic_validation']: body+='<h2>Geographic stress check</h2><pre>'+escape(json.dumps(s['geographic_validation'],indent=2))+'</pre>'
 if PROJECT=='property': body+='<p>90% split-conformal intervals use a separate calibration partition. Coverage assumes exchangeability and is marginal, not conditional on each individual property. Symmetric limits can extend below zero; this is statistical uncertainty, not a plausible negative house value. These historic aggregate house values are not current appraisals.</p>'
 else:
  body+='<h2>Calibration and thresholds</h2><pre>'+escape((OUT/'calibration_validation.json').read_text())+'</pre><p>Probability calibration and threshold selection use disjoint partitions. Threshold selection is not performed on the historical test set.</p>'
  if PROJECT=='disease': body+='<p>Educational use only. No clinical validation, diagnosis, future-onset prediction or treatment recommendation is established. Age uses days. The height unit remains unresolved; BMI is omitted. Gender codes retain original values.</p>'
 body+='<h2>Group evaluation</h2>'+table(s['subgroups'])+'<h2>Global feature associations</h2><p>Permutation importance on a historical-test sample is descriptive, not causal.</p>'+table(s['feature_importance'])
 body+='<h2>Methods</h2><p><a href="https://scikit-learn.org/stable/modules/calibration.html">Scikit-learn calibration</a> · <a href="https://scikit-learn.org/stable/modules/cross_validation.html">Cross-validation</a> · <a href="https://arxiv.org/abs/2107.07511">Conformal prediction introduction</a></p>'
 style='body{font:16px/1.6 Segoe UI,sans-serif;max-width:1150px;margin:45px auto;padding:0 25px;color:#163044;background:#f8fbfc}h1{font-size:38px}h2{margin-top:36px}.eyebrow{color:#146b62;letter-spacing:.15em}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dce5ea;text-align:left}th{background:#e8f0f4}pre{background:#edf3f6;padding:20px;overflow:auto}a{color:#146b62}'
 (ROOT/'advanced_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
 sources=["from pathlib import Path\nimport sys, json\nroot = Path.cwd()\nif (root / "+repr(ROOT.name)+" / 'config.json').exists(): root = root / "+repr(ROOT.name)+"\nsys.path.insert(0, str(root))\nfrom ml_projects.advanced import OUT, PROJECT, load_artifact, predict\nimport pandas as pd\nsummary = json.loads((OUT / 'summary.json').read_text())\nprint(summary['evaluation_history'])\nprint('Selected model:', summary['winner'])",
 "print(pd.DataFrame([summary['metrics']]).to_string(index=False))\nprint(json.dumps(summary['uncertainty'], indent=2))",
 "parts = json.loads((OUT / 'partitions.json').read_text())\nsets = [set(rows) for rows in parts.values()]\nassert all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i+1:])\nprint({name: len(rows) for name, rows in parts.items()})\nprint('All fitting, calibration, threshold and historical-test partitions are disjoint.')",
 "sample = pd.read_csv(OUT / 'example_inputs.csv')\nresult = predict(load_artifact(), json.loads(sample.to_json(orient='records')))\nprint(pd.DataFrame(result['results']).to_string(index=False))\nprint(result['warnings'])",
 "print(pd.DataFrame(summary['subgroups']).to_string(index=False))\nprint(pd.DataFrame(summary['feature_importance']).head(10).to_string(index=False))"]
 cells=[{'cell_type':'markdown','metadata':{},'source':'# '+title+' — advanced workflow\n\nRun this notebook with a fresh kernel from this project folder. Train the advanced models first with `python -m ml_projects.advanced --train`. This notebook inspects and verifies trained artifacts; it does not silently retrain them.\n\n'+s['evaluation_history']+'\n\nSee MODEL_CARD.md and advanced_report.html for limitations.'}]
 namespace={'__name__':'__advanced_notebook__'}
 for i,source in enumerate(sources,1):
  stream=io.StringIO()
  with contextlib.redirect_stdout(stream): exec(compile(source,'advanced_analysis.ipynb','exec'),namespace)
  cells.append({'cell_type':'code','metadata':{},'source':source,'execution_count':i,'outputs':[{'output_type':'stream','name':'stdout','text':stream.getvalue()}]})
 for i,c in enumerate(cells): c['id']=f'advanced-{PROJECT}-{i}'
 write_json(ROOT/'advanced_analysis.ipynb',{'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'execution_method':'Executed ordinary Python cells with captured stdout; not an interactive IDE execution'},'nbformat':4,'nbformat_minor':5})
 print(PROJECT+': advanced report and five executed notebook cells generated')

if __name__=='__main__': build()
