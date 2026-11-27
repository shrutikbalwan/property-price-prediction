# Try the synthetic demo

The demo needs no original workbook, trained model, account or API key.
It is a fictional interface simulator, not a model trained on synthetic data.
It demonstrates single inputs, CSV batches, threshold controls and downloads.

From this repository directory with Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt
.\.venv\Scripts\python.exe -m ml_projects.app --demo --port 8501
```

Open http://127.0.0.1:8501. Stop with Ctrl+C. On macOS/Linux, use
`python3.13` to create the environment and `.venv/bin/python` afterward.

1. The page must display **SYNTHETIC DEMO** before entering inputs.
2. Run the supplied fictional example, then change an input within its displayed range.
3. For classification demos, adjust the threshold and run again to change the flag.
4. Open Batch analysis, download the three-row example CSV and upload it.
5. Analyze the batch and download results. Every result has `mode=synthetic_demo`;
   the download filename also identifies the demo.
6. Open Demo explanation to inspect the exact invented formula.

No source records are reused, no model artifacts are loaded or modified, and
inputs are not saved by the server. Missing inputs use the fictional example
value; invalid numbers and values outside demo ranges are rejected.
Scores are not calibrated probabilities. Property values use arbitrary units;
classification signals have no clinical meaning. There are no performance
metrics, confidence intervals or real-world claims in this mode.

To run the trained application, reproduce the artifacts as described in
[GITHUB_PREPARATION.md](GITHUB_PREPARATION.md), then omit `--demo`.
Missing trained artifacts produce an explicit error, never a silent demo fallback.

For a short demo recording: show the mode banner, run a single example, run
the supplied CSV, download its marked results, and finish on Demo explanation.
Keep the synthetic label visible in screenshots and recordings.
