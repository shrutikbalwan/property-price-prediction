"""Fictional UI simulator. No training, dataset reads, model loads or saved outputs."""
import csv
import io
import math

NOTICE = ('SYNTHETIC DEMO: invented inputs and formulas only. '
          'Not a trained model, property valuation, customer forecast or medical assessment.')

def demo_bundle(project):
    definitions = {
        'property': [('floor_area', 20, 500, 100, 'Invented area units'),
                     ('rooms', 1, 10, 3, 'Invented room count')],
        'churn': [('months', 0, 120, 12, 'Invented subscription duration'),
                  ('service_score', 0, 10, 5, 'Fictional satisfaction score')],
        'disease': [('signal_a', 0, 10, 4, 'Fictional signal, not a clinical measurement'),
                    ('signal_b', 0, 10, 6, 'Fictional signal, not a clinical measurement')],
    }
    schema = [dict(name=n, numeric=True, min=lo, max=hi, unit=unit)
              for n, lo, hi, default, unit in definitions[project]]
    sample = {n: default for n, lo, hi, default, unit in definitions[project]}
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(sample))
    writer.writeheader()
    writer.writerow(sample)
    writer.writerow({s['name']: s['min'] for s in schema})
    writer.writerow({s['name']: s['max'] for s in schema})
    summary = {'winner': 'fictional_formula', 'threshold': .5, 'demo_notice': NOTICE}
    return schema, sample, summary, output.getvalue().encode('utf-8')

def demo_predict(project, schema, defaults, records, threshold=None):
    if (not isinstance(records, list) or not 1 <= len(records) <= 5000
            or not all(isinstance(row, dict) for row in records)):
        raise ValueError('Provide 1 to 5000 records as objects')
    t = .5 if threshold is None else float(threshold)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError('Threshold must be between 0 and 1')
    warnings = [NOTICE]
    results = []
    for row in records:
        values = {}
        for spec in schema:
            name = spec['name']
            if name not in row:
                raise ValueError('Missing columns: ' + name)
            raw = row[name]
            if raw is None or isinstance(raw, str) and not raw.strip():
                value = defaults[name]
                message = name + ': missing input replaced with the fictional example value'
                if message not in warnings:
                    warnings.append(message)
            else:
                if isinstance(raw, bool):
                    raise ValueError('Invalid numeric value in ' + name)
                try:
                    value = float(raw)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError('Invalid numeric value in ' + name) from None
                if not math.isfinite(value):
                    raise ValueError('Invalid numeric value in ' + name)
            if not spec['min'] <= value <= spec['max']:
                raise ValueError(name + ': use the displayed demo range')
            values[name] = value
        if project == 'property':
            result = {'prediction': 10000 + 1200 * values['floor_area'] + 5000 * values['rooms']}
        else:
            score = ((1 - values['months'] / 120) + (1 - values['service_score'] / 10)) / 2 if project == 'churn' else (values['signal_a'] + values['signal_b']) / 20
            result = {'probability': score, 'prediction': int(score >= t), 'threshold': t}
        result['mode'] = 'synthetic_demo'
        results.append(result)
    return {'results': results, 'warnings': warnings, 'mode': 'synthetic_demo'}
