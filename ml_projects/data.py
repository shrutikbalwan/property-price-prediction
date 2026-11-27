"""Read the supplied workbooks without modifying their contents.

The standard-library reader is limited to flat data worksheets, not general Excel
calculation. It consumes cached values and rejects formula cells without caches.
"""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[1]
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
TARGETS = {'property': 'median_house_value'}
FILES = {'property': 'project 1.xlsx'}


def read_dataset(project):
    path = ROOT / 'datasets' / FILES[project]
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(
                archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        records = []
        for row in sheet.findall('s:sheetData/s:row', NS):
            record = {}
            for cell in row:
                column = ''.join(filter(str.isalpha, cell.attrib['r']))
                value = cell.find('s:v', NS)
                if cell.find('s:f', NS) is not None and value is None:
                    raise ValueError('Uncached formula in source workbook')
                text = value.text if value is not None else ''.join(cell.itertext())
                if cell.get('t') == 's':
                    text = strings[int(text)]
                record[column] = text
            records.append(record)
    header, *records = records
    frame = pd.DataFrame([{name: row.get(key, '') for key, name in header.items()}
                          for row in records])
    frame = frame.replace(r'^\s*$', np.nan, regex=True)
    for column in frame:
        if column not in {'date', 'customerID'}:
            numeric = pd.to_numeric(frame[column], errors='coerce')
            if numeric.notna().sum() == frame[column].notna().sum():
                frame[column] = numeric
    return frame


class PrepareFeatures(TransformerMixin, BaseEstimator):
    """Deterministic, row-local transformations shared by training and inference.

    No statistics are learned here. Missing-value imputation belongs to the
    downstream pipeline. Disease BP bounds are broad data-quality assumptions,
    not diagnostic thresholds. Invalid readings are marked missing, not dropped.
    """
    def __init__(self, project, expanded=False, clean_bp=True):
        self.project = project
        self.expanded = expanded
        self.clean_bp = clean_bp

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = X.copy().replace(r'^\s*$', np.nan, regex=True)
        frame = frame.drop(columns=[TARGETS[self.project]], errors='ignore')
        if self.project == 'churn':
            frame = frame.drop(columns=['customerID'], errors='ignore')
            frame['TotalCharges'] = pd.to_numeric(frame['TotalCharges'], errors='coerce')
        elif self.project == 'disease':
            frame = frame.drop(columns=['id', 'date'], errors='ignore')
            if not self.expanded:
                frame = frame.drop(columns=['country', 'occupation'], errors='ignore')
            frame['age'] = pd.to_numeric(frame['age'], errors='coerce') / 365.25
            for col in ['ap_hi', 'ap_lo', 'height', 'weight']:
                frame[col] = pd.to_numeric(frame[col], errors='coerce')
            if self.clean_bp:
                invalid = ((frame.ap_hi <= 0) | (frame.ap_lo <= 0)
                           | (frame.ap_hi > 300) | (frame.ap_lo > 200)
                           | (frame.ap_lo >= frame.ap_hi))
                frame.loc[invalid, ['ap_hi', 'ap_lo']] = np.nan
            for col in ['gender', 'cholesterol', 'gluc']:
                # The source uses numeric codes stored as floats. Preserve the
                # fitted encoder vocabulary for equivalent int/float CSV inputs.
                frame[col] = frame[col].map(lambda v: str(float(v)) if pd.notna(v) else np.nan)
        return frame


def split_target(frame, project):
    target = TARGETS[project]
    y = frame[target]
    if project == 'churn':
        y = y.map({'No': 0, 'Yes': 1})
    else:
        y = pd.to_numeric(y, errors='raise')
    if y.isna().any():
        raise ValueError('Missing or unrecognized target labels')
    if project != 'property' and not set(y.unique()) <= {0, 1}:
        raise ValueError('Expected binary target')
    return frame.drop(columns=[target]), y
