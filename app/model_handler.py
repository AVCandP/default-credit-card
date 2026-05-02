import os
import joblib
import numpy as np
import pandas as pd

FEATURE_ORDER = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_model(version: str = 'v1'):
    path = os.path.join(MODELS_DIR, f'model_{version}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def preprocess_input(data: dict) -> pd.DataFrame:
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    return pd.DataFrame([[float(data[f]) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
