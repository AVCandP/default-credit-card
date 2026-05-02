"""Train model_v1 (LogisticRegression) and model_v2 (RandomForest) on UCI credit card dataset."""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'UCI_Credit_Card.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def load_and_clean(path: str) -> tuple:
    df = pd.read_csv(path)
    df = df.drop(columns=['ID'])

    # Merge rare/unknown categories
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    X = df.drop(columns=['default.payment.next.month'])
    y = df['default.payment.next.month']
    return X, y


def evaluate(model, X_test, y_test, name: str):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"F1 (default class): {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['no default', 'default']))
    return f1


def main():
    print(f"Loading data from: {DATA_PATH}")
    X, y = load_and_clean(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Default rate in test: {y_test.mean():.3f}")

    # --- Model v1: LogisticRegression ---
    pipe_v1 = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=500)),
    ])
    pipe_v1.fit(X_train, y_train)
    evaluate(pipe_v1, X_test, y_test, 'v1 LogisticRegression')
    path_v1 = os.path.join(MODELS_DIR, 'model_v1.pkl')
    joblib.dump(pipe_v1, path_v1)
    print(f"Saved: {path_v1}")

    # --- Model v2: RandomForestClassifier ---
    pipe_v2 = Pipeline([
        ('clf', RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe_v2.fit(X_train, y_train)
    evaluate(pipe_v2, X_test, y_test, 'v2 RandomForest')
    path_v2 = os.path.join(MODELS_DIR, 'model_v2.pkl')
    joblib.dump(pipe_v2, path_v2)
    print(f"Saved: {path_v2}")


if __name__ == '__main__':
    main()
