import json
import pytest
from app.api import app

SAMPLE = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'ok'
    assert 'models_loaded' in body


def test_predict_v1(client):
    payload = {**SAMPLE, 'model_version': 'v1'}
    r = client.post('/predict', data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 200
    body = r.get_json()
    assert body['prediction'] in [0, 1]
    assert 0.0 <= body['probability'] <= 1.0
    assert body['model_version'] == 'v1'


def test_predict_v2(client):
    payload = {**SAMPLE, 'model_version': 'v2'}
    r = client.post('/predict', data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 200
    body = r.get_json()
    assert body['prediction'] in [0, 1]
    assert body['model_version'] == 'v2'


def test_ab_routing_deterministic(client):
    """Same client_id must always route to the same model version."""
    payload = {**SAMPLE, 'model_version': 'ab', 'client_id': 'test-user-123'}
    results = set()
    for _ in range(5):
        r = client.post('/predict', data=json.dumps(payload),
                        content_type='application/json')
        results.add(r.get_json()['model_version'])
    assert len(results) == 1, "A/B routing is not deterministic for the same client_id"


def test_predict_missing_features(client):
    r = client.post('/predict', data=json.dumps({'model_version': 'v1'}),
                    content_type='application/json')
    assert r.status_code == 400
    assert 'error' in r.get_json()


def test_predict_unknown_version(client):
    payload = {**SAMPLE, 'model_version': 'v99'}
    r = client.post('/predict', data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 400
