import hashlib
import logging
import json
import time
from flask import Flask, request, jsonify
from app.model_handler import load_model, preprocess_input
from app.mq_publisher import publish_prediction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

models = {}


def _load_all_models():
    for v in ('v1', 'v2'):
        try:
            models[v] = load_model(v)
            logger.info(json.dumps({'event': 'model_loaded', 'version': v}))
        except FileNotFoundError as e:
            logger.warning(str(e))


_load_all_models()


def _route_version(client_id: str) -> str:
    """Deterministic 50/50 A/B routing by hash of client_id."""
    h = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
    return 'v1' if h % 2 == 0 else 'v2'


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Credit Card Default Prediction',
        'endpoints': {
            'GET  /health': 'service health check',
            'POST /predict': 'predict default probability',
        },
        'model_versions': list(models.keys()),
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Credit card default prediction endpoint.

    Request JSON fields:
        LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
        PAY_0, PAY_2–PAY_6, BILL_AMT1–6, PAY_AMT1–6  (all numeric)
        model_version (str, optional): 'v1' | 'v2' | 'ab'  (default: 'v1')
        client_id (str, optional): used for A/B routing when model_version='ab'

    Response JSON:
        prediction (int): 0 = no default, 1 = default
        probability (float): probability of default
        model_version (str): version that produced the result
    """
    t0 = time.time()
    try:
        data = request.get_json(force=True)
        version = data.get('model_version', 'v1')

        if version == 'ab':
            client_id = data.get('client_id', 'anonymous')
            version = _route_version(client_id)

        if version not in models:
            return jsonify({'error': f"Model '{version}' not loaded"}), 400

        features = preprocess_input(data)
        model = models[version]
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        payload = {
            'event': 'prediction',
            'model_version': version,
            'prediction': prediction,
            'probability': probability,
            'latency_ms': elapsed_ms,
        }
        logger.info(json.dumps(payload))
        publish_prediction(payload)

        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 4),
            'model_version': version,
        }), 200

    except KeyError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(json.dumps({'event': 'error', 'message': str(e)}))
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
