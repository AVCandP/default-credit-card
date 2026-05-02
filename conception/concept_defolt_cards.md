# Концепция: Сервис прогнозирования дефолта по кредитным картам

**Проект:** Default Credit Card Prediction Service  
**Дата:** 2026-05-02  
**Датасет:** UCI Default of Credit Card Clients (Тайвань, апрель–сентябрь 2005)  
**Область:** Finance / Credit Scoring / MLOps  

---

## 1. Постановка задачи

### 1.1 Бизнес-цель

Построить производственный ML-сервис, который по входному набору признаков клиента (кредитный лимит, демография, история платежей за 6 месяцев) предсказывает вероятность дефолта по кредитной карте в следующем месяце.

**Ключевые бизнес-метрики:**

| Метрика | Описание |
|---|---|
| Снижение ожидаемых кредитных потерь | `Δ_loss = (FN_old − FN_new) × avg_exposure` — сколько дополнительных дефолтов модель v2 перехватывает vs v1 |
| Уровень одобрения при фиксированном риске | Доля одобренных заявок при порог-скоринге ≤ 5 % — рост означает, что модель более точно разграничивает риски |

### 1.2 ML-задача

- **Тип:** Бинарная классификация (0 = нет дефолта, 1 = дефолт)
- **Целевая переменная:** `default.payment.next.month`
- **Признаки:** 23 переменные (LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0–PAY_6, BILL_AMT1–6, PAY_AMT1–6)
- **Дисбаланс классов:** ~22 % дефолтов → приоритет метрики F1 для класса «дефолт»

### 1.3 Технические требования к системе

| Требование | Детали |
|---|---|
| API | Flask POST /predict, GET /health |
| Контейнеризация | Docker образ, python:3.12-slim, порт 5000 |
| Воспроизводимость | requirements.txt, Dockerfile, venv |
| A/B-тестирование | Маршрутизация v1/v2, статистика t-test/z-test |
| Документация | README.md + ARCHITECTURE.md |
| Репозиторий | Структура cookiecutter-data-science |

---

## 2. Структура репозитория

```
default_credit_card/
├── app/
│   ├── __init__.py
│   ├── api.py                  # Flask-приложение
│   └── model_handler.py        # Загрузка модели и препроцессинг
├── models/
│   ├── train_model.py          # Обучение и сохранение модели
│   ├── model_v1.pkl            # Модель v1 (контроль, LogisticRegression)
│   └── model_v2.pkl            # Модель v2 (эксперимент, RandomForest/GBM)
├── notebooks/
│   └── eda_and_training.ipynb  # Разведочный анализ и обучение
├── tests/
│   └── test_api.py             # Интеграционные тесты API
├── data/
│   ├── UCI_Credit_Card.csv
│   └── about_dataset.txt
├── docker/
│   └── Dockerfile
├── conception/
│   └── concept_defolt_cards.md
├── requirements.txt
├── docker-compose.yml          # Бонус: оркестрация
├── ab_test_plan.md
├── ARCHITECTURE.md
└── README.md
```

---

## 3. Пошаговая инструкция реализации

### Шаг 1 — Подготовка среды

```bash
# 1. Создать виртуальное окружение
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 2. Установить зависимости
pip install flask numpy pandas scikit-learn joblib pytest requests
pip freeze > requirements.txt
```

**requirements.txt (минимальный набор):**
```
flask==3.1.0
numpy==2.2.5
pandas==2.2.3
scikit-learn==1.6.1
joblib==1.4.2
pytest==8.3.5
requests==2.32.3
gunicorn==23.0.0
```

---

### Шаг 2 — Разведочный анализ данных (EDA)

Файл: `notebooks/eda_and_training.ipynb`

**Обязательные шаги EDA:**

1. **Загрузка и первичный осмотр**
   ```python
   import pandas as pd
   df = pd.read_csv('data/UCI_Credit_Card.csv')
   df.info()
   df.describe()
   df['default.payment.next.month'].value_counts(normalize=True)
   ```

2. **Анализ пропусков и аномалий**
   - EDUCATION: значения 0, 5, 6 → объединить в категорию «другое»
   - MARRIAGE: значение 0 → объединить с 3 («другое»)
   - PAY_0–PAY_6: значения -2, -1 означают «нет задолженности» / «оплачено»

3. **Корреляционный анализ признаков с целевой переменной**
   - Наиболее важные предикторы: PAY_0, PAY_2, PAY_3 (история последних просрочек)
   - LIMIT_BAL — отрицательная корреляция с дефолтом

4. **Распределение дисбаланса классов**
   - Класс 0 (нет дефолта): ~77.8 %
   - Класс 1 (дефолт): ~22.2 %
   - Решение: параметр `class_weight='balanced'` или `scale_pos_weight`

---

### Шаг 3 — Обучение и сохранение моделей

Файл: `models/train_model.py`

**Модель v1 — Logistic Regression (контроль):**
```python
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/UCI_Credit_Card.csv')
df = df.drop(columns=['ID'])

# Очистка категорий
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

X = df.drop(columns=['default.payment.next.month'])
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline_v1 = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=500))
])

pipeline_v1.fit(X_train, y_train)
y_pred = pipeline_v1.predict(X_test)
print("Model v1 F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(pipeline_v1, 'models/model_v1.pkl')
print("model_v1.pkl saved")
```

**Модель v2 — Random Forest (эксперимент):**
```python
from sklearn.ensemble import RandomForestClassifier

pipeline_v2 = Pipeline([
    ('clf', RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_v2.fit(X_train, y_train)
y_pred_v2 = pipeline_v2.predict(X_test)
print("Model v2 F1:", f1_score(y_test, y_pred_v2))

joblib.dump(pipeline_v2, 'models/model_v2.pkl')
print("model_v2.pkl saved")
```

**Ожидаемые результаты (UCI dataset):**

| Модель | F1 (дефолт) | Precision | Recall |
|---|---|---|---|
| v1 LogisticRegression | ~0.48–0.52 | ~0.62–0.66 | ~0.40–0.44 |
| v2 RandomForest | ~0.47–0.53 | ~0.65–0.70 | ~0.38–0.43 |

---

### Шаг 4 — Flask API

Файл: `app/model_handler.py`
```python
import joblib
import numpy as np

FEATURE_ORDER = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

def load_model(version='v1'):
    path = f'models/model_{version}.pkl'
    return joblib.load(path)

def preprocess_input(data: dict) -> np.ndarray:
    return np.array([[data[f] for f in FEATURE_ORDER]])
```

Файл: `app/api.py`
```python
from flask import Flask, request, jsonify
import numpy as np
from app.model_handler import load_model, preprocess_input

app = Flask(__name__)
models = {
    'v1': load_model('v1'),
    'v2': load_model('v2')
}

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for credit card default prediction.
    
    Request JSON:
        features (dict): client features (23 fields)
        model_version (str): 'v1' or 'v2' (default: 'v1')
    
    Response JSON:
        prediction (int): 0 or 1
        probability (float): probability of default
        model_version (str): version used
    """
    try:
        data = request.get_json()
        version = data.get('model_version', 'v1')
        features = preprocess_input(data)
        model = models[version]
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])
        return jsonify({
            'prediction': prediction,
            'probability': round(probability, 4),
            'model_version': version
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models_loaded': list(models.keys())}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Формат запроса к /predict:**
```json
{
  "LIMIT_BAL": 20000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 24,
  "PAY_0": 2,
  "PAY_2": 2,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -2,
  "PAY_6": -2,
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0,
  "model_version": "v1"
}
```

**Пример ответа:**
```json
{
  "prediction": 1,
  "probability": 0.7823,
  "model_version": "v1"
}
```

**curl-команды для демонстрации:**
```bash
# Проверка здоровья сервиса
curl http://localhost:5000/health

# Предсказание (v1)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0,"model_version":"v1"}'

# Предсказание (v2)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{... "model_version":"v2"}'
```

---

### Шаг 5 — Контейнеризация

Файл: `docker/Dockerfile`
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и моделей
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "app/api.py"]
```

**Команды Docker:**
```bash
# Сборка образа
docker build -f docker/Dockerfile -t default-credit-card:latest .

# Запуск контейнера
docker run -d -p 5000:5000 --name credit-card-api default-credit-card:latest

# Проверка логов
docker logs credit-card-api

# Остановка и удаление
docker stop credit-card-api && docker rm credit-card-api

# Публикация на Docker Hub
docker tag default-credit-card:latest <dockerhub-username>/default-credit-card:latest
docker push <dockerhub-username>/default-credit-card:latest
```

**Файл docker-compose.yml (бонус +3 балла):**
```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

### Шаг 6 — Тесты

Файл: `tests/test_api.py`
```python
import pytest
import json
from app.api import app

SAMPLE_INPUT = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.get_json()['status'] == 'ok'

def test_predict_v1(client):
    payload = {**SAMPLE_INPUT, 'model_version': 'v1'}
    r = client.post('/predict', data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 200
    data = r.get_json()
    assert data['prediction'] in [0, 1]
    assert 0.0 <= data['probability'] <= 1.0
    assert data['model_version'] == 'v1'

def test_predict_v2(client):
    payload = {**SAMPLE_INPUT, 'model_version': 'v2'}
    r = client.post('/predict', data=json.dumps(payload),
                    content_type='application/json')
    assert r.status_code == 200

def test_predict_invalid_input(client):
    r = client.post('/predict', data=json.dumps({}),
                    content_type='application/json')
    assert r.status_code == 400
```

```bash
pytest tests/ -v
```

---

### Шаг 7 — A/B-тестирование

Файл: `ab_test_plan.md`

#### 7.1 Гипотезы

| | Описание |
|---|---|
| **H₀** | F1-скор модели v2 ≤ F1-скор модели v1 (нет улучшения) |
| **H₁** | F1-скор модели v2 > F1-скор модели v1 (есть улучшение) |

#### 7.2 Дизайн эксперимента

```
Входящий трафик (N запросов)
          │
    ┌─────┴─────┐
    │  Router   │  ← случайное распределение по cookie/hash(client_id)
    └─────┬─────┘
    50%   │   50%
  ┌───────┘   └───────┐
  ▼                   ▼
Model v1           Model v2
(control)        (treatment)
LogisticReg     RandomForest
  │                   │
  └──────┬────────────┘
         ▼
    Логирование:
    { request_id, model_version,
      prediction, probability,
      true_label (post-hoc), timestamp }
```

#### 7.3 Метрики и критерии

**Первичная метрика:** F1-score для класса «дефолт» (класс 1)

**Вторичные метрики:**
- **Precision** — критична, если стоимость ложно-положительного результата высока (неоправданный отказ хорошему клиенту ведёт к потере дохода)
- **Recall** — критичен, если стоимость пропущенного дефолта высока (финансовые потери от невыявленного риска)
- Снижение ожидаемых кредитных потерь (бизнес-метрика)

**Статистические параметры:**
- Уровень значимости: α = 0.05
- Мощность теста: 1 − β = 0.80
- Минимальный детектируемый эффект: ΔMDE = +0.02 по F1
- Расчётный размер выборки: ~2000 наблюдений на ветку (≥ 4000 всего)
- Длительность: 2 недели (минимум), 4 недели (рекомендуется)
- Метод: двусторонний z-тест для разности долей (или t-тест для непрерывных метрик)

#### 7.4 Реализация роутера в API

```python
import hashlib

def get_model_version(client_id: str) -> str:
    """Детерминированное распределение 50/50 по hash client_id."""
    h = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
    return 'v1' if h % 2 == 0 else 'v2'
```

Клиент всегда попадает в одну и ту же ветку (детерминизм), исключая эффект новизны.

#### 7.5 Критерии завершения A/B-теста

| Условие | Действие |
|---|---|
| p-value < 0.05 И ΔMDE достигнут | Выкатить v2 на 100 % трафика |
| p-value ≥ 0.05 после 4 недель | Оставить v1, закрыть эксперимент |
| Precision v2 < Precision v1 − 0.03 | Остановить тест, расследовать |

---

### Шаг 8 — Архитектурная документация

Файл: `ARCHITECTURE.md`

#### 8.1 Монолит vs Микросервисы

**Выбор архитектуры:** Монолитная (justified для данного MVP)

| Критерий | Монолит ✓ | Микросервисы |
|---|---|---|
| Команда | 1 разработчик | ≥ 3–5 команд |
| Нагрузка | < 100 RPS | > 1000 RPS |
| Сложность модели | 1–2 модели | > 10 моделей |
| Time-to-market | Быстро | Медленно |
| Operational overhead | Низкий | Высокий |

**Вывод:** для стадии MVP с одной задачей классификации монолит оптимален по соотношению сложность/ценность.

#### 8.2 Message Broker — RabbitMQ (концептуально)

При росте нагрузки или необходимости асинхронной обработки пакетов:

```
Client → REST API → RabbitMQ (queue: prediction_requests)
                         │
                    Worker (consumer)
                         │
                    Model inference
                         │
                    RabbitMQ (queue: prediction_results)
                         │
                    Client (polling / webhook)
```

Преимущества: отвязка API от модели, обработка пиков нагрузки, повтор при сбое.

#### 8.3 Логирование

```python
import logging
import json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# Структурированный лог (JSON для ELK/Loki)
logging.info(json.dumps({
    'event': 'prediction',
    'model_version': version,
    'prediction': prediction,
    'probability': probability,
    'latency_ms': elapsed
}))
```

В production: ELK Stack (Elasticsearch + Logstash + Kibana) или Grafana Loki.

#### 8.4 MLOps-инструменты

| Инструмент | Роль | Когда применять |
|---|---|---|
| **DVC** | Версионирование данных и моделей | Когда датасет > 1 ГБ или много экспериментов |
| **MLflow** | Трекинг экспериментов, model registry | При > 10 запусков обучения, командная работа |
| **Docker Hub** | Хранение образов | Уже используется в этом проекте |

---

## 4. Критерии оценки и чеклист

### 4.1 Полный чеклист сдачи (50 баллов)

#### Инженерная реализация (22 балла)

**API (8 баллов)**
- [ ] POST /predict принимает JSON с 23 признаками
- [ ] POST /predict возвращает prediction (int), probability (float), model_version (str)
- [ ] GET /health возвращает `{"status": "ok"}`
- [ ] Обработка ошибок (400 при неверном входе)
- [ ] Поддержка model_version v1 и v2

**Docker (7 баллов)**
- [ ] Dockerfile использует python:3.12-slim
- [ ] Образ успешно собирается (`docker build`)
- [ ] Контейнер запускается и отвечает на порту 5000
- [ ] Образ опубликован на Docker Hub (публичный)
- [ ] README содержит ссылку на Docker Hub образ

**Воспроизводимость (4 балла)**
- [ ] requirements.txt с точными версиями
- [ ] Инструкции по запуску локально (venv)
- [ ] Инструкции по запуску в Docker

**Модель (3 балла)**
- [ ] Модели сохранены pickle/joblib
- [ ] Загрузка модели из файла при старте сервиса
- [ ] Две версии модели (v1 и v2)

#### Архитектура и оркестрация (16 баллов)

**Обоснование архитектуры (8 баллов)**
- [ ] ARCHITECTURE.md с анализом монолит vs микросервисы
- [ ] Концептуальное описание RabbitMQ для батч-предсказаний
- [ ] Описание логирования (формат, инструменты)
- [ ] Обоснование выбора технологий

**MLOps (5 баллов)**
- [ ] Описание DVC (data versioning)
- [ ] Описание MLflow (experiment tracking)
- [ ] Бизнес-метрики (≥ 2 метрики за пределами F1)

**Docker Compose (3 балла — бонус)**
- [ ] docker-compose.yml с healthcheck
- [ ] Сервис запускается через `docker-compose up`

#### A/B-тестирование (10 баллов)

**Дизайн теста (4 балла)**
- [ ] Чёткая гипотеза H₀/H₁
- [ ] Метрики: первичная (F1) и вторичная (Precision/Recall)
- [ ] Бизнес-обоснование метрик

**Статистика и архитектура (3 балла)**
- [ ] Метод анализа: z-test или t-test с CI
- [ ] Размер выборки / длительность теста
- [ ] Роутер v1/v2 в API

**Демонстрация (3 балла)**
- [ ] curl-запросы к /predict?model_version=v1 и v2
- [ ] Разные ответы от разных версий
- [ ] Логи или скриншоты работы

#### Документация (2 балла)

- [ ] README.md: описание, структура, быстрый старт
- [ ] README.md: curl-примеры с реальным выводом
- [ ] Чистая структура репозитория по cookiecutter-data-science

---

## 5. Порядок разработки (рекомендуемая последовательность)

```
День 1
├── [x] Изучить датасет, EDA (notebooks/eda_and_training.ipynb)
├── [x] Создать структуру репозитория
└── [x] Обучить model_v1 и model_v2, сохранить .pkl

День 2
├── [x] Реализовать app/model_handler.py
├── [x] Реализовать app/api.py (/predict, /health)
└── [x] Написать tests/test_api.py, запустить pytest

День 3
├── [x] Написать Dockerfile, собрать образ
├── [x] Проверить контейнер локально (curl)
├── [x] Опубликовать образ на Docker Hub
└── [x] (Бонус) Написать docker-compose.yml

День 4
├── [x] Написать ab_test_plan.md
├── [x] Написать ARCHITECTURE.md
└── [x] Написать README.md (финальный)

День 5
└── [x] Финальная проверка по чеклисту, загрузка на GitHub
```

---

## 6. Типичные ошибки и как их избежать

| Ошибка | Решение |
|---|---|
| KeyError при /predict | Строго проверять наличие всех 23 признаков в FEATURE_ORDER |
| Модель не найдена в Docker | Убедиться, что `COPY models/ ./models/` есть в Dockerfile |
| Разный порядок признаков v1/v2 | Использовать единый FEATURE_ORDER в model_handler.py |
| Docker образ не запускается на порту 5000 | Проверить `EXPOSE 5000` и `host='0.0.0.0'` в Flask |
| A/B-роутер не детерминирован | Использовать hash(client_id) вместо random() |
| Утечка тестовых данных | Обучать scaler только на train-выборке, трансформировать test |

---

## 7. Итоговые артефакты для сдачи

| Артефакт | Расположение | Обязательность |
|---|---|---|
| `models/model_v1.pkl` | репозиторий или Docker образ | обязательно |
| `models/model_v2.pkl` | репозиторий или Docker образ | обязательно |
| `requirements.txt` | корень репозитория | обязательно |
| `docker/Dockerfile` | репозиторий | обязательно |
| `app/api.py` | репозиторий | обязательно |
| `ab_test_plan.md` | репозиторий | обязательно |
| `ARCHITECTURE.md` | репозиторий | обязательно |
| `README.md` | корень репозитория | обязательно |
| `docker-compose.yml` | корень репозитория | бонус (+3 балла) |
| Docker Hub ссылка | в README.md | обязательно |
| GitHub ссылка | сдача | обязательно |

---

*Концепция разработана на основе задания (task.txt, img1.png, img2.png, img3.png) и датасета UCI Default of Credit Card Clients.*
