# Credit Card Default Prediction Service

Flask-сервис для предсказания дефолта по кредитной карте на основе датасета UCI Default of Credit Card Clients.

**Модели:**
- `v1` — LogisticRegression (class_weight='balanced') | F1=0.46, Recall=0.62
- `v2` — RandomForestClassifier (class_weight='balanced') | F1=0.45, Precision=0.64

---

## Структура проекта

```
default_credit_card/
├── app/
│   ├── __init__.py
│   ├── api.py              # Flask: /predict, /health
│   └── model_handler.py    # загрузка моделей, препроцессинг
├── models/
│   ├── train_model.py      # скрипт обучения
│   ├── model_v1.pkl        # LogisticRegression
│   └── model_v2.pkl        # RandomForest
├── tests/
│   └── test_api.py         # 6 pytest-тестов
├── data/
│   ├── UCI_Credit_Card.csv
│   └── about_dataset.txt
├── conception/
│   └── concept_defolt_cards.md
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── ab_test_plan.md
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

---

## Быстрый старт (локально)

```bash
# 1. Создать и активировать venv
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Обучить модели (если .pkl отсутствуют)
python models/train_model.py

# 4. Запустить сервис
set PYTHONPATH=.               # Windows
# export PYTHONPATH=.          # Linux/macOS
python -m app.api
```

Сервис доступен на `http://localhost:5000`.

---

## Запуск через Docker

```bash
# Сборка образа
docker build -f docker/Dockerfile -t default-credit-card:latest .

# Запуск контейнера
docker run -d -p 5000:5000 --name credit-api default-credit-card:latest

# Проверка
docker logs credit-api
```

### Docker Compose

```bash
docker-compose up --build
```

---

## API

### GET /health

```bash
curl http://localhost:5000/health
```

```json
{"models_loaded": ["v1", "v2"], "status": "ok"}
```

### POST /predict

**Тело запроса** — JSON с 23 признаками + опциональный `model_version`:

| Поле | Тип | Описание |
|---|---|---|
| `LIMIT_BAL` | float | Кредитный лимит (NT$) |
| `SEX` | int | 1=М, 2=Ж |
| `EDUCATION` | int | 1=аспирантура, 2=универ, 3=школа, 4=другое |
| `MARRIAGE` | int | 1=женат/замужем, 2=холост/не замужем, 3=другое |
| `AGE` | int | Возраст |
| `PAY_0`..`PAY_6` | int | Статус платежа (−2/−1=вовремя, 1–9=просрочка в месяцах) |
| `BILL_AMT1`..`BILL_AMT6` | float | Выписка по счёту (NT$) |
| `PAY_AMT1`..`PAY_AMT6` | float | Сумма предыдущего платежа (NT$) |
| `model_version` | str | `"v1"` \| `"v2"` \| `"ab"` (default: `"v1"`) |
| `client_id` | str | Используется для A/B-маршрутизации (при `model_version="ab"`) |

**Пример запроса:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
    "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
    "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
    "model_version": "v1"
  }'
```

**Ответ:**

```json
{"model_version": "v1", "prediction": 1, "probability": 0.7777}
```

| Поле | Описание |
|---|---|
| `prediction` | 0 = нет дефолта, 1 = дефолт |
| `probability` | Вероятность дефолта [0.0, 1.0] |
| `model_version` | Версия, выдавшая результат |

### A/B-маршрутизация

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{ ...features..., "model_version": "ab", "client_id": "user-42" }'
```

Один и тот же `client_id` всегда попадает в одну ветку (детерминированный hash).

---

## Тесты

```bash
set PYTHONPATH=.   # Windows
python -m pytest tests/ -v
```

Покрытие: `/health`, `/predict` (v1, v2, ab), обработка ошибок.

---

## Docker Hub

```bash
docker pull avcyber/default-credit-card:latest
docker run -d -p 5000:5000 avcyber/default-credit-card:latest
```

Образ: https://hub.docker.com/r/avcyber/default-credit-card

---

## Документация

- [ARCHITECTURE.md](ARCHITECTURE.md) — архитектурные решения, MLOps, логирование
- [ab_test_plan.md](ab_test_plan.md) — план A/B-тестирования
- [conception/concept_defolt_cards.md](conception/concept_defolt_cards.md) — полная концепция проекта
