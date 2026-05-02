# Architecture — Credit Card Default Prediction Service

## 1. Выбор архитектуры: Монолит

### Обоснование

Для данного MVP выбрана **монолитная архитектура** со следующим обоснованием:

| Критерий | Монолит (выбрано) | Микросервисы |
|---|---|---|
| Размер команды | 1 разработчик | ≥ 3–5 команд |
| Ожидаемая нагрузка | < 100 RPS | > 1 000 RPS |
| Количество моделей | 2 (v1, v2) | > 10 |
| Time-to-market | Быстро | Медленно (CI/CD per service) |
| Operational overhead | Низкий | Высокий (service mesh, трейсинг) |
| Сложность отладки | Низкая | Высокая |

**Вывод:** микросервисная архитектура избыточна для текущего масштаба. При росте до > 500 RPS или при появлении более 5 независимых моделей целесообразно выделить `model-serving-service` и `feature-engineering-service` отдельно.

---

## 2. Компонентная схема

```
┌─────────────────────────────────────────────┐
│               Docker Container               │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │           Flask Application          │   │
│  │                                      │   │
│  │  GET  /health                        │   │
│  │  POST /predict                       │   │
│  │         │                            │   │
│  │         ▼                            │   │
│  │  ┌─────────────┐  A/B Router         │   │
│  │  │model_handler│  hash(client_id)%2  │   │
│  │  └──────┬──────┘                     │   │
│  │         │                            │   │
│  │    ┌────┴────┐                       │   │
│  │    ▼         ▼                       │   │
│  │ model_v1  model_v2                   │   │
│  │ (LR .pkl) (RF .pkl)                  │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Port: 5000                                 │
└─────────────────────────────────────────────┘
         │
         │ HTTP JSON
         ▼
    External Client
    (curl / frontend / batch job)
```

---

## 3. Message Broker — RabbitMQ (концептуально)

При росте нагрузки или при переходе к батч-предсказаниям (ночной скоринг портфеля) рекомендуется асинхронная архитектура:

```
Client
  │  POST /predict-async
  ▼
REST API  ──►  RabbitMQ  ──►  ML Worker (consumer)
              queue:               │
              predictions          │ joblib.load + predict
                                   │
                              RabbitMQ
                              queue: results
                                   │
                              Client (polling GET /result/{id})
                              или Webhook callback
```

**Преимущества:**
- Отвязка API от модели (разные масштабы)
- Обработка пиков нагрузки без потери запросов
- Автоматический retry при сбое воркера
- Возможность параллельного запуска нескольких воркеров

---

## 4. Логирование

Используется структурированное JSON-логирование (совместимо с ELK Stack / Grafana Loki):

```python
logging.info(json.dumps({
    'event': 'prediction',
    'model_version': version,
    'prediction': prediction,
    'probability': probability,
    'latency_ms': elapsed_ms,
}))
```

**В production** рекомендуется:
- **ELK Stack** (Elasticsearch + Logstash + Kibana) — индексация и визуализация логов
- **Grafana Loki** — более лёгкая альтернатива для контейнерных сред
- **Алерты** на latency_ms > 500 и на долю `prediction=1` (дрейф модели)

---

## 5. MLOps-инструменты

### DVC (Data Version Control)

Используется для версионирования данных и артефактов модели:

```bash
dvc init
dvc add data/UCI_Credit_Card.csv   # версионировать датасет
dvc add models/model_v1.pkl        # версионировать модели
git add data/.gitignore models/.gitignore *.dvc
git commit -m "track data and models with DVC"
dvc push                           # в S3 / GCS / Azure Blob
```

**Когда применять:** при > 1 ГБ данных или при необходимости воспроизвести точный эксперимент из прошлого.

### MLflow

Используется для трекинга экспериментов и model registry:

```python
import mlflow
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("f1_default", f1)
    mlflow.sklearn.log_model(pipeline, "model")
```

**Когда применять:** при командной работе, > 10 запусков обучения, сравнении гиперпараметров.

---

## 6. Бизнес-метрики

| Метрика | Формула | Интерпретация |
|---|---|---|
| Снижение ожидаемых кредитных потерь | `(FN_old − FN_new) × avg_exposure` | Сколько NT$ дополнительно «спасает» новая модель |
| Уровень одобрения при фиксированном риске | `approved / total` при threshold ≤ 5 % | Рост = модель точнее разграничивает риски без увеличения дефолтов |

Обе метрики рассчитываются после пост-анализа A/B-теста с разметкой `true_label`.
