FROM python:3.12-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода и моделей
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "-m", "app.api"]
