"""RabbitMQ consumer worker — reads prediction events from 'predictions' queue.

Run standalone:
    python -m app.worker

In Docker Compose it runs as a separate service alongside the API.
"""
import json
import logging
import os
import signal
import sys

import pika

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [worker] %(message)s'
)
logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
QUEUE_NAME = 'predictions'


def process_event(ch, method, properties, body):
    try:
        event = json.loads(body)
        logger.info(json.dumps({'event': 'consumed', **event}))
        # Место для расширения: запись в БД, мониторинг дрейфа, алерты
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error(json.dumps({'event': 'process_error', 'reason': str(e)}))
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    logger.info(f"Connecting to RabbitMQ: {RABBITMQ_URL}")
    params = pika.URLParameters(RABBITMQ_URL)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE_NAME, durable=True)
    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue=QUEUE_NAME, on_message_callback=process_event)

    def _shutdown(sig, frame):
        logger.info("Shutting down worker...")
        conn.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"Waiting for messages on queue '{QUEUE_NAME}'...")
    ch.start_consuming()


if __name__ == '__main__':
    main()
