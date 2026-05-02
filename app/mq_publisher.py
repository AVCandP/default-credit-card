"""RabbitMQ publisher — publishes prediction events to 'predictions' queue.

Designed to fail silently: if RabbitMQ is unavailable the API continues
to work normally, events are only logged locally.
"""
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672/')
QUEUE_NAME = 'predictions'

_lock = threading.Lock()


def _get_connection():
    import pika
    params = pika.URLParameters(RABBITMQ_URL)
    params.socket_timeout = 2
    return pika.BlockingConnection(params)


def publish_prediction(event: dict) -> None:
    """Publish a prediction event to RabbitMQ in a background thread.

    Falls back to a warning log if the broker is unreachable.
    """
    def _send():
        try:
            with _lock:
                conn = _get_connection()
                ch = conn.channel()
                ch.queue_declare(queue=QUEUE_NAME, durable=True)
                ch.basic_publish(
                    exchange='',
                    routing_key=QUEUE_NAME,
                    body=json.dumps(event).encode(),
                    properties=__import__('pika').BasicProperties(delivery_mode=2),
                )
                conn.close()
        except Exception as e:
            logger.warning(json.dumps({'event': 'mq_publish_failed', 'reason': str(e)}))

    threading.Thread(target=_send, daemon=True).start()
