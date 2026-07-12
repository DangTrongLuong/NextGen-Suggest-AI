import pika
import json
import threading
import time
import logging
from typing import Dict, Any

from app.config.settings import settings
from app.features.indexing.services.indexing_service import indexing_service

logger = logging.getLogger(__name__)

class RabbitMQConsumer:
    def __init__(self):
        self.host = settings.RABBITMQ_HOST
        self.port = settings.RABBITMQ_PORT
        self.username = settings.RABBITMQ_USERNAME
        self.password = settings.RABBITMQ_PASSWORD
        self.queue = settings.RABBITMQ_QUEUE
        self.exchange = settings.RABBITMQ_EXCHANGE
        self.routing_key = settings.RABBITMQ_ROUTING_KEY
        
        self.connection = None
        self.channel = None
        self.is_running = False
        self._thread = None

    def connect(self):
        """Establish connection to RabbitMQ with retry mechanism."""
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=credentials,
            connection_attempts=5,
            retry_delay=5,
            heartbeat=60
        )
        
        logger.info(f"Connecting to RabbitMQ on {self.host}:{self.port}...")
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Declare exchange and queue to ensure they exist
        self.channel.exchange_declare(
            exchange=self.exchange,
            exchange_type='direct',
            durable=True
        )
        self.channel.queue_declare(
            queue=self.queue,
            durable=True
        )
        self.channel.queue_bind(
            queue=self.queue,
            exchange=self.exchange,
            routing_key=self.routing_key
        )
        
        # Set prefetch count to 1 to distribute load evenly
        self.channel.basic_qos(prefetch_count=1)
        logger.info(f"Connected to RabbitMQ. Listening on queue '{self.queue}'")

    def _process_message(self, ch, method, properties, body):
        """Callback to process a message."""
        logger.info("Received indexing message from RabbitMQ.")
        try:
            payload = json.loads(body.decode('utf-8'))
            action = payload.get("action")
            product_uuids = payload.get("product_uuids")
            
            if not product_uuids:
                logger.warning("Empty product_uuids list received in RabbitMQ message.")
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return
                
            logger.info(f"Processing action '{action}' for {len(product_uuids)} UUIDs: {product_uuids}")
            
            if action == "sync":
                indexing_service.sync_uuids_to_qdrant(product_uuids)
            elif action == "delete":
                indexing_service.delete_uuids_from_qdrant(product_uuids)
            else:
                logger.warning(f"Unknown action '{action}' received.")
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info("Successfully processed and acknowledged message.")
            
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON body from RabbitMQ message: {je}")
            # Reject invalid JSON formats (no requeue)
            ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing RabbitMQ indexing message: {e}", exc_info=True)
            # Requeue the message for retry if processing failed due to temporary issues (e.g. network/db offline)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def _run(self):
        """Main loop running inside the background thread."""
        while self.is_running:
            try:
                self.connect()
                self.channel.basic_consume(
                    queue=self.queue,
                    on_message_callback=self._process_message
                )
                self.channel.start_consuming()
            except pika.exceptions.ConnectionClosedByBroker:
                logger.warning("RabbitMQ connection closed by broker. Reconnecting...")
            except pika.exceptions.AMQPChannelError as ce:
                logger.error(f"RabbitMQ channel error: {ce}. Reconnecting...")
            except pika.exceptions.AMQPConnectionError as conne:
                logger.error(f"RabbitMQ connection error: {conne}. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Unexpected RabbitMQ consumer error: {e}. Retrying in 10s...", exc_info=True)
                time.sleep(10)
            finally:
                self.cleanup()

    def start(self):
        """Start the background consumer thread."""
        if self.is_running:
            logger.warning("RabbitMQ consumer is already running.")
            return
            
        self.is_running = True
        self._thread = threading.Thread(target=self._run, name="rabbitmq-consumer-thread", daemon=True)
        self._thread.start()
        logger.info("RabbitMQ consumer background thread started.")

    def stop(self):
        """Stop the background consumer thread."""
        logger.info("Stopping RabbitMQ consumer thread...")
        self.is_running = False
        
        # Stop consuming and close connection if open
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
        except Exception as e:
            logger.warning(f"Error while stopping consumption: {e}")
            
        self.cleanup()
        
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("RabbitMQ consumer thread stopped cleanly.")

    def cleanup(self):
        """Safely close RabbitMQ channel and connection."""
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
        except:
            pass
        try:
            if self.connection and self.connection.is_open:
                self.connection.close()
        except:
            pass
        self.channel = None
        self.connection = None

# Singleton consumer instance
rabbitmq_consumer = RabbitMQConsumer()
