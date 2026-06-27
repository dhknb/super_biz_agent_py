"""Redis/RQ queue wiring."""

from redis import Redis
from rq import Queue

from app.config import config


redis_conn = Redis.from_url(config.redis_url)
index_queue = Queue(config.rq_queue_name, connection=redis_conn)
protocol_pdf_queue = Queue("protocol_pdf_ingest", connection=redis_conn)
