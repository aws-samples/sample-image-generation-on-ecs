import sys

from loguru import logger
from mangum import Mangum

logger.remove()
logger.add(sys.stdout, serialize=True, level="INFO")

from server import app

_mangum = Mangum(app, lifespan="off")


def handler(event, context):
    if event.get("warmup"):
        return {"statusCode": 200, "body": "warm"}
    return _mangum(event, context)
