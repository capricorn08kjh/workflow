import logging
from logging.handlers import TimedRotatingFileHandler
import os

os.makedirs("logs", exist_ok=True)

# 공통 포맷
fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
formatter = logging.Formatter(fmt)

# 매일 0시에 새로운 파일 생성, 7일치 보관
file_handler = TimedRotatingFileHandler(
    "logs/app.log", when="midnight", interval=1, backupCount=7, encoding="utf-8"
)
file_handler.setFormatter(formatter)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False

# error, access 로거도 같은 방식으로
for name in ["uvicorn.error", "uvicorn.access"]:
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.addHandler(file_handler)
    lg.propagate = False