# logging_setup.py
import logging, os, re, glob
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

class DateNamedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    오늘은 logs/2025-09-10.log 로 기록하고,
    자정이 지나면 새 파일 logs/2025-09-11.log 로 '새로 열어서' 기록합니다.
    (기본 핸들러처럼 기존 파일을 rename 하지 않음)
    """
    def __init__(
        self,
        dirpath: str = "logs",
        when: str = "midnight",
        interval: int = 1,
        backupCount: int = 7,
        encoding: Optional[str] = "utf-8",
        utc: bool = False,
        filename_pattern: str = "%Y-%m-%d.log",
    ):
        os.makedirs(dirpath, exist_ok=True)
        self.dirpath = dirpath
        self.filename_pattern = filename_pattern
        initial_name = os.path.join(dirpath, datetime.now().strftime(filename_pattern))
        super().__init__(initial_name, when=when, interval=interval,
                         backupCount=backupCount, encoding=encoding, utc=utc)
        # suffix는 회전에 쓰이지만, 우리는 rename을 안 하므로 의미는 없음
        self.namer = None  # 기본 유지

    def doRollover(self):
        # 기존 스트림 닫기
        if self.stream:
            self.stream.close()
            self.stream = None

        # 새 날짜 파일로 교체
        new_name = os.path.join(self.dirpath, datetime.now().strftime(self.filename_pattern))
        self.baseFilename = os.path.abspath(new_name)
        self.mode = "a"
        self.stream = self._open()

        # 다음 롤오버 시각 계산 (부모 로직 사용)
        currentTime = int(self.rolloverAt)
        self.rolloverAt = self.computeRollover(currentTime)

        # 보관 정책 처리(backupCount일 기준으로 오래된 파일 삭제)
        if self.backupCount > 0:
            self._purge_old_files()

    def _purge_old_files(self):
        # 디렉터리 내 YYYY-MM-DD.log 패턴 파일 수집
        pattern = os.path.join(self.dirpath, "*.log")
        files = sorted(glob.glob(pattern))
        # 파일명에서 날짜 추출
        def to_date(f):
            m = re.search(r"(\d{4}-\d{2}-\d{2})\.log$", f)
            if not m: return None
            try: return datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except: return None
        dated = [(f, to_date(f)) for f in files]
        dated = [x for x in dated if x[1] is not None]
        # 날짜 내림차순 정렬 후 backupCount 개만 남기고 삭제
        dated.sort(key=lambda x: x[1], reverse=True)
        for f, _ in dated[self.backupCount:]:
            try: os.remove(f)
            except Exception: pass

def setup_logging():
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    formatter = logging.Formatter(fmt)

    app_handler = DateNamedTimedRotatingFileHandler(
        dirpath="logs", when="midnight", interval=1, backupCount=7,
        filename_pattern="%Y-%m-%d.log", encoding="utf-8"
    )
    app_handler.setFormatter(formatter)

    access_fmt = "%(asctime)s %(levelname)s [%(name)s] %(client_addr)s \"%(request_line)s\" %(status_code)s"
    access_handler = DateNamedTimedRotatingFileHandler(
        dirpath="logs", when="midnight", interval=1, backupCount=7,
        filename_pattern="%Y-%m-%d.access.log", encoding="utf-8"
    )
    access_handler.setFormatter(logging.Formatter(access_fmt))

    # uvicorn 계열 중복 방지 + 파일 핸들러 연결
    for name in ["uvicorn", "uvicorn.error"]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        if not any(isinstance(h, DateNamedTimedRotatingFileHandler) for h in lg.handlers):
            lg.addHandler(app_handler)
        lg.propagate = False

    al = logging.getLogger("uvicorn.access")
    al.setLevel(logging.INFO)
    if not any(isinstance(h, DateNamedTimedRotatingFileHandler) for h in al.handlers):
        al.addHandler(access_handler)
    al.propagate = False

    # 앱 전용 로거(원하면)
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)
    if not any(isinstance(h, DateNamedTimedRotatingFileHandler) for h in app_logger.handlers):
        app_logger.addHandler(app_handler)
    app_logger.propagate = False