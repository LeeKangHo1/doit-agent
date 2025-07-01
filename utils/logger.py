import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    # 로그 폴더 생성
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    # 커스텀 포맷터 정의
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            original = super().format(record)
            return f"\n=================\n{original}"

    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # ⏰ 시간 기준으로 로그 파일 회전 설정 (매일 자정, 7일 보관)
        file_handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # 콘솔 출력 핸들러
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(CustomFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)

    # 예외 자동 로깅
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    return logger
