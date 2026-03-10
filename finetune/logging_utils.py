"""训练模块日志工具。"""
import logging


APP_LOGGER_NAME = "finetune"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def normalize_log_level(log_level: str) -> str:
    normalized = str(log_level).strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError(f"不支持的日志级别: {log_level}")
    return normalized


def configure_logging(log_level: str = "INFO"):
    normalized = normalize_log_level(log_level)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger(APP_LOGGER_NAME).setLevel(getattr(logging, normalized))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")
