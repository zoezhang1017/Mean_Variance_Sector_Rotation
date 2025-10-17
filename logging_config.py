import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_PATH = Path("logs/mean_variance.log")


def setup_logging(log_path: Path | str = DEFAULT_LOG_PATH,
                  level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Configure application wide logger with rotating file handler.

    Parameters
    ----------
    log_path : Path | str
        Destination path for log file.
    level : int
        Logging level, defaults to INFO.

    Returns
    -------
    logging.Logger
        Configured root logger for the application.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mean_variance")
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logging configured with level %s, file %s", level, log_path)
    return logger
