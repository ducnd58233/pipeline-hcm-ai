import logging.config
from datetime import datetime
import pytz


class TimezoneFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        timezone = pytz.timezone('Asia/Ho_Chi_Minh')
        conversion = datetime.fromtimestamp(record.created, timezone)
        if datefmt:
            return conversion.strftime(datefmt)
        return conversion.strftime("%Y-%m-%d %H:%M:%S %Z")


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": TimezoneFormatter,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "level": "INFO",
            "filename": "app.log",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG",
    },
}


def set_timezone(timezone_str: str):
    """Set the timezone for logging."""
    TimezoneFormatter.timezone = pytz.timezone(timezone_str)
    logging.info(f"Logging timezone set to: {TimezoneFormatter.timezone}")


logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

set_timezone('Asia/Ho_Chi_Minh')
