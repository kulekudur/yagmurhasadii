"""
Logging configuration module.
Implements structured logging with JSON format option.
"""

import logging
import sys
from pythonjsonlogger import jsonlogger
from backend.config.settings import settings


class RequestIDFilter(logging.Filter):
    """Add request ID to log records if available"""
    
    def filter(self, record):
        record.request_id = getattr(record, 'request_id', 'N/A')
        return True


def setup_logging(
    log_level: str = settings.LOG_LEVEL,
    log_format: str = settings.LOG_FORMAT
) -> logging.Logger:
    """
    Configure centralized logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'text')
    
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger("rainwater")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.addFilter(RequestIDFilter())
    
    # Create formatter
    if log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional: File handler
    if settings.LOG_FILE:
        file_handler = logging.FileHandler(settings.LOG_FILE)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestIDFilter())
        logger.addHandler(file_handler)
    
    return logger


# Create logger instance
logger = setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance"""
    return logging.getLogger(name)
