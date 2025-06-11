# pipeline/logger.py
# Author: Bill Lu
# Description: Sets up a JSON-formatted logger with console and rotating file handlers.

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger
from datetime import datetime

# Define log directory and ensure it exists
def get_log_directory() -> str:
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

# Creates and returns a configured logger
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # JSON formatter for structured logs
    fmt = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        rename_fields={
            'asctime': 'timestamp',
            'levelname': 'level',
            'name': 'logger'
        }
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Rotating file handler (daily)
    log_dir = get_log_directory()
    filename = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = TimedRotatingFileHandler(
        filename, when='midnight', backupCount=7, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger