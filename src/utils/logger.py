"""
Logging utilities for DHG-LGB framework.

Provides standardized logging across all modules with file and console output.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'DHG-LGB',
    log_dir: Optional[str] = None,
    log_level: str = 'INFO',
    console: bool = True,
    log_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Parameters
    ----------
    name : str, default='DHG-LGB'
        Logger name
    log_dir : str, optional
        Directory to save log files. If None, logs are only printed to console.
    log_level : str, default='INFO'
        Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    console : bool, default=True
        Whether to output logs to console
    log_file : bool, default=True
        Whether to save logs to file

    Returns
    -------
    logging.Logger
        Configured logger instance

    Examples
    --------
    >>> logger = setup_logger('DHG-LGB', log_dir='results/logs')
    >>> logger.info('Training started')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers to avoid duplication
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(log_dir, f'{name}_{timestamp}.log')

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f'Log file created: {log_file_path}')

    return logger


def get_logger(name: str = 'DHG-LGB') -> logging.Logger:
    """
    Get an existing logger instance.

    Parameters
    ----------
    name : str, default='DHG-LGB'
        Logger name

    Returns
    -------
    logging.Logger
        Logger instance
    """
    return logging.getLogger(name)
