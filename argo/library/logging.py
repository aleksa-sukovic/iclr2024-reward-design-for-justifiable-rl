import sys

from os.path import join
from typing import Optional
from loguru import logger


def configure_logger(log_dir: Optional[str] = None, log_epochs: bool = True, log_iters: bool = False, log_training_stage: bool = True) -> logger:
    logger.remove()

    log_format = "<green>[{time:DD.MM.YYYY at HH:mm:ss}]</green>"
    if log_training_stage:
        log_format = log_format + " <light-blue>{extra[mode]}</>"
    if log_epochs:
        log_format = log_format + " <b>EPOCH {extra[epoch]}</>"
    if log_iters:
        log_format = log_format + " <b>ITER {extra[iter]}</>"
    log_format = log_format + "  {message}"

    logger.add(sys.stdout, colorize=True, format=log_format, backtrace=True, diagnose=True)
    if log_dir:
        logger.add(join(log_dir, "out.log"), format=log_format, backtrace=True, diagnose=True)

    return logger
