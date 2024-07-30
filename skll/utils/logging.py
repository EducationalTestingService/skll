# License: BSD 3 clause
"""
Functions related to logging in SKLL.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""
import logging
import re
import warnings
from functools import partial
from os.path import sep
from typing import Optional

orig_showwarning = warnings.showwarning
SKLEARN_WARNINGS_RE = re.compile(re.escape(f"{sep}sklearn{sep}"))


class MatplotlibCategoryFilter(logging.Filter):
    """
    Class to filter out specific log records from `matplotlib.category`.

    This is useful when generating learning curves which generates unnecessary
    log records from `matplotlib.category`. For more details, see this issue:
    https://github.com/matplotlib/matplotlib/issues/23422
    """

    def filter(self, record):
        """
        Implement the filter method.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be filtered.

        """
        # Check if the log record is from matplotlib.category and contains the specific message
        if (
            record.name == "matplotlib.category"
            and "Using categorical units to plot a list of strings" in record.msg
        ):
            # filter out this record
            return False

        # allow other records through
        return True


def send_sklearn_warnings_to_logger(
    logger, message, category, filename, lineno, file=None, line=None
):
    """
    Return method that sends `sklearn`-specific warnings to a logger.

    This method that can be used to replace warnings.showwarning (via `partial`,
    specifying a `logger` instance).
    """
    if SKLEARN_WARNINGS_RE.search(filename):
        logger.warning(f"{filename}:{lineno}: {category.__name__}:{message}")
    else:
        orig_showwarning(message, category, filename, lineno, file=file, line=line)


def get_skll_logger(
    name: str, filepath: Optional[str] = None, log_level: int = logging.INFO
) -> logging.Logger:
    """
    Create and return logger instances appropriate for use in SKLL code.

    These logger instances can log to both STDERR as well as a file. This
    function will try to reuse any previously created logger based on the
    given name and filepath.

    Parameters
    ----------
    name : str
        The name to be used for the logger.
    filepath : Optional[str], default=None
        The file to be used for the logger via a FileHandler.
        Default: None in which case no file is attached to the
        logger.
    log_level : int, default=logging.INFO
        The level for logging messages

    Returns
    -------
    logger: logging.Logger
        A ``Logger`` instance.

    """
    # first get the logger instance associated with the
    # given name if one already exists
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # if we are given a file path and this existing logger doesn't already
    # have a file handler for this file, then add one.
    if filepath:

        def is_file_handler(handler):
            return isinstance(handler, logging.FileHandler) and handler.stream.name == filepath

        need_file_handler = not any([is_file_handler(handler) for handler in logger.handlers])
        if need_file_handler:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - " "%(message)s")
            file_handler = logging.FileHandler(filepath, mode="w")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    warnings.showwarning = partial(send_sklearn_warnings_to_logger, logger)

    # return the logger instance
    return logger


def close_and_remove_logger_handlers(logger: logging.Logger) -> None:
    """
    Close and remove any handlers attached to a logger instance.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance

    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
