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

orig_showwarning = warnings.showwarning
SKLEARN_WARNINGS_RE = re.compile(re.escape(f"{sep}sklearn{sep}"))


def send_sklearn_warnings_to_logger(logger, message, category, filename,
                                    lineno, file=None, line=None):
    """
    Return method that sends `sklearn`-specific warnings to a logger
    that can be used to replace warnings.showwarning (via `partial`,
    specifying a `logger` instance).
    """

    if SKLEARN_WARNINGS_RE.search(filename):
        logger.warning(f'{filename}:{lineno}: {category.__name__}:{message}')
    else:
        orig_showwarning(message, category, filename, lineno,
                         file=file, line=line)


def get_skll_logger(name, filepath=None, log_level=logging.INFO):
    """
    Create and return logger instances that are appropriate for use
    in SKLL code, e.g., that they can log to both STDERR as well
    as a file. This function will try to reuse any previously created
    logger based on the given name and filepath.

    Parameters
    ----------
    name : str
        The name to be used for the logger.
    filepath : str, optional
        The file to be used for the logger via a FileHandler.
        Default: None in which case no file is attached to the
        logger.
        Defaults to ``None``.
    log_level : str, optional
        The level for logging messages
        Defaults to ``logging.INFO``.

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
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - '
                                          '%(message)s')
            file_handler = logging.FileHandler(filepath, mode='w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

    warnings.showwarning = partial(send_sklearn_warnings_to_logger, logger)

    # return the logger instance
    return logger


def close_and_remove_logger_handlers(logger):
    """
    Close and remove any handlers attached to a logger instance.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance

    Returns
    -------
    NoneType
    """

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
