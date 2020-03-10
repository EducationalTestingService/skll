import re
import sys
import warnings

from os import unlink
from six import StringIO
from sklearn.metrics import roc_curve
from tempfile import NamedTemporaryFile

from skll import (close_and_remove_logger_handlers, get_skll_logger,
                  orig_showwarning)

TEMP_FILES = []
TEMP_FILE_PATHS = []
LOGGERS = []


def trigger_sklearn_warning():
    """
    Do something that will trigger an ``sklearn`` warning.

    This should trigger an ``UndefinedMetricWarning``.
    """

    roc_curve([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])


def tearDown():
    reset()


def reset():
    for i, temp_file in enumerate(TEMP_FILES):
        temp_file.close()
        del TEMP_FILES[i]
    for i, temp_file_path in enumerate(TEMP_FILE_PATHS):
        unlink(temp_file_path)
        del TEMP_FILE_PATHS[i]
    for i, logger in enumerate(LOGGERS):
        close_and_remove_logger_handlers(logger)
        del LOGGERS[i]
    warnings.showwarning = orig_showwarning


def test_get_skll_logger():
    reset()

    temp_file = NamedTemporaryFile("w", delete=False)
    temp_file.close()
    TEMP_FILES.append(temp_file)
    TEMP_FILE_PATHS.append(temp_file.name)
    logger = get_skll_logger("test_get_skll_logger", temp_file.name)
    LOGGERS.append(logger)

    # Send a regular log message
    msg1 = "message 1"
    logger.info(msg1)

    # Send a regular log message
    msg2 = "message 2"
    logger.info(msg2)

    with open(temp_file.name) as temp_file:
        log_lines = temp_file.readlines()
        assert log_lines[0].endswith("INFO - {}\n".format(msg1))
        assert log_lines[1].endswith("INFO - {}\n".format(msg2))

    close_and_remove_logger_handlers(logger)


def test_get_skll_logger_with_warning():
    reset()

    temp_file = NamedTemporaryFile("w", delete=False)
    temp_file.close()
    TEMP_FILES.append(temp_file)
    TEMP_FILE_PATHS.append(temp_file.name)
    logger = get_skll_logger("test_get_skll_logger_with_warning",
                             temp_file.name)
    LOGGERS.append(logger)

    # Send a regular log message
    msg1 = "message 1"
    logger.info(msg1)

    # Trigger an ``sklearn`` warning
    trigger_sklearn_warning()

    # Send a regular log message
    msg2 = "message 2"
    logger.info(msg2)

    with open(temp_file.name) as temp_file:
        log_lines = temp_file.readlines()
        assert log_lines[0].endswith("INFO - {}\n".format(msg1))
        sklearn_warning_re = \
            re.compile(r"WARNING - [^\n]+sklearn.metrics._ranking.py:\d+: "
                       r"UndefinedMetricWarning:No negative samples in "
                       r"y_true, false positive value should be "
                       r"meaningless")
        assert sklearn_warning_re.search("".join(log_lines[1]))
        assert log_lines[-1].endswith("INFO - {}\n".format(msg2))

    # Now make sure that warnings.showwarning works the way
    # it normally works (writes to STDERR) by issuing a warning,
    # capturing it, and, finally, making sure the expected
    # warning shows up correctly in the STDERR stream and,
    # additionally, not in the log file.
    old_stderr = sys.stderr
    try:
        msg3 = "message 3"
        sys.stderr = mystderr = StringIO()
        warnings.warn(msg3)
        err = mystderr.getvalue()
        assert "UserWarning: {}".format(msg3) in err
        with open(temp_file.name) as log_file:
            assert "UserWarning:{}".format(msg3) not in log_file.read()
    finally:
        sys.stderr = old_stderr

    close_and_remove_logger_handlers(logger)


def test_close_and_remove_logger_handlers():
    reset()

    temp_file = NamedTemporaryFile("w", delete=False)
    temp_file.close()
    TEMP_FILES.append(temp_file)
    TEMP_FILE_PATHS.append(temp_file.name)
    logger = get_skll_logger("test_close_and_remove_logger_handlers",
                             temp_file.name)
    LOGGERS.append(logger)
    close_and_remove_logger_handlers(logger)
    assert not logger.handlers
