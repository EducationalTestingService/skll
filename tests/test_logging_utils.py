"""
Tests for SKLL logging utilities.

:author: Nitin Madnani (nmadnani@ets.org)
"""

import re
import sys
import unittest
import warnings
from tempfile import NamedTemporaryFile

from six import StringIO
from sklearn.metrics import roc_curve

from skll.utils.logging import (
    close_and_remove_logger_handlers,
    get_skll_logger,
    orig_showwarning,
)
from tests.utils import unlink

TEMP_FILES = []
TEMP_FILE_PATHS = []
LOGGERS = []


class TestLoggingUtils(unittest.TestCase):
    """Test class for logging utility tests."""

    def reset(self):
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

    def setUp(self):
        self.reset()

    def tearDown(self):
        self.reset()

    def trigger_sklearn_warning(self):
        """
        Do something that will trigger an ``sklearn`` warning.

        This should trigger an ``UndefinedMetricWarning``.
        """
        roc_curve([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def test_get_skll_logger(self):
        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        TEMP_FILES.append(temp_file)
        TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger("test_get_skll_logger", filepath=temp_file.name)
        LOGGERS.append(logger)

        # Send a regular log message
        msg1 = "message 1"
        logger.info(msg1)

        # Send a regular log message
        msg2 = "message 2"
        logger.info(msg2)

        with open(temp_file.name) as tempfh:
            log_lines = tempfh.readlines()
            assert log_lines[0].endswith(f"INFO - {msg1}\n")
            assert log_lines[1].endswith(f"INFO - {msg2}\n")

        close_and_remove_logger_handlers(logger)

    def test_get_skll_logger_with_warning(self):
        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        TEMP_FILES.append(temp_file)
        TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger("test_get_skll_logger_with_warning", filepath=temp_file.name)
        LOGGERS.append(logger)

        # Send a regular log message
        msg1 = "message 1"
        logger.info(msg1)

        # Trigger an ``sklearn`` warning
        self.trigger_sklearn_warning()

        # Send a regular log message
        msg2 = "message 2"
        logger.info(msg2)

        with open(temp_file.name) as temp_file:
            log_lines = temp_file.readlines()
            assert log_lines[0].endswith(f"INFO - {msg1}\n")
            sklearn_warning_re = re.compile(
                r"WARNING - [^\n]+sklearn.metrics._ranking.py:\d+: "
                r"UndefinedMetricWarning:No negative samples in y_true, false "
                r"positive value should be meaningless"
            )
            assert sklearn_warning_re.search("".join(log_lines[1]))
            assert log_lines[-1].endswith(f"INFO - {msg2}\n")

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
            assert f"UserWarning: {msg3}" in err
            with open(temp_file.name) as log_file:
                assert f"UserWarning:{msg3}" not in log_file.read()
        finally:
            sys.stderr = old_stderr

        close_and_remove_logger_handlers(logger)

    def test_close_and_remove_logger_handlers(self):
        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        TEMP_FILES.append(temp_file)
        TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger("test_close_and_remove_logger_handlers", temp_file.name)
        LOGGERS.append(logger)
        close_and_remove_logger_handlers(logger)
        assert not logger.handlers
