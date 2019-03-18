import re
import sys
import unittest
import warnings

from six import StringIO
from joblib import Parallel, delayed
from os import unlink
from tempfile import NamedTemporaryFile

from skll import (close_and_remove_logger_handlers,
                  get_skll_logger,
                  orig_showwarning,
                  warn_once)


def issue_warning(msg):
    warnings.warn(msg)


@warn_once
def issue_warning_once(msg):
    warnings.warn(msg)


class TestLogUtils(unittest.TestCase):

    def setUp(self):
        self.TEMP_FILES = []
        self.TEMP_FILE_PATHS = []
        self.LOGGERS = []


    def tearDown(self):
        for temp_file in self.TEMP_FILES:
            temp_file.close()
        for temp_file_path in self.TEMP_FILE_PATHS:
            unlink(temp_file_path)
        for logger in self.LOGGERS:
            close_and_remove_logger_handlers(logger)
        self.reset()


    def reset(self):
        warnings.showwarning = orig_showwarning


    def testGetSKLLLogger(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger(__name__, temp_file.name)
        self.LOGGERS.append(logger)

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


    def testGetSKLLLoggerWithWarning(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger(__name__, temp_file.name)
        self.LOGGERS.append(logger)

        # Send a regular log message
        msg1 = "message 1"
        logger.info(msg1)

        # Issue a UserWarning, which should redirect to the log
        msg2 = "message from issued warning"
        issue_warning(msg2)

        # Send a regular log message
        msg3 = "message 3"
        logger.info(msg3)

        with open(temp_file.name) as temp_file:
            log_lines = temp_file.readlines()
            assert log_lines[0].endswith("INFO - {}\n".format(msg1))
            user_warning_re = \
                re.compile(r"WARNING - [^\n]+tests/test_logutils\.py:\d+: "
                           r"UserWarning:{}$".format(msg2))
            assert user_warning_re.search(log_lines[1])
            assert log_lines[2].endswith("INFO - {}\n".format(msg3))


    def testGetSKLLLoggerWithWarningWarnOnce(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger(__name__, temp_file.name)
        self.LOGGERS.append(logger)

        # Make a method that issues multiple of the same type of
        # warning, decorate it with warn_once, and then call it and
        # make sure only one of the issued warnings is in the log
        # afterwards
        msg = "message from issued warning"
        @warn_once
        def issue_warnings():
            issue_warning(msg)
            issue_warning(msg)
            issue_warning(msg)
        issue_warnings()
        with open(temp_file.name) as temp_file:
            log_content = temp_file.read()
            user_warning_re = \
                re.compile(r"WARNING - [^\n]+tests/test_logutils\.py:\d+: "
                           r"UserWarning:{}".format(msg))
            self.assertEqual(len(user_warning_re.findall(log_content)), 1)


    def testGetSKLLLoggerWithWarningParallel(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger(__name__, temp_file.name)
        self.LOGGERS.append(logger)

        # Send a regular log message
        msg1 = "message 1"
        logger.info(msg1)

        # Issue UserWarnings in parallel, all of which should
        # redirect to the log
        num_warnings = 3
        msg2 = "using delayed: message from issued warning"
        Parallel()(delayed(issue_warning)("{}: {}".format(msg2, i))
                   for i in range(num_warnings))

        # Send a regular log message
        msg3 = "message 3"
        logger.info(msg3)

        with open(temp_file.name) as temp_file:
            log_lines = temp_file.readlines()
            assert log_lines[0].endswith("INFO - {}\n".format(msg1))
            user_warning_re = \
                re.compile(r"WARNING - [^\n]+tests/test_logutils\.py:\d+: "
                           r"UserWarning:{}: [0-2]".format(msg2))
            self.assertEqual(len(user_warning_re.findall("".join(log_lines[1:4]))),
                             num_warnings)
            assert log_lines[4].endswith("INFO - {}\n".format(msg3))


    def testCloseAndRemoveLoggerHandlers(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger(__name__, temp_file.name)
        self.LOGGERS.append(logger)
        close_and_remove_logger_handlers(logger)
        assert not logger.handlers


    def testShowWarnings(self):
        self.reset()

        # Make a SKLL logger that logs messages to a file, issue a
        # warning, and make sure it gets logged in the log file.
        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger(__name__, temp_file.name)
        self.LOGGERS.append(logger)
        msg1 = "message 1"
        issue_warning(msg1)
        close_and_remove_logger_handlers(logger)
        with open(temp_file.name) as log_file:
            assert "UserWarning:{}".format(msg1) in log_file.read()

        # Now reset original warnings.showwarning, issue a
        # UserWarning, which should work the way
        # warnings.showwarning normally works (writes to STDERR),
        # capture it, and, finally, make sure the expected
        # warning shows up correctly in the STDERR stream and,
        # additionally, not in the log file.
        warnings.showwarning = orig_showwarning
        old_stderr = sys.stderr
        msg2 = "message 2"
        try:
            sys.stderr = mystderr = StringIO()
            issue_warning(msg2)
            err = mystderr.getvalue()
            assert "UserWarning: {}".format(msg2) in err
            with open(temp_file.name) as log_file:
                assert "UserWarning:{}".format(msg2) not in log_file.read()
        finally:
            sys.stderr = old_stderr
