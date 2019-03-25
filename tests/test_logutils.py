import re
import sys
import unittest
import warnings
import numpy as np

from six import StringIO
from os import unlink
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tempfile import NamedTemporaryFile

from skll import (close_and_remove_logger_handlers, get_skll_logger,
                  orig_showwarning)


def trigger_sklearn_warnings():
    """
    Do something that will trigger ``sklearn`` warnings.

    This should trigger a couple ``FutureWarning``s about
    ``n_estimators``'s default value and specifying ``cv``.
    """

    (GridSearchCV(RandomForestClassifier(), {"max_depth": [None, 1, 2]})
     .fit(np.array([1, 1, 1, 2, 2, 2]).reshape(6, 1), [2, 2, 2, 3, 3, 3]))


class TestLogUtils(unittest.TestCase):

    def setUp(self):
        self.TEMP_FILES = []
        self.TEMP_FILE_PATHS = []
        self.LOGGERS = []

    def tearDown(self):
        self.reset()

    def reset(self):
        for i, temp_file in enumerate(self.TEMP_FILES):
            temp_file.close()
            del self.TEMP_FILES[i]
        for i, temp_file_path in enumerate(self.TEMP_FILE_PATHS):
            unlink(temp_file_path)
            del self.TEMP_FILE_PATHS[i]
        for i, logger in enumerate(self.LOGGERS):
            close_and_remove_logger_handlers(logger)
            del self.LOGGERS[i]
        warnings.showwarning = orig_showwarning


    def testGetSKLLLogger(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger("testGetSKLLLogger", temp_file.name)
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
        logger = get_skll_logger("testGetSKLLLoggerWithWarning", temp_file.name)
        self.LOGGERS.append(logger)

        # Send a regular log message
        msg1 = "message 1"
        logger.info(msg1)

        # Trigger a couple ``sklearn`` warnings
        trigger_sklearn_warnings()

        # Send a regular log message
        msg2 = "message 2"
        logger.info(msg2)

        with open(temp_file.name) as temp_file:
            log_lines = temp_file.readlines()
            assert log_lines[0].endswith("INFO - {}\n".format(msg1))
            cv_sklearn_warning_re = \
                re.compile(r"WARNING - [^\n]+site-packages/sklearn/model_selection/_split\.py"
                           r":2053: FutureWarning:You should specify a value for 'cv' instead"
                           r" of relying on the default value\. The default value will change"
                           r" from 3 to 5 in version 0\.22\.")
            assert cv_sklearn_warning_re.search("".join(log_lines[1:-1]))
            n_estimators_sklearn_warning_re = \
                re.compile(r"WARNING - [^\n]+site-packages/sklearn/ensemble/forest\.py:246: "
                           r"FutureWarning:The default value of n_estimators will change "
                           r"from 10 in version 0\.20 to 100 in 0\.22\.")
            assert n_estimators_sklearn_warning_re.search("".join(log_lines[1:-1]))
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


    def testCloseAndRemoveLoggerHandlers(self):
        self.reset()

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.close()
        self.TEMP_FILES.append(temp_file)
        self.TEMP_FILE_PATHS.append(temp_file.name)
        logger = get_skll_logger("testCloseAndRemoveLoggerHandlers", temp_file.name)
        self.LOGGERS.append(logger)
        close_and_remove_logger_handlers(logger)
        assert not logger.handlers
