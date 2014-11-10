# License: BSD 3 clause
"""
Handles reading and writing data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

from __future__ import absolute_import, print_function, unicode_literals

import csv

from six import PY2

from .featureset import FeatureSet
from .readers import (ARFFReader, CSVReader, LibSVMReader, MegaMReader,
                      NDJReader, TSVReader, safe_float, Reader)
from .writers import (ARFFWriter, DelimitedFileWriter, Writer,
                      LibSVMWriter, MegaMWriter, NDJWriter)


# Register dialect for handling ARFF files
if PY2:
    csv.register_dialect('arff', delimiter=b',', quotechar=b"'",
                         escapechar=b'\\', doublequote=False,
                         lineterminator=b'\n', skipinitialspace=True)
else:
    csv.register_dialect('arff', delimiter=',', quotechar="'",
                         escapechar='\\', doublequote=False,
                         lineterminator='\n', skipinitialspace=True)


__all__ = ['Reader', 'safe_float', 'FeatureSet', 'ARFFReader',
           'CSVReader', 'LibSVMReader', 'MegaMReader', 'NDJReader',
           'TSVReader', 'ARFFWriter', 'DelimitedFileWriter', 'LibSVMWriter',
           'MegaMWriter', 'NDJWriter', 'Writer']
