# License: BSD 3 clause
"""
Handles reading and writing data from various types of data files.

:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import csv

from .featureset import FeatureSet
from .readers import (ARFFReader, CSVReader, LibSVMReader, MegaMReader,
                      NDJReader, TSVReader, safe_float, Reader)
from .writers import (ARFFWriter, Writer, TSVWriter, CSVWriter,
                      LibSVMWriter, MegaMWriter, NDJWriter)


# Register dialect for handling ARFF files
csv.register_dialect('arff', delimiter=',', quotechar="'",
                     escapechar='\\', doublequote=False,
                     lineterminator='\n', skipinitialspace=True)


__all__ = ['Reader', 'safe_float', 'FeatureSet', 'ARFFReader',
           'CSVReader', 'LibSVMReader', 'MegaMReader', 'NDJReader',
           'TSVReader', 'ARFFWriter', 'LibSVMWriter', 'TSVWriter',
           'CSVWriter', 'MegaMWriter', 'NDJWriter', 'Writer']
