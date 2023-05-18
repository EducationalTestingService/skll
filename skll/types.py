# License: BSD 3 clause
"""
Custom type annotations for readability.

:author: Nitin Madnani (nmadnani@ets.org)
"""
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from scipy.sparse import csr_matrix

# a class map that maps new labels (string)
# to list of old labels (list of string)
ClassMap = Dict[str, List[str]]

# list of feature dictionaries
FeatureDict = Dict[str, Any]
FeatureDictList = List[FeatureDict]

# a mapping from example ID to fold ID;
# the example ID may be a float or a str
FoldMapping = Dict[Union[float, str], str]

# a float or a string; this is useful
# for SKLL IDs that can be both
IdType = Union[float, str]

# a float, int, or a string; this is useful
# for SKLL labels that can be both
LabelType = Union[float, int, str]

# a generator that yields a three-tuple:
# - an example ID (float or str)
# - a label (int, float, or str)
# - a feature dictionary
FeatGenerator = Generator[Tuple[IdType, Optional[LabelType], FeatureDict], None, None]

# a string path or Path object
PathOrStr = Union[Path, str]

# a sparse matrix for features
SparseFeatureMatrix = csr_matrix
