# License: BSD 3 clause
"""
Custom type annotations for readability.

:author: Nitin Madnani (nmadnani@ets.org)
"""

from pathlib import Path
from typing import Dict, Union

# a string path or Path object
PathOrStr = Union[Path, str]

# a mapping from example ID to fold ID;
# the example ID may be a float or a str
FoldMapping = Dict[Union[float, str], str]
