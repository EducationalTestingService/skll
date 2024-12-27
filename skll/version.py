# License: BSD 3 clause
"""
Define version number.

This module exists solely for version information so we only have to change it
in one place. Based on the suggestion `here. <http://bit.ly/16LbuJF>`_

:author: Dan Blanchard (dblanchard@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

__version__ = "5.1.0"
VERSION = tuple(int(x) for x in __version__.split("."))
