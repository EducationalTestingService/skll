# License: BSD 3 clause
"""
Functions for reading inputs for SKLL experiments.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
"""

import logging
from pathlib import Path
from typing import List, Optional

from skll.data import FeatureSet
from skll.data.readers import Reader
from skll.types import ClassMap, PathOrStr


def load_featureset(
    dir_path: PathOrStr,
    feat_files: List[str],
    suffix: str,
    id_col: str = "id",
    label_col: str = "y",
    ids_to_floats: bool = False,
    quiet: bool = False,
    class_map: Optional[ClassMap] = None,
    feature_hasher: bool = False,
    num_features: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> FeatureSet:
    """
    Load a list of feature files and merge them.

    Parameters
    ----------
    dir_path : :class:`skll.types.PathOrStr`
        Path to the directory that contains the feature files.
    feat_files : List[str]
        A list of feature file prefixes.
    suffix : str
        The suffix to add to feature file prefixes to get the full filenames.
    id_col : str, default="id"
        Name of the column which contains the instance IDs.
        If no column with that name exists, or `None` is
        specified, example IDs will be automatically generated.
    label_col : str, default="y"
        Name of the column which contains the class labels.
        If no column with that name exists, or `None` is
        specified, the data is considered to be unlabeled.
    ids_to_floats : bool, default=False
        Whether to convert the IDs to floats to save memory. Will raise error
        if we encounter non-numeric IDs.
    quiet : bool, default=False
        Do not print "Loading..." status message to stderr.
    class_map : Optional[:class:`skll.types.ClassMap`], default=None
        Mapping from original class labels to new ones. This is
        mainly used for collapsing multiple labels into a single
        class. Anything not in the mapping will be kept the same.
    feature_hasher : bool, default=False
        Should we use a FeatureHasher when vectorizing features?
    num_features : Optional[int], default=None
        The number of features to use with the ``FeatureHasher``.
        This should always be set to the power of 2 greater
        than the actual number of features you're using.
    logger : Optional[logging.Logger], default=None
        A logger instance to use to log messages instead of creating
        a new one by default.

    Returns
    -------
    merged_set : :class:`skll.data.featureset.FeatureSet`
        A ``FeatureSet`` instance containing the specified labels, IDs, features,
        and feature vectorizer.

    """
    # get a logger if one was not provided
    logger = logger if logger else logging.getLogger(__name__)

    # convert to Path object
    dir_path = Path(dir_path)

    # if the training file is specified via train_file, then dir_path
    # actually contains the entire file name
    if dir_path.is_file():
        return Reader.for_path(
            dir_path,
            label_col=label_col,
            id_col=id_col,
            ids_to_floats=ids_to_floats,
            quiet=quiet,
            class_map=class_map,
            feature_hasher=feature_hasher,
            num_features=num_features,
            logger=logger,
        ).read()
    else:
        if len(feat_files) > 1 and feature_hasher:
            logger.warning(
                "Since there are multiple feature files, "
                "feature hashing applies to each specified "
                "feature file separately."
            )
        merged_set = FeatureSet("empty", [])
        for file_name in sorted(dir_path / f"{featfile}{suffix}" for featfile in feat_files):
            fs = Reader.for_path(
                file_name,
                label_col=label_col,
                id_col=id_col,
                ids_to_floats=ids_to_floats,
                quiet=quiet,
                class_map=class_map,
                feature_hasher=feature_hasher,
                num_features=num_features,
                logger=logger,
            ).read()

            if len(merged_set) == 0:
                merged_set = fs
            else:
                merged_set += fs

        return merged_set
