# License: BSD 3 clause
"""
Functions for reading inputs for SKLL experiments.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
"""

from os.path import isfile, join

from skll.data.readers import Reader


def load_featureset(dir_path,
                    feat_files,
                    suffix,
                    id_col='id',
                    label_col='y',
                    ids_to_floats=False,
                    quiet=False,
                    class_map=None,
                    feature_hasher=False,
                    num_features=None,
                    logger=None):
    """
    Load a list of feature files and merge them.

    Parameters
    ----------
    dir_path : str
        Path to the directory that contains the feature files.
    feat_files : list of str
        A list of feature file prefixes.
    suffix : str
        The suffix to add to feature file prefixes to get the full filenames.
    id_col : str, optional
        Name of the column which contains the instance IDs.
        If no column with that name exists, or `None` is
        specified, example IDs will be automatically generated.
        Defaults to ``'id'``.
    label_col : str, optional
        Name of the column which contains the class labels.
        If no column with that name exists, or `None` is
        specified, the data is considered to be unlabeled.
        Defaults to ``'y'``.
    ids_to_floats : bool, optional
        Whether to convert the IDs to floats to save memory. Will raise error
        if we encounter non-numeric IDs.
        Defaults to ``False``.
    quiet : bool, optional
        Do not print "Loading..." status message to stderr.
        Defaults to ``False``.
    class_map : dict, optional
        Mapping from original class labels to new ones. This is
        mainly used for collapsing multiple labels into a single
        class. Anything not in the mapping will be kept the same.
        Defaults to ``None``.
    feature_hasher : bool, optional
        Should we use a FeatureHasher when vectorizing
        features?
        Defaults to ``False``.
    num_features : int, optional
        The number of features to use with the ``FeatureHasher``.
        This should always be set to the power of 2 greater
        than the actual number of features you're using.
        Defaults to ``None``.
    logger : logging.Logger, optional
        A logger instance to use to log messages instead of creating
        a new one by default.
        Defaults to ``None``.

    Returns
    -------
    merged_set : skll.FeatureSet
        A ``FeatureSet`` instance containing the specified labels, IDs, features,
        and feature vectorizer.
    """
    # if the training file is specified via train_file, then dir_path
    # actually contains the entire file name
    if isfile(dir_path):
        return Reader.for_path(dir_path,
                               label_col=label_col,
                               id_col=id_col,
                               ids_to_floats=ids_to_floats,
                               quiet=quiet,
                               class_map=class_map,
                               feature_hasher=feature_hasher,
                               num_features=num_features,
                               logger=logger).read()
    else:
        if len(feat_files) > 1 and feature_hasher:
            logger.warning("Since there are multiple feature files, "
                           "feature hashing applies to each specified "
                           "feature file separately.")
        merged_set = None
        for file_name in sorted(join(dir_path, featfile + suffix) for
                                featfile in feat_files):
            fs = Reader.for_path(file_name,
                                 label_col=label_col,
                                 id_col=id_col,
                                 ids_to_floats=ids_to_floats,
                                 quiet=quiet,
                                 class_map=class_map,
                                 feature_hasher=feature_hasher,
                                 num_features=num_features,
                                 logger=logger).read()
            if merged_set is None:
                merged_set = fs
            else:
                merged_set += fs
        return merged_set
