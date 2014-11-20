.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>

Utility Scripts
===============
In addition to the main script, :doc:`run_experiment <run_experiment>`, SKLL
comes with a number of helpful utility scripts that can be used to prepare
feature files and perform other routine tasks. Each is described briefly below.

.. _compute_eval_from_predictions:

compute_eval_from_predictions
-----------------------------
.. program:: compute_eval_from_predictions

Compute evaluation metrics from prediction files after you have run an
experiment.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. option:: examples_file

    SKLL input file with labeled examples

.. option:: predictions_file

    file with predictions from SKLL

.. option:: metric_names

    metrics to compute

Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _filter_features:

filter_features
---------------
.. program:: filter_features

Filter feature file to remove (or keep) any instances with the specified IDs or
labels.  Can also be used to remove/keep feature columns.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^
.. option:: infile

    Input feature file (ends in ``.arff``, ``.csv``, ``.jsonlines``,
    ``.megam``, ``.ndj``, or ``.tsv``)

.. option:: outfile

    Output feature file (must have same extension as input file)

Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: -f <feature <feature ...>>, --feature <feature <feature ...>>

    A feature in the feature file you would like to keep. If unspecified, no
    features are removed.

.. option:: -I <id <id ...>>, --id <id <id ...>>

    An instance ID in the feature file you would like to keep. If unspecified,
    no instances are removed based on their IDs.

.. option:: -i, --inverse

    Instead of keeping features and/or examples in lists, remove them.

.. option:: -L <label <label ...>>, --label <label <label ...>>

    A label in the feature file you would like to keep. If unspecified, no
    instances are removed based on their labels.

.. option:: -l label_col, --label_col label_col

    Name of the column which contains the class labels in ARFF, CSV, or TSV
    files. For ARFF files, this must be the final column to count as the label.
    (default: ``y``)

.. option:: -q, --quiet

    Suppress printing of "Loading..." messages.

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _generate_predictions:

generate_predictions
--------------------
.. program:: generate_predictions

Loads a trained model and outputs predictions based on input feature files.
Useful if you want to reuse a trained model as part of a larger system without
creating configuration files.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^
.. option:: model_file

    Model file to load and use for generating predictions.

.. option:: input_file

    A csv file, json file, or megam file (with or without the label column),
    with the appropriate suffix.

Optional Arguments
^^^^^^^^^^^^^^^^^^
.. option:: -l <label_col>, --label_col <label_col>

    Name of the column which contains the labels in ARFF, CSV, or TSV files.
    For ARFF files, this must be the final column to count as the label.
    (default: ``y``)

.. option:: -p <positive_label>, --positive_label <positive_label>

    If the model is only being used to predict the probability of a particular
    label, this specifies the index of the label we're predicting. 1 = second
    label, which is default for binary classification. Keep in mind that labels
    are sorted lexicographically. (default: 1)

.. option:: -q, --quiet

    Suppress printing of "Loading..." messages.

.. option:: -t <threshold>, --threshold <threshold>

    If the model we're using is generating probabilities of the positive label,
    return 1 if it meets/exceeds the given threshold and 0 otherwise.

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _join_features:

join_features
-------------
.. program:: join_features

Combine multiple feature files into one larger file.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^
.. option:: infile ...

    Input feature files (ends in ``.arff``, ``.csv``, ``.jsonlines``,
    ``.megam``, ``.ndj``, or ``.tsv``)

.. option:: outfile

    Output feature file (must have same extension as input file)

Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: -l <label_col>, --label_col <label_col>

    Name of the column which contains the labels in ARFF, CSV, or TSV files.
    For ARFF files, this must be the final column to count as the label.
    (default: ``y``)

.. option:: -q, --quiet

    Suppress printing of "Loading..." messages.

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _print_model_weights:

print_model_weights
-------------------
.. program:: print_model_weights

Prints out the weights of a given trained model.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. option:: model_file

    Model file to load.

Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: --k <k>

    Number of top features to print (0 for all) (default: 50)

.. option:: sign {positive,negative,all}

    Show only positive, only negative, or all weights (default: all)

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _skll_convert:

skll_convert
------------
.. program:: skll_convert

Convert between .arff, .csv., .jsonlines, .libsvm, .megam, and .tsv formats.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. option:: infile

    Input feature file (ends in ``.arff``, ``.csv``, ``.jsonlines``,
    ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv``)

.. option:: outfile

    Output feature file (ends in ``.arff``, ``.csv``, ``.jsonlines``,
    ``.libsvm``, ``.megam``, ``.ndj``, or ``.tsv``)


Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: -l <label_col>, --label_col <label_col>

    Name of the column which contains the labels in ARFF, CSV, or TSV files.
    For ARFF files, this must be the final column to count as the label.
    (default: ``y``)

.. option:: -q, --quiet

    Suppress printing of "Loading..." messages.

.. option:: --arff_regression

    Create ARFF files for regression, not classification.

.. option:: --arff_relation ARFF_RELATION

    Relation name to use for ARFF file. (default: ``skll_relation``)

.. option:: --reuse_libsvm_map REUSE_LIBSVM_MAP

    If you want to output multiple files that use the same mapping from labels
    and features to numbers when writing libsvm files, you can specify an
    existing .libsvm file to reuse the mapping from.

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _summarize_results:

summarize_results
-----------------
.. program:: summarize_results

Creates an experiment summary TSV file from a list of JSON files generated by
:ref:`run_experiment <run_experiment>`.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. option:: summary_file

    TSV file to store summary of results.

.. option:: json_file

    JSON results file generated by run_experiment.

Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: -a, --ablation

    The results files are from an ablation run.

.. option:: --version

    Show program's version number and exit.

