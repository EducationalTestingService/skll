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

.. option:: --id_col <id_col>

    Name of the column which contains the instance IDs in ARFF, CSV, or TSV files.
    (default: ``id``)

.. option:: -L <label <label ...>>, --label <label <label ...>>

    A label in the feature file you would like to keep. If unspecified, no
    instances are removed based on their labels.

.. option:: -l <label_col>, --label_col <label_col>

    Name of the column which contains the class labels in ARFF, CSV, or TSV
    files. For ARFF files, this must be the final column to count as the label.
    (default: ``y``)

.. option:: -db, --drop-blanks
    
    Drop all lines/rows that have any blank values.
    (default: ``False``)

.. option:: -rb <replacement>, --replace-blanks-with <replacement>

    Specifies a new value with which to replace blank values in all columns in the
    file. To replace blanks differently in each column, use the SKLL Reader API directly.
    (default: ``None``)
      
.. option:: -q, --quiet

    Suppress printing of ``"Loading..."`` messages.

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _generate_predictions:

generate_predictions
--------------------
.. program:: generate_predictions

Loads a trained model and outputs predictions based on input feature files.
Useful if you want to reuse a trained model as part of a larger system without
creating configuration files. Offers the following modes of operation:

- For non-probabilistic classification and regression, generate the predictions.
- For probabilistic classification, generate either the most likely labels 
  or the probabilities for each class label.
- For binary probablistic classification, generate the positive class label
  only if its probability exceeds the given threshold. The positive class
  label is either read from the model file or inferred the same way as 
  a SKLL learner would.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^
.. option:: model_file

    Model file to load and use for generating predictions.

.. option:: input_file(s)

    One or more csv file(s), jsonlines file(s), or megam file(s) (with or without the
    label column), with the appropriate suffix.

Optional Arguments
^^^^^^^^^^^^^^^^^^
.. option:: -i <id_col>, --id_col <id_col>

    Name of the column which contains the instance IDs in ARFF, CSV, or TSV files.
    (default: ``id``)

.. option:: -l <label_col>, --label_col <label_col>

    Name of the column which contains the labels in ARFF, CSV, or TSV files.
    For ARFF files, this must be the final column to count as the label. 
    (default: ``y``)

.. option:: -o <path>, --output_file <path>
    
    Path to output TSV file. If not specified, predictions will be printed
    to stdout. For probabilistic binary classification, the probability of
    the positive class will always be in the last column.
    
.. option:: -p, --predict_labels
    
    If the model does probabilistic classification, output the class label
    with the highest probability instead of the class probabilities.

.. option:: -q, --quiet

    Suppress printing of ``"Loading..."`` messages.

.. option:: -t <threshold>, --threshold <threshold>

    If the model does binary probabilistic classification, 
    return the positive class label only if it meets/exceeds
    the given threshold and the other class label otherwise.

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

    Suppress printing of ``"Loading..."`` messages.

.. option:: --version

    Show program's version number and exit.

-------------------------------------------------------------------------------

.. _plot_learning_curves:

plot_learning_curves
--------------------
.. program:: plot_learning_curves

Generate learning curve plots from a learning curve output TSV file.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. option:: tsv_file

    Input learning Curve TSV output file.

.. option:: output_dir

    Output directory to store the learning curve plots.

-------------------------------------------------------------------------------

.. _print_model_weights:

print_model_weights
-------------------
.. program:: print_model_weights

Prints out the weights of a given trained model. If the model
was trained using :ref:`feature hashing <feature_hasher>`, 
feature names of the form ``hashed_feature_XX`` will be used
since the original feature names no longer apply.

Positional Arguments
^^^^^^^^^^^^^^^^^^^^

.. option:: model_file

    Model file to load.

Optional Arguments
^^^^^^^^^^^^^^^^^^

.. option:: --k <k>

    Number of top features to print (0 for all) (default: 50)

.. option:: --sign {positive,negative,all}

    Show only positive, only negative, or all weights (default: ``all``)

.. option:: --sort_by_labels

    Oorder the features by classes (default: ``False``). Mutually exclusive
    with the ``--k`` option.

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

    Suppress printing of ``"Loading..."`` messages.

.. option:: --arff_regression

    Create ARFF files for regression, not classification.

.. option:: --arff_relation ARFF_RELATION

    Relation name to use for ARFF file. (default: ``skll_relation``)

.. option:: --no_labels

    Used to indicate that the input data has no labels.

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



