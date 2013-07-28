.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>

Running Experiments
===================
The simplest way to use SKLL is to create configuration files that describe
experiments you would like to run on pre-generated features. This document
describes the supported feature file formats, how to create configuration files
(and layout your directories), and how to use ``run_experiment`` to get things
going.

Feature file formats
--------------------
The following feature file formats are supported:

    **megam**
        An expanded form of the input format for the
        `MegaM classification package <http://www.umiacs.umd.edu/~hal/megam/>`_
        with the ``-fvals`` switch.

        The basic format is::

            # Instance1
            CLASS1  F0 2.5 F1 3 FEATURE_2 -152000
            # Instance2
            CLASS2  F1 7.524

        where the comments before each instance are optional IDs for the following
        line, class names are separated from feature-value pairs with a tab, and
        feature-value pairs are separated by spaces. Any omitted features for a
        given instance are assumed to be zero, so this format is handy when dealing
        with sparse data. We also include several utility scripts for converting
        to/from this MegaM format and for adding/removing features from the files.

    **tsv**
        A simple tab-delimited format with the following restrictions:

        *   The first column contains the class label for each instance.
        *   If there is a column called "id" present, this will be treated as the
            ID for each row.
        *   All other columns contain feature values, and every feature value must
            be specified (making this a poor choice for sparse data).

    **jsonlines**
        A twist on the `JSON <http://www.json.org/>`_ format where every line is a
        JSON dictionary (the entire contents of a normal JSON file). Each dictionary
        is expected to contain the following keys:

        *   *y*: The class label.
        *   *x*: A dictionary of feature values.
        *   *id*: An optional instance ID.


Creating configuration files
----------------------------
The experiment configuration files that run_experiment accepts are standard Python
configuration files that are similar in format to Windows INI files.[#]_ There are
three expected sections in a configuration file: ``Input``, ``Tuning``, and
``Output``. We discuss each of the possible settings that fall under each heading
below.

Input
^^^^^

    **train_location**
        Path to directory containing training data files. There must be a file
        for each featureset.

    **test_location**
        Path to directory containing training data files. There must be a file
        for each featureset.  *If unspecified, cross-validation is performed.*

    **cv_folds_location**
        Path to a csv file (with a header that is ignored) specifyingfolds for
        cross-validation. The first column should consist of training set IDs
        and the second should be a string for the fold ID (e.g., 1 through 5,
        A through D, etc.).  If specified, the CV and grid search will leave
        one fold ID out at a time.

        *Note*: K-1 folds will be used for grid search within CV, so there
        should be more at least 3 fold IDs.

    **featuresets**
        List of lists of prefixes for the files containing the features you
        would like to train/test on.  Each list will end up being a job.

    **featureset_names**
        Optional list of names for the feature sets.  If omitted, then the
        prefixes will be munged together to make names.

    **classifiers**
        List of sklearn models to try using. Acceptable values are described
        below.

        *   *logistic*: Logistic regression using LibLinear
        *   *svm_linear*: SVM using LibLinear
        *   *svm_radial*: SVM using LibSVM
        *   *naivebayes*: Multinomial Naive Bayes
        *   *dtree*: Decision Tree
        *   *rforest*: Random Forest
        *   *gradient*: Gradient Boosting Classifier
        *   *gb_regressor*: Gradient Boosting Regressor
        *   *ridge*: Ridge Regression
        *   *rescaled_ridge*: Ridge Regression, with predictions rescaled and
            constrained to better match the training set.
        *   *svr_linear*: Support Vector Regression with a linear kernel.
        *   *rescaled_svr_linear*: Linear SVR, with predictions rescaled and
            constrained to better match the training set.
        *   *rescaled_gb_regressor*: Gradient Boosting Regressor, with predictions
            rescaled and constrained to better match the training set.

Tuning
^^^^^^



Output
^^^^^^


.. [#] We are considering adding support for JSON configuration files in the future,
   but we have not added this functionality yet.

Using run_experiment
--------------------


