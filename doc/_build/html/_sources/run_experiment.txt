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

    **jsonlines**
        A twist on the `JSON <http://www.json.org/>`_ format where every line is a
        JSON dictionary (the entire contents of a normal JSON file). Each dictionary
        is expected to contain the following keys:

        *   *y*: The class label.
        *   *x*: A dictionary of feature values.
        *   *id*: An optional instance ID.

    **megam**
        An expanded form of the input format for the
        `MegaM classification package <http://www.umiacs.umd.edu/~hal/megam/>`_
        with the ``-fvals`` switch.

        The basic format is::

            # Instance1
            CLASS1    F0 2.5 F1 3 FEATURE_2 -152000
            # Instance2
            CLASS2    F1 7.524

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

Creating configuration files
----------------------------
The experiment configuration files that run_experiment accepts are standard Python
configuration files that are similar in format to Windows INI files. [#]_
There are three expected sections in a configuration file: ``Input``,
``Tuning``, and ``Output``.  A detailed description of each possible settings
for each section is provided below, but to summarize:

*   If you want to do **cross-validation**, specify a path to training
    feature files, but not one to test ones. You also can optionally use
    predetermined folds with the ``cv_folds_location`` setting.

*   If you want to **train a model and evaluate it** on some data, specify both
    a training location, a test location, and a directory to store to store
    results.

*   If you want to just **train a model and generate predictions**, specify
    a training location, a test location, but *do not* specify a results
    directory.

*   A list of classifiers/regressors to try on your feature files is
    required.

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
        would like to train/test on.  Each list will end up being a job. IDs
        are required to be the same in all of the feature files, and a
        ``ValueError`` will be raised if this is not the case.

    **suffix** *(Optional)*
        The file format the training/test files are in. Valid option are ".tsv",
        ".megam", and ".jsonlines" (one complete JSON dict per line in the
        file).

        If you omit this field, it is assumed that the "prefixes" listed
        in ``featuresets`` are actually complete filenames. This can be useful
        if you have feature files that are all in different formats that you
        would like to combine.

    **featureset_names**
        Optional list of names for the feature sets.  If omitted, then the
        prefixes will be munged together to make names.

    **learners** [#]_
        List of scikit-learn models to try using. A separate job will be
        run for each combination of classifier and feature-set.
        Acceptable values are described below.

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

    **fixed_parameters**
        List of dicts containing parameters you want to have fixed for each
        classifier in ``classifiers`` list. Any empty ones will be ignored
        (and the defaults will be used).



Tuning
^^^^^^

    **grid_search**
        Whether or not to perform grid search to find optimal parameters for
        classifier.

    **grid_search_jobs**
        Number of folds to run in parallel when using grid search. Defaults to
        number of grid search folds.

    **objective**
        The objective function to use for tuning. Valid options are:

        *   *f1_score_micro*: Micro-averaged f-score
        *   *f1_score_macro*: Macro-averaged f-score
        *   *f1_score_least_frequent*: F-score of the least frequent class. The
            least frequent class may vary from fold to fold for certain data
            distributions.
        *   *accuracy*: Overall accuracy
        *   *spearman*: Spearman rank-correlation
        *   *pearson*: Pearson correlation
        *   *kendall_tau*: Kendall's tau
        *   *quadratic_weighted_kappa*: The quadratic weighted kappa, where any
            floating point values are rounded
        *   *unweighted_kappa*: Unweighted Cohen's kappa, where the classes
            should be ints

    **param_grids**
        List of parameter grids to search for each classifier. Each parameter
        grid should be a list of of dictionaries mapping from strings to lists
        of parameter values. When you specify an empty list for a classifier,
        the default parameter grid for that classifier will be searched.

        The default parameter grids for each classifier are:

        *logistic*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *svm_linear*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *svm_radial*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *naivebayes*

        .. code-block:: python

           [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}]

        *dtree*

        .. code-block:: python

           [{'max_features': ["auto", None]}]

        *rforest*

        .. code-block:: python

           [{'max_depth': [1, 5, 10, None]}]

        *gradient*

        .. code-block:: python

           [{'max_depth': [1, 3, 5], 'n_estimators': [500]}]

        *gb_regressor*

        .. code-block:: python

           [{'max_depth': [1, 3, 5], 'n_estimators': [500]}]

        *ridge*

        .. code-block:: python

           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *rescaled_ridge*

        .. code-block:: python

           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *svr_linear*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *rescaled_svr_linear*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]


    **scale_features**
        Whether to scale features by their mean (for dense data only) and
        standard deviation.  This defaults to ``False``. It is highly
        recommended that you only use this with dense features.

    **use_dense_features**
        Whether the features should be converted to dense matrices. This
        defaults to ``False``.


Output
^^^^^^

    **probability**
        Whether or not to output probabilities for each class instead of the
        most probable class for each instance. Only really makes a difference
        when storing predictions.

    **results**
        Directory to store result files in. If omitted, the current working
        directory is used, **and we're assumed to just want to generate
        predictions if the test_location is specified.**

    **log**
        Directory to store result files in. If omitted, the current working
        directory is used.

    **models**
        Directory to store trained models in. Can be omitted to not store
        models.

    **predictions**
        Directory to store prediction files in. Can be omitted to not store
        predictions.



Using run_experiment
--------------------
Once you have create the configuration file for your experiment, you can usually
just get your experiment started by running ``run_experiment CONFIGFILE``. That
said, there are a couple options that are specified via command-line arguments
instead of in the configuration file: ``--ablation`` and ``--keep-models``.

    ``--ablation``
        Runs an ablation study where repeated experiments are conducted where
        each feature set in the configuration file is held out.

    ``--keep-models``
        If trained models already exist for any of the learner/featureset
        combinations in your configuration file, just load those models and
        do not retrain/overwrite them.

If you have `Grid Map <http://pypi.python.org/pypi/gridmap>`__ installed,
run_experiment will automatically schedule jobs on your DRMAA-compatible
cluster. However, if you would just like to run things locally, you can specify
the ``--local`` option. [#]_ You can also customize the queue and machines that
are used for running your jobs via the ``--queue`` and ``--machines`` arguments.
For complete details on how to specify these options, just run ``run_experiment
--help``.


.. rubric:: Footnotes

.. [#] We are considering adding support for JSON configuration files in the
   future, but we have not added this functionality yet.
.. [#] This field can also be called "classifiers" for backward-compatibility.
.. [#] This will happen automatically if Grid Map cannot be imported.