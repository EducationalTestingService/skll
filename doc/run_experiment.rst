.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>

Running Experiments
===================
The simplest way to use SKLL is to create configuration files that describe
experiments you would like to run on pre-generated features. This document
describes the supported feature file formats, how to create configuration files
(and layout your directories), and how to use ``run_experiment`` to get things
going.

Quick Example
-------------
If you don't want to read the whole document, and just want an example of how
things work, do the following from the command prompt:

.. code-block:: bash

    $ cd examples
    $ python make_example_iris_data.py          # download a simple dataset
    $ run_experiment --local example.cfg        # run an experiment


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

        *   If the data is labelled, there must be a column with the name
            specified by `tsv_label` in the `Input` section of the configuartion
            file you create for your experiment. This defaults to 'y'.
        *   If there is a column called "id" present, this will be treated as the
            ID for each row.
        *   All other columns contain feature values, and every feature value must
            be specified (making this a poor choice for sparse data).

Creating configuration files
----------------------------
The experiment configuration files that run_experiment accepts are standard
`Python configuration files <http://docs.python.org/2/library/configparser.html>`_
that are similar in format to Windows INI files. [#]_
There are four expected sections in a configuration file: ``General``,
``Input``, ``Tuning``, and ``Output``.  A detailed description of each possible
settings for each section is provided below, but to summarize:

*   If you want to do **cross-validation**, specify a path to training feature
    files, and set ``task`` to ``cross_validate`` You also can optionally use
    predetermined folds with the ``cv_folds_location`` setting.

*   If you want to **train a model and evaluate it** on some data, specify
    a training location, a test location, and a directory to store to store
    results, and set ``task`` to ``evaluate``.

*   If you want to just **train a model and generate predictions**, specify
    a training location, a test location, and set ``task`` to ``predict``.

*   If you want to just **train a model**, specify a training location, and set
    ``task`` to ``train_only``.

*   A list of classifiers/regressors to try on your feature files is
    required.

An example configuration file is available `here <https://github.com/EducationalTestingService/skll/blob/master/examples/example.cfg>`_.

General
^^^^^^^
    **experiment_name**
        A string used to identify this particular experiment configuration. When
        generating result summary files, this name helps prevent overwriting
        previous summaries.

    **task**
        What types of experiment we're trying to run. Valid options are:
        ``cross_validate``, ``evaluate``, ``predict``, and ``train``. See above
        for descriptions.


Input
^^^^^

    **train_location**
        Path to directory containing training data files. There must be a file
        for each featureset.

    **test_location** *(Optional)*
        Path to directory containing test data files. There must be a file
        for each featureset.

    **tsv_label** *(Optional)*
        If you're using TSV files, the class labels for each instance are
        assumed to be in a column with this name. If no column with this name is
        found, the data is assumed to be unlabelled. Defaults to 'y'.

    **ids_to_floats** *(Optional)*
        If you have a dataset with lots of examples, and your input files have
        IDs that look like numbers (can be converted by float()), then setting
        this to True will save you some memory by storing IDs as floats.
        Note that this will cause IDs to be printed as floats in prediction
        files (e.g., "4.0" instead of "4" or "0004" or "4.000").

    **cv_folds_location** *(Optional)*
        Path to a csv file (with a header that is ignored) specifyingfolds for
        cross-validation. The first column should consist of training set IDs
        and the second should be a string for the fold ID (e.g., 1 through 5,
        A through D, etc.).  If specified, the CV and grid search will leave
        one fold ID out at a time. [#]_

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

    **featureset_names** *(Optional)*
        Optional list of names for the feature sets.  If omitted, then the
        prefixes will be munged together to make names.

    **learners** [#]_
        List of scikit-learn models to try using. A separate job will be
        run for each combination of classifier and feature-set.
        Acceptable values are described below. Names in parentheses are
        aliases that can also be used in configuration files.

        *   *LogisticRegression (logistic)*: `Logistic regression using LibLinear <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`_
        *   *LinearSVC (svm_linear)*: `SVM using LibLinear <http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_
        *   *SVC (svm_radial)*: `SVM using LibSVM <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_
        *   *MultinomialNB (naivebayes)*: `Multinomial Naive Bayes <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB>`_
        *   *DecisionTreeClassifier (dtree)*: `Decision Tree Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier>`_
        *   *RandomForestClassifier (rforest)*: `Random Forest Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_
        *   *GradientBoostingClassifier (gradient)*: `Gradient Boosting Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier>`_
        *   *GradientBoostingRegressor (gb_regressor)*: `Gradient Boosting Regressor <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor>`_
        *   *Ridge (ridge)*: `Ridge Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier>`_
        *   *RescaledRidge (rescaled_ridge)*: Ridge Regression, with predictions
            rescaled and constrained to better match the training set.
        *   *SVR (svr_linear)*: `Support Vector Regression <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR>`_
            with a linear kernel. Can use other kernels by specifying a 'kernel'
            fixed parameter in the ``fixed_parameters`` list.
        *   *RescaledSVR (rescaled_svr_linear)*: Linear SVR, with predictions
            rescaled and constrained to better match the training set. Can use
            other kernels by specifying a 'kernel' fixed parameter in the
            ``fixed_parameters`` list.

    **fixed_parameters** *(Optional)*
        List of dicts containing parameters you want to have fixed for each
        classifier in ``learners`` list. Any empty ones will be ignored
        (and the defaults will be used).

        The default fixed parameters (beyond those that scikit-learn sets) are:

        *LogisticRegression*

        .. code-block:: python

           {'random_state': 123456789}

        *LinearSVC*

        .. code-block:: python

           {'random_state': 123456789}

        *SVC*

        .. code-block:: python

           {'cache_size': 1000}

        *DecisionTreeClassifier*

        .. code-block:: python

           {'criterion': 'entropy', 'compute_importances': True, 'random_state': 123456789}

        *RandomForestClassifier*

        .. code-block:: python

           {'n_estimators': 500, 'compute_importances': True, 'random_state': 123456789}

        *GradientBoostingClassifier*

        .. code-block:: python

           {'n_estimators': 500, 'random_state': 123456789}

        *GradientBoostingRegressor*

        .. code-block:: python

           {'n_estimators': 500, 'random_state': 123456789}



Tuning
^^^^^^
    **feature_scaling** *(Optional)*
        Whether to scale features by their mean and/or their standard deviation.
        This defaults to ``none``, which does no scaling of any kind. If you
        scale by mean, your data will automatically be converted to dense, so
        use caution when you have a very large dataset. Valid options are:

        *   *none*: perform no feature scaling at all.
        *   *with_std*: Scale feature values by their standard deviation.
        *   *with_mean*: Center features by subtracting their mean.
        *   *both*: perform both centering and scaling.


        Defaults to ``none``.

    **grid_search** *(Optional)*
        Whether or not to perform grid search to find optimal parameters for
        classifier. Defaults to ``False``.

    **grid_search_jobs** *(Optional)*
        Number of folds to run in parallel when using grid search. Defaults to
        number of grid search folds.

    **min_feature_count** *(Optional)*
        The minimum number of examples for a which each feature must be nonzero
        to be included in the model. Defaults to 1.

    **objective** *(Optional)*
        The objective function to use for tuning. Valid options are:

        *   *accuracy*: Overall `accuracy <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`_
        *   *precision*: `Precision <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`_
        *   *recall*: `Recall <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`_
        *   *f1_score_micro*: Micro-averaged `F1 score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_
        *   *f1_score_macro*: Macro-averaged `F1 score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html>`_
        *   *f1_score_least_frequent*: F1 score of the least frequent class. The
            least frequent class may vary from fold to fold for certain data
            distributions.
        *   *average_precision*: `Area under PR curve <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`_
            (for binary classification)
        *   *roc_auc*: `Area under ROC curve <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`_
            (for binary classification)
        *   *unweighted_kappa*: Unweighted `Cohen's kappa <http://en.wikipedia.org/wiki/Cohen's_kappa>`_ (any floating point
            values are rounded to ints)
        *   *linear_weighted_kappa*: Linear weighted kappa (any floating point
            values are rounded to ints)
        *   *quadratic_weighted_kappa*: Quadratic weighted kappa (any floating
            point values are rounded to ints)
        *   *kendall_tau*: `Kendall's tau <http://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient>`_
        *   *pearson*: `Pearson correlation <http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient>`_
        *   *spearman*: `Spearman rank-correlation <http://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient>`_
        *   *r2*: `R2 <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`_
        *   *mean_squared_error*: `Mean squared error regression loss <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`_


        Defaults to ``f1_score_micro``.

    **param_grids** *(Optional)*
        List of parameter grids to search for each classifier. Each parameter
        grid should be a list of of dictionaries mapping from strings to lists
        of parameter values. When you specify an empty list for a classifier,
        the default parameter grid for that classifier will be searched.

        The default parameter grids for each classifier are:

        *LogisticRegression*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *LinearSVC*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *SVC*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *MultinomialNB*

        .. code-block:: python

           [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}]

        *DecisionTreeClassifier*

        .. code-block:: python

           [{'max_features': ["auto", None]}]

        *RandomForestClassifier*

        .. code-block:: python

           [{'max_depth': [1, 5, 10, None]}]

        *GradientBoostingClassifier*

        .. code-block:: python

           [{'max_depth': [1, 3, 5], 'n_estimators': [500]}]

        *GradientBoostingRegressor*

        .. code-block:: python

           [{'max_depth': [1, 3, 5], 'n_estimators': [500]}]

        *Ridge*

        .. code-block:: python

           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *RescaledRidge*

        .. code-block:: python

           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *SVR*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

        *RescaledSVR*

        .. code-block:: python

           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]


    **pos_label_str** *(Optional)*
        The string label for the positive class in the binary
        classification setting. If unspecified, an arbitrary class is
        picked.


Output
^^^^^^

    **probability** *(Optional)*
        Whether or not to output probabilities for each class instead of the
        most probable class for each instance. Only really makes a difference
        when storing predictions. Defaults to ``False``.

    **results** *(Optional)*
        Directory to store result files in. If omitted, the current working
        directory is used, **and we're assumed to just want to generate
        predictions if the test_location is specified.**

    **log** *(Optional)*
        Directory to store result files in. If omitted, the current working
        directory is used.

    **models** *(Optional)*
        Directory to store trained models in. Can be omitted to not store
        models.

    **predictions** *(Optional)*
        Directory to store prediction files in. Can be omitted to not store
        predictions.

Note: you can use the same directory for ``results``, ``log``, ``models``, and
``predictions``.


Using run_experiment
--------------------
Once you have create the configuration file for your experiment, you can usually
just get your experiment started by running ``run_experiment CONFIGFILE``. That
said, there are a couple options that are specified via command-line arguments
instead of in the configuration file: ``--ablation`` and ``--keep-models``.

    ``--ablation``
        Runs an ablation study where repeated experiments are conducted with
        each feature set in the configuration file held out.

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

The result, log, model, and prediction files generated by run_experiment will
all share the following automatically generated prefix
``EXPERIMENT_FEATURESET_LEARNER``, where the following
definitions hold:

    ``EXPERIMENT``
        The name specified as ``experiment_name`` in the configuration file.

    ``FEATURESET``
        The feature set we're training on joined with "+".

    ``LEARNER``
        The learner the current results/model/etc. was generated using.

For every experiment you run, there will also be a result summary file generated
that is a tab-delimited file summarizing the results for each learner-featureset
combination you have in your configuration file. It is named
``EXPERIMENT_summary.tsv``.


.. rubric:: Footnotes

.. [#] We are considering adding support for JSON configuration files in the
   future, but we have not added this functionality yet.
.. [#] K-1 folds will be used for grid search within CV, so there should be at
   least 3 fold IDs.
.. [#] This field can also be called "classifiers" for backward-compatibility.
.. [#] This will happen automatically if Grid Map cannot be imported.
