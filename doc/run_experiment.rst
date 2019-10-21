.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>

Running Experiments
===================

General Workflow
----------------

To run your own SKLL experiments via the command line, the following general workflow
is recommended.

**Get your data into the correct format**

SKLL can work with several common data formats, all of which are described
:ref:`here <file_formats>`.

If you need to convert between any of the supported formats, because, for
example, you would like to create a single data file that will work both with
SKLL and Weka (or some other external tool), the :ref:`skll_convert` script can
help you out.  It is as easy as:

.. code-block:: bash

    $ skll_convert examples/titanic/train/family.csv examples/titanic/train/family.arff

**Create sparse feature files, if necessary**

:ref:`skll_convert` can also create sparse data files in
:ref:`.jsonlines <ndj>`, :ref:`.libsvm <libsvm>`, :ref:`.megam <megam>`, or
:ref:`.ndj <ndj>` formats.  This is very useful for saving disk space and
memory when you have a large data set with mostly zero-valued features.

**Set up training and testing directories/files**

At a minimum, you will probably want to work with a training set and a testing
set.  If you have multiple feature files that you would like SKLL to join together
for you automatically, you will need to create feature files with the exact
same names and store them in training and testing directories.  You can
specifiy these directories in your config file using
:ref:`train_directory <train_directory>` and
:ref:`test_directory <test_directory>`.  The list of files is specified using
the :ref:`featuresets <featuresets>` setting.

If you're conducting a simpler experiment, where you have a single training
file with all of your features and a similar single testing file, you should
use the :ref:`train_file <train_file>` and :ref:`test_file <test_file>`
settings in your config file.

.. note:: If you would like to split an existing file up into a training
          set and a testing set, you can employ the :ref:`filter_features`
          utility script to select instances you would like to include in
          each file.

**Create an experiment configuration file**

You saw a :ref:`basic configuration file <titanic_config>` in the tutorial. For your
own experiment, you will need to refer to the :ref:`create_config` section.

**Run configuration file through run_experiment**

There are a few meta-options for experiments that are specified directly to the
:ref:`run_experiment <run_experiment>` command rather than in a configuration
file.  For example, if you would like to run an ablation experiment, which
conducts repeated experiments using different combinations of the features in
your config, you should use the :option:`run_experiment --ablation` option. A
complete list of options is available :ref:`here <run_experiment>`.

Next, we describe the numerous file formats that SKLL supports for reading
in features.

.. _file_formats:

Feature files
-------------

SKLL supports the following feature file formats:

.. _arff:

arff
^^^^
The same file format used by `Weka <https://www.cs.waikato.ac.nz/ml/weka/>`__
with the following added restrictions:

*   Only simple numeric, string, and nomimal values are supported.
*   Nominal values are converted to strings.
*   If the data has instance IDs, there should be an attribute with the name
    specified by :ref:`id_col <id_col>` in the :ref:`Input` section of the configuration file you create for your experiment. This defaults to ``id``.  If there is no such attribute, IDs will be generated automatically.
*   If the data is labelled, there must be an attribute with the name specified
    by :ref:`label_col <label_col>` in the :ref:`Input` section of the
    configuartion file you create for your experiment. This defaults to ``y``.
    This must also be the final attribute listed (like in Weka).

.. _csv:

csv/tsv
^^^^^^^

A simple comma or tab-delimited format. SKLL underlyingly uses 
[pandas](https://pandas.pydata.org) to read these files which is
extremely fast but at the cost of some extra memory consumption.

When using this file format, the following restrictions apply:

*   If the data is labelled, there must be a column with the name
    specified by :ref:`label_col <label_col>` in the :ref:`Input` section of the
    configuration file you create for your experiment. This defaults to
    ``y``.
*   If the data has instance IDs, there should be a column with the name
    specified by :ref:`id_col <id_col>` in the :ref:`Input` section of the configuration file you create for your experiment. This defaults to ``id``.  If there is no such column, IDs will be generated automatically.
*   All other columns contain feature values, and every feature value
    must be specified (making this a poor choice for sparse data).

.. warning:: 
 
    1. SKLL will raise an error if there are blank values in **any** of the
       columns. You must either drop all rows with blank values in any column
       or replace the blanks with a value you specify. To drop or replace via
       the command line, use the :ref:`filter_features <filter_features>` script.
       You can also drop/replace via the SKLL Reader API, specifically :py:mod:`skll.data.readers.CSVReader` and :py:mod:`skll.data.readers.TSVReader`.

    2. Dropping blanks will drop **all** rows with blanks in **any** of
       the columns. If you care only about **some** of the columns in the file
       and do not want to rows to be dropped due to blanks in the other columns,
       you should remove the columns you do not care about before dropping the
       blanks. For example, consider a hypothetical file ``in.csv`` that contains
       feature columns named ``A`` through ``G`` with the IDs stored in a column
       named ``ID`` and the labels stored in a column named ``CLASS``. You only
       care about columns ``A``, ``C``, and ``F`` and want to drop all rows in
       the file that have blanks in any of these 3 columns but **do not** want
       to lose data due to there being blanks in any of the other columns. On
       the command line, you can run the following two commands:

        .. code-block:: bash

            $ filter_features -f A C F --id_col ID --label_col class in.csv temp.csv
            $ filter_features --id_col ID --label_col CLASS --drop_blanks temp.csv out.csv

       If you are using the SKLL Reader API, you can accomplish the same in a
       single step by also passing using the keyword argument ``pandas_kwargs`` 
       when instantiating either a :py:mod:`skll.data.readers.CSVReader` or a 
       :py:mod:`skll.data.readers.TSVReader`. For our example:

        .. code-block:: python

            r = CSVReader.for_path('/path/to/in.csv',
                                   label_col='CLASS',
                                   id_col='ID',
                                   drop_blanks=True,
                                   pandas_kwargs={'usecols': ['A', 'C', 'F', 'ID', 'CLASS']})
            fs = r.read()

       Make sure to include the ID and label columns in the `usecols` list 
       otherwise ``pandas`` will drop them too.

.. _ndj:

jsonlines/ndj *(Recommended)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A twist on the `JSON <http://www.json.org/>`__ format where every line is a
either JSON dictionary (the entire contents of a normal JSON file), or a
comment line starting with ``//``. Each dictionary is expected to contain the
following keys:

*   **y**: The class label.
*   **x**: A dictionary of feature values.
*   **id**: An optional instance ID.

This is the preferred file format for SKLL, as it is sparse and can be slightly
faster to load than other formats.

.. _libsvm:

libsvm
^^^^^^

While we can process the standard input file format supported by
`LibSVM <https://www.csie.ntu.edu.tw/~cjlin/libsvm/>`__,
`LibLinear <https://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__,
and `SVMLight <http://svmlight.joachims.org>`__, we also support specifying
extra metadata usually missing from the format in comments at the of each line.
The comments are not mandatory, but without them, your labels and features will
not have names.  The comment is structured as follows::

    ID | 1=ClassX | 1=FeatureA 2=FeatureB

The entire format would like this::

    2 1:2.0 3:8.1 # Example1 | 2=ClassY | 1=FeatureA 3=FeatureC
    1 5:7.0 6:19.1 # Example2 | 1=ClassX | 5=FeatureE 6=FeatureF

.. note::
    IDs, labels, and feature names cannot contain the following
    characters:  ``|`` ``#`` ``=``

.. _megam:

megam
^^^^^

An expanded form of the input format for the
`MegaM classification package <http://users.umiacs.umd.edu/~hal/megam/>`__ with
the ``-fvals`` switch.

The basic format is::

    # Instance1
    CLASS1    F0 2.5 F1 3 FEATURE_2 -152000
    # Instance2
    CLASS2    F1 7.524

where the **optional** comments before each instance specify the ID for the
following line, class names are separated from feature-value pairs with a tab,
and feature-value pairs are separated by spaces. Any omitted features for a
given instance are assumed to be zero, so this format is handy when dealing
with sparse data. We also include several utility scripts for converting
to/from this MegaM format and for adding/removing features from the files.

.. _create_config:

Configuration file fields
-------------------------

The experiment configuration files that ``run_experiment`` accepts are standard
`Python configuration files <https://docs.python.org/3/library/configparser.html>`__
that are similar in format to Windows INI files. [#]_
There are four expected sections in a configuration file: :ref:`General`,
:ref:`Input`, :ref:`Tuning`, and :ref:`Output`.  A detailed description of each
field in each section is provided below, but to summarize:

.. _cross_validate:

*   If you want to do **cross-validation**, specify a path to training feature
    files, and set :ref:`task` to ``cross_validate``. Please note that the
    cross-validation currently uses
    `StratifiedKFold <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html>`__.
    You also can optionally use predetermined folds with the
    :ref:`folds_file <folds_file>` setting.

    .. note::

        When using classifiers, SKLL will automatically reduce the
        number of cross-validation folds to be the same as the minimum
        number of examples for any of the classes in the training data.

.. _evaluate:

*   If you want to **train a model and evaluate it** on some data, specify a
    training location, a test location, and a directory to store results,
    and set :ref:`task` to ``evaluate``.

.. _predict:

*   If you want to just **train a model and generate predictions**, specify
    a training location, a test location, and set :ref:`task` to ``predict``.

.. _train:

*   If you want to just **train a model**, specify a training location, and set
    :ref:`task` to ``train``.

.. _learning_curve:

*   If you want to **generate a learning curve** for your data, specify a training location and set :ref:`task` to ``learning_curve``. The learning curve is generated using essentially the same underlying process as in `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve>`__ except that the SKLL feature pre-processing pipline is used while training the various models and computing the scores.

    .. note::

        Ideally, one would first do cross-validation experiments with grid search and/or ablation and get a well-performing set of features and hyper-parameters for a set of learners. Then, one would explicitly specify those features (via :ref:`featuresets <featuresets>`) and hyper-parameters (via :ref:`fixed_parameters <fixed_parameters>`) in the config file for the learning curve and explore the impact of the size of the training data.

.. _learners_required:

*   A :ref:`list of classifiers/regressors <learners>` to try on your feature
    files is required.

Example configuration files are available `here <https://github.com/EducationalTestingService/skll/tree/master/examples/>`__ under the ``boston``, ``iris``, and ``titanic`` sub-directories.

.. _general:

General
^^^^^^^

Both fields in the General section are required.

.. _experiment_name:

experiment_name
"""""""""""""""

A string used to identify this particular experiment configuration. When
generating result summary files, this name helps prevent overwriting previous
summaries.

.. _task:

task
""""

What types of experiment we're trying to run. Valid options are:
:ref:`cross_validate <cross_validate>`, :ref:`evaluate <evaluate>`,
:ref:`predict <predict>`, :ref:`train <train>`, :ref:`learning_curve <learning_curve>`.

.. _input:

Input
^^^^^

The Input section must specify the machine learners to use via the :ref:`learners` 
field as welll as the data and features to be used when
training the model. This can be done by specifying either (a) 
:ref:`train_file <train_file>`  in which case all of the features in
the file will be used, or (b) :ref:`train_directory <train_directory>` along
with :ref:`featuresets <featuresets>`.

.. _learners:

learners
""""""""
List of scikit-learn models to be used in the experiment. Acceptable values
are described below.  Custom learners can also be specified. See 
:ref:`custom_learner_path <custom_learner_path>`.

.. _classifiers:

Classifiers:

    *   **AdaBoostClassifier**: `AdaBoost Classification <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier>`__.  Note that the default base estimator is a ``DecisionTreeClassifier``. A different base estimator can be used by specifying a ``base_estimator`` fixed parameter in the :ref:`fixed_parameters <fixed_parameters>` list. The following additional base estimators are supported: ``MultinomialNB``, ``SGDClassifier``, and ``SVC``. Note that the last two base require setting an additional ``algorithm`` fixed parameter with the value ``'SAMME'``.
    *   **DummyClassifier**: `Simple rule-based Classification <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html#sklearn.dummy.DummyClassifier>`__
    *   **DecisionTreeClassifier**: `Decision Tree Classification <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier>`__
    *   **GradientBoostingClassifier**: `Gradient Boosting Classification <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier>`__
    *   **KNeighborsClassifier**: `K-Nearest Neighbors Classification <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`__
    *   **LinearSVC**: `Support Vector Classification using LibLinear <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`__
    *   **LogisticRegression**: `Logistic Regression Classification using LibLinear <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`__
    *   **MLPClassifier**: `Multi-layer Perceptron Classification <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier>`__
    *   **MultinomialNB**: `Multinomial Naive Bayes Classification <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB>`__
    *   **RandomForestClassifier**: `Random Forest Classification <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`__
    *   **RidgeClassifier**: `Classification using Ridge Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html#sklearn.linear_model.RidgeClassifier>`__
    *   **SGDClassifier**: `Stochastic Gradient Descent Classification <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`__
    *   **SVC**: `Support Vector Classification using LibSVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`__

.. _regressors:

Regressors:

    *   **AdaBoostRegressor**: `AdaBoost Regression <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor>`__. Note that the default base estimator is a ``DecisionTreeRegressor``. A different base estimator can be used by specifying a ``base_estimator`` fixed parameter in the :ref:`fixed_parameters <fixed_parameters>` list. The following additional base estimators are supported: ``SGDRegressor``, and ``SVR``.
    *   **BayesianRidge**: `Bayesian Ridge Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge>`__
    *   **DecisionTreeRegressor**: `Decision Tree Regressor <https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor>`__
    *   **DummyRegressor**: `Simple Rule-based Regression <https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html#sklearn.dummy.DummyRegressor>`__
    *   **ElasticNet**: `ElasticNet Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet>`__
    *   **GradientBoostingRegressor**: `Gradient Boosting Regressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor>`__
    *   **HuberRegressor**: `Huber Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor>`__
    *   **KNeighborsRegressor**: `K-Nearest Neighbors Regression <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor>`__
    *   **Lars**: `Least Angle Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars>`__
    *   **Lasso**: `Lasso Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso>`__
    *   **LinearRegression**: `Linear Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression>`__
    *   **LinearSVR**: `Support Vector Regression using LibLinear <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR>`__
    *   **MLPRegressor**: `Multi-layer Perceptron Regression <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor>`__
    *   **RandomForestRegressor**: `Random Forest Regression <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor>`__
    *   **RANSACRegressor**: `RANdom SAmple Consensus Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor>`__. Note that the default base estimator is a ``LinearRegression``. A different base regressor can be used by specifying a ``base_estimator`` fixed parameter in the :ref:`fixed_parameters <fixed_parameters>` list.
    *   **Ridge**: `Ridge Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge>`__
    *   **SGDRegressor**: `Stochastic Gradient Descent Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html>`__
    *   **SVR**: `Support Vector Regression using LibSVM <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR>`__
    *   **TheilSenRegressor**: `Theil-Sen Regression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor>`__

    For all regressors, you can also prepend ``Rescaled`` to the
    beginning of the full name (e.g., ``RescaledSVR``) to get a version
    of the regressor where predictions are rescaled and constrained to
    better match the training set.

.. _featuresets:

featuresets
"""""""""""
List of lists of prefixes for the files containing the features you would like
to train/test on.  Each list will end up being a job. IDs are required to be
the same in all of the feature files, and a :py:exc:`ValueError` will be raised
if this is not the case.  Cannot be used in combination with
:ref:`train_file <train_file>` or :ref:`test_file <test_file>`.

.. note::

    If specifying :ref:`train_directory <train_directory>` or
    :ref:`test_directory <test_directory>`, :ref:`featuresets <featuresets>`
    is required.


.. _train_file:

train_file 
""""""""""

Path to a file containing the features to train on.  Cannot be used in
combination with :ref:`featuresets <featuresets>`,
:ref:`train_directory <train_directory>`, or :ref:`test_directory <test_directory>`.

.. note::

    If :ref:`train_file <train_file>` is not specified,
    :ref:`train_directory <train_directory>` must be.

.. _train_directory:

train_directory 
"""""""""""""""

Path to directory containing training data files. There must be a file for each
featureset.  Cannot be used in combination with :ref:`train_file <train_file>`
or :ref:`test_file <test_file>`.

.. note::

    If :ref:`train_directory <train_directory>` is not specified,
    :ref:`train_file <train_file>` must be.

The following is a list of the other optional fields in this section 
in alphabetical order.

.. _class_map:

class_map *(Optional)*
""""""""""""""""""""""

If you would like to collapse several labels into one, or otherwise modify your
labels (without modifying your original feature files), you can specify a
dictionary mapping from new class labels to lists of original class labels. For
example, if you wanted to collapse the labels ``beagle`` and ``dachsund`` into a
``dog`` class, you would specify the following for ``class_map``:

.. code-block:: python

   {'dog': ['beagle', 'dachsund']}

Any labels not included in the dictionary will be left untouched.

One other use case for ``class_map`` is to deal with classification labels that
would be converted to ``float`` improperly. All ``Reader`` sub-classes use the
:py:mod:`skll.data.readers.safe_float` function internally to read labels. This function tries to
convert a single label first to ``int``, then to ``float``. If neither
conversion is possible, the label remains a ``str``. Thus, care must be taken
to ensure that labels do not get converted in unexpected ways. For example,
consider the situation where there are classification labels that are a mixture
of ``int``-converting and ``float``-converting labels:

.. code-block:: python

    import numpy as np
    from skll.data.readers import safe_float
    np.array([safe_float(x) for x in ["2", "2.2", "2.21"]]) # array([2.  , 2.2 , 2.21])

The labels will all be converted to floats and any classification model
generated with this data will predict labels such as ``2.0``, ``2.2``, etc.,
not ``str`` values that exactly match the input labels, as might be expected.
``class_map`` could be used to map the original labels to new values that do
not have the same characteristics.

.. _custom_learner_path:

custom_learner_path *(Optional)*
""""""""""""""""""""""""""""""""

Path to a ``.py`` file that defines a custom learner.  This file will be
imported dynamically.  This is only required if a custom learner is specified
in the list of :ref:`learners`.

All Custom learners must implement the ``fit`` and
``predict`` methods. Custom classifiers must either (a) inherit from an existing scikit-learn classifier, or (b) inherit from both `sklearn.base.BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`__. *and* from `sklearn.base.ClassifierMixin <https://scikit-learn.org/stable/modules/generated/sklearn.base.ClassifierMixin.html>`__.

Similarly, Custom regressors must either (a) inherit from an existing scikit-learn regressor, or (b) inherit from both `sklearn.base.BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`__. *and* from `sklearn.base.RegressorMixin <https://scikit-learn.org/stable/modules/generated/sklearn.base.RegressorMixin.html>`__.

Learners that require dense matrices should implement a method ``requires_dense``
that returns ``True``.

.. _feature_hasher:

feature_hasher *(Optional)*
"""""""""""""""""""""""""""

If "true", this enables a high-speed, low-memory vectorizer that uses
feature hashing for converting feature dictionaries into NumPy arrays
instead of using a
`DictVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`__.  This flag will drastically
reduce memory consumption for data sets with a large number of
features. If enabled, the user should also specify the number of
features in the :ref:`hasher_features <hasher_features>` field.  For additional
information see `the scikit-learn documentation <https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing>`__.

.. warning:: Due to the way SKLL experiments are architected, if the features
             for an experiment are spread across multiple files on disk, feature
             hashing will be applied to each file *separately*. For example, if
             you have F feature files and you choose H as the number of hashed
             features (via :ref:`hasher_features <hasher_features>`), you will
             end up with F x H features in the end. If this is not the
             desired behavior, use the :ref:`join_features <join_features>` 
             utility script to combine all feature files into a single file
             before running the experiment.


.. _feature_scaling:

feature_scaling *(Optional)*
""""""""""""""""""""""""""""

Whether to scale features by their mean and/or their standard deviation. If you
scale by mean, your data will automatically be converted to dense, so use
caution when you have a very large dataset. Valid options are:

none
    Perform no feature scaling at all.

with_std
    Scale feature values by their standard deviation.

with_mean
    Center features by subtracting their mean.

both
    Perform both centering and scaling.

Defaults to none.

.. _featureset_names:

featureset_names *(Optional)*
"""""""""""""""""""""""""""""

Optional list of names for the feature sets.  If omitted, then the prefixes
will be munged together to make names.

.. _folds_file:

folds_file *(Optional)*
""""""""""""""""""""""""""""""

Path to a csv file specifying the mapping of instances in the training data
to folds. This can be specified when the :ref:`task` is either ``train`` or
``cross_validate``. For the ``train`` task, if :ref:`grid_search <grid_search>`
is ``True``, this file, if specified, will be used to define the
cross-validation used for the grid search (leave one fold ID out at a time).
Otherwise, it will be ignored.

For the ``cross_validate`` task, this file will be used to define the outer
cross-validation loop and, if :ref:`grid_search <grid_search>` is ``True``, also for the
inner grid-search cross-validation loop. If the goal of specifiying the folds
file is to ensure that the model does not learn to differentiate based on a confound:
e.g. the data from the same person is always in the same fold, it makes sense to
keep the same folds for both the outer and the inner cross-validation loops.

However, sometimes the goal of specifying the folds file is simply for the
purpose of comparison to another existing experiment or another context
in which maintaining the constitution of the folds in the inner
grid-search loop is not required. In this case, users may set the parameter
:ref:`use_folds_file_for_grid_search <use_folds_file_for_grid_search>`
to ``False`` which will then direct the inner grid-search cross-validation loop
to simply use the number specified via :ref:`grid_search_folds <grid_search_folds>`
instead of using the folds file. This will likely lead to shorter execution times as
well depending on how many folds are in the folds file and the value
of :ref:`grid_search_folds <grid_search_folds>`.

The format of this file must be as follows: the first row must be a header.
This header row is ignored, so it doesn't matter what the header row contains,
but it must be there. If there is no header row, whatever row is in its place
will be ignored. The first column should consist of training set IDs and the
second should be a string for the fold ID (e.g., 1 through 5, A through D, etc.).
If specified, the CV and grid search will leave one fold ID out at a time. [#]_

.. _fixed_parameters:

fixed_parameters *(Optional)*
"""""""""""""""""""""""""""""

List of dictionaries containing parameters you want to have fixed for each
learner in :ref:`learners` list. Any empty ones will be ignored
(and the defaults will be used). If :ref:`grid_search` is ``True``,
there is a potential for conflict with specified/default parameter grids
and fixed parameters.

The default fixed parameters (beyond those that scikit-learn sets) are:

    AdaBoostClassifier and AdaBoostRegressor
      .. code-block:: python

        {'n_estimators': 500, 'random_state': 123456789}

    DecisionTreeClassifier and DecisionTreeRegressor
      .. code-block:: python

        {'random_state': 123456789}

    DummyClassifier
        .. code-block:: python
    
           {'random_state': 123456789}
    
    ElasticNet
        .. code-block:: python
    
           {'random_state': 123456789}
    
    GradientBoostingClassifier and GradientBoostingRegressor
        .. code-block:: python
    
           {'n_estimators': 500, 'random_state': 123456789}
    
    Lasso:
        .. code-block:: python
    
           {'random_state': 123456789}
    
    LinearSVC and LinearSVR
        .. code-block:: python
    
           {'random_state': 123456789}
    
    LogisticRegression
        .. code-block:: python
    
            {'max_iter': 1000, multi_class': 'auto', random_state': 123456789, 'solver': 'liblinear'}

        .. note:: The regularization ``penalty`` used by default is ``"l2"``. However, ``"l1"``, ``"elasticnet"``, and ``"none"`` (no regularization) are also available. There is a dependency between the ``penalty`` and the ``solver``. For example, the ``"elasticnet"`` penalty can *only* be used in conjunction with the ``"saga"`` solver. See more information in the ``scikit-learn`` documentation `here <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`__.

    MLPClassifier and MLPRegressor:
        .. code-block:: python
    
           {'learning_rate': 'invscaling', max_iter': 500}
    
    RandomForestClassifier and RandomForestRegressor
        .. code-block:: python
    
           {'n_estimators': 500, 'random_state': 123456789}
    
    RANSACRegressor
        .. code-block:: python
    
           {'loss': 'squared_loss', 'random_state': 123456789}
    
    Ridge and RidgeClassifier
        .. code-block:: python
    
           {'random_state': 123456789}
    
    SVC and SVR
        .. code-block:: python
    
           {'cache_size': 1000, 'gamma': 'scale'}
    
    SGDClassifier
        .. code-block:: python
    
           {'loss': 'log', 'max_iter': 1000, random_state': 123456789, 'tol': 1e-3}

    SGDRegressor
        .. code-block:: python
    
           {'max_iter': 1000, 'random_state': 123456789, 'tol': 1e-3}
    
    TheilSenRegressor
        .. code-block:: python
    
           {'random_state': 123456789}

    .. _imbalanced_data:

    .. note::

        The `fixed_parameters` field offers us a way to deal with imbalanced
        data sets by using the parameter ``class_weight`` for the following 
        classifiers: ``DecisionTreeClassifier``, ``LogisticRegression``, 
        ``LinearSVC``, ``RandomForestClassifier``, ``RidgeClassifier``, 
        ``SGDClassifier``, and ``SVC``.

        Two possible options are available. The first one is ``balanced``,
        which automatically adjust weights inversely proportional to class
        frequencies, as shown in the following code:

        .. code-block:: python

           {'class_weight': 'balanced'}

        The second option allows you to assign a specific weight per each
        class. The default weight per class is 1. For example:

        .. code-block:: python

           {'class_weight': {1: 10}}

        Additional examples and information can be seen `here <https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_weighted_samples.html>`__.

.. _hasher_features:

hasher_features *(Optional)*
""""""""""""""""""""""""""""

The number of features used by the `FeatureHasher <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html>`__ if the
:ref:`feature_hasher <feature_hasher>` flag is enabled.

.. note::

    To avoid collisions, you should always use the power of two larger than the
    number of features in the data set for this setting. For example, if you
    had 17 features, you would want to set the flag to 32.

.. _id_col:

id_col *(Optional)*
"""""""""""""""""""
If you're using :ref:`ARFF <arff>`, :ref:`CSV <csv>`, or :ref:`TSV <csv>`
files, the IDs for each instance are assumed to be in a column with this
name. If no column with this name is found, the IDs are generated
automatically. Defaults to ``id``.

.. _ids_to_floats:

ids_to_floats *(Optional)*
""""""""""""""""""""""""""

If you have a dataset with lots of examples, and your input files have IDs that
look like numbers (can be converted by float()), then setting this to True will
save you some memory by storing IDs as floats. Note that this will cause IDs to
be printed as floats in prediction files (e.g., ``4.0`` instead of ``4`` or
``0004`` or ``4.000``).

.. _label_col:

label_col *(Optional)*
""""""""""""""""""""""

If you're using :ref:`ARFF <arff>`, :ref:`CSV <csv>`, or :ref:`TSV <csv>`
files, the class labels for each instance are assumed to be in a column with
this name. If no column with this name is found, the data is assumed to be
unlabelled. Defaults to ``y``. For ARFF files only, this must also be the final
column to count as the label (for compatibility with Weka).

.. _learning_curve_cv_folds_list:

learning_curve_cv_folds_list *(Optional)*
""""""""""""""""""""""""""""""""""""""""""

List of integers specifying the number of folds to use for cross-validation
at each point of the learning curve (training size), one per learner. For
example, if you specify the following learners: ``["SVC", "LogisticRegression"]``,
specifying ``[10, 100]`` as the value of ``learning_curve_cv_folds_list`` will
tell SKLL to use 10 cross-validation folds at each point of the SVC curve and
100 cross-validation folds at each point of the logistic regression curve. Although
more folds will generally yield more reliable results, smaller number of folds
may be better for learners that are slow to train. Defaults to 10 for
each learner.

.. _learning_curve_train_sizes:

learning_curve_train_sizes *(Optional)*
""""""""""""""""""""""""""""""""""""""""""

List of floats or integers representing relative or absolute numbers
of training examples that will be used to generate the learning curve
respectively. If the type is float, it is regarded as a fraction of
the maximum size of the training set (that is determined by the selected
validation method), i.e. it has to be within (0, 1]. Otherwise it is
interpreted as absolute sizes of the training sets. Note that for classification
the number of samples usually have to be big enough to contain at least
one sample from each class. Defaults to ``[0.1, 0.325, 0.55, 0.775, 1.0]``.

.. _num_cv_folds:

num_cv_folds *(Optional)*
"""""""""""""""""""""""""

The number of folds to use for cross validation. Defaults to 10.

.. _shuffle:

.. _random_folds:

random_folds *(Optional)*
"""""""""""""""""""""""""

Whether to use random folds for cross-validation. Defaults to ``False``.

.. _sampler:

sampler *(Optional)*
""""""""""""""""""""

Whether to use a feature sampler that performs  non-linear transformations 
of the input, which can serve as a basis for linear classification 
or other algorithms. Valid options are:
`Nystroem <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem>`__,
`RBFSampler <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html#sklearn.kernel_approximation.RBFSampler>`__,
`SkewedChi2Sampler <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html#sklearn.kernel_approximation.SkewedChi2Sampler>`__, and
`AdditiveChi2Sampler <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.AdditiveChi2Sampler.html#sklearn.kernel_approximation.AdditiveChi2Sampler>`__.  For additional information see
`the scikit-learn documentation <https://scikit-learn.org/stable/modules/kernel_approximation.html>`__.

.. note:: Using a feature sampler with the ``MultinomialNB`` learner is not allowed
          since it cannot handle negative feature values.


.. _sampler_parameters:

sampler_parameters *(Optional)*
"""""""""""""""""""""""""""""""

dict containing parameters you want to have fixed for  the ``sampler``.
Any empty ones will be ignored (and the defaults will be used).

The default fixed parameters (beyond those that scikit-learn sets) are:

Nystroem
    .. code-block:: python

       {'random_state': 123456789}

RBFSampler
    .. code-block:: python

       {'random_state': 123456789}

SkewedChi2Sampler
    .. code-block:: python

       {'random_state': 123456789}

shuffle *(Optional)*
""""""""""""""""""""

If ``True``, shuffle the examples in the training data before using them for
learning. This happens automatically when doing a grid search but it might be
useful in other scenarios as well, e.g., online learning. Defaults to
``False``.

.. _suffix:

suffix *(Optional)*
"""""""""""""""""""

The file format the training/test files are in. Valid option are
:ref:`.arff <arff>`, :ref:`.csv <csv>`, :ref:`.jsonlines <ndj>`,
:ref:`.libsvm <libsvm>`, :ref:`.megam <megam>`, :ref:`.ndj <ndj>`, and
:ref:`.tsv <csv>`.

If you omit this field, it is assumed that the "prefixes" listed in
:ref:`featuresets <featuresets>` are actually complete filenames. This can be
useful if you have feature files that are all in different formats that you
would like to combine.

.. _test_file:

test_file *(Optional)*
""""""""""""""""""""""

Path to a file containing the features to test on.  Cannot be used in
combination with :ref:`featuresets <featuresets>`,
:ref:`train_directory <train_directory>`, or :ref:`test_directory <test_directory>`

.. _test_directory:

test_directory *(Optional)*
"""""""""""""""""""""""""""

Path to directory containing test data files. There must be a file
for each featureset.  Cannot be used in combination with
:ref:`train_file <train_file>` or :ref:`test_file <test_file>`.

.. _tuning:

Tuning
^^^^^^

Generally, in this section, you would specify fields that pertain to the
hyperparameter tuning for each learner. The most common required field
is :ref:`objectives` although it may also be optional in certain 
circumstances.

.. _objectives:

objectives 
""""""""""

A list of one or more metrics to use as objective functions for tuning the learner
hyperparameters via grid search. Note that ``objectives`` is required by default in most cases unless (a) :ref:`grid_search <grid_search>` is explicitly set to ``False`` or (b) the task is :ref:`learning_curve <learning_curve>`. For (a), any specified objectives are ignored. For (b), specifying objectives will raise an exception.

Available metrics are:

.. _classification_obj:

    **Classification:** The following objectives can be used for classification problems although some are restricted by problem type (binary/multiclass), types of labels (integers/floats/strings), and whether they are contiguous (if integers). Please read carefully.

    .. note:: When doing classification, SKLL internally sorts and maps all the class 
              labels in the data and maps them to integers which can be thought
              of class indices. This happens irrespective of the data type of the
              original labels. For example, if your data has the labels ``['A', 'B', 'C']``,
              SKLL will map them to the indices ``[0, 1, 2]`` respectively. It will do the
              same if you have integer labels (``[1, 2, 3]``) or floating point ones 
              (``[1.0, 1.1, 1.2]``). All of the tuning objectives are computed using
              these integer indices rather than the original class labels. This is why
              some metrics *only* make sense in certain scenarios. For example, SKLL
              only allows using weighted kappa metrics as tuning objectives if the original
              class labels are contiguous integers, e.g., ``[1, 2, 3]`` or ``[4, 5, 6]`` 
              -- or even integer-like floats (e,g., ``[1.0, 2.0, 3.0]``, but not 
              ``[1.0, 1.1, 1.2]``).


    *   **accuracy**: Overall `accuracy <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>`__ 
    *   **average_precision**: `Area under PR curve <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>`__ . To use this metric, :ref:`probability <probability>` must be set to ``True``. (*Binary classification only*).
    *   **balanced_accuracy**: A version of accuracy `specifically designed <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score>`__ for imbalanced binary and multi-class scenarios.
    *   **f1**: The default scikit-learn |F1 link|_
        (F\ :sub:`1` of the positive class for binary classification, or the weighted average F\ :sub:`1` for multiclass classification)
    *   **f1_score_micro**: Micro-averaged |F1 link|_
    *   **f1_score_macro**: Macro-averaged |F1 link|_
    *   **f1_score_weighted**: Weighted average |F1 link|_
    *   **f1_score_least_frequent**: F:\ :sub:`1` score of the least frequent
        class. The least frequent class may vary from fold to fold for certain
        data distributions.
    *   **kendall_tau**: `Kendall's tau <https://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient>`__ . For binary classification and with :ref:`probability <probability>` set to ``True``, the probabilities for the positive class will be used to compute the correlation values. In all other cases, the labels are used. (*Integer labels only*).
    *   **linear_weighted_kappa**: `Linear weighted kappa <http://www.vassarstats.net/kappaexp.html>`__. (*Contiguous integer labels only*).
    *   **lwk_off_by_one**: Same as ``linear_weighted_kappa``, but all
        ranking differences are discounted by one. (*Contiguous integer labels only*).
    *   **neg_log_loss**: The negative of the classification `log loss <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html>`__ . Since scikit-learn `recommends <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`__ using negated loss functions as scorer functions, SKLL does the same for the sake of consistency. To use this metric, :ref:`probability <probability>` must be set to ``True``.
    *   **pearson**: `Pearson correlation <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient>`__ . For binary classification and with :ref:`probability <probability>` set to ``True``, the probabilities for the positive class will be used to compute the correlation values. In all other cases, the labels are used. (*Integer labels only*). 
    *   **precision**: `Precision <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html>`__
    *   **quadratic_weighted_kappa**: `Quadratic weighted kappa <http://www.vassarstats.net/kappaexp.html>`__. (*Contiguous integer labels only*). 
    *   **qwk_off_by_one**: Same as ``quadratic_weighted_kappa``, but all
        ranking differences are discounted by one. (*Contiguous integer labels only*). 
    *   **recall**: `Recall <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html>`__
    *   **roc_auc**: `Area under ROC curve <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html>`__ .To use this metric, :ref:`probability <probability>` must be set to ``True``. (*Binary classification only*).
    *   **spearman**: `Spearman rank-correlation <https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient>`__. For binary classification and with :ref:`probability <probability>` set to ``True``, the probabilities for the positive class will be used to compute the correlation values. In all other cases, the labels are used. (*Integer labels only*).
    *   **unweighted_kappa**: Unweighted `Cohen's kappa <https://en.wikipedia.org/wiki/Cohen's_kappa>`__. 
    *   **uwk_off_by_one**: Same as ``unweighted_kappa``, but all ranking
        differences are discounted by one. In other words, a ranking of
        1 and a ranking of 2 would be considered equal. 

.. |F1 link| replace:: F\ :sub:`1` score
.. _F1 link: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    **Regression:** The following objectives can be used for regression problems. 

    *   **explained_variance**: A `score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score>`__ indicating how much of the variance in the given data can be by the model.
    *   **kendall_tau**: `Kendall's tau <https://en.wikipedia.org/wiki/Kendall_tau_rank_correlation_coefficient>`__ 
    *   **linear_weighted_kappa**: Linear weighted kappa (any floating point values are rounded to ints)
    *   **lwk_off_by_one**: Same as ``linear_weighted_kappa``, but all
        ranking differences are discounted by one.
    *   **max_error**: The `maximum residual error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error>`__.
    *   **neg_mean_absolute_error**: The negative of the `mean absolute error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error>`__ regression loss. Since scikit-learn `recommends <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`__ using negated loss functions as scorer functions, SKLL does the same for the sake of consistency.
    *   **neg_mean_squared_error**: The negative of the `mean squared error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`__ regression loss. Since scikit-learn `recommends <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>`__ using negated loss functions as scorer functions, SKLL does the same for the sake of consistency.
    *   **pearson**: `Pearson correlation <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient>`__
    *   **quadratic_weighted_kappa**: Quadratic weighted kappa (any floating point values are rounded to ints)
    *   **qwk_off_by_one**: Same as ``quadratic_weighted_kappa``, but all
        ranking differences are discounted by one.
    *   **r2**: `R2 <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`__
    *   **spearman**: `Spearman rank-correlation <https://en.wikipedia.org/wiki/Spearman's_rank_correlation_coefficient>`__
    *   **unweighted_kappa**: Unweighted `Cohen's kappa <https://en.wikipedia.org/wiki/Cohen's_kappa>`__ (any floating point values are rounded to ints)
    *   **uwk_off_by_one**: Same as ``unweighted_kappa``, but all ranking
        differences are discounted by one. In other words, a ranking of
        1 and a ranking of 2 would be considered equal.

The following is a list of the other optional fields in this section in alphabetical order.

.. _grid_search:

grid_search *(Optional)*
""""""""""""""""""""""""

Whether or not to perform grid search to find optimal parameters for
classifier. Defaults to ``True`` since optimizing model hyperparameters
almost always leads to better performance. Note that for the
:ref:`learning_curve <learning_curve>` task, grid search is not allowed
and setting it to ``True`` will generate a warning and be ignored.

.. note:: 

    1. In versions of SKLL before v2.0, this option was set to
       ``False`` by default but that was changed since the benefits
       of hyperparameter tuning significantly outweight the cost
       in terms of model fitting time. Instead, SKLL must explicly
       opt out of hyperparameter tuning if they so desire.

    2. Although SKLL only uses the combination of hyperparameters in
       the grid that maximizes the grid search objective, the results
       for all other points on the grid that were tried are also available.
       See the ``grid_search_cv_results`` attribute in the ``.results.json`` 
       file. 

.. _grid_search_folds:

grid_search_folds *(Optional)*
""""""""""""""""""""""""""""""

The number of folds to use for grid search. Defaults to 3.

.. _grid_search_jobs:

grid_search_jobs *(Optional)*
"""""""""""""""""""""""""""""

Number of folds to run in parallel when using grid search. Defaults to
number of grid search folds.

.. _min_feature_count:

min_feature_count *(Optional)*
""""""""""""""""""""""""""""""

The minimum number of examples for which the value of a feature must be nonzero
to be included in the model. Defaults to 1.

.. _param_grids:

param_grids *(Optional)*
""""""""""""""""""""""""

List of parameter grids to search for each learner. Each parameter
grid should be a list of dictionaries mapping from strings to lists
of parameter values. When you specify an empty list for a learner,
the default parameter grid for that learner will be searched.

The default parameter grids for each learner are:

    AdaBoostClassifier and AdaBoostRegressor
        .. code-block:: python
    
            [{'learning_rate': [0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    BayesianRidge
        .. code-block:: python
    
            [{'alpha_1': [1e-6, 1e-4, 1e-2, 1, 10],
              'alpha_2': [1e-6, 1e-4, 1e-2, 1, 10],
              'lambda_1': [1e-6, 1e-4, 1e-2, 1, 10],
              'lambda_2': [1e-6, 1e-4, 1e-2, 1, 10]}]
    
    DecisionTreeClassifier and DecisionTreeRegressor
        .. code-block:: python
    
           [{'max_features': ["auto", None]}]
    
    ElasticNet
        .. code-block:: python
    
           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    GradientBoostingClassifier and GradientBoostingRegressor
        .. code-block:: python
    
           [{'max_depth': [1, 3, 5]}]
    
    HuberRegressor
        .. code-block:: python
    
            [{'epsilon': [1.05, 1.35, 1.5, 2.0, 2.5, 5.0],
              'alpha': [1e-4, 1e-3, 1e-3, 1e-1, 1, 10, 100, 1000]}]
    
    KNeighborsClassifier and KNeighborsRegressor
        .. code-block:: python
    
            [{'n_neighbors': [1, 5, 10, 100],
              'weights': ['uniform', 'distance']}]
    
    Lasso
        .. code-block:: python
    
           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    LinearSVC
        .. code-block:: python
    
           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    LogisticRegression
        .. code-block:: python
    
           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    MLPClassifier and MLPRegressor:
        .. code-block:: python
    
           [{'activation': ['logistic', 'tanh', 'relu'],
             'alpha': [1e-4, 1e-3, 1e-3, 1e-1, 1],
             'learning_rate_init': [0.001, 0.01, 0.1]}],
    
    MultinomialNB
        .. code-block:: python
    
           [{'alpha': [0.1, 0.25, 0.5, 0.75, 1.0]}]
    
    RandomForestClassifier and RandomForestRegressor
        .. code-block:: python
    
           [{'max_depth': [1, 5, 10, None]}]
    
    Ridge and RidgeClassifier
        .. code-block:: python
    
           [{'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    SGDClassifier and SGDRegressor
        .. code-block:: python
    
            [{'alpha': [0.000001, 0.00001, 0.0001, 0.001, 0.01],
              'penalty': ['l1', 'l2', 'elasticnet']}]
    
    SVC
        .. code-block:: python
    
           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0],
             'gamma': ['auto', 0.01, 0.1, 1.0, 10.0, 100.0]}]
    
    SVR
        .. code-block:: python
    
           [{'C': [0.01, 0.1, 1.0, 10.0, 100.0]}]

    .. note::
           Note that learners not listed here do not have any default
           parameter grids in SKLL either because either there are no
           hyper-parameters to tune or decisions about which parameters
           to tune (and how) depend on the data being used for the
           experiment and are best left up to the user.


.. _pos_label_str:

pos_label_str *(Optional)*
""""""""""""""""""""""""""

A string denoting the label of the class to be
treated as the positive class in a binary classification
setting. If unspecified, the class represented by the label
that appears second when sorted is chosen as the positive
class. For example, if the two labels in data are "A" and
"B" and ``pos_label_str`` is not specified, "B" will be chosen
as the positive class.

.. _use_folds_file_for_grid_search:

use_folds_file_for_grid_search *(Optional)*
"""""""""""""""""""""""""""""""""""""""""""

Whether to use the specified :ref:`folds_file <folds_file>` for the inner grid-search
cross-validation loop when :ref:`task` is set to ``cross_validate``.
Defaults to ``True``.

.. note::

    This flag is ignored for all other tasks, including the
    ``train`` task where a specified :ref:`folds_file <folds_file>` is
    *always* used for the grid search.

.. _output:

Output
^^^^^^

The fields in this section generally pertain to the 
:ref:`output files<experiment_output_files>` produced
by the experiment. The most common fields are ``logs``, ``models``, 
``predictions``, and ``results``. These fields are mostly optional
although they may be required in certain cases. A common option 
is to use the same directory for all of these fields.

.. _log:

log *(Optional)*
""""""""""""""""

Directory to store SKLL :ref:`log files <output_log_files>` in. 
If omitted, the current working directory is used. 

.. _models:

models *(Optional)*
"""""""""""""""""""

Directory in which to store :ref:`trained models <output_model_files>`.
Can be omitted to not store models except when using the :ref:`train <train>`
task, where this path *must* be specified. On the other hand, this path must 
*not* be specified for the :ref:`learning_curve <learning_curve>` task.

.. _metrics:

metrics *(Optional)*
""""""""""""""""""""
For the ``evaluate`` and ``cross_validate`` tasks, this is an optional
list of additional metrics that will be computed *in addition to*
the tuning objectives and added to the results files. However, for the 
:ref:`learning_curve <learning_curve>` task, this list is **required**. 
Possible values are all of the same functions as those available for the 
:ref:`tuning objectives <objectives>`  (with the same caveats).

.. note::

    If the list of metrics overlaps with the grid search tuning 
    :ref:`objectives <objectives>`, then, for each job, the objective
    that overlaps is *not* computed again as a metric. Recall that
    each SKLL job can only contain a single tuning objective. Therefore,
    if, say, the ``objectives`` list is ``['accuracy', 'roc_auc']`` and the
    ``metrics`` list is ``['roc_auc', 'average_precision']``, then in the
    second job, ``roc_auc`` is used as the objective but *not* computed
    as an additional metric.


.. _pipeline:

pipeline *(Optional)*
"""""""""""""""""""""

Whether or not the final learner object should contain a ``pipeline``
attribute that contains a scikit-learn `Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`__ object composed
of copies of each of the following steps of training the learner:

    * feature vectorization (`vectorizer`)
    * feature selection (`selector`)
    * feature sampling (`sampler`)
    * feature scaling (`scaler`)
    * main estimator (`estimator`)

The strings in the parentheses represent the name given to each
step in the pipeline.

The goal of this attribute is to allow better interoperability
between SKLL learner objects and scikit-learn. The user can
train the model in SKLL and then further tweak or analyze
the pipeline in scikit-learn, if needed. Each component of the
pipeline is a (deep) copy of the component that was fit as part
of the SKLL model training process. We use copies since we do
not want the  original SKLL model to be affected if the user
modifies the components of the pipeline in scikit-learn space.

Here's an example of how to use this attribute.

.. code-block:: python

    from sklearn.preprocessing import LabelEncoder

    from skll import Learner
    from skll.data import Reader

    # train a classifier and a regressor using the SKLL API
    fs1 = Reader.for_path('examples/iris/train/example_iris_features.jsonlines').read()
    learner1 = Learner('LogisticRegression', pipeline=True)
    _ = learner1.train(fs1, grid_search=True, grid_objective='f1_score_macro')

    fs2 = Reader.for_path('examples/boston/train/example_boston_features.jsonlines').read()
    learner2 = Learner('RescaledSVR', feature_scaling='both', pipeline=True)
    _ = learner2.train(fs2, grid_search=True, grid_objective='pearson')

    # now, we can explore the stored pipelines in sklearn space
    enc = LabelEncoder().fit(fs1.labels)

    # first, the classifier
    D1 = {"f0": 6.1, "f1": 2.8, "f2": 4.7, "f3": 1.2}
    pipeline1 = learner1.pipeline
    enc.inverse_transform(pipeline1.predict(D1))

    # then, the regressor
    D2 = {"f0": 0.09178, "f1": 0.0, "f2": 4.05, "f3": 0.0, "f4": 0.51, "f5": 6.416, "f6": 84.1, "f7": 2.6463, "f8": 5.0, "f9": 296.0, "f10": 16.6, "f11": 395.5, "f12": 9.04}
    pipeline2 = learner2.pipeline
    pipeline2.predict(D2)

    # note that without the `pipeline` attribute, one would have to
    # do the following for D1, which is much less readable
    enc.inverse_transform(learner1.model.predict(learner1.scaler.transform(learner1.feat_selector.transform(learner1.feat_vectorizer.transform(D1)))))

.. note::
    1. When using a `DictVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`__ in SKLL along with :ref:`feature_scaling <feature_scaling>` set to either ``with_mean`` or ``both``, the `sparse` attribute of the vectorizer stage in the pipeline is set to ``False`` since centering requires dense arrays.
    2. When feature hashing is used (via a `FeatureHasher <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html>`__ ) in SKLL along with :ref:`feature_scaling <feature_scaling>` set to either ``with_mean`` or ``both`` , a custom pipeline stage (:py:mod:`skll.learner.Densifier`) is inserted in the pipeline between the feature vectorization (here, hashing) stage and the feature scaling stage. This is necessary since a ``FeatureHasher`` does not have a ``sparse`` attribute to turn off -- it *only* returns sparse vectors.
    3. A ``Densifier`` is also inserted in the pipeline when using a `SkewedChi2Sampler <https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.SkewedChi2Sampler.html>`__ for feature sampling since this sampler requires dense input and cannot be made to work with sparse arrays.

.. _predictions:

predictions *(Optional)*
""""""""""""""""""""""""

Directory in which to store :ref:`prediction files <output_prediction_files>`.
Can be omitted to not store predictions. Must *not* be specified for the 
:ref:`learning_curve <learning_curve>` and :ref:`train <train>` tasks.

.. _probability:

probability *(Optional)*
""""""""""""""""""""""""

Whether or not to output probabilities for each class instead of the
most probable class for each instance. Only really makes a difference
when storing predictions. Defaults to ``False``. Note that this also
applies to the tuning objective.

.. _results:

results *(Optional)*
""""""""""""""""""""

Directory in which to store :ref:`result files <output_results_files>`.
If omitted, the current working directory is used. 

.. _save_cv_folds:

save_cv_folds *(Optional)*
""""""""""""""""""""""""""

Whether to save the :ref:`folds file <output_folds_file>` containing the folds for a cross-validation experiment.
Defaults to ``False``.

.. _save_cv_models:

save_cv_models *(Optional)*
"""""""""""""""""""""""""""

Whether to save each of the K :ref:`model files <output_model_files>` trained during 
each step of a K-fold cross-validation experiment.
Defaults to ``False``.

.. _run_experiment:

Using run_experiment
--------------------
.. program:: run_experiment

Once you have created the :ref:`configuration file <create_config>` for your
experiment, you can usually just get your experiment started by running
``run_experiment CONFIGFILE``. [#]_ That said, there are a few options that are
specified via command-line arguments instead of in the configuration file:

.. option:: -a <num_features>, --ablation <num_features>

    Runs an ablation study where repeated experiments are conducted with the
    specified number of feature files in each featureset in the
    configuration file held out. For example, if you have three feature
    files (``A``, ``B``, and ``C``) in your featureset and you specifiy
    ``--ablation 1``, there will be three experiments conducted with
    the following featuresets: ``[[A, B], [B, C], [A, C]]``. Additionally,
    since every ablation experiment includes a run with all the features as a
    baseline, the following featureset will also be run: ``[[A, B, C]]``.

    If you would like to try all possible combinations of feature files, you
    can use the :option:`run_experiment --ablation_all` option instead.

    .. warning::

        Ablation will *not* work if you specify a :ref:`train_file <train_file>`
        and :ref:`test_file <test_file>` since no featuresets are defined in
        that scenario.

.. option:: -A, --ablation_all

    Runs an ablation study where repeated experiments are conducted with all
    combinations of feature files in each featureset.

    .. warning::

        This can create a huge number of jobs, so please use with caution.

.. option:: -k, --keep-models

    If trained models already exist for any of the learner/featureset
    combinations in your configuration file, just load those models and
    do not retrain/overwrite them.

.. option:: -r, --resume

    If result files already exist for an experiment, do not overwrite them.
    This is very useful when doing a large ablation experiment and part of
    it crashes.

.. option:: -v, --verbose

    Print more status information. For every additional time this flag is
    specified, output gets more verbose.

.. option:: --version

    Show program's version number and exit.

**GridMap options**

If you have `GridMap <https://pypi.org/project/gridmap/>`__ installed,
:program:`run_experiment` will automatically schedule jobs on your DRMAA-
compatible cluster. You can use the following options to customize this
behavior.

.. option:: -l, --local

    Run jobs locally instead of using the cluster. [#]_

.. option:: -q <queue>, --queue <queue>

    Use this queue for `GridMap <https://pypi.org/project/gridmap/>`__.
    (default: ``all.q``)

.. option:: -m <machines>, --machines <machines>

    Comma-separated list of machines to add to GridMap's whitelist.  If not
    specified, all available machines are used.

    .. note::

        Full names must be specified, (e.g., ``nlp.research.ets.org``).

.. _experiment_output_files:

Output files
------------

For most of the SKLL tasks the various output files generated by :ref:`run_experiment <run_experiment>` share the automatically generated prefix 
``<EXPERIMENT>_<FEATURESET>_<LEARNER>_<OBJECTIVE>``, where the following definitions hold:

    ``<EXPERIMENT>``
        The value of the as :ref:`experiment_name` field in the configuration file.

    ``<FEATURESET>``
        The components of the feature set that was used for training, joined with "+".

    ``<LEARNER>``
        The learner that was used to generate the current current results/model/etc. 

    ``<OBJECTIVE>``
        The objective function that was used to generate the current results/model/etc.

.. note:: 

    In SKLL terminology, a specific combination of featuresets, learners, 
    and objectives specified in the configuration file is called a ``job``.
    Therefore, an experiment (represented by a configuration file) can  
    contain multiple jobs.
    
    However, if the :ref:`objectives <objectives>` field in the configuration file
    contains only a single value, the job can be disambiguated using only
    the featuresets and the learners since the objective is fixed. Therefore,
    the output files will have the prefix ``<EXPERIMENT>_<FEATURESET>_<LEARNER>``.

The following types of output files can be generated after running an experiment
configuration file through :ref:`run_experiment <run_experiment>`. Note that
some file types may or may not be generated depending on the values of the fields
specified in the :ref:`Output section <output>` of the configuration file.

.. _output_log_files:

Log files
^^^^^^^^^

SKLL produces two types of log files -- one for each job in the experiment
and a single, top level log file for the entire experiment. Each of the job
log files have the usual job prefix as described above whereas the experiment
log file is simply named ``<EXPERIMENT>.log``.

While the job-level log files contain messages that pertain to the specific 
characteristics of the job (e.g., warnings from scikit-learn pertaining to
the specific learner), the experiment-level log file will contain logging
messages that pertain to the overall experiment and configuration file (e.g.,
an incorrect option specified in the configuration file). The  messages in all
SKLL log files are in the following format:

.. code-block:: bash

    <TIMESTAMP> - <LEVEL> - <MSG>

where ``<TIMESTAMP>`` refers to the exact time when the message was logged,
``<LEVEL>`` refers to the level of the logging message (e.g., ``INFO``, ``WARNING``,
etc.), and ``<MSG>`` is the actual content of the message. All of the messages
are also printed to the console in addition to being saved in the job-level log
files and the experiment-level log file.

.. _output_model_files:

Model files
^^^^^^^^^^^
Model files end in ``.model`` and are serialized :py:mod:`skll.learner.Learner`
instances. :ref:`run_experiment <run_experiment>` will re-use existing model
files if they exist, unless it is explicitly told not to. These model files
can also be loaded programmatically via the SKLL API, specifically the 
:py:mod:`skll.learner.Learner.from_file()` method.

.. _output_results_files:

Results files
^^^^^^^^^^^^^

SKLL generates two types of result files: 

1. Files ending in ``.results`` which contain a human-readable summary of the
   job, complete with confusion matrix, objective function score on the test set,
   and values of any additional metrics specified via the :ref:`metrics <metrics>`
   configuration file option. 

2. Files ending in ``.results.json``, which contain all of the same information as the
   ``.results`` files, but in a format more well-suited to automated processing. In
   some cases, ``.results.json`` files may contain *more* information than their
   ``.results`` file counterparts. For example, when doing :ref:`grid search <grid_search>`
   for tuning model hyperparameters, these files contain an additional attribute ``grid_search_cv_results`` containing detailed results from the grid search process.


.. _output_prediction_files:

Prediction files
^^^^^^^^^^^^^^^^

Predictions files are TSV files that contain either the predicted
values (for regression) OR predicted labels/class probabiltiies 
(for classification) for each instance in the test feature set. 
The value of the :ref:`probability <probability>` option decides whether SKLL
outputs the labels or the probabilities.

When the predictions are labels or values, there
are only two columns in the file: one containing the ID for the instance
and the other containing the prediction. The headers for the two columns
in this case are "id" and "prediction".

When the predictions are class probabilities, there are N+1 columns
in these files, where N are the number of classes in the training
data. The header for the column containing IDs is still "id" and the
labels themselves are the headers for the columns containing their
respective probabilities. In the special case of binary classification,
the :ref:`positive class <pos_label_str>` probabilities are always in
the last column.

.. _output_summary_file:

Summary file
^^^^^^^^^^^^

For every experiment you run, there will also be an experiment summary file
generated that is a tab-delimited file summarizing the results for each
job in the experiment. It is named ``<EXPERIMENT>_summary.tsv``. 
For :ref:`learning_curve <learning_curve>` experiments, this summary
file will contain training set sizes and the averaged scores for all
combinations of featuresets, learners, and objectives.

.. _output_folds_file:

Folds file
^^^^^^^^^^

For the :ref:`cross_validate <cross_validate>` task, SKLL can also output
the actual folds and instance IDs used in the cross-validation process, if
the :ref:`save_cv_folds <save_cv_folds>` option is enabled. In this case,
a file called ``<EXPERIMENT>_skll_fold_ids.csv`` is saved to disk.

.. _output_learning_curve_plots:

Learning curve plots
^^^^^^^^^^^^^^^^^^^^

If `seaborn <http://seaborn.pydata.org>`__ is available when running
a :ref:`learning_curve <learning_curve>` experiment,
actual learning curves are also generated as PNG files - one for each feature set
specified in the configuration file. Each PNG file is named ``EXPERIMENT_FEATURESET.png``
and contains a faceted learning curve plot for the featureset with objective
functions on rows and learners on columns. Here's an example of such a plot.

    .. image:: learning_curve.png

If you didn't have seaborn available when running the learning curve
experiment, you can always generate the plots later from the :ref:`summary
file <output_summary_file>` using the 
:ref:`plot_learning_curves <plot_learning_curves>` utility script.

.. rubric:: Footnotes

.. [#] We are considering adding support for YAML configuration files in the
   future, but we have not added this functionality yet.
.. [#] K-1 folds will be used for grid search within CV, so there should be at
   least 3 fold IDs.
.. [#] If you installed SKLL via pip on macOS, you might get an error when
   using ``run_experiment`` to generate learning curves. To get around this,
   add ``MPLBACKEND=Agg`` before the ``run_experiment`` command and re-run.
.. [#] This will happen automatically if GridMap cannot be imported.
