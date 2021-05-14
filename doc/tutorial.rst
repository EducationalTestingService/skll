.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>
.. sectionauthor:: Diane Napolitano <dnapolitano@ets.org>

Tutorial
========

Before doing anything below, you'll want to :ref:`install SKLL <install>`.

Workflow
--------

In general, there are four steps to using SKLL:

1.  Get some data in a :ref:`SKLL-compatible format <file_formats>`.
2.  Create a small :ref:`configuration file <create_config>` describing the
    machine learning experiment you would like to run.
3.  Run that configuration file with :ref:`run_experiment <run_experiment>`.
4.  Examine the results of the experiment.

.. _titanic_example:

Titanic Example
---------------

Let's see how we can apply the basic workflow above to a simple example using
the `Titantic: Machine Learning from Disaster <https://www.kaggle.com/c/titanic/>`__
data from `Kaggle <https://www.kaggle.com>`__.

Get your data into the correct format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step is to get the Titanic data. We have already downloaded the data files
from Kaggle and included them in the `SKLL repository <../examples/titanic>`. Next, we need to get the files
and process them to get them in the right shape.

The provided script, :download:`make_titanic_example_data.py <../examples/make_titanic_example_data.py>`, will split the train and test data files
from Kaggle up into groups of related features and store them in
``dev``, ``test``, ``train``, and ``train+dev`` subdirectories.
The development set that gets created by the script is 20% of the data
that was in the original training set, and ``train`` contains the other 80%.

Create a configuration file for the experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial, we will refer to an "experiment" as having a single data set
split into training and testing portions.  As part of each
experiment, we can train and test several models, either simultaneously or
sequentially, depending whether we're using
`GridMap <https://pypi.org/project/gridmap/>`__ or not.
This will be described in more detail later on, when we are ready to run our
experiment.

You can consult the :ref:`full list of learners currently available <learners>`
in SKLL to get an idea for the things you can do.  As part of this tutorial, we
will use the following classifiers:

*   Decision Tree
*   Multinomial Naïve Bayes
*   Random Forest
*   Support Vector Machine

.. _titanic_config:

.. literalinclude:: ../examples/titanic/evaluate_tuned.cfg
    :language: ini

Let's take a look at the options specified in ``titanic/evaluate_tuned.cfg``.
Here, we are only going to train a model and evaluate its performance on the
development set, because in the :ref:`General` section, :ref:`task` is set to
:ref:`evaluate <evaluate>`.  We will explore the other options for :ref:`task`
later.

In the :ref:`Input` section, we have specified relative paths to the training
and testing directories via the :ref:`train_directory <train_directory>` and
:ref:`test_directory <test_directory>` settings respectively.
:ref:`featuresets <featuresets>` indicates the name of both the training and
testing files.  :ref:`learners` must always be specified in between ``[`` ``]``
brackets, even if you only want to use one learner.  This is similar to the
:ref:`featuresets <featuresets>` option, which requires two sets of brackets,
since multiple sets of different-yet-related features can be provided.  We will
keep our examples simple, however, and only use one set of features per
experiment. The :ref:`label_col <label_col>` and :ref:`id_col <id_col>`
settings specify the columns in the CSV files that specify the class labels and
instances IDs for each example.

The :ref:`Tuning` section defines how we want our model to be tuned.  Setting
:ref:`grid_search <grid_search>` to ``True`` here employs scikit-learn's
`GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>`_
class, which is an implementation of the
`standard, brute-force approach to hyperparameter optimization <https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search>`_.

:ref:`objectives <objectives>` refers to the desired objective functions; here,
``accuracy`` will optimize for overall accuracy.  You can see a list of all the
available objective functions :ref:`here <objectives>`.

In the :ref:`Output` section, we first define the additional evaluation
metrics we want to compute in addition to the tuning objective via the
:ref:`metrics <metrics>` option. The other options are directories
where you'd like all of the relevant output from your experiment to go.
:ref:`results <results>` refers to the results of the experiment in both
human-readable and JSON forms.  :ref:`logs <logs>` specifies where to put log
files containing any status, warning, or error messages generated during
model training and evaluation.  :ref:`predictions <predictions>` refers to
where to store the individual predictions generated for the test set.
:ref:`models <models>` is for specifying a directory to serialize the trained
models.

Running your configuration file through :ref:`run_experiment <run_experiment>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Getting your experiment running is the simplest part of using SKLL, you just
need to type the following into a terminal:

.. code-block:: bash

    $ run_experiment titanic/evaluate_tuned.cfg

That should produce output like::

    2020-03-10 14:25:23,596 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - Task: evaluate
    2020-03-10 14:25:23,596 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - Training on train, Test on dev, feature set ['family.csv', 'misc.csv', 'socioeconomic.csv', 'vitals.csv'] ...
    Loading /Users/nmadnani/work/skll/examples/titanic/train/family.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/train/misc.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/train/socioeconomic.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/train/vitals.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/family.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/misc.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/socioeconomic.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/vitals.csv...           done
    2020-03-10 14:25:23,662 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - Featurizing and training new RandomForestClassifier model
    2020-03-10 14:25:23,663 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - WARNING - Training data will be shuffled to randomize grid search folds.  Shuffling may yield different results compared to scikit-learn.
    2020-03-10 14:25:28,129 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - Best accuracy grid search score: 0.798
    2020-03-10 14:25:28,130 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - Hyperparameters: bootstrap: True, ccp_alpha: 0.0, class_weight: None, criterion: gini, max_depth: 5, max_features: auto, max_leaf_nodes: None, max_samples: None, min_impurity_decrease: 0.0, min_impurity_split: None, min_samples_leaf: 1, min_samples_split: 2, min_weight_fraction_leaf: 0.0, n_estimators: 500, n_jobs: None, oob_score: False, random_state: 123456789, verbose: 0, warm_start: False
    2020-03-10 14:25:28,130 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - Evaluating predictions
    2020-03-10 14:25:28,172 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_RandomForestClassifier - INFO - using probabilities for the positive class to compute "roc_auc" for evaluation.
    2020-03-10 14:25:28,178 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - Task: evaluate
    2020-03-10 14:25:28,178 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - Training on train, Test on dev, feature set ['family.csv', 'misc.csv', 'socioeconomic.csv', 'vitals.csv'] ...
    Loading /Users/nmadnani/work/skll/examples/titanic/train/family.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/train/misc.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/train/socioeconomic.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/train/vitals.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/family.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/misc.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/socioeconomic.csv...           done
    Loading /Users/nmadnani/work/skll/examples/titanic/dev/vitals.csv...           done
    2020-03-10 14:25:28,226 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - Featurizing and training new DecisionTreeClassifier model
    2020-03-10 14:25:28,226 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - WARNING - Training data will be shuffled to randomize grid search folds.  Shuffling may yield different results compared to scikit-learn.
    2020-03-10 14:25:28,269 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - Best accuracy grid search score: 0.754
    2020-03-10 14:25:28,269 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - Hyperparameters: ccp_alpha: 0.0, class_weight: None, criterion: gini, max_depth: None, max_features: None, max_leaf_nodes: None, min_impurity_decrease: 0.0, min_impurity_split: None, min_samples_leaf: 1, min_samples_split: 2, min_weight_fraction_leaf: 0.0, presort: deprecated, random_state: 123456789, splitter: best
    2020-03-10 14:25:28,269 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - Evaluating predictions
    2020-03-10 14:25:28,272 - Titanic_Evaluate_Tuned_family.csv+misc.csv+socioeconomic.csv+vitals.csv_DecisionTreeClassifier - INFO - using probabilities for the positive class to compute "roc_auc" for evaluation.

We could squelch the warnings about shuffling by setting
:ref:`shuffle <shuffle>` to ``True`` in the :ref:`Input` section.

The reason we see the loading messages repeated is that we are running the
different learners sequentially, whereas SKLL is designed to take advantage
of a cluster to execute everything in parallel via GridMap.


Examine the results
^^^^^^^^^^^^^^^^^^^

As a result of running our experiment, there will be a whole host of files in
our :ref:`results <results>` directory.  They can be broken down into three
types of files:

1.  ``.results`` files, which contain a human-readable summary of the
    experiment, complete with confusion matrix.
2.  ``.results.json`` files, which contain all of the same information as the
    ``.results`` files, but in a format more well-suited to automated
    processing.
3.  A summary ``.tsv`` file, which contains all of the information in all of
    the ``.results.json`` files with one line per file.  This is very nice if
    you're trying many different learners and want to compare their performance.
    If you do additional experiments later (with a different config file), but
    would like one giant summary file, you can use the :ref:`summarize_results`
    command.

An example of a human-readable results file for our Titanic experiment is::

    Experiment Name: Titanic_Evaluate_Tuned
    SKLL Version: 2.1
    Training Set: train
    Training Set Size: 569
    Test Set: dev
    Test Set Size: 143
    Shuffle: False
    Feature Set: ["family.csv", "misc.csv", "socioeconomic.csv", "vitals.csv"]
    Learner: RandomForestClassifier
    Task: evaluate
    Feature Scaling: none
    Grid Search: True
    Grid Search Folds: 3
    Grid Objective Function: accuracy
    Additional Evaluation Metrics: ['roc_auc']
    Scikit-learn Version: 0.22.2.post1
    Start Timestamp: 10 Mar 2020 14:25:23.595787
    End Timestamp: 10 Mar 2020 14:25:28.175375
    Total Time: 0:00:04.579588


    Fold:
    Model Parameters: {"bootstrap": true, "ccp_alpha": 0.0, "class_weight": null, "criterion": "gini", "max_depth": 5, "max_features": "auto", "max_leaf_nodes": null, "max_samples": null, "min_impurity_decrease": 0.0, "min_impurity_split": null, "min_samples_leaf": 1, "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 500, "n_jobs": null, "oob_score": false, "random_state": 123456789, "verbose": 0, "warm_start": false}
    Grid Objective Score (Train) = 0.797874315418175
    +----+------+------+-------------+----------+-------------+
    |    |    0 |    1 |   Precision |   Recall |   F-measure |
    +====+======+======+=============+==========+=============+
    |  0 | [79] |    8 |       0.849 |    0.908 |       0.878 |
    +----+------+------+-------------+----------+-------------+
    |  1 |   14 | [42] |       0.840 |    0.750 |       0.792 |
    +----+------+------+-------------+----------+-------------+
    (row = reference; column = predicted)
    Accuracy = 0.8461538461538461
    Objective Function Score (Test) = 0.8461538461538461

    Additional Evaluation Metrics (Test):
     roc_auc = 0.9224137931034483

IRIS Example on Binder
----------------------
If you prefer using an interactive Jupyter notebook to learn about SKLL, you can do so by clicking the launch button below.

.. image:: https://static.mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/EducationalTestingService/skll/main?filepath=examples%2FTutorial.ipynb
