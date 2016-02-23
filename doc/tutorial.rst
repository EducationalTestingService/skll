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
4.  Examine results

Titanic Example
---------------

Let's see how we can apply the basic workflow above to a simple example using
the `Titantic: Machine Learning from Disaster <http://www.kaggle.com/c/titanic-gettingStarted/>`__
data from `Kaggle <http://www.kaggle.com>`__.

Get your data into the correct format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step to getting the Titanic data is logging into Kaggle and
downloading `train.csv <http://www.kaggle.com/c/titanic-gettingStarted/download/train.csv>`__
and `test.csv <http://www.kaggle.com/c/titanic-gettingStarted/download/test.csv>`__.
Once you have those files, you'll also want to grab the
`examples folder <https://github.com/EducationalTestingService/skll/tree/stable/examples>`__
on our GitHub page and put ``train.csv`` and ``test.csv`` in ``examples``.

The provided script, :download:`make_titanic_example_data.py <../examples/make_titanic_example_data.py>`,
will split the training and test data files from Kaggle up into groups
of related features and store them in ``dev``, ``test``, ``train``, and
``train+dev`` subdirectories.  The development set that gets created by the
script is 20% of the data that was in the original training set, and ``train``
contains the other 80%.

Create a configuration file for the experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For this tutorial, we will refer to an "experiment" as having a single data set
split into training and testing portions.  As part of each
experiment, we can train and test several models, either simultaneously or
sequentially, depending whether we're using
`GridMap <https://github.com/EducationalTestingService/gridmap>`__ or not.
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
`GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV>`_
class, which is an implementation of the
`standard, brute-force approach to hyperparameter optimization <http://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search>`_.

:ref:`objectives <objectives>` refers to the desired objective functions; here,
``accuracy`` will optimize for overall accuracy.  You can see a list of all the 
available objective functions :ref:`here <objectives>`.

In the :ref:`Output` section, the arguments to each of these are directories
where you'd like all of the relevant output from your experiment to go.
:ref:`results <results>` refers to the results of the experiment in both
human-readable and JSON forms.  :ref:`log <log>` specifies where to put log
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

    Loading train/family.csv...           done
    Loading train/misc.csv...           done
    Loading train/socioeconomic.csv...           done
    Loading train/vitals.csv...           done
    Loading dev/family.csv...           done
    Loading dev/misc.csv...           done
    Loading dev/socioeconomic.csv...           done
    Loading dev/vitals.csv...           done
    2014-11-21 22:58:36,056 - skll.learner - WARNING - Training data will be shuffled to randomize grid search folds.  Shuffling may yield different results compared to scikit-learn.
    Loading train/family.csv...           done
    Loading train/misc.csv...           done
    Loading train/socioeconomic.csv...           done
    Loading train/vitals.csv...           done
    Loading dev/family.csv...           done
    Loading dev/misc.csv...           done
    Loading dev/socioeconomic.csv...           done
    Loading dev/vitals.csv...           done
    2014-11-21 22:58:40,180 - skll.learner - WARNING - Training data will be shuffled to randomize grid search folds.  Shuffling may yield different results compared to scikit-learn.
    Loading train/family.csv...           done
    Loading train/misc.csv...           done
    Loading train/socioeconomic.csv...           done
    Loading train/vitals.csv...           done
    Loading dev/family.csv...           done
    Loading dev/misc.csv...           done
    Loading dev/socioeconomic.csv...           done
    Loading dev/vitals.csv...           done
    2014-11-21 22:58:40,430 - skll.learner - WARNING - Training data will be shuffled to randomize grid search folds.  Shuffling may yield different results compared to scikit-learn.
    Loading train/family.csv...           done
    Loading train/misc.csv...           done
    Loading train/socioeconomic.csv...           done
    Loading train/vitals.csv...           done
    Loading dev/family.csv...           done
    Loading dev/misc.csv...           done
    Loading dev/socioeconomic.csv...           done
    Loading dev/vitals.csv...           done
    2014-11-21 22:58:41,132 - skll.learner - WARNING - Training data will be shuffled to randomize grid search folds.  Shuffling may yield different results compared to scikit-learn.

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

An example of a human-readable results file for our Titanic config file is::

    Experiment Name: Titanic_Evaluate_Tuned
    SKLL Version: 1.0.0
    Training Set: train
    Training Set Size: 712
    Test Set: dev
    Test Set Size: 179
    Shuffle: False
    Feature Set: ["family.csv", "misc.csv", "socioeconomic.csv", "vitals.csv"]
    Learner: RandomForestClassifier
    Task: evaluate
    Feature Scaling: none
    Grid Search: True
    Grid Search Folds: 3
    Grid Objective Function: accuracy
    Using Folds File: False
    Scikit-learn Version: 0.15.2
    Start Timestamp: 21 Nov 2014 22:58:35.940243
    End Timestamp: 21 Nov 2014 22:58:40.072254
    Total Time: 0:00:04.132011


    Fold:
    Model Parameters: {"max_depth": 10, "compute_importances": null, "min_density": null, "bootstrap": true, "n_estimators": 500, "verbose": 0, "min_samples_split": 2, "max_features": "auto", "min_samples_leaf": 1, "criterion": "gini", "random_state": 123456789, "max_leaf_nodes": null, "n_jobs": 1, "oob_score": false}
    Grid Objective Score (Train) = 0.8089887640449438
    +---+-------+------+-----------+--------+-----------+
    |   |     0 |    1 | Precision | Recall | F-measure |
    +---+-------+------+-----------+--------+-----------+
    | 0 | [101] |   14 |     0.871 |  0.878 |     0.874 |
    +---+-------+------+-----------+--------+-----------+
    | 1 |    15 | [49] |     0.778 |  0.766 |     0.772 |
    +---+-------+------+-----------+--------+-----------+
    (row = reference; column = predicted)
    Accuracy = 0.8379888268156425
    Objective Function Score (Test) = 0.8379888268156425


Running your own experiments
----------------------------

Once you've gone through the Titanic example, you will hopefully be interested
in trying out SKLL with your own data.  To do so, you'll still need to get your
data in an appropriate format first.

Get your data into the correct format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Supported formats
"""""""""""""""""

SKLL can work with several common data formats, each of which are described
:ref:`here <file_formats>`.

If you need to convert between any of the supported formats, because, for
example, you would like to create a single data file that will work both with
SKLL and Weka (or some other external tool), the :ref:`skll_convert` script can
help you out.  It is as easy as:

.. code-block:: bash

    $ skll_convert examples/titanic/train/family.csv examples/titanic/train/family.arff

Creating sparse files
"""""""""""""""""""""

:ref:`skll_convert` can also create sparse data files in
:ref:`.jsonlines <ndj>`, :ref:`.libsvm <libsvm>`, :ref:`.megam <megam>`, or
:ref:`.ndj <ndj>` formats.  This is very useful for saving disk space and
memory when you have a large data set with mostly zero-valued features.

Training and testing directories
""""""""""""""""""""""""""""""""

At minimum you will probably want to work with a training set and a testing
set.  If you have multiple feature files that you would like SKLL to join together
for you automatically, you will need to create feature files with the exact
same names and store them in training and testing directories.  You can
specifiy these directories in your config file using
:ref:`train_directory <train_directory>` and
:ref:`test_directory <test_directory>`.  The list of files is specified using
the :ref:`featuresets <featuresets>` setting.

Single-file training and testing sets
"""""""""""""""""""""""""""""""""""""

If you're conducting a simpler experiment, where you have a single training
file with all of your features and a similar single testing file, you should
use the :ref:`train_file <train_file>` and :ref:`test_file <test_file>`
settings in your config file.

If you would like to split an existing file up into a training set and a testing
set, you can employ the :ref:`filter_features` tool to select instances you
would like to include in each file.

Creating a configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that you've seen a :ref:`basic configuration file <titanic_config>`, you
should look at the extensive option available in our
:ref:`config file reference <create_config>`.

Running your configuration file through :ref:`run_experiment <run_experiment>`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few meta-options for experiments that are specified directly to the
:ref:`run_experiment <run_experiment>` command rather than in a configuration
file.  For example, if you would like to run an ablation experiment, which
conducts repeated experiments using different combinations of the features in
your config, you should use the :option:`run_experiment --ablation` option. A
complete list of options is available :ref:`here <run_experiment>`.
