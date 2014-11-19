.. sectionauthor:: Dan Blanchard <dblanchard@ets.org>
.. sectionauthor:: Diane Napolitano <dnapolitano@ets.org>

Tutorial
========

For this tutorial, we're going to make use of the examples provided in the ``examples/`` directory in your copy of SKLL.  Of course, the provided examples are already perfect and ready to use.  If this weren't the case, you would need to...

Get your data into the correct format
-------------------------------------

SKLL can work with several common data formats, each of which are described :ref:`here <file_formats>`.

Should you need your data to be in any number of these formats, or should you like to have say, one ARFF file for use with SKLL and Weka, the :ref:`skll_convert` script can help you out.  This is as easy as:

.. code-block:: bash

    $ skll_convert examples/iris/train/example_iris_features.jsonlines examples/iris/train/example_iris_features.arff

``skll_convert`` can also create sparse data files, very useful for speeding up large experiments with lots of zero-valued features.

At minimum you will probably want to work with a training set and a testing set.  You will probably want to place these in **clearly-labeled** directories, since both files **need** to have the same name, and be in the same format, unless they are listed explicitly in your config file.  This will be discussed in the next section.

At this time, utilities necessary to split data into training and testing are not part of SKLL.


Create a configuration file for the experiment you'd like to run
----------------------------------------------------------------

For this tutorial, we will refer to an "experiment" as having a single data set split into (at minimum) training and testing portions.  As part of each experiment, we can train and test several models, either simultaneously or sequentially, depending on the availability of a grid engine.  This will be described in more detail later on, when we are ready to run our experiment.

You can consult :ref:`the full list of learners currently available in SKLL <learners>` to get an idea for the things you can do.  As part of this tutorial, we will use the following learners:

* Random Forest (``RandomForestClassifier``), C-Support Vector Classification (``SVC``), Linear Support Vector Classification (``LinearSVC``), Logistic Regression, Multinomial Naïve Bays (``MultinomialNB``) with the **Iris** example;
* Random Forest, Decision Tree, C-SVC, and Multinomial Naïve Bayes with the **Titanic** example;
* Random Forest Regression (``RandomForestRegressor``), Support Vector Regression (``SVR``), and Linear Regression with the **Boston** example.

Optionally, you can also tune your models in such a way that they maximize a certain evaluation metric, which we refer to as an "objective function".  The full list of available objective functions can be found :ref:`here <objective>`.

Configuration File for the Iris Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look at the options specified in ``iris/evaluate.cfg``.  Here, we are only going to train a model and evaluate its performance, because in the ``General`` section, ``task`` is set to "evaluate".  We will explore the other options for ``task`` later.

In the ``Input`` section, you may want to adjust ``train_location`` and ``test_location`` to point to the directories containing the Iris training and testing data (most likely ``skll/examples/iris/train`` and ``skll/examples/iris/test`` respectively, relative to your installation of SKLL).  ``featuresets`` indicates the name of both the training and testing files.  ``suffix`` is the suffix, indicating the type of file that both the training and testing sets are in.  ``learners`` must always be specified in between ``[`` ``]`` brackets, even if you only want to use one learner.  This is similar to the ``featuresets`` option, which requires two sets of brackets, since multiple sets of different-yet-similar features can be provided.  We will keep our examples simple, however, and only use one set of features per experiment.

The ``Tuning`` section defines how we want our model to be tuned.  Setting ``grid_search`` to "True" here employs scikit-learn's `GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV>`_ class, which is an implementation of the `standard, brute-force approach to hyperparameter optimization <http://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search>`_.  ``objective`` refers to the desired objective function; here, "f1_score_micro" will optimize for micro-averaged F1.

In the ``Output`` section, the arguments to each of these are directories you'd like all of the output of your experiment to go.  ``results`` refers to the results of the experiment, and ``log`` refers to log files containing any status, warning, or error messages generated during model training and evaluation.

This experiment is not configured to serialize the constructed models.

Configuration File for the Iris Cross-Validation Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``iris/cross_val.cfg`` almost identical to ``iris/evaluate.cfg``, except that it is configured to do three-fold cross-validation on the training set.  This is indicated by setting ``task`` to "cross_validate" in the ``General`` section.

Configuration File for the Titanic Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^