Step 2: Tutorial!
=================

For this tutorial, we're going to make use of the examples provided in the ``examples/`` directory in your copy of SKLL.  Of course, the provided examples are already perfect and ready to use.  If this weren't the case, you would need to...

1. Get your data into the correct format
----------------------------------------

SKLL can work with several common data formats, each of which are described :ref:`here <file_formats>`.

Should you need your data to be in any number of these formats, or should you like to have say, one ARFF file for use with SKLL and Weka, the :ref:`skll_convert` script can help you out.  This is as easy as::

    $ skll_convert examples/iris/train/example_iris_features.jsonlines examples/iris/train/example_iris_features.arff

``skll_convert`` can also create sparse data files, very useful for speeding up large experiments with lots of zero-valued features.

2. Create a configuration file for the experiment you'd like to run
-------------------------------------------------------------------

For this tutorial, we will refer to an "experiment" as having a single data set split into (at minimum) training and testing portions.  As part of each experiment, we can train and test several models, either simultaneously or sequentially, depending on the availability of a grid engine.  This will be described in more detail later on, when we are ready to run our experiment.

You can consult :ref:`the full list of learners currently available in SKLL <learners>` to get an idea for the things you can do.  As part of this tutorial, we will use the following learners:

* Random Forest (``RandomForestClassifier``), C-Support Vector Classification (``SVC``), Linear Support Vector Classification (``LinearSVC``), Logistic Regression, Multinomial Naïve Bays (``MultinomialNB``) with the **Iris** data set;
* Random Forest, Decision Tree, C-SVC, and Multinomial Naïve Bayes with the **Titanic** example;
* RandomForest Regression (``RandomForestRegressor``), Support Vector Regression (``SVR``), and Linear Regression with the **Boston** example.

