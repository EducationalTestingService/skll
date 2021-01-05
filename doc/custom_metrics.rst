.. _custom_metrics:

Using Custom Metrics
====================

Although SKLL comes with a huge number of built-in metrics for both classification and regression,
there might be occasions when you want to use a custom metric function for hyper-parameter
tuning or for evaluation. This section shows you how to do that.

Writing Custom Metric Functions
-------------------------------

First, let's look at how to write valid custom metric functions. A valid custom metric function
must take two array-like positional arguments: the first being the true labels or scores, and the
second being the predicted labels or scores. This function can also take three optional keyword arguments:

1. ``greater_is_better``: a boolean keyword argument that indicates whether a higher value of the metric indicates better performance (``True``) or vice versa (``False``). The default value is ``True``.
2. ``needs_proba``: a boolean keyword argument that indicates whether the metric function requires probability estimates. The default value is ``False``.
3. ``needs_threshold``: a boolean keyword argument that indicates whether the metric function takes a continuous decision certainty. The default value is ``False``.

Note that these keyword arguments are identical to the keyword arguments for the `sklearn.metrics.make_scorer() <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer>`_ function and serve the same purpose.

In short, custom metric functions take two required positional arguments (order matters) and three optional keyword arguments. Here's a simple example of a custom metric function: F\ :sub:`β` with β=0.75 defined in a file called ``custom.py``.

.. code-block:: python
   :caption: custom.py

    from sklearn.metrics import fbeta_score

    def f075(y_true, y_pred):
        return fbeta_score(y_true, y_pred, beta=0.75)


Obviously, you may write much more complex functions that aren't directly
available in scikit-learn. Once you have written your metric function, the next
step is to use it in your SKLL experiment.

Using in Configuration Files
----------------------------

The first way of using custom metric functions is via your SKLL experiment
configuration file if you are running SKLL via the command line. To do so:

1. Add a field called :ref:`custom_metric_path <custom_metric_path>` in the
   :ref:`Input <input>` section of your configuration file and set its value to be the path to the ``.py`` file containing your custom metric function.
2. Add the name of your custom metric function to either the :ref:`objectives`
   field in the :ref:`Tuning <tuning>` section (if you wish to use it to tune the model hyper-parameters) or to the :ref:`metrics <metrics>` field in
   the :ref:`Output <output>` section if you wish to only use it for evaluation. You can also add it to both.

Here's an example configuration file using data from the
:ref:`SKLL Titanic example <titanic_example>` that illustrates this. This file
assumes that the file ``custom.py`` above is located in the same directory.

.. code-block:: cfg

   [General]
   experiment_name = titanic
   task = evaluate

   [Input]
   train_directory = train
   test_directory = dev
   featuresets = [["family.csv", "misc.csv", "socioeconomic.csv", "vitals.csv"]]
   learners = ["RandomForestClassifier", "DecisionTreeClassifier", "SVC", "MultinomialNB"]
   label_col = Survived
   id_col = PassengerId
   custom_metric_path = custom.py

   [Tuning]
   grid_search = true
   objectives = ['f075']

   [Output]
   metrics = ['roc_auc']
   probability = true
   log = output
   results = output
   predictions = output
   models = output


And that's it! SKLL will dynamically load and use your custom metric function when you :ref:`run your experiment <run_experiment>`. Custom metric functions can be used for both hyper-parameter tuning and for evaluation.

Using via the API
-----------------

To use a custom metric function via the SKLL API, you first need to register
the custom metric function using the ``register_custom_metric()`` function and
then just use the metric name either as a grid search objective, an output
metric, or both.

Here's a short example that shows how to use the ``f075()`` custom metric
function we defined above via the SKLL API. Again, we assume that ``custom.py``
is located in the current directory.

.. code-block:: python

    from skll import Learner
    from skll.data import CSVReader
    from skll.metrics import register_custom_metric

    # register the custom function with SKLL
    _ = register_custom_metric("custom.py", "f075")

    # let's assume the training data lives in a file called "train.csv"
    # we load that into a SKLL FeatureSet
    fs = CSVReader.for_path("train.csv").read()

    # instantiate a learner and tune its parameters using the custom metric
    learner = Learner('LogisticRegression')
    learner.train(fs, grid_objective="f075")

    ...

As with configuration files, custom metric functions can be used for
both training as well as evaluation with the API.

.. note::

    1. When using the API, if you have multiple metric functions defined in a
       Python source file, you must register each one individually using
       ``register_custom_metric()``.
    2. When using the API, if you try to re-register the same metric in the
       same Python session, it will raise a ``NameError``. Therefore, if you
       edit your custom metric, you must start a new Python session to be able
       to see the changes.

.. warning::

    1. When using a configuration file or the API, if the names of any of your
       custom metric functions conflict with names of :ref:`metrics <objectives>`
       that already exist in either SKLL or scikit-learn, it will raise a
       ``NameError``. You should rename the metric function in that case.
    2. Unlike for the built-in metrics, SKLL does not check whether your custom
       metric function is appropriate for classification or regression. You
       must make that decision for yourself.
