Quickstart
==========

Here is a quick run-down of how you accomplish common tasks.

Load a ``FeatureSet`` from a file::

    from skll.data import Reader

    example_reader = Reader.for_path('myexamples.csv')
    train_examples = example_reader.read()


Or, work with an existing ``pandas`` ``DataFrame``::

    from skll.data import FeatureSet

    # assuming the data labels are in a column called "y"
    train_examples = FeatureSet.from_data_frame(my_data_frame,
                                                "A Name for My Data",
                                                labels_column="y")


Train a linear svm (using the already loaded ``train_examples``)::

    from skll.learner import Learner

    learner = Learner('LinearSVC')
    learner.train(train_examples)


Evaluate a trained model::

    test_examples = Reader.for_path('test.tsv').read()
    conf_matrix, accuracy, prf_dict, model_params, obj_score = learner.evaluate(test_examples)


Perform ten-fold cross-validation with a radial SVM::

    learner = Learner('SVC')
    fold_result_list, grid_search_scores = learner.cross-validate(train_examples)

``fold_result_list`` in this case is a list of the results returned by
``learner.evaluate`` for each fold, and ``grid_search_scores`` is the highest
objective function value achieved when tuning the model.


Generate predictions from a trained model::

    predictions = learner.predict(test_examples)
