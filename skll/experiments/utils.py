# License: BSD 3 clause
"""
Utility classes and functions for running SKLL experiments.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Dan Blanchard (dblanchard@ets.org)
:author: Michael Heilman (mheilman@ets.org)
"""

import json
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from tabulate import tabulate  # type: ignore

from skll.types import EvaluateTaskResults, LabelType
from skll.utils.logging import get_skll_logger


class NumpyTypeEncoder(json.JSONEncoder):
    """
    Serialize results to JSON in a numpy-compatible way.

    This class is used when serializing results, particularly the input label
    values if the input has int-valued labels.  Numpy int64 objects can't
    be serialized by the json module, so we must convert them to int objects.

    A related issue where this was adapted from:
    https://stackoverflow.com/questions/11561932/why-does-json-dumpslistnp-arange5-fail-while-json-dumpsnp-arange5-tolis
    """

    def default(self, obj):
        """Encode given object."""
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class PipelineTypeEncoder(json.JSONEncoder):
    """Serialize ``sklearn.pipeline.Pipeline`` objects."""

    def default(self, obj):
        """Encode given pipeline."""
        if isinstance(obj, Pipeline):
            pipeline_steps = str(obj.named_steps)
            pipeline_steps = re.sub(r"\n", "", pipeline_steps)
            pipeline_steps = re.sub(r"\s+", " ", pipeline_steps)
            return pipeline_steps
        return json.JSONEncoder.default(self, obj)


def _check_job_results(job_results: List[List[Dict[str, Any]]]) -> None:
    """
    See if we have a complete results dictionary for every job.

    Parameters
    ----------
    job_results : List[List[Dict[str, Any]]]
        A list of job result dictionaries.

    """
    logger = get_skll_logger("experiment")
    logger.info("Checking job results")
    for result_dicts in job_results:
        if not result_dicts or "task" not in result_dicts[0]:
            logger.error("There was an error running the experiment:\n" f"{result_dicts}")


def _create_learner_result_dicts(
    task_results: List[EvaluateTaskResults],
    grid_scores: Union[List[None], List[float]],
    grid_search_cv_results_dicts: Union[List[None], List[Dict[str, Any]]],
    learner_result_dict_base: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Create result dictionaries used to write JSON and plain-text results.

    Parameters
    ----------
    task_results : List[:class:`skll.types.EvaluateTaskResults`]
        The task results list.
    grid_scores : Union[List[None], List[float]]
        List of grid scores or a list of ``None`` instances for tasks that do not
        involve grid search.
    grid_search_cv_results_dicts : Union[List[None], List[Dict[str, Any]]]
        A list of dictionaries of grid search CV results, one per fold,
        with keys such as "params", "mean_test_score", etc, that are
        mapped to values associated with each hyperparameter set
        combination. Or a list of ``None`` instances for tasks that do not
        involve any grid search.
    learner_result_dict_base : Dict[str, Any]
        Base dictionary for all learner results.

    Returns
    -------
    res : List[Dict[str, Any]]
        The results of the learners, as a list of dictionaries.

    """
    res = []

    num_folds = len(task_results)
    accuracy_sum = 0.0
    pearson_sum = 0.0
    additional_metric_score_sums: Dict[str, float] = {}
    score_sum = None
    prec_sum_dict: Dict[LabelType, float] = defaultdict(float)
    recall_sum_dict: Dict[LabelType, float] = defaultdict(float)
    f_sum_dict: Dict[LabelType, float] = defaultdict(float)
    result_table = None

    for k, (
        (conf_matrix, fold_accuracy, result_dict, model_params, score, additional_scores),
        grid_score,
        grid_search_cv_results,
    ) in enumerate(zip(task_results, grid_scores, grid_search_cv_results_dicts), start=1):
        # create a new dict for this fold
        learner_result_dict = {}
        learner_result_dict.update(learner_result_dict_base)

        # initialize some variables to blanks so that the
        # set of columns is fixed.
        learner_result_dict["result_table"] = ""
        learner_result_dict["accuracy"] = ""
        learner_result_dict["pearson"] = ""
        learner_result_dict["score"] = ""
        learner_result_dict["fold"] = ""

        if learner_result_dict_base["task"] == "cross_validate":
            learner_result_dict["fold"] = k

        # before we dump the model parameters to disk, we need to handle
        # certain ensemble meta-estimators that have a base estimator
        # since the estimator object is not JSON-serializable; for these
        # meta-estimators, we convert the "estimator" parameter back to
        # a string
        if (estimator := model_params.get("estimator")) and isinstance(estimator, BaseEstimator):
            model_params["estimator"] = estimator.__class__.__name__

        # include model parameters dump for regular learners only;
        # we need to use a special JSON encoder for voting learners
        # that contain ``Pipeline`` objects
        learner_result_dict["model_params"] = json.dumps(model_params, cls=PipelineTypeEncoder)
        if grid_score is not None:
            learner_result_dict["grid_score"] = grid_score
            learner_result_dict["grid_search_cv_results"] = grid_search_cv_results

        if conf_matrix:
            learner_result_dict["conf_matrix"] = conf_matrix
            labels = sorted(task_results[0][2].keys())
            headers = [""] + labels + ["Precision", "Recall", "F-measure"]
            rows = []
            label_metrics = {}
            for i, actual_label in enumerate(labels):
                conf_matrix_row = (
                    conf_matrix[i][:i] + [str([conf_matrix[i][i]])] + conf_matrix[i][i + 1 :]
                )
                label_prf_dict: Dict[str, float] = result_dict[actual_label]
                label_prec = _get_stat_float(label_prf_dict, "Precision")
                label_recall = _get_stat_float(label_prf_dict, "Recall")
                label_f = _get_stat_float(label_prf_dict, "F-measure")
                label_metrics[actual_label] = {
                    "precision": label_prec,
                    "recall": label_recall,
                    "f-measure": label_f,
                }
                if not math.isnan(label_prec):
                    prec_sum_dict[actual_label] += float(label_prec)
                if not math.isnan(label_recall):
                    recall_sum_dict[actual_label] += float(label_recall)
                if not math.isnan(label_f):
                    f_sum_dict[actual_label] += float(label_f)
                result_row = [actual_label] + conf_matrix_row + [label_prec, label_recall, label_f]
                rows.append(result_row)
            learner_result_dict["label_metrics"] = label_metrics
            result_table = tabulate(
                rows, headers=headers, stralign="right", floatfmt=".3f", tablefmt="grid"
            )
            result_table_str = f"{result_table}\n(row = reference; column = " "predicted)"
            learner_result_dict["result_table"] = result_table_str
            if isinstance(fold_accuracy, float):
                learner_result_dict["accuracy"] = fold_accuracy
                accuracy_sum += fold_accuracy

        # if there is no confusion matrix, then we must be dealing
        # with a regression model
        else:
            learner_result_dict.update(result_dict)  # type: ignore
            pearson_sum += float(learner_result_dict["pearson"])

        # get the scores for all the metrics and compute the sums
        if score is not None:
            if score_sum is None:
                score_sum = score
            else:
                score_sum += score
            learner_result_dict["score"] = score
        learner_result_dict["additional_scores"] = additional_scores
        for metric, score in additional_scores.items():
            if score is not None:
                additional_metric_score_sums[metric] = (
                    additional_metric_score_sums.get(metric, 0) + score
                )
        res.append(learner_result_dict)

    if num_folds > 1:
        learner_result_dict = {}
        learner_result_dict.update(learner_result_dict_base)

        learner_result_dict["fold"] = "average"

        if result_table:
            headers = ["Label", "Precision", "Recall", "F-measure"]
            rows = []
            for actual_label in labels:
                # Convert sums to means
                prec_mean = prec_sum_dict[actual_label] / num_folds
                recall_mean = recall_sum_dict[actual_label] / num_folds
                f_mean = f_sum_dict[actual_label] / num_folds
                rows.append([actual_label] + [prec_mean, recall_mean, f_mean])

            result_table = tabulate(rows, headers=headers, floatfmt=".3f", tablefmt="psql")
            learner_result_dict["result_table"] = str(result_table)
            learner_result_dict["accuracy"] = accuracy_sum / num_folds
        else:
            learner_result_dict["pearson"] = pearson_sum / num_folds

        if score_sum is not None:
            learner_result_dict["score"] = score_sum / num_folds
        scoredict = {}
        for metric, score_sum in additional_metric_score_sums.items():
            scoredict[metric] = score_sum / num_folds
        learner_result_dict["additional_scores"] = scoredict
        res.append(learner_result_dict)

    return res


def _get_stat_float(label_result_dict: Dict[str, float], stat: str) -> float:
    """
    Extract precision, recall, and f-score values from the given dictionary.

    Parameters
    ----------
    label_result_dict : Dict[str, float]
        Dictionary containing the stats to retrieve for a particular label.
    stat : str
        The statistic we're looking for in the dictionary.

    Returns
    -------
    float
        The value of the stat if it's in the dictionary, and NaN otherwise.

    """
    if stat in label_result_dict and label_result_dict[stat] is not None:
        return label_result_dict[stat]
    else:
        return float("nan")
