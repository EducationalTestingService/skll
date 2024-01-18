"""
Utility classes and functions for logging to Weights & Biases.

:author: Tamar Lavee (tlavee@ets.org)
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import wandb

from skll.config import _setup_config_parser


class WandbLogger:
    """Interface for Weights and Biases logging."""

    def __init__(self, wandb_credentials: Optional[Dict[str, str]], config_file_path: str):
        """
        Initialize the wandb_run if wandb_credentials are provided.

        Parameters
        ----------
        wandb_credentials : Optional[Dict[str, str]]
            A dictionary containing the W&B entity and project names that will be
            used to initialize the wandb run. If ``None``, logging to W&B will not be performed.
        config_file_path : str
            The path to this experiment's config file
        """
        self.wandb_run = None
        if wandb_credentials:
            self.wandb_run = wandb.init(
                project=wandb_credentials["wandb_project"],
                entity=wandb_credentials["wandb_entity"],
                config=self.get_config_dict(config_file_path),
            )

    def get_config_dict(self, config_file_path: str) -> Dict[str, Any]:
        """
        Load a configuration file into a dictionary, to be logged to W&B as a run config.

        Parameters
        ----------
        config_file_path : str
            Path to the config file

        Returns
        -------
        Dictionary containing all SKLL configuration fields.

        This also includes default values when for fields that are missing in the file.
        """
        config_parser = _setup_config_parser(config_file_path, validate=False)
        return {
            section: {key: val for [key, val] in config_parser.items(section)}
            for section in config_parser.sections()
        }

    def log_plot(self, plot_file_path: str) -> None:
        """
        Log a plot to W&B as an image if logging to W&B is enabled.

        Parameters
        ----------
        plot_file_path : str
            The full path to the plot file.
        """
        plot_name = Path(plot_file_path).stem
        if self.wandb_run:
            self.wandb_run.log({plot_name: wandb.Image(plot_file_path)})

    def log_evaluation_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log evaluation results to W&B, if logging to W&B is enabled.

        Log basic task info, general scores, and label-specific evaluation scores
        and confusion matrix for classification tasks.

        The input is either an "evaluate" task results or a single
        fold results in a "cross validate" task.

        The metrics are logged to a section named by the job name and
        fold, if available.

        Parameters
        ----------
        task_results : Dict[str,Any]
            The evaluation results of a single job of "evaluate" task or
            a single fold of a "cross_validate" task.
        """
        if self.wandb_run:
            metric_dict = {}
            task_prefix = task_results["job_name"]
            # if this is a fold's result, add fold name to the prefix.
            if task_results.get("fold"):
                task_prefix = f"{task_prefix}_fold_{task_results['fold']}"

            # log basic info and scores
            for metric in [
                "train_set_size",
                "test_set_size",
                "pearson",
                "model_params",
                "accuracy",
            ]:
                metric_dict[f"{task_prefix}/{metric}"] = task_results.get(metric, "N/A")

            # log confusion matrix as a custom chart
            if task_results.get("conf_matrix"):
                chart = self.generate_conf_matrix_chart(
                    task_results["conf_matrix"], sorted(task_results["label_metrics"].keys())
                )
                metric_dict[f"{task_prefix}/confusion_matrix"] = chart
                # log Precision, recall and f-measure for each label
                for label, label_metric_dict in task_results["label_metrics"].items():
                    for name, value in label_metric_dict.items():
                        metric_dict[f"{task_prefix}/label_{label}_{name}"] = value

            # log objective scores for train and test
            if task_results.get("grid_score"):
                metric_dict[f"{task_prefix}/grid_objective_score (train)"] = task_results[
                    "grid_score"
                ]
            if task_results.get("score"):
                metric_dict[f"{task_prefix}/objective_score (test)"] = task_results["score"]

            # log additional scores
            if task_results.get("additional_scores"):
                for metric, score in task_results["additional_scores"].items():
                    score = "" if np.isnan(score) else score
                    metric_dict[f"{task_prefix}/{metric}"] = score

            self.wandb_run.log(metric_dict)

    def log_learning_curve_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log learning curve results to W&B, if logging to W&B is enabled.

        Log basic task info as well as meand and stds of learning curve results.

        Parameters
        ----------
        task_results : Dict[str,Any]
            The learning curve task results.
        """
        if self.wandb_run:
            metric_dict = {}
            for metric in [
                "train_set_size",
                "test_set_size",
                "given_curve_train_sizes",
                "learning_curve_train_scores_means",
                "learning_curve_test_scores_means",
                "learning_curve_fit_times_means",
                "learning_curve_train_scores_stds",
                "learning_curve_test_scores_stds",
                "learning_curve_fit_times_stds",
                "computed_curve_train_sizes",
            ]:
                metric_dict[metric] = task_results.get(metric, "N/A")

        self.wandb_run.log(metric_dict)

    def log_train_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log train results to W&B, if logging to W&B is enabled.

        Log basic task info and the model file location.

        Parameters
        ----------
        task_results : Dict[str,Any]
            The learning curve task results.
        """
        if self.wandb_run:
            task_prefix = task_results["job_name"]
            # metric_dict = {}
            for metric in ["train_set_size", "test_set_size", "model_file"]:
                self.log_summary(task_prefix, metric, task_results.get(metric, "N/A"))
            # self.wandb_run.log(metric_dict)

    def log_predict_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log predict results to W&B, if logging to W&B is enabled.

        Log basic task info to the run, and predictions file as a table.

        Parameters
        ----------
        task_results : Dict[str,Any]
            The learning curve task results.
        """
        if self.wandb_run:
            task_prefix = task_results["job_name"]
            metric_dict = {}
            for metric in ["train_set_size", "test_set_size"]:
                metric_dict[f"{task_prefix}/{metric}"] = task_results.get(metric, "N/A")
            self.wandb_run.log(metric_dict)
            predictions_df = pd.read_csv(task_results["predictions_file"], sep="\t")
            predictions_table = wandb.Table(dataframe=predictions_df, allow_mixed_types=True)
            self.wandb_run.log({f"{task_prefix}/predictions": predictions_table})

    def generate_conf_matrix_chart(self, confusion_matrix, labels) -> wandb.Visualize:
        """
        Generate a wandb chart object from confusion matrix data.

        Assumes the input task results contains a confusion matrix.

        Parameters
        ----------
        confusion_matrix : List[List[int]]
            the confusion matrix values
        labels : List[str]
            label names

        Returns
        -------
        wandb.Visualize
            a Visualize object of the confusion matrix chart

        """
        conf_matrix_data = []
        for i, row in enumerate(confusion_matrix):
            for j, val in enumerate(row):
                conf_matrix_data.append([labels[i], labels[j], val])
        table = wandb.Table(columns=["Predicted", "Actual", "Count"], data=conf_matrix_data)
        chart = wandb.visualize("wandb/confusion_matrix/v1", table)
        return chart

    def log_summary(self, task_prefix, metric_name, metric_value) -> None:
        """
        Add a metric to the W&B run summary if logging to W&B is enabled".

        Parameters
        ----------
        task_prefix : str
            task name to be used as section in the summary table
        metric_name : str
            The metric name
        metric_value : Any
            The metric value
        """
        if self.wandb_run:
            self.wandb_run.summary[f"{task_prefix}/{metric_name}"] = metric_value
