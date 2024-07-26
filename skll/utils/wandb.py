"""
Utility classes and functions for logging to Weights & Biases.

:author: Tamar Lavee (tlavee@ets.org)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import wandb
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from skll.config import _setup_config_parser
from skll.types import PathOrStr


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
        self.wandb_run: Optional[Union[Run, RunDisabled]] = None
        if wandb_credentials:
            self.wandb_run = wandb.init(
                project=wandb_credentials["wandb_project"],
                entity=wandb_credentials["wandb_entity"],
                config=get_config_dict(config_file_path),
            )
        self.label_metrics_table: Optional[wandb.Table] = None

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

    def log_summary_file(self, summary_file_path: PathOrStr) -> None:
        """
        Log a task summary file to W&B if logging to W&B is enabled.

        The summary is logged as a table to the wandb run.

        Parameters
        ----------
        summary_file_path : PathOrStr
            The path to the summary tsv file

        """
        if self.wandb_run:
            summary_df = pd.read_csv(summary_file_path, sep="\t")
            summary_table = wandb.Table(dataframe=summary_df, allow_mixed_types=True)
            self.wandb_run.log({"Summary": summary_table})

    def log_evaluation_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log evaluation results to W&B, if logging to W&B is enabled.

        Log basic task info, general scores, and label-specific evaluation scores
        and confusion matrix for classification tasks.

        The input is either "evaluate" task results or a single
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
            task_prefix = task_results["job_name"]
            # if this is a fold's result, add fold name to the prefix.
            if fold_num := task_results.get("fold"):
                task_prefix = f"{task_prefix}_fold_{fold_num}"

            # log confusion matrix as a custom chart
            if confusion_matrix := task_results.get("conf_matrix"):
                self.log_conf_matrix_chart(
                    task_prefix, confusion_matrix, sorted(task_results["label_metrics"].keys())
                )

                # log Precision, recall and f-measure for each label
                if not self.label_metrics_table:
                    self.label_metrics_table = wandb.Table(
                        columns=["Job Name", "Label", "Precision", "Recall", "F-measure"],
                        allow_mixed_types=True,
                    )
                for label, label_metric_dict in task_results["label_metrics"].items():
                    self.label_metrics_table.add_data(
                        task_prefix,
                        label,
                        label_metric_dict.get("precision"),
                        label_metric_dict.get("recall"),
                        label_metric_dict.get("f-measure"),
                    )

    def log_label_metric_table(self) -> None:
        """
        Log the full label metric table to W&B, if logging to W&B is enabled.

        This table is populated when evaluation results are logged for
        classification learners. This method should be called after all task
        results of a `evaluate` or `cross_validate` task have been logged.
        """
        if self.wandb_run and self.label_metrics_table:
            self.wandb_run.log({"classification metrics": self.label_metrics_table})

    def log_train_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log train results to W&B, if logging to W&B is enabled.

        Log basic task info and the model file location.

        Parameters
        ----------
        task_results : Dict[str, Any]
            The train task results.

        """
        if self.wandb_run:
            task_prefix = task_results["job_name"]
            for metric in ["train_set_size", "test_set_size", "model_file"]:
                self.log_to_summary(task_prefix, metric, task_results.get(metric, "N/A"))

    def log_predict_results(self, task_results: Dict[str, Any]) -> None:
        """
        Log predict results to W&B, if logging to W&B is enabled.

        Log basic task info to the run, and predictions file as a table.

        Parameters
        ----------
        task_results : Dict[str, Any]
            The predict task results.

        """
        if self.wandb_run:
            task_prefix = task_results["job_name"]
            for metric in ["train_set_size", "test_set_size"]:
                self.log_to_summary(task_prefix, metric, task_results.get(metric, "N/A"))
            predictions_df = pd.read_csv(task_results["predictions_file"], sep="\t")
            predictions_table = wandb.Table(dataframe=predictions_df, allow_mixed_types=True)
            self.wandb_run.log({f"{task_prefix}/predictions": predictions_table})

    def log_conf_matrix_chart(self, task_prefix, confusion_matrix, labels) -> None:
        """
        Log a confusion matrix to wandb if logging to wandb is enabled.

        A chart object is created from the confusion matrix data and logged
        to the wandb run.

        Parameters
        ----------
        task_prefix: str
            The task's name, to be used in the matrix name.
        confusion_matrix : List[List[int]]
            the confusion matrix values
        labels : List[str]
            label names

        """
        if self.wandb_run:
            conf_matrix_data = []
            for row_idx, row in enumerate(confusion_matrix):
                for col_idx, cell_value in enumerate(row):
                    conf_matrix_data.append([labels[row_idx], labels[col_idx], cell_value])
            table = wandb.Table(columns=["Predicted", "Actual", "Count"], data=conf_matrix_data)
            chart = wandb.visualize("wandb/confusion_matrix/v1", table)
            self.wandb_run.log({f"{task_prefix}/confusion_matrix": chart})

    def log_to_summary(self, task_prefix, metric_name, metric_value) -> None:
        """
        Add a metric to the W&B run summary if logging to W&B is enabled.

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


def get_config_dict(config_file_path: str) -> Dict[str, Any]:
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
