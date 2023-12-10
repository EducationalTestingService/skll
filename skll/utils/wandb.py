"""
Utility classes and functions for logging to Weights & Biases.

:author: Tamar Lavee (tlavee@ets.org)
"""
from typing import Any, Dict, Optional

import wandb


class WandbLogger:
    """Interface for Weights and Biases logging."""

    def __init__(self, wandb_credentials: Optional[Dict[str, str]]):
        """
        Initialize the wandb_run if wandb_credentials are provided.

        Parameters
        ----------
        wandb_credentials : Optional[Dict[str, str]]
            A dictionary containing the W&B entity and project names that will be
            used to initialize the wandb run. If ``None``, logging to W&B will not be performed.
        """
        self.wandb_run = None
        if wandb_credentials:
            self.wandb_run = wandb.init(
                project=wandb_credentials["wandb_project"], entity=wandb_credentials["wandb_entity"]
            )

    def log_configuration(self, conf_dict: Dict[str, Any]):
        """
        Log a configuration dictionary to W&B if logging to W&B is enabled.

        Parameters
        ----------
        conf_dict : Dict[str, Any]
            A dictionary mapping configuration field names to their values.
        """
        if self.wandb_run:
            self.wandb_run.config.update(conf_dict)
