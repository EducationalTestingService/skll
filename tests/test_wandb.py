"""
Tests for wandb logging utility class.

:author: Tamar Lavee (tlavee@ets.org)
"""

import unittest
from unittest.mock import Mock, patch

from skll.utils.testing import config_dir, fill_in_config_options
from skll.utils.wandb import WandbLogger, get_config_dict


class TestWandb(unittest.TestCase):
    """Test cases for wandb interface class."""

    def test_init_wandb_enabled(self):
        """Test initialization with wandb credentials specified."""
        mock_wandb_run = Mock()
        config_template_path = config_dir / "test_wandb.template.cfg"
        values_to_fill_dict = {"experiment_name": "test_wandb", "task": "train"}
        config_path = fill_in_config_options(
            config_template_path, values_to_fill_dict, "conf_wandb"
        )
        config_dict = get_config_dict(config_path)
        with patch("skll.utils.wandb.wandb.init", return_value=mock_wandb_run) as mock_wandb_init:
            WandbLogger(
                {"wandb_entity": "wandb_entity", "wandb_project": "wandb_project"}, config_path
            )
            mock_wandb_init.assert_called_with(
                project="wandb_project", entity="wandb_entity", config=config_dict
            )

    def test_init_wandb_disabled(self):
        """Test initialization with no wandb credentials."""
        mock_wandb_run = Mock()
        with patch("skll.utils.wandb.wandb.init", return_value=mock_wandb_run) as mock_wandb_init:
            WandbLogger({}, "")
            mock_wandb_init.assert_not_called()
