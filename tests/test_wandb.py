"""
Tests for wandb logging utility class.

:author: Tamar Lavee (tlavee@ets.org)
"""

import unittest
from unittest.mock import Mock, patch

from skll.utils.wandb import WandbLogger


class TestWandb(unittest.TestCase):
    """Test cases for wandb interface class."""

    def test_init_wandb_enabled(self):
        """Test initialization with wandb credentials specified."""
        mock_wandb_run = Mock()
        with patch("skll.utils.wandb.wandb.init", return_value=mock_wandb_run) as mock_wandb_init:
            WandbLogger({"wandb_entity": "wandb_entity", "wandb_project": "wandb_project"})
            mock_wandb_init.assert_called_with(project="wandb_project", entity="wandb_entity")

    def test_init_wandb_disabled(self):
        """Test initialization with no wandb credentials."""
        mock_wandb_run = Mock()
        with patch("skll.utils.wandb.wandb.init", return_value=mock_wandb_run) as mock_wandb_init:
            WandbLogger({})
            mock_wandb_init.assert_not_called()
