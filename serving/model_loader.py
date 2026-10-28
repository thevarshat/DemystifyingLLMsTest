"""
Model weight initialization and checkpoint loader.
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from utils.checkpoint import load_checkpoint
from utils.logging_utils import get_logger

logger = get_logger("model_loader")


class ModelLoader:
    """
    Manages filesystem resolution, weight deserialization, and parameter loading
    into transformer model instances.
    """

    DEFAULT_WEIGHTS_FILE = "model_weights.pt"
    DEFAULT_WEIGHTS_DIR = "weights"

    def __init__(self, weights_path: Optional[str] = None):
        if weights_path is not None:
            self.weights_path = weights_path
        else:
            self.weights_path = os.path.join(self.DEFAULT_WEIGHTS_DIR, self.DEFAULT_WEIGHTS_FILE)

        self.weights_filename = os.path.basename(self.weights_path)

    def load_weights(
        self,
        model: nn.Module,
        strict: bool = False,
        map_location: str = "cpu"
    ) -> nn.Module:
        """
        Load checkpoint weights from disk into the provided model module.

        Args:
            model: PyTorch model instance.
            strict: Enforce exact key matching.
            map_location: Target device placement for loaded weights.

        Returns:
            Model with updated parameter states.
        """
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(
                f"Model weights file could not be found at: '{self.weights_path}'. "
                f"Verify filename and extension."
            )

        logger.info(f"Loading weights from {self.weights_path}...")
        state_dict = load_checkpoint(self.weights_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        logger.info("Model weights loaded successfully.")
        return model
