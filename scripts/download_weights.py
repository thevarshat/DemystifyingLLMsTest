#!/usr/bin/env python3
"""
Weight Initialization & Checkpoint Helper.

Initializes or verifies model weights for the Broken Transformer Challenge.
By default, generates a deterministic tiny seeded checkpoint so local tests
and pipeline checks run smoothly on CPU without large external downloads.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import DEFAULT_WEIGHTS_PATH
from config.model_config import ModelConfig


def initialize_seeded_weights(output_path: str = DEFAULT_WEIGHTS_PATH, seed: int = 42):
    """
    Generate deterministic starter weights for the transformer model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.manual_seed(seed)

    config = ModelConfig()
    print(f"Initializing model weights with seed={seed} for config: {config}")

    from inference.engine.inference_engine import FullTransformerModel
    model = FullTransformerModel(config)
    state_dict = model.state_dict()

    torch.save(state_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Successfully saved weights checkpoint to '{output_path}' ({file_size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download or initialize transformer weights.")
    parser.add_argument("--output", type=str, default=DEFAULT_WEIGHTS_PATH, help="Output path for weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic initialization")
    args = parser.parse_args()

    initialize_seeded_weights(args.output, args.seed)


if __name__ == "__main__":
    main()
