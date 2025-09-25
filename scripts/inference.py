# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any
import numpy as np
import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy


def run_inference(
    observations: Dict[str, Any],
    model_path: str = "nvidia/GR00T-N1.5-3B",
    embodiment_tag: str = "gr1",
    data_config: str = "fourier_gr1_arms_waist",
    denoising_steps: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Run inference on given observations using a GR00T model.
    Returns:
        Dictionary of predicted actions.
    """
    # Load modality configuration
    cfg = load_data_config(data_config)
    modality_config = cfg.modality_config()
    modality_transform = cfg.transform()

    # Load policy (GR00T checkpoint + transforms)
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=denoising_steps,
        device=device,
    )

    # Run inference and return results
    actions = policy.get_action(observations)
    return actions
