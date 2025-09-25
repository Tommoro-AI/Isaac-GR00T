# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

"""
GR00T Inference Server

This script launches a GR00T inference server.
It loads the model and exposes it through either:
- ZMQ server (default)
- HTTP server (FastAPI + Uvicorn, if --http-server is enabled)

Usage (ZMQ):
    python scripts/inference_server.py --model-path nvidia/GR00T-N1.5-3B --port 5555

Usage (HTTP):
    pip install uvicorn fastapi json-numpy
    python scripts/inference_server.py --http-server --host 0.0.0.0 --port 8000
"""

from dataclasses import dataclass
from typing import Literal

import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy
from gr00t.eval.robot import RobotInferenceServer


@dataclass
class Args:
    # Model and config
    model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    data_config: str = "fourier_gr1_arms_waist"
    denoising_steps: int = 4

    # Server settings
    host: str = "0.0.0.0"
    port: int = 5555
    api_token: str | None = None

    # Switch between ZMQ and HTTP
    http_server: bool = False


def main(args: Args):
    # Load modality configuration and transforms
    cfg = load_data_config(args.data_config)
    modality_config = cfg.modality_config()
    modality_transform = cfg.transform()

    # Initialize the policy (loads the pretrained checkpoint)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
    )

    # Launch the appropriate server
    if args.http_server:
        # REST API (requires uvicorn, fastapi, json-numpy)
        from gr00t.eval.http_server import HTTPInferenceServer
        server = HTTPInferenceServer(
            policy, port=args.port, host=args.host, api_token=args.api_token
        )
    else:
        # Default ZMQ server
        server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)

    server.run()


if __name__ == "__main__":
    main(tyro.cli(Args))
