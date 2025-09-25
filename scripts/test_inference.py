# scripts/test_inference_gr1.py
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from inference import run_inference

def main():
    # Shapes expected by "fourier_gr1_arms_waist":
    # video.ego_view:            (T,H,W,C) -> we pass (1,256,256,3) and it will be batched internally
    # state.left_arm/right_arm:  (T,7)
    # state.left_hand/right_hand:(T,6)
    # state.waist:               (T,3)
    # language:                  list[str] under "annotation.human.coarse_action"
    obs = {
        "video.ego_view":  np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.left_arm":  np.random.rand(1, 7).astype("float32"),
        "state.right_arm": np.random.rand(1, 7).astype("float32"),
        "state.left_hand": np.random.rand(1, 6).astype("float32"),
        "state.right_hand":np.random.rand(1, 6).astype("float32"),
        "state.waist":     np.random.rand(1, 3).astype("float32"),
        "annotation.human.coarse_action": ["pick up the object"],
    }

    print("Running GR-1 (fourier_gr1_arms_waist) dummy inference…")
    actions = run_inference(
        observations=obs,
        model_path="nvidia/GR00T-N1.5-3B",
        embodiment_tag="gr1",
        data_config="fourier_gr1_arms_waist",
        denoising_steps=4,
    )

    print("Predicted actions:")
    for k, v in actions.items():
        shape = getattr(v, "shape", None)
        print(f"  {k}: {shape if shape is not None else type(v)}")

if __name__ == "__main__":
    main()
