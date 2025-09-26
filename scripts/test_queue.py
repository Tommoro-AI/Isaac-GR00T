import os
import sys
import argparse
import numpy as np
import torch
from typing import Dict, Any

from inference_queue import ActionQueue


# --------- OBS BUILDERS ---------
def _make_obs_gr1() -> Dict[str, Any]:
    # fourier_gr1_arms_waist (T=1)
    return {
        "video.ego_view":  np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.left_arm":  np.random.rand(1, 7).astype("float32"),
        "state.right_arm": np.random.rand(1, 7).astype("float32"),
        "state.left_hand": np.random.rand(1, 6).astype("float32"),
        "state.right_hand":np.random.rand(1, 6).astype("float32"),
        "state.waist":     np.random.rand(1, 3).astype("float32"),
        "annotation.human.coarse_action": ["pick up the object"],
    }


def _make_obs_genie() -> Dict[str, Any]:
    # agibot_genie1 (from metadata: video + states + language)
    return {
        # --- video (uint8, B H W C) ---
        "video.top_head":   np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.hand_left":  np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.hand_right": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),

        # --- state (float32) ---
        "state.left_arm_joint_position":  np.random.rand(1, 7).astype("float32"),
        "state.right_arm_joint_position": np.random.rand(1, 7).astype("float32"),
        "state.left_effector_position":   np.random.rand(1, 1).astype("float32"),
        "state.right_effector_position":  np.random.rand(1, 1).astype("float32"),
        "state.head_position":            np.random.rand(1, 2).astype("float32"),
        "state.waist_position":           np.random.rand(1, 2).astype("float32"),

        # --- language ---
        "annotation.language.action_text": ["pick up the object"],
    }


def _make_obs_droid() -> Dict[str, Any]:
    # oxe_droid (from metadata you provided)
    return {
        # --- video (uint8, B H W C); fps=15 but we pass single frame (T/B=1) here ---
        "video.exterior_image_1": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.exterior_image_2": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "video.wrist_image":      np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),

        # --- state (float32) ---
        "state.eef_position":     np.random.rand(1, 3).astype("float32"),
        "state.eef_rotation":     np.random.rand(1, 3).astype("float32"),  # euler_angles_rpy
        "state.gripper_position": np.random.rand(1, 1).astype("float32"),
        "state.joint_position":   np.random.rand(1, 7).astype("float32"),

        # --- language (choose one of the available keys) ---
        "annotation.language.language_instruction": ["pick up the object"],
        # Other options in your metadata:
        #   language_instruction_2, language_instruction_3
    }


# --------- PROFILE MAP ---------
PROFILES = {
    "gr1": {
        "model_path": "nvidia/GR00T-N1.5-3B",
        "embodiment_tag": "gr1",
        "data_config": "fourier_gr1_arms_waist",
        "obs_fn": _make_obs_gr1,
    },
    "genie": {  # agibot_genie1
        "model_path": "nvidia/GR00T-N1.5-3B",
        "embodiment_tag": "agibot_genie1",
        "data_config": "agibot_genie1",  # adjust if your repo uses a different name
        "obs_fn": _make_obs_genie,
    },
    "droid": {  # oxe_droid
        "model_path": "nvidia/GR00T-N1.5-3B",
        "embodiment_tag": "oxe_droid",
        "data_config": "oxe_droid",  # adjust if your repo uses a different name
        "obs_fn": _make_obs_droid,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ActionQueue test with selectable observation profile")
    p.add_argument("profile", choices=PROFILES.keys(), help="Select obs profile: gr1 | genie | droid")
    p.add_argument("--denoising-steps", type=int, default=4)
    p.add_argument("--actions-per-chunk", type=int, default=None)
    p.add_argument("--debug", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    prof = PROFILES[args.profile]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Profile: {args.profile} | model={prof['model_path']} | "
          f"embodiment={prof['embodiment_tag']} | data_config={prof['data_config']}")

    queue = ActionQueue(
        model_path=prof["model_path"],
        embodiment_tag=prof["embodiment_tag"],
        data_config=prof["data_config"],
        denoising_steps=args.denoising_steps,
        device=device,
        actions_per_chunk=args.actions_per_chunk,
        debug=True,
    )

    obs = prof["obs_fn"]()

    print("[STEP] prime() …")
    queue.prime(obs)
    print(f"[CHECK] buffered after prime: {queue.buffered_len()}")

    a1 = queue.get_one()
    if a1 is None:
        print("[ERROR] get_one returned None.")
    else:
        print(f"[OK] one action: shape={a1.shape}, buffer_remaining={queue.buffered_len()}")

    print("[STEP] get_actions(n=3) … (auto-refill if needed)")
    batch = queue.get_actions(3)
    print(f"[CHECK] got {len(batch)} actions; buffer_remaining={queue.buffered_len()}")
    if batch:
        print(f"       first action shape: {batch[0].shape}")

    print("[STEP] reset(flush)")
    flushed = queue.reset(return_flushed=True)
    print(f"[CHECK] flushed={len(flushed)}; buffer={queue.buffered_len()}")

    print("[STEP] post-reset one more get_one (auto-refill expected)")
    a2 = queue.get_one()
    if a2 is None:
        print("[ERROR] post-reset get_one returned None.")
    else:
        print(f"[OK] post-reset one action: shape={a2.shape}, buffer_remaining={queue.buffered_len()}")

    print("[DONE] queue test finished.")


if __name__ == "__main__":
    main()
