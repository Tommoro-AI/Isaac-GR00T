import os
import numpy as np
import torch

from typing import Dict, Any
from inference_queue import ActionQueue  

def _make_dummy_obs() -> Dict[str, Any]:
    # Shapes/keys expected by 'fourier_gr1_arms_waist'; T=1 to match state_horizon=1
    return {
        "video.ego_view":  np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
        "state.left_arm":  np.random.rand(1, 7).astype("float32"),
        "state.right_arm": np.random.rand(1, 7).astype("float32"),
        "state.left_hand": np.random.rand(1, 6).astype("float32"),
        "state.right_hand":np.random.rand(1, 6).astype("float32"),
        "state.waist":     np.random.rand(1, 3).astype("float32"),
        "annotation.human.coarse_action": ["pick up the object"],
    }

def main():
    # Silence Albumentations update check
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    queue = ActionQueue(
        model_path="nvidia/GR00T-N1.5-3B",
        embodiment_tag="gr1",
        data_config="fourier_gr1_arms_waist",
        denoising_steps=4,
        device=device,
        actions_per_chunk=None,   # set an int if you want to cap per refill
        debug=True,               # turn on to print policy output schema once
    )

    obs = _make_dummy_obs()

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
