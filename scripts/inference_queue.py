from collections import deque
from typing import Deque, Dict, Any, List, Optional
import numpy as np
import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

ActionStep = np.ndarray           # (action_dim,)
ActionDict = Dict[str, Any]       # may include {"action": (T, action_dim)} or similar

def _to_numpy(x):
    """Convert tensors/lists to numpy; return None if not possible."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return None


def _extract_action_chunk(d: Dict[str, Any]) -> Optional[np.ndarray]:
    ORDER = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.waist",
    ]

    parts = []
    T = None
    for k in ORDER:
        if k not in d:
            continue
        arr = _to_numpy(d[k])
        if arr is None:
            continue
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            continue
        if T is None:
            T = arr.shape[0]
        if arr.shape[0] != T:
            # time dimension mismatch; skip this effector
            continue
        parts.append(arr)

    if parts:
        return np.concatenate(parts, axis=1)  # (T, sum(dims))

    # TODO: support other embodiments

    return None


class ActionQueue:

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.5-3B",
        embodiment_tag: str = "gr1",
        data_config: str = "fourier_gr1_arms_waist",
        denoising_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        actions_per_chunk: Optional[int] = None,
        debug: bool = False,
    ):
        self.device = device
        self.actions_per_chunk = actions_per_chunk
        self.debug = debug

        # Load data config and transforms once
        cfg = load_data_config(data_config)
        self._modality_config = cfg.modality_config()
        self._modality_transform = cfg.transform()

        # Load policy once
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=self._modality_config,
            modality_transform=self._modality_transform,
            denoising_steps=denoising_steps,
            device=device,
        )

        # Action buffer and last observation
        self._queue: Deque[ActionStep] = deque()
        self._last_obs: Optional[Dict[str, Any]] = None

    # ----------------- Public API -----------------

    # Function to call if we get a new trigger from outside container
    def prime(self, observations: Dict[str, Any]) -> None:
        self._last_obs = observations
        self.reset()
        self._refill_from_inference()

    # get n actions
    def get_actions(self, n: int = 1) -> List[ActionStep]:
        out: List[ActionStep] = []
        while len(out) < n:
            if not self._queue:
                # try to refill if no buffered actions
                if self._last_obs is None:
                    break
                self._refill_from_inference()
            out.append(self._queue.popleft())
        return out

    # only get one action
    def get_one(self) -> Optional[ActionStep]:
        got = self.get_actions(1)
        return got[0] if got else None

    # reset the buffer
    def reset(self, return_flushed: bool = False) -> List[ActionStep]:
        """Clear buffer. If return_flushed=True, return the flushed actions."""
        if return_flushed:
            flushed = list(self._queue)
        self._queue.clear()
        return flushed if return_flushed else []

    def buffered_len(self) -> int:
        return len(self._queue)

    def has_buffer(self) -> bool:
        return bool(self._queue)

    # ----------------- Internal -----------------

    def _refill_from_inference(self) -> None:
        # Call groot 1.5 policy.get_action(last_obs) and push an action chunk into the buffer.
        try:
            out: ActionDict = self.policy.get_action(self._last_obs)
        except Exception as e:
            print(f"[ActionQueue] Inference error: {e}")
            return

        if self.debug:
            try:
                summary = {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in out.items()}
                print(f"[ActionQueue] policy output: {summary}")
            except Exception:
                pass

        action_chunk = _extract_action_chunk(out)
        if action_chunk is None:
            print(f"[ActionQueue] get_action() returned no usable action. Keys: {list(out.keys())}")
            return

        if self.actions_per_chunk is not None:
            action_chunk = action_chunk[: self.actions_per_chunk]

        # Push each timestep into the queue
        for t in range(action_chunk.shape[0]):
            self._queue.append(action_chunk[t])
