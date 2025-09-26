from collections import deque
from typing import Deque, Dict, Any, List, Optional
import numpy as np
import torch

from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

ActionStep = np.ndarray            # (action_dim,)
ActionDict = Dict[str, Any]        # may include {"action": (T, action_dim)} or similar


# ---------------- Profiles ----------------
PROFILE_MAP = {
    "gr1": {
        "embodiment_tag": "gr1",
        "data_config": "fourier_gr1_arms_waist",
        "hints_any": {
            "state.left_arm", "state.right_arm",
            "state.left_hand", "state.right_hand",
            "video.ego_view",
        },
        # action keys order expected from policy output:
        "action_order": [
            "action.left_arm", "action.right_arm",
            "action.left_hand", "action.right_hand",
            "action.waist",
        ],
    },
    "genie1_gripper": {
        "embodiment_tag": "agibot_genie1",
        "data_config": "agibot_genie1",
        "hints_any": {
            "state.left_arm_joint_position", "state.right_arm_joint_position",
            "state.left_effector_position", "state.right_effector_position",
            "state.head_position", "state.waist_position",
            "video.top_head", "video.hand_left", "video.hand_right",
        },
        "action_order": [
            "action.left_arm_joint_position",
            "action.right_arm_joint_position",
            "action.left_effector_position",
            "action.right_effector_position",
            "action.head_position",
            "action.waist_position",
            "action.robot_velocity",  # optional
        ],
    },
    "oxe_droid": {
        "embodiment_tag": "oxe_droid",
        "data_config": "oxe_droid",
        "hints_any": {
            # video keys from the metadata you shared
            "video.exterior_image_1", "video.exterior_image_2", "video.wrist_image",
            # state keys from the metadata you shared
            "state.eef_position", "state.eef_rotation",
            "state.gripper_position", "state.joint_position",
        },
        # Primary action fields first; optional deltas/velocities appended if present
        "action_order": [
            "action.eef_position",
            "action.eef_rotation",
            "action.gripper_position",
            "action.joint_position",
            # optional (include if present in policy output)
            "action.eef_position_delta",
            "action.eef_rotation_delta",
            "action.gripper_velocity",
            "action.joint_velocity",
        ],
    },
}


# ---------------- Utilities ----------------
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.array(x)
    except Exception:
        return None


def _as_TD(x) -> Optional[np.ndarray]:
    """
    Normalize arrays to (T, D):
      - scalar -> (1,1)
      - (D,)   -> (1,D)
      - (T,)   -> (T,1)
      - (T,D)  -> (T,D)
      - (T,...) with more dims -> flattened to (T, prod(...))
    """
    arr = _to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr[None, None]
    elif arr.ndim == 1:
        # Heuristic: if length>1, treat as (D,), else as (T,)
        arr = arr[None, :] if arr.shape[0] > 1 else arr[:, None]
    elif arr.ndim > 2:
        T = arr.shape[0]
        arr = arr.reshape(T, -1)
    return arr


def _infer_profile_from_obs(obs: Dict[str, Any]) -> Optional[str]:
    obs_keys = set(obs.keys())
    for key, meta in PROFILE_MAP.items():
        if any(h in obs_keys for h in meta["hints_any"]):
            return key
    return None


def _extract_action_chunk(d: Dict[str, Any]) -> Optional[np.ndarray]:
    # direct packed block
    if "action" in d:
        arr = _as_TD(d["action"])
        if arr is not None and arr.ndim == 2:
            return arr

    ORDER_GR1   = PROFILE_MAP["gr1"]["action_order"]
    ORDER_GENIE = PROFILE_MAP["genie1_gripper"]["action_order"]
    ORDER_DROID = PROFILE_MAP["oxe_droid"]["action_order"]

    keys = set(d.keys())
    # Prioritize orders whose keys appear in the dict
    priority = []
    if any(k in keys for k in ORDER_DROID): priority.append(ORDER_DROID)
    if any(k in keys for k in ORDER_GENIE): priority.append(ORDER_GENIE)
    if any(k in keys for k in ORDER_GR1):   priority.append(ORDER_GR1)
    # Fallback try-all if nothing matched
    if not priority:
        priority = [ORDER_DROID, ORDER_GENIE, ORDER_GR1]

    for ORDER in priority:
        parts: List[np.ndarray] = []
        T: Optional[int] = None
        any_found = False

        for k in ORDER:
            if k not in d:
                continue
            any_found = True
            arr = _as_TD(d[k])
            if arr is None or arr.ndim != 2:
                continue
            if T is None:
                T = arr.shape[0]
            if arr.shape[0] != T:
                # time mismatch; skip this component
                continue
            parts.append(arr)

        if any_found and parts:
            try:
                return np.concatenate(parts, axis=1)
            except Exception:
                pass  # try next order

    # Unsupported embodiment
    # TODO: Add support for additional embodiments here
    pass
    return None


# ---------------- ActionQueue ----------------
class ActionQueue:
    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.5-3B",
        embodiment_tag: Optional[str] = None,
        data_config: Optional[str] = None,
        denoising_steps: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        actions_per_chunk: Optional[int] = None,
        debug: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.actions_per_chunk = actions_per_chunk
        self.debug = debug
        self.denoising_steps = denoising_steps

        # fixed selection for this instance (resolved at init or on first prime)
        self.embodiment_tag: Optional[str] = embodiment_tag
        self.data_config: Optional[str] = data_config

        # policy & transforms (lazy-initialized)
        self.policy: Optional[Gr00tPolicy] = None
        self._modality_config = None
        self._modality_transform = None

        # buffer state
        self._queue: Deque[ActionStep] = deque()
        self._last_obs: Optional[Dict[str, Any]] = None

    # -------- public API --------
    def prime(self, observations: Dict[str, Any]) -> None:
        self._last_obs = observations
        self.reset()

        if self.policy is None:
            # resolve embodiment/data_config if not provided
            if not (self.embodiment_tag and self.data_config):
                detected = _infer_profile_from_obs(observations)
                if detected is None:
                    print("[ActionQueue] Could not infer embodiment from observation keys.")
                    # TODO: pass explicit embodiment_tag/data_config when constructing ActionQueue
                    return
                self.embodiment_tag = PROFILE_MAP[detected]["embodiment_tag"]
                self.data_config = PROFILE_MAP[detected]["data_config"]

            # load once
            try:
                cfg = load_data_config(self.data_config)
                self._modality_config = cfg.modality_config()
                self._modality_transform = cfg.transform()
                self.policy = Gr00tPolicy(
                    model_path=self.model_path,
                    embodiment_tag=self.embodiment_tag,
                    modality_config=self._modality_config,
                    modality_transform=self._modality_transform,
                    denoising_steps=self.denoising_steps,
                    device=self.device,
                )
                if self.debug:
                    print(f"[ActionQueue] Loaded policy (tag={self.embodiment_tag}, cfg={self.data_config})")
            except Exception as e:
                print(f"[ActionQueue] Failed to load policy (tag={self.embodiment_tag}, cfg={self.data_config}): {e}")
                return

        self._refill_from_inference()

    def get_actions(self, n: int = 1) -> List[ActionStep]:
        """Pop up to n actions from the buffer, auto-refilling if possible."""
        out: List[ActionStep] = []
        while len(out) < n:
            if not self._queue:
                if self._last_obs is None or self.policy is None:
                    break
                self._refill_from_inference()
            if not self._queue:
                break
            out.append(self._queue.popleft())
        return out

    def get_one(self) -> Optional[ActionStep]:
        got = self.get_actions(1)
        return got[0] if got else None

    def reset(self, return_flushed: bool = False) -> List[ActionStep]:
        """Clear buffer. If return_flushed=True, return flushed actions."""
        flushed = list(self._queue) if return_flushed else []
        self._queue.clear()
        return flushed

    def buffered_len(self) -> int:
        return len(self._queue)

    def has_buffer(self) -> bool:
        return bool(self._queue)

    # -------- internal --------
    def _refill_from_inference(self) -> None:
        """Call policy.get_action(self._last_obs) and enqueue the resulting (T, D) chunk."""
        if self.policy is None or self._last_obs is None:
            return
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

        for t in range(action_chunk.shape[0]):
            self._queue.append(action_chunk[t])
