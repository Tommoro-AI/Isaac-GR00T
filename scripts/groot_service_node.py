import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import CompressedImage, JointState

import numpy as np
import cv2
import torch
import threading
import time
import subprocess

import wandb
from collections import deque

from gr00t.model.gr00t_n1 import GR00T_N1_5
from scripts.gr00t_finetune import ArgsConfig, main as train_main
from physical_ai_interfaces.srv import StartInference
from physical_ai_interfaces.srv import GetImageTopicList, GetJointTopicList
from gr00t_vla_interfaces.msg import TrainingInfo, TrainingStatus


QOS_SENSOR = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
)

ACTION = "action"
OBS_IMAGE = "obs_image"

class DummyPolicy:
    """GR00T 실패시를 위한 간단 더미 정책 (6자유도)"""
    def __init__(self, action_dim=6):
        self.action_dim = action_dim
        self._q = deque()

    def predict_action_chunk(self, batch):
        # (n_steps, batch, dim)
        return torch.rand((1, 1, self.action_dim))

    def select_action(self, batch):
        if not self._q:
            self._q.extend(self.predict_action_chunk(batch))
        # [1, dim] 텐서 반환
        return self._q.popleft()[0]

class Gr00tSrvNode(Node):
    def __init__(self):
        super().__init__('gr00t_srv_node')

        # service for train/inference
        self.srv = self.create_service(StartInference, 'start_inference', self.start_callback)

        # inference pipeline members
        self.model = None          # GR00T_N1_5 or DummyPolicy
        self.pub = None            # JointTrajectory publisher
        self.img_subs = {}
        self.joint_subs = {}
        self.timer = None

        self.image_topics: list[str] = []
        self.joint_topics: list[str] = []

        self.latest_images = {}    # {cam_i: np.ndarray(H,W,3)}
        self.latest_joint = None   # JointState
        self.joint_names = [f"joint{i+1}" for i in range(6)]

        # training status publisher/timer
        qos = QoSProfile(depth=10)
        self.training_pub = self.create_publisher(TrainingStatus, 'training/status', qos)
        self.training_info = None
        self.training_thread = None
        self.training_status_timer = None
        self.is_training = False
        self.current_step = 0
        self.last_loss = 0.0
        self._stop_event = threading.Event()

        # prefetch topic lists if available
        self._bootstrap_topic_cache()

    # ---------------------- Service ---------------------- #
    def start_callback(self, request, response):
        mode = (request.mode or "").strip().lower()

        if mode == "train":
            # 모든 추론 리소스 해제
            self._stop_inference_pipeline()

            self.get_logger().info("Start training... (thread + status topic)")
            # 1) TrainingInfo 초기화
            self.training_info = TrainingInfo()
            self.training_info.dataset = request.config
            self.training_info.policy_type = "gr00t"
            self.training_info.output_folder_name = "gr00t_ckpt"
            self.training_info.policy_device = "cuda"
            self.training_info.sub_task_name = "demo"

            # 2) 상태 변수 초기화
            self.is_training = True
            self.current_step = 0
            self.last_loss = 0.0
            self._stop_event.clear()

            # 3) wandb run (없어도 동작)
            try:
                wandb.init(project="GR00T-demo", name="training-run", reinit=True)
            except Exception as e:
                self.get_logger().warn(f"[wandb] init failed: {e}")

            # 4) 상태 퍼블리시 타이머
            if self.training_status_timer is None:
                self.training_status_timer = self.create_timer(0.5, self._publish_training_status)

            # 5) 학습 스레드 시작 (실제 fine-tune 호출부)
            self.training_thread = threading.Thread(
                target=self._train_worker, args=(request.config,), daemon=True
            )
            self.training_thread.start()

            response.accepted = True
            response.message = "Training started"
            return response

        elif mode == "inference":
            self._stop_inference_pipeline()

            self.get_logger().info("Start inference... (launch inference.py)")

            try:
                cmd = [
                    "python3",
                    "/workspace/gr00t/scripts/inference.py",  # TODO: 경로 확인
                    "--config", request.config
                ]
                subprocess.Popen(cmd)  # 비동기 실행
                response.accepted = True
                response.message = "Inference started via inference.py"
            except Exception as e:
                self.get_logger().error(f"[inference] error: {e}")
                response.accepted = False
                response.message = str(e)

            return response

        else:
            response.accepted = False
            response.message = f"Unknown mode {request.mode}"
            return response

    # ---------------------- Training ---------------------- #
    def _train_worker(self, dataset_path: str):
        """
        실제 파이프라인에서 fine-tune을 돌릴 때는 gr00t_finetune.py 스크립트를 실행
        """
        try:
            cmd = [
                "python3",
                "/workspace/gr00t/scripts/gr00t_finetune.py", # TODO: 경로 확인
                "--dataset_path", dataset_path,
                "--output_dir", "/tmp/gr00t_ckpt",
                "--base_model_path", "nvidia/GR00T-N1.5-3B"
            ]
            subprocess.run(cmd, check=True)
        except Exception as e:
            self.get_logger().error(f"[train] error: {e}")
        finally:
            self.is_training = False
            self._publish_training_status()

            # wandb 종료
            try:
                if wandb.run:
                    wandb.finish()
            except Exception:
                pass

            # 타이머 정지
            if self.training_status_timer is not None:
                self.destroy_timer(self.training_status_timer)
                self.training_status_timer = None


    def _publish_training_status(self):
        msg = TrainingStatus()
        msg.training_info = self.training_info if self.training_info else TrainingInfo()
        msg.current_step = int(self.current_step)
        msg.is_training = bool(self.is_training)
        msg.error = ""
        try:
            msg.wandb_url = wandb.run.get_url() if wandb.run else ""
        except Exception:
            msg.wandb_url = ""
        self.training_pub.publish(msg)
        self.get_logger().info(f"[TrainingStatus] step={msg.current_step} is_training={msg.is_training} loss={self.last_loss:.4f}")

    # ---------------------- Inference ---------------------- #
    def _on_img(self, name: str, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                self.latest_images[name] = img
        except Exception as e:
            self.get_logger().warn(f"[img] decode failed {name}: {e}")

    def _on_joint(self, msg: JointState):
        self.latest_joint = msg

    def _inference_timer_callback(self):
        if self.model is None or self.pub is None:
            return
        # 최소 한 카메라라도 최신 프레임 있어야 함
        if not any(v is not None for v in self.latest_images.values()):
            return

        # TODO: 여기서 latest_images(dict) → 모델 입력 포맷으로 전처리
        # ex) RGB resize/normalize, state concat 등
        batch = {"obs": torch.tensor([1.0])}

        try:
            # GR00T는 select_action 유틸이 없을 수도 있어 더미 래퍼로 통일
            if hasattr(self.model, "select_action"):
                action = self.model.select_action(batch)
            else:
                # 모델에 맞는 forward/predict 함수 연결 필요 시 이 부분에서
                # action = your_adapter(self.model, batch)
                action = self.model.predict_action_chunk(batch)[0, 0]
            action = action.detach().cpu().tolist()
        except Exception as e:
            self.get_logger().warn(f"[inference] select_action failed: {e}")
            return

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = action[:len(self.joint_names)]
        pt.time_from_start = Duration(sec=1)
        traj.points.append(pt)

        self.pub.publish(traj)

    # ---------------------- Utilities ---------------------- #
    def _bootstrap_topic_cache(self):
        # 이미지 토픽
        try:
            cli = self.create_client(GetImageTopicList, '/image/get_available_list')
            if cli.wait_for_service(timeout_sec=1.0):
                fut = cli.call_async(GetImageTopicList.Request())
                rclpy.spin_until_future_complete(self, fut)
                res = fut.result()
                if res and res.success and res.image_topic_list:
                    self.image_topics = list(res.image_topic_list)
                    self.get_logger().info(f"[topics] images: {self.image_topics}")
        except Exception as e:
            self.get_logger().warn(f"[topics] image list failed: {e}")

        # 조인트 토픽
        try:
            cli = self.create_client(GetJointTopicList, '/joint/get_available_list')
            if cli.wait_for_service(timeout_sec=1.0):
                fut = cli.call_async(GetJointTopicList.Request())
                rclpy.spin_until_future_complete(self, fut)
                res = fut.result()
                if res and res.success and res.joint_topic_list:
                    self.joint_topics = list(res.joint_topic_list)
                    self.get_logger().info(f"[topics] joints: {self.joint_topics}")
        except Exception as e:
            self.get_logger().warn(f"[topics] joint list failed: {e}")

    def _stop_inference_pipeline(self):
        if self.timer is not None:
            try: self.timer.cancel()
            except Exception: pass
            self.timer = None

        for name, sub in list(self.img_subs.items()):
            try: self.destroy_subscription(sub)
            except Exception: pass
            del self.img_subs[name]

        for name, sub in list(self.joint_subs.items()):
            try: self.destroy_subscription(sub)
            except Exception: pass
            del self.joint_subs[name]

        if self.pub is not None:
            try: self.destroy_publisher(self.pub)
            except Exception: pass
            self.pub = None

        # 버퍼 초기화
        for k in list(self.latest_images.keys()):
            self.latest_images[k] = None
        self.latest_joint = None

        # 모델도 초기화
        self.model = None

def main(args=None):
    rclpy.init(args=args)
    node = Gr00tSrvNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()