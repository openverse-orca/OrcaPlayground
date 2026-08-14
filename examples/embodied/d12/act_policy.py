import logging
import os

import numpy as np
import torch

from .act_model import ACTLite, ACTLiteVision, ACTDet, ACTDetVision

logger = logging.getLogger(__name__)


class ACTPolicy:
    """ACT 策略推理封装，支持 ACT-Lite 和 ACT-V 两种模型。

    核心功能:
    - 加载 checkpoint 和归一化统计量
    - 两种执行模式: temporal ensemble (每步推理) / chunk 执行 (每 K 步推理一次)
    - EMA 低通滤波消除高频噪声
    - 30 维动作向量 → OSC 控制器所需的末端位姿 + 夹爪
    - ACT-Lite 可选 phase 条件化 (追加 phase 维度到 state)
    """

    ACTION_DIM = 30
    JOINT_MOTOR_SLICE = slice(0, 14)
    END_POS_SLICE = slice(14, 20)
    END_ORIENT_SLICE = slice(20, 28)
    EFFECTOR_MOTOR_SLICE = slice(28, 30)

    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str | None = None,
        device_str: str = "cuda",
        max_steps: int = 3000,
        ema_alpha: float = 0.5,
        ensemble_lambda: float = 0.1,
        ref_trajectory_path: str | None = None,
        exec_mode: str = "ensemble",
        use_phase: bool = False,
    ):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.max_steps = max_steps
        self.ema_alpha = ema_alpha
        self.ensemble_lambda = ensemble_lambda
        self.exec_mode = exec_mode
        self.use_phase = use_phase

        if norm_stats_path is None:
            norm_stats_path = os.path.join(os.path.dirname(checkpoint_path), "norm_stats.pt")

        norm_stats = torch.load(norm_stats_path, map_location="cpu", weights_only=True)
        self.state_mean = norm_stats["state_mean"].numpy()
        self.state_std = norm_stats["state_std"].numpy()
        self.action_mean = norm_stats["action_mean"].numpy()
        self.action_std = norm_stats["action_std"].numpy()

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        self.model_type = "act_vision" if "num_cameras" in cfg else "act_lite"
        if cfg.get("deterministic", False):
            self.model_type = "act_det_vision" if "num_cameras" in cfg else "act_det"
        self.chunk_size = cfg["chunk_size"]
        self.action_dim = cfg["action_dim"]
        self.state_dim = cfg["state_dim"]
        self.delta_action = cfg.get("delta_action", False)

        if self.model_type == "act_det_vision":
            self.model = ACTDetVision(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                num_cameras=cfg["num_cameras"],
                img_size=cfg["img_size"],
                vision_backbone=cfg.get("vision_backbone", "lightweight"),
                chunk_size=cfg["chunk_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
            ).to(self.device)
            self.num_cameras = cfg["num_cameras"]
            self.img_size = cfg["img_size"]
            self.camera_names = cfg.get("camera_names", [])
        elif self.model_type == "act_det":
            self.model = ACTDet(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                chunk_size=cfg["chunk_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
            ).to(self.device)
            self.num_cameras = 0
            self.img_size = 0
            self.camera_names = []
        elif self.model_type == "act_vision":
            self.model = ACTLiteVision(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                num_cameras=cfg["num_cameras"],
                img_size=cfg["img_size"],
                vision_backbone=cfg.get("vision_backbone", "lightweight"),
                chunk_size=cfg["chunk_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                latent_dim=cfg["latent_dim"],
            ).to(self.device)
            self.num_cameras = cfg["num_cameras"]
            self.img_size = cfg["img_size"]
            self.camera_names = cfg.get("camera_names", [])
        else:
            self.model = ACTLite(
                state_dim=cfg["state_dim"],
                action_dim=cfg["action_dim"],
                chunk_size=cfg["chunk_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                latent_dim=cfg["latent_dim"],
            ).to(self.device)
            self.num_cameras = 0
            self.img_size = 0
            self.camera_names = []

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        if self.model_type == "act_lite" and use_phase and cfg["state_dim"] == 14:
            logger.info("Phase conditioning enabled for ACT-Lite: expanding state_dim 14->15")
            self._expand_model_for_phase(cfg)
            self.state_dim = 15
            self._expand_norm_stats_for_phase()

        self._all_time_actions = None
        self._prev_action = None
        self._ref_states = None
        self._current_chunk = None
        self._chunk_idx = 0
        self.t = 0

        if ref_trajectory_path:
            self._load_ref_trajectory(ref_trajectory_path)

        logger.info(
            f"ACTPolicy: type={self.model_type} state_dim={self.state_dim} "
            f"action_dim={self.action_dim} chunk={self.chunk_size} "
            f"exec={self.exec_mode} phase={self.use_phase} device={self.device}"
        )

    def _expand_norm_stats_for_phase(self):
        s_mean = np.append(self.state_mean, np.float32(0.5))
        s_std = np.append(self.state_std, np.float32(0.2887))
        self.state_mean = s_mean
        self.state_std = s_std

    def _expand_model_for_phase(self, cfg: dict):
        old_proj_weight = self.model.state_proj[0].weight.data.clone()
        old_proj_bias = self.model.state_proj[0].bias.data.clone()
        new_proj = torch.nn.Linear(15, old_proj_weight.shape[0])
        new_proj.weight.data[:, :14] = old_proj_weight
        new_proj.weight.data[:, 14] = 0.0
        new_proj.bias.data = old_proj_bias
        self.model.state_proj[0] = new_proj

        old_cvae_weight = self.model.cvae_encoder[0].weight.data.clone()
        old_cvae_bias = self.model.cvae_encoder[0].bias.data.clone()
        new_cvae_in = cfg["action_dim"] * cfg["chunk_size"] + 15
        new_cvae = torch.nn.Linear(new_cvae_in, old_cvae_weight.shape[0])
        new_cvae.weight.data[:, :old_cvae_weight.shape[1]] = old_cvae_weight
        new_cvae.weight.data[:, old_cvae_weight.shape[1]] = 0.0
        new_cvae.bias.data = old_cvae_bias
        self.model.cvae_encoder[0] = new_cvae

        self.model.state_dim = 15

    def _load_ref_trajectory(self, path: str):
        try:
            import h5py
            with h5py.File(path, "r") as f:
                self._ref_states = f["state/joint/position"][:].astype(np.float32)
            if self.max_steps > self._ref_states.shape[0]:
                self.max_steps = self._ref_states.shape[0]
            logger.info(f"Ref trajectory loaded: shape={self._ref_states.shape}, max_steps={self.max_steps}")
        except Exception as e:
            logger.warning(f"Failed to load ref trajectory: {e}")
            self._ref_states = None

    def reset(self):
        self.t = 0
        self._all_time_actions = None
        self._prev_action = None
        self._current_chunk = None
        self._chunk_idx = 0

    def get_state(self, env, agent_conf: dict) -> np.ndarray:
        if self._ref_states is not None:
            t_idx = min(self.t, self._ref_states.shape[0] - 1)
            state = self._ref_states[t_idx].copy()
        else:
            joint_names = agent_conf["l_arm"]["joint_names"] + agent_conf["r_arm"]["joint_names"]
            qpos = env.query_joint_qpos(joint_names)
            state = np.array([qpos[j] for j in joint_names], dtype=np.float32).flatten()

        needs_phase = (self.state_dim == 15) or (self.model_type == "act_vision")
        if needs_phase:
            phase = np.float32(self.t / max(self.max_steps - 1, 1))
            state = np.append(state, phase)

        return (state - self.state_mean[:len(state)]) / self.state_std[:len(state)]

    @torch.no_grad()
    def _predict_action_chunk(self, state: np.ndarray, images: np.ndarray | None = None) -> np.ndarray:
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if self.model_type == "act_vision" and images is not None:
            images_t = torch.from_numpy(images).float().unsqueeze(0).to(self.device)
            pred_actions, _, _ = self.model(state_t, images_t)
        elif self.model_type == "act_vision":
            pred_actions, _, _ = self.model(state_t)
        else:
            pred_actions, _, _ = self.model(state_t)

        return pred_actions.squeeze(0).cpu().numpy()

    def _temporal_ensemble_action(self, state: np.ndarray, images: np.ndarray | None = None) -> np.ndarray:
        t = self.t
        K = self.chunk_size

        if self._all_time_actions is None:
            buf_len = self.max_steps + K
            self._all_time_actions = np.zeros((buf_len, K, self.action_dim), dtype=np.float32)

        chunk = self._predict_action_chunk(state, images)
        self._all_time_actions[t, :chunk.shape[0], :] = chunk

        k_start = max(0, t - K + 1)
        num = t + 1 - k_start
        indices = np.arange(num)
        offsets = (num - 1) - indices

        selected = self._all_time_actions[k_start + indices, offsets]
        weights = np.exp(-self.ensemble_lambda * offsets)
        weights /= weights.sum()

        return np.sum(selected * weights[:, np.newaxis], axis=0)

    def _chunk_exec_action(self, state: np.ndarray, images: np.ndarray | None = None) -> np.ndarray:
        if self._current_chunk is None or self._chunk_idx >= self._current_chunk.shape[0]:
            chunk = self._predict_action_chunk(state, images)
            self._current_chunk = np.array([
                chunk[i] * self.action_std + self.action_mean for i in range(chunk.shape[0])
            ])
            self._chunk_idx = 0

        action = self._current_chunk[self._chunk_idx]
        self._chunk_idx += 1
        return action

    def predict(self, state: np.ndarray, images: np.ndarray | None = None) -> np.ndarray:
        if self.exec_mode == "chunk":
            action = self._chunk_exec_action(state, images)
        else:
            action = self._temporal_ensemble_action(state, images)
            action = action * self.action_std + self.action_mean

        if self.delta_action:
            if self._prev_action is not None:
                action = self._prev_action + action
            self._prev_action = action.copy()
        else:
            if self._prev_action is not None:
                action = self.ema_alpha * action + (1.0 - self.ema_alpha) * self._prev_action
            self._prev_action = action.copy()

        self.t += 1
        return action

    @staticmethod
    def parse_action(action: np.ndarray) -> dict:
        l_pos = action[ACTPolicy.END_POS_SLICE][:3].astype(np.float32)
        r_pos = action[ACTPolicy.END_POS_SLICE][3:6].astype(np.float32)
        l_quat_xyzw = action[ACTPolicy.END_ORIENT_SLICE][:4].astype(np.float32)
        r_quat_xyzw = action[ACTPolicy.END_ORIENT_SLICE][4:8].astype(np.float32)
        effector_motor = action[ACTPolicy.EFFECTOR_MOTOR_SLICE].astype(np.float32)

        for q in [l_quat_xyzw, r_quat_xyzw]:
            n = np.linalg.norm(q)
            if n > 1e-6:
                q /= n

        return {
            "l_pos": l_pos,
            "r_pos": r_pos,
            "l_quat_xyzw": l_quat_xyzw,
            "r_quat_xyzw": r_quat_xyzw,
            "l_grip": effector_motor[0],
            "r_grip": effector_motor[1],
        }
