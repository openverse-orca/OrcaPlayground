import numpy as np
import mujoco
from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv


class D12Env(OrcaGymEulerEnv):

    def __init__(
        self,
        frame_skip: int,
        orcagym_addr: str,
        agent_names: list[str],
        time_step: float,
        **kwargs,
    ):
        super().__init__(
            frame_skip=frame_skip,
            orcagym_addr=orcagym_addr,
            agent_names=agent_names,
            time_step=time_step,
            **kwargs,
        )
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv

    def apply_joint_qpos_dict(self, joint_qpos_dict: dict) -> None:
        """将 {joint_name: qpos} 字典合并到完整 qpos 数组并应用（Euler 兼容）。"""
        full_qpos = self.data.qpos.copy()
        for jname, jqpos in joint_qpos_dict.items():
            addr = self.jnt_qposadr(jname)
            arr = np.atleast_1d(np.asarray(jqpos, dtype=full_qpos.dtype))
            full_qpos[addr:addr + len(arr)] = arr
        self.set_joint_qpos(full_qpos)

    @property
    def opt(self):
        """OSC 控制器兼容：返回 sim_config（含 timestep 等 opt 字段）。"""
        return self.sim_config

    def mj_fullM(self) -> np.ndarray:
        """计算完整质量矩阵 (nv, nv)（OSC 控制器需要，Euler 体系暂缺此公共方法）。"""
        # OrcaGymEuler.__getattribute__ 拦截 _sim 访问，改走 DataView 的 _mj_data/_mj_model。
        # DataView 用 __getattr__ 兜底（不拦截已存在属性），_mj_data/_mj_model 在 __dict__ 中。
        mj_data = self.data._mj_data  # noqa: SLF001  Euler 未暴露 qM，待 OrcaGym 侧扩展后移除
        mj_model = self.data._mj_model  # noqa: SLF001
        nv = mj_model.nv
        mass_matrix = np.ndarray(shape=(nv, nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(mj_model, mass_matrix, mj_data.qM)
        return mass_matrix.reshape((nv, nv))

    def reset_model(self) -> tuple[dict, dict]:
        self.ctrl = np.zeros(self.nu, dtype=np.float32)
        self.mj_forward()
        return self._get_obs(), {}

    def _get_obs(self) -> dict:
        return {
            "qpos": np.array(self.data.qpos, dtype=np.float32),
            "qvel": np.array(self.data.qvel, dtype=np.float32),
            "ctrl": np.array(self.ctrl, dtype=np.float32),
        }
