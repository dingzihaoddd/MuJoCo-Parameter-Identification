"""
MuJoCo 仿真器封装
------------------
提供前向动力学仿真、轨迹记录、以及模型参数读写功能。
用于"仿真辨识仿真"流程中的仿真环节。
"""

import numpy as np
import mujoco


class Simulator:
    """MuJoCo 单关节摆锤仿真器。

    封装了模型加载、前向仿真、轨迹记录和参数修改功能。
    可通过 set_params() 修改动力学参数，run() 执行仿真并返回轨迹。

    Parameters
    ----------
    model_path : str
        MuJoCo XML 模型文件路径。
    """

    def __init__(self, model_path: str, timestep: float | None = None):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)

        if timestep is not None:
            self._model.opt.timestep = timestep

        # 预计算模型元素的索引
        self._body_link_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "link"
        )
        self._joint_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_JOINT, "hinge"
        )
        # MuJoCo 中 joint 的 dof 地址
        self._dof_id = self._model.jnt_dofadr[self._joint_id]

        # 保存原始质量，用于缩放惯量时计算比例
        self._original_mass = float(self._model.body_mass[self._body_link_id])
        self._original_inertia = self._model.body_inertia[
            self._body_link_id
        ].copy()

    # ── 参数读写 ──────────────────────────────────────────────

    def get_params(self) -> dict:
        """返回当前模型的可辨识参数。

        Returns
        -------
        dict
            包含 mass, damping, frictionloss, armature 四个参数。
        """
        dof = self._dof_id
        return {
            "mass": float(self._model.body_mass[self._body_link_id]),
            "damping": float(self._model.dof_damping[dof]),
            "frictionloss": float(self._model.dof_frictionloss[dof]),
            "armature": float(self._model.dof_armature[dof]),
        }

    def set_params(self, params: dict) -> None:
        """设置模型参数。

        质量修改时会按比例同步缩放惯量矩阵，保证几何一致性。

        Parameters
        ----------
        params : dict
            可包含 mass, damping, frictionloss, armature 中的任意键。
        """
        dof = self._dof_id
        bid = self._body_link_id

        if "mass" in params:
            new_mass = float(params["mass"])
            scale = new_mass / self._model.body_mass[bid]
            self._model.body_mass[bid] = new_mass
            self._model.body_inertia[bid] = self._model.body_inertia[bid] * scale

        if "damping" in params:
            self._model.dof_damping[dof] = float(params["damping"])

        if "frictionloss" in params:
            self._model.dof_frictionloss[dof] = float(params["frictionloss"])

        if "armature" in params:
            self._model.dof_armature[dof] = float(params["armature"])

    # ── 仿真 ──────────────────────────────────────────────────

    def reset(self, q0: float = 0.0, qd0: float = 0.0) -> None:
        """将仿真状态重置到给定的初始位置和速度。

        Parameters
        ----------
        q0 : float
            初始关节角 (rad)。
        qd0 : float
            初始关节角速度 (rad/s)。
        """
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[0] = q0
        self._data.qvel[0] = qd0
        mujoco.mj_forward(self._model, self._data)

    def step(self, tau: float) -> None:
        """执行单步仿真。

        Parameters
        ----------
        tau : float
            关节力矩控制输入 (N·m)。
        """
        self._data.ctrl[0] = tau
        mujoco.mj_step(self._model, self._data)

    def run(
        self,
        tau_seq: np.ndarray,
        q0: float = 0.0,
        qd0: float = 0.0,
    ) -> dict:
        """在给定力矩序列下运行完整仿真，返回轨迹。

        Parameters
        ----------
        tau_seq : np.ndarray, shape (N,)
            关节力矩序列 (N·m)，每步一个值。
        q0 : float
            初始关节角 (rad)。
        qd0 : float
            初始关节角速度 (rad/s)。

        Returns
        -------
        dict
            q:   关节位置序列, shape (N,)
            qd:  关节速度序列, shape (N,)
            tau: 关节力矩序列 (输入回传), shape (N,)
        """
        N = len(tau_seq)
        self.reset(q0, qd0)

        q_traj = np.empty(N)
        qd_traj = np.empty(N)
        qdd_traj = np.empty(N)  # 同步记录加速度

        for i in range(N):
            self.step(float(tau_seq[i]))
            q_traj[i] = self._data.qpos[0]
            qd_traj[i] = self._data.qvel[0]
            qdd_traj[i] = self._data.qacc[0]

        return {"q": q_traj, "qd": qd_traj, "qdd": qdd_traj, "tau": tau_seq.copy()}
