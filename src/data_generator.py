"""
训练数据生成器
--------------
使用已知的"真实"参数运行仿真，产生用于参数辨识的训练轨迹。
支持多种激励信号类型，确保参数的可辨识性。
"""

import numpy as np
from .simulator import Simulator


def generate_sinusoidal_sweep(
    duration: float = 5.0,
    dt: float = 0.001,
    amp: float = 5.0,
    f_start: float = 0.1,
    f_end: float = 5.0,
) -> np.ndarray:
    """生成对数扫频正弦信号 (chirp)。

    频率从低到高扫过，能够充分激励系统的各频率成分，
    是参数辨识中最常用的激励信号之一。

    Parameters
    ----------
    duration : float
        信号时长 (s)。
    dt : float
        采样间隔 (s)，应与 MuJoCo 仿真步长一致。
    amp : float
        力矩幅值 (N·m)。
    f_start : float
        起始频率 (Hz)。
    f_end : float
        终止频率 (Hz)。

    Returns
    -------
    np.ndarray, shape (N,)
        力矩序列。
    """
    t = np.arange(0, duration, dt)
    # 对数扫频: 频率随时间指数增长
    k = (f_end / f_start) ** (1.0 / duration)
    freq = f_start * k**t
    phase = 2 * np.pi * f_start * (k**t - 1) / np.log(k)
    return amp * np.sin(phase)


def generate_random_steps(
    duration: float = 5.0,
    dt: float = 0.001,
    amp: float = 5.0,
    step_interval: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """生成随机阶跃信号。

    每隔一段时间随机切换力矩幅值，提供丰富的时域激励。

    Parameters
    ----------
    duration : float
        信号时长 (s)。
    dt : float
        采样间隔 (s)。
    amp : float
        力矩幅值上限 (N·m)。
    step_interval : float
        阶跃切换间隔 (s)。
    seed : int
        随机种子。

    Returns
    -------
    np.ndarray, shape (N,)
        力矩序列。
    """
    rng = np.random.default_rng(seed)
    n_total = int(duration / dt)
    n_per_step = int(step_interval / dt)

    tau = np.empty(n_total)
    for i in range(0, n_total, n_per_step):
        end = min(i + n_per_step, n_total)
        tau[i:end] = rng.uniform(-amp, amp)
    return tau


def generate_training_data(
    sim: Simulator,
    duration: float = 5.0,
    dt: float = 0.001,
    excitation: str = "sweep",
    q0: float = 0.0,
    qd0: float = 0.0,
    seed: int = 42,
) -> dict:
    """生成训练数据：用真实参数运行仿真，记录轨迹。

    Parameters
    ----------
    sim : Simulator
        已加载仿真器（参数已设置为真实值）。
    duration : float
        仿真时长 (s)。
    dt : float
        仿真步长 (s)。
    excitation : str
        激励信号类型: "sweep" (扫频), "random" (随机阶跃)。
    q0 : float
        初始位置 (rad)。
    qd0 : float
        初始速度 (rad/s)。
    seed : int
        随机种子，保证可复现。

    Returns
    -------
    dict
        tau_seq: 输入力矩序列
        q_true:  真实位置轨迹
        qd_true: 真实速度轨迹
        dt:      仿真步长
        true_params: 真实参数值
    """
    np.random.seed(seed)

    if excitation == "sweep":
        tau_seq = generate_sinusoidal_sweep(duration, dt)
    elif excitation == "random":
        tau_seq = generate_random_steps(duration, dt, seed=seed)
    else:
        raise ValueError(f"未知激励类型: {excitation}")

    traj = sim.run(tau_seq, q0, qd0)

    return {
        "tau_seq": traj["tau"],
        "q_true": traj["q"],
        "qd_true": traj["qd"],
        "dt": dt,
        "true_params": sim.get_params(),
    }
