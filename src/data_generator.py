"""
训练数据生成器
--------------
使用已知的"真实"参数运行仿真，产生用于参数辨识的训练轨迹。
支持多种激励信号类型，确保参数的可辨识性。
"""

import numpy as np
from .simulator import Simulator


def generate_sine(
    duration: float = 5.0,
    dt: float = 0.005,
    amp: float = 5.0,
    frequency: float = 0.7,
) -> np.ndarray:
    """生成单频正弦信号。

    使用接近共振的频率激励系统，使响应幅度对阻尼/摩擦参数敏感。

    Parameters
    ----------
    duration : float
        信号时长 (s)。
    dt : float
        采样间隔 (s)。
    amp : float
        力矩幅值 (N·m)。
    frequency : float
        正弦频率 (Hz)。

    Returns
    -------
    np.ndarray, shape (N,)
        力矩序列。
    """
    t = np.arange(0, duration, dt)
    return amp * np.sin(2 * np.pi * frequency * t)


def generate_multisine(
    duration: float = 5.0,
    dt: float = 0.005,
    amp: float = 5.0,
    frequencies: list | None = None,
    seed: int = 42,
) -> np.ndarray:
    """生成多频率谐波叠加信号。

    同时包含低频和高频成分，能在较短的时间内提供丰富的激励，
    有利于分离质量（重力效应）和电枢惯量（惯性效应）。

    Parameters
    ----------
    duration : float
        信号时长 (s)。
    dt : float
        采样间隔 (s)。
    amp : float
        力矩幅值 (N·m)。
    frequencies : list | None
        叠加的频率列表 (Hz)，默认 [0.1, 0.5, 1.5, 5.0]。
    seed : int
        随机种子，用于各频率分量的相位。

    Returns
    -------
    np.ndarray, shape (N,)
        力矩序列。
    """
    if frequencies is None:
        frequencies = [0.1, 0.5, 1.5, 5.0]

    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, dt)
    signal = np.zeros_like(t)
    for f in frequencies:
        phase = rng.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * f * t + phase)
    # 归一化到 [-amp, amp]
    signal *= amp / np.max(np.abs(signal))
    return signal


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
    f_start: float = 0.05,
    f_end: float = 10.0,
    amp: float = 5.0,
    multi_q0: list | None = None,
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
        激励信号类型。
    q0 : float
        初始位置 (rad)，仅在 multi_q0 为 None 时使用。
    qd0 : float
        初始速度 (rad/s)。
    seed : int
        随机种子。
    f_start : float
        扫频起始频率 (Hz)。
    f_end : float
        扫频终止频率 (Hz)。
    amp : float
        力矩幅值 (N·m)。
    multi_q0 : list | None
        多个初始角度列表。若提供，则在每个角度下运行仿真并拼接轨迹。
        这有助于打破参数耦合，提高可辨识性。

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

    if excitation == "sine":
        tau_seq = generate_sine(duration, dt, amp, frequency=f_start)
    elif excitation == "sweep":
        tau_seq = generate_sinusoidal_sweep(duration, dt, amp, f_start, f_end)
    elif excitation == "random":
        tau_seq = generate_random_steps(duration, dt, seed=seed)
    elif excitation == "multisine":
        tau_seq = generate_multisine(duration, dt, amp, seed=seed)
    else:
        raise ValueError(f"未知激励类型: {excitation}")

    if multi_q0 is not None:
        # 多个初始角度：在每个角度下运行仿真，拼接轨迹
        q_list, qd_list, tau_list = [], [], []
        for q0_val in multi_q0:
            traj = sim.run(tau_seq, q0_val, qd0)
            q_list.append(traj["q"])
            qd_list.append(traj["qd"])
            tau_list.append(tau_seq)
        return {
            "tau_seq": np.concatenate(tau_list),
            "q_true": np.concatenate(q_list),
            "qd_true": np.concatenate(qd_list),
            "dt": dt,
            "true_params": sim.get_params(),
        }
    else:
        traj = sim.run(tau_seq, q0, qd0)
        return {
            "tau_seq": traj["tau"],
            "q_true": traj["q"],
            "qd_true": traj["qd"],
            "dt": dt,
            "true_params": sim.get_params(),
        }
