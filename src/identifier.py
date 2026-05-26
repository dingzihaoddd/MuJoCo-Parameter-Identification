"""
梯度下降参数辨识器
------------------
通过最小化仿真轨迹与真实轨迹之间的误差，辨识动力学参数。
使用 scipy.optimize.minimize (L-BFGS-B) 进行有界优化。
"""

import numpy as np
from scipy.optimize import minimize
from .simulator import Simulator


# ── 参数名与边界 ─────────────────────────────────────────────

PARAM_NAMES = ["mass", "damping", "frictionloss", "armature"]

# 参数搜索边界 (min, max)
PARAM_BOUNDS = {
    "mass": (0.01, 20.0),
    "damping": (0.0, 5.0),
    "frictionloss": (0.0, 2.0),
    "armature": (0.0, 0.5),
}


def _params_to_array(params: dict) -> np.ndarray:
    """将参数字典转为固定顺序的 numpy 数组。"""
    return np.array([params[name] for name in PARAM_NAMES], dtype=np.float64)


def _array_to_params(arr: np.ndarray) -> dict:
    """将 numpy 数组转回参数字典。"""
    return {name: float(arr[i]) for i, name in enumerate(PARAM_NAMES)}


def _get_bounds() -> list:
    """返回 scipy 格式的参数边界列表。"""
    return [PARAM_BOUNDS[name] for name in PARAM_NAMES]


def identify_params(
    sim: Simulator,
    tau_seq: np.ndarray,
    q_true: np.ndarray,
    qd_true: np.ndarray,
    initial_guess: dict | None = None,
    q0: float = 0.0,
    qd0: float = 0.0,
    weight_q: float = 1.0,
    weight_qd: float = 0.1,
    method: str = "L-BFGS-B",
    maxiter: int = 200,
    verbose: bool = True,
) -> dict:
    """通过梯度下降从轨迹数据中辨识动力学参数。

    损失函数 = weight_q * MSE(q) + weight_qd * MSE(qd)

    优化器在每次迭代中：
    1. 将当前参数写入仿真器
    2. 用相同的力矩输入运行完整仿真
    3. 计算仿真轨迹与真实轨迹之间的加权均方误差

    Parameters
    ----------
    sim : Simulator
        仿真器实例（参数将在优化过程中被反复修改）。
    tau_seq : np.ndarray, shape (N,)
        输入力矩序列。
    q_true : np.ndarray, shape (N,)
        真实位置轨迹。
    qd_true : np.ndarray, shape (N,)
        真实速度轨迹。
    initial_guess : dict | None
        初始参数猜测。若不提供，则在真值附近随机扰动生成。
    q0, qd0 : float
        初始状态，应与生成训练数据时一致。
    weight_q : float
        位置误差权重。
    weight_qd : float
        速度误差权重。
    method : str
        scipy 优化方法，默认 L-BFGS-B。
    maxiter : int
        最大迭代次数。
    verbose : bool
        是否打印优化过程。

    Returns
    -------
    dict
        recovered_params:  辨识出的参数
        initial_guess:     初始猜测值
        history:           每次迭代的参数与损失记录
        opt_result:        scipy 优化结果对象
    """
    current_params = sim.get_params()

    if initial_guess is None:
        # 在真实参数附近随机扰动，模拟"不知道真值"的场景
        rng = np.random.default_rng(42)
        guess = {}
        for name in PARAM_NAMES:
            true_val = current_params[name]
            # 扰动范围：真值的 0.3 到 2.5 倍
            factor = rng.uniform(0.3, 2.5)
            guess[name] = true_val * factor
        initial_guess = guess

    # 将初始猜测写入仿真器，记录初始损失
    sim.set_params(initial_guess)

    # 优化历史记录
    history = {
        "iter": [],
        "params": [],
        "loss": [],
    }

    sim_ref = sim  # 闭包引用
    tau_ref = tau_seq
    q_ref = q_true
    qd_ref = qd_true
    w_q = weight_q
    w_qd = weight_qd
    n = len(tau_seq)

    def objective(x: np.ndarray) -> float:
        """优化目标函数：运行仿真并计算轨迹误差。"""
        params = _array_to_params(x)
        sim_ref.set_params(params)

        try:
            traj = sim_ref.run(tau_ref, q0, qd0)
        except Exception:
            # 如果参数导致仿真崩溃，返回一个很大的惩罚值
            return 1e12

        q_err = traj["q"] - q_ref
        qd_err = traj["qd"] - qd_ref

        # 加权均方误差
        loss = (w_q * np.mean(q_err**2) + w_qd * np.mean(qd_err**2))

        # 记录历史
        history["iter"].append(len(history["iter"]))
        history["params"].append(params.copy())
        history["loss"].append(float(loss))

        return float(loss)

    def callback(xk: np.ndarray):
        """每次迭代后的回调，打印进度。"""
        if verbose and len(history["iter"]) > 0:
            last_loss = history["loss"][-1]
            params = _array_to_params(xk)
            print(
                f"  iter {len(history['iter']):4d}  "
                f"loss={last_loss:.6e}  "
                f"mass={params['mass']:.3f}  "
                f"damp={params['damping']:.4f}  "
                f"fric={params['frictionloss']:.4f}  "
                f"arm={params['armature']:.4f}"
            )

    x0 = _params_to_array(initial_guess)
    bounds = _get_bounds()

    if verbose:
        print("\n开始参数辨识...")
        print(f"  初始猜测: {initial_guess}")
        print(f"  真实参数: {sim_ref.get_params() if False else '(见最终对比)'}")
        print()

    result = minimize(
        objective,
        x0,
        method=method,
        bounds=bounds,
        callback=callback,
        options={"maxiter": maxiter, "disp": False},
    )

    recovered_params = _array_to_params(result.x)

    if verbose:
        print(f"\n优化完成: {result.message}")
        print(f"  最终损失: {result.fun:.6e}")
        print(f"  迭代次数: {result.nit}")

    # 恢复仿真器到辨识出的参数
    sim.set_params(recovered_params)

    return {
        "recovered_params": recovered_params,
        "initial_guess": initial_guess,
        "history": history,
        "opt_result": result,
    }
