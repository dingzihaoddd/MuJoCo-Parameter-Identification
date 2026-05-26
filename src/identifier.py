"""
梯度下降参数辨识器
------------------
通过最小化仿真轨迹与真实轨迹之间的误差，辨识动力学参数。
使用 scipy.optimize.minimize (L-BFGS-B) 进行有界优化。

参数在内部被缩放到 [0, 1] 范围，以改善有限差分梯度的数值精度。
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from .simulator import Simulator


# ── 所有可辨识参数及其边界 ───────────────────────────────────

ALL_PARAM_BOUNDS = {
    "mass": (0.01, 20.0),
    "damping": (0.0, 5.0),
    "frictionloss": (0.0, 2.0),
    "armature": (0.0, 0.5),
}

PARAM_NAMES = list(ALL_PARAM_BOUNDS.keys())


def identify_params(
    sim: Simulator,
    tau_seq: np.ndarray,
    q_true: np.ndarray,
    qd_true: np.ndarray,
    param_names: list[str] | None = None,
    param_bounds: dict | None = None,
    initial_guess: dict | None = None,
    q0: float = 0.0,
    qd0: float = 0.0,
    weight_q: float = 1.0,
    weight_qd: float = 1.0,
    method: str = "differential_evolution",
    maxiter: int = 500,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """通过梯度下降从轨迹数据中辨识动力学参数。

    损失函数 = weight_q * MSE(q) + weight_qd * MSE(qd)

    参数在内部被缩放到 [0,1] 范围，确保有限差分梯度在各维度精度一致。

    Parameters
    ----------
    sim : Simulator
        仿真器实例。
    tau_seq : np.ndarray, shape (N,)
        输入力矩序列。
    q_true : np.ndarray, shape (N,)
        真实位置轨迹。
    qd_true : np.ndarray, shape (N,)
        真实速度轨迹。
    param_names : list[str] | None
        要辨识的参数名列表。默认为全部 4 个参数。
    initial_guess : dict | None
        初始参数猜测。
    q0, qd0 : float
        初始状态。
    weight_q : float
        位置误差权重。
    weight_qd : float
        速度误差权重。
    method : str
        scipy 优化方法。
    maxiter : int
        最大迭代次数。
    seed : int | None
        随机种子。
    verbose : bool
        是否打印优化过程。

    Returns
    -------
    dict
        recovered_params, initial_guess, history, opt_result
    """
    if param_names is None:
        param_names = PARAM_NAMES

    # 使用自定义边界或默认边界
    bounds_source = param_bounds if param_bounds is not None else ALL_PARAM_BOUNDS
    bounds_list = [bounds_source[name] for name in param_names]
    n_params = len(param_names)

    def params_to_array(params: dict) -> np.ndarray:
        return np.array([params[name] for name in param_names], dtype=np.float64)

    def array_to_params(arr: np.ndarray) -> dict:
        return {name: float(arr[i]) for i, name in enumerate(param_names)}

    def physical_to_scaled(physical: np.ndarray) -> np.ndarray:
        scaled = np.empty(n_params)
        for i, (lo, hi) in enumerate(bounds_list):
            scaled[i] = (physical[i] - lo) / (hi - lo)
        return scaled

    def scaled_to_physical(scaled: np.ndarray) -> np.ndarray:
        physical = np.empty(n_params)
        for i, (lo, hi) in enumerate(bounds_list):
            physical[i] = lo + scaled[i] * (hi - lo)
        return physical

    # 保存仿真器当前完整参数，只覆盖要优化的参数
    full_params = sim.get_params()

    if initial_guess is None:
        rng = np.random.default_rng(42 if seed is None else seed)
        guess = dict(full_params)
        for name in param_names:
            lo, hi = bounds_source[name]
            center = full_params[name]
            factor = np.exp(rng.normal(0, 0.5))
            guess[name] = float(np.clip(center * factor, lo, hi))
        initial_guess = guess

    # 初始猜测写入仿真器（只更新要优化的参数，其他保持原值）
    sim.set_params(initial_guess)

    history = {"iter": [], "params": [], "loss": []}
    sim_ref = sim
    tau_ref = tau_seq
    q_ref = q_true
    qd_ref = qd_true
    w_q = weight_q
    w_qd = weight_qd

    def objective_scaled(x_scaled: np.ndarray) -> float:
        x_physical = scaled_to_physical(x_scaled)
        opt_params = array_to_params(x_physical)

        current = sim_ref.get_params()
        current.update(opt_params)
        sim_ref.set_params(current)

        try:
            traj = sim_ref.run(tau_ref, q0, qd0)
        except Exception:
            return 1e12

        q_err = traj["q"] - q_ref
        qd_err = traj["qd"] - qd_ref
        loss = w_q * np.mean(q_err**2) + w_qd * np.mean(qd_err**2)

        history["iter"].append(len(history["iter"]))
        history["params"].append(current.copy())
        history["loss"].append(float(loss))
        return float(loss)

    def callback(xk_scaled: np.ndarray):
        if verbose and len(history["loss"]) > 0:
            xk = scaled_to_physical(xk_scaled)
            params = array_to_params(xk)
            parts = [f"loss={history['loss'][-1]:.6e}"]
            for i, name in enumerate(param_names):
                parts.append(f"{name}={params[name]:.4f}")
            print(f"  iter {len(history['iter']):4d}  " + "  ".join(parts))

    x0_physical = params_to_array(initial_guess)
    x0_scaled = physical_to_scaled(x0_physical)

    if verbose:
        print(f"\n开始参数辨识 ({n_params} 个参数: {param_names})...")
        print(f"  方法: {method}")
        print(f"  初始猜测: {initial_guess}")
        print()

    if method == "differential_evolution":
        bounds_scaled = [(0.0, 1.0)] * n_params
        if verbose:
            print("  运行差分进化全局搜索...")

        def objective_for_de(x_scaled):
            return objective_scaled(x_scaled)

        result_de = differential_evolution(
            objective_for_de,
            bounds_scaled,
            seed=seed if seed is not None else 42,
            maxiter=maxiter,
            popsize=15,
            polish=True,  # 用 L-BFGS-B 精细化最终结果
            disp=False,
        )

        # DE 返回的结果包装成与 minimize 兼容的格式
        result = result_de
        # 回调打印最终结果
        if verbose:
            xk = scaled_to_physical(result.x)
            params = array_to_params(xk)
            parts = [f"loss={result.fun:.6e}"]
            for i, name in enumerate(param_names):
                parts.append(f"{name}={params[name]:.4f}")
            print("  final " + "  ".join(parts))

    elif method == "Nelder-Mead":
        # Nelder-Mead 不支持 bounds，使用惩罚函数
        def objective_with_penalty(x_scaled):
            if np.any(x_scaled < 0) or np.any(x_scaled > 1):
                return 1e12
            return objective_scaled(x_scaled)

        result = minimize(
            objective_with_penalty,
            x0_scaled,
            method="Nelder-Mead",
            callback=callback,
            options={
                "maxiter": maxiter,
                "maxfev": 50000,
                "xatol": 1e-8,
                "fatol": 1e-12,
            },
        )
    elif method == "L-BFGS-B":
        bounds_scaled = [(0.0, 1.0)] * n_params
        eps_val = 0.05
        if verbose:
            print(f"  eps={eps_val:.4f}")
        result = minimize(
            objective_scaled,
            x0_scaled,
            method="L-BFGS-B",
            bounds=bounds_scaled,
            callback=callback,
            options={
                "maxiter": maxiter,
                "maxfun": 50000,
                "eps": eps_val,
            },
        )
    else:
        bounds_scaled = [(0.0, 1.0)] * n_params
        result = minimize(
            objective_scaled,
            x0_scaled,
            method=method,
            bounds=bounds_scaled,
            callback=callback,
            options={"maxiter": maxiter, "maxfun": 50000},
        )

    recovered_physical = scaled_to_physical(result.x)
    recovered_opt_params = array_to_params(recovered_physical)

    recovered_params = sim.get_params()
    recovered_params.update(recovered_opt_params)
    sim.set_params(recovered_params)

    if verbose:
        print(f"\n优化完成: {result.message}")
        print(f"  最终损失: {result.fun:.6e}")
        print(f"  迭代次数: {result.nit}")
        print(f"  函数调用: {result.nfev}")

    sim.set_params(recovered_params)

    return {
        "recovered_params": recovered_params,
        "initial_guess": initial_guess,
        "history": history,
        "opt_result": result,
    }
