"""
MuJoCo 动力学参数辨识 — 仿真辨识仿真
=====================================

流程:
  1. 在 MuJoCo 中设定"真实"动力学参数
  2. 施加已知力矩输入，记录位置/速度轨迹（模拟采集实验数据）
  3. 用一组偏离真值的初始猜测参数，运行相同的力矩输入
  4. 通过梯度下降最小化轨迹误差，逐步逼近真实参数
  5. 对比辨识结果与真值

用法:
  python main.py
"""

import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulator import Simulator
from src.data_generator import generate_training_data
from src.identifier import identify_params, PARAM_NAMES

SIM_DT = 0.005
SIM_DURATION = 5.0
N_RESTARTS = 1
OPT_PARAMS = ["damping", "frictionloss"]
# 缩小搜索边界，聚焦在实际可能的范围
OPT_BOUNDS = {
    "damping": (0.001, 1.0),
    "frictionloss": (0.001, 0.5),
}


def main():
    # ── 结果目录 ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", timestamp)
    os.makedirs(result_dir, exist_ok=True)
    print(f"结果目录: {result_dir}")

    # ── 1. 加载模型 ──────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "pendulum.xml")
    sim = Simulator(model_path, timestep=SIM_DT)

    # ── 2. 真实参数 ──────────────────────────────────────────
    true_params = sim.get_params()
    print("=" * 60)
    print("MuJoCo 动力学参数辨识 — 仿真辨识仿真")
    print("=" * 60)
    print(f"\n真实参数 (模型默认值):")
    for name in PARAM_NAMES:
        print(f"  {name:15s} = {true_params[name]:.6f}")

    # ── 3. 生成训练数据 (正弦激励，共振频率) ─────────────────
    print(f"\n[1/3] 生成训练数据 (步长={SIM_DT}s, 时长={SIM_DURATION}s)...")
    data = generate_training_data(
        sim,
        duration=SIM_DURATION,
        dt=SIM_DT,
        excitation="sine",
        f_start=0.7,
        amp=5.0,
        q0=np.deg2rad(30),
        qd0=0.0,
        multi_q0=[np.deg2rad(a) for a in [10, 30, 60, -20, -45]],
        seed=42,
    )
    print(f"  采样点数: {len(data['tau_seq'])} (5 个初始角度 x {SIM_DURATION}s)")
    print(f"  激励类型: 正弦 0.7 Hz, 多初始角度 [10, 30, 60, -20, -45] deg")

    # ── 4. 参数辨识 (多次随机重启) ──────────────────────────
    print(f"\n[2/3] 开始参数辨识 ({N_RESTARTS} 次随机重启)...")
    best_result = None
    best_loss = np.inf

    for trial in range(N_RESTARTS):
        seed = 100 + trial * 17
        result = identify_params(
            sim,
            tau_seq=data["tau_seq"],
            q_true=data["q_true"],
            qd_true=data["qd_true"],
            param_names=OPT_PARAMS,
            param_bounds=OPT_BOUNDS,
            initial_guess=None,
            q0=np.deg2rad(30),
            qd0=0.0,
            weight_q=1.0,
            weight_qd=0.01,
            maxiter=50,
            seed=seed,
            verbose=False,
            segments=data.get("segments"),
            segment_len=data.get("segment_len"),
        )
        loss = result["opt_result"].fun
        params = result["recovered_params"]
        print(
            f"  restart {trial+1}/{N_RESTARTS}: seed={seed}, loss={loss:.6e}, "
            f"damp={params['damping']:.4f}, fric={params['frictionloss']:.4f}"
        )
        if loss < best_loss:
            best_loss = loss
            best_result = result

    print(f"\n  最佳结果: loss={best_loss:.6e}")
    result = best_result

    # ── 5. 结果对比 ──────────────────────────────────────────
    print("\n[3/3] 结果对比")
    print("-" * 60)
    print(f"{'参数':<15s} {'真实值':>10s} {'辨识值':>10s} {'误差%':>10s}")
    print("-" * 60)
    for name in PARAM_NAMES:
        true_val = true_params[name]
        recovered_val = result["recovered_params"][name]
        error_pct = abs(recovered_val - true_val) / true_val * 100
        marker = " *" if name in OPT_PARAMS else ""
        print(
            f"{name:<15s} {true_val:10.4f} {recovered_val:10.4f} {error_pct:9.2f}%{marker}"
        )
    print("-" * 60)

    final_loss = result["opt_result"].fun
    print(f"\n最终损失: {final_loss:.6e}")
    print(f"迭代次数: {result['opt_result'].nit}")
    print(f"函数调用: {result['opt_result'].nfev}")
    print(f"优化状态: {result['opt_result'].message}")

    # ── 6. 保存结果到文件 ────────────────────────────────────
    _save_results(result_dir, data, result, sim, true_params)

    # ── 7. 绘制结果 ──────────────────────────────────────────
    try:
        _plot_results(result_dir, data, result, sim, true_params)
        print("\n图表已生成。")
    except Exception as e:
        print(f"\n(图表生成跳过: {e})")

    print(f"\n完成! 结果保存在: {result_dir}")


def _save_results(result_dir, data, result, sim, true_params):
    """保存数值结果到文本文件。"""
    path = os.path.join(result_dir, "results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("MuJoCo 动力学参数辨识结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"优化参数: {OPT_PARAMS}\n")
        f.write(f"最终损失: {result['opt_result'].fun:.6e}\n")
        f.write(f"迭代次数: {result['opt_result'].nit}\n")
        f.write(f"函数调用: {result['opt_result'].nfev}\n")
        f.write(f"优化状态: {result['opt_result'].message}\n\n")
        f.write(f"{'参数':<15s} {'真实值':>10s} {'辨识值':>10s} {'误差%':>10s}\n")
        f.write("-" * 60 + "\n")
        for name in PARAM_NAMES:
            true_val = true_params[name]
            recovered_val = result["recovered_params"][name]
            error_pct = abs(recovered_val - true_val) / true_val * 100
            marker = " *" if name in OPT_PARAMS else ""
            f.write(f"{name:<15s} {true_val:10.4f} {recovered_val:10.4f} {error_pct:9.2f}%{marker}\n")
    print(f"  结果文本保存至: {path}")


def _plot_results(result_dir, data, result, sim, true_params):
    """绘制辨识前后的轨迹对比和损失曲线。"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sim.set_params(result["recovered_params"])
    # 多段数据：分别用每段初始条件仿真
    if "segments" in data and data["segments"] is not None:
        q_list, qd_list = [], []
        for seg_idx, (seg_q0, seg_qd0) in enumerate(data["segments"]):
            n = data["segment_len"]
            seg_tau = data["tau_seq"][seg_idx*n:(seg_idx+1)*n]
            traj = sim.run(seg_tau, seg_q0, seg_qd0)
            q_list.append(traj["q"])
            qd_list.append(traj["qd"])
        q_pred = np.concatenate(q_list)
        qd_pred = np.concatenate(qd_list)
        pred = {"q": q_pred, "qd": qd_pred}
    else:
        pred = sim.run(data["tau_seq"], q0=np.deg2rad(30), qd0=0.0)

    t = np.arange(len(data["tau_seq"])) * data["dt"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("MuJoCo 动力学参数辨识结果", fontsize=14)

    ax = axes[0, 0]
    ax.plot(t, data["q_true"], "b-", alpha=0.7, linewidth=1, label="真实")
    ax.plot(t, pred["q"], "r--", alpha=0.7, linewidth=1, label="辨识后")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("位置 (rad)")
    ax.set_title("关节位置轨迹对比")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, data["qd_true"], "b-", alpha=0.7, linewidth=1, label="真实")
    ax.plot(t, pred["qd"], "r--", alpha=0.7, linewidth=1, label="辨识后")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("速度 (rad/s)")
    ax.set_title("关节速度轨迹对比")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(t, data["tau_seq"], "g-", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("力矩 (N·m)")
    ax.set_title("输入力矩 (正弦 0.7 Hz)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    iters = result["history"]["iter"]
    losses = result["history"]["loss"]
    ax.semilogy(iters, losses, "k.-", markersize=3)
    ax.set_xlabel("函数评估次数")
    ax.set_ylabel("损失 (log)")
    ax.set_title("损失下降曲线")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    x = np.arange(len(PARAM_NAMES))
    width = 0.3
    true_vals = [true_params[n] for n in PARAM_NAMES]
    recovered_vals = [result["recovered_params"][n] for n in PARAM_NAMES]
    ax.bar(x - width / 2, true_vals, width, label="真实值", color="steelblue")
    ax.bar(x + width / 2, recovered_vals, width, label="辨识值", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel("参数值")
    ax.set_title("参数辨识结果对比")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 2]
    errors = [
        abs(result["recovered_params"][n] - true_params[n]) / true_params[n] * 100
        for n in PARAM_NAMES
    ]
    bars = ax.bar(PARAM_NAMES, errors, color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"])
    ax.set_ylabel("相对误差 (%)")
    ax.set_title("各参数辨识误差")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, err in zip(bars, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{err:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plot_path = os.path.join(result_dir, "results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  结果图保存至: {plot_path}")


if __name__ == "__main__":
    main()
