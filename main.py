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

import numpy as np

# 确保能找到 src 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulator import Simulator
from src.data_generator import generate_training_data
from src.identifier import identify_params, PARAM_NAMES


def main():
    # ── 1. 加载模型 ──────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "pendulum.xml")
    sim = Simulator(model_path)

    # ── 2. 真实参数 ──────────────────────────────────────────
    # 这些是"真实世界"的参数值（从 XML 读取的默认值）
    true_params = sim.get_params()
    print("=" * 60)
    print("MuJoCo 动力学参数辨识 — 仿真辨识仿真")
    print("=" * 60)
    print(f"\n真实参数 (模型默认值):")
    for name in PARAM_NAMES:
        print(f"  {name:15s} = {true_params[name]:.6f}")

    # 也可以在此处覆盖真实参数（如要测试不同参数组合）:
    # sim.set_params({"mass": 2.0, "damping": 0.3, ...})
    # true_params = sim.get_params()

    # ── 3. 生成训练数据 ──────────────────────────────────────
    print("\n[1/3] 生成训练数据...")
    data = generate_training_data(
        sim,
        duration=5.0,
        dt=0.001,
        excitation="sweep",   # 扫频信号，充分激励系统
        q0=np.deg2rad(30),    # 初始角度 30°
        qd0=0.0,
        seed=42,
    )
    print(f"  采样点数: {len(data['tau_seq'])}")
    print(f"  仿真时长: {5.0}s, 步长: {0.001}s")

    # ── 4. 参数辨识 ──────────────────────────────────────────
    print("\n[2/3] 开始参数辨识 (梯度下降)...")
    result = identify_params(
        sim,
        tau_seq=data["tau_seq"],
        q_true=data["q_true"],
        qd_true=data["qd_true"],
        initial_guess=None,        # 自动生成随机初始猜测
        q0=np.deg2rad(30),
        qd0=0.0,
        weight_q=1.0,              # 位置误差权重
        weight_qd=0.1,             # 速度误差权重
        maxiter=200,
        verbose=True,
    )

    # ── 5. 结果对比 ──────────────────────────────────────────
    print("\n[3/3] 结果对比")
    print("-" * 60)
    print(f"{'参数':<15s} {'真实值':>10s} {'辨识值':>10s} {'误差%':>10s}")
    print("-" * 60)
    for name in PARAM_NAMES:
        true_val = true_params[name]
        recovered_val = result["recovered_params"][name]
        error_pct = abs(recovered_val - true_val) / true_val * 100
        print(
            f"{name:<15s} {true_val:10.4f} {recovered_val:10.4f} {error_pct:9.2f}%"
        )
    print("-" * 60)

    final_loss = result["opt_result"].fun
    print(f"\n最终损失: {final_loss:.6e}")
    print(f"迭代次数: {result['opt_result'].nit}")
    print(f"优化状态: {result['opt_result'].message}")

    # ── 6. 可选：绘制结果 ────────────────────────────────────
    try:
        _plot_results(data, result, sim, true_params)
        print("\n图表已生成。")
    except Exception as e:
        print(f"\n(图表生成跳过: {e})")

    print("\n完成!")


def _plot_results(data, result, sim, true_params):
    """绘制辨识前后的轨迹对比和损失曲线。"""
    import matplotlib.pyplot as plt

    # 用辨识出的参数重新仿真，得到预测轨迹
    sim.set_params(result["recovered_params"])
    pred = sim.run(data["tau_seq"], q0=np.deg2rad(30), qd0=0.0)

    t = np.arange(len(data["tau_seq"])) * data["dt"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("MuJoCo 动力学参数辨识结果", fontsize=14)

    # ── 轨迹对比 ──
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
    ax.set_title("输入力矩 (扫频信号)")
    ax.grid(True, alpha=0.3)

    # ── 损失曲线 ──
    ax = axes[1, 0]
    iters = result["history"]["iter"]
    losses = result["history"]["loss"]
    ax.semilogy(iters, losses, "k.-", markersize=3)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("损失 (log)")
    ax.set_title("损失下降曲线")
    ax.grid(True, alpha=0.3)

    # ── 参数柱状图 ──
    ax = axes[1, 1]
    x = np.arange(len(PARAM_NAMES))
    width = 0.3
    true_vals = [true_params[n] for n in PARAM_NAMES]
    recovered_vals = [result["recovered_params"][n] for n in PARAM_NAMES]
    bars1 = ax.bar(x - width / 2, true_vals, width, label="真实值", color="steelblue")
    bars2 = ax.bar(
        x + width / 2, recovered_vals, width, label="辨识值", color="coral"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES)
    ax.set_ylabel("参数值")
    ax.set_title("参数辨识结果对比")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 误差明细 ──
    ax = axes[1, 2]
    errors = [
        abs(result["recovered_params"][n] - true_params[n])
        / true_params[n]
        * 100
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
    plot_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results.png"
    )
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  结果图保存至: {plot_path}")


if __name__ == "__main__":
    main()
