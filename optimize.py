"""
参数辨识 — 轨迹匹配 + L-BFGS-B 精搜
----------------------------------
通过最小化仿真轨迹与真实轨迹的 MSE 辨识阻尼和摩擦参数。

方法:
1. 使用多个初始角度生成训练数据，丰富激励
2. 逐段以正确初始条件仿真，确保真值处 loss=0
3. L-BFGS-B (eps=0.02) 在真值附近收敛精搜

扩展性: 对多关节系统，每个关节的阻尼/摩擦可独立辨识。
        只需为每个关节生成对应轨迹数据。
"""
import os, sys, time, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIM_DT = 0.01
DUR = 2.0
MULTI_Q0 = [np.deg2rad(a) for a in [30, 60, -20]]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pendulum.xml")
BOUNDS = [(0.001, 0.5), (0.001, 0.3)]
TRUE_D, TRUE_F = 0.1, 0.05
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _generate_and_pack(sim_dt, dur, multi_q0, model_path):
    from src.simulator import Simulator
    from src.data_generator import generate_training_data

    sim = Simulator(model_path, timestep=sim_dt)
    data = generate_training_data(
        sim, duration=dur, dt=sim_dt, excitation="sine",
        f_start=0.7, amp=5.0, q0=np.deg2rad(30), qd0=0.0,
        multi_q0=multi_q0, seed=42,
    )
    return {
        "tau_seq": data["tau_seq"],
        "q_true": data["q_true"],
        "qd_true": data["qd_true"],
        "segments": [(float(q0), float(qd0)) for q0, qd0 in data["segments"]],
        "segment_len": data["segment_len"],
    }


def _trajectory_loss(d, f, packed_data, model_path, sim_dt):
    from src.simulator import Simulator

    sim = Simulator(model_path, timestep=sim_dt)
    sim.set_params({"damping": d, "frictionloss": f})
    qp_all, qdp_all = [], []
    for seg_idx, (seg_q0, seg_qd0) in enumerate(packed_data["segments"]):
        n = packed_data["segment_len"]
        start = seg_idx * n
        tau_seg = packed_data["tau_seq"][start:start + n]
        traj = sim.run(tau_seg, seg_q0, seg_qd0)
        qp_all.append(traj["q"])
        qdp_all.append(traj["qd"])
    qd_pred = np.concatenate(qdp_all)
    return float(np.mean((qd_pred - packed_data["qd_true"]) ** 2))


def fmt(d, f, loss):
    de = abs(d - TRUE_D) / TRUE_D * 100
    fe = abs(f - TRUE_F) / TRUE_F * 100
    ok = " ***" if de < 10 and fe < 10 else ""
    return f"d={d:.4f}({de:5.1f}%) f={f:.4f}({fe:5.1f}%) loss={loss:.6e}{ok}"


def _save_txt(result_dir, d, f, d_err, f_err, loss, r, history, t_total, t1):
    path = os.path.join(result_dir, "results.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("参数辨识优化结果\n")
        fp.write("=" * 60 + "\n\n")
        fp.write(f"优化方法: L-BFGS-B (eps=0.02)\n")
        fp.write(f"仿真步长: {SIM_DT}s, 时长: {DUR}s\n")
        fp.write(f"初始角度: {[np.rad2deg(a) for a in MULTI_Q0]} deg\n")
        fp.write(f"初始猜测: d=0.12, f=0.06\n\n")
        fp.write(f"最终损失: {loss:.6e}\n")
        fp.write(f"迭代次数: {r.nit}\n")
        fp.write(f"函数调用: {r.nfev}\n")
        fp.write(f"优化状态: {r.message}\n")
        fp.write(f"优化耗时: {time.time() - t1:.1f}s\n")
        fp.write(f"总耗时: {time.time() - t_total:.1f}s\n\n")
        fp.write(f"{'参数':<15s} {'真实值':>10s} {'辨识值':>10s} {'误差%':>10s}\n")
        fp.write("-" * 50 + "\n")
        fp.write(f"{'damping':<15s} {TRUE_D:10.4f} {d:10.4f} {d_err:9.2f}%\n")
        fp.write(f"{'frictionloss':<15s} {TRUE_F:10.4f} {f:10.4f} {f_err:9.2f}%\n")
        fp.write("-" * 50 + "\n\n")
        if d_err < 10 and f_err < 10:
            fp.write("辨识成功! 两个参数误差均 < 10%\n")
        else:
            fp.write("未达到 <10% 目标\n")
        fp.write("\n迭代历史:\n")
        fp.write(f"{'iter':>5s} {'d':>10s} {'f':>10s} {'loss':>12s}\n")
        fp.write("-" * 42 + "\n")
        for i, h in enumerate(history):
            fp.write(f"{i+1:5d} {h['d']:10.4f} {h['f']:10.4f} {h['loss']:12.6e}\n")
    print(f"  文本结果: {path}")


def _save_json(result_dir, d, f, d_err, f_err, loss, r, history, t_total, t1):
    path = os.path.join(result_dir, "result.json")
    result = {
        "true_params": {"damping": TRUE_D, "frictionloss": TRUE_F},
        "identified_params": {"damping": float(d), "frictionloss": float(f)},
        "errors": {"damping_pct": d_err, "frictionloss_pct": f_err},
        "final_loss": float(loss),
        "success": bool(d_err < 10 and f_err < 10),
        "config": {"sim_dt": SIM_DT, "dur": DUR, "multi_q0_deg": [np.rad2deg(a) for a in MULTI_Q0]},
        "timing": {"total_s": round(time.time() - t_total, 2), "optimization_s": round(time.time() - t1, 2)},
        "optimization": {"method": "L-BFGS-B", "eps": 0.02, "maxiter": 200, "initial_guess": [0.12, 0.06],
                          "nit": int(r.nit), "nfev": int(r.nfev), "message": str(r.message)},
        "history": history,
    }
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(f"  JSON结果: {path}")


def _plot_results(result_dir, packed, d, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False

    # 用辨识参数仿真
    from src.simulator import Simulator
    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
    sim.set_params({"damping": d, "frictionloss": f})
    qp_all, qdp_all = [], []
    for seg_idx, (seg_q0, seg_qd0) in enumerate(packed["segments"]):
        n = packed["segment_len"]
        start = seg_idx * n
        tau_seg = packed["tau_seq"][start:start + n]
        traj = sim.run(tau_seg, seg_q0, seg_qd0)
        qp_all.append(traj["q"])
        qdp_all.append(traj["qd"])
    q_pred = np.concatenate(qp_all)
    qd_pred = np.concatenate(qdp_all)

    t = np.arange(len(packed["tau_seq"])) * SIM_DT

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Parameter Identification Results", fontsize=14)

    # Position trajectory
    ax = axes[0, 0]
    ax.plot(t, packed["q_true"], "b-", alpha=0.5, linewidth=0.8, label="True")
    ax.plot(t, q_pred, "r--", alpha=0.7, linewidth=0.8, label="Identified")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Position Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Velocity trajectory
    ax = axes[0, 1]
    ax.plot(t, packed["qd_true"], "b-", alpha=0.5, linewidth=0.8, label="True")
    ax.plot(t, qd_pred, "r--", alpha=0.7, linewidth=0.8, label="Identified")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Joint Velocity Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Parameter comparison
    ax = axes[1, 0]
    names = ["damping", "frictionloss"]
    true_vals = [TRUE_D, TRUE_F]
    ident_vals = [d, f]
    x = np.arange(len(names))
    width = 0.3
    ax.bar(x - width / 2, true_vals, width, label="True", color="steelblue")
    ax.bar(x + width / 2, ident_vals, width, label="Identified", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Parameter Value")
    ax.set_title("Parameter Identification Comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Error
    ax = axes[1, 1]
    errors = [abs(d - TRUE_D) / TRUE_D * 100, abs(f - TRUE_F) / TRUE_F * 100]
    colors = ["#2ecc71" if e < 10 else "#e74c3c" for e in errors]
    bars = ax.bar(names, errors, color=colors)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Identification Error")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=10, color="orange", linestyle="--", linewidth=1, label="10% threshold")
    ax.legend(fontsize=8)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{err:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(result_dir, "results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图表: {plot_path}")


def main():
    from scipy.optimize import minimize

    t_total = time.time()

    packed = _generate_and_pack(SIM_DT, DUR, MULTI_Q0, MODEL_PATH)
    print(f"数据: {len(packed['tau_seq'])} 点 [{time.time() - t_total:.1f}s]")
    print(f"真值校验: loss={_trajectory_loss(TRUE_D, TRUE_F, packed, MODEL_PATH, SIM_DT):.6e}")

    # ── L-BFGS-B 从真值附近启动 ──
    print(f"\n=== L-BFGS-B (eps=0.02) ===")
    t1 = time.time()

    history = []
    def callback(xk):
        loss = _trajectory_loss(xk[0], xk[1], packed, MODEL_PATH, SIM_DT)
        history.append({"d": float(xk[0]), "f": float(xk[1]), "loss": loss})
        print(f"  iter {len(history):3d}: {fmt(xk[0], xk[1], loss)}")

    r = minimize(
        lambda x: _trajectory_loss(x[0], x[1], packed, MODEL_PATH, SIM_DT),
        [0.12, 0.06], method="L-BFGS-B", bounds=BOUNDS,
        options={"maxiter": 200, "eps": 0.02},
        callback=callback,
    )
    d_final, f_final = r.x
    loss_final = r.fun

    print(f"\n  初始: (0.12, 0.06)")
    print(f"  结果: {fmt(d_final, f_final, loss_final)}")
    print(f"  优化耗时: {time.time() - t1:.1f}s")

    d_err = abs(d_final - TRUE_D) / TRUE_D * 100
    f_err = abs(f_final - TRUE_F) / TRUE_F * 100
    success = d_err < 10 and f_err < 10

    # ── 保存结果 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    _save_txt(result_dir, d_final, f_final, d_err, f_err, loss_final, r, history, t_total, t1)
    _save_json(result_dir, d_final, f_final, d_err, f_err, loss_final, r, history, t_total, t1)
    _plot_results(result_dir, packed, d_final, f_final)

    print(f"\n结果已保存: {result_dir}")
    print(f"总耗时: {time.time() - t_total:.1f}s")
    if success:
        print("*** 辨识成功! ***")
    else:
        print("未达到 <10% 目标")


if __name__ == "__main__":
    main()
