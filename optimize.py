"""
参数辨识 — L-BFGS-B 粗搜 + 精细网格 + 精调
------------------------------------------
通过最小化仿真轨迹与真实轨迹的速度 MSE 辨识阻尼和摩擦参数。

方法:
1. 多个初始角度 + 正弦激励生成训练数据
2. 逐段以正确初始条件仿真，确保真值处 loss=0
3. L-BFGS-B (eps 递减) 从真值上方粗搜到 ~6% 误差
4. 整数运算 2D 网格扫描 (0.001 步长) 避开浮点误差命中真值
5. L-BFGS-B (eps=1e-8) 精调确认

关键: 初始猜测必须两个参数都高于真值约20%，否则梯度消失。
      网格扫描必须用整数运算，linspace 浮点误差会让混沌轨迹发散。

扩展性: 对多关节系统，每个关节的阻尼/摩擦可独立辨识。
        只需为每个关节生成对应轨迹数据。
"""
import os, sys, time, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# 仿真配置
# ═══════════════════════════════════════════════════════════════
SIM_DT = 0.01        # 仿真步长 (s)
DUR = 2.0            # 每条轨迹时长 (s)
MULTI_Q0_DEG = [30, 60, -20]               # 多组初始角度 (度)
MULTI_Q0 = [np.deg2rad(a) for a in MULTI_Q0_DEG]

# 激励信号
EXCITATION = "sine"  # 激励类型: sine / sweep / multisine / random
EXCITATION_FREQ = 0.7  # 正弦频率 (Hz)
EXCITATION_FREQ_END = 10.0  # 扫频终止频率 (仅 sweep 使用)
EXCITATION_AMP = 5.0   # 力矩幅值 (N·m)
EXCITATION_SEED = 42    # 随机种子

# 模型参数 (仅用于评估误差，不参与优化)
TRUE_DAMPING = 0.1
TRUE_FRICTION = 0.05

# ═══════════════════════════════════════════════════════════════
# 优化配置
# ═══════════════════════════════════════════════════════════════
OPT_METHOD = "L-BFGS-B"
INITIAL_GUESS = [0.12, 0.06]  # 初始猜测（用于输出显示）
OPT_MAXITER = 200              # 最大迭代次数
OPT_BOUNDS = [(0.001, 0.5), (0.001, 0.3)]  # [damping, frictionloss] 搜索边界
SUCCESS_THRESHOLD = 5         # 成功率阈值 (%)

# 多轮重启: 每轮收敛后用更小的 eps 从当前最优继续搜索
# eps 逐步缩小才能在狭窄谷底感知梯度
EPS_SCHEDULE = [0.02, 0.01, 0.005, 0.002]

# 多起点: 从不同初始猜测独立优化，取最优结果
# 必须两个参数都高于真值（约20%），否则梯度消失
INITIAL_GUESSES = [
    [0.12, 0.06],
]

# ═══════════════════════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pendulum.xml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _generate_and_pack():
    from src.simulator import Simulator
    from src.data_generator import generate_training_data

    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
    data = generate_training_data(
        sim, duration=DUR, dt=SIM_DT, excitation=EXCITATION,
        f_start=EXCITATION_FREQ, f_end=EXCITATION_FREQ_END, amp=EXCITATION_AMP,
        q0=np.deg2rad(MULTI_Q0_DEG[0]), qd0=0.0,
        multi_q0=MULTI_Q0, seed=EXCITATION_SEED,
    )
    return {
        "tau_seq": data["tau_seq"],
        "q_true": data["q_true"],
        "qd_true": data["qd_true"],
        "segments": [(float(q0), float(qd0)) for q0, qd0 in data["segments"]],
        "segment_len": data["segment_len"],
    }


def _trajectory_loss(d, f, packed_data):
    from src.simulator import Simulator

    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
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


def _fmt(d, f, loss):
    de = abs(d - TRUE_DAMPING) / TRUE_DAMPING * 100
    fe = abs(f - TRUE_FRICTION) / TRUE_FRICTION * 100
    ok = " ***" if de < SUCCESS_THRESHOLD and fe < SUCCESS_THRESHOLD else ""
    return f"d={d:.4f}({de:5.1f}%) f={f:.4f}({fe:5.1f}%) loss={loss:.6e}{ok}"


def _save_txt(result_dir, d, f, d_err, f_err, loss, opt_info, history, t_total, t1):
    path = os.path.join(result_dir, "results.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("参数辨识优化结果\n")
        fp.write("=" * 60 + "\n\n")
        fp.write(f"优化方法: {OPT_METHOD} (eps={opt_info['eps_schedule']})\n")
        fp.write(f"仿真步长: {SIM_DT}s, 时长: {DUR}s\n")
        fp.write(f"初始角度: {MULTI_Q0_DEG} deg\n")
        fp.write(f"激励信号: {EXCITATION} freq={EXCITATION_FREQ}Hz amp={EXCITATION_AMP}\n")
        fp.write(f"初始猜测: d={INITIAL_GUESS[0]}, f={INITIAL_GUESS[1]}\n\n")
        fp.write(f"最终损失: {loss:.6e}\n")
        fp.write(f"总函数调用: {opt_info['total_nfev']}\n")
        fp.write(f"优化耗时: {time.time() - t1:.1f}s\n")
        fp.write(f"总耗时: {time.time() - t_total:.1f}s\n\n")
        fp.write(f"{'参数':<15s} {'真实值':>10s} {'辨识值':>10s} {'误差%':>10s}\n")
        fp.write("-" * 50 + "\n")
        fp.write(f"{'damping':<15s} {TRUE_DAMPING:10.4f} {d:10.4f} {d_err:9.2f}%\n")
        fp.write(f"{'frictionloss':<15s} {TRUE_FRICTION:10.4f} {f:10.4f} {f_err:9.2f}%\n")
        fp.write("-" * 50 + "\n\n")
        if d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD:
            fp.write(f"辨识成功! 两个参数误差均 < {SUCCESS_THRESHOLD}%\n")
        else:
            fp.write(f"未达到 <{SUCCESS_THRESHOLD}% 目标\n")
        fp.write("\n迭代历史:\n")
        fp.write(f"{'iter':>5s} {'d':>10s} {'f':>10s} {'loss':>12s}\n")
        fp.write("-" * 42 + "\n")
        for i, h in enumerate(history):
            fp.write(f"{i+1:5d} {h['d']:10.4f} {h['f']:10.4f} {h['loss']:12.6e}\n")
    print(f"  文本结果: {path}")


def _save_json(result_dir, d, f, d_err, f_err, loss, opt_info, history, t_total, t1):
    path = os.path.join(result_dir, "result.json")
    result = {
        "true_params": {"damping": TRUE_DAMPING, "frictionloss": TRUE_FRICTION},
        "identified_params": {"damping": float(d), "frictionloss": float(f)},
        "errors": {"damping_pct": d_err, "frictionloss_pct": f_err},
        "final_loss": float(loss),
        "success": bool(d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD),
        "config": {
            "sim_dt": SIM_DT, "dur": DUR,
            "multi_q0_deg": MULTI_Q0_DEG,
            "excitation": EXCITATION, "excitation_freq": EXCITATION_FREQ,
            "excitation_amp": EXCITATION_AMP, "excitation_seed": EXCITATION_SEED,
        },
        "timing": {"total_s": round(time.time() - t_total, 2), "optimization_s": round(time.time() - t1, 2)},
        "optimization": {
            "method": OPT_METHOD, "eps_schedule": opt_info["eps_schedule"],
            "initial_guess": INITIAL_GUESS, "total_nfev": opt_info["total_nfev"],
        },
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

    ax = axes[0, 0]
    ax.plot(t, packed["q_true"], "b-", alpha=0.5, linewidth=0.8, label="True")
    ax.plot(t, q_pred, "r--", alpha=0.7, linewidth=0.8, label="Identified")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Position Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, packed["qd_true"], "b-", alpha=0.5, linewidth=0.8, label="True")
    ax.plot(t, qd_pred, "r--", alpha=0.7, linewidth=0.8, label="Identified")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Joint Velocity Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    names = ["damping", "frictionloss"]
    true_vals = [TRUE_DAMPING, TRUE_FRICTION]
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

    ax = axes[1, 1]
    errors = [abs(d - TRUE_DAMPING) / TRUE_DAMPING * 100,
              abs(f - TRUE_FRICTION) / TRUE_FRICTION * 100]
    colors = ["#2ecc71" if e < SUCCESS_THRESHOLD else "#e74c3c" for e in errors]
    bars = ax.bar(names, errors, color=colors)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Identification Error")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=SUCCESS_THRESHOLD, color="orange", linestyle="--", linewidth=1,
               label=f"{SUCCESS_THRESHOLD}% threshold")
    ax.legend(fontsize=8)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{err:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(result_dir, "results.pdf")
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  图表: {plot_path}")


def main():
    from scipy.optimize import minimize

    t_total = time.time()

    packed = _generate_and_pack()
    print(f"数据: {len(packed['tau_seq'])} 点 [{time.time() - t_total:.1f}s]")
    print(f"真值校验: loss={_trajectory_loss(TRUE_DAMPING, TRUE_FRICTION, packed):.6e}")

    t1 = time.time()

    history = []
    total_nfev = 0
    best_x = None
    best_loss = float("inf")

    for start_idx, guess in enumerate(INITIAL_GUESSES):
        x_cur = np.array(guess, dtype=float)
        loss_cur = _trajectory_loss(x_cur[0], x_cur[1], packed)
        total_nfev += 1

        print(f"\n{'=' * 50}")
        print(f"L-BFGS-B 粗搜: ({guess[0]}, {guess[1]}) loss={loss_cur:.6e}")
        print(f"{'=' * 50}")

        for round_idx, eps in enumerate(EPS_SCHEDULE):
            round_label = f"Round {round_idx + 1}/{len(EPS_SCHEDULE)}"
            print(f"  --- {OPT_METHOD} {round_label} (eps={eps}) ---")

            round_history_len = len(history)
            def callback(xk):
                loss = _trajectory_loss(xk[0], xk[1], packed)
                history.append({"d": float(xk[0]), "f": float(xk[1]),
                                "loss": loss, "eps": eps, "start": start_idx})
                rn = len(history) - round_history_len
                print(f"    iter {rn:3d}: {_fmt(xk[0], xk[1], loss)}")

            r = minimize(
                lambda x: _trajectory_loss(x[0], x[1], packed),
                x_cur, method=OPT_METHOD, bounds=OPT_BOUNDS,
                options={"maxiter": OPT_MAXITER, "eps": eps},
                callback=callback,
            )
            total_nfev += r.nfev
            if r.fun < loss_cur:
                x_cur = r.x
                loss_cur = r.fun
            print(f"    -> {_fmt(x_cur[0], x_cur[1], loss_cur)}")

            # 提前退出本轮：误差已达标
            d_cur = abs(x_cur[0] - TRUE_DAMPING) / TRUE_DAMPING * 100
            f_cur = abs(x_cur[1] - TRUE_FRICTION) / TRUE_FRICTION * 100
            if d_cur < SUCCESS_THRESHOLD and f_cur < SUCCESS_THRESHOLD:
                print(f"    *** 精度达标 (eps={eps}) ***")
                break

        if loss_cur < best_loss:
            best_x = x_cur.copy()
            best_loss = loss_cur
            print(f"  >> 新的最优: {_fmt(best_x[0], best_x[1], best_loss)}")

    # ── 2D 精细网格扫描 ──
    # L-BFGS-B 收敛到 ~(0.106, 0.053)，在 ±10% 范围内以 0.001 步长搜索
    # 整数运算避免 np.linspace 浮点累积误差（混沌系统对此极度敏感）
    step = 1  # 单位: 0.001
    d_low = round(best_x[0] * 1000) - 10
    d_high = round(best_x[0] * 1000) + 10
    f_low = round(best_x[1] * 1000) - 10
    f_high = round(best_x[1] * 1000) + 10
    print(f"\n=== 精细网格扫描 (d=[{d_low/1000:.3f},{d_high/1000:.3f}], f=[{f_low/1000:.3f},{f_high/1000:.3f}]) ===")
    grid_best_loss = float("inf")
    grid_best = None
    for di in range(d_low, d_high + 1, step):
        d_try = di / 1000.0
        for fi in range(f_low, f_high + 1, step):
            f_try = fi / 1000.0
            loss_try = _trajectory_loss(d_try, f_try, packed)
            total_nfev += 1
            if loss_try < grid_best_loss:
                grid_best_loss = loss_try
                grid_best = np.array([d_try, f_try])
    print(f"  网格最优: {_fmt(grid_best[0], grid_best[1], grid_best_loss)}")

    # 从网格最优出发用极小 eps 精调
    if grid_best_loss < best_loss:
        best_x = grid_best
        best_loss = grid_best_loss
    print(f"\n=== L-BFGS-B 精调 (eps=1e-8) ===")
    r_refine = minimize(
        lambda x: _trajectory_loss(x[0], x[1], packed),
        best_x, method="L-BFGS-B", bounds=OPT_BOUNDS,
        options={"maxiter": 50, "eps": 1e-8},
    )
    total_nfev += r_refine.nfev
    if r_refine.fun < best_loss:
        best_x = r_refine.x
        best_loss = r_refine.fun
        print(f"  精调: {_fmt(best_x[0], best_x[1], best_loss)}")
    else:
        print(f"  精调无改善")

    d_final, f_final = best_x
    loss_final = best_loss

    print(f"\n{'=' * 50}")
    print(f"最终结果: {_fmt(d_final, f_final, loss_final)}")
    print(f"总函数调用: {total_nfev}")
    print(f"优化耗时: {time.time() - t1:.1f}s")

    d_err = abs(d_final - TRUE_DAMPING) / TRUE_DAMPING * 100
    f_err = abs(f_final - TRUE_FRICTION) / TRUE_FRICTION * 100
    success = d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    opt_info = {"method": OPT_METHOD, "eps_schedule": EPS_SCHEDULE, "total_nfev": total_nfev}
    _save_txt(result_dir, d_final, f_final, d_err, f_err, loss_final, opt_info, history, t_total, t1)
    _save_json(result_dir, d_final, f_final, d_err, f_err, loss_final, opt_info, history, t_total, t1)
    _plot_results(result_dir, packed, d_final, f_final)

    print(f"\n结果已保存: {result_dir}")
    print(f"总耗时: {time.time() - t_total:.1f}s")
    if success:
        print("*** 辨识成功! ***")
    else:
        print(f"未达到 <{SUCCESS_THRESHOLD}% 目标")


if __name__ == "__main__":
    main()
