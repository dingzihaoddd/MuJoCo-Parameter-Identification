"""
PD位置伺服 + 多短轨迹参数辨识
==============================
用PD控制器跟踪正弦/扫频参考位置轨迹，生成多条短轨迹。
优化: 所有短轨迹的MSE之和最小化，L-BFGS-B + eps schedule。
"""
import os, sys, time, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# Config
# =====================================================================
SIM_DT = 0.01
PD_KP = 30.0
PD_KD = 2.0

TRUE_DAMPING = 0.1
TRUE_FRICTION = 0.05

# 参考轨迹配置: 不同频率/幅值/初始角度，参考起始位置与q0一致
TRAJ_CONFIGS = [
    {"type": "sine",  "duration": 2.0, "amp": 0.3, "frequency": 0.3},
    {"type": "sine",  "duration": 2.0, "amp": 0.3, "frequency": 0.5},
    {"type": "sine",  "duration": 2.0, "amp": 0.3, "frequency": 0.7},
    {"type": "sine",  "duration": 2.0, "amp": 0.3, "frequency": 1.0},
    {"type": "sweep", "duration": 2.0, "amp": 0.2, "frequency": 0.1, "f_end": 3.0},
    {"type": "sine",  "duration": 2.0, "amp": 0.8, "frequency": 0.5,  "q0": np.deg2rad(30)},
    {"type": "sine",  "duration": 2.0, "amp": 0.5, "frequency": 0.7,  "q0": np.deg2rad(-20)},
    {"type": "sine",  "duration": 2.0, "amp": 1.0, "frequency": 0.3,  "q0": np.deg2rad(45)},
    # 低速轨迹: 小幅值低频，速度主要走 sign 切换区，区分 frictionloss
    {"type": "sine",  "duration": 3.0, "amp": 0.05, "frequency": 0.15},
    {"type": "sine",  "duration": 3.0, "amp": 0.03, "frequency": 0.1,  "q0": np.deg2rad(10)},
]

INITIAL_GUESS = [0.12, 0.06]
OPT_MAXITER = 2000
OPT_BOUNDS = [(0.001, 0.5), (0.001, 0.3)]
SUCCESS_THRESHOLD = 1

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pendulum.xml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# =====================================================================
# PD trajectory simulation (inline, using Simulator.step)
# =====================================================================
def _run_pd_traj(sim, q_ref, q0, qd0):
    """用PD控制器跑一条轨迹（理想速度始终为0），返回 q, qd, tau 序列。"""
    from src.simulator import Simulator
    sim.reset(q0, qd0)
    N = len(q_ref)
    q_traj = np.empty(N)
    qd_traj = np.empty(N)
    tau_traj = np.empty(N)
    for i in range(N):
        q = float(sim._data.qpos[0])
        qd = float(sim._data.qvel[0])
        tau = PD_KP * (q_ref[i] - q) - PD_KD * qd
        sim.step(tau)
        q_traj[i] = sim._data.qpos[0]
        qd_traj[i] = sim._data.qvel[0]
        tau_traj[i] = tau
    return q_traj, qd_traj, tau_traj


# =====================================================================
# Reference signal generators (position only, ideal velocity = 0)
# =====================================================================
def _sine_ref(duration, dt, amp, freq, q0=0.0):
    omega = 2 * np.pi * freq
    phase = np.arcsin(np.clip(q0 / amp, -1, 1))
    t = np.arange(0, duration, dt)
    return amp * np.sin(omega * t + phase)


def _sweep_ref(duration, dt, amp, f_start, f_end):
    t = np.arange(0, duration, dt)
    k = (f_end / f_start) ** (1.0 / duration)
    phase = 2 * np.pi * f_start * (k**t - 1) / np.log(k)
    return amp * np.sin(phase)


# =====================================================================
# Data generation
# =====================================================================
def _generate_data():
    from src.simulator import Simulator
    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
    sim.set_params({"damping": TRUE_DAMPING, "frictionloss": TRUE_FRICTION})

    trajectories = []
    for cfg in TRAJ_CONFIGS:
        dur, amp_val = cfg["duration"], cfg["amp"]
        freq = cfg.get("frequency", 0.5)
        f_end = cfg.get("f_end", 3.0)
        q0 = cfg.get("q0", 0.0)
        qd0 = cfg.get("qd0", 0.0)

        if cfg["type"] == "sine":
            q_ref = _sine_ref(dur, SIM_DT, amp_val, freq, q0)
        else:
            q_ref = _sweep_ref(dur, SIM_DT, amp_val, freq, f_end)
        qd_ref = np.zeros_like(q_ref)

        q_true, qd_true, tau_true = _run_pd_traj(sim, q_ref, q0, qd0)
        trajectories.append({
            "q_ref": q_ref, "qd_ref": qd_ref,
            "q_true": q_true, "qd_true": qd_true, "tau_true": tau_true,
            "q0": q0, "qd0": qd0, "config": cfg,
        })

    return trajectories


# =====================================================================
# Loss: 1-step prediction MSE (reset to true state each step)
# =====================================================================
def _one_step_loss(d, f, trajectories):
    """每步从真实状态重置，施加PD力矩走一步，对比预测速度与真实速度。"""
    from src.simulator import Simulator
    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
    sim.set_params({"damping": d, "frictionloss": f})

    total_err = 0.0
    total_points = 0
    for t in trajectories:
        q_ref = t["q_ref"]
        q_true = t["q_true"]
        qd_true = t["qd_true"]
        N = len(q_ref)

        for i in range(N):
            if i == 0:
                q0, qd0 = t["q0"], t["qd0"]
            else:
                q0, qd0 = q_true[i - 1], qd_true[i - 1]

            sim.reset(q0, qd0)
            tau = PD_KP * (q_ref[i] - q0) - PD_KD * qd0
            sim.step(tau)
            err = (float(sim._data.qvel[0]) - qd_true[i]) ** 2
            total_err += err
            total_points += 1

    return float(total_err / total_points)


# =====================================================================
# Output helpers
# =====================================================================
def _fmt(d, f, loss):
    de = abs(d - TRUE_DAMPING) / TRUE_DAMPING * 100
    fe = abs(f - TRUE_FRICTION) / TRUE_FRICTION * 100
    ok = " ***" if de < SUCCESS_THRESHOLD and fe < SUCCESS_THRESHOLD else ""
    return f"d={d:.4f}({de:5.1f}%) f={f:.4f}({fe:5.1f}%) loss={loss:.6e}{ok}"


def _print_diagnostics(trajectories):
    print(f"\nPD tracking (Kp={PD_KP}, Kd={PD_KD}):")
    print(f"{'#':>3s} {'type':<6s} {'f(Hz)':>6s} {'q0(°)':>7s} {'RMS err':>8s} {'max|tau|':>10s}")
    print("-" * 50)
    for i, t in enumerate(trajectories):
        cfg = t["config"]
        freq = cfg.get("frequency", cfg.get("f_start", "?"))
        rms = np.sqrt(np.mean((t["q_true"] - t["q_ref"]) ** 2))
        max_tau = np.max(np.abs(t["tau_true"]))
        q0_deg = np.rad2deg(t["q0"])
        ok = " !" if rms > 0.2 * cfg["amp"] else ""
        print(f"{i+1:3d} {cfg['type']:<6s} {freq:6.1f} {q0_deg:7.1f} {rms:8.4f} {max_tau:10.1f}{ok}")
    print()


def _save_results(result_dir, d, f, d_err, f_err, loss, opt_info, history, t_total, t1):
    path = os.path.join(result_dir, "results.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("Parameter Identification (PD Servo + Multi Trajectory)\n")
        fp.write("=" * 60 + "\n\n")
        fp.write(f"Method: L-BFGS-B, 1-step prediction MSE\n")
        fp.write(f"dt={SIM_DT}s, Kp={PD_KP}, Kd={PD_KD}\n")
        fp.write(f"Trajectories: {len(TRAJ_CONFIGS)}\n")
        fp.write(f"Initial guess: d={INITIAL_GUESS[0]}, f={INITIAL_GUESS[1]}\n\n")
        fp.write(f"Final loss: {loss:.6e}\n")
        fp.write(f"Total evals: {opt_info['total_nfev']}\n")
        fp.write(f"Opt time: {time.time() - t1:.1f}s\n")
        fp.write(f"Total time: {time.time() - t_total:.1f}s\n\n")
        fp.write(f"{'Param':<15s} {'True':>10s} {'Identified':>10s} {'Error%':>10s}\n")
        fp.write("-" * 50 + "\n")
        fp.write(f"{'damping':<15s} {TRUE_DAMPING:10.4f} {d:10.4f} {d_err:9.2f}%\n")
        fp.write(f"{'frictionloss':<15s} {TRUE_FRICTION:10.4f} {f:10.4f} {f_err:9.2f}%\n")
        fp.write("-" * 50 + "\n\n")
        if d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD:
            fp.write(f"SUCCESS: both < {SUCCESS_THRESHOLD}%\n")
        else:
            fp.write(f"FAILED\n")
        fp.write("\nIteration history:\n")
        for i, h in enumerate(history):
            fp.write(f"{i+1:5d} {h['d']:10.4f} {h['f']:10.4f} {h['loss']:12.6e}\n")
    print(f"  txt: {path}")

    jpath = os.path.join(result_dir, "result.json")
    with open(jpath, "w") as fp:
        json.dump({
            "true": {"damping": TRUE_DAMPING, "frictionloss": TRUE_FRICTION},
            "identified": {"damping": float(d), "frictionloss": float(f)},
            "errors_pct": {"damping": d_err, "frictionloss": f_err},
            "final_loss": float(loss),
            "success": bool(d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD),
            "config": {"dt": SIM_DT, "kp": PD_KP, "kd": PD_KD, "n_traj": len(TRAJ_CONFIGS)},
            "history": history,
        }, fp, indent=2, ensure_ascii=False)
    print(f"  json: {jpath}")


def _plot_results(result_dir, trajectories, d, f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False

    from src.simulator import Simulator
    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
    sim.set_params({"damping": d, "frictionloss": f})

    n = len(trajectories)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (t, ax) in enumerate(zip(trajectories, axes.flat)):
        cfg = t["config"]
        q_pred, _, _ = _run_pd_traj(sim, t["q_ref"], t["q0"], t["qd0"])
        time_axis = np.arange(len(t["q_ref"])) * SIM_DT

        ax.plot(time_axis, t["q_ref"], "k:", alpha=0.35, lw=0.8, label="ref")
        ax.plot(time_axis, t["q_true"], "b-", alpha=0.5, lw=0.8, label="true")
        ax.plot(time_axis, q_pred, "r--", alpha=0.7, lw=0.8, label="identified")

        freq = cfg.get("frequency", cfg.get("f_start", "?"))
        title = f"{cfg['type']} {freq}Hz"
        if t["q0"] != 0:
            title += f" q0={np.rad2deg(t['q0']):.0f}°"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("q (rad)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    plt.suptitle("PD Servo Parameter Identification — Per-Trajectory", fontsize=13)
    plt.tight_layout()
    ppath = os.path.join(result_dir, "results.pdf")
    plt.savefig(ppath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  plot: {ppath}")


# =====================================================================
# Main
# =====================================================================
def main():
    from scipy.optimize import minimize

    t_total = time.time()

    # 1. Generate data
    print(f"Generating {len(TRAJ_CONFIGS)} PD-servo trajectories...")
    trajectories = _generate_data()
    total_steps = sum(len(t["q_true"]) for t in trajectories)
    print(f"  {total_steps} total steps across {len(trajectories)} trajectories")

    _print_diagnostics(trajectories)

    loss_truth = _one_step_loss(TRUE_DAMPING, TRUE_FRICTION, trajectories)
    print(f"Loss at truth: {loss_truth:.6e}")

    all_qd = np.concatenate([t["qd_true"] for t in trajectories])
    print(f"Velocity: RMS={np.sqrt(np.mean(all_qd**2)):.2f} max|qd|={np.max(np.abs(all_qd)):.2f}")

    # 2. Optimize
    t1 = time.time()
    history = []
    total_nfev = 0
    x_cur = np.array(INITIAL_GUESS, dtype=float)

    print(f"\n{'='*50}")
    print(f"L-BFGS-B, 1-step prediction MSE")
    print(f"{'='*50}")
    print(f"  Start: {_fmt(x_cur[0], x_cur[1], _one_step_loss(x_cur[0], x_cur[1], trajectories))}")

    eps_seq = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    for eps in eps_seq:
        round_start = len(history)

        def cb(xk):
            loss = _one_step_loss(xk[0], xk[1], trajectories)
            history.append({"d": float(xk[0]), "f": float(xk[1]), "loss": loss, "eps": eps})
            rn = len(history) - round_start
            print(f"    {rn:3d}: {_fmt(xk[0], xk[1], loss)}")

        r = minimize(
            lambda x: _one_step_loss(x[0], x[1], trajectories),
            x_cur, method="L-BFGS-B", bounds=OPT_BOUNDS,
            options={"maxiter": OPT_MAXITER, "eps": eps, "gtol": 1e-12}, callback=cb,
        )
        total_nfev += r.nfev
        if r.fun < _one_step_loss(x_cur[0], x_cur[1], trajectories):
            x_cur = r.x
        loss_cur = _one_step_loss(x_cur[0], x_cur[1], trajectories)
        total_nfev += 1
        print(f"    -> {_fmt(x_cur[0], x_cur[1], loss_cur)}")

    # Nelder-Mead refinement for the narrow coupling valley
    print(f"\n  Nelder-Mead refinement...")
    r_nm = minimize(
        lambda x: _one_step_loss(x[0], x[1], trajectories),
        x_cur, method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-10, "fatol": 1e-20},
        callback=lambda xk: None,
    )
    if r_nm.fun < _one_step_loss(x_cur[0], x_cur[1], trajectories):
        x_cur = r_nm.x
        history.append({"d": float(x_cur[0]), "f": float(x_cur[1]), "loss": float(r_nm.fun), "eps": "NM"})
    total_nfev += r_nm.nfev
    print(f"    NM -> {_fmt(x_cur[0], x_cur[1], _one_step_loss(x_cur[0], x_cur[1], trajectories))}")

    d_final, f_final = float(x_cur[0]), float(x_cur[1])
    loss_final = _one_step_loss(d_final, f_final, trajectories)

    print(f"\n{'='*50}")
    print(f"Final: {_fmt(d_final, f_final, loss_final)}")
    print(f"Evals: {total_nfev}, opt time: {time.time()-t1:.1f}s")

    d_err = abs(d_final - TRUE_DAMPING) / TRUE_DAMPING * 100
    f_err = abs(f_final - TRUE_FRICTION) / TRUE_FRICTION * 100

    # 3. Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_DIR, ts)
    os.makedirs(result_dir, exist_ok=True)

    opt_info = {"total_nfev": total_nfev}
    _save_results(result_dir, d_final, f_final, d_err, f_err, loss_final, opt_info, history, t_total, t1)
    _plot_results(result_dir, trajectories, d_final, f_final)

    print(f"\nDone: {result_dir}")
    if d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD:
        print("*** SUCCESS ***")
    else:
        print(f"Failed: > {SUCCESS_THRESHOLD}%")


if __name__ == "__main__":
    main()
