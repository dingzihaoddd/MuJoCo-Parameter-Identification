"""
Parameter identification using 1-step prediction loss + L-BFGS-B.
--------------------------------------------------------------
Single-step MuJoCo prediction from true state avoids chaos
accumulation. Uses RK4 integrator (default from XML) which handles
friction smoothly via higher-order integration.
"""
import os, sys, time, json
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# Simulation config
# =====================================================================
SIM_DT = 0.01
DUR = 2.0
MULTI_Q0_DEG = [30, 60, -20]
MULTI_Q0 = [np.deg2rad(a) for a in MULTI_Q0_DEG]

EXCITATION = "sine"
EXCITATION_FREQ = 0.7
EXCITATION_FREQ_END = 10.0
EXCITATION_AMP = 5.0
EXCITATION_SEED = 42

TRUE_DAMPING = 0.1
TRUE_FRICTION = 0.05

# =====================================================================
# Optimization config
# =====================================================================
INITIAL_GUESS = [0.12, 0.06]
OPT_MAXITER = 500
OPT_BOUNDS = [(0.001, 0.5), (0.001, 0.3)]
SUCCESS_THRESHOLD = 5

# =====================================================================
# Paths
# =====================================================================
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


def _fmt(d, f, loss):
    de = abs(d - TRUE_DAMPING) / TRUE_DAMPING * 100
    fe = abs(f - TRUE_FRICTION) / TRUE_FRICTION * 100
    ok = " ***" if de < SUCCESS_THRESHOLD and fe < SUCCESS_THRESHOLD else ""
    return f"d={d:.4f}({de:5.1f}%) f={f:.4f}({fe:5.1f}%) loss={loss:.6e}{ok}"


def _one_step_loss(d, f, packed_data):
    """1-step prediction loss from true state at each step."""
    from src.simulator import Simulator

    sim = Simulator(MODEL_PATH, timestep=SIM_DT)
    sim.set_params({"damping": d, "frictionloss": f})

    total_err = 0.0
    total_points = 0

    for seg_idx, (seg_q0, seg_qd0) in enumerate(packed_data["segments"]):
        n = packed_data["segment_len"]
        seg_start = seg_idx * n

        for i in range(n):
            abs_idx = seg_start + i
            if i == 0:
                q0 = seg_q0
                qd0 = seg_qd0
            else:
                q0 = packed_data["q_true"][abs_idx - 1]
                qd0 = packed_data["qd_true"][abs_idx - 1]

            tau_val = float(packed_data["tau_seq"][abs_idx])
            sim.reset(q0, qd0)
            sim.step(tau_val)
            err = (float(sim._data.qvel[0]) - packed_data["qd_true"][abs_idx]) ** 2
            total_err += err
            total_points += 1

    return float(total_err / total_points)


def _save_txt(result_dir, d, f, d_err, f_err, loss, opt_info, history, t_total, t1):
    path = os.path.join(result_dir, "results.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("Parameter Identification Results\n")
        fp.write("=" * 60 + "\n\n")
        fp.write(f"Method: L-BFGS-B with 1-step prediction loss\n")
        fp.write(f"Sim dt: {SIM_DT}s, dur: {DUR}s\n")
        fp.write(f"Initial angles: {MULTI_Q0_DEG} deg\n")
        fp.write(f"Excitation: {EXCITATION} freq={EXCITATION_FREQ}Hz amp={EXCITATION_AMP}\n")
        fp.write(f"Initial guess: d={INITIAL_GUESS[0]}, f={INITIAL_GUESS[1]}\n\n")
        fp.write(f"Final loss: {loss:.6e}\n")
        fp.write(f"Total function evals: {opt_info['total_nfev']}\n")
        fp.write(f"Optimization time: {time.time() - t1:.1f}s\n")
        fp.write(f"Total time: {time.time() - t_total:.1f}s\n\n")
        fp.write(f"{'Param':<15s} {'True':>10s} {'Identified':>10s} {'Error%':>10s}\n")
        fp.write("-" * 50 + "\n")
        fp.write(f"{'damping':<15s} {TRUE_DAMPING:10.4f} {d:10.4f} {d_err:9.2f}%\n")
        fp.write(f"{'frictionloss':<15s} {TRUE_FRICTION:10.4f} {f:10.4f} {f_err:9.2f}%\n")
        fp.write("-" * 50 + "\n\n")
        if d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD:
            fp.write(f"SUCCESS: both params < {SUCCESS_THRESHOLD}%\n")
        else:
            fp.write(f"FAILED: threshold {SUCCESS_THRESHOLD}% not met\n")
        fp.write("\nIteration history:\n")
        fp.write(f"{'iter':>5s} {'d':>10s} {'f':>10s} {'loss':>12s}\n")
        fp.write("-" * 42 + "\n")
        for i, h in enumerate(history):
            fp.write(f"{i+1:5d} {h['d']:10.4f} {h['f']:10.4f} {h['loss']:12.6e}\n")
    print(f"  Results txt: {path}")


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
            "method": "L-BFGS-B", "loss_type": "1-step prediction",
            "initial_guess": INITIAL_GUESS, "total_nfev": opt_info["total_nfev"],
        },
        "history": history,
    }
    with open(path, "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(f"  JSON result: {path}")


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
    print(f"  Plot: {plot_path}")


def main():
    from scipy.optimize import minimize

    t_total = time.time()

    packed = _generate_and_pack()
    print(f"Data: {len(packed['tau_seq'])} points [{time.time() - t_total:.1f}s]")
    loss_truth = _one_step_loss(TRUE_DAMPING, TRUE_FRICTION, packed)
    print(f"1-step loss at truth: {loss_truth:.6e}")

    # Data diagnostics
    all_qd = np.concatenate([packed["qd_true"][i * packed["segment_len"]:(i + 1) * packed["segment_len"]]
                             for i in range(len(packed["segments"]))])
    print(f"Velocity stats: RMS={np.sqrt(np.mean(all_qd**2)):.1f}, "
          f"|qd|<0.5: {np.sum(np.abs(all_qd)<0.5)}/{len(all_qd)}")

    t1 = time.time()
    history = []
    total_nfev = 0

    x_cur = np.array(INITIAL_GUESS, dtype=float)

    print(f"\n{'=' * 50}")
    print(f"L-BFGS-B with 1-step prediction loss")
    print(f"{'=' * 50}")
    print(f"  Start: {_fmt(x_cur[0], x_cur[1], _one_step_loss(x_cur[0], x_cur[1], packed))}")

    eps_seq = [0.02, 0.01, 0.005, 0.002, 0.001]
    for eps in eps_seq:
        round_hist_len = len(history)
        def callback(xk):
            loss = _one_step_loss(xk[0], xk[1], packed)
            history.append({"d": float(xk[0]), "f": float(xk[1]), "loss": loss, "eps": eps})
            rn = len(history) - round_hist_len
            print(f"    iter {rn:3d}: {_fmt(xk[0], xk[1], loss)}")

        r = minimize(
            lambda x: _one_step_loss(x[0], x[1], packed),
            x_cur, method="L-BFGS-B", bounds=OPT_BOUNDS,
            options={"maxiter": OPT_MAXITER, "eps": eps},
            callback=callback,
        )
        total_nfev += r.nfev
        if r.fun < _one_step_loss(x_cur[0], x_cur[1], packed):
            x_cur = r.x
        loss_cur = _one_step_loss(x_cur[0], x_cur[1], packed)
        total_nfev += 1
        print(f"    -> {_fmt(x_cur[0], x_cur[1], loss_cur)}")

    d_final = float(x_cur[0])
    f_final = float(x_cur[1])
    loss_final = _one_step_loss(d_final, f_final, packed)

    print(f"\n{'=' * 50}")
    print(f"Final: {_fmt(d_final, f_final, loss_final)}")
    print(f"Total evals: {total_nfev}")
    print(f"Optimization time: {time.time() - t1:.1f}s")

    d_err = abs(d_final - TRUE_DAMPING) / TRUE_DAMPING * 100
    f_err = abs(f_final - TRUE_FRICTION) / TRUE_FRICTION * 100
    success = d_err < SUCCESS_THRESHOLD and f_err < SUCCESS_THRESHOLD

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(result_dir, exist_ok=True)

    opt_info = {"method": "L-BFGS-B", "total_nfev": total_nfev}
    _save_txt(result_dir, d_final, f_final, d_err, f_err, loss_final, opt_info, history, t_total, t1)
    _save_json(result_dir, d_final, f_final, d_err, f_err, loss_final, opt_info, history, t_total, t1)
    _plot_results(result_dir, packed, d_final, f_final)

    print(f"\nResults saved: {result_dir}")
    print(f"Total time: {time.time() - t_total:.1f}s")
    if success:
        print("*** SUCCESS ***")
    else:
        print(f"Failed: > {SUCCESS_THRESHOLD}% error")


if __name__ == "__main__":
    main()
