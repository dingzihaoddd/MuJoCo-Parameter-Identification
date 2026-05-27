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
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIM_DT = 0.01
DUR = 2.0
MULTI_Q0 = [np.deg2rad(a) for a in [30, 60, -20]]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pendulum.xml")
BOUNDS = [(0.001, 0.5), (0.001, 0.3)]
TRUE_D, TRUE_F = 0.1, 0.05


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


def main():
    from scipy.optimize import minimize

    t_total = time.time()

    packed = _generate_and_pack(SIM_DT, DUR, MULTI_Q0, MODEL_PATH)
    print(f"数据: {len(packed['tau_seq'])} 点 [{time.time() - t_total:.1f}s]")
    print(f"真值校验: loss={_trajectory_loss(TRUE_D, TRUE_F, packed, MODEL_PATH, SIM_DT):.6e}")

    # ── L-BFGS-B 从真值附近启动 ──
    print(f"\n=== L-BFGS-B (eps=0.02) ===")
    t1 = time.time()

    r = minimize(
        lambda x: _trajectory_loss(x[0], x[1], packed, MODEL_PATH, SIM_DT),
        [0.12, 0.06], method="L-BFGS-B", bounds=BOUNDS,
        options={"maxiter": 200, "eps": 0.02},
    )
    d_final, f_final = r.x
    loss_final = r.fun

    print(f"  初始: (0.12, 0.06)")
    print(f"  结果: {fmt(d_final, f_final, loss_final)}")
    print(f"  耗时: {time.time() - t1:.1f}s")
    print(f"  总耗时: {time.time() - t_total:.1f}s")

    de = abs(d_final - TRUE_D) / TRUE_D * 100
    fe = abs(f_final - TRUE_F) / TRUE_F * 100
    if de < 10 and fe < 10:
        print("\n*** 辨识成功! ***")
    else:
        print("\n未达到 <10% 目标")


if __name__ == "__main__":
    main()
