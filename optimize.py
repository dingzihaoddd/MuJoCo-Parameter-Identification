"""
参数辨识优化 v6 — eps=0.02 + 多起点 L-BFGS-B
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.simulator import Simulator
from src.data_generator import generate_training_data
from src.identifier import identify_params
from scipy.optimize import minimize

SIM_DT = 0.01
DUR = 2.0
MULTI_Q0 = [np.deg2rad(a) for a in [30, 60, -20]]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pendulum.xml")
TRUE = {"damping": 0.1, "frictionloss": 0.05}
BOUNDS_DICT = {"damping": (0.001, 0.5), "frictionloss": (0.001, 0.3)}
BOUNDS_LIST = [(0.001, 0.5), (0.001, 0.3)]


def _data(sim, exc="sine", f0=0.7):
    return generate_training_data(
        sim, duration=DUR, dt=SIM_DT, excitation=exc,
        f_start=f0, amp=5.0, q0=np.deg2rad(30), qd0=0.0,
        multi_q0=MULTI_Q0, seed=42,
    )


def _run(sim, data, d, f):
    sim.set_params({"damping": d, "frictionloss": f})
    qp, qdp = [], []
    for i, (q0, qd0) in enumerate(data["segments"]):
        n = data["segment_len"]
        t = sim.run(data["tau_seq"][i*n:(i+1)*n], q0, qd0)
        qp.append(t["q"]); qdp.append(t["qd"])
    return np.concatenate(qp), np.concatenate(qdp)


def _loss(sim, data, d, f):
    _, qdp = _run(sim, data, d, f)
    return float(np.mean((qdp - data["qd_true"])**2))


def fmt(d, f, loss):
    de = abs(d-0.1)/0.1*100; fe = abs(f-0.05)/0.05*100
    ok = " ***" if de < 10 and fe < 10 else ""
    return f"d={d:.4f}({de:5.1f}%) f={f:.4f}({fe:5.1f}%) loss={loss:.6e}{ok}"


def multi_start_lbfgsb(data, sim, n_starts=30, eps=0.02):
    """多起点 L-BFGS-B，返回最佳结果"""
    best_x, best_l = None, np.inf
    rng = np.random.default_rng(42)

    def obj(x):
        return _loss(sim, data, x[0], x[1])

    for _ in range(n_starts):
        x0 = [10**rng.uniform(-3, np.log10(0.5)), 10**rng.uniform(-3, np.log10(0.3))]
        r = minimize(obj, x0, method="L-BFGS-B", bounds=BOUNDS_LIST,
                     options={"maxiter": 100, "maxfun": 2000, "eps": eps})
        if r.fun < best_l:
            best_l, best_x = r.fun, r.x
            if best_l < 1e-6:  # 已找到全局最小
                break
    return best_x[0], best_x[1], best_l


def main():
    t0 = time.time()
    sim = Simulator(MODEL_PATH, timestep=SIM_DT)

    # 验证真值
    for exc in ["sine", "sweep", "random"]:
        d = _data(sim, exc, 0.7 if exc != "sweep" else 0.1)
        l = _loss(sim, d, 0.1, 0.05)
        print(f"真值校验 [{exc}]: loss={l:.6e}")

    d_sin = _data(sim, "sine", 0.7)
    d_sw = _data(sim, "sweep")
    d_rd = _data(sim, "random")

    print(f"\n数据准备: {time.time()-t0:.1f}s\n")

    all_r = []

    # ── 多起点 L-BFGS-B (eps=0.02) ──
    print("=" * 60)
    print("多起点 L-BFGS-B (eps=0.02)")
    print("=" * 60)

    for exc_name, data in [("正弦", d_sin), ("扫频", d_sw), ("随机阶跃", d_rd)]:
        t1 = time.time()
        d_val, f_val, loss = multi_start_lbfgsb(data, sim, n_starts=30, eps=0.02)
        print(f"  LBx30+{exc_name}: {fmt(d_val, f_val, loss)} [{time.time()-t1:.1f}s]")
        all_r.append((f"LBx30+{exc_name}", d_val, f_val, loss))

    # ── 测试不同 eps 值 ──
    print("\n" + "=" * 60)
    print("测试 eps 值 (正弦, 30 起点)")
    print("=" * 60)
    for eps_val in [0.005, 0.01, 0.02, 0.03, 0.05]:
        t1 = time.time()
        d_val, f_val, loss = multi_start_lbfgsb(d_sin, sim, n_starts=30, eps=eps_val)
        print(f"  eps={eps_val:.3f}: {fmt(d_val, f_val, loss)} [{time.time()-t1:.1f}s]")
        all_r.append((f"LB eps={eps_val}", d_val, f_val, loss))

    # ── DE + L-BFGS-B polish (eps=0.02) ──
    print("\n" + "=" * 60)
    print("DE + L-BFGS-B polish (eps=0.02)")
    print("=" * 60)
    from scipy.optimize import differential_evolution

    for exc_name, data in [("正弦", d_sin), ("扫频", d_sw), ("随机阶跃", d_rd)]:
        t1 = time.time()

        # 先用 DE 粗搜
        def obj(x):
            return _loss(sim, data, x[0], x[1])
        r_de = differential_evolution(obj, bounds=BOUNDS_LIST, seed=42,
                                       maxiter=30, popsize=10, polish=False, disp=False)
        d_de, f_de = r_de.x

        # 再用 L-BFGS-B (eps=0.02) 精细搜索
        r_lb = minimize(obj, r_de.x, method="L-BFGS-B", bounds=BOUNDS_LIST,
                        options={"maxiter": 100, "maxfun": 2000, "eps": 0.02})
        d_final, f_final = r_lb.x
        print(f"  DE→LB+{exc_name}: DE=({d_de:.4f},{f_de:.4f}) → LB=({d_final:.4f},{f_final:.4f}) {fmt(d_final, f_final, r_lb.fun)} [{time.time()-t1:.1f}s]")
        all_r.append((f"DE→LB+{exc_name}", d_final, f_final, r_lb.fun))

    # ── 使用 identify_params 的 L-BFGS-B（eps=0.02 已内置） ──
    print("\n" + "=" * 60)
    print("identify_params L-BFGS-B (内置 eps=0.02)")
    print("=" * 60)
    from src.identifier import identify_params
    t1 = time.time()
    r = identify_params(
        sim, tau_seq=d_sin["tau_seq"], q_true=d_sin["q_true"], qd_true=d_sin["qd_true"],
        param_names=["damping", "frictionloss"], param_bounds=BOUNDS_DICT, initial_guess=None,
        q0=0, qd0=0, weight_q=0.0, weight_qd=1.0,
        method="L-BFGS-B", maxiter=200, seed=42, verbose=False,
        segments=d_sin.get("segments"), segment_len=d_sin.get("segment_len"),
    )
    p = r["recovered_params"]
    de = abs(p["damping"]-0.1)/0.1*100; fe = abs(p["frictionloss"]-0.05)/0.05*100
    print(f"  identify_params: {fmt(p['damping'], p['frictionloss'], r['opt_result'].fun)} [{time.time()-t1:.1f}s]")
    all_r.append(("identify_params", p["damping"], p["frictionloss"], r["opt_result"].fun))

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"汇总 (总时间: {time.time()-t0:.1f}s)")
    print(f"{'='*60}")
    print(f"{'策略':<25s} {'d':>8s} {'f':>8s} {'d_err%':>7s} {'f_err%':>7s} {'loss':>10s} {'':>5s}")
    print("-" * 75)
    good = []
    for label, d, f, loss in sorted(all_r, key=lambda x: x[3]):
        de = abs(d-0.1)/0.1*100; fe = abs(f-0.05)/0.05*100
        ok = " ***" if de < 10 and fe < 10 else ""
        print(f"{label:<25s} {d:8.4f} {f:8.4f} {de:6.1f}% {fe:6.1f}% {loss:10.6e}{ok}")
        if de < 10 and fe < 10:
            good.append(label)

    print(f"\n成功: {len(good)}/{len(all_r)} 策略达到 < 10%")
    if good:
        for g in good:
            print(f"  *** {g}")


if __name__ == "__main__":
    main()
