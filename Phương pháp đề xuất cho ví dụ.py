import numpy as np
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION & DATA
# ============================================================================
RISK_FREE_RATE = 0.05
ALPHA = 0.9
STEPS = 50
SR_SCALE = 500.0  

# Dữ liệu trực tiếp (Dictionary các tuple)
DATA = {
    "STK_01": (0.8803, 0.9923, 1.1083), "STK_02": (0.8796, 1.0018, 1.0838),
    "STK_03": (0.8408, 1.0150, 1.3210), "STK_04": (0.7130, 1.0324, 1.6114),
    "STK_05": (0.7548, 1.0110, 1.3230), "STK_06": (0.7637, 1.0053, 1.2073),
    "STK_07": (0.7686, 1.0025, 1.4505), "STK_08": (0.7894, 1.0141, 1.4221),
    "STK_09": (0.8446, 0.9888, 1.1218), "STK_10": (0.7490, 1.0127, 1.2777),
    "STK_11": (0.4476, 0.9854, 1.2594), "STK_12": (0.6494, 0.9999, 1.3699),
    "STK_13": (0.7589, 1.0089, 1.3729), "STK_14": (0.7176, 1.0150, 1.2800),
    "STK_15": (0.7585, 1.0268, 1.4698), "STK_16": (0.8993, 1.0100, 1.3110),
    "STK_17": (0.8251, 0.9878, 1.0248), "STK_18": (0.8114, 1.0044, 1.2644),
    "STK_19": (0.8529, 1.0335, 1.2825), "STK_20": (0.7177, 1.0868, 1.5928)
}

# ============================================================================
# HYPERVOLUME FUNCTION
# ============================================================================
def calculate_hypervolume(pareto_front, reference_point):
    sorted_front = sorted(pareto_front, key=lambda x: x[0], reverse=True)
    hypervolume = 0.0
    
    for i in range(len(sorted_front)):
        if i == 0:
            width = reference_point[0] - sorted_front[i][0]
        else:
            width = sorted_front[i-1][0] - sorted_front[i][0]
        
        height = reference_point[1] - sorted_front[i][1]
        
        if width > 0 and height > 0:
            hypervolume += width * height
    return hypervolume

def filter_pareto(points_positive):
    pareto_front = []
    for i, p in enumerate(points_positive):
        is_dominated = False
        for j, other in enumerate(points_positive):
            if i == j:
                continue
            if (
                (other[0] >= p[0] and other[1] >= p[1])
                and (other[0] > p[0] or other[1] > p[1])
            ):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(p)
    return np.array(pareto_front)

# ============================================================================
# TRIANGULAR FUZZY NUMBER FUNCTIONS
# ============================================================================
def expected_tfn(fuzzy):
    return (fuzzy[0] + 4.0 * fuzzy[1] + fuzzy[2]) / 6.0

def variance_tfn(fuzzy):
    return (
        (fuzzy[1] - fuzzy[0]) ** 2
        + (fuzzy[2] - fuzzy[1]) ** 2
        + (fuzzy[1] - fuzzy[0]) * (fuzzy[2] - fuzzy[1])
    ) / 18.0

def value_at_risk_tfn(fuzzy, alpha=ALPHA):
    if alpha <= 0.5:
        return 2 * alpha * fuzzy[1] + (1 - 2 * alpha) * fuzzy[0]
    else:
        return (2 * alpha - 1) * fuzzy[2] + (2 - 2 * alpha) * fuzzy[1]

def sharpe_ratio_tfn(fuzzy, rf=RISK_FREE_RATE):
    var = variance_tfn(fuzzy)
    if var <= 1e-10:
        return 0.0
    raw_sr = (expected_tfn(fuzzy) - rf) / var
    
    # --- UPDATE: Chia cho hệ số tỉ lệ SR ---
    return raw_sr / SR_SCALE

def var_ratio_tfn(fuzzy, rf=RISK_FREE_RATE):
    var_val = value_at_risk_tfn(fuzzy)
    if abs(var_val) <= 1e-10:
        return 0.0
        return (expected_tfn(fuzzy) - rf) / var_val

def get_portfolio_tfn(weights, FPS):
    return (weights.reshape(-1, 1).T @ FPS).reshape(-1)

# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    print(f"--- START OPTIMIZATION (SR SCALED BY 1/{int(SR_SCALE)}) ---")
    start_time = time.time()

    # 1. Prepare Data
    tickers = list(DATA.keys())
    FPS = np.array([
        [v[0], v[1], v[2]] for v in DATA.values()
    ])
    
    print(f"✅ Loaded {len(tickers)} stocks.")

    num_assets = len(tickers)
    x0 = np.ones(num_assets) / num_assets
    bounds = [(0, 1) for _ in range(num_assets)]
    base_cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # 2. Anchor points
    print("Calculating anchor points...")
    
    # Maximize Scaled SR
    res_sr = minimize(
        lambda x: -sharpe_ratio_tfn(get_portfolio_tfn(x, FPS)),
        x0, method="SLSQP", bounds=bounds, constraints=base_cons
    )

    # Maximize Raw VR
    res_vr = minimize(
        lambda x: -var_ratio_tfn(get_portfolio_tfn(x, FPS)),
        x0, method="SLSQP", bounds=bounds, constraints=base_cons
    )

    max_sr = -res_sr.fun # Đã được scale
    max_vr = -res_vr.fun # Giá trị gốc

    vr_at_max_sr = var_ratio_tfn(get_portfolio_tfn(res_sr.x, FPS))
    sr_at_max_vr = sharpe_ratio_tfn(get_portfolio_tfn(res_vr.x, FPS))
    
    print(f"  Max SR (scaled): {max_sr:.4f}")
    print(f"  Max VR: {max_vr:.4f}")

    # 3. ε-constraint scan
    results_negative = []

    # ---- Scan VR constraint
    curr_x = res_sr.x
    for ev in np.linspace(vr_at_max_sr, max_vr, STEPS):
        cons = [
            base_cons,
            {"type": "ineq",
             "fun": lambda x, ev=ev: var_ratio_tfn(get_portfolio_tfn(x, FPS)) - ev}
        ]
        res = minimize(
            lambda x: -sharpe_ratio_tfn(get_portfolio_tfn(x, FPS)),
            curr_x, method="SLSQP", bounds=bounds, constraints=cons
        )
        if res.success:
            curr_x = res.x
            results_negative.append([
                -sharpe_ratio_tfn(get_portfolio_tfn(res.x, FPS)),
                -var_ratio_tfn(get_portfolio_tfn(res.x, FPS))
            ])

    # ---- Scan SR constraint
    curr_x = res_vr.x
    for es in np.linspace(sr_at_max_vr, max_sr, STEPS):
        cons = [
            base_cons,
            {"type": "ineq",
             "fun": lambda x, es=es: sharpe_ratio_tfn(get_portfolio_tfn(x, FPS)) - es}
        ]
        res = minimize(
            lambda x: -var_ratio_tfn(get_portfolio_tfn(x, FPS)),
            curr_x, method="SLSQP", bounds=bounds, constraints=cons
        )
        if res.success:
            curr_x = res.x
            results_negative.append([
                -sharpe_ratio_tfn(get_portfolio_tfn(res.x, FPS)),
                -var_ratio_tfn(get_portfolio_tfn(res.x, FPS))
            ])

    results_negative = np.array(results_negative)

    # 4. Post-processing
    if len(results_negative) == 0:
        print("❌ No feasible solution found.")
        return

    points_positive = -results_negative
    pareto_positive = filter_pareto(points_positive)
    pareto_positive = pareto_positive[pareto_positive[:, 0].argsort()]

    # 5. Hypervolume
    objectives = pareto_positive
    margin = 0.1
    ref_point = [
        np.max(objectives[:, 0]) + margin,
        np.max(objectives[:, 1]) + margin
    ]

    hv_value = calculate_hypervolume(objectives, ref_point)
    duration = time.time() - start_time

    # 6. Output
    print("\n" + "=" * 45)
    print(f"⏱ Runtime:        {duration:.2f} seconds")
    print(f"📊 Pareto points: {len(pareto_positive)}")
    print(f"💎 Hypervolume:   {hv_value:.10f}")
    print("=" * 45)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        points_positive[:, 0], points_positive[:, 1],
        c="gray", alpha=0.3, label="Scanned Points"
    )
    plt.plot(
        pareto_positive[:, 0], pareto_positive[:, 1],
        c="red", linestyle='-', linewidth=2, label="Pareto Front"
    )
    plt.scatter(
        pareto_positive[:, 0], pareto_positive[:, 1],
        c="red", s=30, zorder=3
    )
    
    plt.xlabel(f"Sharpe Ratio / {int(SR_SCALE)} (SR)") # <--- Nhãn trục X
    plt.ylabel("VaR Ratio (VR)")
    plt.title(f"Pareto Front: SR vs VR")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.show()

if __name__ == "__main__":
    main()