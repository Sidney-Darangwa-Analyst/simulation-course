import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
TrendWave Retail - Simple Monte Carlo Simulation

This is a beginner-friendly version of the model in trendwave_simulation.py.
It does the same analysis, but keeps the code short and easy to read:
 - Part B: Baseline Monte Carlo simulation
 - Part C1: One-at-a-time (OAT) sensitivity analysis with a tornado chart
 - Part C2: Scenario comparison (baseline vs economic slowdown vs viral win)

Run with:
    python trendwave-sim.py
"""

# ----------------------------- Basic settings ----------------------------- #
np.random.seed(42)          # Fix random seed so results are repeatable
N = 10_000                  # Number of Monte Carlo iterations
DAYS = 180                  # Operating days
DAILY_STAFF = 450           # $ per day
TOTAL_STAFF = DAILY_STAFF * DAYS
MONTHS = 6                  # 6-month lease


def dollars(x):
    """Format a number as whole dollars."""
    return f"${x:,.0f}"


def simulate_profits(
    n,
    traffic_mean, traffic_std,
    conv_low, conv_mode, conv_high,
    atv_mean, atv_std,
    cogs_low, cogs_high,
    rent_low, rent_mode, rent_high,
    setup_low, setup_mode, setup_high,
    fixed=None,
):
    """
    Run a Monte Carlo simulation of profit.

    Each iteration draws ONE value for each input that represents
    the 180-day average (we do not simulate day by day).
    """
    if fixed is None:
        fixed = {}

    # Draw random inputs from the specified distributions
    traffic = np.random.normal(traffic_mean, traffic_std, n)
    traffic = np.maximum(traffic, 0)  # no negative foot traffic

    conv = np.random.triangular(conv_low, conv_mode, conv_high, n)

    atv = np.random.normal(atv_mean, atv_std, n)
    atv = np.maximum(atv, 0)  # no negative ticket size

    cogs_pct = np.random.uniform(cogs_low, cogs_high, n)

    rent_monthly = np.random.triangular(rent_low, rent_mode, rent_high, n)

    setup_cost = np.random.triangular(setup_low, setup_mode, setup_high, n)

    # Apply fixed overrides for sensitivity analysis (e.g., fixed["traffic"])
    if "traffic" in fixed:
        traffic = np.full(n, fixed["traffic"])
    if "conv" in fixed:
        conv = np.full(n, fixed["conv"])
    if "atv" in fixed:
        atv = np.full(n, fixed["atv"])
    if "cogs_pct" in fixed:
        cogs_pct = np.full(n, fixed["cogs_pct"])
    if "rent_monthly" in fixed:
        rent_monthly = np.full(n, fixed["rent_monthly"])
    if "setup_cost" in fixed:
        setup_cost = np.full(n, fixed["setup_cost"])

    # Revenue and cost calculations
    revenue = traffic * conv * atv * DAYS
    cogs = cogs_pct * revenue
    rent_total = rent_monthly * MONTHS

    profit = revenue - cogs - rent_total - TOTAL_STAFF - setup_cost
    return profit


def describe(profits):
    """Return key stats for a profit array, including a 95% CI for the mean."""
    mean = np.mean(profits)
    std = np.std(profits, ddof=1)
    p5 = np.percentile(profits, 5)
    p95 = np.percentile(profits, 95)
    prob_loss = np.mean(profits < 0) * 100

    n = len(profits)
    sem = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem

    return {
        "mean": mean,
        "std": std,
        "p5": p5,
        "p95": p95,
        "prob_loss": prob_loss,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


if __name__ == "__main__":
    # ----------------------- Baseline input settings ----------------------- #
    base = {
        "traffic_mean": 1200,
        "traffic_std": 250,
        "conv_low": 0.02,
        "conv_mode": 0.04,
        "conv_high": 0.07,
        "atv_mean": 55,
        "atv_std": 12,
        "cogs_low": 0.35,
        "cogs_high": 0.50,
        "rent_low": 10000,
        "rent_mode": 12000,
        "rent_high": 18000,
        "setup_low": 25000,
        "setup_mode": 35000,
        "setup_high": 50000,
    }

    # ===================================================================== #
    # PART B - BASELINE MONTE CARLO SIMULATION
    # ===================================================================== #
    print("=" * 65)
    print("PART B - BASELINE MONTE CARLO SIMULATION")
    print("=" * 65)

    np.random.seed(42)
    profits_base = simulate_profits(N, **base)
    stats_base = describe(profits_base)

    print(f"Mean profit:                {dollars(stats_base['mean'])}")
    print(f"Standard deviation:         {dollars(stats_base['std'])}")
    print(f"5th percentile:             {dollars(stats_base['p5'])}")
    print(f"95th percentile:            {dollars(stats_base['p95'])}")
    print(
        f"Probability of loss:        {stats_base['prob_loss']:.1f}% (profit < 0)"
    )
    print(
        "95% confidence interval for mean profit: "
        f"[{dollars(stats_base['ci_low'])}, {dollars(stats_base['ci_high'])}]"
    )

    # Histogram for baseline profit
    plt.figure(figsize=(10, 6))
    plt.hist(profits_base, bins=50, color="skyblue", edgecolor="black")
    plt.xlabel("Profit ($)")
    plt.ylabel("Frequency")
    plt.title("TrendWave Pop-Up Store — Profit Distribution (10,000 Iterations)")

    plt.axvline(stats_base["mean"], color="blue", linestyle="--", linewidth=2, label="Mean")
    plt.axvline(stats_base["p5"], color="red", linestyle="--", linewidth=2, label="5th percentile")
    plt.axvline(stats_base["p95"], color="red", linestyle="--", linewidth=2, label="95th percentile")
    plt.axvline(0, color="black", linestyle="--", linewidth=2, label="Zero profit")
    plt.legend()

    plt.tight_layout()
    plt.savefig("part_b_histogram.png", dpi=300)
    plt.show()

    # ===================================================================== #
    # PART C1 - ONE-AT-A-TIME (OAT) SENSITIVITY ANALYSIS
    # ===================================================================== #
    print("\n" + "=" * 65)
    print("PART C1 - ONE-AT-A-TIME (OAT) SENSITIVITY ANALYSIS")
    print("=" * 65)

    # Low and high values to test for each input
    ranges = [
        ("Daily foot traffic", "traffic", 700, 1700),
        ("Conversion rate", "conv", 0.02, 0.07),
        ("Avg transaction value", "atv", 31, 79),
        ("COGS %", "cogs_pct", 0.35, 0.50),
        ("Monthly rent", "rent_monthly", 10000, 18000),
        ("Setup cost", "setup_cost", 25000, 50000),
    ]

    results = []

    for name, key, low, high in ranges:
        # Note: we always reset the seed so the random noise is comparable
        np.random.seed(42)
        low_fixed = {
            "traffic": None,
            "conv": None,
            "atv": None,
            "cogs_pct": None,
            "rent_monthly": None,
            "setup_cost": None,
        }
        low_fixed[key] = low
        low_fixed = {k: v for k, v in low_fixed.items() if v is not None}
        profits_low = simulate_profits(N, **base, fixed=low_fixed)
        low_mean = np.mean(profits_low)

        np.random.seed(42)
        high_fixed = {
            "traffic": None,
            "conv": None,
            "atv": None,
            "cogs_pct": None,
            "rent_monthly": None,
            "setup_cost": None,
        }
        high_fixed[key] = high
        high_fixed = {k: v for k, v in high_fixed.items() if v is not None}
        profits_high = simulate_profits(N, **base, fixed=high_fixed)
        high_mean = np.mean(profits_high)

        spread = abs(high_mean - low_mean)
        results.append((name, low_mean, high_mean, spread))

    # Sort by spread (largest impact at the top)
    results.sort(key=lambda x: x[3], reverse=True)

    print(f"{'Input':<25} {'Low Mean Profit':>20} {'High Mean Profit':>20} {'Spread':>15}")
    print("-" * 85)
    for name, low_mean, high_mean, spread in results:
        print(
            f"{name:<25} "
            f"{dollars(low_mean):>20} "
            f"{dollars(high_mean):>20} "
            f"{dollars(spread):>15}"
        )

    # Tornado diagram (horizontal bar chart)
    labels = [r[0] for r in results]
    low_means = [r[1] for r in results]
    high_means = [r[2] for r in results]
    spreads = [r[3] for r in results]

    y_pos = np.arange(len(labels))

    # For each bar, the left end is the smaller of low and high means
    left = [min(l, h) for l, h in zip(low_means, high_means)]
    width = spreads

    # Color green if high is better, red if high is worse
    colors = ["green" if h >= l else "red" for l, h in zip(low_means, high_means)]

    plt.figure(figsize=(10, 6))
    plt.barh(y_pos, width, left=left, color=colors, edgecolor="black", alpha=0.8)
    plt.axvline(stats_base["mean"], color="black", linestyle="--", linewidth=2, label="Baseline mean")

    plt.yticks(y_pos, labels)
    plt.xlabel("Mean Profit ($)")
    plt.title("Tornado Diagram — Sensitivity Analysis")
    plt.gca().invert_yaxis()
    plt.legend()

    plt.tight_layout()
    plt.savefig("part_c1_tornado.png", dpi=300)
    plt.show()

    # ===================================================================== #
    # PART C2 - SCENARIO COMPARISON
    # ===================================================================== #
    print("\n" + "=" * 65)
    print("PART C2 - SCENARIO COMPARISON")
    print("=" * 65)

    # Baseline scenario (already simulated)
    stats_baseline = stats_base

    # Economic slowdown scenario
    econ = base.copy()
    econ["traffic_mean"] = 800
    econ["traffic_std"] = 200
    econ["conv_low"] = 0.015
    econ["conv_mode"] = 0.03
    econ["conv_high"] = 0.05
    econ["atv_mean"] = 40
    econ["atv_std"] = 10

    np.random.seed(42)
    profits_econ = simulate_profits(N, **econ)
    stats_econ = describe(profits_econ)

    # Viral marketing win scenario
    viral = base.copy()
    viral["traffic_mean"] = 1800
    viral["traffic_std"] = 300
    viral["conv_low"] = 0.03
    viral["conv_mode"] = 0.06
    viral["conv_high"] = 0.10
    # ATV stays at baseline

    np.random.seed(42)
    profits_viral = simulate_profits(N, **viral)
    stats_viral = describe(profits_viral)

    # Print comparison table
    print(
        f"{'Metric':<20} {'Baseline':>15} {'Econ Slowdown':>15} {'Viral Win':>15}"
    )
    print("-" * 70)

    print(
        f"{'Mean Profit':<20} "
        f"{dollars(stats_baseline['mean']):>15} "
        f"{dollars(stats_econ['mean']):>15} "
        f"{dollars(stats_viral['mean']):>15}"
    )

    print(
        f"{'Prob. of Loss (%)':<20} "
        f"{stats_baseline['prob_loss']:>14.1f}% "
        f"{stats_econ['prob_loss']:>14.1f}% "
        f"{stats_viral['prob_loss']:>14.1f}%"
    )

    print(
        f"{'5th Percentile':<20} "
        f"{dollars(stats_baseline['p5']):>15} "
        f"{dollars(stats_econ['p5']):>15} "
        f"{dollars(stats_viral['p5']):>15}"
    )

    print(
        f"{'95th Percentile':<20} "
        f"{dollars(stats_baseline['p95']):>15} "
        f"{dollars(stats_econ['p95']):>15} "
        f"{dollars(stats_viral['p95']):>15}"
    )

    # Overlay histogram of the three scenarios
    all_profits = np.concatenate([profits_base, profits_econ, profits_viral])
    bins = np.linspace(all_profits.min(), all_profits.max(), 60)

    plt.figure(figsize=(10, 6))
    plt.hist(profits_base, bins=bins, alpha=0.4, label="Baseline", color="blue", edgecolor="black")
    plt.hist(profits_econ, bins=bins, alpha=0.4, label="Economic Slowdown", color="red", edgecolor="black")
    plt.hist(profits_viral, bins=bins, alpha=0.4, label="Viral Marketing Win", color="green", edgecolor="black")

    plt.axvline(stats_baseline["mean"], color="blue", linestyle="--", linewidth=2, label="Baseline mean")
    plt.axvline(stats_econ["mean"], color="red", linestyle="--", linewidth=2, label="Econ slowdown mean")
    plt.axvline(stats_viral["mean"], color="green", linestyle="--", linewidth=2, label="Viral win mean")

    plt.xlabel("Profit ($)")
    plt.ylabel("Frequency")
    plt.title("Scenario Comparison — Profit Distributions")
    plt.legend()

    plt.tight_layout()
    plt.savefig("part_c2_overlay.png", dpi=300)
    plt.show()

