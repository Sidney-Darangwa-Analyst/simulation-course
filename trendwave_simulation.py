import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.patches as mpatches


# =============================================================================
# TrendWave Retail — Pop-Up Store Monte Carlo Simulation
# =============================================================================
# This script evaluates the profitability of a 180-day pop-up store at
# Riverside Mall using Monte Carlo simulation, one-at-a-time sensitivity
# analysis (tornado diagram), and scenario comparison.
#
# Run end-to-end with:
#   python trendwave_simulation.py
# =============================================================================

# Global random seed (required: set before any draws)
np.random.seed(42)


# =============================================================================
# Global Constants and Helper Functions
# =============================================================================

OPERATING_DAYS = 180
DAILY_STAFF_COST = 450
TOTAL_STAFFING_COST = DAILY_STAFF_COST * OPERATING_DAYS
MONTHS = 6
N_ITER = 10_000


def fmt_dollar(x: float) -> str:
    """Format a numeric value as a whole-dollar currency string."""
    return f"${x:,.0f}"


def simulate_profit(
    n_iter: int,
    traffic_params: dict,
    conv_params: dict,
    atv_params: dict,
    cogs_params: dict,
    rent_params: dict,
    setup_params: dict,
    fixed_overrides: dict | None = None,
) -> np.ndarray:
    """
    Simulate profit outcomes for the TrendWave pop-up store.

    Each iteration uses a single draw per input to represent the 180-day
    average, not day-by-day variation.

    Parameters
    ----------
    n_iter : int
        Number of Monte Carlo iterations.
    traffic_params, conv_params, atv_params, cogs_params, rent_params,
    setup_params : dict
        Distribution parameters for each uncertain input.
    fixed_overrides : dict or None
        Optional mapping from input name to fixed value for sensitivity
        analysis (e.g., {"foot_traffic": 700}).

    Returns
    -------
    np.ndarray
        Array of simulated profit values.
    """
    if fixed_overrides is None:
        fixed_overrides = {}

    # --- Draw uncertain inputs (baseline stochastic behavior) ---

    # Daily foot traffic — Normal, truncated at zero to avoid negatives.
    daily_traffic = np.random.normal(
        loc=traffic_params["mean"],
        scale=traffic_params["std"],
        size=n_iter,
    )
    daily_traffic = np.maximum(daily_traffic, 0.0)

    # Conversion rate — Triangular between [low, high].
    conversion_rate = np.random.triangular(
        left=conv_params["low"],
        mode=conv_params["mode"],
        right=conv_params["high"],
        size=n_iter,
    )

    # Average transaction value — Normal, truncated at zero.
    avg_transaction_value = np.random.normal(
        loc=atv_params["mean"],
        scale=atv_params["std"],
        size=n_iter,
    )
    avg_transaction_value = np.maximum(avg_transaction_value, 0.0)

    # COGS % of revenue — Uniform.
    cogs_pct = np.random.uniform(
        low=cogs_params["low"],
        high=cogs_params["high"],
        size=n_iter,
    )

    # Monthly rent — Triangular.
    monthly_rent = np.random.triangular(
        left=rent_params["low"],
        mode=rent_params["mode"],
        right=rent_params["high"],
        size=n_iter,
    )

    # Setup cost — one-time Triangular.
    setup_cost = np.random.triangular(
        left=setup_params["low"],
        mode=setup_params["mode"],
        right=setup_params["high"],
        size=n_iter,
    )

    # --- Apply any fixed overrides for sensitivity analysis ---
    if "foot_traffic" in fixed_overrides:
        daily_traffic = np.full(n_iter, fixed_overrides["foot_traffic"], dtype=float)

    if "conversion_rate" in fixed_overrides:
        conversion_rate = np.full(n_iter, fixed_overrides["conversion_rate"], dtype=float)

    if "avg_transaction_value" in fixed_overrides:
        avg_transaction_value = np.full(
            n_iter, fixed_overrides["avg_transaction_value"], dtype=float
        )

    if "cogs_pct" in fixed_overrides:
        cogs_pct = np.full(n_iter, fixed_overrides["cogs_pct"], dtype=float)

    if "monthly_rent" in fixed_overrides:
        monthly_rent = np.full(n_iter, fixed_overrides["monthly_rent"], dtype=float)

    if "setup_cost" in fixed_overrides:
        setup_cost = np.full(n_iter, fixed_overrides["setup_cost"], dtype=float)

    # --- Core profit logic ---

    # Total revenue over 180 days.
    total_revenue = daily_traffic * conversion_rate * avg_transaction_value * OPERATING_DAYS

    # Cost of goods sold as a % of revenue.
    cogs = cogs_pct * total_revenue

    # Total rent over the 6-month lease.
    total_rent = monthly_rent * MONTHS

    # Staffing is fixed.
    total_staffing = TOTAL_STAFFING_COST

    # Profit = Revenue - COGS - Rent - Staffing - Setup Cost
    profit = total_revenue - cogs - total_rent - total_staffing - setup_cost

    return profit


def summarize_results(profit_samples: np.ndarray) -> dict:
    """
    Compute summary statistics for a profit distribution, including
    a 95% confidence interval for the mean using the t-distribution.
    """
    mean_profit = np.mean(profit_samples)
    std_profit = np.std(profit_samples, ddof=1)
    p5 = np.percentile(profit_samples, 5)
    p95 = np.percentile(profit_samples, 95)
    prob_loss = np.mean(profit_samples < 0) * 100.0  # percentage

    n = len(profit_samples)
    sem = std_profit / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_profit - t_crit * sem
    ci_high = mean_profit + t_crit * sem

    return {
        "mean": mean_profit,
        "std": std_profit,
        "p5": p5,
        "p95": p95,
        "prob_loss": prob_loss,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def run_scenario(
    name: str,
    n_iter: int,
    traffic_params: dict,
    conv_params: dict,
    atv_params: dict,
    cogs_params: dict,
    rent_params: dict,
    setup_params: dict,
) -> dict:
    """
    Wrapper to run a full Monte Carlo scenario and return both the raw
    profit samples and summary statistics.
    """
    # Use the same seed per scenario for comparability.
    np.random.seed(42)
    profits = simulate_profit(
        n_iter=n_iter,
        traffic_params=traffic_params,
        conv_params=conv_params,
        atv_params=atv_params,
        cogs_params=cogs_params,
        rent_params=rent_params,
        setup_params=setup_params,
    )
    stats_dict = summarize_results(profits)
    stats_dict["profits"] = profits
    stats_dict["name"] = name
    return stats_dict


if __name__ == "__main__":
    # =========================================================================
    # Baseline Distribution Parameters
    # =========================================================================
    baseline_traffic = {"mean": 1200, "std": 250}
    baseline_conv = {"low": 0.02, "mode": 0.04, "high": 0.07}
    baseline_atv = {"mean": 55, "std": 12}
    baseline_cogs = {"low": 0.35, "high": 0.50}
    baseline_rent = {"low": 10000, "mode": 12000, "high": 18000}
    baseline_setup = {"low": 25000, "mode": 35000, "high": 50000}

    # =========================================================================
    # PART B — BASELINE MONTE CARLO SIMULATION
    # =========================================================================
    print("=================================================================")
    print("PART B — BASELINE MONTE CARLO SIMULATION")
    print("=================================================================")

    # Reset seed as specified (seed=42, n=10,000).
    np.random.seed(42)
    baseline_profits = simulate_profit(
        n_iter=N_ITER,
        traffic_params=baseline_traffic,
        conv_params=baseline_conv,
        atv_params=baseline_atv,
        cogs_params=baseline_cogs,
        rent_params=baseline_rent,
        setup_params=baseline_setup,
    )
    baseline_stats = summarize_results(baseline_profits)

    print(f"Mean profit:                {fmt_dollar(baseline_stats['mean'])}")
    print(f"Standard deviation:         {fmt_dollar(baseline_stats['std'])}")
    print(f"5th percentile:             {fmt_dollar(baseline_stats['p5'])}")
    print(f"95th percentile:            {fmt_dollar(baseline_stats['p95'])}")
    print(
        f"Probability of loss:        {baseline_stats['prob_loss']:.1f}% "
        f"(profit < 0)"
    )
    print(
        "95% confidence interval for mean profit: "
        f"[{fmt_dollar(baseline_stats['ci_low'])}, "
        f"{fmt_dollar(baseline_stats['ci_high'])}]"
    )

    # Histogram of baseline profit distribution.
    fig_b, ax_b = plt.subplots(figsize=(10, 6))
    ax_b.hist(baseline_profits, bins=50, color="skyblue", edgecolor="black")
    ax_b.set_xlabel("Profit ($)")
    ax_b.set_ylabel("Frequency")
    ax_b.set_title(
        "TrendWave Pop-Up Store — Profit Distribution (10,000 Iterations)"
    )

    # Vertical reference lines: mean, 5th, 95th percentiles, and zero.
    ax_b.axvline(
        baseline_stats["mean"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    ax_b.axvline(
        baseline_stats["p5"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="5th Percentile",
    )
    ax_b.axvline(
        baseline_stats["p95"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="95th Percentile",
    )
    ax_b.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Break-even (0)",
    )
    ax_b.legend()
    plt.tight_layout()
    plt.savefig("part_b_histogram.png", dpi=300)
    plt.show()

    # =========================================================================
    # PART C.1 — ONE-AT-A-TIME (OAT) SENSITIVITY ANALYSIS
    # =========================================================================
    print("\n=================================================================")
    print("PART C.1 — ONE-AT-A-TIME (OAT) SENSITIVITY ANALYSIS")
    print("=================================================================")

    sensitivity_specs = [
        {
            "name": "Daily foot traffic",
            "key": "foot_traffic",
            "low": 700,
            "high": 1700,
        },
        {
            "name": "Conversion rate",
            "key": "conversion_rate",
            "low": 0.02,
            "high": 0.07,
        },
        {
            "name": "Avg transaction value",
            "key": "avg_transaction_value",
            "low": 31,
            "high": 79,
        },
        {
            "name": "COGS %",
            "key": "cogs_pct",
            "low": 0.35,
            "high": 0.50,
        },
        {
            "name": "Monthly rent",
            "key": "monthly_rent",
            "low": 10000,
            "high": 18000,
        },
        {
            "name": "Setup cost",
            "key": "setup_cost",
            "low": 25000,
            "high": 50000,
        },
    ]

    oat_results = []

    for spec in sensitivity_specs:
        # Fix input at low value; all other inputs remain stochastic.
        np.random.seed(42)
        low_profits = simulate_profit(
            n_iter=N_ITER,
            traffic_params=baseline_traffic,
            conv_params=baseline_conv,
            atv_params=baseline_atv,
            cogs_params=baseline_cogs,
            rent_params=baseline_rent,
            setup_params=baseline_setup,
            fixed_overrides={spec["key"]: spec["low"]},
        )
        low_mean = np.mean(low_profits)

        # Fix input at high value; all other inputs remain stochastic.
        np.random.seed(42)
        high_profits = simulate_profit(
            n_iter=N_ITER,
            traffic_params=baseline_traffic,
            conv_params=baseline_conv,
            atv_params=baseline_atv,
            cogs_params=baseline_cogs,
            rent_params=baseline_rent,
            setup_params=baseline_setup,
            fixed_overrides={spec["key"]: spec["high"]},
        )
        high_mean = np.mean(high_profits)

        spread = abs(high_mean - low_mean)

        oat_results.append(
            {
                "input": spec["name"],
                "low_mean": low_mean,
                "high_mean": high_mean,
                "spread": spread,
            }
        )

    # Sort by spread descending for tornado diagram (largest at top).
    oat_results_sorted = sorted(oat_results, key=lambda x: x["spread"], reverse=True)

    # Print formatted sensitivity table.
    print(f"{'Input':<25} {'Low Mean Profit':>20} {'High Mean Profit':>20} {'Spread':>15}")
    print("-" * 85)
    for row in oat_results_sorted:
        print(
            f"{row['input']:<25} "
            f"{fmt_dollar(row['low_mean']):>20} "
            f"{fmt_dollar(row['high_mean']):>20} "
            f"{fmt_dollar(row['spread']):>15}"
        )

    # Tornado diagram.
    labels = [row["input"] for row in oat_results_sorted]
    spreads = [row["spread"] for row in oat_results_sorted]
    low_means = [row["low_mean"] for row in oat_results_sorted]
    high_means = [row["high_mean"] for row in oat_results_sorted]

    y_pos = np.arange(len(labels))

    fig_t, ax_t = plt.subplots(figsize=(10, 6))

    # Create horizontal bars spanning from low_mean to high_mean.
    colors = []
    for low, high in zip(low_means, high_means):
        # Green if higher value yields better profit, otherwise red.
        if high >= low:
            colors.append("green")
        else:
            colors.append("red")

    bar_left = [min(l, h) for l, h in zip(low_means, high_means)]
    bar_width = spreads

    ax_t.barh(
        y_pos,
        bar_width,
        left=bar_left,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    ax_t.axvline(
        baseline_stats["mean"],
        color="black",
        linestyle="--",
        linewidth=2,
        label="Baseline mean profit",
    )

    ax_t.set_yticks(y_pos)
    ax_t.set_yticklabels(labels)
    ax_t.set_xlabel("Mean Profit ($)")
    ax_t.set_title("Tornado Diagram — Sensitivity Analysis")
    ax_t.invert_yaxis()  # Largest spread at the top.

    # Legend indicating direction of impact.
    green_patch = mpatches.Patch(color="green", label="Higher value improves profit")
    red_patch = mpatches.Patch(color="red", label="Higher value reduces profit")
    ax_t.legend(handles=[green_patch, red_patch])

    plt.tight_layout()
    plt.savefig("part_c1_tornado.png", dpi=300)
    plt.show()

    # =========================================================================
    # PART C.2 — SCENARIO COMPARISON
    # =========================================================================
    print("\n=================================================================")
    print("PART C.2 — SCENARIO COMPARISON")
    print("=================================================================")

    # Baseline scenario (original distributions).
    baseline_scenario = run_scenario(
        name="Baseline",
        n_iter=N_ITER,
        traffic_params=baseline_traffic,
        conv_params=baseline_conv,
        atv_params=baseline_atv,
        cogs_params=baseline_cogs,
        rent_params=baseline_rent,
        setup_params=baseline_setup,
    )

    # Economic Slowdown scenario.
    econ_traffic = {"mean": 800, "std": 200}
    econ_conv = {"low": 0.015, "mode": 0.03, "high": 0.05}
    econ_atv = {"mean": 40, "std": 10}
    econ_scenario = run_scenario(
        name="Economic Slowdown",
        n_iter=N_ITER,
        traffic_params=econ_traffic,
        conv_params=econ_conv,
        atv_params=econ_atv,
        cogs_params=baseline_cogs,
        rent_params=baseline_rent,
        setup_params=baseline_setup,
    )

    # Viral Marketing Win scenario.
    viral_traffic = {"mean": 1800, "std": 300}
    viral_conv = {"low": 0.03, "mode": 0.06, "high": 0.10}
    viral_scenario = run_scenario(
        name="Viral Marketing Win",
        n_iter=N_ITER,
        traffic_params=viral_traffic,
        conv_params=viral_conv,
        atv_params=baseline_atv,
        cogs_params=baseline_cogs,
        rent_params=baseline_rent,
        setup_params=baseline_setup,
    )

    scenarios = [baseline_scenario, econ_scenario, viral_scenario]

    # Console comparison table.
    print(
        f"{'Metric':<20} {'Baseline':>15} {'Econ Slowdown':>15} "
        f"{'Viral Win':>15}"
    )
    print("-" * 70)

    # Mean Profit
    print(
        f"{'Mean Profit':<20} "
        f"{fmt_dollar(baseline_scenario['mean']):>15} "
        f"{fmt_dollar(econ_scenario['mean']):>15} "
        f"{fmt_dollar(viral_scenario['mean']):>15}"
    )

    # Probability of Loss
    print(
        f"{'Prob. of Loss (%)':<20} "
        f"{baseline_scenario['prob_loss']:>14.1f}% "
        f"{econ_scenario['prob_loss']:>14.1f}% "
        f"{viral_scenario['prob_loss']:>14.1f}%"
    )

    # 5th Percentile
    print(
        f"{'5th Percentile':<20} "
        f"{fmt_dollar(baseline_scenario['p5']):>15} "
        f"{fmt_dollar(econ_scenario['p5']):>15} "
        f"{fmt_dollar(viral_scenario['p5']):>15}"
    )

    # 95th Percentile
    print(
        f"{'95th Percentile':<20} "
        f"{fmt_dollar(baseline_scenario['p95']):>15} "
        f"{fmt_dollar(econ_scenario['p95']):>15} "
        f"{fmt_dollar(viral_scenario['p95']):>15}"
    )

    # Overlay histogram of the three scenario profit distributions.
    profits_baseline = baseline_scenario["profits"]
    profits_econ = econ_scenario["profits"]
    profits_viral = viral_scenario["profits"]

    all_profits = np.concatenate([profits_baseline, profits_econ, profits_viral])
    bins = np.linspace(np.min(all_profits), np.max(all_profits), 60)

    fig_c2, ax_c2 = plt.subplots(figsize=(10, 6))
    ax_c2.hist(
        profits_baseline,
        bins=bins,
        alpha=0.4,
        label="Baseline",
        color="blue",
        edgecolor="black",
    )
    ax_c2.hist(
        profits_econ,
        bins=bins,
        alpha=0.4,
        label="Economic Slowdown",
        color="red",
        edgecolor="black",
    )
    ax_c2.hist(
        profits_viral,
        bins=bins,
        alpha=0.4,
        label="Viral Marketing Win",
        color="green",
        edgecolor="black",
    )

    # Vertical dashed lines for each mean.
    ax_c2.axvline(
        baseline_scenario["mean"],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Baseline mean",
    )
    ax_c2.axvline(
        econ_scenario["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Econ Slowdown mean",
    )
    ax_c2.axvline(
        viral_scenario["mean"],
        color="green",
        linestyle="--",
        linewidth=2,
        label="Viral Win mean",
    )

    ax_c2.set_xlabel("Profit ($)")
    ax_c2.set_ylabel("Frequency")
    ax_c2.set_title("Scenario Comparison — Profit Distributions")
    ax_c2.legend()

    plt.tight_layout()
    plt.savefig("part_c2_overlay.png", dpi=300)
    plt.show()

