"""
NovaBrew Product Launch - Monte Carlo Risk Analysis
1. Baseline: Profit distribution with 95% CI
2. Sensitivity: One-at-a-time tornado (most influential at top)
3. Scenarios: Competitor Entry vs Supply Chain Disruption
"""

import numpy as np
import matplotlib.pyplot as plt

N_SIMS = 10_000
SEED = 42


def run_simulation(market_size=None, capture_rate=None, mfg_cost=None,
                   fixed_cost=None, return_rate=None, price=None, n_sims=N_SIMS):
    """
    Run Monte Carlo profit simulation.
    Pass None to sample from baseline distribution, or scalar/array to override.
    """
    def sample_or_fill(val, sample_fn):
        if val is None:
            return sample_fn()
        return np.full(n_sims, val) if np.isscalar(val) else val

    np.random.seed(SEED)
    market_size = sample_or_fill(market_size, lambda: np.random.normal(500_000, 75_000, n_sims))
    capture_rate = sample_or_fill(capture_rate, lambda: np.random.triangular(0.02, 0.04, 0.07, n_sims))
    mfg_cost = sample_or_fill(mfg_cost, lambda: np.random.normal(120, 15, n_sims))
    fixed_cost = sample_or_fill(fixed_cost, lambda: np.random.triangular(1_500_000, 2_000_000, 3_000_000, n_sims))
    return_rate = sample_or_fill(return_rate, lambda: np.random.uniform(0.03, 0.15, n_sims))
    price = sample_or_fill(price if price is not None else 299, lambda: np.full(n_sims, 299))

    units_sold = market_size * capture_rate
    revenue = units_sold * price * (1 - return_rate)
    cogs = units_sold * mfg_cost
    return_processing = units_sold * return_rate * 30  # $30 per return
    profit = revenue - cogs - return_processing - fixed_cost
    return profit


# -----------------------------------------------------------------------------
# STEP 1: BASELINE (Price = $299)
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: BASELINE SIMULATION")
print("=" * 60)

np.random.seed(SEED)
profits = run_simulation(price=299)
mean_p = np.mean(profits)
std_p = np.std(profits)
se = std_p / np.sqrt(N_SIMS)
ci = (mean_p - 1.96 * se, mean_p + 1.96 * se)
p5, p50, p95 = np.percentile(profits, [5, 50, 95])
p_neg = 100 * np.mean(profits < 0)

print(f"  Mean profit:      ${mean_p:,.2f}")
print(f"  Std dev:          ${std_p:,.2f}")
print(f"  95% CI:           [${ci[0]:,.2f}, ${ci[1]:,.2f}]")
print(f"  5th / 50th / 95th: ${p5:,.2f} / ${p50:,.2f} / ${p95:,.2f}")
print(f"  P(Profit < 0):    {p_neg:.2f}%")

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.hist(profits / 1e6, bins=50, edgecolor="black", alpha=0.7)
ax1.axvline(mean_p / 1e6, color="green", linestyle="--", linewidth=2, label=f"Mean = ${mean_p/1e6:.2f}M")
ax1.axvline(0, color="red", linestyle="-", linewidth=1)
ax1.set_xlabel("Profit ($ millions)")
ax1.set_ylabel("Frequency")
ax1.set_title("Baseline: Profit Distribution (10,000 runs)")
ax1.legend()
plt.tight_layout()
plt.savefig("Risk-Analysis-baseline.png")
plt.close()
print("Saved: Risk-Analysis-baseline.png\n")


# -----------------------------------------------------------------------------
# STEP 2: SENSITIVITY (One-at-a-time tornado)
# Low/high values from case. Fix one input; others from baseline distributions.
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 2: SENSITIVITY ANALYSIS (Tornado)")
print("=" * 60)

# Input ranges: (name, low, high)
INPUTS = [
    ("Market size", 350_000, 650_000),
    ("Capture rate", 0.02, 0.07),
    ("Unit price", 249, 349),
    ("Mfg cost", 90, 150),
    ("Fixed launch cost", 1_500_000, 3_000_000),
    ("Return rate", 0.03, 0.15),
]
KWARGS_KEYS = ["market_size", "capture_rate", "price", "mfg_cost", "fixed_cost", "return_rate"]

low_profits = []
high_profits = []
for i, (name, lo, hi) in enumerate(INPUTS):
    kw = {KWARGS_KEYS[i]: lo}
    np.random.seed(SEED)
    low_profits.append(np.mean(run_simulation(**kw)))
    kw = {KWARGS_KEYS[i]: hi}
    np.random.seed(SEED)
    high_profits.append(np.mean(run_simulation(**kw)))

ranges = [high_profits[i] - low_profits[i] for i in range(len(INPUTS))]
order = np.argsort(np.abs(ranges))[::-1]  # most influential first
sorted_names = [INPUTS[i][0] for i in order]
sorted_low_m = [low_profits[i] / 1e6 for i in order]
sorted_high_m = [high_profits[i] / 1e6 for i in order]
sorted_range_m = [abs(ranges[i]) / 1e6 for i in order]

for i in order:
    print(f"  {INPUTS[i][0]:20s}: Low ${low_profits[i]/1e6:7.2f}M | High ${high_profits[i]/1e6:7.2f}M")

fig2, ax2 = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(sorted_names))[::-1]
widths = np.array(sorted_range_m)
colors = ["#2ecc71" if sorted_high_m[i] > sorted_low_m[i] else "#e74c3c" for i in range(len(sorted_names))]
ax2.barh(y_pos, widths, left=0, color=colors, edgecolor="black")
for i, y in enumerate(y_pos):
    ax2.text(-0.12, y, f"${sorted_low_m[i]:.2f}M", ha="right", va="center", fontsize=8)
    ax2.text(widths[i] + 0.08, y, f"${sorted_high_m[i]:.2f}M", ha="left", va="center", fontsize=8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_names)
ax2.set_xlabel("Range of mean profit ($ millions)")
ax2.set_title("Tornado: Most Influential Inputs at Top")
ax2.set_xlim(left=-0.5, right=max(widths) * 1.15)
plt.tight_layout()
plt.savefig("Risk-Analysis-tornado.png")
plt.close()
print("Saved: Risk-Analysis-tornado.png\n")


# -----------------------------------------------------------------------------
# STEP 3: SCENARIOS
# A: Competitor Entry (capture down, price $249)
# B: Supply Chain Disruption (mfg & fixed cost up)
# -----------------------------------------------------------------------------
print("=" * 60)
print("STEP 3: SCENARIO COMPARISON")
print("=" * 60)

np.random.seed(SEED)
capture_a = np.random.triangular(0.01, 0.025, 0.04, N_SIMS)
profits_a = run_simulation(capture_rate=capture_a, price=249)
m_a, p5_a, p95_a = np.mean(profits_a), np.percentile(profits_a, 5), np.percentile(profits_a, 95)
p_neg_a = 100 * np.mean(profits_a < 0)

np.random.seed(SEED)
mfg_b = np.random.normal(155, 20, N_SIMS)
fixed_b = np.random.triangular(2_000_000, 2_800_000, 4_000_000, N_SIMS)
profits_b = run_simulation(mfg_cost=mfg_b, fixed_cost=fixed_b)
m_b, p5_b, p95_b = np.mean(profits_b), np.percentile(profits_b, 5), np.percentile(profits_b, 95)
p_neg_b = 100 * np.mean(profits_b < 0)

print("\nScenario A (Competitor Entry): Capture Tri(1%,2.5%,4%), Price $249")
print(f"  Mean: ${m_a:,.2f}  |  5th-95th: [${p5_a:,.2f}, ${p95_a:,.2f}]  |  P(<0): {p_neg_a:.2f}%")
print("\nScenario B (Supply Chain): Mfg N($155,$20), Fixed Tri($2M,$2.8M,$4M)")
print(f"  Mean: ${m_b:,.2f}  |  5th-95th: [${p5_b:,.2f}, ${p95_b:,.2f}]  |  P(<0): {p_neg_b:.2f}%")

bins = np.linspace(
    min(profits.min(), profits_a.min(), profits_b.min()) / 1e6,
    max(profits.max(), profits_a.max(), profits_b.max()) / 1e6, 50
)
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.hist(profits / 1e6, bins=bins, alpha=0.5, label="Baseline", color="gray", density=True)
ax3.hist(profits_a / 1e6, bins=bins, alpha=0.5, label="Competitor Entry", color="orange", density=True)
ax3.hist(profits_b / 1e6, bins=bins, alpha=0.5, label="Supply Chain Disruption", color="red", density=True)
ax3.axvline(0, color="black", linestyle="-", linewidth=1)
ax3.set_xlabel("Profit ($ millions)")
ax3.set_ylabel("Density")
ax3.set_title("Scenario Comparison")
ax3.legend()
plt.tight_layout()
plt.savefig("Risk-Analysis-scenarios.png")
plt.close()
print("\nSaved: Risk-Analysis-scenarios.png")
