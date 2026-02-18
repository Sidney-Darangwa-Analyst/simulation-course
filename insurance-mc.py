import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_months = 10_000
poisson_mean = 8
exp_mean = 5000
admin_cost = 2000

total_costs = np.zeros(n_months)

for month in range(n_months):
    n_claims = np.random.poisson(poisson_mean)
    claim_costs = np.random.exponential(exp_mean, size=n_claims)
    total_costs[month] = claim_costs.sum() + admin_cost

mean_cost = np.mean(total_costs)
std_cost = np.std(total_costs)
n = n_months
se = std_cost / np.sqrt(n)
ci_half = 1.96 * se
ci_low = mean_cost - ci_half
ci_high = mean_cost + ci_half

min_cost = np.min(total_costs)
max_cost = np.max(total_costs)
p5 = np.percentile(total_costs, 5)
p95 = np.percentile(total_costs, 95)

print("Total monthly cost (10,000 months):")
print(f"  Mean:           ${mean_cost:,.2f}")
print(f"  95% CI (mean):  [${ci_low:,.2f}, ${ci_high:,.2f}]  (Mean +/- 1.96*std/sqrt(n))")
print(f"  Std dev:        ${std_cost:,.2f}")
print(f"  Min:            ${min_cost:,.2f}")
print(f"  Max:            ${max_cost:,.2f}")
print(f"  5th percentile: ${p5:,.2f}")
print(f"  95th percentile: ${p95:,.2f}")

above_50k = np.sum(total_costs > 50000)
pct_above_50k = 100 * above_50k / n_months
print(f"\nMonths where total cost exceeds $50,000: {above_50k} ({pct_above_50k:.2f}%)")

plt.figure(figsize=(10, 6))
plt.hist(total_costs, bins=50, edgecolor="black", alpha=0.7)
plt.axvline(mean_cost, color="green", linestyle="--", linewidth=2, label=f"Mean = ${mean_cost:,.0f}")
plt.axvline(p95, color="red", linestyle="--", linewidth=2, label=f"95th %ile = ${p95:,.0f}")
plt.xlabel("Total monthly cost ($)")
plt.ylabel("Frequency")
plt.title("Monte Carlo: Total monthly cost (10,000 months)")
plt.legend()
plt.tight_layout()
plt.savefig("insurance_histogram.png")
plt.close()
print("\nPlot saved as insurance_histogram.png")
