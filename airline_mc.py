import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

seats = 180
tickets_sold = 215
ticket_price = 200
fixed_cost = 30_000
bump_penalty = 800
show_up_prob = 0.88
n_flights = 10_000

profits = np.zeros(n_flights)
bumped_per_flight = np.zeros(n_flights, dtype=int)

for i in range(n_flights):
    show_ups = np.random.binomial(tickets_sold, show_up_prob)
    bumped = max(0, show_ups - seats)
    revenue = tickets_sold * ticket_price
    profit = revenue - fixed_cost - (bumped * bump_penalty)
    profits[i] = profit
    bumped_per_flight[i] = bumped

revenue_per_flight = tickets_sold * ticket_price
mean_profit = np.mean(profits)
std_profit = np.std(profits)
min_profit = np.min(profits)
max_profit = np.max(profits)
p5 = np.percentile(profits, 5)
p95 = np.percentile(profits, 95)
flights_with_bumps = np.sum(bumped_per_flight > 0)
prob_bumping = 100 * flights_with_bumps / n_flights
avg_bumped = np.mean(bumped_per_flight)
total_bumped = np.sum(bumped_per_flight)
flights_losing_money = np.sum(profits < 0)
prob_losing_money = 100 * flights_losing_money / n_flights

print("Airline Overbooking Monte Carlo (10,000 flights)")
print("-" * 50)
print(f"Tickets sold:              {tickets_sold}")
print(f"Revenue per flight:        ${revenue_per_flight:,}")
print(f"Mean profit:               ${mean_profit:,.2f}")
print(f"Std deviation of profit:   ${std_profit:,.2f}")
print(f"Min profit (worst flight): ${min_profit:,.2f}")
print(f"Max profit (best flight):  ${max_profit:,.2f}")
print(f"5th percentile:            ${p5:,.2f}")
print(f"95th percentile:           ${p95:,.2f}")
print(f"Prob of bumping any passenger: {prob_bumping:.2f}%")
print(f"Avg bumped passengers:     {avg_bumped:.2f}")
print(f"Probability of losing money:   {prob_losing_money:.2f}%")
print(f"Number of bumped passengers (total): {total_bumped:,}")

plt.figure(figsize=(10, 6))
plt.hist(profits, bins=50, edgecolor="black", alpha=0.7)
plt.axvline(mean_profit, color="red", linestyle="--", linewidth=2, label=f"Mean = ${mean_profit:,.0f}")
plt.xlabel("Profit ($)")
plt.ylabel("Frequency")
plt.title("Monte Carlo: Flight profits (10,000 flights)")
plt.legend()
plt.tight_layout()
plt.savefig("airline_histogram.png")
plt.close()
print("\nPlot saved as airline_histogram.png")
