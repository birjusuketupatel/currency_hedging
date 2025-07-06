import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f

# === Load data ===
df = pd.read_csv("equity_excess_returns.csv")
hedged = df['equity_excess_return_usd_hedged'].dropna()
unhedged = df['equity_excess_return_usd_unhedged'].dropna()
dynamically_hedged = df['equity_excess_return_usd_dynamically_hedged'].dropna()

# === Basic stats ===
hedged_mean, hedged_std, n1 = hedged.mean(), hedged.std(ddof=1), len(hedged)
unhedged_mean, unhedged_std, n2 = unhedged.mean(), unhedged.std(ddof=1), len(unhedged)
dynamic_mean, dynamic_std, n3 = dynamically_hedged.mean(), dynamically_hedged.std(ddof=1), len(dynamically_hedged)

print(f"Hedged:   Mean = {hedged_mean:.4%}, Std Dev = {hedged_std:.4%} (n = {n1})")
print(f"Unhedged: Mean = {unhedged_mean:.4%}, Std Dev = {unhedged_std:.4%} (n = {n2})")
print(f"Dynamically Hedged: Mean = {dynamic_mean:.4%}, Std Dev = {dynamic_std:.4%} (n = {n2})")

# === One-sided Welch's T-test: H0: mean_hedged <= mean_unhedged
t_stat, p_two_sided = ttest_ind(hedged, unhedged, equal_var=False)
p_mean = p_two_sided / 2 if t_stat > 0 else 1.0  # one-sided p-value

print(f"\nOne-sided Welch's T-test (Hedged > Unhedged):")
print(f"  t = {t_stat:.4f}, one-sided p = {p_mean:.4f}")
if p_mean < 0.05:
    print("Reject H0: Hedged mean return is significantly greater.")
else:
    print("Fail to reject H0: No significant difference in means.")

# === One-sided F-test: H0: var_hedged >= var_unhedged
f_stat = (hedged_std ** 2) / (unhedged_std ** 2)
df1 = n1 - 1
df2 = n2 - 1
p_var = f.cdf(f_stat, df1, df2)  # one-sided, lower tail

print(f"\nOne-sided F-test (Var(hedged) < Var(unhedged)):")
print(f"  F = {f_stat:.4f}, one-sided p = {p_var:.4f}")
if p_var < 0.05:
    print("Reject H0: Hedged variance is significantly smaller.")
else:
    print("Fail to reject H0: No significant difference in variances.")

# === Plot histograms with vertical mean lines ===
plt.figure(figsize=(10, 6))
plt.hist(hedged, bins=50, alpha=0.6, label='Hedged', color='red', density=True)
plt.hist(unhedged, bins=50, alpha=0.6, label='Unhedged', color='blue', density=True)
plt.axvline(hedged_mean, color='red', linestyle='--', linewidth=1.5, label='Hedged Mean')
plt.axvline(unhedged_mean, color='blue', linestyle='--', linewidth=1.5, label='Unhedged Mean')
plt.title("Distribution of Hedged vs. Unhedged USD Excess Returns")
plt.xlabel("Annual Excess Return")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Tail Risk Bucketing ===
thresholds = [-0.2, -0.4, -0.6]
labels = ['≤ -20%', '≤ -40%', '≤ -60%']

hedged_probs = [(hedged <= t).mean() for t in thresholds]
unhedged_probs = [(unhedged <= t).mean() for t in thresholds]
dynamic_probs = [(dynamically_hedged <= t).mean() for t in thresholds]

# === Plot bar chart of tail probabilities ===
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar(x - width, hedged_probs, width, label='Hedged', color='red')
plt.bar(x, dynamic_probs, width, label='Dynamically Hedged', color='green')
plt.bar(x + width, unhedged_probs, width, label='Unhedged', color='blue')

plt.ylabel("Probability")
plt.title("Tail Risk Comparison (Hedged vs. Dynamic vs. Unhedged)")
plt.xticks(x, labels)
plt.ylim(0, max(max(hedged_probs), max(unhedged_probs), max(dynamic_probs)) * 1.2)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# === Print values for clarity ===
print("\nTail Risk Probabilities:")
for label, h, d, u in zip(labels, hedged_probs, dynamic_probs, unhedged_probs):
    print(f"{label}: Hedged = {h:.2%}, Dynamically Hedged = {d:.2%}, Unhedged = {u:.2%}")
