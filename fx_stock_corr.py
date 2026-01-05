import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import winsorize

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

#df = df[df["year"] >= 1971]

# === Compute FX returns ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Extract US risk-free rate and left join on dataset ===
us_rf = df[df['country'] == 'USA'][['year', 'bill_rate']].rename(columns={'bill_rate': 'us_bill_rate'})
df = df.merge(us_rf, on='year', how='left')

# === Compute currency carry return ===
df['carry_excess_return'] = (1 + df['bill_rate']) * df['fx_return'] - (1 + df['us_bill_rate'])

# === Compute hedged USD excess return ===
df['equity_excess_return'] = (df['eq_tr'] - df['bill_rate']) * df['fx_return']

# === Determine unconditional mean of carry excess returns ===
df = df[df["country"] != "USA"]
carry = df['carry_excess_return'].dropna()

carry_mean = carry.mean()
carry_std = carry.std(ddof=1)
n = len(carry)

t_stat, p_two_sided = stats.ttest_1samp(carry, 0.0)

print("=== Carry Excess Return: Mean Test ===")
print(f"Mean      = {carry_mean:.4%}")
print(f"Std Dev   = {carry_std:.4%}")
print(f"n         = {n}")
print(f"t-stat    = {t_stat:.3f}")
print(f"p-value   = {p_two_sided:.4f}")

if p_two_sided < 0.05:
    print("Reject H0: Mean carry excess return ≠ 0")
else:
    print("Fail to reject H0: Mean not significantly different from zero")

# === Determine correlation between carry excess returns and local currency equity excess returns ===
reg_df = df[['carry_excess_return', 'equity_excess_return']].dropna()

Y = reg_df['carry_excess_return']
X = sm.add_constant(reg_df['equity_excess_return'])

model = sm.OLS(Y, X).fit()

print("\n=== Regression: Carry on Equity Excess Return ===")
print(model.summary())

beta = model.params['equity_excess_return']
beta_p = model.pvalues['equity_excess_return']

print(f"\nBeta estimate = {beta:.3f}")
print(f"Beta p-value  = {beta_p:.4f}")

if beta_p < 0.05:
    print("Reject H0: Beta ≠ 0 (carry loads on equity risk)")
else:
    print("Fail to reject H0: No significant equity exposure")


# === Plot histogram ===
plt.figure(figsize=(7,4))
plt.hist(carry, bins=40, density=True, alpha=0.7)
plt.axvline(carry_mean, color='red', linestyle='--', label='Mean')
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Histogram of Currency Carry Excess Returns")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
sns.regplot(
    x='equity_excess_return',
    y='carry_excess_return',
    data=reg_df,
    scatter_kws={'alpha': 0.4},
    line_kws={'color': 'red'}
)

plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.xlabel("Equity Excess Return")
plt.ylabel("Carry Excess Return")
plt.title("Carry vs Equity Excess Return")
plt.tight_layout()
plt.show()