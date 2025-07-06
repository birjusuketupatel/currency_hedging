import pandas as pd
import numpy as np
from scipy.stats import norm

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Compute FX returns (local per USD) ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Extract US risk-free rate and merge ===
us_rf = df[df['country'] == 'USA'][['year', 'bill_rate']].rename(columns={'bill_rate': 'us_bill_rate'})
df = df.merge(us_rf, on='year', how='left')

# === Compute delta_rf (foreign - US interest rate) ===
df['delta_rf'] = df['bill_rate'] - df['us_bill_rate']

# === Compute local equity excess return ===
df['equity_excess_return'] = df['eq_tr'] - df['bill_rate']

# === Compute USD-hedged excess return ===
df['equity_excess_return_usd_hedged'] = df['equity_excess_return'] * df['fx_return']

# === Function to compute dynamically hedged return ===
def compute_dynamic_hedged_return(df, threshold, hedge_ratio):
    fully_hedged = df['equity_excess_return'] * df['fx_return']
    unhedged = (1 + df['eq_tr']) * df['fx_return'] - (1 + df['us_bill_rate'])
    condition = df['delta_rf'] > threshold
    dynamic_return = np.where(
        condition,
        hedge_ratio * fully_hedged + (1 - hedge_ratio) * unhedged,
        fully_hedged
    )
    return pd.Series(dynamic_return, index=df.index)

# === Grid of parameters ===
threshold_grid = np.arange(0.00, 0.055, 0.005)   # e.g., 0.00 to 0.05
hedge_ratio_grid = np.arange(-0.20, 1.10, 0.1)  # e.g., 0.00 to 1.00

# === Drop rows with missing required data
df = df.dropna(subset=['equity_excess_return_usd_hedged', 'eq_tr', 'us_bill_rate', 'fx_return'])

# === Results container
results = []

for threshold in threshold_grid:
    for hedge_ratio in hedge_ratio_grid:
        dynamic = compute_dynamic_hedged_return(df, threshold, hedge_ratio)

        combined = pd.DataFrame({
            'dynamic': dynamic,
            'hedged': df['equity_excess_return_usd_hedged']
        }).dropna()

        if len(combined) < 2:
            continue

        d = combined['dynamic']
        h = combined['hedged']
        n = len(combined)

        d_mean, d_std = d.mean(), d.std(ddof=1)
        h_mean, h_std = h.mean(), h.std(ddof=1)

        sharpe_d = d_mean / d_std if d_std != 0 else np.nan
        sharpe_h = h_mean / h_std if h_std != 0 else np.nan
        sharpe_diff = sharpe_d - sharpe_h

        # === Jobson-Korkie with Memmel correction ===
        cov = np.cov(d, h, ddof=1)[0, 1]
        var_diff = (1 / n) * (
            (d_std**2 + sharpe_d**2) / d_std**2 +
            (h_std**2 + sharpe_h**2) / h_std**2 -
            2 * (cov + sharpe_d * sharpe_h) / (d_std * h_std)
        )
        t_stat = sharpe_diff / np.sqrt(var_diff) if var_diff > 0 else np.nan
        p_val = 1 - norm.cdf(t_stat) if t_stat is not np.nan else np.nan  # one-sided test

        results.append({
            'threshold': threshold,
            'hedge_ratio': hedge_ratio,
            'dynamic_sharpe': sharpe_d,
            'hedged_sharpe': sharpe_h,
            'sharpe_diff': sharpe_diff,
            't_stat': t_stat,
            'p_val_one_sided': p_val
        })

# === Results DataFrame ===
results_df = pd.DataFrame(results).sort_values(by='dynamic_sharpe', ascending=False)

# === Display top rows
print("\n=== Sharpe Ratio Comparison with Statistical Test ===")
print(results_df[['threshold', 'hedge_ratio', 'dynamic_sharpe', 'hedged_sharpe',
                  'sharpe_diff', 't_stat', 'p_val_one_sided']].head(5))