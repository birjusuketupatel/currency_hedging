import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Set dynamic hedging parameters ===
threshold = 0.01    # Minimum interest rate differential required to hedge
hedge_ratio = 0.5   # Portion of FX risk hedged

# === Compute FX returns ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Extract US risk-free rate (used for unhedged excess return) ===
us_rf = df[df['country'] == 'USA'][['year', 'bill_rate']].rename(columns={'bill_rate': 'us_bill_rate'})

# === Merge US bill rates into all countries by year ===
df = df.merge(us_rf, on='year', how='left')

# === Compute difference between US and foreign risk-free rate ===
df['delta_rf'] = df['bill_rate'] - df['us_bill_rate']

# === Compute local excess return ===
df['equity_excess_return'] = df['eq_tr'] - df['bill_rate']

# === USD-hedged excess return ===
df['equity_excess_return_usd_hedged'] = df['equity_excess_return'] * df['fx_return']

# === USD-unhedged excess return ===
df['equity_excess_return_usd_unhedged'] = (1 + df['eq_tr']) * df['fx_return'] - (1 + df['us_bill_rate'])

def calc_dynamic_hedging_return(df, threshold, hedge_ratio):
    """
    Compute the dynamically FX-hedged equity excess return:
    - Fully hedged if delta_rf <= threshold
    - Partially hedged (weighted average of hedged and unhedged) if delta_rf > threshold

    Parameters:
        df (DataFrame): The input DataFrame with required columns.
        threshold (float): Interest rate differential threshold for reducing the hedge.
        hedge_ratio (float): Proportion of FX risk to hedge when delta_rf > threshold.

    Returns:
        Series: New return series with dynamic FX hedging.
    """
    # Fully hedged return
    fully_hedged = df['equity_excess_return'] * df['fx_return']

    # Unhedged return
    unhedged = (1 + df['eq_tr']) * df['fx_return'] - (1 + df['us_bill_rate'])

    # Dynamically hedged return
    condition = df['delta_rf'] > threshold
    strategy_return = np.where(
        condition,
        hedge_ratio * fully_hedged + (1 - hedge_ratio) * unhedged,
        fully_hedged
    )

    return strategy_return

# Create new return column
df['equity_excess_return_usd_dynamically_hedged'] = calc_dynamic_hedging_return(
    df, threshold=threshold, hedge_ratio=hedge_ratio
)

# === Drop rows with missing required values ===
required_cols = [
    'equity_excess_return',
    'fx_return',
    'equity_excess_return_usd_hedged',
    'equity_excess_return_usd_unhedged',
    'equity_excess_return_usd_dynamically_hedged'
]
df = df.dropna(subset=required_cols)

# === Drop leading years per country with incomplete data ===
def drop_leading_missing(data, required_cols):
    cleaned = []
    for country, group in data.groupby('country'):
        first_valid_idx = group[required_cols].dropna().index.min()
        group = group.loc[group.index >= first_valid_idx]
        cleaned.append(group)
    return pd.concat(cleaned)

df = drop_leading_missing(df, required_cols)

# === Build cumulative return indices ===
def build_index_with_gaps(data, return_col, index_col):
    data[index_col] = 1.0
    for country, group in data.groupby('country'):
        idx = group.index
        returns = group[return_col].fillna(0)
        cumulative = (1 + returns).cumprod()
        data.loc[idx, index_col] = cumulative
    return data

df = build_index_with_gaps(df, 'equity_excess_return', 'equity_index_local')
df = build_index_with_gaps(df, 'equity_excess_return_usd_hedged', 'equity_index_usd_hedged')
df = build_index_with_gaps(df, 'equity_excess_return_usd_unhedged', 'equity_index_usd_unhedged')
df = build_index_with_gaps(df, 'equity_excess_return_usd_dynamically_hedged', 'equity_index_usd_dynamically_hedged')

# === Plot for a single example country ===
country = 'Switzerland'
plot_df = df[df['country'] == country].copy()

# Identify missing years after first valid year
first_valid_year = plot_df['year'].min()
last_valid_year = plot_df['year'].max()
expected_years = set(range(first_valid_year, last_valid_year + 1))
actual_years = set(plot_df['year'])
missing_years = sorted(expected_years - actual_years)

# === Plot log cumulative returns ===
plt.figure(figsize=(12, 6))

plt.plot(plot_df['year'], np.log(plot_df['equity_index_usd_hedged']),
         label='USD Hedged Excess Return', marker='s', color='red')
plt.plot(plot_df['year'], np.log(plot_df['equity_index_usd_unhedged']),
         label='USD Unhedged Excess Return', marker='^', color='blue')
plt.plot(plot_df['year'], np.log(plot_df['equity_index_usd_dynamically_hedged']),
         label='USD Dynamically Hedged Excess Return', marker='o', color='green')

# Highlight missing years
for i, year in enumerate(missing_years):
    plt.axvline(x=year, color='gray', linestyle='--', alpha=0.6,
                label='Missing Year' if i == 0 else "")

plt.title(f"Equity Excess Returns – {country}")
plt.xlabel("Year")
plt.ylabel("Log Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Export cleaned and computed data ===
output_cols = [
    'country',
    'year',
    'fx_return',
    'delta_rf',
    'equity_excess_return',
    'equity_excess_return_usd_hedged',
    'equity_excess_return_usd_unhedged',
    'equity_excess_return_usd_dynamically_hedged',
    'equity_index_local',
    'equity_index_usd_hedged',
    'equity_index_usd_unhedged',
    'equity_index_usd_dynamically_hedged'
]
df[output_cols].to_csv("equity_excess_returns.csv", index=False)
print("Saved metrics to 'equity_excess_returns.csv'")
