import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import seaborn as sns

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Drop data from after the Euro was adopted ===
df = df[df['year'] < 1999]

# === Compute FX returns ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Extract US risk-free rate and left join on dataset ===
us_rf = df[df['country'] == 'USA'][['year', 'bill_rate']].rename(columns={'bill_rate': 'us_bill_rate'})
df = df.merge(us_rf, on='year', how='left')

# === Compute difference between US and foreign risk-free rate ===
df['delta_rf'] = df['bill_rate'] - df['us_bill_rate']

# === Compute currency carry return ===
df['carry_excess_return'] = (1 + df['bill_rate']) * df['fx_return'] - (1 + df['us_bill_rate'])

# === Drop rows with missing required values ===
required_cols = [
    'delta_rf',
    'fx_return',
    'carry_excess_return'
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

df = build_index_with_gaps(df, 'carry_excess_return', 'carry_index')

# === Plot for a single example country ===
country = 'Sweden'
plot_df = df[df['country'] == country].copy()

# === Identify missing years after first valid year ===
first_valid_year = plot_df['year'].min()
last_valid_year = plot_df['year'].max()
expected_years = set(range(first_valid_year, last_valid_year + 1))
actual_years = set(plot_df['year'])
missing_years = sorted(expected_years - actual_years)

# === Plot log cumulative returns ===
plt.figure(figsize=(12, 6))

plt.plot(plot_df['year'], np.log(plot_df['carry_index']),
         label='Carry Excess Return', marker='o', color='green')

# === Highlight missing years ===
for i, year in enumerate(missing_years):
    plt.axvline(x=year, color='gray', linestyle='--', alpha=0.6,
                label='Missing Year' if i == 0 else "")

plt.title(f"Carry Excess Returns – {country}")
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
    'delta_rf',
    'fx_return',
    'carry_excess_return',
    'carry_index'
]
df[output_cols].to_csv("carry_excess_returns.csv", index=False)
print("Saved metrics to 'carry_excess_returns.csv'")

# === Histogram of carry_excess_return across all countries ===
plt.figure(figsize=(10, 5))
plt.hist(df['carry_excess_return'], bins=30, edgecolor='black', alpha=0.75)
plt.title("Histogram of Carry Excess Returns – All Countries")
plt.xlabel("Carry Excess Return")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print mean and standard deviation ===
mean_return = df['carry_excess_return'].mean()
std_return = df['carry_excess_return'].std()

print(f"Mean Carry Excess Return: {mean_return:.4f}")
print(f"Standard Deviation:       {std_return:.4f}")

# === Drop missing values for regression ===
regression_df = df[['delta_rf', 'carry_excess_return']].dropna()

# === Regress carry_excess_return on delta_rf ===
X = regression_df['delta_rf']
y = regression_df['carry_excess_return']
X = sm.add_constant(X)  # Adds intercept term

# === Fit the OLS model ===
model = sm.OLS(y, X).fit()

# === Print regression summary ===
print(model.summary())

# === Scatterplot with regression line ===
plt.figure(figsize=(10, 6))
sns.regplot(x='delta_rf', y='carry_excess_return', data=regression_df, ci=95, line_kws={'color': 'red'})
plt.title("Carry Excess Return vs. Interest Rate Differential (Δrf)")
plt.xlabel("Interest Rate Differential (Δrf)")
plt.ylabel("Carry Excess Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Create 4 quartile bins based on delta_rf ===
df['delta_rf_quartile'] = pd.qcut(df['delta_rf'], q=4, duplicates='drop')

# === Compute average carry return in each quartile and convert to percent ===
quartile_means = df.groupby('delta_rf_quartile', as_index=False, observed=True)['carry_excess_return'].mean()
quartile_means['carry_excess_return'] *= 100  # convert to percent

# === Format bin labels for x-axis in percent ===
quartile_labels = [
    f"{interval.left*100:.1f}% to {interval.right*100:.1f}%"
    for interval in quartile_means['delta_rf_quartile']
]
quartile_means['label'] = quartile_labels

# === Plot ===
plt.figure(figsize=(10, 6))
sns.barplot(
    x='label',
    y='carry_excess_return',
    hue='label',
    data=quartile_means,
    palette='muted',
    legend=False
)

plt.title("Average Carry Excess Return by Δrf Quartile")
plt.xlabel("Interest Rate Differential (Δrf) Quartile")
plt.ylabel("Average Carry Excess Return (%)")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()