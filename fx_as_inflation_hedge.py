import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# === Load and prepare dataset ===
df = pd.read_csv("JSTdatasetR6.csv")
df = df.sort_values(['country', 'year'])

# === Compute FX returns ===
df['xrusd'] = df.groupby('country')['xrusd'].ffill()
df['fx_return'] = df.groupby('country')['xrusd'].transform(lambda x: x.shift(1) / x)

# === Extract US risk-free rate (used for unhedged excess return) ===
us_rf = df[df['country'] == 'USA'][['year', 'bill_rate']].rename(columns={'bill_rate': 'us_bill_rate'})

# === Merge US bill rates into all countries by year ===
df = df.merge(us_rf, on='year', how='left')

# === Construct unified short rate ===
df['short_rate'] = df['bill_rate']

# Canada: use STIR (percent → decimal)
mask_canada = (df['country'] == 'Canada') & (df['short_rate'].isna())
df.loc[mask_canada, 'short_rate'] = df.loc[mask_canada, 'stir'] / 100.0


# === Calculate currency carry return ===
df['currency_carry'] = (1 + df['short_rate']) * df['fx_return'] - df['us_bill_rate']
df['delta_rf'] = df['short_rate'] - df['us_bill_rate']

# === Compute inflation rate ===
df['cpi'] = df.groupby('country')['cpi'].ffill()
df['inf'] = df.groupby('country')['cpi'].transform(lambda x: x / x.shift(1))

# == Merge with US inflation ==
us_inf = df[df['country'] == 'USA'][['year', 'inf']].rename(columns={'inf': 'us_inf'})
df = df.merge(us_inf, on='year', how='left')

# === Filter for only data after end of Bretton Woods (fiat currency period) ===
df = df[df['year'] > 1971]

# === Filter number of countries in sample to those with unique currencies ===
# Canada (CAD)
# Germany (DM/EUR)
# Australia (AUD)
# UK (GBP)
# Switzerland (CHF)
# Japan (JPY)
# Sweden (SEK)
# Norway (NO)
countries = ['USA', 'Canada', 'Germany', 'Australia', 'UK', 'Switzerland', 'Japan', 'Sweden', 'Norway']
df = df[df['country'].isin(countries)]

# === Add portfolio that is simple average of all country FX returns ===
avg_portfolio = (
    df[df['country'] != 'USA']
    .groupby('year', as_index=False)
    .agg({
        'currency_carry': 'mean',
        'us_inf': 'first'
    })
)

avg_portfolio['country'] = 'Average'

df = pd.concat([df, avg_portfolio], ignore_index=True)

# === Select only relevent rows ===
df = df[['country', 'year', 'us_inf', 'currency_carry', 'delta_rf']]

# === Calculate correlations between US inflation and foreign currency returns ===
results = []

for country, g in df.groupby('country'):
    if country == 'USA':
        continue  # skip self-regression

    y = g['currency_carry']
    X = sm.add_constant(g['us_inf'])

    model = sm.OLS(y, X).fit()

    results.append({
        'country': country,
        'alpha': model.params['const'],
        'beta_us_inf': model.params['us_inf'],
        't_alpha': model.tvalues['const'],
        't_beta': model.tvalues['us_inf'],
        'n_obs': int(model.nobs),
        'r2': model.rsquared
    })

# === Print panel regression results ===
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('beta_us_inf', ascending=False)
results_df[['alpha','beta_us_inf','t_alpha','t_beta','r2']] = results_df[['alpha','beta_us_inf','t_alpha','t_beta','r2']].round(3)

print(results_df)

# === Display currency return time series for all countries and average of all countries ===
for country, g in df.groupby('country'):
    if country == 'USA':
        continue

    g = g.sort_values('year')
    carry_ret = g['currency_carry'] - 1

    if country == "Average":
        plt.plot(
            g['year'],
            carry_ret,
            linewidth=4.0,
            alpha=1.0,
            label="Average"
        )
    else:
        plt.plot(
            g['year'],
            carry_ret,
            alpha=0.6,
            label=country
        )

plt.axhline(0, linewidth=0.8)

plt.xlabel("Year")
plt.ylabel("Annual Carry Return")
plt.title("Foreign Carry Returns (FX + Foreign Bills − US Bills)")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# === Build top rate spread strategy (long max delta_rf each year; else 0 excess) ===
base = df[(df['country'].isin(countries)) & (df['country'] != 'USA')].copy()

top_spread = (
    base.sort_values(['year', 'delta_rf'], ascending=[True, False])
        .groupby('year')
        .first()
        .reset_index()
)

top_spread['excess_ret'] = np.where(top_spread['delta_rf'] > 0, top_spread['currency_carry'] - 1.0, 0.0)
top_spread['cum_excess'] = (1 + top_spread['excess_ret']).cumprod()

# === Average cumulative excess return ===
avg = df[df['country'] == 'Average'].sort_values('year').copy()
avg['excess_ret'] = avg['currency_carry'] - 1.0
avg['cum_excess'] = (1 + avg['excess_ret']).cumprod()

plt.figure(figsize=(9, 4))
plt.plot(avg['year'], avg['cum_excess'], linewidth=4.0, label="Average")

plt.axhline(1, linewidth=0.8)
plt.xlabel("Year")
plt.ylabel("Cumulative USD Excess Return")
plt.title("Average Foreign Currency Return")
plt.tight_layout()
plt.show()

# === Overlay cumulative curves ===
plt.figure(figsize=(9, 4))
plt.plot(avg['year'], avg['cum_excess'], linewidth=4.0, label="Average")
plt.plot(top_spread['year'], top_spread['cum_excess'], linewidth=2.0, label="Top Rate Spread (max ΔRF)")

plt.axhline(1, linewidth=0.8)
plt.xlabel("Year")
plt.ylabel("Cumulative USD Excess Return")
plt.title("Cumulative Excess Return: Average vs Top Rate Spread (Fiat Period)")
plt.legend()
plt.tight_layout()
plt.show()

print(top_spread[['year','country','delta_rf','excess_ret']])

# === Summary statistics for top spread strategy ===

summary = {}

summary['mean_excess_ret'] = top_spread['excess_ret'].mean()
summary['std_excess_ret'] = top_spread['excess_ret'].std(ddof=1)
summary['sharpe'] = (
    summary['mean_excess_ret'] / summary['std_excess_ret']
    if summary['std_excess_ret'] != 0 else np.nan
)
summary['avg_delta_rf'] = top_spread['delta_rf'].mean()
summary['n_years'] = top_spread['year'].nunique()

summary_df = (
    pd.DataFrame([summary])
    .rename(columns={
        'mean_excess_ret': 'Mean Excess Return',
        'std_excess_ret': 'Std Dev Excess Return',
        'sharpe': 'Sharpe Ratio',
        'avg_delta_rf': 'Average ΔRF',
        'n_years': 'Years'
    })
)

summary_df[['Mean Excess Return',
            'Std Dev Excess Return',
            'Sharpe Ratio',
            'Average ΔRF']] = (
    summary_df[['Mean Excess Return',
                'Std Dev Excess Return',
                'Sharpe Ratio',
                'Average ΔRF']].round(4)
)

print(summary_df)
