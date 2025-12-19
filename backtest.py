import pandas as pd
from scipy.stats import spearmanr

proj = pd.read_csv("projections12-17.csv")
dk   = pd.read_csv("contest-standings-12-17.csv")

proj['Name'] = proj['Name'].str.lower().str.strip()
dk['Player'] = dk['Player'].str.lower().str.strip()

df = dk.merge(
    proj,
    left_on='Player',
    right_on='Name',
    how='left'
)

df = df.dropna(subset=['Projection'])
df['Error'] = df['FPTS'] - df['Projection']

metrics = {
    'MAE': df['Error'].abs().mean(),
    'RMSE': (df['Error']**2).mean()**0.5,
    'Bias': df['Error'].mean(),
    'Spearman': spearmanr(df['Projection'], df['FPTS'])[0]
}

print(metrics)

df['HitCeiling'] = df['FPTS'] >= df['Ceiling_MC']

ceiling_rate = (
    df.groupby(pd.qcut(df['Ceiling_MC'], 5))
      ['HitCeiling']
      .mean()
)

print(ceiling_rate)

df['Busted'] = df['FPTS'] <= df['Floor_MC']

bust_rate = (
    df.groupby(pd.qcut(df['Floor_MC'], 5))
      ['Busted']
      .mean()
)

print(bust_rate)

df['Value'] = df['FPTS'] / df['Salary']

value_corr = spearmanr(
    df['Projection'] / df['Salary'],
    df['Value']
)[0]

print("Value Spearman:", value_corr)

lineups = (
    dk.groupby('EntryId')
      .agg(
          Points=('Points', 'first'),
          Rank=('Rank', 'first')
      )
)

lineups['FinishPct'] = 1 - lineups['Rank'] / lineups['Rank'].max()
print(lineups['FinishPct'].describe())

exposure = (
    df.groupby('Name')
      .agg(
          MeanFPTS=('FPTS', 'mean'),
          Projection=('Projection', 'mean'),
          Exposure=('EntryId', 'count')
      )
)

exposure['Error'] = exposure['MeanFPTS'] - exposure['Projection']

print(exposure.sort_values('Exposure', ascending=False).head(20))
