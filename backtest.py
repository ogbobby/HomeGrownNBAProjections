import pandas as pd
from scipy.stats import spearmanr

# ----------------------------
# Load data
# ----------------------------
proj = pd.read_csv("projections12-22.csv")
dk   = pd.read_csv("contest-standings-12-22.csv")

# ----------------------------
# Normalize names FIRST
# ----------------------------
proj['Name'] = proj['Name'].str.lower().str.strip()
dk['Player'] = dk['Player'].str.lower().str.strip()

# ----------------------------
# Coverage check (correct)
# ----------------------------
common = set(dk['Player']) & set(proj['Name'])
coverage = len(common) / dk['Player'].nunique()

if coverage < 0.95:
    raise ValueError(
        f"Projection/contest mismatch: only {coverage:.1%} of players matched"
    )

# ----------------------------
# STEP 2: Aggregate actual FPTS
# ----------------------------
actuals = (
    dk.groupby('Player', as_index=False)['FPTS']
      .mean()
)

# ----------------------------
# Merge projections with actuals
# ----------------------------
df = proj.merge(
    actuals,
    left_on='Name',
    right_on='Player',
    how='inner'
)

# ----------------------------
# Projection accuracy metrics
# ----------------------------
df['Error'] = df['FPTS'] - df['Projection']

metrics = {
    'MAE': df['Error'].abs().mean(),
    'RMSE': (df['Error']**2).mean()**0.5,
    'Bias': df['Error'].mean(),
    'Spearman': spearmanr(df['Projection'], df['FPTS'])[0]
}

print(metrics)

# ----------------------------
# Ceiling calibration
# ----------------------------
#df['HitCeiling'] = df['FPTS'] >= (0.85 * df['Ceiling_MC'])
df['HitCeiling'] = df['FPTS'] >= df['Ceiling_MC']

ceiling_rate = (
    df.groupby(pd.qcut(df['Ceiling_MC'], 5, duplicates='drop'))
      ['HitCeiling']
      .mean()
)
df['CeilingResidual'] = df['ActualFPTS'] - df['Ceiling_MC']
df.groupby('VolatilityTier')['CeilingResidual'].mean()

print(ceiling_rate)

# ----------------------------
# Floor calibration
# ----------------------------
df['Busted'] = df['FPTS'] <= df['Floor_MC']

bust_rate = (
    df.groupby(pd.qcut(df['Floor_MC'], 5, duplicates='drop'))
      ['Busted']
      .mean()
)

print(bust_rate)

# ----------------------------
# Value correlation
# ----------------------------
df['Value'] = df['FPTS'] / df['Salary']

value_corr = spearmanr(
    df['Projection'] / df['Salary'],
    df['Value']
)[0]

print("Value Spearman:", value_corr)

# ----------------------------
# Lineup finish analysis
# ----------------------------
lineups = (
    dk.groupby('EntryId')
      .agg(
          Points=('Points', 'first'),
          Rank=('Rank', 'first')
      )
)

lineups['FinishPct'] = 1 - lineups['Rank'] / lineups['Rank'].max()
print(lineups['FinishPct'].describe())

# ----------------------------
# Exposure analysis
# ----------------------------
exposure = (
    dk.merge(proj, left_on='Player', right_on='Name')
      .groupby('Name')
      .agg(
          MeanFPTS=('FPTS', 'mean'),
          Projection=('Projection', 'mean'),
          Exposure=('EntryId', 'count')
      )
)

exposure['Error'] = exposure['MeanFPTS'] - exposure['Projection']
print(exposure.sort_values('Exposure', ascending=False).head(20))
