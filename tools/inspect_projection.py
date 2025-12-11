import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simple import SimpleNBAProjection


m = SimpleNBAProjection(dk_salaries_path='DKSalaries.csv')
if not m.load_dk_salaries():
    raise SystemExit('Could not load DK CSV')

m.STARTER_BONUS_SCALE = 4.0
m.LAST5_WEIGHT = 0.35
m.LAST10_WEIGHT = 0.2
m.ROLE_WEIGHT = 0.45

print('Running a quick projection for inspection...')
df = m.run(save_csv=None, n_sims=50)
print('\nColumns:', df.columns.tolist())
print('\nSample rows:')
print(df[['Name','Projection','Floor']].head(15).to_string(index=False))

# also print first 15 Name_L
print('\nName_L sample:')
print(df['Name'].str.lower().str.strip().head(15).to_string(index=False))

# inspect actuals_debug and normalized name overlap
import pandas as pd, unicodedata, re
def normalize_name(s):
    if pd.isna(s):
        return ''
    s = str(s).lower().strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s

try:
    actuals = pd.read_csv('tools/actuals_debug.csv')
    actuals['norm'] = actuals['PLAYER_NAME_L'].apply(normalize_name)
    df['norm'] = df['Name'].apply(normalize_name)
    set_proj = set(df['norm'].unique())
    set_act = set(actuals['norm'].unique())
    inter = set_proj & set_act
    print('\nProjection unique names:', len(set_proj), 'Actuals unique names:', len(set_act), 'Intersection:', len(inter))
    print('Sample intersection (10):', list(inter)[:10])
    print('Sample actuals not in proj (10):', list(set_act - set_proj)[:10])
except FileNotFoundError:
    print('tools/actuals_debug.csv not found')
