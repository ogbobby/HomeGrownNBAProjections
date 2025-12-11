"""Run a narrowed parameter grid across a set of dates and aggregate results.

This script calls tools/backtest_grid.py repeatedly with --date and grid args,
then renames the per-run output and produces an aggregated CSV.
"""
import subprocess
import sys
import os
from datetime import date, timedelta
import glob
import pandas as pd

# configuration: narrowed grid
starter_scales = [3.5, 4.0, 4.5]
last5_weights = [0.40, 0.45, 0.50]
last10_weights = [0.20, 0.25]

# dates to run (inclusive)
dates = [
    '2025-12-01', '2025-12-02', '2025-12-03', '2025-12-04',
    '2025-12-05', '2025-12-06', '2025-12-07'
]

base_cmd = [sys.executable, 'tools/backtest_grid.py']

# build comma args
starter_arg = ','.join(map(str, starter_scales))
last5_arg = ','.join(map(str, last5_weights))
last10_arg = ','.join(map(str, last10_weights))

out_files = []
for d in dates:
    cmd = base_cmd + [
        '--date', d,
        '--starter-scales', starter_arg,
        '--last5-weights', last5_arg,
        '--last10-weights', last10_arg,
        '--n-sims', '200',
        '--backup-delta-threshold', '0.8',
        '--boost-minutes-scale', '1.1'
    ]
    print('\nRunning narrowed grid for', d)
    print('  cmd:', ' '.join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    # backtest_grid.py writes results to tools/backtest_grid_results.csv by default
    src = 'tools/backtest_grid_results.csv'
    dest = f'tools/backtest_grid_results_{d}.csv'
    if os.path.exists(src):
        os.replace(src, dest)
        out_files.append(dest)
        print('Saved per-date results to', dest)
    else:
        print('Expected output', src, 'not found for date', d)

# aggregate any per-date CSVs we produced
if not out_files:
    print('No per-date result files produced; exiting')
    sys.exit(1)

print('\nAggregating per-date results...')
all_df = pd.concat([pd.read_csv(f).assign(date=f.split('_')[-1].split('.csv')[0]) for f in sorted(out_files)], ignore_index=True)
required = ['starter_scale','last5','last10','role','mae','floor_hit_rate','num_merged']
for c in required:
    if c not in all_df.columns:
        print('Missing column in combined results:', c)
        sys.exit(1)

agg = all_df.groupby(['starter_scale','last5','last10','role']).agg(
    mae_mean=('mae','mean'),
    mae_std=('mae','std'),
    floor_mean=('floor_hit_rate','mean'),
    num_merged_mean=('num_merged','mean'),
    dates_count=('mae','count')
).reset_index()
agg = agg.sort_values('mae_mean')
out = 'tools/backtest_grid_results_aggregate_narrow_2025-12-01_07.csv'
agg.to_csv(out, index=False)
print('Saved aggregated narrowed-grid results to', out)
print('\nTop combos:')
print(agg.head(10).to_string(index=False))
