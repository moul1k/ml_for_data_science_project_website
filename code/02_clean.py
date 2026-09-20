from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'exoplanet_outputs'
for folder in ['data/processed', 'figures', 'reports']:
    (root / folder).mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(root / 'data/raw/nasa_exoplanets.csv')
columns = list(raw.columns)
print('Rows, columns:', raw.shape)
print('Duplicate names:', raw.pl_name.duplicated().sum())
print(raw.isna().sum().rename('missing').to_frame())
print(raw.describe(include='all').T)

df = raw.copy()
df['pl_name'] = df['pl_name'].astype('string').str.strip()
df['discoverymethod'] = df['discoverymethod'].astype('string').str.strip()
df['discoverymethod'] = df['discoverymethod'].replace('', pd.NA)
df = df.dropna(subset=['pl_name']).drop_duplicates(subset=['pl_name']).copy()
numeric = [col for col in columns if col not in ['pl_name', 'discoverymethod']]
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

invalid = {}
positive = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax',
            'st_teff', 'st_mass', 'st_rad', 'sy_dist']
for col in positive:
    mask = df[col].notna() & (df[col] <= 0)
    invalid[col] = int(mask.sum())
    df.loc[mask, col] = np.nan
mask = df.pl_orbeccen.notna() & ~df.pl_orbeccen.between(0, 1)
invalid['pl_orbeccen'] = int(mask.sum())
df.loc[mask, 'pl_orbeccen'] = np.nan
mask = df.disc_year.notna() & ~df.disc_year.between(1, datetime.now().year)
invalid['disc_year'] = int(mask.sum())
df.loc[mask, 'disc_year'] = np.nan

df.to_csv(root / 'data/processed/exoplanets_clean.csv', index=False)
missing = pd.DataFrame({'missing': df.isna().sum(), 'available': df.notna().sum()})
missing.to_csv(root / 'reports/missingness.csv')
df.describe(include='all').T.to_csv(root / 'reports/summary_statistics.csv')
report = {'raw_rows': len(raw), 'clean_rows': len(df),
          'dropped_missing_or_duplicate_names': len(raw)-len(df),
          'invalid_values_set_missing': invalid,
          'note': 'Missing values retained; each figure drops only its required missing fields.'}
(root / 'reports/cleaning_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print(missing)
print(df.head())

raw.head(8).to_csv(root / 'reports/raw_preview.csv', index=False)
df.head(8).to_csv(root / 'reports/clean_preview.csv', index=False)
for data, name in [(raw, 'raw'), (df, 'clean')]:
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.axis('off')
    preview = data[['pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbper', 'discoverymethod']].head(5).fillna('NA')
    table = ax.table(cellText=preview.astype(str).values, colLabels=preview.columns,
                     loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax.set_title(f'{name.title()} data preview', pad=16)
    fig.savefig(root / f'figures/{name}_preview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
