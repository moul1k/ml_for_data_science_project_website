from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

root = Path(__file__).resolve().parents[1] / 'exoplanet_outputs'
(root / 'figures').mkdir(parents=True, exist_ok=True)
(root / 'reports').mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook')
df = pd.read_csv(root / 'data/processed/exoplanets_clean.csv')

figures = []
def save(name, title, count, description):
    plt.title(title)
    plt.tight_layout()
    plt.savefig(root / f'figures/{name}.png', dpi=200, bbox_inches='tight')
    plt.close()
    figures.append({'file': f'figures/{name}.png', 'title': title,
                    'n': int(count), 'description': description})

def available(*cols):
    return df[list(cols)].replace([np.inf, -np.inf], np.nan).dropna()

x = available('pl_rade').pl_rade
plt.figure(figsize=(8, 5))
sns.histplot(x, bins=40)
plt.xlabel('Planet radius (Earth radii)')
plt.ylabel('Number of planets')
save('01_radius', 'Planet radius distribution', len(x),
     'Distribution among planets with reported positive radii; missing radii are excluded.')

x = available('pl_bmasse').pl_bmasse
plt.figure(figsize=(8, 5))
sns.histplot(x, bins=40)
plt.xscale('log')
plt.xlabel('Best available planet mass (Earth masses, log scale)')
plt.ylabel('Number of planets')
save('02_mass', 'Planet mass distribution', len(x),
     'Log-scaled masses among planets with mass estimates; different mass-estimation methods may be represented.')

x = available('pl_orbper').pl_orbper
plt.figure(figsize=(8, 5))
sns.histplot(x, bins=40)
plt.xscale('log')
plt.xlabel('Orbital period (days, log scale)')
plt.ylabel('Number of planets')
save('03_period', 'Orbital period distribution', len(x),
     'Reported positive orbital periods on a logarithmic axis; missing periods are excluded.')

x = available('pl_rade').pl_rade
plt.figure(figsize=(8, 3.5))
sns.boxplot(x=x)
plt.xlabel('Planet radius (Earth radii)')
save('04_radius_boxplot', 'Planet radius: spread and outliers', len(x),
     'Boxplot whiskers use a statistical convention; points beyond them are not automatically errors.')

x = available('pl_bmasse', 'pl_rade')
plt.figure(figsize=(8, 5))
sns.scatterplot(data=x, x='pl_bmasse', y='pl_rade', alpha=.35, s=18, legend=False)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Best available planet mass (Earth masses, log scale)')
plt.ylabel('Planet radius (Earth radii, log scale)')
save('05_mass_radius', 'Planet mass versus radius', len(x),
     'Only planets with both measurements appear; association does not establish causation.')

x = available('pl_orbper', 'pl_orbsmax')
plt.figure(figsize=(8, 5))
sns.scatterplot(data=x, x='pl_orbper', y='pl_orbsmax', alpha=.35, s=18, legend=False)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Orbital period (days, log scale)')
plt.ylabel('Semi-major axis (AU, log scale)')
save('06_period_distance', 'Orbital period versus orbital distance', len(x),
     'A relationship is expected from orbital mechanics; host-star mass also matters.')

x = df.discoverymethod.dropna().value_counts()
plt.figure(figsize=(9, 5))
sns.barplot(x=x.values, y=x.index, color='steelblue')
plt.xlabel('Number of confirmed planets')
plt.ylabel('Discovery method')
save('07_methods', 'Planets by discovery method', x.sum(),
     'These are catalogue counts, not a comparison of intrinsic method quality or universe-wide occurrence.')

x = df.disc_year.dropna().astype(int).value_counts().sort_index()
plt.figure(figsize=(9, 5))
plt.plot(x.index, x.values, marker='o', markersize=2)
plt.xlabel('Discovery year')
plt.ylabel('Number of planets in catalogue')
save('08_year', 'Discoveries by year', x.sum(),
     'Counts reflect discoveries recorded in this catalogue; the current year may be incomplete.')

x = available('st_teff', 'pl_rade')
plt.figure(figsize=(8, 5))
sns.scatterplot(data=x, x='st_teff', y='pl_rade', alpha=.35, s=18, legend=False)
plt.yscale('log')
plt.xlabel('Host-star effective temperature (K)')
plt.ylabel('Planet radius (Earth radii, log scale)')
save('09_star_temp', 'Host-star temperature versus planet radius', len(x),
     'Only systems with both measurements appear; discovery selection may influence the pattern.')

features = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax',
            'pl_orbeccen', 'st_teff', 'st_mass', 'st_rad', 'sy_dist']
x = df[features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(x, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
save('10_correlations', 'Pairwise Pearson correlations', len(df),
     'Each correlation uses available pairs and may have a different sample size; Pearson r measures linear association only.')
df[features].notna().astype(int).T.dot(df[features].notna().astype(int)).to_csv(root / 'reports/correlation_pair_counts.csv')

print('Planets:', len(df))
print('Most common discovery method:', df.discoverymethod.value_counts().head(1).to_dict())
print('Median radius:', df.pl_rade.median())
print('Median mass:', df.pl_bmasse.median())
print('Median orbital period:', df.pl_orbper.median())
print('Pairwise mass–radius correlation:', df[['pl_bmasse','pl_rade']].corr().iloc[0,1])
print('Figures:', len(figures))
assert len(figures) == 10
(root / 'reports/figures.json').write_text(json.dumps(figures, indent=2))
with (root / 'reports/key_numbers.txt').open('w') as f:
    f.write(f"Planets: {len(df)}\nMedian radius (Earth radii): {df.pl_rade.median():.3f}\n"
            f"Median mass (Earth masses): {df.pl_bmasse.median():.3f}\n"
            f"Median orbital period (days): {df.pl_orbper.median():.3f}\n"
            f"Mass-radius Pearson r: {df[['pl_bmasse','pl_rade']].corr().iloc[0,1]:.3f}\n")

archive = root.parent / 'exoplanet_outputs.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for file in root.rglob('*'):
        if file.is_file():
            z.write(file, file.relative_to(root))
print('Saved:', archive)
