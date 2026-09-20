from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
import requests

root = Path(__file__).resolve().parents[1] / 'exoplanet_outputs'
for folder in ['data/raw', 'reports']:
    (root / folder).mkdir(parents=True, exist_ok=True)
retrieved = datetime.now(timezone.utc).isoformat()

endpoint = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
columns = ['pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbsmax',
           'pl_orbeccen', 'st_teff', 'st_mass', 'st_rad', 'disc_year',
           'discoverymethod', 'sy_dist']
query = f"SELECT {', '.join(columns)} FROM pscomppars"
response = requests.get(endpoint, params={'query': query, 'format': 'csv'}, timeout=120)
response.raise_for_status()
(root / 'data/raw/nasa_exoplanets.csv').write_bytes(response.content)
raw = pd.read_csv(root / 'data/raw/nasa_exoplanets.csv')
assert set(columns).issubset(raw.columns), 'NASA returned unexpected columns'
assert len(raw) > 0, 'NASA returned no planets'
print('NASA:', raw.shape, '\nGET:', response.url)
print(raw.head())

other_url = ('https://raw.githubusercontent.com/OpenExoplanetCatalogue/'
             'oec_tables/master/comma_separated/open_exoplanet_catalogue.txt')
other = requests.get(other_url, timeout=120)
other.raise_for_status()
(root / 'data/raw/open_exoplanet_catalogue.csv').write_bytes(other.content)
print('Secondary catalogue bytes:', len(other.content))
print('Source:', other_url)

sources = {'retrieved_utc': retrieved, 'nasa_api_endpoint': endpoint,
           'nasa_get_url': response.url, 'nasa_query': query,
           'nasa_table': 'pscomppars', 'secondary_download_url': other_url,
           'secondary_note': 'Supplementary source; not independent of NASA, may be dated.'}
(root / 'reports/sources.json').write_text(json.dumps(sources, indent=2))
