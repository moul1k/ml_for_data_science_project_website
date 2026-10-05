# Exoplanet research — Python scripts
```
ml_for_data_science_project_website/
├── code/
│   ├── 01_collect.py
│   ├── 02_clean.py
│   ├── 03_eda.py
│   ├── 04_unsupervised.py
│   └── requirements.txt
└── exoplanet_outputs/          
```

Run from a terminal (or Colab after uploading the files into a repository folder):

```bash
pip install -r code/requirements.txt
python code/01_collect.py
python code/02_clean.py
python code/03_eda.py
```

The scripts resolve their output location relative to their own `code/` folder, not the current working directory. They download NASA's `pscomppars` table and a supplementary catalogue, save raw/clean CSVs and reports, generate 10 figures plus two data previews, and create `exoplanet_outputs.zip` beside the `code/` folder. The supplement is saved for provenance; it is not merged into the primary dataset. NASA results can change when re-downloaded.

Note: these are a minimal split of the provided notebook, not a new analysis. A live API run requires internet access. Commit the scripts; separately choose which data outputs and figure images to track for the website.

## Module 2: fixed-snapshot unsupervised learning

Run from the repository root after installing `code/requirements.txt`:

```bash
python code/04_unsupervised.py
```

This stage reads the existing `data/processed/exoplanets_clean.csv`; it does not re-download the catalogue or replace Module 1 outputs. It writes numeric input matrices to `data/processed/module2/`, figures to `figures/module2/`, and model diagnostics to `reports/module2/`.

The pipeline selects complete rows for eight physical features, logs seven positive columns and standardizes all eight. Names and discovery methods remain outside the model matrix. It fits k-means for k=2–8, constructs an all-row cosine average-linkage hierarchy, and evaluates silhouette on a shared fixed 2,000-row subset. PCA and an independently computed centered NumPy SVD are checked for score/variance equivalence and reconstruction error. Five 80% subsample refits and a test omitting semi-major axis quantify sensitivity.

`preparation.json` records the exact source SHA-256, transforms, scaler parameters, row counts and library versions. `results.json`, CSV reports, row metadata and evaluation indices support every reported finding. Figure overview schematics are original synthetic teaching illustrations, explicitly separate from observed exoplanet results. No course slide images are redistributed.
