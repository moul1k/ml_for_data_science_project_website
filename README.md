# Exploring Exoplanets — website draft

This is a static, minimal research site built from the uploaded Colab result archive and prepared for GitHub Pages.

## Use

Open `index.html` locally, or enable GitHub Pages from the repository's `main` branch and root directory. All dataset, figure and report links use relative paths.

## Verified result inputs

- NASA `pscomppars` snapshot retrieved 2026-09-20 18:07 UTC; 6,366 rows, 12 columns.
- No rows were dropped by the Colab cleaning process; missing values were retained.
- Supplementary Open Exoplanet Catalogue file was downloaded but NOT merged or statistically analyzed.
- Figures 02 and 03 were re-rendered in THIS WEBSITE COPY from the cleaned CSV using logarithmically spaced histogram bins; the uploaded export is unchanged. The other eight figure files are copied unchanged.

## Before submission

1. Add the actual Colab notebook or public repository code link. The uploaded ZIP had no code or notebook.
2. Add at least one appropriately credited introduction image. Current solar SVG is a navigation illustration rather than a sourced astronomical photo.
3. Review the drafted two introductory paragraphs and all figure interpretations as the student's own work; check the course's rules for AI assistance.
4. Check the cited supplementary dataset against your assignment expectations: it is a separate download but not an independent validation sample.
5. Review the scientific question/target definition before beginning model modules. Models and conclusions are placeholders, not fabricated results.
6. Confirm all hosted GitHub Pages links in a fresh browser after deployment.

## Provenance

Exact API request, collection timestamp and secondary URL are in `reports/sources.json`. Charts' original counts and descriptions are in `reports/figures.json`; cleaned summary statistics in `reports/summary_statistics.csv`.

NASA composite-table caveat: https://exoplanetarchive.ipac.caltech.edu/docs/pscp_about.html
