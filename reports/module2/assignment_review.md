# Module 2 requirement review

The analysis uses the existing 2026-09-20 NASA snapshot. No scientific findings are inferred from the synthetic overview diagrams.

| Requirement | Evidence in the website |
| --- | --- |
| Introduction has at least three paragraphs and two images | Sun: three topic paragraphs, original transit and radial-velocity schematics, existing ten questions |
| Clustering overview: partitional, hierarchical, distance measures, discovery aim, two images | Venus: Overview, synthetic centroid and hierarchy diagrams |
| Unlabeled numeric preparation, sample image and linked data | Venus: Data Prep, `model_preview.png`, numeric sample and full model matrix |
| Python k-means and cosine hierarchical clustering | `code/04_unsupervised.py`: k-means++ with 30 starts; full-row cosine average linkage |
| At least three k values, silhouette visualization and best-k discussion | k=2 through 8; elbow, sampled silhouette, detailed selected-partition silhouette and k=2/3/4 comparison |
| Dendrogram, other clustering image, method agreement, suggested hierarchical k | Full-data dendrogram with cut line; shared PCA projections, original-unit mass-radius plot, ARI and contingency table |
| Topic-specific clustering conclusions | Physical medians, sample stability, redundant-feature sensitivity and selection limitations |
| PCA explanation, eigenvalues/eigenvectors, dimensionality reduction, two images | Earth: Overview and two original synthetic diagrams |
| PCA data format, sample image and data link | Earth: Data Prep and linked shared finite numeric input |
| PCA code link and at least two empirical visualizations | Earth: Code; variance curves, held-out-method projection and variable-score correlations |
| PCA loadings and topic-specific conclusions | First-two-direction coefficient table, correlations, interpretation and retention rationale |
| Week 6 centered SVD additions | Earth: SVD and Reconstruction; singular spectrum, cumulative energy, rank-four reconstruction error and PCA/SVD equivalence |

## Scientific limits

- Complete-case analysis retains 5,141 of 6,366 planets and excludes all microlensing records; results do not represent the full catalogue or galactic occurrence rates.
- Silhouette scores are descriptive estimates from a common seeded 2,000-row subset, not full-data scores or confidence intervals. The full matrix is used to fit both methods.
- Both algorithms favour k=2 over the tested range, but their memberships agree only moderately. Five random 80% subsamples assess sample sensitivity; correlated-feature omission substantially changes memberships. Shared host systems are not resampled as independent units.
- The source does not contain measurement uncertainties, per-value provenance or planet-composition labels. Archive-derived masses/radii and redundant orbital quantities can strengthen apparent correlations.
- Scripts satisfy the stated Python-code-link requirement. No original Colab notebook has been supplied. Institutional or Canvas-specific submission details not contained in the request still need the student's final review.

## Reproduction

Run `python code/04_unsupervised.py` after installing `code/requirements.txt`. The source checksum, feature transforms, scaler parameters and library versions are in `preparation.json`. Model selections, ARI sensitivity checks and numerical PCA/SVD checks are in `results.json`. Detailed CSVs retain assignments, profiles, loadings, scores, linkage and sampled silhouette indices.

Course reference material: Dr. Osita Onyejekwe, CSCI 5612, Week 4 Clustering; Week 5 PCA and Dimensionality Reduction; Week 6 Singular Value Decomposition, supplied instructor-filled copies. Course PDFs are referenced in the text without publishing copies.
