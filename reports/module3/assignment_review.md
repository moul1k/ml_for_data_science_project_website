# Module 3 requirement review

## Scope and reproducibility

The shared script `code/05_supervised.py` reads the existing NASA snapshot; it does not download new observations, alter earlier results or replace the other website sections. The target is the archive's recorded discovery method, restricted to Transit and Radial Velocity. It is not a planet-composition or habitability label. Every reported empirical result is derived from saved model outputs.

| Rubric item | Evidence |
| --- | --- |
| Introduction: at least four topic paragraphs, two images, ten questions | Sun page; added supervised-stage paragraph, retained previous images/questions |
| NB Overview, Data Prep, Code, Results, Conclusions | Mars page, five linked subsections |
| Multinomial and Bernoulli descriptions | NB Overview; Gaussian distinguished from the required multinomial fit |
| Labeled sample image and sample download | Shared `labeled_preview.png` and `labeled_sample.csv` |
| Train/test sample images and linked sets | Both tabs show `train_preview.png` / `test_preview.png`, full and sample CSVs |
| Disjoint split and numeric input explained | Both Data Prep sections; `split_manifest.csv` and `training_cv_folds.csv` |
| Multinomial NB Python code link | `code/05_supervised.py`, training-fold quantile event encoding + MultinomialNB |
| NB confusion matrix, accuracy and interpretation | NB Results; baseline, both class recalls, uncertainty, likelihood contrasts |
| DT Overview with two supporting images | Original `tree_concept.png` and theoretical `tree_impurity.png` |
| Gini, entropy, information gain and worked example | Actual depth-two root counts; `worked_split.json`; both impurity decreases |
| Infinite tree discussion | General representation space distinguished from finite training partitions |
| Three different fitted trees | Gini depths 2 and 3; entropy depth 4; full PNGs and text rules |
| DT confusion matrices and accuracy | All three matrices, training/CV/test comparison and class-specific results |
| Topic-specific conclusions and limitations | Both tabs; observational selection, minority errors, derived estimates, group proxy |

## Scientific audit notes

- Eligibility retains 5,836 planets from 5,908 target-method records: 72 lack a finite positive measurement in the six predictors. The remaining 458 catalogue records have other discovery labels and are outside this binary question.
- The split contains 4,702 training / 1,134 test planets. Row IDs, planet names and name-proxy groups are disjoint. Verified host identities are absent; aliases can still cross the split. All 53 names without the recognized suffix are retained as individual name groups.
- The source-row indices are zero-based positions in the unchanged cleaned CSV. They support traceability and are never features. `labeled.csv`, `train.csv` and `test.csv` contain six numeric predictors and a string target; split metadata lives separately.
- NB has six bin events per row. Its selected 18 zero-one columns are non-negative integer counts, but a bag of variable-specific events is an approximate representation of correlated continuous measurements. It does not turn physical measurements into independent repeated events.
- Training-fold transformations are fitted before validation transformation. Only training CV chooses bins, smoothing and the primary tree. The three tree configurations are defined before test evaluation. All holdout results are descriptive final comparisons, not a second tuning stage.
- Conditional bootstrap intervals resample held-out proxy groups with models fixed. They omit training, selection and survey-shift uncertainty. No formal model-superiority significance claim is made.
- Radius/mass may include archive-derived values. Omitting mass with other parameters fixed is a limited sensitivity check; it cannot establish independent physical measurement provenance.
- The worked example uses the real Gini tree root; computing its entropy gain does not assert that the entropy criterion would choose the same threshold.

## Course references reviewed

Dr. Osita Onyejekwe, CSCI 5612, supplied instructor-filled lecture copies:

- Week 08, Naive Bayes: pages 5-8 (Bayes classification), 14-19 (assumptions/likelihoods), 23-29 (counts, training vocabulary, smoothing, limitations).
- Week 09, Decision Trees: pages 5-12 (trees, impurity, worked splits), 15-20 (overfitting/pruning/validation), 23-26 (thresholds, regions and limitations).

No lecture-slide images or PDF copies are published. The rubric additionally requests a Bernoulli NB definition, which is included even though it is not a fitted model here. Neither tab has a course-basis footer.

## Before submission

The supplied rubric says the writing, analysis, explanations, results and conclusions must be the student's own. This implementation and prose were produced with AI assistance at the user's request. The student must reconcile that assistance with the course rule, independently reproduce and understand the analysis, and prepare their own submission text as required. Site implementation alone does not certify compliance with that authorship requirement.

Canvas submission is separate. A links-only Word draft was created in Downloads as `Exoplanet_Project_Part3_Submission.docx`. Its hyperlink relationships and text were checked, but visual rendering could not run because LibreOffice is unavailable in this environment. It needs a visual check in Word before submission. Nothing was submitted to Canvas.
