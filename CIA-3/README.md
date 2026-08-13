# CIA-3 — MCA 521-4 Machine Learning (ML for Social Good, Ensemble Challenge)

**Project: Mission Crisis — Disaster Message Triage.** Classifies real disaster messages into
**Critical / High / Low** priority so relief coordinators can route responders to the most
life-threatening requests first.

## What's in this folder

| Item | What it is |
|---|---|
| `mission-crisis-triage/` | **The submission.** Working copy — notebook, script, data, figures, results, docs. |
| `mission-crisis-triage.zip` | Packaged copy of that same folder, ready to upload. Verified identical to the folder. |
| `_duplicates_to_delete/` | Four stray copies that were sitting loose at this level (byte-identical to the ones inside the folder). Safe to delete. |

Everything lives in `mission-crisis-triage/`. If you edit anything there, re-zip before submitting.

## Inside `mission-crisis-triage/`

| File | Purpose |
|---|---|
| `Mission_Crisis_Disaster_Triage.ipynb` | Main deliverable — 45 cells, executed, outputs saved (20/21 code cells have output; the one without is a silent setup cell) |
| `pipeline.py` | Same pipeline as a plain script, jupytext "percent" format (`# %%` = one cell) |
| `data/disaster_messages.csv`, `data/disaster_categories.csv` | Raw Figure Eight / Appen dataset, ~26k messages, 36 binary aid labels |
| `figures/` | 7 PNGs — EDA, model comparison, confusion matrices, 4 SHAP plots |
| `results_comparison.csv` | Final metric table, all four models, held-out test set |
| `model_summary.json` | Run summary — best model, split sizes (18188/3898/3898), class balance |
| `ethics_statement.md` | Bias, privacy, uncertainty, FN-vs-FP cost, human oversight, deployment limits |
| `video_pitch_script.md` | Timed script for the 3-minute pitch video |
| `README.md` | Full project README (dataset citation, priority rule, repro steps) |

## Notebook structure (sections)

1. Real-World Impact Framing
2. Setup
3. Data Wrangling & Feature Engineering — load/merge, parse 36 categories, quality audit, engineer the priority target, EDA, features, stratified split, leakage-safe preprocessing, class imbalance
4. Ensemble Architecture — Decision Tree baseline → Random Forest (bagging) → XGBoost (boosting) → heterogeneous Stacking → comparison + confusion matrices
5. Explainability & Ethics — SHAP global and local, bias/fairness/privacy discussion
6. Live Prediction Demo (for the video)
7. Conclusion

## Headline results (held-out test set)

| Model | Accuracy | F1 (macro) | ROC-AUC (OvR) | Critical recall |
|---|---|---|---|---|
| Baseline — Decision Tree | 0.622 | 0.607 | 0.753 | 0.457 |
| Bagging — Random Forest | 0.690 | 0.683 | 0.833 | 0.570 |
| Boosting — XGBoost | 0.708 | 0.701 | 0.854 | **0.610** |
| **Stacking (RF + XGB + LogReg)** | **0.723** | **0.714** | **0.861** | 0.589 |

Stacking wins on macro-F1 and ROC-AUC (+17.6% relative macro-F1 over baseline); XGBoost alone
has the best Critical-class recall. Both are reported because missing a Critical message is the
costlier error — see `ethics_statement.md`.

## Submission checklist

- [x] Executed notebook with visible outputs
- [x] Baseline + bagging + boosting + stacking, compared on one untouched test set
- [x] EDA, feature engineering, leakage-safe pipeline, class-imbalance handling
- [x] SHAP explainability (global + local)
- [x] Ethics statement
- [x] Pitch script — **video still needs recording**
