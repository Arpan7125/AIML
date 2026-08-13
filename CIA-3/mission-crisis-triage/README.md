# Mission Crisis — Disaster Message Triage
### ML for Social Good Ensemble Challenge (CIA 3, MCA 521-4 Machine Learning)

An end-to-end ensemble ML pipeline that classifies incoming disaster messages into
**Critical / High / Low** priority, so relief coordinators can route limited responder
capacity to the most life-threatening requests first.

## Files in this submission

| File | Purpose |
|---|---|
| `Mission_Crisis_Disaster_Triage.ipynb` | Main notebook — full pipeline, already executed with real outputs |
| `pipeline.py` | Same pipeline as a plain script (jupytext "percent" format — every `# %%` is one notebook cell) |
| `data/disaster_messages.csv`, `data/disaster_categories.csv` | Raw dataset (see citation below) |
| `figures/` | Saved EDA, model-comparison, confusion-matrix, and SHAP plots |
| `results_comparison.csv` | Final metric table for all four models on the held-out test set |
| `model_summary.json` | Machine-readable run summary (best model, split sizes, class balance) |
| `ethics_statement.md` | Standalone copy of the responsible-use / ethics discussion |
| `video_pitch_script.md` | Timed script/storyboard for the required 3-minute pitch video |

## Dataset

**Disaster Response Messages** — real messages sent during actual disaster events (the
corpus is heavily weighted toward the 2010 Haiti earthquake response), originally labelled
by **Figure Eight / Appen** and distributed for the Udacity Data Science Nanodegree.

- Public mirror used here: `disaster_messages.csv` + `disaster_categories.csv`
  (also distributed on Kaggle as *"Disaster Response Messages"*).
- **Citation:** Figure Eight / Appen, *Multilingual Disaster Response Messages*, originally
  released via Figure Eight and distributed through Kaggle / the Udacity Data Science
  Nanodegree, 2018.
- ~26,000 rows, 36 binary aid-category labels + `genre` (direct / news / social) per message.
- No personally identifiable or confidential data was added; this is the public research
  release as-is.

## How the priority label was built

The dataset doesn't ship a single "priority" column, so one was engineered from the 36 aid
categories using a fixed, domain-informed (not learned) rule — see Section 3.4 of the
notebook for the exact category lists:

- **Critical** — any life-threat category fires (search & rescue, medical help, security,
  fire, earthquake, missing people).
- **High** — related to the disaster and requests aid, but not immediately life-threatening
  (water, food, shelter, infrastructure, weather, etc.).
- **Low** — unrelated / purely informational.

## Reproducing this run

```bash
pip install --break-system-packages pandas scikit-learn xgboost shap matplotlib seaborn \
    jupytext nbconvert nbformat ipykernel

# option A — run as a script
python3 pipeline.py

# option B — run as a notebook
jupytext --to notebook pipeline.py -o Mission_Crisis_Disaster_Triage.ipynb
jupyter nbconvert --to notebook --execute --inplace Mission_Crisis_Disaster_Triage.ipynb
```

Random seed is fixed (`RANDOM_STATE = 42`) throughout for reproducibility. Note on search
budget: `RandomizedSearchCV` grids are deliberately small (2-fold, 3 candidates) so the whole
pipeline finishes in a few minutes on modest hardware; widen `rf_param_dist` /
`xgb_param_dist` in Section 4 for a more exhaustive search if you have more compute.

## Headline results (held-out test set, never touched during tuning)

| Model | Accuracy | F1 (macro) | ROC-AUC (OvR, macro) | Critical-class recall |
|---|---|---|---|---|
| Baseline — Decision Tree | 0.622 | 0.607 | 0.753 | 0.457 |
| Bagging — Random Forest | 0.690 | 0.683 | 0.833 | 0.570 |
| Boosting — XGBoost | 0.708 | 0.701 | 0.854 | 0.610 |
| **Stacking (RF + XGB + LogReg)** | **0.723** | **0.714** | **0.861** | 0.589 |

The stacking ensemble wins on macro-F1 and ROC-AUC (+17.6% relative macro-F1 over baseline);
XGBoost alone has the single best Critical-class recall. Both numbers are reported explicitly
because, for this use case, missing a Critical message is the costlier error — see
`ethics_statement.md`.
