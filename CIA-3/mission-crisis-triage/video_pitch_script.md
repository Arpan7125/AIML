# 3-Minute Pitch & Demo Script — Mission Crisis: Disaster Message Triage

Record your screen over the notebook (scroll to the matching section as you talk). Suggested
pacing below adds up to ~3:00 — practice once to fit your own speaking speed.

## 0:00–0:35 — Problem & beneficiaries
> "During a disaster, relief agencies get flooded with thousands of messages — but only a
> fraction describe an actual life-threatening emergency. Manually reading every message
> to find the ones that matter is too slow, and a delay can cost lives. I built a model
> that automatically triages incoming disaster messages into Critical, High, and Low
> priority, so responders see the most urgent requests first. The beneficiaries are disaster
> relief NGOs and emergency coordination centres — and, indirectly, the people whose
> messages get seen faster."

*(Show: Section 1 of the notebook — problem framing.)*

## 0:35–1:15 — Data cleaning, EDA, feature engineering
> "I used a real dataset of about 26,000 messages sent during actual disasters, labelled by
> Figure Eight. I cleaned it — removed duplicates, dropped an invalid label, handled missing
> translations — then engineered a 3-class priority target from the 36 original aid
> categories. The class distribution is imbalanced, as you'd expect — Critical messages are
> the rarest — so I used class-weighting during training. For features, I combined TF-IDF
> text vectors with engineered signals like urgency-word counts and exclamation marks."

*(Show: Section 3 — EDA chart `figures/eda_overview.png`, the priority-assignment rule.)*

## 1:15–2:10 — Baseline vs. bagging, boosting, and stacking
> "I compared four models on the same untouched test set: a Decision Tree baseline, a tuned
> Random Forest for bagging, a tuned XGBoost for boosting, and a heterogeneous stacking
> ensemble combining all three with cross-validation to avoid leakage. The stacking ensemble
> wins on macro-F1 and ROC-AUC — a 17.6% relative improvement over the baseline. But because
> missing a Critical message is the costliest mistake here, I also tracked Critical-class
> recall separately, where XGBoost alone actually comes out slightly ahead."

*(Show: `figures/model_comparison.png` and `figures/confusion_matrices.png`, and the results
table.)*

## 2:10–3:00 — Live prediction, explanation, ethics, limitations
> "Here's a live prediction on a synthetic message that was never in the training data:
> [read the demo message] — the model correctly flags it Critical with 90% confidence. Using
> SHAP, I can see exactly why: words like 'earthquake' and 'rescue' are driving that
> prediction. That's also an honest limitation — this dataset is dominated by one real
> earthquake, so the model has partly learned that specific disaster rather than disasters in
> general, which is exactly why I don't treat this as a fully autonomous system. It's decision
> support: a human coordinator makes the final call, especially on anything flagged Critical
> or low-confidence."

*(Show: Section 6 live-demo cell running live, plus one SHAP plot from Section 5.)*
