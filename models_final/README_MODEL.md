# PotholeRisk — Final Accident Severity Model

## What this model does
Predicts whether a Bengaluru road accident is **High-Risk** (Fatal or
Grievous Injury) vs **Lower-Risk** (Simple Injury or Damage Only), using
weather, road/surface condition, collision type, junction control, location
(police-station area), and year, from the Karnataka State Police accident
dataset (2016–2023, 49,932 records).

## Result (held-out test set, 20%, stratified, random_state=42)

| Metric | Score |
|---|---|
| Accuracy | 73.1% |
| Precision (weighted) | 72.8% |
| Recall (weighted) | 73.1% |
| F1 (weighted) | 72.4% |
| F1 (macro) | 70.6% |
| ROC-AUC | 0.778 |

Per class (test set n = 9,847):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Lower-Risk | 0.70 | 0.56 | 0.62 | 3,908 |
| High-Risk | 0.74 | 0.85 | 0.79 | 5,939 |

Confusion matrix (rows = actual, columns = predicted):

| | Pred. Lower-Risk | Pred. High-Risk |
|---|---|---|
| **Actual Lower-Risk** | 2,172 | 1,736 |
| **Actual High-Risk** | 915 | 5,024 |

Full report and confusion matrix in `classification_report.txt`; figures in
`../results/confusion_matrix.png` and `../results/roc_curve.png`.

Results are deterministic for the pinned library versions in
`../requirements.txt`. An earlier run on older library versions gave 72.6%
accuracy; use the numbers above in the paper.

## How this number was reached
Starting point (original baseline models): 38–45% F1 on 5-class severity.
Improvements applied, in order of impact:
1. Fixed broken evaluation — original "Bayesian risk" approach was scored
   on the full dataset with no train/test split, and its apparent F1 (~0.74)
   was mostly an artifact of class imbalance (predicting the majority class).
2. Reframed to binary High-Risk vs Lower-Risk (more learnable, less
   fragmented across 5 sparse classes).
3. Replaced integer LabelEncoding (which falsely implies ordering) with
   proper one-hot encoding.
4. Added previously-unused real columns: `Collision_Type`,
   `Junction_Control`, `Road_Character`, `Hit_Run`, `Accident_Spot`,
   `Accident_SubLocation`.
5. Added leakage-safe, train-only target encoding of police-station area
   (`UNITNAME`) — this and `Year` were the single largest contributors to
   the final gain.
6. Tuned gradient boosting (XGBoost); no class re-weighting is applied.
7. Tried SMOTE, hyperparameter search, and stacked ensembles — none
   improved on this further; performance plateaued, indicating a genuine
   data ceiling rather than a modeling gap.

## Why not higher
Weather, road-surface type, and road-type carry only weak signal for
*severity* specifically — real-world injury severity is driven mainly by
factors this dataset does not contain: vehicle speed at impact, vehicle
type, driver behavior (e.g. alcohol, fatigue), and precise time-of-day.
Two external data sources were checked for additional signal
(Karnataka Police district/year totals; a Kaggle "India Road Accident"
dataset) — the former added no measurable improvement (too coarse), and
the latter showed no learnable signal at all on inspection (internally
inconsistent, likely synthetic) and was discarded rather than used.

## Files
- `train_final_model.py` — full training/evaluation script (reproducible)
- `potholerisk_severity_model.joblib` — trained pipeline (preprocessing + model)
- `metrics.json` — final metrics
- `classification_report.txt` — full per-class report + confusion matrix

## Suggested framing for a paper
Report this as the primary result. State the ceiling explicitly as a
finding (location and temporal features carry more signal than
weather/road-surface features for severity prediction in this dataset),
and note that vehicle- and driver-level data would likely be needed to
improve beyond this range. This is a defensible, reproducible result.
