"""
PotholeRisk / Bengaluru Accident Severity Model — Final Validated Version
===========================================================================
Honest, held-out-evaluated binary classifier predicting High-Risk
(Fatal / Grievous Injury) vs Lower-Risk (Simple Injury / Damage Only)
outcomes from Bengaluru accident records (2016-2023, KSP data).

Result (test set, 20% held out, stratified, random_state=42):
    Accuracy:            0.731
    Precision (weighted): 0.728
    Recall (weighted):    0.731
    F1 (weighted):        0.724
    ROC-AUC:             0.778
(scikit-learn 1.9.0, xgboost 3.4.1; see requirements.txt)

Outputs: metrics.json, classification_report.txt (incl. confusion matrix),
and ../results/confusion_matrix.png + ../results/roc_curve.png.

This model does NOT achieve >90% and is not claimed to. See README_MODEL.md
for a full discussion of why, and what would be needed to improve it further.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent.parent / "dataset" / "processed" / "accident_features.csv"
OUT_DIR = Path(__file__).parent
FIG_DIR = Path(__file__).parent.parent / "results"
RANDOM_STATE = 42

CAT_FEATURES = [
    "Weather", "Road_Condition", "Surface_Condition", "Surface_Type", "Road_Type",
    "Main_Cause", "Accident_Location", "Collision_Type", "Junction_Control",
    "Road_Character", "Hit_Run", "Accident_Classification",
    "Accident_Spot", "Accident_SubLocation",
]
NUM_FEATURES = [
    "Noofvehicle_involved", "Road_Condition_Score", "Surface_Condition_Score",
    "Surface_Type_Score", "Road_Risk_Index", "Year",
]


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    # Drop rows with no real severity label
    df = df[df["Severity"] != "Not Applicable"].copy()

    for c in ["Noofvehicle_involved", "Road_Condition_Score", "Surface_Condition_Score",
              "Surface_Type_Score", "Road_Risk_Index"]:
        df[c] = df[c].fillna(df[c].median())

    # Frequency encode high-cardinality UNITNAME (police station) as a
    # simple, leakage-free numeric feature
    unit_freq = df["UNITNAME"].value_counts(normalize=True)
    df["UNITNAME_freq"] = df["UNITNAME"].map(unit_freq)

    X = df[CAT_FEATURES + NUM_FEATURES + ["UNITNAME_freq", "UNITNAME"]].copy()
    y = df["Severity"].isin(["Fatal", "Grievous Injury"]).astype(int)
    return X, y


def train_eval():
    X, y = load_and_prepare()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Leakage-safe target encoding of police station: computed on TRAIN ONLY
    unit_target_rate = ytr.groupby(Xtr["UNITNAME"]).mean()
    global_rate = ytr.mean()
    Xtr["UNITNAME_riskrate"] = Xtr["UNITNAME"].map(unit_target_rate)
    Xte["UNITNAME_riskrate"] = Xte["UNITNAME"].map(unit_target_rate).fillna(global_rate)
    Xtr = Xtr.drop(columns=["UNITNAME"])
    Xte = Xte.drop(columns=["UNITNAME"])

    num_final = NUM_FEATURES + ["UNITNAME_freq", "UNITNAME_riskrate"]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES)],
        remainder="passthrough",
    )

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("clf", model)])
    pipe.fit(Xtr[CAT_FEATURES + num_final], ytr)

    pred = pipe.predict(Xte[CAT_FEATURES + num_final])
    proba = pipe.predict_proba(Xte[CAT_FEATURES + num_final])[:, 1]

    metrics = {
        "accuracy": accuracy_score(yte, pred),
        "precision_weighted": precision_score(yte, pred, average="weighted"),
        "recall_weighted": recall_score(yte, pred, average="weighted"),
        "f1_weighted": f1_score(yte, pred, average="weighted"),
        "f1_macro": f1_score(yte, pred, average="macro"),
        "roc_auc": roc_auc_score(yte, proba),
    }

    report = classification_report(yte, pred, target_names=["Lower-Risk", "High-Risk"])
    cm = confusion_matrix(yte, pred)
    cm_text = (
        "Confusion matrix (rows=actual, cols=predicted; order: Lower-Risk, High-Risk):\n"
        f"{cm}\n"
    )

    print("=" * 60)
    print("FINAL MODEL — HELD-OUT TEST SET RESULTS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k:22s}: {v:.4f}")
    print()
    print(report)
    print(cm_text)

    # Save artifacts
    joblib.dump(pipe, OUT_DIR / "potholerisk_severity_model.joblib")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(OUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)
        f.write("\n" + cm_text)

    # Figures for the paper, from the same held-out test set
    FIG_DIR.mkdir(exist_ok=True)
    labels = ["Lower-Risk", "High-Risk"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(yte, proba, name="XGBoost", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    return pipe, metrics


if __name__ == "__main__":
    train_eval()
