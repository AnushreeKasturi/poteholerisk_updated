from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv(Path(__file__).parent.parent / 'dataset' / 'processed' / 'accident_features.csv')

# Drop garbage/placeholder label
df = df[df['Severity'] != 'Not Applicable'].copy()

cat_features = [
    'Weather','Road_Condition','Surface_Condition','Surface_Type','Road_Type',
    'Main_Cause','Accident_Location','Collision_Type','Junction_Control',
    'Road_Character','Hit_Run','Accident_Classification'
]
num_features = [
    'Noofvehicle_involved','Road_Condition_Score','Surface_Condition_Score',
    'Surface_Type_Score','Road_Risk_Index'
]
for c in num_features:
    df[c] = df[c].fillna(df[c].median())

X = df[cat_features + num_features].copy()
y5 = df['Severity'].copy()
y2 = df['Severity'].isin(['Fatal','Grievous Injury']).astype(int)

def run(y, label):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
    ], remainder='passthrough')

    models = {
        'LogReg (balanced)': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'RandomForest (balanced)': RandomForestClassifier(n_estimators=400, max_depth=18, class_weight='balanced', random_state=42, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=300, random_state=42),
    }
    print(f"\n=========== TARGET: {label}  (train={len(Xtr)}, test={len(Xte)}, classes={sorted(y.unique())}) ===========")
    results=[]
    for name, clf in models.items():
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1w = f1_score(yte, pred, average='weighted')
        f1m = f1_score(yte, pred, average='macro')
        results.append((name, acc, f1w, f1m))
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}  Weighted-F1: {f1w:.4f}  Macro-F1: {f1m:.4f}")

    # XGBoost needs numeric-encoded target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)
    xgb_pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], remainder='passthrough')
    Xtr_t = xgb_pre.fit_transform(Xtr)
    Xte_t = xgb_pre.transform(Xte)
    classes, counts = np.unique(ytr_enc, return_counts=True)
    weight_map = {c: len(ytr_enc)/(len(classes)*ct) for c,ct in zip(classes,counts)}
    sw = np.array([weight_map[v] for v in ytr_enc])
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                         colsample_bytree=0.8, eval_metric='mlogloss' if len(classes)>2 else 'logloss',
                         random_state=42, n_jobs=-1)
    xgb.fit(Xtr_t, ytr_enc, sample_weight=sw)
    pred = xgb.predict(Xte_t)
    acc = accuracy_score(yte_enc, pred)
    f1w = f1_score(yte_enc, pred, average='weighted')
    f1m = f1_score(yte_enc, pred, average='macro')
    print(f"\n--- XGBoost (balanced) ---")
    print(f"Accuracy: {acc:.4f}  Weighted-F1: {f1w:.4f}  Macro-F1: {f1m:.4f}")
    print(classification_report(yte_enc, pred, target_names=[str(c) for c in le.classes_]))
    results.append(('XGBoost (balanced)', acc, f1w, f1m))
    return results

r5 = run(y5, "5-class Severity")
r2 = run(y2, "Binary High-Risk (Fatal/Grievous vs rest)")

print("\n\n===== SUMMARY =====")
print("5-class:")
for r in r5: print(f"  {r[0]:30s} acc={r[1]:.3f} f1_weighted={r[2]:.3f} f1_macro={r[3]:.3f}")
print("Binary:")
for r in r2: print(f"  {r[0]:30s} acc={r[1]:.3f} f1_weighted={r[2]:.3f} f1_macro={r[3]:.3f}")
