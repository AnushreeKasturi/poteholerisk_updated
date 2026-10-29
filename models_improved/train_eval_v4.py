from pathlib import Path
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier

df = pd.read_csv(Path(__file__).parent.parent / 'dataset' / 'processed' / 'accident_features.csv')
df = df[df['Severity'] != 'Not Applicable'].copy()

# Real official govt data: district x year crash/death totals (Karnataka State Police via data.opencity.in)
govt = {
    ('Bengaluru City',2020):(3233,646), ('Bengaluru City',2021):(3213,654),
    ('Bengaluru City',2022):(3822,772), ('Bengaluru City',2023):(4974,915),
    ('Bengaluru Dist',2020):(1850,567), ('Bengaluru Dist',2021):(1743,519),
    ('Bengaluru Dist',2022):(2129,613), ('Bengaluru Dist',2023):(2451,694),
}
def fat_rate(row):
    key=(row['DISTRICTNAME'], row['Year'])
    if key in govt:
        crashes,deaths = govt[key]
        return deaths/crashes
    return np.nan
df['District_Year_FatalityRate'] = df.apply(fat_rate, axis=1)
median_rate = df['District_Year_FatalityRate'].median()
df['District_Year_FatalityRate'] = df['District_Year_FatalityRate'].fillna(median_rate)

unit_freq = df['UNITNAME'].value_counts(normalize=True)
df['UNITNAME_freq'] = df['UNITNAME'].map(unit_freq)

cat_features = ['Weather','Road_Condition','Surface_Condition','Surface_Type','Road_Type',
    'Main_Cause','Accident_Location','Collision_Type','Junction_Control','Road_Character',
    'Hit_Run','Accident_Classification','Accident_Spot','Accident_SubLocation']
num_features = ['Noofvehicle_involved','Road_Condition_Score','Surface_Condition_Score',
    'Surface_Type_Score','Road_Risk_Index','Year','UNITNAME_freq','District_Year_FatalityRate']
for c in ['Noofvehicle_involved','Road_Condition_Score','Surface_Condition_Score','Surface_Type_Score','Road_Risk_Index']:
    df[c] = df[c].fillna(df[c].median())

X = df[cat_features + num_features + ['UNITNAME']].copy()
y2 = df['Severity'].isin(['Fatal','Grievous Injury']).astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y2, test_size=0.2, random_state=42, stratify=y2)
unit_target_rate = ytr.groupby(Xtr['UNITNAME']).mean()
global_rate = ytr.mean()
Xtr['UNITNAME_riskrate'] = Xtr['UNITNAME'].map(unit_target_rate)
Xte['UNITNAME_riskrate'] = Xte['UNITNAME'].map(unit_target_rate).fillna(global_rate)
Xtr = Xtr.drop(columns=['UNITNAME']); Xte = Xte.drop(columns=['UNITNAME'])
num_final = num_features + ['UNITNAME_riskrate']

pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)], remainder='passthrough')
Xtr_t = pre.fit_transform(Xtr[cat_features+num_final])
Xte_t = pre.transform(Xte[cat_features+num_final])

xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                     colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=4)
xgb.fit(Xtr_t, ytr)
pred = xgb.predict(Xte_t)
print("--- XGBoost + real govt district/year fatality-rate feature ---")
print("Accuracy:", accuracy_score(yte, pred))
print("Precision(weighted):", precision_score(yte, pred, average='weighted'))
print("Recall(weighted):", recall_score(yte, pred, average='weighted'))
print("F1(weighted):", f1_score(yte, pred, average='weighted'))

importances = pd.Series(xgb.feature_importances_, index=pre.get_feature_names_out())
print(importances.sort_values(ascending=False).head(10))
