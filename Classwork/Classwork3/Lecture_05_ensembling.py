import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

df = pd.read_csv("healthcare-dataset-stroke-data.csv", sep=",")
# shuflle the data
print(df.head())
df = df.sample(frac=1, random_state=42)
print(df.isna().sum())  #  there are no missing values
print(df.shape)  #  5110 rows and 12 columns
print(df.dtypes)  # variables are float and class is int
print(df.head())
# we have 201 missing values in bmi, which constitutes 3.9% of the data, not a lot
# drop the rows with missing values
df.dropna(inplace=True)
# X = df[['age', 'gender', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]
# X['isMale'] = (X['gender'] == 'Male').astype(int)
# X['isMarried'] = (X['ever_married'] == 'Yes').astype(int)
# X['isResident'] = (X['Residence_type'] == 'Urban').astype(int)
# X['isChild'] = (X['work_type'] == 'children').astype(int)
# X['isLazy'] = (X['work_type'] == 'Never_worked').astype(int)
# X['isGovt'] = (X['work_type'] == 'Govt_job').astype(int)
# X['isSmoker'] = (X['smoking_status'] == 'smokes').astype(int)
# X['everSmoked'] = (X['smoking_status'] == 'formerly smoked').astype(int)
# X['isPrivate'] = (X['work_type'] == 'Private').astype(int)
# X = X.drop(['gender', 'ever_married', 'Residence_type', 'smoking_status', 'work_type'], axis=1)
X= pd.get_dummies(df, columns=['ever_married','work_type','Residence_type', 'smoking_status'])
X = X.drop(['id', 'gender', 'stroke'], axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestClassifier(random_state=42, n_estimators=500, max_depth=10, min_samples_split=2, min_samples_leaf=1, oob_score=True, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


ada = AdaBoostClassifier(random_state=42, n_estimators=500, learning_rate=0.1, algorithm='SAMME.R', estimator=DecisionTreeClassifier(max_depth=10))
ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)


xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)



metrics = pd.DataFrame(index=['accuracy', 'recall', 'precision', 'f1', 'roc_auc'])
metrics['rf'] = [accuracy_score(y_test, rf_pred), recall_score(y_test, rf_pred), precision_score(y_test, rf_pred), f1_score(y_test, rf_pred), roc_auc_score(y_test, rf_pred)]
metrics['ada'] = [accuracy_score(y_test, ada_pred), recall_score(y_test, ada_pred), precision_score(y_test, ada_pred), f1_score(y_test, ada_pred), roc_auc_score(y_test, ada_pred)]
metrics['xgb'] = [accuracy_score(y_test, xgb_pred), recall_score(y_test, xgb_pred), precision_score(y_test, xgb_pred), f1_score(y_test, xgb_pred), roc_auc_score(y_test, xgb_pred)]

print(metrics)