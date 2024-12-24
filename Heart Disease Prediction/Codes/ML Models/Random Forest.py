import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

data = pd.read_csv(r"D:\Python\Heart Disease Prediction\Final Datasets\Final Data.csv")


x = data.drop(columns = ["Unnamed: 0", "Severity"], axis = 1)
y = data["Severity"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
parameters = {"n_estimators": [50, 100, 200],
    "criterion": ["entropy", "gini"],
    "min_samples_split": [2],
    "min_samples_leaf": [1],
    "max_features": ["sqrt"],
    "max_leaf_nodes": [None],
    "min_impurity_decrease": [0.0],
    "bootstrap": [True],
    "oob_score": [False],
    "n_jobs": [None],
    "random_state": [None]}

model = RandomForestClassifier()
search = GridSearchCV(estimator=model, param_grid=parameters, cv = 5)
search.fit(X_train, y_train)

print(search.score(X_test, y_test))

prediction = search.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
precision = precision_score(y_test, prediction, average= "weighted")
f1score = f1_score(y_test, prediction, average= "weighted")

print(f"Precision: {precision}")
print(f"F1 Score: {f1score}")

single_instance = pd.DataFrame([[1, 0,	1, 1.2791878172588833, 1.5, -0.833333333, 0.6666666666666666, 0, 1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2]], columns = x.columns)
single_prediction = search.predict(single_instance)
print(f"Prediction for single instance: {single_prediction}")