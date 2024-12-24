import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score

data = pd.read_csv(r"D:\Python\Heart Disease Prediction\Final Datasets\Final Data.csv")

x = data.drop(columns = ["Severity"], axis = 1)
y = data["Severity"]

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

parameters = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

model = tree.DecisionTreeClassifier(random_state = 42)

search = GridSearchCV(estimator = model, param_grid = parameters, cv = 5, n_jobs=1)

search.fit(X_train, y_train)
print(search.score(X_test, y_test))

prediction = search.predict(X_test)
print(prediction)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")

precision = precision_score(y_test, prediction, average = "weighted")
f1score = f1_score(y_test, prediction, average= "weighted")

print(f"precision: {precision}")
print(f"f1_score: {f1score}")

