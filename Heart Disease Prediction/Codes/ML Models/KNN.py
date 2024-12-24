import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score

data = pd.read_csv(r"D:\Python\Heart Disease Prediction\Final Datasets\Final Data.csv")
x = data.drop(columns=["Unnamed: 0", "Severity"], axis=1)
y = data["Severity"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

parameters = {
    "n_neighbors": [1],  # Common range for K
    "weights": ["distance"],
    "algorithm": ["auto"],
    "p": [1],  # 1 for Manhattan, 2 for Euclidean distance
    "leaf_size": [5],  # Adjust leaf size
}

model = KNeighborsClassifier()
search = GridSearchCV(estimator = model, param_grid = parameters, cv = 5)

search.fit(X_train, y_train)

prediction = search.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, prediction)}")
print(f"Precision: {precision_score(y_test, prediction, average = 'weighted')}")
print(f"F1 Score: {f1_score(y_test, prediction, average = 'weighted')}")

single_instance = pd.DataFrame([[1, 0,	1, 1.2791878172588833, 1.5, -0.833333333, 0.6666666666666666, 0, 1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 2]], columns = x.columns)
single_prediction = search.predict(single_instance)
print(f"Prediction for single instance: {single_prediction}")