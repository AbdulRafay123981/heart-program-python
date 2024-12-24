import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\Python\Heart Disease Prediction\Disease_Data.csv")

#Encoding Gender column
dummies_Gender = pd.get_dummies(data["Gender"], dtype = int)
data = pd.concat([data, dummies_Gender], axis = "columns")

#Encoding Chest Pain Column
dummies_CP = pd.get_dummies(data["Chest Pain"], dtype = int)
data = pd.concat([data, dummies_CP], axis = "columns")

#Encoding Rest ECG Column
dummies_ECG = pd.get_dummies(data["Rest ECG"], dtype = int)
data = pd.concat([data, dummies_ECG], axis = "columns")

#Encoding Slope
dummies_slope = pd.get_dummies(data["Slope"], dtype = int)
data = pd.concat([data, dummies_slope], axis = "columns")

#Encoding thal column
dummies_thal = pd.get_dummies(data["Thal"], dtype= int)
data = pd.concat([data, dummies_thal], axis = "columns")

data = data.drop(["Unnamed: 0", "Gender", "Chest Pain", "Rest ECG", "Slope", "Thal"], axis = "columns")

data.to_csv(r"D:\Python\Heart Disease Prediction\Encoded_Data.csv")
print(data.to_string())

