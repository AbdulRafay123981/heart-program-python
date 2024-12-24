import pandas as pd
data = pd.read_csv(r"D:\Python\Datasets\heart_disease_uci.csv")

#dropping the ID column
data = data.drop(columns = {"id", "dataset"})

#renaming columns
data = data.rename(columns = {"age": "Age", "sex": "Gender", "cp": "Chest Pain",
                              "trestbps": "Rest BP", "chol": "Cholesterol",
                              "fbs": "FBS", "restecg": "Rest ECG",
                              "thalch": "Max Heart Rate (exercise)",
                              "exang": "EI Angina", "oldpeak": "Old Peak",
                              "slope": "Slope", "ca": "CA", "thal": "Thal",
                              "num": "Severity"})

#Improving data in Chest Pain Column
data["Chest Pain"].replace("atypical angina", "Atypical Angina", inplace = True)
data["Chest Pain"].replace("asymptomatic", "Asymptomatic", inplace = True)
data["Chest Pain"].replace("typical angina", "Typical Angina", inplace = True)
data["Chest Pain"].replace("non-anginal", "Non-Anginal", inplace = True)

#cleaning data in Rest BP
mean_BP = data["Rest BP"].mean()
data["Rest BP"].fillna(mean_BP.astype(int), inplace = True)

#Cleaning data in cholesterol columns
mean_ch = data["Cholesterol"].median()
data["Cholesterol"].fillna(mean_ch, inplace = True)
data["Cholesterol"].replace(0, mean_ch, inplace = True)

#cleaning data in FBS column
data["FBS"].fillna("True", inplace = True)

#correcting data in Rest ECG
data["Rest ECG"].fillna("Normal", inplace = True)
data["Rest ECG"].replace("lv hypertrophy", "LV Hypertrophy", inplace = True)
data["Rest ECG"].replace("st-t abnormality", "ST-T Abnormality", inplace = True)
data["Rest ECG"].replace("normal", "Normal", inplace = True)

#Correcting data in Heart rate column
mean_HR = data["Max Heart Rate (exercise)"].mean()
data["Max Heart Rate (exercise)"].fillna(mean_HR.astype(int), inplace = True)

#Filling missing values in Angina column
data["EI Angina"].fillna("True", inplace = True)

#correcting data in old peak column
median_OP = data["Old Peak"].median()
data["Old Peak"].fillna(median_OP, inplace = True)

#Correcting data in slope column
data["Slope"].fillna("Upsloping", inplace = True)
data["Slope"].replace("downsloping", "Downsloping", inplace = True)
data["Slope"].replace("flat", "Flat", inplace = True)
data["Slope"].replace("upsloping", "Upsloping", inplace = True)

#Correcting CA column
mean = data["CA"].mean()
data["CA"].fillna(mean.astype(int), inplace = True)

#Filling missing values in thal column
data["Thal"].fillna("Normal", inplace = True)
data["Thal"].replace("fixed defect", "Fixed Defect", inplace = True)
data["Thal"].replace("normal", "Normal", inplace = True)
data["Thal"].replace("reversable defect", "Reversable Defect", inplace = True)

data.to_csv(r"D:\Python\Datasets\Disease_Data.csv")
print(data.to_string())