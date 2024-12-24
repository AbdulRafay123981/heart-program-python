import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load your dataset
data = pd.read_csv(r"D:\Python\Heart Disease Prediction\Final Data\Final_data(2).csv")

# Specify the columns to scale
columns_to_scale = ["Age", "Cholesterol", "Rest BP", "Max Heart Rate (exercise)", "Old Peak"]

# Initialize the RobustScaler
scaling = RobustScaler()

# Create new column names for scaled values
scaled_columns = [col + '_scaled' for col in columns_to_scale]

# Apply scaling and add as new columns
data[scaled_columns] = scaling.fit_transform(data[columns_to_scale])

data = data.drop(["Unnamed: 0", "r", "Age", "Rest BP", "Cholesterol", "Max Heart Rate (exercise)", "Old Peak"], axis = 1)
data = data.rename(columns = {"Normal": "Normal (ECG)", "Normal.1": "Normal (Thal)",
                              "Age_scaled": "Age", "Cholesterol_scaled": "Cholesterol",
                              "Rest BP_scaled": "Rest BP",
                              "Max Heart Rate (exercise)_scaled": "Exercise Heart Rate",
                              "Old Peak_scaled": "Old Peak"})

desired_order = ["Age", "Female", "Male", "Cholesterol", "Rest BP", "Exercise Heart Rate", "Old Peak",
                 "FBS", "EI Angina", "CA", "Severity", "Asymptomatic",
                 "Atypical Angina", "Non-Anginal", "Typical Angina", "LV Hypertrophy",
                 "Normal (ECG)", "ST-T Abnormality", "Downsloping", "Flat", "Upsloping",
                 "Fixed Defect", "Normal (Thal)", "Reversable Defect"]
data = data[desired_order]


# Display the updated dataset with both original and scaled columns
print(data.to_string())

# Optionally, save the updated DataFrame to a new CSV if you want to keep the changes
data.to_csv(r"D:\Python\Heart Disease Prediction\Final Datasets\Final_Scaled Data.csv", index=False)
