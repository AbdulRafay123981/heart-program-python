import pandas as pd
from imblearn.over_sampling import SMOTE

# Load the original dataset
data = pd.read_csv(r"D:\Python\Heart Disease Prediction\Final Datasets\Final Balanced Data.csv")

# Split features and target
x = data.drop(columns=["Severity"])
y = data["Severity"]

# Apply SMOTE
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)

# Combine resampled features and target into a new DataFrame
resampled_data = pd.DataFrame(x_resampled, columns=x.columns)
resampled_data['Severity'] = y_resampled

# Save the resampled data to a CSV file
resampled_data.to_csv(r"D:\Python\Heart Disease Prediction\Final Datasets\Final Balanced Data (2).csv", index=False)

# Print the first few rows of the resampled data for verification
print(resampled_data.to_string())
