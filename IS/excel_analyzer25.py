import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming the path is correct and the file is in the right location
path = 'TEST SHEET 433-3 (170 Data sets 263 to 432 and target 433 dated 08 02  2024.xlsx'

# Load the data
df = pd.read_excel(path)
df = df.drop(columns=['Unnamed: 0'])  # Dropping an unnamed column
df = df.drop(index=0)  # Dropping the first row

# Split the data into features and target
X = df.iloc[:, :-1]
Y = pd.to_numeric(df.iloc[:, -1], errors='coerce')

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the SVM classifier with parameters that encourage overfitting
model = SVC(kernel='rbf', C=1000, gamma='scale')  # 'scale' uses 1 / (n_features * X.var()) as the value of gamma

# Fit the model on the scaled features
model.fit(X_scaled, Y)

X_test = X.iloc[-1:, :]  # This ensures all features are included
X_test_scaled = scaler.transform(X_test)  # Scale the test data
Y_test = model.predict(X_test_scaled)

print(Y_test)
