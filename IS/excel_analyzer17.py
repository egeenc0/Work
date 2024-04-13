import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming the path is correct and the file is in the right location
path = 'TEST SHEET 433-5 (180  Data sets 253 to 432 and target 433) dated 08 02 2024.xlsx'

# Load the data
df = pd.read_excel(path)
df = df.drop(columns=['Unnamed: 0'])  # Dropping an unnamed column
df = df.drop(index=0)  # Dropping the first row

# Split the data into features and target
X_train = df.iloc[:, :-1]
Y_train = pd.to_numeric(df.iloc[:, -1], errors='coerce')

# Initialize and fit the RandomForestRegressor
model = RandomForestRegressor(n_estimators=10000,  # An extremely high number of trees
                              min_samples_split=2,  # Allows splitting on all samples
                              min_samples_leaf=1,  # Each leaf can have just 1 sample
                              max_depth=None,  # No limit on tree depth
                              bootstrap=False,  # Use the whole dataset for each tree
                              random_state=42)

model.fit(X_train, Y_train)

# Predict using the last row of the dataset as test data
X_test = df.iloc[-1:, :-1]
Y_test = model.predict(X_test)

print(Y_test)
