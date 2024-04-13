import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.naive_bayes import GaussianNB


path = 'TEST SHEET 433-1B (78 Data sests 336 to 413 and target 433  dated  08 02 2024.xlsx'


df = pd.read_excel(path)
df = df.drop(columns=['Unnamed: 0'])
df = df.drop(index=0)

X_train = df.iloc[:, :-1]
Y_train = pd.to_numeric(df.iloc[:, -1], errors='coerce')

model = GaussianNB()
model.fit(X_train, Y_train)


X_test = df.iloc[-1:, :-1]
Y_test = model.predict(X_test)

print(Y_test)