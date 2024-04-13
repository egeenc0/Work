import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np

df = pd.read_csv('TEST SHEET 388-5 (135  Data sets 253 to 387 and target 388) dated 18 01 2024(Formatted)_Corrected (3).csv')

# X_train, son satır hariç tüm satırlar ve sondaki 7 sütun hariç tüm sütunlar
X_train = df.iloc[:-1, :-7]

# X_test, yalnızca son satır ve sondaki 7 sütun hariç tüm sütunlar
X_test = df.iloc[-1:, :-7]

# Y_train, son satır hariç tüm satırlardan yalnızca sondaki 7 sütun
Y_train = df.iloc[:-1, -7:]

model = GaussianNB()

print(X_train.shape,Y_train.shape)

model.fit(X_train, Y_train)

Y_test = model.predict(X_test)
print(Y_test)