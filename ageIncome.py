# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:56:26 2023

@author: khali
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
# Chargement des données à partir du fichier CSV
data = pd.read_csv('age_income.csv')
print(data)
X = data['age'].values.reshape(-1, 1)
y = data['income'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
new_X = [[25], [30], [55]]
predicted_y = model.predict(new_X)
print(predicted_y)
