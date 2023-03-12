# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:05:09 2023

@author: khali
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv("ventes_produits.csv")
print(data)
X_train, X_test, y_train, y_test = train_test_split(data[['prix', 'promotion']],
data['ventes'], test_size=0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print("Perte de la Précision : {:.2f}%".format(accuracy * 100))
prix = 10.99
promotion = 0.2
ventes = model.predict([[prix, promotion]])
print("Les ventes prévues pour un prix de {} avec une promotion de {} sont de {} unités".format(prix, promotion, ventes[0]))