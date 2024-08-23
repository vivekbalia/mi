# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:50:18 2022

@author: PADMESH
"""

import pandas as pd
import numpy as np
dataset=pd.read_csv('E:\\ML_dataset\\diabetes.csv')
#segregate the columns into data and target
X=dataset.iloc[:,2:4]
X
Y=dataset.iloc[:,4]
Y
#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109) # 70% training and 30% test
#scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train
X_test=scaler.fit_transform(X_test)
X_test
#load the model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#build the model
model=KNeighborsClassifier(n_neighbors=7)
#train the data
model.fit(X_train,y_train)
#predict class level the test data
y_pred=model.predict(X_test)
y_pred
#find the accuracy
metrics.accuracy_score(y_test,y_pred)
