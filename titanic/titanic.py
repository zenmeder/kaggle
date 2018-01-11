#!/usr/local/bin/ python3
# -*- coding:utf-8 -*-
# __author__ = "zenmeder"

import pandas as pd
import numpy as np
# read data from train set
data = pd.read_csv('train.csv')
data = data.iloc[:,[1,2,4,5,6,7,9,11]]
# print(data.isnull().sum())
data = data.dropna()
def gender(x):
    return 0 if x=='male' else 1
def embark(x):
    d = {'C':0,'Q':1,'S':2}
    return d[x]
data.iloc[:]['Sex'] = data.iloc[:]['Sex'].apply(gender)
data.iloc[:]['Embarked'] = data.iloc[:]['Embarked'].apply(embark)

X_train = data.iloc[:,1:]
y_train = data.iloc[:,0]
test_data = pd.read_csv('test.csv')
test_data = test_data.iloc[:,[0,1,3,4,5,6,8,10]]
test_data['Age'] = test_data['Age'].fillna('35').apply(float)
test_data['Sex'] = test_data['Sex'].apply(gender)
test_data['Embarked'] = test_data['Embarked'].apply(embark)

test_data['Fare'] = test_data['Fare'].fillna('1').apply(float)
X_test = test_data.iloc[:,1:]
# 1. LR 76%
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
# lr.fit(X_train,y_train)
# y_prob = lr.predict_proba(X_test)[:,1]
# y_pred = np.where(y_prob>0.5,1,0)
# 2. SVM 59%
# from sklearn.svm import SVC
# svc = SVC()
# svc.fit(X_train, y_train)
# y_test = svc.predict(X_test)
# df = pd.DataFrame({'PassengerId':test_data.iloc[:,0], 'Survived':y_test})
# df.to_csv('SVMsubmission.csv', index=False)
# 3. PCA
from sklearn.decomposition import PCA
pca = PCA()
# standarised data
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
pca.fit_transform(X_train,y_train)
print(pca.score(X_test))
