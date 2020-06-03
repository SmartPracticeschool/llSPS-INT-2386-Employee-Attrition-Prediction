# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 04:47:53 2019

@author: Rammohan
"""

#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import os
os.chdir("C:/Users/Wesilin/Desktop/Machine Learning/employee-attrition")
os.getcwd()

#Read the dataset/Data Preprocessing
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df

df.head()

#checking null values in the dataset
df.isnull().any()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() 

df["Attrition"]=le.fit_transform(df["Attrition"])
df["BusinessTravel"]=le.fit_transform(df["BusinessTravel"])
df["Department"]=le.fit_transform(df["Department"])
df["EducationField"]=le.fit_transform(df["EducationField"])
df["Gender"]=le.fit_transform(df["Gender"])
df["JobRole"]=le.fit_transform(df["JobRole"])
df["MaritalStatus"]=le.fit_transform(df["MaritalStatus"])
df["OverTime"]=le.fit_transform(df["OverTime"])
df["Over18"]=le.fit_transform(df["Over18"])

df1=df.drop(labels="Attrition",axis=1)
df1.head()

x=df1.iloc[:,[0,2,4,17,21]]
y=df.iloc[:,1]



#split the data intlo train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

x_test


#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=10000, random_state=1, n_jobs=-1)

# Train the classifier
clf.fit(x_train, y_train)

import pickle
pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

print(model.predict([[42, 30, 5,2345,1]]))











