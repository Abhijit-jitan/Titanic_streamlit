# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:40:08 2020

@author: ms101
"""


#Model training and saving to pickle file
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

## importing the data
PATH = "C:/Users/ms101/OneDrive/DataScience_ML/projects/Titanic/"

X_train = pd.read_csv(PATH + "X_train.csv")
y_train =  pd.read_csv(PATH + "y_train.csv").squeeze()

## scaling is not performed here as RandomForest performed better without
X_train.drop(['cabin_adv_A',
       'cabin_adv_B', 'cabin_adv_C', 'cabin_adv_D', 'cabin_adv_E',
       'cabin_adv_F', 'cabin_adv_G', 'cabin_adv_T', 'cabin_adv_n',
       'name_title_Capt', 'name_title_Col', 'name_title_Don',
       'name_title_Dona', 'name_title_Dr', 'name_title_Jonkheer',
       'name_title_Lady', 'name_title_Major', 'name_title_Master',
       'name_title_Miss', 'name_title_Mlle', 'name_title_Mme', 'name_title_Mr',
       'name_title_Mrs', 'name_title_Ms', 'name_title_Rev', 'name_title_Sir',
       'name_title_the Countess'], axis = 1 ,inplace = True)

## fitting model

forest_clf = RandomForestClassifier(n_jobs = -1, random_state = 13).fit(X_train,y_train)

pickle.dump(forest_clf, open(PATH + "model.pickle","wb"))