from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import timeit as timer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def lr():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    xtrain = train.loc[:, train.columns != 'income']
    ytrain = train['income']
    xtest = test.loc[:, test.columns != 'income']
    ytest = test['income']
    
    
    Standardize_Var = ['age']
    Standardize_transformer = Pipeline(steps=[('standard', StandardScaler())])
    Normalize_Var = ['education.num','capital.gain','capital.loss','hours.per.week']
    Normalize_transformer = Pipeline(steps=[('norm', MinMaxScaler())])


    preprocessor = ColumnTransformer(transformers= 
                                     [('standard', Standardize_transformer, Standardize_Var), 
                                      ('norm', Normalize_transformer, Normalize_Var)])
    preprocessor = preprocessor.fit(xtrain)

    xtrain_scaled = preprocessor.transform(xtrain)
    xtest_scaled = preprocessor.transform(xtest)

    xtrain[['age','education.num','capital.gain','capital.loss','hours.per.week']] = xtrain_scaled
    xtest[['age','education.num','capital.gain','capital.loss','hours.per.week']] = xtest_scaled
    
    
    lr = LogisticRegression(max_iter=35)
    t1 = timer.default_timer()
    lr.fit(xtrain, ytrain)
    time_fit = timer.default_timer() - t1
    
    t1 = timer.default_timer()
    predictions = lr.predict(xtest)
    time_predict = timer.default_timer() - t1
    
    
    print("Fit time: ", time_fit)
    print("Predict time: ", time_predict)
    print("Accuracy: ", accuracy_score(ytest,predictions) )
    print("F1 score: ", f1_score(ytest,predictions) )
