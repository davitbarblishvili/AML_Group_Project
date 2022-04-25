import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import timeit as timer
import time
from tensorflow.python.keras.layers import Activation, Dense, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


from sklearn.svm import LinearSVC, SVC

import warnings
warnings.filterwarnings("ignore")

def min_max_scale(X_dev, X_test):
    scaler = MinMaxScaler()
    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)
    return X_dev, X_test

def run_random_forest_ros(X_dev, y_dev ,X_test, y_test):
    X_dev_scaled, X_test_scaled = min_max_scale(X_dev, X_test)

    ros = RandomOverSampler(random_state=42)
    ros.fit(X_dev_scaled, y_dev)

    X_resampled, Y_resampled = ros.fit_resample(X_dev_scaled, y_dev)
    round(Y_resampled.value_counts(normalize=True)
          * 100, 2).astype('str') + ' %'
    ran_for = RandomForestClassifier(random_state=42)
    ran_for.fit(X_resampled, Y_resampled)

    rf_best = RandomForestClassifier(
        max_depth=102, n_estimators=40, random_state=42)

    t_start = time.time()
    rf_best.fit(X_resampled, Y_resampled)
    t_end = time.time()

    p_start = time.time()
    Y_pred_rf_best = rf_best.predict(X_test_scaled)
    p_end = time.time()

    accuracy_score_ = round(accuracy_score(y_test, Y_pred_rf_best) * 100, 2)
    f1_score_ = round(f1_score(y_test, Y_pred_rf_best) * 100, 2)


    results = {
        'Model': "Naive Bayes",
        "Accuracy":{accuracy_score_},
        'F1-Score': {f1_score_}, 
        "Fit-Time":{t_end - t_start}, 
        "Predict-Time": {p_end - p_start}
    }
    return results

def run_knn(neighbors, distance, X_dev, y_dev, X_test, y_test):

    X_dev_scaled, X_test_scaled = min_max_scale(X_dev, X_test)
    model = KNeighborsClassifier(n_neighbors=neighbors, p=distance)

    t_start = time.time()
    model.fit(X_dev_scaled, y_dev)
    t_end = time.time()

    p_start = time.time()
    pred_KNN = model.predict(X_test_scaled)
    p_end = time.time()
    report = metrics.classification_report(
        y_test, pred_KNN,  output_dict=True)

    f1_score = report['macro avg']['f1-score']
    accuracy_ = report['accuracy']

    
    results = {
        'Model': "Naive Bayes",
        "Accuracy":{accuracy_},
        'F1-Score': {f1_score}, 
        "Fit-Time":{t_end - t_start}, 
        "Predict-Time": {p_end - p_start}
    }
    return results

def run_GNB(X_dev_g,y_dev,X_test_g, y_test):

    X_dev = X_dev_g
    X_test = X_test_g
    # X_dev, X_test = min_max_scale(X_dev_g, X_test_g)
    
    gnb = GaussianNB()

    t_start = time.time()
    gnb.fit(X_dev, y_dev)
    t_end = time.time()

    p_start = time.time()
    ypred_gnb = gnb.predict(X_test)
    p_end = time.time()

    #'Model', 'Accuracy', 'F1-Score', 'Fit-Time', 'Predict-Time'
    results = {
        'Model': "Naive Bayes",
        "Accuracy":round(accuracy_score(y_test, ypred_gnb) * 100, 2),
        'F1-Score': round(f1_score(y_test, ypred_gnb) * 100, 2), 
        "Fit-Time":t_end - t_start, 
        "Predict-Time": p_end - p_start
    }
    return results


def run_SVM(X_dev, y_dev, X_test, y_test):

    min_max_scale(X_dev, X_test)
    
    
    svm_poly = SVC(kernel="poly")
    t_start_poly = time.time()
    svm_poly.fit(X_dev, y_dev.ravel(order='C'))
    t_end_poly = time.time()
    pred_train4 = svm_poly.predict(X_dev)
    p_start_poly = time.time()
    pred_test4 = svm_poly.predict(X_test)
    p_end_poly = time.time()


    results = {
        'Model': "SVM",
        "Accuracy": round(accuracy_score(y_test, pred_test4) * 100, 2),
        'F1-Score': round(f1_score(y_test, pred_test4) * 100, 2), 
        "Fit-Time":t_end_poly - t_start_poly, 
        "Predict-Time":p_end_poly - p_start_poly
    }
    return results

def cnn(xtrain,ytrain, xtest,  ytest):
    #turn all data into binary
    xtrain['education.num']=np.where(xtrain['education.num'] >= 13, 1, 0) #turned to check if higher than college degree
    xtrain['capital.gain']=np.where(xtrain['capital.gain'] > 0, 1, 0)
    xtrain['capital.loss']=np.where(xtrain['capital.loss'] > 0, 1, 0)
    xtrain['hours.per.week']=np.where(xtrain['hours.per.week'] >= 40, 1, 0) #turned to check if full time
    xtrain = xtrain.drop(columns='age')
    
    #repeat for test
    xtest['education.num']=np.where(xtest['education.num'] >= 13, 1, 0) #turned to check if higher than college degree
    xtest['capital.gain']=np.where(xtest['capital.gain'] > 0, 1, 0)
    xtest['capital.loss']=np.where(xtest['capital.loss'] > 0, 1, 0)
    xtest['hours.per.week']=np.where(xtest['hours.per.week'] >= 40, 1, 0) #turned to check if full time
    xtest = xtest.drop(columns='age')
    
    #reshape for cnn (as tf tensor)
    xtrain = xtrain.values.reshape(xtrain.shape[0],xtrain.shape[1],1)
    xtrain = tf.cast(xtrain, tf.float32)

    xtest = xtest.values.reshape(xtest.shape[0],xtest.shape[1],1)
    xtest = tf.cast(xtest, tf.float32)
    ytrain = tf.keras.utils.to_categorical(ytrain, 2)
    ytest = tf.keras.utils.to_categorical(ytest, 2)
    
    
    #create model
    model = tf.keras.Sequential()
    # #two convoluted layers with dropout/pooling
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())

    model.add(Conv1D(filters=120, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    #equivalent to 1d sigmoid
    model.add(Dense(2, activation='softmax'))
    model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape= xtrain.shape )
    
    
    
    
    #fit and get predict times
    t1 = timer.default_timer()
    fit_data = model.fit(xtrain,ytrain,epochs=15,batch_size=512, verbose=0)
    time_fit = timer.default_timer() - t1
    t1 = timer.default_timer()
    scores = model.evaluate(xtest,ytest,verbose=0)
    time_predict = timer.default_timer() - t1


    #get f1 score
    predict = model.predict(xtest)

    results = {
        'Model': "CNN",
        "Accuracy":  scores[1],
        'F1-Score': f1_score(tf.argmax(ytest,axis=1).numpy(),tf.argmax(input=predict, axis=1).numpy()), 
        "Fit-Time":time_fit, 
        "Predict-Time":time_predict
    }
    return results
    


def lr(xtrain, ytrain, xtest, ytest):
  
    
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
    

    results = {
        'Model': "LR",
        "Accuracy":  accuracy_score(ytest,predictions) ,
        'F1-Score': f1_score(ytest,predictions), 
        "Fit-Time":time_fit, 
        "Predict-Time":time_predict
    }
    return results


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_x = train_df.drop('income', axis=1)
train_y = train_df['income']

test_x = test_df.drop('income', axis=1)
test_y = test_df['income']


results = pd.DataFrame(columns = ['Model', 'Accuracy', 'F1-Score', 'Fit-Time', 'Predict-Time'])


#Logistic Regression
results = results.append( lr(train_x,train_y,test_x,test_y) , ignore_index = True)
#SVM
results = results.append( run_SVM(train_x,train_y,test_x,test_y) , ignore_index = True)
#KNN
results = results.append( run_knn(20,2, train_x,train_y,test_x,test_y) , ignore_index = True)
#Naive Bayes
results = results.append( run_GNB(train_x,train_y,test_x,test_y) , ignore_index = True)
#Random Forest
results = results.append( run_random_forest_ros(train_x,train_y,test_x,test_y) , ignore_index = True)
#CNN
results = results.append(cnn(train_x,train_y,test_x,test_y) , ignore_index = True)


results.to_csv("results_1.csv",index=False)