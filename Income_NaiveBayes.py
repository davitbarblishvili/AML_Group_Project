import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# i_df = pd.read_csv('df_preprocessed.csv', index_col=0)
# train_df, test_df = train_test_split(i_df, test_size=0.2, random_state=42,stratify=i_df['income'])

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_dev = train_df.drop('income', axis=1)
y_dev = train_df['income']
X_test = test_df.drop('income', axis=1)
y_test = test_df['income']

def min_max_scale(X_dev, X_test):
    scaler = MinMaxScaler()
    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)

def run_GNB():
    print("====================================")
    print("Naive Bayes")
    print("---------")
    
    min_max_scale(X_dev, X_test)
    
    gnb = GaussianNB()
    print(f"Training on X_dev with {X_dev.shape[0]} samples")
    t_start = time.time()
    gnb.fit(X_dev, y_dev)
    t_end = time.time()
    print(f"train time = {t_end - t_start}")
    p_start = time.time()
    ypred_gnb = gnb.predict(X_test)
    p_end = time.time()
    print(f"prediction time = {p_end - p_start}")
    print("test accuracy = ", accuracy_score(y_test, ypred_gnb))
    print('Accuracy score:', round(accuracy_score(y_test, ypred_gnb) * 100, 2))
    print('F1 score:', round(f1_score(y_test, ypred_gnb) * 100, 2))
    print()
    results = {
        "train_time": t_end - t_start,
        "pred_time": p_end - p_start,
        "test_acc": accuracy_score(y_test, ypred_gnb),
        "acc_score": round(accuracy_score(y_test, ypred_gnb) * 100, 2),
        "f1_score": round(f1_score(y_test, ypred_gnb) * 100, 2)
    }
    return results
    
def run_GNB_SMOTE():
    print("====================================")
    print("Naive Bayes with SMOTE")
    print("---------")
    
    min_max_scale(X_dev, X_test)
    
    gnb = GaussianNB()
    smote = SMOTE(random_state=42)
    X_dev_smote, y_dev_smote = smote.fit_resample(X_dev, y_dev)
    print(f"Training on X_dev_smote with {X_dev_smote.shape[0]} samples")
    t_start = time.time()
    gnb.fit(X_dev_smote, y_dev_smote)
    t_end = time.time()
    print(f"train time = {t_end - t_start}")
    p_start = time.time()
    ypred_gnb_smote = gnb.predict(X_test)
    p_end = time.time()
    print(f"prediction time = {p_end - p_start}")
    print("test accuracy= ", accuracy_score(y_test, ypred_gnb_smote))
    print('Accuracy score:', round(accuracy_score(y_test, ypred_gnb_smote) * 100, 2))
    print('F1 score:', round(f1_score(y_test, ypred_gnb_smote) * 100, 2))
    print()
    results = {
        "train_time": t_end - t_start,
        "pred_time": p_end - p_start,
        "test_acc": accuracy_score(y_test, ypred_gnb_smote),
        "acc_score": round(accuracy_score(y_test, ypred_gnb_smote) * 100, 2),
        "f1_score": round(f1_score(y_test, ypred_gnb_smote) * 100, 2)
    }
    return results
    
def run_GNB_ros():
    print("====================================")
    print("Naive Bayes with Random Oversampling")
    print("---------")
    
    min_max_scale(X_dev, X_test)
    
    gnb = GaussianNB()
    ros = RandomOverSampler(random_state=42)
    X_dev_ros, y_dev_ros = ros.fit_resample(X_dev, y_dev)
    print(f"Training on X_dev_ros with {X_dev_ros.shape[0]} samples")
    t_start = time.time()
    gnb.fit(X_dev_ros, y_dev_ros)
    t_end = time.time()
    print(f"train time = {t_end - t_start}")
    p_start = time.time()
    ypred_gnb_ros = gnb.predict(X_test)
    p_end = time.time()
    print(f"prediction time = {p_end - p_start}")
    print("test accuracy= ", accuracy_score(y_test, ypred_gnb_ros))
    print('Accuracy score:', round(accuracy_score(y_test, ypred_gnb_ros) * 100, 2))
    print('F1 score:', round(f1_score(y_test, ypred_gnb_ros) * 100, 2))
    print()
    results = {
        "train_time": t_end - t_start,
        "pred_time": p_end - p_start,
        "test_acc": accuracy_score(y_test, ypred_gnb_ros),
        "acc_score": round(accuracy_score(y_test, ypred_gnb_ros) * 100, 2),
        "f1_score": round(f1_score(y_test, ypred_gnb_ros) * 100, 2)
    }
    return results

run_GNB()
run_GNB_SMOTE()
run_GNB_ros()
    