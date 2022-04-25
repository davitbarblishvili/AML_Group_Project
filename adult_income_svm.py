
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC, SVC
import timeit as timer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_dev = train.drop('income', axis=1)
y_dev = train['income']
X_test = test.drop('income', axis=1)
y_test = test['income']

def min_max_scale(X_dev, X_test):
    scaler = MinMaxScaler()
    X_dev = scaler.fit_transform(X_dev)
    X_test = scaler.transform(X_test)

def run_SVM():
    print("====================================")
    print("SVM with Polynomial kernel")
    print("---------")

    min_max_scale(X_dev, X_test)
    
    
    svm_poly = SVC(kernel="poly")
    t_start_poly = timer.default_timer()
    svm_poly.fit(X_dev, y_dev.ravel(order='C'))
    t_end_poly = timer.default_timer()
    pred_train4 = svm_poly.predict(X_dev)
    p_start_poly = timer.default_timer()
    pred_test4 = svm_poly.predict(X_test)
    p_end_poly = timer.default_timer()

    print(f"Rbf kernel SVM train time = {t_end_poly - t_start_poly}")
    print(f"Rbf kernel SVM prediction time = {p_end_poly - p_start_poly}")
    print("test accuracy= ", accuracy_score(y_test, pred_test4))
    print('Accuracy score:', round(accuracy_score(y_test, pred_test4) * 100, 2))
    print('F1 score:', round(f1_score(y_test, pred_test4) * 100, 2))
    print()

    results = {
        "train_time": t_end_poly - t_start_poly,
        "pred_time": p_end_poly - p_start_poly,
        "test_acc": accuracy_score(y_test, pred_test4),
        "acc_score": round(accuracy_score(y_test, pred_test4) * 100, 2),
        "f1_score": round(f1_score(y_test, pred_test4) * 100, 2)
    }
    return results

run_SVM()


