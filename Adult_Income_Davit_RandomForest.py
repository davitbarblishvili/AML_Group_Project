import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_dev = train_df.drop('income', axis=1)
y_dev = train_df['income']
X_test = test_df.drop('income', axis=1)
y_test = test_df['income']

scaler = MinMaxScaler()
X_dev = scaler.fit_transform(X_dev)
X_test = scaler.transform(X_test)


def run_random_forest():

    ros = RandomOverSampler(random_state=42)
    ros.fit(X_dev, y_dev)

    X_resampled, Y_resampled = ros.fit_resample(X_dev, y_dev)
    round(Y_resampled.value_counts(normalize=True)
          * 100, 2).astype('str') + ' %'
    ran_for = RandomForestClassifier(random_state=42)
    ran_for.fit(X_resampled, Y_resampled)
    Y_pred_ran_for = ran_for.predict(X_test)

    prev_acc_score = round(accuracy_score(y_test, Y_pred_ran_for) * 100, 2)
    prev_f1 = round(f1_score(y_test, Y_pred_ran_for) * 100, 2)

    n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
    max_depth = [int(x) for x in np.linspace(40, 150, num=15)]
    param_dist = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
    }
    rf_tuned = RandomForestClassifier(random_state=42)
    rf_cv = RandomizedSearchCV(
        estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)

    rf_cv.fit(X_resampled, Y_resampled)

    rf_best = RandomForestClassifier(
        max_depth=102, n_estimators=40, random_state=42)

    t_start = time.time()
    rf_best.fit(X_resampled, Y_resampled)
    t_end = time.time()

    p_start = time.time()
    Y_pred_rf_best = rf_best.predict(X_test)
    p_end = time.time()

    accuracy_score_ = round(accuracy_score(y_test, Y_pred_rf_best) * 100, 2)
    f1_score_ = round(f1_score(y_test, Y_pred_rf_best) * 100, 2)

    print(f"Random Forest train time = {t_end - t_start}")
    print(f"Random Forest prediction time = {p_end - p_start}")
    print(f"random forest accuracy = {accuracy_score_}")
    print(f"random forest f1_score = {f1_score_}")


run_random_forest()
