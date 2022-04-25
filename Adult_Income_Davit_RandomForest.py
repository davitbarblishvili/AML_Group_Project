import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


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

    return X_dev, X_test


def run_random_forest_ros():
    X_dev_scaled, X_test_scaled = min_max_scale(X_dev, X_test)

    ros = RandomOverSampler(random_state=42)
    ros.fit(X_dev_scaled, y_dev)

    X_resampled, Y_resampled = ros.fit_resample(X_dev_scaled, y_dev)
    round(Y_resampled.value_counts(normalize=True)
          * 100, 2).astype('str') + ' %'
    ran_for = RandomForestClassifier(random_state=42)
    ran_for.fit(X_resampled, Y_resampled)

    #n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
    #max_depth = [int(x) for x in np.linspace(40, 150, num=15)]
    # param_dist = {
    #    'n_estimators': n_estimators,
    #    'max_depth': max_depth,
    # }
    #rf_tuned = RandomForestClassifier(random_state=42)
    # rf_cv = RandomizedSearchCV(
    #    estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)
    #rf_cv.fit(X_resampled, Y_resampled)

    # from RandomizedSearchCV I obtained the best parameters to be:
    # max_depth = 102, n_estimators = 40
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

    print(f"Random Forest (ROS) train time = {t_end - t_start}")
    print(f"Random Forest (ROS) prediction time = {p_end - p_start}")
    print(f"random forest (ROS) accuracy = {accuracy_score_}")
    print(f"random forest (ROS) f1_score = {f1_score_}\n")


def run_random_forest_smote():
    X_dev_scaled, X_test_scaled = min_max_scale(X_dev, X_test)

    smote = SMOTE(random_state=42)
    smote.fit(X_dev_scaled, y_dev)

    X_resampled, Y_resampled = smote.fit_resample(X_dev_scaled, y_dev)
    round(Y_resampled.value_counts(normalize=True)
          * 100, 2).astype('str') + ' %'
    ran_for = RandomForestClassifier(random_state=42)
    ran_for.fit(X_resampled, Y_resampled)

    #n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
    #max_depth = [int(x) for x in np.linspace(40, 150, num=15)]
    # param_dist = {
    #    'n_estimators': n_estimators,
    #    'max_depth': max_depth,
    # }
    #rf_tuned = RandomForestClassifier(random_state=42)
    # rf_cv = RandomizedSearchCV(
    # estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)

    #rf_cv.fit(X_resampled, Y_resampled)

    # from RandomizedSearchCV I obtained the best parameters to be:
    # max_depth = 102, n_estimators = 40
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

    print(f"Random Forest (SMOTE) train time = {t_end - t_start}")
    print(f"Random Forest (SMOTE) prediction time = {p_end - p_start}")
    print(f"random forest (SMOTE) accuracy = {accuracy_score_}")
    print(f"random forest (SMOTE) f1_score = {f1_score_}\n")


def run_random_forest_rus():
    X_dev_scaled, X_test_scaled = min_max_scale(X_dev, X_test)

    rus = RandomUnderSampler(random_state=42)
    rus.fit(X_dev_scaled, y_dev)

    X_resampled, Y_resampled = rus.fit_resample(X_dev_scaled, y_dev)
    round(Y_resampled.value_counts(normalize=True)
          * 100, 2).astype('str') + ' %'
    ran_for = RandomForestClassifier(random_state=42)
    ran_for.fit(X_resampled, Y_resampled)

    #n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
    #max_depth = [int(x) for x in np.linspace(40, 150, num=15)]
    # param_dist = {
    #    'n_estimators': n_estimators,
    #    'max_depth': max_depth,
    # }
    #rf_tuned = RandomForestClassifier(random_state=42)
    # rf_cv = RandomizedSearchCV(
    # estimator=rf_tuned, param_distributions=param_dist, cv=5, random_state=42)

    #rf_cv.fit(X_resampled, Y_resampled)

    # from RandomizedSearchCV I obtained the best parameters to be:
    # max_depth = 102, n_estimators = 40
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

    print(f"Random Forest (RUS) train time = {t_end - t_start}")
    print(f"Random Forest (RUS) prediction time = {p_end - p_start}")
    print(f"random forest (RUS) accuracy = {accuracy_score_}")
    print(f"random forest (RUS) f1_score = {f1_score_}\n")


# RandomForestClassifier with RandomOverSampler (best model out of 3)
run_random_forest_ros()

# RandomForestClassifier with RandomUnderSampler
run_random_forest_rus()

# RandomForestClassifier with SMOTE
run_random_forest_smote()
