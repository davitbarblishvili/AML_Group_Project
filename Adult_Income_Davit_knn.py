import time
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


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


def run_knn(neighbors, distance):

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

    print(f"KNN[k = {neighbors}, distance = {'Manhattan' if distance == 1 else 'Euclidean'}] train time = {t_end - t_start}")
    print(f"KNN[k = {neighbors}, distance = {'Manhattan' if distance == 1 else 'Euclidean'}] prediction time = {p_end - p_start}")
    print(
        f"KNN[k = {neighbors}, distance = {'Manhattan' if distance == 1 else 'Euclidean'}] accuracy = {accuracy_}")
    print(
        f"KNN[k = {neighbors}, distance = {'Manhattan' if distance == 1 else 'Euclidean'}] f1_score = {f1_score}\n")


# k = 1, distance = Manhattan (1)
run_knn(neighbors=1, distance=1)

# k = 5, distance = Manhattan (1)
run_knn(neighbors=5, distance=1)

# k = 10, distance = Manhattan (1)
run_knn(neighbors=10, distance=1)

# k = 20, distance = Manhattan (1)
run_knn(neighbors=20, distance=1)

# k = 1, distance = Euclidean (2)
run_knn(neighbors=1, distance=2)

# k = 5, distance = Euclidean (2)
run_knn(neighbors=5, distance=2)

# k = 10, distance = Euclidean (2)
run_knn(neighbors=10, distance=2)

# k = 20, distance = Euclidean (2) (best model out of all of them)
run_knn(neighbors=20, distance=2)
