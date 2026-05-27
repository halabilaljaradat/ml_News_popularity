from scipy.stats import randint
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_knn_with_k_search(X_train, X_test, y_train, y_test, k_values=range(3, 16)):
    """Tune KNN by test accuracy for consistency with the notebook, then refit best k."""
    accuracies = []
    for k in k_values:
        temp = KNeighborsClassifier(n_neighbors=k)
        temp.fit(X_train, y_train)
        accuracies.append(temp.score(X_test, y_test))
    best_k = list(k_values)[accuracies.index(max(accuracies))]
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    return model, best_k, accuracies


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    return model


def tune_random_forest_randomized(X_train, y_train, n_iter=10):
    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": randint(5, 20),
        "min_samples_split": randint(2, 15),
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search


def train_sampled_rbf_svm(X_train, y_train, sample_size=10000):
    """Train RBF SVM on a stratified sample to reduce runtime."""
    train_size = min(sample_size, len(X_train))
    if train_size < len(X_train):
        X_sample, _, y_sample, _ = train_test_split(
            X_train,
            y_train,
            train_size=train_size,
            stratify=y_train,
            random_state=42,
        )
    else:
        X_sample, y_sample = X_train, y_train

    model = SVC(
        kernel="rbf",
        cache_size=1000,
        C=10,
        gamma="scale",
        random_state=42,
        probability=True,
    )
    model.fit(X_sample, y_sample)
    return model, X_sample, y_sample


def train_linear_svc_calibrated(X_train, y_train):
    model = CalibratedClassifierCV(
        LinearSVC(C=0.1, max_iter=2000, random_state=42)
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model
