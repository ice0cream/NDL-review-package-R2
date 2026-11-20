import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV


class PLSClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = None
        self.lb = None
        self.classes_ = None

    def fit(self, X, y):
        self.lb = LabelBinarizer()
        y_encoded = self.lb.fit_transform(y)
        self.classes_ = self.lb.classes_
        current_n = min(self.n_components, X.shape[1], X.shape[0] - 1)
        if current_n < 1:
            current_n = 1
        self.pls = PLSRegression(n_components=current_n, max_iter=10000, tol=1e-6)
        self.pls.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_scores = self.pls.predict(X)
        if y_scores.shape[1] == 1:
            return self.lb.inverse_transform(y_scores)
        else:
            return self.lb.classes_[np.argmax(y_scores, axis=1)]


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        self.classes_ = np.unique(y)
        try:
            self.encoder_ = OneHotEncoder(sparse_output=False)
        except TypeError:
            self.encoder_ = OneHotEncoder(sparse=False)
        Xs = self.scaler_.fit_transform(X)
        Y = self.encoder_.fit_transform(np.array(y).reshape(-1, 1))
        n_comp = min(self.n_components, Xs.shape[1], Xs.shape[0] - 1)
        self.pls_ = PLSRegression(n_components=n_comp)
        self.pls_.fit(Xs, Y)
        return self

    def predict(self, X):
        Xs = self.scaler_.transform(X)
        Y_pred = self.pls_.predict(Xs)
        pred_idx = np.argmax(Y_pred, axis=1)
        return self.classes_[pred_idx]


def make_plsda(n_components):
    return PLSDAClassifier(n_components=n_components)


def make_lda():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ])


def make_logistic(random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state))
    ])


def make_svm_linear(random_state):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=False, random_state=random_state))
    ])


def make_svm_rbf(C=1, gamma="scale", random_state=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, probability=False, random_state=random_state))
    ])


def make_rf_classifier(n_estimators, random_state, class_weight="balanced", n_jobs=-1):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs
    )


def make_knn(n_neighbors):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=n_neighbors))
    ])


def make_svr(C=30, gamma="scale", epsilon=1.0):
    return SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)


def make_rf_regressor(n_estimators, random_state, min_samples_leaf=2, n_jobs=-1):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )


def make_plsr(n_components):
    return PLSRegression(n_components=n_components, max_iter=5000, tol=1e-7)


def make_ridge_cv():
    return RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
