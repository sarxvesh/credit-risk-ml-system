from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    return rf