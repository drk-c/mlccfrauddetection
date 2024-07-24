from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    return log_reg

def train_random_forest(X_train, y_train):
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    return rf_clf