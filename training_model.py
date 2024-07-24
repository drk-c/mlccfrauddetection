from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# logistic regression model
def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

# random forest model
def train_random_forest(X_train, y_train):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    return rfc

# other models include 
# KNN, decision tree, linear regression, gradient boosting, etc

# steps are essentially the same...