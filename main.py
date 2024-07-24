import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model, print_evaluation

# Zip file paths
file_paths = [
    r'C:\Users\DChai\Downloads\creditcard_2023.csv.zip',
    r'C:\Users\DChai\Downloads\application_data.csv.zip',
    r'C:\Users\DChai\Downloads\creditcard.csv.zip',
]

# Load and preprocess data
data = load_data(file_paths)
X_scaled, y = preprocess_data(data)

# Split data: training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
log_reg_model = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate models
log_reg_accuracy, log_reg_report, log_reg_matrix = evaluate_model(log_reg_model, X_test, y_test)
rf_accuracy, rf_report, rf_matrix = evaluate_model(rf_model, X_test, y_test)

# Print results
print_evaluation("Logistic Regression", log_reg_accuracy, log_reg_report, log_reg_matrix)
print_evaluation("Random Forest", rf_accuracy, rf_report, rf_matrix)
