from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    X = data.drop(columns=['Class'])
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y