from sklearn.ensemble import RandomForestClassifier
import joblib
def train_random_forest(X_train, y_train, model_path="models/migraine_rf_model.pkl"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model