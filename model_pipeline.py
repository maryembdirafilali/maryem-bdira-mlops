import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_data(data_path="Churn_Modelling.csv"):
    """Load the CSV, preprocess it and return scaled train/test sets."""
    encoder = LabelEncoder()
    data = pd.read_csv(data_path)
    data = data.drop(["Surname", "Geography"], axis=1)
    data["Gender"] = encoder.fit_transform(data["Gender"])
    data = data.dropna()
    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")
    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train):
    """Train a RandomForest classifier and return it."""
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Compute and print the accuracy of the model on the test set."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return accuracy


def predict(features, model_path="model.joblib"):
    """Load the saved model and predict on already scaled features."""
    model = joblib.load(model_path)
    return model.predict(features)


def save_model(model, model_path="model.joblib"):
    """Save the trained model to disk with joblib."""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path="model.joblib"):
    """Load a saved model from disk and return it."""
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
