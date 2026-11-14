
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model configuration
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# MLflow experiment
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted"
    )

    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model", "RandomForest")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)

    # Log model
    mlflow.sklearn.log_model(
        model,
        name="iris_model",
        skops_trusted_types=[
              "sklearn.tree._tree.Tree"
    ]
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Model training completed!")
