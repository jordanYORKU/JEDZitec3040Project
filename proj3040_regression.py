"""
Logistic Regression Method for ITEC 3040 Final Project
------------------------------------------------------
This module implements a logistic regression model to predict
event attendance (Attended = 0/1) using the shared preprocessing
pipeline defined in `preprocess_party.py`.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import cleaned data and preprocessing pipeline
from preprocess_party import X_train, X_test, y_train, y_test, preprocessor


def run_regression_model():
    """
    Train and evaluate a Logistic Regression model on the event dataset.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline (preprocessor + logistic regression).
    y_pred : ndarray
        Predicted labels for the test set.
    acc : float
        Accuracy score on the test set.
    cm : ndarray of shape (2, 2)
        Confusion matrix.
    report : str
        Text classification report (precision, recall, F1-score).
    """
    # Build pipeline
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print results
    print("----- Logistic Regression Results (Attended) -----")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    return model, y_pred, acc, cm, report


if __name__ == "__main__":
    run_regression_model()
