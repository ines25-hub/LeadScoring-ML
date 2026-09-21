from sklearn.metrics import accuracy_score, classification_report

from preprocessing import X_test_prepared, y_test
from train_random_forest import model


# Predictions
y_pred = model.predict(X_test_prepared)


# Probabilities
y_probability = model.predict_proba(X_test_prepared)[:, 1]


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===== RANDOM FOREST EVALUATION =====")
print("Accuracy:", accuracy)


# Classification report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))


# Example probabilities
print("\n===== EXAMPLE CONVERSION PROBABILITIES =====")
print(y_probability[:10])