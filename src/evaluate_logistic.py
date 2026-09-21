from sklearn.metrics import accuracy_score, classification_report

from preprocessing import X_test_prepared, y_test
from train_logistic import model


# Predictions
y_pred = model.predict(X_test_prepared)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===== LOGISTIC REGRESSION EVALUATION =====")
print("Accuracy:", accuracy)


# Classification report
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))