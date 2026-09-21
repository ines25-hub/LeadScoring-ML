import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocessing import (
    X_train_prepared,
    y_train,
    preprocessor
)


# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train model
model.fit(X_train_prepared, y_train)


# Save model
joblib.dump(model, "models/random_forest_model.pkl")

# Save preprocessor
joblib.dump(preprocessor, "models/preprocessor.pkl")


print("Random Forest model trained successfully!")
print("Model saved successfully!")
print("Preprocessor saved successfully!")