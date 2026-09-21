from sklearn.linear_model import LogisticRegression

from preprocessing import (
    X_train_prepared,
    y_train
)


# Create model
model = LogisticRegression(max_iter=1000)


# Train model
model.fit(X_train_prepared, y_train)


print("Logistic Regression model trained successfully!")