import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Load dataset
df = pd.read_csv("data/online_shoppers_intention.csv")


# Separate features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Categorical columns
categorical_columns = [
    "Month",
    "VisitorType"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder=StandardScaler()
)


# Prepare training data
X_train_prepared = preprocessor.fit_transform(X_train)


# Prepare test data
X_test_prepared = preprocessor.transform(X_test)


print("Training data:", X_train_prepared.shape)
print("Test data:", X_test_prepared.shape)