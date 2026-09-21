import pandas as pd


file_path = "data/online_shoppers_intention.csv"
df = pd.read_csv(file_path)

print("===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TARGET DISTRIBUTION =====")
print(df["Revenue"].value_counts())

print("\n===== TARGET PERCENTAGES =====")
print(df["Revenue"].value_counts(normalize=True) * 100)