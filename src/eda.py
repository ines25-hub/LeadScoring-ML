import pandas as pd


# Load dataset
file_path = "data/online_shoppers_intention.csv"
df = pd.read_csv(file_path)


# Basic statistics
print("===== BASIC STATISTICS =====")
print(df.describe())


# Numerical features: converted vs not converted
print("\n===== CONVERTED VS NOT CONVERTED =====")

comparison = df.groupby("Revenue")[
    [
        "Administrative",
        "Administrative_Duration",
        "Informational",
        "Informational_Duration",
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
    ]
].mean()

print(comparison)


# Visitor type
print("\n===== CONVERSION RATE BY VISITOR TYPE =====")
print(df.groupby("VisitorType")["Revenue"].mean() * 100)


# Weekend
print("\n===== CONVERSION RATE BY WEEKEND =====")
print(df.groupby("Weekend")["Revenue"].mean() * 100)


# Month
print("\n===== CONVERSION RATE BY MONTH =====")
print(df.groupby("Month")["Revenue"].mean() * 100)