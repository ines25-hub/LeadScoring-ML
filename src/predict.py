import joblib
import pandas as pd

# Load the saved model and preprocessor
model = joblib.load("models/random_forest_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


def predict_conversion(visitor_data):

    # Convert visitor data into a DataFrame
    data = pd.DataFrame([visitor_data])

    # Prepare the data
    data_prepared = preprocessor.transform(data)

    # Get conversion probability
    probability = model.predict_proba(data_prepared)[0][1]

    # Convert probability to score out of 100
    score = round(probability * 100)

    # Classification
    if score < 40:
        classification = "Low"
    elif score < 70:
        classification = "Medium"
    else:
        classification = "High"

    # Get feature importance
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    top_features = importance_df.sort_values(
        by="importance",
        ascending=False
    ).head(3)

    main_factors = [
        feature.replace("remainder__", "")
        for feature in top_features["feature"]
    ]

    # Recommendation
    if classification == "High":
        recommendation = "Prioritize this visitor for commercial follow-up."
    elif classification == "Medium":
        recommendation = "Continue monitoring this visitor."
    else:
        recommendation = "Low priority for commercial follow-up."

    return {
        "probability": round(float(probability) * 100, 2),
        "score": score,
        "classification": classification,
        "main_factors": main_factors,
        "recommendation": recommendation
    }