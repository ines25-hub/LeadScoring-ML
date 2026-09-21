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

    # Recommendation
    if classification == "High":
        recommendation = "Prioritize this visitor for commercial follow-up."
    elif classification == "Medium":
        recommendation = "Continue monitoring this visitor."
    else:
        recommendation = "Low priority for commercial follow-up."

    return {
        "probability": round(probability * 100, 2),
        "score": score,
        "classification": classification,
        "recommendation": recommendation
    }

if __name__ == "__main__":

    test_visitor = {
        "Administrative": 3,
        "Administrative_Duration": 100,
        "Informational": 1,
        "Informational_Duration": 30,
        "ProductRelated": 20,
        "ProductRelated_Duration": 500,
        "BounceRates": 0.01,
        "ExitRates": 0.02,
        "PageValues": 20,
        "SpecialDay": 0,
        "Month": "Nov",
        "OperatingSystems": 2,
        "Browser": 2,
        "Region": 1,
        "TrafficType": 2,
        "VisitorType": "Returning_Visitor",
        "Weekend": False
    }

    result = predict_conversion(test_visitor)

    print("\n===== PREDICTION =====")
    print(result)