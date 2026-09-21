# Lead Scoring ML

Machine Learning component for predicting the probability that a website visitor will convert into a client.

## Project Objective

The system analyzes visitor behavior and produces:

- Conversion probability
- Lead Score out of 100
- Classification: Low, Medium, or High
- Main factors influencing the prediction
- Recommendation for follow-up

These outputs correspond to the Lead Scoring MVP specification.

## Dataset

Dataset used:

`Online Shoppers Purchasing Intention Dataset`

The dataset contains 12,330 visitor sessions and 18 columns.

The target variable is:

`Revenue`

- False: visitor did not convert
- True: visitor converted

## Machine Learning

The project uses a Random Forest Classifier.

The data preprocessing includes:

- Train/test split
- Standardization of numerical features
- One-hot encoding of categorical features

The trained model is saved in:

`models/random_forest_model.pkl`

The preprocessing object is saved in:

`models/preprocessor.pkl`

## Model Evaluation

Random Forest results on the test set:

- Accuracy: 89.94%
- True-class precision: 74%
- True-class recall: 54%
- True-class F1-score: 63%

## Prediction

The prediction system returns:

```json
{
  "probability": 54.0,
  "score": 54,
  "classification": "Medium",
  "main_factors": [
    "PageValues",
    "ProductRelated_Duration",
    "ExitRates"
  ],
  "recommendation": "Continue monitoring this visitor."
}