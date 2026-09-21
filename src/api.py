from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_conversion


app = FastAPI(
    title="Lead Scoring ML API",
    version="1.0.0"
)


class VisitorData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


@app.get("/")
def home():
    return {
        "message": "Lead Scoring ML API is running"
    }


@app.post("/predict")
def predict(visitor_data: VisitorData):
    result = predict_conversion(visitor_data.model_dump())

    return result