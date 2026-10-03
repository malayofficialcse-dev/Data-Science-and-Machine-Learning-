from fastapi import FastAPI

from app.schemas.prediction import (
    HousePredictionRequest
)

from app.services.prediction_service import (
    predict_price
)

app = FastAPI(
    title="House Price Prediction API",
    description="Linear Regression ML API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "status":"healthy"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def prediction(request:HousePredictionRequest) :
    price = predict_price(
        area=request.area,
        bedrooms=request.bedrooms,
        bathrooms=request.bathrooms,
        age=request.age
    )

    return {
        "predicted_price":round(
            price,
            2
        )
    }
