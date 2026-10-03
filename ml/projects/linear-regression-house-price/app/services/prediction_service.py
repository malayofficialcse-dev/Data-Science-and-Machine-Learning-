import joblib
import pandas as pd

Model_path = "model/linear_regression.pkl"
Scaler_path = "model/scaler.pkl"

model = joblib.load(
    Model_path
)

scaler = joblib.load(
    Scaler_path
)

def predict_price (
    area :float,
    bedrooms:int,
    bathrooms:int,
    age:float
):
    data = pd.DataFrame(
        [[area, bedrooms, bathrooms, age]],
        columns=["area", "bedrooms", "bathrooms", "age"],
    )

    data_scaled = scaler.transform(
        data
    )

    prediction = model.predict(
        data_scaled
    )

    return float(prediction[0])