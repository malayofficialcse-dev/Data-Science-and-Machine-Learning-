import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


Data_path = "data/house_prices.csv"

df = pd.read_csv(Data_path)

print("\nDataset : ")
print(df.head())

print("\nShape : ")
print(df.shape)

X = df[
    [
        "area",
        "bedrooms",
        "bathrooms",
        "age"
    ]
]

y = df["price"]


X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train
)

y_pred = model.predict(
    X_test_scaled
)

#Evaluation
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\nModel Performance")
print("----------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)

# save the model

os.makedirs(
    "model",
    exist_ok=True
)

joblib.dump(
    model,
    "model/linear_regression.pkl"
)

joblib.dump(scaler,"model/scaler.pkl")

print("\nModel saved successfully.")