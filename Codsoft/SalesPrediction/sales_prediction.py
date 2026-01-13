import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("sales.csv")

print("Dataset Shape:", data.shape)
print("\nMissing Values:\n", data.isnull().sum())


data = data.dropna(subset=["sales"])

data["region"] = data["region"].fillna("Unknown")
data["product"] = data["product"].fillna("Unknown")

data["advertising_spend"] = data["advertising_spend"].fillna(
    data["advertising_spend"].median()
)
data["units_sold"] = data["units_sold"].fillna(
    data["units_sold"].median()
)


data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year

data = data.drop(columns=["date"])

X = data[[
    "region",
    "product",
    "advertising_spend",
    "units_sold",
    "month",
    "year"
]]

y = data["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_features = ["region", "product"]
numeric_features = [
    "advertising_spend",
    "units_sold",
    "month",
    "year"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 2))

new_data = pd.DataFrame({
    "region": ["South"],
    "product": ["Electronics"],
    "advertising_spend": [50000],
    "units_sold": [320],
    "month": [7],
    "year": [2026]
})

predicted_sales = pipeline.predict(new_data)

print("\nPredicted Sales:", round(predicted_sales[0], 2))
