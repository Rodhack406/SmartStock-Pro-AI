import pandas as pd
import joblib
from data_loader import load_sales_data
from features import FEATURES

model = joblib.load("xgb_model.pkl")


def generate_predictions():
    df = load_sales_data()

    df["product_index"] = df["product_id"].astype("category").cat.codes
    df["created_at"] = pd.to_datetime(df["created_at"])

    df["date"] = df["created_at"].dt.date

    daily = df.groupby(["product_index", "date"], as_index=False).agg({
        "quantity": "sum",
        "price": "mean"
    })

    daily.rename(columns={
        "quantity": "quantity_sold",
        "price": "avg_price"
    }, inplace=True)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day

    daily = daily.sort_values(["product_index", "date"])

    daily["lag_1"] = daily.groupby("product_index")["quantity_sold"].shift(1)
    daily["lag_3"] = daily.groupby("product_index")["quantity_sold"].shift(3)
    daily["lag_7"] = daily.groupby("product_index")["quantity_sold"].shift(7)

    daily["rolling_mean_3"] = daily.groupby("product_index")["quantity_sold"]\
        .transform(lambda x: x.rolling(3).mean())

    daily["rolling_mean_7"] = daily.groupby("product_index")["quantity_sold"]\
        .transform(lambda x: x.rolling(7).mean())

    daily["rolling_std_7"] = daily.groupby("product_index")["quantity_sold"]\
        .transform(lambda x: x.rolling(7).std())

    daily["sales_change"] = daily["quantity_sold"] - daily["lag_1"]
    daily["sales_growth_rate"] = daily["sales_change"] / (daily["lag_1"] + 1)

    daily = daily.dropna()

    X = daily[FEATURES]

    daily["predicted_sales"] = model.predict(X)

    return daily