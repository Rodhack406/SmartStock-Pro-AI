import pandas as pd
from data_loader import load_sales_data


def build_features():
    df = load_sales_data()

    # ----------------------------
    # PRODUCT ENCODING
    # ----------------------------
    product_map = {
        pid: idx for idx, pid in enumerate(sorted(df["product_id"].unique()))
    }
    df["product_index"] = df["product_id"].map(product_map)

    # ensure datetime
    df["created_at"] = pd.to_datetime(df["created_at"])

    # ----------------------------
    # DAILY AGGREGATION FIRST (IMPORTANT FIX)
    # ----------------------------
    df["date"] = df["created_at"].dt.date

    daily = df.groupby(
        ["product_index", "date"],
        as_index=False
    ).agg({
        "quantity": "sum",
        "total": "sum",
        "price": "mean"
    })

    daily.rename(columns={
        "quantity": "quantity_sold",
        "total": "revenue",
        "price": "avg_price"
    }, inplace=True)

    # rebuild datetime context AFTER grouping
    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day

    daily = daily.sort_values(["product_index", "date"])

    # ----------------------------
    # LAGS
    # ----------------------------
    daily["lag_1"] = daily.groupby("product_index")["quantity_sold"].shift(1)
    daily["lag_3"] = daily.groupby("product_index")["quantity_sold"].shift(3)
    daily["lag_7"] = daily.groupby("product_index")["quantity_sold"].shift(7)

    # ----------------------------
    # ROLLING
    # ----------------------------
    daily["rolling_mean_3"] = daily.groupby("product_index")["quantity_sold"]\
        .transform(lambda x: x.rolling(3).mean())

    daily["rolling_mean_7"] = daily.groupby("product_index")["quantity_sold"]\
        .transform(lambda x: x.rolling(7).mean())

    daily["rolling_std_7"] = daily.groupby("product_index")["quantity_sold"]\
        .transform(lambda x: x.rolling(7).std())

    # ----------------------------
    # MOMENTUM
    # ----------------------------
    daily["sales_change"] = daily["quantity_sold"] - daily["lag_1"]
    daily["sales_growth_rate"] = daily["sales_change"] / (daily["lag_1"] + 1)

    # ----------------------------
    # CLEAN
    # ----------------------------
    daily = daily.dropna().reset_index(drop=True)

    return daily