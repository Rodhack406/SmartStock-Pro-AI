import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL connection string (Supabase)
DB_URL = "postgresql://postgres.sknimloxbgrojobvygov:GoretteMulundano@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

engine = create_engine(DB_URL)


def load_sales_data():
    query = """
    SELECT
        product_id,
        quantity,
        price,
        total,
        created_at
    FROM sales;
    """

    df = pd.read_sql(query, engine)

    # convert datetime safely
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # remove broken rows (important for ML stability)
    df = df.dropna(subset=["created_at"])

    return df


def load_products():
    query = """
    SELECT
        id,
        name,
        category,
        price,
        stock
    FROM products;
    """

    df = pd.read_sql(query, engine)

    return df


# TEST RUN (remove later in production pipeline)
if __name__ == "__main__":
    df_sales = load_sales_data()
    df_products = load_products()

    print("SALES DATA:")
    print(df_sales.head())

    print("\nPRODUCTS DATA:")
    print(df_products.head())