import pandas as pd

ADD_ON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

ENGINEERED_NUMERIC = ["service_count", "monthly_per_service", "charge_consistency"]
ENGINEERED_CATEGORICAL = []


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Number of add-on services subscribed — proxy for customer engagement
    df["service_count"] = (
        df[ADD_ON_COLS].apply(lambda col: (col == "Yes").astype(int)).sum(axis=1)
    )

    # Cost per service — high ratio signals poor value perception
    df["monthly_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    # Ratio of actual vs expected spend — detects billing changes over tenure
    expected_total = df["MonthlyCharges"] * df["tenure"]
    df["charge_consistency"] = (df["TotalCharges"] / expected_total.replace(0, 1)).clip(0, 2)

    return df
