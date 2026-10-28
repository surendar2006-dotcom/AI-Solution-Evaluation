"""Train and evaluate simple predictive models on the included synthetic data."""
from pathlib import Path
import time
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "historical_data.csv"
FEATURES = ["month_index", "marketing_spend", "website_visits", "average_price"]
TARGET = "sales"


def evaluate(name, model, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"\n{name}")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  Fit + prediction time: {elapsed:.4f} seconds")
    return {"model": name, "MAE": mae, "RMSE": rmse, "seconds": elapsed}


def main():
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.sort_values("month_index").dropna(subset=FEATURES + [TARGET])
    if len(df) < 10:
        raise ValueError("At least 10 valid rows are required for the demo.")

    # Chronological split: earliest 80% for training, latest 20% for testing.
    split_at = int(len(df) * 0.8)
    X_train = df.iloc[:split_at][FEATURES]
    y_train = df.iloc[:split_at][TARGET]
    X_test = df.iloc[split_at:][FEATURES]
    y_test = df.iloc[split_at:][TARGET]

    results = []
    results.append(evaluate(
        "Baseline (predict training mean)",
        DummyRegressor(strategy="mean"),
        X_train, y_train, X_test, y_test
    ))
    results.append(evaluate(
        "Linear Regression",
        make_pipeline(StandardScaler(), LinearRegression()),
        X_train, y_train, X_test, y_test
    ))
    results.append(evaluate(
        "Random Forest Regressor",
        RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2),
        X_train, y_train, X_test, y_test
    ))

    print("\nComparison (lower MAE/RMSE is better):")
    print(pd.DataFrame(results).round(3).to_string(index=False))
    print("\nReminder: these are results on synthetic demonstration data only.")


if __name__ == "__main__":
    main()
