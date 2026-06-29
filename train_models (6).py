"""
train_models.py  (advanced ML version)
----------------------------------------
Reads the railway transactions CSV, builds:
  1. A daily aggregate table used to train FORECASTING models for revenue & ticket
     volume, using XGBoost with cyclical date encoding, lag/rolling features, and
     time-series cross-validated hyperparameter search.
  2. A per-transaction feature table used to train a RISK model (XGBoost classifier)
     predicting On Time / Delayed / Cancelled, with stratified CV hyperparameter
     search and class-imbalance handling.

Outputs (saved to ./artifacts/):
  - daily_agg.csv             daily aggregated history (for dashboard charts)
  - full_data.csv             full cleaned transaction-level data
  - forecast_revenue.joblib   XGBoost regressor: features -> daily revenue
  - forecast_volume.joblib    XGBoost regressor: features -> daily ticket count
  - risk_model.joblib         XGBoost classifier: journey features -> status
  - encoders.joblib           label encoders / metadata / model metrics
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
)
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "railway.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date of Journey"] = pd.to_datetime(df["Date of Journey"])
    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])
    df["Railcard"] = df["Railcard"].fillna("None")
    df["Departure Hour"] = pd.to_datetime(df["Departure Time"], format="%H:%M:%S").dt.hour
    return df


# ---------------------------------------------------------------------------
# 1. DAILY AGGREGATION + ADVANCED FORECASTING MODELS
# ---------------------------------------------------------------------------
def build_daily_agg(df):
    daily = df.groupby("Date of Journey").agg(
        revenue=("Price", "sum"),
        tickets=("Transaction ID", "count"),
        delayed=("Journey Status", lambda s: (s == "Delayed").sum()),
        cancelled=("Journey Status", lambda s: (s == "Cancelled").sum()),
    ).reset_index()
    daily = daily.sort_values("Date of Journey").reset_index(drop=True)
    daily["dow"] = daily["Date of Journey"].dt.dayofweek
    daily["day_index"] = (daily["Date of Journey"] - daily["Date of Journey"].min()).dt.days
    daily["month"] = daily["Date of Journey"].dt.month
    daily["delay_rate"] = daily["delayed"] / daily["tickets"]
    daily["cancel_rate"] = daily["cancelled"] / daily["tickets"]
    daily.to_csv(os.path.join(OUT_DIR, "daily_agg.csv"), index=False)
    return daily


def build_forecast_features(daily, target_col, min_date, for_training=True):
    """
    Builds a richer feature set for forecasting:
      - cyclical day-of-week and month encoding (sin/cos)
      - linear trend (day_index)
      - lag features (previous 1/7 days) and 7-day rolling mean
    When for_training=False, lag/rolling features use the *last known* values
    (carried forward), since future actuals don't exist yet.
    """
    d = daily.copy()
    d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    if for_training:
        d["lag_1"] = d[target_col].shift(1)
        d["lag_7"] = d[target_col].shift(7)
        d["roll_mean_7"] = d[target_col].shift(1).rolling(7).mean()
        d = d.dropna().reset_index(drop=True)

    feature_cols = ["day_index", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    if for_training:
        feature_cols += ["lag_1", "lag_7", "roll_mean_7"]
    return d, feature_cols


def train_forecast_models(daily):
    min_date = daily["Date of Journey"].min()
    models = {}
    metrics = {}
    feature_meta = {}

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2],
    }

    for target, fname in [("revenue", "forecast_revenue.joblib"),
                           ("tickets", "forecast_volume.joblib")]:
        d, feature_cols = build_forecast_features(daily, target, min_date, for_training=True)
        X, y = d[feature_cols], d[target]

        # time-aware split: last 20% chronologically held out
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        tscv = TimeSeriesSplit(n_splits=4)
        base = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=20, cv=tscv,
            scoring="neg_mean_absolute_error", random_state=RANDOM_STATE, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        preds = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        resid_std = float(np.std(y_test.values - preds))
        print(f"[forecast:{target}] best params = {search.best_params_}")
        print(f"[forecast:{target}] MAE on holdout = {mae:.2f}  (mean={y_test.mean():.1f})")
        metrics[target] = {
            "mae": float(mae), "mean": float(y_test.mean()), "resid_std": resid_std,
            "best_params": search.best_params_,
        }

        # refit best params on ALL data (with lag features) for production model
        final_model = XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE,
                                    n_jobs=-1, **search.best_params_)
        final_model.fit(X, y)
        joblib.dump(final_model, os.path.join(OUT_DIR, fname))
        models[target] = final_model
        feature_meta[target] = feature_cols

    # store recent history needed to build lag features for future predictions
    recent_history = daily[["Date of Journey", "revenue", "tickets", "dow", "month", "day_index"]].tail(14).copy()
    return min_date, metrics, feature_meta, recent_history


# ---------------------------------------------------------------------------
# 2. PER-JOURNEY RISK MODEL (XGBoost, tuned)
# ---------------------------------------------------------------------------
RISK_CATEGORICAL = [
    "Purchase Type", "Payment Method", "Railcard", "Ticket Class",
    "Ticket Type", "Departure Station", "Arrival Destination",
]
RISK_NUMERIC = ["Price", "Departure Hour", "dow"]


def train_risk_model(df):
    df = df.copy()
    df["dow"] = df["Date of Journey"].dt.dayofweek

    encoders = {}
    X_parts = []
    for col in RISK_CATEGORICAL:
        le = LabelEncoder()
        X_parts.append(pd.Series(le.fit_transform(df[col].astype(str)), name=col))
        encoders[col] = le

    X = pd.concat(X_parts, axis=1)
    for col in RISK_NUMERIC:
        X[col] = df[col].values

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["Journey Status"])
    class_names = list(label_enc.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # sample weights to counter class imbalance (XGBoost has no class_weight param)
    class_counts = np.bincount(y_train)
    weight_map = {i: len(y_train) / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    sample_weight = np.array([weight_map[label] for label in y_train])

    param_dist = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2],
    }

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    base = XGBClassifier(
        objective="multi:softprob", num_class=len(class_names),
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=25, cv=skf,
        scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1
    )
    search.fit(X_train, y_train, sample_weight=sample_weight)
    clf = search.best_estimator_

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    print(f"[risk model] best params = {search.best_params_}")
    print(f"[risk model] holdout accuracy = {acc:.3f}  macro-F1 = {f1m:.3f}")
    print("Feature importances:")
    for col, imp in sorted(zip(X.columns, clf.feature_importances_), key=lambda t: -t[1]):
        print(f"   {col:20s} {imp:.3f}")

    # refit best params on ALL data for production
    full_counts = np.bincount(y)
    full_weight_map = {i: len(y) / (len(full_counts) * c) for i, c in enumerate(full_counts)}
    full_sample_weight = np.array([full_weight_map[label] for label in y])
    final_clf = XGBClassifier(
        objective="multi:softprob", num_class=len(class_names),
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1, **search.best_params_
    )
    final_clf.fit(X, y, sample_weight=full_sample_weight)
    joblib.dump(final_clf, os.path.join(OUT_DIR, "risk_model.joblib"))

    feature_importance = {col: float(imp) for col, imp in zip(X.columns, final_clf.feature_importances_)}

    meta = {
        "encoders": encoders,
        "label_encoder": label_enc,
        "categorical_cols": RISK_CATEGORICAL,
        "numeric_cols": RISK_NUMERIC,
        "classes": class_names,
        "risk_accuracy": float(acc),
        "risk_f1_macro": float(f1m),
        "risk_best_params": search.best_params_,
        "feature_importance": feature_importance,
        "stations": sorted(set(df["Departure Station"]) | set(df["Arrival Destination"])),
        "ticket_classes": sorted(df["Ticket Class"].unique().tolist()),
        "ticket_types": sorted(df["Ticket Type"].unique().tolist()),
        "payment_methods": sorted(df["Payment Method"].unique().tolist()),
        "railcards": sorted(df["Railcard"].unique().tolist()),
        "purchase_types": sorted(df["Purchase Type"].unique().tolist()),
        "reason_counts": df["Reason for Delay"].value_counts().to_dict(),
        "route_stats": df.groupby(["Departure Station", "Arrival Destination"]).agg(
            avg_price=("Price", "mean"),
            trips=("Transaction ID", "count"),
            delay_rate=("Journey Status", lambda s: (s == "Delayed").mean()),
            cancel_rate=("Journey Status", lambda s: (s == "Cancelled").mean()),
        ).reset_index().to_dict("records"),
    }
    joblib.dump(meta, os.path.join(OUT_DIR, "encoders.joblib"))
    return meta


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['Date of Journey'].nunique()} unique journey days")

    df.to_csv(os.path.join(OUT_DIR, "full_data.csv"), index=False)

    daily = build_daily_agg(df)
    min_date, forecast_metrics, feature_meta, recent_history = train_forecast_models(daily)

    meta = train_risk_model(df)
    meta["min_date"] = min_date
    meta["forecast_metrics"] = forecast_metrics
    meta["forecast_feature_cols"] = feature_meta
    meta["recent_history"] = recent_history.to_dict("records")
    joblib.dump(meta, os.path.join(OUT_DIR, "encoders.joblib"))

    print("\nAll artifacts saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
