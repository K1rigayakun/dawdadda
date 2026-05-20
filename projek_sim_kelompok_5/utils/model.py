from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = ["brand", "processor", "gpu", "os", "segment", "ram_gb", "storage_gb", "screen_size", "weight_kg"]
NUMERIC = ["ram_gb", "storage_gb", "screen_size", "weight_kg"]
CATEGORICAL = ["brand", "processor", "gpu", "os", "segment"]


@st.cache_resource(show_spinner=False)
def train_price_model(df: pd.DataFrame):
    data = df.dropna(subset=FEATURES + ["price"]).copy()
    x = data[FEATURES]
    y = data["price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = RandomForestRegressor(n_estimators=260, min_samples_leaf=3, random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    return pipeline, metrics, x_test, y_test, predictions


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    return (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(16)
    )


def format_idr(value: float) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")
