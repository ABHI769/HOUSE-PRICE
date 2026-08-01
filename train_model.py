"""Train a house price prediction model on synthetic Indian housing data."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"

CITY_TIERS = ["metro", "tier1", "tier2", "tier3"]
PROPERTY_TYPES = ["apartment", "villa", "independent_house", "studio"]


def generate_housing_data(n_samples: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    city_tier = rng.choice(CITY_TIERS, n_samples, p=[0.25, 0.30, 0.30, 0.15])
    property_type = rng.choice(PROPERTY_TYPES, n_samples, p=[0.45, 0.20, 0.25, 0.10])

    area_sqft = rng.integers(400, 4500, n_samples).astype(float)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n_samples), 1, 5)
    age_years = rng.integers(0, 41, n_samples)
    floor = rng.integers(0, 21, n_samples)
    total_floors = np.clip(floor + rng.integers(1, 15, n_samples), 1, 30)
    parking = rng.integers(0, 4, n_samples)
    furnishing = rng.choice(["unfurnished", "semi", "fully"], n_samples, p=[0.4, 0.35, 0.25])

    tier_multiplier = {"metro": 1.8, "tier1": 1.3, "tier2": 0.9, "tier3": 0.6}
    type_multiplier = {
        "apartment": 1.0,
        "villa": 1.6,
        "independent_house": 1.35,
        "studio": 0.75,
    }
    furnish_multiplier = {"unfurnished": 1.0, "semi": 1.08, "fully": 1.15}

    base_price_per_sqft = 4200
    price = (
        area_sqft
        * base_price_per_sqft
        * np.vectorize(tier_multiplier.get)(city_tier)
        * np.vectorize(type_multiplier.get)(property_type)
        * np.vectorize(furnish_multiplier.get)(furnishing)
        * (1 + bedrooms * 0.04)
        * (1 + parking * 0.03)
        * np.maximum(0.55, 1 - age_years * 0.008)
        * (1 + np.where(floor > 0, 0.02, 0))
    )

    noise = rng.normal(1.0, 0.08, n_samples)
    price_inr = np.clip(price * noise, 800_000, 25_000_000)

    return pd.DataFrame(
        {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "floor": floor,
            "total_floors": total_floors,
            "parking": parking,
            "city_tier": city_tier,
            "property_type": property_type,
            "furnishing": furnishing,
            "price_inr": price_inr.round(0),
        }
    )


def train_and_save():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    df = generate_housing_data()
    df.to_csv(DATA_DIR / "housing_data.csv", index=False)

    feature_cols = [
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "age_years",
        "floor",
        "total_floors",
        "parking",
        "city_tier",
        "property_type",
        "furnishing",
    ]
    categorical = ["city_tier", "property_type", "furnishing"]
    numeric = [c for c in feature_cols if c not in categorical]

    X = df[feature_cols]
    y = df["price_inr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.08,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    model_path = MODEL_DIR / "house_price_model.joblib"
    joblib.dump(pipeline, model_path)

    metadata = {
        "feature_cols": feature_cols,
        "categorical_cols": categorical,
        "numeric_cols": numeric,
        "city_tiers": CITY_TIERS,
        "property_types": PROPERTY_TYPES,
        "furnishing_options": ["unfurnished", "semi", "fully"],
        "base_currency": "INR",
        "r2_score": round(score, 4),
        "sample_count": len(df),
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"R² score on test set: {score:.4f}")
    return pipeline, metadata


if __name__ == "__main__":
    train_and_save()
