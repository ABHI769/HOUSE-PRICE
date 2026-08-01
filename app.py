import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from currency import convert_from_inr, format_price, get_supported_currencies

app = Flask(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "house_price_model.joblib"
METADATA_PATH = Path(__file__).parent / "models" / "model_metadata.json"

model = None
metadata = {}


def load_model():
    global model, metadata
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python train_model.py' first."
        )
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)


@app.route("/")
def index():
    if model is None or not metadata:
        load_model()

    return render_template(
        "index.html",
        city_tiers=metadata.get("city_tiers", []),
        property_types=metadata.get("property_types", []),
        furnishing_options=metadata.get("furnishing_options", []),
        currencies=get_supported_currencies(),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        currency = data.get("currency", "INR").upper()

        features = {
            "area_sqft": float(data["area_sqft"]),
            "bedrooms": int(data["bedrooms"]),
            "bathrooms": int(data["bathrooms"]),
            "age_years": int(data["age_years"]),
            "floor": int(data["floor"]),
            "total_floors": int(data["total_floors"]),
            "parking": int(data["parking"]),
            "city_tier": data["city_tier"],
            "property_type": data["property_type"],
            "furnishing": data["furnishing"],
        }

        if features["floor"] > features["total_floors"]:
            return jsonify({"error": "Floor cannot exceed total floors"}), 400
        if features["area_sqft"] <= 0:
            return jsonify({"error": "Area must be greater than 0"}), 400

        input_df = pd.DataFrame([features])
        price_inr = float(model.predict(input_df)[0])
        price_inr = max(price_inr, 0)

        converted = convert_from_inr(price_inr, currency)

        return jsonify(
            {
                "price_inr": round(price_inr, 2),
                "price": round(converted, 2),
                "currency": currency,
                "formatted_inr": format_price(price_inr, "INR"),
                "formatted": format_price(converted, currency),
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/currencies")
def currencies():
    return jsonify(get_supported_currencies())


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
