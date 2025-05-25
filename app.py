from flask import Flask, request, jsonify
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load model and scaler
try:
    with open("xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    logging.error(f"Failed to load scaler: {e}")
    raise

# Define expected features
feature_columns = [
    "shipping date (DateOrders)", "Order Status", "Delivery Status", "Late_delivery_risk",
    "Shipping Mode", "Order City", "Order State", "Order Region", "Order Country",
    "Latitude", "Longitude", "Customer Street", "Customer City", "Customer State",
    "Customer Country", "order date (DateOrders)", "Order Item Quantity", "Market"
]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        features = data["features"]

        # Check for missing features
        missing = [key for key in feature_columns if key not in features]
        if missing:
            return jsonify({"error": f"Missing required features: {missing}"}), 400

        # Ensure correct feature order
        try:
            values = [features[col] for col in feature_columns]
        except KeyError as ke:
            return jsonify({"error": f"Missing expected feature: {str(ke)}"}), 400

        df = pd.DataFrame([values], columns=feature_columns)

        # Scale and predict
        scaled_input = scaler.transform(df)
        prediction = model.predict(scaled_input)

        return jsonify({"prediction": prediction.tolist()})

    except ValueError as ve:
        logging.exception("Value error during prediction")
        return jsonify({"error": f"ValueError: {str(ve)}"}), 400

    except Exception as e:
        logging.exception("Unhandled exception during prediction")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/test", methods=["GET"])
def test_sample():
    try:
        sample_input = {
            "shipping date (DateOrders)": 27148.0,
            "Order Status": 2.0,
            "Delivery Status": 0.0,
            "Late_delivery_risk": 0.0,
            "Shipping Mode": 3.0,
            "Order City": 331.0,
            "Order State": 475.0,
            "Order Region": 15.0,
            "Order Country": 70.0,
            "Latitude": 18.251453,
            "Longitude": -66.037056,
            "Customer Street": 3454.0,
            "Customer City": 66.0,
            "Customer State": 36.0,
            "Customer Country": 1.0,
            "order date (DateOrders)": 5960.0,
            "Order Item Quantity": 1.0,
            "Market": 3.0
        }

        values = [sample_input[col] for col in feature_columns]
        df = pd.DataFrame([values], columns=feature_columns)
        scaled_input = scaler.transform(df)
        prediction = model.predict(scaled_input)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        logging.exception("Error in /test route")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
