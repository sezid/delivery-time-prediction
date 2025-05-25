from flask import Flask, request, jsonify
import xgboost as xgb
import pickle
import numpy as np
import pandas as pd
import logging
import os

app = Flask(__name__)

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

# Categorical columns that have label encoders
categorical_columns = [
    "Order Status", "Delivery Status", "Shipping Mode", "Order City", "Order State",
    "Order Region", "Order Country", "Customer Street", "Customer City",
    "Customer State", "Customer Country", "Market"
]

# Date columns
date_columns = ["shipping date (DateOrders)", "order date (DateOrders)"]

# Load LabelEncoders for categorical columns
label_encoders = {}
for col in categorical_columns:
    le_path = f"le_{col.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    if os.path.exists(le_path):
        with open(le_path, "rb") as f:
            label_encoders[col] = pickle.load(f)
        logging.info(f"Loaded LabelEncoder for column: {col}")
    else:
        logging.warning(f"LabelEncoder file not found for column: {col}, expected at {le_path}")

def preprocess_input(df):
    # 1. Parse date columns to numeric (timestamp)
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')  # convert to datetime, invalid => NaT
        if df[col].isnull().any():
            raise ValueError(f"Invalid date format detected in column: {col}")
        # Convert datetime to timestamp (float seconds since epoch)
        df[col] = df[col].astype(np.int64) // 10**9  # Convert nanoseconds to seconds (int)
    
    # 2. Apply label encoding for categorical columns
    for col in categorical_columns:
        if col in df.columns:
            le = label_encoders.get(col, None)
            if le is None:
                raise ValueError(f"LabelEncoder not loaded for column: {col}")
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError as ve:
                # Instead of error, map unseen to -1 or a default value
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    return df

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

        # Ensure correct feature order and build DataFrame
        values = [features[col] for col in feature_columns]
        df = pd.DataFrame([values], columns=feature_columns)

        # Preprocess: parse dates and label encode categories
        df_processed = preprocess_input(df)

        # Scale and predict
        scaled_input = scaler.transform(df_processed)
        prediction = model.predict(scaled_input)

        return jsonify({"prediction": prediction.tolist()})

    except ValueError as ve:
        logging.exception("Value error during prediction")
        return jsonify({"error": f"ValueError: {str(ve)}"}), 400
    except Exception as e:
        logging.exception("Unhandled exception during prediction")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Sample input JSON for /test route
sample_input = {
   "features": {
    "shipping date (DateOrders)": "4/1/2016 00:00",
    "Order Status": "PROCESSING",
    "Delivery Status": "Late delivery",
    "Late_delivery_risk": 1,
    "Shipping Mode": "Standard Class",
    "Order City": "Los Angeles",
    "Order State": "California",
    "Order Region": "West of USA",
    "Order Country": "Estados Unidos",
    "Latitude": 18.20397568,
    "Longitude": -66.37054443,
    "Customer Street": "8916 Round Zephyr Ridge",
    "Customer City": "Caguas",
    "Customer State": "PR",
    "Customer Country": "Puerto Rico",
    "order date (DateOrders)": "4/1/2016 00:00",
    "Order Item Quantity": 1,
    "Market": "West of USA"
  }
}



@app.route("/test", methods=["GET"])
def test():
    try:
        features = sample_input["features"]
        values = [features[col] for col in feature_columns]
        df = pd.DataFrame([values], columns=feature_columns)
        df_processed = preprocess_input(df)
        scaled_input = scaler.transform(df_processed)
        prediction = model.predict(scaled_input)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        logging.exception("Error during test prediction")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
