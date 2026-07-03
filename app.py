import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Import the feature engineering function from train
from train import engineer_features

app = Flask(__name__)

# Paths to the saved artifacts
PREPROCESSOR_PATH = "models/preprocessor.joblib"
MODEL_PATH = "models/ensemble_model.joblib"

# Global variables for loaded artifacts
preprocessor = None
model = None

def load_artifacts():
    global preprocessor, model
    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        print("Loading preprocessor and ensemble model...")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
        print("Artifacts loaded successfully.")
    else:
        print("Artifact files not found. Please train the model first.")

# Load models at startup
load_artifacts()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if preprocessor is None or model is None:
        # Try to load again in case they were trained after startup
        load_artifacts()
        
    if preprocessor is None or model is None:
        return jsonify({
            "status": "unhealthy", 
            "message": "Model artifacts not loaded."
        }), 503
        
    return jsonify({
        "status": "healthy", 
        "message": "Model is loaded and ready for inference."
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a customer or list of customers.
    Accepts JSON input.
    """
    if preprocessor is None or model is None:
        load_artifacts()
        if preprocessor is None or model is None:
            return jsonify({"error": "Model artifacts are not loaded on server."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    # If it's a single dictionary, wrap it in a list
    is_single = isinstance(data, dict)
    if is_single:
        data = [data]

    try:
        # Convert JSON request data to DataFrame
        df_input = pd.DataFrame(data)
        
        # Ensure customerID is present, if not, generate dummy IDs
        if 'customerID' not in df_input.columns:
            df_input['customerID'] = [f"INPUT-{i}" for i in range(len(df_input))]
            
        # Ensure Churn target is not present (or dropped if present)
        if 'Churn' in df_input.columns:
            df_input = df_input.drop(columns=['Churn'])

        # Check for required features (optional but good practice)
        required_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Age', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'SupportTickets', 'DataUsageGB'
        ]
        
        missing = [col for col in required_features if col not in df_input.columns]
        if missing:
            return jsonify({
                "error": "Missing required fields in the input JSON.",
                "missing_fields": missing
            }), 400

        # Run feature engineering
        df_engineered = engineer_features(df_input)
        
        # Drop identifiers
        X_eval = df_engineered.drop(columns=['customerID'])

        # Preprocess features
        X_eval_processed = preprocessor.transform(X_eval)

        # Generate predictions and probabilities
        predictions = model.predict(X_eval_processed)
        probabilities = model.predict_proba(X_eval_processed)[:, 1]

        results = []
        for i in range(len(df_input)):
            cust_id = df_input.iloc[i].get('customerID', f"INPUT-{i}")
            results.append({
                "customerID": cust_id,
                "churn_prediction": "Yes" if predictions[i] == 1 else "No",
                "churn_probability": round(float(probabilities[i]), 4)
            })

        if is_single:
            return jsonify(results[0]), 200
        else:
            return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Bind to 0.0.0.0 so it can be accessed from outside containers
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
