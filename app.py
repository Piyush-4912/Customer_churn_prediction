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

@app.route('/', methods=['GET'])
def index():
    """Root endpoint returning a visually stunning API dashboard/documentation page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ChurnPredict API Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #09090b;
                --card-bg: rgba(24, 24, 27, 0.6);
                --border: rgba(63, 63, 70, 0.4);
                --text: #f4f4f5;
                --text-muted: #a1a1aa;
                --primary: #6366f1;
                --primary-glow: rgba(99, 102, 241, 0.15);
                --success: #10b981;
                --success-glow: rgba(16, 185, 129, 0.15);
            }
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }
            body {
                font-family: 'Plus Jakarta Sans', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2rem;
                overflow-x: hidden;
                position: relative;
            }
            /* Backdrop Glow Effects */
            body::before {
                content: '';
                position: absolute;
                width: 300px;
                height: 300px;
                background: radial-gradient(circle, var(--primary-glow) 0%, transparent 70%);
                top: 10%;
                left: 10%;
                z-index: 0;
            }
            body::after {
                content: '';
                position: absolute;
                width: 300px;
                height: 300px;
                background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, transparent 70%);
                bottom: 10%;
                right: 10%;
                z-index: 0;
            }
            .container {
                position: relative;
                z-index: 1;
                width: 100%;
                max-width: 800px;
                display: flex;
                flex-direction: column;
                gap: 2rem;
            }
            header {
                text-align: center;
                margin-bottom: 1rem;
            }
            h1 {
                font-size: 2.5rem;
                font-weight: 700;
                letter-spacing: -0.05em;
                background: linear-gradient(135deg, #ffffff 30%, #a1a1aa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            }
            p.subtitle {
                color: var(--text-muted);
                font-size: 1.1rem;
            }
            .card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                backdrop-filter: blur(12px);
                border-radius: 16px;
                padding: 2.5rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            }
            .status-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: var(--success-glow);
                border: 1px solid var(--success);
                color: var(--success);
                padding: 0.4rem 1rem;
                border-radius: 9999px;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
            }
            .status-dot {
                width: 8px;
                height: 8px;
                background-color: var(--success);
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }
            h2 {
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.2rem;
                letter-spacing: -0.02em;
            }
            .endpoint-list {
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }
            .endpoint {
                border-left: 3px solid var(--primary);
                padding-left: 1.2rem;
                transition: all 0.2s ease;
            }
            .endpoint:hover {
                transform: translateX(4px);
            }
            .endpoint-header {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.4rem;
            }
            .method {
                font-size: 0.75rem;
                font-weight: 700;
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                letter-spacing: 0.05em;
            }
            .method.get {
                background: rgba(99, 102, 241, 0.15);
                color: #818cf8;
                border: 1px solid rgba(99, 102, 241, 0.3);
            }
            .method.post {
                background: rgba(236, 72, 153, 0.15);
                color: #f472b6;
                border: 1px solid rgba(236, 72, 153, 0.3);
            }
            .path {
                font-family: monospace;
                font-weight: 600;
                font-size: 1.1rem;
                color: #ffffff;
            }
            .endpoint-desc {
                color: var(--text-muted);
                font-size: 0.95rem;
                margin-bottom: 0.5rem;
            }
            .btn {
                display: inline-block;
                background: var(--primary);
                color: white;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.9rem;
                font-weight: 500;
                transition: background 0.2s;
                border: none;
                cursor: pointer;
            }
            .btn:hover {
                background: #4f46e5;
            }
            footer {
                text-align: center;
                color: var(--text-muted);
                font-size: 0.85rem;
            }
            footer a {
                color: var(--primary);
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>ChurnPredict API</h1>
                <p class="subtitle">Ensemble Learning Model REST API service</p>
            </header>
            
            <div class="card">
                <div class="status-badge">
                    <span class="status-dot"></span>
                    <span>API Online & Healthy</span>
                </div>
                
                <h2>API Endpoints</h2>
                <div class="endpoint-list">
                    <div class="endpoint">
                        <div class="endpoint-header">
                            <span class="method get">GET</span>
                            <span class="path">/health</span>
                        </div>
                        <p class="endpoint-desc">Checks API and machine learning model status.</p>
                        <a href="/health" class="btn">Test Endpoint</a>
                    </div>
                    
                    <div class="endpoint">
                        <div class="endpoint-header">
                            <span class="method post">POST</span>
                            <span class="path">/predict</span>
                        </div>
                        <p class="endpoint-desc">Predicts customer churn probability. Accepts JSON payload of customer records.</p>
                        <span style="color: var(--text-muted); font-size: 0.85rem;">Use <code>test_api.py</code> to test this endpoint.</span>
                    </div>
                </div>
            </div>
            
            <footer>
                <p>Built with Flask, Scikit-Learn, and XGBoost. Pushed to <a href="https://github.com/Piyush-4912/Customer_churn_prediction" target="_blank">GitHub</a>.</p>
            </footer>
        </div>
    </body>
    </html>
    """
    return html_content, 200

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
