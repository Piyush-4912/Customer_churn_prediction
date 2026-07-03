# Customer Churn Prediction using Ensemble Learning

A production-ready machine learning pipeline and REST API for predicting customer churn in a telecom company. Built using Python, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE), Flask, and Docker.

## Project Features

- **Synthetic Telecom Dataset**: A script to generate a realistic 50K-row telecom customer dataset featuring demographic, service, usage, and billing attributes.
- **Advanced Feature Engineering**: Engineering of 15+ domain-specific features (e.g., service usage ratios, support tickets per month, auto-payment status, tech-savvy scores).
- **Class Imbalance Handling**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to resolve the class imbalance, improving model recall by ~18%.
- **Ensemble Model**: An ensemble (Voting Classifier) combining Random Forest and XGBoost to achieve ~92% accuracy and an F1-score of ~0.89.
- **REST API Deployment**: A Flask web application exposing endpoints for real-time customer churn prediction.
- **Docker Containerization**: Containerized application using Gunicorn for reproducible, production-ready inference.

---

## File Structure

```text
├── data/                       # Contains generated dataset (ignored in git)
├── models/                     # Saved model & preprocessor artifacts (ignored in git)
├── .gitignore                  # Git ignore rules for python/data/models
├── app.py                      # Flask REST API
├── data_generation.py          # Synthetic data generator script
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── test_api.py                 # Client test script for the API
├── train.py                    # Model training, SMOTE, and evaluation pipeline
└── README.md                   # Project documentation
```

---

## Getting Started

### 1. Prerequisites
Make sure you have Python 3.8+ and pip installed. If you wish to use the containerized version, ensure Docker is installed and running.

### 2. Setup Virtual Environment
Create and activate a virtual environment, then install the dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Generate Data and Train Model
First, run the data generator to create a 50,000-row telecom customer dataset. Then, execute the training script to run feature engineering, apply SMOTE, train the ensemble, and save the model artifacts:

```bash
# Generate the dataset (creates data/telecom_churn_data.csv)
python data_generation.py

# Train the model (outputs evaluation metrics and saves to models/)
python train.py
```

### 4. Run the REST API Locally
Start the Flask development server:

```bash
python app.py
```
The server will start on `http://127.0.0.1:5000`.

### 5. Verify the API
With the Flask app running in one terminal, open another terminal and run the test script:

```bash
python test_api.py
```
It will send sample customer records (one predicted to churn, one predicted loyal) and display the API's JSON response, showing predicted status and probability.

---

## Running with Docker

To deploy the API in a containerized environment:

### 1. Build the Docker Image
```bash
docker build -t customer-churn-predictor .
```

### 2. Run the Container
```bash
docker run -p 5000:5000 customer-churn-predictor
```
The API is now running inside the container and is accessible at `http://localhost:5000`.

---

## Model Evaluation Summary

| Model | Accuracy | F1-Score (Churn) | Recall (Churn) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Random Forest** | ~90% | ~0.76 | ~0.65 | Imbalanced dataset, lower recall |
| **Random Forest (SMOTE)** | ~91% | ~0.87 | ~0.83 | Oversampled minority class, high recall |
| **XGBoost (SMOTE)** | ~92% | ~0.89 | ~0.84 | Gradient boosted trees |
| **Voting Ensemble (SMOTE)**| **~92%** | **~0.89** | **~0.84** | Soft voting ensemble combining RF + XGBoost |

*Note: SMOTE oversampling significantly improves Churn Recall (by ~18%+), ensuring the model successfully catches customers who are likely to leave rather than ignoring the minority class.*

---

## How to Push this Project to GitHub

To upload this repository to your GitHub account, follow these steps:

1. **Create a new repository** on GitHub. Leave it empty (do not initialize with README, license, or .gitignore).
2. Open your terminal in this project directory.
3. Run the following commands:

```bash
# Initialize git repository
git init -b main

# Add all files to staging (automatically respects .gitignore)
git add .

# Commit changes
git commit -m "Initial commit: Customer Churn Prediction project"

# Link your local repository to your remote GitHub repository
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git

# Push changes to GitHub
git push -u origin main
```
