import requests
import json

def test_api():
    url = "http://127.0.0.1:5000/predict"
    
    # Test payloads
    customers = [
        {
            "customerID": "TEST-CHURN-01",
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "Age": 68,
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.50,
            "TotalCharges": 268.50,
            "SupportTickets": 4,
            "DataUsageGB": 120.5
        },
        {
            "customerID": "TEST-LOYAL-02",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Age": 45,
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 65.00,
            "TotalCharges": 3900.00,
            "SupportTickets": 0,
            "DataUsageGB": 45.2
        }
    ]
    
    headers = {"Content-Type": "application/json"}
    
    # Test Health Check first
    try:
        health_url = "http://127.0.0.1:5000/health"
        health_response = requests.get(health_url)
        print(f"Health Check Status: {health_response.status_code}")
        print(f"Health Check Response: {health_response.json()}\n")
    except Exception as e:
        print(f"Could not connect to health endpoint: {e}\n")
        
    # Test predicting single customer (Customer 1)
    print("Testing single customer prediction (expected Churn)...")
    try:
        response = requests.post(url, data=json.dumps(customers[0]), headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error testing single prediction: {e}\n")

    # Test predicting multiple customers
    print("Testing batch prediction (expected Churn and Loyal)...")
    try:
        response = requests.post(url, data=json.dumps(customers), headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error testing batch prediction: {e}\n")

if __name__ == "__main__":
    test_api()
