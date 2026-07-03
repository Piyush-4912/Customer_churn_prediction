import os
import numpy as np
import pandas as pd

def generate_telecom_data(num_rows=50000, seed=42):
    np.random.seed(seed)
    
    # 1. Demographic Features
    customer_ids = [f"CUST-{i:05d}" for i in range(1, num_rows + 1)]
    gender = np.random.choice(["Male", "Female"], size=num_rows)
    senior_citizen = np.random.choice([0, 1], p=[0.85, 0.15], size=num_rows)
    partner = np.random.choice(["Yes", "No"], size=num_rows)
    dependents = np.random.choice(["Yes", "No"], p=[0.7, 0.3], size=num_rows)
    age = np.random.randint(18, 85, size=num_rows)
    
    # 2. Account & contract Features
    tenure = np.random.randint(1, 73, size=num_rows)  # 1 to 72 months
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.20, 0.25], size=num_rows)
    paperless_billing = np.random.choice(["Yes", "No"], p=[0.6, 0.4], size=num_rows)
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.35, 0.25, 0.20, 0.20],
        size=num_rows
    )
    
    # 3. Services Features
    phone_service = np.random.choice(["Yes", "No"], p=[0.9, 0.1], size=num_rows)
    
    # Multiple lines depends on phone service
    multiple_lines = []
    for p_serv in phone_service:
        if p_serv == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
    multiple_lines = np.array(multiple_lines)
    
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], p=[0.3, 0.5, 0.2], size=num_rows)
    
    # Other services depend on internet service
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for i_serv in internet_service:
        if i_serv == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.30, 0.70]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            
    online_security = np.array(online_security)
    online_backup = np.array(online_backup)
    device_protection = np.array(device_protection)
    tech_support = np.array(tech_support)
    streaming_tv = np.array(streaming_tv)
    streaming_movies = np.array(streaming_movies)
    
    # 4. Usage & Charges
    monthly_charges = np.zeros(num_rows)
    for i in range(num_rows):
        base = 20.0
        if phone_service[i] == "Yes":
            base += 10.0
            if multiple_lines[i] == "Yes":
                base += 10.0
        if internet_service[i] == "DSL":
            base += 30.0
        elif internet_service[i] == "Fiber optic":
            base += 50.0
            
        # Add charges for add-on services if they have internet
        if internet_service[i] != "No":
            if online_security[i] == "Yes": base += 8.0
            if online_backup[i] == "Yes": base += 10.0
            if device_protection[i] == "Yes": base += 10.0
            if tech_support[i] == "Yes": base += 8.0
            if streaming_tv[i] == "Yes": base += 12.0
            if streaming_movies[i] == "Yes": base += 12.0
            
        # Add some variation
        monthly_charges[i] = base + np.random.normal(0, 3.0)
    
    # clip to reasonable range
    monthly_charges = np.clip(monthly_charges, 18.0, 125.0)
    
    # total charges is tenure * monthly_charges + some noise
    total_charges = tenure * monthly_charges + np.random.normal(0, 15.0, size=num_rows)
    total_charges = np.clip(total_charges, 18.0, None)
    
    # 5. Customer Service metrics
    support_tickets = np.random.negative_binomial(1, 0.4, size=num_rows) # mostly 0, 1, 2, but some higher
    support_tickets = np.clip(support_tickets, 0, 12)
    
    data_usage = np.zeros(num_rows)
    for i in range(num_rows):
        if internet_service[i] == "No":
            data_usage[i] = 0.0
        elif internet_service[i] == "DSL":
            data_usage[i] = np.random.gamma(2.0, 25.0)  # avg 50 GB
        else:
            data_usage[i] = np.random.gamma(5.0, 50.0)  # avg 250 GB
    
    # Churn logic (imbalanced, roughly 15-20% churn rate)
    log_odds = -2.5
    
    # Add influences (positive increases churn probability)
    log_odds += (contract == "Month-to-month") * 2.2
    log_odds += (contract == "One year") * 0.5
    log_odds += (tenure < 12) * 1.5
    log_odds += (tenure > 48) * -1.8
    log_odds += (internet_service == "Fiber optic") * 1.2
    log_odds += (tech_support == "No") * 0.8
    log_odds += (online_security == "No") * 0.6
    log_odds += (support_tickets) * 0.55
    log_odds += (payment_method == "Electronic check") * 0.7
    log_odds += (monthly_charges > 85.0) * 0.5
    log_odds += (senior_citizen == 1) * 0.3
    log_odds += (dependents == "No") * 0.3
    
    # interaction terms
    log_odds += ((contract == "Month-to-month") & (support_tickets > 2)) * 1.1
    
    # Noise
    log_odds += np.random.normal(0, 0.4, size=num_rows)
    
    # Calculate probability
    prob = 1 / (1 + np.exp(-log_odds))
    
    # Churn decision
    churn_labels = (np.random.rand(num_rows) < prob).astype(int)
    churn = np.where(churn_labels == 1, "Yes", "No")
    
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Age": age,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": np.round(total_charges, 2),
        "SupportTickets": support_tickets,
        "DataUsageGB": np.round(data_usage, 2),
        "Churn": churn
    })
    
    print(f"Generated dataset with {num_rows} rows.")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    print(f"Churn Distribution:")
    for val, count in churn_counts.items():
        print(f"  {val}: {count} ({churn_pct[val]:.2f}%)")
        
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_telecom_data(50000)
    df.to_csv("data/telecom_churn_data.csv", index=False)
    print("Dataset saved to data/telecom_churn_data.csv")
