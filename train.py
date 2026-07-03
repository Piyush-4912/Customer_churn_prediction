import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def engineer_features(df_input):
    """
    Engineers new features to improve customer churn prediction accuracy.
    """
    df = df_input.copy()
    
    # 1. Total Services Used
    services = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    df['TotalServices'] = 0
    for service in services:
        df['TotalServices'] += (df[service] == 'Yes').astype(int)
    # Add Internet service itself as a service
    df['TotalServices'] += (df['InternetService'] != 'No').astype(int)
    
    # 2. Support Tickets per month of tenure
    df['SupportTicketsPerMonth'] = df['SupportTickets'] / (df['tenure'] + 1)
    
    # 3. Monthly charges relative to average charges per tenure month
    df['ChargesPerMonthRatio'] = (df['TotalCharges'] / (df['tenure'] + 1)) / (df['MonthlyCharges'] + 1e-5)
    
    # 4. Data usage efficiency (GB per Dollar spent)
    df['DataUsageCostRatio'] = df['DataUsageGB'] / (df['MonthlyCharges'] + 1e-5)
    
    # 5. Contract length mapping in months
    contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['ContractLengthMonths'] = df['Contract'].map(contract_mapping)
    
    # 6. Total contract value
    df['TotalContractValue'] = df['ContractLengthMonths'] * df['MonthlyCharges']
    
    # 7. Interaction: High tickets and short contract
    df['HighTicketShortContract'] = ((df['SupportTickets'] > 2) & (df['Contract'] == 'Month-to-month')).astype(int)
    
    # 8. Demographic Interaction: Senior Citizen & Single
    df['SeniorSingle'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
    
    # 9. Additional engineered features (ratio of tenure to contract length)
    df['TenureToContractRatio'] = df['tenure'] / (df['ContractLengthMonths'] + 1e-5)
    
    # 10. Auto-payment method flag
    df['IsAutoPay'] = df['PaymentMethod'].str.contains('automatic').astype(int)
    
    # 11. Tech savvy flag (uses security, backup, protection, and support)
    df['TechSavvyScore'] = (
        (df['OnlineSecurity'] == 'Yes').astype(int) + 
        (df['OnlineBackup'] == 'Yes').astype(int) + 
        (df['DeviceProtection'] == 'Yes').astype(int) + 
        (df['TechSupport'] == 'Yes').astype(int)
    )
    
    # 12. Fiber Optic but no Support
    df['FiberNoSupport'] = ((df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')).astype(int)
    
    # 13. Average charge per service
    df['AvgChargePerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1)
    
    # 14. Support tickets to total service ratio
    df['TicketsPerServiceRatio'] = df['SupportTickets'] / (df['TotalServices'] + 1)
    
    # 15. Standardized monthly usage variation (TotalCharges deviation from linear growth)
    df['ChargesDeviation'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
    
    return df

def main():
    # Load dataset
    data_path = "data/telecom_churn_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run data_generation.py first.")
        
    df = pd.read_csv(data_path)
    
    # Map target variable Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Process features
    print("Engineering 15+ features...")
    df_engineered = engineer_features(df)
    
    # Separate features and target
    X = df_engineered.drop(columns=['customerID', 'Churn'])
    y = df_engineered['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int32', 'int64', 'float64']).columns.tolist()
    
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Setup preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit preprocessor on training data
    print("Fitting preprocessor pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for reference if needed
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + encoded_cat_cols
    print(f"Total features after preprocessing: {len(all_feature_names)}")
    
    # --- 1. Baseline Model (Imbalanced Data) ---
    print("\n--- Training Baseline Random Forest (Imbalanced Data) ---")
    rf_baseline = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12, n_jobs=-1)
    rf_baseline.fit(X_train_processed, y_train)
    y_pred_baseline = rf_baseline.predict(X_test_processed)
    
    base_acc = accuracy_score(y_test, y_pred_baseline)
    base_rec = recall_score(y_test, y_pred_baseline)
    base_f1 = f1_score(y_test, y_pred_baseline)
    
    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Baseline Recall (Churn=1): {base_rec:.4f}")
    print(f"Baseline F1-Score (Churn=1): {base_f1:.4f}")
    print("Baseline Classification Report:")
    print(classification_report(y_test, y_pred_baseline))
    
    # --- 2. Apply SMOTE to handle Class Imbalance ---
    print("\n--- Applying SMOTE to Training Set ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)
    
    print(f"Original training shape: {X_train_processed.shape}, Churn distribution: {np.bincount(y_train)}")
    print(f"Resampled training shape: {X_train_res.shape}, Churn distribution: {np.bincount(y_train_res)}")
    
    # --- 3. Train Random Forest on SMOTE Data ---
    print("\n--- Training Random Forest on SMOTE Balanced Data ---")
    rf_smote = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=15, min_samples_leaf=2, n_jobs=-1)
    rf_smote.fit(X_train_res, y_train_res)
    y_pred_rf = rf_smote.predict(X_test_processed)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    
    print(f"RF (SMOTE) Accuracy: {rf_acc:.4f}")
    print(f"RF (SMOTE) Recall (Churn=1): {rf_rec:.4f} (Improvement over baseline: {(rf_rec - base_rec)*100:+.2f}%)")
    print(f"RF (SMOTE) F1-Score (Churn=1): {rf_f1:.4f}")
    
    # --- 4. Train XGBoost on SMOTE Data ---
    print("\n--- Training XGBoost on SMOTE Balanced Data ---")
    xgb_model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train_res, y_train_res)
    y_pred_xgb = xgb_model.predict(X_test_processed)
    
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_rec = recall_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    
    print(f"XGB (SMOTE) Accuracy: {xgb_acc:.4f}")
    print(f"XGB (SMOTE) Recall (Churn=1): {xgb_rec:.4f} (Improvement over baseline: {(xgb_rec - base_rec)*100:+.2f}%)")
    print(f"XGB (SMOTE) F1-Score (Churn=1): {xgb_f1:.4f}")
    
    # --- 5. Ensemble: Soft Voting Classifier ---
    print("\n--- Training Ensemble (Voting Classifier: RF + XGBoost) ---")
    # We define models configured similarly
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=15, min_samples_leaf=2, n_jobs=-1)
    xgb_clf = XGBClassifier(
        random_state=42,
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft',
        n_jobs=-1
    )
    ensemble.fit(X_train_res, y_train_res)
    
    # Evaluate Ensemble
    y_pred_ens = ensemble.predict(X_test_processed)
    ens_acc = accuracy_score(y_test, y_pred_ens)
    ens_rec = recall_score(y_test, y_pred_ens)
    ens_f1 = f1_score(y_test, y_pred_ens)
    
    print(f"\nEnsemble Model Evaluation:")
    print(f"Accuracy: {ens_acc:.4f} (Target: ~92%)")
    print(f"F1-Score (Churn=1): {ens_f1:.4f} (Target: ~0.89)")
    print(f"Recall (Churn=1): {ens_rec:.4f} (Improvement over baseline: {(ens_rec - base_rec)*100:+.2f}%)")
    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, y_pred_ens))
    
    # Save the models and pipelines
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(ensemble, "models/ensemble_model.joblib")
    print("Saved preprocessor to models/preprocessor.joblib")
    print("Saved ensemble model to models/ensemble_model.joblib")

if __name__ == "__main__":
    main()
