# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("--- Starting Model Training ---")

# 1. Load the prepared dataset
try:
    df_train = pd.read_csv('fraud_training_prepared.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: 'fraud_training_prepared.csv' not found. Please run 'prepare_data.py' first.")
    exit()

# 2. Prepare data for training
X_train = df_train.drop('isFraud', axis=1)
y_train = df_train['isFraud']

# 3. Train the model
print(f"Training RandomForest model on {len(X_train):,} samples...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")

# 4. Save the trained model and the column list to files
MODEL_FILE = "fraud_model.joblib"
COLUMNS_FILE = "model_columns.pkl"

joblib.dump(model, MODEL_FILE)
joblib.dump(X_train.columns, COLUMNS_FILE)

print(f"\nModel saved to: {MODEL_FILE}")
print(f"Model columns saved to: {COLUMNS_FILE}")
print("--- Training Script Finished ---")