# main.py - Final version with dashboard data generation
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse 
from fastapi.staticfiles import StaticFiles
import io
import numpy as np
import os
import time

# --- Part 1: Train the Model on Startup ---
print("--- Initializing AI Engine ---")

# Define directory paths
REPORTS_DIR = "reports"
STATIC_DIR = "static"
# Create directories if they don't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

try:
    print("Loading the prepared training dataset...")
    df_train = pd.read_csv('fraud_training_prepared.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'fraud_training_prepared.csv' not found.")
    print("Please run the 'prepare_data.py' script first.")
    exit()

print("Training the master AI model...")
X_train = df_train.drop('isFraud', axis=1)
y_train = df_train['isFraud']
MODEL_COLUMNS = X_train.columns
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("--- AI Engine Ready ---")


# --- Part 2: The FastAPI Web Server ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Part 3: The API Endpoints ---
@app.post("/analyze_batch")
async def analyze_batch(file: UploadFile = File(...)):
    print("Received new file for analysis...")
    contents = await file.read()
    df_new = pd.read_csv(io.BytesIO(contents))
    
    total_transactions = len(df_new)
    
    # Live Data Preparation Pipeline
    print("Preparing uploaded data...")
    features_to_keep = ['TransactionAmt', 'ProductCD', 'card4', 'card6', 'addr1', 'P_emaildomain']
    
    required_cols_check = ['TransactionID'] + features_to_keep
    if not all(col in df_new.columns for col in required_cols_check):
        missing_cols = [col for col in required_cols_check if col not in df_new.columns]
        return {"error": f"Missing required columns in uploaded file: {missing_cols}"}

    df_clean = df_new[required_cols_check].copy()

    for col in ['card4', 'card6', 'P_emaildomain']:
        df_clean[col] = df_clean[col].fillna('missing')
    df_clean['addr1'] = df_clean['addr1'].fillna(df_clean['addr1'].median())

    df_clean = pd.get_dummies(df_clean, columns=['ProductCD', 'card4', 'card6', 'P_emaildomain'], dummy_na=False)

    # Align columns with the trained model
    for col in MODEL_COLUMNS:
        if col not in df_clean.columns:
            df_clean[col] = 0
    df_clean = df_clean[MODEL_COLUMNS]
    
    # Prediction
    print("Making predictions...")
    predictions = model.predict(df_clean)
    probabilities = model.predict_proba(df_clean)[:, 1]
    
    df_new['is_fraud_prediction'] = predictions
    df_new['risk_score'] = probabilities
    
    fraudulent_transactions = df_new[df_new['is_fraud_prediction'] == 1]
    
    print(f"Analysis complete. Found {len(fraudulent_transactions)} suspicious transactions.")
    
    download_link = None
    dashboard_data = {} # Initialize dashboard data dictionary
    if not fraudulent_transactions.empty:
        timestamp = int(time.time())
        report_filename = f"fraud_report_{timestamp}.csv"
        report_path = os.path.join(REPORTS_DIR, report_filename)
        fraudulent_transactions.to_csv(report_path, index=False)
        print(f"Full report saved to {report_path}")
        download_link = f"/reports/{report_filename}"

        # --- NEW: Generate data for the dashboard ---
        dashboard_data = {
            "total_fraud_amount": fraudulent_transactions['TransactionAmt'].sum(),
            "fraud_by_card_type": fraudulent_transactions['card4'].value_counts().to_dict(),
            "fraud_by_product_code": fraudulent_transactions['ProductCD'].value_counts().to_dict(),
        }

    TOP_N = 50
    display_results = fraudulent_transactions.sort_values(by='risk_score', ascending=False).head(TOP_N)
    results_df = display_results.replace({np.nan: None})
    
    return {
        "total_transactions": total_transactions,
        "suspicious_count": len(fraudulent_transactions),
        "suspicious_transactions": results_df.to_dict('records'),
        "download_link": download_link,
        "dashboard_data": dashboard_data # Add the new dashboard data to the response
    }

# --- Part 4: Serve Static Files and Frontend ---
# Mount the 'reports' directory so files can be downloaded
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Create a root endpoint to serve the index.html file
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))