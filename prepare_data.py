# prepare_data.py
import pandas as pd

print("Starting data preparation...")

# --- 1. Load a manageable sample of the data ---
# The full files are huge. We'll load the first 80,000 rows for the hackathon.
print("Loading a sample of the datasets...")
df_trans = pd.read_csv('train_transaction.csv', nrows=80000)
df_id = pd.read_csv('train_identity.csv', nrows=80000)

# --- 2. Merge the two datasets ---
# The files are linked by the 'TransactionID' column.
print("Merging transaction and identity data...")
df = pd.merge(df_trans, df_id, on='TransactionID', how='left')

# --- 3. Select a few useful features ---
# The full dataset has ~400 columns. Let's pick a few good ones to start.
print("Selecting key features...")
features_to_keep = [
    'isFraud',
    'TransactionAmt', # The amount of the transaction
    'ProductCD',      # Code for the product
    'card4',          # Card type (e.g., visa, mastercard)
    'card6',          # Card type (e.g., credit, debit)
    'addr1',          # An address feature
    'P_emaildomain',  # Purchaser's email domain
    'DeviceType'      # Mobile or Desktop
]
df_clean = df[features_to_keep].copy()

# --- 4. Clean the data ---
# Handle missing values and convert text columns to numbers the AI can understand.
print("Cleaning data and handling missing values...")

# Fill missing text data with 'missing'
for col in ['card4', 'card6', 'P_emaildomain', 'DeviceType']:
    df_clean[col] = df_clean[col].fillna('missing')

# Fill missing numerical data with the median
df_clean['addr1'] = df_clean['addr1'].fillna(df_clean['addr1'].median())

# Convert categorical text columns into numerical ones (One-Hot Encoding)
df_clean = pd.get_dummies(df_clean, columns=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType'], dummy_na=False)

# --- 5. Save the final, prepared dataset ---
output_file = 'fraud_training_prepared.csv'
print(f"Saving the prepared data to '{output_file}'...")
df_clean.to_csv(output_file, index=False)

print("\nData preparation complete!")
print(f"Your new training file '{output_file}' is ready.")