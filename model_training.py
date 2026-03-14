import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer # New library needed for handling missing data
import joblib 
import numpy as np # New library needed

# Define the column names for the raw Cleveland dataset
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num' # 'num' is the original target
]

TARGET = 'target'
FEATURES = COLUMNS[:-1] # All columns except the target

try:
    # --- 1. Load Raw Data (No header, '?' for missing values) ---
    data = pd.read_csv(
        'data/heart_disease_raw.csv', 
        names=COLUMNS, 
        na_values='?' # Treat '?' as NaN (Not a Number)
    )
except FileNotFoundError:
    print("Error: heart_disease_raw.csv not found in the 'data/' directory.")
    print("Please ensure you extracted and renamed the 'processed.cleveland.data' file.")
    exit()

# --- 2. Clean and Pre-process Data ---

# Remove rows with missing data (only 6 out of 303 instances are missing)
data.dropna(inplace=True) 

# Convert the 'num' target column (0-4) into a binary 'target' (0 or 1)
# 0 means no disease, 1-4 means presence of heart disease
data[TARGET] = np.where(data['num'] > 0, 1, 0)
data.drop('num', axis=1, inplace=True) # Remove the original target column

X = data[FEATURES].astype(float) # Ensure all features are numeric
y = data[TARGET]

# --- 3. Train Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Logistic Regression
model = LogisticRegression(max_iter=10000) # Increased max_iter for convergence
model.fit(X_train, y_train)

# --- 4. Save Model and Features ---
# Ensure the 'model' directory exists
import os
os.makedirs('model', exist_ok=True) 

joblib.dump(model, 'model/heart_disease_model.pkl')
joblib.dump(FEATURES, 'model/model_features.pkl')

print("---")
print("✅ Model trained and saved successfully as model/heart_disease_model.pkl")
print(f"Data shape used: {data.shape}")
print(f"Features used: {FEATURES}")
print("---")