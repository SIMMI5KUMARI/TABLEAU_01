import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# --- Configuration ---
DATA_PATH = 'cardio_train.csv'  # Ensure this file is in your main directory
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'heart_disease_model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.pkl')

def train_and_save_model():
    """Loads the Cardiovascular Disease dataset, trains a Logistic Regression model, and saves the results."""
    print("--- Starting Initial Model Training ---")
    
    # 1. Load Data
    try:
        # The dataset might be semicolon-separated, so we check the delimiter
        df = pd.read_csv(DATA_PATH, sep=';')
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {DATA_PATH}. Please check the file name and location.")
        return
    
    print(f"Successfully loaded {len(df)} records.")

    # 2. Data Cleaning and Preparation
    
    # Drop the 'id' column as it's not a feature
    df = df.drop('id', axis=1)
    
    # Check for and handle duplicates
    df.drop_duplicates(inplace=True)
    print(f"Data after removing duplicates: {len(df)} records.")

    # Feature Engineering/Cleaning (Specific to the Kaggle Cardiovascular Dataset)
    # The 'ap_hi' (systolic) and 'ap_lo' (diastolic) columns often have outliers.
    # We clean extreme outliers that would skew the model severely.
    df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200) & (df['ap_hi'] > 50) & (df['ap_lo'] > 50)]
    df = df[(df['ap_hi'] >= df['ap_lo'])] # Systolic should be greater than diastolic
    
    # Convert 'age' from days to years (approximate)
    df['age'] = (df['age'] / 365.25).round().astype(int)

    # 3. Define Features (X) and Target (y)
    
    # The target variable is 'cardio' (1=patient has cardiovascular disease, 0=healthy)
    target_column = 'cardio'
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Save the final list of feature columns (CRITICAL for Flask app consistency)
    feature_list = X.columns.tolist()
    
    # 4. Training
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluation and Saving
    
    # Check accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Trained Successfully.")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the trained model and the feature list
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_list, FEATURES_PATH)
    
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Features list saved to: {FEATURES_PATH}")
    print("\n--- Initial Model Training Complete ---")


if __name__ == '__main__':
    train_and_save_model()