import pandas as pd
import numpy as np
import joblib
from src.preprocess import preprocess_data
from src.features import feature_engineering
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def train_data(data):
    
    # Read the data 
    df = pd.read_csv(data)
    
    # Split the data 
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocess the train and test data
    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    # Apply feature engineering on train and test data
    X_train, y_train, encoder = feature_engineering(X_train, y_train, fit=True)
    X_test = feature_engineering(X_test, fit=False, encoder=encoder)
    
    # Remove outliers from targets
    mask = y_train <= 500
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    # Log transformation
    y_train_log = np.log(y_train)
    
    # Model
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    