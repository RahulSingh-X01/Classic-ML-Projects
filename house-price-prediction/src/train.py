import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess_data
from features import feature_engineering
from evaluate import evaluate
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def train_data(data):
    
    # Read the data 
    df = pd.read_csv(data)
    
    # Remove outliers
    df = df[df['price'] <= 500]
    
    # Split the data 
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocess the train and test data
    X_train, num_imputer, cat_imputer = preprocess_data(X_train, fit=True)
    X_test = preprocess_data(X_test, num_imputer=num_imputer, cat_imputer=cat_imputer, fit=False)
    
    # Apply feature engineering on train and test data
    X_train, y_train, location_encoder, ohe, loc_ppsf = feature_engineering(X_train, y_train, fit=True)
    X_test = feature_engineering(X_test,fit=False,location_encoder=location_encoder,ohe=ohe, loc_ppsf=loc_ppsf)
    
    # To align indices
    y_test = y_test.loc[X_test.index]
    
    # Log transformation
    y_train_log = np.log1p(y_train)
    
    # Model
    model = XGBRegressor(
    n_estimators=1000,
    max_depth=4,          
    learning_rate=0.05,   
    subsample=0.8,
    colsample_bytree=0.8, 
    reg_alpha=0.1,        
    reg_lambda=2.0,       
    random_state=42
)

    # Model training
    model.fit(X_train, y_train_log)
    
    # Model evaluation
    evaluate(model, X_train, y_train_log, X_test, y_test)

    # Save model and encoders
    joblib.dump(model, "house-price-prediction/models/xgb_model.pkl")
    joblib.dump(location_encoder, "house-price-prediction/models/location_encoder.pkl")
    joblib.dump(ohe, "house-price-prediction/models/ohe.pkl")
    joblib.dump(num_imputer, "house-price-prediction/models/num_imputer.pkl")
    joblib.dump(cat_imputer, "house-price-prediction/models/cat_imputer.pkl")
    joblib.dump(loc_ppsf, "house-price-prediction/models/loc_ppsf.pkl")
    
    print("✅ Model trained and saved.")
    
if __name__ == "__main__":
    train_data(r"house-price-prediction\data\Bengaluru_House_Data.csv")
    
