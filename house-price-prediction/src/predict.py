import pandas as pd
import numpy as np
import joblib
from preprocess import preprocess_data
from features import feature_engineering

model = joblib.load("house-price-prediction/models/xgb_model.pkl")
ohe = joblib.load("house-price-prediction/models/ohe.pkl")
location_encoder = joblib.load("house-price-prediction/models/location_encoder.pkl")
num_imputer = joblib.load("house-price-prediction/models/num_imputer.pkl")
cat_imputer = joblib.load("house-price-prediction/models/cat_imputer.pkl")
loc_ppsf = joblib.load("house-price-prediction/models/loc_ppsf.pkl")

def predict(area_type, availability, location, size, total_sqft, bath, balcony):
    
    # Convert user input to DataFrame
    df = pd.DataFrame([{
    'area_type': area_type,
    'availability': availability,
    'location': location,
    'size': size,
    'total_sqft': total_sqft,
    'bath': bath,
    'balcony': balcony
    }])
    
    # Preprocess the user input
    df = preprocess_data(df, num_imputer=num_imputer, cat_imputer=cat_imputer, fit=False)
    
    # Feature engineer the user input 
    df = feature_engineering(df, fit=False, location_encoder=location_encoder, ohe=ohe, loc_ppsf=loc_ppsf)
    

    