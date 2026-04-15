import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Convert total_sqft from string to numerical
def convert_string(x):
    try:
        x = str(x)

        if '-' in x:
            a, b = x.split('-')
            return (float(a) + float(b)) / 2
        
        return float(x)

    except:
        return np.nan
    
    
# Main preprocessing fucntion
def preprocess_data(df, num_imputer=None, cat_imputer=None, fit=True):
    df = df.copy()

    # Drop useless column
    if 'society' in df.columns:
        df = df.drop('society', axis=1)

    # Convert total_sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_string)

    # Drop rows where total_sqft is NaN
    df = df.dropna(subset=['total_sqft'])

    # Impute missing values of numerical and categorical columns
    numerical_cols = ['bath', 'balcony']
    categorical_cols = ['area_type', 'availability', 'location', 'size']
    
    if fit:
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    else:
        # Reuse the imputers fitted on train data
        df[numerical_cols] = num_imputer.transform(df[numerical_cols])
        df[categorical_cols] = cat_imputer.transform(df[categorical_cols])
    
    

    # Encode avaliability into 1(ready) or 0(not ready)
    df['availability'] = df['availability'].str.strip().str.lower()
    df['availability'] = df['availability'].apply(
        lambda x: 1 if x == 'ready to move' else 0
    )

    if fit:
        return df, num_imputer, cat_imputer
    else:
        return df
