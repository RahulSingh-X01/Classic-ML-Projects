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
def preprocess_data(df):
    df = df.copy()

    # Drop useless column
    if 'society' in df.columns:
        df = df.drop('society', axis=1)

    # Convert total_sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_string)

    # Drop rows where total_sqft is NaN
    df = df.dropna(subset=['total_sqft'])

    # Impute numerical columns with median values
    numerical_cols = ['total_sqft', 'bath', 'balcony']
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute categorical columns with most frequent(mode) values
    categorical_cols = ['area_type', 'availability', 'location', 'size']
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Encode avaliability into 1(ready) or 0(not ready)
    df['availability'] = df['availability'].str.strip().str.lower()
    df['availability'] = df['availability'].apply(
        lambda x: 1 if x == 'ready to move' else 0
    )

    return df
