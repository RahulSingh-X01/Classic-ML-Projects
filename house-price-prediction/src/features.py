import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder


def feature_engineering(X, y=None, fit=True, location_encoder=None, ohe=None):
    X = X.copy()
    
    if fit and y is None:
        raise ValueError("y must be provided when fit=True")
    
    if not fit and (location_encoder is None or ohe is None):
        raise ValueError("encoders must be provided when fit=False")

    # Extract numerical values from size column
    X['size_num'] = X['size'].str.extract(r'(\d+)').astype(float)
    X['size_num'] = X['size_num'].fillna(X['size_num'].median())

    # Remove outliers only during training
    if fit:
        sqft_per_room = X['total_sqft'] / X['size_num']
        mask = (sqft_per_room >= 300) & (sqft_per_room <= 1500)
        X = X[mask]
        y = y[mask]

    # Feature engineering 
    X['bath_per_bhk'] = X['bath'] / X['size_num']
    X['sqft_per_bhk'] = X['total_sqft'] / X['size_num']

    # Luxury feature
    X['is_luxury'] = (
        (X['size_num'] >= 4) &
        (X['total_sqft'] >= 2000)
    ).astype(int)
    
    # Log transformation
    X['total_sqft'] = np.log1p(X['total_sqft'])

    # Total rooms
    X['total_rooms'] = X['size_num'] + X['bath'] + X['balcony']
    
    # One-Hot encoding 'area_type' feature
    if fit:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        area_encoded = ohe.fit_transform(X[['area_type']])

    else:
        area_encoded = ohe.transform(X[['area_type']])

    area_df = pd.DataFrame(
        area_encoded,
        columns=ohe.get_feature_names_out(['area_type']),
        index=X.index
    )

    X = pd.concat([X, area_df], axis=1)
    
    X = X.drop(['area_type', 'size'], axis=1)

    # Target encoding for high cardinality feature
    if fit:
        location_encoder = TargetEncoder(cols=['location'], smoothing=10)
        X = location_encoder.fit_transform(X, y)
    else:
        X = location_encoder.transform(X)

    if fit:
        return X, y, location_encoder, ohe
    
    else:
        return X, location_encoder, ohe