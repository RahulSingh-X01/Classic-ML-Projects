import numpy as np
from category_encoders import TargetEncoder


def feature_engineering(X, y=None, fit=True, encoder=None):
    X = X.copy()

    # Extract numerical values from size column
    X['size_num'] = X['size'].str.extract(r'(\d+)').astype(int)

    # Remove outliers only during training
    if fit:
        sqft_per_room = X['total_sqft'] / X['size_num']
        mask = (sqft_per_room >= 300) & (sqft_per_room <= 1500)
        X = X[mask]
        y = y.loc[X.index]

    # Feature engineering 
    X['bath_per_bhk'] = X['bath'] / X['size_num']
    X['sqft_per_bhk'] = X['total_sqft'] / X['size_num']

    # Log transformation
    X['total_sqft'] = np.log(X['total_sqft'])

    # Luxury feature
    X['is_luxury'] = (
        (X['size_num'] >= 4) &
        (X['total_sqft'] >= np.log(2000))
    ).astype(int)

    # Total rooms
    X['total_rooms'] = X['size_num'] + X['bath'] + X['balcony']

    # Target encoding for high cardinality feature
    if fit:
        encoder = TargetEncoder(cols=['location'], smoothing=10)
        X = encoder.fit_transform(X, y)
    else:
        X = encoder.transform(X)

    # Drop unused columns
    X = X.drop(['size'], axis=1)

    if fit:
        return X, y, encoder
    return X
