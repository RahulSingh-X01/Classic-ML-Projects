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
