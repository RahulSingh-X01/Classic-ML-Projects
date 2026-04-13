import pandas as pd
import numpy as np
import joblib
from src.preprocess import preprocess_data
from src.features import feature_engineering
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def train_data(data):
    