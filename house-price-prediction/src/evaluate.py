import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

def evaluate(model, X_train, y_train_log, X_test, y_test):
    