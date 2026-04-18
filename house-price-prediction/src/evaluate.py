import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score

def evaluate(model, X_train, y_train_log, X_test, y_test):
    
    # Train score
    y_train_pred = model.predict(X_train)
    
    # CV score on train
    cv_scores = cross_val_score(model, X_train, y_train_log, cv=5, scoring='r2')
    mean_cv = cv_scores.mean()
    
    # Train metrics
    train_rmse = root_mean_squared_error(y_train_log, y_train_pred)
    train_r2 = root_mean_squared_error(y_train_log, y_train_pred)

    