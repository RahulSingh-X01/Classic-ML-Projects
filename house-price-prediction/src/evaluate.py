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
    train_r2 = r2_score(y_train_log, y_train_pred)

    # Test prediction converted back to original
    y_test_pred = np.expm1(model.predict(X_test))
    
    # Test metrics
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    