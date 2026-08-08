from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd

def evaluate(model, X_test,y_test):
    predictions=model.predict(X_test)
    r2_Score=r2_score(y_test,predictions)
    rmse=root_mean_squared_error(y_test,predictions)
    mae=mean_absolute_error(y_test,predictions)
    return r2_Score,rmse, mae
