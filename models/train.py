from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def train_RandForestReg(X_train,y_train):
    model=RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    print('model training...')
    model.fit(X_train,y_train)
    print('Random Forest Trained Successfully')    
    return model

def train_ridge(X_train,y_train):
    model=RidgeCV(alphas = [0.001, 0.01, 0.1, 1, 5, 10, 50, 100])
    print('model ridge training... ')
    model.fit(X_train,y_train)
    print('Model Ridge Trained Successfully')
    return model

def train_XgBoost(X_train,y_train):
    base_model = XGBRegressor(
        n_estimators = 400,
        max_depth = 7,
        learning_rate =  0.05,
        subsample =0.8,
        colsample_bytree = 0.8,
        random_state=42
    )
    model = MultiOutputRegressor(
        base_model
    )
    print('training XGBoost... ')
    model.fit(X_train,y_train)
    print('Model XGBoost Trained successfully')
    return model
