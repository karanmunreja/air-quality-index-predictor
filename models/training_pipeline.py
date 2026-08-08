from models.train import train_RandForestReg, train_ridge, train_XgBoost
from models.evaluate import evaluate
from models.lstm import train_lstm
import pandas as pd
import os


RESULTS_FILE = "model_comparison_results.csv"

def train_and_get_best_model(X_train, X_test, y_train, y_test,X_train_seq,X_test_seq, y_train_seq, y_test_seq, forecast_name):
    results=[]
    models={}
    model=train_RandForestReg(X_train,y_train)
    r2_Score, rmse, mae=evaluate(model,X_test,y_test)
    results.append({
    "Horizon": forecast_name,
    'Model':'Random Forest',
    'r2_Score' : r2_Score,
    'rmse':rmse,
    'mae':mae
})  
    models['Random Forest']=model
    model=train_ridge(X_train,y_train)
    r2_Score, rmse, mae=evaluate(model,X_test,y_test)
    results.append({
    "Horizon": forecast_name,
    'Model':'Ridge',
    'r2_Score' : r2_Score,
    'rmse':rmse,
    'mae':mae
})
    models['Ridge']=model
    model=train_XgBoost(X_train,y_train)
    r2_Score, rmse, mae=evaluate(model,X_test,y_test)
    results.append({
    "Horizon": forecast_name,
    'Model':'XGBoost',
    'r2_Score' : r2_Score,
    'rmse':rmse,
    'mae':mae
})
    models['XGBoost']=model
    model = train_lstm(X_train_seq, y_train_seq)
    r2_Score,rmse,mae = evaluate(model,X_test_seq,y_test_seq)   
    results.append({
    "Horizon": forecast_name,
    'Model':'LSTM',
    'r2_Score' : r2_Score,
    'rmse':rmse,
    'mae':mae
})
    models['LSTM']=model
    results_df=pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        results_df.to_csv(RESULTS_FILE,mode="a", header=False,index=False)
    else:
        results_df.to_csv(RESULTS_FILE,mode="w",header=True,index=False)
    best=results_df.loc[results_df['r2_Score'].idxmax()]
    best_model_name=best['Model']
    best_model=models[best_model_name]
    return best_model_name, best_model, best

