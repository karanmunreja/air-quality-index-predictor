from models.train import train_RandForestReg, train_ridge, train_XgBoost
from models.evaluate import evaluate_multioutput
from models.lstm import train_lstm
import pandas as pd
import os


RESULTS_FILE = "model_comparison_results.csv"

def train_and_get_best_model(X_train, X_test, y_train, y_test, X_train_seq, X_test_seq, y_train_seq, y_test_seq):
    results=[]
    models={}
    model=train_RandForestReg(X_train,y_train)
    metrics=evaluate_multioutput(model,X_test,y_test)
    metrics["average_r2"] = (metrics["r2_24"] + metrics["r2_48"]+ metrics["r2_72"]) / 3
    results.append({
    'Model':'Random Forest',
    **metrics
})  
    models['Random Forest']=model
    model=train_ridge(X_train,y_train)
    metrics=evaluate_multioutput(model,X_test,y_test)
    metrics["average_r2"] = (metrics["r2_24"] + metrics["r2_48"]+ metrics["r2_72"]) / 3
    results.append({
    'Model':'Ridge',
    **metrics
})
    models['Ridge']=model
    model=train_XgBoost(X_train,y_train)
    metrics=evaluate_multioutput(model,X_test,y_test)
    metrics["average_r2"] = (metrics["r2_24"] + metrics["r2_48"]+ metrics["r2_72"]) / 3
    results.append({
    'Model':'XGBoost',
    **metrics
})
    models['XGBoost']=model
    model = train_lstm(X_train_seq, y_train_seq)
    metrics = evaluate_multioutput(model,X_test_seq,y_test_seq) 
    metrics["average_r2"] = (metrics["r2_24"] + metrics["r2_48"]+ metrics["r2_72"]) / 3  
    results.append({
    'Model':'LSTM',
    **metrics
})
    models['LSTM']=model
    results_df=pd.DataFrame(results)
    if os.path.exists(RESULTS_FILE):
        results_df.to_csv(RESULTS_FILE,mode="a", header=False,index=False)
    else:
        results_df.to_csv(RESULTS_FILE,mode="w",header=True,index=False)
    best=results_df.loc[results_df['average_r2'].idxmax()]
    best_model_name=best['Model']
    best_model=models[best_model_name]
    return best_model_name, best_model, best

