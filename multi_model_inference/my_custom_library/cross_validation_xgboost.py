import xgboost as xgb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error



def cross_validation(df, K, hyperparameters):
    """
    Perform XGBoost cross validation on a dataset.
    :param df: pandas.DataFrame
    :param K: int
    :param hyperparameters: dict
    """
    train_indices = list(df.sample(frac=1).index)
    k_folds = np.array_split(train_indices, K)
    if K == 1:
        K = 2

    rmse_list = []
    for i in range(len(k_folds)):
        training_folds = [fold for j, fold in enumerate(k_folds) if j != i]
        training_indices = np.concatenate(training_folds)
        x_train, y_train = df.iloc[training_indices, 1:], df.iloc[training_indices, :1]
        x_validation, y_validation = df.iloc[k_folds[i], 1:], df.iloc[k_folds[i], :1]
        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dvalidation = xgb.DMatrix(data=x_validation, label=y_validation)

        model = xgb.train(
            params=hyperparameters,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
        )
        eval_results = model.eval(dvalidation)
        rmse_list.append(float(eval_results.split("eval-rmse:")[1]))
    return rmse_list, model



def cross_validation_catboost(df, K, hyperparameters):
    """
    Perform CatBoost cross validation on a dataset.
    :param df: pandas.DataFrame
    :param K: int
    :param hyperparameters: dict
    """
    train_indices = list(df.sample(frac=1).index)
    k_folds = np.array_split(train_indices, K)
    if K == 1:
        K = 2

    rmse_list = []
    for i in range(len(k_folds)):
        training_folds = [fold for j, fold in enumerate(k_folds) if j != i]
        training_indices = np.concatenate(training_folds)
        x_train, y_train = df.iloc[training_indices, 1:], df.iloc[training_indices, :1]
        x_validation, y_validation = df.iloc[k_folds[i], 1:], df.iloc[k_folds[i], :1]
        
        model = CatBoostRegressor(**hyperparameters)
        model.fit(x_train, y_train, eval_set=(x_validation, y_validation), logging_level='Silent')
        

        eval_results = model.predict(x_validation)
        rms = mean_squared_error(y_validation, eval_results, squared=True)
        print(f'[{i}]\tvalidation-rmse:{rms}')
        rmse_list.append(rms)
    return rmse_list, model



