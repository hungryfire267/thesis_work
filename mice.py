import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import chi2
from simulation import get_ci_width, get_coverage_prob, summary_type
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import time


def data_dropper(data, variable): 
    y = data[variable]
    X = data.drop([variable], axis=1)
    return X, y

def get_parameters(X, y, n, p): 
    V = np.linalg.inv(X.T @ X)
    beta_1 = (V @ X.T @ y).to_numpy()
    sigma_squared_hat = np.sum((y - beta_1 @ X.T) ** 2)/ (n - p)
    L = np.linalg.cholesky(V)
    return beta_1, sigma_squared_hat, L

def get_model(X_complete, y_complete, type): 
    model_regression = None
    if (type == "dt"): 
        dt_parameters = {
            "max_depth": [i for i in range(5, 20)] + [None], 
            "min_samples_split": [2 ** (i) for i in range(1,9)],
            "min_samples_leaf": [2 ** (i) for i in range(9)],
            "max_leaf_nodes": [2 ** (i) for i in range(1,9)] + [None]
        }
        model = DecisionTreeRegressor(random_state=5) 
        model_regression = RandomizedSearchCV(
            estimator=model, param_distributions=dt_parameters, cv=10,
            scoring="neg_mean_squared_error", random_state=12
        )
        model_regression.fit(X_complete, y_complete)
    elif (type == "lgbm"): 
        lgb_parameters = {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5], 
            "n_estimators": [50, 100, 150, 200], 
            "num_leaves": [31, 63, 127, 255],
            "min_child_samples": [5, 10, 20, 50, 100],
            "max_depth": [i for i in range(5, 20)] + [-1],
            "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
            "reg_lambda": [0, 0.01, 0.1, 0.5, 1]
        }
    
        model = lgb.LGBMRegressor(n_jobs=-1, verbose=-1, random_state=5)
        model_regression = RandomizedSearchCV( 
            estimator=model, param_distributions=lgb_parameters, cv=10, 
            scoring="neg_mean_squared_error", random_state=12
        )
        model_regression.fit(X_complete, y_complete)
    return model_regression
        


def single_blr(X_complete, y_complete, X_missing, X_complete_pmm, z_complete, z_missing): 
    n_complete, p = X_complete.shape
    n_missing = X_missing.shape[0]
    n_full = n_complete + n_missing
    complete_beta, complete_variance, L = get_parameters(X_complete, y_complete, n_full, p)
    g = chi2.rvs(n_complete - p, size = 1)
    Z = np.random.normal(0, 1, p)
    sigma_squared_star = complete_variance * (n_full - p)/g[0]
    beta_star = complete_beta + np.sqrt(sigma_squared_star) * L @ Z
    imputed_values = beta_star @ X_missing.T + z_missing * np.sqrt(sigma_squared_star)
    complete_values = beta_star @ X_complete_pmm.T + z_complete * np.sqrt(sigma_squared_star)
    return imputed_values, complete_values

def get_sigma_squared_ml(model, X_complete, y_complete): 
    kf = KFold(n_splits = 5, shuffle=True)
    sigma_squared_hat_list = []
    for train_index, test_index in kf.split(X_complete): 
        X_train, X_test =  X_complete.iloc[train_index], X_complete.iloc[test_index]
        y_train, y_test = y_complete.iloc[train_index], y_complete.iloc[test_index]
        y_pred_test = model.predict(X_test)
        sigma_squared_hat = np.var(y_test - y_pred_test)
        sigma_squared_hat_list.append(sigma_squared_hat)
    return np.mean(sigma_squared_hat_list)

def single_dt(X_complete, y_complete, X_missing, z_complete, z_missing, parameters_dict): 
    model = DecisionTreeRegressor(
        min_samples_split=parameters_dict["min_samples_split"],
        min_samples_leaf=parameters_dict["min_samples_leaf"], 
        max_leaf_nodes=parameters_dict["max_leaf_nodes"], 
        max_depth=parameters_dict["max_depth"], 
    )
    model.fit(X_complete, y_complete)
    sigma_squared_hat = get_sigma_squared_ml(model, X_complete, y_complete)
    
    imputed_noise = z_missing * np.sqrt(sigma_squared_hat)
    imputed_values = model.predict(X_missing) + imputed_noise
    imputed_values_df = pd.Series(imputed_values, index=X_missing.index)
    
    complete_noise = z_complete * np.sqrt(sigma_squared_hat)
    complete_values = model.predict(X_complete) + complete_noise
    complete_values_df = pd.Series(complete_values, index = X_complete.index)
    
    return imputed_values_df, complete_values_df

def single_lgb(X_complete, y_complete, X_missing, z_complete, z_missing, parameters_dict): 
    model = lgb.LGBMRegressor(
        reg_lambda=parameters_dict["reg_lambda"],
        reg_alpha=parameters_dict["reg_alpha"],
        num_leaves=parameters_dict["num_leaves"],
        n_estimators=parameters_dict["n_estimators"], 
        min_child_samples=parameters_dict["min_child_samples"],
        max_depth=parameters_dict["max_depth"],
        learning_rate=parameters_dict["learning_rate"]
    )
    model.fit(X_complete, y_complete)
    
    sigma_squared_hat = get_sigma_squared_ml(model, X_complete, y_complete)
    
    imputed_noise = z_missing * np.sqrt(sigma_squared_hat)
    imputed_values = model.predict(X_missing) + imputed_noise
    imputed_values_df = pd.Series(imputed_values, index=X_missing.index)
    
    complete_noise = z_complete * np.sqrt(sigma_squared_hat)
    complete_values = model.predict(X_complete) + complete_noise
    complete_values_df = pd.Series(complete_values, index=X_complete.index)
    return imputed_values_df, complete_values_df
    
    
def get_single_data(data, framework, model_type, model):
    imp_data = data.copy()
    miss_indices = list(imp_data[imp_data["x1"].isnull()].index)
    n_missing = len(miss_indices)
    n_complete = data.shape[0] - n_missing
    z_missing = np.random.normal(0, 1, n_missing)
    z_complete = np.random.normal(0, 1, n_complete)
    for iteration in range(8): 
        if (iteration == 1 and (model_type == "dt" or model_type == "lgbm")): 
            break

        data_complete = imp_data[imp_data["x1"].notnull()].copy()
        data_missing = imp_data.loc[miss_indices, :].copy()
        X_complete, y_complete = data_dropper(data_complete, "x1")
        
        X_missing, _ = data_dropper(data_missing, "x1")
        if (model_type == "blr"): 
            X_complete["intercept"] = 1
            X_missing["intercept"] = 1
            
            if (iteration == 0): 
                X_complete_pmm = X_complete.copy()
                
            imputed_values, complete_values = single_blr(
                X_complete, y_complete, X_missing, X_complete_pmm, z_complete, z_missing
            )
        elif (model_type == "dt"): 
            parameters_dict = model.best_params_
            imputed_values, complete_values = single_dt(
                X_complete, y_complete, X_missing, z_complete, z_missing, parameters_dict
            )
        else:
            parameters_dict = model.best_params_
            imputed_values, complete_values = single_lgb(
                X_complete, y_complete, X_missing, z_complete, z_missing, parameters_dict
            )
            
        if (framework == "regression"):
            imp_data.loc[miss_indices, "x1"] = imputed_values
        elif (framework == "pmm3"):
            neighbours = 3
            imputed_values = pmm(neighbours, complete_values, imputed_values)
            imp_data.loc[miss_indices, "x1"] = imputed_values
        elif (framework == "pmm5"): 
            neighbours = 5
            imputed_values = pmm(neighbours, complete_values, imputed_values)
            imp_data.loc[miss_indices, "x1"] = imputed_values
        elif (framework == "pmm10"):
            neighbours = 10
            imputed_values = pmm(neighbours, complete_values, imputed_values)
            imp_data.loc[miss_indices, "x1"] = imputed_values
    X_final, y_final = data_dropper(imp_data, "y")
    return X_final, y_final

def rubin_rule(estimates, variances, num_imputations): 
    avg_estimate = np.mean(estimates, axis=0)
    avg_variance = np.mean(variances, axis=0)
    between_variance = np.var(estimates, axis=0, ddof=1)
    total_variance = avg_variance + (1 + 1/num_imputations) * between_variance
    return avg_estimate, total_variance


def get_MICE_estimates(X, y, num_imputations, framework, model_type, linearity, model): 
    data = pd.concat([X, y], axis=1)
    estimates, variances = [], []
    for imputation in range(num_imputations): 
        X_final, y_final = get_single_data(data, framework, model_type, model)
        
        if (linearity == False): 
            X_final["x1_squared"] = np.power(X_final["x1"], 2)
            X_final = X_final[["x1", "x1_squared", "x2", "x3"]]
        
        X_final = sm.add_constant(X_final)
        
        linear_model = sm.OLS(y_final, X_final).fit()
        
        y_pred = X_final @ linear_model.params.values
        
        estimates.append(linear_model.params.values)
        variances.append(np.power(linear_model.bse.values, 2)) 
    avg_estimate, total_variance = rubin_rule(estimates, variances, num_imputations)
    return avg_estimate, total_variance, y_pred

def generate_mice_results(
    X_list, X_true_list, y_list, num_imputations, framework, model_type, full_beta, linearity
):
    ci_list, width_list, mse_pred_list = [], [], []
    
    estimates_list = []
    i = 0
    for X, X_true, y in zip(X_list, X_true_list, y_list): 
        if (i == 0): 
            data = pd.concat([X, y], axis=1)
            data_complete = data[data["x1"].notnull()].copy()
            X_complete, y_complete = data_dropper(data_complete, "x1")
            model = get_model(X_complete, y_complete, model_type)
        else:
            estimate, variance, y_pred = get_MICE_estimates(
                X, y, num_imputations, framework, model_type, linearity, model
            )   
            estimates_list.append(estimate)
            ci_lb, ci_ub, width = get_ci_width(estimate, variance)
            ci_list.append([ci_lb, ci_ub])
            width_list.append(width)
            
            y_actual = X_true @ full_beta
            mse_pred = mean_squared_error(y_actual, y_pred)
            mse_pred_list.append(mse_pred)
        i += 1
    cov_prob = get_coverage_prob(ci_list, full_beta)
    avg_beta = np.mean(estimates_list, axis=0)
    bias = avg_beta - full_beta
    variance = np.var(estimates_list, axis=0, ddof=1)
    squared_errors = (estimates_list - full_beta) ** 2
    mse = np.mean(squared_errors, axis=0)
    mse_pred = np.mean(mse_pred_list)
    
    return bias, variance, mse, cov_prob, width_list, mse_pred

def pmm(neighbours, complete_values, imputed_values):
    indices = list(imputed_values.index)
    for index in indices:
        distances = np.abs(complete_values - imputed_values[index]).sort_values()
        sorted_indices = distances[:neighbours].index
        optimal_index = np.random.choice(sorted_indices, 1, replace=False)[0]
        imputed_values.loc[index] = complete_values.loc[optimal_index]
    return imputed_values

def get_MICE_results_df(X_list, X_true_list, y_list, beta, framework, linearity): 
    if (framework == "regression"): 
        framework_type = "MICE"
    elif (framework == "pmm3"): 
        framework_type = "PMM3"
    elif (framework == "pmm5"): 
        framework_type = "PMM5"
    elif (framework == "pmm10"): 
        framework_type = "PMM10"
    
    print(f"Imputing {framework_type} models")
    
    print(f"Imputing {framework_type}-BLR")
    start_blr = time.time()
    bias_blr, var_blr, mse_blr, cov_blr, _, mse_reg_blr = generate_mice_results(
        X_list, X_true_list, y_list, 10, framework, "blr", beta, linearity
    )
    finish_blr = time.time()
    total_time_blr = finish_blr - start_blr
    
    print(f"Imputing {framework_type}-DT")
    start_dt = time.time() 
    bias_dt, var_dt, mse_dt, cov_dt, _, mse_reg_dt = generate_mice_results( 
        X_list, X_true_list, y_list, 10, framework, "dt", beta, linearity
    )
    finish_dt = time.time()
    total_time_dt = finish_dt - start_dt 
    
    print(f"Imputing {framework_type}-LGB")
    start_lgb = time.time()
    bias_lgb, var_lgb, mse_lgb, cov_lgb, _, mse_reg_lgb = generate_mice_results(
        X_list, X_true_list, y_list, 10, framework, "lgbm", beta, linearity
    )
    finish_lgb = time.time() 
    total_time_lgb = finish_lgb - start_lgb
    
    print("Creating results dataframe")
    mice_blr_df = summary_type(
        f"{framework_type}-BLR", bias_blr, var_blr, mse_blr, cov_blr
    )
    mice_dt_df = summary_type(
        f"{framework_type}-DT", bias_dt, var_dt, mse_dt, cov_dt
    )
    mice_lgb_df = summary_type(
        f"{framework_type}-LGBM", bias_lgb, var_lgb, mse_lgb, cov_lgb
    )

    
    df_list = [
        mice_blr_df, mice_dt_df, mice_lgb_df
    ]
    mse_list = [mse_reg_blr, mse_reg_dt, mse_reg_lgb]
    total_time_list = [
        total_time_blr, total_time_dt, total_time_lgb
    ]
    
    return df_list, mse_list, total_time_list