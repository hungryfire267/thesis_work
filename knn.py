import numpy as np 
import pandas as pd
from simulation import get_ci_width, get_coverage_prob, summary_type
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import time

def knn_impute_dataset(X, neighbour): 
    imputer = KNNImputer(n_neighbors=neighbour, weights="distance")
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(
        X_imputed, columns=["x1", "x2", "x3"]
    )
    return X_imputed_df

def get_KNN_estimates(X, y, scaler, k, linearity): 
    X_imputed = knn_impute_dataset(X, k)
    X_imputed_inverse = pd.DataFrame(
        scaler.inverse_transform(X_imputed), columns=["x1", "x2", "x3"]
    )
    
    if (linearity == False): 
        X_imputed_inverse["x1_squared"] = np.power(X_imputed_inverse["x1"], 2)
        X_imputed_inverse = X_imputed_inverse[["x1", "x1_squared", "x2", "x3"]]
    
    X_imputed_final = sm.add_constant(X_imputed_inverse)
    linear_model = sm.OLS(y, X_imputed_final).fit() 
    estimates = linear_model.params.values
    variances = np.power(linear_model.bse.values, 2)
    y_pred = X_imputed_final @ estimates
    return estimates, variances, y_pred

def generate_KNN_results(X_list, X_true_list, y_list, scaler_list, k, full_beta, linearity):
    ci_list, width_list, mse_pred_list = [], [], []
    
    estimates_list = []
    i = 0
    for X, X_true, y, scaler in zip(X_list, X_true_list, y_list, scaler_list): 
        if (i == 0): 
            i += 1
            continue
        estimate, variance, y_pred = get_KNN_estimates(X, y, scaler, k, linearity)
        estimates_list.append(estimate)
        ci_lb, ci_ub, width = get_ci_width(estimate, variance)
        ci_list.append([ci_lb, ci_ub])
        width_list.append(width)
        
        y_actual = X_true @ full_beta
        mse_pred = mean_squared_error(y_actual, y_pred)
        mse_pred_list.append(mse_pred)
        
    cov_prob = get_coverage_prob(ci_list, full_beta)
    avg_beta = np.mean(estimates_list, axis=0)
    bias = avg_beta - full_beta
    variance = np.var(estimates_list, axis=0, ddof=1)
    squared_errors = (estimates_list - full_beta) ** 2
    mse = np.mean(squared_errors, axis=0)
    mse_pred = np.mean(mse_pred_list)
    
    return bias, variance, mse, cov_prob, width_list, mse_pred
    
    
def get_KNN_results_df(X_list, X_true_list, y_list, scaler_list, beta, linearity):
    print("Imputing WKNN models")
    
    print("Imputing WKNN-5")
    start_knn5 = time.time()    
    bias_knn5, var_knn5, mse_knn5, cov_knn5, _, mse_reg_knn5 = generate_KNN_results(
        X_list, X_true_list, y_list, scaler_list, 5, beta, linearity
    )
    end_knn5 = time.time() 
    time_knn5 = end_knn5 - start_knn5

    print("Imputing WKNN-10")
    start_knn10 = time.time()
    bias_knn10, var_knn10, mse_knn10, cov_knn10, _, mse_reg_knn10 = generate_KNN_results(
        X_list, X_true_list, y_list, scaler_list, 10, beta, linearity
    )
    end_knn10 = time.time()
    time_knn10 = end_knn10 - start_knn10

    print("Imputing WKNN-20")
    start_knn20 = time.time()
    bias_knn20, var_knn20, mse_knn20, cov_knn20, _, mse_reg_knn20 = generate_KNN_results(
       X_list, X_true_list, y_list, scaler_list, 20, beta, linearity
    )
    end_knn20 = time.time() 
    time_knn20 = end_knn20 - start_knn20

    print("Creating results dataframe")
    knn5_df = summary_type("WKNN-5", bias_knn5, var_knn5, mse_knn5, cov_knn5)
    knn10_df = summary_type("WKNN-10", bias_knn10, var_knn10, mse_knn10, cov_knn10)
    knn20_df = summary_type("WKNN-20", bias_knn20, var_knn20, mse_knn20, cov_knn20)
    
    df_list = [knn5_df, knn10_df, knn20_df]
    mse_list = [mse_reg_knn5, mse_reg_knn10, mse_reg_knn20] 
    total_time_list = [time_knn5, time_knn10, time_knn20]
    
    return df_list, mse_list, total_time_list