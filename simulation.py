import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import statsmodels.api as sm

def get_mean_cov(): 
    mean = [3, 2, 8, 6]
    A = np.array([
            [1.2, 3.4, 2.8, 0.6], 
            [1.6, 2.4, 3.8, 0.8],
            [2.9, 3.2, 1.7, 1.4], 
            [2.4, 1.3, 2.8, 1.7]
    ])
    cov = A.T @ A
    return mean, cov

def get_beta(mean, cov, linearity):
    mean_y = mean[0]
    mean_x = mean[1:]
    
    sigma_xx = cov[1:, 1:]
    sigma_xy = cov[1:, 0]

    beta = np.linalg.inv(sigma_xx) @ sigma_xy
    beta_0 = mean_y - beta @ mean_x
    
    if (linearity == True): 
        full_beta = np.hstack([beta_0, beta])
    else: 
        full_beta = np.hstack([beta_0, beta[0], 0.01, beta[1:]])
    return full_beta

def transform_X(X_true, linearity): 
    X_data = X_true.copy()
    if (linearity == False): 
        X_data["x1_squared"] = np.power(X_data["x1"], 2)
        X_data = X_data[["x1", "x1_squared", "x2", "x3"]]
    X_true_final = sm.add_constant(X_data)
    return X_true_final

def get_data(mean, cov, sample_size, miss_prop, seed, linearity): 
    np.random.seed(seed)
    X = np.random.multivariate_normal(mean, cov, sample_size)
    data = pd.DataFrame(X, columns=["y", "x1", "x2", "x3"])
    true_data = data.copy()
    if (linearity == False): 
        data["y"] = 0.01 * np.power(data["x1"], 2) + data["y"]
    miss_size = int(miss_prop * sample_size)
    
    prob_missing = 1 / (1 + np.exp(-data["x2"]))
    missing_indices = np.random.choice(
        data.index, size=miss_size, replace=False, p=prob_missing/prob_missing.sum()
    )
    data.loc[missing_indices, "x1"] = np.nan
    return data, true_data

def generate_datasets(num_datasets, sample_size, miss_prop, framework, linearity): 
    X_list, X_true_list, y_list, scale_list = [], [], [], []
    for seed in range(1, num_datasets + 1): 
        mean, cov = get_mean_cov()
        data, true_data = get_data(mean, cov, sample_size, miss_prop, seed, linearity)
        
        y = data["y"]
        X = data.drop(["y"], axis=1)
        X_true = true_data.drop(["y"], axis=1).copy()
        
        if (linearity == False): 
            X_true["x1_squared"] = np.power(X_true["x1"], 2)
            X_true = X_true[["x1", "x1_squared", "x2", "x3"]]
        X_true_final = sm.add_constant(X_true)
        X_true_list.append(X_true_final)
        
        if (framework == "KNN"):
            X_scaled, scaler = scale_datasets(X)
            X_list.append(X_scaled)
            scale_list.append(scaler)
        else: 
            X_list.append(X)
        y_list.append(y)
    return X_list, X_true_list, y_list, scale_list

def scale_datasets(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=["x1", "x2", "x3"]
    )
    return X_scaled, scaler


def get_ci_width(estimates, variances): 
    ci_lb = estimates - 1.96 * np.sqrt(variances)
    ci_ub = estimates + 1.96 * np.sqrt(variances)
    width = ci_ub - ci_lb
    return ci_lb, ci_ub, width[1]

def get_coverage_prob(ci_list, full_beta): 
    num_datasets = len(ci_list)
    count = np.zeros(len(full_beta))
    for ci in ci_list: 
        ci_lb = ci[0]
        ci_ub = ci[1]
        for i in range(len(full_beta)): 
            if (full_beta[i] >= ci_lb[i] and full_beta[i] <= ci_ub[i]): 
                count[i] += 1
    return count/num_datasets

def summary_type(model_type, avg_bias, avg_var, avg_mse, cov):
    model_list, beta_list, stat_list, score_list = [], [], [], []
    stats = ["Bias", "Var", "MSE", "Cov Prob"]
    for i, (bias, variance, mse, cov_prob) in enumerate(zip(avg_bias, avg_var, avg_mse, cov)):
        model_list.extend([model_type] * 4)
        beta_list.extend([f"Beta_{i}"] * 4)
        stat_list.extend(stats)
        score_list.extend([bias, variance, mse, cov_prob])

    # Create the DataFrame
    new_dict = { 
        "Type": model_list,
        "Beta": beta_list, 
        "Stat": stat_list,
        "Score": score_list        
    }
    new_df = pd.DataFrame(new_dict)
    return new_df

def summary_data(combined_df, linearity): 
    overall_df = pd.concat(combined_df, axis=0).reset_index()
    overall_df = overall_df.drop(["index"], axis=1)

    summary_df = overall_df.pivot_table(
        index='Type', columns=['Beta', 'Stat'], values='Score'
    )
    if (linearity == True):
        summary_df = summary_df.reindex(
            columns=pd.MultiIndex.from_product(
                [['Beta_0', 'Beta_1', 'Beta_2', 'Beta_3'], ['Bias', 'Var', 'MSE', "Cov Prob"]]
            )
        )
    else: 
        summary_df = summary_df.reindex(
            columns=pd.MultiIndex.from_product(
                [['Beta_0', 'Beta_1', 'Beta_2', 'Beta_3', 'Beta_4'], ['Bias', 'Var', 'MSE', "Cov Prob"]]
            )
        )
    return summary_df


