"""
Ridge Regression with GroupKFold Cross-Validation
Implementation for ADA Project - Formula 1 Lap Time Prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def make_Xy(df, features, categorical, target, fit_cols=None, medians=None):
    """
    Build feature matrix X and target vector y from a dataframe.
    
    Handles numerical and categorical features separately:
    - Numerical features: Fill NaN with median (computed on train, applied to val/test)
    - Categorical features: One-hot encode
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature column names (numerical)
    categorical : list
        List of categorical feature column names
    target : str
        Target variable column name
    fit_cols : list, optional
        Column names from training set (for validation/test sets)
    medians : dict, optional
        Median values from training set (for validation/test sets)
        
    Returns:
    --------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    cols : list
        Column names (for validation/test alignment)
    meds : dict
        Median values (for validation/test imputation)
    """
    # Make a copy to avoid modifying original dataframe
    df = df.copy()
    
    # Create is_new_tyre feature if TyreAgeAtStart exists but is_new_tyre doesn't
    if 'is_new_tyre' in features and 'is_new_tyre' not in df.columns:
        if 'TyreAgeAtStart' in df.columns:
            df['is_new_tyre'] = (df['TyreAgeAtStart'] == 0).astype(int)
        else:
            # If neither exists, create a dummy feature (all zeros)
            df['is_new_tyre'] = 0
    
    # Extract target
    y = df[target].values
    
    # Filter features to only those that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    # Numerical features
    X_num = df[available_features].copy()
    
    # Compute or use provided medians
    if medians is None:
        medians = X_num.median().to_dict()
    
    # Fill NaN with medians
    X_num = X_num.fillna(medians)
    
    # Categorical features (one-hot encoding)
    if categorical:
        X_cat = pd.get_dummies(df[categorical], drop_first=True, dtype=float)
        
        # Align columns for validation/test sets
        if fit_cols is not None:
            # Identify categorical columns from training set (not in numerical features)
            train_cat_cols = [c for c in fit_cols if c not in available_features]
            
            # Add missing categorical columns with zeros
            for col in train_cat_cols:
                if col not in X_cat.columns:
                    X_cat[col] = 0.0
            
            # Select only the categorical columns that were in training
            # (in case validation/test has new categories)
            X_cat = X_cat[[c for c in train_cat_cols if c in X_cat.columns]]
        
        # Combine numerical and categorical
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_num
    
    # Store column names
    cols = X.columns.tolist() if fit_cols is None else fit_cols
    
    # Ensure column alignment for validation/test
    if fit_cols is not None:
        # Add missing columns with zeros
        for col in fit_cols:
            if col not in X.columns:
                X[col] = 0.0
        # Reorder and select only training columns
        X = X[fit_cols]
    
    return X.values, y, cols, medians


def cv_ridge_groupkfold(df, features, categorical, target, group_col='name', 
                        alphas=[0.1, 1, 10, 100], n_splits=5):
    """
    Perform GroupKFold cross-validation for Ridge regression.
    
    Groups data by a grouping column (e.g., circuit name) to prevent data leakage.
    Tests multiple alpha (regularization) values and returns the best one.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Training dataframe
    features : list
        List of numerical feature column names
    categorical : list
        List of categorical feature column names
    target : str
        Target variable column name
    group_col : str, default='name'
        Column name to group by (e.g., circuit name)
    alphas : list, default=[0.1, 1, 10, 100]
        List of alpha values to test
    n_splits : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    best_alpha : float
        Best alpha value (lowest CV error)
    results_df : pd.DataFrame
        DataFrame with CV results for each alpha
    cols : list
        Feature column names (for final model fitting)
    meds : dict
        Median values for imputation (for final model fitting)
    """
    # Extract groups
    groups = df[group_col].values
    
    # Initialize GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Store results
    results = []
    
    # Validate inputs
    if not alphas:
        raise ValueError("alphas list cannot be empty")
    
    print(f"\nRunning {n_splits}-fold cross-validation with GroupKFold (grouped by '{group_col}')...")
    print(f"Testing {len(alphas)} alpha values: {alphas}\n")
    
    # Track best alpha (initialize with first alpha)
    best_alpha = alphas[0]
    best_score = float('inf')
    
    # For each alpha value
    for alpha in alphas:
        fold_scores = []
        
        # For each fold
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
            # Split data
            df_fold_train = df.iloc[train_idx]
            df_fold_val = df.iloc[val_idx]
            
            # Build feature matrices
            X_train, y_train, cols, meds = make_Xy(
                df_fold_train, features, categorical, target
            )
            X_val, y_val, _, _ = make_Xy(
                df_fold_val, features, categorical, target, 
                fit_cols=cols, medians=meds
            )
            
            # Train Ridge model
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            fold_scores.append({'mae': mae, 'rmse': rmse})
        
        # Compute mean scores across folds
        mean_mae = np.mean([s['mae'] for s in fold_scores])
        mean_rmse = np.mean([s['rmse'] for s in fold_scores])
        std_mae = np.std([s['mae'] for s in fold_scores])
        
        results.append({
            'alpha': alpha,
            'mean_mae': mean_mae,
            'std_mae': std_mae,
            'mean_rmse': mean_rmse
        })
        
        print(f"  Alpha {alpha:8.1f}: MAE = {mean_mae:.4f} ± {std_mae:.4f}  |  RMSE = {mean_rmse:.4f}")
        
        # Track best alpha
        if mean_mae < best_score:
            best_score = mean_mae
            best_alpha = alpha
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Build final feature matrix for returning column names and medians
    X_full, y_full, cols, meds = make_Xy(df, features, categorical, target)
    
    print(f"\n✓ Best alpha: {best_alpha} (MAE = {best_score:.4f})")
    
    return best_alpha, results_df, cols, meds


def metrics(y_true, y_pred):
    """
    Compute regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    dict
        Dictionary with MAE, RMSE, and R² metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
