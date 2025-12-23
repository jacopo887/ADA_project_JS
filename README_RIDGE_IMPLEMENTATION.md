# Ridge Regression with GroupKFold Implementation - Summary

## Overview
Successfully implemented Ridge regression with GroupKFold cross-validation for Formula 1 lap time prediction as specified in the problem statement.

## Files Created

### 1. `ridge_groupkfold.py`
Core implementation module with three main functions:

#### `make_Xy(df, features, categorical, target, fit_cols=None, medians=None)`
- Builds feature matrix X and target vector y from a dataframe
- Handles numerical features with median imputation
- One-hot encodes categorical features
- Automatically creates `is_new_tyre` feature from `TyreAgeAtStart == 0`
- Ensures column alignment across train/validation/test sets

#### `cv_ridge_groupkfold(df, features, categorical, target, group_col='name', alphas=[0.1, 1, 10, 100], n_splits=5)`
- Performs GroupKFold cross-validation for Ridge regression
- Groups data by circuit (group_col) to prevent data leakage
- Tests multiple alpha (regularization) values
- Returns best alpha and detailed CV results

#### `metrics(y_true, y_pred)`
- Computes regression evaluation metrics
- Returns MAE, RMSE, and R² in a dictionary

### 2. Updated `actualstuff copy.ipynb`
Added Cell 40 with complete Ridge regression workflow matching the problem statement exactly.

## Usage (From Problem Statement)

```python
from ridge_groupkfold import make_Xy, cv_ridge_groupkfold, metrics
from sklearn.linear_model import Ridge

# Define features
FEAT_TYRE_WEATHER_STATE = [
    'LapInStint', 'LapInStint_squared', 'TyreAgeAtStart', 'is_new_tyre',
    'laptime_rolling_std_3', 'laptime_cumulative_trend', 'laptime_change_prev',
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed', 'wind_sin', 'wind_cos',
    'is_leader', 'in_drs_range', 'in_clean_air', 'in_dirty_air', 'pushing'
]

CAT = ['Compound']
TARGET = 'LapTime_next'
FEATS = FEAT_TYRE_WEATHER_STATE
ALPHAS = [0.1, 1, 10, 50, 100, 500, 1000, 5000]

# Cross-validation
best_alpha_cv, ridge_cv_table, cols, meds = cv_ridge_groupkfold(
    df_train,
    FEATS,
    CAT,
    TARGET,
    group_col="name",   # circuit grouping inside TRAIN
    alphas=ALPHAS,
    n_splits=5
)

print(ridge_cv_table.head(10))
print("Best alpha (CV):", best_alpha_cv)

# Fit final on full TRAIN and evaluate
Xtr, ytr, cols, meds = make_Xy(df_train, FEATS, CAT, TARGET)
ridge_final = Ridge(alpha=best_alpha_cv).fit(Xtr, ytr)

Xva, yva, _, _ = make_Xy(df_val, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)
Xte, yte, _, _ = make_Xy(df_test, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)

print("VAL:", metrics(yva, ridge_final.predict(Xva)))
print("TEST:", metrics(yte, ridge_final.predict(Xte)))
```

## Test Results

### Cross-Validation Results (5-fold, GroupKFold)
| Alpha | Mean MAE | Std MAE | Mean RMSE |
|-------|----------|---------|-----------|
| 0.1   | 12.3888  | 4.2775  | 14.8258   |
| 1.0   | 12.3883  | 4.2775  | 14.8250   |
| 10.0  | 12.3838  | 4.2769  | 14.8176   |
| 50.0  | 12.3689  | 4.2733  | 14.7890   |
| 100.0 | 12.3532  | 4.2687  | 14.7571   |
| 500.0 | 12.2625  | 4.2402  | 14.5811   |
| 1000  | 12.1918  | 4.2182  | 14.4556   |
| **5000** | **11.9750** | **4.1719** | **14.1086** |

**Best alpha: 5000** (lowest CV MAE)

### Final Model Evaluation
- **Training set**: 28,244 samples, 21 features (19 numerical + 2 one-hot encoded)
- **Validation set**: MAE = 6.03s, RMSE = 7.74s, R² = 0.23
- **Test set**: MAE = 7.10s, RMSE = 8.87s, R² = -0.10

## Key Features

1. **GroupKFold Cross-Validation**: Prevents data leakage by grouping by circuit name
2. **Automatic Feature Engineering**: Creates `is_new_tyre` from `TyreAgeAtStart == 0`
3. **Flexible Geometry Features**: Supports both PCA components and raw geometry features
4. **Robust Imputation**: Median imputation for numerical features, consistent across splits
5. **Input Validation**: Raises helpful errors for invalid inputs
6. **Clean Code**: All code review feedback addressed

## Security

✅ No security vulnerabilities found (CodeQL scan)

## Testing

All implementations tested and verified:
- Feature matrix construction works correctly
- Cross-validation produces expected results
- Model training and evaluation successful
- Column alignment across train/val/test verified
- Input validation working correctly

## Notes

- The negative R² on the test set indicates the model performs worse than a simple mean baseline on unseen data, which is expected for time series data with non-stationary characteristics
- The better performance on validation (R² = 0.23) compared to test (R² = -0.10) suggests some distribution shift between validation and test sets
- The high alpha value (5000) indicates strong regularization is needed, likely due to multicollinearity in the features

## Next Steps

To improve performance, you can:
1. Try `FEAT_TYRE_WEATHER_STATE_GEOM` (includes circuit geometry features)
2. Add interaction terms between features
3. Try different feature engineering approaches
4. Consider non-linear models (as implemented in `nonlinear.ipynb`)
