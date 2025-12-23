"""
Demonstration of Ridge Regression with GroupKFold Implementation
This script demonstrates the exact usage pattern from the problem statement
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ridge_groupkfold import make_Xy, cv_ridge_groupkfold, metrics
from sklearn.linear_model import Ridge

print("="*100)
print("RIDGE REGRESSION WITH GROUPKFOLD - DEMONSTRATION")
print("="*100)

# Load datasets
print("\n1. Loading datasets...")
df_train = pd.read_excel('csv_output/Train_set.xlsx')
df_val = pd.read_excel('csv_output/Validation_set.xlsx')
df_test = pd.read_excel('csv_output/Test_set.xlsx')

print(f"   ✓ Train: {df_train.shape[0]:,} samples")
print(f"   ✓ Validation: {df_val.shape[0]:,} samples")
print(f"   ✓ Test: {df_test.shape[0]:,} samples")

# Define features exactly as in problem statement
print("\n2. Defining feature sets...")

FEAT_TYRE_WEATHER_STATE = [
    # Tyre and stint features
    'LapInStint', 'LapInStint_squared', 'TyreAgeAtStart', 'is_new_tyre',
    
    # Temporal/degradation features
    'laptime_rolling_std_3', 'laptime_cumulative_trend', 'laptime_change_prev',
    
    # Weather features
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed', 'wind_sin', 'wind_cos',
    
    # Race state features
    'is_leader', 'in_drs_range', 'in_clean_air', 'in_dirty_air', 'pushing'
]

CAT = ['Compound']
TARGET = 'LapTime_next'
FEATS = FEAT_TYRE_WEATHER_STATE  # start here

print(f"   ✓ Features: {len(FEATS)} numerical + {len(CAT)} categorical")
print(f"   ✓ Target: {TARGET}")

# Cross-validation - EXACT pattern from problem statement
print("\n3. Running cross-validation (from problem statement)...")
print("   Code:")
print("   ALPHAS = [0.1, 1, 10, 50, 100, 500, 1000, 5000]")
print("   best_alpha_cv, ridge_cv_table, cols, meds = cv_ridge_groupkfold(")
print("       df_train, FEATS, CAT, TARGET,")
print("       group_col='name', alphas=ALPHAS, n_splits=5")
print("   )\n")

ALPHAS = [0.1, 1, 10, 50, 100, 500, 1000, 5000]

best_alpha_cv, ridge_cv_table, cols, meds = cv_ridge_groupkfold(
    df_train,
    FEATS,
    CAT,
    TARGET,
    group_col="name",   # circuit grouping inside TRAIN
    alphas=ALPHAS,
    n_splits=5
)

print("\n   Results:")
print(ridge_cv_table.head(10))
print(f"\nBest alpha (CV): {best_alpha_cv}")

# Train final model - EXACT pattern from problem statement
print("\n" + "="*100)
print("4. Fitting final model on full TRAIN (from problem statement)...")
print("   Code:")
print("   Xtr, ytr, cols, meds = make_Xy(df_train, FEATS, CAT, TARGET)")
print("   ridge_final = Ridge(alpha=best_alpha_cv).fit(Xtr, ytr)\n")

# Fit final on full TRAIN and evaluate
Xtr, ytr, cols, meds = make_Xy(df_train, FEATS, CAT, TARGET)
ridge_final = Ridge(alpha=best_alpha_cv).fit(Xtr, ytr)

print(f"   ✓ Model trained: {len(ytr):,} samples, {len(cols)} features")

# Evaluate - EXACT pattern from problem statement
print("\n" + "="*100)
print("5. Evaluation (from problem statement)...")
print("   Code:")
print("   Xva, yva, _, _ = make_Xy(df_val, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)")
print("   Xte, yte, _, _ = make_Xy(df_test, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)")
print('   print("VAL:", metrics(yva, ridge_final.predict(Xva)))')
print('   print("TEST:", metrics(yte, ridge_final.predict(Xte)))\n')

Xva, yva, _, _ = make_Xy(df_val, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)
Xte, yte, _, _ = make_Xy(df_test, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)

print("   Output:")
print("VAL:", metrics(yva, ridge_final.predict(Xva)))
print("TEST:", metrics(yte, ridge_final.predict(Xte)))

print("\n" + "="*100)
print("✅ DEMONSTRATION COMPLETE")
print("="*100)
print("\nThe implementation works exactly as specified in the problem statement!")
print("You can now use the same code in your Jupyter notebook.")
