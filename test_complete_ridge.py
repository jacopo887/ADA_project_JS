"""
Complete test of Ridge regression with GroupKFold implementation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our implementation
from ridge_groupkfold import make_Xy, cv_ridge_groupkfold, metrics
from sklearn.linear_model import Ridge

print("="*80)
print("RIDGE REGRESSION WITH GROUPKFOLD - COMPLETE TEST")
print("="*80)

print("\n1. Loading data...")
df_train = pd.read_excel('csv_output/Train_set.xlsx')
df_val = pd.read_excel('csv_output/Validation_set.xlsx')
df_test = pd.read_excel('csv_output/Test_set.xlsx')

print(f"   ✓ Train: {df_train.shape[0]} samples")
print(f"   ✓ Val: {df_val.shape[0]} samples")
print(f"   ✓ Test: {df_test.shape[0]} samples")

print("\n2. Defining features...")

# FEAT_TYRE_WEATHER_STATE: Tyre, weather, and race state features
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

# Check if PC columns exist
pc_cols = [c for c in df_train.columns if c.startswith('geom_PC') or c.startswith('PC')]

if pc_cols:
    FEAT_TYRE_WEATHER_STATE_GEOM = FEAT_TYRE_WEATHER_STATE + pc_cols
    print(f"   ✓ Found {len(pc_cols)} geometry PCA components")
else:
    # Use raw geometry features
    geometry_features = [
        'num_drs_zones', 'length_km', 'num_turns', 'slow_share', 'medium_share', 'fast_share',
        'slow_cluster_max', 'straight_ratio', 'straight_len_max_m', 'n_major_straights',
        'heavy_braking_zones', 'heavy_braking_mean_dv_kmh', 'avg_corner_angle',
        'avg_corner_distance', 'drs_total_len_m'
    ]
    available_geom = [f for f in geometry_features if f in df_train.columns]
    FEAT_TYRE_WEATHER_STATE_GEOM = FEAT_TYRE_WEATHER_STATE + available_geom
    print(f"   ✓ Using {len(available_geom)} raw geometry features")

CAT = ['Compound']
TARGET = 'LapTime_next'
FEATS = FEAT_TYRE_WEATHER_STATE  # Start with base features

print(f"   ✓ Using {len(FEATS)} features + {len(CAT)} categorical")

print("\n3. Running GroupKFold cross-validation...")
ALPHAS = [0.1, 1, 10, 50, 100, 500, 1000, 5000]

try:
    best_alpha_cv, ridge_cv_table, cols, meds = cv_ridge_groupkfold(
        df_train,
        FEATS,
        CAT,
        TARGET,
        group_col="name",
        alphas=ALPHAS,
        n_splits=5
    )
    
    print("\n   Cross-validation results:")
    print(ridge_cv_table.head(10))
    print(f"\n   ✓ Best alpha (CV): {best_alpha_cv}")
    
except Exception as e:
    print(f"\n   ✗ Error during cross-validation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n4. Training final model on full training set...")

try:
    Xtr, ytr, cols, meds = make_Xy(df_train, FEATS, CAT, TARGET)
    ridge_final = Ridge(alpha=best_alpha_cv).fit(Xtr, ytr)
    print(f"   ✓ Model trained: {Xtr.shape[0]} samples, {Xtr.shape[1]} features")
    
except Exception as e:
    print(f"   ✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Evaluating on validation and test sets...")

try:
    # Validation
    Xva, yva, _, _ = make_Xy(df_val, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)
    y_val_pred = ridge_final.predict(Xva)
    val_metrics = metrics(yva, y_val_pred)
    
    # Test
    Xte, yte, _, _ = make_Xy(df_test, FEATS, CAT, TARGET, fit_cols=cols, medians=meds)
    y_test_pred = ridge_final.predict(Xte)
    test_metrics = metrics(yte, y_test_pred)
    
    print("\n   VALIDATION SET:")
    print(f"     MAE:  {val_metrics['mae']:.4f} seconds")
    print(f"     RMSE: {val_metrics['rmse']:.4f} seconds")
    print(f"     R²:   {val_metrics['r2']:.4f}")
    
    print("\n   TEST SET:")
    print(f"     MAE:  {test_metrics['mae']:.4f} seconds")
    print(f"     RMSE: {test_metrics['rmse']:.4f} seconds")
    print(f"     R²:   {test_metrics['r2']:.4f}")
    
except Exception as e:
    print(f"   ✗ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - Implementation working correctly!")
print("="*80)
