"""
Test script for Ridge regression with GroupKFold implementation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Load the train/val/test sets
df_train = pd.read_excel('csv_output/Train_set.xlsx')
df_val = pd.read_excel('csv_output/Validation_set.xlsx')
df_test = pd.read_excel('csv_output/Test_set.xlsx')

print(f"✓ Train set: {df_train.shape}")
print(f"✓ Val set: {df_val.shape}")
print(f"✓ Test set: {df_test.shape}")

print("\nChecking required columns...")

# Check which features exist
required_features = [
    'LapInStint', 'LapInStint_squared', 'TyreAgeAtStart', 'is_new_tyre',
    'laptime_rolling_std_3', 'laptime_cumulative_trend', 'laptime_change_prev',
    'AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed', 'wind_sin', 'wind_cos',
    'is_leader', 'in_drs_range', 'in_clean_air', 'in_dirty_air', 'pushing'
]

target_col = 'LapTime_next'
cat_col = 'Compound'
group_col = 'name'

existing_features = [f for f in required_features if f in df_train.columns]
missing_features = [f for f in required_features if f not in df_train.columns]

print(f"  Existing features: {len(existing_features)}/{len(required_features)}")
if missing_features:
    print(f"  Missing features: {missing_features}")

# Check target
if target_col in df_train.columns:
    print(f"✓ Target '{target_col}' exists")
else:
    print(f"✗ Target '{target_col}' missing")
    
# Check categorical
if cat_col in df_train.columns:
    print(f"✓ Categorical '{cat_col}' exists")
else:
    print(f"✗ Categorical '{cat_col}' missing")
    
# Check grouping column
if group_col in df_train.columns:
    print(f"✓ Group column '{group_col}' exists")
    print(f"  Unique circuits in train: {df_train[group_col].nunique()}")
else:
    print(f"✗ Group column '{group_col}' missing")

# List all available columns
print(f"\nAvailable columns ({len(df_train.columns)}):")
print(df_train.columns.tolist()[:50])  # First 50 columns

# Check for any lap time column
laptime_cols = [c for c in df_train.columns if 'laptime' in c.lower() or 'lap_time' in c.lower()]
print(f"\nLap time related columns: {laptime_cols[:10]}")

# Check for compound-related columns
compound_cols = [c for c in df_train.columns if 'compound' in c.lower() or 'tyre' in c.lower() or 'tire' in c.lower()]
print(f"Tyre/Compound related columns: {compound_cols[:10]}")

print("\n" + "="*80)
print("Data check complete!")
print("="*80)
