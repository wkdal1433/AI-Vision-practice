import random
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb

import warnings
warnings.filterwarnings(action='ignore')

# Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)  # Seed 고정

# Load Data
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
building_info_df = pd.read_csv('./building_info.csv')

# Merge Building Info
train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

# Data Pre-Processing
# Replace '-' values with NaN
train_df.replace('-', np.nan, inplace=True)
test_df.replace('-', np.nan, inplace=True)

# Convert columns to appropriate data types
train_df = train_df.astype({'강수량(mm)': float, '태양광용량(kW)': float, 'ESS저장용량(kWh)': float})
test_df = test_df.astype({'강수량(mm)': float, '태양광용량(kW)': float, 'ESS저장용량(kWh)': float})

# Fill missing values with column means
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Time Features
def add_time_features(df):
    df['month'] = df['일시'].apply(lambda x: int(x[4:6]))
    df['day'] = df['일시'].apply(lambda x: int(x[6:8]))
    df['time'] = df['일시'].apply(lambda x: int(x[9:11]))
    return df

train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

# Additional Pre-Processing and Feature Engineering
def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(columns=['num_date_time', '일시', '일조(hr)', '일사(MJ/m2)'], inplace=True)

    # Feature Scaling - Min-Max Scaling
    scaler = StandardScaler()
    numeric_cols = ['기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
    
    # Fill missing values with 0 before scaling
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Feature Engineering - Lagging Features
    lagging_periods = [3, 6, 12, 24]  # Use data from the previous 3, 6, 12, and 24 hours
    for col in numeric_cols:
        for lag in lagging_periods:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Train-Validation Split
X = train_df.drop(columns=['전력소비량(kWh)'])
y = train_df['전력소비량(kWh)']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM Model
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)

# Model Evaluation
def evaluate_model(model, X, y):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    return mse, mae

lgb_mse, lgb_mae = evaluate_model(lgb_model, X_val, y_val)

print("LightGBM:")
print("Validation MSE:", lgb_mse)
print("Validation MAE:", lgb_mae)

# Final Predictions
final_test_preds = lgb_model.predict(test_df)
final_test_preds = final_test_preds.flatten()

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = final_test_preds
submission.to_csv('./modified_baseline_submission.csv', index=False)