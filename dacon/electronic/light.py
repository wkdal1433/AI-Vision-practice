import random
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings(action='ignore') 

# Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

# Load Data with explicit encoding option and replace '-' with NaN
train_df = pd.read_csv('./train.csv', encoding='utf-8', na_values=['-'])
test_df = pd.read_csv('./test.csv', encoding='utf-8', na_values=['-'])
building_info_df = pd.read_csv('./building_info.csv', encoding='utf-8', na_values=['-'])

# Replace '-' with NaN and convert to numeric for building_info dataset
building_info_df.replace('-', np.nan, inplace=True)
building_info_df = building_info_df.apply(pd.to_numeric)

# Merge Building Info with Train and Test Data
train_df = train_df.merge(building_info_df, on='건물번호', how='left')
test_df = test_df.merge(building_info_df, on='건물번호', how='left')

# Extract Time Features (요일, 계절 등)
train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])

train_df['요일'] = train_df['일시'].dt.dayofweek
test_df['요일'] = test_df['일시'].dt.dayofweek

# 기온(C)에 따른 계절 추출
train_df['계절'] = train_df['기온(C)'].apply(lambda x: '봄' if 5 <= x <= 10 else ('여름' if 20 <= x <= 25 else ('가을' if 11 <= x <= 15 else '겨울')))
test_df['계절'] = test_df['기온(C)'].apply(lambda x: '봄' if 5 <= x <= 10 else ('여름' if 20 <= x <= 25 else ('가을' if 11 <= x <= 15 else '겨울')))

# One-Hot Encoding for Categorical Features
train_df = pd.get_dummies(train_df, columns=['건물유형', '계절'])
test_df = pd.get_dummies(test_df, columns=['건물유형', '계절'])

# Replace '-' with NaN and convert to numeric for train and test datasets
train_df.replace('-', np.nan, inplace=True)
test_df.replace('-', np.nan, inplace=True)
train_df = train_df.apply(pd.to_numeric)
test_df = test_df.apply(pd.to_numeric)

# Train Data Pre-Processing
train_df.fillna(train_df.mean(), inplace=True)

train_x = train_df.drop(columns=['num_date_time', '일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']

# Feature Scaling
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)

# Split train and validation data
X_train, X_val, y_train, y_val = train_test_split(train_x_scaled, train_y, test_size=0.2, random_state=42)

# XGBoost Regression Model with Hyperparameter Tuning
xgb_model = xgb.XGBRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
grid_search_xgb = GridSearchCV(xgb_model, param_grid, cv=5, n_jobs=-1)
grid_search_xgb.fit(X_train, y_train)

best_xgb_model = grid_search_xgb.best_estimator_

# LightGBM Regression Model with Hyperparameter Tuning
lgb_model = lgb.LGBMRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}
grid_search_lgb = GridSearchCV(lgb_model, param_grid, cv=5, n_jobs=-1)
grid_search_lgb.fit(X_train, y_train)

best_lgb_model = grid_search_lgb.best_estimator_

# Evaluation on Validation Data
val_preds_xgb = best_xgb_model.predict(X_val)
val_preds_lgb = best_lgb_model.predict(X_val)
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, val_preds_xgb))
val_rmse_lgb = np.sqrt(mean_squared_error(y_val, val_preds_lgb))
print(f"Validation RMSE (XGBoost): {val_rmse_xgb:.4f}")
print(f"Validation RMSE (LightGBM): {val_rmse_lgb:.4f}")

# Select the better model between XGBoost and LightGBM
best_model = best_xgb_model if val_rmse_xgb < val_rmse_lgb else best_lgb_model

# Test Data Pre-Processing
test_df.fillna(test_df.mean(), inplace=True)

test_df['일시'] = pd.to_datetime(test_df['일시'])
test_df['요일'] = test_df['일시'].dt.dayofweek
test_df['계절'] = test_df['기온(C)'].apply(lambda x: '봄' if 5 <= x <= 10 else ('여름' if 20 <= x <= 25 else ('가을' if 11 <= x <= 15 else '겨울')))
test_df = pd.get_dummies(test_df, columns=['건물유형', '계절'])

# Feature Scaling
test_x_scaled = scaler.transform(test_df.drop(columns=['num_date_time', '일시']))

# Ensemble Predictions
ensemble_preds = (best_xgb_model.predict(test_x_scaled) + best_lgb_model.predict(test_x_scaled)) / 2

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = ensemble_preds
submission.to_csv('./modified_submission.csv', index=False)