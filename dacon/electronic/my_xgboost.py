import random
import pandas as pd
import numpy as np
import os

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime

import warnings
warnings.filterwarnings(action='ignore')

# Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

# Load Data
# Load Data with explicit encoding option
train_df = pd.read_csv('./train.csv', encoding='utf-8')
test_df = pd.read_csv('./test.csv', encoding='utf-8')
building_info_df = pd.read_csv('./building_info.csv', encoding='utf-8')

train_df = train_df.merge(building_info_df, left_on='num', right_on='건물 번호', how='left')
test_df = test_df.merge(building_info_df, left_on='num', right_on='건물 번호', how='left')

# Convert '일시' column to datetime format
train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])

# Feature Engineering
def feature_engineering(df):
    df['month'] = df['일시'].dt.month
    df['day'] = df['일시'].dt.day
    df['hour'] = df['일시'].dt.hour
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Train Data Pre-Processing
train_df.fillna(train_df.mean(), inplace=True)

train_x = train_df.drop(columns=['num_date_time', '일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']

# Feature Scaling
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)

# Split train and validation data
X_train, X_val, y_train, y_val = train_test_split(train_x_scaled, train_y, test_size=0.2, random_state=42)

# Regression Model with Hyperparameter Tuning
model = xgb.XGBRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluation on Validation Data
val_preds = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
val_mae = mean_absolute_error(y_val, val_preds)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAE: {val_mae:.4f}")

# Test Data Pre-Processing
test_df.fillna(test_df.mean(), inplace=True)

# Convert '일시' column to datetime format
test_df['일시'] = pd.to_datetime(test_df['일시'])

test_x = test_df.drop(columns=['num_date_time', '일시', '전력소비량(kWh)'])

# Feature Engineering for Test Data
test_x = feature_engineering(test_x)

# Feature Scaling for Test Data
test_x_scaled = scaler.transform(test_x)

# Inference
test_preds = best_model.predict(test_x_scaled)

# Add '전력소비량(kWh)' column to test data and fill with predictions
test_df['전력소비량(kWh)'] = test_preds

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = test_df['전력소비량(kWh)']
submission.to_csv('./modified_submission.csv', index=False)