import random
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings(action='ignore')

# Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)  # Seed 고정

# Load Data with specified encoding
train_df = pd.read_csv('./train.csv', encoding='utf-8')
test_df = pd.read_csv('./test.csv', encoding='utf-8')
building_info_df = pd.read_csv('./building_info.csv')

# Check the column containing the special characters in train_df
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        try:
            train_df[col] = train_df[col].astype(float)
        except ValueError as e:
            print(f"Error occurred in column '{col}':")
            print(train_df[col].unique())

# Check the column containing the special characters in test_df
for col in test_df.columns:
    if test_df[col].dtype == 'object':
        try:
            test_df[col] = test_df[col].astype(float)
        except ValueError as e:
            print(f"Error occurred in column '{col}':")
            print(test_df[col].unique())
# Merge train_df and building_info_df based on '건물번호'
train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')

# Building Info Column Names
building_info_columns = ['건물번호', '건물유형', '연면적(m2)', '냉방면적(m2)', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']

# Test Data Column Names
test_columns = ['num_date_time', '건물번호', '일시', '기온(C)', '강수량(mm)', '풍속(m/s)', '습도(%)']

# Train Data Pre-Processing
train_df.fillna(train_df.mean(), inplace=True)

# 시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
train_df['month'] = train_df['일시'].apply(lambda x: int(x[4:6]))
train_df['day'] = train_df['일시'].apply(lambda x: int(x[6:8]))
train_df['time'] = train_df['일시'].apply(lambda x: int(x[9:11]))

train_x = train_df.drop(columns=['num_date_time', '일시', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']

# Feature Scaling
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)

# Regression Model Fit
model = RandomForestRegressor()
model.fit(train_x_scaled, train_y)

# Test Data Pre-Processing
test_df.fillna(test_df.mean(), inplace=True)

# 시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
test_df['month'] = test_df['일시'].apply(lambda x: int(x[4:6]))
test_df['day'] = test_df['일시'].apply(lambda x: int(x[6:8]))
test_df['time'] = test_df['일시'].apply(lambda x: int(x[9:11]))

# Merge test_df and building_info_df based on '건물번호'
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

# Feature Scaling for test data
test_x_scaled = scaler.transform(test_df.drop(columns=['num_date_time', '일시', '전력소비량(kWh)']))

# Inference
preds = model.predict(test_x_scaled)

# Add predictions to '전력소비량(kWh)' column in test data
test_df['전력소비량(kWh)'] = preds

# Evaluation
actual_values = test_df['전력소비량(kWh)']
mse = mean_squared_error(actual_values, preds)
mae = mean_absolute_error(actual_values, preds)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = preds
submission.to_csv('./modified_baseline_submission.csv', index=False)