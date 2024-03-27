import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings(action='ignore')

# Fixed Random-Seed
def seed_everything(seed):
    np.random.seed(seed)

seed_everything(42)  # Seed 고정

# Load Data
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
building_info_df = pd.read_csv('./building_info.csv')

# Replace '-' with NaN in building_info_df
building_info_df.replace('-', np.nan, inplace=True)

# Merge building info into train and test data
train_df = pd.merge(train_df, building_info_df, on='건물번호', how='left')
test_df = pd.merge(test_df, building_info_df, on='건물번호', how='left')

# Check and handle missing values in train and test data
def handle_missing_values(df):
    # Fill numerical columns with mean
    numeric_cols = df.select_dtypes(include=np.number).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Fill categorical columns with most frequent value
    categorical_cols = df.select_dtypes(include='object').columns
    imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer.fit_transform(df[categorical_cols])

handle_missing_values(train_df)
handle_missing_values(test_df)

# Feature Engineering

# 시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
train_df['month'] = train_df['일시'].apply(lambda x: int(x[4:6]))
train_df['day'] = train_df['일시'].apply(lambda x: int(x[6:8]))
train_df['time'] = train_df['일시'].apply(lambda x: int(x[9:11]))

test_df['month'] = test_df['일시'].apply(lambda x: int(x[4:6]))
test_df['day'] = test_df['일시'].apply(lambda x: int(x[6:8]))
test_df['time'] = test_df['일시'].apply(lambda x: int(x[9:11]))

# Additional time-related features
train_df['hour_of_day'] = train_df['time']
train_df['morning'] = (train_df['hour_of_day'] >= 6) & (train_df['hour_of_day'] < 12)
train_df['afternoon'] = (train_df['hour_of_day'] >= 12) & (train_df['hour_of_day'] < 18)
train_df['evening'] = (train_df['hour_of_day'] >= 18) & (train_df['hour_of_day'] < 24)
train_df['weekend'] = (train_df['day'] % 7 == 5) | (train_df['day'] % 7 == 6)

test_df['hour_of_day'] = test_df['time']
test_df['morning'] = (test_df['hour_of_day'] >= 6) & (test_df['hour_of_day'] < 12)
test_df['afternoon'] = (test_df['hour_of_day'] >= 12) & (test_df['hour_of_day'] < 18)
test_df['evening'] = (test_df['hour_of_day'] >= 18) & (test_df['hour_of_day'] < 24)
test_df['weekend'] = (test_df['day'] % 7 == 5) | (test_df['day'] % 7 == 6)

# Interaction Features
train_df['temp_hum_interaction'] = train_df['기온(C)'] * train_df['습도(%)']
test_df['temp_hum_interaction'] = test_df['기온(C)'] * test_df['습도(%)']

# Data Center and University Binary Feature
train_df['is_datacenter'] = train_df['건물유형'].apply(lambda x: 1 if x == 'data center' else 0)
train_df['is_university'] = train_df['건물유형'].apply(lambda x: 1 if x == 'university' else 0)

test_df['is_datacenter'] = test_df['건물유형'].apply(lambda x: 1 if x == 'data center' else 0)
test_df['is_university'] = test_df['건물유형'].apply(lambda x: 1 if x == 'university' else 0)

# Drop unnecessary columns
train_df.drop(columns=['건물번호', '일조(hr)', '일사(MJ/m2)'], inplace=True)
test_df.drop(columns=['건물번호'], inplace=True)

# Label Encoding for categorical features
le = LabelEncoder()
train_df['건물유형'] = le.fit_transform(train_df['건물유형'])
test_df['건물유형'] = le.transform(test_df['건물유형'])

# Prepare Data for ARIMA model
# ARIMA Model
# We will use the '전력소비량(kWh)' data for the ARIMA model
# We'll take the log transformation of the target variable to make it more stationary

train_df['전력소비량(kWh)_log'] = np.log(train_df['전력소비량(kWh)'])

# Fit the ARIMA model with different orders until convergence
orders_to_try = [(p, d, q) for p in range(5) for d in range(2) for q in range(5)]
for order in orders_to_try:
    try:
        model_arima = ARIMA(train_df['전력소비량(kWh)_log'], order=order)
        model_fit = model_arima.fit()
        break
    except:
        continue
else:
    print("ARIMA Model Fit Error: Unable to find a converging order.")
    exit(1)

# Fill '전력소비량(kWh)' column in test_df with np.nan
test_df['전력소비량(kWh)'] = np.nan

# Make predictions for the test data
test_df['전력소비량(kWh)_log_pred'] = model_fit.forecast(steps=len(test_df))  # Forecast the next 'len(test_df)' steps

# Convert the log-transformed predictions back to the original scale
test_df['전력소비량(kWh)_pred'] = np.exp(test_df['전력소비량(kWh)_log_pred'])

# Fill NaN values in '전력소비량(kWh)_pred' with mean value of the log-transformed target variable
mean_log_target = train_df['전력소비량(kWh)_log'].mean()
test_df['전력소비량(kWh)_pred'].fillna(np.exp(mean_log_target), inplace=True)

# Evaluation
actual_values = test_df['전력소비량(kWh)']
predicted_values = test_df['전력소비량(kWh)_pred']
mse = mean_squared_error(actual_values, predicted_values)
mae = mean_absolute_error(actual_values, predicted_values)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = test_df['전력소비량(kWh)_pred']
submission.to_csv('./arima_submission.csv', index=False)