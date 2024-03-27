import random
import pandas as pd
import numpy as np
import os
from prophet import Prophet


from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Regression Model Fit
model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05)

# Assuming train_df is already prepared with necessary features and 'power_consumption' column
train_data = train_df[['일시', '전력소비량(kWh)']]
train_data = train_data.rename(columns={'일시': 'ds', '전력소비량(kWh)': 'y'})

model.fit(train_data)

# Test Data Pre-Processing
test_df.fillna(test_df.mean(), inplace=True)

# 시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
test_df['month'] = test_df['일시'].apply(lambda x: int(x[4:6]))
test_df['day'] = test_df['일시'].apply(lambda x: int(x[6:8]))
test_df['time'] = test_df['일시'].apply(lambda x: int(x[9:11]))

# Test 데이터에 대한 날짜(ds) 컬럼 추가
test_df = test_df[['일시']].rename(columns={'일시': 'ds'})

# Add '전력소비량(kWh)' column to test data and fill with NaN values
test_df['전력소비량(kWh)'] = np.nan

# Inference
test_x = test_df.drop(columns=['ds', '전력소비량(kWh)'])
preds = model.predict(test_x)

# Add predictions to '전력소비량(kWh)' column in test data
test_df['전력소비량(kWh)'] = preds
test_dataframe = pd.read_csv('./test.csv')

test_df.insert(0, 'num_date_time', test_dataframe['num_date_time'])


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