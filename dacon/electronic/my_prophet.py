# Import

import pandas as pd
import numpy as np
import os

from neuralprophet import NeuralProphet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Fixed Random-Seed

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

# Load Data

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

# Train Data Pre-Processing

# 결측치를 해당 열의 평균값으로 채우기
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# 시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
train_df['ds'] = pd.to_datetime(train_df['일시'])  # '일시' 컬럼을 datetime 형태로 변환하여 'ds' 컬럼으로 추가
train_df.rename(columns={'전력소비량(kWh)': 'y'}, inplace=True)  # NeuralProphet의 Target column 이름은 'y'로 사용

train_df = train_df[['ds', 'y']]  # 필요한 column만 선택

# Initialize and fit NeuralProphet model

model = NeuralProphet(
    n_forecasts=24,      # 24시간 예측
    n_lags=168,          # 168시간(1주일)의 과거 데이터를 기반으로 예측
    changepoints_range=0.95,
)
model.fit(train_df, freq='H')  # 'freq' 인자로 시계열 데이터의 주기를 지정, 여기서는 1시간('H') 단위로 데이터가 주기성을 가짐

# Test Data Pre-Processing

test_df['ds'] = pd.to_datetime(test_df['일시'])  # '일시' 컬럼을 datetime 형태로 변환하여 'ds' 컬럼으로 추가
test_df = test_df[['ds']]  # 필요한 column만 선택

# Inference

forecast = model.predict(test_df)

# Get actual values (전력소비량)

actual_values = test_df['전력소비량(kWh)']  # 이 부분은 '전력소비량(kWh)' 컬럼의 실제 값이라고 가정합니다

# 평가 지표 계산

preds = forecast['yhat1'].values  # neuralprophet의 예측 결과는 'yhat1' 컬럼에 있습니다
mse = mean_squared_error(actual_values, preds)
mae = mean_absolute_error(actual_values, preds)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Submission

submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = preds
submission.to_csv('./neuralprophet_submission.csv', index=False)