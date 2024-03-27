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

# Initialize and fit NeuralProphet model per '건물번호'

results_df = pd.DataFrame(columns=['건물번호', 'final_return'])  # 추론 결과를 저장하기 위한 dataframe 생성
unique_building_numbers = train_df['건물번호'].unique()

for building_number in unique_building_numbers:
    train_building = train_df[train_df['건물번호'] == building_number][['ds', 'y']]
    m = NeuralProphet(
        yearly_seasonality=False,  # 년간 계절성 설정
        weekly_seasonality=False,  # 주간 계절성 설정
        daily_seasonality=False,   # 일간 계절성 설정
        epochs=10,                 # 학습 횟수 설정
        n_lags=3,
        n_forecasts=3
    )
    metrics = m.fit(train_building, freq="D")

    df_future = m.make_future_dataframe(train_building, periods=15,  n_historic_predictions=10)
    forecast = m.predict(df_future)

    preds = forecast['yhat3'][-3:]

    final_return = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]
    # 결과 저장
    results_df = results_df.append({'건물번호': building_number, 'final_return': final_return}, ignore_index=True)
    print('건물번호 {0} 실행 완료'.format(building_number))

# Submission

submission = pd.read_csv('./sample_submission.csv')
baseline_submission = submission[['건물번호']].merge(results_df[['건물번호', 'final_return']], on='건물번호', how='left')
baseline_submission.to_csv('baseline_submission.csv', index=False)