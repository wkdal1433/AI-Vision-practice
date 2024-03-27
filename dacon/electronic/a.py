# 필요한 라이브러리 import
from prophet import Prophet
import pandas as pd
import numpy as np
import random
import os

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
model = RandomForestRegressor()
model.fit(train_x, train_y)

# Test Data Pre-Processing
test_df.fillna(test_df.mean(), inplace=True)

# 시계열 특성을 학습에 반영하기 위해 일시를 월, 일, 시간으로 나눕니다
test_df['month'] = test_df['일시'].apply(lambda x: int(x[4:6]))
test_df['day'] = test_df['일시'].apply(lambda x: int(x[6:8]))
test_df['time'] = test_df['일시'].apply(lambda x: int(x[9:11]))

# Add '전력소비량(kWh)' column to test data and fill with NaN values
test_df['전력소비량(kWh)'] = np.nan

# Inference using NeuralProphet model
# 이전에 사용한 neuralprophet 코드를 이용해 추론한 후 예측 결과를 가져옵니다.
train = pd.read_csv('./train.csv')
results_df = pd.DataFrame(columns=['건물번호', 'final_return'])
unique_codes = train['건물번호'].unique()

iter_num = 1  # 변수 추가: 몇 번째 실행 중인지를 나타내는 변수

for code in tqdm(unique_codes):
    train_close = train[train['건물번호'] == code][['일시', '전력소비량(kWh)']]
    train_close['ds'] = pd.to_datetime(train_close['일시'].str.slice(0, 8), format='%Y%m%d')
    train_close.set_index('일시', inplace=True)
    train_close.rename(columns={"전력소비량(kWh)": "y"}, inplace=True)

    m = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        epochs=10,
        n_lags=3,
        n_forecasts=15
    )
    metrics = m.fit(train_close, freq="D")

    df_future = m.make_future_dataframe(train_close, periods=15, n_historic_predictions=10)
    forecast = m.predict(df_future)

    preds = forecast['yhat15'][-15:]

    final_return = (preds.iloc[-1] - preds.iloc[0]) / preds.iloc[0]

    # 결과 저장
    results_df = results_df.append({'건물번호': code, 'final_return': final_return}, ignore_index=True)
    print('{0} 번째 실행중 '.format(iter_num))  # 변수 추가: 몇 번째 실행 중인지 출력
    iter_num += 1

# Combine the predictions from NeuralProphet and RandomForestRegressor
# 추론 결과를 NeuralProphet 모델의 결과와 RandomForestRegressor 모델의 결과를 결합합니다.
merged_df = pd.merge(test_df, results_df, on='건물번호', how='left')
merged_df['전력소비량(kWh)'].fillna(merged_df['final_return'], inplace=True)

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = merged_df['전력소비량(kWh)']
submission.to_csv('./modified_baseline_submission.csv', index=False)

# Evaluation
actual_values = merged_df['전력소비량(kWh)']
preds = merged_df['전력소비량(kWh)']

mse = mean_squared_error(actual_values, preds)
mae = mean_absolute_error(actual_values, preds)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)