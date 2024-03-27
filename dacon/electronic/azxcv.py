import random
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from neuralprophet import NeuralProphet
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
building_info_df = pd.read_csv('./building_info.csv')

# Data Pre-Processing
def preprocess_data(df):
    df['일시'] = pd.to_datetime(df['일시'], format='%Y%m%d %H')
    df = pd.merge(df, building_info_df, on='건물번호', how='left')
    df.drop(columns=['num_date_time', '건물번호'], inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Feature Scaling
scaler = StandardScaler()
numerical_columns = train_df.select_dtypes(include=np.number).columns.tolist()
scaler.fit(train_df[numerical_columns])
train_df[numerical_columns] = scaler.transform(train_df[numerical_columns])
test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

# NeuralProphet Model
def train_neural_prophet(df):
    model = NeuralProphet(
        n_forecasts=1,
        n_lags=12,  # 데이터 주기에 맞는 적절한 lag 값 사용
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        batch_size=64,
        epochs=100,
    )
    model.fit(df, freq='H')
    return model

# 훈련 데이터로 NeuralProphet 모델 학습
train_x = train_df.drop(columns=['일시', '전력소비량(kWh)'])
train_y = train_df['전력소비량(kWh)']
neuralprophet_model = train_neural_prophet(train_df)

# NeuralProphet을 사용하여 테스트 데이터셋에 대한 예측 수행
forecast = neuralprophet_model.predict(test_df, freq='H')
preds = forecast['yhat1'].values

# 예측값에 스케일링 역변환 적용
preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

# 평가
actual_values = test_df['전력소비량(kWh)']
mse = mean_squared_error(actual_values, preds)
mae = mean_absolute_error(actual_values, preds)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Submission
submission = pd.read_csv('./sample_submission.csv')
submission['answer'] = preds
submission.to_csv('./modified_neuralprophet_submission.csv', index=False)